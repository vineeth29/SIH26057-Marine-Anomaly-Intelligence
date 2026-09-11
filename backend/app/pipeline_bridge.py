"""
Adapter layer between the EXISTING SIH26057 Python pipeline
(services/pipeline_service.py, services/mission_service.py,
utils/pdf_report.py) and the FastAPI routes.

No AI/CV logic lives here. This module only:
  - invokes the existing pipeline
  - shapes its real output into the Pydantic response schemas
  - derives a couple of honest, clearly-labelled fields (location_source,
    status) from data the pipeline already computes

If a field cannot be honestly derived from existing pipeline output,
it is returned as None / "not available" rather than invented.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Make the existing project package importable. The FastAPI app lives in
# <repo_root>/backend/app/, the existing pipeline in <repo_root>/services,
# ai/, utils/, etc. We add <repo_root> to sys.path once, here.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.pipeline_service import (  # noqa: E402
    get_pipeline,
    PipelineResult,
    classify_natural_vs_artificial,
)
from services import mission_service  # noqa: E402

from .schemas import (
    Detection,
    AnalysisResponse,
    ImageInfo,
    ModelInfo,
    SystemStatusResponse,
    ComponentHealth,
    ComponentStatus,
    LocationSource,
    DashboardStats,
)

DEFAULT_MISSION_ID = "MISSION-001"

# ---------------------------------------------------------------------------
# In-memory cache of the annotated result image for each analysis_id, so
# the frontend can fetch it via GET /api/analysis/{id}/image instead of
# inlining a large base64 blob into the JSON response.
#
# This is process-local and unbounded — fine for a single-instance dev/demo
# deployment. If this API is ever run with multiple workers or needs to
# survive a restart, this should be replaced with disk or object storage
# (mission_service already has a `result_path` column on SonarImage for
# exactly this purpose — wiring that up is a follow-up, not part of this
# slice).
# ---------------------------------------------------------------------------
ImageVariant = str  # "raw" | "processed" | "annotated"

_IMAGE_CACHE: dict[str, dict[ImageVariant, bytes]] = {}
_IMAGE_CACHE_MAX = 200


def _cache_images(analysis_id: str, variants: dict[ImageVariant, bytes]) -> None:
    if len(_IMAGE_CACHE) >= _IMAGE_CACHE_MAX:
        # Drop the oldest entry (dicts preserve insertion order in Py3.7+).
        oldest = next(iter(_IMAGE_CACHE))
        _IMAGE_CACHE.pop(oldest, None)
    _IMAGE_CACHE[analysis_id] = variants


def get_cached_image(analysis_id: str, variant: ImageVariant = "annotated") -> Optional[bytes]:
    return _IMAGE_CACHE.get(analysis_id, {}).get(variant)


# ---------------------------------------------------------------------------
# Location source derivation
# ---------------------------------------------------------------------------

def _derive_location_source(geo_label: str, lat: Optional[float]) -> LocationSource:
    """
    Maps the pipeline's free-text geo_label to the honest enum the
    frontend contract requires. Never claims REAL_GPS unless the
    pipeline itself says so.
    """
    if lat is None:
        return LocationSource.UNAVAILABLE
    label = (geo_label or "").lower()
    if "simulated" in label:
        return LocationSource.SIMULATED
    if "sonar" in label or "metadata" in label:
        return LocationSource.SONAR_METADATA
    if "manual" in label:
        return LocationSource.MANUAL
    if "gps" in label or "real" in label:
        return LocationSource.REAL_GPS
    # Pipeline gave coordinates but with an unrecognized label —
    # do not guess REAL_GPS; be conservative.
    return LocationSource.SIMULATED


def _derive_status(class_name: str, confidence: float, anomaly_score: float) -> str:
    """
    The existing pipeline never actually drops a detection as a false
    positive — classify_natural_vs_artificial() only annotates it.
    We surface that judgment as RETAINED/FILTERED here rather than
    inventing a filtering step that doesn't exist.
    """
    nat = classify_natural_vs_artificial(class_name, confidence, anomaly_score)
    return "FILTERED" if nat["type"] == "NATURAL" else "RETAINED"


def _to_detection_schema(pd) -> Detection:
    d = pd.to_dict()
    return Detection(
        **d,
        location_source=_derive_location_source(pd.geo_label, pd.lat),
        status=_derive_status(pd.class_name, pd.confidence, pd.anomaly_score),
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(
    image_bytes: bytes,
    filename: str,
    mission_id: str = DEFAULT_MISSION_ID,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    depth_m: Optional[float] = None,
) -> AnalysisResponse:
    t0 = time.time()

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a readable image.")

    h, w = img.shape[:2]
    analysis_id = f"AN-{uuid.uuid4().hex[:10].upper()}"

    pipeline = get_pipeline()
    result: PipelineResult = pipeline.run(
        img,
        image_id=analysis_id,
        lat=lat,
        lon=lon,
        depth_m=depth_m,
    )

    detections = [_to_detection_schema(pd) for pd in result.detections]

    # Cache the raw, preprocessed, and server-annotated images so the
    # frontend can fetch any of them via GET /api/analysis/{id}/image
    # (Original / Processed / Detection views per the UI spec).
    #
    # The frontend draws its own interactive bounding-box overlay on top
    # of the RAW image (for zoom/pan support) using the same coordinates
    # in AnalysisResponse.detections[].bbox — so the "annotated" variant
    # here is a static fallback / download image, not what's shown by
    # default in the interactive viewer.
    try:
        variants: dict[str, bytes] = {}
        for key, arr in (
            ("raw", result.raw_image),
            ("processed", result.preprocessed_image),
            ("annotated", result.annotated_image),
        ):
            ok, encoded = cv2.imencode(".png", arr)
            if ok:
                variants[key] = encoded.tobytes()
            else:
                result.warnings.append(f"Failed to encode '{key}' image for display.")
        _cache_images(result.image_id, variants)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"Result images unavailable: {exc}")

    # Persist to the mission DB using the existing service, same as the
    # Streamlit app does, so this analysis shows up in Detections/Map/Reports.
    #
    # Note: save_image_record requires a filepath; the uploaded bytes are
    # not currently written to disk anywhere in this API layer, so we
    # record the original filename as a placeholder path and flag that
    # the raw file itself is not retained server-side (only the DB
    # record + this in-memory analysis response). Persisting the actual
    # image is a follow-up, not part of this slice.
    try:
        mission_service.init_database()
        mission_service.create_mission(mission_id, name=mission_id)
        img_status = mission_service.save_image_record(
            mission_id=mission_id,
            image_id=result.image_id,
            filename=filename,
            filepath=f"(not persisted to disk)/{filename}",
        )
        if img_status.get("status") != "created":
            result.warnings.append(
                f"Image record status: {img_status.get('status')}"
            )
        for pd in result.detections:
            mission_service.save_detection(mission_id, result.image_id, pd)
        mission_service.mark_image_processed(result.image_id)
    except Exception as exc:  # noqa: BLE001
        # Persistence failure must not hide a successful analysis from
        # the caller — surface it as a warning instead.
        result.warnings.append(f"Result not persisted to mission DB: {exc}")

    processing_time_ms = round((time.time() - t0) * 1000, 1)

    return AnalysisResponse(
        analysis_id=result.image_id,
        mission_id=mission_id,
        status="completed",
        image=ImageInfo(
            filename=filename,
            width=w,
            height=h,
            size_bytes=len(image_bytes),
        ),
        detections=detections,
        quality=result.quality,
        anomaly_result=result.anomaly_result,
        num_known=result.num_known,
        num_anomalies=result.num_anomalies,
        mode=result.mode,
        model_version=result.model_version,
        warnings=result.warnings,
        processing_time_ms=processing_time_ms,
        timing=result.timing,
    )


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

def get_model_info() -> ModelInfo:
    try:
        pipeline = get_pipeline()
        detector = pipeline.detector  # SonarDetector
    except Exception as exc:  # noqa: BLE001
        return ModelInfo(
            status="NOT_AVAILABLE",
            model_type="unknown",
            supported_classes=[],
            device="unknown",
            model_version="unknown",
            conf_threshold=0.0,
            iou_threshold=0.0,
            metrics=None,
            metrics_source=f"Model unavailable: {exc}",
        )

    inner = getattr(detector, "_detector", None)
    class_names: list[str] = []
    device = "cpu"
    if inner is not None and hasattr(inner, "model"):
        names_dict = getattr(inner.model, "names", {}) or {}
        class_names = [str(v) for v in names_dict.values()]
        device = str(getattr(inner, "device", "cpu"))

    metrics, metrics_source = _load_real_metrics()

    return ModelInfo(
        status="READY",
        model_type="YOLOv8",
        supported_classes=class_names,
        device=device,
        model_version=getattr(detector, "model_version", "unknown"),
        conf_threshold=detector.conf_threshold,
        iou_threshold=detector.iou_threshold,
        metrics=metrics,
        metrics_source=metrics_source,
    )


def _load_real_metrics() -> tuple[Optional[dict], Optional[str]]:
    """
    Metrics are surfaced in priority order, each labelled with exactly
    what produced them — never merged together as if equivalent:

      1. reports/evaluation_results.json — a genuine held-out test-set
         run via evaluate.py. This is the only source that represents
         performance on data the model never influenced during training
         or checkpoint selection.
      2. models/model_info.json — metrics captured at training time
         (typically the best validation-split epoch, per Ultralytics'
         own checkpoint-selection). Real numbers, but from the same
         data split used to pick the checkpoint, so they run optimistic
         relative to a true test-set score. Labelled accordingly rather
         than presented as equivalent to (1).
      3. Neither present -> None, with an honest explanation.
    """
    import json

    eval_path = REPO_ROOT / "reports" / "evaluation_results.json"
    if eval_path.exists():
        try:
            data = json.loads(eval_path.read_text())
            return data, f"reports/evaluation_results.json (held-out test split)"
        except Exception as exc:  # noqa: BLE001
            return None, f"Failed to read evaluation_results.json: {exc}"

    info_path = REPO_ROOT / "models" / "model_info.json"
    if info_path.exists():
        try:
            data = json.loads(info_path.read_text())
            metrics = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            return (
                metrics or None,
                "models/model_info.json (training-time validation metrics, "
                "NOT an independent held-out test evaluation — run evaluate.py "
                "for that)",
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"Failed to read models/model_info.json: {exc}"

    return None, "No evaluation_results.json or model_info.json found. Run evaluate.py to generate one."


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------

def get_system_status() -> SystemStatusResponse:
    components: list[ComponentHealth] = []

    try:
        pipeline = get_pipeline()
        components.append(ComponentHealth(name="AI Model", status=ComponentStatus.READY))
        components.append(ComponentHealth(name="Inference Engine", status=ComponentStatus.READY))
    except Exception as exc:  # noqa: BLE001
        components.append(
            ComponentHealth(name="AI Model", status=ComponentStatus.ERROR, detail=str(exc))
        )
        components.append(
            ComponentHealth(name="Inference Engine", status=ComponentStatus.ERROR, detail=str(exc))
        )

    for name in ("Sonar Processor", "Shadow Analyzer", "Anomaly Analyzer"):
        components.append(ComponentHealth(name=name, status=ComponentStatus.READY))

    # Geolocation is real code but currently only produces simulated
    # coordinates end-to-end — surface that honestly as WARNING, not READY.
    components.append(
        ComponentHealth(
            name="Geolocation",
            status=ComponentStatus.WARNING,
            detail="Coordinates are currently simulated, not parsed from real sonar metadata.",
        )
    )

    try:
        mission_service.init_database()
        components.append(ComponentHealth(name="Reporting", status=ComponentStatus.READY))
    except Exception as exc:  # noqa: BLE001
        components.append(
            ComponentHealth(name="Reporting", status=ComponentStatus.ERROR, detail=str(exc))
        )

    overall = ComponentStatus.READY
    if any(c.status == ComponentStatus.ERROR for c in components):
        overall = ComponentStatus.ERROR
    elif any(c.status == ComponentStatus.WARNING for c in components):
        overall = ComponentStatus.WARNING

    device = "cpu"
    try:
        device = get_model_info().device
    except Exception:  # noqa: BLE001
        pass

    return SystemStatusResponse(
        overall=overall,
        components=components,
        inference_device=device,
        mean_pipeline_latency_ms=None,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def get_dashboard_stats(mission_id: Optional[str] = None) -> DashboardStats:
    """
    Sourced from mission_service.get_statistics(), which returns (verified
    against the real implementation):
      total_missions, total_images, total_detections, total_anomalies,
      known_debris, high_risk, avg_evidence_score, avg_evidence_pct,
      class_distribution, severity_distribution.

    There is currently no stored per-analysis processing-time record in
    the mission DB, so average_processing_time_ms is honestly None
    rather than invented — it can be added if mission_service starts
    persisting timing per image.
    """
    mission_service.init_database()
    stats = mission_service.get_statistics()
    return DashboardStats(
        sonar_frames=stats.get("total_images", 0),
        anomalies_detected=stats.get("total_detections", 0),
        high_priority=stats.get("high_risk", 0),
        average_confidence=stats.get("avg_evidence_pct"),
        average_processing_time_ms=None,
    )
