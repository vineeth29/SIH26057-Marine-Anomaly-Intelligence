from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response

from . import pipeline_bridge
from .schemas import (
    AnalysisResponse,
    ModelInfo,
    SystemStatusResponse,
    DashboardStats,
    HealthResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        model_info = pipeline_bridge.get_model_info()
        return HealthResponse(
            status="ok",
            model_loaded=model_info.status == "READY",
            version="0.1.0",
        )
    except Exception as exc:  # noqa: BLE001
        return HealthResponse(status="error", model_loaded=False, version="0.1.0")


@router.get("/model", response_model=ModelInfo)
def get_model() -> ModelInfo:
    return pipeline_bridge.get_model_info()


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status() -> SystemStatusResponse:
    return pipeline_bridge.get_system_status()


@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats() -> DashboardStats:
    return pipeline_bridge.get_dashboard_stats()


@router.post("/sonar/analyze", response_model=AnalysisResponse)
async def analyze_sonar(
    file: UploadFile = File(...),
    mission_id: str = Form(default="MISSION-001"),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None),
    depth_m: float | None = Form(default=None),
) -> AnalysisResponse:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a sonar image (PNG/JPEG).",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = pipeline_bridge.run_analysis(
            image_bytes=image_bytes,
            filename=file.filename or "upload.png",
            mission_id=mission_id,
            lat=lat,
            lon=lon,
            depth_m=depth_m,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # Do not leak internal paths/tracebacks to the client.
        raise HTTPException(
            status_code=500,
            detail="Analysis failed on the server. Please try again.",
        ) from exc

    return result


@router.get("/analysis/{analysis_id}/image")
def get_analysis_image(analysis_id: str, variant: str = "raw") -> Response:
    """
    Serves an image for a completed analysis.

    variant:
      raw        - original uploaded image, no overlay (default; the
                   frontend draws its own interactive bbox overlay on
                   top of this using AnalysisResponse.detections[].bbox)
      processed  - after sonar preprocessing (CLAHE/denoise)
      annotated  - server-baked image with boxes drawn by the existing
                   annotate_image() in pipeline_service.py (static
                   fallback / what a downloaded image would look like)

    Backed by an in-memory cache populated at analyze time — see
    pipeline_bridge._IMAGE_CACHE for the current-process-only caveat.
    """
    if variant not in ("raw", "processed", "annotated"):
        raise HTTPException(
            status_code=400,
            detail="variant must be one of: raw, processed, annotated",
        )
    png_bytes = pipeline_bridge.get_cached_image(analysis_id, variant)
    if png_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="Image not available for this analysis (not found or server restarted).",
        )
    return Response(content=png_bytes, media_type="image/png")
