"""
FastAPI routing layer for SIH26057 Sonar Intelligence System.
"""
from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response, Query

from . import pipeline_bridge
from .schemas import (
    MapTargetsResponse, MapTarget,
    AnalysisResponse,
    BatchAnalysisResponse,
    ModelInfo,
    SystemStatusResponse,
    DashboardStats,
    HealthResponse,
    MissionCreateRequest,
    MissionSummary,
    MissionDetail,
    DetectionRecord,
    OperatorReviewRequest,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        model_info = pipeline_bridge.get_model_info()
        return HealthResponse(
            status="ok",
            model_loaded=model_info.status == "READY",
            version="0.1.0",
        )
    except Exception:
        return HealthResponse(status="error", model_loaded=False, version="0.1.0")


@router.get("/model", response_model=ModelInfo)
def get_model() -> ModelInfo:
    return pipeline_bridge.get_model_info()


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status() -> SystemStatusResponse:
    return pipeline_bridge.get_system_status()


@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(mission_id: Optional[str] = Query(default=None)) -> DashboardStats:
    return pipeline_bridge.get_dashboard_stats(mission_id=mission_id)


# ---------------------------------------------------------------------------
# Sonar Ingestion & Analysis
# ---------------------------------------------------------------------------

@router.post("/sonar/analyze", response_model=AnalysisResponse)
async def analyze_sonar(
    file: UploadFile = File(...),
    mission_id: str = Form(default="MISSION-001"),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None),
    depth_m: float | None = Form(default=None),
) -> AnalysisResponse:
    content_type = file.content_type or ""
    if not content_type.startswith("image/") and not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a sonar image (PNG/JPEG/TIFF).",
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
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed on server: {exc}",
        ) from exc

    return result


@router.post("/sonar/batch-analyze", response_model=BatchAnalysisResponse)
async def batch_analyze_sonar(
    files: List[UploadFile] = File(...),
    mission_id: str = Form(default="MISSION-001"),
) -> BatchAnalysisResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    files_data = []
    for f in files:
        b = await f.read()
        if b:
            files_data.append((b, f.filename or "upload.png"))

    if not files_data:
        raise HTTPException(status_code=400, detail="Uploaded files are empty.")

    return pipeline_bridge.run_batch_analysis(files_data=files_data, mission_id=mission_id)


@router.get("/analysis/{analysis_id}/image")
def get_analysis_image(analysis_id: str, variant: str = "raw") -> Response:
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


@router.get("/analysis/{analysis_id}/pdf")
@router.get("/reports/analysis/{analysis_id}/pdf")
def get_analysis_pdf_report(analysis_id: str, mission_id: str = Query(default="MISSION-001")) -> Response:
    pdf_bytes = pipeline_bridge.get_analysis_pdf(analysis_id, mission_id=mission_id)
    if pdf_bytes is None:
        raise HTTPException(
            status_code=500,
            detail="PDF report could not be compiled.",
        )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=Sonar_Report_{analysis_id}.pdf"
        },
    )


@router.get("/reports/pdf")
def get_general_pdf_report(
    analysis_id: Optional[str] = Query(default="AN-LATEST"),
    mission_id: str = Query(default="MISSION-001"),
) -> Response:
    pdf_bytes = pipeline_bridge.get_analysis_pdf(analysis_id or "AN-LATEST", mission_id=mission_id)
    if pdf_bytes is None:
        raise HTTPException(
            status_code=500,
            detail="PDF report could not be compiled.",
        )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=Sonar_Report_{analysis_id or 'latest'}.pdf"
        },
    )


# ---------------------------------------------------------------------------
# Missions API
# ---------------------------------------------------------------------------

@router.get("/missions", response_model=List[MissionSummary])
def list_missions() -> List[dict]:
    return pipeline_bridge.get_missions()


@router.post("/missions")
def create_mission_endpoint(req: MissionCreateRequest) -> dict:
    return pipeline_bridge.create_mission(
        mission_id=req.mission_id,
        name=req.name,
        area=req.area,
        operator=req.operator,
        data_label=req.data_label or "SURVEY",
    )


@router.get("/missions/{mission_id}", response_model=MissionDetail)
def get_mission_detail(mission_id: str) -> dict:
    m = pipeline_bridge.get_mission(mission_id)
    if not m:
        raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found.")
    return m


@router.get("/missions/{mission_id}/images")
def get_mission_images_endpoint(mission_id: str) -> list:
    return pipeline_bridge.get_mission_images(mission_id)


@router.get("/missions/{mission_id}/detections", response_model=List[DetectionRecord])
def get_mission_detections_endpoint(mission_id: str) -> list:
    return pipeline_bridge.get_detections(mission_id=mission_id, limit=500)


# ---------------------------------------------------------------------------
# Detections & Operator Feedback
# ---------------------------------------------------------------------------

@router.get("/detections", response_model=List[DetectionRecord])
def list_detections(
    mission_id: Optional[str] = Query(default=None),
    limit: int = Query(default=200, le=1000),
) -> list:
    return pipeline_bridge.get_detections(mission_id=mission_id, limit=limit)


@router.post("/detections/{detection_id}/review")
@router.patch("/detections/{detection_id}/review")
def review_detection(detection_id: str, req: OperatorReviewRequest) -> dict:
    ok = pipeline_bridge.update_operator_feedback(
        detection_id=detection_id,
        status=req.status,
        label=req.label,
        note=req.note,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found.")
    return {"status": "updated", "detection_id": detection_id}


# ---------------------------------------------------------------------------
# Data Exports (CSV / JSON)
# ---------------------------------------------------------------------------

@router.get("/export/csv")
@router.get("/missions/{mission_id}/export/csv")
def export_csv_endpoint(mission_id: Optional[str] = None) -> Response:
    csv_str = pipeline_bridge.export_csv(mission_id=mission_id)
    filename = f"sonar_detections_{mission_id or 'all'}.csv"
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/export/json")
@router.get("/missions/{mission_id}/export/json")
def export_json_endpoint(mission_id: Optional[str] = None) -> Response:
    json_str = pipeline_bridge.export_json(mission_id=mission_id)
    filename = f"sonar_detections_{mission_id or 'all'}.json"
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@router.get("/analysis/latest", response_model=Optional[AnalysisResponse])
def get_latest_analysis_endpoint() -> Optional[AnalysisResponse]:
    return pipeline_bridge.get_latest_analysis()

@router.get("/map/targets", response_model=MapTargetsResponse)
def get_map_targets_endpoint(
    mission_id: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    location_source: Optional[str] = Query(default=None),
) -> MapTargetsResponse:
    return pipeline_bridge.get_map_targets(
        mission_id=mission_id,
        severity=severity,
        location_source=location_source
    )
