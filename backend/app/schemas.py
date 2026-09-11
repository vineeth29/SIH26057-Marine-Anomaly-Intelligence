"""
Pydantic response/request models for the SIH26057 API layer.

These schemas are a typed view over the EXISTING pipeline outputs
(services/pipeline_service.py PipelineResult / PipelineDetection).
No field here is invented — every value is sourced from the real
pipeline, mission DB, or model config at request time.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LocationSource(str, Enum):
    """
    Honest provenance label for a detection's coordinates.
    Mirrors PipelineDetection.geo_label from the existing pipeline:
      - "Simulated Location" -> SIMULATED
      - a real GPS/telemetry tag -> REAL_GPS
      - parsed from sonar ping/metadata header -> SONAR_METADATA
      - operator-entered -> MANUAL
      - no coordinates at all -> UNAVAILABLE
    """
    REAL_GPS = "REAL_GPS"
    SONAR_METADATA = "SONAR_METADATA"
    MANUAL = "MANUAL"
    SIMULATED = "SIMULATED"
    UNAVAILABLE = "UNAVAILABLE"


class ComponentStatus(str, Enum):
    READY = "READY"
    WARNING = "WARNING"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

class ShadowDetails(BaseModel):
    """Only populated with keys the shadow analyzer actually returns."""
    model_config = {"extra": "allow"}


class FusionBreakdown(BaseModel):
    """Only populated with keys evidence fusion actually returns."""
    model_config = {"extra": "allow"}


class Detection(BaseModel):
    detection_id: str
    class_name: str
    display_name: str
    confidence: float
    confidence_pct: float
    anomaly_score: float
    shadow_score: float
    evidence_score: float
    evidence_pct: float
    severity: str
    severity_color: str
    bbox: List[float]
    bbox_width_px: int
    bbox_height_px: int
    object_area_px: int
    is_anomaly: bool
    mode: str
    model_version: str
    track_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    depth_m: Optional[float] = None
    location_source: LocationSource
    height_estimate: str
    reasons: List[str] = Field(default_factory=list)
    shadow_details: Dict[str, Any] = Field(default_factory=dict)
    fusion_breakdown: Dict[str, Any] = Field(default_factory=dict)
    status: str  # "RETAINED" | "FILTERED" — derived, never invented


# ---------------------------------------------------------------------------
# Analysis (POST /api/sonar/analyze response)
# ---------------------------------------------------------------------------

class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int
    size_bytes: int


class QualityInfo(BaseModel):
    """Passthrough of whatever the real preprocessor's quality dict contains."""
    model_config = {"extra": "allow"}


class AnomalyInfo(BaseModel):
    """Passthrough of the real anomaly_result dict."""
    model_config = {"extra": "allow"}


class AnalysisResponse(BaseModel):
    analysis_id: str
    mission_id: str
    status: str  # "completed" | "failed"
    image: ImageInfo
    detections: List[Detection]
    quality: QualityInfo
    anomaly_result: AnomalyInfo
    num_known: int
    num_anomalies: int
    mode: str  # "REAL" — DEMO mode no longer exists in this pipeline
    model_version: str
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float
    timing: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    status: str  # "READY" | "NOT_AVAILABLE"
    model_type: str
    supported_classes: List[str]
    device: str
    model_version: str
    conf_threshold: float
    iou_threshold: float
    # Only present if a real evaluation report was found on disk.
    metrics: Optional[Dict[str, float]] = None
    metrics_source: Optional[str] = None


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------

class ComponentHealth(BaseModel):
    name: str
    status: ComponentStatus
    detail: Optional[str] = None


class SystemStatusResponse(BaseModel):
    overall: ComponentStatus
    components: List[ComponentHealth]
    inference_device: str
    mean_pipeline_latency_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# Missions / dashboard
# ---------------------------------------------------------------------------

class DashboardStats(BaseModel):
    sonar_frames: int
    anomalies_detected: int
    high_priority: int
    average_confidence: Optional[float] = None
    average_processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
