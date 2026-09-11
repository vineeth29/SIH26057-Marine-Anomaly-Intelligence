"""
Pydantic response/request models for the SIH26057 API layer.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class LocationSource(str, Enum):
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
    model_config = {"extra": "allow"}


class FusionBreakdown(BaseModel):
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
    status: str


class DetectionRecord(BaseModel):
    detection_id: str
    image_id: Optional[str] = None
    mission_id: Optional[str] = None
    class_name: str
    confidence: float
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    shadow_score: float = 0.0
    evidence_score: float = 0.0
    severity: str = "UNKNOWN"
    bbox: List[int] = Field(default_factory=list)
    lat: Optional[float] = None
    lon: Optional[float] = None
    depth_m: Optional[float] = None
    mode: str = "REAL"
    model_version: str = "best"
    operator_status: Optional[str] = None
    operator_label: Optional[str] = None
    operator_note: Optional[str] = None
    created_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int
    size_bytes: int


class QualityInfo(BaseModel):
    model_config = {"extra": "allow"}


class AnomalyInfo(BaseModel):
    model_config = {"extra": "allow"}


class AnalysisResponse(BaseModel):
    analysis_id: str
    mission_id: str
    status: str
    image: ImageInfo
    detections: List[Detection]
    quality: QualityInfo
    anomaly_result: AnomalyInfo
    num_known: int
    num_anomalies: int
    mode: str
    model_version: str
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float
    timing: Dict[str, float] = Field(default_factory=dict)


class BatchAnalysisResponse(BaseModel):
    total_images: int
    successful_count: int
    failed_count: int
    results: List[AnalysisResponse]
    total_processing_time_ms: float


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    status: str
    model_type: str
    supported_classes: List[str]
    device: str
    model_version: str
    conf_threshold: float
    iou_threshold: float
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
# Missions / dashboard / reviews
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


class MissionCreateRequest(BaseModel):
    mission_id: str
    name: str
    area: Optional[str] = "Survey Grid Alpha"
    operator: Optional[str] = "Sonar Operator"
    data_label: Optional[str] = "SURVEY"


class MissionSummary(BaseModel):
    mission_id: str
    name: str
    date: Optional[str] = None
    area: Optional[str] = None
    status: Optional[str] = None
    data_label: Optional[str] = None
    image_count: int = 0
    detection_count: int = 0
    anomaly_count: int = 0


class MissionDetail(BaseModel):
    mission_id: str
    name: str
    date: Optional[str] = None
    area: Optional[str] = None
    status: Optional[str] = None
    data_label: Optional[str] = None
    image_count: int = 0
    detection_count: int = 0
    anomaly_count: int = 0
    high_risk_count: int = 0


class OperatorReviewRequest(BaseModel):
    status: str
    label: Optional[str] = None
    note: Optional[str] = None

class MapTarget(BaseModel):
    detection_id: str
    mission_id: str
    image_id: Optional[str] = None
    class_name: str
    display_name: str
    latitude: float
    longitude: float
    location_source: LocationSource
    confidence: float
    evidence_score: Optional[float] = None
    severity: str
    timestamp: Optional[str] = None
    review_status: Optional[str] = None
    depth_m: Optional[float] = None
    heading_deg: Optional[float] = None


class MapTargetsResponse(BaseModel):
    targets: List[MapTarget]
    total_geolocated: int
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int
