// Mirrors backend/app/schemas.py. Keep these in sync manually for now —
// if the API grows, consider generating this from the OpenAPI schema.

export type LocationSource =
  | "REAL_GPS"
  | "SONAR_METADATA"
  | "MANUAL"
  | "SIMULATED"
  | "UNAVAILABLE";

export type ComponentStatusValue = "READY" | "WARNING" | "ERROR" | "UNAVAILABLE";

export type Severity = "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";

export type DetectionStatus = "RETAINED" | "FILTERED";

export interface Detection {
  detection_id: string;
  class_name: string;
  display_name: string;
  confidence: number;
  confidence_pct: number;
  anomaly_score: number;
  shadow_score: number;
  evidence_score: number;
  evidence_pct: number;
  severity: Severity;
  severity_color: string;
  bbox: number[]; // [x1, y1, x2, y2]
  bbox_width_px: number;
  bbox_height_px: number;
  object_area_px: number;
  is_anomaly: boolean;
  mode: string;
  model_version: string;
  track_id: string | null;
  lat: number | null;
  lon: number | null;
  depth_m: number | null;
  location_source: LocationSource;
  height_estimate: string;
  reasons: string[];
  shadow_details: Record<string, unknown>;
  fusion_breakdown: Record<string, unknown>;
  status: DetectionStatus;
}

export interface ImageInfo {
  filename: string;
  width: number;
  height: number;
  size_bytes: number;
}

export interface AnalysisResponse {
  analysis_id: string;
  mission_id: string;
  status: "completed" | "failed";
  image: ImageInfo;
  detections: Detection[];
  quality: Record<string, unknown>;
  anomaly_result: Record<string, unknown>;
  num_known: number;
  num_anomalies: number;
  mode: string;
  model_version: string;
  warnings: string[];
  processing_time_ms: number;
  timing: Record<string, number>;
}

export interface ModelInfo {
  status: "READY" | "NOT_AVAILABLE";
  model_type: string;
  supported_classes: string[];
  device: string;
  model_version: string;
  conf_threshold: number;
  iou_threshold: number;
  metrics: Record<string, number> | null;
  metrics_source: string | null;
}

export interface ComponentHealth {
  name: string;
  status: ComponentStatusValue;
  detail: string | null;
}

export interface SystemStatusResponse {
  overall: ComponentStatusValue;
  components: ComponentHealth[];
  inference_device: string;
  mean_pipeline_latency_ms: number | null;
}

export interface DashboardStats {
  sonar_frames: number;
  anomalies_detected: number;
  high_priority: number;
  average_confidence: number | null;
  average_processing_time_ms: number | null;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  version: string;
}
