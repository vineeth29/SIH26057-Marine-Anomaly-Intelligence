# REST API Reference

The backend exposes a REST API via FastAPI at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is accessible at `/docs`.

---

## Endpoints

### 1. System Status
`GET /api/system/status`

**Response:**
```json
{
  "overall": "OPERATIONAL",
  "yolo_model": "LOADED",
  "yolo_classes": 5,
  "anomaly_model": "LOADED",
  "database": "CONNECTED",
  "inference_device": "cpu",
  "mean_pipeline_latency_ms": 54.0
}
```

---

### 2. Dashboard Statistics
`GET /api/dashboard/stats`

**Response:**
```json
{
  "sonar_frames": 48,
  "anomalies_detected": 12,
  "high_priority": 3,
  "medium_priority": 5,
  "low_priority": 4,
  "average_confidence": 84.6
}
```

---

### 3. Pipeline Analyze (File Upload)
`POST /api/pipeline/analyze`

**Form Data:**
- `file`: Image binary (PNG, JPG, BMP)
- `mission_id` (optional): string
- `latitude` (optional): float
- `longitude` (optional): float

**Response:**
```json
{
  "analysis_id": "AN-20260911-8842",
  "detections": [
    {
      "detection_id": "DET-001",
      "class_id": 4,
      "class_name": "mine_cylinder",
      "display_name": "Mine Cylinder",
      "confidence": 0.88,
      "evidence_score": 0.84,
      "severity": "HIGH",
      "severity_color": "#EF4444",
      "bbox": [120, 85, 240, 190],
      "shadow_score": 0.82,
      "anomaly_score": 0.79,
      "location_source": "GPS",
      "latitude": 12.9234,
      "longitude": 80.1256
    }
  ],
  "quality_score": 87.5,
  "processing_time_ms": 52.4
}
```

---

### 4. PDF Survey Report Export
`GET /api/reports/analysis/{analysis_id}/pdf` or `GET /api/reports/pdf?mission_id={mission_id}`

**Response:** `application/pdf` binary stream.

---

### 5. CSV / JSON Telemetry Export
- `GET /api/export/csv`: Returns RFC 4180 formatted CSV table.
- `GET /api/export/json`: Returns JSON array of all detections and missions.
