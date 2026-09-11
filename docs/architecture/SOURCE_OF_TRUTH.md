# Source of Truth: Canonical Architecture & Runtime Specification

This document defines the single canonical source of truth for the SIH26057 platform to prevent divergence across workspace iterations.

---

## Canonical File Locations

| Subsystem | Canonical Path | Description |
|---|---|---|
| **Root Directory** | `.` (Repository Root) | Top-level consolidated repository |
| **Active Backend** | `backend/app/api.py` | FastAPI application serving all REST endpoints |
| **Pipeline Bridge** | `backend/app/pipeline_bridge.py` | Connects raw request to multi-stage AI pipeline |
| **Active Frontend** | `frontend/src/` | Vite + React + TypeScript operator interface |
| **Production Model** | `models/best.pt` | 5-class YOLOv8n detector checkpoint (22.5 MB) |
| **Anomaly Model** | `models/anomaly/autoencoder.pt` | PyTorch Conv2D Autoencoder checkpoint (1.25 MB) |
| **Evidence Fusion** | `ai/fusion/confidence_fusion.py` | Multi-signal fusion & severity scoring engine |
| **Shadow Analyzer** | `ai/shadow_analysis/shadow_analyzer.py`| Acoustic highlight-shadow geometric analyzer |
| **Dropout Detector** | `ai/quality/dropout_detector.py` | Acoustic signal loss and dropout checker |
| **FP Filter** | `ai/fp_filter/false_positive_filter.py` | Geometric and contrast false-positive filter |
| **Database** | `database/models.py` (`sih26057.db`) | SQLite database storing missions, detections & telemetry |
| **PDF Engine** | `utils/pdf_report.py` | ReportLab PDF report generation engine |

---

## Runtime Network Topology

| Component | Default Port | Protocol | Base URL |
|---|---|---|---|
| **FastAPI Backend** | `8000` | HTTP / JSON | `http://127.0.0.1:8000` |
| **React Frontend** | `5173` | HTTP / Web | `http://127.0.0.1:5173` |

### Startup Commands

#### Backend:
```bash
python -m uvicorn backend.app.api:app --host 127.0.0.1 --port 8000
```

#### Frontend:
```bash
npm --prefix frontend run dev
```

---

## Verification Criteria
Every valid deployment must satisfy:
1. `python -m pytest` executes 101+ tests with 0 failures.
2. `npm --prefix frontend run build` completes with 0 errors.
3. `GET /api/system/status` returns status `OPERATIONAL` or `READY`.
4. `POST /api/pipeline/analyze` successfully runs YOLO, Shadow Analyzer, Autoencoder, and Evidence Fusion.
