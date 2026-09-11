# SIH26057 — Canonical Source of Truth

**Project Name:** AI-Powered Automated Underwater Marine Debris and Target Detection System using Side-Scan Sonar (SSS)  
**Problem Statement ID:** SIH26057  
**Consolidation Date:** 2026-09-11  

---

## 1. Canonical Repository Location

This repository is the single consolidated **Source of Truth** for the SIH26057 project:

```
c:\Users\vinee\Downloads\sih26057-standalone (1)\sih26057-standalone
```

All legacy implementations, research scripts, evaluation metrics, and documentation from `SIH-2026-Marine-Debris-Detection-main` have been selectively audited, verified, and consolidated here.

---

## 2. Active System Architecture

| Subsystem | Canonical Implementation Path | Technology Stack | Port |
|---|---|---|---|
| **Frontend UI** | `frontend/` | React 18, Vite, TypeScript, TailwindCSS, Leaflet GIS | `http://localhost:5173` |
| **Backend REST API** | `backend/app/` | FastAPI, Pydantic v2, Uvicorn | `http://localhost:8000` |
| **Detection Engine** | `ai/detection/yolo_detector.py` | PyTorch / Ultralytics YOLOv8s | Embedded |
| **Anomaly Detector** | `ai/anomaly/anomaly_detector.py` | Deep Autoencoder (PyTorch) | Embedded |
| **Shadow Analyzer** | `ai/shadow_analysis/shadow_analyzer.py` | OpenCV Physics-informed Shadow Tracking | Embedded |
| **Evidence Fusion** | `ai/fusion/confidence_fusion.py` | Multi-Factor Composite Fusion Engine | Embedded |
| **FP Filter** | `ai/fp_filter/false_positive_filter.py` | Geometry, Texture & Aspect Ratio Filter | Embedded |
| **Quality Screening** | `ai/quality/dropout_detector.py`, `utils/quality.py` | Laplacian variance, Column Dropout | Embedded |
| **Database** | `database/models.py`, `sih26057.db` | SQLite / SQLAlchemy | `sih26057.db` |
| **PDF Reporting** | `utils/pdf_report.py` | ReportLab | Export Service |

---

## 3. Active Production Models

### Primary Object Detection Model
- **File Path:** `models/best.pt`
- **Architecture:** YOLOv8s (22,520,746 bytes)
- **SHA256:** `898ba4c1cafa23b9f55d18b3bfdbe615e26264a38d109fb391c303c16aa57314`
- **Validation Metric:** mAP50 = 0.712 (Input size: 640×640)
- **Target Classes (5 Classes):**
  1. `crab_pot` (Debris / Hazard)
  2. `submarine_pipeline` (Infrastructure)
  3. `shipwreck` (Maritime Hazard / Cultural)
  4. `ghost_net` (Abandoned Fishing Gear / High Threat)
  5. `mine_cylinder` (Submerged Ordnance / High Threat)

### Seabed Regional Anomaly Model
- **File Path:** `models/anomaly/autoencoder.pt`
- **Architecture:** Deep Autoencoder (1,257,419 bytes)
- **SHA256:** `7952578229a013ed0cd7ad1b800dfa729fa2a7e2dd8380691f91366c313cbde5`
- **Status:** Validated baseline reconstruction model (`_ANOMALY_MODEL_STATUS = "VALIDATED"`)

---

## 4. Multi-Modal Evidence Fusion & Severity Classification

Evidence score is synthesized using configured component weights:
- **Detector Confidence ($W_d = 0.45$)**
- **Acoustic Shadow Presence & Length ($W_s = 0.20$)**
- **Texture Consistency ($W_t = 0.15$)**
- **Shape / Geometry Profile ($W_g = 0.10$)**
- **Regional Anomaly Residual ($W_a = 0.10$)**

### Severity Rating
- **HIGH SEVERITY (Red `#FF3B30`):** Target in `HIGH_RISK_CLASSES` (`ghost_net`, `mine_cylinder`, `shipwreck`, `submarine_pipeline`, `container`) with Evidence $\ge 0.55$, or any target with Evidence $\ge 0.70$.
- **MEDIUM SEVERITY (Amber `#FF9500`):** Targets with Evidence $\ge 0.40$ or medium-risk classes (`metal_debris`, `tire`, `plastic_debris`).
- **LOW SEVERITY (Green `#34C759`):** Natural formations, low-threat debris (`bottle_object`, `rock`), or Evidence $< 0.40$.

---

## 5. System Execution & Startup

### Start Backend Service
```powershell
# From repository root
uvicorn backend.app.main:app --reload --port 8000
```

### Start Frontend Application
```powershell
cd frontend
npm run dev
```

### Run Test Suite
```powershell
python -m pytest tests/ -v
```
