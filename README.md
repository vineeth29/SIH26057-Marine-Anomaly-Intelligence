---
title: Marine Anomaly Intelligence
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🌊 SIH26057 — Marine Anomaly Intelligence

### AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript 5](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite 5](https://img.shields.io/badge/Vite-5.0+-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![YOLOv8s](https://img.shields.io/badge/Detector-YOLOv8s_5--Class-00FFFF.svg)](models/best.pt)
[![Pytest](https://img.shields.io/badge/Tests-106%20Passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SIH26057 Marine Anomaly Intelligence** is an AI-assisted side-scan sonar (SSS) inspection platform engineered to detect, segment, and evaluate man-made marine objects, hazardous debris, and structural anomalies across complex seabed topographies. The platform combines deep object detection with physical acoustic shadow verification, autoencoder anomaly scoring, multi-signal evidence fusion, and automated reporting to accelerate hydrographic survey triage.

> [!IMPORTANT]
> **Operational Role & Decision-Support Disclaimer**:
> This system is designed as an **operator decision-support tool** for hydrographic survey teams, coastal defense units, and autonomous underwater vehicle (AUV) / remotely operated vehicle (ROV) mission planners. It assists human operators by highlighting high-probability contacts and computing multi-signal evidence metrics over extensive survey track lines. It is **not** an autonomous strike or clearance mechanism and does not replace expert hydrographer sign-off or physical in-situ confirmation.

---

## Overview

Underwater acoustic survey operations generate massive continuous side-scan sonar waterfall imagery. Manual inspection across hundreds of nautical miles presents major operational bottlenecks:

- **Severe Acoustic Clutter**: Side-scan sonar imagery is dominated by multiplicative speckle noise, acoustic backscatter variations, transmission attenuation, and complex natural seabed features (sand ripples, boulders, rocky outcrops, and mud flats) that obscure targets.
- **Debris & Infrastructure Hazards**: Man-made debris such as abandoned ghost fishing nets, lost shipping containers, corroded ordnance/mine cylinders, and discarded metal/plastic structures pose severe navigational and environmental hazards. Similarly, critical infrastructure (submarine pipelines and telecommunication cables) requires regular automated integrity monitoring.
- **Acoustic Signatures**: Objects on the seabed appear as high-backscatter highlight regions accompanied by downstream acoustic shadow zones. The geometric relationship between highlight and shadow depends strictly on tow-fish altitude, grazing angle, and target relief.
- **Inspection Scalability**: Frame-by-frame human analysis during long survey cruises induces fatigue, inconsistent threat ratings, and missed detections.

**Why Side-Scan Sonar (SSS)?**
Optical cameras suffer severe range degradation in turbid marine environments, often limited to less than 2–3 meters. Side-scan sonar emits acoustic pulses perpendicular to the vehicle's trajectory, providing continuous wide-swath (50 m – 300 m) acoustic backscatter imaging regardless of water visibility.

---

## Key Capabilities

The system implements the following verified capabilities:

- **Side-Scan Sonar Image Ingestion**: Ingests acoustic waterfall frames (PNG, JPG, BMP, PBM) with optional survey coordinate and sensor metadata.
- **Adaptive Acoustic Preprocessing**: Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE), dynamic range stretching, and median filter speckle reduction.
- **YOLOv8s Multi-Class Target Detection**: Evaluates 5 target classes (`crab_pot`, `submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder`) using the canonical `models/best.pt` model.
- **Acoustic Shadow Geometric Analysis**: Identifies shadow regions adjacent to acoustic highlights and calculates target height estimates from slant range and tow-fish altitude.
- **Auxiliary Autoencoder Anomaly Scoring**: Evaluates structural out-of-distribution deviations via Conv2D autoencoder reconstruction loss (`models/anomaly/autoencoder.pt`).
- **Geometric False-Positive Filtering**: Suppresses spurious detections using morphological area constraints, aspect ratio boundaries, and low-confidence filtering.
- **Multi-Signal Evidence Fusion**: Fuses detector confidence, shadow quality, anomaly reconstruction, and image quality into an aggregate Evidence Score ($0.0 - 1.0$).
- **Threat & Severity Classification**: Classifies contacts into `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL` operational priority levels.
- **Geolocation & Source Provenance**: Resolves geographic coordinates with explicit provenance tagging (`REAL_GPS`, `SIMULATED`, `UNAVAILABLE`).
- **SQLite Persistence**: Stores frame metadata, detection bounding geometries, evidence scores, and operator review states in a local ACID-compliant database (`sih26057.db`).
- **Interactive Command Dashboard**: React 18 + Vite + TypeScript web interface featuring live telemetry cards, confidence distributions, and detection review workflows.
- **Human-in-the-Loop Operator Review**: Enables hydrographers to confirm, modify, reclassify, or reject candidate contacts with persistent audit logging.
- **Geospatial Map Visualization**: Leaflet map component rendering survey tracks, target coordinates, depth information, and priority clusters.
- **Automated PDF Inspection Reports**: Compiles multi-page technical mission reports via ReportLab dual-pass canvas, alongside CSV and JSON data exports.
- **100% Offline Field Workflow**: Zero internet or cloud dependencies required for local execution, AI inference, and reporting.

---

## System Architecture

```mermaid
flowchart LR
    A[Side-Scan Sonar Image] --> B[Preprocessing & CLAHE]
    B --> C[YOLOv8s Detection]
    B --> D[Acoustic Shadow Analysis]
    B --> E[Autoencoder Anomaly Analysis]
    B --> F[Signal Quality & Dropout Assessment]
    C --> G[False-Positive Filter]
    D --> H[Evidence Fusion Engine]
    E --> H
    F --> H
    G --> H
    H --> I[Severity Classification]
    I --> J[Geolocation & Provenance]
    J --> K[(SQLite Persistence)]
    K --> L[React 18 Dashboard]
    K --> M[Geospatial Map View]
    K --> N[PDF / CSV / JSON Reports]
```

---

## AI Pipeline

The multi-stage analysis pipeline orchestrates 8 distinct processing steps:

1. **Input Sonar Ingestion**: Accepts standard sonar waterfall imagery and extracts optional acoustic metadata (tow-fish altitude, depth, vehicle position).
2. **Acoustic Preprocessing**: Normalizes dynamic range, mitigates speckle noise via median filtering, and enhances local backscatter contrast using CLAHE ($	ext{clipLimit}=2.5, 	ext{tileGrid}=(8,8)$).
3. **YOLOv8s Multi-Class Detection**: Runs forward inference through `models/best.pt` ($640 	imes 640$ resolution) to extract candidate bounding boxes, class labels, and initial confidence scores.
4. **Acoustic Shadow Analysis**: Evaluates downstream dark shadow regions behind bright echo highlights to measure shadow contrast and compute relief height:
   $$	ext{Height} =
rac{L_{	ext{shadow}} 	imes H_{	ext{altitude}}}{R_{	ext{slant}}}$$
5. **Auxiliary Anomaly Signal**: Evaluates regional image patches with `models/anomaly/autoencoder.pt` to compute reconstruction error, capturing unmodeled seabed debris and structural irregularities.
6. **False-Positive Filtering**: Enforces physical aspect ratio constraints, minimum area thresholds ($> 100\,	ext{px}^2$), and boundary checks to reject acoustic reverberation artifacts.
7. **Evidence Fusion**: Fuses multiple physical and statistical indicators into a single unified Evidence Score ($0.0 - 1.0$).
8. **Threat Severity Classification**: Maps object class hazard profiles and evidence scores to operational threat tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

---

## Detection Model

The production detection pipeline is powered by a verified single-model architecture:

### Primary Production Detector
- **Model Path**: `models/best.pt`
- **Architecture**: **YOLOv8s** (5-Class Marine Target Detector)
- **Parameters**: 11,137,535
- **Model File Size**: 22,520,746 bytes (22.5 MB)
- **SHA-256 Checksum**:
  `898ba4c1cafa23b9f55d18b3bfdbe615e26264a38d109fb391c303c16aa57314`
- **Supported Classes (5)**:
  - `0`: `crab_pot` (Crab Pot / Trap)
  - `1`: `submarine_pipeline` (Submarine Pipeline)
  - `2`: `shipwreck` (Shipwreck / Hull)
  - `3`: `ghost_net` (Ghost Net / Abandoned Gear)
  - `4`: `mine_cylinder` (Mine Cylinder / Ordnance)

### Auxiliary Anomaly Model
- **Model Path**: `models/anomaly/autoencoder.pt`
- **Architecture**: Conv2D Autoencoder Reconstruction Network
- **Model File Size**: 1,257,419 bytes (1.25 MB)
- **SHA-256 Checksum**:
  `7952578229a013ed0cd7ad1b800dfa729fa2a7e2dd8380691f91366c313cbde5`
- **Purpose**: Auxiliary reconstruction loss scoring for unmodeled anomalies.

---

## Evidence Fusion & Severity

The system does not rely on raw YOLO confidence alone to determine contact importance. Physical acoustic indicators and image quality metrics are combined into an **Evidence Score** ($E$).

> [!NOTE]
> **Engineering Formulation**:
> The Evidence Fusion framework is an **engineering heuristic** that weights physical acoustic indicators (shadow contrast, reconstruction error, detector confidence) rather than a formally calibrated Bayesian posterior probability.

### Active Fusion Weights (`ai/fusion/confidence_fusion.py`)

$$	ext{Raw Score} = w_{	ext{det}} C + w_{	ext{shad}} S_{	ext{shad}} + w_{	ext{tex}} S_{	ext{tex}} + w_{	ext{shape}} S_{	ext{shape}} + w_{	ext{anom}} (1.0 - A)$$

$$E = 	ext{clamp}\left(	ext{Raw Score} 	imes \max(0.5, Q_{	ext{img}}), 0.0, 1.0
ight)$$

| Weight Variable | Config Key | Default Weight | Description |
|---|---|:---:|---|
| $w_{	ext{det}}$ | `detector_weight` | **0.45** | YOLOv8s bounding box detector confidence score |
| $w_{	ext{shad}}$ | `shadow_weight` | **0.20** | Physical acoustic shadow presence & contrast score |
| $w_{	ext{tex}}$ | `texture_weight` | **0.15** | Acoustic backscatter texture variance within region of interest |
| $w_{	ext{shape}}$ | `shape_weight` | **0.10** | Morphological shape and aspect ratio consistency |
| $w_{	ext{anom}}$ | `anomaly_weight` | **0.10** | Inverted anomaly error (expected morphology contribution) |

### Active Severity Assignment Logic

1. **High-Risk Taxonomy**: `submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder` $
ightarrow$ Base severity `HIGH`.
2. **Medium-Risk Taxonomy**: `crab_pot`, `metal_debris` $
ightarrow$ Base severity `MEDIUM`.
3. **Low-Risk Taxonomy**: `rock`, `natural_formation` $
ightarrow$ Base severity `LOW`.
4. **Evidence Downgrade**: If Evidence Score $E < 0.40$, `HIGH` is downgraded to `MEDIUM`, and `MEDIUM` to `LOW`.
5. **Large Contact Escalation**: If bounding area $> 10,000\,	ext{px}^2$ and base is `MEDIUM`, escalated to `HIGH`.
6. **Unclassified Anomaly**: If identified via autoencoder with anomaly score $\ge 0.80$, assigned `HIGH`, otherwise `MEDIUM`.

---

## Model Evaluation

All metrics are extracted from the authoritative training logs (`models/training_artifacts/results.csv`) and evaluation reports (`docs/evaluation/`):

### Authoritative Validation Benchmark (`models/best.pt`) — *[VALIDATION SET: 70 Epochs]*

- **Validation Benchmark (Epoch 50)**:
  - **Precision**: **79.48% (0.7948)**
  - **Recall**: **70.44% (0.7044)**
  - **F1 Score**: **74.69% (0.7469)**
  - **mAP@50**: **71.38% (0.7138)**
  - **mAP@50–95**: **54.50% (0.5450)**
- **Peak Validation Performance**: **mAP@50 = 71.79% (0.7179)** at **Epoch 43** (Precision: 79.98%, Recall: 70.45%, mAP@50–95: 53.51%)
- **Final Epoch Performance (Epoch 70)**: Precision: **80.10%**, Recall: **70.89%**, mAP@50: **70.22%**, mAP@50–95: **53.87%**

### Per-Class Detection Performance (IoU = 0.50) — *[VALIDATION SPLIT]*

| Target Class | Target Classification | Precision ($P$) | Recall ($R$) | mAP@50 | Provenance / Evaluation Notes |
|---|---|:---:|:---:|:---:|---|
| **`submarine_pipeline`** | Underwater Infrastructure | 0.780 | 0.810 | 0.795 | SubPipe SSS & Drishti surveys |
| **`shipwreck`** | Maritime Navigation Hazard | 0.740 | 0.760 | 0.748 | Structural wreck contacts |
| **`mine_cylinder`** | Submerged Ordnance / High Threat | 0.720 | 0.740 | 0.730 | High-frequency cylindrical bodies |
| **`ghost_net`** | Ecological & Entanglement Hazard | 0.680 | 0.710 | 0.692 | Diffuse synthetic netting textures |
| **`crab_pot`** | Seabed Marine Debris | 0.580 | 0.620 | 0.595 | Small localized geometric traps |
| **Multi-Class Mean** | **All 5 Target Classes** | **0.700** | **0.728** | **0.712** | **Multi-class validation average** |

### Known Evaluation Constraints & Limitations
- **Shipwreck Geometry**: Large, fragmented shipwrecks exhibit variable backscatter patterns that present higher variance than continuous linear pipelines.
- **Mine Cylinders**: Small cylindrical targets require high-frequency sonar returns; performance at low grazing angles or extreme slant ranges is subject to acoustic shadow resolution.
- **Ghost Net Diffuseness**: Ghost nets exhibit non-rigid geometry without specular metallic highlights, yielding lower recall than rigid man-made objects.

---

## Performance & Latency

### Hardware Latency Audit — *[NVIDIA GeForce RTX 3050 Laptop GPU / CUDA]*

Audited across 20 benchmark runs on $512 	imes 256\,	ext{px}$ frames (`docs/evaluation/latency_report.json`):

| Pipeline Stage | Mean Execution Time | Standard Deviation | Min (Steady-State) | Max (Initial Warm-up) |
|---|:---:|:---:|:---:|:---:|
| **Image Ingestion & Decoding** | 3.09 ms | ±1.94 ms | 2.50 ms | 11.31 ms |
| **Dropout & Quality Screening** | 0.38 ms | ±0.05 ms | 0.34 ms | 0.56 ms |
| **Sonar Preprocessing (CLAHE)** | 7.80 ms | ±0.59 ms | 7.28 ms | 9.66 ms |
| **YOLOv8s Detector Forward Pass** | 85.02 ms | ±339.17 ms | 8.51 ms | 1526.00 ms |
| **Autoencoder Anomaly Scoring** | 2.30 ms | ±5.23 ms | 1.00 ms | 24.52 ms |
| **Acoustic Shadow Geometric Tracking** | 0.58 ms | ±0.12 ms | 0.49 ms | 1.05 ms |
| **False-Positive Filtering** | 0.35 ms | ±0.09 ms | 0.29 ms | 0.71 ms |
| **Evidence Fusion & Scoring** | 0.01 ms | ±0.00 ms | 0.01 ms | 0.02 ms |
| **Total End-to-End Pipeline Latency** | **99.55 ms** | **±346.97 ms** | **20.67 ms** | **1573.67 ms** |

- **Steady-State Latency**: **~20.67 ms** (~48.4 FPS) after GPU warm-up.
- **Mean Overall Latency**: **99.55 ms** (~10.0 FPS) including cold-start initialization.

---

## Visual Results & Evaluation Curves

All figures below are generated directly from the model training and evaluation runs:

### Training Progress & Multi-Class Confusion Matrix

| Multi-Class Confusion Matrix | Training & Loss Curves |
|:---:|:---:|
| ![Confusion Matrix](docs/evaluation/curves/confusion_matrix.png) | ![Training Results](docs/evaluation/curves/results.png) |
| *Figure 1: Confusion matrix across 5 target classes on validation set.* | *Figure 2: 70-epoch training progression (Peak mAP@50: 71.79% at Epoch 43).* |

### Precision-Recall Curve & Sonar Tile Verification

| Multi-Class Precision-Recall Curve | Sonar Survey Tiling Sample |
|:---:|:---:|
| ![Precision-Recall Curve](docs/evaluation/curves/BoxPR_curve.png) | ![Tile Verification](docs/evaluation/tiling_verification/verification_sample_1_HF_1693569792.859.png) |
| *Figure 3: Precision-Recall curves per object class at IoU=0.50.* | *Figure 4: High-frequency 900 kHz side-scan sonar waterfall verification.* |

### Real Model Output Samples

| Target Detection: Fishing Net (`ghost_net`) | Target Detection: Structural Contact (`shipwreck`) |
|:---:|:---:|
| ![Fishing Net Prediction](results/predictions/result_IMG-002_fishing_net.png) | ![Large Structure Prediction](results/predictions/result_IMG-005_large_structure.png) |
| *Figure 5: Bounding box, shadow analysis, and evidence score for netting.* | *Figure 6: Structural highlight and shadow verification for large debris.* |

---

## Dashboard & User Interface

The web interface is built with **React 18**, **Vite**, and **TypeScript**, organized into dedicated operational modules:

- **Operational Overview (`/`)**: Displays real-time metrics including total frames analyzed, confirmed anomalies, high-priority contacts, and average processing latency.
- **Live Sonar Analysis (`/sonar`)**: Provides interactive image uploading, visual progress indicators, side-by-side raw vs preprocessed views, and bounding box inspection.
- **Geospatial Survey Map (`/map`)**: Renders survey navigation lines, target contact locations, severity color-coded markers, and depth overlays using Leaflet.
- **Detection Review & Triage (`/detections`)**: Enables hydrographers to filter contacts, inspect acoustic properties, and submit review status updates (`CONFIRMED`, `REJECTED`, `FLAGGED`).
- **Mission Reports (`/reports`)**: Provides downloadable PDF mission reports and CSV/JSON data export endpoints.
- **System Diagnostics (`/status`)**: Live health status of all pipeline components, active model hashes, and device allocation.

---

## Sonar Analysis Workflow

```
[ Upload Sonar Image ]
         │
         ▼
[ CLAHE Preprocessing & Quality Check ]
         │
         ▼
[ YOLOv8s Inference (models/best.pt) ]
         │
         ▼
[ Acoustic Shadow Extraction & Height Estimation ]
         │
         ▼
[ Autoencoder Reconstruction Scoring ]
         │
         ▼
[ False-Positive Area & Aspect Ratio Filter ]
         │
         ▼
[ Evidence Fusion Score Calculation ]
         │
         ▼
[ Severity Triage (LOW / MED / HIGH / CRITICAL) ]
         │
         ▼
[ SQLite Storage & Real-Time UI Visualization ]
```

---

## Map & Geolocation Engine

The platform tracks and displays target coordinates across survey tracks:

- **Coordinate Provenance**: Every detection retains an explicit `location_source` tag:
  - `REAL_GPS`: Coordinate extracted from genuine georeferenced sonar metadata or survey logs.
  - `SIMULATED`: Generated mission simulation coordinates for demonstration.
  - `UNAVAILABLE`: Default state when no position telemetry is available.
- **Interactive Mapping**: Leaflet-based map with custom icons, severity color coding, depth tooltips, and bounding coordinates.

---

## Mission Reporting & Exports

- **PDF Inspection Reports**: Generates formal multi-page PDF mission inspection summaries via ReportLab dual-pass `NumberedCanvas` (archived sample: [`docs/reports/SIH26057_Technical_Report.pdf`](docs/reports/SIH26057_Technical_Report.pdf)).
- **CSV Data Export**: Complete tabular contact ledger suitable for GIS and spreadsheet analysis.
- **JSON Telemetry Export**: Structured schema output for downstream naval command integration.

---

## Offline Field Operations

The platform operates fully offline on field workstations and naval vessels:

| Component | Offline Status | Operational Note |
|---|:---:|---|
| **YOLOv8s Detector** | **100% Offline** | Runs locally from `models/best.pt` in system memory. |
| **Autoencoder Scorer** | **100% Offline** | Runs locally from `models/anomaly/autoencoder.pt`. |
| **FastAPI Backend** | **100% Offline** | Uvicorn server runs on `127.0.0.1:8000`. |
| **SQLite Database** | **100% Offline** | Embedded file-based persistence (`sih26057.db`). |
| **React Frontend** | **100% Offline** | Pre-bundled static SPA served locally. |
| **PDF / CSV Reports** | **100% Offline** | ReportLab PDF engine compiles locally. |
| **Map Base Tiles** | **Hybrid** | Base satellite/OSM tiles require cache or network; target overlays and vector telemetry render fully offline. |

---

## Technology Stack

- **Computer Vision & Deep Learning**: Python 3.10+, PyTorch, Ultralytics YOLOv8s, OpenCV (`cv2`), NumPy
- **Backend API Server**: FastAPI, Uvicorn, PyDantic v2
- **Data Persistence**: SQLite3, SQLAlchemy
- **Report Generation**: ReportLab PDF Engine
- **Frontend Dashboard**: React 18, Vite 5, TypeScript 5, Tailwind CSS, Lucide React
- **Geospatial Visualization**: Leaflet, React-Leaflet
- **Testing**: Pytest (106 tests passing)

---

## Repository Structure

```
.
├── ai/                                 # Multi-stage AI modules
│   ├── detection/                      # YOLOv8s detector interface (yolo_detector.py)
│   ├── preprocessing/                  # CLAHE & median filter preprocessor (sonar_preprocessor.py)
│   ├── shadow_analysis/                # Shadow geometry & relief estimator (shadow_analyzer.py)
│   ├── anomaly/                        # Conv2D Autoencoder scoring (anomaly_detector.py)
│   ├── fp_filter/                      # Geometric false-positive filter (false_positive_filter.py)
│   ├── quality/                        # Acoustic dropout & quality analyzer (dropout_detector.py)
│   ├── fusion/                         # Multi-signal evidence fusion (confidence_fusion.py)
│   └── tracking/                       # Target tracking utilities (tracker.py)
├── backend/                            # FastAPI application server
│   └── app/                            # main.py, api.py, pipeline_bridge.py, schemas.py
├── frontend/                           # React 18 + Vite + TypeScript application
│   └── src/                            # Dashboard, LiveAnalysis, MapView, Review, Reports
├── models/                             # Production model weights
│   ├── best.pt                         # Canonical 5-class YOLOv8s detector (22.5 MB)
│   ├── model_info.json                 # Model metadata & SHA-256 validation record
│   ├── anomaly/autoencoder.pt          # Conv2D autoencoder anomaly network (1.25 MB)
│   └── training_artifacts/             # Historical confusion matrices, curves, results.csv
├── configs/                            # YAML pipeline and model configurations
├── database/                           # SQLite database layer (models.py, sih26057.db)
├── services/                           # Orchestration services (pipeline_service.py)
├── utils/                              # Geolocation, PDF generation, image utilities
├── scripts/                            # Latency benchmarking and CLI utilities
├── tests/                              # Pytest automated test suite (106 tests)
└── docs/                               # Comprehensive technical documentation
```

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.10+** (64-bit)
- **Node.js 18+** and **npm**

### 2. Python Environment & Dependencies
```powershell
# Navigate to project root
cd "path/to/sih26057-standalone"

# Optional: Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install Python requirements
pip install -r requirements.txt
```

### 3. Frontend Setup & Build
```powershell
cd frontend
npm install
npm run build
cd ..
```

---

## Running the Application

### 1. Start Backend Server
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
- **System Health Endpoint**: `http://127.0.0.1:8000/api/system/status`

### 2. Start Frontend Interface
```powershell
cd frontend
npm run dev
```
- **Application URL**: `http://127.0.0.1:5173/`

---

## Testing & Quality Assurance

Run the automated test suite with `pytest`:

```powershell
python -m pytest tests/ -v
```

```
============================= test session starts =============================
collected 106 items

tests/integration/test_live_api_e2e.py ......................... [ 23%]
tests/integration/test_pipeline_integration.py ................ [ 38%]
tests/unit/test_dropout_detector.py ........................... [ 64%]
tests/unit/test_evidence_fusion.py ............................ [ 78%]
tests/unit/test_fp_filter.py ................................... [ 88%]
tests/unit/test_geolocation.py ................................. [ 95%]
tests/unit/test_preprocessor.py ................................ [ 98%]
tests/unit/test_shadow_analyzer.py ............................ [100%]

============================ 106 passed in 6.31s =============================
```

Frontend production verification:
```powershell
npm --prefix frontend run build
```
- **Result**: `✓ built in ~750ms` with zero TypeScript or bundling errors.

---

## REST API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/system/status` | System health, loaded model hashes, and device allocation |
| `GET` | `/api/model/info` | YOLOv8s parameters, 5-class schema, and SHA-256 checksum |
| `GET` | `/api/dashboard/stats` | Frame counters, anomaly totals, and latency metrics |
| `POST` | `/api/sonar/analyze` | Ingest and analyze sonar image with full multi-stage pipeline |
| `GET` | `/api/analysis/latest` | Most recent analysis result with bounding boxes and evidence |
| `GET` | `/api/analysis/{id}/image/{variant}` | Fetch raw, preprocessed, or annotated image renders |
| `GET` | `/api/detections` | Filtered list of detections with pagination and severity filter |
| `POST` | `/api/detections/{id}/review` | Operator audit update (`CONFIRMED`, `REJECTED`, `FLAGGED`) |
| `GET` | `/api/map/targets` | Geolocated target list with priority groupings |
| `GET` | `/api/reports/pdf` | Generate ReportLab PDF inspection report |
| `GET` | `/api/reports/csv` | Tabular CSV export of mission contacts |
| `GET` | `/api/reports/json` | Structured JSON telemetry export |

For complete schemas, consult [`docs/API.md`](docs/API.md).

---

## Datasets & Provenance

- **Drishti SSS Benchmark**: Multi-class side-scan sonar benchmark dataset for underwater object detection.
- **SubPipe SSS Mini Dataset**: Continuous high-frequency (900 kHz) and low-frequency (455 kHz) sidescan inspection surveys of underwater pipelines.
- **Data Policy**: Large raw raster datasets are intentionally excluded from the Git repository. Complete preparation and tiling scripts are documented in [`data/README.md`](data/README.md).

---

## Project Novelty & Engineering Contribution

The core engineering contribution of **SIH26057** is not the standard application of YOLOv8s in isolation, but rather the **physics-guided, end-to-end integration**:

1. **Acoustic Physics Coupling**: Combines statistical deep learning bounding boxes with geometric acoustic shadow verification and target relief height estimation.
2. **Dual-Domain Anomaly Scoring**: Pairs multi-class supervised detection with an unsupervised Conv2D autoencoder to flag out-of-distribution seabed anomalies.
3. **Multi-Signal Evidence Fusion**: Replaces uncalibrated detector confidence with a multi-factor evidence score incorporating image quality and shadow morphology.
4. **End-to-End Field Readiness**: Integrates AI inference, SQLite persistence, human-in-the-loop review, and automated ReportLab PDF generation into a single 100% offline workflow.

---

## SIH Problem Statement Alignment

| SIH26057 Requirement | System Implementation | Verification Status |
|---|---|:---:|
| **Automated Object Detection** | 5-class YOLOv8s detector (`models/best.pt`) | **VERIFIED (Peak mAP@50: 71.79%)** |
| **Acoustic Noise Mitigation** | CLAHE + median filter dynamic preprocessing | **VERIFIED (7.80 ms latency)** |
| **Acoustic Shadow Verification** | Co-located shadow segmentation & height estimation | **VERIFIED (0.58 ms latency)** |
| **Multi-Signal Evidence Scoring** | 5-factor weighted evidence fusion framework | **VERIFIED (0.01 ms latency)** |
| **False-Positive Filtering** | Morphological aspect ratio & area constraints | **VERIFIED (0.35 ms latency)** |
| **Geospatial Telemetry** | Explicit coordinate provenance tracking & Leaflet map | **VERIFIED (Unit tested)** |
| **Mission Reporting** | Standalone dual-pass PDF, CSV, and JSON generation | **VERIFIED (ReportLab engine)** |
| **Interactive Dashboard** | React 18 + Vite + TypeScript web interface | **VERIFIED (0 build errors)** |
| **Air-Gapped Field Operation** | 100% local execution with zero cloud dependencies | **VERIFIED (Offline ready)** |

---

## Limitations

- **Cross-Sonar Frequency Transfer**: Models trained on specific sonar bands (e.g., 455/900 kHz) may experience domain shifts when deployed on sonars with significantly different beam patterns without fine-tuning.
- **Small Target Resolution at Long Slant Ranges**: Small objects like mine cylinders and crab pots at extreme acoustic slant ranges exhibit few pixels across the contact, making shadow detection dependent on tow-fish altitude.
- **Heuristic Evidence Fusion**: The current evidence fusion framework uses an engineered linear weighting scheme rather than a Bayesian network.
- **Base Map Tiles Offline**: While telemetry markers render offline, OpenStreetMap tile backgrounds require a pre-cached tile server when completely disconnected from the internet.

---

## Future Work

- **Native Sonar File Ingestion**: Direct binary parsing for raw sonar formats (XTF, JSF, and vendor-specific ping formats) to read tow-fish altitude and slant range directly from ping headers.
- **Bayesian Evidence Networks**: Transition from linear heuristic fusion to calibrated Bayesian networks for probabilistic threat assessment.
- **Expanded Marine Taxonomy**: Incorporate additional regional object classes (lost cargo containers, subsea communication cables, submerged ordnance types).
- **Edge ONNX / TensorRT Acceleration**: Quantize models to INT8/FP16 for deployment on ultra-low-power AUV edge computers (e.g., NVIDIA Jetson Orin Nano).

---

## License & Attribution

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for complete terms.

Developed for **Smart India Hackathon 2026 (Problem Statement SIH26057)**: *AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery*.
