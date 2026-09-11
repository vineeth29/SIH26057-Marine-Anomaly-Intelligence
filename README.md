# SIH26057 — Automated Underwater Marine Debris & Target Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg)](https://www.typescriptlang.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent inspection and multi-modal evidence fusion platform for detecting underwater debris, pipelines, hazardous submerged ordnance, and acoustic anomalies using **Side-Scan Sonar (SSS)** imagery.

---

## Key Capabilities

- **Adaptive Sonar Enhancement**: Contrast-Limited Adaptive Histogram Equalization (CLAHE), dynamic range stretching, and bilateral speckle filtering.
- **Transducer Quality & Dropout Screening**: Laplacian sharpness analysis, column dropout identification, and heave/pitch artifact detection.
- **Multi-Class Neural Detection**: YOLOv8s neural detector trained on multi-class targets (`submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder`, `crab_pot`).
- **Physics-Informed Acoustic Shadow Validation**: Shadow ray casting, target elevation estimation, and aspect-ratio validation.
- **Deep Anomaly Autoencoder**: Unsupervised reconstruction residual scoring for open-set acoustic seabed anomalies.
- **Multi-Modal Evidence Fusion**: Synthesis of detector confidence, shadow metrics, texture, and anomaly residuals into a unified threat score.
- **Modern Full-Stack Experience**: High-performance FastAPI REST backend paired with a React 18 / Vite / TypeScript dashboard featuring Leaflet GIS mapping and ReportLab PDF reporting.

---

## Architecture Overview

```
Sonar Image ──► Quality Screening ──► Preprocessing (CLAHE) ──► YOLOv8s + Autoencoder
                     │                                                   │
                     ▼                                                   ▼
             Dropout Detection                                   Shadow & FP Filtering
                                                                         │
                                                                         ▼
                                                                Evidence Fusion Engine
                                                                         │
                                                                         ▼
                                                         [ FastAPI ] ──► [ React UI / PDF ]
```

See [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) and [docs/architecture/SOURCE_OF_TRUTH.md](docs/architecture/SOURCE_OF_TRUTH.md) for full architectural specifications.

---

## Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend)

### 2. Backend Setup & Startup
```powershell
# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI backend (port 8000)
uvicorn backend.app.main:app --reload --port 8000
```
API Documentation: `http://localhost:8000/docs`

### 3. Frontend Setup & Startup
```powershell
# In a new terminal
cd frontend
npm install
npm run dev
```
Dashboard UI: `http://localhost:5173`

### 4. Running Test Suite
```powershell
# Execute 100+ unit and integration tests
python -m pytest tests/ -v
```

---

## Documentation Index

- [Source of Truth Specification](docs/architecture/SOURCE_OF_TRUTH.md)
- [System Architecture](docs/architecture/ARCHITECTURE.md)
- [Multi-Modal Evidence Fusion Methodology](docs/methodology/EVIDENCE_FUSION.md)
- [Model Evaluation & Latency Benchmarks](docs/evaluation/EVALUATION_REPORT.md)
- [Dataset Preparation Guide](data/README.md)
- [Backend & Frontend Standalone Setup](SETUP.md)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
