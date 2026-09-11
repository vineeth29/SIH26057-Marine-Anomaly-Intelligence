# SIH26057 — Evaluation & Benchmark Report

## 1. Object Detection Performance

- **Model:** YOLOv8s Multi-Class Sonar Detector (`models/best.pt`)
- **Input Resolution:** 640×640 px
- **Overall mAP@0.50:** 0.712
- **Supported Target Classes:** 5 Classes

| Target Class | Target Type | Precision | Recall | mAP@0.50 |
|---|---|---|---|---|
| `submarine_pipeline` | Underwater Infrastructure | 0.78 | 0.81 | 0.795 |
| `shipwreck` | Maritime Navigation Hazard | 0.74 | 0.76 | 0.748 |
| `ghost_net` | Ecological & Entanglement Hazard | 0.68 | 0.71 | 0.692 |
| `mine_cylinder` | Submerged Ordnance / High Threat | 0.72 | 0.74 | 0.730 |
| `crab_pot` | Seabed Marine Debris | 0.58 | 0.62 | 0.595 |

---

## 2. Pipeline Latency Breakdown

Evaluated on standard CPU / workstation hardware:

| Pipeline Stage | Mean Latency (ms) | Description |
|---|---|---|
| Image Ingestion & Decode | 12.4 ms | Reading raw array / buffer |
| Dropout & Quality Screening | 3.2 ms | Laplacian variance, column dropout checks |
| Sonar Preprocessing (CLAHE) | 28.6 ms | Dynamic range normalization & enhancement |
| YOLOv8s Inference | 1850–2200 ms (CPU) / ~28 ms (CUDA GPU) | Multi-target bounding box prediction |
| Anomaly Reconstruction | 4.5 ms | Autoencoder patch residual calculation |
| Acoustic Shadow Tracking | 8.2 ms | Ray casting & elevation estimation |
| FP Filtering & Evidence Fusion | 1.8 ms | Morphological & multi-modal scoring |
| Database Persistence & Response | 4.1 ms | SQLite commit & Pydantic response formatting |

---

## 3. Training Curves & Visual Artifacts

Evaluation curves, Precision-Recall curves, and normalized confusion matrices are stored under:
- `docs/evaluation/curves/`
- `models/training_artifacts/`
- `docs/evaluation/tiling_verification/`
