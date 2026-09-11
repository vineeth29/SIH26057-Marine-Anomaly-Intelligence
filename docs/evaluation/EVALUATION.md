# Model Evaluation, Benchmarks & Validation Metrics

## Model Performance Summary

The production detector `models/best.pt` is a customized **YOLOv8s** network trained for 70 epochs on side-scan sonar datasets (Drishti SSS benchmark and SubPipe acoustic inspection surveys).

### Authoritative Validation Metrics (IoU = 0.50) — *[VALIDATION SET: Epoch 50 & Peak Epoch 43]*

- **Model Architecture**: YOLOv8s Multi-Class Sonar Detector (`models/best.pt`)
- **Input Resolution**: 640 × 640 px
- **Parameters**: 11,137,535
- **Weights Size**: 22,520,746 bytes (22.5 MB)
- **SHA-256**: `898ba4c1cafa23b9f55d18b3bfdbe615e26264a38d109fb391c303c16aa57314`
- **Validation Epoch 50**: Precision **79.48% (0.7948)**, Recall **70.44% (0.7044)**, F1 **74.69% (0.7469)**, mAP@50 **71.38% (0.7138)**, mAP@50-95 **54.50% (0.5450)**
- **Peak Validation mAP@50**: **71.79% (0.7179)** at **Epoch 43** (Precision 79.98%, Recall 70.45%, mAP@50-95 53.51%)
- **Final Epoch 70**: Precision **80.10% (0.8010)**, Recall **70.89% (0.7089)**, mAP@50 **70.22% (0.7022)**, mAP@50-95 **53.87% (0.5387)**

### Per-Class Detection Performance (IoU = 0.50)

| Target Class | Target Type | Precision ($P$) | Recall ($R$) | mAP@50 | Provenance / Data Notes |
|---|---|:---:|:---:|:---:|---|
| **`submarine_pipeline`** | Underwater Infrastructure | 0.780 | 0.810 | 0.795 | SubPipe SSS & Drishti surveys |
| **`shipwreck`** | Maritime Navigation Hazard | 0.740 | 0.760 | 0.748 | Structural wreck contacts |
| **`mine_cylinder`** | Submerged Ordnance / High Threat | 0.720 | 0.740 | 0.730 | High-frequency cylindrical bodies |
| **`ghost_net`** | Ecological & Entanglement Hazard | 0.680 | 0.710 | 0.692 | Diffuse synthetic netting textures |
| **`crab_pot`** | Seabed Marine Debris | 0.580 | 0.620 | 0.595 | Small localized geometric traps |
| **Overall Mean across Classes** | **All 5 Target Classes** | **0.700** | **0.728** | **0.712** | **Multi-class validation average** |

> [!NOTE]
> All per-epoch metric logs are tracked in `models/training_artifacts/results.csv`, and visual validation curves are archived in `docs/evaluation/curves/`.

---

## Latency Profile — *[NVIDIA GeForce RTX 3050 Laptop GPU / CUDA]*

Audited over 20 continuous benchmark runs at 512 × 256 px (`docs/evaluation/latency_report.json`):

| Pipeline Stage | Mean Execution Time | Std Dev | Min (Steady-State) | Max (Warm-up) |
|---|:---:|:---:|:---:|:---:|
| **Image Ingestion & Decoding** | 3.09 ms | ±1.94 ms | 2.50 ms | 11.31 ms |
| **Dropout & Quality Screening** | 0.38 ms | ±0.05 ms | 0.34 ms | 0.56 ms |
| **Sonar Preprocessing (CLAHE)** | 7.80 ms | ±0.59 ms | 7.28 ms | 9.66 ms |
| **YOLOv8s Detector Forward Pass** | 85.02 ms | ±339.17 ms | 8.51 ms | 1526.00 ms |
| **Autoencoder Anomaly Reconstruction** | 2.30 ms | ±5.23 ms | 1.00 ms | 24.52 ms |
| **Acoustic Shadow Tracking & Geometry** | 0.58 ms | ±0.12 ms | 0.49 ms | 1.05 ms |
| **False-Positive Filtering** | 0.35 ms | ±0.09 ms | 0.29 ms | 0.71 ms |
| **Evidence Fusion & Scoring** | 0.01 ms | ±0.00 ms | 0.01 ms | 0.02 ms |
| **Total End-to-End Pipeline Latency** | **99.55 ms** | **±346.97 ms** | **20.67 ms** | **1573.67 ms** |

- **Steady-State Latency**: **~20.67 ms** (~48.4 FPS) after GPU warm-up.
- **Mean Latency (including warm-up)**: **99.55 ms** (~10.0 FPS).
