# SIH26057 System Architecture

## End-to-End Pipeline Architecture

```
                                  [ RAW SIDE-SCAN SONAR IMAGE ]
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │   Quality Assessor    │ ───► Sharpness, Contrast, Noise,
                                    │ & Dropout Detector    │      Column Dropout Runs Check
                                    └───────────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │   Sonar Preprocessor  │ ───► Grayscale Conversion, Dynamic Range
                                    │       (CLAHE)         │      Normalization, Adaptive Bilateral Denoising
                                    └───────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
            ┌─────────────────────┐                           ┌─────────────────────┐
            │   YOLOv8s Detector  │                           │   Autoencoder Anom  │
            │   (5 Target Classes)│                           │   Detector (PyTorch)│
            └─────────────────────┘                           └─────────────────────┘
                       │                                                 │
            Bboxes & Confidence                                 Reconstruction Residual
                       │                                                 │
                       ▼                                                 │
            ┌─────────────────────┐                                     │
            │  Acoustic Shadow    │                                     │
            │     Analyzer        │                                     │
            └─────────────────────┘                                     │
                       │                                                 │
            Shadow Length & Height                                       │
                       │                                                 │
                       ▼                                                 │
            ┌─────────────────────┐                                     │
            │ False-Positive (FP) │                                     │
            │       Filter        │                                     │
            └─────────────────────┘                                     │
                       │ (Aspect Ratio, Area, Texture)                   │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │ Multi-Modal Evidence  │ ───► Fused Threat Score (0–100%)
                                    │     Fusion Engine     │      Severity Categorization
                                    └───────────────────────┘
                                                │
                       ┌────────────────────────┼────────────────────────┐
                       ▼                        ▼                        ▼
            ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
            │ SQLAlchemy / SQLite │  │  FastAPI REST API   │  │  ReportLab PDF /    │
            │   (sih26057.db)     │  │  (backend/app/)     │  │  CSV / JSON Export  │
            └─────────────────────┘  └─────────────────────┘  └─────────────────────┘
                                                │
                                                ▼
                                     [ React 18 / Vite UI ]
```

---

## Component Responsibilities

1. **`ai/preprocessing/sonar_preprocessor.py`**:
   - Contrast-Limited Adaptive Histogram Equalization (CLAHE)
   - Dynamic range stretching and Gaussian/Bilateral denoising
   - Background intensity flattening to handle acoustic attenuation with range

2. **`ai/quality/dropout_detector.py` & `utils/quality.py`**:
   - Focus / sharpness index using Laplacian variance
   - Acoustic transducer dropout run detection (blank columns)
   - Discontinuity and heave/pitch artifact detection

3. **`ai/detection/yolo_detector.py`**:
   - Neural target inference using trained YOLOv8s weights (`models/best.pt`)
   - NMS and IoU filtering across 5 key target categories

4. **`ai/shadow_analysis/shadow_analyzer.py`**:
   - Physics-informed ray casting and thresholding to detect acoustic shadow cast behind elevated objects
   - Target height estimation based on shadow length and slant range geometry

5. **`ai/fp_filter/false_positive_filter.py`**:
   - Morphological aspect-ratio analysis (distinguishing long linear pipelines from compact debris)
   - Rejecting spurious detections with non-physical geometry or noise artifacts

6. **`ai/anomaly/anomaly_detector.py`**:
   - Unsupervised deep autoencoder reconstruction error calculation for open-set / uncatalogued seabed anomalies

7. **`ai/fusion/confidence_fusion.py`**:
   - Weighted multi-factor evidence aggregation combining detector confidence, acoustic shadow presence, texture metrics, shape profile, and anomaly residual
   - Dynamic weight redistribution when acoustic shadows are unavailable or obscured
