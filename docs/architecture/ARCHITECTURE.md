# System Architecture & AI Pipeline Specification

## High-Level Architecture

The SIH26057 system implements a decoupled, high-performance architecture consisting of:
1. **Perception Engine (`ai/`)**: Modular computer vision and acoustic signal processing pipelines.
2. **Execution Bridge (`backend/app/pipeline_bridge.py`)**: Synchronous & asynchronous execution manager.
3. **Application Layer (`backend/app/api.py`)**: Asynchronous FastAPI service exposing authenticated REST endpoints.
4. **Presentation Layer (`frontend/`)**: Modern reactive interface providing situational awareness for sonar operators.

---

## AI Processing Pipeline

```
Raw Sonar Tile (H x W x C)
       │
       ▼
[01. Sonar Preprocessing]
       ├── Grayscale conversion & Contrast Normalization
       ├── CLAHE enhancement (tileGridSize=(8,8), clipLimit=2.0)
       └── Metric computation (Mean, Std, Entropy, Dynamic Range)
       │
       ├───► [02. Autoencoder Anomaly Detection]
       │         └── Latent reconstruction error (MSE) -> Anomaly Score ($0.0 - 1.0$)
       │
       ├───► [03. Dropout & Signal Quality Analysis]
       │         └── Row-wise brightness & zero-run detection -> Blank fraction
       │
       ▼
[04. YOLOv8s Target Detection]
       └── Multi-class inference -> [Bounding Boxes, Class IDs, Raw Confidence]
       │
       ▼
[05. Acoustic Shadow Geometric Analysis]
       ├── Highlight-to-shadow gradient calculation
       ├── Shadow length and offset validation
       └── Shadow score computation ($S_{shadow} \in [0, 1]$)
       │
       ▼
[06. False Positive Rejection Filter]
       ├── Minimum area threshold ($>200\,\text{px}^2$)
       ├── Aspect ratio constraint ($AR \in [0.1, 10.0]$)
       └── Shadow-contrast compatibility check
       │
       ▼
[07. Multi-Signal Evidence Fusion]
       ├── $E = w_{det} C_{det} + w_{shad} S_{shad} + w_{anom} S_{anom} + w_{snr} S_{snr}$
       ├── Threat Severity Classification (HIGH / MEDIUM / LOW)
       └── Geolocation mapping (GPS / Dead Reckoning / Simulated Demo fallback)
       │
       ▼
[08. Persistence & Operator Dispatch]
       ├── SQLite storage
       └── Real-time WebSocket / REST delivery
```

---

## False Positive Suppression Strategy

In sidescan sonar imagery, seabed reverberations, sand ripples, and thermocline artifacts frequently trigger spurious detections in pure deep-learning models. Our 3-stage validation mitigates this:
1. **Acoustic Shadow Verification**: Genuine seabed protrusions block acoustic return, creating an acoustic shadow behind the target relative to the nadir track. Detections lacking appropriate shadow regions are penalized.
2. **Quality & Dropout Gate**: High-noise or signal-loss tiles are flagged, preventing false triggers caused by acoustic beam dropouts.
3. **Geometric Prior Bounds**: Enforces realistic physical aspect ratio and boundary checks.
