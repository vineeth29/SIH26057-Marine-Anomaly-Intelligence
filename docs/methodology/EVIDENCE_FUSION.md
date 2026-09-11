# Multi-Modal Evidence Fusion & Threat Scoring

## Mathematical Formulation

In Side-Scan Sonar (SSS) imagery, standalone detector confidence can be prone to false positives caused by reverberation, speckle noise, and natural seabed structures. Our system combines physical acoustic properties with deep neural detection through multi-modal evidence fusion.

$$E = W_d \cdot C_d + W_s \cdot S_s + W_t \cdot T_s + W_g \cdot G_s + W_a \cdot A_s$$

Where:
- $C_d \in [0, 1]$: YOLOv8 Detector Confidence
- $S_s \in [0, 1]$: Acoustic Shadow Verification Score
- $T_s \in [0, 1]$: Object Texture Consistency Score
- $G_s \in [0, 1]$: Geometric Aspect Ratio & Shape Fit Score
- $A_s \in [0, 1]$: Regional Anomaly Residual Score

---

## Weight Redistribution for Low-Angle / Obscured Sonar

When acoustic shadows are physically absent (e.g. flat seabed debris or targets directly beneath the vehicle nadir track):

1. The shadow weight $W_s$ is dynamically removed.
2. The remaining weights are normalized such that:

$$\sum W_i = 1.0$$

---

## Acoustic Shadow Height Estimation

Target elevation above the seabed is calculated from acoustic cast shadow geometry:

$$H_t = \frac{L_s \cdot H_s}{R_s}$$

Where:
- $H_t$: Estimated target height (meters)
- $L_s$: Cast shadow length on the seabed (meters)
- $H_s$: Sonar altitude above seabed (meters)
- $R_s$: Slant range to target (meters)

When altitude or slant range metadata is not provided in telemetry, relative shadow ratio is computed from pixel span and reported as an acoustic prominence metric.
