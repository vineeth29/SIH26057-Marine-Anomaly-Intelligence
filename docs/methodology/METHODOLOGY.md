# Mathematical & Physical Methodology

## 1. Acoustic Image Preprocessing

Given a raw sonar intensity matrix $I(x, y) \in [0, 255]$:
1. **Dynamic Range Normalization**:
   $$I_{norm}(x, y) = \frac{I(x, y) - I_{min}}{I_{max} - I_{min}} \times 255$$
2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE)**:
   Divides the raster into contextual tiles ($8 \times 8$) and applies localized histogram clipping ($C_{limit} = 2.0$) to prevent acoustic speckle amplification.

---

## 2. Multi-Signal Evidence Score Formulation

The final evidence score $E \in [0, 1]$ represents the integrated confidence from independent sensing modalities:

$$E = \text{clamp}\left( w_{det} \cdot C_{det} + w_{shad} \cdot S_{shad} + w_{anom} \cdot S_{anom} + w_{snr} \cdot S_{snr},\, 0.0,\, 1.0 \right)$$

### Standard Weights
- **Detector Confidence Weight ($w_{det}$)**: $0.45$
- **Acoustic Shadow Score Weight ($w_{shad}$)**: $0.25$
- **Autoencoder Anomaly Weight ($w_{anom}$)**: $0.20$
- **Signal-to-Noise Ratio Weight ($w_{snr}$)**: $0.10$

$$\sum w_i = 1.00$$

---

## 3. Threat Severity Classification

Severity $S \in \{\text{HIGH}, \text{MEDIUM}, \text{LOW}\}$ is determined by combining the target classification risk tier and the computed evidence score:

| Class / Condition | Evidence Threshold | Severity | Visual Indicator |
|---|---|---|---|
| `mine_cylinder`, `submarine_pipeline` | $E \ge 0.50$ | **HIGH** | Red (`#EF4444`) |
| `shipwreck`, `ghost_net` | $E \ge 0.65$ | **HIGH** | Red (`#EF4444`) |
| `shipwreck`, `ghost_net` | $0.40 \le E < 0.65$ | **MEDIUM** | Amber (`#F59E0B`) |
| `crab_pot` | $E \ge 0.60$ | **MEDIUM** | Amber (`#F59E0B`) |
| Any target | $E < 0.40$ | **LOW** | Blue/Green (`#10B981`) |

---

## 4. Acoustic Target Height Estimation

Target elevation above seabed $h_t$ is estimated from shadow length $L_s$, sensor altitude $H$, and slant range $R$:

$$h_t = \frac{H \cdot L_s}{R + L_s}$$

When slant range telemetry is unavailable, a calibrated geometric approximation based on pixel shadow extension is utilized.
