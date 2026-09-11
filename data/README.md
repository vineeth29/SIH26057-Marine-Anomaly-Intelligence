# Datasets & Sonar Data Documentation

## SSS Dataset Overview

The system processes high-resolution side-scan sonar (SSS) imagery across multiple frequency bands (e.g. 450 kHz and 900 kHz).

### Integrated Datasets
1. **Drishti SSS Dataset**: Side-scan sonar dataset containing marine debris annotations (`crab_pot`, `ghost_net`, `shipwreck`, `mine_cylinder`).
2. **SubPipe SSS Dataset**: Industrial submarine pipeline inspection dataset with continuous linear infrastructure targets.
3. **Synthetic Mission Benchmark**: Generated high-fidelity sonar simulation frames for end-to-end integration and calibration testing (`scripts/setup_demo.py`).

---

## Data Preparation & Tiling Pipeline

Raw side-scan sonar waterfall logs (e.g. $5000 \times 1000$ pixels) are tiled using sliding windows with 25% overlap:
- Tile Size: $640 \times 640$ pixels
- Normalization: CLAHE + dynamic range normalization
- Script: `scripts/tile_subpipe.py` and `scripts/build_combined_dataset.py`

To generate synthetic demonstration frames:
```bash
python scripts/setup_demo.py
```
