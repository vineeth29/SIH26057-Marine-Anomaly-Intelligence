# Sonar Data Directory — SIH26057

## Dataset Policy
To maintain repository agility, large raw sonar recordings and tile archives (e.g. SubPipe 96k raw tiles) are **not tracked in Git**.

## Datasets Supported

### 1. SubPipe (Underwater Pipeline Inspection Dataset)
- **Source:** [REMARO Network SubPipe Dataset](https://github.com/remaro-network/SubPipe-dataset)
- **Target Category:** `submarine_pipeline`
- **Data Format:** High-Frequency (HF) and Low-Frequency (LF) Side-Scan Sonar waterfall strips.

### 2. Drishti / Combined Multi-Class Debris Dataset
- **Configuration:** `configs/drishti_data.yaml`
- **Classes:** `submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder`, `crab_pot`

## Dataset Preparation & Tiling Scripts
Use the curated scripts in `scripts/` to generate training and evaluation datasets:
```bash
# Inspect raw dataset
python scripts/inspect_subpipe.py --path <path-to-subpipe>

# Prepare and tile large sonar strips into 640x640 training tiles
python scripts/tile_subpipe.py --input-dir <path> --output-dir data/tiled --tile-size 640 --overlap 0.2

# Validate dataset balance and annotations
python scripts/validate_tiling.py --dataset-dir data/tiled
```
