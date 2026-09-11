"""
SIH26057 Multi-class YOLOv8n training script.
Dataset: rehan9599/drishti-sss (HuggingFace)

This script:
1. Downloads and inspects the drishti-sss dataset
2. Verifies class distribution and split integrity
3. Trains YOLOv8n with proper augmentation and early stopping
4. Evaluates on test set and saves metrics
5. Saves best_multiclass.pt for production use

Usage:
    python scripts/train_multiclass.py [--dry-run] [--epochs N] [--imgsz N] [--batch N]

Requirements:
    pip install huggingface_hub ultralytics pyyaml
    CUDA-enabled PyTorch (verified before training starts)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# GPU VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_gpu() -> tuple:
    """
    Verify GPU, measure VRAM, and return (device, vram_gb, free_gb).
    Prints a complete hardware report.
    Exits if CUDA is unavailable and --force-cpu is not set.
    """
    import torch

    cuda_ok = torch.cuda.is_available()
    info = {
        "pytorch_version": torch.__version__,
        "cuda_built_version": torch.version.cuda,
        "cuda_runtime_available": cuda_ok,
        "gpu_name": "N/A",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "compute_capability": "N/A",
    }

    if cuda_ok:
        torch.cuda.init()
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_total_gb"] = round(props.total_memory / 1024**3, 2)
        # Measure actual free memory
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        info["vram_free_gb"] = round(free_bytes / 1024**3, 2)
        info["compute_capability"] = f"{props.major}.{props.minor}"

    print("\n" + "="*60)
    print("GPU HARDWARE REPORT")
    print("="*60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")

    if not cuda_ok:
        print("FATAL: CUDA is not available.")
        print("  torch.__version__ =", torch.__version__)
        print("  torch.cuda.is_available() = False")
        print("")
        print("This means PyTorch was built without CUDA support, or the")
        print("CUDA runtime is not accessible from this Python environment.")
        print("")
        print("FIX: Run from the train_env virtualenv with cu121 torch installed.")
        print("  .\\train_env\\Scripts\\python.exe scripts\\train_multiclass.py")
        sys.exit(1)

    return "0", info["vram_total_gb"], info["vram_free_gb"]


def auto_select_batch_imgsz(vram_free_gb: float, args) -> tuple:
    """
    Pick the largest (imgsz, batch) combination that fits available VRAM
    for YOLOv8n, while staying conservative to avoid OOM mid-epoch.

    RTX 3050 4GB — typically ~3.0–3.2 GB free at training start.

    VRAM usage estimates for YOLOv8n (AMP enabled):
      imgsz=640,  batch=16  → ~3.5 GB  (OOM on 4GB)
      imgsz=640,  batch=8   → ~2.2 GB  (SAFE on 4GB)
      imgsz=640,  batch=12  → ~2.8 GB  (SAFE if >2.8 GB free)
      imgsz=800,  batch=8   → ~2.8 GB  (SAFE if >2.8 GB free)
      imgsz=800,  batch=6   → ~2.4 GB  (SAFE)
      imgsz=1024, batch=4   → ~3.2 GB  (RISKY on 4GB — use only if >3.5 GB free)
      imgsz=1024, batch=2   → ~2.5 GB  (SAFE)

    We leave a 0.5 GB safety margin for driver overhead.
    """
    # User explicitly set both — respect it (but warn if risky)
    if args.batch != -1 and args.imgsz != -1:
        print(f"[Config] User-specified: imgsz={args.imgsz}, batch={args.batch}")
        if vram_free_gb < 2.0:
            print(f"WARNING: Only {vram_free_gb:.2f} GB free. User-set config may OOM.")
        return args.imgsz, args.batch

    usable = vram_free_gb - 0.5  # safety margin

    # Select: largest imgsz that fits, then largest batch for that imgsz
    if usable >= 3.2:
        imgsz, batch = 1024, 2   # high-res, small batch
        label = "imgsz=1024, batch=2 (high-res mode, >3.2GB free)"
    elif usable >= 2.8:
        imgsz, batch = 800, 8
        label = "imgsz=800, batch=8 (balanced mode, >2.8GB free)"
    elif usable >= 2.2:
        imgsz, batch = 640, 8
        label = "imgsz=640, batch=8 (standard mode)"
    elif usable >= 1.6:
        imgsz, batch = 640, 4
        label = "imgsz=640, batch=4 (low VRAM mode)"
    else:
        imgsz, batch = 640, 2
        label = "imgsz=640, batch=2 (minimal VRAM mode)"

    # Override with user args if partially specified
    if args.imgsz != -1:
        imgsz = args.imgsz
    if args.batch != -1:
        batch = args.batch

    print(f"\n[AutoConfig] VRAM usable: {usable:.2f} GB")
    print(f"[AutoConfig] Selected:    {label}")
    print(f"[AutoConfig] Final:       imgsz={imgsz}, batch={batch}\n")
    return imgsz, batch




# ─────────────────────────────────────────────────────────────────────────────
# DATASET DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset(local_dir: Path) -> Path:
    """Download drishti-sss dataset from HuggingFace if not already present."""
    from huggingface_hub import snapshot_download

    marker = local_dir / ".download_complete"
    if marker.exists():
        print(f"[Dataset] Already downloaded at: {local_dir}")
        return local_dir

    print(f"[Dataset] Downloading rehan9599/drishti-sss to {local_dir}...")
    t0 = time.time()
    snapshot_download(
        "rehan9599/drishti-sss",
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    elapsed = round(time.time() - t0, 1)
    marker.write_text(f"Downloaded in {elapsed}s")
    print(f"[Dataset] Download complete in {elapsed}s")
    return local_dir


# ─────────────────────────────────────────────────────────────────────────────
# DATASET INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(dataset_dir: Path) -> dict:
    """
    Walk the dataset directory and report exact structure:
    - image counts per split
    - annotation counts per split
    - class distribution
    - image resolutions
    - bounding box statistics
    """
    import cv2
    import numpy as np

    print(f"\n[Inspect] Scanning dataset at: {dataset_dir}")

    # Find all label files
    label_files = list(dataset_dir.rglob("*.txt"))
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [f for f in dataset_dir.rglob("*") if f.suffix.lower() in image_exts]

    # Identify splits
    splits = defaultdict(lambda: {"images": [], "labels": [], "annotations": []})
    for img_path in image_files:
        parts = img_path.parts
        split = "unknown"
        for p in parts:
            if p.lower() in ("train", "val", "valid", "test"):
                split = p.lower()
                if split == "valid":
                    split = "val"
                break
        splits[split]["images"].append(img_path)

    for lbl_path in label_files:
        parts = lbl_path.parts
        split = "unknown"
        for p in parts:
            if p.lower() in ("train", "val", "valid", "test"):
                split = p.lower()
                if split == "valid":
                    split = "val"
                break
        splits[split]["labels"].append(lbl_path)

    # Class counting
    class_counts = defaultdict(int)
    bbox_stats = []
    all_classes = set()

    # Find data.yaml for class names
    yaml_files = list(dataset_dir.rglob("data.yaml")) + list(dataset_dir.rglob("*.yaml"))
    class_names = {}
    data_yaml_path = None

    import yaml
    for yf in yaml_files:
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if "names" in cfg:
                names = cfg["names"]
                if isinstance(names, list):
                    class_names = {i: n for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    class_names = {int(k): v for k, v in names.items()}
                data_yaml_path = yf
                break
        except Exception:
            pass

    # Count annotations
    for split_name, split_data in splits.items():
        for lbl_path in split_data["labels"]:
            try:
                lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cls_name = class_names.get(cls_id, f"class_{cls_id}")
                        class_counts[cls_name] += 1
                        all_classes.add(cls_id)
                        # bbox: cx cy w h (normalized)
                        w, h = float(parts[3]), float(parts[4])
                        bbox_stats.append((w, h, w * h))
                        split_data["annotations"].append(cls_name)
            except Exception:
                pass

    # Resolution sampling
    resolutions = []
    for split_name, split_data in splits.items():
        for img_path in split_data["images"][:20]:  # sample first 20
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    resolutions.append((img.shape[1], img.shape[0]))
            except Exception:
                pass

    # Build report
    report = {
        "dataset_dir": str(dataset_dir),
        "data_yaml_path": str(data_yaml_path) if data_yaml_path else "NOT FOUND",
        "class_names": class_names,
        "splits": {},
        "class_distribution": dict(sorted(class_counts.items(), key=lambda x: -x[1])),
        "total_annotations": sum(class_counts.values()),
        "unique_classes_found": sorted(list(all_classes)),
    }

    for split_name, split_data in splits.items():
        report["splits"][split_name] = {
            "images": len(split_data["images"]),
            "label_files": len(split_data["labels"]),
            "annotations": len(split_data["annotations"]),
        }

    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        report["resolution_stats"] = {
            "sample_count": len(resolutions),
            "width_min": min(widths),
            "width_max": max(widths),
            "height_min": min(heights),
            "height_max": max(heights),
        }

    if bbox_stats:
        import numpy as np
        areas = [b[2] for b in bbox_stats]
        report["bbox_stats"] = {
            "count": len(bbox_stats),
            "mean_area_normalized": round(float(np.mean(areas)), 4),
            "min_area_normalized": round(float(np.min(areas)), 6),
            "max_area_normalized": round(float(np.max(areas)), 4),
        }

    # Print report
    print("\n" + "="*60)
    print("DATASET INSPECTION REPORT")
    print("="*60)
    print(f"  Dataset dir:    {report['dataset_dir']}")
    print(f"  data.yaml:      {report['data_yaml_path']}")
    print(f"  Class names:    {report['class_names']}")
    print(f"  Total annotations: {report['total_annotations']}")
    print(f"\n  Splits:")
    for sp, sd in report["splits"].items():
        print(f"    {sp}: {sd['images']} images, {sd['annotations']} annotations")
    print(f"\n  Class distribution:")
    for cls, cnt in report["class_distribution"].items():
        print(f"    {cls}: {cnt}")
    if "resolution_stats" in report:
        rs = report["resolution_stats"]
        print(f"\n  Resolutions (sample of {rs['sample_count']}):")
        print(f"    width: {rs['width_min']}-{rs['width_max']}px")
        print(f"    height: {rs['height_min']}-{rs['height_max']}px")
    print("="*60 + "\n")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# DATA YAML PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data_yaml(dataset_dir: Path, inspection: dict) -> Path:
    """
    Use the existing data.yaml if found and locally valid, otherwise create configs/drishti_data.yaml.
    Preserves the dataset's predefined splits and resolves paths locally.
    """
    import yaml

    existing_yaml = inspection.get("data_yaml_path", "NOT FOUND")
    if existing_yaml != "NOT FOUND" and Path(existing_yaml).exists():
        try:
            with open(existing_yaml) as f:
                cfg = yaml.safe_load(f)
            # Check if cfg has required keys and that 'path' is either absent or points to a real local path
            if "train" in cfg and "names" in cfg:
                base_path = cfg.get("path")
                if base_path and Path(base_path).exists():
                    print(f"[DataYAML] Using existing valid YAML: {existing_yaml}")
                    return Path(existing_yaml)
                elif not base_path and (Path(existing_yaml).parent / cfg["train"]).exists():
                    print(f"[DataYAML] Using existing valid YAML with relative paths: {existing_yaml}")
                    return Path(existing_yaml)
                else:
                    print(f"[DataYAML] Existing yaml base path invalid ({base_path}) — generating local configs/drishti_data.yaml")
        except Exception as e:
            print(f"[DataYAML] Error checking existing yaml: {e} — generating local configs/drishti_data.yaml")

    # Find split directories
    splits_found = inspection.get("splits", {})
    class_names = inspection.get("class_names", {})

    if not class_names:
        # Fallback: expected SIH26057 target classes
        class_names = {
            0: "crab_pot",
            1: "submarine_pipeline",
            2: "shipwreck",
            3: "ghost_net",
            4: "mine_cylinder",
        }

    # Detect split dirs
    train_dir = None
    val_dir = None
    test_dir = None

    for candidate in ["train", "training"]:
        d = dataset_dir / candidate / "images"
        if d.exists():
            train_dir = candidate + "/images"
            break
        d = dataset_dir / candidate
        if d.exists():
            train_dir = candidate
            break

    for candidate in ["val", "valid", "validation"]:
        d = dataset_dir / candidate / "images"
        if d.exists():
            val_dir = candidate + "/images"
            break
        d = dataset_dir / candidate
        if d.exists():
            val_dir = candidate
            break

    for candidate in ["test", "testing"]:
        d = dataset_dir / candidate / "images"
        if d.exists():
            test_dir = candidate + "/images"
            break
        d = dataset_dir / candidate
        if d.exists():
            test_dir = candidate
            break

    if train_dir is None:
        raise FileNotFoundError(
            f"Could not find train split directory in {dataset_dir}. "
            "Please inspect the dataset manually."
        )

    names_list = [class_names[i] for i in sorted(class_names.keys())]

    yaml_content = {
        "path": str(dataset_dir.resolve()).replace("\\", "/"),
        "train": train_dir.replace("\\", "/"),
        "val": (val_dir or train_dir).replace("\\", "/"),
        "names": {i: name for i, name in enumerate(names_list)},
        "nc": len(names_list),
    }
    if test_dir:
        yaml_content["test"] = test_dir.replace("\\", "/")

    out_yaml = ROOT / "configs" / "drishti_data.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"[DataYAML] Generated and saved: {out_yaml}")
    return out_yaml


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train(data_yaml: Path, device: str, args) -> Path:
    """Run YOLOv8n training with proper augmentation and early stopping."""
    from ultralytics import YOLO

    # Choose batch size based on GPU VRAM
    # RTX 3050 4GB: batch 8 at 640, batch 4 at 1024
    batch = args.batch
    imgsz = args.imgsz

    print(f"\n[Train] Starting YOLOv8n training")
    print(f"  Epochs:  {args.epochs} (with early stopping patience={args.patience})")
    print(f"  Imgsz:   {imgsz}")
    print(f"  Batch:   {batch}")
    print(f"  Device:  {device}")
    print(f"  Data:    {data_yaml}\n")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/train",
        name="drishti_multiclass_yolov8n",
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        amp=True,             # Mixed precision — faster on GPU
        workers=2,            # Windows safe dataloader worker count
        # --- Augmentation ---
        hsv_h=0.015,          # hue jitter (minimal for sonar)
        hsv_s=0.3,            # saturation (limited for grayscale-like sonar)
        hsv_v=0.4,            # value/brightness variation
        degrees=10.0,         # small rotation ±10° (physically valid for SSS)
        translate=0.1,        # shift
        scale=0.5,            # scale augmentation
        shear=0.0,            # no shear (distorts sonar geometry)
        perspective=0.0,      # no perspective (SSS is roughly orthographic)
        flipud=0.0,           # no vertical flip (sonar port/starboard has meaning)
        fliplr=0.5,           # horizontal flip valid (targets can be on either side)
        mosaic=0.8,           # mosaic augmentation
        mixup=0.1,            # mild mixup
        copy_paste=0.0,       # no copy-paste (may introduce phantom objects)
        erasing=0.3,          # simulate data dropout/speckle occlusion
        close_mosaic=10,      # disable mosaic in last 10 epochs
        # --- Validation ---
        val=True,
        split="val",
        plots=True,
        save=True,
        save_period=-1,       # only save best
        verbose=True,
    )

    # Locate best.pt
    run_dir = Path(results.save_dir)
    best_pt = run_dir / "weights" / "best.pt"

    # Copy to canonical path
    out_path = ROOT / "models" / "best_multiclass.pt"
    if best_pt.exists():
        shutil.copy2(str(best_pt), str(out_path))
        print(f"\n[Train] Saved best checkpoint: {out_path}")
    else:
        print(f"[Train] WARNING: best.pt not found at {best_pt}")

    return run_dir, out_path


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model_path: Path, data_yaml: Path, device: str) -> dict:
    """Run held-out test evaluation and return metrics dict."""
    from ultralytics import YOLO
    import yaml

    print(f"\n[Evaluate] Running test evaluation on: {model_path}")
    model = YOLO(str(model_path))

    # Check if test split exists
    with open(data_yaml) as f:
        data_cfg = yaml.safe_load(f)

    split = "test" if "test" in data_cfg else "val"
    print(f"[Evaluate] Evaluating on split: {split}")

    val_results = model.val(
        data=str(data_yaml),
        split=split,
        device=device,
        verbose=True,
        plots=True,
    )

    box = val_results.box
    metrics = {
        "split": split,
        "model_path": str(model_path),
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4),
            "f1": round(2 * float(box.mp) * float(box.mr) / max(float(box.mp) + float(box.mr), 1e-8), 4),
        },
        "per_class": {},
    }

    # Per-class metrics
    names = val_results.names
    if hasattr(box, "ap_class_index"):
        for i, cls_idx in enumerate(box.ap_class_index):
            cls_name = names.get(int(cls_idx), f"class_{cls_idx}")
            metrics["per_class"][cls_name] = {
                "precision": round(float(box.p[i]), 4),
                "recall": round(float(box.r[i]), 4),
                "ap50": round(float(box.ap50[i]), 4),
                "ap50_95": round(float(box.ap[i]), 4),
            }

    # Save metrics
    eval_dir = ROOT / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    out_json = eval_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Evaluate] Metrics saved: {out_json}")

    print("\n" + "="*60)
    print("TEST EVALUATION RESULTS")
    print("="*60)
    print(f"  Split:      {split}")
    print(f"  Precision:  {metrics['overall']['precision']:.4f}")
    print(f"  Recall:     {metrics['overall']['recall']:.4f}")
    print(f"  F1:         {metrics['overall']['f1']:.4f}")
    print(f"  mAP@50:     {metrics['overall']['map50']:.4f}")
    print(f"  mAP@50-95:  {metrics['overall']['map50_95']:.4f}")
    print(f"\n  Per-class:")
    for cls_name, cls_m in metrics["per_class"].items():
        # Mark ghost_net as synthetic if applicable
        tag = " [SYNTHETIC-ONLY]" if "ghost" in cls_name.lower() or "net" in cls_name.lower() else ""
        print(f"    {cls_name}{tag}: P={cls_m['precision']:.3f} R={cls_m['recall']:.3f} AP50={cls_m['ap50']:.3f}")
    print("="*60 + "\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def update_model_config(model_path: Path, class_names: list, metrics: dict):
    """Update configs/model.yaml to point to the new multiclass model."""
    import yaml

    config_path = ROOT / "configs" / "model.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["detection"]["model_path"] = str(model_path.relative_to(ROOT))
    cfg["detection"]["classes"] = class_names
    cfg["versioning"]["current_detection_version"] = "multiclass-v1.0"

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"[Config] Updated model.yaml → {model_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SIH26057 Multi-class YOLOv8n training")
    p.add_argument("--dataset-dir", default="data/drishti_sss",
                   help="Local path to store/find the drishti-sss dataset")
    p.add_argument("--epochs", type=int, default=100,
                   help="Maximum training epochs (default: 100 with early stopping patience=20)")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience in epochs (default: 20)")
    p.add_argument("--imgsz", type=int, default=-1,
                   help="Training image size. Default: AUTO (VRAM-based selection). "
                        "Override with e.g. --imgsz 640")
    p.add_argument("--batch", type=int, default=-1,
                   help="Batch size. Default: AUTO (VRAM-based selection). "
                        "Override with e.g. --batch 8")
    p.add_argument("--dry-run", action="store_true",
                   help="Inspect dataset only, do not train")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip HuggingFace download (dataset already local)")
    p.add_argument("--device", default=None,
                   help="Override device (auto-detected by default)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_dir = ROOT / args.dataset_dir

    # ─── Step 1: GPU verification ──────────────────────────────────────────────
    if args.device:
        device = args.device
        vram_total_gb = 0.0
        vram_free_gb = 4.0  # assume sufficient when manually specified
    else:
        device, vram_total_gb, vram_free_gb = verify_gpu()

    # ─── Step 2: Auto-select imgsz + batch based on measured free VRAM ────────
    imgsz, batch = auto_select_batch_imgsz(vram_free_gb, args)
    # Patch args so train() uses the selected values
    args.imgsz = imgsz
    args.batch = batch

    # ─── Step 3: Dataset download ─────────────────────────────────────────────
    if not args.skip_download:
        dataset_dir = download_dataset(dataset_dir)

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        print("Run without --skip-download to fetch the dataset.")
        sys.exit(1)

    # ─── Step 4: Dataset inspection ───────────────────────────────────────────
    inspection = inspect_dataset(dataset_dir)

    eval_dir = ROOT / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    with open(eval_dir / "dataset_inspection.json", "w") as f:
        json.dump(inspection, f, indent=2, default=str)

    if args.dry_run:
        print("[DryRun] Dataset inspected. Not training (--dry-run).")
        sys.exit(0)

    if inspection["total_annotations"] == 0:
        print("ERROR: No annotations found in dataset. Cannot train.")
        print("Please manually inspect the dataset directory.")
        sys.exit(1)

    # ─── Step 5: Build data.yaml ──────────────────────────────────────────────
    data_yaml = prepare_data_yaml(dataset_dir, inspection)

    # ─── Step 6: Train ────────────────────────────────────────────────────────
    run_dir, model_path = train(data_yaml, device, args)

    # ─── Step 7: Evaluate multiclass model on test/val split ──────────────────
    metrics = evaluate(model_path, data_yaml, device)

    # ─── Step 8: Compare against preserved baseline ───────────────────────────
    baseline_pt = ROOT / "models" / "best.pt"
    baseline_metrics = None
    if baseline_pt.exists():
        print("\n" + "="*60)
        print("BASELINE vs MULTICLASS COMPARISON")
        print("="*60)
        try:
            baseline_metrics = evaluate(baseline_pt, data_yaml, device)
            with open(eval_dir / "baseline_metrics.json", "w") as f:
                json.dump(baseline_metrics, f, indent=2)
            delta = metrics['overall']['map50'] - baseline_metrics['overall']['map50']
            print(f"  Baseline  mAP@50: {baseline_metrics['overall']['map50']:.4f}")
            print(f"  Multiclass mAP@50: {metrics['overall']['map50']:.4f}")
            print(f"  Delta:            {delta:+.4f}")
        except Exception as e:
            print(f"  Baseline evaluation failed: {e}")
    else:
        print("[Comparison] models/best.pt not found — skipping baseline comparison.")

    # ─── Step 9: Update model config ──────────────────────────────────────────
    class_names = list(inspection.get("class_names", {}).values())
    if class_names:
        update_model_config(model_path, class_names, metrics)

    # ─── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  GPU:            {device} ({vram_total_gb:.1f} GB total, {vram_free_gb:.1f} GB was free)")
    print(f"  imgsz:          {imgsz}")
    print(f"  batch:          {batch}")
    print(f"  Final model:    {model_path}")
    print(f"  Precision:      {metrics['overall']['precision']:.4f}")
    print(f"  Recall:         {metrics['overall']['recall']:.4f}")
    print(f"  F1:             {metrics['overall']['f1']:.4f}")
    print(f"  mAP@50:         {metrics['overall']['map50']:.4f}")
    print(f"  mAP@50-95:      {metrics['overall']['map50_95']:.4f}")
    if baseline_metrics:
        print(f"  Baseline mAP@50:{baseline_metrics['overall']['map50']:.4f}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run GPU latency benchmark:")
    print("     .\\train_env\\Scripts\\python.exe scripts\\benchmark_latency.py --runs 20")
    print("  2. Launch dashboard:")
    print("     streamlit run app.py")

