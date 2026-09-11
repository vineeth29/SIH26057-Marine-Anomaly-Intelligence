"""
Phase 5: Train YOLOv8n on Combined SIH26057 Dataset
====================================================
Fine-tunes the multiclass model on data/sih26057_combined/ (Drishti-SSS + SubPipe Tiled).

Requirements:
  - Base checkpoint: models/best_multiclass.pt (NOT overwritten)
  - Target checkpoint: models/best_combined.pt
  - Dataset: data/sih26057_combined/dataset.yaml
  - Safe VRAM allocation for NVIDIA RTX 3050 Laptop GPU (4GB)
  - Complete logging of per-class validation metrics, hardware, SHA-256
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DATA_YAML = ROOT / "data" / "sih26057_combined" / "dataset.yaml"
BASE_MODEL = ROOT / "models" / "best_multiclass.pt"
TARGET_MODEL = ROOT / "models" / "best_combined.pt"
EVAL_DIR = ROOT / "evaluation"
RUNS_DIR = ROOT / "runs" / "train"

CLASS_NAMES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_gpu_info():
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
        }
    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info(0)
    return {
        "cuda_available": True,
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_total_gb": round(props.total_memory / (1024**3), 2),
        "vram_free_gb": round(free_b / (1024**3), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on Combined SIH26057 Dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (safe for 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--resume", type=str, default="", help="Path to last.pt to resume training")
    parser.add_argument("--device", type=str, default="0", help="Device: '0' for GPU, 'cpu' for CPU")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5: COMBINED SIH26057 MODEL TRAINING")
    print("=" * 70)

    # 1. Validate inputs
    if not DATA_YAML.exists():
        print(f"FATAL: Dataset config not found: {DATA_YAML}")
        sys.exit(1)
    if not BASE_MODEL.exists():
        print(f"FATAL: Base model not found: {BASE_MODEL}")
        sys.exit(1)

    gpu_info = get_gpu_info()
    print("\n[Hardware & Environment]")
    for k, v in gpu_info.items():
        print(f"  {k}: {v}")
    if not gpu_info["cuda_available"]:
        print("WARNING: CUDA is not available. Training on CPU will be extremely slow.")

    base_sha = sha256(BASE_MODEL)
    print(f"\n[Base Model]")
    print(f"  Path: {BASE_MODEL}")
    print(f"  SHA-256: {base_sha}")

    if args.resume:
        print(f"\n[Resuming Training]")
        print(f"  Resume Checkpoint: {args.resume}")
        model = YOLO(args.resume)
        start_time = time.time()
        train_results = model.train(resume=True, workers=0)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"combined_yolov8n_{timestamp}"
        run_dir = RUNS_DIR / run_name

        print(f"\n[Run Configuration]")
        print(f"  Run Directory: {run_dir}")
        print(f"  Epochs: {args.epochs} (patience={args.patience})")
        print(f"  Batch size: {args.batch}")
        print(f"  Image size: {args.imgsz}")
        print(f"  Optimizer: AdamW, lr0=0.0005, lrf=0.01, weight_decay=0.0005")
        print(f"  Augmentations: mosaic=0.8, fliplr=0.5, flipud=0.0, close_mosaic=10")

        # 2. Initialize Model from best_multiclass.pt
        model = YOLO(str(BASE_MODEL))

        start_time = time.time()

        # 3. Execute Training
        train_results = model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.01,
            weight_decay=0.0005,
            amp=True,
            workers=args.workers,
            mosaic=0.8,
            fliplr=0.5,
            flipud=0.0,
            shear=0.0,
            perspective=0.0,
            close_mosaic=10,
            project=str(RUNS_DIR),
            name=run_name,
            device=args.device if gpu_info["cuda_available"] else "cpu",  # noqa: E501
            save=True,
            val=True,
            plots=True,
            verbose=True,
        )

    duration_sec = time.time() - start_time
    duration_min = round(duration_sec / 60.0, 2)

    # 4. Locate best checkpoint
    actual_run_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir") else run_dir
    best_pt = actual_run_dir / "weights" / "best.pt"
    last_pt = actual_run_dir / "weights" / "last.pt"

    if not best_pt.exists():
        print(f"WARNING: best.pt not found at {best_pt}, trying last.pt")
        best_pt = last_pt if last_pt.exists() else None

    if best_pt is None or not best_pt.exists():
        print(f"FATAL: No trained checkpoint found in {actual_run_dir}")
        sys.exit(1)

    # 5. Save best checkpoint as models/best_combined.pt (WITHOUT overwriting best_multiclass.pt)
    TARGET_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(best_pt), str(TARGET_MODEL))
    target_sha = sha256(TARGET_MODEL)

    print("\n" + "=" * 70)
    print("TRAINING FINISHED & CHECKPOINT SAVED")
    print("=" * 70)
    print(f"  Best Weights: {best_pt}")
    print(f"  Saved Target: {TARGET_MODEL}")
    print(f"  Target SHA-256: {target_sha}")
    print(f"  Training Duration: {duration_min} minutes ({round(duration_sec, 1)}s)")

    # 6. Evaluate Best Checkpoint on Combined Validation Set
    print("\n[Evaluating Best Combined Checkpoint on Validation Set...]")
    eval_model = YOLO(str(TARGET_MODEL))
    val_metrics = eval_model.val(
        data=str(DATA_YAML),
        split="val",
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if gpu_info["cuda_available"] else "cpu",  # noqa: E501
        plots=False,
        verbose=False,
    )

    # Per-class metrics
    class_indices = val_metrics.box.ap_class_index if hasattr(val_metrics.box, "ap_class_index") else []
    per_class = {}
    for idx, c_idx in enumerate(class_indices):
        c_name = CLASS_NAMES.get(int(c_idx), f"class_{c_idx}")
        p = float(val_metrics.box.p[idx]) if idx < len(val_metrics.box.p) else 0.0
        r = float(val_metrics.box.r[idx]) if idx < len(val_metrics.box.r) else 0.0
        ap50 = float(val_metrics.box.ap50[idx]) if idx < len(val_metrics.box.ap50) else 0.0
        ap50_95 = float(val_metrics.box.ap[idx]) if idx < len(val_metrics.box.ap) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class[c_name] = {
            "class_id": int(c_idx),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "map50": round(ap50, 4),
            "map50_95": round(ap50_95, 4),
        }

    overall_p = float(val_metrics.box.mp)
    overall_r = float(val_metrics.box.mr)
    overall_map50 = float(val_metrics.box.map50)
    overall_map50_95 = float(val_metrics.box.map)
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0

    best_epoch = int(getattr(val_metrics, "epoch", -1))
    if hasattr(model, "trainer") and hasattr(model.trainer, "best_epoch"):
        best_epoch = int(model.trainer.best_epoch) + 1

    training_summary = {
        "timestamp": timestamp,
        "run_name": run_name,
        "run_dir": str(actual_run_dir),
        "base_model": {
            "path": str(BASE_MODEL),
            "sha256": base_sha,
        },
        "combined_model": {
            "path": str(TARGET_MODEL),
            "sha256": target_sha,
        },
        "hardware": gpu_info,
        "training_params": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "optimizer": "AdamW",
            "lr0": 0.0005,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "amp": True,
            "workers": args.workers,
            "mosaic": 0.8,
            "fliplr": 0.5,
            "flipud": 0.0,
            "close_mosaic": 10,
        },
        "duration_seconds": round(duration_sec, 2),
        "duration_minutes": duration_min,
        "best_epoch": best_epoch,
        "validation_metrics": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f1, 4),
            "map50": round(overall_map50, 4),
            "map50_95": round(overall_map50_95, 4),
            "per_class": per_class,
        },
        "notes": {
            "crab_pot": "Class 0 has 0 annotations in dataset (preserved in taxonomy)",
            "ghost_net": "Class 3 is synthetic-only",
            "submarine_pipeline": "Class 1 contains both Drishti-SSS and SubPipe tiled data",
        }
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EVAL_DIR / "combined_training_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)

    print(f"\nTraining summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
