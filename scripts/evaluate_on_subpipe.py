"""
SubPipeMini2 Model Evaluation Script
=====================================

Evaluates models/best_multiclass.pt on the SubPipeMini2 (SubPipeMiniSSS) dataset.

- Never modifies the original SubPipe dataset.
- Converts PBM images in-memory (not persisted to disk).
- Evaluates HF (900 kHz) and LF (455 kHz) sonar splits separately.
- Handles class remapping: SubPipe class 0 (Pipeline) → model class 1 (submarine_pipeline).
- Evaluates 1,365 annotated images for detection metrics.
- Evaluates 701 background images for false-positive rate.
- Saves results to evaluation/subpipe_model_evaluation.json ONLY.
- Does NOT overwrite evaluation/metrics.json or evaluation/evaluation_results.json.

Run:
    .\\train_env\\Scripts\\python.exe scripts\\evaluate_on_subpipe.py
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CHECKPOINT = ROOT / "models" / "best_multiclass.pt"
SUBPIPE_ROOT = Path(r"data\subpipe\SubPipeMini2\SubPipeMiniSSS")
DATA_DIR = SUBPIPE_ROOT / "DATA"
OUTPUT_JSON = ROOT / "evaluation" / "subpipe_model_evaluation.json"

# SubPipe YOLO class 0 = "Pipeline"
# best_multiclass.pt class 1 = "submarine_pipeline"
# This is the authoritative cross-dataset remapping.
SUBPIPE_CLASS_TO_MODEL_CLASS = {
    0: 1  # Pipeline → submarine_pipeline in drishti taxonomy
}

# IoU threshold for TP/FP matching
IOU_THRESHOLD = 0.50
CONF_THRESHOLD = 0.25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pbm_to_bgr(path: Path) -> np.ndarray:
    """Read a PBM sonar image into a standard BGR numpy array (in-memory only)."""
    img = cv2.imread(str(path))
    if img is None:
        # Try PIL as fallback
        from PIL import Image
        pil_img = Image.open(str(path))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def parse_yolo_labels(lbl_path: Path):
    """Return list of (class_id, cx, cy, w, h) tuples from a YOLO .txt label."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                               float(parts[3]), float(parts[4])))
            except ValueError:
                pass
    return boxes


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized coords to absolute xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def iou(box_a, box_b):
    """Compute IoU between two xyxy boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_split(name, img_dir, lbl_dir, model, annotated_stems, unannotated_stems):
    """
    Evaluate model on annotated and background images.

    Returns dict with per-split metrics, TP/FP/FN, FP-on-background rate.
    """
    from ultralytics import YOLO

    total_TP = 0
    total_FP = 0
    total_FN = 0
    latencies_ms = []
    per_image_results = []

    # For AP computation
    all_scores = []   # confidence of each prediction
    all_matches = []  # 1 if TP, 0 if FP

    total_gt_boxes = 0

    print(f"\n  [{name}] Evaluating {len(annotated_stems)} annotated images...")

    for stem in annotated_stems:
        img_path = img_dir / (stem + ".pbm")
        lbl_path = lbl_dir / (stem + ".txt")

        if not img_path.exists():
            # Try .bpm extension (observed in inspection)
            img_path_alt = img_dir / (stem + ".bpm")
            if img_path_alt.exists():
                img_path = img_path_alt
            else:
                continue

        # Load image in-memory
        img_bgr = read_pbm_to_bgr(img_path)
        if img_bgr is None:
            continue
        img_h, img_w = img_bgr.shape[:2]

        # Parse ground truth (SubPipe class 0 = Pipeline)
        gt_raw = parse_yolo_labels(lbl_path)
        # Only keep Pipeline class (0), remap to model class 1
        gt_boxes_xyxy = []
        for cls_id, cx, cy, w, h in gt_raw:
            if cls_id == 0:  # Pipeline
                gt_boxes_xyxy.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
        total_gt_boxes += len(gt_boxes_xyxy)

        if not gt_boxes_xyxy:
            # Image was labeled but all labels are non-Pipeline → skip
            continue

        # Run inference
        t0 = time.perf_counter()
        results = model.predict(
            img_bgr,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=0 if hasattr(model, 'device') else 'cpu'
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        # Extract predictions for class 1 (submarine_pipeline = Pipeline in cross-dataset mapping)
        pred_boxes = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                if cls == 1:  # submarine_pipeline
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    pred_boxes.append((conf, x1, y1, x2, y2))

        # Sort by confidence descending
        pred_boxes.sort(key=lambda x: -x[0])

        # Match predictions to GTs (greedy IoU matching)
        matched_gt = set()
        for conf, px1, py1, px2, py2 in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(gt_boxes_xyxy):
                if idx in matched_gt:
                    continue
                iou_val = iou((px1, py1, px2, py2), gt_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx

            if best_iou >= IOU_THRESHOLD and best_idx >= 0:
                total_TP += 1
                matched_gt.add(best_idx)
                all_scores.append(conf)
                all_matches.append(1)
            else:
                total_FP += 1
                all_scores.append(conf)
                all_matches.append(0)

        total_FN += len(gt_boxes_xyxy) - len(matched_gt)
        per_image_results.append({
            "stem": stem,
            "gt_boxes": len(gt_boxes_xyxy),
            "pred_boxes": len(pred_boxes),
            "latency_ms": round(latency_ms, 1)
        })

    # Background (unannotated) images — check for false positives
    print(f"  [{name}] Evaluating {len(unannotated_stems)} background images for FP rate...")
    bg_images_with_fp = 0
    bg_total_fps = 0
    bg_latencies_ms = []

    for stem in unannotated_stems:
        img_path = img_dir / (stem + ".pbm")
        if not img_path.exists():
            img_path_alt = img_dir / (stem + ".bpm")
            if not img_path_alt.exists():
                continue
            img_path = img_path_alt

        img_bgr = read_pbm_to_bgr(img_path)
        if img_bgr is None:
            continue

        t0 = time.perf_counter()
        results = model.predict(img_bgr, conf=CONF_THRESHOLD, verbose=False)
        bg_latencies_ms.append((time.perf_counter() - t0) * 1000)

        fp_count = 0
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                if int(boxes.cls[i].item()) == 1:  # submarine_pipeline detections on background
                    fp_count += 1

        if fp_count > 0:
            bg_images_with_fp += 1
        bg_total_fps += fp_count

    # Compute precision/recall/F1
    precision = total_TP / max(total_TP + total_FP, 1)
    recall = total_TP / max(total_TP + total_FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Compute AP@50 using area under P-R curve (11-point interpolation)
    ap50 = compute_ap(all_scores, all_matches, total_gt_boxes)

    # Latency stats
    lat_all = latencies_ms + bg_latencies_ms
    lat_stats = {}
    if lat_all:
        lat_stats = {
            "mean_ms": round(float(np.mean(lat_all)), 2),
            "median_ms": round(float(np.median(lat_all)), 2),
            "min_ms": round(float(np.min(lat_all)), 2),
            "max_ms": round(float(np.max(lat_all)), 2),
            "p95_ms": round(float(np.percentile(lat_all, 95)), 2),
        }

    # FP rate on background
    fp_rate_bg = bg_images_with_fp / max(len(unannotated_stems), 1)

    return {
        "name": name,
        "annotated_images_evaluated": len(per_image_results),
        "background_images_evaluated": len(unannotated_stems),
        "total_gt_boxes": total_gt_boxes,
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "ap50": round(ap50, 4),
        "ap50_percent": round(ap50 * 100, 2),
        "background_images_with_any_FP": bg_images_with_fp,
        "background_total_FP_detections": bg_total_fps,
        "false_positive_rate_on_backgrounds": round(fp_rate_bg, 4),
        "false_positive_rate_pct": round(fp_rate_bg * 100, 2),
        "latency": lat_stats,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
    }


def compute_ap(scores, matches, n_gt):
    """
    Compute AP@50 using sorted detections and 101-point interpolation.
    scores: list of confidences
    matches: list of 1 (TP) or 0 (FP)
    n_gt: total number of ground truth boxes
    """
    if not scores or n_gt == 0:
        return 0.0

    # Sort by descending score
    sorted_indices = np.argsort(-np.array(scores))
    matches_sorted = np.array(matches)[sorted_indices]

    tp_cumsum = np.cumsum(matches_sorted)
    fp_cumsum = np.cumsum(1 - matches_sorted)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recalls = tp_cumsum / (n_gt + 1e-8)

    # Add sentinels
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [recalls[-1] if len(recalls) > 0 else 0.0]))

    # Monotonically decreasing precision
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_thresholds:
        prec_at_r = precisions[recalls >= r]
        if len(prec_at_r) > 0:
            ap += np.max(prec_at_r)
    return ap / 101.0


def main():
    print("=" * 70)
    print("SUBPIPE MODEL EVALUATION")
    print("Checkpoint: models/best_multiclass.pt")
    print("Dataset:    SubPipeMini2 / SubPipeMiniSSS")
    print("=" * 70)

    # ── Step 1: Checkpoint Verification ──────────────────────────────────────
    if not CHECKPOINT.exists():
        print(f"FATAL: Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    chk_sha256 = sha256(CHECKPOINT)
    chk_abs = str(CHECKPOINT.resolve())
    chk_size_mb = round(CHECKPOINT.stat().st_size / 1024**2, 3)

    print(f"\nCheckpoint Verification:")
    print(f"  Absolute path: {chk_abs}")
    print(f"  SHA-256:       {chk_sha256}")
    print(f"  Size (MB):     {chk_size_mb}")

    # ── Step 2: Load Model ────────────────────────────────────────────────────
    from ultralytics import YOLO
    model = YOLO(str(CHECKPOINT))
    print(f"\nModel loaded:")
    print(f"  Names: {model.names}")
    print(f"  nc:    {len(model.names)}")

    # Confirm class 1 is submarine_pipeline
    assert model.names.get(1) == "submarine_pipeline", \
        f"Expected class 1 = submarine_pipeline, got {model.names.get(1)}"
    print(f"  ✅ Class 1 confirmed: submarine_pipeline")
    print(f"  Cross-dataset mapping: SubPipe class 0 (Pipeline) → model class 1 (submarine_pipeline)")

    # ── Step 3: Collect image/label stems ─────────────────────────────────────
    def gather_stems(img_dir, lbl_dir):
        img_stems = {p.stem for p in img_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in (".pbm", ".bpm")}
        lbl_stems = {p.stem for p in lbl_dir.iterdir()
                     if p.is_file() and p.suffix == ".txt"
                     and p.stem.lower() not in ("classes",)}
        annotated = sorted(img_stems & lbl_stems)
        unannotated = sorted(img_stems - lbl_stems)
        return annotated, unannotated

    hf_img = DATA_DIR / "SSS_HF_images" / "Image"
    hf_lbl = DATA_DIR / "SSS_HF_images" / "YOLO_Annotation"
    lf_img = DATA_DIR / "SSS_LF_images" / "Image"
    lf_lbl = DATA_DIR / "SSS_LF_images" / "YOLO_Annotation"

    hf_annotated, hf_unannotated = gather_stems(hf_img, hf_lbl)
    lf_annotated, lf_unannotated = gather_stems(lf_img, lf_lbl)

    print(f"\nSplit Summary:")
    print(f"  HF annotated: {len(hf_annotated)}, background: {len(hf_unannotated)}")
    print(f"  LF annotated: {len(lf_annotated)}, background: {len(lf_unannotated)}")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    hf_results = evaluate_split("HF_900kHz", hf_img, hf_lbl, model, hf_annotated, hf_unannotated)
    lf_results = evaluate_split("LF_455kHz", lf_img, lf_lbl, model, lf_annotated, lf_unannotated)

    # ── Step 5: Combined summary ──────────────────────────────────────────────
    comb_TP = hf_results["TP"] + lf_results["TP"]
    comb_FP = hf_results["FP"] + lf_results["FP"]
    comb_FN = hf_results["FN"] + lf_results["FN"]
    comb_GT = hf_results["total_gt_boxes"] + lf_results["total_gt_boxes"]
    comb_prec = comb_TP / max(comb_TP + comb_FP, 1)
    comb_rec = comb_TP / max(comb_TP + comb_FN, 1)
    comb_f1 = 2 * comb_prec * comb_rec / max(comb_prec + comb_rec, 1e-8)
    # Simple average of split APs
    comb_ap50 = (hf_results["ap50"] + lf_results["ap50"]) / 2

    comb_bg_fp = hf_results["background_images_with_any_FP"] + lf_results["background_images_with_any_FP"]
    comb_bg_total = len(hf_unannotated) + len(lf_unannotated)
    comb_fp_rate = comb_bg_fp / max(comb_bg_total, 1)

    combined = {
        "total_annotated_images": len(hf_annotated) + len(lf_annotated),
        "total_background_images": comb_bg_total,
        "total_gt_boxes": comb_GT,
        "TP": comb_TP,
        "FP": comb_FP,
        "FN": comb_FN,
        "precision": round(comb_prec, 4),
        "recall": round(comb_rec, 4),
        "f1": round(comb_f1, 4),
        "ap50_mean": round(comb_ap50, 4),
        "ap50_mean_pct": round(comb_ap50 * 100, 2),
        "background_images_with_any_FP": comb_bg_fp,
        "false_positive_rate_on_backgrounds": round(comb_fp_rate, 4),
        "false_positive_rate_pct": round(comb_fp_rate * 100, 2),
    }

    full_report = {
        "evaluation_name": "SubPipeMini2 / best_multiclass.pt",
        "checkpoint": {
            "absolute_path": chk_abs,
            "sha256": chk_sha256,
            "size_mb": chk_size_mb,
        },
        "model_class_names": model.names,
        "cross_dataset_mapping": {
            "subpipe_class_0_Pipeline": "→ model class 1 (submarine_pipeline)",
            "notes": "SubPipe has 1 class (Pipeline/class 0). Model class 1 = submarine_pipeline. Only class-1 predictions are compared."
        },
        "dataset": {
            "root": str(SUBPIPE_ROOT),
            "hf_image_resolution": "5000x500 px (PBM, 10:1 aspect)",
            "lf_image_resolution": "2500x500 px (PBM, 5:1 aspect)",
        },
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "combined": combined,
        "hf_sonar_900khz": hf_results,
        "lf_sonar_455khz": lf_results,
    }

    # ── Step 6: Save — never touch metrics.json or evaluation_results.json ────
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # ── Step 7: Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUBPIPE MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n  Checkpoint: {chk_abs}")
    print(f"  SHA-256:    {chk_sha256}")
    print(f"\n  COMBINED (HF + LF):")
    print(f"    Annotated images evaluated: {combined['total_annotated_images']}")
    print(f"    Background images tested:   {combined['total_background_images']}")
    print(f"    Ground truth boxes:         {combined['total_gt_boxes']}")
    print(f"    TP: {combined['TP']}  FP: {combined['FP']}  FN: {combined['FN']}")
    print(f"    Precision:     {combined['precision']:.4f} ({combined['precision']*100:.1f}%)")
    print(f"    Recall:        {combined['recall']:.4f}    ({combined['recall']*100:.1f}%)")
    print(f"    F1:            {combined['f1']:.4f}")
    print(f"    mAP@50 (mean): {combined['ap50_mean']:.4f} ({combined['ap50_mean_pct']:.1f}%)")
    print(f"    FP rate (BG):  {combined['false_positive_rate_pct']:.1f}% ({combined['background_images_with_any_FP']}/{combined['total_background_images']} background images triggered a FP)")
    print(f"\n  HF (900 kHz):")
    print(f"    TP={hf_results['TP']} FP={hf_results['FP']} FN={hf_results['FN']}")
    print(f"    Precision={hf_results['precision']:.4f}  Recall={hf_results['recall']:.4f}  AP50={hf_results['ap50']:.4f}")
    print(f"    FP rate (BG): {hf_results['false_positive_rate_pct']:.1f}%")
    if hf_results['latency']:
        print(f"    Latency mean: {hf_results['latency']['mean_ms']:.1f}ms  median: {hf_results['latency']['median_ms']:.1f}ms")
    print(f"\n  LF (455 kHz):")
    print(f"    TP={lf_results['TP']} FP={lf_results['FP']} FN={lf_results['FN']}")
    print(f"    Precision={lf_results['precision']:.4f}  Recall={lf_results['recall']:.4f}  AP50={lf_results['ap50']:.4f}")
    print(f"    FP rate (BG): {lf_results['false_positive_rate_pct']:.1f}%")
    if lf_results['latency']:
        print(f"    Latency mean: {lf_results['latency']['mean_ms']:.1f}ms  median: {lf_results['latency']['median_ms']:.1f}ms")
    print(f"\n  Saved: {OUTPUT_JSON}")
    print("=" * 70)
    print("\nNo training or fine-tuning performed. Evaluation only.")


if __name__ == "__main__":
    main()
