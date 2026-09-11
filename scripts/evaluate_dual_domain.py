"""
Dual-Domain Model Evaluation Script
====================================
Evaluates both baseline (models/best_multiclass.pt) and combined model (models/best_combined.pt)
across both domains without modifying any original dataset or overwriting existing baseline logs:

Domain A: Drishti-SSS 700-image held-out test set
  - Baseline -> evaluation/evaluation_results_before_combined.json
  - Combined -> evaluation/evaluation_results_combined_model.json

Domain B: SubPipe held-out source images (aspect-ratio-aware tiled inference & stitching)
  - Baseline -> evaluation/subpipe_model_evaluation_before_combined.json
  - Combined -> evaluation/subpipe_model_evaluation_combined.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DRISHTI_YAML = ROOT / "data" / "drishti_sss" / "dataset.yaml"
SUBPIPE_DATA = Path(r"data\subpipe\SubPipeMini2\SubPipeMiniSSS\DATA")
SUBPIPE_MANIFEST = ROOT / "data" / "subpipe_tiled" / "manifests" / "tiling_manifest.json"

BASE_MODEL_PATH = ROOT / "models" / "best_multiclass.pt"
COMBINED_MODEL_PATH = ROOT / "models" / "best_combined.pt"
EVAL_DIR = ROOT / "evaluation"

CLASS_NAMES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder",
}

TILE_WIDTH = 640
TILE_HEIGHT = 500
STRIDE_HF = 480
STRIDE_LF = 465
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.45


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pbm_to_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        from PIL import Image
        pil_img = Image.open(str(path))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def parse_yolo_labels(lbl_path: Path):
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
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def box_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def nms(boxes, iou_thresh):
    """boxes: list of (conf, x1, y1, x2, y2) sorted by conf desc."""
    if not boxes:
        return []
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        boxes = [b for b in boxes if box_iou(current[1:], b[1:]) < iou_thresh]
    return keep


def get_tile_ranges(img_width: int, tile_width: int, target_stride: int):
    ranges = []
    curr_x = 0
    while curr_x + tile_width <= img_width:
        ranges.append((curr_x, curr_x + tile_width))
        curr_x += target_stride
    if ranges[-1][1] < img_width:
        ranges.append((img_width - tile_width, img_width))
    return ranges


def evaluate_drishti_test(model_path: Path, output_json: Path):
    print(f"\nEvaluating {model_path.name} on Drishti-SSS Test Set...")
    model = YOLO(str(model_path))
    device = "0" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    metrics = model.val(
        data=str(DRISHTI_YAML),
        split="test",
        batch=16,
        imgsz=640,
        device=device,
        plots=False,
        verbose=False,
    )
    total_val_time = (time.perf_counter() - t0)

    # Benchmark single-image latency
    test_img_dir = ROOT / "data" / "drishti_sss" / "test" / "images"
    sample_imgs = list(test_img_dir.glob("*.png"))[:50]
    latencies = []
    if sample_imgs:
        for s_img in sample_imgs:
            im = cv2.imread(str(s_img))
            if im is not None:
                t_start = time.perf_counter()
                model.predict(im, conf=CONF_THRESHOLD, device=device, verbose=False)
                latencies.append((time.perf_counter() - t_start) * 1000)

    avg_latency_ms = round(float(np.mean(latencies)), 2) if latencies else 0.0

    class_indices = metrics.box.ap_class_index if hasattr(metrics.box, "ap_class_index") else []
    per_class = {}
    for idx, c_idx in enumerate(class_indices):
        c_name = CLASS_NAMES.get(int(c_idx), f"class_{c_idx}")
        p = float(metrics.box.p[idx]) if idx < len(metrics.box.p) else 0.0
        r = float(metrics.box.r[idx]) if idx < len(metrics.box.r) else 0.0
        ap50 = float(metrics.box.ap50[idx]) if idx < len(metrics.box.ap50) else 0.0
        ap50_95 = float(metrics.box.ap[idx]) if idx < len(metrics.box.ap) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class[c_name] = {
            "class_id": int(c_idx),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "ap50": round(ap50, 4),
            "ap50_95": round(ap50_95, 4),
        }

    overall_p = float(metrics.box.mp)
    overall_r = float(metrics.box.mr)
    overall_map50 = float(metrics.box.map50)
    overall_map50_95 = float(metrics.box.map)
    overall_f1 = (2 * overall_p * overall_r / (overall_p + overall_r)) if (overall_p + overall_r) > 0 else 0.0

    result = {
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "dataset": "data/drishti_sss (700 test images)",
        "metrics": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f1, 4),
            "map50": round(overall_map50, 4),
            "map50_95": round(overall_map50_95, 4),
            "per_class": per_class,
        },
        "latency_ms_per_image": avg_latency_ms,
        "evaluation_duration_sec": round(total_val_time, 2),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved Drishti test results to: {output_json}")
    return result


def evaluate_subpipe_tiled(model_path: Path, output_json: Path):
    print(f"\nEvaluating {model_path.name} on SubPipe Held-out Test Source Images (Tiled Inference)...")
    model = YOLO(str(model_path))
    device = "0" if torch.cuda.is_available() else "cpu"

    if not SUBPIPE_MANIFEST.exists():
        print(f"FATAL: Manifest not found: {SUBPIPE_MANIFEST}")
        sys.exit(1)

    with open(SUBPIPE_MANIFEST, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    test_entries = [e for e in manifest_data.get("manifest", []) if e.get("split") == "test"]
    # Group test source images
    test_sources = {}
    for e in test_entries:
        key = (e["domain"], e["source_stem"])
        if key not in test_sources:
            test_sources[key] = {
                "domain": e["domain"],
                "source_stem": e["source_stem"],
                "orig_dims": e["orig_dims"],
            }

    print(f"  Total held-out test source images: {len(test_sources)}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_boxes = 0

    bg_total_images = 0
    bg_images_with_fp = 0
    bg_total_fps = 0

    image_latencies_ms = []

    for (domain, stem), info in test_sources.items():
        if domain == "HF":
            img_dir = SUBPIPE_DATA / "SSS_HF_images" / "Image"
            lbl_dir = SUBPIPE_DATA / "SSS_HF_images" / "YOLO_Annotation"
            stride = STRIDE_HF
        else:
            img_dir = SUBPIPE_DATA / "SSS_LF_images" / "Image"
            lbl_dir = SUBPIPE_DATA / "SSS_LF_images" / "YOLO_Annotation"
            stride = STRIDE_LF

        img_path = img_dir / f"{stem}.pbm"
        if not img_path.exists():
            img_path = img_dir / f"{stem}.bpm"
        lbl_path = lbl_dir / f"{stem}.txt"

        if not img_path.exists():
            continue

        img_bgr = read_pbm_to_bgr(img_path)
        if img_bgr is None:
            continue
        img_h, img_w = img_bgr.shape[:2]

        # Load GT boxes (SubPipe class 0 Pipeline -> model class 1)
        raw_gt = parse_yolo_labels(lbl_path)
        gt_boxes = []
        for cls_id, cx, cy, w, h in raw_gt:
            if cls_id == 0:
                gt_boxes.append(yolo_to_xyxy(cx, cy, w, h, img_w, img_h))

        is_background = (len(gt_boxes) == 0)
        if is_background:
            bg_total_images += 1
        else:
            total_gt_boxes += len(gt_boxes)

        # Aspect-ratio-aware tiled inference
        t_start = time.perf_counter()
        tile_ranges = get_tile_ranges(img_w, TILE_WIDTH, stride)
        stitched_preds = []

        for x_start, x_end in tile_ranges:
            tile_crop = img_bgr[0:TILE_HEIGHT, x_start:x_end]
            res = model.predict(tile_crop, conf=CONF_THRESHOLD, device=device, verbose=False)
            if res and res[0].boxes is not None:
                boxes = res[0].boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    if cls == 1:  # submarine_pipeline
                        tx1, ty1, tx2, ty2 = boxes.xyxy[i].tolist()
                        gx1 = tx1 + x_start
                        gy1 = ty1
                        gx2 = tx2 + x_start
                        gy2 = ty2
                        stitched_preds.append((conf, gx1, gy1, gx2, gy2))

        # Apply Non-Maximum Suppression (NMS) to fuse detections across tiles
        stitched_preds.sort(key=lambda x: -x[0])
        final_preds = nms(stitched_preds, NMS_IOU_THRESHOLD)
        img_lat = (time.perf_counter() - t_start) * 1000
        image_latencies_ms.append(img_lat)

        if is_background:
            if len(final_preds) > 0:
                bg_images_with_fp += 1
                bg_total_fps += len(final_preds)
        else:
            matched_gt = set()
            for conf, px1, py1, px2, py2 in final_preds:
                best_iou = 0.0
                best_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if idx in matched_gt:
                        continue
                    iou_val = box_iou((px1, py1, px2, py2), gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_idx = idx
                if best_iou >= IOU_THRESHOLD and best_idx >= 0:
                    total_tp += 1
                    matched_gt.add(best_idx)
                else:
                    total_fp += 1
            total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bg_fp_rate = (bg_images_with_fp / bg_total_images * 100.0) if bg_total_images > 0 else 0.0
    avg_latency = float(np.mean(image_latencies_ms)) if image_latencies_ms else 0.0

    result = {
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "dataset": "SubPipe Held-out Test Source Images (Aspect-Ratio-Aware Tiled Inference)",
        "test_source_images": len(test_sources),
        "annotated_gt_boxes": total_gt_boxes,
        "metrics": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "background_evaluation": {
            "total_background_images": bg_total_images,
            "background_images_with_fp": bg_images_with_fp,
            "total_false_positives_on_background": bg_total_fps,
            "background_fp_rate_pct": round(bg_fp_rate, 2),
        },
        "latency_ms_per_full_sonar_image": round(avg_latency, 2),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved SubPipe test results to: {output_json}")
    return result


def main():
    print("=" * 70)
    print("DUAL-DOMAIN EVALUATION: BASELINE VS COMBINED MODEL")
    print("=" * 70)

    # 1. Evaluate baseline model
    if BASE_MODEL_PATH.exists():
        print(f"\n--- BASELINE MODEL: {BASE_MODEL_PATH} ---")
        evaluate_drishti_test(BASE_MODEL_PATH, EVAL_DIR / "evaluation_results_before_combined.json")
        evaluate_subpipe_tiled(BASE_MODEL_PATH, EVAL_DIR / "subpipe_model_evaluation_before_combined.json")
    else:
        print(f"WARNING: Baseline model not found at {BASE_MODEL_PATH}")

    # 2. Evaluate combined model
    if COMBINED_MODEL_PATH.exists():
        print(f"\n--- COMBINED MODEL: {COMBINED_MODEL_PATH} ---")
        evaluate_drishti_test(COMBINED_MODEL_PATH, EVAL_DIR / "evaluation_results_combined_model.json")
        evaluate_subpipe_tiled(COMBINED_MODEL_PATH, EVAL_DIR / "subpipe_model_evaluation_combined.json")
    else:
        print(f"WARNING: Combined model not found at {COMBINED_MODEL_PATH}")


if __name__ == "__main__":
    main()
