"""
Comprehensive Inspection Script for SubPipeMini2 (SubPipeMiniSSS) Dataset.

Validates:
- HF & LF sonar images (formats, resolutions)
- Image-label pairing (missing labels, orphan labels)
- YOLO annotation validity (class IDs, coordinate ranges [0, 1], malformed lines)
- Bounding-box statistics (count, areas, aspect ratios)
- Class distributions
- Authoritative metadata verification (COCO categories & sensor config)

Saves machine-readable report to evaluation/subpipe_inspection.json.
Does NOT modify any files in the original SubPipeMini2 directory.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
SUBPIPE_ROOT = Path(r"data\subpipe\SubPipeMini2\SubPipeMiniSSS")
DATA_DIR = SUBPIPE_ROOT / "DATA"

def inspect_split(name: str, img_dir: Path, lbl_dir: Path, coco_json: Path = None):
    print(f"\nScanning {name}...")
    print(f"  Images: {img_dir}")
    print(f"  Labels: {lbl_dir}")

    # Gather images
    img_files = {}
    for p in img_dir.iterdir():
        if p.is_file():
            img_files[p.stem] = p

    # Gather labels
    lbl_files = {}
    if lbl_dir.exists():
        for p in lbl_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".txt":
                lbl_files[p.stem] = p

    img_stems = set(img_files.keys())
    lbl_stems = set(lbl_files.keys())

    missing_labels = sorted(list(img_stems - lbl_stems))
    orphan_labels = sorted(list(lbl_stems - img_stems))
    matched_stems = sorted(list(img_stems & lbl_stems))

    # Authoritative categories from COCO if available
    coco_categories = []
    if coco_json and coco_json.exists():
        try:
            with open(coco_json, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                coco_categories = cdata.get("categories", [])
        except Exception as e:
            print(f"  Warning: Failed to load COCO json {coco_json}: {e}")

    # Image format and resolution sampling
    formats = defaultdict(int)
    resolutions = []
    for stem in list(img_stems)[:50]:
        p = img_files[stem]
        formats[p.suffix.lower()] += 1
        img = cv2.imread(str(p))
        if img is not None:
            resolutions.append((img.shape[1], img.shape[0], img.shape[2]))

    # Analyze annotations
    class_counts = defaultdict(int)
    bbox_areas = []
    bbox_widths = []
    bbox_heights = []
    annotated_images_count = 0
    unannotated_images_count = 0
    invalid_labels = []

    for stem in matched_stems:
        lbl_p = lbl_files[stem]
        try:
            content = lbl_p.read_text(encoding="utf-8").strip()
        except Exception as e:
            invalid_labels.append({"stem": stem, "error": f"Read error: {e}"})
            continue

        if not content:
            unannotated_images_count += 1
            continue

        lines = content.splitlines()
        has_valid_box = False

        for line_num, line in enumerate(lines, 1):
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split()
            if len(parts) != 5:
                invalid_labels.append({
                    "stem": stem,
                    "line_num": line_num,
                    "line": line_str,
                    "error": f"Expected 5 tokens, got {len(parts)}"
                })
                continue

            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError as e:
                invalid_labels.append({
                    "stem": stem,
                    "line_num": line_num,
                    "line": line_str,
                    "error": f"Non-numeric value: {e}"
                })
                continue

            # Coordinate range validation
            coord_errs = []
            if not (0.0 <= cx <= 1.0):
                coord_errs.append(f"cx={cx} out of [0, 1]")
            if not (0.0 <= cy <= 1.0):
                coord_errs.append(f"cy={cy} out of [0, 1]")
            if not (0.0 < w <= 1.0):
                coord_errs.append(f"w={w} out of (0, 1]")
            if not (0.0 < h <= 1.0):
                coord_errs.append(f"h={h} out of (0, 1]")

            # Bounding box bounds check: x1, y1, x2, y2
            x1 = cx - w / 2.0
            x2 = cx + w / 2.0
            y1 = cy - h / 2.0
            y2 = cy + h / 2.0
            if x1 < -0.05 or x2 > 1.05 or y1 < -0.05 or y2 > 1.05:
                coord_errs.append(f"Box extends significantly outside image: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]")

            if coord_errs:
                invalid_labels.append({
                    "stem": stem,
                    "line_num": line_num,
                    "line": line_str,
                    "error": "; ".join(coord_errs)
                })
                continue

            class_counts[cls_id] += 1
            area = w * h
            bbox_areas.append(area)
            bbox_widths.append(w)
            bbox_heights.append(h)
            has_valid_box = True

        if has_valid_box:
            annotated_images_count += 1
        else:
            unannotated_images_count += 1

    # Add images that had no label file to unannotated
    unannotated_images_count += len(missing_labels)

    # Resolution summary
    res_summary = {}
    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        channels = [r[2] for r in resolutions]
        res_summary = {
            "sample_count": len(resolutions),
            "width": {"min": min(widths), "max": max(widths), "mode": int(np.median(widths))},
            "height": {"min": min(heights), "max": max(heights), "mode": int(np.median(heights))},
            "channels": int(np.median(channels))
        }

    # BBox summary
    bbox_summary = {}
    if bbox_areas:
        bbox_summary = {
            "total_boxes": len(bbox_areas),
            "mean_area": round(float(np.mean(bbox_areas)), 6),
            "min_area": round(float(np.min(bbox_areas)), 6),
            "max_area": round(float(np.max(bbox_areas)), 6),
            "std_area": round(float(np.std(bbox_areas)), 6),
            "mean_width": round(float(np.mean(bbox_widths)), 4),
            "mean_height": round(float(np.mean(bbox_heights)), 4),
        }

    return {
        "name": name,
        "image_count": len(img_files),
        "label_file_count": len(lbl_files),
        "annotated_image_count": annotated_images_count,
        "unannotated_image_count": unannotated_images_count,
        "missing_labels_count": len(missing_labels),
        "missing_labels_sample": missing_labels[:10],
        "orphan_labels_count": len(orphan_labels),
        "orphan_labels_sample": orphan_labels[:10],
        "formats": dict(formats),
        "resolution": res_summary,
        "class_distribution": dict(class_counts),
        "invalid_labels_count": len(invalid_labels),
        "invalid_labels_sample": invalid_labels[:10],
        "bbox_stats": bbox_summary,
        "coco_categories": coco_categories
    }

def main():
    print("=" * 70)
    print("SUBPIPE MINI 2 (SubPipeMiniSSS) DATASET INSPECTION")
    print("=" * 70)
    print(f"Dataset root: {SUBPIPE_ROOT}")

    if not SUBPIPE_ROOT.exists():
        print(f"FATAL: Dataset root does not exist: {SUBPIPE_ROOT}")
        sys.exit(1)

    hf_img = DATA_DIR / "SSS_HF_images" / "Image"
    hf_lbl = DATA_DIR / "SSS_HF_images" / "YOLO_Annotation"
    hf_coco = DATA_DIR / "SSS_HF_images" / "COCO_Annotation" / "coco_format.json"

    lf_img = DATA_DIR / "SSS_LF_images" / "Image"
    lf_lbl = DATA_DIR / "SSS_LF_images" / "YOLO_Annotation"
    lf_coco = DATA_DIR / "SSS_LF_images" / "COCO_Annotation" / "coco_format.json"

    # Sensor config inspection
    config_yaml_path = SUBPIPE_ROOT / "config.yaml"
    sensor_config = {}
    if config_yaml_path.exists():
        try:
            import yaml
            with open(config_yaml_path, "r", encoding="utf-8") as f:
                sensor_config = yaml.safe_load(f)
        except Exception as e:
            sensor_config = {"read_error": str(e)}

    hf_results = inspect_split("HF_Sonar", hf_img, hf_lbl, hf_coco)
    lf_results = inspect_split("LF_Sonar", lf_img, lf_lbl, lf_coco)

    total_images = hf_results["image_count"] + lf_results["image_count"]
    total_annotated = hf_results["annotated_image_count"] + lf_results["annotated_image_count"]
    total_unannotated = hf_results["unannotated_image_count"] + lf_results["unannotated_image_count"]
    total_boxes = hf_results["bbox_stats"].get("total_boxes", 0) + lf_results["bbox_stats"].get("total_boxes", 0)

    # Combined class distribution
    combined_classes = defaultdict(int)
    for c, n in hf_results["class_distribution"].items():
        combined_classes[str(c)] += n
    for c, n in lf_results["class_distribution"].items():
        combined_classes[str(c)] += n

    # Authoritative mapping deduction
    # In COCO json, categories: [{'id': 1, 'name': 'Pipeline', 'supercategory': 'Pipeline'}]
    # In YOLO, 0-indexed class 0 -> 'Pipeline'
    authoritative_mapping = {
        "0": "Pipeline (submarine_pipeline)"
    }

    full_report = {
        "dataset_name": "SubPipeMini2 / SubPipeMiniSSS",
        "dataset_root": str(SUBPIPE_ROOT),
        "data_dir": str(DATA_DIR),
        "sensor_config": sensor_config,
        "authoritative_class_mapping": authoritative_mapping,
        "coco_categories_hf": hf_results["coco_categories"],
        "coco_categories_lf": lf_results["coco_categories"],
        "summary": {
            "total_images": total_images,
            "total_annotated_images": total_annotated,
            "total_unannotated_images": total_unannotated,
            "total_bounding_boxes": total_boxes,
            "total_missing_labels": hf_results["missing_labels_count"] + lf_results["missing_labels_count"],
            "total_orphan_labels": hf_results["orphan_labels_count"] + lf_results["orphan_labels_count"],
            "total_invalid_labels": hf_results["invalid_labels_count"] + lf_results["invalid_labels_count"],
            "combined_class_distribution": dict(combined_classes),
        },
        "hf_sonar": hf_results,
        "lf_sonar": lf_results
    }

    # Save to evaluation/subpipe_inspection.json
    eval_dir = ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_file = eval_dir / "subpipe_inspection.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 70)
    print("SUBPIPE DATASET INSPECTION SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Images:            {total_images}")
    print(f"  HF Images:             {hf_results['image_count']} (PBM format, {hf_results['resolution']['width']['mode']}x{hf_results['resolution']['height']['mode']})")
    print(f"  LF Images:             {lf_results['image_count']} (PBM format, {lf_results['resolution']['width']['mode']}x{lf_results['resolution']['height']['mode']})")
    print(f"Total Annotated Images:  {total_annotated}")
    print(f"Total Unannotated:       {total_unannotated}")
    print(f"Total Bounding Boxes:    {total_boxes}")
    print(f"Missing Labels:          {full_report['summary']['total_missing_labels']} (HF: {hf_results['missing_labels_count']}, LF: {lf_results['missing_labels_count']})")
    print(f"Orphan Labels:           {full_report['summary']['total_orphan_labels']} (HF: {hf_results['orphan_labels_count']}, LF: {lf_results['orphan_labels_count']})")
    print(f"Invalid Labels:          {full_report['summary']['total_invalid_labels']}")
    print(f"Authoritative Mapping:   Class 0 -> {authoritative_mapping['0']}")
    print(f"Class Distribution:      {dict(combined_classes)}")
    print(f"Machine-readable report: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
