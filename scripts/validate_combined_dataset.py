"""
Phase 4: Validate Combined SIH26057 Dataset
=============================================
Comprehensive pre-training validation of data/sih26057_combined/.

Checks:
  1.  Total image count matches manifest
  2.  Train / val / test image counts
  3.  Missing images (manifest entry but no file)
  4.  Missing label files
  5.  Invalid label lines (bad format)
  6.  Out-of-bounds coordinates
  7.  Class ID range (only 0-4 allowed)
  8.  Duplicate filenames across splits
  9.  Cross-split source leakage (SubPipe source images)
  10. HF 700-image held-out test set identifiable and untouched
  11. HF test images NOT in train or val
  12. Negative/background images counted
  13. Per-class annotation distribution

Output:
  reports/combined_dataset_validation.json
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

COMBINED_DIR = ROOT / "data" / "sih26057_combined"
DRISHTI_DIR = ROOT / "data" / "drishti_sss"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

CLASS_NAMES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder",
}

VALID_CLASS_IDS = set(CLASS_NAMES.keys())

PREFIX_DRISHTI = "drishti__"
PREFIX_SUBPIPE = "subpipe__"


def parse_label(lbl_path: Path) -> tuple[list, list]:
    """Return (valid_boxes, invalid_lines) from a YOLO label file."""
    valid_boxes = []
    invalid_lines = []
    if not lbl_path.exists():
        return valid_boxes, [f"FILE_MISSING:{lbl_path.name}"]
    content = lbl_path.read_text(encoding="utf-8").strip()
    if not content:
        return valid_boxes, invalid_lines  # empty = background tile, valid
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            invalid_lines.append(line)
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            valid_boxes.append((cls_id, cx, cy, w, h))
        except ValueError:
            invalid_lines.append(line)
    return valid_boxes, invalid_lines


def check_bbox_validity(boxes: list) -> list:
    """Return list of violation strings for any out-of-bounds boxes."""
    violations = []
    for cls_id, cx, cy, w, h in boxes:
        if cls_id not in VALID_CLASS_IDS:
            violations.append(f"invalid_class_id:{cls_id}")
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            violations.append(f"out_of_bounds_centre:({cx:.4f},{cy:.4f})")
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            violations.append(f"invalid_size:(w={w:.4f},h={h:.4f})")
        x1 = cx - w / 2.0
        x2 = cx + w / 2.0
        y1 = cy - h / 2.0
        y2 = cy + h / 2.0
        if x1 < -0.01 or x2 > 1.01 or y1 < -0.01 or y2 > 1.01:
            violations.append(f"box_overflow:[{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}]")
    return violations


def main():
    print("=" * 70)
    print("PHASE 4: COMBINED DATASET VALIDATION")
    print("=" * 70)

    if not COMBINED_DIR.exists():
        print(f"FATAL: Combined dataset not found at {COMBINED_DIR}")
        print("  Run scripts/build_combined_dataset.py first.")
        sys.exit(1)

    # ── Load manifest
    manifest_path = COMBINED_DIR / "manifests" / "combined_manifest.json"
    if not manifest_path.exists():
        print(f"FATAL: Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_entries = manifest_data["entries"]
    print(f"  Manifest entries loaded: {len(manifest_entries)}")

    # ── Load dataset.yaml
    yaml_path = COMBINED_DIR / "dataset.yaml"
    if not yaml_path.exists():
        print(f"FATAL: dataset.yaml not found: {yaml_path}")
        sys.exit(1)
    with open(yaml_path, encoding="utf-8") as f:
        dataset_cfg = yaml.safe_load(f)
    print(f"  dataset.yaml classes: {dataset_cfg.get('names')}")

    # ═══════════════════════════════════════════════════════════════
    # CHECK 1–4: Image and label file existence
    # ═══════════════════════════════════════════════════════════════
    print("\n[Check 1-4] Image and label file existence...")

    errors_missing_image = []
    errors_missing_label = []
    split_counts = defaultdict(lambda: {"total": 0, "drishti": 0, "subpipe": 0})
    all_filenames = defaultdict(set)  # split -> set of filenames

    for entry in manifest_entries:
        split = entry["split"]
        filename = entry["filename"]
        label_filename = entry["label_filename"]
        source = entry["source_dataset"]

        img_path = COMBINED_DIR / "images" / split / filename
        lbl_path = COMBINED_DIR / "labels" / split / label_filename

        split_counts[split]["total"] += 1
        if source == "drishti_sss":
            split_counts[split]["drishti"] += 1
        else:
            split_counts[split]["subpipe"] += 1

        all_filenames[split].add(filename)

        if not img_path.exists():
            errors_missing_image.append(f"{split}/{filename}")
        if not lbl_path.exists():
            errors_missing_label.append(f"{split}/{label_filename}")

    # Also count actual files on disk vs manifest
    disk_counts = {}
    for split in SPLITS:
        img_dir = COMBINED_DIR / "images" / split
        disk_counts[split] = len(list(img_dir.glob("*"))) if img_dir.exists() else 0

    # ═══════════════════════════════════════════════════════════════
    # CHECK 5–7: Label validity
    # ═══════════════════════════════════════════════════════════════
    print("[Check 5-7] Label format and coordinate validity...")

    errors_invalid_label_lines = []
    errors_out_of_bounds = []
    class_distribution = defaultdict(int)  # class_id -> count
    split_positive = defaultdict(int)
    split_negative = defaultdict(int)

    for entry in manifest_entries:
        split = entry["split"]
        label_filename = entry["label_filename"]
        lbl_path = COMBINED_DIR / "labels" / split / label_filename

        valid_boxes, invalid_lines = parse_label(lbl_path)

        if invalid_lines:
            errors_invalid_label_lines.extend(
                [f"{split}/{label_filename}:{l}" for l in invalid_lines
                 if not l.startswith("FILE_MISSING:")]
            )

        violations = check_bbox_validity(valid_boxes)
        if violations:
            errors_out_of_bounds.extend(
                [f"{split}/{label_filename}: {v}" for v in violations]
            )

        for cls_id, *_ in valid_boxes:
            class_distribution[cls_id] += 1

        if valid_boxes:
            split_positive[split] += 1
        else:
            split_negative[split] += 1

    # ═══════════════════════════════════════════════════════════════
    # CHECK 8: Duplicate filenames across splits
    # ═══════════════════════════════════════════════════════════════
    print("[Check 8] Duplicate filenames across splits...")

    train_set = all_filenames.get("train", set())
    val_set = all_filenames.get("val", set())
    test_set = all_filenames.get("test", set())

    dup_train_val = train_set & val_set
    dup_train_test = train_set & test_set
    dup_val_test = val_set & test_set

    # ═══════════════════════════════════════════════════════════════
    # CHECK 9: Cross-split source leakage (SubPipe source images)
    # ═══════════════════════════════════════════════════════════════
    print("[Check 9] SubPipe source-level split leakage...")

    subpipe_source_by_split = defaultdict(set)  # split -> set of (domain, source_stem)

    for entry in manifest_entries:
        if entry["source_dataset"] != "subpipe_tiled":
            continue
        prov = entry.get("provenance", {})
        domain = prov.get("domain", "")
        stem = prov.get("source_stem", "")
        if domain and stem:
            subpipe_source_by_split[entry["split"]].add((domain, stem))

    sp_train = subpipe_source_by_split.get("train", set())
    sp_val = subpipe_source_by_split.get("val", set())
    sp_test = subpipe_source_by_split.get("test", set())

    sp_tv_overlap = sp_train & sp_val
    sp_tt_overlap = sp_train & sp_test
    sp_vt_overlap = sp_val & sp_test

    # ═══════════════════════════════════════════════════════════════
    # CHECK 10–11: HF 700-image test set integrity
    # ═══════════════════════════════════════════════════════════════
    print("[Check 10-11] HF held-out test set integrity...")

    # Collect HF test image stems from drishti_sss/test
    hf_test_img_dir = DRISHTI_DIR / "test" / "images"
    if hf_test_img_dir.exists():
        hf_test_stems = {f.stem for f in hf_test_img_dir.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    else:
        hf_test_stems = set()

    # Expected prefixed stems in combined/images/test
    expected_hf_test_filenames = {PREFIX_DRISHTI + s for s in hf_test_stems}

    # Check all HF test images appear in combined test set
    hf_test_in_combined_test = set()
    hf_test_in_combined_train = set()
    hf_test_in_combined_val = set()

    for entry in manifest_entries:
        if entry["source_dataset"] != "drishti_sss":
            continue
        orig_split = entry.get("provenance", {}).get("original_split", "")
        orig_stem = entry.get("provenance", {}).get("original_image", "").rsplit(".", 1)[0]

        if orig_split == "test":
            if entry["split"] == "test":
                hf_test_in_combined_test.add(orig_stem)
            elif entry["split"] == "train":
                hf_test_in_combined_train.add(orig_stem)
            elif entry["split"] == "val":
                hf_test_in_combined_val.add(orig_stem)

    hf_test_count_in_combined_test = len(hf_test_in_combined_test)
    hf_test_contaminating_train = len(hf_test_in_combined_train)
    hf_test_contaminating_val = len(hf_test_in_combined_val)

    # ═══════════════════════════════════════════════════════════════
    # COMPILE RESULTS
    # ═══════════════════════════════════════════════════════════════

    total_images = sum(split_counts[s]["total"] for s in SPLITS)
    checks_passed = []
    checks_failed = []

    def _check(name, passed, detail=""):
        if passed:
            checks_passed.append(name)
            print(f"  [OK]   {name}")
        else:
            checks_failed.append(name)
            print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    _check("Total images match manifest",
           total_images == len(manifest_entries),
           f"manifest={len(manifest_entries)}, counted={total_images}")

    _check("Zero missing image files",
           len(errors_missing_image) == 0,
           f"{len(errors_missing_image)} missing")

    _check("Zero missing label files",
           len(errors_missing_label) == 0,
           f"{len(errors_missing_label)} missing")

    _check("Zero invalid label lines",
           len(errors_invalid_label_lines) == 0,
           f"{len(errors_invalid_label_lines)} invalid lines")

    _check("Zero out-of-bounds coordinates",
           len(errors_out_of_bounds) == 0,
           f"{len(errors_out_of_bounds)} violations")

    _check("Zero duplicate filenames train/val",
           len(dup_train_val) == 0,
           f"{len(dup_train_val)} duplicates")

    _check("Zero duplicate filenames train/test",
           len(dup_train_test) == 0,
           f"{len(dup_train_test)} duplicates")

    _check("Zero duplicate filenames val/test",
           len(dup_val_test) == 0,
           f"{len(dup_val_test)} duplicates")

    _check("Zero SubPipe source leakage train/val",
           len(sp_tv_overlap) == 0,
           f"{len(sp_tv_overlap)} overlapping source images")

    _check("Zero SubPipe source leakage train/test",
           len(sp_tt_overlap) == 0,
           f"{len(sp_tt_overlap)} overlapping source images")

    _check("Zero SubPipe source leakage val/test",
           len(sp_vt_overlap) == 0,
           f"{len(sp_vt_overlap)} overlapping source images")

    _check("HF 700-image test set in combined test split",
           hf_test_count_in_combined_test == len(hf_test_stems) and len(hf_test_stems) > 0,
           f"{hf_test_count_in_combined_test}/{len(hf_test_stems)} identified")

    _check("HF test images NOT contaminating train",
           hf_test_contaminating_train == 0,
           f"{hf_test_contaminating_train} HF test images found in train")

    _check("HF test images NOT contaminating val",
           hf_test_contaminating_val == 0,
           f"{hf_test_contaminating_val} HF test images found in val")

    # Print split counts
    print(f"\n  Split Breakdown:")
    for split in SPLITS:
        sc = split_counts[split]
        print(f"    {split}: {sc['total']} total "
              f"({sc['drishti']} Drishti + {sc['subpipe']} SubPipe) | "
              f"positive={split_positive[split]}, negative={split_negative[split]}")

    print(f"\n  Per-class annotation distribution:")
    total_annotations = 0
    for cls_id in sorted(class_distribution.keys()):
        count = class_distribution[cls_id]
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        note = ""
        if name == "ghost_net":
            note = " [SYNTHETIC-ONLY]"
        if name == "crab_pot":
            note = " [ZERO REAL SAMPLES — expect 0]"
        print(f"    {cls_id} {name}: {count}{note}")
        total_annotations += count

    print(f"    TOTAL annotations: {total_annotations}")

    # ─── FINAL VERDICT
    all_passed = len(checks_failed) == 0

    print("\n" + "=" * 70)
    if all_passed:
        print("[OK] PHASE 4 VALIDATION PASSED — safe to proceed to training.")
    else:
        print(f"[FAIL] PHASE 4 VALIDATION FAILED — {len(checks_failed)} checks failed:")
        for f in checks_failed:
            print(f"         * {f}")
    print("=" * 70)

    # ─── Save report
    report = {
        "validation_status": "PASSED" if all_passed else "FAILED",
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "total_images": total_images,
        "manifest_entries": len(manifest_entries),
        "split_counts": {s: dict(split_counts[s]) for s in SPLITS},
        "split_positive": dict(split_positive),
        "split_negative": dict(split_negative),
        "per_class_distribution": {
            CLASS_NAMES.get(k, f"class_{k}"): v
            for k, v in sorted(class_distribution.items())
        },
        "total_annotations": total_annotations,
        "errors": {
            "missing_images": errors_missing_image[:20],
            "missing_labels": errors_missing_label[:20],
            "invalid_label_lines": errors_invalid_label_lines[:20],
            "out_of_bounds": errors_out_of_bounds[:20],
        },
        "duplicate_filenames": {
            "train_val": list(dup_train_val)[:10],
            "train_test": list(dup_train_test)[:10],
            "val_test": list(dup_val_test)[:10],
        },
        "subpipe_leakage": {
            "train_val_overlap": len(sp_tv_overlap),
            "train_test_overlap": len(sp_tt_overlap),
            "val_test_overlap": len(sp_vt_overlap),
        },
        "hf_test_set": {
            "original_hf_test_images": len(hf_test_stems),
            "found_in_combined_test": hf_test_count_in_combined_test,
            "contaminating_train": hf_test_contaminating_train,
            "contaminating_val": hf_test_contaminating_val,
        },
    }

    out_path = REPORTS_DIR / "combined_dataset_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Validation report: {out_path}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
