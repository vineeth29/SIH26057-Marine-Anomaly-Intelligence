"""
Phase 3: Build Combined SIH26057 Dataset
=========================================
Merges Drishti-SSS (HuggingFace) and SubPipe tiled data into:
  data/sih26057_combined/

Source datasets are NEVER modified:
  - data/drishti_sss/           (read-only)
  - data/subpipe_tiled/         (read-only)
  - original SubPipeMini2/      (never touched)

Final classes:
  0 = crab_pot
  1 = submarine_pipeline
  2 = shipwreck
  3 = ghost_net
  4 = mine_cylinder

SubPipe labels already remapped to class 1 (submarine_pipeline).
"""

import json
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DRISHTI_DIR = ROOT / "data" / "drishti_sss"
SUBPIPE_TILED_DIR = ROOT / "data" / "subpipe_tiled"
SUBPIPE_MANIFEST = SUBPIPE_TILED_DIR / "manifests" / "tiling_manifest.json"
OUTPUT_DIR = ROOT / "data" / "sih26057_combined"

CLASS_NAMES = {
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder",
}

SPLITS = ["train", "val", "test"]

# File prefix to avoid filename collisions
PREFIX_DRISHTI = "drishti__"
PREFIX_SUBPIPE = "subpipe__"


def safe_copy(src: Path, dst: Path):
    """Copy src to dst, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def count_boxes(lbl_path: Path) -> tuple[int, list[int]]:
    """Return (box_count, list_of_class_ids) from a YOLO label file."""
    if not lbl_path.exists():
        return 0, []
    boxes = 0
    classes = []
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                classes.append(int(parts[0]))
                boxes += 1
            except ValueError:
                pass
    return boxes, classes


def main():
    print("=" * 70)
    print("PHASE 3: BUILD COMBINED SIH26057 DATASET")
    print("=" * 70)

    # ── Verify sources exist
    if not DRISHTI_DIR.exists():
        print(f"FATAL: Drishti-SSS not found at {DRISHTI_DIR}")
        sys.exit(1)
    if not SUBPIPE_TILED_DIR.exists():
        print(f"FATAL: SubPipe tiled not found at {SUBPIPE_TILED_DIR}")
        sys.exit(1)
    if not SUBPIPE_MANIFEST.exists():
        print(f"FATAL: SubPipe manifest not found at {SUBPIPE_MANIFEST}")
        sys.exit(1)

    # ── Load SubPipe manifest (for provenance)
    with open(SUBPIPE_MANIFEST, "r", encoding="utf-8") as f:
        subpipe_manifest_data = json.load(f)
    subpipe_manifest = {e["tile_name"]: e for e in subpipe_manifest_data["manifest"]}

    # ── Wipe and recreate output dir
    if OUTPUT_DIR.exists():
        print(f"  Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    for split in SPLITS:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifests").mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    stats = {
        split: {
            "drishti": {"images": 0, "positive": 0, "negative": 0, "boxes": 0},
            "subpipe": {"images": 0, "positive": 0, "negative": 0, "boxes": 0},
        }
        for split in SPLITS
    }

    # ═══════════════════════════════════════════════════════════════
    # SOURCE 1: DRISHTI-SSS
    # ═══════════════════════════════════════════════════════════════
    print("\n[Source 1] Drishti-SSS (HuggingFace)")

    for split in SPLITS:
        img_src_dir = DRISHTI_DIR / split / "images"
        lbl_src_dir = DRISHTI_DIR / split / "labels"

        if not img_src_dir.exists():
            print(f"  WARNING: Drishti {split}/images not found — skipping.")
            continue

        img_files = sorted([f for f in img_src_dir.iterdir()
                            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        print(f"  {split}: {len(img_files)} images")

        for img_path in img_files:
            out_stem = f"{PREFIX_DRISHTI}{split}__{img_path.stem}"
            out_img = OUTPUT_DIR / "images" / split / (out_stem + img_path.suffix)
            out_lbl = OUTPUT_DIR / "labels" / split / (out_stem + ".txt")

            # Copy image
            safe_copy(img_path, out_img)

            # Copy label (create empty if missing)
            lbl_src = lbl_src_dir / (img_path.stem + ".txt")
            if lbl_src.exists():
                safe_copy(lbl_src, out_lbl)
                num_boxes, classes = count_boxes(lbl_src)
            else:
                out_lbl.write_text("", encoding="utf-8")
                num_boxes, classes = 0, []

            is_positive = num_boxes > 0
            stats[split]["drishti"]["images"] += 1
            stats[split]["drishti"]["boxes"] += num_boxes
            if is_positive:
                stats[split]["drishti"]["positive"] += 1
            else:
                stats[split]["drishti"]["negative"] += 1

            manifest_entries.append({
                "filename": out_stem + img_path.suffix,
                "label_filename": out_stem + ".txt",
                "source_dataset": "drishti_sss",
                "original_file": str(img_path),
                "split": split,
                "is_positive": is_positive,
                "num_boxes": num_boxes,
                "classes_present": sorted(set(classes)),
                "provenance": {
                    "original_image": img_path.name,
                    "original_split": split,
                },
            })

    # ═══════════════════════════════════════════════════════════════
    # SOURCE 2: SUBPIPE TILED
    # ═══════════════════════════════════════════════════════════════
    print("\n[Source 2] SubPipe Tiled (aspect-ratio-aware tiles)")

    for split in SPLITS:
        img_src_dir = SUBPIPE_TILED_DIR / "images" / split
        lbl_src_dir = SUBPIPE_TILED_DIR / "labels" / split

        if not img_src_dir.exists():
            print(f"  WARNING: SubPipe tiled {split}/images not found — skipping.")
            continue

        img_files = sorted([f for f in img_src_dir.iterdir()
                            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        print(f"  {split}: {len(img_files)} tiles")

        for img_path in img_files:
            out_stem = PREFIX_SUBPIPE + img_path.stem
            out_img = OUTPUT_DIR / "images" / split / (out_stem + img_path.suffix)
            out_lbl = OUTPUT_DIR / "labels" / split / (out_stem + ".txt")

            # Copy image
            safe_copy(img_path, out_img)

            # Copy label
            lbl_src = lbl_src_dir / (img_path.stem + ".txt")
            if lbl_src.exists():
                safe_copy(lbl_src, out_lbl)
                num_boxes, classes = count_boxes(lbl_src)
            else:
                out_lbl.write_text("", encoding="utf-8")
                num_boxes, classes = 0, []

            is_positive = num_boxes > 0
            stats[split]["subpipe"]["images"] += 1
            stats[split]["subpipe"]["boxes"] += num_boxes
            if is_positive:
                stats[split]["subpipe"]["positive"] += 1
            else:
                stats[split]["subpipe"]["negative"] += 1

            # Build provenance from manifest
            tile_name = img_path.stem
            tile_meta = subpipe_manifest.get(tile_name, {})

            manifest_entries.append({
                "filename": out_stem + img_path.suffix,
                "label_filename": out_stem + ".txt",
                "source_dataset": "subpipe_tiled",
                "original_file": str(img_path),
                "split": split,
                "is_positive": is_positive,
                "num_boxes": num_boxes,
                "classes_present": sorted(set(classes)),
                "provenance": {
                    "tile_name": tile_name,
                    "source_stem": tile_meta.get("source_stem", ""),
                    "domain": tile_meta.get("domain", ""),
                    "tile_range": tile_meta.get("tile_range", []),
                    "orig_dims": tile_meta.get("orig_dims", []),
                    "original_split": split,
                },
            })

    # ═══════════════════════════════════════════════════════════════
    # DATASET YAML
    # ═══════════════════════════════════════════════════════════════
    dataset_yaml = {
        "path": str(OUTPUT_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 5,
        "names": {i: name for i, name in CLASS_NAMES.items()},
    }
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True)
    print(f"\n  dataset.yaml written: {yaml_path}")

    # ═══════════════════════════════════════════════════════════════
    # MANIFEST
    # ═══════════════════════════════════════════════════════════════
    combined_manifest = {
        "dataset": "sih26057_combined",
        "classes": CLASS_NAMES,
        "note_ghost_net": "ghost_net annotations are synthetic-only (from Drishti-SSS)",
        "note_crab_pot": "crab_pot has zero known real samples — no fabricated annotations",
        "note_subpipe": "SubPipe pipeline mapped to class 1 (submarine_pipeline)",
        "splits": {},
        "total_images": len(manifest_entries),
        "entries": manifest_entries,
    }

    # Fill split summary
    for split in SPLITS:
        d = stats[split]["drishti"]
        s = stats[split]["subpipe"]
        combined_manifest["splits"][split] = {
            "drishti_images": d["images"],
            "drishti_positive": d["positive"],
            "drishti_negative": d["negative"],
            "drishti_boxes": d["boxes"],
            "subpipe_images": s["images"],
            "subpipe_positive": s["positive"],
            "subpipe_negative": s["negative"],
            "subpipe_boxes": s["boxes"],
            "total_images": d["images"] + s["images"],
            "total_positive": d["positive"] + s["positive"],
            "total_negative": d["negative"] + s["negative"],
            "total_boxes": d["boxes"] + s["boxes"],
        }

    manifest_path = OUTPUT_DIR / "manifests" / "combined_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(combined_manifest, f, indent=2)
    print(f"  Combined manifest written: {manifest_path}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMBINED DATASET BUILD SUMMARY")
    print("=" * 70)
    total_images = 0
    total_positive = 0
    total_negative = 0
    total_boxes = 0

    for split in SPLITS:
        d = stats[split]["drishti"]
        s = stats[split]["subpipe"]
        t_imgs = d["images"] + s["images"]
        t_pos = d["positive"] + s["positive"]
        t_neg = d["negative"] + s["negative"]
        t_boxes = d["boxes"] + s["boxes"]
        total_images += t_imgs
        total_positive += t_pos
        total_negative += t_neg
        total_boxes += t_boxes

        print(f"\n  {split.upper()}:")
        print(f"    Drishti: {d['images']} images ({d['positive']} pos, "
              f"{d['negative']} neg, {d['boxes']} boxes)")
        print(f"    SubPipe: {s['images']} tiles ({s['positive']} pos, "
              f"{s['negative']} neg, {s['boxes']} boxes)")
        print(f"    TOTAL:   {t_imgs} images | {t_pos} positive | "
              f"{t_neg} negative | {t_boxes} boxes")

    print(f"\n  OVERALL:")
    print(f"    Total images:    {total_images}")
    print(f"    Total positive:  {total_positive}")
    print(f"    Total negative:  {total_negative}")
    print(f"    Total boxes:     {total_boxes}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    print("\n[OK] Phase 3 complete. Proceed to Phase 4 validation.")


if __name__ == "__main__":
    main()
