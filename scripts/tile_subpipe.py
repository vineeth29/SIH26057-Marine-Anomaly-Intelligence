"""
SubPipe Tiling Script (Phase 1)
================================

Converts wide SubPipe sonar imagery (5000x500 and 2500x500) into 640x500 tiles
matching the native geometry of Drishti-SSS (640x500).

- Never modifies original SubPipe directory.
- Splits by SOURCE IMAGE to ensure zero data leakage across train/val/test.
- Transforms YOLO bounding boxes (SubPipe class 0 -> class 1 submarine_pipeline).
- Preserves background/negative tiles.
- Creates comprehensive manifest tracking provenance.
"""

import os
import sys
import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SUBPIPE_DATA = Path(r"data\subpipe\SubPipeMini2\SubPipeMiniSSS\DATA")
OUTPUT_DIR = ROOT / "data" / "subpipe_tiled"

TILE_WIDTH = 640
TILE_HEIGHT = 500  # native height of SubPipe sonar scans
STRIDE_HF = 480    # Overlap = 160px for 5000px image -> 11 tiles
STRIDE_LF = 465    # Overlap = 175px for 2500px image -> 5 tiles

# Split ratios (stratified by source image)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

def parse_yolo_file(lbl_path: Path):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append((cls_id, cx, cy, w, h))
            except ValueError:
                pass
    return boxes

def get_tile_ranges(img_width: int, tile_width: int, target_stride: int):
    """Generate (x_start, x_end) ranges spanning the entire width."""
    ranges = []
    curr_x = 0
    while curr_x + tile_width <= img_width:
        ranges.append((curr_x, curr_x + tile_width))
        curr_x += target_stride
    
    # Ensure the very end of the image is covered
    if ranges[-1][1] < img_width:
        ranges.append((img_width - tile_width, img_width))
    elif ranges[-1][1] > img_width:
        ranges[-1] = (img_width - tile_width, img_width)
    return ranges

def main():
    print("=" * 70)
    print("PHASE 1: SUBPIPE ASPECT-RATIO-AWARE TILING")
    print("=" * 70)
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Verify sources
    hf_img_dir = SUBPIPE_DATA / "SSS_HF_images" / "Image"
    hf_lbl_dir = SUBPIPE_DATA / "SSS_HF_images" / "YOLO_Annotation"
    lf_img_dir = SUBPIPE_DATA / "SSS_LF_images" / "Image"
    lf_lbl_dir = SUBPIPE_DATA / "SSS_LF_images" / "YOLO_Annotation"
    
    if not hf_img_dir.exists() or not lf_img_dir.exists():
        print(f"FATAL: SubPipe source directory not found at {SUBPIPE_DATA}")
        sys.exit(1)
        
    # Gather image files
    hf_imgs = sorted([p for p in hf_img_dir.iterdir() if p.suffix.lower() in (".pbm", ".bpm")])
    lf_imgs = sorted([p for p in lf_img_dir.iterdir() if p.suffix.lower() in (".pbm", ".bpm")])
    
    print(f"Found {len(hf_imgs)} HF source images (5000x500)")
    print(f"Found {len(lf_imgs)} LF source images (2500x500)")
    print(f"Total source images: {len(hf_imgs) + len(lf_imgs)}")
    
    # Build dataset index with annotation status
    all_sources = []
    
    for p in hf_imgs:
        lbl_p = hf_lbl_dir / (p.stem + ".txt")
        boxes = parse_yolo_file(lbl_p)
        all_sources.append({
            "path": p,
            "stem": p.stem,
            "domain": "HF",
            "img_w": 5000,
            "img_h": 500,
            "stride": STRIDE_HF,
            "lbl_path": lbl_p,
            "gt_boxes": boxes,
            "is_annotated": len(boxes) > 0
        })
        
    for p in lf_imgs:
        lbl_p = lf_lbl_dir / (p.stem + ".txt")
        boxes = parse_yolo_file(lbl_p)
        all_sources.append({
            "path": p,
            "stem": p.stem,
            "domain": "LF",
            "img_w": 2500,
            "img_h": 500,
            "stride": STRIDE_LF,
            "lbl_path": lbl_p,
            "gt_boxes": boxes,
            "is_annotated": len(boxes) > 0
        })

    # Stratified split by (domain, is_annotated)
    strata = {}
    for item in all_sources:
        key = (item["domain"], item["is_annotated"])
        strata.setdefault(key, []).append(item)
        
    train_sources, val_sources, test_sources = [], [], []
    
    for key, items in strata.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        train_sources.extend(items[:n_train])
        val_sources.extend(items[n_train:n_train + n_val])
        test_sources.extend(items[n_train + n_val:])
        
    print(f"\nStratified Source Image Split:")
    print(f"  Train: {len(train_sources)} images")
    print(f"  Val:   {len(val_sources)} images")
    print(f"  Test:  {len(test_sources)} images")
    
    # Prepare output directories
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifests").mkdir(parents=True, exist_ok=True)
    
    manifest_entries = []
    split_assignments = [
        ("train", train_sources),
        ("val", val_sources),
        ("test", test_sources)
    ]
    
    total_tiles_created = 0
    total_transformed_boxes = 0
    total_original_boxes_count = 0
    boxes_recovered_set = set() # (stem, box_idx)
    
    print("\nTiling images and mapping bounding boxes...")
    
    for split_name, sources in split_assignments:
        for src_idx, item in enumerate(sources):
            # Load image
            pil_img = Image.open(item["path"])
            img_w, img_h = pil_img.size
            
            # Ground truth boxes in absolute pixel coordinates
            gt_boxes_abs = []
            for b_idx, (cls_id, cx, cy, bw, bh) in enumerate(item["gt_boxes"]):
                total_original_boxes_count += 1
                bx1 = (cx - bw / 2.0) * img_w
                by1 = (cy - bh / 2.0) * img_h
                bx2 = (cx + bw / 2.0) * img_w
                by2 = (cy + bh / 2.0) * img_h
                gt_boxes_abs.append((b_idx, cls_id, bx1, by1, bx2, by2, bw * img_w, bh * img_h))
                
            # Get tile ranges
            ranges = get_tile_ranges(img_w, TILE_WIDTH, item["stride"])
            
            for t_idx, (x_start, x_end) in enumerate(ranges):
                tile_crop = pil_img.crop((x_start, 0, x_end, img_h))
                tile_name = f"{item['domain']}_{item['stem']}_tile{t_idx:02d}_{x_start}_{x_end}"
                tile_img_path = OUTPUT_DIR / "images" / split_name / f"{tile_name}.png"
                tile_lbl_path = OUTPUT_DIR / "labels" / split_name / f"{tile_name}.txt"
                
                tile_crop.save(tile_img_path)
                total_tiles_created += 1
                
                # Transform boxes into tile coordinates
                tile_boxes = []
                for b_idx, cls_id, bx1, by1, bx2, by2, orig_bw_px, orig_bh_px in gt_boxes_abs:
                    ix1 = max(bx1, x_start)
                    ix2 = min(bx2, x_end)
                    iy1 = by1
                    iy2 = by2
                    
                    if ix2 > ix1:
                        inter_w = ix2 - ix1
                        # Condition to retain box: visible width >= 8px or at least 15% of original box
                        if inter_w >= 8 or (inter_w / max(orig_bw_px, 1e-5)) >= 0.15:
                            # Map to tile local coords
                            tx1 = ix1 - x_start
                            tx2 = ix2 - x_start
                            ty1 = iy1
                            ty2 = iy2
                            
                            # Normalized YOLO coords in [0, 1] relative to tile (640, 500)
                            tcx = (tx1 + tx2) / (2.0 * TILE_WIDTH)
                            tcy = (ty1 + ty2) / (2.0 * TILE_HEIGHT)
                            tw = (tx2 - tx1) / float(TILE_WIDTH)
                            th = (ty2 - ty1) / float(TILE_HEIGHT)
                            
                            # Clamp strictly to [0, 1]
                            tcx = min(max(tcx, 0.0), 1.0)
                            tcy = min(max(tcy, 0.0), 1.0)
                            tw = min(max(tw, 0.001), 1.0)
                            th = min(max(th, 0.001), 1.0)
                            
                            # Remap SubPipe class 0 -> class 1 (submarine_pipeline)
                            target_cls = 1
                            tile_boxes.append((target_cls, tcx, tcy, tw, th))
                            boxes_recovered_set.add((item["stem"], b_idx))
                            total_transformed_boxes += 1
                            
                # Write YOLO annotation file
                with open(tile_lbl_path, "w", encoding="utf-8") as f:
                    for t_cls, tcx, tcy, tw, th in tile_boxes:
                        f.write(f"{t_cls} {tcx:.6f} {tcy:.6f} {tw:.6f} {th:.6f}\n")
                        
                manifest_entries.append({
                    "tile_name": tile_name,
                    "split": split_name,
                    "source_stem": item["stem"],
                    "domain": item["domain"],
                    "orig_dims": [img_w, img_h],
                    "tile_range": [x_start, x_end],
                    "num_boxes": len(tile_boxes),
                    "is_positive": len(tile_boxes) > 0
                })
                
            if (src_idx + 1) % 200 == 0 or (src_idx + 1) == len(sources):
                print(f"  [{split_name}] Processed {src_idx + 1}/{len(sources)} source images...")
                
    # Save manifest
    manifest_path = OUTPUT_DIR / "manifests" / "tiling_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_source_images": len(all_sources),
            "total_tiles_created": total_tiles_created,
            "total_original_boxes": total_original_boxes_count,
            "distinct_original_boxes_recovered": len(boxes_recovered_set),
            "box_recovery_rate": round(len(boxes_recovered_set) / max(total_original_boxes_count, 1) * 100, 2),
            "manifest": manifest_entries
        }, f, indent=2)
        
    print(f"\n[OK] Tiling completed successfully!")
    print(f"  Saved to: {OUTPUT_DIR}")
    print(f"  Total tiles: {total_tiles_created}")
    print(f"  Original boxes: {total_original_boxes_count}")
    print(f"  Boxes captured: {len(boxes_recovered_set)} ({len(boxes_recovered_set)/max(total_original_boxes_count,1)*100:.2f}%)")
    print(f"  Manifest written: {manifest_path}")

if __name__ == "__main__":
    main()
