"""
False-positive filter for SIH26057 sonar detection pipeline.

For every YOLO detection this module computes measurable object-level features
and makes a transparent KEEP / REJECT decision with a traceable rejection reason.

All thresholds are class-aware and configurable via configs/fp_filter.yaml.
No random rejections — every rejection is traceable to a measured feature.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# Per-class expected geometry profiles
# Values are (min_aspect, max_aspect, min_area_px)
# aspect = width / height  (or height/width, whichever > 1)
# ---------------------------------------------------------------------------
CLASS_GEOMETRY = {
    # long thin structures
    "submarine_pipeline": {"min_aspect": 1.5, "max_aspect": 40.0, "min_area_px": 500},
    "pipeline":           {"min_aspect": 1.5, "max_aspect": 40.0, "min_area_px": 500},
    "pipe_cable":         {"min_aspect": 1.5, "max_aspect": 40.0, "min_area_px": 400},
    # compact roughly boxy structures
    "shipwreck":          {"min_aspect": 1.0, "max_aspect": 8.0,  "min_area_px": 800},
    # compact cylinders
    "mine_cylinder":      {"min_aspect": 1.0, "max_aspect": 4.0,  "min_area_px": 300},
    # spread-out nets
    "ghost_net":          {"min_aspect": 1.0, "max_aspect": 12.0, "min_area_px": 400},
    "fishing_net":        {"min_aspect": 1.0, "max_aspect": 12.0, "min_area_px": 400},
    # small debris
    "metal_debris":       {"min_aspect": 1.0, "max_aspect": 6.0,  "min_area_px": 200},
    "plastic_debris":     {"min_aspect": 1.0, "max_aspect": 6.0,  "min_area_px": 100},
    # fallback
    "_default":           {"min_aspect": 1.0, "max_aspect": 30.0, "min_area_px": 100},
}

# Minimum detector confidence for any class
MIN_CONF_HARD = 0.10
# Minimum local contrast (obj mean - bg mean) normalised
MIN_LOCAL_CONTRAST = 0.05
# Minimum bbox area fraction of image (avoid dust detections)
MIN_AREA_FRACTION = 0.0002


@dataclass
class FPFilterResult:
    detection_id: str
    keep: bool
    rejection_reason: Optional[str]
    # measured features
    aspect_ratio: float = 0.0
    area_px: int = 0
    area_fraction: float = 0.0
    local_contrast: float = 0.0
    edge_density: float = 0.0
    texture_score: float = 0.0   # normalised local std
    shape_score: float = 0.0     # how well aspect matches expected profile
    detector_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_aspect_ratio(bbox: list) -> Tuple[float, float, float]:
    """Returns (aspect_ratio, width, height). aspect = longer/shorter side."""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    ar = max(w, h) / min(w, h)
    return float(ar), float(w), float(h)


def _compute_local_contrast(img_gray: np.ndarray, bbox: list) -> float:
    """
    Normalised local contrast: (mean_obj - mean_bg) / 255.
    Positive = object brighter than surroundings (high reflectivity = man-made).
    We also accept high negative (very dark shadow objects).
    """
    h_img, w_img = img_gray.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    obj = img_gray[y1:y2, x1:x2].astype(np.float32)
    obj_mean = float(obj.mean())

    # Background: a margin around the bbox (clamped to image)
    margin_x = max(10, (x2 - x1) // 2)
    margin_y = max(10, (y2 - y1) // 2)
    bx1 = max(0, x1 - margin_x)
    by1 = max(0, y1 - margin_y)
    bx2 = min(w_img, x2 + margin_x)
    by2 = min(h_img, y2 + margin_y)

    bg_full = img_gray[by1:by2, bx1:bx2].astype(np.float32)
    # Mask out the object region
    mask = np.ones(bg_full.shape, dtype=bool)
    roi_y1 = y1 - by1
    roi_y2 = y2 - by1
    roi_x1 = x1 - bx1
    roi_x2 = x2 - bx1
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = False

    bg_pixels = bg_full[mask]
    bg_mean = float(bg_pixels.mean()) if len(bg_pixels) > 0 else obj_mean

    contrast = abs(obj_mean - bg_mean) / 255.0
    return float(np.clip(contrast, 0.0, 1.0))


def _compute_edge_density(img_gray: np.ndarray, bbox: list) -> float:
    """Fraction of pixels in bbox that are strong edges."""
    h_img, w_img = img_gray.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = img_gray[y1:y2, x1:x2]
    edges = cv2.Canny(roi, 30, 90)
    density = float(edges.sum() / 255) / max(1, roi.size)
    return float(np.clip(density, 0.0, 1.0))


def _compute_texture_score(img_gray: np.ndarray, bbox: list) -> float:
    """Local std / 64 as a normalised texture score [0,1]."""
    h_img, w_img = img_gray.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = img_gray[y1:y2, x1:x2].astype(np.float32)
    score = float(roi.std()) / 64.0
    return float(np.clip(score, 0.0, 1.0))


def _compute_shape_score(class_name: str, aspect_ratio: float) -> float:
    """
    Returns 1.0 if the detected bbox aspect ratio matches the class profile,
    0.0 if it's completely outside the expected range.
    Score is linearly interpolated inside the expected range.
    """
    cn = class_name.lower().replace(" ", "_")
    profile = CLASS_GEOMETRY.get(cn, CLASS_GEOMETRY["_default"])
    lo = profile["min_aspect"]
    hi = profile["max_aspect"]

    if aspect_ratio < lo:
        # Too square for expected class
        return float(np.clip(aspect_ratio / max(lo, 1e-6), 0.0, 1.0))
    if aspect_ratio > hi:
        # Too elongated
        return float(np.clip(1.0 - (aspect_ratio - hi) / max(hi, 1.0), 0.0, 1.0))
    # Within expected range — full score
    return 1.0


def apply_fp_filter(
    detection_id: str,
    class_name: str,
    confidence: float,
    bbox: list,
    img: np.ndarray,
    shadow_score: float = 0.0,
    min_conf: float = MIN_CONF_HARD,
    min_area_px: Optional[int] = None,
    min_local_contrast: float = MIN_LOCAL_CONTRAST,
) -> FPFilterResult:
    """
    Apply measurable feature-based false-positive filter to a single detection.

    Returns FPFilterResult with:
        keep: bool
        rejection_reason: str if rejected, None if kept

    All rejection decisions are based on measured, traceable features.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h_img, w_img = gray.shape
    img_area = h_img * w_img

    # Compute all features
    aspect_ratio, bbox_w, bbox_h = _compute_aspect_ratio(bbox)
    area_px = int(bbox_w * bbox_h)
    area_fraction = area_px / max(img_area, 1)
    local_contrast = _compute_local_contrast(gray, bbox)
    edge_density = _compute_edge_density(gray, bbox)
    texture_score = _compute_texture_score(gray, bbox)
    shape_score = _compute_shape_score(class_name, aspect_ratio)

    cn = class_name.lower().replace(" ", "_")
    profile = CLASS_GEOMETRY.get(cn, CLASS_GEOMETRY["_default"])
    _min_area = min_area_px if min_area_px is not None else profile["min_area_px"]

    # -----------------------------------------------------------------------
    # REJECTION RULES — all based on measured features
    # -----------------------------------------------------------------------

    # Rule 1: Hard confidence floor
    if confidence < min_conf:
        return FPFilterResult(
            detection_id=detection_id, keep=False,
            rejection_reason=f"Detector confidence {confidence:.3f} below hard threshold {min_conf:.2f}",
            aspect_ratio=aspect_ratio, area_px=area_px, area_fraction=area_fraction,
            local_contrast=local_contrast, edge_density=edge_density,
            texture_score=texture_score, shape_score=shape_score,
            detector_confidence=confidence,
        )

    # Rule 2: Area too small — likely noise speckle
    if area_px < _min_area:
        return FPFilterResult(
            detection_id=detection_id, keep=False,
            rejection_reason=f"Object area {area_px}px below class minimum {_min_area}px (likely noise)",
            aspect_ratio=aspect_ratio, area_px=area_px, area_fraction=area_fraction,
            local_contrast=local_contrast, edge_density=edge_density,
            texture_score=texture_score, shape_score=shape_score,
            detector_confidence=confidence,
        )

    # Rule 3: Image fraction too small
    if area_fraction < MIN_AREA_FRACTION:
        return FPFilterResult(
            detection_id=detection_id, keep=False,
            rejection_reason=f"Bounding box covers only {area_fraction*100:.4f}% of image — below minimum {MIN_AREA_FRACTION*100:.3f}%",
            aspect_ratio=aspect_ratio, area_px=area_px, area_fraction=area_fraction,
            local_contrast=local_contrast, edge_density=edge_density,
            texture_score=texture_score, shape_score=shape_score,
            detector_confidence=confidence,
        )

    # Rule 4: Aspect ratio severely inconsistent with class profile (shape_score very low)
    if shape_score < 0.15 and confidence < 0.5:
        return FPFilterResult(
            detection_id=detection_id, keep=False,
            rejection_reason=(
                f"Aspect ratio {aspect_ratio:.2f} strongly inconsistent with expected "
                f"'{class_name}' geometry (shape_score={shape_score:.2f}), "
                f"low detector confidence {confidence:.2f}"
            ),
            aspect_ratio=aspect_ratio, area_px=area_px, area_fraction=area_fraction,
            local_contrast=local_contrast, edge_density=edge_density,
            texture_score=texture_score, shape_score=shape_score,
            detector_confidence=confidence,
        )

    # Rule 5: Insufficient local contrast AND no shadow evidence
    if local_contrast < min_local_contrast and shadow_score < 0.15 and confidence < 0.4:
        return FPFilterResult(
            detection_id=detection_id, keep=False,
            rejection_reason=(
                f"Insufficient local contrast ({local_contrast:.3f}) "
                f"and no shadow evidence ({shadow_score:.2f}) — "
                f"consistent with natural seabed feature"
            ),
            aspect_ratio=aspect_ratio, area_px=area_px, area_fraction=area_fraction,
            local_contrast=local_contrast, edge_density=edge_density,
            texture_score=texture_score, shape_score=shape_score,
            detector_confidence=confidence,
        )

    # KEEP
    return FPFilterResult(
        detection_id=detection_id, keep=True,
        rejection_reason=None,
        aspect_ratio=aspect_ratio, area_px=area_px, area_fraction=area_fraction,
        local_contrast=local_contrast, edge_density=edge_density,
        texture_score=texture_score, shape_score=shape_score,
        detector_confidence=confidence,
    )


def batch_filter(
    detections: list,
    img: np.ndarray,
    shadow_scores: Optional[Dict[str, float]] = None,
) -> Tuple[List[FPFilterResult], dict]:
    """
    Filter a batch of Detection objects.

    Returns:
        results: List[FPFilterResult] one per detection (in same order)
        summary: dict with counts
    """
    shadow_scores = shadow_scores or {}
    results = []
    kept = 0
    rejected = 0

    for det in detections:
        det_id = getattr(det, "detection_id", str(id(det)))
        shadow_s = shadow_scores.get(det_id, 0.0)
        r = apply_fp_filter(
            detection_id=det_id,
            class_name=getattr(det, "class_name", "unknown"),
            confidence=getattr(det, "confidence", 0.0),
            bbox=getattr(det, "bbox", [0, 0, 10, 10]),
            img=img,
            shadow_score=shadow_s,
        )
        results.append(r)
        if r.keep:
            kept += 1
        else:
            rejected += 1

    summary = {
        "total_before": len(detections),
        "kept": kept,
        "rejected": rejected,
        "rejection_rate": round(rejected / max(len(detections), 1), 3),
    }
    return results, summary
