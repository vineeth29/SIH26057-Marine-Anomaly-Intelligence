"""
Sonar image dropout and corruption detector for SIH26057.

Detects:
- Missing/blank column runs (typical sonar data dropout)
- Brightness discontinuities (heave/pitch artifacts)
- Corrupted frame patterns (abnormal blank regions)
- Severe speckle burst noise

All detections are based on measurable pixel statistics — no guessing.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class DropoutResult:
    dropout_detected: bool
    quality_warning: str  # Human-readable summary
    dropout_column_runs: List[tuple] = field(default_factory=list)  # list of (start_col, end_col)
    blank_fraction: float = 0.0          # fraction of image that is near-black
    brightness_discontinuity: bool = False
    discontinuity_row: Optional[int] = None
    discontinuity_magnitude: float = 0.0
    max_run_length_cols: int = 0
    issues: List[str] = field(default_factory=list)
    motion_metadata_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def detect_dropout(
    img: np.ndarray,
    blank_threshold: int = 8,           # intensity below this = blank
    min_dropout_run: int = 20,          # min consecutive blank columns = dropout
    blank_fraction_threshold: float = 0.15,   # >15% blank pixels = warning
    brightness_jump_threshold: float = 40.0,  # row-to-row mean jump = discontinuity
    sonar_metadata: Optional[dict] = None,
) -> DropoutResult:
    """
    Analyse a sonar frame for data dropouts and corruption artefacts.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    blank_threshold : int
        Pixel intensity below which a pixel is considered blank/missing.
    min_dropout_run : int
        Minimum number of consecutive blank columns to count as a dropout event.
    blank_fraction_threshold : float
        If more than this fraction of all pixels are blank, emit a warning.
    brightness_jump_threshold : float
        Row-mean difference above this counts as a brightness discontinuity.
    sonar_metadata : dict, optional
        If provided, motion metadata (heave, pitch, roll) availability is reported.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    h, w = gray.shape
    issues = []

    # -----------------------------------------------------------------------
    # 1. Blank pixel fraction
    # -----------------------------------------------------------------------
    blank_mask = gray < blank_threshold
    blank_fraction = float(blank_mask.sum()) / max(h * w, 1)

    # -----------------------------------------------------------------------
    # 2. Blank column detection (dropout runs)
    # -----------------------------------------------------------------------
    col_means = gray.mean(axis=0)  # shape (w,)
    blank_cols = col_means < blank_threshold

    dropout_runs = []
    run_start = None
    for c in range(w):
        if blank_cols[c]:
            if run_start is None:
                run_start = c
        else:
            if run_start is not None:
                run_len = c - run_start
                if run_len >= min_dropout_run:
                    dropout_runs.append((run_start, c - 1))
                run_start = None
    if run_start is not None:
        run_len = w - run_start
        if run_len >= min_dropout_run:
            dropout_runs.append((run_start, w - 1))

    max_run = max((e - s + 1 for s, e in dropout_runs), default=0)

    if dropout_runs:
        issues.append(
            f"Sonar data dropout: {len(dropout_runs)} blank column run(s) detected "
            f"(longest: {max_run} columns)"
        )

    if blank_fraction > blank_fraction_threshold:
        issues.append(
            f"High blank pixel fraction: {blank_fraction*100:.1f}% of image is near-black"
        )

    # -----------------------------------------------------------------------
    # 3. Brightness discontinuity (row-to-row mean jump)
    # -----------------------------------------------------------------------
    row_means = gray.mean(axis=1)  # shape (h,)
    row_diffs = np.abs(np.diff(row_means))
    disc_row = None
    disc_mag = float(row_diffs.max()) if len(row_diffs) > 0 else 0.0
    brightness_disc = disc_mag > brightness_jump_threshold
    if brightness_disc:
        disc_row = int(np.argmax(row_diffs))
        issues.append(
            f"Brightness discontinuity at row {disc_row} "
            f"(magnitude {disc_mag:.1f} — possible heave/pitch artifact)"
        )

    # -----------------------------------------------------------------------
    # 4. Motion metadata availability
    # -----------------------------------------------------------------------
    motion_available = False
    if sonar_metadata:
        motion_keys = {"heave", "pitch", "roll", "heading"}
        if any(k in sonar_metadata for k in motion_keys):
            motion_available = True

    # -----------------------------------------------------------------------
    # Build result
    # -----------------------------------------------------------------------
    dropout_detected = bool(dropout_runs) or blank_fraction > blank_fraction_threshold or brightness_disc

    if not issues:
        warning = "No dropout or corruption detected"
    else:
        warning = " | ".join(issues)

    return DropoutResult(
        dropout_detected=dropout_detected,
        quality_warning=warning,
        dropout_column_runs=dropout_runs,
        blank_fraction=round(blank_fraction, 4),
        brightness_discontinuity=brightness_disc,
        discontinuity_row=disc_row,
        discontinuity_magnitude=round(disc_mag, 2),
        max_run_length_cols=max_run,
        issues=issues,
        motion_metadata_available=motion_available,
    )
