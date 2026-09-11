"""
Unit tests for dropout detector.
Verifies blank-column detection, blank-fraction flagging,
brightness discontinuity, and motion metadata handling.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import cv2

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ai.quality.dropout_detector import detect_dropout, DropoutResult


def make_clean_image(h: int = 200, w: int = 400) -> np.ndarray:
    img = np.random.randint(60, 180, (h, w), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_dropout_image(h: int = 200, w: int = 400, dropout_start: int = 150, dropout_width: int = 40) -> np.ndarray:
    """Image with a column-dropout region (typical sonar data loss)."""
    img = np.random.randint(60, 180, (h, w), dtype=np.uint8)
    img[:, dropout_start:dropout_start + dropout_width] = 3  # near-black columns
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def make_discontinuity_image(h: int = 200, w: int = 400, jump_row: int = 100) -> np.ndarray:
    """Image with a severe brightness jump at a specific row."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:jump_row, :] = 60
    img[jump_row:, :] = 180  # big jump
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestDropoutDetector:
    def test_returns_dropout_result(self):
        img = make_clean_image()
        r = detect_dropout(img)
        assert isinstance(r, DropoutResult)

    def test_to_dict_has_required_keys(self):
        img = make_clean_image()
        r = detect_dropout(img)
        d = r.to_dict()
        for k in [
            "dropout_detected", "quality_warning", "dropout_column_runs",
            "blank_fraction", "brightness_discontinuity",
            "motion_metadata_available", "issues"
        ]:
            assert k in d, f"Missing key: {k}"

    def test_clean_image_no_dropout(self):
        img = make_clean_image()
        r = detect_dropout(img)
        # Clean image should not have significant dropout
        assert r.blank_fraction < 0.10

    def test_dropout_image_detected(self):
        img = make_dropout_image(dropout_width=50)
        r = detect_dropout(img)
        assert r.dropout_detected is True
        assert len(r.dropout_column_runs) > 0

    def test_dropout_run_boundaries_correct(self):
        img = make_dropout_image(dropout_start=150, dropout_width=50)
        r = detect_dropout(img)
        if r.dropout_column_runs:
            start, end = r.dropout_column_runs[0]
            assert start >= 140  # approximately correct
            assert end <= 210

    def test_blank_fraction_computed(self):
        img = make_dropout_image(dropout_width=80)
        r = detect_dropout(img)
        assert r.blank_fraction > 0.0

    def test_brightness_discontinuity_detected(self):
        img = make_discontinuity_image(jump_row=100)
        r = detect_dropout(img, brightness_jump_threshold=30.0)
        assert r.brightness_discontinuity is True
        assert r.discontinuity_row is not None

    def test_no_discontinuity_uniform_image(self):
        img = np.full((200, 400, 3), 100, dtype=np.uint8)
        r = detect_dropout(img, brightness_jump_threshold=30.0)
        assert r.brightness_discontinuity is False

    def test_motion_metadata_absent(self):
        img = make_clean_image()
        r = detect_dropout(img, sonar_metadata=None)
        assert r.motion_metadata_available is False

    def test_motion_metadata_present(self):
        img = make_clean_image()
        meta = {"heave": 0.2, "pitch": 1.5, "heading": 270.0}
        r = detect_dropout(img, sonar_metadata=meta)
        assert r.motion_metadata_available is True

    def test_grayscale_input(self):
        gray = np.random.randint(60, 180, (200, 400), dtype=np.uint8)
        r = detect_dropout(gray)
        assert isinstance(r, DropoutResult)

    def test_quality_warning_is_string(self):
        img = make_clean_image()
        r = detect_dropout(img)
        assert isinstance(r.quality_warning, str)
        assert len(r.quality_warning) > 0
