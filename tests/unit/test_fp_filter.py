"""
Unit tests for the false-positive filter module.
Verifies KEEP/REJECT decisions are based on measurable features
with traceable rejection reasons.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import cv2

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ai.fp_filter.false_positive_filter import apply_fp_filter, FPFilterResult, batch_filter


def make_test_img(h: int = 200, w: int = 400) -> np.ndarray:
    img = np.full((h, w, 3), 80, dtype=np.uint8)
    # bright target
    img[80:120, 150:220] = 220
    return img


class TestApplyFPFilter:
    def test_returns_fp_filter_result(self):
        img = make_test_img()
        r = apply_fp_filter("DET-001", "submarine_pipeline", 0.7, [150, 80, 220, 120], img)
        assert isinstance(r, FPFilterResult)

    def test_keep_high_confidence(self):
        """High-confidence detection with reasonable geometry should be kept."""
        img = make_test_img()
        r = apply_fp_filter("DET-001", "submarine_pipeline", 0.75, [100, 80, 220, 95], img)
        # Wide, thin bbox matches pipeline profile → should be kept
        assert r.keep is True
        assert r.rejection_reason is None

    def test_reject_very_low_confidence(self):
        """Sub-threshold confidence must be rejected."""
        img = make_test_img()
        r = apply_fp_filter("DET-002", "submarine_pipeline", 0.05, [150, 80, 220, 120], img)
        assert r.keep is False
        assert r.rejection_reason is not None
        assert len(r.rejection_reason) > 0

    def test_reject_tiny_area(self):
        """Tiny 5x5 bbox must be rejected as noise."""
        img = make_test_img()
        r = apply_fp_filter("DET-003", "shipwreck", 0.6, [100, 100, 105, 105], img)
        assert r.keep is False
        assert "area" in r.rejection_reason.lower() or "noise" in r.rejection_reason.lower()

    def test_rejection_reason_is_string(self):
        """Any rejected detection must have a string rejection reason."""
        img = make_test_img()
        r = apply_fp_filter("DET-004", "shipwreck", 0.03, [100, 100, 110, 110], img)
        assert r.keep is False
        assert isinstance(r.rejection_reason, str)
        assert len(r.rejection_reason) > 5

    def test_no_rejection_reason_when_kept(self):
        """Kept detections must have no rejection reason."""
        img = make_test_img()
        r = apply_fp_filter("DET-005", "mine_cylinder", 0.8, [100, 80, 160, 130], img)
        if r.keep:
            assert r.rejection_reason is None

    def test_computed_features_are_in_range(self):
        img = make_test_img()
        r = apply_fp_filter("DET-006", "shipwreck", 0.6, [100, 80, 200, 130], img)
        assert 0.0 <= r.texture_score <= 1.0
        assert 0.0 <= r.shape_score <= 1.0
        assert 0.0 <= r.local_contrast <= 1.0
        assert 0.0 <= r.edge_density <= 1.0
        assert r.aspect_ratio >= 1.0
        assert r.area_px > 0

    def test_aspect_ratio_correctly_computed(self):
        img = make_test_img()
        # Wide box: 200x20 → aspect = 10
        r = apply_fp_filter("DET-007", "submarine_pipeline", 0.7, [50, 90, 250, 110], img)
        assert abs(r.aspect_ratio - 10.0) < 1.0

    def test_to_dict_has_all_fields(self):
        img = make_test_img()
        r = apply_fp_filter("DET-008", "shipwreck", 0.6, [100, 80, 200, 130], img)
        d = r.to_dict()
        for k in [
            "detection_id", "keep", "rejection_reason", "aspect_ratio",
            "area_px", "area_fraction", "local_contrast", "edge_density",
            "texture_score", "shape_score", "detector_confidence"
        ]:
            assert k in d, f"Missing key in FPFilterResult.to_dict(): {k}"


class TestBatchFilter:
    def test_batch_returns_correct_count(self):
        img = make_test_img()

        class FakeDet:
            def __init__(self, did, cls, conf, bbox):
                self.detection_id = did
                self.class_name = cls
                self.confidence = conf
                self.bbox = bbox

        dets = [
            FakeDet("D1", "shipwreck", 0.8, [100, 80, 200, 130]),
            FakeDet("D2", "submarine_pipeline", 0.03, [0, 0, 5, 5]),
        ]
        results, summary = batch_filter(dets, img)
        assert len(results) == 2
        assert summary["total_before"] == 2
        assert summary["kept"] + summary["rejected"] == 2

    def test_summary_has_all_keys(self):
        img = make_test_img()
        results, summary = batch_filter([], img)
        for k in ["total_before", "kept", "rejected", "rejection_rate"]:
            assert k in summary
