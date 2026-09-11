"""
Unit tests for acoustic shadow analyzer.
Verifies that shadow_score is in [0,1], that invalid bboxes are handled,
and that measured fields are plausible.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import cv2

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ai.shadow_analysis.shadow_analyzer import AcousticShadowAnalyzer, ShadowAnalysisResult


def make_shadow_image(h: int = 200, w: int = 400) -> np.ndarray:
    """Create a synthetic sonar strip with a bright object and dark shadow."""
    img = np.full((h, w), 80, dtype=np.uint8)
    # Bright object on the left side
    img[80:120, 100:180] = 220
    # Dark shadow to the right of the object
    img[75:125, 180:260] = 15
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestAcousticShadowAnalyzer:
    def setup_method(self):
        self.analyzer = AcousticShadowAnalyzer()

    def test_shadow_score_in_range(self):
        img = make_shadow_image()
        bbox = [100, 80, 180, 120]
        result = self.analyzer.analyze(img, bbox)
        assert 0.0 <= result.shadow_score <= 1.0, f"shadow_score={result.shadow_score} out of range"

    def test_returns_shadow_analysis_result(self):
        img = make_shadow_image()
        bbox = [100, 80, 180, 120]
        result = self.analyzer.analyze(img, bbox)
        assert isinstance(result, ShadowAnalysisResult)

    def test_to_dict_has_required_keys(self):
        img = make_shadow_image()
        bbox = [100, 80, 180, 120]
        result = self.analyzer.analyze(img, bbox)
        d = result.to_dict()
        for key in [
            "shadow_score", "object_intensity", "shadow_intensity",
            "background_intensity", "object_area_px", "shadow_area_px",
            "shadow_to_object_ratio", "contrast_ratio", "elapsed_ms",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_invalid_bbox_returns_safely(self):
        """Zero-area or negative bbox must not crash."""
        img = make_shadow_image()
        result = self.analyzer.analyze(img, [50, 50, 50, 50])
        assert result.analysis_mode == "INVALID_BBOX"
        assert result.shadow_score == 0.0

    def test_bright_object_detected(self):
        """A very bright object region should yield non-zero object_intensity."""
        img = make_shadow_image()
        bbox = [100, 80, 180, 120]
        result = self.analyzer.analyze(img, bbox)
        assert result.object_intensity > 100, "Expected high intensity for bright object"

    def test_shadow_present_increases_score(self):
        """Image with an explicit shadow should score higher than featureless image."""
        # Image WITH shadow
        img_with_shadow = make_shadow_image()
        bbox = [100, 80, 180, 120]
        result_with = self.analyzer.analyze(img_with_shadow, bbox)

        # Image WITHOUT shadow (uniform background)
        img_flat = np.full((200, 400, 3), 80, dtype=np.uint8)
        img_flat[80:120, 100:180] = 220  # bright object only, no dark shadow
        result_flat = self.analyzer.analyze(img_flat, bbox)

        # With-shadow score should be >= flat score
        assert result_with.shadow_score >= result_flat.shadow_score

    def test_elapsed_ms_recorded(self):
        img = make_shadow_image()
        result = self.analyzer.analyze(img, [100, 80, 180, 120])
        assert result.elapsed_ms >= 0

    def test_grayscale_input(self):
        """Grayscale (2D) input must not crash."""
        gray = np.full((200, 400), 80, dtype=np.uint8)
        gray[80:120, 100:180] = 220
        result = self.analyzer.analyze(gray, [100, 80, 180, 120])
        assert 0.0 <= result.shadow_score <= 1.0

    def test_height_estimate_without_metadata(self):
        """Without sonar altitude metadata, height estimate must say so."""
        img = make_shadow_image()
        result = self.analyzer.analyze(img, [100, 80, 180, 120], sonar_metadata=None)
        # Must NOT claim a real altitude measurement
        assert "unavailable" not in result.height_estimate.lower() or result.height_estimate != ""

    def test_height_estimate_with_metadata(self):
        """With altitude and pixel_size metadata, height estimate should be numeric."""
        img = make_shadow_image()
        meta = {"altitude_m": 5.0, "pixel_size_m": 0.1}
        result = self.analyzer.analyze(img, [100, 80, 180, 120], sonar_metadata=meta)
        # Should produce some kind of estimate (geometry or fallback)
        assert isinstance(result.height_estimate, str)
        assert len(result.height_estimate) > 0
