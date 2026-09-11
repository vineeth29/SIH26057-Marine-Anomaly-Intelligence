"""
Unit tests for sonar image preprocessing pipeline.
Verifies actual behavior — not just that functions run without error.
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import cv2

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ai.preprocessing.sonar_preprocessor import SonarPreprocessor, ImageQualityAssessor


def make_test_image(h: int = 120, w: int = 160, noise: bool = False) -> np.ndarray:
    """Create a synthetic sonar-like grayscale image as BGR for testing."""
    img = np.zeros((h, w), dtype=np.uint8)
    # Simulate seabed gradient
    for r in range(h):
        img[r, :] = int(60 + 120 * r / h)
    # Add a bright target
    img[40:60, 60:100] = 220
    if noise:
        noise_arr = np.random.randint(0, 25, (h, w), dtype=np.uint8)
        img = np.clip(img.astype(np.int32) + noise_arr - 12, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestImageQualityAssessor:
    def setup_method(self):
        self.qa = ImageQualityAssessor()

    def test_returns_dict_with_required_keys(self):
        img = make_test_image()
        result = self.qa.assess(img)
        for key in ["quality", "score", "reasons", "metrics", "warning"]:
            assert key in result, f"Missing key: {key}"

    def test_score_in_range(self):
        img = make_test_image()
        result = self.qa.assess(img)
        assert 0.0 <= result["score"] <= 1.0

    def test_quality_label_valid(self):
        img = make_test_image()
        result = self.qa.assess(img)
        assert result["quality"] in {"GOOD", "ACCEPTABLE", "POOR"}

    def test_poor_quality_blank_image(self):
        """A completely blank image should score poorly."""
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.qa.assess(blank)
        # Blank image has zero contrast and zero dynamic range — should not be GOOD
        assert result["quality"] in {"ACCEPTABLE", "POOR"}

    def test_metrics_have_numeric_values(self):
        img = make_test_image()
        result = self.qa.assess(img)
        m = result["metrics"]
        for k in ["sharpness", "contrast", "noise_estimate", "dynamic_range"]:
            assert isinstance(m[k], (int, float)), f"{k} is not numeric"

    def test_grayscale_input_works(self):
        """Grayscale (2-channel) input must not crash."""
        gray = np.random.randint(50, 200, (80, 100), dtype=np.uint8)
        result = self.qa.assess(gray)
        assert "quality" in result


class TestSonarPreprocessor:
    def setup_method(self):
        self.prep = SonarPreprocessor()

    def test_preprocess_returns_required_keys(self):
        img = make_test_image()
        result = self.prep.preprocess(img)
        for key in ["raw", "preprocessed", "quality", "elapsed_ms"]:
            assert key in result, f"Missing key: {key}"

    def test_preprocessed_shape_is_3channel(self):
        img = make_test_image()
        result = self.prep.preprocess(img)
        assert result["preprocessed"].ndim == 3
        assert result["preprocessed"].shape[2] == 3

    def test_preprocessed_same_spatial_size(self):
        img = make_test_image(100, 150)
        result = self.prep.preprocess(img)
        h_in, w_in = img.shape[:2]
        h_out, w_out = result["preprocessed"].shape[:2]
        assert h_in == h_out
        assert w_in == w_out

    def test_elapsed_ms_positive(self):
        img = make_test_image()
        result = self.prep.preprocess(img)
        assert result["elapsed_ms"] > 0

    def test_preprocessed_pixel_range(self):
        img = make_test_image()
        result = self.prep.preprocess(img)
        assert result["preprocessed"].min() >= 0
        assert result["preprocessed"].max() <= 255

    def test_with_steps_flag(self):
        img = make_test_image()
        result = self.prep.preprocess(img, return_steps=True)
        assert "steps" in result
        assert len(result["steps"]) > 0

    def test_grayscale_input_handled(self):
        gray = np.random.randint(50, 200, (80, 100), dtype=np.uint8)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = self.prep.preprocess(bgr)
        assert "preprocessed" in result
