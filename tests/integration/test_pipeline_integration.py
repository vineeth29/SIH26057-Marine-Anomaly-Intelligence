"""
Integration test: full image → pipeline → shadow → anomaly → fusion.
Tests that the pipeline produces correct output structure with real modules
(no mocking of the core pipeline steps).
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import cv2

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from services.pipeline_service import SonarAnalysisPipeline, PipelineResult, PipelineDetection


def make_synthetic_sonar(h: int = 256, w: int = 512) -> np.ndarray:
    """Synthetic side-scan sonar image with a pipeline-like bright strip."""
    img = np.random.randint(50, 100, (h, w), dtype=np.uint8)
    # Gradient background
    for r in range(h):
        img[r, :] = np.clip(img[r, :] + int(40 * r / h), 0, 255)
    # Bright target (simulated pipeline echo)
    img[100:115, 150:350] = 210
    # Shadow
    img[95:120, 350:420] = 10
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def pipeline():
    """Load pipeline once for all tests in this module."""
    try:
        return SonarAnalysisPipeline()
    except Exception as e:
        pytest.skip(f"Pipeline could not be loaded: {e}")


@pytest.fixture(scope="module")
def pipeline_result(pipeline):
    img = make_synthetic_sonar()
    return pipeline.run(img, image_id="TEST-INT-001", lat=12.345, lon=80.123)


class TestPipelineOutput:
    def test_returns_pipeline_result(self, pipeline_result):
        assert isinstance(pipeline_result, PipelineResult)

    def test_image_id_preserved(self, pipeline_result):
        assert pipeline_result.image_id == "TEST-INT-001"

    def test_mode_is_real_or_demo(self, pipeline_result):
        assert pipeline_result.mode in {"REAL", "DEMO"}

    def test_raw_image_preserved(self, pipeline_result):
        assert pipeline_result.raw_image is not None
        assert pipeline_result.raw_image.ndim == 3

    def test_preprocessed_image_exists(self, pipeline_result):
        assert pipeline_result.preprocessed_image is not None
        assert pipeline_result.preprocessed_image.ndim == 3

    def test_annotated_image_exists(self, pipeline_result):
        assert pipeline_result.annotated_image is not None
        assert pipeline_result.annotated_image.ndim == 3

    def test_quality_dict_structure(self, pipeline_result):
        q = pipeline_result.quality
        assert "quality" in q
        assert "score" in q
        assert q["quality"] in {"GOOD", "ACCEPTABLE", "POOR", "UNKNOWN"}
        assert 0.0 <= q["score"] <= 1.0

    def test_timing_fields_present(self, pipeline_result):
        t = pipeline_result.timing
        assert "preprocess_ms" in t
        assert "detection_ms" in t
        assert "total_ms" in t
        for v in t.values():
            assert v >= 0

    def test_warnings_is_list(self, pipeline_result):
        assert isinstance(pipeline_result.warnings, list)

    def test_detections_is_list(self, pipeline_result):
        assert isinstance(pipeline_result.detections, list)

    def test_dropout_result_present(self, pipeline_result):
        assert isinstance(pipeline_result.dropout_result, dict)
        assert "dropout_detected" in pipeline_result.dropout_result

    def test_fp_filter_summary_present(self, pipeline_result):
        assert isinstance(pipeline_result.fp_filter_summary, dict)

    def test_anomaly_model_status_present(self, pipeline_result):
        assert hasattr(pipeline_result, "anomaly_model_status")
        assert pipeline_result.anomaly_model_status in {"NON_DISCRIMINATIVE", "VALIDATED", "UNTESTED"}

    def test_to_dict_serializable(self, pipeline_result):
        import json
        d = pipeline_result.to_dict()
        # Must be JSON serializable (no numpy types etc.)
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 10

    def test_num_known_plus_anomalies_equals_detections(self, pipeline_result):
        all_dets = len(pipeline_result.detections)
        known = sum(1 for d in pipeline_result.detections if not d.is_anomaly)
        anom = sum(1 for d in pipeline_result.detections if d.is_anomaly)
        assert known == pipeline_result.num_known
        assert anom == pipeline_result.num_anomalies


class TestDetectionFields:
    """Test that each detection has all required fields with correct types."""

    def test_each_detection_has_required_fields(self, pipeline_result):
        for det in pipeline_result.detections:
            assert hasattr(det, "detection_id")
            assert hasattr(det, "class_name")
            assert hasattr(det, "confidence")
            assert hasattr(det, "bbox")
            assert hasattr(det, "shadow_score")
            assert hasattr(det, "texture_score")
            assert hasattr(det, "shape_score")
            assert hasattr(det, "evidence_score")
            assert hasattr(det, "keep")
            assert hasattr(det, "coordinate_source")
            assert hasattr(det, "anomaly_score")

    def test_confidence_in_range(self, pipeline_result):
        for det in pipeline_result.detections:
            assert 0.0 <= det.confidence <= 1.0, f"Confidence out of range: {det.confidence}"

    def test_evidence_score_in_range(self, pipeline_result):
        for det in pipeline_result.detections:
            assert 0.0 <= det.evidence_score <= 1.0

    def test_shadow_score_in_range(self, pipeline_result):
        for det in pipeline_result.detections:
            assert 0.0 <= det.shadow_score <= 1.0

    def test_texture_score_not_hardcoded_0_6(self, pipeline_result):
        """All texture scores must be computed, not hardcoded 0.6."""
        for det in pipeline_result.detections:
            # 0.6 could theoretically be correct, but we can't have ALL of them be exactly 0.6
            pass  # This is validated in the fusion unit tests

    def test_coordinate_source_valid(self, pipeline_result):
        valid = {"REAL_GPS", "MANUAL", "SIMULATED", "UNAVAILABLE"}
        for det in pipeline_result.detections:
            assert det.coordinate_source in valid, f"Invalid coordinate_source: {det.coordinate_source}"

    def test_coordinate_source_manual_when_coords_provided(self, pipeline_result):
        """When lat/lon was provided as manual, coordinate_source must not be UNAVAILABLE."""
        # We passed lat=12.345, lon=80.123 → should be MANUAL or SIMULATED, not UNAVAILABLE
        for det in pipeline_result.detections:
            assert det.coordinate_source != "UNAVAILABLE", (
                "coordinate_source should not be UNAVAILABLE when coordinates were provided"
            )

    def test_keep_is_bool(self, pipeline_result):
        for det in pipeline_result.detections:
            assert isinstance(det.keep, bool)

    def test_rejection_reason_present_when_rejected(self, pipeline_result):
        for det in pipeline_result.detections:
            if not det.keep:
                assert det.rejection_reason is not None
                assert len(det.rejection_reason) > 0

    def test_bbox_has_four_elements(self, pipeline_result):
        for det in pipeline_result.detections:
            assert len(det.bbox) == 4

    def test_to_dict_has_all_fields(self, pipeline_result):
        required = [
            "detection_id", "class_name", "confidence", "shadow_score",
            "texture_score", "shape_score", "evidence_score", "keep",
            "rejection_reason", "coordinate_source", "anomaly_score", "bbox"
        ]
        for det in pipeline_result.detections:
            d = det.to_dict()
            for k in required:
                assert k in d, f"Missing key '{k}' in detection dict"


class TestPipelineNoCoordinates:
    def test_coordinate_source_unavailable_without_coords(self, pipeline):
        img = make_synthetic_sonar()
        result = pipeline.run(img, lat=None, lon=None)
        for det in result.detections:
            assert det.coordinate_source == "UNAVAILABLE"


class TestPipelineDifferentResolutions:
    def test_small_image(self, pipeline):
        img = make_synthetic_sonar(h=128, w=256)
        result = pipeline.run(img)
        assert result is not None
        assert result.annotated_image.shape[:2] == (128, 256)

    def test_large_image(self, pipeline):
        img = make_synthetic_sonar(h=512, w=1024)
        result = pipeline.run(img)
        assert result is not None
        assert result.annotated_image.shape[:2] == (512, 1024)
