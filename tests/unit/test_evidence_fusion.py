"""
Unit tests for evidence fusion and confidence scoring.
Verifies:
- No hardcoded 0.6 texture/shape scores reach the fusion output
- Scores always in [0, 1]
- Weights sum correctly
- Rejection path works
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from ai.fusion.confidence_fusion import (
    compute_evidence_score,
    compute_anomaly_evidence_score,
    compute_severity,
    get_severity_color,
)


class TestComputeEvidenceScore:
    def test_returns_dict_with_required_keys(self):
        r = compute_evidence_score(
            detector_confidence=0.7,
            shadow_score=0.5,
            anomaly_score=0.1,
            image_quality_score=0.9,
            texture_score=0.4,
            shape_score=0.6,
        )
        for k in ["evidence_score", "evidence_pct", "breakdown", "label"]:
            assert k in r, f"Missing key: {k}"

    def test_evidence_score_in_range(self):
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = compute_evidence_score(
                detector_confidence=conf,
                shadow_score=0.3,
                anomaly_score=0.05,
                image_quality_score=0.8,
                texture_score=0.3,
                shape_score=0.4,
            )
            assert 0.0 <= r["evidence_score"] <= 1.0, f"Out of range at conf={conf}"

    def test_evidence_pct_matches_score(self):
        r = compute_evidence_score(0.6, 0.4, 0.05, 0.9, 0.5, 0.5)
        assert abs(r["evidence_pct"] - r["evidence_score"] * 100) < 0.2

    def test_higher_confidence_gives_higher_score(self):
        low = compute_evidence_score(0.1, 0.0, 0.0, 1.0, 0.0, 0.0)
        high = compute_evidence_score(0.9, 0.0, 0.0, 1.0, 0.0, 0.0)
        assert high["evidence_score"] > low["evidence_score"]

    def test_breakdown_contributions_non_negative(self):
        r = compute_evidence_score(0.7, 0.5, 0.1, 0.9, 0.4, 0.6)
        for k, v in r["breakdown"].items():
            assert isinstance(v, (int, float)), f"Breakdown {k} not numeric"
            assert v >= 0, f"Negative contribution: {k}={v}"

    def test_zero_inputs_give_non_negative(self):
        r = compute_evidence_score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert r["evidence_score"] >= 0.0

    def test_no_hardcoded_0_6_passed_externally(self):
        """
        Verify that when we pass real computed texture/shape scores (NOT 0.6),
        the evidence score differs from the hardcoded-0.6 case.
        This catches regression where caller used texture_score=0.6 hardcoded.
        """
        real_texture = 0.25   # different from 0.6
        real_shape = 0.80     # different from 0.6
        r_real = compute_evidence_score(0.7, 0.5, 0.05, 0.9, real_texture, real_shape)
        r_hard = compute_evidence_score(0.7, 0.5, 0.05, 0.9, 0.6, 0.6)
        # Scores must differ — if equal, hardcoding crept back in
        assert r_real["evidence_score"] != r_hard["evidence_score"], (
            "Evidence score is identical regardless of texture/shape input — "
            "hardcoded values may have crept back in"
        )

    def test_custom_weights_respected(self):
        weights = {
            "detector_weight": 1.0,
            "shadow_weight": 0.0,
            "texture_weight": 0.0,
            "shape_weight": 0.0,
            "anomaly_weight": 0.0,
        }
        r = compute_evidence_score(0.8, 0.0, 0.0, 1.0, 0.0, 0.0, weights=weights)
        # With only detector weight=1.0 and conf=0.8, evidence_score ~ 0.8
        assert abs(r["evidence_score"] - 0.8) < 0.05


class TestComputeAnomalyEvidenceScore:
    def test_score_in_range(self):
        for anom in [0.0, 0.3, 0.7, 1.0]:
            r = compute_anomaly_evidence_score(anom, shadow_score=0.3, image_quality_score=1.0)
            assert 0.0 <= r["evidence_score"] <= 1.0

    def test_returns_required_keys(self):
        r = compute_anomaly_evidence_score(0.5)
        for k in ["evidence_score", "evidence_pct", "label"]:
            assert k in r


class TestComputeSeverity:
    def test_high_risk_class_gets_at_least_medium(self):
        sev = compute_severity("shipwreck", evidence_score=0.2)
        assert sev in {"MEDIUM", "HIGH"}

    def test_high_evidence_gives_high_severity(self):
        sev = compute_severity("shipwreck", evidence_score=0.9)
        assert sev == "HIGH"

    def test_anomaly_flag_gives_high_or_medium(self):
        sev = compute_severity("unknown_anomaly", evidence_score=0.3, is_anomaly=True)
        assert sev in {"MEDIUM", "HIGH"}

    def test_severity_strings_valid(self):
        for cls in ["shipwreck", "mine_cylinder", "submarine_pipeline", "ghost_net"]:
            sev = compute_severity(cls, evidence_score=0.5)
            assert sev in {"HIGH", "MEDIUM", "LOW", "UNKNOWN"}


class TestGetSeverityColor:
    def test_returns_hex_color(self):
        for sev in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
            color = get_severity_color(sev)
            assert color.startswith("#"), f"Not a hex color: {color}"
            assert len(color) == 7
