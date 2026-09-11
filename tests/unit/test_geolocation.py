"""
Unit tests for geolocation utilities.
Verifies coordinate_source values, no fake GPS, and correct labeling.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.geolocation import build_geolocation, format_coords_display


class TestBuildGeolocation:
    def test_unavailable_when_no_coords(self):
        geo = build_geolocation(lat=None, lon=None)
        assert geo["available"] is False
        assert geo["status"] == "UNAVAILABLE"
        assert geo["lat"] is None
        assert geo["lon"] is None

    def test_simulated_when_is_simulated_true(self):
        geo = build_geolocation(lat=12.345, lon=80.123, is_simulated=True)
        assert geo["available"] is True
        assert geo["status"] == "SIMULATED"
        assert "simulated" in geo["label"].lower() or "Simulated" in geo["label"]

    def test_real_gps_when_is_simulated_false(self):
        geo = build_geolocation(lat=12.345, lon=80.123, is_simulated=False)
        assert geo["status"] == "REAL" or geo["status"] == "REAL_GPS"
        # Must NOT say simulated
        assert "simulated" not in geo["label"].lower()

    def test_coordinates_rounded(self):
        geo = build_geolocation(lat=12.3456789, lon=80.1234567, is_simulated=True)
        assert geo["lat"] is not None
        # Should be rounded to at most 6 decimal places
        assert len(str(geo["lat"]).split(".")[-1]) <= 6

    def test_depth_included_when_provided(self):
        geo = build_geolocation(lat=10.0, lon=20.0, depth_m=15.5, is_simulated=True)
        assert geo["depth_m"] == 15.5

    def test_no_fake_gps_without_coords(self):
        """Critical: no GPS fix can be claimed without coordinates."""
        geo = build_geolocation()
        assert geo["available"] is False
        assert geo["lat"] is None
        assert geo["lon"] is None
        assert "GPS" not in geo.get("label", "") or "unavailable" in geo.get("label", "").lower()

    def test_timestamp_auto_filled(self):
        geo = build_geolocation(lat=10.0, lon=20.0, is_simulated=True)
        assert geo.get("timestamp") is not None


class TestFormatCoordsDisplay:
    def test_unavailable_label(self):
        geo = build_geolocation(lat=None, lon=None)
        display = format_coords_display(geo)
        assert "unavailable" in display.lower() or "Unavailable" in display

    def test_simulated_label_contains_coords(self):
        geo = build_geolocation(lat=12.345, lon=80.123, is_simulated=True)
        display = format_coords_display(geo)
        assert "12" in display
        assert "80" in display

    def test_simulated_tag_present(self):
        geo = build_geolocation(lat=12.345, lon=80.123, is_simulated=True)
        display = format_coords_display(geo)
        assert "simulated" in display.lower() or "[Simulated]" in display

    def test_gps_tag_when_real(self):
        geo = build_geolocation(lat=12.345, lon=80.123, is_simulated=False)
        display = format_coords_display(geo)
        assert "GPS" in display or "Real" in display or "gps" in display.lower()
