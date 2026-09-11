import io
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def create_synthetic_sonar_image():
    img = np.full((300, 300, 3), 40, dtype=np.uint8)
    # Highlight
    cv2.circle(img, (150, 150), 20, (220, 220, 220), -1)
    # Shadow
    cv2.ellipse(img, (190, 150), (30, 15), 0, 0, 360, (5, 5, 5), -1)
    _, buf = cv2.imencode(".png", img)
    return io.BytesIO(buf.tobytes())

def test_system_status_endpoint(client):
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data.get("overall") in ["OPERATIONAL", "READY"]
    assert "inference_device" in data

def test_dashboard_stats_endpoint(client):
    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "sonar_frames" in data
    assert "anomalies_detected" in data
    assert "high_priority" in data

def test_map_targets_endpoint(client):
    response = client.get("/api/map/targets")
    assert response.status_code == 200
    data = response.json()
    assert "targets" in data
    assert "total_geolocated" in data

def test_pipeline_analyze_and_pdf_generation(client):
    img_buf = create_synthetic_sonar_image()
    files = {"file": ("test_sonar.png", img_buf, "image/png")}
    data = {
        "mission_id": "MISSION-001",
        "lat": 12.9234,
        "lon": 80.1256,
        "depth_m": 18.5
    }
    
    # 1. Post Analyze
    resp = client.post("/api/sonar/analyze", files=files, data=data)
    assert resp.status_code == 200
    res_data = resp.json()
    assert "analysis_id" in res_data
    analysis_id = res_data["analysis_id"]

    # 2. Raw and Processed Image Retrieval
    img_resp = client.get(f"/api/analysis/{analysis_id}/image?variant=raw")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"

    # 3. PDF Download
    pdf_resp = client.get(f"/api/reports/analysis/{analysis_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF")

def test_csv_and_json_export_endpoints(client):
    csv_resp = client.get("/api/export/csv")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]

    json_resp = client.get("/api/export/json")
    assert json_resp.status_code == 200
    jdata = json_resp.json()
    assert "detections" in jdata
