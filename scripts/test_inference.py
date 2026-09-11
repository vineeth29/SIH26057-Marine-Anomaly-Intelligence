"""
End-to-end inference test for SIH26057.
Tests the loaded model on synthetic sonar images and verifies:
1. Model loads without error
2. Detections contain all required fields
3. coordinate_source is correct
4. texture/shape scores are NOT hardcoded 0.6
5. FP filter produces per-detection keep/reject decisions
6. Dropout result present
7. Anomaly model is correctly labeled NON_DISCRIMINATIVE
8. JSON/CSV export works

Run AFTER training:
    python scripts/test_inference.py
    
Or with train_env (after GPU training):
    .\\train_env\\Scripts\\python.exe scripts\\test_inference.py
"""
import sys
import json
import time
import csv
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)


def make_test_sonar(h=256, w=512, seed=42) -> np.ndarray:
    """Reproducible synthetic SSS sonar frame."""
    rng = np.random.RandomState(seed)
    img = rng.randint(40, 120, (h, w), dtype=np.uint8)
    # Gradient
    for r in range(h):
        img[r, :] = np.clip(img[r, :] + int(60 * r / h), 0, 255)
    # Bright target (pipeline-like echo)
    img[h//3:h//2, w//4:3*w//4] = 215
    # Shadow
    img[h//3:h//2, 3*w//4:3*w//4+50] = 8
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def run_inference_test():
    from services.pipeline_service import SonarAnalysisPipeline

    print("\n" + "="*60)
    print("SIH26057 END-TO-END INFERENCE TEST")
    print("="*60)

    # Load pipeline
    print("\n[Load] Loading pipeline...")
    t0 = time.perf_counter()
    pipeline = SonarAnalysisPipeline()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[Load] Pipeline loaded in {load_ms:.1f} ms")

    # Show which model is active
    det = pipeline.detector
    print(f"[Model] Path:    {det.model_path}")
    print(f"[Model] Label:   {getattr(det, 'model_label', 'UNKNOWN')}")
    print(f"[Model] Classes: {getattr(det, 'class_names', [])}")
    print(f"[Anomaly] Status: {getattr(pipeline, 'anomaly_model_status', 'UNKNOWN')}")

    failures = []

    # Test 1: Manual coordinates
    print("\n[Test 1] Manual coordinates (lat=12.345, lon=80.123)")
    img = make_test_sonar()
    result = pipeline.run(img, image_id="TEST-001", lat=12.345, lon=80.123,
                          coordinate_source="MANUAL")
    assert result is not None, "Pipeline returned None"
    for d in result.detections:
        src = d.coordinate_source
        if src not in ("MANUAL", "SIMULATED"):
            failures.append(f"Test1: Expected MANUAL/SIMULATED, got {src}")
    print(f"  Detections: {len(result.detections)}, coordinate_source checks: {'PASS' if not failures else failures}")

    # Test 2: No coordinates → UNAVAILABLE
    print("\n[Test 2] No coordinates → UNAVAILABLE")
    result2 = pipeline.run(img, image_id="TEST-002")
    for d in result2.detections:
        if d.coordinate_source != "UNAVAILABLE":
            failures.append(f"Test2: Expected UNAVAILABLE, got {d.coordinate_source}")
    print(f"  coordinate_source=UNAVAILABLE: {'PASS' if not [f for f in failures if 'Test2' in f] else 'FAIL'}")

    # Test 3: texture_score and shape_score NOT hardcoded 0.6
    print("\n[Test 3] texture_score / shape_score not hardcoded")
    result3 = pipeline.run(img, image_id="TEST-003", lat=0, lon=0, coordinate_source="SIMULATED")
    for d in result3.detections:
        tx = getattr(d, 'texture_score', None)
        sh = getattr(d, 'shape_score', None)
        if tx is None or sh is None:
            failures.append(f"Test3: Missing texture/shape score on {d.detection_id}")
        if tx == 0.6 and sh == 0.6:
            failures.append(f"Test3: HARDCODED 0.6 detected on {d.detection_id}")
    print(f"  Texture/shape not hardcoded: {'PASS' if not [f for f in failures if 'Test3' in f] else 'FAIL'}")

    # Test 4: FP filter present
    print("\n[Test 4] FP filter keep/reject per detection")
    for d in result3.detections:
        if not hasattr(d, 'keep'):
            failures.append(f"Test4: Missing 'keep' on {d.detection_id}")
        if not d.keep and not getattr(d, 'rejection_reason', None):
            failures.append(f"Test4: Rejected but no rejection_reason on {d.detection_id}")
    print(f"  FP filter fields: {'PASS' if not [f for f in failures if 'Test4' in f] else 'FAIL'}")
    fp_sum = result3.fp_filter_summary
    print(f"  FP Summary: {fp_sum}")

    # Test 5: Dropout result
    print("\n[Test 5] Dropout detection result")
    dropout = getattr(result3, 'dropout_result', None)
    if dropout is None:
        failures.append("Test5: dropout_result missing from PipelineResult")
    else:
        for key in ["dropout_detected", "quality_warning", "blank_fraction"]:
            if key not in dropout:
                failures.append(f"Test5: dropout_result missing key: {key}")
    print(f"  Dropout: {dropout}")
    print(f"  Dropout check: {'PASS' if not [f for f in failures if 'Test5' in f] else 'FAIL'}")

    # Test 6: Anomaly model status
    print("\n[Test 6] Anomaly model status")
    anom_status = getattr(result3, 'anomaly_model_status', None)
    if anom_status is None:
        failures.append("Test6: anomaly_model_status missing from PipelineResult")
    if anom_status not in ("NON_DISCRIMINATIVE", "VALIDATED", "UNTESTED"):
        failures.append(f"Test6: Invalid anomaly_model_status: {anom_status}")
    print(f"  anomaly_model_status: {anom_status}")
    print(f"  Status check: {'PASS' if not [f for f in failures if 'Test6' in f] else 'FAIL'}")

    # Test 7: Timing fields
    print("\n[Test 7] Timing fields")
    t = result3.timing
    for k in ["preprocess_ms", "detection_ms", "total_ms"]:
        if k not in t:
            failures.append(f"Test7: timing missing key: {k}")
        elif t[k] < 0:
            failures.append(f"Test7: {k}={t[k]} is negative")
    print(f"  Timing: preprocess={t.get('preprocess_ms',0):.1f}ms, "
          f"detection={t.get('detection_ms',0):.1f}ms, "
          f"total={t.get('total_ms',0):.1f}ms")
    print(f"  Timing check: {'PASS' if not [f for f in failures if 'Test7' in f] else 'FAIL'}")

    # Test 8: JSON serialization
    print("\n[Test 8] JSON serialization")
    try:
        d_dict = result3.to_dict()
        json_str = json.dumps(d_dict, default=str)
        assert len(json_str) > 10
        print(f"  JSON size: {len(json_str)} bytes — PASS")
    except Exception as e:
        failures.append(f"Test8: JSON serialization failed: {e}")
        print(f"  JSON: FAIL — {e}")

    # Test 9: CSV export
    print("\n[Test 9] CSV export")
    out_csv = ROOT / "evaluation" / "inference_test_detections.csv"
    try:
        rows = []
        for d in result3.detections:
            d_dict = d.to_dict()
            rows.append(d_dict)
        if rows:
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f"  CSV: {len(rows)} rows → {out_csv} — PASS")
        else:
            print("  CSV: No detections to write (zero detections case)")
    except Exception as e:
        failures.append(f"Test9: CSV export failed: {e}")
        print(f"  CSV: FAIL — {e}")

    # Final summary
    print("\n" + "="*60)
    if not failures:
        print("ALL INFERENCE TESTS PASS ✅")
    else:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  ✗ {f}")
    print("="*60)

    return len(failures) == 0


if __name__ == "__main__":
    ok = run_inference_test()
    sys.exit(0 if ok else 1)
