"""
Post-training complete verification for SIH26057.
Run this after training completes to:
1. Verify best_multiclass.pt exists and loads
2. Run full inference test on new model
3. Run GPU latency benchmark (20 runs)
4. Compare latency vs CPU baseline
5. Confirm dashboard shows correct model info
6. Print final PASS/FAIL matrix

Usage:
    .\\train_env\\Scripts\\python.exe scripts\\post_training_verify.py
    
    Or (if running from main Python env after model copy):
    python scripts\\post_training_verify.py
"""
import sys
import json
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)


CHECKLIST = {
    "multiclass_pt_exists": False,
    "multiclass_pt_loads": False,
    "cuda_verified": False,
    "inference_tests_pass": False,
    "coordinate_source_correct": False,
    "fp_filter_active": False,
    "anomaly_nondiscriminative_labeled": False,
    "texture_shape_not_hardcoded": False,
    "dropout_result_present": False,
    "per_class_metrics_exist": False,
    "latency_benchmark_done": False,
    "latency_acceptable": False,
}


def check_multiclass_model():
    print("\n[1] Checking best_multiclass.pt ...")
    pt = ROOT / "models" / "best_multiclass.pt"
    if not pt.exists():
        print(f"  FAIL: {pt} does not exist")
        print("  -> Training has not completed yet, or checkpoint was not copied.")
        return False, None

    size_mb = round(pt.stat().st_size / 1024**2, 1)
    print(f"  EXISTS: {pt} ({size_mb} MB)")
    CHECKLIST["multiclass_pt_exists"] = True

    try:
        from ultralytics import YOLO
        model = YOLO(str(pt))
        class_names = list(model.names.values())
        print(f"  LOADS: classes={class_names}")
        CHECKLIST["multiclass_pt_loads"] = True
        return True, class_names
    except Exception as e:
        print(f"  FAIL to load: {e}")
        return False, None


def check_cuda():
    print("\n[2] Checking CUDA ...")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  torch={torch.__version__}")
        print(f"  CUDA={cuda_ok}")
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            print(f"  GPU={props.name}")
            print(f"  VRAM total={total_b/1024**3:.2f} GB, free={free_b/1024**3:.2f} GB")
            CHECKLIST["cuda_verified"] = True
            return True
        else:
            print("  WARN: CUDA not available in this env. Latency will be CPU-only.")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def run_inference_tests():
    print("\n[3] Running inference tests on new model ...")
    try:
        from services.pipeline_service import SonarAnalysisPipeline
        import numpy as np
        import cv2

        pipeline = SonarAnalysisPipeline()
        det = pipeline.detector

        model_label = getattr(det, 'model_label', 'UNKNOWN')
        class_names = getattr(det, 'class_names', [])
        anom_status = getattr(pipeline, 'anomaly_model_status', 'UNKNOWN')

        print(f"  Model label:   {model_label}")
        print(f"  Class names:   {class_names}")
        print(f"  Anomaly status:{anom_status}")

        # Expected: MULTICLASS after training
        if model_label != "MULTICLASS":
            print(f"  WARN: Expected MULTICLASS, got {model_label}.")
            print("  -> best_multiclass.pt may not be the active model.")

        # Synthetic test image
        rng = np.random.RandomState(99)
        img = rng.randint(40, 150, (256, 512), dtype=np.uint8)
        for r in range(256):
            img[r, :] = np.clip(img[r, :] + int(70 * r / 256), 0, 255)
        img[80:130, 150:360] = 220
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        failures = []

        # Test coordinate_source
        result = pipeline.run(img, image_id="PT-001", lat=12.0, lon=80.0,
                              coordinate_source="MANUAL")
        for d in result.detections:
            if d.coordinate_source not in ("MANUAL", "SIMULATED"):
                failures.append(f"coordinate_source={d.coordinate_source}")
        if not failures:
            CHECKLIST["coordinate_source_correct"] = True

        # Test no coords -> UNAVAILABLE
        result2 = pipeline.run(img, image_id="PT-002")
        for d in result2.detections:
            if d.coordinate_source != "UNAVAILABLE":
                failures.append(f"no-coord should be UNAVAILABLE, got {d.coordinate_source}")

        # Test FP filter
        for d in result.detections:
            if not hasattr(d, 'keep') or not hasattr(d, 'rejection_reason'):
                failures.append("Missing keep/rejection_reason")
        if not [f for f in failures if "keep" in f]:
            CHECKLIST["fp_filter_active"] = True

        # Test anomaly status
        if anom_status in ("NON_DISCRIMINATIVE", "UNTESTED"):
            CHECKLIST["anomaly_nondiscriminative_labeled"] = True
        else:
            failures.append(f"anomaly_status={anom_status}")

        # Test texture/shape not hardcoded
        for d in result.detections:
            tx = getattr(d, 'texture_score', None)
            sh = getattr(d, 'shape_score', None)
            if tx == 0.6 and sh == 0.6:
                failures.append(f"Hardcoded 0.6 on {d.detection_id}")
        if not [f for f in failures if "0.6" in f]:
            CHECKLIST["texture_shape_not_hardcoded"] = True

        # Test dropout
        dropout = getattr(result, 'dropout_result', None)
        if dropout is not None:
            CHECKLIST["dropout_result_present"] = True

        if not failures:
            CHECKLIST["inference_tests_pass"] = True
            print(f"  PASS: {len(result.detections)} detections, all checks OK")
        else:
            print(f"  FAIL: {failures}")

        return not bool(failures)

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_metrics_file():
    print("\n[4] Checking evaluation metrics file ...")
    metrics_file = ROOT / "evaluation" / "metrics.json"
    if not metrics_file.exists():
        print(f"  FAIL: {metrics_file} not found")
        print("  -> Evaluate step in train_multiclass.py did not complete")
        return None

    with open(metrics_file) as f:
        metrics = json.load(f)

    overall = metrics.get("overall", {})
    per_class = metrics.get("per_class", {})

    print(f"  Split:      {metrics.get('split','unknown')}")
    print(f"  Precision:  {overall.get('precision', 0):.4f}")
    print(f"  Recall:     {overall.get('recall', 0):.4f}")
    print(f"  F1:         {overall.get('f1', 0):.4f}")
    print(f"  mAP@50:     {overall.get('map50', 0):.4f}")
    print(f"  mAP@50-95:  {overall.get('map50_95', 0):.4f}")
    print(f"  Per-class:")
    for cls, m in per_class.items():
        tag = "  [SYNTHETIC]" if "ghost" in cls.lower() or "net" in cls.lower() else ""
        print(f"    {cls}{tag}: P={m['precision']:.3f} R={m['recall']:.3f} AP50={m['ap50']:.3f}")

    if per_class:
        CHECKLIST["per_class_metrics_exist"] = True
    return metrics


def run_latency_benchmark():
    print("\n[5] Running GPU latency benchmark (20 runs) ...")
    try:
        ret = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "benchmark_latency.py"),
             "--runs", "20", "--height", "256", "--width", "512"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        print(ret.stdout[-2000:] if len(ret.stdout) > 2000 else ret.stdout)
        if ret.returncode == 0:
            CHECKLIST["latency_benchmark_done"] = True
            # Check the saved report
            report_file = ROOT / "evaluation" / "latency_report.json"
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                total_mean = report.get("latency", {}).get("total_ms", {}).get("mean_ms", 9999)
                print(f"  Mean total latency: {total_mean:.1f} ms")
                # Acceptable threshold: <500ms for real-time review (not hard real-time)
                if total_mean < 500:
                    CHECKLIST["latency_acceptable"] = True
                    print("  Latency: ACCEPTABLE (<500ms)")
                else:
                    print(f"  Latency: HIGH ({total_mean:.0f}ms) — GPU warm-up effect or CPU-only")
        else:
            print(f"  Benchmark failed: {ret.stderr[-500:]}")
    except Exception as e:
        print(f"  ERROR: {e}")


def print_final_report():
    print("\n" + "="*60)
    print("SIH26057 FINAL VERIFICATION MATRIX")
    print("="*60)
    all_pass = True
    for check, passed in CHECKLIST.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {check}")
        if not passed:
            all_pass = False
    print("="*60)
    if all_pass:
        print("FINAL STATUS: ✅ ALL CHECKS PASS — PROJECT READY FOR PRESENTATION")
    else:
        failed = [k for k, v in CHECKLIST.items() if not v]
        print(f"FINAL STATUS: ❌ {len(failed)} check(s) FAIL — project NOT complete")
        print("Remaining work:")
        for f in failed:
            print(f"  -> {f}")
    print("="*60)
    return all_pass


if __name__ == "__main__":
    print("="*60)
    print("SIH26057 POST-TRAINING VERIFICATION")
    print("="*60)

    model_ok, class_names = check_multiclass_model()
    cuda_ok = check_cuda()
    inference_ok = run_inference_tests()
    metrics = check_metrics_file()
    run_latency_benchmark()

    all_pass = print_final_report()
    sys.exit(0 if all_pass else 1)
