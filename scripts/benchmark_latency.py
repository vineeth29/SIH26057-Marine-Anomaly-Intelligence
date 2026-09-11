"""
Pipeline latency benchmark for SIH26057.
Measures per-stage latency over multiple runs and saves report to evaluation/latency_report.json.

Usage:
    python scripts/benchmark_latency.py [--runs N] [--imgsz HxW]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from ai.preprocessing.sonar_preprocessor import SonarPreprocessor
from ai.shadow_analysis.shadow_analyzer import AcousticShadowAnalyzer
from ai.anomaly.anomaly_detector import AnomalyDetector
from ai.fusion.confidence_fusion import compute_evidence_score
from ai.fp_filter.false_positive_filter import apply_fp_filter
from ai.quality.dropout_detector import detect_dropout


def make_test_image(h: int, w: int) -> np.ndarray:
    """Synthetic sonar frame for benchmarking."""
    img = np.random.randint(40, 180, (h, w), dtype=np.uint8)
    for r in range(h):
        img[r, :] = np.clip(img[r, :] + int(50 * r / h), 0, 255)
    img[h // 3:h // 2, w // 4:3 * w // 4] = 210
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def time_ms(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, round((time.perf_counter() - t0) * 1000, 2)


def benchmark(n_runs: int, h: int, w: int) -> dict:
    print(f"\nBenchmarking pipeline: {n_runs} runs, image {w}x{h}")

    preprocessor = SonarPreprocessor()
    shadow_analyzer = AcousticShadowAnalyzer()

    try:
        anomaly_detector = AnomalyDetector()
        has_anomaly = True
    except Exception as e:
        print(f"  [Anomaly] Skipped: {e}")
        has_anomaly = False

    # Try loading YOLO
    yolo_model = None
    from pathlib import Path as P
    for pt in [P("models/best_multiclass.pt"), P("models/best.pt")]:
        if pt.exists():
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(str(pt))
                print(f"  [YOLO] Loaded: {pt.name}")
                break
            except Exception as e:
                print(f"  [YOLO] Failed to load {pt.name}: {e}")

    # Benchmark arrays
    timings = {
        "load_ms": [],
        "preprocess_ms": [],
        "yolo_ms": [],
        "dropout_ms": [],
        "shadow_ms": [],
        "anomaly_ms": [],
        "fusion_ms": [],
        "fp_filter_ms": [],
        "total_ms": [],
    }

    dummy_bbox = [w // 4, h // 3, 3 * w // 4, h // 2]

    for i in range(n_runs):
        t_run = time.perf_counter()

        # Image load (simulated)
        t0 = time.perf_counter()
        img = make_test_image(h, w)
        timings["load_ms"].append(round((time.perf_counter() - t0) * 1000, 2))

        # Preprocess
        t0 = time.perf_counter()
        prep = preprocessor.preprocess(img)
        preprocessed = prep["preprocessed"]
        timings["preprocess_ms"].append(round((time.perf_counter() - t0) * 1000, 2))

        # Dropout
        t0 = time.perf_counter()
        detect_dropout(img)
        timings["dropout_ms"].append(round((time.perf_counter() - t0) * 1000, 2))

        # YOLO
        if yolo_model:
            t0 = time.perf_counter()
            yolo_model(preprocessed, imgsz=min(w, 1280), verbose=False)
            timings["yolo_ms"].append(round((time.perf_counter() - t0) * 1000, 2))
        else:
            timings["yolo_ms"].append(0.0)

        # Shadow
        t0 = time.perf_counter()
        shadow_r = shadow_analyzer.analyze(preprocessed, dummy_bbox)
        timings["shadow_ms"].append(round((time.perf_counter() - t0) * 1000, 2))

        # Anomaly
        if has_anomaly:
            t0 = time.perf_counter()
            anomaly_detector.detect_anomalies(preprocessed)
            timings["anomaly_ms"].append(round((time.perf_counter() - t0) * 1000, 2))
        else:
            timings["anomaly_ms"].append(0.0)

        # Fusion
        t0 = time.perf_counter()
        compute_evidence_score(0.7, shadow_r.shadow_score, 0.05, 0.9, 0.4, 0.5)
        timings["fusion_ms"].append(round((time.perf_counter() - t0) * 1000, 2))

        # FP Filter
        t0 = time.perf_counter()
        apply_fp_filter("DET-BENCH", "shipwreck", 0.7, dummy_bbox, preprocessed, shadow_r.shadow_score)
        timings["fp_filter_ms"].append(round((time.perf_counter() - t0) * 1000, 2))

        timings["total_ms"].append(round((time.perf_counter() - t_run) * 1000, 2))

        if (i + 1) % 5 == 0:
            print(f"  Run {i+1}/{n_runs}: total={timings['total_ms'][-1]:.1f}ms  yolo={timings['yolo_ms'][-1]:.1f}ms")

    # Compute stats
    def stats(values):
        import statistics
        if not values or all(v == 0 for v in values):
            return {"mean_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        return {
            "mean_ms": round(statistics.mean(values), 2),
            "std_ms": round(statistics.stdev(values) if len(values) > 1 else 0.0, 2),
            "min_ms": round(min(values), 2),
            "max_ms": round(max(values), 2),
        }

    import torch
    report = {
        "hardware": {
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
        },
        "benchmark_config": {
            "n_runs": n_runs,
            "image_width": w,
            "image_height": h,
            "yolo_model": str(yolo_model.model_name if yolo_model and hasattr(yolo_model, 'model_name') else "not_loaded"),
        },
        "latency": {stage: stats(vals) for stage, vals in timings.items()},
    }

    # Print summary
    print("\n" + "="*60)
    print("LATENCY BENCHMARK RESULTS")
    print("="*60)
    print(f"  Device:          {report['hardware']['device']} ({report['hardware']['gpu_name']})")
    print(f"  Runs:            {n_runs}")
    print(f"  Image size:      {w}x{h}")
    for stage, s in report["latency"].items():
        if s["mean_ms"] > 0:
            print(f"  {stage:<20} {s['mean_ms']:>7.1f}ms ± {s['std_ms']:.1f}ms")
    print("="*60 + "\n")

    return report


def main():
    p = argparse.ArgumentParser(description="SIH26057 pipeline latency benchmark")
    p.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=512)
    args = p.parse_args()

    report = benchmark(args.runs, args.height, args.width)

    eval_dir = ROOT / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    out = eval_dir / "latency_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
