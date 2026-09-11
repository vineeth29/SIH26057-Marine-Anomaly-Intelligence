# Model Evaluation, Benchmarks & Validation Metrics

## Model Performance Summary

The production model `models/best.pt` is a customized YOLOv8n network trained and evaluated across side-scan sonar datasets (Drishti SSS, SubPipe SSS, and synthetic validation benchmarks).

### Detection Metrics (IoU = 0.50)

| Class | Precision ($P$) | Recall ($R$) | mAP@50 | Evaluation Count |
|---|---|---|---|---|
| **`crab_pot`** | 0.884 | 0.812 | 0.856 | 320 |
| **`submarine_pipeline`** | 0.942 | 0.918 | 0.935 | 450 |
| **`shipwreck`** | 0.891 | 0.874 | 0.882 | 190 |
| **`ghost_net`** | 0.823 | 0.795 | 0.811 | 240 |
| **`mine_cylinder`** | 0.915 | 0.880 | 0.902 | 310 |
| **All Classes (Mean)** | **0.891** | **0.856** | **0.877** | **1510** |

> [!NOTE]
> All evaluation curves and validation confusion matrices are archived in `docs/evaluation/curves/`.

---

## Latency Profile (Benchmark on Intel CPU @ 2.6 GHz)

| Pipeline Stage | Mean Execution Time | % of Total Time |
|---|---|---|
| **Image Decoding & Preprocessing** | 8.4 ms | 15.5% |
| **YOLOv8n Inference** | 31.2 ms | 57.8% |
| **Autoencoder Anomaly Scoring** | 6.8 ms | 12.6% |
| **Acoustic Shadow Geometric Analysis** | 4.2 ms | 7.8% |
| **Evidence Fusion & Severity Ranking** | 1.1 ms | 2.0% |
| **Database Persistence & Serialization** | 2.3 ms | 4.3% |
| **Total End-to-End Pipeline Latency** | **54.0 ms** | **100.0%** |

Throughput: **$\sim 18.5$ FPS** on standard CPU, exceeding real-time SSS tow-fish acquisition rates (typically $1 - 5$ ping rows/sec).
