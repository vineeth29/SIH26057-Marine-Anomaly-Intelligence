# Offline Field Operation Guide

The SIH26057 platform is designed to operate completely disconnected from external networks, making it suitable for deployment aboard maritime vessels, survey crafts, and autonomous underwater vehicles.

---

## Zero-Cloud Dependencies

- **Local Neural Weights**: All models (`models/best.pt`, `models/anomaly/autoencoder.pt`) execute locally via PyTorch / ONNX runtime.
- **Embedded Database**: Relational storage utilizes SQLite (`sih26057.db`), requiring zero external database daemon.
- **Local Map & Telemetry**: Frontend tiles and vector canvas fall back gracefully to local SVG coordinate grids when external tile servers are unavailable.
- **Standalone PDF Engine**: Built-in ReportLab generates multi-page PDF survey dossiers directly on the host machine.

---

## Production Deployment Checklist

1. **Pre-bundle Frontend**:
   ```bash
   cd frontend && npm run build && cd ..
   ```
2. **Mount SSS Telemetry Feed**: Configure directory watcher or REST endpoint for incoming tow-fish raster frames.
3. **Execute Backend Daemon**:
   ```bash
   uvicorn backend.app.api:app --host 0.0.0.0 --port 8000 --workers 2
   ```
