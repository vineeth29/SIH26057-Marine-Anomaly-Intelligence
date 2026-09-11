# SIH26057 — Standalone Backend + Frontend Setup

This package is a self-contained copy of the pieces the FastAPI backend
actually imports at runtime, so it can run without the rest of the original
Streamlit project. Nothing here is a stub or a mock — `backend/app/api.py`
calls the real pipeline in `services/pipeline_service.py`, which loads the
real trained weights in `models/best.pt` and `models/anomaly/autoencoder.pt`.

## What's included and why

Every one of these is imported (directly or transitively) by
`backend/app/pipeline_bridge.py`:

```
services/            pipeline_service.py, mission_service.py
database/            models.py (SQLAlchemy schema)
ai/                  preprocessing/, detection/, anomaly/, shadow_analysis/, fusion/
utils/               pdf_report.py (reusable; not yet wired to an endpoint)
configs/             model.yaml, preprocessing.yaml, confidence.yaml, etc.
models/              best.pt (trained YOLOv8), anomaly/autoencoder.pt
backend/             the FastAPI app itself
frontend/             React/Vite app (src/ only — run `npm install` to get node_modules)
```

`configs/*.yaml` and `models/*.pt` are loaded via **relative paths**
(e.g. `Path("models/best.pt")`), not paths resolved against `__file__`.
This means **the server must be started with your current working
directory at the project root** (where this file lives) — see the run
command below. Starting it from inside `backend/` or anywhere else will
fail to find the model and config files.

## Verified dependency set

This exact list was installed and tested end-to-end in a clean sandbox
(Python 3.12) to produce a working server:

```
torch                    (2.14.0 tested; CPU-only build is sufficient)
opencv-python-headless   (see note below — NOT opencv-python)
ultralytics
numpy
PyYAML
Pillow
sqlalchemy
fastapi
uvicorn[standard]
python-multipart
```

**Note on `opencv-python-headless` vs `opencv-python`:** the original
`requirements.txt` lists `opencv-python`, which bundles GUI bindings
(`cv2.imshow`, etc.) that need system display libraries. Since this
backend runs headless (no display, server/API context) and the codebase
does not call any GUI functions, `opencv-python-headless` was used
instead — it provides the identical `cv2` API used by this project.
`pip install opencv-python` also works fine if you prefer to match the
original file exactly; either satisfies `import cv2`.

Install everything with:

```bash
pip install -r requirements.txt -r backend/requirements.txt
```

(`requirements.txt` at the project root is the original project's full
list, which also includes Streamlit-only packages not needed to run the
API — they don't conflict, so installing them is harmless if you'd
rather not maintain a separate minimal list.)

## Running the backend

From the project root (this directory):

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

You should see:

```
INFO:     Application startup complete.
[SonarDetector] Loaded YOLO model from models/best.pt
```

That second line is the real trained model loading — if you instead see
a `FileNotFoundError` about `models/best.pt`, you are not running from
the project root.

Verify it's alive:

```bash
curl http://127.0.0.1:8000/api/health
# {"status":"ok","model_loaded":true,"version":"0.1.0"}
```

Interactive API docs: http://127.0.0.1:8000/docs

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

`frontend/.env` already points `VITE_API_BASE_URL` at
`http://localhost:8000/api`, matching the backend command above.

## Known limitations (carried over from the source project, not introduced here)

- `GET /api/model`'s `metrics` field, when populated from
  `models/model_info.json`, reflects training-time validation metrics
  (the split Ultralytics used for checkpoint selection) — not an
  independent held-out test evaluation. `metrics_source` says so
  explicitly. Run `evaluate.py` against a genuine test split to get a
  `reports/evaluation_results.json`, which takes priority when present.
- Geolocation is currently simulated end-to-end, not parsed from real
  sonar metadata — `GET /api/system/status` flags this with a `WARNING`
  on the `Geolocation` component rather than reporting it as `READY`.
- The annotated-image cache in `pipeline_bridge.py` is in-memory and
  per-process; it does not survive a server restart and won't work
  correctly if you ever run multiple uvicorn workers.
- The `crab_pot` class exists in the model's embedded class list (it's
  index 0 in the trained checkpoint) but has no severity/classification
  entry in `ai/fusion/confidence_fusion.py` or
  `services/pipeline_service.py`, since it was excluded from training
  data — a `crab_pot` detection (unlikely, given that) will report
  `UNCERTAIN` type and `MEDIUM` severity by default rather than a
  judgment the model was never meaningfully trained to support.

## Model update history

- **Current model** (`models/best.pt`): YOLOv8s, 4 real classes —
  `submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder` (plus
  an unused `crab_pot` class-index carried over from the label map).
  Training-time validation mAP50 ≈ 0.712 per `models/model_info.json`.
  Full per-epoch training log and confusion matrix from this run are in
  `models/training_artifacts/`.
- **Previous model**: YOLOv8-Nano, single class (`Pipeline`), SubPipe
  dataset only. Superseded by the current model above; not included in
  this package.
- Updating the model required three code changes beyond swapping the
  `.pt` file, since class names are looked up in a few explicit
  registries rather than derived generically:
  `ai/fusion/confidence_fusion.py` (`HIGH_RISK_CLASSES` etc.),
  `services/pipeline_service.py` (`ARTIFICIAL_CLASSES`,
  `CLASS_DISPLAY_NAMES`). `ai/detection/yolo_detector.py`'s
  `KNOWN_CLASSES` and `configs/model.yaml`'s `classes:` list were also
  updated for accuracy, though neither is actually read by any code —
  they exist for human reference only.
