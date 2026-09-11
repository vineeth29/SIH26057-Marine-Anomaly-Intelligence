# Setup and Installation Guide

## 1. Environment Preparation

### Recommended Environment
- OS: Windows 10/11, Ubuntu 20.04+, or macOS
- Python: 3.10, 3.11, or 3.14
- Node.js: 18.x or 20.x

### Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

---

## 2. Dependencies Installation

```bash
# Core backend and AI requirements
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

---

## 3. Running Test Suites

```bash
# Run complete test suite (101 unit + integration tests)
python -m pytest

# Run latency benchmark
python scripts/benchmark_latency.py
```

---

## 4. Running the Complete System

### Terminal 1 (FastAPI Backend):
```bash
python -m uvicorn backend.app.api:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2 (React Operator Frontend):
```bash
npm --prefix frontend run dev
```

Visit `http://127.0.0.1:5173` to access the mission console.
