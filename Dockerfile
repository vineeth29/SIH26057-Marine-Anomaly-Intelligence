# ── Stage 1: Build React Frontend ──
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python ML & FastAPI Backend ──
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

WORKDIR /app

# Install system dependencies needed for OpenCV and imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU build first for speed and lean image size
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy \
    pandas \
    scipy \
    Pillow \
    PyYAML \
    sqlalchemy \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    reportlab

# Copy application code and models
COPY ai/ ./ai/
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY data/ ./data/
COPY database/ ./database/
COPY models/ ./models/
COPY services/ ./services/
COPY utils/ ./utils/
COPY sih26057.db ./

# Copy built frontend assets from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose container port
EXPOSE 7860

# Run FastAPI via uvicorn from the repository root
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
