"""
SIH26057 backend API — thin FastAPI layer over the existing
Python pipeline (services/, ai/, utils/). No AI logic lives here.

Run with:
    uvicorn backend.app.main:app --reload --port 8000
(from the repository root)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router

app = FastAPI(
    title="SIH26057 Marine Anomaly Intelligence API",
    version="0.1.0",
)

# Vite dev server default ports. Adjust/add production origin(s) as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
