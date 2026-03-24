from fastapi import FastAPI, File, UploadFile

from app.config import settings
from app.schemas import AnalysisResponse, DetectionResult, HealthResponse


app = FastAPI(
    title="Sound Dashboard API",
    version="0.1.0",
    description=(
        "Backend scaffold for sound event detection and selective suppression."
    ),
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/config")
async def config() -> dict:
    return settings.model_dump()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)) -> AnalysisResponse:
    # This is a stub response that lets the frontend and backend contract stabilize
    # before the actual audio feature extraction and model inference are added.
    detections = [
        DetectionResult(
            label="speech",
            confidence=0.92,
            start_ms=0,
            end_ms=1000,
        )
    ]

    return AnalysisResponse(
        filename=file.filename,
        status="stub",
        message="Analysis pipeline not implemented yet; returning placeholder data.",
        sample_rate_hz=settings.sample_rate_hz,
        detections=detections,
    )
