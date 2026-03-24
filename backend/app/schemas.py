from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class DetectionResult(BaseModel):
    label: str
    confidence: float
    start_ms: int
    end_ms: int


class AnalysisResponse(BaseModel):
    filename: str
    status: str
    message: str
    sample_rate_hz: int
    detections: list[DetectionResult]
