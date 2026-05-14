"""
Pydantic schemas for the Crab Bimodal Emotion Classifier API.

Separated from app.py so that:
1. FastAPI auto-generates OpenAPI docs from these models.
2. Updating response/request schemas only requires editing this one file.
3. Other services (LangGraph nodes) can import these for type-checking.
"""
from pydantic import BaseModel, Field
from typing import Optional

INTERVIEW_CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]


class EmotionProbabilities(BaseModel):
    """Softmax probabilities for the 3 interview emotion classes."""
    Excited: float = Field(..., ge=0.0, le=1.0)
    Unconfident: float = Field(..., ge=0.0, le=1.0)
    Neutral_3Class: float = Field(..., ge=0.0, le=1.0)


class SingleClassifyResponse(BaseModel):
    """Response for a single audio classification request."""
    primary_label: str = Field(..., description="argmax of softmax (3-class)")
    primary_confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: EmotionProbabilities
    latency_ms: float = Field(..., description="Server-side inference latency in milliseconds")


class BatchResultItem(BaseModel):
    """One result within a batch classification response."""
    filename: str
    primary_label: str
    primary_confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: EmotionProbabilities


class BatchClassifyResponse(BaseModel):
    """Response for a batch audio classification request."""
    batch_size: int
    total_latency_ms: float
    avg_latency_ms: float
    results: list[BatchResultItem]


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""
    status: str = Field(..., description="'ok' when service is ready")
    model: str = Field(..., description="Model identifier, e.g. 'crab-bimodal-scheme2'")
    device: str = Field(..., description="Compute device, 'cuda' or 'cpu'")
    classes: list[str] = Field(..., description="Ordered list of output class names")
    gpu_name: Optional[str] = Field(None, description="GPU model name if available")
    vram_mb: Optional[int] = Field(None, description="Total GPU VRAM in MB if available")


class TimelineItem(BaseModel):
    """One window result in the long-audio timeline."""
    window: str = Field(..., description="Time range, e.g. '0.0~12.0s'")
    label: str = Field(..., description="Predicted emotion label for this window")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: EmotionProbabilities


class LongClassifyResponse(BaseModel):
    """Response for long-audio sliding-window classification."""
    final_label: str = Field(..., description="Overall emotion (probability-averaged)")
    final_confidence: float = Field(..., ge=0.0, le=1.0)
    avg_probabilities: EmotionProbabilities
    timeline: list[TimelineItem]
    total_windows: int
    audio_duration_sec: float = Field(..., description="Total audio length in seconds")
    latency_ms: float = Field(..., description="Server-side inference latency in milliseconds")
