"""
Pydantic schemas for the Crab Bimodal Emotion Classifier API.

Separated from app.py so that:
1. FastAPI auto-generates OpenAPI docs from these models.
2. Updating response/request schemas only requires editing this one file.
3. Other services (LangGraph nodes) can import these for type-checking.
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional

INTERVIEW_CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
EmotionLabel = Literal["Excited", "Unconfident", "Neutral_3Class"]
SingleInputMode = Literal["audio_text", "audio_only"]
BatchInputMode = Literal["audio_text", "audio_only", "mixed"]


class ErrorResponse(BaseModel):
    """Stable JSON error envelope used by application-generated errors."""
    detail: str
    error_code: str = Field(..., description="Stable machine-readable error code")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


class EmotionProbabilities(BaseModel):
    """Softmax probabilities for the 3 interview emotion classes."""
    Excited: float = Field(..., ge=0.0, le=1.0)
    Unconfident: float = Field(..., ge=0.0, le=1.0)
    Neutral_3Class: float = Field(..., ge=0.0, le=1.0)


class RuntimeLimits(BaseModel):
    """Public resource policy needed by callers and reproducible benchmarks."""
    max_single_file_bytes: int = Field(..., gt=0)
    max_batch_total_bytes: int = Field(..., gt=0)
    max_long_file_bytes: int = Field(..., gt=0)
    max_text_chars: int = Field(..., gt=0)
    max_long_audio_sec: float = Field(..., gt=0.0)
    max_long_windows: int = Field(..., gt=0)
    max_batch_size: int = Field(..., gt=0)


class SingleClassifyResponse(BaseModel):
    """Response for a single audio classification request."""
    model_id: str = Field(..., description="Stable semantic model identifier")
    input_mode: SingleInputMode = Field(..., description="'audio_text' or 'audio_only'")
    primary_label: EmotionLabel = Field(..., description="argmax of softmax (3-class)")
    primary_confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: EmotionProbabilities
    audio_duration_sec: float = Field(..., gt=0.0)
    queue_wait_ms: float = Field(..., ge=0.0)
    inference_ms: float = Field(..., ge=0.0)
    server_total_ms: float = Field(..., ge=0.0)
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Server-side inference latency in milliseconds",
    )


class BatchResultItem(BaseModel):
    """One result within a batch classification response."""
    filename: str
    primary_label: EmotionLabel
    primary_confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: EmotionProbabilities
    audio_duration_sec: float = Field(..., gt=0.0)


class BatchClassifyResponse(BaseModel):
    """Response for a batch audio classification request."""
    model_id: str = Field(..., description="Stable semantic model identifier")
    input_mode: BatchInputMode = Field(
        ...,
        description="'audio_text', 'audio_only', or 'mixed'",
    )
    batch_size: int = Field(..., ge=1, le=16)
    total_latency_ms: float = Field(..., ge=0.0)
    avg_latency_ms: float = Field(..., ge=0.0)
    queue_wait_ms: float = Field(..., ge=0.0)
    inference_ms: float = Field(..., ge=0.0)
    server_total_ms: float = Field(..., ge=0.0)
    results: list[BatchResultItem]


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""
    status: Literal["ok"] = Field(..., description="'ok' when service is ready")
    api_version: str
    model: str = Field(..., description="Model identifier, e.g. 'crab-bimodal-scheme2'")
    model_id: str = Field(..., description="Stable semantic model identifier")
    ft_mode: str = Field(..., description="Checkpoint mode, e.g. 'lora' or 'full_ft'")
    ssl_type: str = Field(..., description="Audio base model")
    text_model_path: str = Field(..., description="Text base model")
    hf_offline: bool = Field(..., description="Whether Hugging Face is forced offline")
    device: str = Field(..., description="Compute device, 'cuda' or 'cpu'")
    classes: list[EmotionLabel] = Field(
        ...,
        description="Ordered list of output class names",
    )
    gpu_name: Optional[str] = Field(None, description="GPU model name if available")
    vram_mb: Optional[int] = Field(None, description="Total GPU VRAM in MB if available")
    queue_depth: int = Field(..., ge=0)
    queue_capacity: int = Field(..., ge=0)
    active_jobs: int = Field(..., ge=0, le=1)
    request_inflight: int = Field(..., ge=0)
    request_capacity: int = Field(..., ge=1)
    limits: RuntimeLimits


class LiveResponse(BaseModel):
    """Process liveness; does not claim that the model is ready."""
    status: Literal["alive"]
    version: str


class ReadyResponse(BaseModel):
    """Model readiness and bounded-queue state."""
    status: Literal["ready"]
    api_version: str
    model_id: str
    device: str
    queue_depth: int = Field(..., ge=0)
    queue_capacity: int = Field(..., ge=0)
    active_jobs: int = Field(..., ge=0, le=1)
    request_inflight: int = Field(..., ge=0)
    request_capacity: int = Field(..., ge=1)


class TimelineItem(BaseModel):
    """One window result in the long-audio timeline."""
    window: str = Field(..., description="Time range, e.g. '0.0~12.0s'")
    label: EmotionLabel = Field(
        ...,
        description="Predicted emotion label for this window",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: EmotionProbabilities


class LongClassifyResponse(BaseModel):
    """Response for long-audio sliding-window classification."""
    model_id: str = Field(..., description="Stable semantic model identifier")
    input_mode: SingleInputMode = Field(..., description="'audio_text' or 'audio_only'")
    final_label: EmotionLabel = Field(
        ...,
        description="Overall emotion (probability-averaged)",
    )
    final_confidence: float = Field(..., ge=0.0, le=1.0)
    avg_probabilities: EmotionProbabilities
    timeline: list[TimelineItem]
    total_windows: int = Field(..., ge=1)
    audio_duration_sec: float = Field(
        ...,
        ge=0.0,
        description="Total audio length in seconds",
    )
    queue_wait_ms: float = Field(..., ge=0.0)
    inference_ms: float = Field(..., ge=0.0)
    server_total_ms: float = Field(..., ge=0.0)
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Server-side inference latency in milliseconds",
    )
