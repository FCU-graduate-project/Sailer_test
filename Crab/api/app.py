"""
Crab Bimodal Emotion API — FastAPI application.

Endpoints:
  GET  /v1/health                     → HealthResponse
  POST /v1/emotion/classify           → SingleClassifyResponse
  POST /v1/emotion/classify-batch     → BatchClassifyResponse
  POST /v1/emotion/classify-long      → LongClassifyResponse

Key design decisions:
  - asyncio.Lock() around GPU calls to prevent concurrent CUDA access
  - GPU warm-up during lifespan to eliminate first-request latency spike
  - text defaults to empty string "" so the API works audio-only (graceful degradation)
  - Batch endpoint allows up to 16 files in a single GPU forward pass
  - Long-audio endpoint uses sliding window (12s window, 6s stride) with batch GPU inference
"""

import asyncio
import io
import time
import logging
import os
import sys

# Ensure the root directory is in PYTHONPATH so 'src.models.ser' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional

from .inference import CrabEmotionPredictor, CLASSES
from .schemas import (
    SingleClassifyResponse,
    BatchClassifyResponse,
    BatchResultItem,
    EmotionProbabilities,
    HealthResponse,
    LongClassifyResponse,
    TimelineItem,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = os.environ.get(
    "CRAB_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "experiments", "interview_scheme1"),
)
DEVICE = os.environ.get("CRAB_DEVICE", "cuda")

logger = logging.getLogger("crab_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

predictor: Optional[CrabEmotionPredictor] = None
_gpu_lock = asyncio.Lock()  # Serialise GPU access to prevent CUDA collisions


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, run GPU warm-up, clean up on shutdown."""
    global predictor

    abs_model_dir = os.path.abspath(MODEL_DIR)
    logger.info(f"Loading Crab Bimodal models from: {abs_model_dir}")
    predictor = CrabEmotionPredictor(model_dir=abs_model_dir, device=DEVICE)
    logger.info("Models loaded successfully.")

    # GPU warm-up: run a dummy forward pass to initialise all CUDA kernels.
    # Without this, the first real request takes 1–2s extra.
    logger.info("Running GPU warm-up (dummy forward)...")
    predictor.warmup()
    logger.info("GPU warm-up complete. API is ready to serve.")

    yield

    del predictor
    logger.info("Models unloaded. Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Crab Bimodal Emotion API",
    version="1.0.0",
    description=(
        "WavLM-Large + RoBERTa-Large → 3-class Interview Emotion Classifier\n\n"
        "Classes: **Excited** · **Unconfident** · **Neutral_3Class**"
    ),
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# GET /v1/health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """Check service health and model status."""
    gpu_name = None
    vram_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        model=os.path.basename(MODEL_DIR),
        device=str(predictor.device),
        classes=CLASSES,
        gpu_name=gpu_name,
        vram_mb=vram_mb,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/emotion/classify  (single)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/emotion/classify", response_model=SingleClassifyResponse)
async def classify_single(
    audio: UploadFile = File(..., description="Audio file (WAV/MP3/FLAC, any sample rate)"),
    text: str = Form(default="", description="Transcript text. Empty string if unavailable."),
):
    """Classify a single audio clip into one of 3 interview emotions."""
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    async with _gpu_lock:
        t0 = time.perf_counter()
        try:
            probs = predictor.predict_single(audio_bytes, text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        latency_ms = (time.perf_counter() - t0) * 1000

    primary_label = max(probs, key=probs.get)
    return SingleClassifyResponse(
        primary_label=primary_label,
        primary_confidence=round(probs[primary_label], 4),
        probabilities=EmotionProbabilities(**{k: round(v, 4) for k, v in probs.items()}),
        latency_ms=round(latency_ms, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/emotion/classify-batch  (batch)
# ─────────────────────────────────────────────────────────────────────────────

MAX_BATCH = 16

@app.post("/v1/emotion/classify-batch", response_model=BatchClassifyResponse)
async def classify_batch(
    files: list[UploadFile] = File(..., description="Multiple audio files"),
    texts: list[str] = Form(default=[], description="Transcripts for each file (optional)"),
):
    """Classify multiple audio clips in a single GPU-parallel batch.

    If `texts` is empty, all clips use empty string (audio-only mode).
    If `texts` is provided, its length must match `files`.
    """
    n = len(files)
    if n == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    if n > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {n} exceeds maximum ({MAX_BATCH})",
        )
    if len(texts) == 0:
        texts = [""] * n
    
    logger.info(f"Batch inference request: Received {n} files and {len(texts)} texts.")
    
    if len(texts) != n:
        raise HTTPException(
            status_code=400,
            detail=f"files ({n}) and texts ({len(texts)}) count mismatch. Ensure clients send 'texts' fields repetitively in multipart form data.",
        )

    audio_bytes_list = [await f.read() for f in files]
    filenames = [f.filename or f"audio_{i}" for i, f in enumerate(files)]

    async with _gpu_lock:
        t0 = time.perf_counter()
        try:
            batch_probs = predictor.predict_batch(audio_bytes_list, texts)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        total_ms = (time.perf_counter() - t0) * 1000

    results = []
    for fname, probs in zip(filenames, batch_probs):
        primary_label = max(probs, key=probs.get)
        results.append(
            BatchResultItem(
                filename=fname,
                primary_label=primary_label,
                primary_confidence=round(probs[primary_label], 4),
                probabilities=EmotionProbabilities(
                    **{k: round(v, 4) for k, v in probs.items()}
                ),
            )
        )

    return BatchClassifyResponse(
        batch_size=n,
        total_latency_ms=round(total_ms, 2),
        avg_latency_ms=round(total_ms / n, 2),
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/emotion/classify-long  (sliding window for long audio)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/emotion/classify-long", response_model=LongClassifyResponse)
async def classify_long(
    audio: UploadFile = File(..., description="Audio file of any length (WAV/MP3/FLAC)"),
    text: str = Form(default="", description="Full transcript text (optional, shared across all windows)"),
    window_sec: float = Form(default=12.0, description="Window size in seconds (max 12)"),
    stride_sec: float = Form(default=6.0, description="Stride in seconds (overlap = window - stride)"),
):
    """Classify long audio using sliding-window segmentation.

    The audio is sliced into overlapping windows, each processed through
    the full WavLM + RoBERTa pipeline in GPU-parallel batches.

    Returns:
      - final_label: overall emotion via probability averaging
      - timeline: per-window emotion predictions for temporal analysis
    """
    # Validate parameters
    if window_sec < 1.0 or window_sec > 12.0:
        raise HTTPException(status_code=400, detail=f"window_sec must be 1.0~12.0, got {window_sec}")
    if stride_sec < 0.5 or stride_sec > window_sec:
        raise HTTPException(status_code=400, detail=f"stride_sec must be 0.5~window_sec, got {stride_sec}")

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    async with _gpu_lock:
        t0 = time.perf_counter()
        try:
            result = predictor.predict_long(
                audio_bytes, text,
                window_sec=window_sec,
                stride_sec=stride_sec,
                max_batch=MAX_BATCH,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        latency_ms = (time.perf_counter() - t0) * 1000

    # Convert timeline dicts to Pydantic models
    timeline_items = [
        TimelineItem(
            window=t["window"],
            label=t["label"],
            confidence=t["confidence"],
            probabilities=EmotionProbabilities(**t["probabilities"]),
        )
        for t in result["timeline"]
    ]

    return LongClassifyResponse(
        final_label=result["final_label"],
        final_confidence=result["final_confidence"],
        avg_probabilities=EmotionProbabilities(**result["avg_probabilities"]),
        timeline=timeline_items,
        total_windows=result["total_windows"],
        audio_duration_sec=result["audio_duration_sec"],
        latency_ms=round(latency_ms, 2),
    )
