"""Production-prepared FastAPI service for the Chinese N3 emotion model.

The service deliberately runs one GPU worker. Requests may wait in a bounded
queue, but concurrent CUDA forwards are never allowed.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
import logging
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from typing import Any, Callable, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, Security, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

# Make `src.models.ser` importable when launched from either workspace or Crab.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .config import SETTINGS
from .inference import CrabEmotionPredictor, InferenceInputError
from .schemas import (
    BatchClassifyResponse,
    BatchResultItem,
    EmotionProbabilities,
    ErrorResponse,
    HealthResponse,
    LiveResponse,
    LongClassifyResponse,
    ReadyResponse,
    RuntimeLimits,
    SingleClassifyResponse,
    TimelineItem,
)


API_VERSION = "1.3.0"
MODEL_DIR = SETTINGS.model_dir
MODEL_ID = SETTINGS.model_id
DEVICE = SETTINGS.device
MAX_BATCH = 16
MAX_SINGLE_FILE_BYTES = SETTINGS.max_single_file_bytes
MAX_BATCH_TOTAL_BYTES = SETTINGS.max_batch_total_bytes
MAX_LONG_FILE_BYTES = SETTINGS.max_long_file_bytes
MAX_TEXT_CHARS = SETTINGS.max_text_chars
MAX_INFLIGHT_REQUESTS = SETTINGS.max_inflight_requests
GPU_QUEUE_DEPTH = SETTINGS.gpu_queue_depth
GPU_QUEUE_TIMEOUT_SEC = SETTINGS.gpu_queue_timeout_sec
REQUIRE_AUTH = SETTINGS.require_auth
API_TOKEN = SETTINGS.api_token
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024

logger = logging.getLogger("crab_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class ApiFailure(Exception):
    """Stable API error safe to return to a caller."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(detail)
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        self.headers = headers or {}


@dataclass(frozen=True)
class GpuTiming:
    queue_wait_ms: float
    inference_ms: float


class RuntimeMetrics:
    """Small dependency-free metrics store; never records audio or transcript."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._requests: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._server_ms: deque[float] = deque(maxlen=10_000)
        self._queue_ms: deque[float] = deque(maxlen=10_000)
        self._inference_ms: deque[float] = deque(maxlen=10_000)

    def record_request(self, path: str, status_code: int, server_ms: float) -> None:
        with self._lock:
            self._requests[f"{path}|{status_code}"] += 1
            self._server_ms.append(server_ms)

    def record_error(self, error_code: str) -> None:
        with self._lock:
            self._errors[error_code] += 1

    def record_inference(self, timing: GpuTiming) -> None:
        with self._lock:
            self._queue_ms.append(timing.queue_wait_ms)
            self._inference_ms.append(timing.inference_ms)

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
        ordered = sorted(values)

        def percentile(p: float) -> float:
            index = min(len(ordered) - 1, int((len(ordered) - 1) * p))
            return round(ordered[index], 3)

        return {
            "count": len(ordered),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": round(ordered[-1], 3),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            requests = dict(self._requests)
            errors = dict(self._errors)
            server = list(self._server_ms)
            queue = list(self._queue_ms)
            inference = list(self._inference_ms)
        return {
            "uptime_sec": round(time.time() - self.started_at, 3),
            "requests_total": requests,
            "errors_total": errors,
            "timing_ms": {
                "server_total": self._summary(server),
                "queue_wait": self._summary(queue),
                "inference": self._summary(inference),
            },
        }


metrics = RuntimeMetrics()
predictor: Optional[CrabEmotionPredictor] = None
_gpu_lock = asyncio.Lock()
_gpu_executor: ThreadPoolExecutor | None = None
_executor_state_lock = threading.Lock()
_queue_state_lock = threading.Lock()
_gpu_outstanding = 0
_gpu_waiting = 0
_gpu_active = 0
_request_inflight = 0
_drain_tasks: set[asyncio.Task] = set()
_request_id_context: ContextVar[str | None] = ContextVar(
    "crab_request_id",
    default=None,
)


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _queue_snapshot() -> dict[str, int]:
    with _queue_state_lock:
        return {
            "queue_depth": _gpu_waiting,
            "queue_capacity": GPU_QUEUE_DEPTH,
            "active_jobs": _gpu_active,
            "outstanding_jobs": _gpu_outstanding,
            "request_inflight": _request_inflight,
            "request_capacity": MAX_INFLIGHT_REQUESTS,
        }


def _try_reserve_request() -> bool:
    """Bound requests before multipart files are copied into Python bytes."""
    global _request_inflight
    with _queue_state_lock:
        if _request_inflight >= MAX_INFLIGHT_REQUESTS:
            return False
        _request_inflight += 1
        return True


def _release_request() -> None:
    global _request_inflight
    with _queue_state_lock:
        _request_inflight -= 1


def _ensure_executor() -> ThreadPoolExecutor:
    global _gpu_executor
    with _executor_state_lock:
        if _gpu_executor is None:
            _gpu_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="crab-gpu",
            )
        return _gpu_executor


def _safe_request_id(raw: str | None) -> str:
    if raw and len(raw) <= 64 and re.fullmatch(r"[A-Za-z0-9._:-]+", raw):
        return raw
    return uuid4().hex


def _error_response(
    status_code: int,
    error_code: str,
    detail: str,
    request_id: str | None,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        detail=detail,
        error_code=error_code,
        request_id=request_id,
    )
    response = JSONResponse(
        status_code=status_code,
        content=_model_dump(payload),
        headers=headers,
    )
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


def _request_limit(path: str) -> int | None:
    if path == "/v1/emotion/classify":
        return MAX_SINGLE_FILE_BYTES + _MULTIPART_OVERHEAD_BYTES
    if path == "/v1/emotion/classify-batch":
        return MAX_BATCH_TOTAL_BYTES + _MULTIPART_OVERHEAD_BYTES
    if path == "/v1/emotion/classify-long":
        return MAX_LONG_FILE_BYTES + _MULTIPART_OVERHEAD_BYTES
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load, validate and warm the model before declaring readiness."""
    global predictor, _gpu_lock, _gpu_executor
    _gpu_lock = asyncio.Lock()
    abs_model_dir = os.path.abspath(MODEL_DIR)
    logger.info("Loading Crab N3 model from %s", abs_model_dir)
    predictor = CrabEmotionPredictor(
        model_dir=abs_model_dir,
        model_id=MODEL_ID,
        device=DEVICE,
    )
    _ensure_executor()
    predictor.warmup()
    logger.info("N3 warm-up complete; service is ready")
    try:
        yield
    finally:
        # A disconnected client may have left an in-flight executor job behind.
        # Keep its drain task alive until it has released the GPU lock and queue
        # counters, then stop the single worker.  This also makes repeated
        # TestClient/app lifecycles deterministic.
        if _drain_tasks:
            await asyncio.gather(*tuple(_drain_tasks), return_exceptions=True)
            _drain_tasks.clear()
        predictor = None
        with _executor_state_lock:
            executor = _gpu_executor
            _gpu_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        logger.info("N3 runtime and GPU executor shut down cleanly")


app = FastAPI(
    title="Crab Bimodal Emotion API",
    version=API_VERSION,
    description=(
        "N3 XLS-R + XLM-R LoRA → 3-class Chinese Interview Emotion Classifier"
    ),
    lifespan=lifespan,
)
documented_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
    description=(
        "Required on every endpoint except /v1/live when "
        "CRAB_REQUIRE_AUTH=true"
    ),
)


@app.exception_handler(ApiFailure)
async def api_failure_handler(request: Request, exc: ApiFailure) -> JSONResponse:
    metrics.record_error(exc.error_code)
    return _error_response(
        exc.status_code,
        exc.error_code,
        exc.detail,
        getattr(request.state, "request_id", None),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_failure_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.info(
        "request_id=%s validation_failed fields=%d",
        getattr(request.state, "request_id", None),
        len(exc.errors()),
    )
    metrics.record_error("VALIDATION_ERROR")
    return _error_response(
        422,
        "VALIDATION_ERROR",
        "Request fields are missing or invalid",
        getattr(request.state, "request_id", None),
    )


@app.exception_handler(StarletteHTTPException)
async def http_failure_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Normalize framework/parser errors that happen before a route executes."""
    messages = {
        400: ("MALFORMED_REQUEST", "Malformed request body"),
        404: ("NOT_FOUND", "Endpoint not found"),
        405: ("METHOD_NOT_ALLOWED", "HTTP method not allowed"),
    }
    error_code, detail = messages.get(
        exc.status_code,
        ("HTTP_ERROR", "HTTP request failed"),
    )
    metrics.record_error(error_code)
    headers = dict(exc.headers or {})
    return _error_response(
        exc.status_code,
        error_code,
        detail,
        getattr(request.state, "request_id", None),
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled_failure_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Keep response-validation and route-shaping failures on the JSON contract."""
    request_id = getattr(request.state, "request_id", None)
    elapsed_ms = (
        (time.perf_counter() - request.state.started_at) * 1000
        if hasattr(request.state, "started_at")
        else 0.0
    )
    logger.error(
        "request_id=%s unhandled_request_failure=true",
        request_id,
        exc_info=exc,
    )
    metrics.record_error("INTERNAL_ERROR")
    return _error_response(
        500,
        "INTERNAL_ERROR",
        "Internal server error",
        request_id,
        headers={
            "X-Server-Total-Ms": f"{elapsed_ms:.3f}",
            "Server-Timing": f"app;dur={elapsed_ms:.3f}",
        },
    )


@app.middleware("http")
async def request_controls_and_telemetry(request: Request, call_next):
    request_id = _safe_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    request.state.started_at = time.perf_counter()
    token = _request_id_context.set(request_id)
    path = request.url.path
    request_reserved = False

    try:
        limit = _request_limit(path)
        if request.method == "POST" and limit is not None:
            raw_length = request.headers.get("content-length")
            if raw_length is None:
                response = _error_response(
                    411,
                    "LENGTH_REQUIRED",
                    "Content-Length is required for inference uploads",
                    request_id,
                )
            else:
                try:
                    content_length = int(raw_length)
                except ValueError:
                    content_length = -1
                if content_length < 0:
                    response = _error_response(
                        400,
                        "INVALID_CONTENT_LENGTH",
                        "Content-Length must be a non-negative integer",
                        request_id,
                    )
                elif content_length > limit:
                    response = _error_response(
                        413,
                        "REQUEST_TOO_LARGE",
                        f"Request body exceeds the {limit}-byte limit",
                        request_id,
                    )
                else:
                    response = None
            if response is not None:
                metrics.record_error("REQUEST_REJECTED")
                return response

        if REQUIRE_AUTH and path != "/v1/live":
            authorization = request.headers.get("Authorization", "")
            scheme, _, supplied = authorization.partition(" ")
            valid = (
                scheme.lower() == "bearer"
                and bool(API_TOKEN)
                and secrets.compare_digest(supplied, API_TOKEN or "")
            )
            if not valid:
                metrics.record_error("UNAUTHORIZED")
                response = _error_response(
                    401,
                    "UNAUTHORIZED",
                    "A valid bearer token is required",
                    request_id,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                return response

        if request.method == "POST" and limit is not None:
            if not _try_reserve_request():
                metrics.record_error("REQUEST_CAPACITY_FULL")
                response = _error_response(
                    503,
                    "REQUEST_CAPACITY_FULL",
                    "Inference request capacity is full",
                    request_id,
                    headers={"Retry-After": "1"},
                )
                return response
            request_reserved = True

        response = await call_next(request)
        return response
    finally:
        if request_reserved:
            _release_request()
        elapsed_ms = (time.perf_counter() - request.state.started_at) * 1000
        response_obj = locals().get("response")
        status_code = getattr(response_obj, "status_code", 500)
        metrics.record_request(path, status_code, elapsed_ms)
        if response_obj is not None:
            response_obj.headers["X-Request-ID"] = request_id
            response_obj.headers["X-Server-Total-Ms"] = f"{elapsed_ms:.3f}"
            response_obj.headers["Server-Timing"] = f"app;dur={elapsed_ms:.3f}"
        logger.info(
            "request_id=%s method=%s path=%s status=%s server_total_ms=%.3f",
            request_id,
            request.method,
            path,
            status_code,
            elapsed_ms,
        )
        _request_id_context.reset(token)


def _require_predictor() -> CrabEmotionPredictor:
    if predictor is None:
        raise ApiFailure(503, "MODEL_NOT_READY", "Model is not ready")
    return predictor


def _reserve_gpu_job() -> None:
    global _gpu_outstanding, _gpu_waiting
    with _queue_state_lock:
        capacity = 1 + GPU_QUEUE_DEPTH
        if _gpu_outstanding >= capacity:
            raise ApiFailure(
                503,
                "GPU_QUEUE_FULL",
                "GPU inference queue is full",
                headers={"Retry-After": "1"},
            )
        _gpu_outstanding += 1
        _gpu_waiting += 1


async def _run_gpu_inference(
    callable_: Callable[..., Any],
    *args,
    **kwargs,
) -> tuple[Any, GpuTiming]:
    """Run one cancellation-safe job through a bounded single-GPU queue."""
    global _gpu_outstanding, _gpu_waiting, _gpu_active
    _reserve_gpu_job()
    acquired = False
    cleanup_deferred = False
    phase = "waiting"
    queue_started = time.perf_counter()
    try:
        try:
            await asyncio.wait_for(
                _gpu_lock.acquire(),
                timeout=GPU_QUEUE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError as exc:
            raise ApiFailure(
                503,
                "GPU_QUEUE_TIMEOUT",
                "Timed out waiting for GPU capacity",
                headers={"Retry-After": "1"},
            ) from exc
        acquired = True
        phase = "active"
        queue_wait_ms = (time.perf_counter() - queue_started) * 1000
        with _queue_state_lock:
            _gpu_waiting -= 1
            _gpu_active += 1

        inference_started = time.perf_counter()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            _ensure_executor(),
            partial(callable_, *args, **kwargs),
        )
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError:
            logger.warning(
                "request_id=%s cancelled_during_gpu=true; draining job",
                _request_id_context.get(),
            )
            async def drain_and_release() -> None:
                try:
                    await asyncio.shield(future)
                except Exception:
                    logger.exception("Cancelled GPU inference completed with an error")
                finally:
                    _finish_gpu_job(acquired=True, phase="active")

            cleanup_deferred = True
            drain_task = asyncio.create_task(drain_and_release())
            _drain_tasks.add(drain_task)
            drain_task.add_done_callback(_drain_tasks.discard)
            raise
        inference_ms = (time.perf_counter() - inference_started) * 1000
        timing = GpuTiming(queue_wait_ms=queue_wait_ms, inference_ms=inference_ms)
        metrics.record_inference(timing)
        return result, timing
    finally:
        if not cleanup_deferred:
            _finish_gpu_job(acquired=acquired, phase=phase)


def _finish_gpu_job(*, acquired: bool, phase: str) -> None:
    """Release queue accounting exactly once, including cancelled GPU jobs."""
    global _gpu_outstanding, _gpu_waiting, _gpu_active
    if acquired:
        _gpu_lock.release()
    with _queue_state_lock:
        if phase == "waiting":
            _gpu_waiting -= 1
        else:
            _gpu_active -= 1
        _gpu_outstanding -= 1


def _input_mode(texts: list[str]) -> str:
    has_text = [bool(text.strip()) for text in texts]
    if all(has_text):
        return "audio_text"
    if any(has_text):
        return "mixed"
    return "audio_only"


def _validate_text_chars(text: str) -> None:
    if len(text) > MAX_TEXT_CHARS:
        raise ApiFailure(
            400,
            "TEXT_TOO_LARGE",
            f"Transcript exceeds the {MAX_TEXT_CHARS}-character request limit",
        )


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    try:
        value = await upload.read(limit + 1)
    finally:
        await upload.close()
    if len(value) > limit:
        raise ApiFailure(
            413,
            "FILE_TOO_LARGE",
            f"Audio file exceeds the {limit}-byte limit",
        )
    if not value:
        raise ApiFailure(400, "EMPTY_AUDIO", "Audio file is empty")
    return value


def _server_elapsed_ms(request: Request) -> float:
    return (time.perf_counter() - request.state.started_at) * 1000


def _safe_inference_failure(exc: Exception) -> ApiFailure:
    if isinstance(exc, InferenceInputError):
        return ApiFailure(400, exc.code, exc.public_message)
    message = str(exc).lower()
    if "cuda" in message and "out of memory" in message:
        return ApiFailure(
            503,
            "GPU_OUT_OF_MEMORY",
            "GPU capacity is temporarily unavailable",
            headers={"Retry-After": "1"},
        )
    return ApiFailure(500, "INFERENCE_FAILED", "Inference failed")


def _gpu_identity(active_predictor: CrabEmotionPredictor) -> tuple[str | None, int | None]:
    try:
        import torch

        device = active_predictor.device
        if device.type != "cuda" or not torch.cuda.is_available():
            return None, None
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        return (
            torch.cuda.get_device_name(index),
            properties.total_memory // (1024 * 1024),
        )
    except Exception:
        return None, None


@app.get("/v1/live", response_model=LiveResponse)
async def live() -> LiveResponse:
    return LiveResponse(status="alive", version=API_VERSION)


@app.get(
    "/v1/ready",
    response_model=ReadyResponse,
    dependencies=[Security(documented_bearer)],
    responses={
        401: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def ready() -> ReadyResponse:
    active_predictor = _require_predictor()
    queue = _queue_snapshot()
    return ReadyResponse(
        status="ready",
        api_version=API_VERSION,
        model_id=active_predictor.model_id,
        device=str(active_predictor.device),
        queue_depth=queue["queue_depth"],
        queue_capacity=queue["queue_capacity"],
        active_jobs=queue["active_jobs"],
        request_inflight=queue["request_inflight"],
        request_capacity=queue["request_capacity"],
    )


@app.get(
    "/v1/health",
    response_model=HealthResponse,
    dependencies=[Security(documented_bearer)],
    responses={
        401: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def health() -> HealthResponse:
    active_predictor = _require_predictor()
    gpu_name, vram_mb = _gpu_identity(active_predictor)
    queue = _queue_snapshot()
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        model=active_predictor.model_id,
        model_id=active_predictor.model_id,
        ft_mode=active_predictor.ft_mode,
        ssl_type=active_predictor.ssl_type,
        text_model_path=active_predictor.text_model_path,
        hf_offline=active_predictor.offline,
        device=str(active_predictor.device),
        classes=active_predictor.classes,
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        queue_depth=queue["queue_depth"],
        queue_capacity=queue["queue_capacity"],
        active_jobs=queue["active_jobs"],
        request_inflight=queue["request_inflight"],
        request_capacity=queue["request_capacity"],
        limits=RuntimeLimits(
            max_single_file_bytes=MAX_SINGLE_FILE_BYTES,
            max_batch_total_bytes=MAX_BATCH_TOTAL_BYTES,
            max_long_file_bytes=MAX_LONG_FILE_BYTES,
            max_text_chars=MAX_TEXT_CHARS,
            max_long_audio_sec=SETTINGS.max_long_audio_sec,
            max_long_windows=SETTINGS.max_long_windows,
            max_batch_size=MAX_BATCH,
        ),
    )


@app.get(
    "/v1/metrics",
    dependencies=[Security(documented_bearer)],
    responses={401: {"model": ErrorResponse}},
)
async def service_metrics() -> dict[str, Any]:
    snapshot = metrics.snapshot()
    snapshot["queue"] = _queue_snapshot()
    snapshot["model_ready"] = predictor is not None
    if predictor is not None:
        snapshot["model_id"] = predictor.model_id
        try:
            import torch

            device = predictor.device
            if device.type == "cuda" and torch.cuda.is_available():
                index = device.index if device.index is not None else torch.cuda.current_device()
                snapshot["gpu_memory_mb"] = {
                    "allocated": round(torch.cuda.memory_allocated(index) / 1024**2, 2),
                    "reserved": round(torch.cuda.memory_reserved(index) / 1024**2, 2),
                }
        except Exception:
            pass
    return snapshot


@app.post(
    "/v1/emotion/classify",
    response_model=SingleClassifyResponse,
    dependencies=[Security(documented_bearer)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        411: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def classify_single(
    request: Request,
    audio: UploadFile = File(...),
    text: str = Form(default=""),
) -> SingleClassifyResponse:
    _validate_text_chars(text)
    audio_bytes = await _read_limited(audio, MAX_SINGLE_FILE_BYTES)
    active_predictor = _require_predictor()
    try:
        result, timing = await _run_gpu_inference(
            active_predictor.predict_single,
            audio_bytes,
            text,
            return_metadata=True,
        )
    except ApiFailure:
        raise
    except Exception as exc:
        failure = _safe_inference_failure(exc)
        if failure.status_code >= 500:
            logger.exception("Unexpected single-item inference failure")
        raise failure from exc

    probabilities = result["probabilities"]
    label = max(probabilities, key=probabilities.get)
    server_total_ms = _server_elapsed_ms(request)
    logger.info(
        "request_id=%s endpoint=single mode=%s audio_sec=%.3f "
        "queue_ms=%.3f inference_ms=%.3f",
        request.state.request_id,
        _input_mode([text]),
        result["audio_duration_sec"],
        timing.queue_wait_ms,
        timing.inference_ms,
    )
    return SingleClassifyResponse(
        model_id=active_predictor.model_id,
        input_mode=_input_mode([text]),
        primary_label=label,
        primary_confidence=round(probabilities[label], 4),
        probabilities=EmotionProbabilities(
            **{key: round(value, 4) for key, value in probabilities.items()}
        ),
        audio_duration_sec=round(result["audio_duration_sec"], 3),
        queue_wait_ms=round(timing.queue_wait_ms, 3),
        inference_ms=round(timing.inference_ms, 3),
        server_total_ms=round(server_total_ms, 3),
        latency_ms=round(timing.inference_ms, 3),
    )


@app.post(
    "/v1/emotion/classify-batch",
    response_model=BatchClassifyResponse,
    dependencies=[Security(documented_bearer)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        411: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def classify_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    texts: list[str] = Form(default=[]),
) -> BatchClassifyResponse:
    count = len(files)
    if count == 0:
        raise ApiFailure(400, "NO_FILES", "No files were provided")
    if count > MAX_BATCH:
        raise ApiFailure(
            400,
            "BATCH_TOO_LARGE",
            f"Batch size exceeds the maximum of {MAX_BATCH}",
        )
    if not texts:
        texts = [""] * count
    if len(texts) != count:
        raise ApiFailure(
            400,
            "BATCH_TEXT_COUNT_MISMATCH",
            "The number of texts must equal the number of files",
        )
    for text in texts:
        _validate_text_chars(text)

    filenames = [item.filename or f"audio_{index}" for index, item in enumerate(files)]
    audio_items: list[bytes] = []
    total_bytes = 0
    for upload in files:
        value = await _read_limited(upload, MAX_SINGLE_FILE_BYTES)
        total_bytes += len(value)
        if total_bytes > MAX_BATCH_TOTAL_BYTES:
            raise ApiFailure(
                413,
                "BATCH_BYTES_TOO_LARGE",
                f"Batch audio exceeds the {MAX_BATCH_TOTAL_BYTES}-byte limit",
            )
        audio_items.append(value)

    active_predictor = _require_predictor()
    try:
        predictions, timing = await _run_gpu_inference(
            active_predictor.predict_batch,
            audio_items,
            texts,
            return_metadata=True,
        )
    except ApiFailure:
        raise
    except Exception as exc:
        failure = _safe_inference_failure(exc)
        if failure.status_code >= 500:
            logger.exception("Unexpected batch inference failure")
        raise failure from exc
    if len(predictions) != count:
        raise ApiFailure(
            500,
            "RESULT_COUNT_MISMATCH",
            "Inference result count mismatch",
        )

    response_items: list[BatchResultItem] = []
    for filename, prediction in zip(filenames, predictions):
        probabilities = prediction["probabilities"]
        label = max(probabilities, key=probabilities.get)
        response_items.append(
            BatchResultItem(
                filename=filename,
                primary_label=label,
                primary_confidence=round(probabilities[label], 4),
                probabilities=EmotionProbabilities(
                    **{
                        key: round(value, 4)
                        for key, value in probabilities.items()
                    }
                ),
                audio_duration_sec=round(prediction["audio_duration_sec"], 3),
            )
        )
    server_total_ms = _server_elapsed_ms(request)
    return BatchClassifyResponse(
        model_id=active_predictor.model_id,
        input_mode=_input_mode(texts),
        batch_size=count,
        total_latency_ms=round(timing.inference_ms, 3),
        avg_latency_ms=round(timing.inference_ms / count, 3),
        queue_wait_ms=round(timing.queue_wait_ms, 3),
        inference_ms=round(timing.inference_ms, 3),
        server_total_ms=round(server_total_ms, 3),
        results=response_items,
    )


@app.post(
    "/v1/emotion/classify-long",
    response_model=LongClassifyResponse,
    dependencies=[Security(documented_bearer)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        411: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def classify_long(
    request: Request,
    audio: UploadFile = File(...),
    text: str = Form(default=""),
    window_sec: float = Form(default=12.0),
    stride_sec: float = Form(default=6.0),
) -> LongClassifyResponse:
    if not 1.0 <= window_sec <= 12.0:
        raise ApiFailure(
            400,
            "INVALID_WINDOW",
            "window_sec must be between 1.0 and 12.0",
        )
    if not 0.5 <= stride_sec <= window_sec:
        raise ApiFailure(
            400,
            "INVALID_STRIDE",
            "stride_sec must be between 0.5 and window_sec",
        )
    _validate_text_chars(text)
    audio_bytes = await _read_limited(audio, MAX_LONG_FILE_BYTES)
    active_predictor = _require_predictor()
    try:
        result, timing = await _run_gpu_inference(
            active_predictor.predict_long,
            audio_bytes,
            text,
            window_sec=window_sec,
            stride_sec=stride_sec,
            max_batch=MAX_BATCH,
        )
    except ApiFailure:
        raise
    except Exception as exc:
        failure = _safe_inference_failure(exc)
        if failure.status_code >= 500:
            logger.exception("Unexpected long-audio inference failure")
        raise failure from exc

    timeline = [
        TimelineItem(
            window=item["window"],
            label=item["label"],
            confidence=item["confidence"],
            probabilities=EmotionProbabilities(**item["probabilities"]),
        )
        for item in result["timeline"]
    ]
    server_total_ms = _server_elapsed_ms(request)
    return LongClassifyResponse(
        model_id=active_predictor.model_id,
        input_mode=_input_mode([text]),
        final_label=result["final_label"],
        final_confidence=result["final_confidence"],
        avg_probabilities=EmotionProbabilities(**result["avg_probabilities"]),
        timeline=timeline,
        total_windows=result["total_windows"],
        audio_duration_sec=result["audio_duration_sec"],
        queue_wait_ms=round(timing.queue_wait_ms, 3),
        inference_ms=round(timing.inference_ms, 3),
        server_total_ms=round(server_total_ms, 3),
        latency_ms=round(timing.inference_ms, 3),
    )
