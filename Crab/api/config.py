"""Environment-backed, fail-fast configuration for the N3 API."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}")
    return value


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    value = default if raw is None else float(raw)
    if value <= minimum:
        raise RuntimeError(f"{name} must be > {minimum}")
    return value


@dataclass(frozen=True)
class ApiSettings:
    model_dir: str
    model_id: str | None
    device: str
    hf_offline: bool
    max_single_file_bytes: int
    max_batch_total_bytes: int
    max_long_file_bytes: int
    max_text_chars: int
    max_long_audio_sec: float
    max_long_windows: int
    max_inflight_requests: int
    gpu_queue_depth: int
    gpu_queue_timeout_sec: float
    require_auth: bool
    api_token: str | None

    @classmethod
    def from_env(cls) -> "ApiSettings":
        api_dir = Path(__file__).resolve().parent
        model_dir = os.environ.get(
            "CRAB_MODEL_DIR",
            str(api_dir.parent / "experiments" / "nnime_v1_warmstart_lora"),
        )
        require_auth = _bool("CRAB_REQUIRE_AUTH", False)
        token = os.environ.get("CRAB_API_TOKEN")
        if require_auth and (token is None or len(token) < 24):
            raise RuntimeError(
                "CRAB_REQUIRE_AUTH=true requires CRAB_API_TOKEN with at least 24 characters"
            )
        gpu_queue_depth = _int("CRAB_GPU_QUEUE_DEPTH", 8, minimum=0)
        return cls(
            model_dir=model_dir,
            model_id=os.environ.get("CRAB_MODEL_ID"),
            device=os.environ.get("CRAB_DEVICE", "cuda"),
            hf_offline=_bool("CRAB_HF_OFFLINE", True),
            max_single_file_bytes=_int(
                "CRAB_MAX_SINGLE_FILE_BYTES", 10 * 1024 * 1024
            ),
            max_batch_total_bytes=_int(
                "CRAB_MAX_BATCH_TOTAL_BYTES", 64 * 1024 * 1024
            ),
            max_long_file_bytes=_int(
                "CRAB_MAX_LONG_FILE_BYTES", 100 * 1024 * 1024
            ),
            max_text_chars=_int("CRAB_MAX_TEXT_CHARS", 4096),
            max_long_audio_sec=_float("CRAB_MAX_LONG_AUDIO_SEC", 300.0),
            max_long_windows=_int("CRAB_MAX_LONG_WINDOWS", 64),
            max_inflight_requests=_int(
                "CRAB_MAX_INFLIGHT_REQUESTS",
                1 + gpu_queue_depth,
            ),
            gpu_queue_depth=gpu_queue_depth,
            gpu_queue_timeout_sec=_float("CRAB_GPU_QUEUE_TIMEOUT_SEC", 30.0),
            require_auth=require_auth,
            api_token=token,
        )


SETTINGS = ApiSettings.from_env()
