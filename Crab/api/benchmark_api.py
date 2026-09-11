"""Reproducible N3 API benchmark matrix.

The matrix separates client-observed latency from the server inference time
reported by the API. It can run in-process through FastAPI TestClient or
against a real HTTP server on localhost/another host.

Examples:
    Crab/.venv/bin/python -m Crab.api.benchmark_api --transport testclient

    Crab/.venv/bin/python -m Crab.api.benchmark_api \
      --transport http --base-url http://127.0.0.1:8001 \
      --output-json /tmp/n3-benchmark.json \
      --output-csv /tmp/n3-benchmark.csv
"""

from __future__ import annotations

import argparse
from array import array
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import logging
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Callable
import wave

import requests

from .model_profile import INTERVIEW_CLASSES, N3_MODEL_ID
from .schemas import (
    BatchClassifyResponse,
    HealthResponse,
    LongClassifyResponse,
    SingleClassifyResponse,
)


DEFAULT_TEXT = "我對這份工作很有信心，也很期待加入團隊。"
ALLOWED_MODES = {"audio_text", "audio_only"}


def _model_validate(model, payload):
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model.parse_obj(payload)


def parse_csv_numbers(raw: str, value_type: type) -> list:
    values = [value_type(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile with interpolation for small benchmark sets."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def make_wav(duration_sec: float, sample_rate: int = 16000) -> bytes:
    """Generate deterministic low-amplitude PCM audio for speed/load tests."""
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    sample_count = max(1, int(round(duration_sec * sample_rate)))
    pattern_len = 1600
    pattern = array(
        "h",
        (
            int(
                32767
                * (
                    0.015 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate)
                    + 0.005 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate)
                )
            )
            for index in range(pattern_len)
        ),
    )
    samples = (pattern * ((sample_count + pattern_len - 1) // pattern_len))[:sample_count]
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return output.getvalue()


class ApiClient:
    """Small transport adapter shared by the benchmark scenarios."""

    def __init__(
        self,
        transport: str,
        base_url: str,
        timeout_sec: float,
        model_dir: Path,
        device: str,
        token: str | None,
    ) -> None:
        self.transport = transport
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.model_dir = model_dir
        self.device = device
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._context = None
        self._client = None
        self._previous_auth: tuple[bool, str | None] | None = None
        self._thread_local = threading.local()

    def __enter__(self):
        if self.transport == "testclient":
            os.environ["CRAB_MODEL_DIR"] = str(self.model_dir.resolve())
            os.environ["CRAB_MODEL_ID"] = N3_MODEL_ID
            os.environ["CRAB_DEVICE"] = self.device
            os.environ["CRAB_HF_OFFLINE"] = "1"
            logging.getLogger("httpx").setLevel(logging.WARNING)
            from fastapi.testclient import TestClient
            from . import app as app_module

            if self.token:
                self._previous_auth = (
                    app_module.REQUIRE_AUTH,
                    app_module.API_TOKEN,
                )
                app_module.REQUIRE_AUTH = True
                app_module.API_TOKEN = self.token
            self._context = TestClient(app_module.app)
            self._client = self._context.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._context is not None:
            self._context.__exit__(exc_type, exc, traceback)
        if self._previous_auth is not None:
            from . import app as app_module

            app_module.REQUIRE_AUTH, app_module.API_TOKEN = self._previous_auth
            self._previous_auth = None

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def get(self, path: str):
        if self.transport == "testclient":
            return self._client.get(path, headers=self.headers)
        return self._session().get(
            f"{self.base_url}{path}",
            headers=self.headers,
            timeout=self.timeout_sec,
        )

    def post(self, path: str, multipart: list[tuple]):
        if self.transport == "testclient":
            return self._client.post(path, files=multipart, headers=self.headers)
        return self._session().post(
            f"{self.base_url}{path}",
            files=multipart,
            headers=self.headers,
            timeout=self.timeout_sec,
        )


def code_revision() -> str:
    """Return a non-secret reproducibility marker for generated reports."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return f"{head}+dirty" if dirty else head
    except Exception:
        return "unavailable"


def _validate_probabilities(probabilities: dict) -> None:
    if list(probabilities) != list(INTERVIEW_CLASSES):
        raise ValueError("probability class keys/order mismatch")
    values = [float(value) for value in probabilities.values()]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probability is outside [0, 1]")
    if abs(sum(values) - 1.0) > 0.002:
        raise ValueError("rounded probability sum is outside tolerance")


def validate_response(endpoint: str, payload: dict, expected_mode: str) -> tuple[dict, int]:
    """Validate every public result and return split timings/work-item count."""
    if payload.get("model_id") != N3_MODEL_ID:
        raise ValueError(f"model_id mismatch: {payload.get('model_id')!r}")
    if payload.get("input_mode") != expected_mode:
        raise ValueError(
            f"input_mode mismatch: expected {expected_mode!r}, "
            f"got {payload.get('input_mode')!r}"
        )

    if endpoint == "single":
        _model_validate(SingleClassifyResponse, payload)
        _validate_probabilities(payload["probabilities"])
        if payload["primary_label"] != max(
            payload["probabilities"],
            key=payload["probabilities"].get,
        ):
            raise ValueError("single primary_label is not probability argmax")
        work_items = 1
    elif endpoint == "batch":
        _model_validate(BatchClassifyResponse, payload)
        if len(payload["results"]) != payload["batch_size"]:
            raise ValueError("batch results length does not match batch_size")
        for index, item in enumerate(payload["results"]):
            _validate_probabilities(item["probabilities"])
            if item["primary_label"] != max(
                item["probabilities"],
                key=item["probabilities"].get,
            ):
                raise ValueError(f"batch item {index} label is not probability argmax")
            if float(item["audio_duration_sec"]) <= 0:
                raise ValueError(f"batch item {index} has invalid duration")
        work_items = int(payload["batch_size"])
    elif endpoint == "long":
        _model_validate(LongClassifyResponse, payload)
        if len(payload["timeline"]) != payload["total_windows"]:
            raise ValueError("timeline length does not match total_windows")
        _validate_probabilities(payload["avg_probabilities"])
        for index, item in enumerate(payload["timeline"]):
            _validate_probabilities(item["probabilities"])
            if item["label"] != max(
                item["probabilities"],
                key=item["probabilities"].get,
            ):
                raise ValueError(f"long window {index} label is not probability argmax")
        work_items = int(payload["total_windows"])
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")

    timings = {
        "queue": float(payload["queue_wait_ms"]),
        "inference": float(payload["inference_ms"]),
        "server": float(payload["server_total_ms"]),
    }
    if timings["server"] + 0.1 < timings["inference"]:
        raise ValueError("server_total_ms is smaller than inference_ms")
    return timings, work_items


@dataclass(frozen=True)
class Scenario:
    name: str
    endpoint: str
    input_mode: str
    audio_sec: float
    request_batch: int
    concurrency: int
    request_count: int
    invoke: Callable


def make_single_invoker(client: ApiClient, audio: bytes, text: str, mode: str):
    def invoke():
        multipart = [("audio", ("benchmark.wav", audio, "audio/wav"))]
        if mode == "audio_text":
            multipart.append(("text", (None, text)))
        return client.post("/v1/emotion/classify", multipart)

    return invoke


def make_batch_invoker(
    client: ApiClient,
    audio: bytes,
    text: str,
    mode: str,
    batch_size: int,
):
    def invoke():
        multipart = [
            ("files", (f"benchmark_{index}.wav", audio, "audio/wav"))
            for index in range(batch_size)
        ]
        if mode == "audio_text":
            multipart.extend(("texts", (None, text)) for _ in range(batch_size))
        return client.post("/v1/emotion/classify-batch", multipart)

    return invoke


def make_long_invoker(client: ApiClient, audio: bytes, text: str, mode: str):
    def invoke():
        multipart = [
            ("audio", ("benchmark-long.wav", audio, "audio/wav")),
            ("window_sec", (None, "12")),
            ("stride_sec", (None, "6")),
        ]
        if mode == "audio_text":
            multipart.append(("text", (None, text)))
        return client.post("/v1/emotion/classify-long", multipart)

    return invoke


def run_scenario(scenario: Scenario, warmup: int) -> dict:
    for _ in range(warmup):
        response = scenario.invoke()
        if response.status_code != 200:
            raise RuntimeError(
                f"Warm-up failed for {scenario.name}: "
                f"{response.status_code} {response.text}"
            )
        validate_response(scenario.endpoint, response.json(), scenario.input_mode)

    client_latencies: list[float] = []
    queue_latencies: list[float] = []
    inference_latencies: list[float] = []
    server_latencies: list[float] = []
    response_sizes: list[int] = []
    work_items = 0
    failures: list[str] = []

    def one_request() -> tuple[float, dict, int, int]:
        started = time.perf_counter()
        response = scenario.invoke()
        client_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        payload = response.json()
        timings, items = validate_response(
            scenario.endpoint,
            payload,
            scenario.input_mode,
        )
        header_server_ms = float(
            response.headers.get("X-Server-Total-Ms", timings["server"])
        )
        timings["server"] = header_server_ms
        return client_ms, timings, len(response.content), items

    wall_started = time.perf_counter()
    if scenario.concurrency == 1:
        for _ in range(scenario.request_count):
            try:
                client_ms, timings, response_size, items = one_request()
                client_latencies.append(client_ms)
                queue_latencies.append(timings["queue"])
                inference_latencies.append(timings["inference"])
                server_latencies.append(timings["server"])
                response_sizes.append(response_size)
                work_items += items
            except Exception as exc:
                failures.append(str(exc))
    else:
        with ThreadPoolExecutor(max_workers=scenario.concurrency) as executor:
            futures = [executor.submit(one_request) for _ in range(scenario.request_count)]
            for future in as_completed(futures):
                try:
                    client_ms, timings, response_size, items = future.result()
                    client_latencies.append(client_ms)
                    queue_latencies.append(timings["queue"])
                    inference_latencies.append(timings["inference"])
                    server_latencies.append(timings["server"])
                    response_sizes.append(response_size)
                    work_items += items
                except Exception as exc:
                    failures.append(str(exc))
    wall_sec = time.perf_counter() - wall_started

    successes = len(client_latencies)
    requests_total = successes + len(failures)

    def stats(values: list[float], prefix: str) -> dict:
        if not values:
            return {
                f"{prefix}_mean_ms": None,
                f"{prefix}_p50_ms": None,
                f"{prefix}_p95_ms": None,
                f"{prefix}_min_ms": None,
                f"{prefix}_max_ms": None,
            }
        return {
            f"{prefix}_mean_ms": statistics.mean(values),
            f"{prefix}_p50_ms": percentile(values, 0.50),
            f"{prefix}_p95_ms": percentile(values, 0.95),
            f"{prefix}_min_ms": min(values),
            f"{prefix}_max_ms": max(values),
        }

    row = {
        "scenario": scenario.name,
        "endpoint": scenario.endpoint,
        "input_mode": scenario.input_mode,
        "audio_sec": scenario.audio_sec,
        "request_batch": scenario.request_batch,
        "concurrency": scenario.concurrency,
        "requests": requests_total,
        "successes": successes,
        "failures": len(failures),
        "success_rate": successes / requests_total if requests_total else 0.0,
        "wall_sec": wall_sec,
        "work_items": work_items,
        "requests_per_sec": successes / wall_sec if wall_sec else 0.0,
        "work_items_per_sec": work_items / wall_sec if wall_sec else 0.0,
        "response_bytes_mean": (
            statistics.mean(response_sizes) if response_sizes else None
        ),
        "first_errors": failures[:3],
    }
    row.update(stats(client_latencies, "client"))
    row.update(stats(queue_latencies, "queue"))
    row.update(stats(inference_latencies, "inference"))
    row.update(stats(server_latencies, "server"))
    if client_latencies and server_latencies:
        client_minus_server = [
            max(0.0, client_ms - server_ms)
            for client_ms, server_ms in zip(client_latencies, server_latencies)
        ]
        row.update(stats(client_minus_server, "client_minus_server"))
    else:
        row.update(stats([], "client_minus_server"))
    return row


def print_matrix(rows: list[dict]) -> None:
    print()
    print(
        f"{'scenario':34} {'mode':10} {'n':>3} {'c':>2} "
        f"{'ok':>7} {'client p50':>11} {'client p95':>11} "
        f"{'server p50':>11} {'items/s':>9}"
    )
    print("-" * 118)
    for row in rows:
        ok = f"{row['successes']}/{row['requests']}"

        def metric(name: str) -> str:
            value = row[name]
            return "-" if value is None else f"{value:.2f}"

        print(
            f"{row['scenario'][:34]:34} {row['input_mode']:10} "
            f"{row['request_batch']:>3} {row['concurrency']:>2} "
            f"{ok:>7} {metric('client_p50_ms'):>11} "
            f"{metric('client_p95_ms'):>11} {metric('server_p50_ms'):>11} "
            f"{row['work_items_per_sec']:>9.2f}"
        )


def write_reports(rows: list[dict], metadata: dict, json_path: Path | None, csv_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"metadata": metadata, "results": rows}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {json_path}")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [key for key in rows[0] if key != "first_errors"] if rows else []
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fieldnames})
        print(f"CSV report: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the N3 API benchmark matrix")
    parser.add_argument("--transport", choices=("testclient", "http"), default="testclient")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments"
        / "nnime_v1_warmstart_lora",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--durations", default="1,3,6,12")
    parser.add_argument("--batch-duration", type=float, default=3.0)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--long-durations", default="60,300")
    parser.add_argument("--modes", default="audio_text,audio_only")
    parser.add_argument("--concurrencies", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--token",
        default=os.environ.get("CRAB_API_TOKEN"),
        help="Bearer token; defaults to CRAB_API_TOKEN",
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    durations = parse_csv_numbers(args.durations, float)
    batch_sizes = parse_csv_numbers(args.batch_sizes, int)
    long_durations = parse_csv_numbers(args.long_durations, float)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    concurrencies = parse_csv_numbers(args.concurrencies, int)
    if any(mode not in ALLOWED_MODES for mode in modes):
        parser.error(f"--modes must be drawn from {sorted(ALLOWED_MODES)}")
    if "audio_text" in modes and not args.text.strip():
        parser.error("--text must be non-blank when --modes includes audio_text")
    if args.repeats <= 0 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup cannot be negative")
    if any(size < 1 or size > 16 for size in batch_sizes):
        parser.error("--batch-sizes values must be between 1 and 16")
    if any(value <= 0 for value in concurrencies):
        parser.error("--concurrencies values must be positive")
    if args.transport == "testclient" and any(value != 1 for value in concurrencies):
        print("TestClient matrix skips concurrency >1; use --transport http for queue/load tests.")

    rows: list[dict] = []
    with ApiClient(
        transport=args.transport,
        base_url=args.base_url,
        timeout_sec=args.timeout_sec,
        model_dir=args.model_dir,
        device=args.device,
        token=args.token,
    ) as client:
        health_response = client.get("/v1/health")
        if health_response.status_code != 200:
            raise RuntimeError(
                f"Health check failed: {health_response.status_code} "
                f"{health_response.text}"
            )
        health = health_response.json()
        _model_validate(HealthResponse, health)
        if health["model_id"] != N3_MODEL_ID:
            raise RuntimeError(f"Unexpected model_id: {health['model_id']}")

        for mode in modes:
            for duration in durations:
                audio = make_wav(duration)
                scenario = Scenario(
                    name=f"single-{duration:g}s",
                    endpoint="single",
                    input_mode=mode,
                    audio_sec=duration,
                    request_batch=1,
                    concurrency=1,
                    request_count=args.repeats,
                    invoke=make_single_invoker(client, audio, args.text, mode),
                )
                rows.append(run_scenario(scenario, args.warmup))

        batch_audio = make_wav(args.batch_duration)
        for mode in modes:
            for batch_size in batch_sizes:
                scenario = Scenario(
                    name=f"batch-{batch_size}x{args.batch_duration:g}s",
                    endpoint="batch",
                    input_mode=mode,
                    audio_sec=args.batch_duration,
                    request_batch=batch_size,
                    concurrency=1,
                    request_count=args.repeats,
                    invoke=make_batch_invoker(
                        client,
                        batch_audio,
                        args.text,
                        mode,
                        batch_size,
                    ),
                )
                rows.append(run_scenario(scenario, args.warmup))

        for mode in modes:
            for duration in long_durations:
                long_audio = make_wav(duration)
                scenario = Scenario(
                    name=f"long-{duration:g}s",
                    endpoint="long",
                    input_mode=mode,
                    audio_sec=duration,
                    request_batch=1,
                    concurrency=1,
                    request_count=args.repeats,
                    invoke=make_long_invoker(client, long_audio, args.text, mode),
                )
                rows.append(run_scenario(scenario, args.warmup))

        if args.transport == "http":
            for concurrency in concurrencies:
                if concurrency == 1:
                    continue
                scenario = Scenario(
                    name=f"single-{args.batch_duration:g}s-concurrency-{concurrency}",
                    endpoint="single",
                    input_mode="audio_text",
                    audio_sec=args.batch_duration,
                    request_batch=1,
                    concurrency=concurrency,
                    request_count=max(args.repeats * concurrency, concurrency),
                    invoke=make_single_invoker(
                        client,
                        batch_audio,
                        args.text,
                        "audio_text",
                    ),
                )
                rows.append(run_scenario(scenario, args.warmup))

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transport": args.transport,
        "base_url": args.base_url if args.transport == "http" else "in-process TestClient",
        "model_id": health["model_id"],
        "api_version": health["api_version"],
        "code_revision": code_revision(),
        "device": health["device"],
        "gpu_name": health.get("gpu_name"),
        "vram_mb": health.get("vram_mb"),
        "repeats": args.repeats,
        "warmup_requests_per_scenario": args.warmup,
        "auth": "bearer-token-provided" if args.token else "disabled",
        "runtime_policy": {
            "queue_capacity": health["queue_capacity"],
            "request_capacity": health["request_capacity"],
            "limits": health["limits"],
        },
        "synthetic_audio": True,
        "note": (
            "Speed/load benchmark only. Synthetic audio labels are not a quality metric. "
            "Each response is fully validated. Reports split client end-to-end, exact "
            "server total header, queue wait, and serialized inference time. "
            "client_minus_server is transport/client overhead, subject to clock-free "
            "same-client measurement."
        ),
    }
    print_matrix(rows)
    write_reports(rows, metadata, args.output_json, args.output_csv)
    failures = sum(row["failures"] for row in rows)
    print(f"\nScenarios: {len(rows)}; failed requests: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
