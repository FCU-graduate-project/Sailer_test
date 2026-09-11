"""Time-based HTTP soak test for the N3 emotion API.

This intentionally targets a real HTTP server rather than TestClient. It
checks every successful response against the public schema, samples health and
local GPU memory, and exits non-zero when configured service-level gates fail.

Example:
    Crab/.venv/bin/python -m Crab.api.soak_test \
      --base-url http://127.0.0.1:8001 \
      --duration-sec 600 --concurrency 2 \
      --output-json /tmp/n3-soak.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import requests

from .benchmark_api import (
    DEFAULT_TEXT,
    code_revision,
    make_wav,
    percentile,
    validate_response,
)
from .model_profile import N3_MODEL_ID
from .schemas import HealthResponse


def _model_validate(model, payload):
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model.parse_obj(payload)


@dataclass
class SharedResults:
    lock: threading.Lock = field(default_factory=threading.Lock)
    client_latencies_ms: list[float] = field(default_factory=list)
    queue_latencies_ms: list[float] = field(default_factory=list)
    inference_latencies_ms: list[float] = field(default_factory=list)
    server_latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    status_counts: Counter = field(default_factory=Counter)
    label_counts: Counter = field(default_factory=Counter)
    health_samples: list[dict] = field(default_factory=list)

    def record_success(
        self,
        client_ms: float,
        timings: dict[str, float],
        status_code: int,
        labels: list[str],
    ) -> None:
        with self.lock:
            self.client_latencies_ms.append(client_ms)
            self.queue_latencies_ms.append(timings["queue"])
            self.inference_latencies_ms.append(timings["inference"])
            self.server_latencies_ms.append(timings["server"])
            self.status_counts[str(status_code)] += 1
            self.label_counts.update(labels)

    def record_error(self, status_code: int | None, message: str) -> None:
        with self.lock:
            key = "exception" if status_code is None else str(status_code)
            self.status_counts[key] += 1
            if len(self.errors) < 20:
                self.errors.append(message[:500])

    def record_health(self, sample: dict) -> None:
        with self.lock:
            self.health_samples.append(sample)


def gpu_sample(enabled: bool = True) -> dict | None:
    """Read load-generator-local first-GPU state when explicitly enabled."""
    if not enabled:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = completed.stdout.strip().splitlines()[0]
        memory_mb, utilization_pct, temperature_c = [
            float(value.strip()) for value in first_line.split(",")
        ]
        return {
            "memory_used_mb": memory_mb,
            "utilization_pct": utilization_pct,
            "temperature_c": temperature_c,
        }
    except Exception:
        return None


def request_multipart(audio: bytes, text: str, batch_size: int) -> tuple[str, list[tuple]]:
    if batch_size == 1:
        return (
            "single",
            [
                ("audio", ("soak.wav", audio, "audio/wav")),
                ("text", (None, text)),
            ],
        )
    multipart = [
        ("files", (f"soak_{index}.wav", audio, "audio/wav"))
        for index in range(batch_size)
    ]
    multipart.extend(("texts", (None, text)) for _ in range(batch_size))
    return "batch", multipart


def health_probe(
    base_url: str,
    timeout_sec: float,
    sample_local_gpu: bool,
    token: str | None,
) -> dict:
    started = time.perf_counter()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(
        f"{base_url}/v1/health",
        headers=headers,
        timeout=timeout_sec,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(f"Health HTTP {response.status_code}: {response.text[:300]}")
    payload = response.json()
    _model_validate(HealthResponse, payload)
    if payload["model_id"] != N3_MODEL_ID:
        raise RuntimeError(f"Health model_id mismatch: {payload['model_id']!r}")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": elapsed_ms,
        "status": "ok",
        "model_id": payload["model_id"],
        "api_version": payload["api_version"],
        "runtime_policy": {
            "queue_capacity": payload["queue_capacity"],
            "request_capacity": payload["request_capacity"],
            "limits": payload["limits"],
        },
        "gpu": gpu_sample(sample_local_gpu),
    }


def summarize_metrics(values: list[float], prefix: str) -> dict:
    if not values:
        return {
            f"{prefix}_mean_ms": None,
            f"{prefix}_p50_ms": None,
            f"{prefix}_p95_ms": None,
            f"{prefix}_p99_ms": None,
            f"{prefix}_min_ms": None,
            f"{prefix}_max_ms": None,
        }
    return {
        f"{prefix}_mean_ms": statistics.mean(values),
        f"{prefix}_p50_ms": percentile(values, 0.50),
        f"{prefix}_p95_ms": percentile(values, 0.95),
        f"{prefix}_p99_ms": percentile(values, 0.99),
        f"{prefix}_min_ms": min(values),
        f"{prefix}_max_ms": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an HTTP soak test against N3")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-sec", type=float, default=3.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument(
        "--token",
        default=os.environ.get("CRAB_API_TOKEN"),
        help="Bearer token; defaults to CRAB_API_TOKEN",
    )
    parser.add_argument("--health-interval-sec", type=float, default=30.0)
    parser.add_argument("--request-interval-ms", type=float, default=0.0)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--min-success-rate", type=float, default=0.999)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-gpu-memory-growth-mb", type=float)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    args.base_url = args.base_url.rstrip("/")
    sample_local_gpu = args.base_url in {
        "http://127.0.0.1:8001",
        "http://localhost:8001",
    }
    expected_mode = "audio_text" if args.text.strip() else "audio_only"
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.batch_size < 1 or args.batch_size > 16:
        parser.error("--batch-size must be between 1 and 16")
    if args.audio_sec <= 0:
        parser.error("--audio-sec must be positive")
    if args.health_interval_sec <= 0:
        parser.error("--health-interval-sec must be positive")
    if not 0.0 <= args.min_success_rate <= 1.0:
        parser.error("--min-success-rate must be between 0 and 1")
    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")

    initial_health = health_probe(
        args.base_url,
        args.timeout_sec,
        sample_local_gpu,
        args.token,
    )
    print(
        f"Health OK: {initial_health['model_id']} "
        f"({initial_health['latency_ms']:.1f} ms)"
    )

    audio = make_wav(args.audio_sec)
    endpoint, multipart_template = request_multipart(audio, args.text, args.batch_size)
    path = (
        "/v1/emotion/classify"
        if endpoint == "single"
        else "/v1/emotion/classify-batch"
    )
    shared = SharedResults()
    shared.record_health(initial_health)
    stop_event = threading.Event()
    start_barrier = threading.Barrier(args.concurrency + 1)
    deadline = time.monotonic() + args.duration_sec
    request_counter = 0
    counter_lock = threading.Lock()

    def claim_request() -> bool:
        nonlocal request_counter
        with counter_lock:
            if args.max_requests is not None and request_counter >= args.max_requests:
                return False
            request_counter += 1
            return True

    def worker(worker_id: int) -> None:
        session = requests.Session()
        start_barrier.wait()
        while not stop_event.is_set() and time.monotonic() < deadline:
            if not claim_request():
                stop_event.set()
                break
            try:
                started = time.perf_counter()
                response = session.post(
                    f"{args.base_url}{path}",
                    files=multipart_template,
                    headers=(
                        {"Authorization": f"Bearer {args.token}"}
                        if args.token
                        else {}
                    ),
                    timeout=args.timeout_sec,
                )
                client_ms = (time.perf_counter() - started) * 1000.0
                if response.status_code != 200:
                    shared.record_error(
                        response.status_code,
                        f"worker={worker_id} HTTP {response.status_code}: "
                        f"{response.text[:300]}",
                    )
                else:
                    payload = response.json()
                    timings, _ = validate_response(
                        endpoint,
                        payload,
                        expected_mode,
                    )
                    if endpoint == "single":
                        labels = [payload["primary_label"]]
                    else:
                        labels = [item["primary_label"] for item in payload["results"]]
                    shared.record_success(
                        client_ms,
                        timings,
                        response.status_code,
                        labels,
                    )
            except Exception as exc:
                shared.record_error(None, f"worker={worker_id}: {type(exc).__name__}: {exc}")
            if args.request_interval_ms > 0:
                stop_event.wait(args.request_interval_ms / 1000.0)
        session.close()

    def health_sampler() -> None:
        while not stop_event.wait(args.health_interval_sec):
            try:
                shared.record_health(
                    health_probe(
                        args.base_url,
                        args.timeout_sec,
                        sample_local_gpu,
                        args.token,
                    )
                )
            except Exception as exc:
                shared.record_health(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "gpu": gpu_sample(sample_local_gpu),
                    }
                )

    sampler = threading.Thread(target=health_sampler, name="health-sampler", daemon=True)
    workers = [
        threading.Thread(target=worker, args=(index,), name=f"soak-worker-{index}")
        for index in range(args.concurrency)
    ]
    sampler.start()
    for thread in workers:
        thread.start()

    wall_started = time.perf_counter()
    start_barrier.wait()
    try:
        for thread in workers:
            thread.join()
    except KeyboardInterrupt:
        print("Interrupted; finishing partial report.")
        stop_event.set()
        for thread in workers:
            thread.join()
    finally:
        stop_event.set()
        sampler.join(timeout=min(5.0, args.health_interval_sec))
    wall_sec = time.perf_counter() - wall_started

    try:
        shared.record_health(
            health_probe(
                args.base_url,
                args.timeout_sec,
                sample_local_gpu,
                args.token,
            )
        )
    except Exception as exc:
        shared.record_health(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "gpu": gpu_sample(sample_local_gpu),
            }
        )

    successes = len(shared.client_latencies_ms)
    failures = sum(shared.status_counts.values()) - successes
    total = successes + failures
    success_rate = successes / total if total else 0.0
    health_failures = sum(
        sample.get("status") != "ok" for sample in shared.health_samples
    )
    gpu_memories = [
        sample["gpu"]["memory_used_mb"]
        for sample in shared.health_samples
        if sample.get("gpu") is not None
    ]
    gpu_growth_mb = (
        max(gpu_memories) - min(gpu_memories)
        if len(gpu_memories) >= 2
        else None
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model_id": N3_MODEL_ID,
        "api_version": initial_health["api_version"],
        "code_revision": code_revision(),
        "auth": "bearer-token-provided" if args.token else "disabled",
        "runtime_policy": initial_health["runtime_policy"],
        "endpoint": endpoint,
        "duration_target_sec": args.duration_sec,
        "duration_actual_sec": wall_sec,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "audio_sec": args.audio_sec,
        "input_mode": expected_mode,
        "timeout_sec": args.timeout_sec,
        "health_interval_sec": args.health_interval_sec,
        "request_interval_ms": args.request_interval_ms,
        "max_requests": args.max_requests,
        "requests": total,
        "successes": successes,
        "failures": failures,
        "success_rate": success_rate,
        "requests_per_sec": successes / wall_sec if wall_sec else 0.0,
        "items_per_sec": (
            successes * args.batch_size / wall_sec if wall_sec else 0.0
        ),
        "status_counts": dict(shared.status_counts),
        "label_counts": dict(shared.label_counts),
        "health_failures": health_failures,
        "health_samples": shared.health_samples,
        "gpu_memory_growth_mb": gpu_growth_mb,
        "first_errors": shared.errors,
        "synthetic_audio": True,
        "gpu_sample_source": (
            "load-generator-local-first-gpu"
            if sample_local_gpu
            else "disabled-for-remote-base-url"
        ),
        "gate_thresholds": {
            "min_success_rate": args.min_success_rate,
            "max_p95_ms": args.max_p95_ms,
            "max_gpu_memory_growth_mb": args.max_gpu_memory_growth_mb,
        },
    }
    summary.update(summarize_metrics(shared.client_latencies_ms, "client"))
    summary.update(summarize_metrics(shared.queue_latencies_ms, "queue"))
    summary.update(summarize_metrics(shared.inference_latencies_ms, "inference"))
    summary.update(summarize_metrics(shared.server_latencies_ms, "server"))

    gates = {
        "has_requests": total > 0,
        "success_rate": success_rate >= args.min_success_rate,
        "health": health_failures == 0,
    }
    if args.max_p95_ms is not None:
        gates["p95_latency"] = (
            summary["client_p95_ms"] is not None
            and summary["client_p95_ms"] <= args.max_p95_ms
        )
    if args.max_gpu_memory_growth_mb is not None:
        gates["gpu_memory_growth"] = (
            gpu_growth_mb is not None
            and gpu_growth_mb <= args.max_gpu_memory_growth_mb
        )
    summary["gates"] = gates
    summary["passed"] = all(gates.values())

    print(
        f"Soak complete: {successes}/{total} valid responses "
        f"({success_rate:.5%}), {summary['requests_per_sec']:.2f} req/s"
    )
    if summary["client_p95_ms"] is not None:
        print(
            f"Client latency p50/p95/p99: "
            f"{summary['client_p50_ms']:.2f}/"
            f"{summary['client_p95_ms']:.2f}/"
            f"{summary['client_p99_ms']:.2f} ms"
        )
    print(f"Health failures: {health_failures}; GPU memory growth: {gpu_growth_mb} MB")
    print(f"Gates: {gates}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.output_json}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
