"""Industry-style HTTP load profile for the N3 API.

Supports closed-loop concurrency, fixed-rate and Poisson arrivals, multiple
measurement windows, stability selection, and saturation sweeps. Only stable
windows are used for the headline result when stability is reached.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import threading
import time
from typing import Any

import requests

from .benchmark_api import (
    DEFAULT_TEXT,
    code_revision,
    make_wav,
    percentile,
    validate_response,
)
from .model_profile import N3_MODEL_ID


@dataclass
class Record:
    completed_at: float
    status_code: int | None
    client_ms: float
    queue_ms: float | None = None
    inference_ms: float | None = None
    server_ms: float | None = None
    error_code: str | None = None
    scheduler_lag_ms: float = 0.0


class Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.records: list[Record] = []
        self.dropped = 0

    def append(self, record: Record) -> None:
        with self.lock:
            self.records.append(record)

    def drop(self) -> None:
        with self.lock:
            self.dropped += 1


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }


def summarize_records(records: list[Record], duration_sec: float) -> dict[str, Any]:
    successes = [record for record in records if record.status_code == 200]
    statuses = Counter(
        "exception" if record.status_code is None else str(record.status_code)
        for record in records
    )
    errors = Counter(
        record.error_code or "UNKNOWN"
        for record in records
        if record.status_code != 200
    )
    total = len(records)
    return {
        "requests": total,
        "successes": len(successes),
        "failures": total - len(successes),
        "success_rate": len(successes) / total if total else 0.0,
        "throughput_rps": len(successes) / duration_sec if duration_sec else 0.0,
        "status_counts": dict(statuses),
        "error_codes": dict(errors),
        "client": metric_summary([record.client_ms for record in successes]),
        "queue": metric_summary(
            [record.queue_ms for record in successes if record.queue_ms is not None]
        ),
        "inference": metric_summary(
            [
                record.inference_ms
                for record in successes
                if record.inference_ms is not None
            ]
        ),
        "server": metric_summary(
            [record.server_ms for record in successes if record.server_ms is not None]
        ),
        "scheduler_lag": metric_summary(
            [record.scheduler_lag_ms for record in records]
        ),
    }


def coefficient_range(values: list[float]) -> float:
    if not values:
        return math.inf
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0 if max(values) == 0 else math.inf
    return (max(values) - min(values)) / mean


def select_stable_windows(
    windows: list[dict[str, Any]],
    required: int,
    tolerance: float,
) -> list[int]:
    """Return the first consecutive stable window indexes, or an empty list."""
    for end in range(required, len(windows) + 1):
        indexes = list(range(end - required, end))
        selected = [windows[index] for index in indexes]
        if any(window["requests"] == 0 for window in selected):
            continue
        throughputs = [window["throughput_rps"] for window in selected]
        p95_values = [
            window["client"]["p95_ms"]
            for window in selected
            if window["client"]["p95_ms"] is not None
        ]
        if len(p95_values) != required:
            continue
        if (
            coefficient_range(throughputs) <= tolerance
            and coefficient_range(p95_values) <= tolerance
        ):
            return indexes
    return []


class HttpLoadClient:
    def __init__(
        self,
        base_url: str,
        audio: bytes,
        text: str,
        timeout_sec: float,
        token: str | None,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/v1/emotion/classify"
        self.audio = audio
        self.text = text
        self.timeout_sec = timeout_sec
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.local = threading.local()

    def request(self, scheduled_at: float | None = None) -> Record:
        now = time.perf_counter()
        scheduler_lag_ms = (
            max(0.0, now - scheduled_at) * 1000 if scheduled_at is not None else 0.0
        )
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            self.local.session = session
        started = time.perf_counter()
        try:
            response = session.post(
                self.url,
                files={"audio": ("load.wav", self.audio, "audio/wav")},
                data={"text": self.text},
                headers=self.headers,
                timeout=self.timeout_sec,
            )
            client_ms = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                try:
                    error_code = response.json().get("error_code", "UNKNOWN")
                except Exception:
                    error_code = "INVALID_ERROR_BODY"
                return Record(
                    completed_at=time.perf_counter(),
                    status_code=response.status_code,
                    client_ms=client_ms,
                    error_code=error_code,
                    scheduler_lag_ms=scheduler_lag_ms,
                )
            payload = response.json()
            timings, _ = validate_response("single", payload, "audio_text")
            server_ms = float(
                response.headers.get("X-Server-Total-Ms", timings["server"])
            )
            return Record(
                completed_at=time.perf_counter(),
                status_code=200,
                client_ms=client_ms,
                queue_ms=timings["queue"],
                inference_ms=timings["inference"],
                server_ms=server_ms,
                scheduler_lag_ms=scheduler_lag_ms,
            )
        except Exception as exc:
            return Record(
                completed_at=time.perf_counter(),
                status_code=None,
                client_ms=(time.perf_counter() - started) * 1000,
                error_code=type(exc).__name__,
                scheduler_lag_ms=scheduler_lag_ms,
            )


def run_closed_loop(
    client: HttpLoadClient,
    concurrency: int,
    duration_sec: float,
) -> tuple[list[Record], int]:
    recorder = Recorder()
    deadline = time.perf_counter() + duration_sec
    barrier = threading.Barrier(concurrency + 1)

    def worker() -> None:
        barrier.wait()
        while time.perf_counter() < deadline:
            recorder.append(client.request())

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    return recorder.records, recorder.dropped


def run_open_loop(
    client: HttpLoadClient,
    mode: str,
    rate: float,
    duration_sec: float,
    max_workers: int,
    max_pending: int,
    seed: int,
) -> tuple[list[Record], int]:
    recorder = Recorder()
    pending = threading.BoundedSemaphore(max_pending)
    randomizer = random.Random(seed)
    started = time.perf_counter()
    deadline = started + duration_sec
    scheduled = started

    def submit_one(executor: ThreadPoolExecutor, due: float) -> None:
        if not pending.acquire(blocking=False):
            recorder.drop()
            return

        def work() -> None:
            try:
                recorder.append(client.request(due))
            finally:
                pending.release()

        executor.submit(work)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while scheduled < deadline:
            delay = scheduled - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            submit_one(executor, scheduled)
            if mode == "fixed":
                scheduled += 1.0 / rate
            else:
                scheduled += randomizer.expovariate(rate)
    return recorder.records, recorder.dropped


def build_windows(
    records: list[Record],
    run_started: float,
    warmup_sec: float,
    measurement_sec: float,
    window_sec: float,
) -> tuple[list[dict[str, Any]], list[list[Record]]]:
    bucket_count = max(1, math.ceil(measurement_sec / window_sec))
    buckets: list[list[Record]] = [[] for _ in range(bucket_count)]
    measurement_start = run_started + warmup_sec
    for record in records:
        offset = record.completed_at - measurement_start
        if 0 <= offset < measurement_sec:
            index = int(offset // window_sec)
            buckets[index].append(record)
    summaries = []
    for index, bucket in enumerate(buckets):
        relative_start = index * window_sec
        duration = min(window_sec, measurement_sec - relative_start)
        summaries.append(
            {
                "index": index,
                "start_sec": warmup_sec + relative_start,
                "end_sec": warmup_sec + relative_start + duration,
                "duration_sec": duration,
                **summarize_records(bucket, duration),
            }
        )
    return summaries, buckets


def run_level(args, client: HttpLoadClient, level: float) -> dict[str, Any]:
    duration_sec = args.warmup_sec + args.measurement_sec
    run_started = time.perf_counter()
    if args.mode == "closed-loop":
        records, dropped = run_closed_loop(client, int(level), duration_sec)
    else:
        records, dropped = run_open_loop(
            client,
            args.mode,
            float(level),
            duration_sec,
            args.max_workers,
            args.max_pending,
            args.seed,
        )
    windows, buckets = build_windows(
        records,
        run_started,
        args.warmup_sec,
        args.measurement_sec,
        args.window_sec,
    )
    stable_indexes = select_stable_windows(
        windows,
        args.stable_windows,
        args.stability_tolerance,
    )
    if stable_indexes:
        accepted_records = [
            record
            for index in stable_indexes
            for record in buckets[index]
        ]
        accepted_duration = sum(
            windows[index]["duration_sec"] for index in stable_indexes
        )
        source = "stable_windows"
    else:
        accepted_records = [record for bucket in buckets for record in bucket]
        accepted_duration = sum(window["duration_sec"] for window in windows)
        source = "all_measurement_windows_unstable"
    headline = summarize_records(accepted_records, accepted_duration)
    headline["dropped_by_load_generator"] = dropped
    return {
        "level": level,
        "level_unit": "concurrency" if args.mode == "closed-loop" else "requests_per_sec",
        "stable": bool(stable_indexes),
        "stability_tolerance": args.stability_tolerance,
        "accepted_window_indexes": stable_indexes,
        "headline_source": source,
        "headline": headline,
        "windows": windows,
    }


def saturation_summary(levels: list[dict[str, Any]], args) -> dict[str, Any]:
    last_good = None
    first_failure = None
    previous_throughput = None
    plateau_count = 0
    for result in levels:
        headline = result["headline"]
        p95 = headline["client"]["p95_ms"]
        failure_reasons: list[str] = []
        if not result["stable"]:
            failure_reasons.append("unstable_measurement")
        if headline["success_rate"] < args.min_success_rate:
            failure_reasons.append("success_rate")
        if args.max_p95_ms is not None and (
            p95 is None or p95 > args.max_p95_ms
        ):
            failure_reasons.append("p95_latency")
        if headline["dropped_by_load_generator"] > 0:
            failure_reasons.append("load_generator_drop")
        throughput = headline["throughput_rps"]
        if previous_throughput is not None:
            gain = (
                (throughput - previous_throughput) / previous_throughput
                if previous_throughput > 0
                else math.inf
            )
            plateau_count = plateau_count + 1 if gain < args.min_throughput_gain else 0
            if plateau_count >= 2:
                failure_reasons.append("throughput_plateau")
        previous_throughput = throughput
        result["failure_reasons"] = failure_reasons
        if failure_reasons and first_failure is None:
            first_failure = result["level"]
        if not failure_reasons and first_failure is None:
            last_good = result["level"]
    return {
        "last_good_level": last_good,
        "first_failure_or_saturation_level": first_failure,
        "criteria": {
            "requires_stable_measurement": True,
            "min_success_rate": args.min_success_rate,
            "max_p95_ms": args.max_p95_ms,
            "min_throughput_gain": args.min_throughput_gain,
            "plateau_consecutive_levels": 2,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="N3 API open/closed-loop load test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--mode",
        choices=("closed-loop", "fixed", "poisson"),
        required=True,
    )
    parser.add_argument(
        "--levels",
        required=True,
        help="Comma-separated concurrency values or request rates",
    )
    parser.add_argument("--audio-sec", type=float, default=3.0)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--warmup-sec", type=float, default=5.0)
    parser.add_argument("--measurement-sec", type=float, default=20.0)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--stable-windows", type=int, default=3)
    parser.add_argument("--stability-tolerance", type=float, default=0.20)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--max-pending", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--min-throughput-gain", type=float, default=0.05)
    parser.add_argument(
        "--token",
        default=os.environ.get("CRAB_API_TOKEN"),
        help="Bearer token; defaults to CRAB_API_TOKEN",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    levels = [float(value.strip()) for value in args.levels.split(",") if value.strip()]
    if not levels or any(value <= 0 for value in levels):
        parser.error("--levels must contain positive values")
    if args.mode == "closed-loop" and any(not value.is_integer() for value in levels):
        parser.error("closed-loop levels must be integer concurrency values")
    if args.measurement_sec < args.window_sec * args.stable_windows:
        parser.error("measurement-sec must contain at least stable-windows windows")
    if args.window_sec <= 0 or args.warmup_sec < 0:
        parser.error("window-sec must be positive and warmup-sec non-negative")

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    health_response = requests.get(
        f"{args.base_url.rstrip('/')}/v1/health",
        headers=headers,
        timeout=args.timeout_sec,
    )
    if health_response.status_code != 200:
        raise RuntimeError(
            f"Health failed: HTTP {health_response.status_code} {health_response.text[:300]}"
        )
    health = health_response.json()
    if health.get("model_id") != N3_MODEL_ID:
        raise RuntimeError(f"Unexpected model_id {health.get('model_id')!r}")

    client = HttpLoadClient(
        args.base_url,
        make_wav(args.audio_sec),
        args.text,
        args.timeout_sec,
        args.token,
    )
    results = []
    for level in levels:
        print(f"Running {args.mode} level={level:g} ...", flush=True)
        result = run_level(args, client, level)
        results.append(result)
        headline = result["headline"]
        print(
            f"  stable={result['stable']} success={headline['success_rate']:.3%} "
            f"throughput={headline['throughput_rps']:.2f} req/s "
            f"client_p95={headline['client']['p95_ms']}",
            flush=True,
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model_id": health["model_id"],
        "api_version": health["api_version"],
        "code_revision": code_revision(),
        "auth": "bearer-token-provided" if args.token else "disabled",
        "runtime_policy": {
            "queue_capacity": health["queue_capacity"],
            "request_capacity": health["request_capacity"],
            "limits": health["limits"],
        },
        "mode": args.mode,
        "workload": {
            "audio_sec": args.audio_sec,
            "input_mode": "audio_text" if args.text.strip() else "audio_only",
            "synthetic_audio": True,
            "warmup_sec": args.warmup_sec,
            "measurement_sec": args.measurement_sec,
            "window_sec": args.window_sec,
        },
        "results": results,
        "saturation": saturation_summary(results, args),
        "interpretation": (
            "Headline values use the first stable consecutive windows when found. "
            "Unstable levels cannot qualify as last-good. "
            "Queue, inference, exact server-total header, and client end-to-end are "
            "reported independently. A remote A4500 run is still required to add "
            "network/proxy behavior."
        ),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.output_json}")

    saturation = report["saturation"]
    print(
        "Last good:",
        saturation["last_good_level"],
        "First failure/saturation:",
        saturation["first_failure_or_saturation_level"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
