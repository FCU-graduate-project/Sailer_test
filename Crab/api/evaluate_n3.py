"""Offline N3 loader, HTTP contract, and NNIME regression evaluation.

Run from the repository root:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      Crab/.venv/bin/python -m Crab.api.evaluate_n3

The validated N3 score was produced with test batch size 1. This evaluator
therefore uses ``predict_single`` for every row and fails if the full Test
split no longer reproduces the recorded macro-F1 within tolerance.

Use `--via-api` to exercise FastAPI multipart/Pydantic boundaries through
TestClient, or `--audio-only` to measure the unsupported-without-ablation
fallback separately from the validated bimodal score.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import torchaudio
from sklearn.metrics import accuracy_score, f1_score, recall_score

from .inference import CrabEmotionPredictor
from .model_profile import INTERVIEW_CLASSES, N3_MODEL_ID


CRAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = CRAB_ROOT / "experiments" / "nnime_v1_warmstart_lora"
DEFAULT_CSV = CRAB_ROOT / "data" / "nnime_crab_format.csv"
EXPECTED_TEST_MACRO_F1 = 0.5877


def _checkpoint_faithful_audio(path: Path) -> tuple[bytes, bool]:
    """Explicitly crop legacy evaluation rows to the training 12-second policy."""
    info = torchaudio.info(str(path))
    if info.sample_rate <= 0 or info.num_frames / info.sample_rate <= 12.0:
        return path.read_bytes(), False
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform[:, : round(12.0 * sample_rate)]
    output = io.BytesIO()
    torchaudio.save(output, waveform, sample_rate, format="wav")
    return output.getvalue(), True


def _label_index(row: dict[str, str]) -> int:
    values = [float(row[name]) for name in INTERVIEW_CLASSES]
    return int(np.argmax(values))


def _load_rows(csv_path: Path, split: str) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["Split_Set"] == split]
    if not rows:
        raise RuntimeError(f"No rows for Split_Set={split!r} in {csv_path}")
    missing = [row["FileName"] for row in rows if not Path(row["FileName"]).is_file()]
    if missing:
        raise RuntimeError(
            f"{len(missing)} audio files are missing; first missing path: {missing[0]}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate N3 and reproduce NNIME metrics")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--split", default="Test")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Smoke-test only; skips full regression threshold when set",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--via-api",
        action="store_true",
        help="Run every row through FastAPI TestClient multipart requests",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Send empty text and report an ablation without enforcing 0.5877",
    )
    parser.add_argument("--expected-macro-f1", type=float, default=EXPECTED_TEST_MACRO_F1)
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()

    rows = _load_rows(args.csv.resolve(), args.split)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            parser.error("--max-samples must be positive")
        rows = rows[: args.max_samples]

    from peft import PeftModel

    client_context = None
    client = None
    if args.via_api:
        os.environ["CRAB_MODEL_DIR"] = str(args.model_dir.resolve())
        os.environ["CRAB_MODEL_ID"] = N3_MODEL_ID
        os.environ["CRAB_DEVICE"] = args.device
        os.environ["CRAB_HF_OFFLINE"] = "1"
        logging.getLogger("httpx").setLevel(logging.WARNING)

        from fastapi.testclient import TestClient
        from . import app as app_module

        client_context = TestClient(app_module.app)
        client = client_context.__enter__()
        health_response = client.get("/v1/health")
        if health_response.status_code != 200:
            raise RuntimeError(
                f"Health check failed: {health_response.status_code} "
                f"{health_response.text}"
            )
        predictor = app_module.predictor
    else:
        predictor = CrabEmotionPredictor(
            model_dir=str(args.model_dir.resolve()),
            model_id=N3_MODEL_ID,
            device=args.device,
            offline=True,
        )

    if not isinstance(predictor.ssl_model, PeftModel):
        raise RuntimeError("Audio encoder is not a PeftModel")
    if not isinstance(predictor.text_model, PeftModel):
        raise RuntimeError("Text encoder is not a PeftModel")

    model_id = predictor.model_id
    ft_mode = predictor.ft_mode
    offline = predictor.offline

    y_true: list[int] = []
    y_pred: list[int] = []
    server_latency_ms = 0.0
    explicitly_cropped_rows = 0
    started = time.perf_counter()
    try:
        for index, row in enumerate(rows, start=1):
            audio_path = Path(row["FileName"])
            audio_bytes, was_cropped = _checkpoint_faithful_audio(audio_path)
            explicitly_cropped_rows += int(was_cropped)
            text = "" if args.audio_only else (row.get("Text", "") or "")
            if client is None:
                probabilities = predictor.predict_single(
                    audio_bytes,
                    text,
                )
                probability_tolerance = 1e-5
            else:
                response = client.post(
                    "/v1/emotion/classify",
                    files={
                        "audio": (
                            audio_path.name,
                            audio_bytes,
                            "audio/wav",
                        )
                    },
                    data={"text": text},
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"HTTP row {index} failed: {response.status_code} "
                        f"{response.text}"
                    )
                payload = response.json()
                if payload["model_id"] != N3_MODEL_ID:
                    raise RuntimeError(
                        f"HTTP row {index} model mismatch: {payload['model_id']}"
                    )
                expected_mode = "audio_only" if args.audio_only else "audio_text"
                if payload["input_mode"] != expected_mode:
                    raise RuntimeError(
                        f"HTTP row {index} input_mode mismatch: "
                        f"{payload['input_mode']}"
                    )
                probabilities = payload["probabilities"]
                server_latency_ms += float(payload["latency_ms"])
                probability_tolerance = 1e-3

            ordered = [probabilities[name] for name in INTERVIEW_CLASSES]
            if abs(sum(ordered) - 1.0) > probability_tolerance:
                raise RuntimeError(
                    f"Probability sum mismatch at row {index}: {sum(ordered)}"
                )
            y_true.append(_label_index(row))
            y_pred.append(int(np.argmax(ordered)))
            if index % 100 == 0 or index == len(rows):
                print(f"evaluated {index}/{len(rows)}", flush=True)
        elapsed = time.perf_counter() - started
    finally:
        if client_context is not None:
            client_context.__exit__(None, None, None)
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(INTERVIEW_CLASSES))),
        average=None,
        zero_division=0,
    )
    result = {
        "model_id": model_id,
        "ft_mode": ft_mode,
        "offline": offline,
        "evaluation_path": "http_testclient" if args.via_api else "direct_core",
        "input_mode": "audio_only" if args.audio_only else "audio_text",
        "split": args.split,
        "samples": len(rows),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "uar": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "per_class_f1": {
            name: float(per_class_f1[index])
            for index, name in enumerate(INTERVIEW_CLASSES)
        },
        "elapsed_sec": elapsed,
        "items_per_sec": len(rows) / elapsed,
        "server_inference_sec": (
            server_latency_ms / 1000.0 if args.via_api else None
        ),
        "explicitly_cropped_to_12s": explicitly_cropped_rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.max_samples is None and args.split == "Test" and not args.audio_only:
        delta = abs(result["macro_f1"] - args.expected_macro_f1)
        if delta > args.tolerance:
            raise RuntimeError(
                f"N3 regression failed: macro-F1={result['macro_f1']:.6f}, "
                f"expected={args.expected_macro_f1:.6f} ± {args.tolerance:.6f}"
            )
        print(
            f"N3 regression PASS: macro-F1={result['macro_f1']:.6f} "
            f"(expected {args.expected_macro_f1:.6f} ± {args.tolerance:.6f})"
        )
    elif args.audio_only:
        print("N3 audio-only ablation complete; bimodal threshold intentionally skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
