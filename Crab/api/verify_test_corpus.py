"""Send every generated compatibility-corpus case through a real HTTP API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests

from .benchmark_api import validate_response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--token", default=os.environ.get("CRAB_API_TOKEN"))
    args = parser.parse_args()
    manifest = json.loads(
        (args.corpus_dir / "manifest.json").read_text(encoding="utf-8")
    )
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    failures = []
    for case in manifest["cases"]:
        path = args.corpus_dir / case["file"]
        response = requests.post(
            f"{args.base_url.rstrip('/')}/v1/emotion/classify",
            files={"audio": (path.name, path.read_bytes())},
            data={"text": "我對這份工作很有信心"},
            headers=headers,
            timeout=args.timeout_sec,
        )
        if response.status_code != case["expected_status"]:
            failures.append(
                f"{path.name}: expected {case['expected_status']}, "
                f"got {response.status_code} {response.text[:200]}"
            )
            continue
        payload = response.json()
        if response.status_code == 200:
            try:
                validate_response("single", payload, "audio_text")
            except Exception as exc:
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
        elif payload.get("error_code") != case["error_code"]:
            failures.append(
                f"{path.name}: expected error {case['error_code']}, "
                f"got {payload.get('error_code')}"
            )
    print(f"cases={len(manifest['cases'])} failures={len(failures)}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
