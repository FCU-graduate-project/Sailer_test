"""Generate a deterministic, non-sensitive API compatibility corpus."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import wave


PROFILES = [
    ("mono_8k_1s", 8000, 1, 1.0),
    ("mono_16k_3s", 16000, 1, 3.0),
    ("stereo_16k_6s", 16000, 2, 6.0),
    ("mono_44100_3s", 44100, 1, 3.0),
    ("stereo_48k_12s", 48000, 2, 12.0),
]


def wav_bytes(sample_rate: int, channels: int, duration_sec: float) -> bytes:
    path_samples = round(sample_rate * duration_sec)
    mono = array(
        "h",
        (
            round(1200 * math.sin(2 * math.pi * 220 * index / sample_rate))
            for index in range(path_samples)
        ),
    )
    samples = array("h")
    for value in mono:
        samples.extend([value] * channels)
    from io import BytesIO

    output = BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return output.getvalue()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-compressed", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    wav_paths = []
    for name, sample_rate, channels, duration_sec in PROFILES:
        path = args.output_dir / f"{name}.wav"
        path.write_bytes(wav_bytes(sample_rate, channels, duration_sec))
        wav_paths.append(path)
        manifest.append(
            {
                "file": path.name,
                "format": "wav",
                "sample_rate": sample_rate,
                "channels": channels,
                "duration_sec": duration_sec,
                "expected_status": 200,
                "sha256": digest(path),
            }
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        source = wav_paths[1]
        for extension, codec in (("flac", "flac"), ("mp3", "libmp3lame")):
            target = args.output_dir / f"mono_16k_3s.{extension}"
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-codec:a",
                    codec,
                    str(target),
                ],
                check=False,
            )
            if completed.returncode == 0:
                manifest.append(
                    {
                        "file": target.name,
                        "format": extension,
                        "sample_rate": 16000,
                        "channels": 1,
                        "duration_sec": 3.0,
                        "expected_status": 200,
                        "sha256": digest(target),
                    }
                )
            elif args.require_compressed:
                raise SystemExit(f"Could not generate {extension}")
    elif args.require_compressed:
        raise SystemExit("ffmpeg is required for compressed corpus cases")

    invalid = args.output_dir / "invalid_audio.bin"
    invalid.write_bytes(b"not an audio container")
    empty = args.output_dir / "empty.wav"
    empty.write_bytes(b"")
    manifest.extend(
        [
            {
                "file": invalid.name,
                "format": "invalid",
                "expected_status": 400,
                "error_code": "INVALID_AUDIO",
                "sha256": digest(invalid),
            },
            {
                "file": empty.name,
                "format": "empty",
                "expected_status": 400,
                "error_code": "EMPTY_AUDIO",
                "sha256": digest(empty),
            },
        ]
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "synthetic": True,
                "contains_personal_data": False,
                "cases": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
