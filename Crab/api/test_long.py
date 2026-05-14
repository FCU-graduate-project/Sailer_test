"""
Test the /v1/emotion/classify-long endpoint.

Since MSP-Podcast files are all < 12s, this script concatenates
multiple clips into a ~60s synthetic interview audio, then sends
it to the long-audio endpoint to demonstrate sliding-window
segmentation and timeline tracking.

Usage:
    cd /home/brant/Project/SAILER_test
    # Start the API first:
    #   export CRAB_MODEL_DIR=Crab/experiments/interview_scheme2
    #   Crab/.venv/bin/python -m uvicorn Crab.api.app:app --host 0.0.0.0 --port 8001
    # Then run:
    Crab/.venv/bin/python -m Crab.api.test_long
"""

import argparse
import csv
import io
import os
import random
import sys
import time

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def concatenate_wav_files(wav_paths: list[str], target_duration_sec: float = 60.0) -> bytes:
    """Concatenate WAV files into a single long WAV, up to target_duration_sec."""
    import wave
    import struct

    # Read all files
    all_frames = b""
    total_sec = 0.0
    params_set = False
    params = None

    for path in wav_paths:
        try:
            with wave.open(path, "rb") as wf:
                if not params_set:
                    params = wf.getparams()
                    params_set = True
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                dur = n_frames / wf.getframerate()
                all_frames += frames
                total_sec += dur
                if total_sec >= target_duration_sec:
                    break
        except Exception as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")
            continue

    if not params_set or total_sec < 1.0:
        raise ValueError(f"Could not build long audio (only {total_sec:.1f}s)")

    # Write concatenated WAV to bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(all_frames)

    print(f"  Built synthetic audio: {total_sec:.1f}s from {len(wav_paths)} clips")
    return buf.getvalue()


def build_long_audio_from_csv(csv_path: str, audio_dir: str, target_sec: float = 60.0) -> bytes:
    """Load MSP clips from CSV and concatenate them into a long audio."""
    wav_paths = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = os.path.join(audio_dir, row["FileName"])
            if os.path.exists(path):
                wav_paths.append(path)

    if len(wav_paths) == 0:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    # Shuffle for variety
    random.seed(42)
    random.shuffle(wav_paths)

    print(f"  Found {len(wav_paths)} audio files. Concatenating to ~{target_sec}s...")
    return concatenate_wav_files(wav_paths, target_duration_sec=target_sec)


# ─────────────────────────────────────────────────────────────────────────────
# Test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_with_short_audio(url: str, audio_dir: str, csv_path: str):
    """Test classify-long with a single short MSP clip (should return 1 window)."""
    print(f"\n{'='*60}")
    print(f"  Test 1: Short audio (single MSP clip, expect 1 window)")
    print(f"{'='*60}")

    # Find a single file
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = os.path.join(audio_dir, row["FileName"])
            if os.path.exists(path):
                break

    with open(path, "rb") as f:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{url}/v1/emotion/classify-long",
            files={"audio": ("short_clip.wav", f, "audio/wav")},
        )
        client_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        print(f"  ❌ Error: {resp.status_code} - {resp.text}")
        return

    data = resp.json()
    print(f"  ✅ Status: 200")
    print(f"  Audio duration: {data['audio_duration_sec']}s")
    print(f"  Windows: {data['total_windows']}")
    print(f"  Final: {data['final_label']} ({data['final_confidence']:.3f})")
    print(f"  Server latency: {data['latency_ms']:.1f}ms | Client: {client_ms:.1f}ms")
    print(f"  Timeline:")
    for t in data["timeline"]:
        print(f"    {t['window']:>14s}  {t['label']:16s}  conf={t['confidence']:.3f}")


def test_with_long_audio(url: str, audio_bytes: bytes, duration_label: str):
    """Test classify-long with a concatenated long audio."""
    print(f"\n{'='*60}")
    print(f"  Test 2: Long audio (~{duration_label})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/v1/emotion/classify-long",
        files={"audio": ("long_interview.wav", audio_bytes, "audio/wav")},
        data={"window_sec": "12.0", "stride_sec": "6.0"},
    )
    client_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        print(f"  ❌ Error: {resp.status_code} - {resp.text}")
        return

    data = resp.json()
    print(f"  ✅ Status: 200")
    print(f"  Audio duration: {data['audio_duration_sec']}s")
    print(f"  Windows: {data['total_windows']}")
    print(f"  Final: {data['final_label']} ({data['final_confidence']:.3f})")
    print(f"  Avg Probabilities:")
    for cls, prob in data["avg_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:16s}  {prob:.3f}  {bar}")

    print(f"\n  Server latency: {data['latency_ms']:.1f}ms | Client: {client_ms:.1f}ms")
    print(f"  Throughput: {data['audio_duration_sec'] / (data['latency_ms']/1000):.0f}x realtime")

    print(f"\n  📊 Emotion Timeline:")
    print(f"  {'Window':>14s}  {'Label':16s}  {'Conf':>6s}  {'Excited':>8s}  {'Unconf':>8s}  {'Neutral':>8s}")
    print(f"  {'─'*14}  {'─'*16}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
    for t in data["timeline"]:
        p = t["probabilities"]
        print(
            f"  {t['window']:>14s}  {t['label']:16s}  {t['confidence']:.3f}"
            f"   {p['Excited']:.3f}    {p['Unconfident']:.3f}    {p['Neutral_3Class']:.3f}"
        )


def test_audio_only_vs_with_text(url: str, audio_bytes: bytes):
    """Compare long-audio inference with and without text."""
    print(f"\n{'='*60}")
    print(f"  Test 3: Audio-only vs Audio+Text comparison")
    print(f"{'='*60}")

    # Audio only
    t0 = time.perf_counter()
    resp1 = requests.post(
        f"{url}/v1/emotion/classify-long",
        files={"audio": ("test.wav", audio_bytes, "audio/wav")},
    )
    ms1 = (time.perf_counter() - t0) * 1000

    # Audio + text
    t0 = time.perf_counter()
    resp2 = requests.post(
        f"{url}/v1/emotion/classify-long",
        files={"audio": ("test.wav", audio_bytes, "audio/wav")},
        data={"text": "I am very excited about this opportunity and believe I am a strong fit for this position."},
    )
    ms2 = (time.perf_counter() - t0) * 1000

    if resp1.status_code != 200 or resp2.status_code != 200:
        print(f"  ❌ Error occurred")
        return

    d1 = resp1.json()
    d2 = resp2.json()

    print(f"  {'':20s}  {'Audio Only':>14s}  {'Audio + Text':>14s}")
    print(f"  {'─'*20}  {'─'*14}  {'─'*14}")
    print(f"  {'Final Label':20s}  {d1['final_label']:>14s}  {d2['final_label']:>14s}")
    print(f"  {'Confidence':20s}  {d1['final_confidence']:>14.3f}  {d2['final_confidence']:>14.3f}")
    for cls in ["Excited", "Unconfident", "Neutral_3Class"]:
        print(f"  {cls:20s}  {d1['avg_probabilities'][cls]:>14.3f}  {d2['avg_probabilities'][cls]:>14.3f}")
    print(f"  {'Latency (ms)':20s}  {ms1:>14.1f}  {ms2:>14.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test classify-long endpoint")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--duration", type=float, default=60.0, help="Target duration in seconds")
    parser.add_argument(
        "--csv", default="/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme2.csv"
    )
    parser.add_argument(
        "--audio-dir", default="/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios/"
    )
    args = parser.parse_args()

    # Health check
    try:
        health = requests.get(f"{args.url}/v1/health", timeout=5)
        if health.status_code == 200:
            info = health.json()
            print(f"\n✅ API is healthy: {info['model']} on {info['device']}")
        else:
            print(f"\n⚠️  Health check returned {health.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to {args.url}. Is the server running?")
        sys.exit(1)

    # Test 1: Short audio
    test_with_short_audio(args.url, args.audio_dir, args.csv)

    # Build long audio
    print(f"\n🔧 Building synthetic {args.duration}s audio from MSP clips...")
    long_audio = build_long_audio_from_csv(args.csv, args.audio_dir, target_sec=args.duration)

    # Test 2: Long audio
    test_with_long_audio(args.url, long_audio, f"{args.duration}s")

    # Test 3: Audio-only vs with text
    test_audio_only_vs_with_text(args.url, long_audio)

    print(f"\n{'='*60}")
    print(f"  All tests completed! 🎉")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
