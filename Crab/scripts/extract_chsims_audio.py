"""
P1: Extract audio from CH-SIMS v2(s) mp4 clips → 16 kHz mono wav.

Source:  datasets/chsims_v2s/ch-simsv2s/Raw/<folder>/<clip>.mp4
Target:  datasets/chsims_v2s/ch-simsv2s/Audio/<folder>/<clip>.wav

Properties match Crab pipeline expectations:
  - sample rate 16000
  - 1 channel (mono)
  - PCM 16-bit
  - duration: preserved (no truncation; Crab trims to 12s at training time)

Idempotent: skips wav that already exist with non-zero size.
Parallel: uses multiprocessing pool (default 8 workers).
"""
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

ROOT = Path("/home/brant/Project/SAILER_test/datasets/chsims_v2s/ch-simsv2s")
RAW  = ROOT / "Raw"
OUT  = ROOT / "Audio"

FFMPEG = "/home/brant/bin/ffmpeg"


def extract_one(mp4_path_str):
    """Run ffmpeg to extract 16kHz mono wav from one mp4."""
    mp4 = Path(mp4_path_str)
    rel = mp4.relative_to(RAW)            # e.g. aqgy3_0001/00000.mp4
    wav = OUT / rel.with_suffix(".wav")
    wav.parent.mkdir(parents=True, exist_ok=True)

    if wav.exists() and wav.stat().st_size > 1024:
        return ("skip", str(rel))

    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", str(mp4),
        "-vn",                 # drop video
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz
        "-acodec", "pcm_s16le",
        str(wav),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return ("fail", f"{rel}: {r.stderr.strip()[:200]}")
        if not wav.exists() or wav.stat().st_size == 0:
            return ("fail", f"{rel}: empty output")
        return ("ok", str(rel))
    except subprocess.TimeoutExpired:
        return ("fail", f"{rel}: timeout")
    except Exception as e:
        return ("fail", f"{rel}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit",   type=int, default=None, help="for testing")
    args = ap.parse_args()

    mp4s = sorted(RAW.glob("*/*.mp4"))
    if args.limit:
        mp4s = mp4s[: args.limit]
    print(f"Found {len(mp4s)} mp4 files under {RAW}")
    print(f"Output → {OUT}")
    print(f"Workers: {args.workers}")

    OUT.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "skip": 0, "fail": 0}
    failures = []

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, str(m)): m for m in mp4s}
        done = 0
        last_print = t0
        for fut in as_completed(futures):
            status, info = fut.result()
            stats[status] += 1
            if status == "fail":
                failures.append(info)
            done += 1
            now = time.time()
            if now - last_print > 2 or done == len(mp4s):
                elapsed = now - t0
                rate = done / max(elapsed, 1e-3)
                eta  = (len(mp4s) - done) / max(rate, 1e-6)
                print(f"  [{done:5d}/{len(mp4s)}] ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}  "
                      f"{rate:.1f} clip/s  eta={eta:.0f}s", flush=True)
                last_print = now

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s. ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}")
    if failures:
        print(f"\n⚠ {len(failures)} failures:")
        for f in failures[:20]:
            print(f"  - {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures)-20} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
