"""
Latency benchmark for the Crab Bimodal Emotion API.

Usage:
    cd /home/brant/Project/SAILER_test
    python -m Crab.api.test_latency --wav path/to/test.wav

Runs:
  1. N consecutive single-file requests → P50/P95/Mean/Min/Max
  2. 1 batch request with M files → total & per-item latency
  3. Speedup comparison: sequential vs batch
"""

import argparse
import time
import statistics
import sys

import requests


def run_single_test(url: str, wav_path: str, text: str, n: int = 10) -> list[float]:
    """Send n sequential single-file requests and collect latencies."""
    print(f"\n{'='*60}")
    print(f"  [Single] {n} consecutive requests")
    print(f"{'='*60}")

    latencies = []
    for i in range(n):
        with open(wav_path, "rb") as f:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{url}/v1/emotion/classify",
                files={"audio": ("test.wav", f, "audio/wav")},
                data={"text": text},
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            print(f"  [{i+1:02d}] ERROR {resp.status_code}: {resp.text}")
            continue

        result = resp.json()
        latencies.append(elapsed_ms)
        server_ms = result.get("latency_ms", 0)
        print(
            f"  [{i+1:02d}] {result['primary_label']:16s} "
            f"conf={result['primary_confidence']:.3f}  "
            f"client={elapsed_ms:.1f}ms  "
            f"server={server_ms:.1f}ms"
        )

    if not latencies:
        print("  No successful requests!")
        return latencies

    sorted_lat = sorted(latencies)
    p95_idx = min(int(n * 0.95), len(sorted_lat) - 1)

    print(f"\n  ── Single Request Stats {'─'*36}")
    print(f"  Mean   : {statistics.mean(latencies):>8.1f} ms")
    print(f"  Median : {statistics.median(latencies):>8.1f} ms")
    print(f"  P95    : {sorted_lat[p95_idx]:>8.1f} ms")
    if n < 20:
        print(f"  ⚠️  n={n} is small; P95 estimate is unreliable (recommend --n 20+)")
    print(f"  Min    : {min(latencies):>8.1f} ms")
    print(f"  Max    : {max(latencies):>8.1f} ms")

    return latencies


def run_batch_test(
    url: str, wav_path: str, text: str, batch_size: int = 10
) -> tuple[float, float]:
    """Send 1 batch request containing batch_size copies of the same file."""
    print(f"\n{'='*60}")
    print(f"  [Batch] 1 request × {batch_size} files")
    print(f"{'='*60}")

    # Prepare multipart files and form data
    file_handles = []
    files_list = []
    texts_list = []
    for i in range(batch_size):
        fh = open(wav_path, "rb")
        file_handles.append(fh)
        files_list.append(("files", (f"audio_{i}.wav", fh, "audio/wav")))
        texts_list.append(("texts", text))

    t0 = time.perf_counter()
    resp = requests.post(
        f"{url}/v1/emotion/classify-batch",
        files=files_list,
        data=texts_list,
    )
    total_ms = (time.perf_counter() - t0) * 1000

    # Clean up file handles
    for fh in file_handles:
        fh.close()

    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text}")
        return total_ms, 0

    result = resp.json()

    print(f"  Batch size        : {result['batch_size']}")
    print(f"  Total latency     : {result['total_latency_ms']:.1f} ms  (client: {total_ms:.1f} ms)")
    print(f"  Avg per item      : {result['avg_latency_ms']:.1f} ms")
    print(f"\n  Results:")
    for r in result["results"]:
        print(
            f"    {r['filename']:20s}  "
            f"{r['primary_label']:16s}  "
            f"conf={r['primary_confidence']:.3f}"
        )

    return total_ms, result.get("avg_latency_ms", 0)


def print_comparison(
    single_latencies: list[float], batch_total_ms: float, batch_size: int
):
    """Print side-by-side comparison of sequential vs batch performance."""
    if not single_latencies or batch_total_ms <= 0:
        print("\n  ⚠️  Cannot compute comparison (missing data).")
        return

    single_mean = statistics.mean(single_latencies)
    sequential_total = single_mean * batch_size
    speedup = sequential_total / batch_total_ms

    print(f"\n{'='*60}")
    print(f"  [Comparison] Single × {batch_size} vs Batch × {batch_size}")
    print(f"{'='*60}")
    print(f"  Single × {batch_size} (sequential) : {sequential_total:>8.1f} ms")
    print(f"  Batch  × {batch_size} (parallel)   : {batch_total_ms:>8.1f} ms")
    print(f"  ─────────────────────────────────────")
    print(f"  🚀 Speedup             : {speedup:.2f}x")
    print(f"  📊 Throughput (single) : {1000/single_mean:.1f} req/s")
    batch_avg = batch_total_ms / batch_size
    print(f"  📊 Throughput (batch)  : {1000/batch_avg:.1f} req/s (amortised)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Crab Emotion API latency benchmark")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--wav", required=True, help="Path to a test WAV file")
    parser.add_argument(
        "--text",
        default="I think I am a strong fit for this position.",
        help="Transcript text to send",
    )
    parser.add_argument("--n", type=int, default=10, help="Single-request repetitions")
    parser.add_argument("--batch", type=int, default=10, help="Batch size")
    args = parser.parse_args()

    # Health check first
    try:
        health = requests.get(f"{args.url}/v1/health", timeout=5)
        if health.status_code == 200:
            info = health.json()
            print(f"\n✅ API is healthy: {info['model']} on {info['device']}")
            if info.get("gpu_name"):
                print(f"   GPU: {info['gpu_name']} ({info.get('vram_mb', '?')} MB)")
            print(f"   Classes: {info['classes']}")
        else:
            print(f"\n⚠️  Health check returned {health.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to {args.url}. Is the server running?")
        sys.exit(1)

    single_latencies = run_single_test(args.url, args.wav, args.text, n=args.n)
    batch_total, _ = run_batch_test(args.url, args.wav, args.text, batch_size=args.batch)
    print_comparison(single_latencies, batch_total, args.batch)


if __name__ == "__main__":
    main()
