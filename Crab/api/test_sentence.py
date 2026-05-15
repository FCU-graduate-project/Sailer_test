"""
Sentence-level emotion analysis using Whisper + Crab API.

This script is STANDALONE — it does NOT modify any existing API code.
It works as a client that:
  1. Transcribes audio (local faster-whisper OR OpenRouter API)
  2. Groups words / segments into sentences
  3. Merges short sentences (< min) and splits long ones (> max)
  4. Slices the audio into sentence-aligned segments
  5. Sends each segment + its transcript to the existing Crab API
  6. Displays a timeline with actual text + emotion labels

Usage:
    cd /home/brant/Project/SAILER_test
    # Make sure the Crab API is running on port 8001

    # Local Whisper (default)
    Crab/.venv/bin/python -m Crab.api.test_sentence

    # OpenRouter Whisper (no local GPU needed)
    export OPENROUTER_API_KEY=sk-or-v1-...
    Crab/.venv/bin/python -m Crab.api.test_sentence --backend openrouter --wav /path/to/clip.wav
"""

import argparse
import base64
import io
import os
import sys
import time
import wave
import random
import csv
import struct

import requests
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Whisper transcription with word-level timestamps
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_with_timestamps(
    wav_path: str,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
) -> list[dict]:
    """Transcribe audio and return word-level timestamps.

    Returns a list of dicts:
        [{"word": "我", "start": 0.0, "end": 0.3}, ...]
    """
    from faster_whisper import WhisperModel

    print(f"  Loading Whisper model: {model_size} on {device}...")
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Whisper model loaded in {load_ms:.0f}ms")

    print(f"  Transcribing: {os.path.basename(wav_path)}")
    t0 = time.perf_counter()
    segments, info = model.transcribe(
        wav_path,
        word_timestamps=True,
        language=None,  # auto-detect
    )

    words = []
    full_text_parts = []
    for segment in segments:
        full_text_parts.append(segment.text)
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

    transcribe_ms = (time.perf_counter() - t0) * 1000
    full_text = "".join(full_text_parts).strip()
    detected_lang = info.language
    lang_prob = info.language_probability

    print(f"  Transcription done in {transcribe_ms:.0f}ms")
    print(f"  Detected language: {detected_lang} ({lang_prob:.1%})")
    print(f"  Full transcript: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")
    print(f"  Total words: {len(words)}")

    # Free GPU memory
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return words


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter transcription (no local GPU)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_openrouter(
    wav_path: str,
    model: str = "openai/whisper-large-v3",
    language: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Transcribe via OpenRouter's audio transcription endpoint.

    Returns a dict with exactly one of:
        {"words": [{"word", "start", "end"}, ...], "sentences": None}
            when word-level timestamps are available
        {"words": None, "sentences": [{"text", "start", "end"}, ...]}
            when only segment-level timestamps are available

    The caller decides which path to feed into the sentence builder.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key not found. "
            "Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY."
        )

    print(f"  Reading audio: {os.path.basename(wav_path)}")
    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    fmt = os.path.splitext(wav_path)[1].lstrip(".").lower() or "wav"
    if fmt == "wave":
        fmt = "wav"

    payload = {
        "model": model,
        "input_audio": {"data": audio_b64, "format": fmt},
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }
    if language:
        payload["language"] = language

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"  Calling OpenRouter ({model})...")
    t0 = time.perf_counter()
    resp = requests.post(
        "https://openrouter.ai/api/v1/audio/transcriptions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"OpenRouter returned non-JSON (HTTP {resp.status_code}): {resp.text[:300]}") from e

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"OpenRouter error: {data['error']}")

    detected_lang = data.get("language", "?")
    print(f"  OpenRouter responded in {elapsed_ms:.0f}ms (lang={detected_lang})")

    words_data = data.get("words") or []
    if words_data:
        words = [
            {"word": str(w["word"]).strip(), "start": float(w["start"]), "end": float(w["end"])}
            for w in words_data
        ]
        print(f"  Got {len(words)} word-level timestamps")
        return {"words": words, "sentences": None}

    segments = data.get("segments") or []
    if segments:
        sentences = [
            {"text": str(s["text"]).strip(), "start": float(s["start"]), "end": float(s["end"])}
            for s in segments
        ]
        print(f"  No word-level — falling back to {len(sentences)} segment(s)")
        return {"words": None, "sentences": sentences}

    raise RuntimeError(
        f"OpenRouter returned no timestamps. Response keys: {list(data.keys())}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sentence segmentation from word timestamps
# ─────────────────────────────────────────────────────────────────────────────

# Sentence-ending punctuation (covers Chinese & English)
SENTENCE_ENDERS = {".", "。", "?", "？", "!", "！", "…"}

# Soft break punctuation (clause boundaries) — used as fallback split points
# when a sentence exceeds max_duration_sec and we want to avoid mid-clause cuts.
SOFT_BREAK_CHARS = {",", ";", ":", "，", "；", "：", "、"}

def _merge_two(a: dict, b: dict) -> dict:
    """Merge two adjacent sentence entries, preserving words if either side has them."""
    out = {
        "text": a["text"] + " " + b["text"],
        "start": a["start"],
        "end": b["end"],
    }
    combined_words = (a.get("words") or []) + (b.get("words") or [])
    if combined_words:
        out["words"] = combined_words
    return out


def _find_punct_break(
    words: list[dict],
    start_idx: int,
    end_idx: int,
    punct_set: set,
    min_chunk_dur: float = 0.0,
) -> int | None:
    """Find the latest word in [start_idx, end_idx] that ends with a char in punct_set
    and produces a chunk of at least min_chunk_dur. Pass min_chunk_dur=0 to honor any
    punctuation regardless of resulting chunk size."""
    chunk_start_time = words[start_idx]["start"]
    for i in range(end_idx, start_idx - 1, -1):
        word_text = words[i]["word"]
        if any(word_text.endswith(p) for p in punct_set):
            chunk_dur = words[i]["end"] - chunk_start_time
            if chunk_dur >= min_chunk_dur or i == start_idx:
                return i
    return None


def _split_long_words(words: list[dict], max_duration_sec: float) -> list[dict]:
    """Split a long sentence's words into chunks ≤ max, preferring punctuation breaks.

    Greedy: include as many words as fit under max_duration_sec, then back off to
    the latest internal break point. Priority: sentence-ending > soft-break > word.
    """
    chunks: list[dict] = []
    n = len(words)
    start_idx = 0
    # Soft breaks (commas) get a 25%-of-max floor to avoid pathological tiny chunks
    # from listy commas; hard breaks (periods) are always honored.
    soft_min_chunk = max_duration_sec * 0.25

    while start_idx < n:
        chunk_start_time = words[start_idx]["start"]

        # Find the furthest word we can include without exceeding max
        end_idx = start_idx
        for i in range(start_idx, n):
            if words[i]["end"] - chunk_start_time > max_duration_sec:
                break
            end_idx = i

        # If we can take all remaining words, emit final chunk and stop
        if end_idx == n - 1:
            bucket = words[start_idx:end_idx + 1]
            chunks.append({
                "text": " ".join(w["word"] for w in bucket),
                "start": bucket[0]["start"],
                "end": bucket[-1]["end"],
            })
            break

        # Pick the best split point inside [start_idx, end_idx]
        # Priority: hard break (always honored) > soft break (≥25% chunk)
        #         > balanced equal-time word-aligned split (fallback when no punctuation)
        split_at = _find_punct_break(words, start_idx, end_idx, SENTENCE_ENDERS)
        if split_at is None:
            split_at = _find_punct_break(words, start_idx, end_idx, SOFT_BREAK_CHARS, soft_min_chunk)
        if split_at is None:
            # Equal-time balanced fallback: aim the cut at the word whose end time is
            # closest to (remaining_duration / n_remaining_chunks). Produces ~equal
            # halves rather than degenerate (11.9s + 1.1s) splits.
            remaining_dur = words[n - 1]["end"] - chunk_start_time
            n_remaining = int(np.ceil(remaining_dur / max_duration_sec))
            target_end_time = chunk_start_time + remaining_dur / n_remaining

            split_at = end_idx
            best_diff = abs(words[end_idx]["end"] - target_end_time)
            for i in range(start_idx, end_idx + 1):
                diff = abs(words[i]["end"] - target_end_time)
                if diff < best_diff:
                    best_diff = diff
                    split_at = i

        bucket = words[start_idx:split_at + 1]
        chunks.append({
            "text": " ".join(w["word"] for w in bucket),
            "start": bucket[0]["start"],
            "end": bucket[-1]["end"],
        })
        start_idx = split_at + 1

    return chunks


def merge_and_split_sentences(
    raw_sentences: list[dict],
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 12.0,
) -> list[dict]:
    """Merge sentences shorter than min and split sentences longer than max.

    Each entry may optionally carry a `words` field (list of {word, start, end}).
    When present, long sentences are split *by word time* so each chunk's text
    actually matches its audio. Without words (e.g. OpenRouter segment fallback),
    falls back to equal-time chunking with the full sentence text repeated.
    """
    if not raw_sentences:
        return []

    # Merge short sentences with neighbors
    merged = [raw_sentences[0]]
    for s in raw_sentences[1:]:
        prev = merged[-1]
        if (prev["end"] - prev["start"]) < min_duration_sec:
            merged[-1] = _merge_two(prev, s)
        else:
            merged.append(s)

    # If the last one is still too short, fold it into its predecessor
    if len(merged) > 1:
        last = merged[-1]
        if (last["end"] - last["start"]) < min_duration_sec:
            merged[-2] = _merge_two(merged[-2], last)
            merged.pop()

    # Split sentences longer than max
    final = []
    for s in merged:
        duration = s["end"] - s["start"]
        if duration <= max_duration_sec:
            final.append(s)
            continue

        words = s.get("words")
        if words:
            # Word-level: greedy split preferring hard/soft punctuation breaks
            final.extend(_split_long_words(words, max_duration_sec))
        else:
            # No word-level info — equal-time chunks with duplicated full text
            n_chunks = int(np.ceil(duration / max_duration_sec))
            chunk_dur = duration / n_chunks
            for i in range(n_chunks):
                final.append({
                    "text": s["text"] if i == 0 else f"[cont.] {s['text']}",
                    "start": s["start"] + i * chunk_dur,
                    "end": min(s["start"] + (i + 1) * chunk_dur, s["end"]),
                })

    return final


def words_to_sentences(
    words: list[dict],
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 12.0,
) -> list[dict]:
    """Group words into sentences based on punctuation, then merge/split.

    Returns:
        [{"text": "我認為我很適合...", "start": 0.0, "end": 3.5}, ...]
    """
    if not words:
        return []

    raw_sentences = []
    current_words = []

    for w in words:
        current_words.append(w)
        word_text = w["word"]
        if any(word_text.endswith(p) for p in SENTENCE_ENDERS):
            raw_sentences.append({
                "text": " ".join(cw["word"] for cw in current_words),
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "words": current_words[:],
            })
            current_words = []

    if current_words:
        raw_sentences.append({
            "text": " ".join(cw["word"] for cw in current_words),
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "words": current_words[:],
        })

    return merge_and_split_sentences(raw_sentences, min_duration_sec, max_duration_sec)


# ─────────────────────────────────────────────────────────────────────────────
# Audio slicing
# ─────────────────────────────────────────────────────────────────────────────

def slice_audio(wav_path: str, start_sec: float, end_sec: float) -> bytes:
    """Extract a segment from a WAV file and return as bytes."""
    import torchaudio

    waveform, sr = torchaudio.load(wav_path)
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = waveform[:, start_sample:end_sample]

    buf = io.BytesIO()
    torchaudio.save(buf, segment, sr, format="wav")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Send to Crab API
# ─────────────────────────────────────────────────────────────────────────────

def classify_segment(
    api_url: str,
    audio_bytes: bytes,
    text: str,
    filename: str = "segment.wav",
) -> dict:
    """Send a single segment to the Crab classify endpoint."""
    resp = requests.post(
        f"{api_url}/v1/emotion/classify",
        files={"audio": (filename, audio_bytes, "audio/wav")},
        data={"text": text},
    )
    if resp.status_code != 200:
        return {"error": resp.text}
    return resp.json()


def classify_segments_batch(
    api_url: str,
    segments: list[dict],
    wav_path: str,
    batch_size: int = 16,
) -> list[dict]:
    """Classify all segments via the batch API endpoint."""
    results = []

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]

        files_list = []
        texts_list = []
        for i, seg in enumerate(batch):
            audio_bytes = slice_audio(wav_path, seg["start"], seg["end"])
            files_list.append(("files", (f"seg_{batch_start+i}.wav", audio_bytes, "audio/wav")))
            texts_list.append(("texts", seg["text"]))

        resp = requests.post(
            f"{api_url}/v1/emotion/classify-batch",
            files=files_list,
            data=texts_list,
        )

        if resp.status_code != 200:
            print(f"  ❌ Batch error: {resp.status_code} - {resp.text}")
            results.extend([{"error": resp.text}] * len(batch))
        else:
            data = resp.json()
            results.extend(data["results"])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Build synthetic long audio from MSP clips (for testing)
# ─────────────────────────────────────────────────────────────────────────────

def build_long_wav_from_msp(
    csv_path: str,
    audio_dir: str,
    output_path: str,
    target_sec: float = 60.0,
) -> str:
    """Concatenate MSP clips into a single long WAV file on disk."""
    import torchaudio

    wav_paths = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = os.path.join(audio_dir, row["FileName"])
            if os.path.exists(path):
                wav_paths.append(path)

    random.seed(42)
    random.shuffle(wav_paths)

    all_waveforms = []
    total_sec = 0.0
    target_sr = 16000

    for path in wav_paths:
        try:
            wf, sr = torchaudio.load(path)
            if sr != target_sr:
                wf = torchaudio.functional.resample(wf, sr, target_sr)
            wf = wf.mean(dim=0, keepdim=True)  # mono
            all_waveforms.append(wf)
            total_sec += wf.shape[1] / target_sr
            if total_sec >= target_sec:
                break
        except Exception:
            continue

    import torch
    combined = torch.cat(all_waveforms, dim=1)
    torchaudio.save(output_path, combined, target_sr)
    print(f"  Built synthetic WAV: {total_sec:.1f}s → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_ICONS = {
    "Excited": "🔥",
    "Unconfident": "😰",
    "Neutral_3Class": "😐",
}


def display_results(sentences: list[dict], api_results: list[dict]):
    """Pretty-print the sentence-level emotion timeline."""
    print(f"\n{'='*80}")
    print(f"  📊 Sentence-Level Emotion Timeline")
    print(f"{'='*80}")

    # Compute overall average
    avg_probs = {"Excited": 0.0, "Unconfident": 0.0, "Neutral_3Class": 0.0}
    valid_count = 0

    for i, (seg, res) in enumerate(zip(sentences, api_results)):
        if "error" in res:
            print(f"\n  [{i+1}] ❌ Error: {res['error']}")
            continue

        duration = seg["end"] - seg["start"]
        label = res.get("primary_label", res.get("label", "?"))
        conf = res.get("primary_confidence", res.get("confidence", 0))
        icon = EMOTION_ICONS.get(label, "❓")

        # Get probabilities
        probs = res.get("probabilities", {})

        print(f"\n  [{i+1}] {seg['start']:.1f}s ~ {seg['end']:.1f}s ({duration:.1f}s)")
        print(f"      📝 \"{seg['text']}\"")
        print(f"      {icon} {label}  (confidence: {conf:.3f})")
        if probs:
            print(f"      Excited={probs.get('Excited', 0):.3f}  "
                  f"Unconfident={probs.get('Unconfident', 0):.3f}  "
                  f"Neutral={probs.get('Neutral_3Class', 0):.3f}")

        for cls in avg_probs:
            avg_probs[cls] += probs.get(cls, 0)
        valid_count += 1

    if valid_count > 0:
        avg_probs = {k: v / valid_count for k, v in avg_probs.items()}
        final_label = max(avg_probs, key=avg_probs.get)
        icon = EMOTION_ICONS.get(final_label, "❓")

        print(f"\n{'─'*80}")
        print(f"  🎯 Overall Result (probability-averaged over {valid_count} sentences):")
        print(f"     {icon} {final_label} ({avg_probs[final_label]:.3f})")
        for cls in ["Excited", "Unconfident", "Neutral_3Class"]:
            bar = "█" * int(avg_probs[cls] * 40)
            print(f"     {cls:16s}  {avg_probs[cls]:.3f}  {bar}")
        print(f"{'='*80}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sentence-level emotion analysis (Whisper + Crab)")
    parser.add_argument("--url", default="http://localhost:8001", help="Crab API base URL")
    parser.add_argument("--wav", default=None, help="Path to an audio file (WAV/MP3/FLAC)")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration for synthetic audio")
    parser.add_argument(
        "--backend", default="local", choices=["local", "openrouter"],
        help="Transcription backend: local faster-whisper or OpenRouter API",
    )
    parser.add_argument("--whisper-model", default="large-v3", help="Local Whisper model size")
    parser.add_argument("--device", default="cuda", help="Device for local Whisper")
    parser.add_argument(
        "--openrouter-model", default="openai/whisper-large-v3",
        help="OpenRouter model id (only used when --backend openrouter)",
    )
    parser.add_argument(
        "--language", default=None,
        help="Force language code (e.g. 'en', 'zh'). Default: auto-detect",
    )
    parser.add_argument("--min-sentence", type=float, default=3.0, help="Min sentence duration (sec)")
    parser.add_argument("--max-sentence", type=float, default=12.0, help="Max sentence duration (sec)")
    parser.add_argument(
        "--csv", default="/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme2.csv"
    )
    parser.add_argument(
        "--audio-dir", default="/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios/"
    )
    args = parser.parse_args()

    # ── Health check ──
    try:
        health = requests.get(f"{args.url}/v1/health", timeout=5)
        if health.status_code == 200:
            info = health.json()
            print(f"\n✅ Crab API is healthy: {info['model']} on {info['device']}")
        else:
            print(f"\n⚠️  Health check returned {health.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to {args.url}. Is the Crab API running?")
        sys.exit(1)

    # ── Get or build the audio file ──
    if args.wav:
        wav_path = args.wav
        print(f"\n📁 Using provided audio: {wav_path}")
    else:
        print(f"\n🔧 Building synthetic {args.duration}s audio from MSP clips...")
        wav_path = "/tmp/sentence_test_long.wav"
        build_long_wav_from_msp(args.csv, args.audio_dir, wav_path, target_sec=args.duration)

    # ── Step 1: Transcription ──
    print(f"\n{'='*80}")
    print(f"  Step 1: Transcription (backend={args.backend})")
    print(f"{'='*80}")

    words = None
    raw_segment_sentences = None

    if args.backend == "local":
        words = transcribe_with_timestamps(
            wav_path,
            model_size=args.whisper_model,
            device=args.device,
        )
    else:  # openrouter
        result = transcribe_openrouter(
            wav_path,
            model=args.openrouter_model,
            language=args.language,
        )
        words = result["words"]
        raw_segment_sentences = result["sentences"]

    if words is not None:
        if not words:
            print("  ❌ No words detected in audio!")
            sys.exit(1)
        print(f"\n  Sample words:")
        for w in words[:8]:
            print(f"    {w['start']:6.2f}s ~ {w['end']:6.2f}s  \"{w['word']}\"")
        if len(words) > 8:
            print(f"    ... ({len(words) - 8} more words)")
    elif not raw_segment_sentences:
        print("  ❌ No segments returned!")
        sys.exit(1)

    # ── Step 2: Sentence segmentation ──
    print(f"\n{'='*80}")
    print(f"  Step 2: Sentence Segmentation (min={args.min_sentence}s, max={args.max_sentence}s)")
    print(f"{'='*80}")

    if words is not None:
        sentences = words_to_sentences(
            words,
            min_duration_sec=args.min_sentence,
            max_duration_sec=args.max_sentence,
        )
    else:
        print(f"  Using {len(raw_segment_sentences)} segment-level entries from OpenRouter")
        sentences = merge_and_split_sentences(
            raw_segment_sentences,
            min_duration_sec=args.min_sentence,
            max_duration_sec=args.max_sentence,
        )

    print(f"  Total sentences: {len(sentences)}")
    for i, s in enumerate(sentences):
        dur = s["end"] - s["start"]
        text_preview = s["text"][:50] + ("..." if len(s["text"]) > 50 else "")
        print(f"    [{i+1}] {s['start']:6.1f}s ~ {s['end']:6.1f}s ({dur:4.1f}s)  \"{text_preview}\"")

    # ── Step 3: Crab API inference ──
    print(f"\n{'='*80}")
    print(f"  Step 3: Crab API Batch Inference ({len(sentences)} segments)")
    print(f"{'='*80}")

    t0 = time.perf_counter()
    results = classify_segments_batch(args.url, sentences, wav_path)
    total_ms = (time.perf_counter() - t0) * 1000

    print(f"  Inference completed in {total_ms:.0f}ms")

    # ── Step 4: Display results ──
    display_results(sentences, results)


if __name__ == "__main__":
    main()
