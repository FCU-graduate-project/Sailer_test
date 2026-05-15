"""
Sliding Window Emotion Analysis with Whisper Text Alignment.

This script combines the Fixed Sliding Window approach (0-12s, 6-18s)
with Whisper word-level timestamps to provide BOTH audio and text
modalities to the Crab API for every window.

Usage:
    Crab/.venv/bin/python -m Crab.api.test_long_text --wav /path/to/interview.wav
"""

import argparse
import io
import os
import sys
import time
import requests
import numpy as np
import torchaudio

def transcribe_with_timestamps(wav_path: str, model_size: str = "large-v3", device: str = "cuda") -> list[dict]:
    from faster_whisper import WhisperModel
    print(f"  Loading Whisper model: {model_size} on {device}...")
    t0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type="float16")
    print(f"  Whisper loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
    
    print(f"  Transcribing {os.path.basename(wav_path)}...")
    segments, _ = model.transcribe(wav_path, word_timestamps=True)
    
    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    
    # Free memory
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
        
    return words

def build_windows(audio_duration: float, window_sec: float, stride_sec: float) -> list[tuple[float, float]]:
    windows = []
    start = 0.0
    while start < audio_duration:
        end = min(start + window_sec, audio_duration)
        if (end - start) < 1.0: # Skip very short trailing windows
            break
        windows.append((start, end))
        if end == audio_duration:
            break
        start += stride_sec
    return windows

def slice_audio(waveform, sr, start_sec: float, end_sec: float) -> bytes:
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = waveform[:, start_sample:end_sample]
    buf = io.BytesIO()
    torchaudio.save(buf, segment, sr, format="wav")
    return buf.getvalue()

def main():
    parser = argparse.ArgumentParser(description="Sliding window with text alignment")
    parser.add_argument("--url", default="http://localhost:8001", help="Crab API base URL")
    parser.add_argument("--wav", required=True, help="Path to audio file")
    parser.add_argument("--window", type=float, default=12.0, help="Window size in seconds")
    parser.add_argument("--stride", type=float, default=6.0, help="Stride in seconds")
    args = parser.parse_args()

    # 1. Health check
    try:
        requests.get(f"{args.url}/v1/health", timeout=5)
    except Exception:
        print(f"❌ Cannot connect to API at {args.url}")
        sys.exit(1)

    # 2. Get Audio Duration & Load Audio
    waveform, sr = torchaudio.load(args.wav)
    audio_duration = waveform.shape[1] / sr
    print(f"\n📁 Audio duration: {audio_duration:.1f}s")

    # 3. Whisper Transcription
    words = transcribe_with_timestamps(args.wav)
    
    # 4. Build Windows and Align Text
    windows = build_windows(audio_duration, args.window, args.stride)
    
    print(f"\n🔧 Created {len(windows)} sliding windows ({args.window}s window, {args.stride}s stride).")
    
    segments_to_send = []
    for start, end in windows:
        # Extract words that fall into this window
        window_words = [w["word"] for w in words if w["start"] >= start and w["end"] <= end]
        window_text = " ".join(window_words).strip()
        segments_to_send.append({
            "start": start,
            "end": end,
            "text": window_text
        })
        
    # 5. Send to API Batch
    print(f"\n🚀 Sending {len(segments_to_send)} windows to Crab API in chunks...")
    t0 = time.perf_counter()
    
    results = []
    batch_size = 16
    for i in range(0, len(segments_to_send), batch_size):
        chunk = segments_to_send[i:i+batch_size]
        files_list = []
        texts_list = []
        for j, seg in enumerate(chunk):
            audio_bytes = slice_audio(waveform, sr, seg["start"], seg["end"])
            files_list.append(("files", (f"win_{i+j}.wav", audio_bytes, "audio/wav")))
            texts_list.append(("texts", seg["text"]))
            
        resp = requests.post(f"{args.url}/v1/emotion/classify-batch", files=files_list, data=texts_list)
        if resp.status_code != 200:
            print(f"❌ API Error: {resp.text}")
            return
        results.extend(resp.json()["results"])
        
    api_ms = (time.perf_counter() - t0) * 1000
    
    # 6. Display Results
    print(f"\n================================================================================")
    print(f"  📊 Sliding Window Timeline (WITH TEXT)")
    print(f"================================================================================\n")
    
    avg_probs = {"Excited": 0.0, "Unconfident": 0.0, "Neutral_3Class": 0.0}
    
    for i, (seg, res) in enumerate(zip(segments_to_send, results)):
        probs = res["probabilities"]
        label = res["primary_label"]
        conf = res["primary_confidence"]
        
        for cls in avg_probs:
            avg_probs[cls] += probs.get(cls, 0)
            
        print(f"  [{i+1}] {seg['start']:5.1f}s ~ {seg['end']:5.1f}s | {label:14s} (conf: {conf:.3f})")
        
        text_preview = seg["text"][:80] + ("..." if len(seg["text"]) > 80 else "")
        if not text_preview:
            text_preview = "(No speech detected)"
        print(f"        📝 \"{text_preview}\"")
        print(f"        Excited: {probs.get('Excited', 0):.3f} | Unconfident: {probs.get('Unconfident', 0):.3f} | Neutral: {probs.get('Neutral_3Class', 0):.3f}\n")

    if len(results) > 0:
        avg_probs = {k: v / len(results) for k, v in avg_probs.items()}
        final_label = max(avg_probs, key=avg_probs.get)
        print(f"────────────────────────────────────────────────────────────────────────────────")
        print(f"  🎯 Overall Result (averaged over {len(results)} windows): {final_label}")
        for cls in ["Excited", "Unconfident", "Neutral_3Class"]:
            print(f"     {cls:16s}  {avg_probs[cls]:.3f}  {'█' * int(avg_probs[cls] * 40)}")
        print(f"================================================================================")
        print(f"  Crab API inference time: {api_ms:.0f}ms")

if __name__ == "__main__":
    main()
