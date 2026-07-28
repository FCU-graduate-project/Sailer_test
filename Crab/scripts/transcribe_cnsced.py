"""
Transcribe CNSCED audio (15,785 wav) with faster-whisper large-v3, Chinese.

Output: Crab/data/cnsced_transcripts.csv
  columns: rel_path, split, transcript, duration_sec, language

Features:
- Resume support (skip already-transcribed rows)
- Incremental flush every 100 files (crash-safe)
- Progress log every 100 files
- Stereo 44.1kHz -> mono 16kHz on the fly
- Filters out empty transcripts (Whisper may return "" for silence)

Estimated runtime: ~2-3 hr on RTX 3090 with int8_float16.
"""
import csv
import os
import sys
import time
from pathlib import Path

import librosa
from faster_whisper import WhisperModel

CNSCED_ROOT = Path("/home/brant/Project/SAILER_test/datasets/CNSCED")
OUT_CSV = Path("/home/brant/Project/SAILER_test/Crab/data/cnsced_transcripts.csv")
SPLITS = ["train", "val", "test"]

FLUSH_EVERY = 100
PROGRESS_EVERY = 100


def load_already_done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            done.add(row["rel_path"])
    return done


def list_all_audio():
    rows = []
    for split in SPLITS:
        d = CNSCED_ROOT / split
        for wav in sorted(d.glob("*.wav")):
            rel = f"{split}/{wav.name}"
            rows.append((rel, split, wav))
    return rows


def main():
    todo = list_all_audio()
    done = load_already_done(OUT_CSV)
    new_todo = [r for r in todo if r[0] not in done]
    print(f"[init] total={len(todo)}  already_done={len(done)}  remaining={len(new_todo)}", flush=True)
    if not new_todo:
        print("[init] all transcribed; exiting", flush=True)
        return

    print("[init] loading faster-whisper large-v3 (int8_float16, GPU)...", flush=True)
    t0 = time.time()
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    print(f"[init] model loaded in {time.time()-t0:.1f}s", flush=True)

    write_header = not OUT_CSV.exists()
    f = open(OUT_CSV, "a", encoding="utf-8", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["rel_path", "split", "transcript", "duration_sec", "language"])
        f.flush()

    t_start = time.time()
    n = 0
    fail = 0
    buf = []
    for rel, split, wav_path in new_todo:
        try:
            audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
            dur = len(audio) / sr
            segs, info = model.transcribe(
                audio,
                language="zh",
                task="transcribe",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segs).strip()
            buf.append([rel, split, text, f"{dur:.3f}", info.language])
        except Exception as e:
            fail += 1
            buf.append([rel, split, "<ERROR>", "0", str(e)[:80]])

        n += 1
        if n % FLUSH_EVERY == 0:
            for row in buf:
                w.writerow(row)
            f.flush()
            buf = []
        if n % PROGRESS_EVERY == 0:
            elapsed = time.time() - t_start
            rate = n / elapsed
            eta = (len(new_todo) - n) / rate / 60
            print(f"[{n}/{len(new_todo)}] rate={rate:.1f}/s  elapsed={elapsed/60:.1f}min  ETA={eta:.1f}min  fail={fail}", flush=True)

    # flush remaining
    for row in buf:
        w.writerow(row)
    f.flush()
    f.close()
    print(f"[done] total transcribed={n}  fail={fail}  elapsed={(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
