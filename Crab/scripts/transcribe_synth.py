"""
Transcribe the 20 unique synthetic texts (one wav per speaker folder).

The synthetic dataset structure:
  datasets/output_multi_text20/{NN_lang}/{emotion}/text{ID}_{emotion}_alpha{0.3..0.9}.wav
  - 00_en..09_en : 10 EN speakers, text1..text10
  - 10_cn..19_cn : 10 CN speakers, text11..text20
  - 8 emotions × 7 alpha = 56 wav per folder

We only need ONE transcript per (folder, text_id) since the text is shared across
all (emotion, alpha) combos within a folder. → Total 20 transcripts.

Output: Crab/data/synth_transcripts.csv
        columns: folder, text_id, lang, transcript
"""
from pathlib import Path
import csv
import sys
import time

from transformers import pipeline
import torch

ROOT = Path("/home/brant/Project/SAILER_test/datasets/output_multi_text20")
OUT  = Path("/home/brant/Project/SAILER_test/Crab/data/synth_transcripts.csv")

# pick a neutral-ish wav from each folder: 'peaceful' emotion, alpha 0.5
# (peaceful is closest to neutral; alpha 0.5 is mid-intensity)
PICK_EMOTION = "peaceful"
PICK_ALPHA   = "0.5"


def discover_folders():
    rows = []
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name  # e.g., "10_cn"
        lang = name.split("_")[-1]  # en or cn
        # pick the wav
        emo_dir = folder / PICK_EMOTION
        if not emo_dir.exists():
            print(f"⚠ skip {folder} — no {PICK_EMOTION}/ dir")
            continue
        wavs = sorted(emo_dir.glob(f"*_alpha{PICK_ALPHA}.wav"))
        if not wavs:
            print(f"⚠ skip {folder} — no alpha={PICK_ALPHA} wav in {emo_dir}")
            continue
        wav = wavs[0]
        # parse text_id from filename: text11_peaceful_alpha0.5.wav -> text11
        text_id = wav.name.split("_")[0]
        rows.append({"folder": name, "text_id": text_id, "lang": lang, "wav": str(wav)})
    return rows


def main():
    rows = discover_folders()
    print(f"Found {len(rows)} folders to transcribe")
    for r in rows:
        print(f"  {r['folder']:8} text_id={r['text_id']:6} lang={r['lang']:3} wav={Path(r['wav']).name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading whisper-large-v3 on {device}…")
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print("Transcribing…")
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["folder", "text_id", "lang", "transcript"])
        writer.writeheader()
        for r in rows:
            lang_code = "zh" if r["lang"] == "cn" else "en"
            result = asr(
                r["wav"],
                generate_kwargs={"language": lang_code, "task": "transcribe"},
            )
            text = result["text"].strip()
            print(f"  [{r['folder']}] {text}")
            writer.writerow({
                "folder":     r["folder"],
                "text_id":    r["text_id"],
                "lang":       r["lang"],
                "transcript": text,
            })

    print(f"\n✅ Done in {time.time()-t0:.1f}s")
    print(f"   Saved → {OUT}")


if __name__ == "__main__":
    main()
