"""
Diagnostic baseline: run current EN-trained Crab on all 1,120 synthetic clips.

Extract (pooled audio embedding, 3-class probs) for each clip.
Save → Crab/data/diagnostic_results.npz for subsequent analysis (B / C / D).

Resolves BILINGUAL_WORK_LOG.md A.2/A.3/A.4 — data collection step.
Analysis happens in diagnostic_analyze.py.
"""
from pathlib import Path
import sys
import time
import csv

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import numpy as np

# Add Crab to path so `src.models.ser` and `api.inference` resolve
CRAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRAB_ROOT))

from api.inference import CrabEmotionPredictor, MAX_AUDIO_LEN, TARGET_SR, TEXT_MAX_LEN, CLASSES

DATA_ROOT  = Path("/home/brant/Project/SAILER_test/datasets/output_multi_text20")
MODEL_DIR  = CRAB_ROOT / "experiments" / "interview_scheme1"
TRANSCRIPT = CRAB_ROOT / "data" / "synth_transcripts.csv"
OUT_NPZ    = CRAB_ROOT / "data" / "diagnostic_results.npz"

BATCH_SIZE = 8
EMOTIONS = ["angry", "depressed", "disgust", "fear", "happy", "peaceful", "sad", "surprise"]
ALPHAS   = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]


def load_transcripts():
    """folder_name -> transcript text."""
    m = {}
    with TRANSCRIPT.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m[row["folder"]] = row["transcript"]
    return m


def enumerate_clips(folders_filter=None):
    """Return list of (wav_path, folder, lang, emotion, alpha)."""
    rows = []
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        if folders_filter and folder.name not in folders_filter:
            continue
        lang = folder.name.split("_")[-1]
        for emo in EMOTIONS:
            emo_dir = folder / emo
            if not emo_dir.exists():
                continue
            for alpha in ALPHAS:
                wavs = sorted(emo_dir.glob(f"*_alpha{alpha}.wav"))
                if not wavs:
                    continue
                rows.append({
                    "wav":    str(wavs[0]),
                    "folder": folder.name,
                    "lang":   lang,
                    "emotion": emo,
                    "alpha":  float(alpha),
                })
    return rows


def preprocess(wav_path, wav_mean, wav_std):
    """Mirror CrabEmotionPredictor._preprocess_audio (using soundfile for loading)."""
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)  # [T, C]
    waveform = torch.from_numpy(data.T)  # [C, T]
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    waveform = waveform.mean(dim=0)
    waveform = waveform[:MAX_AUDIO_LEN]
    num_valid = waveform.shape[0]
    waveform = (waveform - wav_mean) / (wav_std + 1e-6)
    return waveform.float(), num_valid


def main():
    print(f"Loading Crab from {MODEL_DIR}…")
    pred = CrabEmotionPredictor(model_dir=str(MODEL_DIR))
    device = pred.device

    transcripts = load_transcripts()
    clips = enumerate_clips()
    print(f"Found {len(clips)} clips total")
    cn = sum(1 for c in clips if c["lang"] == "cn")
    en = sum(1 for c in clips if c["lang"] == "en")
    print(f"  CN: {cn}   EN: {en}")
    assert cn == 560 and en == 560, f"unexpected clip count: cn={cn} en={en}"

    n = len(clips)
    audio_embs = np.zeros((n, 1024), dtype=np.float32)
    probs_arr  = np.zeros((n, len(CLASSES)), dtype=np.float32)

    t0 = time.time()
    for batch_start in range(0, n, BATCH_SIZE):
        batch = clips[batch_start: batch_start + BATCH_SIZE]

        # 1. preprocess audio
        wavs, vlens = [], []
        for c in batch:
            wav, vlen = preprocess(c["wav"], pred.wav_mean, pred.wav_std)
            wavs.append(wav)
            vlens.append(vlen)
        max_len = max(vlens)
        padded = torch.zeros(len(batch), max_len)
        amask  = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, (w, vl) in enumerate(zip(wavs, vlens)):
            padded[i, :vl] = w[:vl]
            amask[i, :vl]  = 1
        padded = padded.to(device)
        amask  = amask.to(device)

        # 2. tokenize text (per-folder transcript)
        texts = [transcripts[c["folder"]] for c in batch]
        tok = pred.tokenizer(
            texts, return_tensors="pt", max_length=TEXT_MAX_LEN,
            padding="max_length", truncation=True,
        )
        input_ids = tok["input_ids"].to(device)
        tmask     = tok["attention_mask"].to(device)

        # 3. forward
        with torch.no_grad():
            audio_feat = pred.ssl_model(padded, attention_mask=amask).last_hidden_state  # [B,T',1024]
            text_feat  = pred.text_model(input_ids=input_ids, attention_mask=tmask).last_hidden_state
            logits     = pred.ser_model(audio_feat, text_feat)
            probs      = F.softmax(logits, dim=1)

        # 4. pool audio embedding (mean over valid time steps)
        #    audio_feat downsamples T -> T' (~320x reduction for WavLM)
        T_prime = audio_feat.shape[1]
        time_scale = max_len / T_prime
        for i, vl in enumerate(vlens):
            valid_T = max(1, int(vl / time_scale))
            pooled = audio_feat[i, :valid_T].mean(dim=0).cpu().numpy()
            audio_embs[batch_start + i] = pooled

        probs_arr[batch_start: batch_start + len(batch)] = probs.cpu().numpy()

        if (batch_start // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            done = batch_start + len(batch)
            rate = done / elapsed
            eta  = (n - done) / max(rate, 1e-6)
            print(f"  [{done:4d}/{n}] elapsed={elapsed:.1f}s rate={rate:.1f} clip/s eta={eta:.0f}s")

    # save
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_NPZ,
        audio_embs = audio_embs,
        probs      = probs_arr,
        folders    = np.array([c["folder"] for c in clips]),
        langs      = np.array([c["lang"]   for c in clips]),
        emotions   = np.array([c["emotion"] for c in clips]),
        alphas     = np.array([c["alpha"]  for c in clips], dtype=np.float32),
        classes    = np.array(CLASSES),
    )
    print(f"\n✅ Done in {time.time()-t0:.1f}s")
    print(f"   Saved → {OUT_NPZ}   shape: embs={audio_embs.shape} probs={probs_arr.shape}")


if __name__ == "__main__":
    main()
