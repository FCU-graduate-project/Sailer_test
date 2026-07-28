"""
Per-language eval for any Crab-family checkpoint on a CSV that carries a
'Language' column (currently: bilingual_strategyA.csv).

Usage:
  .venv/bin/python scripts/eval_per_language.py \
      --model_dir experiments/strategyA_xlsr_xlmr_lora \
      --df_path data/bilingual_strategyA.csv \
      --split Test \
      --ssl_type facebook/wav2vec2-xls-r-300m \
      --text_model FacebookAI/xlm-roberta-large \
      --run_tag strategyA_test

Logs three wandb runs (Crab_Bilingual_ZH project):
  • <tag>__overall   — mixed test set headline (matches train run's test/macro_f1)
  • <tag>__EN        — MSP Test1 subset only
  • <tag>__ZH        — EmotionTalk Test subset only

The model_dir must contain a LoRA checkpoint produced by train_crab_lora.py:
  text_lora_adapter/, audio_lora_adapter/, final_ser.pt, train_norm_stat.pkl
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import wandb
from peft import PeftModel
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, recall_score,
                             precision_recall_fscore_support)
from transformers import AutoModel, AutoTokenizer

CRAB = Path("/home/brant/Project/SAILER_test/Crab")
sys.path.insert(0, str(CRAB))

import src.models as net  # noqa: E402
from src.data.dataset.dataset import load_norm_stat  # noqa: E402

WANDB_PROJECT = "Crab_Bilingual_ZH"
WANDB_ENTITY  = "d1249119-feng-chia-university"
CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]

TARGET_SR = 16000
MAX_DUR_SEC = 12
MAX_SAMPLES = TARGET_SR * MAX_DUR_SEC


def load_wav(path: str) -> np.ndarray:
    import torchaudio
    wav, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.mean(dim=0).numpy()  # mono
    if len(wav) > MAX_SAMPLES:
        wav = wav[:MAX_SAMPLES]
    return wav


class CrabLoraInfer:
    """Loads a Crab checkpoint with auto mode detection:
        • LoRA          → text_lora_adapter/ + audio_lora_adapter/ present
        • Full / partial FT → final_ssl.pt + final_text.pt present (no adapter dirs)
    Either layout uses final_ser.pt for the cross-modal head.
    """
    def __init__(self, model_dir, ssl_type, text_model, device="cuda",
                 fusion_hidden_dim=512, num_emotions=3):
        self.device = device
        self.dir = Path(model_dir)

        ssl = AutoModel.from_pretrained(ssl_type)
        if (self.dir / "audio_lora_adapter").exists():
            ssl = PeftModel.from_pretrained(ssl, str(self.dir / "audio_lora_adapter"))
            print(f"  loaded audio LoRA from {self.dir/'audio_lora_adapter'}")
        elif (self.dir / "final_ssl.pt").exists():
            ssl_state = torch.load(self.dir / "final_ssl.pt", map_location=device)
            ssl.load_state_dict(ssl_state)
            print(f"  loaded audio full state_dict from {self.dir/'final_ssl.pt'}")
        else:
            print(f"  ⚠️ no audio weights in {self.dir} — using HF pretrained only")
        self.ssl = ssl.to(device).eval()

        text = AutoModel.from_pretrained(text_model)
        if (self.dir / "text_lora_adapter").exists():
            text = PeftModel.from_pretrained(text, str(self.dir / "text_lora_adapter"))
            print(f"  loaded text LoRA from {self.dir/'text_lora_adapter'}")
        elif (self.dir / "final_text.pt").exists():
            text_state = torch.load(self.dir / "final_text.pt", map_location=device)
            text.load_state_dict(text_state)
            print(f"  loaded text full state_dict from {self.dir/'final_text.pt'}")
        else:
            print(f"  ⚠️ no text weights in {self.dir} — using HF pretrained only")
        self.text = text.to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)

        # PeftModel nests under .base_model.model; plain HF models expose config directly.
        a_dim = (self.ssl.base_model.model.config.hidden_size
                 if isinstance(self.ssl, PeftModel) else self.ssl.config.hidden_size)
        t_dim = (self.text.base_model.model.config.hidden_size
                 if isinstance(self.text, PeftModel) else self.text.config.hidden_size)
        self.ser = net.MultiModalEmotionClassifierDeep(
            features1_dim=a_dim, features2_dim=t_dim,
            fusion_hidden_dim=fusion_hidden_dim, num_emotions=num_emotions, dropout=0.5,
        ).to(device).eval()
        ser_state = torch.load(self.dir / "final_ser.pt", map_location=device)
        self.ser.load_state_dict(ser_state)

        nstat = self.dir / "train_norm_stat.pkl"
        if nstat.exists():
            self.wav_mean, self.wav_std = load_norm_stat(str(nstat))
            print(f"  loaded norm_stat mean={self.wav_mean:.4f} std={self.wav_std:.4f}")
        else:
            self.wav_mean, self.wav_std = 0.0, 1.0
            print(f"  ⚠️ no train_norm_stat.pkl at {nstat}; using 0/1")

    @torch.no_grad()
    def predict_one(self, wav_path, text):
        wav = load_wav(wav_path)
        wav = (wav - self.wav_mean) / (self.wav_std + 1e-6)
        x_audio = torch.tensor(wav).float().unsqueeze(0).to(self.device)
        mask_audio = torch.ones_like(x_audio)
        enc = self.tokenizer(text or "", return_tensors="pt", padding="max_length",
                             truncation=True, max_length=128)
        x_text = enc["input_ids"].to(self.device)
        mask_text = enc["attention_mask"].to(self.device)
        ssl_h = self.ssl(x_audio, attention_mask=mask_audio).last_hidden_state
        text_h = self.text(input_ids=x_text, attention_mask=mask_text).last_hidden_state
        logits = self.ser(ssl_h, text_h)
        return torch.softmax(logits, -1).cpu().numpy()[0]


def load_subset(csv_path, split, language=None):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Split_Set"] != split:
                continue
            if language is not None and r.get("Language") != language:
                continue
            rows.append(r)
    return rows


def eval_subset(infer, rows, label_name, wav_base_dir=""):
    y_true, y_pred = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        # label
        matched = False
        for j, c in enumerate(CLASSES):
            if int(float(r[c])) == 1:
                y_true.append(j); matched = True; break
        if not matched:
            y_true.append(int(np.argmax([float(r[c]) for c in CLASSES])))
        # wav path: absolute (bilingual CSV) or relative (single-lang CSV)
        wav_path = r["FileName"]
        if not os.path.isabs(wav_path) and wav_base_dir:
            wav_path = os.path.join(wav_base_dir, wav_path)
        probs = infer.predict_one(wav_path, r["Text"])
        y_pred.append(int(np.argmax(probs)))
        if (i + 1) % 200 == 0 or (i + 1) == len(rows):
            print(f"  [{label_name}] {i+1}/{len(rows)}  "
                  f"rate={ (i+1)/(time.time()-t0):.1f}/s", flush=True)
    return np.array(y_true), np.array(y_pred)


def log_metrics(y_true, y_pred, label):
    if len(y_true) == 0:
        print(f"  [{label}] no samples"); return None
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 labels=list(range(len(CLASSES))),
                                                 zero_division=0)
    pred_dist = Counter(y_pred); true_dist = Counter(y_true)
    print(f"\n=== [{label}] ===")
    print(classification_report(y_true, y_pred,
                                labels=list(range(len(CLASSES))),
                                target_names=CLASSES, digits=4, zero_division=0))
    print(f"  macro-F1 {macro_f1:.4f}  UAR {uar:.4f}  acc {acc:.4f}")
    payload = {"n": len(y_true), "macro_f1": macro_f1, "weighted_f1": weighted_f1,
               "uar": uar, "accuracy": acc}
    for i, c in enumerate(CLASSES):
        payload[f"f1_{c}"] = f[i]; payload[f"precision_{c}"] = p[i]
        payload[f"recall_{c}"] = r[i]; payload[f"support_{c}"] = int(s[i])
        payload[f"pred_pct_{c}"] = 100*pred_dist[i]/len(y_pred)
        payload[f"true_pct_{c}"] = 100*true_dist[i]/len(y_true)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--df_path", required=True)
    ap.add_argument("--split", default="Test")
    ap.add_argument("--ssl_type", required=True)
    ap.add_argument("--text_model", required=True)
    ap.add_argument("--run_tag", required=True,
                    help="wandb run prefix; will spawn <tag>__overall/__EN/__ZH")
    ap.add_argument("--fusion_hidden_dim", type=int, default=512)
    ap.add_argument("--wav_base_dir", default="",
                    help="prepend to relative FileName entries (single-lang CSV); "
                         "ignored for absolute paths (bilingual CSV)")
    args = ap.parse_args()

    infer = CrabLoraInfer(args.model_dir, args.ssl_type, args.text_model,
                          fusion_hidden_dim=args.fusion_hidden_dim)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for subset_label, language in [("overall", None), ("EN", "EN"), ("ZH", "ZH")]:
        rows = load_subset(args.df_path, args.split, language)
        print(f"\n>>> Eval {subset_label} ({len(rows)} samples)")
        if not rows:
            print(f"  skip — empty subset"); continue

        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                   name=f"{args.run_tag}__{subset_label}_{ts}", reinit=True,
                   config={"model_dir": args.model_dir, "split": args.split,
                           "subset": subset_label, "ssl_type": args.ssl_type,
                           "text_model": args.text_model})
        y_true, y_pred = eval_subset(infer, rows, subset_label, args.wav_base_dir)
        payload = log_metrics(y_true, y_pred, subset_label)
        for k, v in payload.items():
            wandb.log({f"test/{k}": v})
        wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=y_true.tolist(), preds=y_pred.tolist(),
            class_names=CLASSES)})
        rep = classification_report(y_true, y_pred,
                                    labels=list(range(len(CLASSES))),
                                    target_names=CLASSES, digits=4,
                                    zero_division=0)
        wandb.log({"test/classification_report": wandb.Html(f"<pre>{rep}</pre>")})
        wandb.finish()


if __name__ == "__main__":
    main()
