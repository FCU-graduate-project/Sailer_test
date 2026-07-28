"""
Confusion matrix per language for any Crab-family model on any 3-class CSV.

Auto-detects LoRA vs full-FT models:
  - LoRA model dir (has text_lora_adapter/): uses CrabLoraInfer (eval_per_language.py)
  - Full-FT model dir (has final_text.pt): uses api.inference.CrabEmotionPredictor

Per language (or 'overall' if --languages omitted) emits THREE artefacts:
  <output_dir>/<tag>_confusion_<lang>.png   — 2-panel heatmap (counts + row-norm)
  <output_dir>/<tag>_confusion_<lang>.txt   — classification report + raw cm
  <output_dir>/<tag>_confusion_<lang>.csv   — raw cm counts for paper tables

Usage examples:
  # Strategy A on bilingual test, EN + ZH split
  .venv/bin/python scripts/confusion_matrix_per_language.py \
    --model_dir experiments/strategyA_xlsr_xlmr_lora \
    --df_path  data/bilingual_strategyA.csv --split Test \
    --ssl_type facebook/wav2vec2-xls-r-300m \
    --text_model FacebookAI/xlm-roberta-large \
    --languages EN ZH \
    --tag strategyA

  # Hybrid B on EmotionTalk test (ZH only — no Language column)
  .venv/bin/python scripts/confusion_matrix_per_language.py \
    --model_dir experiments/emotiontalk_hybridB_lora \
    --df_path  data/emotiontalk_crab_format.csv --split Test \
    --ssl_type microsoft/wavlm-large \
    --text_model FacebookAI/xlm-roberta-large \
    --wav_base_dir /home/brant/Project/SAILER_test/datasets/emotiontalk/Audio16k \
    --tag hybridB_zh

  # scheme1 (full FT) on EmotionTalk Test (ZH zero-shot baseline)
  .venv/bin/python scripts/confusion_matrix_per_language.py \
    --model_dir experiments/interview_scheme1 \
    --df_path  data/emotiontalk_crab_format.csv --split Test \
    --wav_base_dir /home/brant/Project/SAILER_test/datasets/emotiontalk/Audio16k \
    --tag scheme1_zh_zeroshot
"""
from pathlib import Path
from collections import Counter
import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

CRAB = Path("/home/brant/Project/SAILER_test/Crab")
sys.path.insert(0, str(CRAB))
sys.path.insert(0, str(CRAB / "scripts"))

CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
CLASS_SHORT = {"Excited": "Excited", "Unconfident": "Unconf", "Neutral_3Class": "Neutral"}


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


def is_lora_model(model_dir):
    return (Path(model_dir) / "text_lora_adapter").exists()


def label_index(r):
    for j, c in enumerate(CLASSES):
        try:
            if int(float(r[c])) == 1:
                return j
        except (ValueError, TypeError):
            continue
    return int(np.argmax([float(r[c]) for c in CLASSES]))


class FullModelAdapter:
    """Wrap api.inference.CrabEmotionPredictor with predict_one signature."""
    def __init__(self, model_dir):
        from api.inference import CrabEmotionPredictor
        self.predictor = CrabEmotionPredictor(model_dir=str(model_dir))

    def predict_one(self, wav_path, text):
        audio_bytes = Path(wav_path).read_bytes()
        res = self.predictor.predict_batch([audio_bytes], [text or ""])[0]
        return [res[c] for c in CLASSES]


def get_inferrer(model_dir, ssl_type, text_model):
    if is_lora_model(model_dir):
        if not ssl_type or not text_model:
            raise ValueError("LoRA model requires --ssl_type and --text_model")
        from eval_per_language import CrabLoraInfer
        return CrabLoraInfer(model_dir, ssl_type, text_model)
    return FullModelAdapter(model_dir)


def render_one(label, rows, infer, wav_base_dir, output_dir, tag):
    print(f"\n>>> [{label}] eval {len(rows)} samples")
    y_true, y_pred = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        y_true.append(label_index(r))
        wav_path = r["FileName"]
        if not os.path.isabs(wav_path) and wav_base_dir:
            wav_path = os.path.join(wav_base_dir, wav_path)
        probs = infer.predict_one(wav_path, r["Text"])
        y_pred.append(int(np.argmax(probs)))
        if (i + 1) % 500 == 0 or (i + 1) == len(rows):
            rate = (i + 1) / (time.time() - t0)
            eta = (len(rows) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(rows)}] {rate:.1f}/s eta={eta:.0f}s", flush=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

    rep = classification_report(y_true, y_pred,
                                labels=list(range(len(CLASSES))),
                                target_names=CLASSES, digits=4,
                                zero_division=0)
    pred_dist = Counter(y_pred); true_dist = Counter(y_true)
    dist_lines = ["\nPrediction distribution vs ground truth:"]
    for i, c in enumerate(CLASSES):
        p_pct = 100 * pred_dist[i] / len(y_pred)
        t_pct = 100 * true_dist[i] / len(y_true)
        dist_lines.append(f"  {c:>16}  pred {pred_dist[i]:>5} ({p_pct:5.1f}%)  | true {true_dist[i]:>5} ({t_pct:5.1f}%)")
    dist_block = "\n".join(dist_lines)

    print(rep)
    print(dist_block)

    prefix = f"{tag}_" if tag else ""
    txt_path = output_dir / f"{prefix}confusion_{label}.txt"
    csv_path = output_dir / f"{prefix}confusion_{label}.csv"
    png_path = output_dir / f"{prefix}confusion_{label}.png"

    with open(txt_path, "w") as f:
        f.write(rep + dist_block + "\n\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(", " + ", ".join(CLASSES) + "\n")
        for cls, row in zip(CLASSES, cm):
            f.write(cls + ", " + ", ".join(str(x) for x in row) + "\n")
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow([""] + CLASSES)
        for c, row in zip(CLASSES, cm):
            w.writerow([c] + list(row))

    # PNG: 2-panel
    short = [CLASS_SHORT[c] for c in CLASSES]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short, yticklabels=short, ax=axes[0],
                cbar_kws={"label": "count"})
    axes[0].set_title(f"{label} — counts (n={len(rows)})")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = cm.astype(float) / np.where(row_sum > 0, row_sum, 1)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1,
                xticklabels=short, yticklabels=short, ax=axes[1],
                cbar_kws={"label": "recall (row-norm)"})
    axes[1].set_title(f"{label} — row-normalized (recall view)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"  saved: {txt_path.name}, {csv_path.name}, {png_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--df_path", required=True)
    ap.add_argument("--split", default="Test")
    ap.add_argument("--ssl_type", default=None)
    ap.add_argument("--text_model", default=None)
    ap.add_argument("--languages", nargs="*", default=None,
                    help="e.g. EN ZH. Omit for 'overall' only.")
    ap.add_argument("--wav_base_dir", default="")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    output_dir = Path(args.output_dir or
                      (Path(args.model_dir) / f"confusion_{args.split}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    infer = get_inferrer(args.model_dir, args.ssl_type, args.text_model)

    subsets = ([(lang, lang) for lang in args.languages]
               if args.languages else [("overall", None)])

    for label, language in subsets:
        rows = load_subset(args.df_path, args.split, language)
        if not rows:
            print(f"[{label}] empty subset — skip")
            continue
        render_one(label, rows, infer, args.wav_base_dir, output_dir, args.tag)


if __name__ == "__main__":
    main()
