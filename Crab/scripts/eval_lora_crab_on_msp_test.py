"""
Post-hoc eval a LoRA-based Crab ckpt on MSP-Podcast Test1 + Test2.

Handles LoRA-adapter checkpoints (like N3 v1-warmstart NNIME LoRA) that store:
  audio_lora_adapter/     (via PeftModel.save_pretrained)
  text_lora_adapter/
  final_ser.pt
  train_norm_stat.pkl

The generic eval_crab_on_msp_test.py assumes Full-FT ckpts (final_ssl.pt / final_text.pt
state_dict override). This script instead loads base HF encoders and wraps them with
PeftModel.from_pretrained to attach the LoRA adapter.

Primary use case:
  Check whether N3 (stage-2 finetuned on 3k NNIME ZH only) catastrophically forgets
  the v1 backbone's EN capability. i.e. answer:
    "N3 boosted ZH by +0.033, but did EN performance survive?"

Usage
-----
  python scripts/eval_lora_crab_on_msp_test.py \
      --model_dir ./experiments/nnime_v1_warmstart_lora \
      --ssl_type facebook/wav2vec2-xls-r-300m \
      --text_model_path FacebookAI/xlm-roberta-large \
      --run_tag n3_v1_nnime_lora
"""
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import sys
import time

import numpy as np
import wandb

CRAB = Path("/home/brant/Project/SAILER_test/Crab")
sys.path.insert(0, str(CRAB))

from api.inference import CrabEmotionPredictor, CLASSES  # noqa: E402

CSV_PATH = CRAB / "data" / "msp2_interview_scheme1.csv"
WAV_BASE = Path("/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios")

WANDB_PROJECT = "Crab_Bilingual_ZH"
WANDB_ENTITY  = "d1249119-feng-chia-university"


def load_split(split_name):
    rows, missing = [], 0
    with CSV_PATH.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Split_Set"] != split_name:
                continue
            wav = WAV_BASE / r["FileName"]
            if not wav.exists():
                missing += 1
                continue
            rows.append(r)
    return rows, missing


def label_index(row):
    for i, c in enumerate(CLASSES):
        try:
            if int(float(row[c])) == 1:
                return i
        except (ValueError, TypeError):
            continue
    vals = []
    for c in CLASSES:
        try:
            vals.append(float(row[c]))
        except (ValueError, TypeError):
            vals.append(0.0)
    return int(np.argmax(vals))


def attach_lora(predictor, model_dir):
    """Wrap predictor's ssl_model and text_model with PeftModel loaded from
    LoRA adapter subdirs. Called after CrabEmotionPredictor.__init__ (base
    HF weights already loaded — final_ssl.pt / final_text.pt do NOT exist
    for LoRA ckpts, so the predictor kept pristine base weights)."""
    from peft import PeftModel

    audio_adapter = Path(model_dir) / "audio_lora_adapter"
    text_adapter  = Path(model_dir) / "text_lora_adapter"

    if not audio_adapter.exists() or not text_adapter.exists():
        raise FileNotFoundError(
            f"LoRA adapter dirs missing under {model_dir}: "
            f"expected audio_lora_adapter/ and text_lora_adapter/"
        )

    print(f"  attaching audio LoRA from {audio_adapter}", flush=True)
    predictor.ssl_model = PeftModel.from_pretrained(
        predictor.ssl_model, str(audio_adapter)).to(predictor.device).eval()

    print(f"  attaching text LoRA from {text_adapter}", flush=True)
    predictor.text_model = PeftModel.from_pretrained(
        predictor.text_model, str(text_adapter)).to(predictor.device).eval()


def run_split(predictor, split_name, run_tag, batch_size,
              ssl_type, text_model_path, model_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"eval_{run_tag}_msp_{split_name}_{ts}"

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
               reinit=True,
               config={"step": "post_hoc_eval_lora",
                       "run_tag": run_tag,
                       "model_dir": model_dir,
                       "dataset": "MSP-Podcast",
                       "split": split_name,
                       "ssl_type": ssl_type,
                       "text_model": text_model_path})

    rows, missing = load_split(split_name)
    print(f"\n[{split_name}] samples: {len(rows)}  (missing wav: {missing})",
          flush=True)

    y_true = [label_index(r) for r in rows]

    y_pred = []
    t0 = time.time()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        audio_bytes_list, texts = [], []
        for r in chunk:
            audio_bytes_list.append((WAV_BASE / r["FileName"]).read_bytes())
            texts.append(r["Text"] if r["Text"] else "")

        try:
            results = predictor.predict_batch(audio_bytes_list, texts)
        except Exception as e:
            print(f"  ERROR at [{i}/{len(rows)}]: {e}", flush=True)
            results = [{c: (1.0/len(CLASSES)) for c in CLASSES}] * len(chunk)

        for res in results:
            probs = [res[c] for c in CLASSES]
            y_pred.append(int(np.argmax(probs)))

        done = i + len(chunk)
        if done % (batch_size * 50) == 0 or done == len(rows):
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(rows) - done) / max(rate, 1e-6)
            print(f"  [{done}/{len(rows)}]  {rate:.1f}/s  eta={eta:.0f}s",
                  flush=True)

    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, accuracy_score, recall_score,
                                 precision_recall_fscore_support)

    print(f"\n=== [{split_name}] Classification Report ===")
    print(classification_report(y_true, y_pred,
                                labels=list(range(len(CLASSES))),
                                target_names=CLASSES, digits=4,
                                zero_division=0))

    print(f"=== [{split_name}] Confusion Matrix (rows=true, cols=pred) ===")
    print("class order:", CLASSES)
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== [{split_name}] Headline ===")
    print(f"  macro-F1   : {macro_f1:.4f}")
    print(f"  weighted-F1: {weighted_f1:.4f}")
    print(f"  UAR        : {uar:.4f}")
    print(f"  accuracy   : {acc:.4f}")

    pred_dist = Counter(y_pred)
    true_dist = Counter(y_true)
    print(f"\n=== [{split_name}] Prediction distribution vs ground truth ===")
    for i, c in enumerate(CLASSES):
        p_pct = 100 * pred_dist[i] / len(y_pred)
        t_pct = 100 * true_dist[i] / len(y_true)
        print(f"  {c:>16s}  pred {pred_dist[i]:>5} ({p_pct:5.1f}%)  | true "
              f"{true_dist[i]:>5} ({t_pct:5.1f}%)")

    wandb.log({
        "test/macro_f1":    macro_f1,
        "test/weighted_f1": weighted_f1,
        "test/UAR":         uar,
        "test/accuracy":    acc,
        "test/n_samples":   len(y_true),
    })
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0)
    for i, c in enumerate(CLASSES):
        wandb.log({f"test/f1_{c}": f[i],
                   f"test/precision_{c}": p[i],
                   f"test/recall_{c}": r[i],
                   f"test/support_{c}": int(s[i]),
                   f"test/pred_pct_{c}": 100 * pred_dist[i] / len(y_pred),
                   f"test/true_pct_{c}": 100 * true_dist[i] / len(y_true)})

    wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=y_true, preds=y_pred, class_names=CLASSES)})
    wandb.finish()

    return {"split": split_name, "n": len(y_true), "macro_f1": macro_f1,
            "weighted_f1": weighted_f1, "uar": uar, "accuracy": acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="LoRA ckpt dir (contains audio_lora_adapter/, "
                        "text_lora_adapter/, final_ser.pt, train_norm_stat.pkl)")
    p.add_argument("--ssl_type", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--text_model_path", default="FacebookAI/xlm-roberta-large")
    p.add_argument("--run_tag", default="crab_lora")
    p.add_argument("--splits", nargs="+", default=["Test1", "Test2"],
                   choices=["Test1", "Test2"])
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    print(f"Loading base Crab from {args.model_dir} ...", flush=True)
    print(f"  ssl_type       : {args.ssl_type}", flush=True)
    print(f"  text_model_path: {args.text_model_path}", flush=True)
    predictor = CrabEmotionPredictor(
        model_dir=args.model_dir,
        ssl_type=args.ssl_type,
        text_model_path=args.text_model_path,
    )

    print("\nAttaching LoRA adapters (base encoders untouched by ckpt) ...", flush=True)
    attach_lora(predictor, args.model_dir)

    summary = []
    for split in args.splits:
        summary.append(run_split(predictor, split, args.run_tag,
                                 args.batch_size,
                                 args.ssl_type, args.text_model_path,
                                 args.model_dir))

    print(f"\n\n=== SUMMARY ({args.run_tag} on MSP-Podcast) ===")
    print(f"{'split':<8} {'n':>6} {'macro-F1':>9} {'wF1':>7} {'UAR':>7} {'acc':>7}")
    for s in summary:
        print(f"{s['split']:<8} {s['n']:>6} {s['macro_f1']:>9.4f} "
              f"{s['weighted_f1']:>7.4f} {s['uar']:>7.4f} {s['accuracy']:>7.4f}")


if __name__ == "__main__":
    main()
