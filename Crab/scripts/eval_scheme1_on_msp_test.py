"""
Run scheme1 (English Crab) on MSP-Podcast Test1 / Test2 — the EN baseline that
was never captured at training time (train_crab.py had no --eval_test flag).

This is the proper anchor for Q3's "EN drop ≤ 0.05" criterion that bilingual
runs (Hybrid B, Strategy A) must respect.

Reads /home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme1.csv,
filters by Split_Set ∈ {Test1, Test2}, runs CrabEmotionPredictor.predict_batch
in chunks, logs per-split metrics to wandb (Crab_Bilingual_ZH project).
"""
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
SCHEME1  = CRAB / "experiments" / "interview_scheme1"
BATCH    = 8

WANDB_PROJECT = "Crab_Bilingual_ZH"
WANDB_ENTITY  = "d1249119-feng-chia-university"

SPLITS = ["Test1", "Test2"]


def load_split(split_name):
    rows = []
    missing = 0
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
    # one-hot over CLASSES. Match the convention used by eval_per_language.py
    # and confusion_matrix_per_language.py: strict `int(float()) == 1`, with
    # argmax fallback for any malformed row (none expected in current CSVs).
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


def run_split(predictor, split_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"eval_scheme1_msp_{split_name}_{ts}"

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
               reinit=True,
               config={"step": "scheme1_EN_baseline",
                       "model": "scheme1",
                       "dataset": "MSP-Podcast",
                       "split": split_name,
                       "ssl_type": "microsoft/wavlm-large",
                       "text_model": "roberta-large"})

    rows, missing = load_split(split_name)
    print(f"\n[{split_name}] samples: {len(rows)}  (missing wav: {missing})", flush=True)

    y_true = [label_index(r) for r in rows]

    y_pred = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        audio_bytes_list, texts = [], []
        for r in chunk:
            audio_bytes_list.append((WAV_BASE / r["FileName"]).read_bytes())
            texts.append(r["Text"] if r["Text"] else "")

        try:
            results = predictor.predict_batch(audio_bytes_list, texts)
        except Exception as e:
            print(f"  ERROR at [{i}/{len(rows)}]: {e}", flush=True)
            # skip this chunk — pad with argmax-of-zeros (Excited=0) to keep alignment
            results = [{c: (1.0/len(CLASSES)) for c in CLASSES}] * len(chunk)

        for res in results:
            probs = [res[c] for c in CLASSES]
            y_pred.append(int(np.argmax(probs)))

        done = i + len(chunk)
        if done % (BATCH * 50) == 0 or done == len(rows):
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(rows) - done) / max(rate, 1e-6)
            print(f"  [{done}/{len(rows)}]  {rate:.1f}/s  eta={eta:.0f}s", flush=True)

    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    from sklearn.metrics import (classification_report, confusion_matrix,
                                 f1_score, accuracy_score, recall_score,
                                 precision_recall_fscore_support)

    print(f"\n=== [{split_name}] Classification Report ===")
    print(classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                target_names=CLASSES, digits=4, zero_division=0))

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
        print(f"  {c:>16s}  pred {pred_dist[i]:>5} ({p_pct:5.1f}%)  | true {true_dist[i]:>5} ({t_pct:5.1f}%)")

    wandb.log({
        "test/macro_f1":    macro_f1,
        "test/weighted_f1": weighted_f1,
        "test/UAR":         uar,
        "test/accuracy":    acc,
        "test/n_samples":   len(y_true),
    })
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 labels=list(range(len(CLASSES))),
                                                 zero_division=0)
    for i, c in enumerate(CLASSES):
        wandb.log({f"test/f1_{c}": f[i],
                   f"test/precision_{c}": p[i],
                   f"test/recall_{c}": r[i],
                   f"test/support_{c}": int(s[i]),
                   f"test/pred_pct_{c}": 100 * pred_dist[i] / len(y_pred),
                   f"test/true_pct_{c}": 100 * true_dist[i] / len(y_true)})

    wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=y_true, preds=y_pred, class_names=CLASSES)})

    rep_txt = classification_report(y_true, y_pred,
                                    labels=list(range(len(CLASSES))),
                                    target_names=CLASSES, digits=4,
                                    zero_division=0)
    wandb.log({"test/classification_report": wandb.Html(f"<pre>{rep_txt}</pre>")})
    wandb.finish()

    return {
        "split": split_name,
        "n": len(y_true),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "uar": uar,
        "accuracy": acc,
    }


def main():
    print(f"Loading scheme1 from {SCHEME1} ...", flush=True)
    predictor = CrabEmotionPredictor(model_dir=str(SCHEME1))

    summary = []
    for split in SPLITS:
        summary.append(run_split(predictor, split))

    print("\n\n=== SUMMARY (scheme1 on MSP-Podcast) ===")
    print(f"{'split':<8} {'n':>6} {'macro-F1':>9} {'wF1':>7} {'UAR':>7} {'acc':>7}")
    for s in summary:
        print(f"{s['split']:<8} {s['n']:>6} {s['macro_f1']:>9.4f} "
              f"{s['weighted_f1']:>7.4f} {s['uar']:>7.4f} {s['accuracy']:>7.4f}")


if __name__ == "__main__":
    main()
