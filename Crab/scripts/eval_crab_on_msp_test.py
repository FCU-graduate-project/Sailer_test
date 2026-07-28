"""
Generic post-hoc eval: run any Crab Full-FT / LoRA checkpoint on MSP-Podcast
Test1 + Test2 splits — needed because train_crab_lora.py filters Split_Set == 'Test'
(exact), which misses MSP's Test1/Test2 convention → --eval_test at end of training
produces nothing for MSP scheme1 CSV.

Usage
-----
  # scheme1-XLMR Full FT ckpt (XLS-R + XLM-R):
  python scripts/eval_crab_on_msp_test.py \
      --model_dir ./experiments/scheme1_xlmr_fullft \
      --ssl_type facebook/wav2vec2-xls-r-300m \
      --text_model_path FacebookAI/xlm-roberta-large \
      --run_tag scheme1_xlmr_fullft

  # (optional) restrict to Test1 only:
  python scripts/eval_crab_on_msp_test.py --model_dir ... --splits Test1

Notes
-----
* Uses api.inference.CrabEmotionPredictor (loads final_ssl.pt / final_text.pt /
  final_ser.pt from model_dir).
* Does NOT currently handle LoRA-adapter-only ckpts (🅒-full) — those need
  PeftModel.from_pretrained wrapping, out of scope here.
* Same reporting shape as scripts/eval_scheme1_on_msp_test.py so scheme1
  vs scheme1-XLMR-FullFT numbers are directly comparable.
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


def run_split(predictor, split_name, run_tag, batch_size,
              ssl_type, text_model_path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"eval_{run_tag}_msp_{split_name}_{ts}"

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
               reinit=True,
               config={"step": "post_hoc_eval",
                       "run_tag": run_tag,
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

    rep_txt = classification_report(y_true, y_pred,
                                    labels=list(range(len(CLASSES))),
                                    target_names=CLASSES, digits=4,
                                    zero_division=0)
    wandb.log({"test/classification_report":
               wandb.Html(f"<pre>{rep_txt}</pre>")})
    wandb.finish()

    return {"split": split_name, "n": len(y_true), "macro_f1": macro_f1,
            "weighted_f1": weighted_f1, "uar": uar, "accuracy": acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="Directory containing final_ssl.pt / final_text.pt / final_ser.pt")
    p.add_argument("--ssl_type", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--text_model_path", default="FacebookAI/xlm-roberta-large")
    p.add_argument("--run_tag", default="crab",
                   help="Short tag used in wandb run names (e.g. scheme1_xlmr_fullft)")
    p.add_argument("--splits", nargs="+", default=["Test1", "Test2"],
                   choices=["Test1", "Test2"])
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    print(f"Loading Crab from {args.model_dir} ...", flush=True)
    print(f"  ssl_type       : {args.ssl_type}", flush=True)
    print(f"  text_model_path: {args.text_model_path}", flush=True)
    predictor = CrabEmotionPredictor(
        model_dir=args.model_dir,
        ssl_type=args.ssl_type,
        text_model_path=args.text_model_path,
    )

    summary = []
    for split in args.splits:
        summary.append(run_split(predictor, split, args.run_tag,
                                 args.batch_size,
                                 args.ssl_type, args.text_model_path))

    print(f"\n\n=== SUMMARY ({args.run_tag} on MSP-Podcast) ===")
    print(f"{'split':<8} {'n':>6} {'macro-F1':>9} {'wF1':>7} {'UAR':>7} {'acc':>7}")
    for s in summary:
        print(f"{s['split']:<8} {s['n']:>6} {s['macro_f1']:>9.4f} "
              f"{s['weighted_f1']:>7.4f} {s['uar']:>7.4f} {s['accuracy']:>7.4f}")


if __name__ == "__main__":
    main()
