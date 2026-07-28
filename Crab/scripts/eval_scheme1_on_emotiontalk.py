"""
Zero-shot baseline (Step ①): run scheme1 (English Crab) on EmotionTalk Chinese test split.

Answers the question: "現在中文情緒測得多差?" — provides the FLOOR baseline
that Hybrid B (②) and Strategy A (③) must beat.

⚠️ scheme1 uses RoBERTa-Large which has NO Chinese tokens → text side falls back
   to byte-level fragments (essentially noise). The remaining signal is from WavLM
   processing Chinese audio (already known to be conservative→Neutral per §9.2).
   Expected: low macroF1, predictions skewed toward Neutral.

Outputs to stdout:
  - per-class precision / recall / F1
  - confusion matrix
  - prediction distribution (mirrors §9.2's «怯» pattern)
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

CSV_PATH = CRAB / "data" / "emotiontalk_crab_format.csv"
WAV_BASE = Path("/home/brant/Project/SAILER_test/datasets/emotiontalk/Audio16k")
SCHEME1  = CRAB / "experiments" / "interview_scheme1"
BATCH    = 8

WANDB_PROJECT = "Crab_Bilingual_ZH"
WANDB_ENTITY  = "d1249119-feng-chia-university"


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"eval_scheme1_zeroshot_emotiontalk_{ts}"
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
               config={"step": "①_zero-shot_floor", "model": "scheme1",
                       "dataset": "EmotionTalk", "split": "Test",
                       "ssl_type": "microsoft/wavlm-large",
                       "text_model": "roberta-large (no Chinese tokens → byte fallback)"})

    print(f"Loading scheme1 from {SCHEME1} ...", flush=True)
    predictor = CrabEmotionPredictor(model_dir=str(SCHEME1))

    # ---- load EmotionTalk test split ----
    rows = [r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8"))
            if r["Split_Set"] == "Test"]
    print(f"Test samples: {len(rows)}", flush=True)

    # ground truth (argmax over one-hot)
    y_true = []
    for r in rows:
        y_true.append(next(i for i, c in enumerate(CLASSES) if r[c] == "1"))

    # ---- batched inference ----
    y_pred = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        audio_bytes_list, texts = [], []
        for r in chunk:
            audio_bytes_list.append((WAV_BASE / r["FileName"]).read_bytes())
            texts.append(r["Text"])

        results = predictor.predict_batch(audio_bytes_list, texts)
        for res in results:
            probs = [res[c] for c in CLASSES]
            y_pred.append(int(np.argmax(probs)))

        done = i + len(chunk)
        if done % (BATCH * 20) == 0 or done == len(rows):
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(rows) - done) / max(rate, 1e-6)
            print(f"  [{done}/{len(rows)}]  {rate:.1f}/s  eta={eta:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.1f}s", flush=True)

    # ---- metrics ----
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("sklearn missing, fallback to manual metrics", flush=True)
        return _manual_metrics(y_true, y_pred)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                target_names=CLASSES, digits=4, zero_division=0))

    print("=== Confusion Matrix (rows=true, cols=pred) ===")
    print("class order:", CLASSES)
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))))

    # macro / weighted F1 spelled out
    from sklearn.metrics import f1_score, accuracy_score, recall_score
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Headline ===")
    print(f"  macro-F1   : {macro_f1:.4f}")
    print(f"  weighted-F1: {weighted_f1:.4f}")
    print(f"  UAR        : {uar:.4f}")
    print(f"  accuracy   : {acc:.4f}")

    # prediction distribution — to see «怯/Neutral collapse» if it happens
    pred_dist = Counter(y_pred)
    true_dist = Counter(y_true)
    print("\n=== Prediction distribution vs ground truth ===")
    for i, c in enumerate(CLASSES):
        p_pct = 100 * pred_dist[i] / len(y_pred)
        t_pct = 100 * true_dist[i] / len(y_true)
        print(f"  {c:>16s}  pred {pred_dist[i]:>4} ({p_pct:5.1f}%)  | true {true_dist[i]:>4} ({t_pct:5.1f}%)")

    # ---- log to wandb ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    wandb.log({
        "test/macro_f1":    macro_f1,
        "test/weighted_f1": weighted_f1,
        "test/UAR":         uar,
        "test/accuracy":    acc,
        "test/n_samples":   len(y_true),
    })
    # per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                 labels=list(range(len(CLASSES))),
                                                 zero_division=0)
    for i, c in enumerate(CLASSES):
        wandb.log({f"test/f1_{c}": f[i], f"test/precision_{c}": p[i],
                   f"test/recall_{c}": r[i], f"test/support_{c}": int(s[i]),
                   f"test/pred_pct_{c}": 100 * pred_dist[i] / len(y_pred),
                   f"test/true_pct_{c}": 100 * true_dist[i] / len(y_true)})
    # confusion matrix as wandb plot
    wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
        probs=None, y_true=y_true, preds=y_pred, class_names=CLASSES)})
    # classification report as text
    rep_txt = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                    target_names=CLASSES, digits=4, zero_division=0)
    wandb.log({"test/classification_report": wandb.Html(f"<pre>{rep_txt}</pre>")})
    wandb.finish()


def _manual_metrics(y_true, y_pred):
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print("CM:\n", cm)
    f1s = []
    for c in range(n):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        f1s.append(f)
        print(f"  {CLASSES[c]}: P={p:.4f} R={r:.4f} F1={f:.4f}")
    print(f"macro-F1: {np.mean(f1s):.4f}")


if __name__ == "__main__":
    main()
