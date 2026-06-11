"""Okeke 4 類 SER:分語言評估(用驗證過的 OkekeSER 推論封裝)。

對任一模型目錄(含 audio/text LoRA adapter + final_ser.pt + train_norm_stat.pkl),
在 okeke_bilingual_4class.csv 的 Test split 上按 Language(EN/ZH)分開算
WAR / UAR / macro-F1 + 各類 P/R/F1 + 混淆矩陣。

用法:
  .venv/bin/python scripts/eval_okeke4_per_lang.py --model_dir experiments/okeke_bilingual_4class --langs ZH EN
  .venv/bin/python scripts/eval_okeke4_per_lang.py --model_dir experiments/okeke_msp_4class      --langs ZH
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

CRAB = Path("/home/brant/Project/SAILER_test/Crab")
sys.path.insert(0, str(CRAB))
from api.okeke_infer import OkekeSER, CLASSES  # noqa: E402

DF = CRAB / "data" / "okeke_bilingual_4class.csv"


def true_idx(r):
    for i, c in enumerate(CLASSES):
        if str(r[c]) == "1":
            return i
    return -1


def load_rows(lang, cap):
    rows = []
    with open(DF, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["Split_Set"] == "Test" and r.get("Language") == lang:
                rows.append(r)
    if cap and len(rows) > cap:
        # deterministic subsample (seed 42), keep class balance roughly via stride
        import random
        random.seed(42)
        rows = random.sample(rows, cap)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--langs", nargs="+", default=["ZH"])
    ap.add_argument("--cap", type=int, default=0, help="max samples per language (0=all)")
    args = ap.parse_args()

    print(f"\n=== 載入模型 {args.model_dir} ===", flush=True)
    t0 = time.time()
    ser = OkekeSER(args.model_dir)
    print(f"  載入 {time.time()-t0:.1f}s, device={ser.device}", flush=True)

    for lang in args.langs:
        rows = load_rows(lang, args.cap)
        if not rows:
            print(f"\n[{lang}] 無 test 樣本,略過"); continue
        print(f"\n=== [{lang}] {len(rows)} 筆 test 推論中 ===", flush=True)
        y_true, y_pred = [], []
        t1 = time.time()
        for n, r in enumerate(rows, 1):
            out = ser.predict(r["FileName"], r.get("Text", "") or "")
            y_true.append(true_idx(r))
            y_pred.append(CLASSES.index(out["label"]))
            if n % 400 == 0:
                print(f"  [{n}/{len(rows)}] {time.time()-t1:.0f}s", flush=True)
        war = accuracy_score(y_true, y_pred)
        uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
        mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"\n>>> [{lang}] model={Path(args.model_dir).name}  "
              f"WAR {war:.4f}  UAR {uar:.4f}  macroF1 {mf1:.4f}  (n={len(rows)})")
        print(classification_report(y_true, y_pred, target_names=CLASSES,
                                    digits=4, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
        print("混淆矩陣 (列=真, 行=預測) 順序", CLASSES)
        for i, c in enumerate(CLASSES):
            print(f"  {c:>8}", cm[i].tolist())


if __name__ == "__main__":
    main()
