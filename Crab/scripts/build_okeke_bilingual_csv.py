"""
Merge MSP-Podcast (EN) + EmotionTalk (ZH) into a single Okeke 4-class bilingual
training CSV — the data feed for run_okeke_bilingual_train.sh (XLS-R + XLM-R + LoRA,
--language_balanced).

Sources (both already 4-class: Angry / Happy / Neutral / Anxious):
  EN: data/okeke_msp_4class.csv               (absolute FileName, Train/Development/Test)
  ZH: data/emotiontalk_okeke4_crab_format.csv (relative FileName → joined with Audio16k)

Output columns: FileName (ABSOLUTE), Text, Split_Set, Language, Angry, Happy, Neutral, Anxious
  Absolute FileName lets one CSV reference two wav roots; train passes --wav_base_dir "".

MSP train ratio control (default 2.5:1 — matches strategyA's measured train ratio
EN 30000 : ZH 11744 = 2.55:1):
  Read ZH train total live → MSP train target = round(ratio × ZH_train) → subsample
  MSP train STRATIFIED per-class (keeps EN train class-balanced). Dev/Test untouched.
  --language_balanced still enforces 50:50 EN:ZH per batch at train time; this cap
  just keeps the upsampling gentle (not fighting a 5:1+ raw imbalance).

Sanity emitted: per-split×language, per-class×language (train), FileName existence
spot-check, duplicate FileName across sources. Also writes inverse-frequency class
weights on the MERGED train split.

Usage:
    python scripts/build_okeke_bilingual_csv.py                 # default ratio 2.5
    python scripts/build_okeke_bilingual_csv.py --target_ratio 2.5 --seed 42
"""
from pathlib import Path
import argparse
import json
from collections import Counter

import pandas as pd

CRAB = Path("/home/brant/Project/SAILER_test/Crab")
MSP_CSV = CRAB / "data" / "okeke_msp_4class.csv"
ET_CSV  = CRAB / "data" / "emotiontalk_okeke4_crab_format.csv"
ET_WAV  = Path("/home/brant/Project/SAILER_test/datasets/emotiontalk/Audio16k")

OUT       = CRAB / "data" / "okeke_bilingual_4class.csv"
OUT_WJSON = CRAB / "data" / "okeke_bilingual_4class_weights.json"

CLASSES = ["Angry", "Happy", "Neutral", "Anxious"]


def normalize_msp(df: pd.DataFrame) -> pd.DataFrame:
    """okeke_msp_4class.csv already has absolute FileName + 4-class one-hot."""
    df = df.copy()
    keep = ["FileName", "Text", "Split_Set"] + CLASSES
    df = df[keep]
    df["Language"] = "EN"
    df["Text"] = df["Text"].fillna("").astype(str)
    for c in CLASSES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def normalize_emotiontalk(df: pd.DataFrame) -> pd.DataFrame:
    """ET CSV has relative FileName under Audio16k → make absolute."""
    df = df.copy()
    keep = ["FileName", "Text", "Split_Set"] + CLASSES
    df = df[keep]
    df["FileName"] = df["FileName"].apply(lambda f: str(ET_WAV / f))
    df["Language"] = "ZH"
    df["Text"] = df["Text"].fillna("").astype(str)
    for c in CLASSES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def class_of(row):
    for c in CLASSES:
        if row[c] == 1:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_ratio", type=float, default=2.5,
                    help="MSP train : ZH train ratio (default 2.5, matches strategyA 2.55:1)")
    ap.add_argument("--en_dev", action="store_true",
                    help="keep English dev rows (DEFAULT: ZH-only dev, so early-stop / best-checkpoint "
                         "selection optimizes CHINESE macro-F1, not the EN-dominated combined dev)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not ET_CSV.exists():
        raise SystemExit(f"ERROR: {ET_CSV} not found — run build_emotiontalk_okeke4_csv.py first.")

    print(f"Reading MSP {MSP_CSV} ...")
    msp = normalize_msp(pd.read_csv(MSP_CSV, low_memory=False))
    print(f"Reading EmotionTalk {ET_CSV} ...")
    et  = normalize_emotiontalk(pd.read_csv(ET_CSV))

    print("MSP raw split counts:", dict(Counter(msp.Split_Set)))
    print("ET  raw split counts:", dict(Counter(et.Split_Set)))

    # keep canonical splits
    msp = msp[msp.Split_Set.isin(["Train", "Development", "Test"])].copy()
    et  = et[et.Split_Set.isin(["Train", "Development", "Test"])].copy()

    # #1 fix: 早停/選 best 以「中文 dev」為準 → 丟掉英文 dev(EN train + EN test 保留)
    if not args.en_dev:
        before = len(msp)
        msp = msp[msp.Split_Set != "Development"].copy()
        print(f"ZH-only dev: 丟掉 {before - len(msp)} 筆 EN Development "
              f"→ run_eval(dev) 變純中文 macro-F1,選 best/早停以中文為準")

    # ── MSP train stratified subsample to hit target ratio ──
    zh_train = et[et.Split_Set == "Train"]
    zh_train_total = len(zh_train)
    msp_train_target = round(args.target_ratio * zh_train_total)
    per_class_cap = max(1, round(msp_train_target / len(CLASSES)))
    print(f"\nZH train total = {zh_train_total}  →  target EN train ≈ {msp_train_target} "
          f"(ratio {args.target_ratio}:1)  →  per-class cap = {per_class_cap}")

    msp_train = msp[msp.Split_Set == "Train"].copy()
    msp_train["_cls"] = msp_train.apply(class_of, axis=1)
    kept = []
    for c in CLASSES:
        sub = msp_train[msp_train["_cls"] == c]
        if len(sub) > per_class_cap:
            sub = sub.sample(n=per_class_cap, random_state=args.seed)
        kept.append(sub)
        print(f"  EN train {c:>8}: {len(msp_train[msp_train['_cls']==c]):>6} → kept {len(sub):>6}")
    msp_train_capped = pd.concat(kept, ignore_index=True).drop(columns=["_cls"])
    msp = pd.concat([msp_train_capped, msp[msp.Split_Set != "Train"]], ignore_index=True)

    merged = pd.concat([msp, et], ignore_index=True)
    print(f"\nMerged rows: {len(merged)}")

    # ── sanity ──
    label_sum = merged[CLASSES].sum(axis=1)
    bad = int((label_sum != 1).sum())
    if bad:
        print(f"⚠️  {bad} rows lack exactly one positive label — dropping")
        merged = merged[label_sum == 1].reset_index(drop=True)

    dup = int(merged.FileName.duplicated().sum())
    if dup:
        print(f"⚠️  {dup} duplicate FileName entries across sources")

    spot = merged.sample(n=min(200, len(merged)), random_state=args.seed)
    missing = sum(1 for f in spot.FileName if not Path(f).exists())
    print(f"Existence spot-check (200 random rows): {missing} missing")

    print("\nSplit × Language:")
    print(merged.groupby(["Split_Set", "Language"]).size().unstack(fill_value=0))
    print("\nClass × Language (Train only):")
    train = merged[merged.Split_Set == "Train"]
    for lang in ["EN", "ZH"]:
        sub = train[train.Language == lang]
        cnt = {c: int(sub[c].sum()) for c in CLASSES}
        print(f"  {lang}: {cnt}  total={len(sub)}")
    en_tr = len(train[train.Language == "EN"]); zh_tr = len(train[train.Language == "ZH"])
    print(f"  → train EN:ZH = {en_tr}:{zh_tr} = {en_tr/zh_tr:.2f}:1" if zh_tr else "")

    # write merged CSV
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"\n✅ wrote {len(merged)} rows → {OUT}")

    # inverse-frequency class weights on MERGED train (both languages)
    train_counts = {c: int(train[c].sum()) for c in CLASSES}
    total = sum(train_counts.values()) or 1
    n = len(CLASSES)
    weights = {c: (total / (n * cnt) if cnt else 0.0) for c, cnt in train_counts.items()}
    OUT_WJSON.write_text(json.dumps({"class_weight": weights}, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"✅ wrote class weights → {OUT_WJSON}")
    print("Merged train class counts / weights:")
    for c in CLASSES:
        print(f"  {c:>8}: {train_counts[c]:>6}  weight={weights[c]:.3f}")


if __name__ == "__main__":
    main()
