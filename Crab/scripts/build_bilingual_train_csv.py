"""
Merge MSP-Podcast (EN) + EmotionTalk (ZH) into a single bilingual training CSV
for Strategy A (XLS-R + XLM-R + LoRA, bilingual co-training).

Output columns: FileName (ABSOLUTE path), Text, Split_Set, Language,
                Excited, Unconfident, Neutral_3Class

Why absolute FileName:
  train_crab_lora.py does `os.path.join(args.wav_base_dir, FileName)`.
  When FileName is absolute, the base dir is discarded — letting one CSV
  reference wavs in two different roots. We then pass --wav_base_dir "".

Split remap:
  MSP-Podcast: Train / Development / Test1 / Test2  →  Train / Development / Test (Test1) / dropped (Test2 reported separately)
  EmotionTalk: Train / Development / Test           →  Train / Development / Test

  Test1 (MSP) ∪ Test (EmotionTalk) form the bilingual "Test" split inside the run.
  Test2 is preserved in a SECOND CSV (test2_only.csv) for a post-hoc EN-only eval.

Labels:
  Both sources already carry hard scheme1 mapping (Excited / Unconfident / Neutral_3Class
  as 0/1). We force them to integer 0/1 to avoid pandas-read inconsistencies.

Sanity checks emitted:
  - per-split counts per language
  - per-class counts per language
  - FileName existence (random 200 sample)
  - duplicate FileName across the two sources
"""
from pathlib import Path
import argparse
import csv
import random
from collections import Counter

import pandas as pd

CRAB = Path("/home/brant/Project/SAILER_test/Crab")
MSP_CSV   = CRAB / "data" / "msp2_interview_scheme1.csv"
ET_CSV    = CRAB / "data" / "emotiontalk_crab_format.csv"
MSP_WAV   = Path("/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios")
ET_WAV    = Path("/home/brant/Project/SAILER_test/datasets/emotiontalk/Audio16k")

OUT       = CRAB / "data" / "bilingual_strategyA.csv"
OUT_TEST2 = CRAB / "data" / "bilingual_strategyA_msp_test2.csv"

CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]


def normalize_msp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["Scheme_Type"] == "scheme1"]
    keep = ["FileName", "Text", "Split_Set"] + CLASSES
    df = df[keep]
    df["FileName"] = df["FileName"].apply(lambda f: str(MSP_WAV / f))
    df["Language"] = "EN"
    df["Text"] = df["Text"].fillna("").astype(str)
    for c in CLASSES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def normalize_emotiontalk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keep = ["FileName", "Text", "Split_Set"] + CLASSES
    df = df[keep]
    df["FileName"] = df["FileName"].apply(lambda f: str(ET_WAV / f))
    df["Language"] = "ZH"
    df["Text"] = df["Text"].fillna("").astype(str)
    for c in CLASSES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msp_subsample_train", type=int, default=0,
                    help="if > 0, randomly sample this many MSP train rows "
                         "(to reduce 10:1 EN:ZH imbalance before sampler).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    print(f"Reading MSP {MSP_CSV} ...")
    msp = normalize_msp(pd.read_csv(MSP_CSV, low_memory=False))
    print(f"Reading EmotionTalk {ET_CSV} ...")
    et  = normalize_emotiontalk(pd.read_csv(ET_CSV))

    print("MSP raw split counts:", dict(Counter(msp.Split_Set)))
    print("ET  raw split counts:", dict(Counter(et.Split_Set)))

    # split remap
    msp_test2 = msp[msp.Split_Set == "Test2"].copy()
    msp = msp[msp.Split_Set.isin(["Train", "Development", "Test1"])].copy()
    msp["Split_Set"] = msp["Split_Set"].replace({"Test1": "Test"})

    et = et[et.Split_Set.isin(["Train", "Development", "Test"])].copy()

    # optional MSP train subsampling
    if args.msp_subsample_train > 0:
        msp_train = msp[msp.Split_Set == "Train"]
        if len(msp_train) > args.msp_subsample_train:
            msp_train = msp_train.sample(n=args.msp_subsample_train,
                                         random_state=args.seed)
            msp = pd.concat([msp_train,
                             msp[msp.Split_Set != "Train"]], ignore_index=True)
            print(f"MSP train subsampled to {args.msp_subsample_train}")

    merged = pd.concat([msp, et], ignore_index=True)
    print(f"\nMerged rows: {len(merged)}")

    # ── sanity ──
    # 1. label one-hot integrity (exactly one '1' per row)
    label_sum = merged[CLASSES].sum(axis=1)
    bad = (label_sum != 1).sum()
    if bad:
        print(f"⚠️  {bad} rows do NOT have exactly one positive label — fixing/dropping")
        merged = merged[label_sum == 1].reset_index(drop=True)

    # 2. dup FileName
    dup = merged.FileName.duplicated().sum()
    if dup:
        print(f"⚠️  {dup} duplicate FileName entries across sources")

    # 3. existence spot-check
    spot = merged.sample(n=min(200, len(merged)), random_state=args.seed)
    missing = sum(1 for f in spot.FileName if not Path(f).exists())
    print(f"Existence spot-check (200 random rows): {missing} missing")

    # 4. distribution table
    print("\nSplit × Language:")
    print(merged.groupby(["Split_Set", "Language"]).size().unstack(fill_value=0))
    print("\nClass × Language (Train only):")
    train = merged[merged.Split_Set == "Train"]
    for lang in ["EN", "ZH"]:
        sub = train[train.Language == lang]
        cnt = {c: int(sub[c].sum()) for c in CLASSES}
        print(f"  {lang}: {cnt}  total={len(sub)}")

    # write
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"\n✅ wrote {len(merged)} rows → {OUT}")

    msp_test2.to_csv(OUT_TEST2, index=False)
    print(f"✅ wrote MSP Test2 ({len(msp_test2)} rows) → {OUT_TEST2}")


if __name__ == "__main__":
    main()
