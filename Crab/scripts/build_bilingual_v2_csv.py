"""
Build bilingual_v2.csv: EN (MSP) + ZH (EmotionTalk + CNSCED + NNIME v3) + Source column.

New column vs v1:
    Source: MSP | EmotionTalk | CNSCED | NNIME  (for 3-layer sampler)

Input:
    data/bilingual_strategyA.csv       (v1: EN 77,467 + ZH 14,612 EmotionTalk = 92,079)
    data/cnsced_crab_format.csv        (10,120)
    data/nnime_crab_format.csv         (4,300)

Output:
    data/bilingual_v2.csv
    columns: FileName, Text, Split_Set, Excited, Unconfident, Neutral_3Class, Language, Source
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path("/home/brant/Project/SAILER_test/Crab/data")

V1 = DATA_DIR / "bilingual_strategyA.csv"
CNSCED = DATA_DIR / "cnsced_crab_format.csv"
NNIME = DATA_DIR / "nnime_crab_format.csv"
OUT = DATA_DIR / "bilingual_v2.csv"


def load_v1_with_source():
    """v1: EN=MSP, ZH=EmotionTalk. Assign Source accordingly."""
    df = pd.read_csv(V1)
    df["Source"] = df["Language"].map({"EN": "MSP", "ZH": "EmotionTalk"})
    return df


def load_zh_with_source(path: Path, source: str):
    df = pd.read_csv(path)
    df["Source"] = source
    return df


def main():
    print("=== Loading ===")
    v1 = load_v1_with_source()
    print(f"  v1: {len(v1)} rows (EN {(v1.Language=='EN').sum()} / ZH {(v1.Language=='ZH').sum()})")

    cnsced = load_zh_with_source(CNSCED, "CNSCED")
    print(f"  CNSCED: {len(cnsced)} rows")

    nnime = load_zh_with_source(NNIME, "NNIME")
    print(f"  NNIME: {len(nnime)} rows")

    # concat
    cols = ["FileName", "Text", "Split_Set", "Excited", "Unconfident",
            "Neutral_3Class", "Language", "Source"]
    v2 = pd.concat([v1[cols], cnsced[cols], nnime[cols]], ignore_index=True)
    print(f"\n=== v2 total: {len(v2)} rows ===")

    # write
    v2.to_csv(OUT, index=False)
    print(f"[write] {OUT}")

    # summary
    print("\n=== Language × Split ===")
    print(v2.groupby(["Language", "Split_Set"]).size().unstack(fill_value=0))

    print("\n=== Source × Split ===")
    print(v2.groupby(["Source", "Split_Set"]).size().unstack(fill_value=0))

    print("\n=== Train: Source × class ===")
    train = v2[v2.Split_Set == "Train"].copy()
    train["class"] = train.apply(
        lambda r: "Excited" if r.Excited else ("Unconfident" if r.Unconfident else "Neutral"),
        axis=1,
    )
    print(train.groupby(["Source", "class"]).size().unstack(fill_value=0))

    print("\n=== Total ZH train contrib per Source ===")
    zh_train = train[train.Language == "ZH"]
    print(zh_train.groupby("Source").size())

    # per-language balance summary
    print("\n=== Overall sampling target (3-layer sampler expectation) ===")
    total_train = len(train)
    n_en_train = (train.Language == "EN").sum()
    n_zh_train = (train.Language == "ZH").sum()
    print(f"  EN train: {n_en_train} → target sample mass 50%")
    print(f"  ZH train: {n_zh_train} → target sample mass 50%")
    for src in ["EmotionTalk", "CNSCED", "NNIME"]:
        n = (train.Source == src).sum()
        print(f"    - {src}: {n} rows → target 1/3 of 50% = 16.67% of each batch")


if __name__ == "__main__":
    main()
