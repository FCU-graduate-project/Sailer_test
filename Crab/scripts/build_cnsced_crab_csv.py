"""
Build CNSCED → Crab format CSV (scheme1 3-class + transcript).

Input:
    data/cnsced_transcripts.csv       (from transcribe_cnsced.py)
    datasets/CNSCED/{train,val,test}/  (raw wav)

Output:
    data/cnsced_crab_format.csv
    columns: FileName, Text, Split_Set, Excited, Unconfident, Neutral_3Class, Language

Scheme1 mapping (strict single-emotion, drop multi/angry/only-aroused):
    0 (Neutral)            -> Neutral_3Class=1
    H (Happy) | B (Surprise) [single, no A] -> Excited=1
    S (Sad)  | F (Fear)     [single, no A]  -> Unconfident=1
    A (Angry) present       -> DROP
    W only (no H/B/S/F/0)   -> DROP  (aroused without emotion)
    Multi-emotion           -> DROP  (except W as arousal axis)

Split remap: train -> Train, val -> Development, test -> Test
    (matches EmotionTalk convention exactly)

Filename convention:
    {F|M}{speaker4}-{utt4}-{emotion_code}.wav
    e.g. F0001-0008-A2_S1_W2.wav
"""
import csv
import re
from pathlib import Path
from collections import Counter

import pandas as pd

CNSCED_ROOT = Path("/home/brant/Project/SAILER_test/datasets/CNSCED")
IN_CSV = Path("/home/brant/Project/SAILER_test/Crab/data/cnsced_transcripts.csv")
OUT_CSV = Path("/home/brant/Project/SAILER_test/Crab/data/cnsced_crab_format.csv")

SPLIT_MAP = {"train": "Train", "val": "Development", "test": "Test"}


def parse_code(rel_path: str):
    """
    Parse emotion code from rel_path like 'train/F0001-0008-A2_S1_W2.wav'.
    Returns:
        letters: set of emotion letters present (excluding W handled separately)
        has_A: bool (angry present)
        has_W_only: bool (only aroused, no other letter)
        code_str: raw code string for logging
    """
    stem = rel_path.split("/")[-1].replace(".wav", "")
    parts = stem.split("-")
    if len(parts) < 3:
        return set(), False, False, "?"
    code = parts[-1]
    # Handle duplicate marker like "0(1)"
    code = re.sub(r"\(\d+\)$", "", code)

    if code == "0":
        return {"0"}, False, False, code

    toks = code.split("_")
    letters = set()
    for t in toks:
        m = re.match(r"([A-Z])", t)
        if m:
            letters.add(m.group(1))

    has_A = "A" in letters
    non_arousal = letters - {"W"}
    has_W_only = (letters == {"W"}) or (len(non_arousal) == 0)
    return letters, has_A, has_W_only, code


def scheme1_label(letters: set, has_A: bool, has_W_only: bool):
    """
    Returns (excited, unconfident, neutral) one-hot, or None if drop.
    Strict: single non-arousal emotion required.
    """
    if letters == {"0"}:
        return (0, 0, 1), "Neutral"
    if has_A:
        return None, "DROP_ANGRY"
    if has_W_only:
        return None, "DROP_W_ONLY"

    non_arousal = letters - {"W"}
    if len(non_arousal) > 1:
        return None, "DROP_MULTI"
    if len(non_arousal) == 0:
        return None, "DROP_EMPTY"

    L = next(iter(non_arousal))
    if L in ("H", "B"):
        return (1, 0, 0), "Excited"
    if L in ("S", "F"):
        return (0, 1, 0), "Unconfident"
    return None, f"DROP_UNKNOWN_{L}"


def main():
    print(f"[load] {IN_CSV}", flush=True)
    df = pd.read_csv(IN_CSV)
    print(f"       {len(df)} rows", flush=True)

    # drop empty / error transcripts (Whisper failed or silent audio)
    n_before = len(df)
    df = df[df["transcript"].notna()].copy()
    df["transcript"] = df["transcript"].astype(str).str.strip()
    df = df[df["transcript"] != ""].copy()
    df = df[df["transcript"] != "<ERROR>"].copy()
    print(f"[filter] dropped {n_before - len(df)} empty/error transcripts", flush=True)

    # apply scheme1 mapping
    rows_out = []
    reason_counter = Counter()
    for _, r in df.iterrows():
        letters, has_A, has_W_only, code = parse_code(r["rel_path"])
        result, reason = scheme1_label(letters, has_A, has_W_only)
        reason_counter[reason] += 1
        if result is None:
            continue
        exc, unc, neu = result
        abs_wav = str(CNSCED_ROOT / r["rel_path"])
        split_out = SPLIT_MAP[r["split"]]
        rows_out.append({
            "FileName": abs_wav,
            "Text": r["transcript"],
            "Split_Set": split_out,
            "Excited": exc,
            "Unconfident": unc,
            "Neutral_3Class": neu,
            "Language": "ZH",
        })

    print(f"\n=== Mapping reasons ===", flush=True)
    for reason, n in reason_counter.most_common():
        print(f"  {reason:20s} {n:>6d}", flush=True)

    # write out
    out_df = pd.DataFrame(rows_out, columns=[
        "FileName", "Text", "Split_Set", "Excited", "Unconfident",
        "Neutral_3Class", "Language",
    ])
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV}  {len(out_df)} rows", flush=True)

    # per-split, per-class summary
    print("\n=== Per split, per scheme1 class ===", flush=True)
    for split in ["Train", "Development", "Test"]:
        sub = out_df[out_df.Split_Set == split]
        e = int((sub["Excited"] == 1).sum())
        u = int((sub["Unconfident"] == 1).sum())
        n = int((sub["Neutral_3Class"] == 1).sum())
        print(f"  {split:12s}  Excited={e:>5}  Unconfident={u:>5}  Neutral={n:>5}  total={len(sub):>5}", flush=True)

    print(f"\n=== Sanity: sample 5 output rows ===", flush=True)
    for _, r in out_df.sample(min(5, len(out_df)), random_state=42).iterrows():
        cls = "Excited" if r.Excited else ("Unconfident" if r.Unconfident else "Neutral")
        wav_short = r.FileName.split("/")[-1]
        text_short = str(r.Text)[:40]
        print(f"  {r.Split_Set:12s}  {cls:12s}  {wav_short:35s}  '{text_short}'", flush=True)


if __name__ == "__main__":
    main()
