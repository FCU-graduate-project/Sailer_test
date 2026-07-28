"""
P2: Build Crab-compatible training CSV from CH-SIMS v2(s) meta.csv.

Input:  datasets/chsims_v2s/ch-simsv2s/meta.csv
Output: Crab/data/chsims_v2s_train.csv

Schema produced:
  wav_path      absolute path to extracted wav
  text          Chinese transcript
  label         multimodal sentiment value (-1.0 .. 1.0)   [regression target]
  label_class   5-class id (0..4) derived from label       [classification target]
  label_T       text-only sentiment                        [aux supervision]
  label_A       audio-only sentiment                       [aux supervision]
  annotation    Negative / Weakly Negative / Neutral / Weakly Positive / Positive
  split         train / valid / test (from meta `mode` column)

The 5-class mapping for label_class follows CH-SIMS-v2 convention:
   -1.0  → 0  Negative
   -0.5  → 1  Weakly Negative
    0.0  → 2  Neutral
    0.5  → 3  Weakly Positive
    1.0  → 4  Positive
"""
from pathlib import Path
import csv

ROOT  = Path("/home/brant/Project/SAILER_test/datasets/chsims_v2s/ch-simsv2s")
META  = ROOT / "meta.csv"
AUDIO = ROOT / "Audio"
OUT   = Path("/home/brant/Project/SAILER_test/Crab/data/chsims_v2s_train.csv")

LABEL_TO_CLASS = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
CLASS_NAMES    = ["Negative", "Weakly Negative", "Neutral", "Weakly Positive", "Positive"]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with META.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"meta.csv rows: {len(rows)}")

    out_rows = []
    skipped  = []
    class_hist = {i: 0 for i in range(5)}
    split_hist = {}

    for r in rows:
        video_id = r["video_id"]
        clip_id  = r["clip_id"]
        wav = AUDIO / video_id / f"{clip_id}.wav"
        if not wav.exists() or wav.stat().st_size < 1024:
            skipped.append(f"{video_id}/{clip_id}")
            continue

        label_val = float(r["label"])
        # snap to nearest of {-1, -0.5, 0, 0.5, 1} just in case of float drift
        snap = min(LABEL_TO_CLASS.keys(), key=lambda x: abs(x - label_val))
        cls  = LABEL_TO_CLASS[snap]

        out_rows.append({
            "wav_path":    str(wav),
            "text":        r["text"],
            "label":       f"{label_val:.4f}",
            "label_class": cls,
            "label_T":     r["label_T"],
            "label_A":     r["label_A"],
            "annotation":  r["annotation"],
            "split":       r["mode"],
        })
        class_hist[cls] += 1
        split_hist[r["mode"]] = split_hist.get(r["mode"], 0) + 1

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["wav_path", "text", "label", "label_class",
                        "label_T", "label_A", "annotation", "split"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\n✅ Wrote {len(out_rows)} rows → {OUT}")
    if skipped:
        print(f"⚠ Skipped {len(skipped)} (missing wav). First 5: {skipped[:5]}")

    print(f"\nClass distribution (5-class sentiment):")
    total = sum(class_hist.values())
    for cls, n in class_hist.items():
        print(f"  {cls} ({CLASS_NAMES[cls]:>17}): {n:>5}   {n/total*100:>5.1f}%")

    print(f"\nSplit distribution:")
    for split, n in sorted(split_hist.items()):
        print(f"  {split:>6}: {n}")


if __name__ == "__main__":
    main()
