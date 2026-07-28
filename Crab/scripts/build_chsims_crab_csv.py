"""
Reformat CH-SIMS v2(s) into the CSV layout that train_crab.py expects.

train_crab.py / load_cat_emo_label needs:
  FileName    relative wav path (joined with --wav_base_dir)
  Text        transcript
  Split_Set   "Train" / "Development" / "Test"
  <one-hot columns per class in --classes_list order>

CH-SIMS sentiment 5-class → one-hot columns:
  -1.0 → Negative
  -0.5 → WeaklyNegative
   0.0 → Neutral
   0.5 → WeaklyPositive
   1.0 → Positive

Also emits a class-weight JSON (inverse-frequency) for weighted CrossEntropy.
"""
from pathlib import Path
import csv
import json

SRC   = Path("/home/brant/Project/SAILER_test/Crab/data/chsims_v2s_train.csv")
OUT   = Path("/home/brant/Project/SAILER_test/Crab/data/chsims_crab_format.csv")
WJSON = Path("/home/brant/Project/SAILER_test/Crab/data/chsims_class_weights.json")
AUDIO_ROOT = Path("/home/brant/Project/SAILER_test/datasets/chsims_v2s/ch-simsv2s/Audio")

CLASSES = ["Negative", "WeaklyNegative", "Neutral", "WeaklyPositive", "Positive"]
LABEL_TO_CLASS = {-1.0: 0, -0.5: 1, 0.0: 2, 0.5: 3, 1.0: 4}
SPLIT_MAP = {"train": "Train", "valid": "Development", "test": "Test"}


def main():
    with SRC.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Source rows: {len(rows)}")

    out_rows = []
    class_counts = {c: 0 for c in CLASSES}
    split_counts = {}

    for r in rows:
        wav_path = Path(r["wav_path"])
        # FileName relative to AUDIO_ROOT, e.g. aqgy3_0001/0001.wav
        try:
            rel = wav_path.relative_to(AUDIO_ROOT)
        except ValueError:
            rel = Path(wav_path.parent.name) / wav_path.name
        file_name = str(rel)

        label_val = float(r["label"])
        snap = min(LABEL_TO_CLASS.keys(), key=lambda x: abs(x - label_val))
        cls_idx = LABEL_TO_CLASS[snap]

        onehot = {c: (1 if i == cls_idx else 0) for i, c in enumerate(CLASSES)}
        class_counts[CLASSES[cls_idx]] += 1

        split = SPLIT_MAP[r["split"]]
        split_counts[split] = split_counts.get(split, 0) + 1

        row = {
            "FileName":  file_name,
            "Text":      r["text"],
            "Split_Set": split,
            **onehot,
        }
        out_rows.append(row)

    fieldnames = ["FileName", "Text", "Split_Set"] + CLASSES
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"✅ Wrote {len(out_rows)} rows → {OUT}")

    # class weights (inverse frequency on TRAIN split only, like train_crab.py expects)
    train_counts = {c: 0 for c in CLASSES}
    for r in out_rows:
        if r["Split_Set"] == "Train":
            for c in CLASSES:
                if r[c] == 1:
                    train_counts[c] += 1
    total_train = sum(train_counts.values())
    n_cls = len(CLASSES)
    class_weight = {
        c: (total_train / (n_cls * cnt) if cnt > 0 else 0.0)
        for c, cnt in train_counts.items()
    }
    WJSON.write_text(json.dumps({"class_weight": class_weight}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Wrote class weights → {WJSON}")

    print("\nClass distribution (all splits):")
    total = sum(class_counts.values())
    for c in CLASSES:
        print(f"  {c:>15}: {class_counts[c]:>5}  {class_counts[c]/total*100:>5.1f}%   weight={class_weight[c]:.3f}")
    print("\nSplit distribution:")
    for s, n in sorted(split_counts.items()):
        print(f"  {s:>12}: {n}")


if __name__ == "__main__":
    main()
