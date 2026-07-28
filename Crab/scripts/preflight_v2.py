"""
Preflight check for LoRA v2 (Strategy A LoRA + expanded ZH data).

Validates 8 categories before launch to ensure the experiment will demonstrate
that expanded data helps the model.

Run: .venv/bin/python scripts/preflight_v2.py
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# fixed seed for reproducible sample verifications
random.seed(42)
np.random.seed(42)

DATA_DIR = Path("/home/brant/Project/SAILER_test/Crab/data")
V1_CSV = DATA_DIR / "bilingual_strategyA.csv"
V2_CSV = DATA_DIR / "bilingual_v2.csv"
CW_JSON = DATA_DIR / "bilingual_class_weights.json"

FEAR_BOOST_RATIO = 3.0
FEAR_BOOST_SOURCE = "EmotionTalk"

ok_all = []


def check(name, condition, detail=""):
    status = "✅" if condition else "❌"
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))
    ok_all.append(condition)


def check_warn(name, condition, detail=""):
    status = "✅" if condition else "⚠️"
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))


def hdr(txt):
    print(f"\n{'='*60}\n{txt}\n{'='*60}")


# ============ 1. Data integrity ============
hdr("1️⃣  Data integrity")
v2 = pd.read_csv(V2_CSV)
print(f"  → v2 rows: {len(v2)}, columns: {list(v2.columns)}")

check("required columns present",
      set(["FileName","Text","Split_Set","Excited","Unconfident",
           "Neutral_3Class","Language","Source"]).issubset(v2.columns))
check("no NaN in critical columns",
      v2[["FileName","Split_Set","Language","Source"]].notna().all().all())
check("no duplicate FileName",
      v2["FileName"].is_unique,
      f"dupes={len(v2) - v2['FileName'].nunique()}")

# label one-hot check
one_hot_sum = (v2[["Excited","Unconfident","Neutral_3Class"]].sum(axis=1))
check("one-hot label valid (each row exactly one 1)",
      (one_hot_sum == 1).all(),
      f"bad_rows={((one_hot_sum != 1)).sum()}")

# text non-empty
text_ok = v2["Text"].notna() & (v2["Text"].astype(str).str.strip() != "")
check("text non-empty",
      text_ok.all(),
      f"empty_text_rows={(~text_ok).sum()}")

# random wav existence check (200 samples)
print("  → sampling 200 wav paths to check existence...")
sampled = v2.sample(200, random_state=42)
missing = sum(1 for p in sampled["FileName"] if not Path(p).exists())
check("random 200 wav files exist (0 missing)",
      missing == 0,
      f"missing={missing}")


# ============ 2. Split distribution ============
hdr("2️⃣  Split × Language × Source")
print(v2.groupby(["Split_Set", "Language", "Source"]).size().to_frame("n").to_string())


# ============ 3. Per-Split, per-class balance ============
hdr("3️⃣  Per-Split × class × Source")
for split in ["Train", "Development", "Test"]:
    sub = v2[v2.Split_Set == split].copy()
    sub["cls"] = sub.apply(
        lambda r: "Excited" if r.Excited else ("Unconfident" if r.Unconfident else "Neutral"),
        axis=1,
    )
    print(f"\n--- {split} ---")
    print(sub.groupby(["Source","cls"]).size().unstack(fill_value=0).to_string())


# ============ 4. Sampler simulation (critical) ============
hdr("4️⃣  Sampler simulation (10,000 draws, verify batch composition)")
train = v2[v2.Split_Set == "Train"].reset_index(drop=True).copy()
n_en_sources = train[train.Language == "EN"]["Source"].nunique()
n_zh_sources = train[train.Language == "ZH"]["Source"].nunique()
lang_counts = train["Language"].value_counts().to_dict()
en_src_counts = train[train.Language == "EN"]["Source"].value_counts().to_dict()
zh_src_counts = train[train.Language == "ZH"]["Source"].value_counts().to_dict()

def compute_weight(row):
    if row.Language == "EN":
        return 0.5 / n_en_sources / en_src_counts[row.Source]
    w = 0.5 / n_zh_sources / zh_src_counts[row.Source]
    if row.Source == FEAR_BOOST_SOURCE and row.Unconfident == 1:
        w *= FEAR_BOOST_RATIO
    return w

train["_w"] = train.apply(compute_weight, axis=1)
weights = train["_w"].values

# simulate 10k draws
sampled_idx = np.random.choice(len(train), size=10000, replace=True,
                                p=weights / weights.sum())
sampled = train.iloc[sampled_idx]

print(f"\n  Simulated 10,000 draws:")
lang_dist = sampled["Language"].value_counts(normalize=True) * 100
src_dist = sampled["Source"].value_counts(normalize=True) * 100
print(f"\n  Language ratio:  EN={lang_dist.get('EN', 0):.1f}% (target 50)  ZH={lang_dist.get('ZH', 0):.1f}% (target 50)")
print(f"  Source ratio in ZH:")
zh_only = sampled[sampled.Language == "ZH"]
if len(zh_only):
    zh_src_dist = zh_only["Source"].value_counts(normalize=True) * 100
    for src in ["EmotionTalk", "CNSCED", "NNIME"]:
        pct = zh_src_dist.get(src, 0)
        print(f"    {src}: {pct:.1f}% of ZH (target ~33.3% base, EmotionTalk boosted slightly by fear-boost)")

# fear boost verification
et_all = sampled[(sampled.Language == "ZH") & (sampled.Source == "EmotionTalk")]
if len(et_all):
    unc_pct = (et_all["Unconfident"] == 1).mean() * 100
    orig_unc_pct = train[(train.Language == "ZH") & (train.Source == "EmotionTalk")]["Unconfident"].mean() * 100
    print(f"\n  Fear boost check:")
    print(f"    EmotionTalk Unconfident in ORIGINAL data: {orig_unc_pct:.1f}%")
    print(f"    EmotionTalk Unconfident in SAMPLED data:  {unc_pct:.1f}%")
    print(f"    → boost effect: {unc_pct/orig_unc_pct:.2f}× (target ~{FEAR_BOOST_RATIO}×)")

check("sampler EN/ZH ratio in [45%, 55%]",
      45 < lang_dist.get("EN", 0) < 55,
      f"EN={lang_dist.get('EN', 0):.1f}%")


# ============ 5. v1 vs v2 config parity ============
hdr("5️⃣  v1 vs v2 config parity (paper clean attribution)")
v1_script = Path("/home/brant/Project/SAILER_test/Crab/bin/run_strategyA_bilingual.sh")
v2_script = Path("/home/brant/Project/SAILER_test/Crab/bin/run_strategyA_v2_bilingual_expanded.sh")

def extract_flags(path):
    txt = path.read_text()
    flags = {}
    for line in txt.split("\n"):
        line = line.strip().rstrip(" \\")
        if line.startswith("--"):
            parts = line.replace("--", "", 1).split(None, 1)
            k = parts[0]
            v = parts[1].strip() if len(parts) > 1 else "True"
            flags[k] = v
    return flags

v1_flags = extract_flags(v1_script)
v2_flags = extract_flags(v2_script)

parity_expected_same = ["ssl_type", "text_model_path", "classes_list", "batch_size",
                        "accumulation_steps", "num_workers", "epochs", "lr",
                        "encoder_lr", "lora_rank", "lora_alpha", "lora_dropout",
                        "contrastive_weight", "grad_clip",
                        "fusion_hidden_dim", "text_max_len"]

print("  Hyperparams that MUST be identical:")
for k in parity_expected_same:
    same = v1_flags.get(k) == v2_flags.get(k)
    check(f"    {k}: v1={v1_flags.get(k)!r} v2={v2_flags.get(k)!r}", same)

print("\n  Flags that SHOULD differ (v1→v2 upgrades):")
expected_diffs = {
    "df_path": ("./data/bilingual_strategyA.csv", "./data/bilingual_v2.csv"),
    "model_path": ("./experiments/strategyA_xlsr_xlmr_lora",
                   "./experiments/strategyA_v2_bilingual_expanded"),
    "language_balanced": ("True", None),   # v1 has it, v2 doesn't (uses zh_source_balanced instead)
    "zh_source_balanced": (None, "True"),
}
for k, (v1_expected, v2_expected) in expected_diffs.items():
    v1_ok = v1_flags.get(k) == v1_expected
    v2_ok = v2_flags.get(k) == v2_expected
    status = "✅" if (v1_ok and v2_ok) else "⚠️"
    print(f"    {status} {k}: v1={v1_flags.get(k)!r} (expected {v1_expected!r}), v2={v2_flags.get(k)!r} (expected {v2_expected!r})")


# ============ 6. Eval strategy — v1 test IS subset of v2 test ============
hdr("6️⃣  Eval comparability (v1 vs v2 test set overlap)")
v1 = pd.read_csv(V1_CSV)
v1_test = set(v1[v1.Split_Set == "Test"]["FileName"])
v2_test = set(v2[v2.Split_Set == "Test"]["FileName"])
overlap = v1_test & v2_test
print(f"  v1 test size: {len(v1_test)}")
print(f"  v2 test size: {len(v2_test)}")
print(f"  overlap: {len(overlap)} ({100*len(overlap)/max(len(v1_test),1):.1f}% of v1)")
print(f"  v2 only  (from CNSCED + NNIME test): {len(v2_test - v1_test)}")

check("v2 test is superset of v1 test",
      v1_test.issubset(v2_test),
      f"missing from v2: {len(v1_test - v2_test)}")


# ============ 7. Class weights JSON check ============
hdr("7️⃣  Class weights JSON")
if CW_JSON.exists():
    weights_json = json.loads(CW_JSON.read_text())
    print(f"  {CW_JSON.name}: {weights_json}")
    check_warn("class_weights_json exists", True,
               "reused from v1 setup (won't retrain per-source imbalance);"
               " sampler handles balance instead — OK")
else:
    check("bilingual_class_weights.json exists", False)


# ============ 8. Resources + runtime estimate ============
hdr("8️⃣  Resources + runtime estimate")

# GPU
try:
    import torch
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB")
except Exception as e:
    print(f"  GPU probe failed: {e}")

# disk
import shutil
free_gb = shutil.disk_usage("/home").free / 1e9
print(f"  Disk free: {free_gb:.0f} GB")
check("disk free ≥ 10 GB (for ckpt saves)",
      free_gb >= 10,
      f"{free_gb:.0f} GB")

# runtime estimate
n_train = (v2.Split_Set == "Train").sum()
per_step = 5.0  # est sec per step (bs=16 accum=4 → per-step batch=4)
epoch_sec = n_train / (16 * 4) * per_step  # very rough
print(f"  Estimated epoch time: ~{epoch_sec/60:.0f} min ({epoch_sec/3600:.1f} hr)")
print(f"  Estimated total (10 epoch): ~{10*epoch_sec/3600:.1f} hr")

# ============ Summary ============
hdr("Summary")
n_ok = sum(ok_all)
n_all = len(ok_all)
print(f"  Passed: {n_ok} / {n_all}")
if n_ok == n_all:
    print("\n  🟢 All checks passed — safe to launch")
    sys.exit(0)
else:
    print(f"\n  🔴 {n_all - n_ok} check(s) failed — investigate before launch")
    sys.exit(1)
