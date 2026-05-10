import pandas as pd
import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Best thresholds from VAD grid search ──────────────────────────────────────
V_TH     = 4.5   # Valence lower bound for Excited
A_TH     = 4.5   # Arousal lower bound for Excited
D_HI_TH  = 4.5   # Dominance lower bound for Excited
D_LOW_TH = 3.8   # Dominance upper bound for Unconfident

# Integer label encoding (must match training script)
LABEL_MAP = {'Excited': 0, 'Unconfident': 1, 'Neutral_3Class': 2}


def main():
    parser = argparse.ArgumentParser(
        description="Map MSP-Podcast to Interview 3-Class (Scheme 2: V-A-D Grid-Search Thresholds)"
    )
    parser.add_argument("--input_csv",  type=str,
                        default="/home/brant/Project/SAILER_test/Crab/data/msp2_processed_labels.csv")
    parser.add_argument("--output_csv", type=str,
                        default="/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme2.csv")
    parser.add_argument("--output_weights_json", type=str,
                        default="/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme2_weights.json")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    original_len = len(df)
    logger.info(f"Original rows: {original_len:,}")

    # ── Step 1: drop rows with missing VAD ────────────────────────────────────
    df = df.dropna(subset=['EmoDom', 'EmoAct', 'EmoVal']).copy()

    # ── Step 2: drop Angry (label noise for interview context) ────────────────
    angry_mask = df['Angry'] == 1
    angry_count = angry_mask.sum()
    df = df[~angry_mask].copy()
    logger.info(f"Dropped Angry: {angry_count:,}  →  Working set: {len(df):,}")

    # ── Step 3: VAD mapping (mutually exclusive, priority: Excited first) ─────
    # Excited:        Arousal > 4.5 AND Dominance > 4.5 AND Valence > 4.5
    # Unconfident:    Dominance < 3.8  (guaranteed disjoint: 3.8 < 4.5)
    # Neutral_3Class: everything else
    mask_excited    = (df['EmoAct'] > A_TH) & (df['EmoDom'] > D_HI_TH) & (df['EmoVal'] > V_TH)
    mask_unconf     = (df['EmoDom'] < D_LOW_TH) & ~mask_excited
    mask_neutral    = ~mask_excited & ~mask_unconf

    df['Interview_Class'] = None
    df.loc[mask_excited, 'Interview_Class'] = 'Excited'
    df.loc[mask_unconf,  'Interview_Class'] = 'Unconfident'
    df.loc[mask_neutral, 'Interview_Class'] = 'Neutral_3Class'

    # Integer label for training
    df['label'] = df['Interview_Class'].map(LABEL_MAP)

    # Hard-label columns — names must match emolist used by load_cat_emo_label
    df['Excited']        = mask_excited.astype(float)
    df['Unconfident']    = mask_unconf.astype(float)
    df['Neutral_3Class'] = mask_neutral.astype(float)

    df['Scheme_Type'] = 'scheme2'

    # ── Step 4: class distribution ────────────────────────────────────────────
    dist = df['Interview_Class'].value_counts()
    total = len(df)
    logger.info("Class distribution:")
    for cls in ['Excited', 'Unconfident', 'Neutral_3Class']:
        n = dist.get(cls, 0)
        logger.info(f"  {cls:<16}: {n:7,}  ({n/total*100:.1f}%)")

    # ── Step 5: compute imbalance weights ─────────────────────────────────────
    # class_weight  → for CrossEntropyLoss(weight=...)
    # sample_weight → for WeightedRandomSampler
    #
    # Formula: w_c = (1 / n_c) * (total / n_classes), normalised so mean = 1
    n_classes = len(LABEL_MAP)
    class_counts = {cls: int(dist.get(cls, 1)) for cls in LABEL_MAP}
    class_weight = {
        cls: total / (n_classes * count)
        for cls, count in class_counts.items()
    }
    # Map to label-ordered list for torch: index = label integer
    class_weight_list = [class_weight[cls] for cls in sorted(LABEL_MAP, key=LABEL_MAP.get)]

    logger.info("Class weights (for CrossEntropyLoss):")
    for cls, w in class_weight.items():
        logger.info(f"  {cls:<16}: {w:.4f}")

    # Per-sample weight (WeightedRandomSampler)
    sample_weight_map = {cls: class_weight[cls] for cls in LABEL_MAP}
    df['sample_weight'] = df['Interview_Class'].map(sample_weight_map)

    # ── Step 6: save CSV ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Saved mapped dataset → {args.output_csv}  ({len(df):,} rows)")

    # ── Step 7: save weights JSON ─────────────────────────────────────────────
    weights_payload = {
        "thresholds": {
            "V_th": V_TH, "A_th": A_TH, "D_hi_th": D_HI_TH, "D_low_th": D_LOW_TH
        },
        "class_counts":      class_counts,
        "class_weight":      class_weight,
        "class_weight_list": class_weight_list,   # ordered by label int [0,1,2]
        "label_map":         LABEL_MAP,
    }
    os.makedirs(os.path.dirname(args.output_weights_json), exist_ok=True)
    with open(args.output_weights_json, 'w') as f:
        json.dump(weights_payload, f, indent=2)
    logger.info(f"Saved class weights → {args.output_weights_json}")

    # ── Usage hint ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training snippet:")
    logger.info("  import torch, json, pandas as pd")
    logger.info("  from torch.utils.data import WeightedRandomSampler")
    logger.info(f"  w = json.load(open('{args.output_weights_json}'))")
    logger.info("  # CrossEntropyLoss")
    logger.info("  cw = torch.tensor(w['class_weight_list'], dtype=torch.float)")
    logger.info("  criterion = torch.nn.CrossEntropyLoss(weight=cw.to(device))")
    logger.info("  # WeightedRandomSampler")
    logger.info(f"  df = pd.read_csv('{args.output_csv}')")
    logger.info("  sampler = WeightedRandomSampler(")
    logger.info("      weights=torch.tensor(df['sample_weight'].values, dtype=torch.float),")
    logger.info("      num_samples=len(df), replacement=True)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
