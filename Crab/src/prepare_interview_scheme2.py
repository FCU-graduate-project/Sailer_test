import pandas as pd
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Map MSP-Podcast to Interview 3-Class (Scheme 2: V-A-D Dimensional Thresholding)")
    parser.add_argument("--input_csv", type=str, default="/home/brant/Project/SAILER_test/Crab/data/msp2_processed_labels.csv")
    parser.add_argument("--output_csv", type=str, default="/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme2.csv")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    original_len = len(df)

    # Verify VAD columns exist
    if not all(col in df.columns for col in ['EmoDom', 'EmoAct', 'EmoVal']):
        logger.error("V-A-D columns (EmoDom, EmoAct, EmoVal) not found in the dataset.")
        return

    # Filter out records without valid VAD scores
    df = df.dropna(subset=['EmoDom', 'EmoAct', 'EmoVal']).copy()

    # Initialize probability columns (hard labels for this scheme)
    df['Excited'] = 0.0
    df['Unconfident'] = 0.0
    df['Neutral_3Class'] = 0.0
    df['Interview_Class'] = None

    # Condition 1: Excited / Confident (High Dominance, High Arousal)
    mask_excited = (df['EmoDom'] > 4.5) & (df['EmoAct'] > 4.5)
    df.loc[mask_excited, 'Interview_Class'] = 'Excited'
    df.loc[mask_excited, 'Excited'] = 1.0

    # Condition 2: Unconfident (Low Dominance)
    mask_unconfident = (df['EmoDom'] < 3.5)
    df.loc[mask_unconfident, 'Interview_Class'] = 'Unconfident'
    df.loc[mask_unconfident, 'Unconfident'] = 1.0

    # Condition 3: Neutral (Comfort zone - Moderate Dominance and Arousal)
    mask_neutral = (df['EmoDom'] >= 3.5) & (df['EmoDom'] <= 4.5) & \
                   (df['EmoAct'] >= 3.5) & (df['EmoAct'] <= 4.5)
    df.loc[mask_neutral, 'Interview_Class'] = 'Neutral_3Class'
    df.loc[mask_neutral, 'Neutral_3Class'] = 1.0

    # Drop unclassified samples (ambiguous edge cases)
    df_filtered = df.dropna(subset=['Interview_Class']).copy()
    filtered_len = len(df_filtered)
    logger.info(f"Filtered by V-A-D strict thresholds. Kept {filtered_len} / {original_len} samples.")

    # Add Scheme type for future merging
    df_filtered['Scheme_Type'] = 'scheme2'

    # Distribution logging
    dist = df_filtered['Interview_Class'].value_counts()
    logger.info("New Class Distribution:")
    for cls, count in dist.items():
        logger.info(f"  {cls}: {count} ({count/len(df_filtered)*100:.2f}%)")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_filtered.to_csv(args.output_csv, index=False)
    logger.info(f"Saved mapped dataset to {args.output_csv}")

if __name__ == "__main__":
    main()
