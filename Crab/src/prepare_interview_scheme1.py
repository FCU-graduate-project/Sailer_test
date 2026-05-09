import pandas as pd
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Map MSP-Podcast to Interview 3-Class (Scheme 1: Valence-Based Hard Mapping)")
    parser.add_argument("--input_csv", type=str, default="/home/brant/Project/SAILER_test/Crab/data/msp2_processed_labels.csv")
    parser.add_argument("--output_csv", type=str, default="/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme1.csv")
    args = parser.parse_args()

    logger.info(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    original_len = len(df)

    # Calculate original dominant class
    emo_cols = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']
    df['Original_Dom'] = df[emo_cols].idxmax(axis=1)

    # Scheme 1: Filter out Angry, Disgust, Contempt
    drop_emotions = ['Angry', 'Disgust', 'Contempt']
    df_filtered = df[~df['Original_Dom'].isin(drop_emotions)].copy()
    filtered_len = len(df_filtered)
    logger.info(f"Filtered out hostile/disgust emotions. Kept {filtered_len} / {original_len} samples.")

    # Map to 3 classes
    df_filtered['Excited'] = df_filtered['Happy'] + df_filtered['Surprise']
    df_filtered['Unconfident'] = df_filtered['Fear'] + df_filtered['Sad']
    df_filtered['Neutral_3Class'] = df_filtered['Neutral']

    # Normalize to get new probabilities
    new_cols = ['Excited', 'Unconfident', 'Neutral_3Class']
    row_sums = df_filtered[new_cols].sum(axis=1)
    
    # Avoid division by zero
    valid_mask = row_sums > 0
    df_filtered = df_filtered[valid_mask].copy()
    row_sums = row_sums[valid_mask]  # Re-align index
    
    for col in new_cols:
        df_filtered[col] = df_filtered[col] / row_sums.values

    # Determine new dominant class
    df_filtered['Interview_Class'] = df_filtered[new_cols].idxmax(axis=1)

    # Distribution logging
    dist = df_filtered['Interview_Class'].value_counts()
    logger.info("New Class Distribution:")
    for cls, count in dist.items():
        logger.info(f"  {cls}: {count} ({count/len(df_filtered)*100:.2f}%)")

    df_filtered['Scheme_Type'] = 'scheme1'
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_filtered.to_csv(args.output_csv, index=False)
    logger.info(f"Saved mapped dataset to {args.output_csv}")

if __name__ == "__main__":
    main()
