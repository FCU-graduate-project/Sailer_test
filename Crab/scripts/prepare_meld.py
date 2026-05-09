import os
import pandas as pd
import subprocess
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = "/home/brant/Project/SAILER_test/datasets/MELD/MELD.Raw"
OUTPUT_DIR = "/home/brant/Project/SAILER_test/datasets/MELD/processed"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audios")

os.makedirs(AUDIO_DIR, exist_ok=True)

def extract_audio(input_file, output_file):
    """Extract audio from mp4 to wav (16kHz, mono)"""
    if os.path.exists(output_file):
        return True
    
    cmd = [
        "ffmpeg", "-i", input_file,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_file, "-y", "-loglevel", "error"
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def prepare_split(split_name, csv_file, video_dir):
    logger.info(f"Processing split: {split_name}")
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file, encoding='latin1') # MELD CSVs often use latin1
    df['split'] = split_name
    
    # MELD CSV has 'Dialogue_ID' and 'Utterance_ID'
    # Filenames are usually dia{Dialogue_ID}_utt{Utterance_ID}.mp4
    df['FileName'] = df.apply(lambda x: f"dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}.mp4", axis=1)
    
    # Target wav filenames
    df['WavName'] = df['FileName'].str.replace('.mp4', '.wav')
    
    # Process audio
    split_audio_dir = os.path.join(AUDIO_DIR, split_name)
    os.makedirs(split_audio_dir, exist_ok=True)
    
    success_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split_name} audio"):
        video_path = os.path.join(video_dir, row['FileName'])
        audio_path = os.path.join(split_audio_dir, row['WavName'])
        
        if os.path.exists(video_path):
            if extract_audio(video_path, audio_path):
                success_count += 1
        else:
            # logger.warning(f"Video not found: {video_path}")
            pass
            
    logger.info(f"Successfully extracted {success_count}/{len(df)} audio files for {split_name}")
    return df

def main():
    # Define paths
    # Note: train_sent_emo.csv might be in the root after extraction
    train_csv = os.path.join(BASE_DIR, "train_sent_emo.csv")
    if not os.path.exists(train_csv):
        # Check if it's named differently or inside train_splits
        # In some versions it's inside the tar
        train_csv = os.path.join(BASE_DIR, "train_splits", "train_sent_emo.csv")
    
    dev_csv = os.path.join(BASE_DIR, "dev_sent_emo.csv")
    test_csv = os.path.join(BASE_DIR, "test_sent_emo.csv")
    
    train_video = os.path.join(BASE_DIR, "train_splits")
    dev_video = os.path.join(BASE_DIR, "dev_splits_complete")
    test_video = os.path.join(BASE_DIR, "output_repeated_splits_test")
    
    train_df = prepare_split("train", train_csv, train_video)
    dev_df = prepare_split("dev", dev_csv, dev_video)
    test_df = prepare_split("test", test_csv, test_video)
    
    # Merge and save
    dfs = [df for df in [train_df, dev_df, test_df] if df is not None]
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        output_csv = os.path.join(OUTPUT_DIR, "meld_processed.csv")
        final_df.to_csv(output_csv, index=False)
        logger.info(f"Final processed CSV saved to: {output_csv}")
        logger.info(f"Total samples: {len(final_df)}")
    else:
        logger.error("No data processed!")

if __name__ == "__main__":
    main()
