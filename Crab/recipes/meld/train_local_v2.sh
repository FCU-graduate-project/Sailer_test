#!/bin/bash
# MELD Baseline Training Script (Local)

PROJ_DIR="/home/brant/Project/SAILER_test"
CRAB_DIR="${PROJ_DIR}/Crab"
SAVE_DIR="${CRAB_DIR}/experiments/meld/baseline"

# Data Paths (Processed)
DF_PATH="${PROJ_DIR}/datasets/MELD/processed/meld_processed.csv"
WAV_DIR="${PROJ_DIR}/datasets/MELD/processed/audios"

# Create save directory
mkdir -p "$SAVE_DIR"

# Activate environment
source ${CRAB_DIR}/.venv/bin/activate

# Run training
python ${CRAB_DIR}/bin/train_meld.py \
  --df_path "$DF_PATH" \
  --wav_base_dir "$WAV_DIR" \
  --model_path "$SAVE_DIR" \
  --ssl_type "microsoft/wavlm-large" \
  --text_model_path "roberta-large" \
  --batch_size 4 \
  --accumulation_steps 32 \
  --epochs 20 \
  --lr 1e-5 \
  --seed 42
