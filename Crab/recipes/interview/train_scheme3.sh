#!/bin/bash

# Training Script for Interview 3-Class System (Scheme 3: Stress-Aware)

# Activate python environment
VENV_PYTHON="/home/brant/Project/SAILER_test/Crab/.venv/bin/python"

# Define Paths
WAV_DIR="/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios/"
CSV_PATH="/home/brant/Project/SAILER_test/Crab/data/msp2_interview_scheme3.csv"
SAVE_DIR="/home/brant/Project/SAILER_test/Crab/experiments/interview_scheme3"

# Create output directory
mkdir -p ${SAVE_DIR}

# Ensure we are in the correct working directory
cd /home/brant/Project/SAILER_test/Crab


echo "=========================================================="
echo "Starting Training: Interview 3-Class (Scheme 3)"
echo "Data Path: ${CSV_PATH}"
echo "Output Dir: ${SAVE_DIR}"
echo "=========================================================="

# Start training
${VENV_PYTHON} bin/train_crab.py \
    --wav_base_dir ${WAV_DIR} \
    --df_path ${CSV_PATH} \
    --model_path ${SAVE_DIR} \
    --classes_list Excited Unconfident Neutral_3Class \
    --ssl_type microsoft/wavlm-large \
    --text_model_path roberta-large \
    --batch_size 4 \
    --accumulation_steps 16 \
    --epochs 20 \
    --lr 1e-5 \
    --fusion_hidden_dim 512 \
    --constrastive_loss \
    --project_name SAILER_CRAB_MSP \
    --run_name Interview_Scheme3 \
    > ${SAVE_DIR}/train_$(date +"%Y%m%d_%H%M%S").log 2>&1
