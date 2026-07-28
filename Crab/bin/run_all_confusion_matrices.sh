#!/usr/bin/env bash
# Generate per-language confusion matrices for all 4 models × relevant splits.
# Outputs PNG + CSV + TXT under each model's <model_dir>/confusion_<split>/
#
# Coverage (6 figures total):
#   1. scheme1   on MSP Test1      (EN baseline,    28000)
#   2. scheme1   on EmotionTalk    (ZH zero-shot,   1447)
#   3. Hybrid B  on EmotionTalk    (ZH proper,      1447)
#   4. Hybrid B  on MSP Test1      (EN forgot?,    28000)
#   5. Strategy A on MSP Test1     (EN proper,     28000)  — via bilingual CSV, Language=EN
#   6. Strategy A on EmotionTalk   (ZH proper,      1447)  — via bilingual CSV, Language=ZH
set -euo pipefail
cd /home/brant/Project/SAILER_test/Crab
VENV=.venv/bin/python
SCRIPT=scripts/confusion_matrix_per_language.py

echo "===== 1. scheme1 on MSP Test1 (EN baseline) ====="
$VENV $SCRIPT \
    --model_dir experiments/interview_scheme1 \
    --df_path data/msp2_interview_scheme1.csv --split Test1 \
    --wav_base_dir /home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios \
    --tag scheme1_MSP_Test1

echo "===== 2. scheme1 on EmotionTalk (ZH zero-shot) ====="
$VENV $SCRIPT \
    --model_dir experiments/interview_scheme1 \
    --df_path data/emotiontalk_crab_format.csv --split Test \
    --wav_base_dir /home/brant/Project/SAILER_test/datasets/emotiontalk/Audio16k \
    --tag scheme1_ZH_zeroshot

echo "===== 3. Hybrid B on EmotionTalk (ZH proper) ====="
$VENV $SCRIPT \
    --model_dir experiments/emotiontalk_hybridB_lora \
    --df_path data/emotiontalk_crab_format.csv --split Test \
    --ssl_type microsoft/wavlm-large \
    --text_model FacebookAI/xlm-roberta-large \
    --wav_base_dir /home/brant/Project/SAILER_test/datasets/emotiontalk/Audio16k \
    --tag hybridB_ZH

echo "===== 4. Hybrid B on MSP Test1 (EN — forgotten?) ====="
$VENV $SCRIPT \
    --model_dir experiments/emotiontalk_hybridB_lora \
    --df_path data/msp2_interview_scheme1.csv --split Test1 \
    --ssl_type microsoft/wavlm-large \
    --text_model FacebookAI/xlm-roberta-large \
    --wav_base_dir /home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios \
    --tag hybridB_MSP_Test1

echo "===== 5+6. Strategy A per-language on bilingual test ====="
$VENV $SCRIPT \
    --model_dir experiments/strategyA_xlsr_xlmr_lora \
    --df_path data/bilingual_strategyA.csv --split Test \
    --ssl_type facebook/wav2vec2-xls-r-300m \
    --text_model FacebookAI/xlm-roberta-large \
    --languages EN ZH \
    --tag strategyA

echo "=== ALL 6 confusion matrices done ==="
