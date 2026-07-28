#!/usr/bin/env bash
set -euo pipefail
cd /home/brant/Project/SAILER_test/Crab
VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
"$VENV" bin/train_crab_lora.py \
  --ssl_type microsoft/wavlm-large \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path ./experiments/interview_scheme1 \
  --df_path ./data/chsims_crab_format.csv \
  --weights_json ./data/chsims_class_weights.json \
  --wav_base_dir /home/brant/Project/SAILER_test/datasets/chsims_v2s/ch-simsv2s/Audio \
  --classes_list Negative WeaklyNegative Neutral WeaklyPositive Positive \
  --batch_size 32 --accumulation_steps 4 --epochs 15 --lr 1e-3 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --contrastive_weight 0.0 --grad_clip 1.0 --early_stop_patience 5 --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path ./experiments/chsims_wavlm_xlmr_lora_noCon \
  --project_name Crab_Bilingual_ZH \
  --run_name "wavlm_xlmr_lora_noCon_${TS}"
echo "=== exited $? ==="
