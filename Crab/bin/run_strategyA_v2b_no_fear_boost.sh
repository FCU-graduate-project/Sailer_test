#!/usr/bin/env bash
# Strategy A LoRA v2b — Retry after v2 (fear_boost 3.0) crashed at epoch 1
#
# v2 result (2026-07-01 17:32-20:04):
#   epoch 0 dev macroF1 = 0.5357 (★ new best)
#   epoch 1 dev macroF1 = 0.3071 (-0.23, model over-predicts Unconfident)
#   Hypothesis: fear_boost 3.0 too aggressive — EmotionTalk Unconfident samples
#   became ~11% of each batch (28.7% of EmotionTalk portion × 39.7% ZH mass).
#   Model learned "output Unconfident wins" → macroF1 collapsed.
#
# v2b changes only ONE thing: --fear_boost_ratio 3.0 → 1.0 (disable boost).
# Everything else identical to v2, cleanly tests the fear-boost hypothesis.
#
# If v2b converges normally (dev macroF1 > 0.60 by epoch 3) → hypothesis confirmed.

set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_v2.csv
if [ ! -f "$CSV" ]; then
  echo "ERROR: $CSV missing" >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_v2b_no_fear_boost

"$VENV" bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json ./data/bilingual_class_weights.json \
  --wav_base_dir "" \
  --zh_source_balanced \
  --fear_boost_source EmotionTalk \
  --fear_boost_ratio 1.0 \
  --classes_list Excited Unconfident Neutral_3Class \
  --batch_size 16 --accumulation_steps 4 --num_workers 3 \
  --epochs 10 \
  --lr 2e-4 --encoder_lr 1e-4 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "strategyA_v2b_no_fear_boost_${TS}"

echo "=== Strategy A v2b exited $? ==="
