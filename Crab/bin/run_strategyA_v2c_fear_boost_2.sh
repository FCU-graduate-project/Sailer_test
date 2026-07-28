#!/usr/bin/env bash
# Strategy A LoRA v2c — Fear boost middle-ground (1.5 skipped, jumped to 2.0)
#
# Ancestry:
#   v2  (fear_boost 3.0) → crashed ep1 macroF1 0.31 (over-predicted Unconfident)
#   v2b (fear_boost 1.0) → converged, best 0.6710 @ ep6, TEST 0.6718 ⭐
#   v2c (fear_boost 2.0) → this run — test the middle
#
# Hypothesis:
#   - v2b Unconfident F1 = 0.51 (recall 0.45) is conservative
#   - Fear boost 2.0 → EmotionTalk-Unconfident ~7.5% per batch
#     (vs 3.0 = ~11% [crash], 1.0 = ~3% [conservative])
#   - Expected: Unconfident recall 0.45 → 0.55-0.60, macroF1 +0.01-0.03
#   - Risk: could still over-predict Unconfident if 2.0 is still too aggressive
#
# ONLY change vs v2b: --fear_boost_ratio 1.0 → 2.0
# Everything else identical (fresh init, same CSV, same LoRA config).

set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_v2.csv
if [ ! -f "$CSV" ]; then
  echo "ERROR: $CSV missing" >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_v2c_fear_boost_2

"$VENV" bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json ./data/bilingual_class_weights.json \
  --wav_base_dir "" \
  --zh_source_balanced \
  --fear_boost_source EmotionTalk \
  --fear_boost_ratio 2.0 \
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
  --run_name "strategyA_v2c_fear_boost_2_${TS}"

echo "=== Strategy A v2c exited $? ==="
