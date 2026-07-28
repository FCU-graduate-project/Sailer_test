#!/usr/bin/env bash
# Strategy A LoRA v2 — Fresh training with expanded ZH data (2026-07-01)
#
# Same architecture + same hyperparams as v1 (Strategy A LoRA), the ONLY differences:
#   • Training CSV: bilingual_v2.csv (EN 30k + EmotionTalk 12k + CNSCED 10k + NNIME 4.3k)
#   • Sampler: 3-layer (--zh_source_balanced) instead of 2-layer (--language_balanced)
#       - Layer 1: EN 50 vs ZH 50 by Language
#       - Layer 2: ZH internal — EmotionTalk / CNSCED / NNIME evenly (1/3 each)
#   • Fear boost: EmotionTalk Unconfident samples ×3 (only Emo has real fear+sad mix)
#
# Fresh start (NO warmstart from v1) so paper attribution stays clean:
#   "single-variable test: only training data changed, everything else identical to v1"
#
# See BILINGUAL_FINETUNE_PLAN.md v3.13 §8.1 for data details.

set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_v2.csv
if [ ! -f "$CSV" ]; then
  echo "ERROR: $CSV missing" >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_v2_bilingual_expanded

"$VENV" bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json ./data/bilingual_class_weights.json \
  --wav_base_dir "" \
  --zh_source_balanced \
  --fear_boost_source EmotionTalk \
  --fear_boost_ratio 3.0 \
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
  --run_name "strategyA_v2_bilingual_expanded_${TS}"

echo "=== Strategy A v2 exited $? ==="
