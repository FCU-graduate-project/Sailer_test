#!/usr/bin/env bash
# NNIME-only LoRA — "老師字面意思:純 NNIME 單獨訓一次看台灣口音表現"
#
# Setup:
#   - Data: NNIME only (Train 3,054 / Dev 608 / Test 638)
#   - Encoders: XLM-R-L + XLS-R-300M (same as Strategy A LoRA family)
#   - FT: LoRA q,v r=16 α=32 (locked sweet spot)
#   - Warmstart: fresh (no scheme1 / no v2b) — paper 乾淨歸因「3k 台灣口音單獨能學到什麼」
#   - Sampler: default (no --language_balanced, no --zh_source_balanced)
#     — 只有 1 語言 1 corpus,不需要 balancing;class imbalance 靠 --weights_json 抵
#   - Class weights: from data/nnime_class_weights.json (inverse freq)
#       Excited 1.26 / Unconfident 1.62 / Neutral 0.63
#
# 預期風險:
#   - 3,054 train / (bs 16 × accum 4) = ~48 gradient steps/epoch
#   - 10 epoch = ~480 total steps (LoRA 通常需 500-2000) → 可能 under-train
#   - 若 dev best 太低 → 加 epochs 到 20 重跑
#
# vs v2b NNIME test baseline:
#   v2b(bilingual full v2)測 NNIME test = 0.5543 macroF1
#   純 NNIME train 有沒有機會超過(overfitting to domain)or 就是不夠

set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/nnime_crab_format.csv
if [ ! -f "$CSV" ]; then
  echo "ERROR: $CSV missing" >&2
  exit 1
fi

WEIGHTS=./data/nnime_class_weights.json
if [ ! -f "$WEIGHTS" ]; then
  echo "ERROR: $WEIGHTS missing" >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/nnime_only_lora

"$VENV" bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json "$WEIGHTS" \
  --wav_base_dir "" \
  --classes_list Excited Unconfident Neutral_3Class \
  --batch_size 16 --accumulation_steps 4 --num_workers 3 \
  --epochs 15 \
  --lr 2e-4 --encoder_lr 1e-4 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "nnime_only_lora_${TS}"

echo "=== NNIME-only LoRA exited $? ==="
