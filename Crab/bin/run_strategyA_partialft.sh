#!/usr/bin/env bash
# Strategy A — PARTIAL FT fallback.
#   Freezes base encoders, unfreezes only the top N=2 transformer layers per
#   encoder. Middle ground between LoRA (cheap, ~6M trainable) and full FT
#   (expensive, ~860M trainable, fragile multilingual forgetting).
#
# Why this exists:
#   If run_strategyA_fullft.sh OOMs or is unbearably slow, run this instead.
#   ~70M trainable, fits comfortably on 24GB, ~15 hr training.
#
# Knobs:
#   --unfreeze_last_n 2  : try 2 first; bump to 3-4 if dev metrics flatline
#   --use_amp            : still on (free win)
#   (no --use_grad_ckpt — VRAM is fine without it, saves wall-clock)
set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_strategyA.csv
if [ ! -f "$CSV" ]; then
  echo "❌ $CSV missing — run scripts/build_bilingual_train_csv.py first." >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_partialft

"$VENV" bin/train_crab_lora.py \
  --ft_mode partial_ft \
  --unfreeze_last_n 2 \
  --use_amp \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json ./data/bilingual_class_weights.json \
  --wav_base_dir "" \
  --language_balanced \
  --classes_list Excited Unconfident Neutral_3Class \
  --batch_size 16 --accumulation_steps 4 --num_workers 3 \
  --epochs 10 \
  --lr 2e-4 --encoder_lr 5e-5 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "strategyA_partialft_${TS}"

echo "=== Strategy A PARTIAL FT exited with code $? ==="
