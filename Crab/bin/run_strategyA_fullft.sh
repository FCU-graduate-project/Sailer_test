#!/usr/bin/env bash
# Strategy A — FULL FT variant.
#   Same architecture as run_strategyA_bilingual.sh (XLS-R-300M + XLM-R-Large)
#   but trains ALL base encoder params instead of LoRA.
#
# VRAM strategy (must squeeze ~30 GB into 24 GB on 4090):
#   • bf16 mixed precision (--use_amp)
#   • gradient checkpointing on both encoders (--use_grad_ckpt)
#   • per-step batch = 4 (bs=16 accum=4 → effective 16, per-step 4)
#     ← matches Strategy A LoRA's batch structure so contrastive has 2-4 positive pairs/batch
#       (per-step bs=1 makes MultiPosConLoss return 0 — DEAD contrastive)
#   • lower encoder LR (full FT needs gentle stepping or it forgets multilingual)
#
# Smoke-tested 2026-06-02 21:13:
#   • bs=4 per-step → 17.7/24 GB (~6 GB headroom) ✅ + ~3.1 it/s
#   • bs=8 per-step → 24.07/24.6 GB (60 MiB margin) ❌ would OOM within 7 hr run
#
# Expectation:
#   • ~7.5 hr training (8 epochs × ~56 min)
#   • macroF1 may be -0.05 to +0.05 vs LoRA — coin flip, but at least clean comparison
#   • If OOM in first 3 batches → kill and run bin/run_strategyA_partialft.sh
set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_strategyA.csv
if [ ! -f "$CSV" ]; then
  echo "❌ $CSV missing — run scripts/build_bilingual_train_csv.py first." >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_fullft

"$VENV" bin/train_crab_lora.py \
  --ft_mode full_ft \
  --use_amp --use_grad_ckpt \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json ./data/bilingual_class_weights.json \
  --wav_base_dir "" \
  --language_balanced \
  --classes_list Excited Unconfident Neutral_3Class \
  --batch_size 16 --accumulation_steps 4 --num_workers 3 \
  --epochs 8 \
  --lr 1e-4 --encoder_lr 1e-5 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 4 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "strategyA_fullft_${TS}"

echo "=== Strategy A FULL FT exited with code $? ==="
