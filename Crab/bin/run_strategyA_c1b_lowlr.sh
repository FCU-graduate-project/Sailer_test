#!/usr/bin/env bash
# C1b — same LoRA target as C1 (q,k,v,o) but encoder_lr halved.
#
# Why this exists:
#   C1 (encoder_lr 1e-4, same as LoRA Strategy A) underperformed massively —
#   dev plateaued at 0.5442 @ epoch 4, vs LoRA Strategy A 0.6552 @ epoch 6.
#   Hypothesis: with 3.7× more LoRA params (q,v 1.5M → q,k,v,o 5.5M), the
#   same encoder_lr is effectively too aggressive — q/k/v/o all moving at
#   the same rate produces attention drift / noisy gradients.
#
# Fix:
#   encoder_lr 1e-4 → 5e-5 (half). Head LR unchanged.
#   Empirically, LoRA literature halves LR when target_modules count doubles.
#
# Expected:
#   If LR was the only issue → match or beat LoRA Strategy A 0.6552.
#   If LoRA target expansion itself is wrong → C1b stays near C1's 0.5442
#     (then we'd conclude q,v is the sweet spot).
set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_strategyA.csv
if [ ! -f "$CSV" ]; then
  echo "❌ $CSV missing — run scripts/build_bilingual_train_csv.py first." >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_c1b_lowlr

"$VENV" bin/train_crab_lora.py \
  --ft_mode lora \
  --lora_target_set expanded \
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
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "strategyA_c1b_lowlr_${TS}"

echo "=== Strategy A C1b (LoRA expanded, encoder_lr=5e-5) exited with code $? ==="
