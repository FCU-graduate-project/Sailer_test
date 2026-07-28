#!/usr/bin/env bash
# C5 — same target as proven Strategy A LoRA baseline (q,v) but rank+alpha doubled.
#
# Why this exists:
#   C1 / C1b proved that EXPANDING the LoRA target (q,v -> q,k,v,o) is wrong:
#     C1  (enc_lr 1e-4): dev plateau 0.5442 vs LoRA 0.6552 -> -0.111
#     C1b (enc_lr 5e-5): dev oscillate 0.18-0.42, best 0.4213 -> -0.121 vs C1
#   The opposite axis was never tested: more capacity ON THE PROVEN TARGET.
#   C5 keeps target = q,v (sweet spot confirmed) and doubles LoRA rank only.
#   LoRA params on q,v alone go ~1.5M -> ~3.0M (NOT the 3.7x of C1/C1b).
#
# Why alpha also scales 32 -> 64:
#   Effective LoRA scale = alpha / rank. Baseline = 32/16 = 2.0.
#   To isolate the CAPACITY axis from the effective-update-magnitude axis,
#   scale alpha proportionally so 64/32 = 2.0 stays identical. Single-variable
#   test of rank (capacity), not effective LR.
#
# Expected (per plan v3.7 §10.1):
#   ~40% probability  q,v capacity is the bottleneck -> C5 >= 0.66 (beats LoRA)
#   ~40% probability  r=16 is already the ceiling      -> C5 ~ 0.65 +/- 0.005
#   ~20% probability  doubled update too strong         -> C5 microloss, retry with alpha=32
#
# If C5 is a tie or worse, the LoRA adapter is right-sized and further gains
# must come from a DIFFERENT axis: C3 (contrastive_weight), C7 (text_max_len),
# or C11 (warmstart from scheme1).
set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_strategyA.csv
if [ ! -f "$CSV" ]; then
  echo "ERROR: $CSV missing - run scripts/build_bilingual_train_csv.py first." >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_c5_rank32

"$VENV" bin/train_crab_lora.py \
  --ft_mode lora \
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
  --lr 2e-4 --encoder_lr 1e-4 \
  --lora_rank 32 --lora_alpha 64 --lora_dropout 0.1 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "strategyA_c5_rank32_${TS}"

echo "=== Strategy A C5 (LoRA q,v, rank=32, alpha=64) exited with code $? ==="
