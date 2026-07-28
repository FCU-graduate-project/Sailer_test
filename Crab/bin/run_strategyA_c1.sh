#!/usr/bin/env bash
# C1 — LoRA target_modules expansion (q,v → q,k,v + audio out_proj).
#
# Same arch / data / hyperparams as run_strategyA_bilingual.sh except:
#   --lora_target_set expanded
#     text  ["query", "value"]            → ["query", "key", "value"]
#     audio ["q_proj", "v_proj"]          → ["q_proj", "k_proj", "v_proj", "out_proj"]
#
# Why: Strategy A LoRA hit overall test 0.6531. Full FT (C14, 891M params, 148× more)
# only managed 0.6408 — overfit at epoch 2. The LoRA bottleneck isn't capacity, it's
# WHERE the capacity is placed: q,v only adapts queries and values, ignoring keys and
# output mixing. Expanding to q,k,v,o lets attention re-aim AND re-mix all 16 heads.
#
# Why text only gets +"key" (not "output.dense"): PEFT does suffix matching, and
# XLM-R has `dense` modules in both attention.output and intermediate (FFN). Adding
# "output.dense" or "dense" would balloon the param count and silently target FFN.
# Safest: just add "key" to text. Trainable params goes from ~6M → ~11M.
#
# Expected: +0.02-0.05 macroF1 across all three axes (overall / EN / ZH),
# biggest gain on EN Unconfident (currently 0.50, was scheme1 0.56)
# and ZH Excited (currently 0.60, biggest gap).
set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_strategyA.csv
if [ ! -f "$CSV" ]; then
  echo "❌ $CSV missing — run scripts/build_bilingual_train_csv.py first." >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_c1_qkvo

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
  --lr 2e-4 --encoder_lr 1e-4 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "strategyA_c1_qkvo_${TS}"

echo "=== Strategy A C1 (LoRA expanded) exited with code $? ==="
