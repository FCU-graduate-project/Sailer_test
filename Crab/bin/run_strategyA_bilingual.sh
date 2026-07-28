#!/usr/bin/env bash
# Step ③ — Strategy A (friend's proposal):
#   XLS-R-300M (replaces WavLM) + XLM-R-Large (replaces RoBERTa), LoRA on both.
#   Trained on bilingual mix: MSP-Podcast (EN) + EmotionTalk (ZH), 3-class scheme1.
#
# Key differences vs Hybrid B (②):
#   • Audio encoder fully swapped → no scheme1 WavLM warm-start (would be useless)
#   • Bilingual co-training with WeightedRandomSampler (50:50 EN:ZH per batch)
#   • No ser warm-start (different audio encoder output distribution)
#
# Pre-flight: must run scripts/build_bilingual_train_csv.py first to produce
#             data/bilingual_strategyA.csv.
#
# Test split inside this run:
#   "Test" = MSP Test1 (EN) ∪ EmotionTalk Test (ZH)   — mixed-language test macroF1
#   MSP Test2 is excluded from this run; eval separately via a small script (TODO).
set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_strategyA.csv
if [ ! -f "$CSV" ]; then
  echo "❌ $CSV missing — run scripts/build_bilingual_train_csv.py first." >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_xlsr_xlmr_lora

# --wav_base_dir "" because FileName in the merged CSV is absolute (two roots).
# --pre_trained_path /tmp/__no_warmstart__ so the WavLM warm-start branch
#   silently skips (no final_ssl.pt there) and XLS-R uses HF pretrained weights.

"$VENV" bin/train_crab_lora.py \
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
  --run_name "strategyA_xlsr_xlmr_bilingual_${TS}"

echo "=== Strategy A training exited with code $? ==="
