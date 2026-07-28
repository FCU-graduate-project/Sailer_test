#!/usr/bin/env bash
# NNIME LoRA — Stage 2 fine-tune from v1 Strategy A LoRA warm-start
# 用意:把 v1 (XLM-R + XLS-R, 42k EN+ZH LoRA) 當「XLM-R+XLS-R 版 scheme1」用
#      避免 Full FT 168k EN 的 ~24 hr 成本,直接繼承 v1 已學到的:
#        - LoRA q,v adapter (audio+text) — 3-class emotion features
#        - ser_model head — 3-class classifier
#
# 對比:
#   N2 (scheme1 WavLM+RoBERTa warmstart): test 0.5028 — RoBERTa 廢中文 → Unconfident F1 0.32
#   本 run (v1 XLM-R+XLS-R warmstart): 預期 XLM-R 讀中文正常 → Unconfident F1 應該 +
#   v2b (fresh 53k mix): test 0.5543 — baseline
#
# Setup:
#   - Encoders: XLM-R-L + XLS-R-300M (v1 相同)
#   - Warmstart: v1 LoRA adapters (text + audio) via --lora_warmstart
#                + v1 final_ser.pt via --warm_start_ser
#   - Continue LoRA training: LR 略降到 head 5e-5 / enc 2e-5 (v1 原本 2e-4/1e-4)
#     防止 catastrophic forget v1 學到的 EN 表示
#   - Data: 純 NNIME (Train 3,054 / Dev 608 / Test 638)
#   - epochs 10 早停 patience 5 (3k 小 corpus,不需要 15)

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

V1=./experiments/strategyA_xlsr_xlmr_lora
if [ ! -d "$V1/text_lora_adapter" ] || [ ! -d "$V1/audio_lora_adapter" ] || [ ! -f "$V1/final_ser.pt" ]; then
  echo "ERROR: v1 checkpoint dirs missing in $V1" >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/nnime_v1_warmstart_lora

"$VENV" bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path "$V1" \
  --lora_warmstart \
  --warm_start_ser \
  --df_path "$CSV" \
  --weights_json "$WEIGHTS" \
  --wav_base_dir "" \
  --classes_list Excited Unconfident Neutral_3Class \
  --batch_size 16 --accumulation_steps 4 --num_workers 3 \
  --epochs 10 \
  --lr 5e-5 --encoder_lr 2e-5 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "nnime_v1_warmstart_lora_${TS}"

echo "=== NNIME v1-warmstart LoRA exited $? ==="
