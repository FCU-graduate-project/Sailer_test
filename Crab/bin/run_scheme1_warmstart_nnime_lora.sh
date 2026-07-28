#!/usr/bin/env bash
# NNIME LoRA with scheme1 (WavLM + RoBERTa) warm-start
# 老師方向:「當時 Crab 一開始映射完 3 類的模型去做 finetune 在 NNIME 上」
#
# Setup:
#   - Encoders: scheme1 的原班 WavLM-L + RoBERTa-L (不是 XLS-R + XLM-R)
#   - Warm-start: 完整載入 scheme1 (final_ssl + final_text + final_ser)
#   - FT method: LoRA q,v r=16 α=32 (安全選項;3k 全 FT 保證 overfit 崩)
#   - Data: 純 NNIME (Train 3,054 / Dev 608 / Test 638)
#   - Class weights: nnime_class_weights.json (inverse freq)
#
# ⚠️ 已知限制:
#   - RoBERTa-L 無中文 vocab → 中文字被 byte-level BPE 拆成無意義 tokens
#   - 就算 warm-start scheme1 的 RoBERTa 權重,中文語意訊號仍極弱
#   - 預期 macroF1 天花板 0.45-0.55 (跑不贏 XLM-R-based 版本)
#   - 但比純 fresh NNIME LoRA (完全崩,0.30) 應該好
#
# 論文角度: "scheme1 (EN encoders) transferred to ZH NNIME = 邊界 case"
#           對比 Strategy A LoRA (XLS-R + XLM-R) 顯示中文 vocab 的必要性

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

SCHEME1=./experiments/interview_scheme1
if [ ! -f "$SCHEME1/final_ssl.pt" ] || [ ! -f "$SCHEME1/final_text.pt" ] || [ ! -f "$SCHEME1/final_ser.pt" ]; then
  echo "ERROR: scheme1 checkpoints missing in $SCHEME1" >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/nnime_scheme1_warmstart_lora

"$VENV" bin/train_crab_lora.py \
  --ssl_type microsoft/wavlm-large \
  --text_model_path roberta-large \
  --pre_trained_path "$SCHEME1" \
  --warm_start_ser \
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
  --run_name "nnime_scheme1_warmstart_lora_${TS}"

echo "=== NNIME scheme1-warmstart LoRA exited $? ==="
