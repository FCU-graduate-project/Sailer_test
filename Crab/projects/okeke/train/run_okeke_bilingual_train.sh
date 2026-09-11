#!/usr/bin/env bash
# Okeke 雙語 4 類 SER LoRA 訓練(MSP-EN + EmotionTalk-ZH, strategyA 配方)
# ⚠️ 這會用 GPU。動工前:① 確認 GPU 空出來(清掉佔用的 ~6.8GB)② nvidia-smi 盯著,接近滿就 Ctrl-C/TaskStop。
# 資料已備:projects/okeke/data/okeke_bilingual_4class.csv + weights。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CRAB_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
cd "$CRAB_ROOT"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$PROJECT_ROOT/data/okeke_bilingual_4class.csv" \
  --weights_json "$PROJECT_ROOT/data/okeke_bilingual_4class_weights.json" \
  --wav_base_dir "" \
  --language_balanced \
  --classes_list Angry Happy Neutral Anxious \
  --ft_mode lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --batch_size 24 --accumulation_steps 2 \
  --epochs 10 --early_stop_patience 5 \
  --lr 2e-4 --encoder_lr 1e-4 \
  --contrastive_weight 2.0 --grad_clip 1.0 \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --use_amp --use_grad_ckpt \
  --num_workers 3 --eval_test \
  --model_path "$PROJECT_ROOT/models/okeke_bilingual_4class" \
  --project_name Crab_Okeke_4class --run_name okeke_bilingual_msp_et_lora

# ── 若想用英文 4 類模型暖啟(收斂更快,可選)──
#   把上面 --pre_trained_path 改成: "$PROJECT_ROOT/models/okeke_msp_4class"
# ── 若 OOM(餘量不足)──
#   --batch_size 8 --accumulation_steps 8   # 有效 batch 仍 64,峰值約砍半
