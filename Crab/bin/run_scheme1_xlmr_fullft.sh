#!/usr/bin/env bash
# scheme1-XLMR-Full FT — 純 encoder swap ablation
# 目的:對比原 scheme1(WavLM+RoBERTa Full FT)= 0.6720 dev
#      這個(XLS-R+XLM-R Full FT)= ?
#      若 ~0.63-0.67 → encoder swap 幾乎沒 cost
#      若 ~0.55 → encoder swap 吃很大
#
# 對比 🅒-full(XLS-R+XLM-R LoRA)= 0.5405 dev
#      這個(XLS-R+XLM-R Full FT)= ?
#      若 明顯高於 🅒-full → Full FT capacity 有價值
#      若 差不多 → LoRA 已足夠,Full FT 不必要
#
# ⚠️ 進程保護:setsid + nohup 雙保險,防 SIGHUP 再犧牲
# ⚠️ VRAM 22 GB 峰,必開 bf16 + grad_ckpt

set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/msp2_interview_scheme1.csv
if [ ! -f "$CSV" ]; then
  echo "ERROR: $CSV missing" >&2
  exit 1
fi

WEIGHTS=./data/msp_scheme1_class_weights.json
if [ ! -f "$WEIGHTS" ]; then
  echo "ERROR: $WEIGHTS missing" >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/scheme1_xlmr_fullft

"$VENV" bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json "$WEIGHTS" \
  --wav_base_dir /home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios \
  --classes_list Excited Unconfident Neutral_3Class \
  --ft_mode full_ft \
  --batch_size 16 --accumulation_steps 4 --num_workers 2 \
  --epochs 8 \
  --lr 1e-4 --encoder_lr 1e-5 \
  --use_amp \
  --use_grad_ckpt \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "scheme1_xlmr_fullft_${TS}"

echo "=== scheme1_xlmr Full FT exited $? ==="
