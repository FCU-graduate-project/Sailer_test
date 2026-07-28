#!/usr/bin/env bash
# N4-B — NNIME LoRA stage 2 with scheme1-XLMR-FullFT warm-start
#
# 純 encoder swap ablation(對照 N2):
#   N2:   WavLM + RoBERTa   (scheme1 原班)  → NNIME test 0.5028
#   N4-B: XLS-R + XLM-R    (scheme1-XLMR-FullFT) → ?
#   → 拆「encoder pair 的差」對 ZH downstream 的影響
#
# Setup:
#   - Encoders: XLS-R-300M + XLM-R-large(取代 N2 的 WavLM + RoBERTa)
#   - Warm-start: scheme1-XLMR-FullFT Ep 6 best(dev 0.6499)
#     載入 final_ssl.pt (1.26 GB) + final_text.pt (2.24 GB) + final_ser.pt
#   - FT method: LoRA q,v r=16 α=32(同 N2)
#   - Data: 純 NNIME (Train 3,054 / Dev 608 / Test 638) — 同 N2
#   - Class weights: nnime_class_weights.json — 同 N2
#   - Hyperparams: **完全同 N2**(head 2e-4 / enc 1e-4,不像 N3 降 25%)
#
# 論文 claim(如果 N4-B > N2):
#   「encoder swap(WavLM+RoBERTa → XLS-R+XLM-R)在 NNIME transfer stage 帶來 accuracy 提升,
#    補回 pretraining stage 的 -0.022 cost」
#
# ⚠️ 進程保護: setsid + nohup + < /dev/null(scheme1-XLMR-FullFT crash 教訓)

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

BACKBONE=./experiments/scheme1_xlmr_fullft
if [ ! -f "$BACKBONE/final_ssl.pt" ] || [ ! -f "$BACKBONE/final_text.pt" ] || [ ! -f "$BACKBONE/final_ser.pt" ]; then
  echo "ERROR: scheme1-XLMR-FullFT checkpoints missing in $BACKBONE" >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/nnime_scheme1xlmr_fullft_warmstart_lora

"$VENV" bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path "$BACKBONE" \
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
  --run_name "nnime_scheme1xlmr_fullft_warmstart_lora_${TS}"

echo "=== N4-B (scheme1-XLMR-FullFT warmstart NNIME LoRA) exited $? ==="
