#!/usr/bin/env bash
# scheme1_xlmr — 🅒-full — XLM-R+XLS-R LoRA on full MSP EN 110k Train
# 目的:製造「XLM-R+XLS-R 版本的 scheme1」(pure EN pretrained backbone)
#      之後 N4 對照 = 這個 ckpt + NNIME LoRA stage 2
#      對比 N3(v1 EN+ZH 混訓 + NNIME) → 拆出 v1 中 ZH 混訓的 unique 貢獻
#
# 完全 match v1 Strategy A LoRA 的超參,只換兩件事:
#   - 資料:v1 30k MSP + 12k EmoTalk → 純 110k MSP EN(單語)
#   - Sampler:v1 用 --language_balanced → default(無 balance,單語不需要)
#
# 資料量 110k vs v2b 53k → wall clock 估 22-26 hr(10 epoch)
# 無 3-layer sampler 開銷 → 實際可能 ~22 hr

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
MODEL_DIR=./experiments/scheme1_xlmr_lora_full

"$VENV" bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json "$WEIGHTS" \
  --wav_base_dir /home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios \
  --classes_list Excited Unconfident Neutral_3Class \
  --batch_size 16 --accumulation_steps 4 --num_workers 2 \
  --epochs 10 \
  --lr 2e-4 --encoder_lr 1e-4 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "scheme1_xlmr_lora_full_${TS}"

echo "=== scheme1_xlmr LoRA full exited $? ==="
