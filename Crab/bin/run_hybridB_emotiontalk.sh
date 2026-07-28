#!/usr/bin/env bash
# Step ② — Hybrid B: WavLM (kept, LoRA) + XLM-R (replaces RoBERTa, LoRA)
# Trains on EmotionTalk 3-class (scheme1 mapping: Excited / Unconfident / Neutral_3Class).
#
# Why "minimal change":
#   The only architectural delta vs scheme1 is the text encoder (RoBERTa→XLM-R).
#   Same WavLM weights are warm-started; same cross-modal fusion head is warm-started
#   (3-class + 1024 hidden_size unchanged). The training task is the SAME 3-class
#   scheme1 mapping, just on Chinese data (EmotionTalk).
#
# Comparable against:
#   • scheme1 zero-shot on EmotionTalk Test    → ZH macroF1 0.4810
#   • scheme1 dev on MSP-Podcast               → EN macroF1 ~0.67
#   • Strategy A (③) on the same Chinese Test  → A/B for "do we need XLS-R?"
#
# Validated settings (re-using contrastive 2.0 from prior CH-SIMS confirm):
#   contrastive 2.0 +0.018 macroF1 on CH-SIMS test split.
set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab
VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/emotiontalk_hybridB_lora

"$VENV" bin/train_crab_lora.py \
  --ssl_type microsoft/wavlm-large \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path ./experiments/interview_scheme1 \
  --warm_start_ser \
  --df_path ./data/emotiontalk_crab_format.csv \
  --weights_json ./data/emotiontalk_class_weights.json \
  --wav_base_dir /home/brant/Project/SAILER_test/datasets/emotiontalk/Audio16k \
  --classes_list Excited Unconfident Neutral_3Class \
  --batch_size 32 --accumulation_steps 8 --num_workers 3 \
  --epochs 15 \
  --lr 2e-4 --encoder_lr 1e-4 \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 5 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "hybridB_emotiontalk_${TS}"

echo "=== Hybrid B training exited with code $? ==="
