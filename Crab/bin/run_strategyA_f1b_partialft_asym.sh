#!/usr/bin/env bash
# F1-B: Partial FT asymmetric — Wang 2022 audio + Lee 2019 text.
#
# Why this exists (deep-research survey 2026-06-04):
#   C1 / C1b / C5 all failed at "more LoRA capacity" (target expansion or rank).
#   The SER literature points to a different partial-FT recipe:
#
#   Audio (Wang/Boumadane/Heba 2022 ICASSP, arXiv 2111.02735):
#     "Partial fine-tuning" for wav2vec2/HuBERT = freeze CNN feature extractor
#     + UNFREEZE ALL transformer layers. PF-hbt-large reported 79.58% WA on
#     IEMOCAP (SD), BEATING entire FT (EF). The CNN front-end is already frozen
#     in train_crab_lora.py:334-337, so passing audio_n=24 = Wang 2022 setup.
#
#   Text (Lee/Tang/Lin 2019, arXiv 1911.03090 "What Would Elsa Do?"):
#     For BERT/RoBERTa/XLM-R classification fine-tuning, top-quarter of layers
#     suffices (top-3 of 12, top-6 of 24). XLM-R-Large = 24 layers -> N=6.
#
# This is the FIRST experiment that targets the FT (non-LoRA) capacity axis
# with a literature-grounded asymmetric design.
#
# Expected:
#   Wang 2022 reports PF > EF -> F1-B SHOULD beat C14 Full FT 0.6408.
#   Whether F1-B beats LoRA Strategy A 0.6531 is the open question this run answers.
#   If F1-B >= 0.66 -> partial FT becomes the new main path (re-write paper around it).
#   If F1-B in [0.64, 0.66] -> matches Wang 2022 prediction but doesn't beat LoRA;
#     add as paper ablation row "partial FT competitive but LoRA still wins on bilingual".
#   If F1-B < 0.64 -> partial FT also fails; full FT family is dead for this data scale.
#
# Estimated trainable params:
#   Audio: ~315M (24 transformer + classifier; CNN ~7M frozen)
#   Text:  ~75M  (top-6 of 24 transformer + classifier)
#   SER head: ~19M
#   TOTAL ~410M  (between LoRA 6M and Full FT 891M; ~half of Full FT)
set -euo pipefail

cd /home/brant/Project/SAILER_test/Crab

CSV=./data/bilingual_strategyA.csv
if [ ! -f "$CSV" ]; then
  echo "ERROR: $CSV missing - run scripts/build_bilingual_train_csv.py first." >&2
  exit 1
fi

VENV=/home/brant/Project/SAILER_test/Crab/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
MODEL_DIR=./experiments/strategyA_f1b_partialft_asym

"$VENV" bin/train_crab_lora.py \
  --ft_mode partial_ft \
  --unfreeze_last_n_audio 24 \
  --unfreeze_last_n_text 6 \
  --use_amp --use_grad_ckpt \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \
  --df_path "$CSV" \
  --weights_json ./data/bilingual_class_weights.json \
  --wav_base_dir "" \
  --language_balanced \
  --classes_list Excited Unconfident Neutral_3Class \
  --batch_size 16 --accumulation_steps 4 --num_workers 3 \
  --epochs 8 \
  --lr 1e-4 --encoder_lr 1e-5 \
  --contrastive_weight 2.0 --grad_clip 1.0 --early_stop_patience 3 \
  --eval_test \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --model_path "$MODEL_DIR" \
  --project_name Crab_Bilingual_ZH \
  --run_name "strategyA_f1b_partialft_asym_${TS}"

echo "=== F1-B Partial FT (audio=24, text=6) exited with code $? ==="
