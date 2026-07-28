#!/bin/bash
# 奧客 4 類 SER:MSP(EN) 4 類 LoRA 微調,warm-start 自 Strategy A,沿用其超參。
# per-step batch=32(accum 1)→ 每批約 8/類,讓 MPCL 對比學習真的有效(非太小 batch)。
cd /home/brant/Project/SAILER_test/Crab
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python bin/train_crab_lora.py \
  --df_path data/okeke_msp_4class.csv --wav_base_dir "" \
  --weights_json data/okeke_msp_4class_weights.json \
  --classes_list Angry Happy Neutral Anxious \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path experiments/strategyA_fullft \
  --model_path experiments/okeke_msp_4class \
  --ft_mode lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --batch_size 16 --accumulation_steps 1 --epochs 8 \
  --use_amp --use_grad_ckpt --contrastive_weight 2.0 \
  --lr 1e-4 --encoder_lr 1e-5 --early_stop_patience 5 --num_workers 0 \
  --project_name Crab_Okeke_4class --run_name msp4_lora_warmA_lr1e4
