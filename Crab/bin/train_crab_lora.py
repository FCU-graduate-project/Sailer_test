# -*- coding: UTF-8 -*-
"""
LoRA fine-tune variant of train_crab.py for bilingual (Chinese) adaptation.

Architecture (unchanged from Crab): WavLM(audio) + XLM-R(text) → BiGRU unimodal
→ cross-attention fusion → classifier, with MPCL contrastive supervision.

DIFFERENCE vs train_crab.py:
  - Encoders are LoRA-adapted (base frozen) instead of fully fine-tuned.
    → far fewer trainable params → safer on small data (e.g. CH-SIMS 2.7k).
  - Only LoRA params + classifier head (ser_model) are optimized.
  - Saves LoRA adapters (small) + ser head, not full encoder weights.

IMPROVEMENTS over train_crab.py (gaps the original didn't handle):
  1. debug + accumulation guard (orig: cur_bs = 2//accum = 0 → crash)
  2. epoch-average train loss (orig: logged only last batch's loss)
  3. gradient clipping
  4. early stopping with patience (small data overfits fast)
  5. final test-split evaluation with best model (orig: never touched test)
  6. real contrastive weight control (orig: --constrastive_loss flag was dead)
  7. unified seed (orig: hardcoded 42 at top AND args.seed=100 → inconsistent)
  8. train/dev overfit-gap logged to W&B
"""
import os
import sys
import argparse
import copy
import json
import logging
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score, classification_report, recall_score, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from safetensors.torch import load_file as load_safetensors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.models as net
from src.utils.losses import MultiPosConLoss
from src.data.podcast import load_cat_emo_label
from src.data.dataset.dataset import load_norm_stat, TxtSet, CAT_EmoSet, CombinedSet, LazyWavSet
from src.utils.etc import set_deterministic


# ───────────────────────────── args ─────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ssl_type", type=str, default="microsoft/wavlm-large")
parser.add_argument("--text_model_path", type=str, default="FacebookAI/xlm-roberta-large")
parser.add_argument("--pre_trained_path", type=str, default="./experiments/interview_scheme1",
                    help="dir containing final_ssl.pt to warm-start WavLM from (English Crab)")
parser.add_argument("--batch_size", type=int, default=32, help="effective batch (=per_step * accum)")
parser.add_argument("--accumulation_steps", type=int, default=4)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-3, help="LR for classifier head (ser_model)")
parser.add_argument("--encoder_lr", type=float, default=None,
                    help="LR for LoRA params; default = lr/10")
parser.add_argument("--model_path", type=str, default="./experiments/chsims_wavlm_xlmr_lora")
parser.add_argument("--df_path", type=str, default="./data/chsims_crab_format.csv")
parser.add_argument("--weights_json", type=str, default="./data/chsims_class_weights.json")
parser.add_argument("--wav_base_dir", type=str,
                    default="/home/brant/Project/SAILER_test/datasets/chsims_v2s/ch-simsv2s/Audio")
parser.add_argument("--text_max_len", type=int, default=128)
parser.add_argument("--fusion_hidden_dim", type=int, default=512)
parser.add_argument("--classes_list", nargs='+', type=str,
                    default=['Negative', 'WeaklyNegative', 'Neutral', 'WeaklyPositive', 'Positive'])
parser.add_argument("--balanced_sampling", action="store_true", default=False,
                    help="class-balanced WeightedRandomSampler (per-class weights from JSON)")
parser.add_argument("--language_balanced", action="store_true", default=False,
                    help="language-balanced (50:50 EN/ZH) WeightedRandomSampler. "
                         "Requires a 'Language' column in --df_path. Mutually exclusive "
                         "with --balanced_sampling (language wins if both are set).")
parser.add_argument("--zh_source_balanced", action="store_true", default=False,
                    help="3-layer sampler: (1) EN 50 vs ZH 50 by Language, "
                         "(2) within ZH, EmotionTalk/CNSCED/NNIME evenly split by Source. "
                         "Requires 'Language' + 'Source' columns in --df_path. "
                         "Supersedes --language_balanced when set.")
parser.add_argument("--fear_boost_source", type=str, default="EmotionTalk",
                    help="Source name to apply fear boost within Unconfident class. "
                         "Only used when --zh_source_balanced + --fear_boost_ratio > 1.")
parser.add_argument("--fear_boost_ratio", type=float, default=1.0,
                    help="multiply sampler weight of (Source==fear_boost_source AND Unconfident==1) "
                         "samples by this factor. 1.0 = disabled. Recommend 3.0.")
parser.add_argument("--warm_start_ser", action="store_true", default=False,
                    help="if set, load final_ser.pt from --pre_trained_path as init for "
                         "the ser_model (cross-modal head). Requires matching num_classes "
                         "and hidden dims. Use when scheme1 ↔ Hybrid B (3-class, both 1024).")
parser.add_argument("--use_tp", action="store_true", default=False)
parser.add_argument("--tp_prob", type=float, default=0.8)
# LoRA
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.1)
parser.add_argument("--lora_target_set", choices=["standard", "expanded"], default="standard",
                    help="standard = q,v (current default); expanded = q,k,v plus o (audio only). "
                         "Text doesn't add output.dense because PEFT suffix matching would also "
                         "hit FFN dense layers and blow up the param count.")
parser.add_argument("--freeze_audio", action="store_true", default=False,
                    help="if set, WavLM is fully frozen (no LoRA); else WavLM gets LoRA too. "
                         "Only honored when --ft_mode=lora.")
parser.add_argument("--lora_warmstart", action="store_true", default=False,
                    help="if set, load existing LoRA adapters from --pre_trained_path/"
                         "text_lora_adapter/ + audio_lora_adapter/ instead of creating fresh. "
                         "Enables 2-stage LoRA fine-tune (e.g. v1 → NNIME stage 2). "
                         "Only honored when --ft_mode=lora.")
# fine-tune mode (lora = current behavior; full_ft = train all base params;
#  partial_ft = unfreeze top-N transformer layers per encoder)
parser.add_argument("--ft_mode", choices=["lora", "full_ft", "partial_ft"], default="lora",
                    help="lora (default) keeps current LoRA-on-frozen-base. "
                         "full_ft trains all base params (high VRAM!). "
                         "partial_ft unfreezes only the top --unfreeze_last_n transformer layers.")
parser.add_argument("--unfreeze_last_n", type=int, default=2,
                    help="for --ft_mode=partial_ft: how many top transformer layers to unfreeze "
                         "per encoder (default for both audio and text if no per-encoder override).")
parser.add_argument("--unfreeze_last_n_audio", type=int, default=None,
                    help="for --ft_mode=partial_ft: override N for audio encoder only "
                         "(defaults to --unfreeze_last_n if not set). "
                         "Wang 2022 ICASSP (arXiv 2111.02735) partial FT for SER uses ALL 24 "
                         "transformer layers; CNN feature extractor already frozen above.")
parser.add_argument("--unfreeze_last_n_text", type=int, default=None,
                    help="for --ft_mode=partial_ft: override N for text encoder only "
                         "(defaults to --unfreeze_last_n if not set). "
                         "Lee 2019 (arXiv 1911.03090) shows top-quarter (e.g. top-6 of 24 for "
                         "XLM-R-Large) is sufficient for classification fine-tuning.")
parser.add_argument("--use_amp", action="store_true", default=False,
                    help="bf16 mixed precision via autocast. Needed for full_ft to fit on 24GB.")
parser.add_argument("--use_grad_ckpt", action="store_true", default=False,
                    help="gradient checkpointing on both encoders. Needed for full_ft VRAM, "
                         "costs ~30%% wall-clock.")
# improvements
parser.add_argument("--contrastive_weight", type=float, default=2.0,
                    help="weight on summed contrastive losses; 0 disables contrastive")
parser.add_argument("--grad_clip", type=float, default=1.0, help="0 disables")
parser.add_argument("--early_stop_patience", type=int, default=5, help="0 disables")
parser.add_argument("--eval_test", action="store_true", default=True,
                    help="evaluate test split at the end with best model")
parser.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader worker processes for wav loading (0=single-thread). "
                         "On 6-core box, 3 is sweet spot.")
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--project_name", type=str, default="Crab_Bilingual_ZH")
parser.add_argument("--run_name", type=str, default=None)
args = parser.parse_args()

# ───────────────────────────── setup ─────────────────────────────
MODEL_PATH = args.model_path
os.makedirs(MODEL_PATH, exist_ok=True)

log_filename = f"logging_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_filepath = os.path.join(MODEL_PATH, log_filename)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# (7) unified seed — single source of truth
set_deterministic(args.seed)
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

SSL_TYPE = args.ssl_type
TEXT_MODEL_PATH = os.path.expanduser(args.text_model_path)
BATCH_SIZE = args.batch_size
ACCUMULATION_STEP = args.accumulation_steps
EPOCHS = args.epochs
LR = args.lr
ENCODER_LR = args.encoder_lr if args.encoder_lr is not None else LR / 10
TEXT_MAX_LEN = args.text_max_len
FUSION_HIDDEN_DIM = args.fusion_hidden_dim
PRE_TRAINED_PATH = args.pre_trained_path
USE_TP = args.use_tp
TP_PROB = args.tp_prob
BALANCED_SAMPLING = args.balanced_sampling
CONTRASTIVE_WEIGHT = args.contrastive_weight
GRAD_CLIP = args.grad_clip
EARLY_STOP_PATIENCE = args.early_stop_patience
classes = args.classes_list

# (1) debug + accumulation guard
debug = args.debug
if debug:
    logger.info("Running in debug mode!")
    BATCH_SIZE = 4
    EPOCHS = 2
    if ACCUMULATION_STEP > BATCH_SIZE:
        logger.warning(f"debug: accumulation_steps {ACCUMULATION_STEP} > batch {BATCH_SIZE}; forcing 1")
        ACCUMULATION_STEP = 1

assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0), \
    f"batch_size {BATCH_SIZE} must be divisible by accumulation_steps {ACCUMULATION_STEP}"

wandb.init(
    project=args.project_name,
    name=args.run_name if args.run_name else f"crab_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=vars(args),
)

logger.info("=" * 80)
logger.info("LoRA Fine-tune — Bimodal Crab (WavLM + XLM-R) on Chinese")
logger.info("=" * 80)
logger.info(f"Log file: {log_filepath}")
logger.info(f"W&B Run: {wandb.run.get_url()}")
for arg, value in vars(args).items():
    logger.info(f"  {arg}: {value}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Device: {device}")
logger.info(f"Classes: {classes}")
emo_list = copy.deepcopy(classes)

# ───────────────────────────── class weights ─────────────────────────────
df = pd.read_csv(args.df_path)
if 'Text' not in df.columns:
    raise ValueError("df must contain a 'Text' column")
train_df = df[df['Split_Set'] == 'Train']

if os.path.exists(args.weights_json):
    with open(args.weights_json) as f:
        weights_data = json.load(f)
    weights_list = [weights_data['class_weight'][cls] for cls in classes]
    logger.info("Class weights loaded from JSON")
else:
    class_frequencies = train_df[classes].sum().to_dict()
    total_samples = len(train_df)
    class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0
                     for cls, freq in class_frequencies.items()}
    weights_list = [class_weights[cls] for cls in classes]

class_weights_tensor = torch.tensor(weights_list, device=device, dtype=torch.float)
logger.info(f"Loss weights: {class_weights_tensor}")

# ───────────────────────────── data ─────────────────────────────
logger.info(f"Loading text tokenizer: {TEXT_MODEL_PATH}")
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)


def load_text_data(df_path, dtype, debug=False):
    d = pd.read_csv(df_path)
    split = {"train": "Train", "dev": "Development", "test": "Test"}[dtype]
    d = d[d['Split_Set'] == split]
    if debug:
        d = d.sample(n=min(100, len(d)), random_state=args.seed).reset_index(drop=True)
    return d['Text'].fillna("").to_numpy()


def collate_fn_bimodal(data):
    n_batch = len(data)
    wav_list = [x[0][0] for x in data]
    dur_list = [x[0][1] for x in data]
    max_len = max(dur_list)
    wav_arr = torch.zeros((n_batch, max_len))
    mask_arr = torch.zeros((n_batch, max_len))
    for i, (wav, dur) in enumerate(zip(wav_list, dur_list)):
        wav_arr[i, :dur] = torch.tensor(wav[:dur])
        mask_arr[i, :dur] = 1
    input_ids_arr = torch.stack([x[1][0] for x in data])
    text_attention_mask_arr = torch.stack([x[1][1] for x in data])
    lab_arr = torch.tensor(np.array([x[2] for x in data]))
    utt_list = [x[3] for x in data]
    return (wav_arr, mask_arr), (input_ids_arr, text_attention_mask_arr), lab_arr, utt_list


total_dataset, total_dataloader = {}, {}
splits = ["train", "dev"] + (["test"] if args.eval_test else [])
cur_bs = BATCH_SIZE // ACCUMULATION_STEP

for dtype in splits:
    cur_utts, cur_labs = load_cat_emo_label(args.df_path, dtype, debug=debug, emolist=emo_list)
    # absolute FileName entries (bilingual CSV) bypass wav_base_dir naturally
    cur_wav_paths = [utt if os.path.isabs(utt) else os.path.join(args.wav_base_dir, utt)
                     for utt in cur_utts]
    cur_texts = load_text_data(args.df_path, dtype, debug=debug)

    if dtype == "train":
        cur_wav_set = LazyWavSet(cur_wav_paths, use_tp=USE_TP, tp_prob=TP_PROB)
        logger.info("Computing normalization stats...")
        cur_wav_set.compute_norm_stats(sample_size=5000)
        cur_wav_set.save_norm_stat(MODEL_PATH + "/train_norm_stat.pkl")
    else:
        wav_mean = total_dataset["train"].datasets[0].wav_mean
        wav_std = total_dataset["train"].datasets[0].wav_std
        cur_wav_set = LazyWavSet(cur_wav_paths, wav_mean=wav_mean, wav_std=wav_std)

    cur_txt_set = TxtSet(cur_texts, text_tokenizer, max_len=TEXT_MAX_LEN)
    cur_emo_set = CAT_EmoSet(cur_labs)
    total_dataset[dtype] = CombinedSet([cur_wav_set, cur_txt_set, cur_emo_set, cur_utts])

    if dtype == "train" and args.zh_source_balanced:
        # 3-layer sampler:
        #   (1) EN 50 vs ZH 50 by Language
        #   (2) within ZH, EmotionTalk/CNSCED/NNIME/... evenly by Source
        #   (3) optional fear-boost: multiply weight of (Source==fear_boost_source AND Unconfident==1)
        train_full = pd.read_csv(args.df_path)
        train_full = train_full[train_full['Split_Set'] == 'Train']
        for col in ("Language", "Source"):
            if col not in train_full.columns:
                raise ValueError(f"--zh_source_balanced needs a '{col}' column in df")
        # per-utterance lookup: language + source + Unconfident bool
        fname_to_lang = dict(zip(train_full['FileName'].astype(str),
                                 train_full['Language'].astype(str)))
        fname_to_src = dict(zip(train_full['FileName'].astype(str),
                                train_full['Source'].astype(str)))
        fname_to_unc = dict(zip(train_full['FileName'].astype(str),
                                train_full['Unconfident'].astype(int)))
        missing = [u for u in cur_utts if u not in fname_to_lang]
        if missing:
            raise ValueError(f"--zh_source_balanced: {len(missing)} FileNames missing "
                             f"Language/Source lookup; example: {missing[0]!r}")

        # counts
        lang_counts = train_full['Language'].value_counts().to_dict()
        # per (lang, source) counts
        zh_src_counts = train_full[train_full['Language'] == 'ZH']['Source'].value_counts().to_dict()
        n_zh_sources = len(zh_src_counts)
        en_src_counts = train_full[train_full['Language'] == 'EN']['Source'].value_counts().to_dict()
        n_en_sources = len(en_src_counts)

        boost_src = args.fear_boost_source
        boost_ratio = float(args.fear_boost_ratio)

        sample_weights = []
        boost_hits = 0
        for u in cur_utts:
            lang = fname_to_lang[u]
            src = fname_to_src[u]
            if lang == 'EN':
                # split 50% mass across EN sources evenly, then across samples within source
                w = 0.5 / n_en_sources / en_src_counts[src]
            else:  # ZH
                w = 0.5 / n_zh_sources / zh_src_counts[src]
                if boost_ratio > 1.0 and src == boost_src and fname_to_unc[u] == 1:
                    w *= boost_ratio
                    boost_hits += 1
            sample_weights.append(w)

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                        replacement=True)
        logger.info(f"3-layer sampler: EN sources={en_src_counts}, ZH sources={zh_src_counts}, "
                    f"total lang={lang_counts}, fear_boost {boost_src}×{boost_ratio} hit {boost_hits} rows")
        total_dataloader[dtype] = DataLoader(total_dataset[dtype], batch_size=cur_bs, sampler=sampler,
                                             pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn_bimodal)
    elif dtype == "train" and args.language_balanced:
        # 50:50 EN / ZH via per-sample weights (inverse-frequency by language).
        # Build a FileName→Language lookup so weights align with cur_utts order,
        # NOT with whatever row order pd.read_csv happens to return.
        train_full = pd.read_csv(args.df_path)
        train_full = train_full[train_full['Split_Set'] == 'Train']
        if 'Language' not in train_full.columns:
            raise ValueError("--language_balanced needs a 'Language' column in df")
        fname_to_lang = dict(zip(train_full['FileName'].astype(str),
                                 train_full['Language'].astype(str)))
        missing = [u for u in cur_utts if u not in fname_to_lang]
        if missing:
            raise ValueError(f"--language_balanced: {len(missing)} FileNames in "
                             f"loader have no Language entry; example: {missing[0]!r}")
        lang_counts = train_full['Language'].value_counts().to_dict()
        # weight ∝ 1/freq so each language contributes ~equal mass
        sample_weights = [1.0 / lang_counts[fname_to_lang[u]] for u in cur_utts]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                        replacement=True)
        logger.info(f"Language-balanced sampler: counts={lang_counts}, "
                    f"aligned to cur_utts ({len(cur_utts)} rows)")
        total_dataloader[dtype] = DataLoader(total_dataset[dtype], batch_size=cur_bs, sampler=sampler,
                                             pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn_bimodal)
    elif dtype == "train" and BALANCED_SAMPLING:
        labs_indices = (np.argmax(cur_labs, axis=1) if np.array(cur_labs).ndim > 1
                        else np.array(cur_labs).astype(int))
        sample_weights = [weights_list[i] for i in labs_indices]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        total_dataloader[dtype] = DataLoader(total_dataset[dtype], batch_size=cur_bs, sampler=sampler,
                                             pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn_bimodal)
    elif dtype == "train":
        total_dataloader[dtype] = DataLoader(total_dataset[dtype], batch_size=cur_bs, shuffle=True,
                                             pin_memory=True, num_workers=args.num_workers, collate_fn=collate_fn_bimodal)
    else:
        total_dataloader[dtype] = DataLoader(total_dataset[dtype], batch_size=cur_bs * 4 if dtype == "dev" else 1,
                                             shuffle=False, pin_memory=True, num_workers=args.num_workers,
                                             collate_fn=collate_fn_bimodal)
    logger.info(f"{dtype}: {len(total_dataset[dtype])} samples, {len(total_dataloader[dtype])} batches")

# ───────────────────────────── models + LoRA ─────────────────────────────
logger.info(f"Loading SSL model: {SSL_TYPE}")
ssl_model = AutoModel.from_pretrained(SSL_TYPE)

# warm-start WavLM from English Crab BEFORE LoRA wrapping
ssl_weights_path = os.path.join(PRE_TRAINED_PATH, "final_ssl.pt")
if os.path.exists(ssl_weights_path):
    ssl_model.load_state_dict(torch.load(ssl_weights_path, map_location="cpu"))
    logger.info(f"Warm-started WavLM from {ssl_weights_path}")
else:
    logger.warning(f"No warm-start WavLM weights at {ssl_weights_path}; using pretrained.")

# freeze WavLM CNN feature extractor (matches train_crab.py)
if hasattr(ssl_model, 'feature_extractor'):
    for p in ssl_model.feature_extractor.parameters():
        p.requires_grad = False

logger.info(f"Loading text model: {TEXT_MODEL_PATH}")
text_model = AutoModel.from_pretrained(TEXT_MODEL_PATH)

# warm-start text encoder from a prior full-FT checkpoint if present
text_weights_path = os.path.join(PRE_TRAINED_PATH, "final_text.pt")
if os.path.exists(text_weights_path):
    text_model.load_state_dict(torch.load(text_weights_path, map_location="cpu"))
    logger.info(f"Warm-started text encoder from {text_weights_path}")
else:
    logger.info(f"No warm-start text weights at {text_weights_path}; using HF pretrained.")

# ── fine-tune mode setup ──
if args.ft_mode == "lora":
    # LoRA target sets:
    #   standard: text=[query,value], audio=[q_proj,v_proj]            (current default)
    #   expanded: text=[query,key,value], audio=[q_proj,k_proj,v_proj,out_proj]   (C1)
    if args.lora_target_set == "expanded":
        text_targets = ["query", "key", "value"]
        audio_targets = ["q_proj", "k_proj", "v_proj", "out_proj"]
    else:
        text_targets = ["query", "value"]
        audio_targets = ["q_proj", "v_proj"]
    logger.info(f"LoRA target_set={args.lora_target_set}: text={text_targets}, audio={audio_targets}")

    text_adapter_dir = os.path.join(PRE_TRAINED_PATH, "text_lora_adapter")
    if args.lora_warmstart and os.path.isdir(text_adapter_dir):
        text_model = PeftModel.from_pretrained(text_model, text_adapter_dir, is_trainable=True)
        logger.info(f"Warm-started text LoRA from {text_adapter_dir}")
        logger.info("Text LoRA:")
        text_model.print_trainable_parameters()
    else:
        text_lora_cfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=text_targets, lora_dropout=args.lora_dropout, bias="none",
        )
        text_model = get_peft_model(text_model, text_lora_cfg)
        logger.info("Text LoRA (fresh):")
        text_model.print_trainable_parameters()

    # audio encoder (WavLM/XLS-R attention), unless frozen
    if not args.freeze_audio:
        audio_adapter_dir = os.path.join(PRE_TRAINED_PATH, "audio_lora_adapter")
        if args.lora_warmstart and os.path.isdir(audio_adapter_dir):
            ssl_model = PeftModel.from_pretrained(ssl_model, audio_adapter_dir, is_trainable=True)
            logger.info(f"Warm-started audio LoRA from {audio_adapter_dir}")
            logger.info("Audio LoRA:")
            ssl_model.print_trainable_parameters()
        else:
            audio_lora_cfg = LoraConfig(
                r=args.lora_rank, lora_alpha=args.lora_alpha,
                target_modules=audio_targets, lora_dropout=args.lora_dropout, bias="none",
            )
            ssl_model = get_peft_model(ssl_model, audio_lora_cfg)
            logger.info("Audio LoRA (fresh):")
            ssl_model.print_trainable_parameters()
    else:
        for p in ssl_model.parameters():
            p.requires_grad = False
        logger.info("Audio encoder fully frozen (no LoRA).")

elif args.ft_mode == "full_ft":
    # Train all base params — no LoRA wrap. Keep ssl feature_extractor frozen
    # (CNN front-end already had this in lora path).
    if args.use_grad_ckpt:
        if hasattr(ssl_model, "gradient_checkpointing_enable"):
            ssl_model.gradient_checkpointing_enable()
            logger.info("Audio: gradient checkpointing enabled")
        if hasattr(text_model, "gradient_checkpointing_enable"):
            text_model.gradient_checkpointing_enable()
            logger.info("Text: gradient checkpointing enabled")
    n_ssl = sum(p.numel() for p in ssl_model.parameters() if p.requires_grad)
    n_text = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
    logger.info(f"Full FT: audio trainable {n_ssl/1e6:.1f}M, text trainable {n_text/1e6:.1f}M")

elif args.ft_mode == "partial_ft":
    # Freeze everything, then unfreeze top N transformer layers per encoder.
    # Audio (WavLM/XLS-R/Wav2Vec2): encoder.layers[-N:]
    # Text (XLM-R/RoBERTa):         encoder.layer[-N:]
    # Asymmetric: --unfreeze_last_n_audio / --unfreeze_last_n_text override the
    # symmetric --unfreeze_last_n default. SER literature supports asymmetry:
    # audio = Wang 2022 partial FT (all 24 transformer, CNN already frozen above),
    # text  = Lee 2019 top-quarter (top-6 of 24).
    n_audio = args.unfreeze_last_n_audio if args.unfreeze_last_n_audio is not None else args.unfreeze_last_n
    n_text  = args.unfreeze_last_n_text  if args.unfreeze_last_n_text  is not None else args.unfreeze_last_n
    for p in ssl_model.parameters():
        p.requires_grad = False
    for p in text_model.parameters():
        p.requires_grad = False

    audio_layers = None
    if hasattr(ssl_model, "encoder") and hasattr(ssl_model.encoder, "layers"):
        audio_layers = ssl_model.encoder.layers
    if audio_layers is not None:
        for layer in audio_layers[-n_audio:]:
            for p in layer.parameters():
                p.requires_grad = True
        logger.info(f"Audio partial FT: unfroze last {n_audio}/{len(audio_layers)} layers")
    else:
        logger.warning("Audio encoder has no .encoder.layers — nothing unfrozen on audio side.")

    text_layers = None
    if hasattr(text_model, "encoder") and hasattr(text_model.encoder, "layer"):
        text_layers = text_model.encoder.layer
    if text_layers is not None:
        for layer in text_layers[-n_text:]:
            for p in layer.parameters():
                p.requires_grad = True
        logger.info(f"Text partial FT: unfroze last {n_text}/{len(text_layers)} layers")
    else:
        logger.warning("Text encoder has no .encoder.layer — nothing unfrozen on text side.")

    if args.use_grad_ckpt:
        if hasattr(ssl_model, "gradient_checkpointing_enable"):
            ssl_model.gradient_checkpointing_enable()
        if hasattr(text_model, "gradient_checkpointing_enable"):
            text_model.gradient_checkpointing_enable()

    n_ssl = sum(p.numel() for p in ssl_model.parameters() if p.requires_grad)
    n_text = sum(p.numel() for p in text_model.parameters() if p.requires_grad)
    logger.info(f"Partial FT: audio trainable {n_ssl/1e6:.1f}M, text trainable {n_text/1e6:.1f}M")

ssl_model.to(device)
text_model.to(device)

# audio/text feat dims — LoRA wrap nests model under base_model.model; full/partial FT does not.
if args.ft_mode == "lora":
    audio_feat_dim = (ssl_model.base_model.model.config.hidden_size if not args.freeze_audio
                      else ssl_model.config.hidden_size)
    text_feat_dim = text_model.base_model.model.config.hidden_size
else:
    audio_feat_dim = ssl_model.config.hidden_size
    text_feat_dim = text_model.config.hidden_size
logger.info(f"audio_feat_dim={audio_feat_dim}, text_feat_dim={text_feat_dim}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
contrastive_criterion_audio = MultiPosConLoss()
contrastive_criterion_text = MultiPosConLoss()
contrastive_criterion_fusion = MultiPosConLoss()

ser_model = net.MultiModalEmotionClassifierDeep(
    features1_dim=audio_feat_dim, features2_dim=text_feat_dim,
    fusion_hidden_dim=FUSION_HIDDEN_DIM, num_emotions=len(classes), dropout=0.5,
).to(device)

# optional ser warm-start (Hybrid B: scheme1 head reused since 3-class + dims match)
if args.warm_start_ser:
    ser_init = os.path.join(PRE_TRAINED_PATH, "final_ser.pt")
    if os.path.exists(ser_init):
        try:
            state = torch.load(ser_init, map_location=device)
            ser_model.load_state_dict(state)
            logger.info(f"Warm-started ser_model from {ser_init}")
        except Exception as e:
            logger.warning(f"ser warm-start failed ({e}); using fresh init")
    else:
        logger.warning(f"--warm_start_ser set but no {ser_init}; using fresh init")

# optimizers — only trainable params (mode-agnostic: any non-empty trainable set gets an opt)
ser_opt = torch.optim.AdamW(ser_model.parameters(), LR)
opts = [ser_opt]
schedulers = [CosineAnnealingLR(ser_opt, T_max=EPOCHS, eta_min=1e-6)]

text_trainable = [p for p in text_model.parameters() if p.requires_grad]
if text_trainable:
    text_opt = torch.optim.AdamW(text_trainable, ENCODER_LR)
    opts.append(text_opt)
    schedulers.append(CosineAnnealingLR(text_opt, T_max=EPOCHS, eta_min=1e-7))

ssl_trainable = [p for p in ssl_model.parameters() if p.requires_grad]
if ssl_trainable:
    ssl_opt = torch.optim.AdamW(ssl_trainable, ENCODER_LR)
    opts.append(ssl_opt)
    schedulers.append(CosineAnnealingLR(ssl_opt, T_max=EPOCHS, eta_min=1e-7))

n_trainable = sum(p.numel() for opt in opts for g in opt.param_groups for p in g['params'])
logger.info(f"Total trainable params: {n_trainable/1e6:.2f}M")
wandb.run.summary["trainable_params_M"] = n_trainable / 1e6


def save_models(tag="best"):
    if args.ft_mode == "lora":
        text_model.save_pretrained(os.path.join(MODEL_PATH, "text_lora_adapter"))
        if not args.freeze_audio:
            ssl_model.save_pretrained(os.path.join(MODEL_PATH, "audio_lora_adapter"))
    else:
        # full_ft / partial_ft: save full state dicts (large — ~2 GB each)
        torch.save(ssl_model.state_dict(), os.path.join(MODEL_PATH, "final_ssl.pt"))
        torch.save(text_model.state_dict(), os.path.join(MODEL_PATH, "final_text.pt"))
    torch.save(ser_model.state_dict(), os.path.join(MODEL_PATH, "final_ser.pt"))
    # write a small mode.json next to weights so downstream reloaders can
    # verify they're loading into a matching wrap shape (avoids LoRA vs FT
    # silent mismatch when an eval script hardcodes one path).
    with open(os.path.join(MODEL_PATH, "mode.json"), "w") as _f:
        json.dump({
            "ft_mode": args.ft_mode,
            "ssl_type": SSL_TYPE,
            "text_model_path": TEXT_MODEL_PATH,
            "freeze_audio": args.freeze_audio,
            "unfreeze_last_n": args.unfreeze_last_n if args.ft_mode == "partial_ft" else None,
            "unfreeze_last_n_audio": args.unfreeze_last_n_audio if args.ft_mode == "partial_ft" else None,
            "unfreeze_last_n_text": args.unfreeze_last_n_text if args.ft_mode == "partial_ft" else None,
            "lora_rank": args.lora_rank if args.ft_mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.ft_mode == "lora" else None,
            "classes": classes,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
        }, _f, indent=2)


_amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if args.use_amp \
           else nullcontext


def run_eval(loader):
    ssl_model.eval(); text_model.eval(); ser_model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            (x_audio, mask_audio), (x_text_ids, mask_text), y, _ = batch
            x_audio = x_audio.to(device).float(); mask_audio = mask_audio.to(device).float()
            x_text_ids = x_text_ids.to(device); mask_text = mask_text.to(device)
            y = y.max(dim=1)[1].to(device).long()
            with _amp_ctx():
                ssl = ssl_model(x_audio, attention_mask=mask_audio).last_hidden_state
                text_hs = text_model(input_ids=x_text_ids, attention_mask=mask_text).last_hidden_state
                logits = ser_model(ssl, text_hs)
            all_logits.append(logits.float()); all_labels.append(y)
    logits = torch.cat(all_logits); labels = torch.cat(all_labels)
    loss = criterion(logits, labels).item()
    y_pred = torch.argmax(logits, 1).cpu().numpy(); y_true = labels.cpu().numpy()
    return {
        "loss": loss,
        "war": accuracy_score(y_true, y_pred),
        "uar": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "y_pred": y_pred, "y_true": y_true,
    }


# ───────────────────────────── train loop ─────────────────────────────
max_f1, best_epoch, epochs_no_improve = 0.0, 0, 0
logger.info("\n" + "=" * 80 + "\nStarting LoRA training\n" + "=" * 80)

for epoch in range(EPOCHS):
    logger.info(f"\nEpoch {epoch}")
    ssl_model.train(); text_model.train(); ser_model.train()
    # Python float accumulators; per batch we sync each tensor exactly once
    # (loss / cls / con) and reuse the scalars for both running totals and
    # wandb.log — was 6 syncs/batch in the original, now 3.
    running_loss, running_cls, running_con, n_batches = 0.0, 0.0, 0.0, 0

    for batch_cnt, batch in enumerate(tqdm(total_dataloader["train"])):
        (x_audio, mask_audio), (x_text_ids, mask_text), y, _ = batch
        x_audio = x_audio.to(device, non_blocking=True).float()
        mask_audio = mask_audio.to(device, non_blocking=True).float()
        x_text_ids = x_text_ids.to(device, non_blocking=True)
        mask_text = mask_text.to(device, non_blocking=True)
        y = y.max(dim=1)[1].to(device, non_blocking=True).long()

        with _amp_ctx():
            ssl = ssl_model(x_audio, attention_mask=mask_audio).last_hidden_state
            text_hs = text_model(input_ids=x_text_ids, attention_mask=mask_text).last_hidden_state
            emo_pred, embeddings = ser_model(ssl, text_hs, return_embeddings=True)
            cls_loss = criterion(emo_pred, y)

            if CONTRASTIVE_WEIGHT > 0:
                con = (contrastive_criterion_audio(embeddings['speech_frame_emb'], y)
                       + contrastive_criterion_text(embeddings['text_frame_emb'], y)
                       + contrastive_criterion_audio(embeddings['speech_pooled_emb'], y)
                       + contrastive_criterion_text(embeddings['text_pooled_emb'], y)
                       + contrastive_criterion_fusion(embeddings['fusion_emb'], y)) / 5
                con = CONTRASTIVE_WEIGHT * con
            else:
                con = torch.zeros((), device=device)

            loss = cls_loss + con
        (loss / ACCUMULATION_STEP).backward()

        if (batch_cnt + 1) % ACCUMULATION_STEP == 0 or (batch_cnt + 1) == len(total_dataloader["train"]):
            if GRAD_CLIP > 0:
                for opt in opts:
                    for g in opt.param_groups:
                        torch.nn.utils.clip_grad_norm_(g['params'], GRAD_CLIP)
            for opt in opts:
                opt.step(); opt.zero_grad(set_to_none=True)

        # (2) compute scalars ONCE per batch and reuse for both running
        # accumulators and wandb. Original called .item()/float() six times
        # (3 in accumulator + 3 in wandb.log) — now 3 syncs/batch.
        loss_f = loss.detach().item()
        cls_f = cls_loss.detach().item()
        con_f = con.detach().item() if isinstance(con, torch.Tensor) else float(con)
        running_loss += loss_f; running_cls += cls_f; running_con += con_f
        n_batches += 1
        global_step = epoch * len(total_dataloader["train"]) + batch_cnt
        wandb.log({"batch/total_loss": loss_f, "batch/cls_loss": cls_f,
                   "batch/contrastive_loss": con_f}, step=global_step)

    for sch in schedulers:
        sch.step()

    train_loss = running_loss / max(n_batches, 1)
    train_cls = running_cls / max(n_batches, 1)
    train_con = running_con / max(n_batches, 1)

    # ── dev eval ──
    dev = run_eval(total_dataloader["dev"])
    logger.info(f"Train loss: {train_loss:.4f} (cls {train_cls:.4f} / con {train_con:.4f})")
    logger.info(f"Dev: loss {dev['loss']:.4f} WAR {dev['war']:.4f} UAR {dev['uar']:.4f} "
                f"macroF1 {dev['macro_f1']:.4f} wF1 {dev['weighted_f1']:.4f}")
    logger.info("\n" + classification_report(dev['y_true'], dev['y_pred'], target_names=classes,
                                              digits=4, zero_division=0))

    epoch_step = (epoch + 1) * len(total_dataloader["train"]) - 1
    wandb_log = {
        "epoch": epoch, "train_loss": train_loss, "train_cls_loss": train_cls,
        "train_contrastive_loss": train_con,
        "dev_loss": dev['loss'], "dev_WAR": dev['war'], "dev_UAR": dev['uar'],
        "dev_macro_f1": dev['macro_f1'], "dev_weighted_f1": dev['weighted_f1'],
        # (8) overfit-gap monitor
        "overfit_gap_loss": dev['loss'] - train_loss,
        # schedulers order: [ser, text?, ssl?] — text/ssl appended only if
        # they have trainable params. Guard so a fully-frozen text encoder
        # doesn't shift the index and mislabel ssl's LR as "lr_text".
        "lr_ser": schedulers[0].get_last_lr()[0],
        "lr_text": schedulers[1].get_last_lr()[0] if len(schedulers) > 1 else 0.0,
        "lr_ssl": schedulers[2].get_last_lr()[0] if len(schedulers) > 2 else 0.0,
    }
    per_class_f1 = f1_score(dev['y_true'], dev['y_pred'], average=None, zero_division=0)
    for i, cls in enumerate(classes):
        if i < len(per_class_f1):
            wandb_log[f"dev_f1/{cls}"] = per_class_f1[i]
    wandb.log(wandb_log, step=epoch_step)

    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        try:
            cm = confusion_matrix(dev['y_true'], dev['y_pred'])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].clip(min=1)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_xlabel('Predicted'); ax.set_ylabel('True')
            ax.set_title(f'Dev Confusion Matrix - Epoch {epoch+1}')
            plt.tight_layout()
            wandb.log({"dev/confusion_matrix": wandb.Image(fig)}, step=epoch_step)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"CM log failed: {e}")

    # ── best model + early stopping ──
    if dev['macro_f1'] > max_f1:
        max_f1, best_epoch, epochs_no_improve = dev['macro_f1'], epoch, 0
        save_models("best")
        wandb.run.summary["best_dev_macro_f1"] = max_f1
        wandb.run.summary["best_epoch"] = best_epoch
        logger.info(f"★ New best macro-F1 {max_f1:.4f} (epoch {epoch}) — saved.")
    else:
        epochs_no_improve += 1
        logger.info(f"No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

    torch.save({
        'epoch': epoch, 'ser_model_state_dict': ser_model.state_dict(),
        'max_f1': max_f1, 'best_epoch': best_epoch,
    }, os.path.join(MODEL_PATH, "checkpoint_latest.pt"))

    # (4) early stopping
    if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
        logger.info(f"Early stopping at epoch {epoch} (no improve for {EARLY_STOP_PATIENCE}).")
        break

logger.info(f"\nTraining done. Best macro-F1 {max_f1:.4f} @ epoch {best_epoch}")

# ───────────────────────── (5) final test eval with best model ─────────────────────────
if args.eval_test and "test" in total_dataloader:
    logger.info(f"Loading best weights + ser head for test eval (ft_mode={args.ft_mode})...")
    # In-memory weights at this point are from the LAST epoch (potentially after
    # early stop's patience window), NOT from the best epoch. Reload all three
    # from disk so test eval reflects the actual saved 'best' checkpoint.
    if args.ft_mode == "lora":
        text_adapter_state = load_safetensors(
            os.path.join(MODEL_PATH, "text_lora_adapter", "adapter_model.safetensors"))
        set_peft_model_state_dict(text_model, text_adapter_state)
        logger.info("  ✓ text LoRA reloaded from disk (best)")
        if not args.freeze_audio:
            audio_adapter_state = load_safetensors(
                os.path.join(MODEL_PATH, "audio_lora_adapter", "adapter_model.safetensors"))
            set_peft_model_state_dict(ssl_model, audio_adapter_state)
            logger.info("  ✓ audio LoRA reloaded from disk (best)")
    else:
        # full_ft / partial_ft: reload full state dicts saved by save_models
        ssl_model.load_state_dict(torch.load(
            os.path.join(MODEL_PATH, "final_ssl.pt"), map_location=device))
        logger.info("  ✓ audio full state_dict reloaded from disk (best)")
        text_model.load_state_dict(torch.load(
            os.path.join(MODEL_PATH, "final_text.pt"), map_location=device))
        logger.info("  ✓ text full state_dict reloaded from disk (best)")
    ser_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "final_ser.pt"),
                                         map_location=device))
    logger.info("  ✓ ser head reloaded from disk (best)")
    test = run_eval(total_dataloader["test"])
    logger.info(f"TEST: loss {test['loss']:.4f} WAR {test['war']:.4f} UAR {test['uar']:.4f} "
                f"macroF1 {test['macro_f1']:.4f} wF1 {test['weighted_f1']:.4f}")
    logger.info("\n" + classification_report(test['y_true'], test['y_pred'], target_names=classes,
                                              digits=4, zero_division=0))
    wandb.run.summary["test_macro_f1"] = test['macro_f1']
    wandb.run.summary["test_WAR"] = test['war']
    wandb.run.summary["test_UAR"] = test['uar']
    try:
        cm = confusion_matrix(test['y_true'], test['y_pred'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis].clip(min=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Test Confusion Matrix (best model)')
        plt.tight_layout()
        wandb.log({"test/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Test CM log failed: {e}")

wandb.finish()
logger.info("Done.")
