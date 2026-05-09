import os
import sys
import argparse
import logging
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from transformers import AutoModel, AutoTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.models as net
from src.data.dataset.dataset import WavSet, TxtSet
from src.utils.etc import set_deterministic
from src.utils.losses import MultiPosConLoss

def get_args():
    parser = argparse.ArgumentParser(description="MELD Baseline Training with WandB")
    parser.add_argument("--df_path", type=str, required=True, help="Path to MELD CSV")
    parser.add_argument("--wav_base_dir", type=str, default="", help="Base directory for audio files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save models")
    parser.add_argument("--ssl_type", type=str, default="microsoft/wavlm-large", help="Audio SSL model")
    parser.add_argument("--text_model_path", type=str, default="roberta-large", help="Text model path")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--head_dim", type=int, default=1024)
    parser.add_argument("--classes_list", nargs="+", default=["neutral", "surprise", "fear", "sadness", "joy", "anger", "disgust"])
    parser.add_argument("--project_name", type=str, default="SAILER_Crab_MELD", help="WandB project name")
    parser.add_argument("--run_name", type=str, default=None, help="WandB run name")
    parser.add_argument("--no_mlcs", action="store_true", help="Disable MLCS contrastive loss")
    return parser.parse_args()

def train():
    args = get_args()
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Logging
    os.makedirs(args.model_path, exist_ok=True)
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.model_path, f"train_meld_{run_time}.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

    # WandB Initialization
    run_name = args.run_name if args.run_name else f"MELD_Baseline_{run_time}"
    wandb.init(project=args.project_name, name=run_name, config=vars(args))

    logger.info(f"Starting MELD training with args: {args}")

    # Load Data
    df = pd.read_csv(args.df_path)
    class_to_id = {cls.lower(): i for i, cls in enumerate(args.classes_list)}
    
    # Check for emotion column
    target_col = None
    for col in ['Emotion', 'emotion', 'Sentiment', 'sentiment']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        df['label'] = df[target_col].str.lower().map(class_to_id)
    else:
        raise ValueError(f"Could not find emotion column in CSV. Available: {df.columns}")

    # Handle missing labels if any
    if df['label'].isnull().any():
        logger.warning(f"Found {df['label'].isnull().sum()} missing labels. Dropping them.")
        df = df.dropna(subset=['label'])

    # Split Data
    train_df = df[df['split'].str.lower() == 'train'].reset_index(drop=True)
    dev_df = df[df['split'].str.lower() == 'dev'].reset_index(drop=True)

    logger.info(f"Data Splits - Train: {len(train_df)}, Dev: {len(dev_df)}")

    # Models
    logger.info(f"Loading Encoders: {args.ssl_type}, {args.text_model_path}")
    ssl_model = AutoModel.from_pretrained(args.ssl_type)
    text_model = AutoModel.from_pretrained(args.text_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)

    if hasattr(ssl_model, 'freeze_feature_encoder'):
        ssl_model.freeze_feature_encoder()
    elif hasattr(ssl_model, 'feature_extractor'):
        for param in ssl_model.feature_extractor.parameters():
            param.requires_grad = False

    classifier = net.MultiModalEmotionClassifierDeep(
        features1_dim=ssl_model.config.hidden_size,
        features2_dim=text_model.config.hidden_size,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_emotions=len(args.classes_list),
        dropout=0.5 # Matched with official Crab 
    )

    ssl_model.to(device)
    text_model.to(device)
    classifier.to(device)

    # Dataset
    class MeldDataset(torch.utils.data.Dataset):
        def __init__(self, data_df, wav_base_dir, tokenizer, max_len=128):
            self.df = data_df
            self.wav_base_dir = wav_base_dir
            self.tokenizer = tokenizer
            self.max_len = max_len
            import librosa
            self.librosa = librosa

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            # Try to find wav in different splits if wav_base_dir is root
            wav_name = row['FileName'].replace('.mp4', '.wav')
            # Check if wav_name is already a path
            wav_path = os.path.join(self.wav_base_dir, row['split'], wav_name)
            if not os.path.exists(wav_path):
                # Fallback to direct path
                wav_path = os.path.join(self.wav_base_dir, wav_name)
                
            try:
                wav, _ = self.librosa.load(wav_path, sr=16000)
            except Exception as e:
                # logger.error(f"Error loading {wav_path}: {e}")
                wav = np.zeros(16000) # Fallback to 1s silence

            text = row['Utterance']
            encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
            
            return {
                'wav': torch.FloatTensor(wav),
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(int(row['label']), dtype=torch.long)
            }

    def collate_fn(batch):
        wavs = [item['wav'] for item in batch]
        max_wav_len = max(len(w) for w in wavs)
        max_wav_len = min(max_wav_len, 128000) # Limit to 8s
        max_wav_len = max(max_wav_len, 8000)   # Ensure at least 0.5s to avoid WavLM mask error
        padded_wavs = []
        for w in wavs:
            if len(w) > max_wav_len:
                padded_wavs.append(w[:max_wav_len])
            else:
                padded_wavs.append(torch.cat([w, torch.zeros(max_wav_len - len(w))]))
        
        return {
            'wav': torch.stack(padded_wavs),
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'label': torch.stack([item['label'] for item in batch])
        }

    train_loader = DataLoader(MeldDataset(train_df, args.wav_base_dir, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dev_loader = DataLoader(MeldDataset(dev_df, args.wav_base_dir, tokenizer), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Training logic
    optimizer = optim.AdamW([
        {'params': ssl_model.parameters(), 'lr': args.lr * 0.1},
        {'params': text_model.parameters(), 'lr': args.lr * 0.1},
        {'params': classifier.parameters(), 'lr': args.lr}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Calculate dynamic class weights for MELD
    label_counts = train_df['label'].value_counts().sort_index()
    counts_list = [label_counts.get(i, 1) for i in range(len(args.classes_list))]
    weights = 1.0 / torch.tensor(counts_list, dtype=torch.float)
    weights = weights / weights.sum() * len(args.classes_list)
    class_weights = weights.to(device)
    logger.info(f"Using class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # MLCS Contrastive Losses
    contrastive_criterion_audio = MultiPosConLoss().to(device)
    contrastive_criterion_text = MultiPosConLoss().to(device)
    contrastive_criterion_fusion = MultiPosConLoss().to(device)

    best_f1 = 0
    for epoch in range(args.epochs):
        classifier.train()
        ssl_model.train()
        text_model.train()
        
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_con_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, batch in enumerate(pbar):
            wav = batch['wav'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)

            # 0. Extract features
            audio_feats = ssl_model(wav).last_hidden_state
            text_feats = text_model(input_ids, attention_mask=attn_mask).last_hidden_state

            # 1. Forward Pass with Embeddings
            logits, embs = classifier(audio_feats, text_feats, return_embeddings=True)
            cls_loss = criterion(logits, label)
            
            # 2. MLCS - Multi-Level Contrastive Loss
            if not args.no_mlcs:
                # Frame-level
                c_loss_s_frame = contrastive_criterion_audio(embs['speech_frame_emb'], label)
                c_loss_t_frame = contrastive_criterion_text(embs['text_frame_emb'], label)
                
                # Pooled-level
                c_loss_s_pooled = contrastive_criterion_audio(embs['speech_pooled_emb'], label)
                c_loss_t_pooled = contrastive_criterion_text(embs['text_pooled_emb'], label)
                
                # Fusion-level
                c_loss_fusion = contrastive_criterion_fusion(embs['fusion_emb'], label)
                
                # Combined Contrastive Loss (Weight = 2.0, Average over 5 levels)
                total_contrastive_loss = 2.0 * (
                    c_loss_s_frame + c_loss_t_frame + 
                    c_loss_s_pooled + c_loss_t_pooled + 
                    c_loss_fusion
                ) / 5
                loss = (cls_loss + total_contrastive_loss) / args.accumulation_steps
            else:
                total_contrastive_loss = torch.tensor(0.0).to(device)
                loss = cls_loss / args.accumulation_steps
            
            loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache() # Clear cache after step

            epoch_loss += loss.item() * args.accumulation_steps
            epoch_cls_loss += cls_loss.item()
            epoch_con_loss += total_contrastive_loss.item()
            pbar.set_postfix(loss=epoch_loss / (i+1))

        scheduler.step()

        # Validation
        classifier.eval()
        ssl_model.eval()
        text_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Validating"):
                wav = batch['wav'].to(device)
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                audio_feats = ssl_model(wav).last_hidden_state
                text_feats = text_model(input_ids, attention_mask=attn_mask).last_hidden_state
                logits = classifier(audio_feats, text_feats)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Detailed Per-class Report
        report = classification_report(all_labels, all_preds, target_names=args.classes_list, digits=4)
        logger.info(f"\nEpoch {epoch+1} Classification Report:\n{report}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "cls_loss": epoch_cls_loss / len(train_loader),
            "contrastive_loss": epoch_con_loss / len(train_loader),
            "dev_acc": acc,
            "dev_f1": f1,
            "lr": optimizer.param_groups[-1]['lr']
        })

        logger.info(f"Epoch {epoch+1} - Dev Acc: {acc:.4f}, Weighted F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            save_dict = {
                'epoch': epoch + 1,
                'classifier': classifier.state_dict(),
                'ssl': ssl_model.state_dict(),
                'text': text_model.state_dict(),
                'f1': f1,
                'args': args
            }
            torch.save(save_dict, os.path.join(args.model_path, "best_model.pth"))
            logger.info(f"New best model saved with F1: {best_f1:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train()
