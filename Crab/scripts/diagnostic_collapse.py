"""
Diagnose why the LoRA classifier collapses to uniform output (train loss stuck at ln5).

Checks:
  1. Variance ACROSS the batch of: ssl feat, text feat, speech_pooled, text_pooled,
     concatenated, normalized, logits. If logits don't vary across samples → constant output.
  2. Can the model OVERFIT a single tiny batch (16 samples, 100 steps)? If train loss
     can't drop below ln5 on 16 samples → hard structural bug.
  3. Gradient norms on ser_model params (are they receiving signal?).
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import pandas as pd

CRAB = "/home/brant/Project/SAILER_test/Crab"
sys.path.insert(0, CRAB)
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import src.models as net

device = "cuda"
AUDIO_ROOT = "/home/brant/Project/SAILER_test/datasets/chsims_v2s/ch-simsv2s/Audio"
CSV = f"{CRAB}/data/chsims_crab_format.csv"
CLASSES = ["Negative", "WeaklyNegative", "Neutral", "WeaklyPositive", "Positive"]
TARGET_SR = 16000

def load_wav(path, mean, std):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(data.T).mean(0)[:12*TARGET_SR]
    return (w - mean) / (std + 1e-6)

# ---- build same model as training ----
print("loading models...")
ssl = AutoModel.from_pretrained("microsoft/wavlm-large")
ssl.load_state_dict(torch.load(f"{CRAB}/experiments/interview_scheme1/final_ssl.pt", map_location="cpu"))
for p in ssl.feature_extractor.parameters(): p.requires_grad = False
ssl = get_peft_model(ssl, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.1, bias="none"))
ssl.to(device)

txt = AutoModel.from_pretrained("FacebookAI/xlm-roberta-large")
txt = get_peft_model(txt, LoraConfig(r=16, lora_alpha=32, target_modules=["query","value"], lora_dropout=0.1, bias="none"))
txt.to(device)
tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

ser = net.MultiModalEmotionClassifierDeep(features1_dim=1024, features2_dim=1024,
        fusion_hidden_dim=512, num_emotions=5, dropout=0.5).to(device)

# ---- one batch of 16 training samples (varied labels) ----
df = pd.read_csv(CSV)
tr = df[df.Split_Set=="Train"]
# pick samples spanning classes
picks = []
for c in CLASSES:
    picks += list(tr[tr[c]==1].head(4).index)
batch = tr.loc[picks[:16]]
print(f"batch labels: {[int(np.argmax(r[CLASSES].values)) for _,r in batch.iterrows()]}")

# fixed norm stats (approx — use 0/1, only relative variance matters here)
mean, std = 0.0, 1.0
wavs = [load_wav(os.path.join(AUDIO_ROOT, r.FileName), mean, std) for _,r in batch.iterrows()]
maxlen = max(w.shape[0] for w in wavs)
x_audio = torch.zeros(len(wavs), maxlen); amask = torch.zeros(len(wavs), maxlen)
for i,w in enumerate(wavs):
    x_audio[i,:w.shape[0]] = w; amask[i,:w.shape[0]] = 1
x_audio, amask = x_audio.to(device), amask.to(device)
texts = batch.Text.fillna("").tolist()
t = tok(texts, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
ids, tmask = t.input_ids.to(device), t.attention_mask.to(device)
y = torch.tensor([int(np.argmax(r[CLASSES].values)) for _,r in batch.iterrows()]).to(device)

def fwd():
    a = ssl(x_audio, attention_mask=amask).last_hidden_state
    h = txt(input_ids=ids, attention_mask=tmask).last_hidden_state
    return a, h

# ---- 1. variance across batch ----
ssl.eval(); txt.eval(); ser.eval()
with torch.no_grad():
    a, h = fwd()
    print("\n=== feature variance across batch (mean over dims of per-sample std) ===")
    print(f"  ssl last_hidden  meanstd={a.mean(1).std(0).mean().item():.5f}  (audio feat varies across samples?)")
    print(f"  txt last_hidden  meanstd={h.mean(1).std(0).mean().item():.5f}  (text feat varies?)")
    logits, emb = ser(a, h, return_embeddings=True)
    print(f"  concatenated/normalized varies: {emb['normalized'].std(0).mean().item():.5f}")
    print(f"  LOGITS std across batch (per-class): {logits.std(0).detach().cpu().numpy()}")
    print(f"  LOGITS mean across batch: {logits.mean(0).detach().cpu().numpy()}")
    print(f"  → if logits std ~0 → constant output regardless of input")
    print(f"  text token counts (non-pad): {tmask.sum(1).tolist()}")

# ---- 2. can it overfit 16 samples? ----
print("\n=== overfit test: 100 steps on these 16 samples (head lr 1e-3, lora 5e-4) ===")
ssl.train(); txt.train(); ser.train()
crit = nn.CrossEntropyLoss()
opt = torch.optim.AdamW([
    {"params": ser.parameters(), "lr": 1e-3},
    {"params": [p for p in txt.parameters() if p.requires_grad], "lr": 5e-4},
    {"params": [p for p in ssl.parameters() if p.requires_grad], "lr": 5e-4},
])
ln5 = math.log(5)
for step in range(101):
    a, h = fwd()
    logits = ser(a, h)
    loss = crit(logits, y)
    opt.zero_grad(); loss.backward()
    if step in (0,1,5,10,25,50,100):
        gn_ser = math.sqrt(sum((p.grad**2).sum().item() for p in ser.parameters() if p.grad is not None))
        gn_txt = math.sqrt(sum((p.grad**2).sum().item() for p in txt.parameters() if p.requires_grad and p.grad is not None))
        pred = logits.argmax(1)
        acc = (pred==y).float().mean().item()
        print(f"  step {step:3d}: loss {loss.item():.4f} (ln5={ln5:.4f})  acc {acc:.2f}  grad_ser {gn_ser:.3e}  grad_txt {gn_txt:.3e}  pred {pred.tolist()}")
    opt.step()

print("\n→ if loss drops toward 0 + acc→1 : model CAN learn, full-run issue is elsewhere (LR sched / data scale)")
print("→ if loss stuck at ln5 + acc stuck : structural bug (constant features / dead path)")
