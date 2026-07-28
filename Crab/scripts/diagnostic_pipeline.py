"""
Decisive test: build a batch via the REAL training pipeline (LazyWavSet/TxtSet/
CAT_EmoSet/CombinedSet/collate_fn_bimodal), then try to memorize it (50 steps).

If the pipeline batch can't be memorized but the standalone probe batch could,
the bug is in the data pipeline (label alignment / audio / tokenization).

Also prints alignment: pipeline label vs CSV-expected label, text, audio stats.
"""
import os, sys, math
import numpy as np
import torch, torch.nn as nn
import pandas as pd

CRAB = "/home/brant/Project/SAILER_test/Crab"
sys.path.insert(0, CRAB)
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import src.models as net
from src.data.podcast import load_cat_emo_label
from src.data.dataset.dataset import LazyWavSet, TxtSet, CAT_EmoSet, CombinedSet
from torch.utils.data import DataLoader

device = "cuda"
CSV = f"{CRAB}/data/chsims_tiny_overfit.csv"
AUDIO = "/home/brant/Project/SAILER_test/datasets/chsims_v2s/ch-simsv2s/Audio"
CLS = ["Negative","WeaklyNegative","Neutral","WeaklyPositive","Positive"]

def collate(data):
    n=len(data); wl=[x[0][0] for x in data]; dl=[x[0][1] for x in data]; ml=max(dl)
    wav=torch.zeros(n,ml); msk=torch.zeros(n,ml)
    for i,(w,d) in enumerate(zip(wl,dl)): wav[i,:d]=torch.tensor(w[:d]); msk[i,:d]=1
    ids=torch.stack([x[1][0] for x in data]); tm=torch.stack([x[1][1] for x in data])
    lab=torch.tensor(np.array([x[2] for x in data])); utt=[x[3] for x in data]
    return (wav,msk),(ids,tm),lab,utt

# build pipeline exactly like training
utts, labs = load_cat_emo_label(CSV, "train", emolist=CLS)
paths = [os.path.join(AUDIO, u) for u in utts]
df = pd.read_csv(CSV); tdf = df[df.Split_Set=="Train"]
texts = tdf["Text"].fillna("").to_numpy()
tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

wavset = LazyWavSet(paths); wavset.compute_norm_stats(sample_size=5000)
print(f"norm stats: mean={wavset.wav_mean:.5f} std={wavset.wav_std:.5f}")
txtset = TxtSet(texts, tok, max_len=128)
emoset = CAT_EmoSet(labs)
ds = CombinedSet([wavset, txtset, emoset, utts])
dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)

(x_audio, amask),(ids,tmask), y_oh, utt = next(iter(dl))
y = y_oh.max(dim=1)[1]

# ALIGNMENT CHECK: pipeline label vs CSV expected
print("\n=== alignment check (first 8) ===")
for i in range(8):
    csv_row = tdf.iloc[i]
    csv_cls = int(np.argmax(csv_row[CLS].values))
    print(f"  idx {i}: utt={utt[i]:20s} pipe_label={int(y[i])}({CLS[int(y[i])]:14s}) csv_label={csv_cls}({CLS[csv_cls]:14s}) {'OK' if int(y[i])==csv_cls else '❌MISMATCH'}  text={texts[i][:20]}")

# AUDIO sanity
print(f"\n=== audio stats ===\n  x_audio min={x_audio.min():.2f} max={x_audio.max():.2f} mean={x_audio.mean():.3f} std={x_audio.std():.3f}")
print(f"  (if |max| >> 20 → over-amplified → WavLM saturate)")
print(f"  text non-pad counts: {tmask.sum(1).tolist()}")
print(f"  labels in batch: {y.tolist()}")

# MEMORIZE TEST on this pipeline batch
print("\n=== memorize pipeline batch (50 steps) ===")
ssl = AutoModel.from_pretrained("microsoft/wavlm-large")
ssl.load_state_dict(torch.load(f"{CRAB}/experiments/interview_scheme1/final_ssl.pt", map_location="cpu"))
for p in ssl.feature_extractor.parameters(): p.requires_grad=False
ssl = get_peft_model(ssl, LoraConfig(r=16,lora_alpha=32,target_modules=["q_proj","v_proj"],lora_dropout=0.1,bias="none")).to(device)
txt = get_peft_model(AutoModel.from_pretrained("FacebookAI/xlm-roberta-large"),
                     LoraConfig(r=16,lora_alpha=32,target_modules=["query","value"],lora_dropout=0.1,bias="none")).to(device)
ser = net.MultiModalEmotionClassifierDeep(1024,1024,512,5,0.5).to(device)
x_audio,amask,ids,tmask,y = x_audio.to(device),amask.to(device),ids.to(device),tmask.to(device),y.to(device)
opt = torch.optim.AdamW([{"params":ser.parameters(),"lr":1e-3},
    {"params":[p for p in txt.parameters() if p.requires_grad],"lr":5e-4},
    {"params":[p for p in ssl.parameters() if p.requires_grad],"lr":5e-4}])
crit = nn.CrossEntropyLoss()
ssl.train(); txt.train(); ser.train()
for s in range(51):
    a = ssl(x_audio.float(), attention_mask=amask.float()).last_hidden_state
    h = txt(input_ids=ids, attention_mask=tmask).last_hidden_state
    logits = ser(a,h); loss = crit(logits,y)
    opt.zero_grad(); loss.backward(); opt.step()
    if s in (0,10,25,50):
        acc=(logits.argmax(1)==y).float().mean().item()
        print(f"  step {s:2d}: loss {loss.item():.4f} (ln5={math.log(5):.4f}) acc {acc:.2f}")
print("\n→ acc→high : pipeline data OK (bug elsewhere)")
print("→ acc stuck/loss~ln5 : pipeline batch is unlearnable → label/audio/text bug")
