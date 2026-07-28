"""
Minimal trainer on 40 mixed pipeline samples: simple setup (1 optimizer, no
scheduler, plain CE, batch 8). If this FITS but train_crab_lora.py doesn't,
the bug is in the script's {3 separate optimizers / cosine scheduler / weighted CE}.
"""
import os, sys, math
import numpy as np, pandas as pd
import torch, torch.nn as nn
CRAB="/home/brant/Project/SAILER_test/Crab"; sys.path.insert(0,CRAB)
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
import src.models as net
from src.data.podcast import load_cat_emo_label
from src.data.dataset.dataset import LazyWavSet, TxtSet, CAT_EmoSet, CombinedSet
from torch.utils.data import DataLoader

device="cuda"; CSV=f"{CRAB}/data/chsims_tiny_overfit.csv"
AUDIO="/home/brant/Project/SAILER_test/datasets/chsims_v2s/ch-simsv2s/Audio"
CLS=["Negative","WeaklyNegative","Neutral","WeaklyPositive","Positive"]

def collate(data):
    n=len(data); wl=[x[0][0] for x in data]; dl=[x[0][1] for x in data]; ml=max(dl)
    wav=torch.zeros(n,ml); msk=torch.zeros(n,ml)
    for i,(w,d) in enumerate(zip(wl,dl)): wav[i,:d]=torch.tensor(w[:d]); msk[i,:d]=1
    ids=torch.stack([x[1][0] for x in data]); tm=torch.stack([x[1][1] for x in data])
    lab=torch.tensor(np.array([x[2] for x in data]))
    return (wav,msk),(ids,tm),lab

utts,labs=load_cat_emo_label(CSV,"train",emolist=CLS)
df=pd.read_csv(CSV); tdf=df[df.Split_Set=="Train"]
texts=tdf["Text"].fillna("").to_numpy()
# take 8 per class = 40 mixed
idx=[]
for c in range(5):
    ci=[i for i in range(len(labs)) if int(np.argmax(labs[i]))==c][:8]
    idx+=ci
paths=[os.path.join(AUDIO,utts[i]) for i in idx]
tok=AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
ws=LazyWavSet(paths); ws.compute_norm_stats(5000)
ts=TxtSet(texts[idx],tok,128); es=CAT_EmoSet(labs[idx])
ds=CombinedSet([ws,ts,es,[utts[i] for i in idx]])
dl=DataLoader(ds,batch_size=8,shuffle=True,collate_fn=collate)
print(f"{len(ds)} samples, {len(dl)} batches/epoch")

ssl=AutoModel.from_pretrained("microsoft/wavlm-large")
ssl.load_state_dict(torch.load(f"{CRAB}/experiments/interview_scheme1/final_ssl.pt",map_location="cpu"))
for p in ssl.feature_extractor.parameters(): p.requires_grad=False
ssl=get_peft_model(ssl,LoraConfig(r=16,lora_alpha=32,target_modules=["q_proj","v_proj"],lora_dropout=0.1,bias="none")).to(device)
txt=get_peft_model(AutoModel.from_pretrained("FacebookAI/xlm-roberta-large"),
    LoraConfig(r=16,lora_alpha=32,target_modules=["query","value"],lora_dropout=0.1,bias="none")).to(device)
ser=net.MultiModalEmotionClassifierDeep(1024,1024,512,5,0.5).to(device)

# SIMPLE setup: 1 optimizer, no scheduler, plain CE
opt=torch.optim.AdamW([{"params":ser.parameters(),"lr":2e-4},
    {"params":[p for p in txt.parameters() if p.requires_grad],"lr":1e-4},
    {"params":[p for p in ssl.parameters() if p.requires_grad],"lr":1e-4}])
crit=nn.CrossEntropyLoss()
ln5=math.log(5)
for ep in range(40):
    ssl.train();txt.train();ser.train(); tot=0;cor=0;ls=0;nb=0
    for (xa,am),(ii,tm),yoh in dl:
        xa,am,ii,tm=xa.to(device).float(),am.to(device).float(),ii.to(device),tm.to(device)
        y=yoh.max(1)[1].to(device)
        a=ssl(xa,attention_mask=am).last_hidden_state
        h=txt(input_ids=ii,attention_mask=tm).last_hidden_state
        lo=ser(a,h); loss=crit(lo,y)
        opt.zero_grad();loss.backward();opt.step()
        ls+=loss.item();nb+=1; cor+=(lo.argmax(1)==y).sum().item(); tot+=len(y)
    if ep%5==0 or ep==39:
        print(f"  epoch {ep:2d}: train_loss {ls/nb:.4f} (ln5={ln5:.4f}) train_acc {cor/tot:.3f}")
print("→ acc→1 : SIMPLE setup fits 40 → bug is script's 3opt/cosine/weighted-CE")
print("→ acc stuck : capacity issue (frozen encoder+LoRA can't fit 40 mixed)")
