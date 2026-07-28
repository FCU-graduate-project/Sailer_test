"""MSP-Podcast 8類 → 奧客遊戲 4類(Angry/Happy/Neutral/Anxious),平衡子採樣。
  Angry = Angry+Disgust+Contempt | Happy = Happy+Surprise | Neutral = Neutral | Anxious = Sad+Fear
輸出: data/okeke_msp_4class.csv(FileName 絕對路徑, Text, Split_Set, 4 one-hot)
       data/okeke_msp_4class_weights.json
子採樣是為了「控訓練時間 + CPU/RAM 安全」,非記憶體 preload(訓練走 LazyWavSet 懶載)。
"""
import json, pandas as pd
from pathlib import Path

CRAB = Path("/home/brant/Project/SAILER_test/Crab")
SRC  = CRAB / "data" / "msp2_processed_labels.csv"
WAV  = Path("/home/brant/Project/SAILER_test/datasets/MSP_Podcast_Data/Audios")
OUT  = CRAB / "data" / "okeke_msp_4class.csv"
WJS  = CRAB / "data" / "okeke_msp_4class_weights.json"
SEED = 42
CAP  = {"Train": 10000, "Development": 1500, "Test": 1500}   # per-class 上限(控時間/CPU)
GROUP = {"Angry":["Angry","Disgust","Contempt"], "Happy":["Happy","Surprise"],
         "Neutral":["Neutral"], "Anxious":["Sad","Fear"]}
CLASSES = list(GROUP)

df = pd.read_csv(SRC)
df = df[df["Text"].notna() & (df["Text"].astype(str).str.strip() != "")]
df["Split_Set"] = df["Split_Set"].replace({"Test1": "Test"})      # Test1→Test
df = df[df["Split_Set"].isin(["Train","Development","Test"])].copy()  # 丟 Test2
for c, src in GROUP.items():
    df[c] = (df[src].sum(axis=1) > 0).astype(int)
df = df[df[CLASSES].sum(axis=1) == 1].copy()                       # 保乾淨單標籤
df["FileName"] = df["FileName"].apply(lambda f: str(WAV / f))
df["_cls"] = df[CLASSES].idxmax(axis=1)

rows = []
for split, cap in CAP.items():
    sub = df[df["Split_Set"] == split]
    for cls in CLASSES:
        g = sub[sub["_cls"] == cls]
        rows.append(g.sample(min(cap, len(g)), random_state=SEED) if len(g) > cap else g)
out = pd.concat(rows).sample(frac=1, random_state=SEED)[["FileName","Text","Split_Set"]+CLASSES]
out.to_csv(OUT, index=False)

# class weights(以 Train 分布算,inverse-freq,mean=1)
tr = out[out["Split_Set"]=="Train"]
n = len(tr); k = len(CLASSES)
cw = {c: (n/(k*tr[c].sum())) if tr[c].sum() else 0.0 for c in CLASSES}
m = sum(cw.values())/k
cw = {c: v/m for c,v in cw.items()}
WJS.write_text(json.dumps({"class_weight": cw}, ensure_ascii=False, indent=2))

print("OUT:", OUT, "rows:", len(out))
for split in ["Train","Development","Test"]:
    s = out[out["Split_Set"]==split]
    print(f"  {split:12} {len(s):6}  ", {c:int(s[c].sum()) for c in CLASSES})
print("weights:", {c: round(v,3) for c,v in cw.items()})
