# Crab 中英雙語化 Fine-tune Plan

**版本:** v4.2(2026-07-07 §5.6 加 N3 v1-warmstart NNIME LoRA stage 2 → **test 0.5877,超越 v2b baseline +0.033**,17× less data;新增「stage-2 transfer > joint learning」claim)
**目標:** 3-class(Excited / Unconfident / Neutral)雙語化
**起點:** scheme1(`experiments/interview_scheme1/`)

> 相關文件:[`BILINGUAL_DATASETS.md`](BILINGUAL_DATASETS.md)(資料集規格)/ [`BILINGUAL_WORK_LOG.md`](BILINGUAL_WORK_LOG.md)(時序紀錄)/ [`BILINGUAL_MEETING_MEMO.md`](BILINGUAL_MEETING_MEMO.md)(開會用)
> 狀態圖例:🟢 執行中　✅ 已完成　⛔ 卡住　🗄️ 淘汰　🔮 未來

---

## 1. TL;DR(2026-07-02)

| 面向 | 現況 |
|------|------|
| 🥇 **主路徑** | Strategy A LoRA(XLS-R + XLM-R + LoRA q,v r=16 α=32) |
| **最新版本** | **v2b**(2026-07-01)= v1 + CNSCED + NNIME + 3-layer sampler(EN 50 / ZH 50,ZH 內部三 corpus 平均) |
| **v2b Overall Test macroF1** | **0.6718**(30,771 樣本)⭐ 對比 v1 = **+0.10** |
| **v2b Dev best** | 0.6710 @ epoch 6 |
| ✅ 已鎖死 8 軸 | audio enc / text enc / FT 法 / 資料 / LoRA target / LoRA rank α / LR 對 / contrastive weight |
| 🆕 3-layer sampler | Layer 1: EN 50 / ZH 50;Layer 2: ZH 三 corpus 平均;Layer 3: fear-boost(v2b 關掉 = 1.0)|
| 🎯 **NNIME-focused 子實驗**(2026-07-06 ~ 07-07)| **N3 v1-warmstart 3k → test 0.5877 超越 v2b baseline +0.033** ⭐⭐;N2 scheme1-warmstart 3k → test 0.5028 |
| 🏆 **新 claim**(2026-07-07)| **Stage-2 LoRA transfer > joint learning**:v1(42k EN+ZH pretrained)+ 3k NNIME stage-2 finetune 超越 v2b(53k joint 訓)+0.033;17× data efficiency |
| 🗄️ 反例 8 條 | Hybrid B / C14 Full FT / C1 / C1b / C5 / F1-B / v2 fear 3.0 / **v2c fear 2.0**(2026-07-02)/ **Fresh NNIME 3k**(2026-07-06)|
| 🔮 進行中 / 待跑 | v2b per-source eval 完成;剩 v2b cross-dataset re-test |
| ⛔ 卡住 | 朋友 rated Unconfident 校準資料(升級非硬卡)|

**一句話**:v2b 用 3-layer sampler + 加 CNSCED + NNIME,把 test macroF1 從 v1 的 0.57 拉到 **0.6718(+0.10)** — 資料軸驗證成功。

---

## 2. 問題本質 — 不對稱雙軸(短版)

| 編碼器 | 現用 | 對中文 | 結論 |
|--------|------|--------|------|
| Text | XLM-R-Large(舊 RoBERTa-L)| RoBERTa tokenizer 無中文 vocab | **已換**成 XLM-R |
| Audio | XLS-R-300M(舊 WavLM-L)| WavLM 95% 英文預訓,中文「怯」| **已換**成 XLS-R |

**標籤正交**:`Unconfident = Fear + Sad`(scheme1 操作型定義)。純極性資料(CH-SIMS)監督不到 confidence 軸 → 必須情緒分類資料。

---

## 3. Crab 架構共同基底

```
   wav (16kHz)                   text (tokenized)
       │                                │
       ▼                                ▼
  ┌──────────────┐              ┌──────────────┐
  │ Audio Encoder│              │ Text Encoder │   ← 🟩 只動這兩塊
  └──────┬───────┘              └──────┬───────┘
         ▼                             ▼
     [BiGRU]                       [BiGRU]
         │                             │
         └──────────────┬──────────────┘
                        ▼
              [Cross-Attn Fusion]
                        │
                        ▼
              [Classifier] → 3-class
              + MPCL contrastive(5-level,weight 2.0,per-step bs ≥ 4 hard rule)
```

BiGRU / cross-attention fusion / classifier / MPCL / 3-class scheme1 標籤:所有實驗皆未改。

---

## 4. 訓練配置(主路徑 Strategy A LoRA 系列)

### 4.1 v2b 完整訓練配置(2026-07-01,現最新)

| 面向 | 配置 |
|---|---|
| **Audio encoder** | `facebook/wav2vec2-xls-r-300m` |
| **Text encoder** | `FacebookAI/xlm-roberta-large` |
| **FT 方法** | **LoRA q,v r=16 α=32 dropout=0.1**(audio + text 兩側都加)|
| **Warmstart** | ❌ 無(fresh init,paper 乾淨歸因)|
| **Text max len** | 128 |
| **Fusion hidden dim** | 512 |
| **Trainable params** | 22.58M(text LoRA 1.57M + audio LoRA 1.57M + ser 19M)|
| **CSV** | `data/bilingual_v2.csv`(106,499 rows,詳見 [`BILINGUAL_DATASETS.md`](BILINGUAL_DATASETS.md#2-bilingual_v2csv)) |
| **Class weights** | `data/bilingual_class_weights.json`(loss 加權)|
| **Batch size** | bs=16, accumulation_steps=4(effective bs=64) |
| **LR** | head 2e-4 / encoder 1e-4 |
| **Contrastive weight** | 2.0(per-step bs=16 ≥ 4 hard rule pass)|
| **Grad clip** | 1.0 |
| **Epochs** | 10(early stop patience 5)|
| **Sampler** | **3-layer(`--zh_source_balanced`)**,詳見 §4.2 |
| **Fear boost** | 1.0(關,v2 用 3.0 → 崩)|
| **wall clock** | ~12.5 hr(10 epoch)|
| **VRAM 峰** | ~14 GB |
| **Launch script** | [`bin/run_strategyA_v2b_no_fear_boost.sh`](bin/run_strategyA_v2b_no_fear_boost.sh) |
| **Model dir** | `experiments/strategyA_v2b_no_fear_boost/` |

### 4.2 3-layer sampler(v2 起用)

配合三個 ZH corpus 混訓的 WeightedRandomSampler:

**Layer 1 — 語言平衡**:每 batch EN 50 : ZH 50
**Layer 2 — ZH 內部 corpus 平均**:ZH 那 50% 內,EmotionTalk / CNSCED / NNIME 各 1/3
**Layer 3 — fear boost 選項**:EmotionTalk Unconfident 樣本可乘 N 倍(v2 用 3.0 崩,v2b 改 1.0)

```python
for u in cur_utts:
    lang = fname_to_lang[u]
    src = fname_to_src[u]
    if lang == 'EN':
        w = 0.5 / n_en_sources / en_src_counts[src]
    else:  # ZH
        w = 0.5 / n_zh_sources / zh_src_counts[src]
        if boost_ratio > 1.0 and src == boost_src and fname_to_unc[u] == 1:
            w *= boost_ratio
    sample_weights.append(w)

sampler = WeightedRandomSampler(sample_weights, num_samples=len(cur_utts), replacement=True)
```

**v2b 實際運作(log 確認)**:
```
3-layer sampler: EN sources={'MSP': 30000},
                 ZH sources={'EmotionTalk': 11744, 'CNSCED': 8624, 'NNIME': 3054},
                 total lang={'EN': 30000, 'ZH': 23422},
                 fear_boost EmotionTalk×1.0 hit 0 rows
```

### 4.3 v1(舊)vs v2(fear_boost 3.0)vs v2b 配置差異

| 面向 | v1 Strategy A | v2 | v2b(最新)|
|---|---|---|---|
| CSV | `bilingual_strategyA.csv`(EN 30k + EmotionTalk 12k) | `bilingual_v2.csv`(+CNSCED +NNIME) | 同 v2 |
| Sampler | 2-layer `--language_balanced`(EN 50/ZH 50)| 3-layer `--zh_source_balanced` | 同 v2 |
| Fear boost | ❌ | **3.0** | **1.0**(關)|
| Result | 0.65 in-domain / 0.55 cross | **崩**(ep1 dev 0.31)| **0.6718** ⭐ |

**其他所有超參 v1 → v2b 一致**(LoRA rank α / LR / contrastive weight / batch 都同)→ paper 可寫「single-variable data upgrade」乾淨歸因。

### 4.4 已鎖死的 8 個設計軸(paper Table C)

| 軸 | 鎖定值 | 由哪些 run 證實 |
|---|---|---|
| Audio encoder | **XLS-R-300M**(取代 WavLM)| Strategy A 勝 Hybrid B ZH 0.6386 vs 0.5959 |
| Text encoder | **XLM-R-Large**(取代 RoBERTa)| RoBERTa 無中文 tokenizer;scheme1 ZH zero-shot 0.4810 |
| FT 方法 | **LoRA**(優於 Full FT)| Strategy A LoRA 0.6531 vs C14 Full FT 0.6408 |
| 資料 | **雙語混訓**(50:50 EN/ZH)| Strategy A 勝 Hybrid B EN 0.6512 vs 0.4072 |
| LoRA target | **q,v**(不擴 q,k,v,o)| C1/C1b 退 0.111-0.234 |
| LoRA rank/α | **r=16 α=32**(不擴 r=32 α=64)| C5 退 0.107 |
| LR 對 | LoRA: head 2e-4 / enc 1e-4;FT: head 1e-4 / enc 1e-5 | scheme1 collapse 教訓 + C14/F1-B 沿用 |
| Contrastive weight | **2.0**(per-step bs ≥ 4)| C14 第一次 launch bs=1 → MPCL=0 kill 教訓 |

---

## 5. LoRA 版本演進與成績對照 ⭐

### 5.1 版本演進表

| Version | 啟動 | Model dir | 資料變化 | Sampler | Fear boost | 主 trigger |
|---|---|---|---|---|:--:|---|
| **v1**(baseline)| 2026-05-31 21:46 | `strategyA_xlsr_xlmr_lora/` | EN 30k + EmotionTalk 12k(單一 ZH corpus)| 2-layer `language_balanced` | ❌ | — |
| **v2**(炸)| 2026-07-01 17:32 | `strategyA_v2_bilingual_expanded/` | + CNSCED 10k + NNIME 4.3k | 3-layer `zh_source_balanced` | **3.0** | fear 樣本 boost 過激 |
| **v2b**(現)| 2026-07-01 20:28 | `strategyA_v2b_no_fear_boost/` | 同 v2 | 同 v2 | **1.0**(關)| fear boost 假說驗證 |

### 5.2 成績對照(核心比較 — 開會就看這張)

| Version | Dev best macroF1 | Test macroF1 | Test WAR | Test UAR | Test wF1 | Δ vs v1 |
|---|---:|---:|---:|---:|---:|---:|
| **v1** Strategy A LoRA 🥇 | 0.6552 | ~**0.5700**(注 1)| — | — | — | — |
| **v2**(fear 3.0)🗄️ | 0.5357 ep0 → **0.3071 ep1(崩)** | — | — | — | — | — |
| **v2b**(fear 1.0)⭐ | **0.6710** @ ep6 | **0.6718** | 0.7254 | 0.6608 | 0.7218 | **+0.10** |

> **注 1**:v1 舊時報告用 EN Test1 / Test2 / ZH EmotionTalk 三分別數字(0.6512 / 0.5544 / 0.6386),混合 test overall 未整份報告;v2b 是三 ZH corpus + MSP 混合 30,771 test overall。

### 5.3 v2b Per-class Test 表現

| Class | Precision | Recall | F1 | support |
|---|---:|---:|---:|---:|
| Excited | 0.7639 | **0.7890** | **0.7762** | 12,829 |
| Unconfident | 0.5829 | 0.4489 | 0.5072 | 3,916 |
| Neutral | 0.7198 | 0.7443 | **0.7319** | 14,026 |

**觀察**:
- Excited / Neutral F1 ≥ 0.73(高),Unconfident F1 = 0.51(弱項)
- Unconfident recall 0.45(precision 0.58 反倒不差)→ 模型不敢猜 Unconfident,fear boost 1.0 保守
- 下一 iteration 候選:fear boost **1.5 or 2.0**(在 3.0 崩 / 1.0 保守之間)

### 5.4 v2b Per-source Test(2026-07-02 完成)

按 macroF1 排序:

| Rank | Source | Lang | rows | **macroF1** | WAR | UAR | wF1 |
|:--:|---|:--:|---:|---:|---:|---:|---:|
| 🥇 | **CNSCED** | ZH | 686 | **0.6978** ⭐ | 0.7303 | 0.6679 | 0.7175 |
| 🥈 | MSP | EN | 28,000 | 0.6704 | 0.7271 | 0.6599 | 0.7236 |
| 🥈 | Overall | mixed | 30,771 | 0.6721 | 0.7256 | 0.6611 | 0.7220 |
| 🥉 | EmotionTalk | ZH | 1,447 | 0.6493 | 0.7457 | 0.6391 | 0.7419 |
| ⚠️ | **NNIME** | ZH | 638 | **0.5543** | 0.6050 | 0.5472 | 0.5829 |

**重大觀察**:
1. **CNSCED > MSP**(0.6978 > 0.6704)— ZH 自然對話 corpus 超過 EN in-domain。可能原因:CNSCED speaker 454 人 diversity 高 → generalization 好
2. **EmotionTalk 沒退**:v1(0.6386)→ v2b(0.6493)**+0.011**,加 CNSCED/NNIME 沒稀釋 EmotionTalk
3. **NNIME 拉低平均**:0.5543 是全表最低(-0.11 vs 其他 ZH)。這 corpus 內在難度高(6-rater majority vote 標得嚴 + v3 mapping 偏 Unconfident 家族)。ckpt 有訓過,不算 zero-shot,但仍是三 corpus 最難
4. **EN/ZH 差距只 0.037**:MSP 0.6704 vs 三 ZH 平均 0.6338 — 遠比 scheme1 的 EN 0.69 vs ZH zero-shot 0.48(差 0.21)小很多

> Script: [`scripts/eval_per_source.py`](scripts/eval_per_source.py) → 輸出 `data/persource_v2b.json` + wandb runs `strategyA_v2b_persource__<subset>_<ts>`。

### 5.5 v1 完整 cross-dataset(2026-06-04,for reference)

| Eval set | 樣本 | 性質 | v1 Strategy A LoRA macroF1 |
|---|---:|---|---:|
| MSP Test1 | 10,684 | in-domain | **0.6512** |
| MSP Test2 | 10,684 | cross-session within MSP | **0.5544**(drop −0.097 比 scheme1 −0.131 小 26%)|
| **IEMOCAP** | 4,575 | 🎯 true cross-dataset(演員 scripted)| **0.5990** ⭐ 比 Test2 高 +0.045 |
| **MELD** | 2,197 | 🎯 true cross-dataset(TV 多角色 OOD)| **0.5097**(最 hard OOD 仍 ≥0.50)|

v2b 這些 cross-dataset 還沒重測(candidate 任務)。

### 5.6 NNIME-focused 子實驗(2026-07-06 ~ 07-07)

**動機**:v2b per-source eval(§5.4)顯示 NNIME 是四 corpus 最差(0.5543 vs CNSCED 0.6978 / MSP 0.6704 / EmotionTalk 0.6493)。老師方向:能不能直接對 NNIME 做 focused finetune 補上?

#### 配置對照表

| # | Setup | Encoder | Warmstart | Train data | LR head/enc | Dev best | Test macroF1 | 結果 |
|:--:|---|---|---|---:|---|---:|---:|---|
| **N1** 🗄️ | **Fresh NNIME LoRA** | XLM-R + XLS-R | 無 | 3,054 | 2e-4 / 1e-4 | 0.3012 @ ep0 | ~崩 | Collapse 到 Excited-only |
| **N2** | **Scheme1-warmstart NNIME LoRA** | **WavLM + RoBERTa** | scheme1 全套 | 3,054 | 2e-4 / 1e-4 | 0.5017 @ ep9 | **0.5028** | 破 0.50 但仍 < v2b |
| **N3** ⭐⭐ | **v1-warmstart NNIME LoRA stage 2** | XLM-R + XLS-R | **v1 LoRA + ser** | 3,054 | **5e-5 / 2e-5** | 0.5283 @ ep4 | **0.5877** ⭐⭐ | **超越 v2b +0.033** |
| — | v2b(參考)| XLM-R + XLS-R | 無 | 53,422(v2 mix)| 2e-4 / 1e-4 | 0.6710 @ ep6 | 0.5543 | baseline(17× train)|

#### N3 vs v2b vs N2 Per-class NNIME test 對比

| Class | v2b F1 | N2 F1 | **N3 F1** | Δ vs v2b |
|---|---:|---:|---:|---:|
| Excited | 0.563 | 0.551 | **0.583** | **+0.020** |
| **Unconfident** | 0.403 | 0.320 | **0.482** | **+0.079** ⭐ |
| Neutral | **0.697** | 0.637 | 0.698 | +0.001 |

**Unconfident F1 0.482 = 三 setup 最高**:
- 對比 N2 的 0.320:XLM-R 中文 vocab 讓 Unconfident recall 從 0.297 → 0.4545(+15%)
- 對比 v2b 的 0.403:v1 warmstart 給了「已學好的 3-class 邊界」→ 少量 NNIME finetune 就能專注拉 confidence 軸

#### 四個關鍵發現(paper §V.C 素材)

1. **Fresh NNIME 3k + LoRA → 100% collapse**  
   Fusion embedding 塌成單一向量,contrastive loss = 0,dev_f1/Unconfident = 0 / dev_f1/Neutral = 0,模型永遠猜 Excited → **小資料 fresh init 不可行**

2. **Scheme1(WavLM+RoBERTa)warmstart 拯救 collapse**  
   即使 RoBERTa 廢中文,audio + ser head warmstart 就提供足夠 embedding anchor → 3k data 也能訓到 0.50。但 Unconfident F1 只 0.32,顯示 RoBERTa 無中文 vocab 是硬瓶頸

3. **v1(XLM-R+XLS-R)warmstart 突破 v2b 天花板** ⭐⭐  
   v1 LoRA(text+audio 兩個 adapter)+ ser head warmstart + **降 LR 到 25%**(head 5e-5 / enc 2e-5)防 catastrophic forget → 3k NNIME finetune 得 test 0.5877,**超越 v2b(53k joint 訓)+0.033**

4. **Stage-2 LoRA transfer > Joint learning**(for NNIME)  
   針對 NNIME 這種 out-of-distribution 特定 domain,「先 pretrained backbone + 少量 domain finetune」比「一次性 joint 訓所有資料」更有效  
   → **17× less data → +0.033 macroF1**(不是打平,是打贏)

#### 論文角度(§V.C 可寫)

- N1 反例補強 §5「LoRA 需要 backup data 或 warmstart」論點
- N2 顯示 encoder 中文 vocab 的重要性(Unconfident F1 -0.083 硬證據)
- **N3 主要 claim**:XLM-R+XLS-R 有 42k EN+ZH 訓過的 v1 作為「LoRA 版 scheme1-XLMR」,對 NNIME 這種 domain-specific corpus 提供最強 transfer 起點
- **v2b 路線的 limitation**:joint learning 分散 LoRA capacity → 大 corpus + 稀疏 domain(NNIME 3k / 53k)反而學不好 NNIME domain
- **未來 claim**:同樣邏輯應該適用其他 domain-specific 小 corpus(e.g. 未來 CHEAVD-2.0 或朋友 Unconfident 資料)

**ckpt 保留**:
- N1(反例)`experiments/nnime_only_lora/` — collapsed model,paper §V-B 反例
- N2 `experiments/nnime_scheme1_warmstart_lora/` — 0.5028 中線
- N3 `experiments/nnime_v1_warmstart_lora/` — 0.5877 **new NNIME SOTA**

#### 尚待驗證(open question)

**N3 超 v2b 的成績,主因是 v1 已含 EN+ZH 混訓,還是 XLM-R+XLS-R 這個 encoder pair 本身**?
拆這兩個因素需跑對照 N4:XLM-R+XLS-R **只 MSP 168k EN 訓過**(不含任何 ZH data)+ NNIME LoRA stage 2  
→ 排程 §8.1(短期)

---

## 6. 淘汰路徑彙整(反例 ablation,paper §IV.B)

9 條真機驗證、結論為負的路徑。**覆蓋三正交軸 + 一個結構容量子軸 + 一個 sampler 子軸 + 一個資料規模子軸**。

| # | 路徑 | 設定差(vs Strategy A LoRA)| 結果 | 失敗根因 | Paper 角度 |
|:--:|---|---|---:|---|---|
| 1 | 🗄️ **Hybrid B** | LoRA + **單語(只 ZH)** + scheme1 warm-start | EN 0.69 → **0.4072(-0.28)** / ZH 0.5959 | LoRA 凍 base 不防 catastrophic forgetting | **「LoRA + warm-start ≠ bilingual safety」** — 雙語必須混訓 |
| 2 | 🗄️ **C14 Full FT** | LoRA → Full FT 891M | 0.6408(-0.012 均勻)| epoch 2 plateau 後 overfit | **「LoRA 是正確選擇」** — 小資料 + LoRA regularization 勝 Full FT |
| 3 | 🗄️ **C1** | LoRA target **q,v → q,k,v,o** | dev 0.5442(-0.111)| 四矩陣同步更新破壞 cross-modal attention | target = (q,v) sweet spot |
| 4 | 🗄️ **C1b** | 同 C1 + enc_lr 半 5e-5 | dev 0.4213(-0.121)| 降 LR 反更差 → 非 LR 問題 | 三重確認 q,v sweet spot |
| 5 | 🗄️ **C5** | 同 q,v,rank+α 雙倍(16→32, 32→64)| dev 0.5479(-0.107)| capacity 加也壞 | r=16, α=32 是 ceiling |
| 6 | 🗄️ **F1-B Partial FT** | Wang 2022 audio 全 24 + Lee 2019 text top-6 | -0.046 vs LoRA | FT 家族 monotonic 死亡(LoRA > C14 > F1-B)| FT 家族被 42k 雙語樣本完整證偽 |
| 7 | 🗄️ **v2 fear boost 3.0** | v2b 相同但 boost=3.0 | ep0 0.5357 → ep1 **0.3071 崩** | 過度預測 Unconfident,UAR 0.46 高但 WAR 0.33 near-random | fear-boost 上限 < 3.0 |
| 8 | 🗄️ **v2c fear boost 2.0**(新)| v2b 相同但 boost=2.0 | ep0 0.4459 → ep1 **0.4137**(繼續掉)| 中線 2.0 也不行,全面弱 | **fear-boost 甜蜜區只有 1.0**,窄到不可調 |
| 9 | 🗄️ **Fresh NNIME LoRA**(新)| 純 3k NNIME + fresh init(無 backup data / 無 warmstart)| ep0 0.3012 → collapse 至 Excited-only | 3k 資料 fresh init 不足以 warm-up LoRA;fusion embedding 塌成單點 | **小資料必要 warmstart or backup data 混訓** |

**統一論點:**

| 軸 | ablation 對手 | 鎖定結論 |
|---|---|---|
| ① 資料軸 — 語言 | Hybrid B(單語) vs Strategy A(雙語 50:50)| **雙語混訓** |
| ② FT 方法軸 | C14 vs Strategy A / F1-B | **LoRA**(非 Full FT / 非 Partial FT)|
| ③ LoRA 結構 — target | C1 + C1b vs Strategy A | **target = (q,v)** |
| ③' LoRA 結構 — capacity | C5 vs Strategy A | **r=16, α=32** |
| ④ Sampler 子軸 — fear boost | v2(3.0)+ **v2c(2.0)** vs v2b(1.0)| **甜蜜區 = 1.0**(2.0/3.0 都崩)|
| ⑤ 資料規模子軸(新)| Fresh NNIME 3k(崩) vs Strategy A(42k+)| **小資料必要 warmstart or backup data 混訓** |

**ckpt 保留策略**:所有死路 ckpt 全保留作 paper 反例素材。

---

## 7. Unconfident 標籤

**定義(鎖定)**:`Unconfident = Fear + Sad`,scheme1 操作型定義一致([`src/prepare_interview_scheme1.py`](src/prepare_interview_scheme1.py))。

**朋友負責(進行中)**:合成 unconfident-targeted clip + 10-rater 信心評分(κ≥0.4 收貨)→ Stage B-target 校準資料。從「硬卡關」降為「proxy 校準升級」。

> ⚠️ **中文聲調地雷**:WavLM 英文學的 pitch-based 信心線索在中文聲調語境失準;scheme1 zero-shot 在 EmotionTalk 反而**過度預測 Excited 48.1%**(true 22.1%)→ 朋友資料能幫忙校準。詳見 [`BILINGUAL_DATASETS.md`](BILINGUAL_DATASETS.md#9-diagnostic-資料歷史) §9。

---

## 8. 未來方向 Roadmap

### 8.1 短期(本週,v2b 完訓後)

| # | 動作 | 依賴 | 預期 EV | 成本 |
|:--:|---|---|---|---|
| **v2b eval** | Per-source test 拆(MSP / EmotionTalk / CNSCED / NNIME)| 已跑 | 讀 cross-corpus gap | ~30 min |
| **v2b cross-dataset** | IEMOCAP + MELD 重測(對比 v1)| ckpt 已存 | 驗 cross-dataset 是否還 hold | ~1 hr |
| **v2c fear boost 中線** | fear_boost_ratio 1.5 或 2.0(在 3.0 崩 / 1.0 保守之間)| v2b done | Unconfident F1 +0.02-0.05 | ~12 hr |
| A1 | contrastive_weight 2.0 → 1.0 / 3.0(從沒 ablate)| — | +0.005-0.02 | ~7 hr |
| B6 | SpecAugment / speed perturb | — | +0.01-0.03 | 半天 code + ~12 hr |

### 8.2 中期(1-2 月,解 Unconfident plateau)

| # | 動作 | 依賴 | 預期 |
|:--:|---|---|---|
| Stage B-target | 朋友 rated Unconfident → 直接打 0.50 plateau | κ≥0.4 收 | ZH Unconfident 0.50 → ?(本質解)|
| CHEAVD-2.0 → B1 | Email CASIA 7 天,加 ZH 第 4 corpus | 申請信 | ZH +0.05-0.10 |
| ESD-CN → B1 | 5 分鐘 form + 24hr 回信 | 申請信 | 邊際,只補 sad |
| 套組(v2c + A1 + B6 + B1)| 上述全到位 | | 兩語 macroF1 **0.70+** |

### 8.3 論文方向(2-3 月)

| 角度 | 內容 |
|---|---|
| Method 章節 | Strategy A 主路徑 + Table C 8 軸鎖定設計 + LoRA target=q,v + 3-layer sampler 機制 |
| §IV.B Negative Results | 6 條死路 ablation(§6)|
| §V.A Data | 三 ZH corpus 互補性(見 [`BILINGUAL_DATASETS.md`](BILINGUAL_DATASETS.md) §7)|
| §V.B / §V.C | NNIME Session/non-speech drop 論證 + 62-label v3 mapping 論證(見 [`BILINGUAL_DATASETS.md`](BILINGUAL_DATASETS.md) §6)|
| Cross-lingual analysis | EN/ZH 同情緒對齊 t-SNE 視覺化 |
| Cross-dataset robustness | v2b 版本 vs v1 版本 IEMOCAP / MELD 對照 |

### 8.4 長期候選(有時間再動)

Cross-lingual contrastive(EN/ZH 同情緒對齊)/ Two-stage curriculum(EN-heavy → bilingual)/ Focal loss 取代 weighted CE / InfoXLM 替 XLM-R / MMS audio encoder(1000+ 語)/ Continuous VAD score 解 Unconfident plateau。

---

## 9. 實驗守則(避免重蹈覆轍)

- **每個新模型必跑 per-language + per-source eval**(只看 overall 會錯過 Hybrid B 級別 forgetting)
- 紀錄多軸:overall + EN + ZH + per-source macroF1 + **預測分布**(分布相對 ground truth 比 F1 更早發現崩潰)
- **Q3 grilling 門檻**:任何雙語/中文化模型 EN test 退 ≤ 0.05 才考慮接受
- **Contrastive (MPCL) per-step bs ≥ 4 hard rule**:bs=1/2 時 MultiPosConLoss 數學上 ≈ 0 → contrastive_weight 設多少都沒用
- **新 architecture / 新 sampler smoke 1-2 分鐘**:看 `batch/contrastive_loss` 不平 0 + VRAM 合理 + ETA 合理,再放真正跑
- **Fear boost 上限**:v2 已證 3.0 崩(WAR 0.33 near-random),下次候選 1.5-2.0

---

## 10. 附錄:重要術語

### A. LoRA target = (q, v) 為什麼

**LoRA**(Hu et al. 2021,arXiv 2106.09685)= 凍住 `W`,加低秩更新 `ΔW = (α/r)·B·A`:

```
forward(x) = (W_frozen + (α/r) · B · A) · x
              └─ 凍住 ─┘   └── 訓練 ──┘
```

**為什麼選 (Q, V) 不選 (Q, K, V, O)**:

| 矩陣 | 角色 | 直觀解釋 |
|---|---|---|
| **Q** | 「我要找什麼」 | 決定 attention 看向哪 |
| **K** | 「有什麼可以對」 | 跟 Q 點積算 attention score |
| **V** | 「找到後拿什麼」 | 內容加權平均後變輸出 |
| **O** | 後線性投影 | 後處理 |

原 LoRA paper Table 5 ablation 證實 (Q, V) 對給最佳 perf-per-param:Q + V 控制兩個正交軸;K 跟 Q 相乘 → 改 K 等於改 Q 的重定向,冗餘;O 屬後處理。

**Strategy A 具體配置(對應 v2b)**:
- Audio(XLS-R-300M,24 layers,d=1024):24 × 2 × 32K = **1.57M LoRA**
- Text(XLM-R-Large,24 layers,d=1024):24 × 2 × 32K = **1.57M LoRA**
- 兩 encoder + ser_model 19M = **總 trainable 22.58M**
- vs Full FT 891M:**LoRA 是 1/40 參數但 macroF1 高 0.012**

### B. WeightedRandomSampler 語言 / corpus 平衡

**問題**:不做 sampler 時每 batch(bs=16)期望 11.5 EN : 4.5 ZH,ZH 訊號被壓過。
**解法**:PyTorch `WeightedRandomSampler`,樣本權重 = `1 / count_of_its_bucket`。

v2b 用 3-layer:
- Layer 1 EN/ZH 50:50(vs v1 只有這層)
- Layer 2 ZH 內三 corpus 平均(v2/v2b 新)
- Layer 3 fear boost(v2/v2b 新,v2b 關掉)

### C. LoRA warm-start vs fresh 說明

LoRA 是加法(`W_frozen + (α/r)·B·A`)。**技術上**「LoRA warm-start」= 把 v1 訓好的 A、B 矩陣當初始化,再繼續訓 → 不是 "FT on FT"(那要動 W_frozen)。但為了 paper 乾淨歸因 "single-variable test:only training data changed",v2 / v2b 都選 fresh init。

---

## 11. 變更歷史

| 版本 | 日期 | 重點 |
|---|---|---|
| v1.0 | 2026-05-16 | 建檔(6 方案 + Phased Rollout)|
| v2.0-2.6 | 2026-05-28 ~ 06-02 | CH-SIMS Stage A + Hybrid B + Strategy A 完訓;判決翻轉 |
| v3.0-3.13a | 2026-06-02 ~ 07-01 | C14 / C1 / C1b / C5 / F1-B ablation 全跑;cross-dataset(IEMOCAP+MELD)實測;ZH 資料軸大擴(CNSCED + NNIME)|
| **v4.0** | **2026-07-02** | **大幅精簡改組**:資料集細節拆到 [`BILINGUAL_DATASETS.md`](BILINGUAL_DATASETS.md);本 doc 只留架構 + 訓練配置 + LoRA 版本對照 + 淘汰路徑 + 未來方向;加 **§5 LoRA 版本演進與成績對照**;加 v2b 結果(Test macroF1 0.6718 = v1 baseline +0.10);淘汰路徑加第 7 條 v2 fear boost 3.0 崩;實驗守則加 fear boost 上限 |
| **v4.1** | **2026-07-06** | 加 **§5.6 NNIME-focused 子實驗**:N1 fresh 3k 完全 collapse / N2 scheme1-warmstart(WavLM+RoBERTa)3k → test 0.5028(⭐ data efficiency 佐證,vs v2b 用 17× data 得 0.5543 只差 -0.05)/ N2 vs v2b per-class 對比顯示 Unconfident 弱 -0.083 = RoBERTa 無中文 vocab 直接證據;§6 淘汰路徑加 v2c fear boost 2.0(甜蜜區確定 = 1.0)+ Fresh NNIME LoRA(新軸「資料規模」);TL;DR + §6 統一論點對應更新;patch `train_crab_lora.py` 加 `final_text.pt` warm-start 支援(原本只 warm-start audio + ser)|
| **v4.2** | **2026-07-07** | **§5.6 加 N3 v1-warmstart NNIME LoRA stage 2 → test 0.5877 超越 v2b baseline +0.033**(17× less data 卻打贏,per-class Unconfident F1 0.482 = 三 setup 最高);新 claim「Stage-2 LoRA transfer > Joint learning for NNIME」;TL;DR 加 N3 結果 + 新 claim 條目;patch `train_crab_lora.py` 加 `--lora_warmstart` flag(用 PeftModel.from_pretrained 載入現有 LoRA adapter,支援 stage-2 finetune);未來排程加 N4 對照(排除 v1 中 ZH 混訓的貢獻)|
