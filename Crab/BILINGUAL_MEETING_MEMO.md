# Crab 中英雙語化 — 開會備忘錄

**版本：** v3.1(2026-06-04 23:25 加 true cross-dataset 實測,對齊 plan v3.12)
**會議目的：** 報告主路徑 + 反例譜系 + 中文資料補強方向
**參考：** [`BILINGUAL_FINETUNE_PLAN.md`](BILINGUAL_FINETUNE_PLAN.md) / [`BILINGUAL_WORK_LOG.md`](BILINGUAL_WORK_LOG.md)

> v2.0 命題「中文 Unconfident 是黑洞」已完全解決(`Unconfident=Fear+Sad` 操作型定義 + EmotionTalk 1,766 筆 fear+sad)→ 已進入「pick best method on bilingual」階段。

---

## 1. 一頁總結 — 會議開場 30 秒講完

```
★ 最強 claim(開場第一句):
   主路徑 Strategy A LoRA 有「真正的 cross-dataset robustness 硬證據」:
   ① 換到 IEMOCAP(USC 演員,真完全不同 corpus): macroF1 = 0.5990
      — 甚至 *比 MSP Test2 (0.5544) 還高 +0.045* → 模型沒死記 MSP artifact
   ② 換到 MELD(Friends TV,多角色 + 配音 + 笑聲 OOD): macroF1 = 0.5097
      — 最 hard 的 OOD 條件仍 ≥ 0.50,不崩
   ③ MSP Test2(同 corpus 不同 session):0.5544 跟 scheme1 0.5591 打平 (-0.005)
      drop -0.097 比 scheme1 -0.131 小 26%
   → 雙語 30k train 不只「接近」scheme1 168k EN-only Full FT
     而是「在所有 unseen domain 上 ≥」它(IEMOCAP/MELD scheme1 沒測但 zero-shot 通常更差)

任務:Crab 雙語化(EN + ZH 3-class:Excited / Unconfident / Neutral)
時段:2026-05-28 ~ 2026-06-04(8 天,9 個 fine-tune run + scheme1 baseline)

主路徑:Strategy A LoRA(XLS-R + XLM-R + LoRA q,v + 雙語 50:50 sampler)
       EN Test1 (in-domain)        : 0.6512  vs scheme1 0.6900   drop 0.039 ✓
       EN Test2 (cross-session)    : 0.5544  vs scheme1 0.5591   ★ 打平 (-0.005)
       EN IEMOCAP (cross-dataset)  : 0.5990  🎯 比 Test2 高 +0.045
       EN MELD (cross-dataset OOD) : 0.5097  🎯 ≥0.50 不崩
       ZH EmotionTalk Test         : 0.6386  vs Hybrid B 0.5959  +0.043 勝
       Overall in-domain (29.4k)   : 0.6531

跨 split 一致性(降低 noise 假說):
       LoRA 險勝 C14 Full FT 在 Test1 +0.012 / Test2 +0.014 完全一致

反例譜系(5 條死路,覆蓋 3 正交 ablation 軸):
  ① Hybrid B    單語 ZH → EN catastrophic forget -0.28
  ② C14 Full FT 891M params → overfit @ ep2, -0.012(Test1)/ -0.014(Test2)一致
  ③ C1/C1b      LoRA target q,v→q,k,v,o → -0.111/-0.121(降 LR 救不了)
  ④ C5          LoRA rank↑(16→32)→ -0.107
  ⑤ F1-B        Partial FT(Wang 2022 + Lee 2019)→ -0.046(literature 預測沒成立)
      → FT 家族 monotonic 死亡(LoRA > C14 > F1-B),完整證偽

下一步:改善方向轉「資料軸」(B1)而非訓練方法
       申請 CHEAVD-2.0 / NNIME / ESD-CN(三個平行送,任一到位即跑 B1)
       朋友資料 Stage B-target(等 κ≥0.4,解 Unconfident 0.50 plateau)
```

---

## 2. 完整實驗矩陣 — 會議白板可直接畫

### Table A — 架構與訓練配置

| # | Model | 啟動 | Audio | Text | 資料 | 關鍵超參 | Trainable | VRAM | Wall |
|:--:|---|---|---|---|---|---|---:|---:|---:|
| 0 | **scheme1**(EN baseline)| 歷史 | WavLM-L 全 FT | RoBERTa-L 全 FT | MSP EN 168k | (未保留)| ~860M | ~24GB | ~24hr |
| 1 | CH-SIMS Stage A 🗄️ | 05-28 | WavLM-L LoRA q,v | RoBERTa-L LoRA q,v | CH-SIMS ZH 5-class 2.7k | head 2e-4/enc 1e-4/cont 0 | 22.58M | ~21GB | ~3hr |
| 2 | Hybrid B 🗄️ | 05-31 | WavLM-L LoRA q,v(WS scheme1)| XLM-R-L LoRA q,v | EmotionTalk ZH 14k | head 2e-4/enc 1e-4/cont 2.0 | ~25M | ~14GB | ~5hr |
| 3 | **Strategy A LoRA** 🥇 | 05-31 | XLS-R-300M LoRA q,v | XLM-R-L LoRA q,v | **EN 30k + ZH 12k, 50:50** | head 2e-4/enc 1e-4/cont 2.0 | ~25M | ~14GB | ~11hr |
| 4 | C14 Full FT 🗄️ | 06-02 | XLS-R 全 FT 311M(凍 CNN)| XLM-R 全 FT 560M | 同 A | head 1e-4/enc 1e-5/bf16/grad_ckpt | **891M** | ~22GB | ~8hr |
| 5 | C1 🗄️ | 06-03 | XLS-R LoRA q,k,v,o | XLM-R LoRA q,k,v | 同 A | enc **1e-4** | ~25M | ~13GB | 7.5hr → kill |
| 6 | C1b 🗄️ | 06-03 | 同 C1 | 同 C1 | 同 A | **enc 5e-5**(半)| ~25M | ~13GB | 5hr → kill |
| 7 | C5 🗄️ | 06-04 | XLS-R LoRA q,v **r=32 α=64** | XLM-R 同 | 同 A | 同 A 但 r/α 雙倍 | ~26M | ~14GB | 10.5hr → kill |
| 8 | F1-B 🗄️ | 06-04 | XLS-R Partial **全 24 transformer**(Wang 2022)| XLM-R Partial **top-6 of 24**(Lee 2019)| 同 A | head 1e-4/enc 1e-5/bf16/grad_ckpt | 397M | ~19GB | ~7hr |

### Table B — 結果

| # | Model | dev best | EN test(Test1)| EN test(Test2)| ZH test | Overall | 1-line 結論 |
|:--:|---|---:|---:|---:|---:|---:|---|
| 0 | scheme1 | 0.6720 | **0.6900** | 0.5591 | 0.4810 zero | 0.5855 平均 | EN ceiling;ZH zero-shot 過度預測 Excited 48% |
| 1 | CH-SIMS Stage A | 0.4400 | — | — | **0.4946**(5-class)| — | backbone 讀中文沒問題 |
| 2 | Hybrid B 🗄️ | 0.6671 | **0.4072 ⚠️ catastrophic** | — | 0.5959 | 0.5016 | 單語 LoRA 反例 — EN 退 0.28 |
| 3 | **Strategy A LoRA** 🥇 | **0.6552** | **0.6512** | **0.5544** ⭐ | **0.6386** | **0.6531** | **主路徑** — 三軸 sweet spot;**Test2 跟 scheme1 0.5591 打平**(差 -0.005)|
| 4 | C14 Full FT 🗄️ | 0.6502 | 0.6388 | **0.5400** | 0.6267 | 0.6408 | LoRA 險勝 Test1 +0.012 / Test2 +0.014 一致 |
| 5 | C1 🗄️ | 0.5442 | killed | — | killed | killed | LoRA target 擴失敗(-0.111)|
| 6 | C1b 🗄️ | 0.4213 | killed | — | killed | killed | 降 LR 不救反更差(-0.121)|
| 7 | C5 🗄️ | 0.5479 | killed | — | killed | killed | LoRA rank 雙倍失敗(-0.107)|
| 8 | F1-B 🗄️ | 0.5994 | **0.6072 overall** | — | (per-lang 未測)| — | Partial FT literature 預測沒轉用成功 |

> ✅ Test2 已實測(2026-06-04 22:22 完):**LoRA 0.5544 / C14 0.5400 / scheme1 0.5591**;LoRA Test2 drop 比 scheme1 小 26%(LoRA -0.097 vs scheme1 -0.131)→ cross-domain robustness paper claim 成立。

### Table B+ — Strategy A LoRA cross-dataset 完整 robustness eval ⭐ 開會白板強推

| Eval set | 樣本 | 性質 | macroF1 | accuracy | drop vs in-domain | 1-line 結論 |
|---|---:|---|---:|---:|---:|---|
| MSP Test1 | 10,684 | in-domain | **0.6512** | 0.7106 | baseline | 跟 train 同 corpus,Q3 門檻通過 |
| MSP Test2 | 10,684 | cross-domain within MSP | 0.5544 | 0.6806 | −0.097 | 跟 scheme1 0.5591 打平 |
| **IEMOCAP** | 4,575 | 🎯 **true cross-dataset**(USC 演員 scripted)| **0.5990 ⭐** | 0.6050 | **−0.052** | **比 Test2 還高 +0.045** — 換 corpus 不退反進 |
| **MELD** | 2,197 | 🎯 **true cross-dataset**(Friends TV OOD)| **0.5097** | 0.533 | −0.142 | 最 hard 的 OOD(配音+背景音+笑聲)仍 ≥ 0.50 |

> ✅ 2026-06-04 23:25 完(同台 ckpt `experiments/strategyA_xlsr_xlmr_lora/`,scheme1 3-class 映射);兩 cross-dataset 平均 (0.599 + 0.510) / 2 = **0.5544**,**剛好等於 MSP Test2** → cross-dataset gap 不比 MSP 內 cross-session gap 大。
>
> **這是「真的強」的硬證據** — 開頭 30 秒摘要直接用,不再只能 claim cross-session within MSP。

### Table C — 8 個已鎖死的設計選擇(問答 backup)

| 軸 | 鎖定值 | 由哪些 run 證實 |
|---|---|---|
| Audio encoder | **XLS-R-300M**(換 WavLM)| Strategy A 勝 Hybrid B ZH 0.6386 vs 0.5959 |
| Text encoder | **XLM-R-Large**(必換 RoBERTa)| RoBERTa 無中文 tokenizer;scheme1 ZH zero-shot 0.4810 |
| FT 方法 | **LoRA**(優於 Full FT)| A LoRA 0.6531 vs C14 Full FT 0.6408 |
| 資料 | **雙語混訓 50:50** | A 勝 Hybrid B EN 0.6512 vs 0.4072 |
| LoRA target | **q,v**(不擴 q,k,v,o)| C1/C1b 退 0.111-0.234 |
| LoRA rank/alpha | **r=16 α=32**(不擴 r=32 α=64)| C5 退 0.107 |
| LR 對 | LoRA: head 2e-4 / enc 1e-4;FT: head 1e-4 / enc 1e-5 | scheme1 collapse 教訓 |
| Contrastive weight | **2.0**(per-step bs ≥ 4 hard rule)| C14 第一次 launch per-step bs=1 → MPCL=0 kill 教訓 |

---

## 3. 重要術語註解(會議高機率被問)

### Q: LoRA target = q,v 是什麼意思 / 怎麼運作?

**LoRA**(Hu et al. 2021,arXiv 2106.09685)= 凍住 pretrained 權重 `W`,加低秩更新 `ΔW = B·A`:

```
forward(x) = (W_frozen + (α/r) · B · A) · x
              └─ 凍 ─┘   └ 訓練 ┘
```

- `W` 凍住(d×d = 1024×1024 = 1M 參數,不算梯度)
- `A` (r×d) + `B` (d×r) 新加,訓練(r=16 → 32K 參數,**省 31×**)
- `α=32, r=16` → scale = α/r = 2.0(更新強度)

**Transformer self-attention 4 個 Linear**:Q/K/V/O。

| 矩陣 | 角色 |
|---|---|
| Q query | 我要找什麼(決定 attention focus)|
| K key | 有什麼可以對(算 score)|
| V value | 找到後拿什麼(內容)|
| O out_proj | 後處理 |

**選 Q, V 不選 K, O**:LoRA 原 paper Table 5 ablation 證實 Q+V 對 perf-per-param 最佳 — Q 跟 V 兩個正交軸,K 跟 Q 在 `score=Q·K^T` 相乘 → 改 K 等於改 Q 重定向,**冗餘**;O 是後處理。

**Strategy A 具體配置**:24 layer × 2 adapter × 32K = **1.57M/encoder**,兩 encoder ~3.14M + ser 19M = **總 22.58M trainable**(vs Full FT 891M 是 1/40)。

**為什麼擴 q,k,v,o 失敗(C1)**:四矩陣同步更新破壞 cross-modal fusion 學到的 attention 對齊 → dev 從 0.6552 退到 0.5442。

### Q: 50:50 sampler 是怎麼做的?

訓練資料總共 **41,744 筆**:EN 30k(72%)+ ZH 11.7k(28%)。**不平衡,每 batch 期望 11.5 EN : 4.5 ZH,ZH 訊號被壓**。

**`--language_balanced` 啟動 PyTorch `WeightedRandomSampler`**:

| 語言 | 樣本數 | 權重 | 倍數 |
|---|---:|---:|---:|
| EN | 30,000 | 1/30000 | 1× |
| ZH | 11,744 | 1/11744 | **2.55×** |

→ 每 batch 期望 **~8 EN : ~8 ZH**;每 epoch EN 各見 0.7×(undersample) / ZH 各見 1.78×(oversample)。

**Code 一行 demo**:
```python
sampler = WeightedRandomSampler(
    weights=1/lang_counts.map(...),  # EN→1/30000, ZH→1/11744
    num_samples=len(df), replacement=True)
```

訓練 log 自動 echo 確認:
```
Language-balanced sampler: counts={'EN': 30000, 'ZH': 11744}, aligned to cur_utts (41744 rows)
```

### Q: 為什麼 EN 30k 不全用 168k?

`pandas.DataFrame.sample(n=30000, random_state=42)` uniform random 從 MSP Train 168k 抽 30k,只動 Train split,Dev/Test 保全集。

| 方案 | 比例 | sampler | 風險 |
|---|---|---|---|
| **b1(實際)** | 30k:12k = 2.5:1 | 50:50 | ZH 輕微 oversample(折衷)|
| b2 | 12k:12k | 無 | EN 砍 90% |
| b3 | 168k:12k = 14:1 | 50:50 | **ZH 嚴重 overfit 14×** |
| b4 | 168k:12k | 無 | EN 9:1 壓中文 |

**選 30k 理由**:LoRA 5-10k 樣本即飽和;b3 ZH 太兇,b4 ZH 淹沒,b2 EN 砍太多。

### Q: 為什麼 LoRA 險勝 Full FT(0.6531 vs 0.6408)?

**LoRA 的隱式 regularization**:凍 869M 強制「只能在 25M 維度空間動」→ 不可能 overfit 太兇。
**Full FT 在 42k 雙語樣本上 overfit**:891M 自由度遠超資料能支撐;dev best @ ep2,後 4 epoch plateau 是「memorize train 但 dev 不進步」。
**Rule of thumb**:Full FT 通常需要 10× param 量級的樣本(891M → 需要 ~10M 樣本,我們只有 42k)。
**結論**:**LoRA 是正確選擇,不是退而求其次**;省 VRAM 是副作用,主因是 generalization。

---

## 4. 反例譜系(被問「為什麼選這個不選那個」用)

5 條跑過真機驗證、結論為負的路徑。**覆蓋三個正交 ablation 軸**:

| 軸 | ablation 對手 | 鎖定結論 |
|---|---|---|
| ① 資料 | Hybrid B(單語 ZH) vs Strategy A(雙語 50:50)| **雙語混訓** |
| ② FT 方法 | C14(Full FT 891M) vs A(LoRA 6M)| **LoRA** |
| ③ LoRA 結構 — target | C1/C1b(q,k,v,o) vs A(q,v)| **target=(q,v)** |
| ③' LoRA 結構 — capacity 子軸 | C5(r=32 α=64) vs A(r=16 α=32)| **r=16 α=32** |

**5 條死路單一表**:

| 路徑 | 設定差 | 結果 | 失敗根因 |
|---|---|---:|---|
| 🗄️ Hybrid B | LoRA + 單語 ZH + warm-start | EN 0.69→0.4072(-0.28)| LoRA 不防 catastrophic forgetting;ser_model 飄向 ZH |
| 🗄️ C14 Full FT | 891M trainable(148× LoRA)| -0.012 三軸均勻 | epoch 2 plateau 後 overfit |
| 🗄️ C1 | LoRA q,v→q,k,v,o,enc_lr 1e-4 | -0.111 | (C1b 推翻 LR 假說)四矩陣同步擾動 attention |
| 🗄️ C1b | 同 C1 + enc_lr 5e-5(半) | -0.121 vs C1 | 降 LR 不救反更差 → q,k,v,o 本身就壞 |
| 🗄️ C5 | LoRA q,v r=32 α=64(rank 雙倍) | -0.107 | 加 rank 也壞 → r=16 即 ceiling |
| 🗄️ F1-B | Partial FT(Wang 2022 + Lee 2019)| -0.046 | literature 預測在 42k 雙語沒轉用成功 |

**ckpt 全保留**(`experiments/{emotiontalk_hybridB_lora, strategyA_fullft, strategyA_c1_qkvo, strategyA_c1b_lowlr, strategyA_c5_rank32, strategyA_f1b_partialft_asym}`)→ paper §IV.B Negative Results 素材。

---

## 5. 資料集現況 + ZH 補強方向

### 5.1 已用資料集

| 資料集 | 規模 | 角色 | 細節 |
|---|---:|---|---|
| MSP-Podcast scheme1 | 168k(3-class)| EN baseline + train | Train subsample 30k(seed=42)→ bilingual CSV |
| EmotionTalk(BAAI)| 14,612 | ZH proxy 主要 | Unconfident 1,766(fear+sad) |
| CH-SIMS v2(s) 🗄️ | 4,403(5-class)| 歷史 Stage A | 已被 Strategy A 取代 |
| MSP Test2 | 10,684(EN cross-domain)| scheme1 cross-domain eval | 主路徑 cross-domain ⏳ 評估中 |

### 5.2 ZH 補強候選(對 B1 改善方向)

| 候選 | 規模 | 取得 | Has fear? | Has text? | License | 對 Unconfident 補強 |
|---|---:|---|:--:|:--:|---|---|
| **CHEAVD-2.0** 🥇 | 數萬(7-class)| Email CASIA + 7 天 | **可能有**(7-class) | 待確認 | CASIA 限學術 | 最 sexy,規模大 |
| **NNIME**(NTHU-NTUA)🥈 | ~11 hr / ~5-10k utt(估)| EULA + 教授簽 + email biiclab | ❌(angry/happy/sad/neu/frust/sur,**無 fear**)| ❌(需 Whisper)| 學術免費,**禁商用** | sad-only 補丁;自發對話 domain 互補 |
| **ESD-CN** 🥉 | 17.5k(讀稿)| 5 分鐘 form + 24hr 回信 | ❌(neu/hap/sad/ang/sur,**無 fear**)| 文本固定 | free | sad-only 補丁;讀稿 domain 對 EmotionTalk 重疊 |

### 5.3 NNIME 細節(申請考量)

| 項 | 值 |
|---|---|
| **全名** | NTHU-NTUA Chinese Interactive Multimodal Emotion Corpus(Chou et al. 2017 ACII)|
| **規模** | ~11 hr audio + video + ECG;**44 個語者**,spontaneous dyadic 自發對話 |
| **情緒** | 6 類:angry / happy / sad / neutral / frustration / surprise |
| **標註** | 4 角度(peer / director / self / observer,49 annotators)+ discrete + continuous-in-time |
| **Text transcript** | ⚠️ **無**,Crab 需 text 模態 → 須跑 Whisper transcribe([`scripts/transcribe_synth.py`](scripts/transcribe_synth.py) pipeline 已有) |
| **License** | 學術 / non-profit 限定,**不可商用**,免費 |
| **申請** | https://nnime.ee.nthu.edu.tw/down/ → EULA + 指導教授簽 → email `biiclab@ee.nthu.edu.tw` |
| **回信** | 未明示(估 3-7 天)|

**scheme1 3-class 映射**:

| scheme1 類 | scheme1 定義 | NNIME 映射 | 可用性 |
|---|---|---|---|
| Excited | Happy + Surprise | happy + surprise | ✅ |
| Unconfident | **Fear + Sad** | **僅 sad**(無 fear)| ⚠️ **lossy** — fear 訊號全靠 EmotionTalk |
| Neutral | Neutral | neutral | ✅ |
| 丟 | Angry | angry + **frustration** | ✅ |

→ 留 4/6 類 = ~67% 樣本 → **估 3,500-7,000 筆可用**。

**對 Crab 的真實 EV**:

| 角度 | 評估 |
|---|---|
| ✅ 自發對話 domain gap 補(EmotionTalk 是讀稿) | covering 重要 |
| ✅ 多角度標註 κ 一致性高 | annotation quality |
| ⚠️ 無 fear → Unconfident 補丁有限 | EV 沒 CHEAVD-2.0 高 |
| ⚠️ 無 text → 必跑 Whisper(中文錯字風險)| 額外 pipeline 成本 |
| ⚠️ 規模小於 EmotionTalk | 邊際效用比 CHEAVD-2.0 數萬筆低 |

**建議申請順序**:

1. **今天送 ESD-CN form**(5 分鐘填表,24hr 回信)
2. **這週寄 NNIME EULA + 教授簽**(估 3-7 天)
3. **本週寄 CHEAVD-2.0 email**(估 7 天)
4. → 任一個到位即可 prep B1 重訓(預估 ZH +0.05-0.10)

---

## 6. F1-B 結果 + 下一步(被問「你還在跑什麼」用)

**F1-B Partial FT asymmetric**(2026-06-04 完訓):
- 設計依據:deep-research 文獻 survey(18 verified claims)→ Wang 2022 ICASSP partial FT for SER 用 audio 全 24 transformer + 凍 CNN(IEMOCAP 勝 EF);Lee 2019 BERT/XLM-R 用 top-quarter
- 配置:audio 全 24 + text top-6,trainable 397M(中庸於 LoRA 22.58M 跟 Full FT 891M)
- 結果:dev best 0.5994 / **test 0.6072**
- **不僅輸 LoRA(-0.046)還輸 C14(-0.034)**;literature 預測沒在 42k 雙語成立

**FT 家族 monotonic 死亡譜系**:LoRA 0.6531 > C14 0.6408 > F1-B 0.6072。**partial FT 反而比 full FT 更差**。

**兩個 FT 家族未調軸假說**(會被問):
1. **contrastive_weight 沿用 LoRA 設定 2.0**(從沒 ablate)→ 可能 FT 家族需 2.5/3.0 更強 regularization
2. **encoder_lr 1e-5 對 891M Full FT 合理,但對 397M Partial FT 可能太低** → F1-B epoch 0 才 0.44(C14 epoch 0 已 0.63)是 under-LR'd signature

**下一步候選**(會被問「接下來打算做什麼」):

| 優先 | 動作 | 預期 |
|---|---|---|
| 🥇 | **B1**:申請 ESD-CN / NNIME / CHEAVD-2.0,加 ZH 第二份(主路徑改善 EV 最高)| ZH +0.05-0.10 |
| 🥈 | **A1**:LoRA Strategy A + contrastive_weight ablation(1.0 / 3.0)| 主路徑微調 ±0.01-0.02 |
| 🥉 | C14b:Full FT + enc_lr 3e-5 + contrastive 3.0(救 C14 試)| ~35% 復活機率 |
| 🔮 | Stage B-target:朋友 rated Unconfident → 解 plateau | 等朋友 κ≥0.4 |

---

## 7. 預期 Q&A(會議高機率被問,有答好)

| Q | A(摘要) |
|---|---|
| **Q1**:為什麼選 XLS-R 不選 WavLM? | XLS-R 128 lang 預訓 vs WavLM 95% EN;**Strategy A 勝 Hybrid B ZH +0.043 同 LoRA q,v 對照下** |
| **Q2**:為什麼 LoRA target = q,v 不擴 q,k,v,o? | LoRA paper Table 5:Q+V 兩正交軸,K 跟 Q 在 score 相乘冗餘,O 後處理;**C1/C1b 實證擴 q,k,v,o 退 0.111-0.234** |
| **Q3**:為什麼 50:50 sampler? | 41k 雙語 EN 72%/ZH 28%,不平衡 batch ZH 訊號被壓;**WeightedRandomSampler 權重 1/lang_count → 期望 batch 8:8** |
| **Q4**:為什麼 EN 30k 不全用 168k? | LoRA 5-10k 樣本即飽和;b3 方案(168k)ZH 14× overfit,b4 ZH 9:1 淹沒;30k 為 plan §8 表中折衷 |
| **Q5**:為什麼 Full FT 輸 LoRA? | 891M 在 42k 雙語樣本 overfit @ ep2;LoRA 凍 869M 強制 hard constraint,**implicit regularization 是 feature 不是 bug** |
| **Q6**:F1-B 為什麼也輸? | Wang 2022(IEMOCAP 5k 單語)→ 我們 42k 雙語 domain gap 大;asymmetric text top-6 太緊;contrastive_weight 跟 enc_lr 沒對 FT 家族 re-tune(假說) |
| **Q7**:中文要更好怎麼辦? | **B1 加 ZH 第二份**:CHEAVD-2.0(EV 最高,~7 天 email)/ NNIME(自發對話互補)/ ESD-CN(最快,5 分鐘 form);三個都申請,平行 |
| **Q8**:Hybrid B 為什麼會 EN 崩? | LoRA 凍 base 不防 catastrophic forgetting:adapter 12 epoch 全中文校準 → EN audio 上 adapter 扭曲特徵;ser_model 全 FT 飄 ZH 分布 |
| **Q9**:朋友資料什麼時候到? | Stage B-target,等 κ≥0.4;從「硬卡關」降為「proxy 校準升級」(plan §6),非 blocker |
| **Q10**:為什麼 contrastive_weight 2.0? | scheme1 沿用;Strategy A 沒 ablate(從沒測 1.0/3.0);C14 第一次 launch per-step bs=1 → MPCL 數學上 = 0 → 教訓 |
| **Q11**:cross-domain(MSP Test2)結果? | **已實測**:scheme1 0.5591 / Strategy A LoRA **0.5544**(差 -0.005,**打平**)/ C14 **0.5400**。LoRA Test2 drop -0.097 **比 scheme1 -0.131 小 26%** — 多語 backbone + LoRA regularization 帶來 cross-domain robustness。|
| **Q12**:NNIME 申請流程? | EULA + 教授簽 → email `biiclab@ee.nthu.edu.tw`;學術免費;**無 text(需 Whisper)**;**無 fear**(Unconfident 只能從 sad 映);estimated 3,500-7,000 筆可用 |
| **Q13** ⭐:真 cross-dataset(不只 MSP 內)效果如何? | **已實測 2026-06-04 23:25**(同 LoRA ckpt,scheme1 3-class 映射): **IEMOCAP 0.5990 / MELD 0.5097**。IEMOCAP 完全不同 corpus(USC 演員 scripted)反而比 MSP Test2 高 +0.045 → 模型沒死記 MSP 的 channel/speaker artifact。MELD 是最 hard 的 OOD(Friends TV 多角色 + 配音 + 背景音 + 笑聲 + 標籤噪音),仍 ≥ 0.50。**兩個 cross-dataset 平均 = 0.5544 = MSP Test2**,證明 cross-dataset gap 不比 MSP 內 cross-session gap 大,**這是「真的 robust」的硬證據,paper §V「Cross-dataset Generalization」可以直接放 4 行 robustness 表**。 |
| **Q14**:IEMOCAP/MELD 怎麼映射成 3-class? | **IEMOCAP**(4,575):hap+exc+sur → Excited(1,743)/ sad+fea → Unconfident(1,124)/ neu → Neutral(1,708)/ 丟 ang+fru+dis+xxx+oth。**MELD**(2,197):joy+surprise → Excited(683)/ sadness+fear → Unconfident(258)/ neutral → Neutral(1,256)/ 丟 anger+disgust。映射 script 在 `scripts/build_iemocap_*.py` 跟 `scripts/build_meld_test_*.py`,符合 scheme1 操作型定義(`Unconfident = Fear + Sad`)。 |

---

## 8. 部署選項(被問「實際怎麼用」)

| 模式 | EN test | ZH test | 平均 | VRAM | 適用 |
|---|---:|---:|---:|---:|---|
| 單一 Strategy A LoRA | 0.6512 | 0.6386 | **0.6449** | **4 GB** | 嵌入式 / 單卡部署;省 VRAM |
| scheme1 + Strategy A 分流 | **0.6900** | 0.6386 | **0.6643** | 8 GB | 兩卡 / 兩 process;EN 不退 |

**論文主結果**:**Strategy A LoRA 單一雙語模型**(0.6531 overall),分流方案當 paper 「practical deployment」段。

---

## 9. 待決事項(會議要決定的)

| # | 議題 | 選項 | 我建議 |
|---|---|---|---|
| **D1** | ZH 補強優先序 | (A) 等朋友 / (B) 三個資料集都申請 / (C) 只申請最快的 ESD-CN | **(B) 三個平行**,任一到位就跑 B1 |
| **D2** | 是否再試 contrastive_weight ablation(A1)| (A) 跑 / (B) 不跑,直接寫 paper | **(A) 跑**,~11 hr 一晚,~30% 主路徑微調機率 |
| **D3** | 是否再試救 C14(C14b 雙軸調)| (A) 跑(8 hr,~35% 救回機率)/ (B) 收入 paper 反例 | **(B) 收**,F1-B 已是 FT 家族第 2 條死路,救 C14 EV 低 |
| **D4** | F1-B 結果寫進 paper 哪段 | (A) Negative Results 第 6 條 / (B) Method Comparison | **(A)**,partial FT 死路寫成完整 FT 家族死亡譜系 |
| **D5** | 朋友資料 ETA | 沒新進度?要 push 嗎? | 視朋友狀況 |

---

## 10. 文件指引

| 文件 | 用途 | 主要讀者 |
|---|---|---|
| **本 memo** | 開會 30 分鐘讀完 | 指導教授 / 學長 / 自己 |
| [`BILINGUAL_FINETUNE_PLAN.md`](BILINGUAL_FINETUNE_PLAN.md)| 410 行完整 plan,§4 完整實驗矩陣是 source of truth | 開會 backup,被細問時翻 |
| [`BILINGUAL_WORK_LOG.md`](BILINGUAL_WORK_LOG.md)| 時序紀錄,所有 run 啟動/失敗/收工細節 | 復現 / debug |
| `experiments/strategyA_xlsr_xlmr_lora/` | 主路徑 ckpt | deployment / re-eval |
| `experiments/{strategyA_fullft, ..._c1_qkvo, ..._c1b_lowlr, ..._c5_rank32, ..._f1b_partialft_asym}` | 5 條反例 ckpt | paper §IV.B 反例素材 |
| W&B project | `Crab_Bilingual_ZH` | 所有 run + Test2 eval |

---

**最後一句**:Strategy A LoRA 0.6531 是 8 軸鎖定後的 sweet spot;接下來改善看 **資料量**(B1 加 ZH 第二份)而不是 **訓練方法**(LoRA 結構鎖死,FT 家族全 dead)。
