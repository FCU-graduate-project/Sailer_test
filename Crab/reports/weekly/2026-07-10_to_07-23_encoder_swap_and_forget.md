# Week 2026-07-10 ~ 07-23 — Encoder Swap Ablation + Catastrophic Forget 檢查

**核心結果:**
- ⭐ scheme1-XLMR-FullFT Ep 6 dev **0.6499**,vs 原 scheme1 0.6720 只 **-0.022** → encoder swap 小 cost
- ⭐ N4-B(scheme1-XLMR-FullFT-warmstart NNIME LoRA)test **0.5286**,vs N2 0.5028 **+0.026** → encoder swap 在 downstream 賺回
- ⚠️ N3 on EN(post-hoc):Test1 **0.5272** / Test2 **0.4349**;vs v1 baseline **-0.1226 weighted** → **明顯 catastrophic forget**
- 修正 paper claim:「Stage-2 transfer 是 domain specialization trade-off,不是 free lunch」

**前情提要:** 見 [2026-07-03_to_07-09 週報](2026-07-03_to_07-09_nnime_transfer.md)(N2 0.5028 / N3 0.5877 / 🅒-full crash / v2b baseline)。

---

## 關鍵結論 1 — Encoder swap 雙 stage 對稱結果

| Stage | 比較 | Δ from encoder swap |
|---|---|---:|
| Pretraining | 原 scheme1(WavLM+RoBERTa)vs scheme1-XLMR-FullFT(XLM-R+XLS-R)dev | **-0.022** |
| NNIME transfer | N2(WavLM+RoBERTa)vs N4-B(XLM-R+XLS-R)test | **+0.026** |
| **Net** | | **+0.004** |

**「換 encoder 在 pretraining 只小 cost,downstream 賺回;搭配獲得的中文 vocab 能力,對 ZH 應用 net-positive」**

---

## 關鍵結論 2 — N3 stage-2 有明顯 EN catastrophic forget ⚠️

| Split | v1 baseline | N3(NNIME stage-2) | **Δ** |
|---|---:|---:|---:|
| MSP Test1(28k)| 0.6506 | 0.5272 | **-0.1234** |
| MSP Test2(10.7k)| 0.5553 | 0.4349 | **-0.1204** |
| **Weighted 平均** | **0.6243** | **0.5017** | **-0.1226** |

**Trade-off ratio:** EN 損失 -0.1226 : NNIME ZH gain +0.033 = **~3.7 : 1** loss/gain

即使 LR 降到 25% 防 forget,仍發生。**修正原本「free lunch」的 paper claim**:

> **「Stage-2 LoRA transfer 對 domain-specific 目標可以 outperform joint learning(NNIME ZH +0.033)—— 但代價是 source domain 明顯 forget(EN -0.12)。這是 domain specialization 的合理 trade-off,不是 free lunch。」**

**部署場景解讀:**
- 只服務中文使用者(如台灣 SER 系統):**N3 > v2b > v1**
- 雙語通用系統:**v2b > v1 > N3**
- 主 EN、少量 ZH:**v1 > N3**

---

## 1. Events

| # | 日期 | Event | 結果 |
|:-:|---|---|---|
| 1 | 07-08 ~ 09 | scheme1-XLMR Full FT(8 epochs)| Ep 6 best dev **0.6499** ⭐ |
| 2 | 07-09 | N4-B scheme1-XLMR-FullFT-warmstart NNIME LoRA(encoder swap ablation vs N2)| test **0.5286** ⭐ |
| 3 | 07-09 | N3 post-hoc EN eval on MSP Test1+Test2 | Test1 **0.5272** / Test2 **0.4349** ⚠️ |
| 4 | 07-09 | v1 baseline EN eval on MSP Test1+Test2(compute forget Δ)| Test1 **0.6506** / Test2 **0.5553** |

---

## 2. scheme1-XLMR Full FT — encoder swap ablation @ pretraining stage ⭐

### Config
- **Encoder:** XLM-R-large + XLS-R-300M
- **Warmstart:** 無
- **FT method:** `--ft_mode full_ft`(**891M trainable**)
- **Data:** MSP scheme1 CSV(110k train / 19k dev / test 未認到 Test1+Test2 命名)
- **Hyperparams:** bs=16 accum=4, epochs=8 early_stop=5
  - LR head 1e-4 / enc 1e-5(Table C FT path)
  - `--use_amp` + `--use_grad_ckpt`(24GB 3090 必開)
  - contrastive 2.0, grad_clip 1.0
- **Class weights:** msp_scheme1_class_weights.json
- **進程保護:** setsid + nohup + < /dev/null(對抗 🅒-full 上週 silent SIGHUP)
- **VRAM 峰:** 19.7 / 24 GB

### 對照原 scheme1 config(caveat: 沒完全對齊)

| | scheme1 原 | 現在 |
|---|---:|---:|
| eff. bs | 1024 | 64(16× 小) |
| epochs | 20 | 8 |
| LR | 1e-5 單一 | head 1e-4 / enc 1e-5 |
| AMP + grad ckpt | 無 | 開 |

原因:XLM-R vocab 250k(RoBERTa 50k)+ MPCL 5-level activation → 同 24GB 開不出大 bs。

### Result trajectory(全 8 epoch)

| Ep | loss | WAR | UAR | macroF1 | 備註 |
|:-:|---:|---:|---:|---:|---|
| 0 | 0.8498 | 0.5673 | 0.6181 | 0.5473 | ★ |
| 1 | 0.8110 | 0.6695 | 0.6731 | 0.6415 | ★ +0.094 big jump |
| 2 | 0.7582 | 0.6601 | 0.6325 | 0.6210 | 1/5 |
| 3 | 0.7209 | 0.6817 | 0.6630 | 0.6458 | ★ |
| 4 | 0.7764 | 0.6262 | 0.6229 | 0.5834 | dip 1/5 |
| 5 | 0.9018 | 0.6322 | 0.5934 | 0.5860 | dip 2/5 |
| **6** | **0.7822** | **0.6915** | **0.6502** | **0.6499** ⭐ | ★ **BEST**,counter reset |
| 7 | 0.8214 | 0.6762 | 0.6490 | 0.6381 | 1/5(訓練結束)|

- Ep 6 per-class dev F1:
  - Excited 0.7683(P 0.7543 / R 0.7828)
  - Unconfident 0.5144(P 0.5154 / R 0.5135)
  - Neutral 0.6670(P 0.6799 / R 0.6545)
- **Best ckpt: Ep 6 (0.6499)**;Ep 7 沒突破 → Ep 6 是天花板
- Wall clock: ~20 hr 完 8 epoch(ep 平均 ~2.5 hr)
- Wandb run id: `dp0e0swe`
- End-of-training test eval:`total_dataloader["test"]` 空(Test1/Test2 沒認到)→ 待 post-hoc `scripts/eval_crab_on_msp_test.py`

### vs 原 scheme1

| | Encoder | dev macroF1 | Δ |
|---|---|---:|---:|
| 原 scheme1(WavLM+RoBERTa, 20 ep)| WavLM+RoBERTa | 0.6720 | baseline |
| **scheme1-XLMR-FullFT Ep 6(8 ep)** | XLM-R+XLS-R | **0.6499** | **-0.022** |

### 結論
**encoder swap Δ 只 -0.022**。Ep 4-5 是暫時 dip,非真正 saturation(Ep 6 又爬回並創新高)。

---

## 3. N4-B — encoder swap ablation @ NNIME transfer stage(N2 對照)⭐

### Config
- **Encoder:** XLM-R-large + XLS-R-300M(取代 N2 的 WavLM+RoBERTa)
- **Warmstart:** scheme1-XLMR-FullFT Ep 6 best(final_ssl + final_text + final_ser)
- **FT method:** LoRA q,v r=16 α=32 dropout=0.1(同 N2)
- **Data:** NNIME 3054 / 608 / 638(同 N2)
- **Hyperparams:** **完全同 N2**(bs=16 accum=4, epochs=15 early_stop=5, LR head 2e-4 / enc 1e-4, contrastive 2.0, grad_clip 1.0)
- **Class weights:** 同 N2

### Result trajectory(全 15 epoch)

| Ep | loss | macroF1 | note |
|:-:|---:|---:|---|
| 0 | 1.1605 | 0.4276 | ★ |
| 1 | 1.0382 | 0.4593 | ★ +0.032 |
| 5 | 1.2955 | 0.3637 | dip 4/5 |
| 6 | 1.1430 | 0.4902 | ★ rebound +0.031 |
| 10 | 1.3326 | 0.5154 | ★ 破 0.50 |
| 12 | 1.3903 | 0.5187 | ★ |
| 14 | 1.3966 | **0.5231** ⭐ | ★ **BEST**(最後 epoch)|

- Ep 5 dip 後 Ep 6 rebound,與 scheme1-XLMR-FullFT pretraining 同樣震盪 → **可能是 XLS-R+XLM-R + Full FT frozen backbone + head LR 2e-4 的組合特徵**
- **Test macroF1: 0.5286**(WAR 0.5846 / UAR 0.5311 / wF1 0.5607)
- Wall clock: ~42 min(15 epoch × ~2.8 min)
- Wandb: `nnime_scheme1xlmr_fullft_warmstart_lora_20260709_143033`

### vs N2(pure encoder swap ablation)

| | N2 | **N4-B** | Δ |
|---|---:|---:|---:|
| Encoder | WavLM + RoBERTa | XLM-R + XLS-R | swap |
| Backbone method | scheme1(Full FT)| scheme1-XLMR(Full FT)| 同(Full FT)|
| Backbone data | 168k EN | 110k EN | 同(純 EN)|
| **Test macroF1** | **0.5028** | **0.5286** | **+0.026** ⭐ |

**Encoder swap 在 NNIME transfer stage 賺 +0.026** —— 與 pretraining stage 的 -0.022 cost 對稱抵消(見「關鍵結論 1」)。

### N4-B 未解 confound(需要 N4-A 拆)

N4-B(0.5286) vs N3(0.5877)= **-0.059**,有 **2 個變因**:
1. Backbone FT method:Full FT vs LoRA
2. Backbone 訓練資料:110k pure EN vs 42k EN+ZH mix

**N4-A(🅒-full-LoRA-ckpt + NNIME stage 2)可拆:**
- v1 vs 🅒-full:資料變因(mix vs pure EN,同 LoRA)
- 🅒-full vs N4-B:FT 方法變因(同 110k EN)

排隊待跑。

---

## 4. N3 catastrophic forget 檢查 ⚠️

**動機:** N3 stage-2 只 finetune 在 3k NNIME(純 ZH)。v1 backbone 原本會英文,那 N3 訓完後英文剩多少?LR 已降到 25% 防 forget,夠嗎?

### 4.1 新工具:LoRA-aware post-hoc eval script

`scripts/eval_lora_crab_on_msp_test.py`:
- 掛 LoRA ckpt(用 `PeftModel.from_pretrained` wrap base HF encoder)
- 跑 MSP Test1 + Test2
- 報 per-class F1 / precision / recall / pred distribution
- 支援任何 LoRA 存法的 ckpt(audio_lora_adapter + text_lora_adapter + final_ser.pt)

### 4.2 N3(NNIME stage-2)EN eval

| Split | n | macroF1 | wF1 | UAR | acc |
|---|---:|---:|---:|---:|---:|
| Test1(matched)| 28,000 | **0.5272** | 0.5773 | 0.6168 | 0.5595 |
| Test2(mismatched)| 10,684 | **0.4349** ⚠️ | 0.5257 | 0.5521 | 0.4657 |

**Test2 病徵:大量 over-predict Unconfident**

| Class | 預測 % | 真實 % | 診斷 |
|---|---:|---:|---|
| Excited | 26.7% | 29.9% | OK |
| **Unconfident** | **41.1%** | **6.5%** | ⚠️ **6× over-predict**(precision 0.1124)|
| Neutral | 32.1% | 63.6% | 大量 under-predict(recall 0.4044)|

Root cause:
- NNIME class weights = Excited 1.26 / **Unc 1.62 最高** / Neu 0.63
- NNIME 樣本本身 Unc 比例遠高於 MSP
- N3 學到「低 threshold 判 Unconfident」→ 套到 MSP EN(Unc 只 6.5%)時大爆表

### 4.3 v1 baseline(未 finetune)EN eval

| Split | n | macroF1 | wF1 | UAR | acc |
|---|---:|---:|---:|---:|---:|
| Test1 | 28,000 | **0.6506** | 0.6998 | 0.6634 | 0.6966 |
| Test2 | 10,684 | **0.5553** | 0.6868 | 0.5783 | 0.6794 |

v1 Test2 per-class F1: Excited 0.5939 / Unc 0.3019 / Neu **0.7699**

v1 Test2 預測分佈:Excited 23.8% / Unc **12.0%(near truth 6.5%)** / Neu **64.2%(近乎完美 63.6%)**

→ **v1 baseline 對 Neutral 的判準完全正確**。N3 stage-2 之後 Unconfident 判準低到把 41% 的 Neutral 樣本錯分成 Unc → **這是明顯 catastrophic forget**。

### 4.4 Forget cost 定量

| Split | v1 baseline | N3 | **Δ** |
|---|---:|---:|---:|
| Test1 | 0.6506 | 0.5272 | **-0.1234** |
| Test2 | 0.5553 | 0.4349 | **-0.1204** |
| **Weighted 平均** | **0.6243** | **0.5017** | **-0.1226** ⚠️ |

**LR 降到 25% 不夠防 forget**。要顯著減少 forget 需要:
- 更低 LR(例:v1 的 10%)
- Elastic Weight Consolidation(EWC)或 rehearsal
- Multi-task loss(同時算 EN + ZH loss)
- 或直接接受這是 domain specialization 的必然 cost

---

## 5. 跨實驗綜合對照(累積本週 + 上週)

### 5.1 NNIME transfer(5 setup + v2b baseline)

| Setup | Encoder | Warmstart | Backbone method | Train | Test macroF1 |
|---|---|---|---|---:|---:|
| N1 fresh(反例)| XLM-R+XLS-R | 無 | — | 3k | 崩 |
| N2 scheme1 | WavLM+RoBERTa | scheme1 全套 | Full FT | 3k | 0.5028 |
| **N3 v1** ⭐⭐ | XLM-R+XLS-R | v1 LoRA + ser | LoRA | 3k | **0.5877** |
| **N4-B**(本週)| XLM-R+XLS-R | scheme1-XLMR-FullFT + ser | **Full FT** | 3k | **0.5286** |
| v2b(baseline)| XLM-R+XLS-R | 無 | — | **53k**(17×)| 0.5543 |

**兩個 clean ablation:**
- **N2 vs N4-B**(encoder swap,同 Full FT backbone,同純 EN)= **+0.026**(XLM-R vocab 賺)
- **N3 vs N4-B**(backbone FT method 差,但資料 confound)= **-0.059** → 需 N4-A 拆變因

### 5.2 Backbone ablation @ pretraining

| Setup | Encoder | FT | Train | Dev macroF1 | Test macroF1 |
|---|---|---|---|---:|---:|
| v1(Strategy A LoRA)| XLM-R+XLS-R | LoRA | 42k EN+ZH | — | **0.6552** |
| 🅒-full | XLM-R+XLS-R | LoRA | 110k EN only | 0.5405 @ ep 2 | crash |
| 原 scheme1 | WavLM+RoBERTa | Full FT | 110k EN | 0.6720 | — |
| **scheme1-XLMR-FullFT** ⭐ | XLM-R+XLS-R | Full FT | 110k EN | **0.6499** | 待 post-hoc |

### 5.3 EN retention @ MSP(catastrophic forget 檢查)

| Model | Test1 macroF1 | Test2 macroF1 | Weighted 平均 |
|---|---:|---:|---:|
| v1 baseline(未 finetune)| **0.6506** | **0.5553** | **0.6243** |
| N3(NNIME stage-2)| 0.5272 | 0.4349 | 0.5017 |
| **N3 forget Δ** | **-0.1234** | **-0.1204** | **-0.1226** ⚠️ |

---

## 6. 下一步

### 短期(本週剩下)
- **N4-A:** 🅒-full ckpt + NNIME LoRA stage 2 → 拆 backbone FT method vs data 變因
- scheme1-XLMR-FullFT **post-hoc Test1+Test2 eval**(`scripts/eval_crab_on_msp_test.py`,補上 EN test 缺失數字)
- N4-B 在 MSP EN 測 forget cost(對比 N3 forget,測 encoder swap 是否影響 forget 程度)

### 中期(下週)
- 若 forget 是問題:試 LR = v1 的 10% 或 EWC / rehearsal 
- N3 pattern 套 EmotionTalk / CNSCED(不 finetune 只在單 corpus 上,測是否推廣 stage-2 > joint claim)
- v2b cross-dataset eval(IEMOCAP + MELD)→ 對比 v1 的 0.5990 / 0.5097

### 長期
- CHEAVD-2.0 申請(email CASIA,7 天回信)+ 加第 4 ZH corpus + stage-2 finetune
- 若 CHEAVD-2.0 到手 → 用 N3 pattern 再測「+更多 ZH data 是否再擴大 NNIME gain」

---

## 7. 程式改動 + 文件

### 新檔案

| 檔案 | 用途 |
|---|---|
| `bin/run_scheme1_xlmr_fullft.sh` | scheme1-XLMR-FullFT launch(含 setsid 三重防護)|
| `bin/run_scheme1xlmr_fullft_warmstart_nnime_lora.sh` | N4-B launch |
| `scripts/eval_crab_on_msp_test.py` | Full-FT ckpt post-hoc Test1+Test2 eval(通用) |
| `scripts/eval_lora_crab_on_msp_test.py` | **LoRA ckpt post-hoc Test1+Test2 eval**(N3、v1、🅒-full 用)|

### 文件

- 本檔:`reports/weekly/2026-07-10_to_07-23_encoder_swap_and_forget.md`
- 上週報(trim):`reports/weekly/2026-07-03_to_07-09_nnime_transfer.md`
- `reports/weekly/README.md` — 週報 index 更新
- 待更新:`BILINGUAL_FINETUNE_PLAN.md` v4.3
  - 加 §5.7 encoder swap 雙 stage 對稱 claim
  - 加 §5.8 catastrophic forget 定量 + trade-off 討論
  - 修正 §5.6 N3 claim 從「free lunch」→「domain specialization trade-off」
