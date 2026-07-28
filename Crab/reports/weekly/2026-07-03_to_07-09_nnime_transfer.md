# Week 2026-07-03 ~ 07-09 — NNIME Transfer Learning 調查

**核心結果:**
- ⭐⭐ N3(v1-warmstart NNIME LoRA)test **0.5877**,超 v2b baseline 0.5543 **+0.033**(用 5.7% 資料量)
- 🅒-full 110k 純 EN LoRA dev **0.5405** < v1 42k EN+ZH mix 0.6552 → 混訓 > 純量

**scheme1-XLMR-FullFT / N4-B / N3 EN eval 的結果 → 見** [2026-07-10_to_07-23 週報](2026-07-10_to_07-23_encoder_swap_and_forget.md)。

---

## 1. Events

| # | 日期 | Event | 結果 |
|:-:|---|---|---|
| 1 | 07-06 | v2c kill(zombie 清理)| — |
| 2 | 07-06 | N2 scheme1-warmstart NNIME LoRA | test **0.5028** |
| 3 | 07-07 | N3 v1-warmstart NNIME LoRA stage 2 | test **0.5877** ⭐⭐ |
| 4 | 07-08 | 🅒-full XLM-R+XLS-R LoRA on 110k MSP EN | dev **0.5405** @ ep 2(silent crash @ ep 6)|
| 5 | 07-08 | scheme1-XLMR-FullFT launched(結果轉入下週)| — |

---

## 2. N2 — Scheme1 warmstart NNIME LoRA

### Config
- **Encoder:** WavLM-large + RoBERTa-large(scheme1 原生)
- **Warmstart:** scheme1 全套(final_ssl.pt + **final_text.pt(新 patch)** + final_ser.pt)
- **FT method:** LoRA q,v r=16 α=32 dropout=0.1
- **Data:** NNIME 3054 / 608 / 638(train/dev/test)
- **Hyperparams:** bs=16 accum=4(eff bs 64), epochs=15 early_stop=5, LR head 2e-4 / enc 1e-4, contrastive 2.0, grad_clip 1.0
- **Class weights:** Excited 1.2599 / Unc 1.6210 / Neu 0.6292

### Result
- Dev best: **0.5017** @ ep 9
- **Test macroF1: 0.5028**(WAR 0.5361 / UAR 0.5049)
- Per-class F1: Excited 0.551 / Unconfident 0.320 / Neutral 0.637
- Wall clock: ~1 hr
- Wandb: `strategyA_v2b_persource__scheme1_warmstart_nnime`

### 結論
破 0.50 但 < v2b。**RoBERTa 無中文 vocab 是硬瓶頸**(Unc F1 只 0.320 vs v2b 0.403)。

---

## 3. N3 — v1 warmstart NNIME LoRA stage 2 ⭐⭐

### Config
- **Encoder:** FacebookAI/xlm-roberta-large + facebook/wav2vec2-xls-r-300m
- **Warmstart:** v1 (Strategy A LoRA, EN+ZH 42k pretrained)全套
  - `--pre_trained_path ./experiments/strategyA_xlsr_xlmr_lora`
  - `--lora_warmstart`(新 flag,`PeftModel.from_pretrained(is_trainable=True)`)
  - `--warm_start_ser`(final_ser.pt)
- **FT method:** LoRA q,v r=16 α=32 dropout=0.1
- **Data:** NNIME 3054 / 608 / 638
- **Hyperparams:** bs=16 accum=4, epochs=10 early_stop=5
  - **LR head 5e-5 / enc 2e-5(v1 的 25%,防 catastrophic forget)**
  - contrastive 2.0, grad_clip 1.0
- **Class weights:** 同 N2

### Result
- Dev best: **0.5283** @ ep 4
- **Test macroF1: 0.5877 ⭐⭐**(WAR 0.6176 / UAR 0.5787 / wF1 0.6095)
- Per-class:

| Class | precision | recall | F1 |
|---|---:|---:|---:|
| Excited | 0.7031 | 0.4972 | 0.5825 |
| Unconfident | 0.5137 | 0.4545 | 0.4823 |
| Neutral | 0.6291 | 0.7842 | 0.6982 |

- Wall clock: ~30 分鐘(10 epoch × ~3 min)
- Wandb: `nnime_v1_warmstart_lora_20260707_164218`

### vs v2b baseline(NNIME per-source)

| Class F1 | v2b(53k) | N3(3k) | Δ |
|---|---:|---:|---:|
| Excited | 0.563 | 0.583 | +0.020 |
| **Unconfident** | 0.403 | **0.482** | **+0.079** ⭐ |
| Neutral | 0.697 | 0.698 | +0.001 |
| **macro** | **0.5543** | **0.5877** | **+0.033** |

### 結論
**用 5.7% 資料量(3k vs 53k)打贏 v2b baseline +0.033**,Unconfident F1 +0.079 為主力。
paper 素材:**Stage-2 LoRA transfer > Joint learning(for NNIME)**。

⚠️ **caveat 待補**:N3 的 EN 保留度未知,catastrophic forget 檢查見下週報告。

---

## 4. 🅒-full — XLM-R+XLS-R LoRA on 110k MSP EN

### Config
- **Encoder:** XLM-R-large + XLS-R-300M(同 v1)
- **Warmstart:** 無
- **FT method:** LoRA q,v r=16 α=32 dropout=0.1
- **Data:** MSP scheme1 CSV,train 110k / dev 19k / test 未認到(Test1+Test2 命名)
- **Hyperparams:** bs=16 accum=4(eff bs 64), epochs=10 early_stop=5, LR head 2e-4 / enc 1e-4(完全 match v1),contrastive 2.0
- **Class weights:** Excited 0.9144 / Unc 1.9330 / Neu 0.7199

### Result
- Dev best: **0.5405** @ ep 2 ⭐(ckpt 保住)
- **Ep 6 silent crash**(school VPN 斷 → SIGHUP 連鎖,無 Traceback)
- Test: 未 eval
- Wall clock: ~11 hr(前 5 epochs)
- Wandb: `scheme1_xlmr_lora_full_20260708_...`
- 修正:下次 launch 加 `setsid + nohup + < /dev/null` 三重防護

### vs v1

| Backbone | Encoder | Data | Test macroF1 |
|---|---|---|---:|
| v1(Strategy A LoRA)| XLM-R+XLS-R | 42k EN+ZH mix | **0.6552** |
| 🅒-full | XLM-R+XLS-R | 110k EN only | (未 eval,dev 0.5405 @ ep 2)|

### 結論
110k EN 純量 < 42k EN+ZH mix。**cross-lingual 混訓是 effective regularizer**,不是資料多就贏。加強 N3 選 v1 backbone 是對的。

---

## 5. 跨實驗綜合對照

### NNIME transfer(本週 3 setup + v2b baseline)

| Setup | Encoder | Warmstart | Train | Test macroF1 | Exc F1 | Unc F1 | Neu F1 |
|---|---|---|---:|---:|---:|---:|---:|
| N1 fresh(反例)| XLM-R+XLS-R | 無 | 3k | 崩 | — | 0.00 | 0.00 |
| N2 scheme1 | WavLM+RoBERTa | scheme1 全套 | 3k | 0.5028 | 0.551 | 0.320 | 0.637 |
| **N3 v1** ⭐⭐ | XLM-R+XLS-R | v1 LoRA + ser | 3k | **0.5877** | 0.583 | **0.482** | 0.698 |
| v2b(上週,baseline)| XLM-R+XLS-R | 無 | **53k**(17×)| 0.5543 | 0.563 | 0.403 | 0.697 |

---

## 6. 下一步

### 短期(下週)
- **N4-B:** scheme1-XLMR-FullFT ckpt + NNIME LoRA stage 2 → 對比 N2,測 encoder swap 在 NNIME transfer stage 的效果
- **N4-A:** 🅒-full-LoRA ckpt + NNIME LoRA stage 2 → 對比 N3 拆混訓 vs 純 EN 貢獻
- N3 catastrophic forget 檢查:on MSP Test1+Test2 measure EN retention

### 中期(1-2 週)
- N3 pattern 套 EmotionTalk / CNSCED → 推廣 stage-2 > joint claim
- v2b cross-dataset eval(IEMOCAP + MELD)→ 對比 v1 的 0.5990 / 0.5097
- CHEAVD-2.0 申請(email CASIA,7 天回信)+ 加第 4 ZH corpus

---

## 7. 程式改動 + 文件

### 新檔案

| 檔案 | 用途 |
|---|---|
| `bin/run_scheme1_warmstart_nnime_lora.sh` | N2 launch |
| `bin/run_v1_warmstart_nnime_lora.sh` | N3 launch |
| `bin/run_scheme1_xlmr_lora_full.sh` | 🅒-full launch |
| `data/nnime_class_weights.json` | N1/N2/N3 共用 |
| `data/msp_scheme1_class_weights.json` | 🅒-full + scheme1-XLMR-FullFT(下週)共用 |

### 程式 patch

| 檔案 | 改動 | 用途 |
|---|---|---|
| `bin/train_crab_lora.py` | 加 `final_text.pt` warm-start | N2 需要 |
| `bin/train_crab_lora.py` | 加 `--lora_warmstart` flag(`PeftModel.from_pretrained(is_trainable=True)`)| N3 需要 |

### 文件

- `BILINGUAL_FINETUNE_PLAN.md` v4.0 → v4.1 → v4.2
  - v4.1: 加 §5.6 NNIME 子實驗(N1 + N2)
  - v4.2: §5.6 加 N3;新 claim「Stage-2 > Joint」;`--lora_warmstart` 說明
- `reports/weekly/README.md` — 新週報 index
- `reports/weekly/2026-07-03_to_07-09_nnime_transfer.md` — 本檔
