# Crab 中英雙語化 Fine-tune Plan

**版本：** v1.0
**日期：** 2026-05-16
**目標：** 讓現有 Crab 雙模態情緒模型在中文 + 英文兩種語言上都有可用準度
**現況：** Crab 目前在英文上訓練，使用 WavLM-Large（audio）+ RoBERTa-Large（text）

---

## 1. 問題本質：兩個獨立的編碼器軸

中英雙語化**不是單一決策**。Crab 有兩個編碼器，各自面對不同的語言適應問題：

| 編碼器 | 現用模型 | Hidden Size | 對中文的問題 |
|--------|---------|-------------|-------------|
| **Text Encoder** | `roberta-large` | 1024 | Tokenizer 沒中文 vocab，byte fallback 切爛中文字 |
| **Audio Encoder** | `microsoft/wavlm-large` | 1024 | 95%+ 預訓練語料為英文（LibriLight），中文聲學特徵（聲調、韻律）沒見過 |

⚠️ **關鍵體認**：兩個編碼器**必須同時處理**才完整。若只動 text encoder（朋友的初版方案）、不動 audio encoder：
- 文字翻譯後再準也沒用，因為 WavLM 對中文音訊輸出的 representation 已經偏斜
- Cross-modal attention 拿到歪斜的 audio frame，無法跟正確的 text token 對齊
- 結果：中文準度被 audio 端綁住，永遠上不去

---

## 2. 五個方案總覽

### Quick Comparison

| 方案 | Text Encoder | Audio Encoder | 中文準度 | 英文準度 | 訓練成本 | 推薦度 |
|------|--------------|---------------|----------|----------|----------|--------|
| **1. 雙 encoder 替換 + bilingual FT** | XLM-R-Large | XLS-R-300M | ⭐⭐⭐⭐ | ⭐⭐⭐ 略降 | 1-2 週 + 數據 | **★ 正式版首選** |
| **2. 只換 Text** | XLM-R-Large | WavLM-Large | ⭐⭐⭐ | ⭐⭐⭐⭐ | 1-2 天 (A4500) | 折衷選項 |
| **3. LoRA on Text only** | roberta-large + LoRA r=16 | WavLM-Large | ⭐⭐ | ⭐⭐⭐⭐ 完全保留 | 數小時 | **★ POC / 試水溫** |
| **4. Translate-then-classify** | roberta-large | WavLM-Large | ⭐ | ⭐⭐⭐⭐ | 0 | 不推薦 |
| **5. 雙模型分流** | 兩個獨立模型 | 兩個獨立模型 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 訓練 ×2 | 不推薦 |

---

## 3. 方案 1：雙 encoder 替換 + bilingual fine-tune（正式版首選）

### 架構變動

```
中/英文音檔 ──→ XLS-R-300M (多語 SSL, 128 langs)        ┐
                                                          ↓ Cross-attention fusion
中/英文文字 ──→ XLM-RoBERTa-Large (多語文字, 100 langs) ┘
                                                          ↓
                                                  3-class output (Excited/Unconfident/Neutral)
```

### 為什麼選這兩個模型

| 候選 | Hidden Size | 多語支援 | 架構相容性 | 採用 |
|------|-------------|----------|-----------|------|
| `FacebookAI/xlm-roberta-large` | 1024 | 100 langs | ✅ 與 roberta-large 一樣 | **採用** |
| `facebook/wav2vec2-xls-r-300m` | 1024 | 128 langs | ✅ 與 wavlm-large 一樣 | **採用** |
| `utter-project/mHuBERT-147` | 768 | 多語 | ❌ 需改架構 | 不採用 |
| `hfl/chinese-roberta-wwm-ext` | 1024 | 純中文 | ✅ 但需另跑英文版 | 不採用 |

**關鍵 insight：兩個替換模型的 `hidden_size` 都是 1024，與現有 Crab 的 cross-attention 維度完全相容，不用改架構。**

### 程式碼變動點

```python
# Crab/api/inference.py
text_model_path = "FacebookAI/xlm-roberta-large"   # 原: "roberta-large"
ssl_model_path = "facebook/wav2vec2-xls-r-300m"    # 原: "microsoft/wavlm-large"
```

需重新訓練的 weights：
- `final_text.pt` — text encoder 上的 task head
- `final_ssl.pt` — cross-attention（text representation 分布變了）
- `final_ser.pt` — classifier head

### 訓練資料需求

| 資料集 | 語言 | 規模 | 取得難度 | 用途 |
|--------|------|------|----------|------|
| MSP-Podcast（已有）| en | ~50k utt | ✓ 已有 | Crab 原訓練集 |
| M3ED | zh | ~24k utt | 學術下載 | 多人對話、多模態 |
| CH-SIMS | zh | 2,281 影片 | 學術下載 | 中文情緒+極性 |
| CHEAVD 2.0 | zh | ~140h | 申請 | 中國影視情感 |
| CASIA-Chinese Emotional Speech | zh | ~9,600 | 學術 | 純語音、6 情緒 |
| ESD - Mandarin subset | zh | 多人 | 開源 | 5 情緒 |
| 自蒐面試錄音 + 標註 | zh | 自定 | 高 | **最對齊使用情境** |

混合訓練策略：MSP-Podcast + M3ED + CH-SIMS 比例約 **1:1** 中英平衡。

### 訓練資源預估

- GPU：A4500 (24GB) 即可
- 時間：1-2 天 / epoch，3-5 epoch 收斂
- 注意：中英資料的 3-class 分布可能差很大，需 **weighted CrossEntropy + class balancing**

---

## 4. 方案 2：只換 Text Encoder（折衷選項）

### 架構

僅替換 text encoder，audio encoder 不動。

```python
# Crab/api/inference.py:~48
text_model_path = "FacebookAI/xlm-roberta-large"   # 原: "roberta-large"
# audio 不變
```

### 需重新訓練

- `final_text.pt`
- `final_ssl.pt`（text 表徵分布變了，cross-attention 要重新對齊）
- `final_ser.pt`（classifier head）

### 優缺點

| 優點 | 缺點 |
|------|------|
| 不用換 audio encoder，省一半實驗 | **中文音訊瓶頸沒解決** |
| 1-2 天訓練（A4500）| 中文準度被 WavLM 限制 |
| 英文準度幾乎不變 | 半套解法 |

**何時用**：你**只有中文文字標註**沒有音檔，或暫時無法重新訓練 audio side。

---

## 5. 方案 3：LoRA on Text only（POC 推薦）

### 架構

```
roberta-large（凍結）
    └─ LoRA adapter (rank=8 or 16)  ← 只訓練這個
```

### 優缺點

| 優點 | 缺點 |
|------|------|
| 額外參數 < 1%，訓練 10x 快 | RoBERTa 本身沒中文 vocab → tokenizer byte fallback 救不到 |
| 英文性能完全保留（base 凍結）| 中文上限不高（中文 token 表達能力差） |
| 數小時內可看到結果 | 不能真正取代方案 1 |

### 何時用

- **快速驗證概念** — 想知道「fine-tune 對中文有沒有用」
- 還沒準備好大規模重訓
- 僅有少量（< 1000 筆）中文標註

### 配方

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],  # RoBERTa attention layers
    lora_dropout=0.1,
    task_type="FEATURE_EXTRACTION"
)
text_encoder = get_peft_model(roberta_large, lora_config)
```

---

## 6. 方案 4：Translate-then-classify（不推薦）

### 流程

```
中文音檔     → WavLM (對中文不準)             ┐
中文 transcript → 翻譯成英文 → 英文 RoBERTa  ┘
                                              → 預測
```

### 為什麼不推薦

| 問題 | 影響 |
|------|------|
| 音檔無法翻譯 | WavLM 對中文音訊還是不準 |
| 文字翻譯丟失語氣/文化詞 | RoBERTa 拿到的 representation 失真 |
| Cross-modal attention 對齊鬆掉 | text 是翻譯版、audio 是原版，時間軸 + 語意都對不上 |

### 唯一適用場景

完全沒有訓練資源 + 完全可接受品質差。**正式產品不要用**。

---

## 7. 方案 5：雙模型分流（不推薦）

### 架構

```
音檔 → 語言偵測 (Whisper langid) ┬→ en → Crab-EN (現有)
                                  └→ zh → Crab-ZH (新訓練)
```

### 優缺點

| 優點 | 缺點 |
|------|------|
| 各語言可獨立最佳化 | VRAM 用量 ×2 |
| 英文準度完全保留 | 維護成本 ×2 |
| 部署可分機器 | 切換語言瞬間有延遲 |

### 為什麼不採用

ROI 太低 — 同樣的訓練工作量，方案 1 用單一模型就解決，不需要為了「英文準度完全不變」付出雙倍維護成本。

---

## 8. 方案 6：從頭預訓練 base（拒絕）

- 從 scratch 預訓練 XLM-R 等級的 base model：**10k+ GPU-hours**
- 沒道理做。Facebook / Microsoft 已經免費釋出，直接用。

---

## 9. 推薦執行路徑（Phased Rollout）

### Phase 1：試水溫（1 週內）

**目標**：在不大改的前提下，知道中文準度天花板在哪。

```
1. 跑方案 4（translate-then-classify）幾筆中文面試片段
   → 觀察「audio 端 WavLM 對中文音訊輸出」的可用度
   → 評估「不動 audio」的下限

2. 同時跑方案 3（LoRA on text）
   → 用少量中文資料 fine-tune RoBERTa 的 LoRA
   → 看文字側能拉多少

3. 比較兩者結果，判斷是否需要走 Phase 2
```

成本：幾小時 ~ 一天，0 訓練資料壓力。

### Phase 2：正式雙語化（1-2 個月）

**前提**：
- ✅ 已蒐集 ≥ 5,000 筆中文情緒標註資料
- ✅ 中文準度要 production-level

**動作**：執行**方案 1**（雙 encoder 替換 + bilingual fine-tune）。

```python
# 1. 修改 inference.py 兩處 model path
# 2. 準備混合資料 loader（MSP + M3ED + CH-SIMS）
# 3. 訓練腳本同 Crab 原訓練流程，loss 加 weighted CE
# 4. A4500 跑 3-5 epoch
# 5. 評估中英 holdout set
```

---

## 10. 兩家分析的差異（給後續討論的 anchor）

| 議題 | 採納誰 | 為什麼 |
|------|--------|--------|
| 文字編碼器選 XLM-R-Large | 朋友（方案 A）| 指出 hidden_size 相容性是關鍵實務知識 |
| LoRA 作為快速試水溫 | 朋友（方案 B）| 引入中間路徑 |
| A4500 1-2 天訓練估算 | 朋友 | 實測經驗值 |
| 訓練資料集名單 | 兩邊合併 | 朋友提了 M3ED/CH-SIMS/CHEAVD，我加 CASIA/ESD |
| 加入 Audio Encoder 替換 | 我 | 朋友的方案 A 漏了這塊，純換 text 中文音訊端不會好 |
| 雙模型分流方案 | 我 | 朋友沒考慮，但 ROI 低可擱置 |
| 拒絕 translate-then-classify | 兩邊都不推 | 跨模態對齊壞掉 |
| 拒絕從頭預訓練 | 兩邊都不推 | 10k+ GPU-hour，沒道理 |

---

## 11. 一句話結論

> **Phase 1 試 LoRA + 翻譯驗證，Phase 2 走「XLM-R-Large + XLS-R-300M」雙 encoder 替換，bilingual fine-tune 一張 A4500 1-2 天。**
>
> **只動 text encoder 不動 audio 是半套**，會在中文音訊上有明顯瓶頸。

---

## 12. 待辦事項

- [ ] **Phase 1 準備**：抓 100 筆中文面試錄音 + transcript 做翻譯版測試
- [ ] **Phase 1 LoRA**：找 1000 筆中文情緒標註資料（CASIA / CH-SIMS subset）
- [ ] **Phase 2 資料**：申請 M3ED / CHEAVD 學術授權
- [ ] **Phase 2 程式**：在 `Crab/src/train_*.py` 加 multilingual data loader
- [ ] **Phase 2 評估**：建立中文 holdout set，定義評估指標
- [ ] **架構驗證**：實際 print 出 `roberta-large` 和 `xlm-roberta-large` 的 `hidden_size` 確認都是 1024
- [ ] **架構驗證**：實際 print 出 `wavlm-large` 和 `wav2vec2-xls-r-300m` 的 hidden 維度確認相容
