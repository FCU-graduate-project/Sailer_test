# Crab 台灣華語優先 Fine-tune Plan

**版本：** v5.0（2026-09-02 重整）
**主要服務目標：** 提升台灣口音華語三分類情緒辨識的 macro-F1
**主要模型起點：** N3 v1-warmstart stage 2
**英文模型：** scheme 1（維持現況，暫不進入本輪搜尋）
**統一 W&B project：** [Crab_Bilingual_ZH](https://wandb.ai/d1249119-feng-chia-university/Crab_Bilingual_ZH)

> 本文件從 v5.0 起是「台灣華語主線」的執行規格。舊版中已完成、失敗或暫緩的方案移至後段整理，不再和下一輪待執行實驗混排。

---

## 1. 執行摘要

目前最佳中文模型是 **N3 v1-warmstart stage 2**：

| 項目 | 現況 |
|---|---:|
| NNIME test macro-F1 | **0.5877** |
| N3 best dev macro-F1 | **0.5283**（epoch 4） |
| 相對 v2b NNIME baseline | **+0.033** |
| Stage 2 訓練資料 | 約 3k NNIME |
| 初始化 | v1 bilingual checkpoint |
| 現行 contrastive loss weight | 2.0 |
| 實際 micro batch | 4 |
| gradient accumulation | 4 |
| optimizer effective batch | 16 |

下一步不直接增加多個複雜模組，而是依照下列順序進行：

1. 先確認 contrastive loss mask、有效 anchor 與 batch 語意正確。
2. 對中文資料正式做 contrastive weight（α）消融。
3. 再測試真正會影響 supervised contrastive learning 的 micro batch。
4. 前兩階段仍無穩定提升時，才進入標註分布、資料與 augmentation。

所有新訓練、驗證結果與最終測試結果都記錄到同一個 W&B project；不再另外建立中文 project。

---

## 2. 服務範圍與成功定義

### 2.1 主要評估對象

- 語言：華語。
- 口音／場景：台灣口音、台灣面試或近似服務語音。
- 任務：Excited / Unconfident / Neutral 三分類。
- 主指標：macro-F1。
- 輔助指標：每類 precision、recall、F1、UAR、WAR、confusion matrix、預測類別分布。

英文 scheme 1 仍保留為英文服務基線，但 v5.0 不以英文平均分數決定中文模型，也不讓英文資料的提升掩蓋中文少數類別退步。

### 2.2 模型晉級條件

開發階段只用固定 dev split 選模。候選模型需同時滿足：

1. 多 seed 的 dev macro-F1 平均優於重新跑出的 N3 對照組。
2. 提升不能只來自單一 seed；必須回報平均、標準差與每個 seed。
3. 最弱類別 F1 不應有明顯退化；若退化超過 0.02，需單獨說明。
4. confusion matrix 與預測類別比例沒有顯著 collapse。
5. 最後勝出設定才可碰 locked test。

實務上，**平均 dev macro-F1 +0.01** 可視為值得晉級的初步門檻；小於 0.01 的差距必須有低變異、少數類別改善或服務 holdout 證據支持，不能只報最佳單次結果。

### 2.3 Test 使用規範

- α 搜尋與 batch 搜尋期間不得反覆看 test。
- 每一個完整 phase 只允許最終 champion 評估一次 locked test。
- test macro-F1 仍與 N3 的 0.5877 比較，但 0.5877 不能拿來調超參數。
- 若建立台灣服務 holdout，該集合亦須鎖定，不能混入訓練或一般 dev。

---

## 3. 已知基線與證據

### 3.1 中文最佳基線：N3

N3 的主要價值不是只靠 NNIME 3k 從頭訓練，而是：

```text
bilingual v1 representation
        ↓ warm start
NNIME LoRA stage 2
        ↓
NNIME test macro-F1 0.5877
```

既有 fresh NNIME 3k 與 scheme1 warm-start 結果均低於 N3，因此本輪以 N3 checkpoint 作為共同初始化，避免重新比較已被否定的起點。

N3 test 各類別現況：

| 類別 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Excited | 0.7031 | 0.4972 | 0.5825 |
| Unconfident | 0.5137 | 0.4545 | 0.4823 |
| Neutral | 0.6291 | 0.7842 | 0.6982 |

目前最需要改善的是 Unconfident，其次是 Excited recall。後續不只看總 macro-F1。

### 3.2 Contrastive weight 2.0 的適用範圍

原始 CRAB 論文在 MELD 英文資料上掃描 α = 0.25 到 3.0：

- UAR 最佳點為 α = 2.0。
- WAR 最佳點為 α = 2.75。
- 論文的主要選擇依據不是台灣華語 macro-F1。

因此，「英文 α = 2.0 最好」可作為中文搜尋中心點，但不足以證明中文也應固定為 2.0。v4.2 將 contrastive weight 列為鎖死軸，v5.0 正式解除這項鎖定。

### 3.3 Batch 記錄修正

`bin/train_crab_lora.py` 中的 `batch_size` 是 optimizer effective batch；實際 forward micro batch 為：

```text
micro_batch = batch_size // accumulation_step
```

N3 使用 `batch_size=16`、`accumulation_step=4`，所以：

- micro batch = 4
- effective batch = 16
- 並非 effective batch 64

gradient accumulation 只累積梯度，不會讓前一個 micro batch 的樣本加入當次 contrastive pair pool。因此對 supervised contrastive loss 而言，N3 每次能配對的 batch 仍只有 4，這是 v5.0 必須檢驗的主要限制。

---

## 4. 實驗共同契約

所有消融必須遵守「一次只改一個軸」。

### 4.1 固定項目

- N3 v1-warmstart stage 2 初始化 checkpoint。
- NNIME train/dev/test split 與資料前處理。
- 類別 mapping 與 class weights。
- audio encoder、text encoder、fusion 與 classifier 結構。
- LoRA target、rank 與 LoRA α。
- optimizer、learning rate、scheduler、epoch 上限與 early stopping 規則。
- 模型選擇指標：dev macro-F1。
- Phase A 首輪 seed：42。
- Phase A 複驗 seeds：42、100、2026。

若因修正 bug 而無法完全復現舊 N3，需先產生「修正後 N3 baseline」，之後所有候選都與它比較，不可把舊分數和新程式結果直接混算。

### 4.2 實驗中可變項

| Phase | 唯一主要變項 |
|---|---|
| Phase 0 | 無；只做正確性與重現性稽核 |
| Phase A | contrastive weight α |
| Phase B | micro batch／contrastive pair pool |
| Phase C | hard label 與 soft rater target |
| Phase D | 資料選擇或 augmentation |

不得在 α sweep 同時更改 batch、learning rate、sampling 或 augmentation。

---

## 5. Phase 0：先確認量測與 loss 正確

### 5.1 N3 可重現性

先用現行設定重跑一個 seed 42 對照組，至少核對：

- best dev macro-F1 與 epoch。
- 三類 precision／recall／F1。
- test 僅在確認這是正式 baseline 時評估一次。
- train classification loss 與 contrastive loss 的量級。
- micro batch 及 effective batch 的真實值。

允許隨機差異，但若 dev macro-F1 與既有 0.5283 差距超過 0.02，先停止超參數搜尋並找出原因。

### 5.2 Contrastive mask 稽核

逐項確認：

- positive pair 的定義是否真的是「同 emotion label」。
- self-pair 對角線是否被排除。
- denominator 是否排除自身且包含應有 negatives。
- 沒有 positive pair 的 anchor 是否安全跳過，而非產生 NaN 或錯誤平均。
- mask 與 logits 位於相同 device、shape 和 dtype 語意正確。
- batch size 4 時，每類 anchor 的覆蓋率是否足夠。

至少新增兩個小型 deterministic 測試：

1. 人工 label batch，可手算 positive pair 數量。
2. 全部 label 唯一的 batch，loss 必須可定義且明確回報 zero-positive anchors。

### 5.3 Phase 0 通過條件

- 無 NaN／Inf。
- 同一 checkpoint 的 eval 重跑結果一致。
- valid-anchor ratio、zero-positive ratio 與平均 positive pairs 可以記錄。
- N3 對照組的偏差在可接受範圍。

Phase 0 未通過，不開始 Phase A。

---

## 6. Phase A：中文 contrastive weight 消融

### 6.1 首輪搜尋

固定所有其他設定，只改 α：

| Run ID | α | 用途 |
|---|---:|---|
| A-000 | 0.00 | 無 contrastive loss 對照 |
| A-025 | 0.25 | 論文搜尋下界 |
| A-050 | 0.50 | 低權重 |
| A-100 | 1.00 | 中低權重 |
| A-150 | 1.50 | 接近現行值 |
| A-200 | 2.00 | N3／論文 UAR 對照 |
| A-250 | 2.50 | 高權重 |
| A-300 | 3.00 | 論文搜尋上界 |

首輪全部使用 seed 42。排序依據：

1. dev macro-F1。
2. 最低類別 F1。
3. Unconfident F1。
4. macro-F1 相同時，優先較低 α，降低過度約束風險。

### 6.2 多 seed 複驗

首輪 top 3 設定以 seeds 42、100、2026 完成複驗。由於 seed 42 已跑過，正常新增 6 runs。

最終 α 不採單次最高值，而依下列資料決定：

- dev macro-F1 mean ± std。
- 每類 F1 mean ± std。
- valid-anchor ratio 與 positive pairs。
- loss 曲線是否穩定。
- 是否出現某 seed 類別 collapse。

### 6.3 決策分支

- 若 α = 0 勝出：中文 stage 2 暫停 contrastive loss，後續把資源放在 classification、sampling 或 soft labels。
- 若 α = 2 穩定勝出：保留 2.0，但結論限定為 NNIME／台灣華語實證。
- 若最佳點落在搜尋邊界 0.25 或 3.0：只往該方向補 1–2 個點，不直接大幅擴張。
- 若各 α 差異小於 run-to-run 變異：視為沒有可靠差異，優先選較簡單或較穩定的設定。
- 若 valid-anchor ratio 過低：不要解讀 α 結果，先執行 Phase B。

### 6.4 預估 run 數

- 首輪：8 runs。
- top 3 複驗：6 個新增 runs。
- 正常合計：14 runs。

此數量不含 Phase 0 baseline，也不包含最後 locked-test evaluation。

---

## 7. Phase B：提升有效 contrastive pair pool

Phase B 的核心不是盲目放大 optimizer batch，而是讓一次 contrastive loss forward 看見更多樣本。

### 7.1 建議矩陣

在選出的 α* 下，優先維持 optimizer effective batch = 16：

| 設定 | Micro batch | Accumulation | Effective batch | 目的 |
|---|---:|---:|---:|---|
| B4 | 4 | 4 | 16 | N3 對照 |
| B8 | 8 | 2 | 16 | 擴大 pair pool |
| B16 | 16 | 1 | 16 | 最大化單次 pair pool |

若 B8／B16 超出 GPU 記憶體：

- 先使用 mixed precision、gradient checkpointing 或縮短不影響標籤的 padding。
- 不可把 accumulation 增加誤稱為 contrastive batch 增加。
- 若只能用 micro batch 4，考慮 label-aware batch sampler 或 memory queue，但必須另立實驗，不能混入 Phase A。

### 7.2 必看診斷

- `mpcl_valid_anchor_ratio`
- `mpcl_zero_positive_anchor_ratio`
- `mpcl_positive_pairs_per_anchor_mean`
- 每類樣本在 batch 中的覆蓋率
- classification 與 contrastive gradient／loss 相對量級
- 每 step 時間、peak GPU memory

### 7.3 Sampler 備選

若隨機 batch 常產生 zero-positive anchor，可測試 emotion-balanced／M-per-class sampler。此方案需注意：

- dev/test sampling 不變。
- 訓練資料的實際 exposure 次數需記錄。
- 與一般 sampler 做同 seed 對照。
- 不同 sampler 不與 micro-batch 改動放在同一 run。

---

## 8. Phase C：利用 NNIME 多評分者資訊

只有在資料中能取得合規且可靠的 rater 分布時才啟動。

### 8.1 候選方法

- hard majority label：現行 baseline。
- soft label distribution：用多評分者比例作為 target。
- KL divergence 或 soft cross-entropy。
- 依 rater agreement 對 classification loss 加權。
- 高歧義樣本降低 contrastive attraction，避免把邊界樣本強拉在一起。

### 8.2 驗證假說

Unconfident 常與 Neutral 或 Excited 邊界模糊。soft target 可能比提高 α 更能處理真實標註不確定性。

但不可同時改 α、soft target 與 sampler。先沿用 Phase A／B champion，只比較 hard 與 soft label。

---

## 9. Phase D：台灣服務資料與 augmentation

### 9.1 資料優先順序

1. 合法可用且與服務音訊近似的台灣華語資料。
2. 現有 NNIME train 中的 speaker／場景分析。
3. 內部服務資料的去識別、授權與人工標註。
4. 其他中文語料，只作輔助，不作台灣口音主測試。

資料切分應以 speaker 為單位，避免同一說話者跨 train/dev/test。NNIME 有 44 位說話者，需在 W&B config 記錄 split 版本與 hash。

### 9.2 建議的服務 holdout

若有合規內部資料，建立獨立 `TW-Service-Holdout`：

- 80% train、10% dev、10% locked test 只是起始參考；實際應以 speaker 分組。
- 保留性別、年齡、裝置、噪音、語速與口音差異。
- 不以 ASR 文字準確率代替情緒標註品質。
- 回報 audio-only、text-only 與 fusion 表現，找出服務 domain shift 來源。

### 9.3 低風險 augmentation

先做不應改變情緒語意的音訊擾動：

- 小幅背景噪音。
- room impulse response。
- codec／取樣率退化。
- 小幅 gain。

pitch、tempo、強烈 time-stretch 可能直接改變情緒線索，必須另作 tone-safety 檢查，不能預設為安全。

---

## 10. W&B 統一記錄規格

### 10.1 固定位置

- Entity：`d1249119-feng-chia-university`
- Project：`Crab_Bilingual_ZH`
- URL：[Crab_Bilingual_ZH](https://wandb.ai/d1249119-feng-chia-university/Crab_Bilingual_ZH)

所有 Phase 0–D 的 train、dev 與最終 test 都寫到此 project。

### 10.2 保留既有欄位

現有欄位與命名先保留，避免舊 dashboard 斷裂，至少包括：

- epoch／step／learning rate
- train total loss
- train classification loss
- train contrastive loss
- dev macro-F1／UAR／WAR
- test macro-F1／UAR／WAR
- best epoch／best dev macro-F1

若程式現有 key 名稱不同，v5.0 實作時以現有 key 為準，不批次改寫歷史 run。

### 10.3 建議新增的必要欄位

這些欄位是本輪判斷 α 與 batch 是否有效所必需：

#### Run identity／可重現性

- `experiment_phase`：`phase0`、`phaseA`、`phaseB`、`phaseC`、`phaseD`
- `experiment_id`：例如 `A-200-s42`
- `seed`
- `git_commit`
- `init_checkpoint`
- `parent_run_id`（若為 warm start）
- `dataset_name`、`dataset_version`、`split_hash`
- `label_mapping_version`

#### 超參數

- `contrastive_weight`
- `micro_batch_size`
- `gradient_accumulation_steps`
- `effective_batch_size`
- `contrastive_batch_size`（通常等於 micro batch，需明確記錄）
- LoRA rank／LoRA alpha／LoRA target modules
- optimizer／learning rate／scheduler／class weights

#### Contrastive 診斷

- `mpcl_valid_anchor_ratio`
- `mpcl_zero_positive_anchor_ratio`
- `mpcl_positive_pairs_per_anchor_mean`
- `mpcl_positive_pairs_per_anchor_min`
- 各類別 valid-anchor ratio

#### 中文主指標

- dev 每類 precision／recall／F1
- dev confusion matrix
- dev prediction class percentage
- final test 每類 precision／recall／F1
- final test confusion matrix
- final test prediction class percentage
- 多 seed 彙總表的 macro-F1 mean／std

### 10.4 Run 組織方式

建議命名：

```text
group: zh-v5-phaseA-alpha-sweep
name:  zh-a200-s42-n3ws
job_type: train-alpha-sweep
tags: [zh-TW, NNIME, N3-warmstart, phaseA, alpha-2.0, seed-42]
```

Phase B 另開 `zh-v5-phaseB-microbatch` group，不和 Phase A 混在同一 group。

### 10.5 Artifact 與授權

- 可上傳：config、程式版本資訊、彙總 metrics、圖表與不含個資的 confusion matrix。
- NNIME 原始音訊、逐筆資料或可還原資料內容不得因 logging 自動上傳。
- checkpoint 是否能上傳 W&B artifact，需先依 NNIME 與其他來源授權確認。
- 本機 output、cache 與 checkpoint 目錄必須包含 run ID，避免平行訓練互相覆寫。

---

## 11. 實際執行順序

### Milestone 0：儀表與 correctness

- 補齊 W&B config／summary 欄位。
- 加入 contrastive mask 單元測試。
- 重跑 N3 seed 42 baseline。
- 建立新的中文結果表。

### Milestone 1：α sweep

- 跑 8 個 seed 42 首輪設定。
- 自動產生 dev macro-F1、per-class F1、anchor diagnostics 排名。
- top 3 跑 seeds 100、2026。
- 選出 α*，不看 test。

### Milestone 2：micro batch

- 在 α* 下跑 B4／B8／B16。
- 若顯存不足，記錄 OOM／peak memory，不靜默改參數。
- champion 做多 seed 複驗。

### Milestone 3：最終評估

- 凍結 config 與 checkpoint。
- 只對 champion 跑 locked NNIME test。
- 若已有 `TW-Service-Holdout`，再跑一次服務測試。
- W&B 建立 summary table，與 N3 0.5877 並列。

### Milestone 4：條件式後續

- α／batch 有提升：再評估 soft labels。
- α／batch 無提升：轉向 rater distribution、service data 與 sampler。
- 不在沒有診斷證據時繼續堆疊 loss。

---

## 12. 停止與回退條件

- 任一 run 出現 NaN／Inf：停止同組排程，先修 correctness。
- N3 baseline 無法在合理範圍內重現：停止 sweep。
- 三個 seed 的改善小於變異：不宣稱提升。
- macro-F1 增加但 Unconfident F1 下降超過 0.02：不直接晉級服務模型。
- valid-anchor ratio 很低：α 結果標為 inconclusive，轉 Phase B／sampler。
- 訓練中斷後 resume 必須沿用同一 W&B run ID；重新開始則建立新 run，不覆寫舊結果。

回退點永遠保留：

1. 目前 N3 checkpoint。
2. Phase 0 重現 baseline。
3. 每個 phase 的正式 champion。

---

## 13. 舊方案整理

### 13.1 已完成且保留為證據

- scheme 1：英文最佳基線。
- bilingual v1：N3 warm-start 的表示來源。
- v2b：大規模 joint bilingual 對照。
- N3 v1-warmstart stage 2：目前中文最佳。
- N2 scheme1-warmstart：低於 N3，證明初始化來源重要。

### 13.2 已知負結果／不優先重跑

- Hybrid B
- C14 Full FT
- C1／C1b／C5
- F1-B
- v2 fear boost 3.0
- v2c fear boost 2.0
- fresh NNIME 3k

除非程式 correctness、資料 split 或評估方式被證明有重大問題，否則不重跑以上路徑。

### 13.3 v5.0 已修正的舊假設

| 舊假設 | v5.0 處理 |
|---|---|
| contrastive weight 2.0 為鎖死軸 | 中文正式做 α 消融 |
| N3 effective batch 為 64 | 修正為 micro 4／effective 16 |
| bilingual overall 指標可代表中文服務 | 改以台灣華語 macro-F1 為主 |
| 單次 best run 可作結論 | 改為 top 設定多 seed 複驗 |
| test 可用於持續選模 | test 鎖定，只給 phase champion |

### 13.4 暫緩項目

- 英文 scheme 1 的進一步優化。
- 新的 multilingual joint training。
- 大型 backbone 更換。
- 多任務／生成式情緒解釋。
- 未確認授權的外部中文情緒語料。

這些項目不是永久取消，而是在台灣華語主線完成 α、batch 與 service holdout 前不占用主要資源。

---

## 14. 結果記錄模板

### 14.1 單一 run

| 欄位 | 值 |
|---|---|
| W&B run URL | |
| Experiment ID | |
| Git commit | |
| Init checkpoint | |
| Seed | |
| Contrastive α | |
| Micro／accum／effective batch | |
| Best epoch | |
| Dev macro-F1 | |
| E／U／N F1 | |
| Valid-anchor ratio | |
| Positive pairs mean | |
| Notes | |

### 14.2 多 seed 候選

| Candidate | α | Micro batch | Seeds | Dev macro-F1 mean ± std | Lowest class F1 | 決策 |
|---|---:|---:|---|---:|---:|---|
| N3 corrected baseline | 2.0 | 4 | 42,100,2026 | | | |
| Candidate 1 | | | 42,100,2026 | | | |
| Candidate 2 | | | 42,100,2026 | | | |

### 14.3 最終 locked test

| Model | Dev macro-F1 mean ± std | NNIME test macro-F1 | Excited F1 | Unconfident F1 | Neutral F1 | TW service holdout | W&B |
|---|---:|---:|---:|---:|---:|---:|---|
| N3 historical | — | 0.5877 | 0.5825 | 0.4823 | 0.6982 | — | |
| v5 champion | | | | | | | |

---

## 15. 版本紀錄

| 版本 | 日期 | 變更 |
|---|---|---|
| v5.0 | 2026-09-02 | 文件改為台灣華語優先；新增 α 消融、micro-batch／anchor 診斷、locked-test 規則與 W&B 統一欄位；整理完成、失敗與暫緩方案 |
| v4.2 | 2026-07-07 | 記錄 N3 v1-warmstart stage 2 test 0.5877 與 stage-2 transfer 結果 |

---

## 16. 本輪授權邊界

目前只完成計畫與欄位規格。以下動作需在下一步明確執行：

- 修改訓練程式與 W&B logging。
- 修正或驗證 contrastive mask。
- 建立 sweep launcher。
- 啟動任何 GPU 訓練。
- 建立或上傳 W&B runs／artifacts。

在正式開跑前，先依本文件完成 Phase 0；中文 α sweep 是第一個正式訓練批次。
