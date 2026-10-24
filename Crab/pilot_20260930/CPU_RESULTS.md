# 入學前前測：已完成的 CPU 證據

2026-09-30。來源腳本：[prepare_cpu.py](prepare_cpu.py)；機器可讀結果與來源 hash：[cpu_v1/summary.json](cpu_v1/summary.json)。**以下只有資料統計，沒有模型處理效應、沒有前測推論的信賴區間。**

## 1. A75 尾端精確零值

來源：`../experiments/n3_phase_a_long12_20260909/n3a75_train_dev_test.csv` 的 471 筆 TTS Train；不是從 576 筆 max12 候選任意重抽。每支 WAV 為 24 kHz、mono、16-bit PCM。

定義：從最後一個樣本往前，連續等於整數 0 的樣本數除以取樣率；遇非零即停止。不是 RMS/VAD 靜音，也不把低能量非零樣本算為零。主要分位數以全部 471 筆計算，包含 99 筆零長度尾段；分位數採 linear interpolation，index=(n−1)q。

| 指標 | 結果 |
|---|---:|
| 句數 | 471 |
| 結尾至少一個零值樣本 | 372 |
| 完全沒有零值尾段 | 99 |
| 中位數 | **16.208333 ms** |
| 第 90 百分位 | **82.416667 ms** |
| 最大值 | **257.375 ms** |
| 嚴格大於 10 ms | **328** |
| 嚴格大於 50 ms | **132** |
| 嚴格大於 100 ms | **11** |

補充：僅計 372 筆非零長度尾段時，中位數 22.479167 ms，P90 92.125 ms；**不是本草案 b/c 條件使用的分位數**。原規格「多數可能只有幾個樣本」目前不宜沿用，已有 328/471 支超過 10 ms。

若在 16 kHz 浮點原始波形加尾段，以四捨五入到最接近樣本數：

| 用途 | 樣本數 | 實際追加長度 |
|---|---:|---:|
| 中位數 b/d | 259 | 16.1875 ms |
| P90 c/e | 1319 | 82.4375 ms |
| 敏感度 f | 16000 | 1000 ms |
| 部署模擬 g | 32000 | 2000 ms |

來源：[逐檔資料](cpu_v1/a75_tail_zero_per_file.csv)、[彙總](cpu_v1/tail_and_eligibility_summary.json)。本次未把這些尾段實際接到音檔，也未觀察模型反應。

## 2. NNIME Test 資格

來源：`../data/nnime_crab_format.csv`；只取既有 Test，共 638 筆。逐檔讀 WAV header，不跑 ASR、不讀模型。

| 條件 | 句數 | 用途 |
|---|---:|---|
| 全部真人 Test | 638 | 既有資料集 |
| 原句 ≤11 秒 | 629 | 加最多 1 秒時可落在 12 秒內 |
| 原句 ≤10 秒 | **626** | 加最多 2 秒仍可落在 12 秒內；建議所有 a～g 共用 |
| >10 秒 | 12 | 建議排除於本前測共同子集；原資料不刪除 |

共同子集標籤：E=179、U=157、N=290。team 4=102、20=222、21=187、22=115。這不是重新切分訓練資料，也不能把 626 筆結果冒稱為完整 638 筆結果。

638 筆對應 session-level WAV 路徑均存在；**檔案存在不代表已找到合格無語音底噪**。已建立 626 列 [noise_review_template.csv](cpu_v1/noise_review_template.csv)，起訖時間及人工確認欄位尚空白，approved 全為 False。

來源：[資格清單](cpu_v1/nnime_test_eligibility.csv)。12 秒截斷由 `../src/data/dataset/dataset.py:268` 的 LazyWavSet 確認；本前測必須另加超長就報错的保護，不得靜默截斷追加刺激。

## 3. 模型盤點，不代表已驗收推論

| 組別 | run 數 | 劑量／seed | CL | 訓練輪數／早停 | 本前測取用 |
|---|---:|---|---:|---|---|
| 新 A full10 | 9 | 25/50/75 × 42/43/44 | 2 | 10／patience 0 | 各自最佳 Dev checkpoint |
| B Match-min1 | 3 | 25/50/75 × 42 | 2 | 10／patience 0 | 各自最佳 Dev checkpoint |
| 原 N3 | 1 | 無 TTS／42 | 2 | 上限 10／patience 5 | 最佳 Dev checkpoint |

A/B 來源：`../experiments/n3_phase_a_multiseed_20260911/campaign.json`、`../experiments/n3_phase_b_matchmin1_s42_20260912/campaign.json` 的 common_cli／arms，並連到各 run_result.json。N3 來源：`../bin/run_v1_warmstart_nnime_lora.sh`，原訓練 run=`bkndrhih`。

13 個模型所需 adapters、head、mode 與 Train normalization 檔均存在。沒有載入任何 checkpoint，也未重新雜湊權重本體或做 forward smoke。因此 [model_inventory_DRAFT.csv](cpu_v1/model_inventory_DRAFT.csv) 全部 ready_for_inference=False。

## 4. 三類先驗差重新核對

來源：V1 實際 `../data/bilingual_strategyA.csv` 的 Train。

| 來源 | E | U | N | 合計 |
|---|---:|---:|---:|---:|
| MSP | 10934 | 5183 | 13883 | 30000 |
| EmotionTalk | 2807 | 1369 | 7568 | 11744 |

\[
d_c=\ln\frac{n_{MSP}(c)}{n_{MSP}(N)}-\ln\frac{n_{ET}(c)}{n_{ET}(N)}.
\]

| 分量 | 重算結果 |
|---|---:|
| E/N 差 | **+0.753024665** |
| U/N 差 | **+0.724567255** |
| 向量方向 atan2(d_U,d_E) | **43.896659°** |

與原規格的 +0.75／+0.72／43.9° 相符。但這是語料標籤先驗的參考向量，**不是整句英文模型反應的數學上限，也不是已證實的模型語言效應**。

### 原 30000 筆是怎麼抽的？

`../scripts/build_bilingual_train_csv.py:99`：先使用 Scheme 1 過濾後的 MSP Train，再做 pandas `sample(n=30000, random_state=42)`，**不放回、非分層抽樣**。本次用 NumPy 1.24.4 的同一 RandomState choice 重現，與 V1 30000 筆的檔名及順序完全一致；原始共識清單過濾後的次序也與 Scheme 1 CSV 一致。

## 5. 七類先驗差：不是七類模型成績

### 來源與排除

- MSP：`../../datasets/MSP_Podcast_Data/Labels/labels_consensus.csv`，`Split_Set=Train`、標籤欄位 `EmoClass`。保留 N/H/S/A/U/F/D；此處 U 原碼代表 Surprise。
- MSP Train 排除：Contempt C=2765、Other O=1746、No agreement X=30276。七類保留 **134403** 筆。
- EmotionTalk：`../../datasets/emotiontalk/Audio/Audio/json/` 的 `emotion_result`，沿用 `build_emotiontalk_crab_csv.py:51` 的 EMO_NORM。沿用 group split：1/12 為 Dev、3/15 為 Test，其餘 Train；核對原始 WAV 存在。本次未出現未解析 group、未知情緒或缺 WAV 的排除。
- EmotionTalk 七類 Train 共 **15413** 筆。把 anger/disgust 移除後，檔名集合精確等於 V1 的 11744 筆，不只總數相等。
- 七類 MSP 30000 筆：回到新的七類 Train 候選池，再按相同不放回 seed42 方法抽樣。**是供未來七類方案參考的新候選，不是歷史 V1 的原 30000 筆**；原清單已排除了 anger/disgust，不可能直接還原成涵蓋七類的同一清單。

### 筆數與比例

每格為筆數（該欄比例）；百分比四捨五入。

| 原始七類 | MSP 全七類 Train | MSP 同選法 30000 筆候選 | EmotionTalk 七類 Train |
|---|---:|---:|---:|
| 中性 neutral | 51151（38.06%） | 11316（37.72%） | 7568（49.10%） |
| 快樂 happy | 37048（27.56%） | 8315（27.72%） | 1720（11.16%） |
| 悲傷 sad | 18256（13.58%） | 4138（13.79%） | 898（5.83%） |
| 生氣 anger | 22609（16.82%） | 5092（16.97%） | 3057（19.83%） |
| 驚訝 surprise | 3220（2.40%） | 677（2.26%） | 1087（7.05%） |
| 恐懼 fear | 794（0.59%） | 180（0.60%） | 471（3.06%） |
| 厭惡 disgust | 1325（0.99%） | 282（0.94%） | 612（3.97%） |
| **合計** | **134403** | **30000** | **15413** |

### MSP 相對 EmotionTalk、各類相對中性的 log-ratio 差

| 類別 c | MSP 全七類 Train − ET | MSP 七類 30000 候選 − ET |
|---|---:|---:|
| 快樂 | +1.159037 | +1.173448 |
| 悲傷 | +1.101226 | +1.125509 |
| 生氣 | +0.090061 | +0.107948 |
| 驚訝 | −0.824893 | −0.875794 |
| 恐懼 | −1.388628 | −1.364190 |
| 厭惡 | −1.138418 | −1.177114 |

來源：[seven_class_counts.csv](cpu_v1/seven_class_counts.csv)、[seven_class_logratio_differences.csv](cpu_v1/seven_class_logratio_differences.csv)、[prior_summary.json](cpu_v1/prior_summary.json)。零計數規則為回報 undefined，不加任意 pseudocount；本次六個差值都可計算。

**解讀限度：**三類 U 把 sad 與 fear 合併，但七類中 sad 差為正、fear 為負；E 合併 happy 與 surprise，也有相反方向。研究計畫改為七類時，不能把三類的 +0.75／+0.72 直接套到七類或宣稱所有非中性類都會上升。語言與資料來源同時改變，仍不能只靠分布數字證明模型使用語言捷徑。
