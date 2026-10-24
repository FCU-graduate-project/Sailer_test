# 入學前前測：準備包（草案，未核准 GPU 執行）

日期：2026-09-30。這是研究計畫前測的獨立準備包，不納入既有 N3 TTS 成績表，也不修改原規格、資料集、模型或 frozen campaign。

原始需求：[入學前前測規格_給3090.md](../入學前前測規格_給3090.md)。本輪只完成 CPU 準備；沒有模型推論、訓練、資料下載、W&B 寫入、queue 提交或搶占。

## 先看這兩份

- [CPU_RESULTS.md](CPU_RESULTS.md)：已查證的尾端零值分布、測試資格、模型清單、三類／七類先驗差。
- [PROTOCOL_DRAFT.md](PROTOCOL_DRAFT.md)：前測設計修訂、比較方式、尚待決定事項與 GPU 放行清單。

## 已完成／未完成

| 工作 | 狀態 |
|---|---|
| A75 471 個 WAV 的精確 PCM 零值尾端統計 | 已完成；逐檔清單與 SHA256 已保存 |
| NNIME 638 筆 Test 的時長與 session 來源 | 已完成；建議所有條件共用 ≤10 秒的 626 筆 |
| A 9＋B 3＋N3 1 模型盤點 | 已完成配置／檔案存在性盤點；未載入權重，尚未做权重 hash 與推論驗收 |
| 三類先驗重算、V1 抽樣方法核對 | 已完成；30000 筆樣本及順序精確重現 |
| 七類全 Train 與同選法 30000 筆候選 | 已完成計數；候選不是歷史 V1 訓練清單 |
| 真實底噪選段／無語音驗收 | 未完成；僅建立空白驗收表 |
| 前測一 GPU 推論與 CI | 未開始；尚未實作／驗收 GPU runner，不能直接提交 queue |
| 前測二自錄 100 句 | 尚待錄音與規格確認 |
| 前測三 ASR 資料／標註／推論 | 暫緩；未下載、未跑 Whisper |

## 機器狀態快照

準備開始時唯讀觀察：RTX 3090 使用率約 97%，總使用 15709 MiB；SSAMBA train worker PID 3902210 使用 15694 MiB。queue health 回報 service=train-queue、test_mode=false；overview 回報以下任務 running：

- Job ID：`01893a110a3842ae9b780fc1e2df793c`
- 名稱：`experiment-two-formal-20260929--train--fusion_heterogeneous_s42`
- Queue：`http://127.0.0.1:8765`

這是當次 `nvidia-smi`、程序清單與 GET `/health`、`/api/v1/overview` 的快照，不是永久狀態。沒有發送 POST／pause／urgent 請求。啟動前測前必須重查，不得據此宣稱 GPU 現在空閒。

## 重現 CPU 準備

唯一新增的程式：[prepare_cpu.py](prepare_cpu.py)。它只讀來源，僅在本資料夾下建立一個新的 `cpu_*` 子目錄；目的地已存在就拒絕執行，避免覆寫。

```bash
cd /home/brant/Project/SAILER_test/Crab/pilot_20260930
/home/brant/Project/SAILER_test/Crab/.venv/bin/python -B -m unittest -v test_prepare_cpu.py
nice -n 15 /home/brant/Project/SAILER_test/Crab/.venv/bin/python -B prepare_cpu.py --out-dir cpu_recheck_01
```

本次產物位於 [cpu_v1/](cpu_v1/)。限制：單執行緒、CUDA 不可見、2 GiB process address-space 上限；未匯入 torch，未開子程序。實際峰值 RSS 為 142808 KiB，約 139.5 MiB；5 項 CPU 單元測試通過。

## 產物與隱私

| 檔案 | 用途 |
|---|---|
| [summary.json](cpu_v1/summary.json) | 執行狀態、來源／輸出 SHA256、記憶體峰值 |
| [tail_and_eligibility_summary.json](cpu_v1/tail_and_eligibility_summary.json) | 尾端分位數、精確閾值計數、資格彙總 |
| [a75_tail_zero_per_file.csv](cpu_v1/a75_tail_zero_per_file.csv) | 471 支音檔逐筆統計及雜湊 |
| [nnime_test_eligibility.csv](cpu_v1/nnime_test_eligibility.csv) | 638 筆 Test 的 ≤11 秒／≤10 秒旗標與來源路徑 |
| [noise_review_template.csv](cpu_v1/noise_review_template.csv) | 626 筆所需底噪的待填驗收表，全部未批准 |
| [model_inventory_DRAFT.csv](cpu_v1/model_inventory_DRAFT.csv) | 13 個模型路徑、seed、劑量、CL、W&B 訓練來源；全部 ready=false |
| [prior_summary.json](cpu_v1/prior_summary.json) | 三類核對、七類差值、原始標籤排除計數 |
| [seven_class_counts.csv](cpu_v1/seven_class_counts.csv) | 七類筆數與比例 |
| [seven_class_logratio_differences.csv](cpu_v1/seven_class_logratio_differences.csv) | 六個相對中性的跨語料對數比 |
| [MSP 七類 30000 筆候選](cpu_v1/msp_seven_train_sample30k_seed42_CANDIDATE.csv) | 只作同抽樣方法的反事實計數，不當訓練授權 |
| [EmotionTalk 標註來源清單](cpu_v1/emotiontalk_train_annotation_inventory.csv) | Train 原始標註位置、雜湊及來源 WAV 是否存在 |

`cpu_*` 已由本資料夾的 `.gitignore` 排除，避免含 NNIME 樣本識別與本機路徑的清單被意外加入公開 repo。未複製音訊或逐字稿，沒有 commit／push。需另確認授權後才能分享清單；Git ignore 不是資料存取控制。
