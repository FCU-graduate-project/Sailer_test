# 2026-09-27：CPU 診斷完成、GPU 驗收待排程

## 本次決定

使用者不做 CL 四組 Test；興奮 TTS 尚未完成，暫不做。錯誤分析與續訓驗收先處理可不占 GPU 的部分，讓另一邊繼續使用 GPU。**本次沒有提交任何 queue 工作、沒有暫停其他工作、没有載入 N3 或進行訓練。**

| 項目 | 本次完成 | 尚未完成 |
|---|---|---|
| A/B 錯誤分析 | 六組既有 **Test** 預測逐筆對照、依長度分組、混淆矩陣、兩張圖；Dev只建立長度清單 | 六組 **Dev** 預測重推與分長度錯誤分析，需要GPU |
| CRAB 暫停／續訓 | 既有CPU測試重跑40/40通過，0錯誤、0略過；程式SHA與9/15一致 | 完整N3模型的連續／恢復數值對照、真實queue插隊、W&B雲端續接，需要GPU |
| 資源隔離 | CPU測試要求1核心、RAM上限3GiB；分析要求1核心、RAM上限2GiB，低優先序、禁swap，兩者CUDA隱藏 | 不以CPU小模型記憶體量推估完整N3訓練峰值 |

## 1. 錯誤分析：目前只有 Test 逐筆資料

[圖表與完整分析](../../experiments/n3_ab_test_cpu_diagnostics_20260927/README.md)／[機器可讀結果](../../experiments/n3_ab_test_cpu_diagnostics_20260927/summary.json)。原始Test分數、模型、音訊、逐字稿均未改；衍生表不匯出逐字稿、不上傳雲端。

目前掃描兩個訓練campaign可見Dev log與彙總，但沒有可用的Dev逐筆預測。檔案中的 `Development` 是Dev，共608筆；Test共638筆。Test對照保留原638筆順序，與音訊雜湊核對一致。

下次Dev重推需先實作／檢查專用入口，再排隊；**本次沒有建立已可提交的Dev工作或宣稱Dev分析完成**。

| Dev重推必須固定 | 規格 |
|---|---|
| 模型 | full10 A25／50／75與B25／50／75，全部seed42、各自best Dev |
| 資料與順序 | 原Development 608筆，依各自凍結CSV核對同一順序；非Test |
| Dev batch | 原trainer為micro4 × 4＝16，shuffle=False；不能直接套Test batch1 |
| 特徵與補齊 | 原collate、FP32、16kHz、最多12秒、文字128、各自保存的Train normalization |
| 重現檢查 | 先與既有best Dev macro-F1及log四捨五入逐類數字核對；不一致先定位，不能覆寫舊分數 |
| 產物 | Dev逐筆檔名／真值／預測／機率，混淆矩陣、每長度區間三類支持數、U precision／recall／F1 |
| 權限與資源 | 共用queue，單卡序列；不在另一邊實驗中擅自插隊，待GPU時段確認 |

## 2. CPU 暫停／續訓複查結果

[40項CPU結果](../../experiments/queue_acceptance_20260927_cpu/cpu_tests.json)／[日誌](../../experiments/queue_acceptance_20260927_cpu/console.log)。Python3.10.20、torch2.2.0+cu121；`gpu_disabled=true`，測試本體3.448秒，主程序maxrss约504MiB，非整組cgroup峰值。

| 已核對 | 證據範圍 |
|---|---|
| 模型／adapters／optimizer／scheduler／RNG／progress | Tiny Linear／LoRA fixture保存和恢復；optimizer狀態由fixture填入，沒有真實N3更新 |
| best與current、early-stop、完成優先序 | 單元測試，不是實際N3訓練曲線驗收 |
| 暫停75、完成0、wrapper exec | 單元測試及小型子程序；不操作正式queue |
| queue manifest／ack | 使用本機queue protocol validator驗證fixture；不等於真實queue插隊成功 |
| W&B同ID／resume=must／step防倒退 | Mock測試；没有新建或修改W&B run |
| RAM限制 | 啟動命令指定MemoryHigh2G／Max3G／Swap0／CPU100%／OOMPolicy=kill；測試後服務正常退出0 |

現行queue已支援per-job RAM限制，且今晚三組N3 Test執行時已驗證生效；**這個工程缺口已補，不再視為阻擋**。剩下的是完整N3的模型恢復與排程驗收，不能拿SSAMBA通過來替代。

## 3. 待GPU時段的實際驗收方案（尚未提交）

先用新專用目錄、相同V1／N3架構、FP32 LoRA、固定seed與資料，保留舊實驗不動。候選小資料規模為Train33／Dev17、3輪，不使用Test；先凍結成員，刻意涵蓋batch尾段與三類。這是工程驗收，數字不計入模型研究成績。

| 次序 | 檢查 | 通過條件 |
|---|---|---|
| 1 | 磁碟／RAM／數值政策前置核對 | 估算完整snapshot大小、保留量與一輪峰值；先決定可重現設定與比較容差，不能事後為通過而放寬 |
| 2 | 不中斷連續3輪 | 保存每輪current/best、optimizer moments／step、scheduler、RNG、樣本順序、Dev與進度 |
| 3 | 第1輪後手動暫停再恢复 | 完成當輪→完整snapshot及ack→退出75；重建程序後執行第2／3輪，與連續基準逐項比較 |
| 4 | 普通驗收工作被急件插隊 | 僅在同意的隔離驗收時段操作；原工作退出75且GPU釋放後才派急件，完成後續回同一工作 |
| 5 | 完成／early-stop與pause同時發生 | 終止條件優先正常退出0，不再排一次續訓；不得改正式研究分數製造結果 |
| 6 | W&B同run與模型輸出 | 雲端同ID、step不回退、不重建run；新完整checkpoint與既有adapter-only評估器另做匯出／reader一致性驗收 |

現有review-only範本保持禁止啟動；沒有移除gate、沒有將正式任務改成epoch_v1。實際GPU驗收driver、凍結清單與queue spec仍需在排定時段前完成，不把本文件當已執行腳本。

主機的CRAB驗證／測試放行仍有效，但「允許驗收」不等於「現在可搶GPU」或「主機根因已修復」。本次不commit／push。
