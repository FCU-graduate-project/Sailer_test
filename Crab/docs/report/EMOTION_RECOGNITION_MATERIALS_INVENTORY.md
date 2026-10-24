# 情緒辨識結報素材清單

盤點日期：2026-09-17｜v0.1｜本機寫作索引，不是最終成果認定或公開資料包

搭配閱讀：[第二章條列式架構與內容配置初版](CHAPTER_2_EMOTION_RECOGNITION_OUTLINE.md)。

本清單回答「目前有什麼可以拿來寫、放在哪裡、還缺什麼」。本次確認文件內容與素材路徑，沒有重新訓練、重新評估或逐一驗證所有歷史 checkpoint；實驗狀態以現有結果文件截至 2026-09-14 的紀錄為準，不表示此次查詢了即時 W&B 或 queue。

## 一、先看我們已經具備的主線

- **任務與選型背景**：8 月企劃書已有情緒模組動機、Crab 選用理由與中文適配段落，可以原句搬移。
- **資料與模型方法**：已有雙語資料規格、三類映射歷史、編碼器替換、V1→N3 訓練設定及程式。
- **中文 TTS 實驗**：已有完整語音擴增、CL 權重、多 seed、Match-min1 切段的配置及結果。
- **資料製作證據**：已有來源對應、manifest、時長圖、自動稽核、試聽頁與人工驗收模板；沒有完成的人工情緒驗收結果。
- **服務工程素材**：已有 N3 API 規格、schema、推論與測試工具；完整面試系統的實際彙整規則、操作畫面和端到端驗證仍須向整合負責人確認。
- **尚未能當成果的項目**：macro-F1 達 65%、所有 TTS 人工合格、切段穩定較好、真實面試泛化能力與心理自信判定。

## 二、主要文字與程式素材

「可沿用」指可以作為寫作材料，不代表原文中的每項研究主張已重新查證。

| 素材／入口 | 已有什麼 | 建議放哪裡 | 使用前注意 |
|---|---|---|---|
| [8 月競賽企劃書](../../2026全國大專校院智慧創新暨跨域整合創作競賽企劃書12.pdf) | 任務、選型、三資料集、中文編碼器、融合、兩階段訓練、歷史分數、UI 圖 | 2.1–2.5、2.7；部分圖跨章引用 | 原文保留；架構稿 P01–P12 已列頁碼與搬移位置。排行榜、語料規模與舊分數仍須溯源 |
| [雙語微調計畫](../../BILINGUAL_FINETUNE_PLAN.md) | 歷史方案、模型版本、訓練決策 | 2.2、2.4、2.5 | 含不同時期方案，不能把所有配置合成同一個 N3 |
| [資料集規格](../../BILINGUAL_DATASETS.md) | 來源、映射、候選／排除決策、不同版本組成 | 2.3 | 是歷史規格；列為現用不代表某次 run 有用。V1 不等於含 CNSCED／NNIME 的其他混合版本 |
| [雙語工作紀錄](../../BILINGUAL_WORK_LOG.md) | 日期、嘗試路線與研究脈絡 | 選型背景、結果討論、附錄 | 選重要轉折，不把日誌全文貼入正文 |
| [7/03–7/09 週報](../weekly/2026-07-09_nnime_transfer/2026-07-03_to_07-09_nnime_transfer.md)、[7/10–7/23 週報](../weekly/2026-07-23_encoder_swap_and_forget/2026-07-10_to_07-23_encoder_swap_and_forget.md) | NNIME 遷移、編碼器替換與歷史比較紀錄 | 2.2、2.4 或歷史結果 | 作溯源入口；Scheme 1 與 N3 的分數須對回相同 split／指標 |
| [N3 TTS 文件總覽](../../N3_PHASE_A_RESULTS_AND_NEXT_STEPS_20260910.md) | 資料、計畫、結果三份主文件的入口 | 寫作導航，不必放正文 | 優先讀新版主文件，不以 archive 取代最新紀錄 |
| [資料處理與驗收](../n3_tts/DATASET_AND_REVIEW.md) | 完整／切段 TTS、Train-only、Match-min1、稽核與人工流程 | 2.3、2.8 | 自動檢查與探索性使用批准，不等於人工情緒通過 |
| [訓練計畫](../n3_tts/TRAINING_PLAN.md) | V1／N3、loss、超參數、A／CL／seed／B 及後續提案 | 2.5 | 已執行配置以各 frozen campaign 為準；未執行提案列未來工作 |
| [結果與證據索引](../n3_tts/RESULTS.md) | Dev／Test、各類 F1、precision／recall、混淆矩陣、原因假說及原始檔連結 | 結果章；本章只引用必要摘要 | 分開 Dev／Test、歷史 patience5 與 full10；不只摘最高 Test |
| [模型程式](../../src/models/ser.py)、[LoRA 訓練程式](../../bin/train_crab_lora.py) | 實際融合架構與訓練實作 | 2.4、2.5 的核對依據 | 正文解釋原理，不貼大量原始碼；需綁定對應 run 版本 |
| [N3 API 文件](../../api/N3_API.md)、[schema](../../api/schemas.py)、[推論程式](../../api/inference.py) | 模型身分、輸入限制、分數輸出、單筆／batch／long 服務 | 2.1、2.6 | 舊 api/README 是歷史 Scheme 1 入口；N3 以 N3_API 為主 |
| [API benchmark](../../api/benchmark_api.py)、[load](../../api/load_test.py)、[soak](../../api/soak_test.py)、[N3 評估工具](../../api/evaluate_n3.py) | 效能、穩定性及回歸的工具 | 系統實作／驗證章或附錄 | 有工具不等於已在最終部署條件驗收；需另附實際輸出與環境 |
| [Queue 整合文件](../queue/N3_QUEUE_INTEGRATION.md)、[queue trainer](../../bin/train_crab_queue.py) | 訓練排程、保存與接續的工程入口 | 附錄或工程章 | 本機文件可能早於其他分支的修訂；不把其他模型驗收當成 CRAB 真實 resume 證據 |

## 三、可放入結果章的實驗素材

以下是 N3 中文 TTS 擴增實驗線的盤點，不是整個專題歷來訓練總數。原 V1／N3 是歷史初始化／基線，不計入這次 19 個微調 run。

| 素材組 | 已有規模與評估 | 原始證據入口 | 可以支持／尚不能支持 |
|---|---|---|---|
| 原 N3 與歷史 A | A25／50／75 共 3 個訓練；原 N3＋A 共 4 個 checkpoint 的 Test | [歷史 A 目錄](../../experiments/n3_phase_a_long12_20260909/)、[campaign](../../experiments/n3_phase_a_long12_20260909/campaign.json)；Test 入口見主結果文件 | 可呈現補 U 後的三類取捨；不是多 seed 因果結論 |
| CL 消融 | 固定 A25，λ=0／1／2／3，共 4 個 run，Dev-only | [CL 目錄](../../experiments/n3_a25_cl_ablation_20260911/)、[campaign](../../experiments/n3_a25_cl_ablation_20260911/campaign.json) | 可比較本次權重；λ=0 只關閉 stage 2 CL，不抹除 V1 的學習歷史 |
| A 多 seed | 3 劑量 × seeds42／43／44＝9 個 run，Dev-only | [A 多 seed 目錄](../../experiments/n3_phase_a_multiseed_20260911/)、[Dev 彙總](../../experiments/n3_phase_a_multiseed_20260911/aggregate_dev.json) | 可列平均、樣本標準差與個別 run；不可借用舊 A 的 Test |
| B Match-min1 | B25／50／75 × seed42＝3 個 run，Dev 與 3 組 Test 已有紀錄 | [B 目錄](../../experiments/n3_phase_b_matchmin1_s42_20260912/)、[配對 Dev](../../experiments/n3_phase_b_matchmin1_s42_20260912/paired_dev_comparison.json) | 與新 A seed42 的同劑量配對目前是 Dev；新 A 未有配對 Test，不能做完整配對 Test 結論 |

- 合計已有 **19 個微調 run、7 組 checkpoint 的獨立 Test 評估**；Test 評估不是另 7 次訓練。
- 可整理的結果圖表：歷史 A 的 Dev／Test 對照、CL 權重曲線、多 seed 平均與離散程度、A/B 配對 Dev、各類 F1 與 U precision／recall、Test 混淆矩陣。
- 數字統一從主結果文件追到 `run_result.json`、彙總 JSON、逐筆預測與 log；本素材表不再維護第二套完整排行榜。
- 現有紀錄未達 macro-F1 65%；B75 Test 60.84% 是本線已記錄的描述性最高值，不直接作為依 Test 挑選的正式最佳模型。
- 短片段可能失去情緒線索是待驗證解釋；未人工驗收是已知缺口，不能寫成已證實的退步原因。

## 四、現成圖表與可再製的圖

| 素材 | 現況 | 建議處理與放置 |
|---|---|---|
| 企劃書 PDF p.4 圖 1 | 模型圖已存在 | 2.4 優先沿用；本次尚未另行抽圖或檢查圖中文字，定稿核對引用與編碼器 |
| 企劃書 PDF p.5 系統圖、p.6–7 UI 圖 | 可作整體系統／介面素材候選 | 跨章引用；未必都是情緒模組畫面，避免全部塞入第二章 |
| [TTS 製作流程圖](../assets/nnime_tts_uonly_v1/pipeline.svg) | 舊規劃圖已存在 | 2.3 候選；先對照後來實際 A／B 規格，不能直接當最終實作圖 |
| [擴增比例圖](../assets/nnime_tts_uonly_v1/ablation_mix.svg)、[時長目標圖](../assets/nnime_tts_uonly_v1/duration_targets.svg) | 舊規劃圖已存在 | 核對分母 U=628、版本和時長限制後再選用 |
| [初期時長比較圖](../n3_experiment_phase1_20260909/duration_comparison.svg)、[實驗路線圖](../n3_experiment_phase1_20260909/experiment_roadmap.svg) | 初期資料稽核／規劃素材 | 保留日期；不能把當時規劃項全寫成已完成 |
| [Match-min1 實際分布圖](../../experiments/datasets/nnime_u_match_min1_20260911/duration_distribution.svg) | 切段版本圖已存在 | 2.3 或結果章，明確標為參考 Train U≥1 秒，而非完整 NNIME 時長分布 |
| [標註校準流程](../assets/nnime_tts_uonly_v1/calibration_workflow.svg) | 流程設計圖已存在 | 品質方法或附錄；標成規劃／待完成人工驗收 |
| 實驗曲線與混淆矩陣 | 各 campaign 的本機 W&B 紀錄有部分圖片；主結果文件另有矩陣表 | 挑定 run／split／最佳 epoch 後再製圖；不隨便取資料夾內任一張當最終結果 |
| V1→N3 與 A／CL／B 關係、推論流程 | 架構稿已有文字圖與 Mermaid 初稿 | 可作正式製圖底稿；推論流程需整合人員核對 |

本次僅盤點檔案與用途，沒有重製 SVG、擷取 PDF 圖片或新增結果曲線。

## 五、資料製作與驗收素材：有工具，但不是已驗收資料集

- [Match-min1 資料包說明](../../experiments/datasets/nnime_u_match_min1_20260911/README.md)：版本、來源及交付說明。
- [試聽頁](../../experiments/datasets/nnime_u_match_min1_20260911/review.html)與[人工驗收空白表](../../experiments/datasets/nnime_u_match_min1_20260911/review_template.csv)：471 支不同候選，B25⊂B50⊂B75；不是 942 支獨立音訊。
- [建置報告](../../experiments/datasets/nnime_u_match_min1_20260911/build_report.json)、[獨立稽核](../../experiments/datasets/nnime_u_match_min1_20260911/independent_audit.json)與[B 訓練前稽核](../../experiments/n3_phase_b_matchmin1_s42_20260912/preflight_audit.json)：可支持來源、格式、成員與分布的可追溯性。
- [NNIME Train 校準包](../calibration/nnime_uonly_v1/README.md)與[標註者指南](../calibration/nnime_uonly_v1/rater/README.md)：已有校準流程及材料；不能冒稱已取得標註一致性結果。
- 所有上述音檔、逐字稿、個別樣本資訊維持本機／授權範圍；正式結報優先使用彙總統計與流程，不直接附上可公開下載的 NNIME 原始資料。
- 如果日後補驗收：另存標註版本、日期、分歧處理與一致性結果，不能倒填成舊實驗開始前已完成。

## 六、還缺什麼，會影響結報完整性？

- **老師先確認即可，不必等實驗結束的事項**
  - 採用第二章八節架構，或按老師要求合併為六節。
  - 全書是否另設「實驗結果與討論」章；若沒有，第二章需另增結果節。
  - 確認中文為主線；英文 Scheme 1 作選型／演進背景，避免主線分散。
- **定稿前必須補齊的證據**
  - 最終各模型名稱、run、資料版本／split、最佳 checkpoint 與對應表格的核對表。
  - 各資料集實際映射後的 Train／Dev／Test 三類筆數、比例與時長；不能拿歷史混合版總數代替 N3 使用量。
  - 原企劃書 0.320、0.403、0.482 的 run／指標／split 溯源，以及「當時最佳」的比較來源。
  - Crab、XLS-R、XLM-R、LoRA 與各資料集的正式文獻與版本；本次只保留舊文獻，尚未外部查證。
  - 系統實際切分、ASR 對齊、彙整規則、回饋頁畫面及一條端到端示例。
  - 若報告主張實際部署效能：須提供最終環境下的延遲／資源／失敗率等實測，而非只有工具檔案。
- **有助提升研究強度，但不能為交稿而補造的事項**
  - 新 A 的保留 Test 與 B 的配對評估；是否執行須另確認實驗方案，不能因寫報告就自行跑。
  - 對照 0% TTS 的同政策多 seed 基線、更多 B seeds 或信賴區間；未完成者明列限制。
  - 人工驗收、錯例分析與長度分層分析；若不做，保留探索性定位。
  - 真實台灣華語面試資料的授權收集與驗證；NNIME 成績不能直接替代實際面試效果。

## 七、寫作時的素材使用順序

- 第一輪：用章節架構稿讓老師確認大綱與圖表安排，不急著塞所有分數。
- 第二輪：搬移 P01–P12 可沿用段落；每項舊敘述的修訂另列，不默默改字或換數字。
- 第三輪：從資料、訓練、結果三份主文件抽取表格，逐項連到原始證據。
- 第四輪：取得系統端實際畫面與規則，完成推論／回饋兩節。
- 第五輪：統一模型名稱、日期、指標、圖號、引用與時態；分清原架構、本組工作、探索結果及未來計畫。

**本次範圍：** 只整理兩份結報準備文件；不搬移或刪除素材、不改 frozen campaign、不啟動實驗、不 commit／push。清單中的本機連結也不代表相關素材已備份到 GitHub。
