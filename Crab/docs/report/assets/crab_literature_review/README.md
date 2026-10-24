# CRAB 文獻回顧：原圖與內容來源

對應正文：[CRAB_LITERATURE_REVIEW.md](../../CRAB_LITERATURE_REVIEW.md)。

本目錄是文件素材，不是新的模型架構、實驗或資料集。

## 來源與版本

- 技術內容唯一主來源：本機 [Crab.pdf](../../../../Crab.pdf)，arXiv:2603.23673v1，2026-03-24。
- 論文：Lucas H. Ueda、João G. T. Lima、Paula D. P. Costa，*Crab: Multi Layer Contrastive Supervision to Improve Speech Emotion Recognition Under Both Acted and Natural Speech Condition*。
- 寫作方式參考：[專題計畫提案書.pdf](../../專題計畫提案書.pdf) 第五部分「文獻回顧與探討」，尤其 5.4 SAILER。僅參考敘事節奏，不沿用其技術主張。
- 正文採「研究問題 → 原架構圖 → 模組拆解 → 原論文公式與變數解釋 → 訓練策略 → 論文實驗 → 選用依據」。不寫成實驗進度報告。

來源 PDF SHA256：

| PDF | SHA256 |
|---|---|
| Crab.pdf | d8403fc62487cff365b1a6e915f97f342cf7557bcbc26823a103e694ff7efe17 |
| 專題計畫提案書.pdf | b4529af2472df3a06c31432814cd8c225b0434c68d1ccd4306110a18be4e1dfd |

## 圖片

| 檔案 | 原文位置 | 擷取方式 |
|---|---|---|
| crab_original_architecture.png | 第5頁 Fig. 1；輸出2150×942 | 原圖內嵌RGB帶透明遮罩；以pdftoppm按原PDF渲染並擷取圖區，保留正確白底與全部文字，不重畫、不換編碼器。 |
| crab_original_alpha_ablation.png | 第8頁 Fig. 3；2898×1717 | pdfimages直接匯出原圖，保留原始UAR/WAR雙軸與數值，已目視核對。 |

兩圖是原論文圖像，不是中文 N3 改版架構，也不是本專題 CL 實驗結果。正文另附中文圖說與來源。原始 PDF、原企劃書及既有結報均未更動。

架構圖的擷取參數（PDF頁面point座標約x48–564、y54–280，未含原英文圖說）：pdftoppm -f 5 -l 5 -r 300 -x 200 -y 225 -W 2150 -H 942 -singlefile -png。原始內嵌圖為3363×1443；直接忽略smask匯出會造成黑色背景，不作交付。

原圖歸原作者；此處僅保留研究引用與來源記錄，不另宣稱圖像可任意再授權。本次未 commit、push 或對外上傳。

## 公式對照

| 正文式號 | CRAB 原式號／頁碼 | 內容 |
|---|---|---|
| (2-1) | (3)，p.4 | 注意力權重，保留分母尺度 sqrt(D)。 |
| (2-2) | (4)，p.4 | 注意力池化。 |
| (2-3) | (5)，p.5 | MPCL 預測分布 q。 |
| (2-4) | (6)，p.5 | 同情緒正樣本的目標分布 c。 |
| (2-5) | (7)，p.5 | MPCL 交叉熵。 |
| (2-6) | (8)，p.5 | CE＋五處 CSL 平均對比損失。 |
| (2-7) | (9)，p.5 | 逆類別頻率權重 N/N_j。 |

數學關係與原式一致，只重新排版、按本章編號。指示函數在 Markdown 用 \mathbb{1} 表達原論文的黑板體 1。

## 結果對照與寫作邊界

- 主結果表：原文 Table II（p.6）、Table VII／VIII（p.10）。
- 單模態對照：Table III（p.8）。
- 訓練目標消融：Table IV（p.8）。
- α 消融：Fig. 3 與 Section IV-A（pp.7–8），UAR 決定 α=2；不能寫成中文 Macro F1 的最佳值。
- 不重述原文 p.10「較最佳非集成方法約高0.08」的近似說法；本稿直接報表中0.4313。也不援用該頁 ABHINAYA 正文與表格數值不一致的部分。
- 不把原論文「CSL不增加推論參數」改寫為整體模型很小、可在RTX3060完成訓練或已通過服務延遲驗證。
- 不把SAILER的Annotation Dropout、Audio Mixing、soft annotator labels、次要情緒／屬性多任務損失寫成CRAB方法。
- 不把本專題的XLS-R、XLM-R、LoRA、NNIME三類映射或TTS擴增寫成CRAB原論文設定。這些僅於選用依據段落交代其屬於方法章。
- 本篇是CRAB專題文獻回顧；對其他基礎方法只整理CRAB原文的回顧與採用情形，沒有冒稱本次已獨立重讀所有被引用論文。

Markdown 使用標準圖片相對連結、行內數學與 $$ 區塊數學。移交時須保留正文旁 assets/crab_literature_review/ 的相對位置。章節與公式號可於整本結報整合時統一調整。
