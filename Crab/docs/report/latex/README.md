# 第二章：論文風格 LaTeX 排版

本版提供「章節架構＋可用正文＋［這邊放什麼］」，供老師查看內容安排。以 A4 單欄、12 pt、2.5 cm 邊界、段落首行縮排與章節式圖表／數學式編號呈現。原 Markdown 初稿維持上一版，本輪修訂以此目錄的 LaTeX／PDF 為準。

## 從哪個檔案開始

- `main.pdf`：已編譯的排版預覽，先看這份。
- `main.tex`：主檔；中文字型、邊界、行距、頁首頁尾均在這裡。
- `chapter2.tex`：可修改的八節內容，含正文、公式、圖說與方括號填寫提示。
- `assets/proposal_architecture-000.jpg`：原企劃書第 4 頁內嵌模型圖，直接抽出，沒有重畫。
- `assets/duration_distribution.pdf`：由既有 Match-min1 建置統計重製的向量分布圖。
- `make_duration_figure.py`：上述統計圖的重製程式；編譯論文不必重新執行。

正文以文字為主，保留兩張必要表格，不將 Markdown 的 10 張表逐一搬入。模型圖使用既有圖片；後續要放的實驗結果、系統整合、驗收與限制分析，改以具體的方括號提示列出。PDF 不放來源搬移說明、日期式進度或內部審閱註記。

閱讀順序採「先總覽、再細節」：2.2 開頭放中文雙模態架構（圖 2-1）與流程概述，接續說明方法選用；2.3 介紹資料；2.4 再依同一張圖解釋模組與公式，不重複放圖。時長分布圖依出現順序編為圖 2-2。

原有 P01–P09、P11、P12 共 11 組段落保留原句；P10 的歷史結果改為「這邊放結果」的配置，原句仍留在上層架構稿。2.4 與 2.5 的核心公式使用本機 `Crab.pdf` 原式（3）至（8）：注意力池化、MPCL 與 MLCS；正文註明論文引用，章內另行編號。

## 編譯

在具備 XeLaTeX、CTeX 及相關套件的環境，從本目錄執行兩次：

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

亦可用 Tectonic：

```bash
tectonic -X compile main.tex --keep-logs
```

主檔使用 `ctexrep`，不可直接以一般 pdfLaTeX 取代 XeLaTeX。若有 Noto Serif CJK TC 字型，優先使用；否則採 CTeX 的 Fandol 字型。更換環境或字型可能改變換行及頁數。上傳到支援 XeLaTeX 的編輯環境時，需要 `main.tex`、`chapter2.tex` 與 `assets/` 中使用的兩張圖片，不需任何模型、音訊、逐字稿或訓練環境。

本機首次編譯使用獨立暫存目錄的 Tectonic 0.17.0，限制虛擬記憶體為 2 GiB、降低優先序，不使用 GPU；沒有安裝系統 TeX 或更動訓練 venv。工具首次會下載 TeX 套件到暫存快取，此動作不會上傳章節內容。

## 圖片與資料來源

模型圖取自 `Crab/2026全國大專校院智慧創新暨跨域整合創作競賽企劃書12.pdf` 的第 4 頁，原圖約 1088×482 像素。正式印刷建議取得製圖原檔或向量圖，放大 JPEG 不會增加細節。原 PDF 未修改。

時長分布圖使用 `Crab/experiments/datasets/nnime_u_match_min1_20260911/build_report.json` 的 `reference_counts` 與各組 `actual_bin_counts`；程式驗證參考集合為 510、B 組為 157／314／471 筆後繪圖。不讀取個別音訊或逐字稿，也不修改實驗資料。其意義是分布符合建置配額，不是人工情緒驗收通過。

## 後續填寫

- 換上系所正式封面、頁碼、字型及章節格式；目前只是排版初版。
- 補齊完整結果章、正式介面截圖、資料版本及各 split 分布。
- 查證舊文的排行榜與因果敘述，補編碼器、LoRA 文獻並統一引用。
- 依正文的方括號提示填入文字、圖表與案例，完成後再移除該項提示。
- 目前仍是本機審閱文件，未 commit／push，亦未上傳任何線上排版服務。
