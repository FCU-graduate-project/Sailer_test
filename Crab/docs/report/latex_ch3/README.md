# 第三章：情緒辨識機制（LaTeX）

接在第二章文獻回顧（CRAB 原理、式 2-1～2-7，於 Notion 撰寫）之後；CRAB 原理與公式不在本章重複。

- `main.pdf`：編譯結果（13 頁）
- `main.tex`：版面設定（A4、12 pt、章節式圖表編號 3-x）；`chapter3.tex`：正文
- `assets/`：圖檔；`make_figures.py` 由既有 CSV／預測檔重繪圖 3-2、3-4、3-5（不讀音訊、不跑模型）
- 灰框［待補］兩處：實際逐句輸入／輸出示例、結果頁截圖（需系統端提供）

## 編譯

```bash
xelatex main.tex && xelatex main.tex      # 或：tectonic -X compile main.tex
python3 make_figures.py                    # 只有改圖時才需要
```

## 數字來源

- TTS 實驗：`docs/n3_tts/RESULTS.md`、`docs/n3_tts/DATASET_AND_REVIEW.md`
- 模型演進、V1、LoRA vs Full FT：`docs/report/CRAB_SAILER_ADMISSIONS_EVIDENCE_20260922.md`、`docs/weekly/2026-07-09_nnime_transfer/2026-07-03_to_07-09_nnime_transfer.md`
- 長度分析與 Test≥3 s 子集：由 `experiments/n3_ab_test_cpu_diagnostics_20260927/test_paired_predictions.csv` 與 `experiments/n3_phase_a_test_20260910/n3_reference/predictions.csv` 事後計算
- 環境音增強：`experiments/n3_u75aug_v1_s42_20261012/README.md`
- API 延遲與回歸：`reports/api/`
