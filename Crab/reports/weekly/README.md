# Weekly Reports — Crab 中英雙語化

每週實驗成果的總結報告。用於固定週會 / 老師 review / paper 撰寫參考。

## Naming convention

`YYYY-MM-DD_to_MM-DD_<theme>.md`(日期範圍)

## 索引

| 週期 | 檔案 | 主題 | 核心 claim |
|---|---|---|---|
| 2026-07-03 ~ 07-09 | [2026-07-03_to_07-09_nnime_transfer.md](2026-07-03_to_07-09_nnime_transfer.md) | NNIME transfer 三 setup 比較(N1/N2/N3)| **N3 v1-warmstart test 0.5877 超越 v2b +0.033** ⭐⭐;🅒-full 110k EN dev 0.5405 < v1 混訓 |
| 2026-07-10 ~ 07-23 | [2026-07-10_to_07-23_encoder_swap_and_forget.md](2026-07-10_to_07-23_encoder_swap_and_forget.md) | Encoder swap 雙 stage ablation(scheme1-XLMR-FullFT + N4-B)+ N3 catastrophic forget 定量 | **Encoder swap net +0.004(pretrain -0.022 + downstream +0.026)**;**N3 forget EN -0.1226 vs +0.033 ZH → 非 free lunch,是 domain specialization trade-off** |

## 撰寫 template(參考 W1 檔為模板)

每份報告應含 9 個 section:

1. **主題 + 核心 claim**(標題下方一句話)
2. **TL;DR**(給老師 30 秒讀,程式碼區塊格式方便直接貼)
3. **本週執行的 events**(表格:日期 / event / 屬性 / result)
4. **關鍵發現**(numbered,paper 素材級別)
5. **核心對照表**(開會白板可以畫的)
6. **下一步 / open questions**(短期 + 中期)
7. **超參數配置附錄**(每個 run 完整 yaml)
8. **程式改動 + 文件更新**
9. **對老師的 one-liner**(只能講一句時說什麼)
