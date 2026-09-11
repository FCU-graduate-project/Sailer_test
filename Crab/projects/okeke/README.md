# OKEKE 四分類期末小作業

**文件狀態：** Historical／paused  
**用途：** 和朋友以既有系統製作的期末小作業；不是目前 N3 中文面試 API
或 A4500 串接主線。

## 目錄

- `docs/`：當時的雙語四分類計畫與紀錄。
- `api/`：四分類推論封裝與歷史 HTTP service。
- `scripts/`：MSP／EmotionTalk 資料建置與分語言評估。
- `train/`：兩個訓練入口。
- `data/`：OKEKE 衍生 CSV 與 class weights；CSV 不納入 Git。
- `models/`：兩組 OKEKE checkpoints；不納入 Git。
- `logs/`：歷史訓練 log；不納入 Git。

共用原始資料仍留在 workspace `datasets/`，MSP 共用 label source 仍留在
`Crab/data/`，避免複製大型資料或破壞其他實驗。

從 `Crab/` 啟動歷史服務：

    .venv/bin/python -m uvicorn projects.okeke.api.ser_service:app \
      --host 127.0.0.1 --port 8100 --workers 1

除非重新啟動這個期末作業，不應把四分類 labels、batch 設定或服務 contract
套用到 N3 三分類 API。全專案文件入口見
[`DOCUMENTATION_INDEX.md`](../../DOCUMENTATION_INDEX.md)。
