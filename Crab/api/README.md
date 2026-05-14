# Crab Bimodal Emotion API

基於 FastAPI 構建的高性能面試情感辨識 REST API。本模組將訓練完成的 Crab 雙模態模型（WavLM-Large + RoBERTa-Large）封裝為生產環境可用的服務，專為 LangGraph 面試模擬器設計。

## 🌟 核心特性

- **雙模態融合 (Bimodal)**：結合語音聲學特徵 (WavLM) 與語意文字特徵 (RoBERTa)，比單純聽聲音更懂面試者的情緒。
- **極速平行批次 (Batch Processing)**：支援一次上傳多個音檔與文字，利用 GPU 進行平行矩陣運算，吞吐量提升 3 倍以上。
- **零啟動延遲 (Warm-up)**：服務啟動時自動進行 GPU 預熱，消除了 PyTorch 首次推論卡頓的問題。
- **無縫相容性**：輸出格式保留了與舊版 SAILER 相同的欄位名稱，降低系統整合成本。

## 📁 目錄結構

```text
Crab/api/
├── app.py                # FastAPI 應用程式與路由
├── inference.py          # 核心推論引擎（模型載入、前處理、Mask處理）
├── schemas.py            # Pydantic 資料模型與類型檢查
├── test_latency.py       # 延遲與性能基準測試腳本
└── requirements.txt      # API 專屬依賴套件
```

## 🚀 快速開始

### 1. 安裝依賴
```bash
cd /home/brant/Project/SAILER_test
Crab/.venv/bin/python -m pip install -r Crab/api/requirements.txt
```

### 2. 啟動服務
```bash
cd /home/brant/Project/SAILER_test/Crab
.venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8001 --workers 1
```

### 3. 性能測試
```bash
cd /home/brant/Project/SAILER_test
# 請替換為真實的 wav 檔案路徑
Crab/.venv/bin/python -m Crab.api.test_latency --wav /path/to/test.wav
```

## 🔌 API 規格簡介

### 輸出類別 (3-Class)
- `Excited` (興奮/積極)
- `Unconfident` (缺乏自信/焦慮)
- `Neutral_3Class` (平靜/中立)

### 主要端點

#### 1. `GET /v1/health`
- **功能**：健康檢查與 GPU 狀態查看。
- **回傳**：模型版本、運行設備、GPU 型號與剩餘顯存。

#### 2. `POST /v1/emotion/classify`
- **功能**：單筆音訊推論。
- **輸入**：`audio` (檔案), `text` (選填文字)。
- **回傳**：主標籤、置信度、三類完整機率、伺服器耗時。

#### 3. `POST /v1/emotion/classify-batch`
- **功能**：批次平行推論。
- **輸入**：`files` (多檔案), `texts` (多文字)。
- **回傳**：Batch 總耗時、平均耗時、每筆音訊的獨立預測結果。

## 📊 性能指標 (RTX 3090 實測)

| 測試項目 | 實測數據 |
| --- | --- |
| 單筆推論平均延遲 (Latency) | **~42.5 ms** |
| 批次處理 (10筆) 總耗時 | **~133.4 ms** |
| 批次處理每筆平均耗時 | **~12.7 ms** |
| **批次平行加速比 (Speedup)** | **3.19x** 🚀 |
| **系統最高吞吐量 (Throughput)** | **~75 req/s** |

## 📝 互動式文件

服務啟動後，可直接訪問以下網址查看完整 API 參數並進行線上測試：
- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
