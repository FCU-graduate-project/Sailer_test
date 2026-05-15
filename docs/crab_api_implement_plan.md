# Crab Bimodal Emotion Classifier — API Implementation Plan

**Version:** 1.0  
**Date:** 2026-05-10  
**Model:** WavLM-Large + RoBERTa-Large → 3-class Interview Emotion  
**Base URL:** `http://localhost:8001`  
**Replaces:** Section 3 (SAILER/Whisper) in `docs/api_spec.md`

---

## 0. 現有規格差異對照

| 面向 | 原始 SAILER (Section 3) | Crab Bimodal (本計畫) |
|---|---|---|
| 架構 | Whisper-Large-V3 單模態 | WavLM-Large + RoBERTa 雙模態 |
| 輸入 | 音訊 only | 音訊 + 文字逐字稿 |
| 輸出類別數 | 8 類原始情緒 | 3 類面試情緒 |
| 輸出標籤 | Neutral/Angry/Sad... | Excited/Unconfident/Neutral_3Class |
| VAD 輸出 | ✅ arousal/valence/dominance | ❌ 不輸出（模型未訓練此頭） |
| 端點路徑 | `POST /classify-emotion` | `POST /v1/emotion/classify` |
| 批次端點 | ❌ 無 | ✅ `POST /v1/emotion/classify-batch` |
| 權重來源 | HuggingFace Hub | 本地 `experiments/interview_scheme1/` |

---

## 1. 目錄結構

```
Crab/
├── api/
│   ├── __init__.py
│   ├── inference.py          # CrabEmotionPredictor 核心推論類別
│   ├── app.py                # FastAPI 應用程式
│   ├── schemas.py            # Pydantic Request / Response 模型
│   └── test_latency.py       # 延遲基準測試腳本
├── experiments/
│   └── interview_scheme1/
│       ├── final_ser.pt      # MultiModalEmotionClassifierDeep 權重
│       ├── final_ssl.pt      # WavLM-Large 權重
│       └── final_text.pt     # RoBERTa-Large 權重
└── docs/
    └── api_spec.md           # 更新 Section 3
```

---

## 2. Phase 1｜`api/schemas.py` — 資料模型定義

定義所有 Request / Response 的 Pydantic schema，讓 FastAPI 自動生成 OpenAPI 文件。

```python
# Crab/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

INTERVIEW_CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]

class EmotionProbabilities(BaseModel):
    Excited: float
    Unconfident: float
    Neutral_3Class: float

class SingleClassifyResponse(BaseModel):
    primary_label: str = Field(..., description="argmax of softmax (3-class)")
    primary_confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: EmotionProbabilities
    latency_ms: float

class BatchResultItem(BaseModel):
    filename: str
    primary_label: str
    primary_confidence: float
    probabilities: EmotionProbabilities

class BatchClassifyResponse(BaseModel):
    batch_size: int
    total_latency_ms: float
    avg_latency_ms: float
    results: list[BatchResultItem]

class HealthResponse(BaseModel):
    status: str           # "ok"
    model: str            # "crab-bimodal-scheme1"
    device: str           # "cuda" / "cpu"
    classes: list[str]
```

**重點設計決策：**
- Response 保留與舊 SAILER 相同的 `primary_label` / `primary_confidence` 欄位名稱，讓 LangGraph Emotion Node 不需大幅修改
- `Neutral_3Class` 鍵名直接用訓練時的 class name，避免重新 mapping 造成 bug

---

## 3. Phase 2｜`api/inference.py` — 推論引擎

### 3.1 模型載入策略

```python
# Crab/api/inference.py
import torch
import torchaudio
import torch.nn.functional as F
from transformers import (
    WavLMModel, WavLMConfig,
    RobertaModel, RobertaTokenizer
)
from src.model.ser_model import MultiModalEmotionClassifierDeep  # 你們的自定義模型

CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
TARGET_SR = 16000
MAX_AUDIO_LEN = 12 * TARGET_SR   # 12 秒上限
MIN_AUDIO_LEN = 3 * TARGET_SR    # 3 秒下限

class CrabEmotionPredictor:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self._load_models()

    def _load_models(self):
        """載入三個模型權重，順序：SSL → Text → SER Head"""

        # 1. WavLM-Large SSL 音訊編碼器
        self.ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        ssl_state = torch.load(f"{self.model_dir}/final_ssl.pt", map_location="cpu")
        self.ssl_model.load_state_dict(ssl_state)
        self.ssl_model.to(self.device).eval()

        # 2. RoBERTa-Large 文字編碼器 + tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.text_model = RobertaModel.from_pretrained("roberta-large")
        text_state = torch.load(f"{self.model_dir}/final_text.pt", map_location="cpu")
        self.text_model.load_state_dict(text_state)
        self.text_model.to(self.device).eval()

        # 3. 分類頭 (MultiModalEmotionClassifierDeep)
        self.ser_model = MultiModalEmotionClassifierDeep(
            audio_dim=1024,    # WavLM-Large hidden size
            text_dim=1024,     # RoBERTa-Large hidden size
            num_classes=len(CLASSES),
            fusion_hidden_dim=512,
            head_dim=1024
        )
        ser_state = torch.load(f"{self.model_dir}/final_ser.pt", map_location="cpu")
        self.ser_model.load_state_dict(ser_state)
        self.ser_model.to(self.device).eval()

    def _preprocess_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """bytes → normalized mono 16kHz Tensor [1, T]"""
        import io
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        waveform = waveform.mean(dim=0, keepdim=True)           # 轉 mono
        waveform = waveform[:, :MAX_AUDIO_LEN]                  # 截斷
        return waveform.float()

    def _preprocess_text(self, text: str, max_len: int = 128):
        """text → tokenized dict on device"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_len,
            padding="max_length",
            truncation=True
        )

    @torch.no_grad()
    def predict_single(self, audio_bytes: bytes, text: str) -> dict:
        """單筆推論，回傳機率 dict"""
        wav = self._preprocess_audio(audio_bytes).to(self.device)
        tok = {k: v.to(self.device) for k, v in self._preprocess_text(text).items()}

        audio_feat = self.ssl_model(wav).last_hidden_state       # [1, T', 1024]
        text_feat  = self.text_model(**tok).last_hidden_state     # [1, L, 1024]

        logits = self.ser_model(audio_feat, text_feat)            # [1, 3]
        probs  = F.softmax(logits, dim=1)[0].cpu().tolist()

        return dict(zip(CLASSES, probs))

    @torch.no_grad()
    def predict_batch(
        self,
        audio_bytes_list: list[bytes],
        texts: list[str]
    ) -> list[dict]:
        """
        GPU 平行批次推論
        將所有音訊 pad 到相同長度後組成單一 Tensor，一次 forward
        """
        wavs = [self._preprocess_audio(ab) for ab in audio_bytes_list]

        # Pad 到 batch 中最長的音訊
        max_len = max(w.shape[1] for w in wavs)
        padded  = torch.zeros(len(wavs), 1, max_len)
        for i, w in enumerate(wavs):
            padded[i, :, :w.shape[1]] = w
        padded = padded.squeeze(1).to(self.device)               # [B, T]

        # Batch tokenize
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}

        audio_feat = self.ssl_model(padded).last_hidden_state     # [B, T', 1024]
        text_feat  = self.text_model(**tok).last_hidden_state     # [B, L, 1024]

        logits = self.ser_model(audio_feat, text_feat)            # [B, 3]
        probs  = F.softmax(logits, dim=1).cpu().tolist()          # list[list[float]]

        return [dict(zip(CLASSES, p)) for p in probs]
```

### 3.2 關鍵實作注意事項

| 項目 | 說明 |
|---|---|
| `map_location="cpu"` | 避免權重載入時直接吃 GPU 記憶體，統一載後再 `.to(device)` |
| `@torch.no_grad()` | 推論不需梯度，節省約 30% 顯存 |
| Audio Padding | batch 推論前必須 pad 到相同長度，否則 `torch.stack` 報錯 |
| `squeeze(1)` | WavLM 輸入是 `[B, T]`，不是 `[B, 1, T]` |
| Text empty string | 若 LangGraph 沒有傳 text，用空字串 `""` 仍可 forward，但準確率會下降 |

---

## 4. Phase 3｜`api/app.py` — FastAPI 伺服器

```python
# Crab/api/app.py
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from .inference import CrabEmotionPredictor, CLASSES
from .schemas import (
    SingleClassifyResponse, BatchClassifyResponse,
    BatchResultItem, EmotionProbabilities, HealthResponse
)

logger = logging.getLogger("crab_api")
predictor: CrabEmotionPredictor = None

MODEL_DIR = "/home/brant/Project/SAILER_test/Crab/experiments/interview_scheme1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """啟動時載入模型，關閉時釋放"""
    global predictor
    logger.info("Loading Crab Bimodal models...")
    predictor = CrabEmotionPredictor(model_dir=MODEL_DIR, device="cuda")
    logger.info("Models loaded. API ready.")
    yield
    del predictor
    logger.info("Models unloaded.")

app = FastAPI(
    title="Crab Bimodal Emotion API",
    version="1.0",
    description="WavLM-Large + RoBERTa-Large → 3-class Interview Emotion Classifier",
    lifespan=lifespan
)

# ─────────────────────────────────────────────
# GET /v1/health
# ─────────────────────────────────────────────
@app.get("/v1/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model="crab-bimodal-scheme1",
        device=str(predictor.device),
        classes=CLASSES
    )

# ─────────────────────────────────────────────
# POST /v1/emotion/classify  (單筆)
# ─────────────────────────────────────────────
@app.post("/v1/emotion/classify", response_model=SingleClassifyResponse)
async def classify_single(
    audio: UploadFile = File(..., description="WAV/MP3, 16kHz mono, 3–15s"),
    text:  str        = Form(default="", description="逐字稿，若為空則給空字串")
):
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")

    t0 = time.perf_counter()
    probs = predictor.predict_single(audio_bytes, text)
    latency_ms = (time.perf_counter() - t0) * 1000

    primary_label = max(probs, key=probs.get)
    return SingleClassifyResponse(
        primary_label=primary_label,
        primary_confidence=round(probs[primary_label], 4),
        probabilities=EmotionProbabilities(**{k: round(v, 4) for k, v in probs.items()}),
        latency_ms=round(latency_ms, 2)
    )

# ─────────────────────────────────────────────
# POST /v1/emotion/classify-batch  (批次)
# ─────────────────────────────────────────────
@app.post("/v1/emotion/classify-batch", response_model=BatchClassifyResponse)
async def classify_batch(
    files: list[UploadFile] = File(..., description="多個音訊檔案"),
    texts: list[str]        = Form(default=[], description="對應每個音訊的逐字稿")
):
    if len(files) == 0:
        raise HTTPException(400, "No files provided")
    if len(texts) == 0:
        texts = [""] * len(files)
    if len(texts) != len(files):
        raise HTTPException(400, f"files ({len(files)}) and texts ({len(texts)}) count mismatch")

    MAX_BATCH = 32
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f"Batch size exceeds maximum ({MAX_BATCH})")

    audio_bytes_list = [await f.read() for f in files]

    t0 = time.perf_counter()
    batch_probs = predictor.predict_batch(audio_bytes_list, texts)
    total_ms = (time.perf_counter() - t0) * 1000

    results = []
    for f, probs in zip(files, batch_probs):
        primary_label = max(probs, key=probs.get)
        results.append(BatchResultItem(
            filename=f.filename,
            primary_label=primary_label,
            primary_confidence=round(probs[primary_label], 4),
            probabilities=EmotionProbabilities(**{k: round(v, 4) for k, v in probs.items()})
        ))

    return BatchClassifyResponse(
        batch_size=len(files),
        total_latency_ms=round(total_ms, 2),
        avg_latency_ms=round(total_ms / len(files), 2),
        results=results
    )
```

---

## 5. Phase 4｜`api/test_latency.py` — 延遲基準測試

```python
# Crab/api/test_latency.py
"""
執行方式：
    python -m Crab.api.test_latency --url http://localhost:8001 --wav path/to/test.wav
"""
import argparse
import time
import statistics
import requests

def run_single_test(url: str, wav_path: str, text: str, n: int = 10):
    print(f"\n{'='*60}")
    print(f"[Single] {n} consecutive requests")
    print(f"{'='*60}")
    latencies = []
    for i in range(n):
        with open(wav_path, "rb") as f:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{url}/v1/emotion/classify",
                files={"audio": ("test.wav", f, "audio/wav")},
                data={"text": text}
            )
            elapsed = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        result = resp.json()
        latencies.append(elapsed)
        print(f"  [{i+1:02d}] {result['primary_label']:16s} "
              f"conf={result['primary_confidence']:.3f}  "
              f"latency={elapsed:.1f}ms  "
              f"(server reports {result['latency_ms']:.1f}ms)")

    print(f"\n  ── Single Request Stats ──────────────────")
    print(f"  Mean   : {statistics.mean(latencies):.1f} ms")
    print(f"  Median : {statistics.median(latencies):.1f} ms")
    print(f"  P95    : {sorted(latencies)[int(n*0.95)]:.1f} ms")
    print(f"  Min    : {min(latencies):.1f} ms")
    print(f"  Max    : {max(latencies):.1f} ms")
    return latencies

def run_batch_test(url: str, wav_path: str, text: str, batch_size: int = 10):
    print(f"\n{'='*60}")
    print(f"[Batch] 1 request × {batch_size} files")
    print(f"{'='*60}")

    files = []
    data  = {}
    for i in range(batch_size):
        files.append(("files", (f"audio_{i}.wav", open(wav_path, "rb"), "audio/wav")))
    # texts 以 Form 傳遞（FastAPI list[str] from Form）
    for i in range(batch_size):
        data[f"texts"] = text   # 簡化：全部相同的 text

    t0 = time.perf_counter()
    resp = requests.post(f"{url}/v1/emotion/classify-batch", files=files, data=data)
    total_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    result = resp.json()

    print(f"  Batch size        : {result['batch_size']}")
    print(f"  Total latency     : {result['total_latency_ms']:.1f} ms  (client: {total_ms:.1f} ms)")
    print(f"  Avg per item      : {result['avg_latency_ms']:.1f} ms")
    print(f"\n  Results:")
    for r in result["results"]:
        print(f"    {r['filename']:20s}  {r['primary_label']:16s}  conf={r['primary_confidence']:.3f}")

    # 關閉檔案
    for _, (_, fobj, _) in files:
        fobj.close()

    return total_ms, result["avg_latency_ms"]

def print_comparison(single_latencies, batch_total_ms, batch_size):
    single_mean = statistics.mean(single_latencies)
    batch_avg   = batch_total_ms / batch_size
    speedup     = (single_mean * batch_size) / batch_total_ms

    print(f"\n{'='*60}")
    print(f"[Comparison] Single × {batch_size} vs Batch × {batch_size}")
    print(f"{'='*60}")
    print(f"  Single × {batch_size} (sequential) : {single_mean * batch_size:.1f} ms")
    print(f"  Batch  × {batch_size} (parallel)   : {batch_total_ms:.1f} ms")
    print(f"  Speedup                   : {speedup:.2f}x")
    print(f"  Throughput (single)       : {1000/single_mean:.1f} req/s")
    print(f"  Throughput (batch)        : {1000/batch_avg:.1f} req/s (amortized)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default="http://localhost:8001")
    parser.add_argument("--wav",  required=True, help="Path to a test WAV file (3–15s, 16kHz mono)")
    parser.add_argument("--text", default="I think I am a strong fit for this position.")
    parser.add_argument("--n",    type=int, default=10, help="Single test repetitions")
    parser.add_argument("--batch",type=int, default=10, help="Batch size for batch test")
    args = parser.parse_args()

    single_latencies = run_single_test(args.url, args.wav, args.text, n=args.n)
    batch_total, batch_avg = run_batch_test(args.url, args.wav, args.text, batch_size=args.batch)
    print_comparison(single_latencies, batch_total, args.batch)
```

---

## 6. Phase 5｜啟動與驗證流程

```bash
# Step 1: 啟動 API
cd /home/brant/Project/SAILER_test
uvicorn Crab.api.app:app --host 0.0.0.0 --port 8001 --workers 1

# Step 2: 健康檢查
curl http://localhost:8001/v1/health

# Step 3: 單筆手動測試
curl -X POST http://localhost:8001/v1/emotion/classify \
  -F "audio=@/path/to/test.wav" \
  -F "text=I believe my skills align well with this role"

# Step 4: 延遲基準測試
python -m Crab.api.test_latency \
  --wav /path/to/test.wav \
  --text "I believe my skills align well with this role" \
  --n 10 \
  --batch 10
```

### 延遲目標

| 情境 | 目標 | 備註 |
|---|---|---|
| 單筆 P50 | < 200 ms | 面試即時回饋可接受範圍 |
| 單筆 P95 | < 400 ms | 偶爾較長音訊 |
| Batch/10 total | < 600 ms | GPU 平行效益 > 2x |
| Batch avg/item | < 80 ms | 遠優於 10 × single |

---

## 7. Phase 6｜更新 `docs/api_spec.md` Section 3

取代原本 Whisper 的 Section 3，改為以下規格：

### 更新後的 Response Schema（與舊版相容欄位標注）

```json
// POST /v1/emotion/classify — 單筆 Response
{
  "primary_label": "Neutral_3Class",       // ← 與舊版同名，LangGraph Node 不需改
  "primary_confidence": 0.80,              // ← 與舊版同名
  "probabilities": {
    "Excited": 0.12,
    "Unconfident": 0.08,
    "Neutral_3Class": 0.80
  },
  "latency_ms": 145.2
}

// POST /v1/emotion/classify-batch — 批次 Response
{
  "batch_size": 3,
  "total_latency_ms": 210.5,
  "avg_latency_ms": 70.1,
  "results": [
    {
      "filename": "audio1.wav",
      "primary_label": "Excited",
      "primary_confidence": 0.72,
      "probabilities": { "Excited": 0.72, "Unconfident": 0.10, "Neutral_3Class": 0.18 }
    }
  ]
}
```

### LangGraph Emotion Node 需要的 mapping 調整

舊 SAILER 回傳 `arousal` / `valence` / `dominance`，新版 Crab 不輸出 VAD。  
LangGraph Feedback Node 的 state 需要調整：

```python
# Before (SAILER)
{
  "detected_emotion": result["primary_label"],     # e.g. "Neutral"
  "emotion_confidence": result["primary_confidence"],
  "emotion_arousal": result["arousal"],
  "emotion_valence": result["valence"],
}

# After (Crab)
{
  "detected_emotion": result["primary_label"],     # e.g. "Neutral_3Class"
  "emotion_confidence": result["primary_confidence"],
  "emotion_probabilities": result["probabilities"] # 新增，傳完整機率給 LLM prompt
  # arousal / valence 移除或改為 None
}
```

---

## 8. 潛在問題與解法

| 問題 | 原因 | 解法 |
|---|---|---|
| WavLM forward 對不同長度 batch 報錯 | attention mask 沒有傳 | `ssl_model(padded, attention_mask=mask)` |
| 首次請求特別慢（> 1s）| CUDA kernel warm-up | lifespan 啟動後跑一次 dummy forward |
| 多個請求同時進來搶 GPU | FastAPI 單 worker 但 async | 加 `asyncio.Lock()` 確保 GPU 序列存取 |
| text 欄位傳 Form list 語法 | FastAPI batch form 需特殊處理 | 改用 JSON body 或 multipart 分段 |
| 音訊長度 < 3s | 面試中短暫停頓可能觸發 | 前端做 VAD buffering，攢夠 3s 再送 |

### Dummy Warm-up（加在 lifespan 內）

```python
import torch
import numpy as np

# 啟動後跑一次 dummy forward，讓 CUDA kernel 預熱
dummy_wav = np.zeros(3 * 16000, dtype=np.float32).tobytes()
predictor.predict_single(dummy_wav, "warm up")
logger.info("GPU warm-up complete.")
```

---

## 9. 執行順序 Checklist

- [ ] **Phase 1** 建立 `schemas.py`
- [ ] **Phase 2** 建立 `inference.py`，確認三個 `.pt` 路徑正確，跑 unit test
- [ ] **Phase 3** 建立 `app.py`，`uvicorn` 啟動確認健康檢查通過
- [ ] **Phase 4** 建立 `test_latency.py`，跑完取得 P50/P95/Speedup 數據
- [ ] **Phase 5** 驗證 batch 結果與 single 結果一致（同音訊同文字應得相同 label）
- [ ] **Phase 6** 更新 `docs/api_spec.md` Section 3，移除 Whisper 規格
- [ ] **Phase 7** 更新 LangGraph Emotion Node 移除 arousal/valence 依賴

---

*此計畫書對應 `docs/api_spec.md` v1.0 Section 3 替換。*
