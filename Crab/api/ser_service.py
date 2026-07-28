"""SER 微服務 —— 把 Okeke 4 類語音情緒模型包成 HTTP API。

跑在 Crab/.venv(torch 2.2+cu121、peft、transformers);「AI 奧客應對演練」遊戲後端
用 HTTP 呼叫,這樣遊戲 server 完全不必裝 CRAB 依賴(避免 torch 版本衝突),GPU 進程也隔離。

啟動:
  cd /home/brant/Project/SAILER_test/Crab
  source .venv/bin/activate
  uvicorn api.ser_service:app --host 127.0.0.1 --port 8100
  # 或:python -m uvicorn api.ser_service:app --host 127.0.0.1 --port 8100

端點:
  GET  /health            → {ok, classes, device}
  POST /predict           → multipart:audio=<檔案>, text=<逐字稿(選填)>
                            回 {label, confidence, probs:{Angry,Happy,Neutral,Anxious}}
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# 讓 `from api.okeke_infer import ...` 找得到(本檔在 Crab/api/ 下)
_CRAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CRAB_ROOT not in sys.path:
    sys.path.insert(0, _CRAB_ROOT)
from api.okeke_infer import CLASSES, OkekeSER  # noqa: E402

app = FastAPI(title="Okeke SER 微服務")
_predictor: OkekeSER | None = None


def _get() -> OkekeSER:
    global _predictor
    if _predictor is None:
        _predictor = OkekeSER()   # 第一次:載 XLS-R + XLM-R + LoRA + SER 頭到 GPU(~5s)
    return _predictor


@app.on_event("startup")
def _startup() -> None:
    _get()   # 啟動就載模型,第一次 /predict 不卡


@app.get("/health")
def health() -> dict:
    return {"ok": True, "classes": CLASSES, "device": str(_get().device)}


@app.post("/predict")
async def predict(audio: UploadFile = File(...), text: str = Form("")) -> dict:
    """收一段音檔(+ 可選逐字稿)→ 回 4 類情緒機率。音檔交給 librosa 讀(wav 最穩,給 CLI/curl 測)。"""
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return _get().predict(tmp.name, text or "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"infer failed: {type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


class PcmReq(BaseModel):
    pcm_b64: str                 # int16 little-endian 單聲道 PCM 的 base64
    sr: int = 16000
    text: str = ""
    words: list | None = None    # [{start,end,word}, ...](有給且語句長 → 依詞級時間戳切段加權)


@app.post("/predict_pcm")
def predict_pcm(req: PcmReq) -> dict:
    """給遊戲端用:JSON + base64 PCM(免上傳檔案、免解碼 webm)。遊戲已用 faster_whisper 解碼成 16k。"""
    try:
        pcm = np.frombuffer(base64.b64decode(req.pcm_b64), dtype="<i2").astype(np.float32) / 32768.0
        if req.sr != 16000:
            import librosa
            pcm = librosa.resample(pcm, orig_sr=req.sr, target_sr=16000)
        if req.words:
            return _get().predict_chunked(pcm, req.words, req.text or "")
        return _get().predict_array(pcm, req.text or "")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"infer failed: {type(e).__name__}: {e}")
