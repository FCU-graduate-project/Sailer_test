# Crab Bimodal Emotion API

基於 FastAPI 構建的高性能面試情感辨識 REST API。本模組將訓練完成的 Crab 雙模態模型（WavLM-Large + RoBERTa-Large）封裝為生產環境可用的服務，專為 LangGraph 面試模擬器設計。

**Version:** 1.1
**Date:** 2026-05-15
**Base URL:** `http://<SERVER_IP>:8001`
**Interactive Docs:** `http://<SERVER_IP>:8001/docs` (Swagger UI)

---

## Table of Contents

1. [Core Features](#-core-features)
2. [Directory Structure](#-directory-structure)
3. [Quick Start](#-quick-start)
4. [Model Overview](#-model-overview)
5. [API Endpoints](#-api-endpoints)
   - [GET /v1/health](#1-get-v1health--health-check)
   - [POST /v1/emotion/classify](#2-post-v1emotionclassify--single-turn-classification)
   - [POST /v1/emotion/classify-batch](#3-post-v1emotionclassify-batch--batch-classification)
   - [POST /v1/emotion/classify-long](#4-post-v1emotionclassify-long--long-audio-timeline)
6. [Error Responses](#-error-responses)
7. [Sentence-Level Integration Guide](#-sentence-level-integration-guide-recommended)
8. [Long-Sentence Splitting Algorithm (>12s)](#-long-sentence-splitting-algorithm-12s)
9. [Best Practices](#-best-practices)
10. [Performance Benchmarks](#-performance-benchmarks-rtx-3090)
11. [Interactive Docs](#-interactive-docs)

---

## 🌟 Core Features

- **雙模態融合 (Bimodal)**：結合語音聲學特徵 (WavLM) 與語意文字特徵 (RoBERTa)，比單純聽聲音更懂面試者的情緒。
- **極速平行批次 (Batch Processing)**：支援一次上傳多個音檔與文字，利用 GPU 進行平行矩陣運算，吞吐量提升 3 倍以上。
- **零啟動延遲 (Warm-up)**：服務啟動時自動進行 GPU 預熱，消除了 PyTorch 首次推論卡頓的問題。
- **無縫相容性**：輸出格式保留了與舊版 SAILER 相同的欄位名稱，降低系統整合成本。
- **長句智能切分**：針對訓練上限（12 秒）以上的句子，採用三層優先序演算法，確保 audio-text 嚴格對齊。

---

## 📁 Directory Structure

```text
Crab/api/
├── app.py                     # FastAPI 應用程式與路由
├── inference.py               # 核心推論引擎（模型載入、前處理、Mask處理）
├── schemas.py                 # Pydantic 資料模型與類型檢查
├── requirements.txt           # API 專屬依賴套件
│
├── test_latency.py            # 延遲與性能基準測試腳本
├── test_accuracy.py           # 模型準確率評估
├── test_long.py               # 長音檔 sliding window 測試
├── test_long_text.py          # 長音檔 + 文字對齊測試
├── test_sentence.py           # ⭐ 句子級切分 + Crab API（推薦使用）
├── test_sentence_openai.py    # OpenRouter Whisper backend 版本
├── generate_demo_audio.py     # 測試用合成音檔產生器
└── README.md                  # 本文件
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd /home/brant/Project/SAILER_test
Crab/.venv/bin/python -m pip install -r Crab/api/requirements.txt
```

### 2. Start the server

```bash
cd /home/brant/Project/SAILER_test/Crab
.venv/bin/python -m uvicorn api.app:app --host 0.0.0.0 --port 8001 --workers 1
```

> 模型路徑由環境變數 `CRAB_MODEL_DIR` 控制（預設 `Crab/experiments/interview_scheme1`）。部署到不同機器時 `export CRAB_MODEL_DIR=/your/path` 即可。

### 3. Sentence-level analysis (recommended)

```bash
cd /home/brant/Project/SAILER_test

# Local Whisper (default, requires GPU)
Crab/.venv/bin/python -m Crab.api.test_sentence --wav /path/to/interview.wav

# OpenRouter Whisper (no local GPU needed)
export OPENROUTER_API_KEY=sk-or-v1-...
Crab/.venv/bin/python -m Crab.api.test_sentence --backend openrouter --wav /path/to/interview.wav
```

### 4. Latency benchmark

```bash
Crab/.venv/bin/python -m Crab.api.test_latency --wav /path/to/test.wav
```

---

## 🧠 Model Overview

| Attribute | Value |
|---|---|
| Architecture | WavLM-Large (Audio) + RoBERTa-Large (Text) + Cross-Attention Fusion |
| Input | Audio (WAV/MP3/FLAC, auto-resampled to 16kHz mono) + Transcript Text (optional) |
| Output | 3-Class Interview Emotion probabilities + argmax label |
| Training Data | MSP-Podcast → Custom mapped 3-Class Interview Scheme |
| Audio Max Length | **12 seconds** (192000 samples @ 16kHz) — hard limit from training collate_fn |
| Text Max Length | **128 tokens** (RoBERTa tokenizer) |

### Emotion Labels (3-Class)

| Index | Label | Description |
|---|---|---|
| 0 | `Excited` | Positive / Confident / Enthusiastic |
| 1 | `Unconfident` | Anxious / Hesitant / Nervous |
| 2 | `Neutral_3Class` | Calm / Professional / Neutral |

---

## 🔌 API Endpoints

### 1. `GET /v1/health` — Health Check

Check if the API server is alive and which model is loaded.

**Response** `200 OK`

```json
{
  "status": "ok",
  "model": "interview_scheme1",
  "device": "cuda",
  "classes": ["Excited", "Unconfident", "Neutral_3Class"],
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "vram_mb": 24576
}
```

| Field | Type | Description |
|---|---|---|
| `status` | `string` | `"ok"` when service is ready |
| `model` | `string` | Loaded model identifier |
| `device` | `string` | `"cuda"` or `"cpu"` |
| `classes` | `string[]` | Ordered list of output class names |
| `gpu_name` | `string \| null` | GPU model name (null if CPU-only) |
| `vram_mb` | `int \| null` | Total GPU VRAM in MB (null if CPU-only) |

---

### 2. `POST /v1/emotion/classify` — Single-Turn Classification

Best for **real-time turn-by-turn** interview simulation. Classify one audio segment.

#### Request (`multipart/form-data`)

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | `file` | **Yes** | Audio file (WAV/MP3/FLAC, any sample rate; 3-12s recommended) |
| `text` | `string` | No (default: `""`) | Transcript of the audio. **Highly recommended** — enables RoBERTa semantic understanding for significantly better accuracy. |

#### Response `200 OK`

```json
{
  "primary_label": "Excited",
  "primary_confidence": 0.954,
  "probabilities": {
    "Excited": 0.954,
    "Unconfident": 0.016,
    "Neutral_3Class": 0.030
  },
  "latency_ms": 45.2
}
```

| Field | Type | Description |
|---|---|---|
| `primary_label` | `string` | Argmax of softmax (one of the 3 classes) |
| `primary_confidence` | `float` | Softmax probability of `primary_label` (0.0 ~ 1.0) |
| `probabilities` | `object` | Full 3-class softmax distribution |
| `latency_ms` | `float` | Server-side GPU inference time in milliseconds |

#### Example (Python)

```python
import requests

url = "http://<SERVER_IP>:8001/v1/emotion/classify"

with open("answer.wav", "rb") as f:
    response = requests.post(
        url,
        files={"audio": ("answer.wav", f, "audio/wav")},
        data={"text": "I am very excited about this opportunity."},
    )

result = response.json()
print(result["primary_label"], result["primary_confidence"])
# "Excited" 0.954
```

#### Example (cURL)

```bash
curl -X POST "http://<SERVER_IP>:8001/v1/emotion/classify" \
  -F "audio=@answer.wav;type=audio/wav" \
  -F "text=I am very excited about this opportunity."
```

---

### 3. `POST /v1/emotion/classify-batch` — Batch Classification

Classify **multiple audio clips** in a single GPU-parallel forward pass. Maximum batch size: **16**.

#### Request (`multipart/form-data`)

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | `file[]` | **Yes** | Multiple audio files (max 16) |
| `texts` | `string[]` | No (default: all `""`) | Transcripts for each file. If provided, length **must match** `files`. |

#### Response `200 OK`

```json
{
  "batch_size": 3,
  "total_latency_ms": 58.4,
  "avg_latency_ms": 19.5,
  "results": [
    {
      "filename": "answer1.wav",
      "primary_label": "Excited",
      "primary_confidence": 0.891,
      "probabilities": {
        "Excited": 0.891,
        "Unconfident": 0.032,
        "Neutral_3Class": 0.077
      }
    },
    {
      "filename": "answer2.wav",
      "primary_label": "Unconfident",
      "primary_confidence": 0.643,
      "probabilities": {
        "Excited": 0.120,
        "Unconfident": 0.643,
        "Neutral_3Class": 0.237
      }
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `batch_size` | `int` | Number of audio clips processed |
| `total_latency_ms` | `float` | Total GPU inference time for the entire batch |
| `avg_latency_ms` | `float` | Average per-clip inference time |
| `results` | `array` | Per-clip results (see fields below) |
| `results[].filename` | `string` | Original filename |
| `results[].primary_label` | `string` | Predicted emotion label |
| `results[].primary_confidence` | `float` | Confidence of prediction |
| `results[].probabilities` | `object` | Full 3-class softmax distribution |

#### Example (Python)

```python
import requests

url = "http://<SERVER_IP>:8001/v1/emotion/classify-batch"

files = [
    ("files", ("seg1.wav", open("seg1.wav", "rb"), "audio/wav")),
    ("files", ("seg2.wav", open("seg2.wav", "rb"), "audio/wav")),
]
data = [
    ("texts", "I think I can do this job well."),
    ("texts", "I'm not sure if I'm qualified."),
]

response = requests.post(url, files=files, data=data)
for item in response.json()["results"]:
    print(f"{item['filename']}: {item['primary_label']} ({item['primary_confidence']:.2f})")
```

---

### 4. `POST /v1/emotion/classify-long` — Long Audio Timeline

Analyze a long audio recording using **sliding windows**. Returns an emotion timeline showing how emotions change over time.

#### Request (`multipart/form-data`)

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `audio` | `file` | **Yes** | WAV/MP3/FLAC, any length | Full interview audio |
| `text` | `string` | No (default: `""`) | — | Full transcript (shared across all windows) |
| `window_sec` | `float` | No (default: `12.0`) | 1.0 ~ 12.0 | Sliding window size in seconds |
| `stride_sec` | `float` | No (default: `6.0`) | 0.5 ~ `window_sec` | Stride in seconds (overlap = window − stride) |

#### Response `200 OK`

```json
{
  "final_label": "Excited",
  "final_confidence": 0.565,
  "avg_probabilities": {
    "Excited": 0.565,
    "Unconfident": 0.036,
    "Neutral_3Class": 0.399
  },
  "total_windows": 20,
  "audio_duration_sec": 120.5,
  "timeline": [
    {
      "window": "0.0~12.0s",
      "label": "Neutral_3Class",
      "confidence": 0.543,
      "probabilities": {
        "Excited": 0.367,
        "Unconfident": 0.090,
        "Neutral_3Class": 0.543
      }
    },
    {
      "window": "6.0~18.0s",
      "label": "Excited",
      "confidence": 0.687,
      "probabilities": {
        "Excited": 0.687,
        "Unconfident": 0.014,
        "Neutral_3Class": 0.298
      }
    }
  ],
  "latency_ms": 1063.3
}
```

| Field | Type | Description |
|---|---|---|
| `final_label` | `string` | Overall emotion via probability averaging across all windows |
| `final_confidence` | `float` | Confidence of `final_label` |
| `avg_probabilities` | `object` | Averaged 3-class distribution across all windows |
| `total_windows` | `int` | Number of sliding windows generated |
| `audio_duration_sec` | `float` | Total audio length in seconds |
| `timeline` | `array` | Per-window predictions ordered by time |
| `timeline[].window` | `string` | Time range, e.g. `"0.0~12.0s"` |
| `timeline[].label` | `string` | Predicted emotion for this window |
| `timeline[].confidence` | `float` | Confidence of this window's prediction |
| `timeline[].probabilities` | `object` | Full 3-class distribution for this window |
| `latency_ms` | `float` | Server-side GPU inference time in milliseconds |

---

## ⚠️ Error Responses

All endpoints return standard HTTP error codes with a JSON detail message.

| Status | Condition |
|---|---|
| `400 Bad Request` | Empty audio file, invalid `window_sec`/`stride_sec` range, batch size exceeds 16, or files/texts count mismatch |
| `422 Unprocessable Entity` | Invalid or corrupt audio format |
| `500 Internal Server Error` | GPU inference failure |

**Error Response Body:**

```json
{
  "detail": "Batch size 20 exceeds maximum (16)"
}
```

---

## 📚 Sentence-Level Integration Guide (Recommended)

For the **highest accuracy**, we recommend splitting audio by sentence boundaries before calling the Crab API. This ensures RoBERTa receives semantically complete sentences and the cross-modal attention can correctly align audio frames to words.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Client (LangGraph)                      │
│                                                          │
│  1. Record user audio                                    │
│  2. Whisper (word_timestamps=True) → transcription       │
│  3. Group words into sentences by punctuation (. ? !)    │
│  4. Merge short sentences (< 3s) with neighbors          │
│  5. Split long sentences (> 12s) — see Plan B section    │
│  6. Slice audio by sentence timestamps                   │
│  7. POST each segment to Crab API                        │
└────────────────────┬─────────────────────────────────────┘
                     │  HTTP POST (audio + text)
                     ▼
┌──────────────────────────────────────────────────────────┐
│              Crab API (Server, port 8001)                │
│                                                          │
│  /v1/emotion/classify       → single sentence            │
│  /v1/emotion/classify-batch → multiple sentences         │
│                                                          │
│  Returns: primary_label, primary_confidence,             │
│           probabilities (3-class)                        │
└──────────────────────────────────────────────────────────┘
```

### Step-by-Step Python Example

```python
import io
import requests
import torchaudio
from faster_whisper import WhisperModel

CRAB_API = "http://<SERVER_IP>:8001"

# ── Step 1: Transcribe with word-level timestamps ──────────────
model = WhisperModel("large-v3", device="cuda")
segments, info = model.transcribe(
    "interview_answer.wav",
    word_timestamps=True,   # ← THIS IS THE KEY SETTING
    language=None,           # auto-detect
)

# Flatten all words with their timestamps
words = []
for segment in segments:
    for word in segment.words:
        words.append({
            "word": word.word.strip(),
            "start": word.start,
            "end": word.end,
        })

# ── Step 2: Group words into sentences by punctuation ──────────
SENTENCE_ENDERS = {".", "。", "?", "？", "!", "！", "…"}

sentences = []
current_words = []
for w in words:
    current_words.append(w)
    if any(w["word"].endswith(p) for p in SENTENCE_ENDERS):
        sentences.append({
            "text": " ".join(cw["word"] for cw in current_words),
            "start": current_words[0]["start"],
            "end": current_words[-1]["end"],
            "words": current_words[:],   # keep for >12s splitting
        })
        current_words = []
if current_words:
    sentences.append({
        "text": " ".join(cw["word"] for cw in current_words),
        "start": current_words[0]["start"],
        "end": current_words[-1]["end"],
        "words": current_words[:],
    })

# ── Step 3: Merge short sentences (< 3s) with neighbors ───────
MIN_DURATION = 3.0
merged = [sentences[0]]
for s in sentences[1:]:
    prev = merged[-1]
    if (prev["end"] - prev["start"]) < MIN_DURATION:
        merged[-1] = {
            "text": prev["text"] + " " + s["text"],
            "start": prev["start"],
            "end": s["end"],
            "words": prev["words"] + s["words"],
        }
    else:
        merged.append(s)
sentences = merged

# ── Step 4: Split long sentences (> 12s) — see next section ────
# (See Long-Sentence Splitting Algorithm)

# ── Step 5: Slice audio & call Crab API per sentence ───────────
waveform, sr = torchaudio.load("interview_answer.wav")

for sent in sentences:
    start_sample = int(sent["start"] * sr)
    end_sample = int(sent["end"] * sr)
    segment = waveform[:, start_sample:end_sample]

    buf = io.BytesIO()
    torchaudio.save(buf, segment, sr, format="wav")
    audio_bytes = buf.getvalue()

    response = requests.post(
        f"{CRAB_API}/v1/emotion/classify",
        files={"audio": ("segment.wav", audio_bytes, "audio/wav")},
        data={"text": sent["text"]},
    )
    result = response.json()
    print(f"[{sent['start']:.1f}s ~ {sent['end']:.1f}s] {result['primary_label']} ({result['primary_confidence']:.2f})")
```

> **完整可執行範例**：見 [`test_sentence.py`](test_sentence.py)，內建轉錄、切句、合併、長句切分、批次推論、漂亮顯示。

### Key Requirements

| Requirement | Detail |
|---|---|
| `word_timestamps=True` | **Must** be enabled in Whisper. Without this, you cannot know where each word falls in the audio and sentence-level slicing is impossible. |
| Minimum segment duration | Recommended **≥ 3 seconds** per segment. Shorter segments may cause less reliable predictions. |
| Maximum segment duration | **≤ 12 seconds** — Crab 模型訓練上限。超過會被靜默截斷。長句切分演算法見下節。 |
| Batch limit | If using `classify-batch`, max 16 segments per request. Split into multiple requests if needed. |

---

## 🧩 Long-Sentence Splitting Algorithm (>12s)

### Why It Matters

Crab 模型在訓練時 collate function 將音檔 **truncate 至 12 秒**（192000 samples @ 16kHz）。若直接傳入更長音檔，會被**靜默截斷**，超過部分的資訊完全丟失：

```python
# Crab/api/inference.py
MAX_AUDIO_SEC = 12
MAX_AUDIO_LEN = MAX_AUDIO_SEC * TARGET_SR   # 192000

# When inference receives > 12s audio:
waveform = waveform[:MAX_AUDIO_LEN]   # Silently dropped beyond 12s ❌
```

文字模態 RoBERTa 同樣有 `TEXT_MAX_LEN = 128 tokens` 的硬上限。

因此，client 端必須在送進 Crab API 之前**將 >12s 的句子切成多個 ≤12s 的 chunk**，且每個 chunk 的音訊與文字必須嚴格對齊（否則 cross-modal attention 失效）。

### Algorithm: Three-Tier Priority Greedy Split

當一句經過合併後仍 > `max_duration_sec`（預設 12s）時，採用以下優先序貪婪切分：

| Priority | 切點類型 | 規則 | 用途 |
|----------|---------|------|------|
| **1** | **Hard break**（句末標點 `. ? ! 。 ？ ！ …`）| 視窗內存在即無條件 honor | 最自然的語意邊界 |
| **2** | **Soft break**（子句標點 `, ; : 、 ， ； ：`）| chunk 必須 ≥ 25% of max | 避免列表式逗號的碎片化 |
| **3** | **Balanced equal-time word-aligned** | 找詞末時間最接近 `remaining/n_remaining` 的詞 | 完全無標點時的平衡 fallback |

### Decision Flowchart

```
┌────────────────────────────────────────────────┐
│  輸入：sentence with word-level timestamps     │
└────────────────────┬───────────────────────────┘
                     ▼
            ┌─────────────────┐
            │ duration > max? │
            └────────┬────────┘
                NO   │   YES
        ┌────────────┴───────────┐
        ▼                        ▼
    passthrough          ┌───────────────────┐
                         │  start_idx = 0    │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌──────────────────────────┐
        ┌───────────────►│  while start_idx < N:    │
        │                └──────────┬───────────────┘
        │                           ▼
        │           ┌─────────────────────────────────────┐
        │           │  end_idx = last word fitting ≤ max  │
        │           └──────────────────┬──────────────────┘
        │                              ▼
        │                     ┌────────────────┐
        │                     │ all rest fit?  │
        │                     └─────┬──────────┘
        │                       NO  │  YES
        │                ┌──────────┴────────┐
        │                ▼                   ▼
        │      ┌───────────────────┐    emit final
        │      │ Priority 1:       │    chunk → DONE
        │      │ hard break in     │
        │      │ [start, end]?     │
        │      └───────┬───────────┘
        │       found  │  not found
        │      ┌───────┴────────┐
        │      ▼                ▼
        │   use it       ┌───────────────────┐
        │      │         │ Priority 2:       │
        │      │         │ soft break        │
        │      │         │ AND ≥ 25% of max? │
        │      │         └───────┬───────────┘
        │      │          found  │  not found
        │      │         ┌───────┴────────┐
        │      │         ▼                ▼
        │      │      use it       ┌────────────────────────┐
        │      │         │         │ Priority 3:            │
        │      │         │         │ balanced equal-time    │
        │      │         │         │ target = rest / n_rem  │
        │      │         │         └──────────┬─────────────┘
        │      │         │                    ▼
        │      │         │              use closest word
        │      │         │                    │
        │      └─────────┴────────────────────┘
        │                ▼
        │      ┌────────────────────────────┐
        │      │ emit chunk[start..split]   │
        │      │ start_idx = split + 1      │
        │      └────────────┬───────────────┘
        │                   │
        └───────────────────┘  (loop until start_idx >= N)
```

### Pseudo-code

```text
function split_long_sentence(words, max_duration):
    chunks = []
    start = 0
    while start < len(words):
        chunk_start_time = words[start].start

        # 1. Find furthest word fitting within max_duration
        end = start
        while end < len(words) and words[end].end - chunk_start_time <= max_duration:
            end += 1
        end -= 1

        # 2. If all remaining fit, emit and stop
        if end == len(words) - 1:
            emit chunk(words[start..end])
            break

        # 3. Priority 1: latest hard break in [start, end] — always honored
        split = find_latest(words[start..end], ending with . ? ! 。 ？ ！ …)

        # 4. Priority 2: latest soft break, requiring chunk ≥ 25% of max
        if split is None:
            split = find_latest(words[start..end], ending with , ; : 、 ， ； ：)
                    AND chunk_duration ≥ 0.25 * max_duration

        # 5. Priority 3: balanced equal-time word-aligned fallback
        if split is None:
            remaining = words[-1].end - chunk_start_time
            n_remaining = ceil(remaining / max_duration)
            target = chunk_start_time + remaining / n_remaining
            split = argmin_i(|words[i].end - target|) for i in [start..end]

        emit chunk(words[start..split])
        start = split + 1

    return chunks
```

**Iterative property**：迴圈每跑一次只 emit 一個 chunk。剩餘部分（即使本身又 > 12s）會在下一次迭代中再次套用整個優先序判斷，所以演算法可以處理任意長度的句子。

### Case Studies

實驗設定：300 秒 MSP-Podcast 合成音檔、faster-whisper large-v3、`max_duration_sec = 12.0`，找到 3 句 > 12s。

#### Case 1 — Hard break split（最理想）

**原句**（13.7s，103.0~116.7s）：
> "She might have been hired as an actress without an agenda. But you might think of the case where you're really torn between, you know, she would always be like, if you don't come for Christmas pictures and make it look like we're a happy family, I'm going to call the police on you."

| Version | chunk | Time (Dur) | Text |
|---------|-------|------------|------|
| Legacy（等時間 + `[cont.]`）| A | 103.0~109.85s (6.85s) | 整句完整文字 |
| Legacy | B | 109.85~116.7s (6.85s) | **`[cont.]`** + 整句完整文字 |
| **New (Plan B)** | A | 103.0~**105.5s (2.5s)** | "She might have been hired as an actress without an agenda." |
| **New (Plan B)** | B | **106.2s**~116.7s **(10.5s)** | "But you might think of the case where you're really torn between... I'm going to call the police on you." |

切點落在 `agenda.` 之後，兩段都是語意完整的獨立句。

#### Case 2 — Soft break split（句末標點剛好出界）

**原句**（12.1s，208.5~220.6s）：
> "Came through as, as if she were describing a murder or some other terrible act that had already had my first ever concert in my life was on the 10th of March, 2012 in Sydney."

| Version | chunk | Time (Dur) | Text |
|---------|-------|------------|------|
| Legacy | A | 208.5~214.55s (6.05s) | 整句完整文字 |
| Legacy | B | 214.55~220.6s (6.05s) | **`[cont.]`** + 整句完整文字 |
| **New (Plan B)** | A | 208.5~**219.0s (10.5s)** | "Came through as, as if she were describing... was on the 10th of March," |
| **New (Plan B)** | B | **219.2s**~220.6s **(1.4s)** | "2012 in Sydney." |

句末標點 `Sydney.` 落在 220.6s，**超出 12s 視窗 0.1 秒**（視窗上限為 220.5s），所以視窗內僅 `March,` 可用，演算法降級到 soft break。

#### Case 3 — Balanced equal-time fallback（轉錄無內部標點）

**原句**（13.0s，163.4~176.4s）：
> "But the reason that animals are being factory farmed and tortured and murdered like somebody else did come to the back of that was attacked on site because Alfred's in a tank and then cyborgs like I got this and there's from another that then went and went out and went dancing."

| Version | chunk | Time (Dur) | Text | Balance |
|---------|-------|------------|------|---------|
| Legacy | A | 163.4~169.9s (6.5s) | 整句完整文字 | 50% / 50% |
| Legacy | B | 169.9~176.4s (6.5s) | **`[cont.]`** + 整句完整文字 | (文字錯位)|
| Pure word boundary | A | 163.4~175.3s **(11.9s)** | "...that then went and went out and" | **91% / 9%** ⚠️ |
| Pure word boundary | B | 175.3~176.4s **(1.1s)** | "went dancing." | (退化) |
| **New (Plan B balanced)** | A | 163.4~**170.1s (6.7s)** | "But the reason... because Alfred's" | **51% / 49%** ✅ |
| **New (Plan B balanced)** | B | **170.1s**~176.4s **(6.3s)** | "in a tank and then cyborgs... went dancing." | (平衡) |

12 秒視窗內沒有任何句點或逗號（轉錄連貫無內部標點），且唯一的句末標點 `dancing.` 在視窗外。Balanced fallback 計算 `target = remaining / n_remaining` 並切在詞末時間最接近 target 的詞，避免「11.9s + 1.1s」這種降低 SNR 的退化結果。

### Why It Matters: Confidence Impact (Case 1 deep dive)

舊版（複製文字 + `[cont.]`）vs 新版（Plan B）對 Crab 預測信心的影響：

| chunk | Version | Time | Audio content | Text content | Excited | Unconfident | Neutral | Verdict |
|-------|---------|------|---------------|--------------|---------|-------------|---------|---------|
| A (前半) | Legacy | 103.0~109.8s | 平鋪敘述 | 整句（含後半戲劇張力）| **0.696** | 0.071 | 0.232 | 🔥 Excited |
| A (前半) | **Plan B** | 103.0~105.5s | 平鋪敘述 | "...without an agenda." | 0.146 | 0.097 | **0.756** | 😐 **Neutral** ✅ |
| B (後半) | Legacy | 109.8~116.7s | 戲劇張力強 | 整句（含前半平鋪內容）| 0.314 | 0.209 | **0.476** | 😐 Neutral |
| B (後半) | **Plan B** | 106.2~116.7s | 戲劇張力強 | "But you might think...police on you." | **0.464** | 0.241 | 0.293 | 🔥 **Excited** ✅ |

**Observation**：舊版的兩塊情緒判讀**剛好顛倒** — chunk A 拿到平鋪音訊但配對戲劇語料 → 誤報 Excited；chunk B 拿到戲劇音訊但被平鋪文字稀釋 → 誤判 Neutral。Plan B 修正 audio-text alignment 後，**兩塊的預測都與音訊真實內容一致**。

### Implementation

| 元件 | 位置 |
|------|------|
| 演算法主體 | [`test_sentence.py:_split_long_words`](test_sentence.py) |
| 標點搜尋輔助 | [`test_sentence.py:_find_punct_break`](test_sentence.py) |
| 整合到 segmentation pipeline | [`test_sentence.py:merge_and_split_sentences`](test_sentence.py) |
| 標點常數集合 | [`test_sentence.py:SENTENCE_ENDERS, SOFT_BREAK_CHARS`](test_sentence.py) |

### Fallback for No Word-Level Timestamps

當 word-level timestamps **不可得**（例如 OpenRouter API 沒回 word timestamps），降級為**等時間切分 + 整句文字複製 + `[cont.]` 前綴標記**。此時 audio-text 對齊性下降，但無更好選擇。建議優先使用支援 word-level 的本機 Whisper。

---

## 💡 Best Practices

1. **Always send `text`**：Providing the transcript alongside audio activates the RoBERTa text encoder, dramatically improving accuracy — especially for detecting `Unconfident` emotions from semantic cues (e.g., "I'm not sure", "I don't think I can").

2. **Use `classify` for real-time**：During a live interview, call `/v1/emotion/classify` after each user turn. This is a single HTTP POST per turn (~45ms latency).

3. **Use `classify-batch` for sentence-level analysis**：If you have Whisper word-level timestamps, slice audio by sentence boundaries on the client side, then send all segments in one batch request for maximum accuracy and throughput. See [`test_sentence.py`](test_sentence.py) for the complete reference implementation.

4. **Use `classify-long` for quick overview**：If you just need a rough emotion timeline without client-side preprocessing, this endpoint handles everything server-side via sliding windows.

5. **Health check before first call**：Call `GET /v1/health` on startup to verify the server is ready and GPU is available.

6. **Respect the 12s limit**：If you have sentences > 12s, use the [Long-Sentence Splitting Algorithm](#-long-sentence-splitting-algorithm-12s) to split them word-aligned. Never send >12s audio directly — it will be silently truncated.

7. **Use `--backend openrouter` for GPU-less deployment**：[`test_sentence.py`](test_sentence.py) supports OpenRouter Whisper as a transcription backend, useful when the client machine has no GPU. Set `OPENROUTER_API_KEY` env var.

---

## 📊 Performance Benchmarks (RTX 3090)

| 測試項目 | 實測數據 |
|---|---|
| 單筆推論平均延遲 (Latency) | **~42.5 ms** |
| 批次處理 (10筆) 總耗時 | **~133.4 ms** |
| 批次處理每筆平均耗時 | **~12.7 ms** |
| **批次平行加速比 (Speedup)** | **3.19x** 🚀 |
| **系統最高吞吐量 (Throughput)** | **~75 req/s** |

---

## 📝 Interactive Docs

服務啟動後，可直接訪問以下網址查看完整 API 參數並進行線上測試：

- Swagger UI: `http://<SERVER_IP>:8001/docs`
- ReDoc: `http://<SERVER_IP>:8001/redoc`
