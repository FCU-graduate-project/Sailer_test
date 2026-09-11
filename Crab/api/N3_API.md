# N3 中文情緒辨識 API

**Current version:** 1.3.0

**Model:** `n3-v1-warmstart-nnime-lora-stage2`

**Deployment state:** Local CUDA／HTTP qualification and A4500 local integration
passed. Production-branch integration and reboot recovery remain pending.

**用途:** 畢業專題／學術用途。

## 已完成

- `mode.json`、SER head、normalization stats、兩個 LoRA adapters 與
  SHA-256 manifest 的 fail-fast 驗證。
- XLS-R-300M + XLM-R-Large 離線載入、GPU warm-up、固定模型 ID。
- Single／batch／long API；batch 內部維持逐筆 forward，確保與 N3
  batch-size-one 評估一致。
- 12 秒單筆音訊與 128-token transcript 採明確拒絕，不再靜默截斷。
- 單 GPU、單 executor、bounded queue、timeout、取消安全與 OOM 安全錯誤。
- request body／file／文字／long-window limits。
- Multipart 解析前的 in-flight admission cap，避免大量請求同時把完整檔案
  複製進 Python memory。
- request ID、queue／inference／server-total／client E2E 分段量測。
- liveness、readiness、health、低敏感度 JSON metrics。
- 正式部署使用 fail-closed Bearer token、TLS、來源限制與 rate limit。

## 安裝與測試前置條件

建議使用獨立 Python 3.10 venv，不與面試後端或 Whisper 共用環境。可重現的
驗證版本在 `requirements.lock`；`requirements.txt` 是開發用相容範圍。

CUDA 環境應從與 PyTorch wheel 相符的官方 index 安裝。3090 與 A4500 已驗證
`torch==2.2.0+cu121`／`torchaudio==2.2.0+cu121`：

```bash
python3.10 -m venv Crab/.venv
Crab/.venv/bin/python -m pip install --upgrade pip
Crab/.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r Crab/api/requirements.lock
```

安裝前應先確認 NVIDIA driver 能執行 CUDA 12.1 wheel；不應直接沿用另一個
服務的 venv。CPU-only 環境可以跑免模型單元測試，但不是本專案的部署目標。

`httpx` 是 FastAPI TestClient／契約測試依賴，已列入 loose requirements 與
lock。FFmpeg 不是 Python package：WAV-only corpus 不需要它，但產生 FLAC／MP3
案例（`--require-compressed`）及驗證壓縮音訊支援時，系統的 `PATH` 必須能找到
`ffmpeg`。安裝方式由主機管理者決定，測試前先確認：

```bash
ffmpeg -version
```

模型權重、LoRA、normalization stats、Hugging Face cache 與 NNIME 語料刻意不進
Git。啟動前須由核准的 artifact／cache 管道 provision；服務會依 manifest
fail fast 驗證。`DOCUMENTATION_INDEX.md` 與部署草案也是 local-only 輔助文件，
不是 checkout 後即可取得的 runtime 依賴。

## 安全啟動

本機測試只綁 localhost：

```bash
cd Crab
.venv/bin/python -m uvicorn api.app:app \
  --host 127.0.0.1 --port 8001 --workers 1
```

A4500 目前採同機 loopback `127.0.0.1:8002`。若未來恢復跨主機部署，必須透過
私網／VPN，並套用 local-only deployment template 的 `CRAB_REQUIRE_AUTH=true`。
不可直接把 Uvicorn port 公開到 Internet，也不可啟動多個 Uvicorn workers，
否則會重複載入 GPU 模型。

## Endpoint

| Endpoint | 語意 |
|---|---|
| `GET /v1/live` | process 存活；不代表模型已就緒 |
| `GET /v1/ready` | 模型可服務與目前 queue 狀態 |
| `GET /v1/health` | 完整模型身分、device、classes、queue |
| `GET /v1/metrics` | 不含音訊／逐字稿的 request 與 timing 摘要 |
| `POST /v1/emotion/classify` | 一筆 ≤12 秒音訊 |
| `POST /v1/emotion/classify-batch` | 最多 16 筆，每筆 ≤12 秒 |
| `POST /v1/emotion/classify-long` | sliding windows；預設最多 300 秒／64 windows |

完整且與 runtime 同步的 contract 以服務的 `/openapi.json` 為準。

## 預設 resource policy

| 項目 | 預設值 | Environment variable |
|---|---:|---|
| Single file | 10 MiB | `CRAB_MAX_SINGLE_FILE_BYTES` |
| Batch audio total | 64 MiB | `CRAB_MAX_BATCH_TOTAL_BYTES` |
| Long file | 100 MiB | `CRAB_MAX_LONG_FILE_BYTES` |
| Transcript request | 4096 characters | `CRAB_MAX_TEXT_CHARS` |
| Model transcript | 128 XLM-R tokens | checkpoint 固定 |
| Single／batch duration | 12 seconds | checkpoint 固定 |
| Long duration | 300 seconds | `CRAB_MAX_LONG_AUDIO_SEC` |
| Long windows | 64 | `CRAB_MAX_LONG_WINDOWS` |
| In-flight inference requests | 9 | `CRAB_MAX_INFLIGHT_REQUESTS` |
| Waiting GPU jobs | 8 | `CRAB_GPU_QUEUE_DEPTH` |
| Queue timeout | 30 seconds | `CRAB_GPU_QUEUE_TIMEOUT_SEC` |

這些是單 GPU 的安全預設，不是跨主機 SLA。正式部署仍須依實際 request
distribution 與 transport／proxy 測量結果調校。

## 遠端 caller 串接條件

- Caller 與 API 必須位於同一個經核准的私網／VPN，或透過受保護的
  reverse proxy；不得直接公開 Uvicorn port。
- Caller 必須送出 Bearer token，並設定有限 timeout、retry 與
  unavailable fallback。
- 中文送出前須確認 ASR language hint、文字正規化，以及 audio/text
  timestamp 對齊。
- 中文 batch 失敗後，整段音訊若超過 12 秒不可打 single endpoint；N3
  會明確回 400。應保留對齊切片或安全降級為 N/A。

## Timing contract

- `queue_wait_ms`：等單 GPU 執行權的時間。
- `inference_ms`／舊欄位 `latency_ms`：decode、preprocess、encoder、SER
  forward 所在的 serialized job。
- `server_total_ms`：route 內處理時間。
- `X-Server-Total-Ms`／`Server-Timing`：middleware 量到的完整 server 時間。
- client end-to-end：由 caller／benchmark 量測。

## 分級驗證指令

以下指令均從 repository root 執行。不要把「產生測試檔」誤認為已驗證 API。

### A. 免模型、免 GPU、免服務

單元／契約測試使用 mock 或暫存 artifacts，不啟動真實模型：

```bash
Crab/.venv/bin/python -m unittest discover \
  -s Crab/api -t . -p 'test_*.py'
```

產生 WAV-only、無個資的 deterministic corpus 也不需要服務：

```bash
Crab/.venv/bin/python -m Crab.api.generate_test_corpus \
  --output-dir /tmp/n3-api-corpus
```

需要同時涵蓋 FLAC／MP3 時，先安裝 FFmpeg，再加上
`--require-compressed`。這一步只建立檔案與 `manifest.json`，不會呼叫 API。

### B. 需要 N3 artifacts、HF cache、GPU 與已啟動的真實 API

先依「安全啟動」啟動一個 worker，確認 `/v1/ready` 成功，再執行：

```bash
Crab/.venv/bin/python -m Crab.api.verify_test_corpus \
  --corpus-dir /tmp/n3-api-corpus --base-url http://127.0.0.1:8001

Crab/.venv/bin/python -m Crab.api.benchmark_api \
  --transport http --base-url http://127.0.0.1:8001

Crab/.venv/bin/python -m Crab.api.load_test \
  --base-url http://127.0.0.1:8001 --mode fixed --levels 5,15,22,30

Crab/.venv/bin/python -m Crab.api.soak_test \
  --base-url http://127.0.0.1:8001 \
  --duration-sec 1800 --concurrency 1 --request-interval-ms 950
```

`verify_test_corpus.py` 會真的逐筆 POST 到 API；benchmark、load、soak 也都需要
活的服務，不能列入 CPU-only 測試。若服務強制驗證，HTTP 工具使用 `--token`
或從 `CRAB_API_TOKEN` 讀取；報告只記「有提供 Bearer」，不可保存 token。
Benchmark/load/soak 報告會記錄 API version、Git revision（含 dirty 狀態）、
queue/request capacity 與 resource limits。

### C. 需要 local-only NNIME 語料的品質回歸

```bash
Crab/.venv/bin/python -m Crab.api.evaluate_n3 \
  --model-dir /path/to/n3-artifacts \
  --csv /path/to/nnime.csv --split Test --device cuda

Crab/.venv/bin/python -m Crab.api.evaluate_n3 \
  --model-dir /path/to/n3-artifacts \
  --csv /path/to/nnime.csv --split Test --device cuda --via-api
```

NNIME 不隨 Git 發布；638-row regression 只能在取得核准資料管道後執行。

合成音訊只驗速度、格式與穩定性。模型品質以 NNIME 638 筆 regression 為準；
正式 N3 品質應提供與音訊對齊的中文 transcript。

## Model artifacts

Checkpoint、LoRA weights、normalization pickle 與 Hugging Face cache 不進
Git，必須由部署流程另外 provision。Repository 只保存可審查的 SHA-256
manifest；服務啟動時會先驗證 artifacts，缺檔或 checksum 不符即 fail fast。
