# OKEKE 雙語 4 類 SER 重訓計劃(MSP + EmotionTalk · LoRA)

> 目標:讓遊戲的 SER 中文也準。沿用已驗證的 **strategyA 雙語 LoRA 配方**,
> 只把類別從 3 類(Excited/Unconfident/Neutral)換成遊戲的 **4 類(Angry/Happy/Neutral/Anxious)**。
> ⚠️ 動工前等 Kaiser 確認;GPU 單卡、AMP+grad ckpt、不准 OOM、一察覺不對就停。

---

## 0. 可行性(已確認 ✅)
EmotionTalk 原始有完整 7 情緒(中英拼寫都正規化過):
`happy / surprise / sad / disgust / anger / fear / neutral`
→ 先前 3 類 scheme **故意丟掉 anger/disgust**;我們 4 類**全用得到**,映射跟 MSP 完全同款:

| 遊戲 4 類 | MSP(8→4) | EmotionTalk(7→4) |
|---|---|---|
| **Angry** | Angry+Disgust+Contempt | anger + disgust |
| **Happy** | Happy+Surprise | happy + surprise |
| **Neutral** | Neutral | neutral |
| **Anxious** | Sad+Fear | sad + fear |

---

## 1. 資料 pipeline(建 `okeke_bilingual_4class.csv`)

**MSP 側**:直接重用既有 `data/okeke_msp_4class.csv`(已是 4 類、含 Text、絕對路徑)→ 加 `Language=EN`。

**EmotionTalk 側**(要新建一個 build 腳本):
- 複製 `scripts/build_emotiontalk_crab_csv.py` → `scripts/build_emotiontalk_okeke4_csv.py`
- 改 `SCHEME_MAP` 為 4 類:
  ```
  "happy":"Happy","surprise":"Happy",
  "sad":"Anxious","fear":"Anxious",
  "anger":"Angry","disgust":"Angry",   # ← 不再丟掉
  "neutral":"Neutral",
  ```
- `CLASSES = ["Angry","Happy","Neutral","Anxious"]`(順序**務必**跟遊戲一致)
- 保留它的 group-based split(Val=G01/G12、Test=G03/G15)、16k 重採樣、Text。→ 加 `Language=ZH`。

**合併**(新建):
- 複製 `scripts/build_bilingual_train_csv.py` → `scripts/build_okeke_bilingual_csv.py`
- 來源改成:MSP=`okeke_msp_4class.csv`、ET=`emotiontalk_okeke4_crab_format.csv`
- `CLASSES = ["Angry","Happy","Neutral","Anxious"]`;產出:
  - `data/okeke_bilingual_4class.csv`(欄:FileName 絕對 / Text / Split_Set / Language / 4 one-hot)
  - `data/okeke_bilingual_4class_weights.json`(類別權重)
  - 沿用 split remap:MSP Test1→Test、Test2 另存 EN-only;ET Train/Dev/Test 照舊
- 內建 diagnostics:**per-split×per-language×per-class 計數、檔案存在抽查、跨來源重複檔名**。

---

## 2. 訓練設定(沿用 strategyA 配方,改 4 類)
新建 `run_okeke_bilingual_train.sh`(= `run_strategyA_bilingual.sh` 改下列):
```
.venv/bin/python bin/train_crab_lora.py \
  --ssl_type facebook/wav2vec2-xls-r-300m \
  --text_model_path FacebookAI/xlm-roberta-large \
  --pre_trained_path /tmp/__no_warmstart__ \      # 同 strategyA:不暖啟、純 HF base + 新 LoRA
  --df_path data/okeke_bilingual_4class.csv \
  --weights_json data/okeke_bilingual_4class_weights.json \
  --wav_base_dir "" \
  --language_balanced \                            # ★ 50:50 EN:ZH 取樣(WeightedRandomSampler)
  --classes_list Angry Happy Neutral Anxious \     # ★ 我們的 4 類
  --ft_mode lora --lora_rank 16 --lora_alpha 32 --lora_dropout 0.1 \
  --batch_size 16 --accumulation_steps 4 \         # 有效 batch 64(同 strategyA)
  --epochs 10 --early_stop_patience 5 \
  --lr 2e-4 --encoder_lr 1e-4 \                    # 同 strategyA(無暖啟用較高 lr)
  --contrastive_weight 2.0 --grad_clip 1.0 \
  --fusion_hidden_dim 512 --text_max_len 128 \
  --use_amp --use_grad_ckpt \                      # ★ 24GB 安全(okeke 用過)
  --num_workers 3 --eval_test \
  --model_path experiments/okeke_bilingual_4class \
  --project_name Crab_Okeke_4class --run_name okeke_bilingual_msp_et_lora
```
**可調**:OOM 就把 `accumulation_steps` 降到 2、或 `batch_size` 降 8;收斂不好再考慮暖啟 `experiments/okeke_msp_4class`。

### 資料比決策(已對照 strategyA 實測 ✅)
strategyA `bilingual_strategyA.csv` 實際 **Train split = MSP(EN) 30,000 : EmotionTalk(ZH) 11,744 = 2.55:1**
(檔案總計 5.3:1 是因 Test/Dev 塞英文,但 eval 不取樣,不影響訓練)。
→ 新 4 類維持同款 **train ≈ 2.5:1**:
- EmotionTalk 4 類撿回 anger+disgust(~4638),估 ZH train 由 11,744 → **~15,000**。
- MSP train 目標 ≈ 2.5×15,000 ≈ **37,500(~9,400/類)**。現有 `okeke_msp_4class.csv` Train 已 10,000/類=40,000 → **~2.6:1,幾乎正中**。
- **動作**:建完 4 類 ZH CSV 後印出確切 ZH train 數,必要時把 MSP 上限由 10,000/類微降到 ~9,400/類校到 2.5:1;`--language_balanced` 仍負責每批 50:50。

---

## 3. 驗證(要能證明「中文變準」)
- in-run:Test split(MSP Test1 ∪ ET Test)的 **WAR / UAR / macro-F1**(早停看 Development)。
- **分語言評估**(重點,報告用):跑 `scripts/eval_per_language.py` + `scripts/confusion_matrix_per_language.py`
  → 報 **EN vs ZH 各自準確率/混淆矩陣**。
- **對照舊模型**:同一份 ZH 測試集,跑「舊 `okeke_msp_4class`」vs「新雙語」→ 中文應明顯提升(這張對比圖放簡報超有力)。

---

## 4. 換進遊戲(1 行)
- 產物在 `experiments/okeke_bilingual_4class/`(需有 `audio_lora_adapter/`、`text_lora_adapter/`、`final_ser.pt`)。
- 改 `api/okeke_infer.py` 的 `_DEFAULT_DIR` 指到新資料夾(或啟動 SER 服務時傳 `--model_dir`)。
- 驗:`/predict_pcm` 仍回 4 類、順序 `[Angry,Happy,Neutral,Anxious]`;遊戲端零改動。
- **保留舊模型**(改名備份),萬一新模型怪可秒回退。

---

## 5. GPU 安全(動工守則)
- `CUDA_VISIBLE_DEVICES=0`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 開訓後**盯 `nvidia-smi`**;接近滿就降 batch/accum。**一 OOM 或不對就 `TaskStop`**。
- 背景跑 + watchdog(repo 有 `bin/watchdog_*.sh` 範式可仿)。
- 不動 CPU-only 路徑。

---

## 6. 動工前 Step 0 檢查(純讀資料、不動 GPU,先跑這些)
1. **EmotionTalk Angry/Anxious 樣本數夠不夠**(anger/disgust 之前被丟,要確認量足以撐 Angry 類)。
2. EmotionTalk 每筆有 **Text(逐字稿)**(文字分支需要;`emotiontalk_crab_format.csv` 已有 Text 欄 → 應 OK,確認新建 4 類版也帶上)。
3. wav 路徑存在:`datasets/emotiontalk/Audio16k`、`datasets/MSP_Podcast_Data/Audios`。
4. build 後看 diagnostics:**ZH 那側 4 類分布**會不會太偏(EmotionTalk 可能某類很少 → 靠 class weights + language_balanced 緩解,必要時上限子採樣)。

### ✅ Step 0 驗證結果(2026-06-11 實查,全綠)
- **資料位置**:在 `/home/brant/Project/SAILER_test/datasets/`(**Crab 上一層**,不是 `Crab/datasets/` — 那是空的,別被騙)。build 腳本內 `ROOT` 已寫死絕對路徑,沒問題。
- **MSP**:`datasets/MSP_Podcast_Data/Audios` = **267,905 wav**;`okeke_msp_4class.csv` 絕對路徑抽查存在 ✅。
- **EmotionTalk**:`datasets/emotiontalk/Audio16k/G*` 對得上 CSV 相對路徑 ✅;**原始 json 38,500 筆**(`Audio/Audio/json/G*`),每筆有 `emotion_result`(含 **anger/disgust**)+ `content`(逐字稿)→ 4 類重建可行。
- **4 類預估量**(由 build log 反推):Angry≈4638(anger+disgust,撿回)/ Happy≈3468 / Neutral≈9378 / Anxious≈1766(最少但夠;盯其 recall)。
- **舊模型 4 件套**齊全:`experiments/okeke_msp_4class/` 有 `audio_lora_adapter`、`text_lora_adapter`、`final_ser.pt`、`train_norm_stat.pkl`(基線對照 + 遊戲推論)。
- **環境**:`Crab/.venv` = torch 2.2.0+cu121 / peft 0.19.1 / transformers 4.47.1 / CUDA True ✅。
- **base 模型**:XLS-R-300M、XLM-RoBERTa-large 已在 HF 快取,**免重下載** ✅。
- **GPU**:RTX 3090 24GB,當前用 ~6.8GB(餘 ~17.7GB,batch16+AMP+grad-ckpt 充足);開訓前確認那 6.8GB 是誰(可能是遊戲 SER 服務)。
- **參考腳本**全在:`build_emotiontalk_crab_csv.py`、`build_bilingual_train_csv.py`、`run_strategyA_bilingual.sh`、`train_crab_lora.py`、`eval_per_language.py`、`confusion_matrix_per_language.py`。
- **尚缺(動工時產,免 GPU)**:`build_emotiontalk_okeke4_csv.py`、`build_okeke_bilingual_csv.py`、`run_okeke_bilingual_train.sh` → 產出 `okeke_bilingual_4class.csv`。
- **路徑提醒**:ET 的 CSV FileName 是**相對**(G…/…wav,base=Audio16k),MSP 是**絕對** → 合併腳本要把 ET 轉絕對(或設 `--wav_base_dir`),沿用 `build_bilingual_train_csv.py` 既有作法即可。

---

## 7. 風險 / 備註
- **EmotionTalk 類別不平衡**:Angry/Anxious 在中文側可能偏少 → class weights + 觀察 per-class recall;真的太少考慮加 CH-SIMS 或 MELD 補。
- **時間**:6/14 簡報在即。重訓 + 評估約數小時。**簡報可先用「未來工作」帶過,真有結果再補一頁對比**。
- **不破壞現況**:全程新檔、新 experiments 目錄;遊戲指向舊模型不變,直到驗證通過才切。

---

## 待辦清單(動工日順序)
- [x] Step 0 四項檢查(2026-06-11,全綠,見上方驗證結果)
- [x] `build_emotiontalk_okeke4_csv.py`(4 類映射)→ `emotiontalk_okeke4_crab_format.csv`
      → **19,250 筆,0 fail,補採樣 4,638 個新 Angry 檔**。ZH train=15,413(Angry 3669/Happy 2807/Neutral 7568/Anxious 1369)。
- [x] `build_okeke_bilingual_csv.py` → `okeke_bilingual_4class.csv` + weights
      → **69,782 筆,train EN:ZH=38532:15413=2.50:1**,存活抽查 0 missing。weights:Angry1.014/Happy1.084/Neutral0.784/Anxious1.226。
- [x] 看 diagnostics(per-lang×per-class 計數)→ 上行
- [x] `run_okeke_bilingual_train.sh` 已寫好備著(**尚未執行,等「動工」**)
- [ ] **動工**:清 GPU → 跑 `run_okeke_bilingual_train.sh`(盯 GPU)
- [ ] 分語言評估 + 對照舊模型
- [ ] 換 `okeke_infer.py` 指向新模型 + 遊戲實測
