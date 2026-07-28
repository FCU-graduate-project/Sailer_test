# Crab 中英雙語化 — 資料集詳細規格

**版本:** v1.0(2026-07-02 從 `BILINGUAL_FINETUNE_PLAN.md §8` 拆出)
**用途:** 集中管理所有已用 / 候選 / 淘汰資料集的規格、映射、drop 決策、license。
**關聯文件:** [`BILINGUAL_FINETUNE_PLAN.md`](BILINGUAL_FINETUNE_PLAN.md)(架構 + 訓練配置)/ [`BILINGUAL_WORK_LOG.md`](BILINGUAL_WORK_LOG.md)(時序紀錄)

---

## 1. 資料集總覽

| # | 資料集 | Lang | 規模(scheme1 可用)| 用途 | 狀態 |
|:--:|---|:--:|---|---|:--:|
| 1 | **MSP-Podcast** | EN | 168,621(train subsample 30k)| EN 主 corpus | ✅ 現用 |
| 2 | **EmotionTalk**(BAAI)| ZH | 14,612(Train 11,744 / Dev 1,421 / Test 1,447)| ZH 主 corpus 1 | ✅ 現用 |
| 3 | **CNSCED**(HF)| ZH | 10,120(Train 8,624 / Dev 810 / Test 686)| ZH 主 corpus 2 | ✅ 現用(2026-07-01)|
| 4 | **NNIME**(NTHU BIIC)| ZH | 4,300(Train 3,054 / Dev 608 / Test 638)| ZH 主 corpus 3 | ✅ 現用(2026-07-01)|
| 5 | **IEMOCAP** | EN | 4,575 | cross-dataset eval | ✅ eval-only |
| 6 | **MELD** | EN | 2,197 | cross-dataset eval | ✅ eval-only |
| 7 | 合成 clip 560 EN + 560 CN | EN+ZH | 1,120 | diagnostic | ✅ 歷史 |
| 8 | CH-SIMS v2(s) | ZH | 4,403(5-class)| 歷史 Stage A | 🗄️ 淘汰 |
| 9 | CHEAVD-2.0 | ZH | 數萬(7-class)| 未來 B1 候選 | 🔮 未送申請 |
| 10 | ESD-CN | ZH | 17.5k(讀稿)| 未來 B1 候選 | 🔮 未送申請 |
| 11 | 10-rater 朋友合成 | ZH | TBA | Stage B-target(校準)| ⛔ 等朋友 |

---

## 2. bilingual_v2.csv(v2b 訓練用 CSV,2026-07-01 建)

**檔案:** `data/bilingual_v2.csv`(106,499 rows,由 [`scripts/build_bilingual_v2_csv.py`](scripts/build_bilingual_v2_csv.py) 建)

**欄位:** `FileName`, `Text`, `Split_Set`, `Excited`, `Unconfident`, `Neutral_3Class`, `Language`, `Source`

**組成:**

| Split | Language | Source | rows |
|:--:|:--:|---|---:|
| **Train** | EN | MSP | 30,000 |
| | ZH | EmotionTalk | 11,744 |
| | ZH | CNSCED | 8,624 |
| | ZH | NNIME | 3,054 |
| | | **Train 合計** | **53,422** |
| **Dev** | EN | MSP | 19,467 |
| | ZH | EmotionTalk | 1,421 |
| | ZH | CNSCED | 810 |
| | ZH | NNIME | 608 |
| | | **Dev 合計** | **22,306** |
| **Test** | EN | MSP | 28,000 |
| | ZH | EmotionTalk | 1,447 |
| | ZH | CNSCED | 686 |
| | ZH | NNIME | 638 |
| | | **Test 合計** | **30,771** |

**Per-class train 分布:**

| Class | EmotionTalk | CNSCED | NNIME | MSP | **合計** |
|---|---:|---:|---:|---:|---:|
| **Excited** | 2,807 | 1,835 | 808 | ~11,000 | ~16,450 |
| **Unconfident** ⭐ | 1,369 | 990 | 628 | ~4,500 | ~7,487 |
| **Neutral** | 7,568 | 5,799 | 1,618 | ~14,500 | ~29,485 |

**vs v1(bilingual_strategyA.csv,只有 MSP + EmotionTalk):** ZH train ×1.99 / ZH Unconfident train ×2.18(2,987 vs 1,369)

---

## 3. MSP-Podcast(EN 主 corpus)

| 項 | 值 |
|---|---|
| **來源** | The MSP-Podcast Corpus(Busso lab, UTD)|
| **License** | 學術授權 |
| **規模** | 168,621 utt 全集 / Train 30,000 uniform subsample(`random_state=42`)|
| **Text** | ✅ 內建 transcript |
| **Split** | 官方切分 |
| **Test 拆分** | **Test1**(in-domain,10,684 utt)/ **Test2**(cross-session within MSP,10,684 utt)|
| **cross-dataset eval** | 已測 IEMOCAP(4,575)+ MELD(2,197)— Strategy A LoRA 分別得 0.5990 / 0.5097 |

**Subsample 理由**:原 168:12 EN:ZH 比例太懸殊,sampler 強拉 ZH 14× → overfit;30k → 2.5:1,sampler 拉 1.78× 溫和;LoRA 5-10k 樣本即飽和;全 168k 訓練時間 3×(報酬遞減)。

---

## 4. EmotionTalk(ZH 主 corpus 1)

| 項 | 值 |
|---|---|
| **來源** | BAAI(北京智源)|
| **License** | 學術授權 |
| **規模** | 14,612 utt(Train 11,744 / Dev 1,421 / Test 1,447)|
| **Modality** | audio + text(內建)|
| **Label 體系** | 3-class 內建可映射 scheme1 |
| **Unconfident 樣本** | **1,766 筆**(唯一有 explicit fear column)⭐ |
| **domain** | 錄音室 dialog(讀稿)|
| **CSV** | `data/emotiontalk_crab_format.csv` |

**scheme1 映射**:對齊 `Unconfident = Fear + Sad` 操作型定義。

---

## 5. CNSCED(ZH 主 corpus 2,2026-07-01 落地)

| 項 | 值 |
|---|---|
| **來源** | `Kaiser20207/CNSCED` @ HuggingFace(已下載到 lab)|
| **License** | 用戶自持,學術報告 OK |
| **規模(raw)** | 15,785 utt(train 12,823 / val 1,509 / test 1,453)|
| **規模(scheme1 可用)** | **10,120**(Train 8,624 / Dev 810 / Test 686);保留率 64.1% |
| **Speaker 數** | **454**(train 354 / val 43 / test 57)⭐ 三 corpus 最高 diversity |
| **Modality** | audio only |
| **Text** | 用 `faster-whisper-large-v3 int8_float16` 補全,15,785 筆完成 fail=0(64 分鐘)|
| **Label 體系** | 6 情緒 code(**H** happy / **B** surprise / **S** sad / **F** fear / **0** neutral / **A** angry drop)+ **W** aroused 副標籤 |
| **多標籤** | ✅ 允許共現(A2_S1_W2 = 中怒 + 弱悲 + 中激動)|
| **Fear 樣本** | ⚠️ **F 只 32 筆全集**(0.2%)— fear coverage 低,主要靠 EmotionTalk + NNIME |
| **CSV** | `data/cnsced_crab_format.csv` |

**scheme1 映射**:H + B → Excited / S + F → Unconfident / 0 → Neutral / A → drop / multi-emotion → drop

**Build script**:[`scripts/transcribe_cnsced.py`](scripts/transcribe_cnsced.py)(Whisper)→ [`scripts/build_cnsced_crab_csv.py`](scripts/build_cnsced_crab_csv.py)(映射)

**domain**:自然對話 + 多 speaker(vs EmotionTalk 錄音室 dialog)

---

## 6. NNIME(ZH 主 corpus 3,2026-07-01 落地)

### 6.1 基本規格

| 項 | 值 |
|---|---|
| **全名** | NTHU-NTUA Chinese Interactive Multimodal Emotion Corpus(Chou et al. 2017, ACII)|
| **來源** | NTHU BIIC lab EULA 授權(Multiple Academic Users License 給指導教授 team)|
| **License** | **academic / non-profit 限定,禁商用**;**禁 team 以外成員 / 禁再散播** |
| **規模(raw Speech)** | 4,738 utt(non-speech Laugh/Sigh/Sobbing 等已 drop)|
| **規模(scheme1 可用)** | **4,300**(Train 3,054 / Dev 608 / Test 638);保留率 90.8% |
| **Speaker 數** | 44(22 teams × A/B,**4B = 22A 同一人**)|
| **Modality** | audio + video + ECG + text |
| **Text** | ✅ 內建 sentence-level transcript(免 Whisper)|
| **CSV** | `data/nnime_crab_format.csv` |
| **domain** | 自發 dyadic 對話(vs EmotionTalk 讀稿、CNSCED 對話)|

**Label 體系**:**6 情緒 filename atmosphere**(angry/frustration/happy/neutral/sad/surprise)+ **62 個 fine-grained rater 標籤**(6 raters × per-sentence,可 multi-label)

**標註策略**:**Native Rater**(外部觀察者,類 observer-report),6 人 majority vote(paper 論證 observer-based 最 objective)

**Split 策略**:team_id anchored → Train 1-3,5-16 / Dev 17-19 / Test 4,20-22(4 跟 22 放同 split 防 4B=22A 語者洩漏)

### 6.2 NNIME 62-label → scheme1 v3 映射(2026-07-01 用戶拍板)

| scheme1 類 | 標籤數 | 內容 |
|---|---:|---|
| **Excited**(9)| 快樂 / 喜樂 / 興奮 / 驚訝 / 期待 / 振作 / 鬥志 / 激動 / **自信**(v3 加)|
| **Unconfident**(31)| **fear 家族**(恐懼 ⭐ 唯一真 fear / 緊張 / 焦慮 / 焦急 / 擔心 / 憂心 / 心虛 / 急 / 壓力 etc)+ **sad 家族**(傷心 / 低落 / 失望 / 愧疚 / 無奈 / 痛 / 痛苦)+ **社交低自信**(尷尬 / 囧 / 害躁 / 低聲下氣)+ **v3 加**:挫折 / 傻眼 / 懷疑 / 疑惑 / 莫名奇妙 |
| **Neutral**(6)| 中性 / 放鬆 / 祥和 / 認真 / 嚴肅 / 其他 |
| **DROP**(16)| **angry 家族**(生氣 / 煩 / 不悅 / 埋怨 / 警告 etc)+ 雜訊 + v3 drop:想睡 / 無聊 / 感性 / 關懷 / 下定決心 / 語重心長 |

### 6.3 v3 mapping 論證(paper §V.C 可寫,會議可答)

- **挫折(Frustration)→ Unconfident**:實證 794 frustration atmosphere 中,Native Rater 標的 sad-cluster(失望+傷心+恐懼)= **19.5%** > angry-cluster = **8.3%**。**中文「挫折」語意上偏 sad-adjacent**(setback → helpless / disappointed),不同於英文 frustration(anger-adjacent)。此決定使 Unconfident train 樣本 +32%(NNIME +152 / 476→628)
- **傻眼 / 懷疑 / 疑惑 / 莫名奇妙 → Unconfident**:negative surprise + uncertainty,belong to「不確定/沒把握」譜系,匹配 scheme1「Unconfident」語意
- **自信 → Excited**:high arousal + 正 valence,語意正好對比 Unconfident(paper 語意 axis 對稱性論證)
- **想睡 / 無聊 → DROP**:能量太低,語音特徵無區辨力
- **關懷 / 感性 / 下定決心 / 語重心長 → DROP**:語意太模糊,任一 scheme1 bucket 都不精準

### 6.4 三個 data slice 的取捨(paper §V.B / 會議問答用)

NNIME 資料集分三塊,只用 **Sentence-Level Speech(4,300 usable / 4,738 raw)**。以下解釋另外兩塊為何 drop。

#### (A) Session Level(201 個 wav,每個 2-3 分鐘,共 11.3 hr)— 確定 drop

**驗證發現**:Session Level 是 Sentence Level 的「未切分原始」版本。
- e.g. `angry_01_A.wav`(session, 135 秒)= 13 個 `angry_01_A_001.wav`, `angry_01_A_002.wav` ...(sentence 短句)接連
- 201 sessions × 平均 23.6 sentences/session = ~4,738 sentences → **跟 sentence-level Speech 完全對應**

**為什麼 drop**:

| 問題 | 說明 |
|---|---|
| ① **是 duplication,不是新資料** | Session 音訊 = Sentence Level 的合體,若加就是同一段訊號訓 2 次 → 過擬合單一 corpus subset,paper 標準會被質疑 |
| ② 標籤粒度不對 | Session 只有 session-level discrete label(peer/director/self 每人一個 label /整段 3 分鐘)。scheme1 是 **per-sentence 分類**,session label 太粗 |
| ③ 時長超 SER 標準 | XLS-R / XLM-R 預設 batch 處理 ≤ 30 秒;3 分鐘直接 truncate 或需 sliding window,VRAM 爆炸 |
| ④ 混合情緒訊號 | 3 分鐘內演員情緒起伏,一個 aggregate label 無法代表全段 |

#### (B) 非語音類別(Laugh / Sigh / Sobbing / Audien_Laugh / Others,共 858 筆,15.3%)— drop

| 類 | 數量 | 占比 | 平均時長 | 可能情緒訊號 |
|---|---:|---:|---:|---|
| **Speech** ⭐ | 4,738 | **84.7%** | 1.2s | 主要訓練 |
| Laugh | 248 | 4.4% | 1.4s | ⚠️ 可能 Excited(happy laugh vs nervous laugh 難分)|
| Audien_Laugh | 200 | 3.6% | 1.8s | ❌ 觀眾背景音,非目標語者 |
| Sigh | 163 | 2.9% | 1.1s | ⚠️ 可能 Neutral / Unconfident(tired sigh)|
| Sobbing | 131 | 2.3% | 1.2s | ⚠️ 可能 Unconfident(crying)|
| Others | 116 | 2.1% | 0.4s | ❌ 太短,unknown |

**為什麼 drop**:

| 問題 | 說明 |
|---|---|
| ① **Text modality mismatch** | Crab 是 audio + text bimodal;非語音沒 text(笑聲/嘆氣沒詞語)→ 只能餵空字串或 `<laugh>` sentinel token,XLM-R 沒訓過,fusion 表示不可用 |
| ② **XLS-R domain mismatch** | XLS-R 預訓 = spontaneous speech;笑聲/哭聲/嘆氣分布不同,capacity 浪費 |
| ③ **Label noise 高** | 同「笑聲」可能 happy laugh / nervous laugh / sarcastic laugh — 無法用單一 emotion class 覆蓋 |
| ④ **Audien_Laugh 非目標語者** | 200 筆背景觀眾笑聲,絕對 drop |

**若一定要加(理論上限)**:排除 Audien_Laugh + Others 後,Laugh(248 大多 Excited)+ Sobbing(131 大多 Unconfident)+ Sigh(163 mix)= 542 筆 → NNIME 4,300 → 4,842(+12.6%)。但要付代價:改 text sentinel token + 混合 domain,工程量 1 週,paper 得寫「audio-only fine-tune sub-branch 對比」。

#### (C) 選擇:**只用 Sentence-Level Speech,4,300 rows**

**paper defense point**:「我們選擇 sentence-level speech 作為訓練子集,對應 SER literature standard(IEMOCAP / MSP-Podcast / EmotionTalk 均為 sentence-level speech)。Session Level 是同段音訊的未切分版本,包含即會產生重複訓練訊號。非語音類別缺乏 text modality,不適合 Crab 的 bimodal 架構;未來 audio-only extension 可以重新考慮。」

---

## 7. 三 ZH corpus 互補性(paper §V.A)

| 面向 | EmotionTalk | CNSCED | NNIME |
|---|---|---|---|
| **domain** | 錄音室 dialog | 自然 dialog(多 speaker)| 自發 dyadic |
| **Speaker 數** | ~75(估)| **454** ⭐ | 44(22 teams × A/B)|
| **Text** | ✅ 內建 | ✅ Whisper 補 | ✅ 內建(sentence-level)|
| **標註 protocol** | 3-class 內建 | 多 label + arousal 副軸(6 code)| **62 fine-grained + 6-rater majority** |
| **Fear 訊號** | ✅ 內建 Unconfident(fear+sad 混)| ⚠️ 32 筆 F | ✅ 明確「恐懼」+ 挫折 sad-adjacent |
| **domain 貢獻** | in-domain baseline | 對話多樣性 + speaker 廣 | 自發 + 高品質標註 |

---

## 8. Cross-dataset eval 資料集(EN 主 corpus 之外)

| 資料集 | 樣本 | 用途 | 標籤映射 | Strategy A LoRA 結果 |
|---|---:|---|---|---:|
| **IEMOCAP** | 4,575 | USC 演員,scripted dyadic,Session 1-5 | hap+exc+sur → Excited(1,743)/ sad+fea → Unconfident(1,124)/ neu → Neutral(1,708)/ 丟 ang+fru+dis+xxx+oth | **0.5990** ⭐ |
| **MELD** | 2,197 | Friends TV,多角色配音 + 背景音 + 笑聲 | joy+surprise → Excited(683)/ sadness+fear → Unconfident(258)/ neutral → Neutral(1,256)/ 丟 anger+disgust | **0.5097** |

**兩個 cross-dataset 平均**:(0.5990 + 0.5097) / 2 = **0.5544**,剛好等於 MSP Test2 → cross-dataset gap 不比 MSP 內 cross-session gap 大。

---

## 9. Diagnostic 資料(歷史)

**合成 clip 1,120 筆(2026-05-27)**:EN Crab on 20 語者 × 8 情緒 × 7 alpha;alpha sensitivity CN/EN **3.37×**;CN 預測 **Excited 13.6% / Unconfident 15.5% / Neutral 70.9%** → 中文「塌 Neutral 怯」。

**真實中文反例(2026-05-31)**:scheme1 zero-shot 在真實 EmotionTalk Test 上 **過度預測 Excited 48.1%**(true 22.1%),**不是**合成資料的「塌 Neutral」。推測:真實中文聲調 pitch movement 被英文 SOTA 誤判為「激動」。**聲調地雷成立,但失準方向是「拉高 Excited」**。

---

## 10. 未來候選(尚未取得)

### 10.1 CHEAVD-2.0 🥇 未來 B1

| 項 | 值 |
|---|---|
| **來源** | CASIA(Email 申請,~7 天回信)|
| **License** | academic |
| **規模** | 數萬筆 |
| **Label** | 7-class(**可能有 fear**,需確認)|
| **EV** | 對 Unconfident 最 sexy,尤其如果 fear 覆蓋足 |
| **申請狀態** | 未送(2026-07-01)|

### 10.2 ESD-CN 🥉 未來 B1

| 項 | 值 |
|---|---|
| **來源** | 官網 5 分鐘 form + 24hr 回信 |
| **License** | academic |
| **規模** | 17.5k(350 句 × 10 speaker × 5 emo,讀稿 corpus)|
| **Label** | **無 fear**,只補 sad/happy/surprise/neutral |
| **EV** | 邊際(讀稿 ≠ 自然對話,且無 fear)|
| **申請狀態** | 未送(2026-07-01)|

### 10.3 朋友合成 + 10-rater ⛔ Stage B-target

| 項 | 值 |
|---|---|
| **內容** | 朋友合成 unconfident-targeted clip |
| **標註** | 10-rater 信心評分,κ≥0.4 收貨 |
| **目的** | Stage B-target 校準資料,直擊 Unconfident 0.50 plateau |
| **狀態** | ⛔ 等朋友(從硬卡關降為 proxy 校準升級)|

---

## 11. 淘汰:CH-SIMS v2(s)🗄️

| 項 | 值 |
|---|---|
| **原用途** | 歷史 Stage A(2026-05-28)|
| **規模** | 4,403(5-class sentiment)|
| **問題** | 5-class 與後續 3-class 不直接可比;純極性標籤 → 監督不到 Unconfident(confidence 軸與 valence 正交)|
| **狀態** | 已被 Strategy A 取代;2026-06-30 用戶刪除 |

---

## 12. 模型備選(hidden_size = 1024 才 drop-in)

| Text | 備註 | | Audio | 備註 |
|---|---|---|---|---|
| **XLM-R-Large**(現)| 100 langs | | **XLS-R-300M**(現)| 128 langs |
| InfoXLM-Large | 跨語常勝(候選)| | WavLM-Large | EN SOTA,ZH 怯 |
| mDeBERTa-v3 / mBERT | 768 ❌ | | MMS-300M | 1000+ langs(候選)|

---

## 13. 變更歷史

| 版本 | 日期 | 重點 |
|---|---|---|
| v1.0 | 2026-07-02 | 從 `BILINGUAL_FINETUNE_PLAN.md` §8 拆出,加 §2 bilingual_v2.csv 組成表 + §8 cross-dataset + §9 diagnostic |
