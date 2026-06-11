"""Okeke 4-class SER 推論封裝(給「AI 奧客應對演練」用)。

載入訓練好的 LoRA 微調模型,輸入一段 wav(+ 可選 transcript)→ 4 類情緒機率
(Angry / Happy / Neutral / Anxious)。辨識對象是「玩家(店員)講話的語氣」。

模型組成(預設 experiments/okeke_bilingual_4class,見 _DEFAULT_DIR;結構與舊 okeke_msp_4class 同):
  - 音訊:facebook/wav2vec2-xls-r-300m + audio_lora_adapter(q_proj/v_proj, r16/a32)
  - 文字:FacebookAI/xlm-roberta-large + text_lora_adapter(query/value)
  - 融合+分類頭:MultiModalEmotionClassifierDeep(final_ser.pt,num_emotions=4)
  - 波形正規化:train_norm_stat.pkl(mean/std)—— 一定要套,否則編碼器看到不同分布→預測爛掉

CLI:
  uv run python api/okeke_infer.py <wav_path> [--text "逐字稿"]
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

# 讓 `from src.models.ser import ...` 找得到(本檔在 Crab/api/ 下)
_CRAB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CRAB_ROOT not in sys.path:
    sys.path.insert(0, _CRAB_ROOT)
from src.models.ser import MultiModalEmotionClassifierDeep  # noqa: E402

CLASSES = ["Angry", "Happy", "Neutral", "Anxious"]   # index 0..3,順序同訓練(build script)
TARGET_SR = 16000
MAX_LEN = 12 * TARGET_SR        # 訓練截斷 12 秒
TEXT_MAX_LEN = 128
_DEFAULT_DIR = os.path.join(_CRAB_ROOT, "experiments", "okeke_bilingual_4class")
# 備援(回退):舊純英文模型 = os.path.join(_CRAB_ROOT, "experiments", "okeke_msp_4class")
# 2026-06-11 換成雙語 4 類:中文 test macro-F1 0.46→0.57(+24.5%)
_SSL_NAME = "facebook/wav2vec2-xls-r-300m"
_TEXT_NAME = "FacebookAI/xlm-roberta-large"

_HARD_BREAK = set(".?!。?!…~")        # 句末:最優先斷
_SOFT_BREAK = set(",;:、,;:::")        # 子句:次優先(且該段需 ≥25% max)


def _ends_with(word: str, charset: set) -> bool:
    s = (word or "").rstrip()
    return bool(s) and s[-1] in charset


def split_long_sentence(words, max_duration: float):
    """把過長語句依「詞級時間戳」切成 ≤max_duration 的段落,回傳 [(start_idx, end_idx), ...]。

    斷點優先序(玩家提供的演算法):
      1. 區間內「最後一個」句末標點(. ? ! 。 ? ! …)—— 一定採用。
      2. 否則「最後一個」子句標點(, ; : 、 …),且該段需 ≥ 25% max_duration。
      3. 否則「等時平衡」:把剩餘時間均分,挑 word.end 最接近目標時刻者。
    words:list[dict],每個有 start/end/word。
    """
    chunks = []
    start = 0
    N = len(words)
    while start < N:
        c0 = float(words[start]["start"])
        end = start
        while end < N and float(words[end]["end"]) - c0 <= max_duration:
            end += 1
        end -= 1
        if end < start:
            end = start                                  # 單一詞就超長 → 至少含一個詞
        if end == N - 1:                                 # 剩下的全裝得下 → 收尾
            chunks.append((start, end))
            break
        # 優先 1:最後一個句末標點
        split = None
        for i in range(end, start - 1, -1):
            if _ends_with(words[i]["word"], _HARD_BREAK):
                split = i
                break
        # 優先 2:最後一個子句標點 + 段長 ≥ 25% max
        if split is None:
            for i in range(end, start - 1, -1):
                if _ends_with(words[i]["word"], _SOFT_BREAK) and (float(words[i]["end"]) - c0) >= 0.25 * max_duration:
                    split = i
                    break
        # 優先 3:等時平衡
        if split is None:
            remaining = float(words[N - 1]["end"]) - c0
            n_rem = max(1, math.ceil(remaining / max_duration))
            target = c0 + remaining / n_rem
            best, best_d = start, abs(float(words[start]["end"]) - target)
            for i in range(start, end + 1):
                d = abs(float(words[i]["end"]) - target)
                if d < best_d:
                    best, best_d = i, d
            split = best
        chunks.append((start, split))
        start = split + 1
    return chunks


class OkekeSER:
    """載入一次、可重複推論。微服務啟動時建一個實例即可。"""

    def __init__(self, model_dir: str = _DEFAULT_DIR, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_dir = model_dir

        # 1. 波形正規化統計(訓練時存)
        with open(os.path.join(model_dir, "train_norm_stat.pkl"), "rb") as f:
            self.wav_mean, self.wav_std = pickle.load(f)
        if self.wav_std < 1e-5:   # 防爆:std 異常小就退回 1.0
            self.wav_std = 1.0

        # 2. 音訊編碼器 + LoRA adapter
        ssl = AutoModel.from_pretrained(_SSL_NAME)
        audio_dim = ssl.config.hidden_size       # 1024
        self.ssl = PeftModel.from_pretrained(ssl, os.path.join(model_dir, "audio_lora_adapter"))
        self.ssl.to(self.device).eval()

        # 3. 文字編碼器 + LoRA adapter + tokenizer
        txt = AutoModel.from_pretrained(_TEXT_NAME)
        text_dim = txt.config.hidden_size        # 1024
        self.text = PeftModel.from_pretrained(txt, os.path.join(model_dir, "text_lora_adapter"))
        self.text.to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(_TEXT_NAME)

        # 4. 融合+分類頭(4 類)
        self.ser = MultiModalEmotionClassifierDeep(
            features1_dim=audio_dim, features2_dim=text_dim,
            fusion_hidden_dim=512, num_emotions=len(CLASSES), dropout=0.5,
        )
        self.ser.load_state_dict(torch.load(os.path.join(model_dir, "final_ser.pt"), map_location="cpu"))
        self.ser.to(self.device).eval()

    @torch.no_grad()
    def _infer(self, wav16, text: str = "") -> dict:
        """核心:np.float32 @16k(單聲道)+ 逐字稿 → {類: 機率}(未取 argmax)。"""
        wav = np.asarray(wav16, dtype=np.float32)[:MAX_LEN]
        wav = (wav - self.wav_mean) / (self.wav_std + 1e-6)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        amask = torch.ones(1, wav_t.shape[1], dtype=torch.long, device=self.device)
        a = self.ssl(wav_t, attention_mask=amask).last_hidden_state

        enc = self.tok(text or "", return_tensors="pt", max_length=TEXT_MAX_LEN,
                       padding="max_length", truncation=True)
        ids = enc["input_ids"].to(self.device)
        tmask = enc["attention_mask"].to(self.device)
        t = self.text(input_ids=ids, attention_mask=tmask).last_hidden_state

        logits = self.ser(a, t)
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
        return {c: float(p) for c, p in zip(CLASSES, probs)}

    @staticmethod
    def _wrap(probs: dict, chunks: int = 1, timeline=None) -> dict:
        d = {c: round(probs[c], 4) for c in CLASSES}
        label = max(d, key=d.get)
        out = {"label": label, "confidence": d[label], "probs": d, "chunks": chunks}
        if timeline is not None:
            out["timeline"] = timeline
        return out

    def predict(self, wav_path: str, text: str = "") -> dict:
        """從檔案推論(給 /predict 與 CLI;wav 最穩)。"""
        wav, _ = librosa.load(wav_path, sr=TARGET_SR)      # 單聲道 16k
        return self._wrap(self._infer(wav, text))

    def predict_array(self, wav16, text: str = "") -> dict:
        """從 np.float32 @16k 推論(給微服務 /predict_pcm 的短語句直通)。"""
        return self._wrap(self._infer(wav16, text))

    def predict_chunked(self, wav16, words, text: str = "", max_dur: float = 12.0) -> dict:
        """長語句:依詞級時間戳切段 → 各段 SER → 依段長加權平均機率。

        ≤max_dur 或無 words → 直通單次推論。回傳含 timeline(每段標籤)。
        """
        wav16 = np.asarray(wav16, dtype=np.float32)
        dur = len(wav16) / TARGET_SR
        if not words or dur <= max_dur:
            return self._wrap(self._infer(wav16, text), chunks=1)
        agg = {c: 0.0 for c in CLASSES}
        total = 0.0
        timeline = []
        for s_idx, e_idx in split_long_sentence(words, max_dur):
            t0 = float(words[s_idx]["start"])
            t1 = float(words[e_idx]["end"])
            seg = wav16[int(t0 * TARGET_SR):int(t1 * TARGET_SR)]
            if len(seg) < int(0.2 * TARGET_SR):           # 太短(<0.2s)→ 跳過
                continue
            ctext = "".join(str(words[k].get("word", "")) for k in range(s_idx, e_idx + 1))
            p = self._infer(seg, ctext)
            w = max(0.1, t1 - t0)
            total += w
            for c in CLASSES:
                agg[c] += p[c] * w
            timeline.append({"t": [round(t0, 2), round(t1, 2)], "label": max(p, key=p.get),
                             "probs": {c: round(p[c], 4) for c in CLASSES}})
        if total == 0:                                    # 全被跳過(極端)→ 退直通
            return self._wrap(self._infer(wav16, text), chunks=1)
        probs = {c: agg[c] / total for c in CLASSES}
        return self._wrap(probs, chunks=len(timeline), timeline=timeline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="wav 檔路徑")
    ap.add_argument("--text", default="", help="逐字稿(可選)")
    ap.add_argument("--model_dir", default=_DEFAULT_DIR)
    args = ap.parse_args()
    ser = OkekeSER(args.model_dir)
    out = ser.predict(args.wav, args.text)
    print(f"預測:{out['label']}  (信心 {out['confidence']})")
    print("各類機率:", out["probs"])


if __name__ == "__main__":
    main()
