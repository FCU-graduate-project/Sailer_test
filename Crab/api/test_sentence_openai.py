import io
import os
import requests
import torchaudio
import base64

# 💡 請在這裡填入你的 OpenRouter API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")

CRAB_API = "http://localhost:8001"
AUDIO_FILE = "/tmp/demo_with_ground_truth.wav"  # 使用我們之前產生的 demo 檔

def main():
    if not OPENAI_API_KEY.startswith("sk-or-v1"):
        print("⚠️ 看起來這不是 OpenRouter 的金鑰。")

    print("🚀 正在讀取音訊並轉換為 Base64...")
    try:
        with open(AUDIO_FILE, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 讀取音訊檔案失敗: {e}")
        return

    print("🚀 正在呼叫 OpenRouter Whisper API 進行轉錄 (含時間戳)...")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 根據 OpenRouter 官方文件，音訊轉錄必須使用 input_audio 物件
    payload = {
        "model": "openai/whisper-large-v3",
        "input_audio": {
            "data": audio_base64,  # 只要純 Base64 字串，不用 data URI 前綴
            "format": "wav"
        },
        "language": "en"
    }
    
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/audio/transcriptions",
            headers=headers,
            json=payload
        )
        resp_data = resp.json()
        print(f"DEBUG resp_data: {resp_data}")
        
        if "error" in resp_data:
            print(f"❌ OpenRouter 報錯: {resp_data['error']}")
            print(f"完整回應: {resp_data}")
            return
            
    except Exception as e:
        print(f"❌ 呼叫 OpenRouter 失敗: {e}")
        return

    print("✅ 轉錄成功！開始進行分句處理...")

    sentences = []
    
    # 1. 整理出所有的單字與時間
    words_data = resp_data.get("words", [])
    
    if not words_data and "segments" in resp_data:
        # 如果沒有獨立的 words，嘗試從 segments 裡撈 (有些模型可能不支援 word level)
        print("ℹ️ 未找到 word-level timestamps，嘗試使用 segment-level...")
        for seg in resp_data["segments"]:
            sentences.append({
                "text": seg["text"].strip(),
                "start": seg["start"],
                "end": seg["end"]
            })
    elif words_data:
        words = []
        for word in words_data:
            words.append({
                "text": word["word"].strip(),
                "start": word["start"],
                "end": word["end"]
            })

        # 2. 依標點符號組合成句子
        current_words = []
        for w in words:
            current_words.append(w)
            if w["text"].endswith((".", "?", "!")):
                sentences.append({
                    "text": " ".join(cw["text"] for cw in current_words),
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"]
                })
                current_words = []
                
        if current_words:
            sentences.append({
                "text": " ".join(cw["text"] for cw in current_words),
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"]
            })

    if not sentences:
        print("❌ 無法從轉錄結果中切分出句子！")
        return

    # 3. 合併小於 3 秒的短句
    MIN_DURATION = 3.0
    merged = [sentences[0]]
    for s in sentences[1:]:
        prev = merged[-1]
        if (prev["end"] - prev["start"]) < MIN_DURATION:
            merged[-1] = {
                "text": prev["text"] + " " + s["text"],
                "start": prev["start"],
                "end": s["end"]
            }
        else:
            merged.append(s)
    sentences = merged

    print(f"📊 成功切分出 {len(sentences)} 個句子。開始丟給 Crab API 進行情緒辨識...")

    # 4. 讀取音訊並切片丟給 Crab API
    waveform, sr = torchaudio.load(AUDIO_FILE)

    print("\n" + "="*80)
    print("  📊 OpenRouter Whisper + Crab API 雙模態分析結果")
    print("="*80)

    for i, sent in enumerate(sentences):
        # 切片
        start_sample = int(sent["start"] * sr)
        end_sample = int(sent["end"] * sr)
        segment = waveform[:, start_sample:end_sample]

        # 轉成 bytes
        buf = io.BytesIO()
        torchaudio.save(buf, segment, sr, format="wav")
        audio_bytes = buf.getvalue()

        # 呼叫你的 Crab API
        try:
            api_resp = requests.post(
                f"{CRAB_API}/v1/emotion/classify",
                files={"audio": ("segment.wav", audio_bytes, "audio/wav")},
                data={"text": sent["text"]}
            )
            result = api_resp.json()
            
            print(f"[{i+1:2d}] {sent['start']:5.1f}s ~ {sent['end']:5.1f}s | {result['primary_label']:14s}")
            print(f"        📝 \"{sent['text']}\"\n")
            
        except Exception as e:
            print(f"[{i+1:2d}] 呼叫 Crab API 失敗: {e}")

if __name__ == "__main__":
    main()
