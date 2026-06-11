"""
Build an Okeke 4-class Crab-format CSV from EmotionTalk.

Same pipeline as build_emotiontalk_crab_csv.py, but maps EmotionTalk's canonical
7 emotions onto the GAME's 4 classes (identical mapping to the MSP 8→4 build):

    Angry   = anger + disgust      <- 3-class build dropped these; we KEEP them
    Happy   = happy + surprise
    Anxious = sad   + fear
    Neutral = neutral

Output columns (matches train_crab_lora.py / load_cat_emo_label):
    FileName    wav path RELATIVE to --wav_base_dir (the 16 kHz resampled dir)
    Text        transcript (content)
    Split_Set   Train / Development / Test   (by group id, same as 3-class)
    <one-hot>   Angry, Happy, Neutral, Anxious   (order MUST match the game)

Resamples the 44.1 kHz wav → 16 kHz mono ONLY for files not already present in
Audio16k (the 3-class build resampled everything EXCEPT anger/disgust, so this run
fills in ~4.6k new Angry-class wavs and skips the rest).

Usage:
    python scripts/build_emotiontalk_okeke4_csv.py --inspect   # sanity first
    python scripts/build_emotiontalk_okeke4_csv.py             # build (resamples new files)
    python scripts/build_emotiontalk_okeke4_csv.py --skip_resample  # CSV only
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import csv
import re
import subprocess
import sys

ROOT       = Path("/home/brant/Project/SAILER_test/datasets/emotiontalk")
AUDIO_ROOT = ROOT / "Audio" / "Audio"   # extracted Audio.tar → contains json/ + wav/
ANN_DIR    = AUDIO_ROOT / "json"        # annotation jsons (nested G#####/.../*.json)
WAV_ROOT   = AUDIO_ROOT / "wav"         # 44.1kHz wavs, same relative tree as file_path
OUT16K     = ROOT / "Audio16k"          # resampled target (wav_base_dir for training)
OUT_CSV   = Path("/home/brant/Project/SAILER_test/Crab/data/emotiontalk_okeke4_crab_format.csv")
OUT_WJSON = Path("/home/brant/Project/SAILER_test/Crab/data/emotiontalk_okeke4_class_weights.json")
FFMPEG    = "/home/brant/bin/ffmpeg"

CLASSES = ["Angry", "Happy", "Neutral", "Anxious"]   # order identical to the game / MSP build

# normalise the many possible spellings of emotion_result → canonical 7
EMO_NORM = {
    "happy": "happy", "happiness": "happy", "高兴": "happy", "开心": "happy",
    "surprise": "surprise", "surprised": "surprise", "惊讶": "surprise",
    "sad": "sad", "sadness": "sad", "伤心": "sad", "悲伤": "sad",
    "disgust": "disgust", "disgusted": "disgust", "厌恶": "disgust",
    "anger": "anger", "angry": "anger", "愤怒": "anger", "生气": "anger",
    "fear": "fear", "fearful": "fear", "恐惧": "fear", "害怕": "fear",
    "neutral": "neutral", "中性": "neutral", "平静": "neutral", "中立": "neutral",
}
# canonical 7 → game 4-class (nothing dropped now)
SCHEME4 = {
    "happy": "Happy", "surprise": "Happy",
    "sad": "Anxious", "fear": "Anxious",
    "anger": "Angry", "disgust": "Angry",   # ← 3-class build dropped these; KEEP
    "neutral": "Neutral",
}

# split by group id (from README: Val=G01/G12, Test=G03/G15, rest=Train)
DEFAULT_VAL_GROUPS  = {1, 12}
DEFAULT_TEST_GROUPS = {3, 15}


def norm_emotion(raw):
    if raw is None:
        return None
    return EMO_NORM.get(str(raw).strip().lower())


def find_group_id(*strings):
    """Pull a G## group id from any of the given strings (file_name / speaker_id)."""
    for s in strings:
        if not s:
            continue
        m = re.search(r"[Gg](\d+)", str(s))
        if m:
            return int(m.group(1))
    return None


def iter_annotations(ann_dir):
    """Yield raw annotation dicts (per-sample json / jsonl / single list)."""
    jsons = sorted(ann_dir.rglob("*.json"))
    jsonls = sorted(ann_dir.rglob("*.jsonl"))

    if jsonls:
        for jl in jsonls:
            with jl.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return

    if len(jsons) == 1:
        obj = json.loads(jsons[0].read_text(encoding="utf-8"))
        if isinstance(obj, list):
            yield from obj
            return
        if isinstance(obj, dict):
            vals = list(obj.values())
            if vals and all(isinstance(v, dict) for v in vals):
                yield from vals
            else:
                yield obj
            return

    for jp in jsons:
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, list):
            yield from obj
        else:
            obj.setdefault("_json_path", str(jp))
            yield obj


def rel_wav_path(ann):
    return (ann.get("file_path") or ann.get("file_name")
            or ann.get("filename") or ann.get("wav") or ann.get("audio"))


def resolve_src_wav(ann, wav_root):
    fp = rel_wav_path(ann)
    if fp:
        p = Path(fp) if Path(fp).is_absolute() else (wav_root / fp)
        if p.exists():
            return p
        cand = list(wav_root.rglob(Path(fp).name))
        if cand:
            return cand[0]
    return None


def resample_one(args):
    src, dst = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1024:
        return ("skip", str(dst))
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
           "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not dst.exists():
            return ("fail", f"{src.name}: {r.stderr.strip()[:150]}")
        return ("ok", str(dst))
    except Exception as e:
        return ("fail", f"{src.name}: {type(e).__name__}")


def do_inspect(ann_dir, wav_root):
    print(f"=== inspecting {ann_dir} ===")
    njson = sum(1 for _ in ann_dir.rglob("*.json"))
    nwav = sum(1 for _ in wav_root.rglob("*.wav"))
    print(f"  jsons={njson}  wavs={nwav}")
    print("\n=== first annotation record ===")
    for i, ann in enumerate(iter_annotations(ann_dir)):
        printable = {k: (str(v)[:120]) for k, v in ann.items()}
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        canon = norm_emotion(ann.get('emotion_result'))
        print(f"  norm_emotion = {canon}  → class = {SCHEME4.get(canon)}")
        print(f"  rel_wav_path = {rel_wav_path(ann)}")
        print(f"  group_id = {find_group_id(rel_wav_path(ann))}")
        print(f"  resolved src wav = {resolve_src_wav(ann, wav_root)}")
        if i >= 1:
            break
    print("\n→ verify: emotion_result maps to 4 class, rel_wav_path resolves, group_id parses.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", type=Path, default=ANN_DIR)
    ap.add_argument("--wav_root", type=Path, default=WAV_ROOT)
    ap.add_argument("--inspect", action="store_true", help="print structure + one record, then exit")
    ap.add_argument("--skip_resample", action="store_true", help="rebuild CSV only, reuse existing 16k wavs")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no_traditional", action="store_true",
                    help="keep Simplified text (default: convert to Traditional s2twp to MATCH the game's STT output)")
    ap.add_argument("--val_groups", type=int, nargs="*", default=sorted(DEFAULT_VAL_GROUPS))
    ap.add_argument("--test_groups", type=int, nargs="*", default=sorted(DEFAULT_TEST_GROUPS))
    args = ap.parse_args()

    if not args.ann_dir.exists():
        sys.exit(f"ERROR: {args.ann_dir} not found.")

    if args.inspect:
        do_inspect(args.ann_dir, args.wav_root)
        return

    val_groups, test_groups = set(args.val_groups), set(args.test_groups)

    # #2 fix: align training text script with the game (STT 用 opencc s2twp 轉繁體 → 訓練也轉)
    cc = None
    if not args.no_traditional:
        try:
            from opencc import OpenCC
            cc = OpenCC("s2twp")
            print("Text → Traditional Chinese (opencc s2twp) — 對齊遊戲 STT 輸出")
        except Exception as e:
            print(f"⚠ opencc 不可用({e})→ 文字保持簡體(請改用有 opencc 的 venv 重跑)")

    rows, resample_jobs = [], []
    skipped = {"no_emotion": 0, "unmapped": 0, "no_wav": 0}
    class_counts = {c: 0 for c in CLASSES}
    split_counts = {}

    for ann in iter_annotations(args.ann_dir):
        canon = norm_emotion(ann.get("emotion_result"))
        if canon is None:
            skipped["no_emotion"] += 1
            continue
        cls = SCHEME4.get(canon)
        if cls is None:                       # unexpected emotion not in 4-class map
            skipped["unmapped"] += 1
            continue
        src = resolve_src_wav(ann, args.wav_root)
        fp = rel_wav_path(ann)
        if src is None or not fp:
            skipped["no_wav"] += 1
            continue

        rel = Path(fp).with_suffix(".wav")
        dst = OUT16K / rel
        resample_jobs.append((src, dst))

        gid = find_group_id(fp)
        split = "Development" if gid in val_groups else "Test" if gid in test_groups else "Train"

        onehot = {c: (1 if c == cls else 0) for c in CLASSES}
        class_counts[cls] += 1
        split_counts[split] = split_counts.get(split, 0) + 1
        text = (ann.get("content") or "").replace("\n", " ").strip()
        if cc is not None and text:
            text = cc.convert(text)
        rows.append({
            "FileName": str(rel),
            "Text": text,
            "Split_Set": split,
            **onehot,
        })

    print(f"Parsed {len(rows)} usable rows.  skipped={skipped}")
    if not rows:
        sys.exit("No rows produced — run --inspect and check field names / mapping.")

    if not args.skip_resample:
        print(f"Resampling {len(resample_jobs)} wav → 16 kHz mono "
              f"({args.workers} workers; already-done files are skipped) ...")
        stats = {"ok": 0, "skip": 0, "fail": 0}
        fails = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(resample_one, j) for j in resample_jobs]
            for n, fut in enumerate(as_completed(futs), 1):
                st, info = fut.result()
                stats[st] += 1
                if st == "fail":
                    fails.append(info)
                if n % 500 == 0 or n == len(futs):
                    print(f"  [{n}/{len(futs)}] {stats}", flush=True)
        if fails:
            print(f"⚠ {len(fails)} resample fails (first 10): {fails[:10]}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["FileName", "Text", "Split_Set"] + CLASSES)
        w.writeheader()
        w.writerows(rows)
    print(f"✅ wrote {len(rows)} rows → {OUT_CSV}")

    # inverse-frequency class weights on TRAIN split (ZH-only; bilingual merge recomputes its own)
    train_counts = {c: 0 for c in CLASSES}
    for r in rows:
        if r["Split_Set"] == "Train":
            for c in CLASSES:
                train_counts[c] += r[c]
    total = sum(train_counts.values()) or 1
    n = len(CLASSES)
    weights = {c: (total / (n * cnt) if cnt else 0.0) for c, cnt in train_counts.items()}
    OUT_WJSON.write_text(json.dumps({"class_weight": weights}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ wrote class weights → {OUT_WJSON}")

    print("\nClass distribution (all splits):")
    tot = sum(class_counts.values())
    for c in CLASSES:
        print(f"  {c:>8}: {class_counts[c]:>5}  ({class_counts[c]/tot*100:4.1f}%)  train={train_counts[c]:>5}  weight={weights[c]:.3f}")
    print("Split distribution:", split_counts)


if __name__ == "__main__":
    main()
