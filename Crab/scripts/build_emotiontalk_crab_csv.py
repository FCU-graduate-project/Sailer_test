"""
Build a Crab-format 3-class CSV from EmotionTalk, using the SAME mapping as the
English Crab (src/prepare_interview_scheme1.py):

    Excited        = Happy + Surprise
    Unconfident    = Fear  + Sad         <- the class CH-SIMS could not supply
    Neutral_3Class = Neutral
    (dropped)      = Anger, Disgust

Output matches train_crab_lora.py / load_cat_emo_label expectations:
    FileName    wav path relative to --wav_base_dir (the 16 kHz resampled dir)
    Text        transcript (content)
    Split_Set   Train / Development / Test
    <one-hot>   Excited, Unconfident, Neutral_3Class

Also resamples EmotionTalk's 44.1 kHz wav → 16 kHz mono (Crab pipeline) and
emits an inverse-frequency class-weight JSON (train split only).

⚠️ The internal tar layout is verified at runtime — ALWAYS run --inspect first:
    python scripts/build_emotiontalk_crab_csv.py --inspect
then build:
    python scripts/build_emotiontalk_crab_csv.py

Train command afterwards (3-class, real Crab head):
    bin/train_crab_lora.py --classes_list Excited Unconfident Neutral_3Class \\
      --df_path data/emotiontalk_crab_format.csv \\
      --weights_json data/emotiontalk_class_weights.json \\
      --wav_base_dir datasets/emotiontalk/Audio16k ...
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
TEXT_DIR   = ROOT / "Text" / "Text"     # extracted Text.tar
OUT16K     = ROOT / "Audio16k"          # resampled target (wav_base_dir for training)
OUT_CSV   = Path("/home/brant/Project/SAILER_test/Crab/data/emotiontalk_crab_format.csv")
OUT_WJSON = Path("/home/brant/Project/SAILER_test/Crab/data/emotiontalk_class_weights.json")
FFMPEG    = "/home/brant/bin/ffmpeg"

CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]

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
# scheme1: canonical 7 → 3-class (None = drop)
SCHEME1 = {
    "happy": "Excited", "surprise": "Excited",
    "fear": "Unconfident", "sad": "Unconfident",
    "neutral": "Neutral_3Class",
    "anger": None, "disgust": None,
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
    """
    Yield dicts from the extracted annotation dir. Handles:
      - per-sample {key}.json files (webdataset style)
      - a single *.jsonl manifest
      - a single *.json list
    Each yielded dict is the raw annotation (expects keys like emotion_result,
    content, file_name, speaker_id — verified via --inspect).
    """
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

    # one big json list vs many per-sample jsons
    if len(jsons) == 1:
        obj = json.loads(jsons[0].read_text(encoding="utf-8"))
        if isinstance(obj, list):
            yield from obj
            return
        if isinstance(obj, dict):
            # dict-of-records or single record
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
            # remember where it came from so we can find the sibling wav
            obj.setdefault("_json_path", str(jp))
            yield obj


def rel_wav_path(ann):
    """Relative wav path from the annotation (EmotionTalk uses 'file_path')."""
    return (ann.get("file_path") or ann.get("file_name")
            or ann.get("filename") or ann.get("wav") or ann.get("audio"))


def resolve_src_wav(ann, wav_root):
    """Find the source 44.1kHz wav: wav_root / file_path."""
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
        print(f"  norm_emotion(emotion_result) = {norm_emotion(ann.get('emotion_result'))}")
        print(f"  rel_wav_path = {rel_wav_path(ann)}")
        print(f"  group_id = {find_group_id(rel_wav_path(ann))}")
        print(f"  resolved src wav = {resolve_src_wav(ann, wav_root)}")
        if i >= 1:
            break
    print("\n→ verify: emotion_result maps, rel_wav_path resolves, group_id parses.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", type=Path, default=ANN_DIR,
                    help="dir with annotation jsons (emotion_result + content + file_path)")
    ap.add_argument("--wav_root", type=Path, default=WAV_ROOT,
                    help="root holding the 44.1kHz wavs (file_path is relative to this)")
    ap.add_argument("--inspect", action="store_true", help="print structure + one record, then exit")
    ap.add_argument("--skip_resample", action="store_true", help="rebuild CSV only, reuse existing 16k wavs")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--val_groups", type=int, nargs="*", default=sorted(DEFAULT_VAL_GROUPS))
    ap.add_argument("--test_groups", type=int, nargs="*", default=sorted(DEFAULT_TEST_GROUPS))
    args = ap.parse_args()

    if not args.ann_dir.exists():
        sys.exit(f"ERROR: {args.ann_dir} not found — run download_emotiontalk.py --extract first.")

    if args.inspect:
        do_inspect(args.ann_dir, args.wav_root)
        return

    val_groups, test_groups = set(args.val_groups), set(args.test_groups)
    rows, resample_jobs = [], []
    skipped = {"no_emotion": 0, "dropped": 0, "no_wav": 0}
    class_counts = {c: 0 for c in CLASSES}
    split_counts = {}

    for ann in iter_annotations(args.ann_dir):
        canon = norm_emotion(ann.get("emotion_result"))
        if canon is None:
            skipped["no_emotion"] += 1
            continue
        cls = SCHEME1.get(canon)
        if cls is None:                       # anger / disgust → drop
            skipped["dropped"] += 1
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
        rows.append({
            "FileName": str(rel),
            "Text": (ann.get("content") or "").replace("\n", " ").strip(),
            "Split_Set": split,
            **onehot,
        })

    print(f"Parsed {len(rows)} usable rows.  skipped={skipped}")
    if not rows:
        sys.exit("No rows produced — run --inspect and check field names / mapping.")

    if not args.skip_resample:
        print(f"Resampling {len(resample_jobs)} wav → 16 kHz mono ({args.workers} workers) ...")
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

    # inverse-frequency class weights on TRAIN split
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
        print(f"  {c:>15}: {class_counts[c]:>5}  ({class_counts[c]/tot*100:4.1f}%)  weight={weights[c]:.3f}")
    print("Split distribution:", split_counts)


if __name__ == "__main__":
    main()
