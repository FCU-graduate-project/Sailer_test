"""
Build NNIME → Crab format CSV.

Ground truth: Native Rater (2021 xlsx, Mandarin_Chinese_Version).
    6 raters E1-E6 per sentence, each can multi-label (colon separated).
    Aggregation: pool all rater atoms → majority scheme1 class.

Split (team_id anchored, prevents speaker leakage):
    Train:  teams {1-3, 5-16}    (15 teams)
    Dev:    teams {17, 18, 19}   (3 teams)
    Test:   teams {4, 20, 21, 22} (4 teams — 4B = 22A same person per README)

Filter: keep only Type == 'Speech' (drop Laugh/Sigh/Sobbing/etc).

Scheme1 mapping (62 labels → E/U/N/DROP):
    Excited:      快樂 喜樂 興奮 驚訝 期待 振作 鬥志 激動
    Unconfident:  fear cluster (恐懼 緊張 焦慮 擔心 心虛 急 壓力 ...)
                + sad cluster (傷心 低落 失望 愧疚 無奈 痛 痛苦)
                + low-conf   (尷尬 囧 害躁 低聲下氣)
    Neutral:      中性 放鬆 祥和 認真 嚴肅 想睡 無聊 其他
    DROP:         angry cluster / frustration / noise / ambig (下定決心 自信 etc)

Text: from sentence-level transcripts (traditional Chinese, no Whisper needed).

Output: data/nnime_crab_format.csv
    columns: FileName, Text, Split_Set, Excited, Unconfident, Neutral_3Class, Language
"""
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

NNIME_ROOT = Path("/home/brant/Project/SAILER_test/datasets/NNIME")
ANNOT_XLSX = NNIME_ROOT / "Emotion Annotation of Sentence-Level by Native Rater/Emotion Annotation of Sentence-Level by Native Rater/Temporary_Version_Updated_20210601/Mandarin_Chinese_Version/Emotion.xlsx"
AUDIO_ROOT = NNIME_ROOT / "Sentence Level/Sentence Level/Speech"
TRANS_ROOT = NNIME_ROOT / "Recordings of Sentence-Level Transcripts/Recordings of Sentence-Level Transcripts/transcripts_speech/20220424"
OUT_CSV = Path("/home/brant/Project/SAILER_test/Crab/data/nnime_crab_format.csv")

# 62-label → scheme1 mapping (v3, user-adjusted 2026-07-01)
# 變更 vs v1:+挫折/傻眼/懷疑/疑惑/莫名奇妙 → Unconfident
#            +自信 → Excited
#            -想睡/無聊 → DROP (原 Neutral)
#            關懷/感性/下定決心/語重心長 → DROP (原本就是 drop)
EXCITED = {"快樂", "喜樂", "興奮", "驚訝", "期待", "振作", "鬥志", "激動",
           "自信"}
UNCONFIDENT = {
    # fear cluster
    "恐懼", "緊張", "緊張，惶恐", "焦慮", "焦急", "著急", "擔心", "擔心，下決心",
    "憂心", "心虛", "不甘心", "心不甘", "急", "急澄清", "壓力",
    # sad cluster
    "傷心", "低落", "失望", "愧疚", "無奈", "痛", "痛苦",
    # low-confidence cluster
    "尷尬", "囧", "害躁", "低聲下氣",
    # v3 additions: frustration + negative surprise + uncertainty
    "挫折", "傻眼", "懷疑", "疑惑", "莫名奇妙",
}
NEUTRAL = {"中性", "放鬆", "祥和", "認真", "嚴肅", "其他"}
DROP_ANY = {  # angry / noise / ambig / weak-emotion
    "不以為然", "不屑", "不悅", "不耐煩", "埋怨", "煩", "煩躁", "生氣", "警告",
    "無法標記(雜訊，觀眾笑聲)",
    "下定決心", "感性", "語重心長", "關懷",
    "想睡", "無聊",
}

TRAIN_TEAMS = {1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
DEV_TEAMS = {17, 18, 19}
TEST_TEAMS = {4, 20, 21, 22}


def parse_team(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except Exception:
        return -1


def split_of(team_id: int) -> str:
    if team_id in TRAIN_TEAMS: return "Train"
    if team_id in DEV_TEAMS: return "Development"
    if team_id in TEST_TEAMS: return "Test"
    return "UNK"


def classify_row(rater_labels, drop_counter: Counter):
    """Given 6 rater strings (each may be 'a:b' multi-atom), return scheme1 class or None."""
    votes = Counter()
    for r_label in rater_labels:
        if pd.isna(r_label):
            continue
        for atom in str(r_label).split(":"):
            atom = atom.strip()
            if not atom:
                continue
            if atom in EXCITED: votes["E"] += 1
            elif atom in UNCONFIDENT: votes["U"] += 1
            elif atom in NEUTRAL: votes["N"] += 1
            elif atom in DROP_ANY: votes["DROP"] += 1
            else: drop_counter[f"UNKNOWN:{atom}"] += 1

    scheme_votes = votes["E"] + votes["U"] + votes["N"]
    drop_votes = votes["DROP"]
    if scheme_votes == 0:
        return None  # all drops or empty
    # if DROP dominates, drop this sample
    if drop_votes > scheme_votes:
        return None
    # top scheme1 class
    scheme_only = {k: v for k, v in votes.items() if k in ("E", "U", "N")}
    return max(scheme_only, key=scheme_only.get)


def load_transcript(name: str) -> str:
    """Load transcript for utterance 'name' from txt file."""
    p = TRANS_ROOT / f"{name}.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def main():
    print(f"[load] {ANNOT_XLSX}", flush=True)
    df = pd.read_excel(ANNOT_XLSX)
    print(f"       {len(df)} total rows", flush=True)

    # keep Speech only
    df = df[df["Type"] == "Speech"].reset_index(drop=True)
    print(f"[filter] Speech only: {len(df)} rows", flush=True)

    # classify each row
    drop_counter = Counter()
    rater_cols = ["E1", "E2", "E3", "E4", "E5", "E6"]

    df["scheme1"] = df[rater_cols].apply(lambda row: classify_row(row.tolist(), drop_counter), axis=1)

    # unknown labels report
    if drop_counter:
        print("\n=== Unknown / uncovered atoms (should be empty) ===")
        for k, n in drop_counter.most_common(20):
            print(f"  {k:30s} {n}")

    # attach team + split
    df["team"] = df["Name"].apply(parse_team)
    df["split"] = df["team"].apply(split_of)
    df = df[df["split"] != "UNK"].reset_index(drop=True)

    # attach transcript + wav path
    print("\n[join] loading transcripts + wav paths...", flush=True)
    df["Text"] = df["Name"].apply(load_transcript)
    df["FileName"] = df["Name"].apply(lambda n: str(AUDIO_ROOT / f"{n}.wav"))

    # verify wav exists
    missing_wav = 0
    for fp in df["FileName"]:
        if not Path(fp).exists():
            missing_wav += 1
    print(f"       missing wav files: {missing_wav}", flush=True)

    # keep only classified + non-empty text
    usable = df[df["scheme1"].isin(["E", "U", "N"])].copy()
    empty_text = (usable["Text"].str.strip() == "").sum()
    usable = usable[usable["Text"].str.strip() != ""].reset_index(drop=True)
    print(f"[final] usable: {len(usable)} rows (dropped {empty_text} with empty text)", flush=True)

    # build one-hot columns
    usable["Excited"] = (usable["scheme1"] == "E").astype(int)
    usable["Unconfident"] = (usable["scheme1"] == "U").astype(int)
    usable["Neutral_3Class"] = (usable["scheme1"] == "N").astype(int)
    usable["Split_Set"] = usable["split"]
    usable["Language"] = "ZH"

    out = usable[["FileName", "Text", "Split_Set", "Excited", "Unconfident",
                  "Neutral_3Class", "Language"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[write] {OUT_CSV}  {len(out)} rows", flush=True)

    # per-split, per-class
    print("\n=== Per split, per scheme1 class ===")
    for split in ["Train", "Development", "Test"]:
        sub = out[out.Split_Set == split]
        e = int((sub["Excited"] == 1).sum())
        u = int((sub["Unconfident"] == 1).sum())
        n = int((sub["Neutral_3Class"] == 1).sum())
        print(f"  {split:12s}  Excited={e:>5}  Unconfident={u:>5}  Neutral={n:>5}  total={len(sub):>5}")

    # sample
    print("\n=== Sample 6 rows ===")
    for _, r in out.sample(min(6, len(out)), random_state=42).iterrows():
        cls = "Excited" if r.Excited else ("Unconfident" if r.Unconfident else "Neutral")
        wav_short = r.FileName.split("/")[-1]
        text_short = str(r.Text)[:40]
        print(f"  {r.Split_Set:12s}  {cls:12s}  {wav_short:28s}  '{text_short}'")


if __name__ == "__main__":
    main()
