"""Build provisional Match-min1 TTS candidates from the exact frozen A parents.

No TTS generation, no new ASR inference, no training, no source overwrites.
Natural punctuation/pause/ASR-token boundaries only; all emotion/transcript QC
remains pending. Chinese ASR tokens can be single characters, so token boundaries
are explicitly NOT certified linguistic word boundaries.
"""
import argparse
import csv
import difflib
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import sys
import unicodedata
import wave
from datetime import datetime, timezone

CRAB = Path(__file__).resolve().parents[1]
A_ROOT = CRAB / "experiments/n3_phase_a_long12_20260909"
GEMINI = Path("/home/brant/Project/gemini/dataset")
MAX12 = GEMINI / "interview_uncertain_500_v2_max12"
MATCH = GEMINI / "interview_uncertain_nnime_u_match_v1"
BINS = [(1, 2, "1-<2"), (2, 4, "2-<4"), (4, 6, "4-<6"),
        (6, 8, "6-<8"), (8, 10, "8-<10"), (10, 12.00001, "10-12")]
BASE_FIELDS = ["FileName", "Text", "Split_Set", "Excited", "Unconfident",
               "Neutral_3Class", "Language", "Source", "ParentID",
               "DurationSeconds", "DurationBin", "Prompt"]
QC_FIELDS = ["QCStatus", "HumanEmotionVerified", "HumanTranscriptVerified",
             "DatasetVersion", "TrainingApproved"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm(text):
    chars, raw = [], []
    for i, ch in enumerate(text):
        if ch.isspace() or unicodedata.category(ch)[0] in "PZS":
            continue
        for c in unicodedata.normalize("NFKC", ch).lower():
            chars.append(c)
            raw.append(i)
    return "".join(chars), raw


def wav_info(path):
    with wave.open(str(path), "rb") as f:
        return {"sr": f.getframerate(), "channels": f.getnchannels(),
                "width": f.getsampwidth(), "frames": f.getnframes(),
                "seconds": f.getnframes() / f.getframerate()}


def bin_index(seconds):
    return next((i for i, (low, high, _) in enumerate(BINS)
                 if low <= seconds < high), None)


def quotas(counts, n):
    ideal = [n * c / sum(counts) for c in counts]
    out = [math.floor(x) for x in ideal]
    for i in sorted(range(len(counts)), key=lambda i: (-(ideal[i] - out[i]), i))[:n - sum(out)]:
        out[i] += 1
    return out


def safety():
    from run_n3_phase_a_safe import cgroup_path, host_available
    path = cgroup_path()
    if (path / "memory.max").read_text().strip() != str(2 * 1024**3):
        raise RuntimeError("Dataset builder requires its separate 2 GiB memory cgroup")
    if (path / "memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("Dataset builder must not use swap")
    if host_available() < 12 * 1024**3:
        raise RuntimeError("Host RAM reserve below 12 GiB")
    if int((path / "memory.current").read_text()) > int(1.8 * 1024**3):
        raise RuntimeError("Dataset builder RAM reached 1.8 GiB")
    return path


def parent_options(parent, arow, hashes):
    record_path = MAX12 / "records" / (parent + ".json")
    record = json.loads(record_path.read_text())
    hashes[str(record_path)] = sha(record_path)
    source = Path(arow["FileName"])
    info = wav_info(source)
    hashes[str(source)] = sha(source)
    if info["channels"] != 1 or info["width"] != 2:
        raise RuntimeError("Unexpected A source WAV format")
    children = record["outputs"]
    child = next(x for x in children if x["file_name"] == source.name)
    if child["sha256"] != hashes[str(source)]:
        raise RuntimeError("Source child hash mismatch: " + parent)
    start = float(record["trim_start_seconds"])
    if record["action"] == "split_at_aligned_punctuation" and child != children[0]:
        start = float(record["cut_seconds"])
    end = start + info["seconds"]
    caches = [MATCH / "alignment_cache" / (parent + ".json"),
              MAX12 / "alignment_cache" / (parent + ".json")]
    cache = next((p for p in caches if p.is_file()), None)
    if cache is None:
        raise RuntimeError("No existing alignment for " + parent)
    hashes[str(cache)] = sha(cache)
    asr = json.loads(cache.read_text())
    source_text = record["source_text"]
    s, raw = norm(source_text)
    ac, char_tokens = [], []
    words = asr["words"]
    for index, word in enumerate(words):
        token, _ = norm(word["word"])
        ac.extend(token)
        char_tokens.extend([index] * len(token))
    matcher = difflib.SequenceMatcher(None, s, "".join(ac), autojunk=False)
    mapped = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapped[block.a + offset] = char_tokens[block.b + offset]
    token_chars = {}
    for si, wi in mapped.items():
        token_chars.setdefault(wi, []).append(si)
    valid = [wi for wi in sorted(token_chars)
             if float(words[wi]["start"]) >= start - 0.025
             and float(words[wi]["end"]) <= end + 0.025
             and float(words[wi]["end"]) > float(words[wi]["start"])]
    options = []
    for ai, wi in enumerate(valid):
        left = min(token_chars[wi])
        wstart = max(start, float(words[wi]["start"]))
        previous_end = float(words[wi - 1]["end"]) if wi > 0 else start
        gap_before = max(0, wstart - previous_end)
        t0 = max(start, wstart - min(0.04, gap_before / 2))
        punct_before = left == 0 or any(unicodedata.category(c).startswith("P")
            for c in source_text[raw[left - 1] + 1:raw[left]])
        for wj in valid[ai:]:
            right = max(token_chars[wj])
            if right <= left:
                continue
            wend = min(end, float(words[wj]["end"]))
            next_start = float(words[wj + 1]["start"]) if wj + 1 < len(words) else end
            gap_after = max(0, next_start - wend)
            t1 = min(end, wend + min(0.06, gap_after / 2))
            # Round once to the actual source sampling grid. Never force a duration.
            first = max(0, round((t0 - start) * info["sr"]))
            last = min(info["frames"], round((t1 - start) * info["sr"]))
            dur = (last - first) / info["sr"]
            b = bin_index(dur)
            if b is None:
                continue
            text = source_text[raw[left]:raw[right] + 1]
            punct_after = right == len(s) - 1 or any(unicodedata.category(c).startswith("P")
                for c in source_text[raw[right] + 1:raw[right + 1]])
            coverage = sum(i in mapped for i in range(left, right + 1)) / (right - left + 1)
            exact_inside = "".join(norm(w["word"])[0] for w in words[wi:wj + 1]) == s[left:right + 1]
            kind = ("punctuation_clause" if punct_before and punct_after else
                    "pause_bounded" if (punct_before or gap_before >= .12)
                    and (punct_after or gap_after >= .12) else "asr_token_window")
            score = {"punctuation_clause": 0.0, "pause_bounded": 0.6, "asr_token_window": 2.0}[kind]
            score += 8 * (1 - coverage) + (0 if exact_inside else 2.0)
            if right - left + 1 < 4:
                score += 1.5
            options.append({"text": text, "start_frame": first, "end_frame": last,
                            "seconds": dur, "bin_index": b, "boundary_type": kind,
                            "candidate_cost": score, "mapped_char_coverage": coverage,
                            "asr_source_exact_inside": exact_inside,
                            "source_char_start": left, "source_char_end": right + 1})
    if not options:
        raise RuntimeError("No >=1s aligned candidate inside A clip: " + parent)
    return {"parent_id": parent, "a_row": arow, "source": str(source), "info": info,
            "parent_source": str(GEMINI / "interview_uncertain_500_v2" / record["source_audio"]),
            "a_parent_offset_seconds": start, "alignment_cache": str(cache),
            "alignment_ratio": matcher.ratio(), "source_text": source_text,
            "options": options}


def pick(option_list, target, target_bin):
    def score(o):
        # Every candidate already respects ASR token endpoints and >=1s.
        # Prefer the target bin but record all departures; prefer stronger boundaries within it.
        return (0 if o["bin_index"] == target_bin else 8 + 2 * abs(o["bin_index"] - target_bin)) + (
            o["candidate_cost"] + .7 * abs(o["seconds"] - target) / max(target, 1))
    best = min(option_list, key=lambda o: (score(o), o["start_frame"], o["end_frame"]))
    return best, score(best)


def emit_audio(source, destination, option):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError("Audio output exists")
    with wave.open(source, "rb") as f:
        params = f.getparams()
        f.setpos(option["start_frame"])
        data = f.readframes(option["end_frame"] - option["start_frame"])
    with wave.open(str(destination), "wb") as f:
        f.setparams(params)
        f.writeframes(data)
    # Preserve PCM, sample rate and amplitude. No new fade, speed change or normalization.
    return wav_info(destination)


def make_visual(root, counts, summary):
    width, height = 920, 380
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font:14px sans-serif;fill:#253047}</style>',
           '<text x="25" y="28" style="font-size:20px">Match-min1: duration distributions (percent)</text>']
    series = [("NNIME Train U >=1s", [100*c/sum(counts) for c in counts], "#475569")]
    colors = ["#2563eb", "#10b981", "#f59e0b"]
    for (name, arm), color in zip(summary.items(), colors):
        series.append((name, [100*c/arm["n_tts"] for c in arm["actual_bin_counts"]], color))
    for i in range(6):
        x = 45 + i * 143
        svg.append(f'<text x="{x}" y="310">{html.escape(BINS[i][2])}</text>')
        for j, (_, vals, color) in enumerate(series):
            h = vals[i] * 5.3
            svg.append(f'<rect x="{x+j*27}" y="{290-h:.2f}" width="22" height="{h:.2f}" fill="{color}"/>')
            svg.append(f'<text x="{x+j*27}" y="{283-h:.2f}" style="font-size:10px">{vals[i]:.1f}</text>')
    for j, (name, _, color) in enumerate(series):
        x = 30 + j * 225
        svg.append(f'<rect x="{x}" y="343" width="13" height="13" fill="{color}"/>')
        svg.append(f'<text x="{x+18}" y="355">{html.escape(name)}</text>')
    svg.append("</svg>")
    (root / "duration_distribution.svg").write_text("\n".join(svg), encoding="utf-8")


def make_review(root, records):
    order = sorted(records, key=lambda r: hashlib.sha256(("review-20260911:"+r["parent_id"]).encode()).hexdigest())
    rows = []
    cards = []
    for i, rec in enumerate(order, 1):
        blind = f"R{i:03d}"
        rows.append({"review_id": blind, "parent_id": rec["parent_id"], "audio": rec["audio"],
                     "candidate_text": rec["text"], "b25": rec["in_u25"], "b50": rec["in_u50"],
                     "emotion": "", "emotion_confidence": "", "transcript_correct": "",
                     "corrected_text": "", "start_end_complete": "", "taiwan_accent_natural": "",
                     "accept": "", "notes": ""})
        # Display no intended emotion or model score. Text can be expanded after first listening.
        cards.append(f'<article data-b25="{str(rec["in_u25"]).lower()}"><h3>{blind} · {rec["actual_seconds"]:.3f} s</h3>'
                     f'<audio controls preload="none" src="{html.escape(rec["audio"])}"></audio>'
                     f'<details><summary>After audio-only judgment: proposed transcript</summary><p>{html.escape(rec["text"])}</p></details>'
                     '<p>Record emotion (E / U / N / unclear / mixed), confidence, cut quality and transcript in review_template.csv.</p></article>')
    write_csv(root / "review_template.csv", rows, list(rows[0]))
    page = ('<!doctype html><meta charset="utf-8"><title>Match-min1 candidate review</title>'
            '<style>body{font-family:sans-serif;max-width:880px;margin:32px auto}article{border-top:1px solid #ddd;padding:16px}audio{width:100%}</style>'
            '<h1>Candidate audio review</h1><p>Listen before expanding transcript. No human labels have been assigned. '
            'Do not equate uncertainty in your own judgment with the Unconfident emotion label.</p>'
            '<label><input type="checkbox" id="b25">Show B25 only</label>'
            + "".join(cards)
            + '<script>document.getElementById("b25").onchange=function(){document.querySelectorAll("article").forEach(x=>x.hidden=this.checked&&x.dataset.b25!=="true")}</script>')
    (root / "review.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise RuntimeError("New dataset version must not already exist")
    guard = safety()
    root.mkdir(parents=True)
    hashes = {str(Path(__file__).resolve()): sha(__file__)}
    progress = {"state": "building", "created_at": utc(), "training_approved": False}
    dump(root / "status.json", progress)
    try:
        nnime_path = CRAB / "data/nnime_crab_format.csv"
        hashes[str(nnime_path)] = sha(nnime_path)
        nnime = read_csv(nnime_path)
        reference = [[] for _ in BINS]
        excluded_under1 = 0
        for r in nnime:
            if r["Split_Set"] == "Train" and float(r["Unconfident"]) == 1:
                info = wav_info(r["FileName"])
                seconds = min(info["seconds"], 12.0)
                hashes[r["FileName"]] = sha(r["FileName"])
                if seconds < 1:
                    excluded_under1 += 1
                else:
                    reference[bin_index(seconds)].append(seconds)
        counts = [len(x) for x in reference]
        if counts != [203,211,54,20,10,12] or excluded_under1 != 118:
            raise RuntimeError("Unexpected NNIME Train U reference")
        for pool in reference:
            pool.sort()
        arm_rows, parent_sets = {}, {}
        for n in (25,50,75):
            path = A_ROOT / f"n3a{n}_train_dev_test.csv"
            hashes[str(path)] = sha(path)
            arm_rows[n] = read_csv(path)
            parent_sets[n] = {r["ParentID"] for r in arm_rows[n] if r.get("ParentID")}
        if not (len(parent_sets[25]) == 157 and len(parent_sets[50]) == 314
                and len(parent_sets[75]) == 471 and parent_sets[25] < parent_sets[50] < parent_sets[75]):
            raise RuntimeError("A parent sets are not the expected nested arms")
        if "utt_000279" in parent_sets[75] or "utt_000405" in parent_sets[75] or "utt_000111" not in parent_sets[75]:
            raise RuntimeError("A replacement provenance mismatch")
        frozen = json.loads((A_ROOT / "tts_audio_hashes.json").read_text())
        source_rows = {r["ParentID"]:r for r in arm_rows[75] if r.get("ParentID")}
        for parent, r in source_rows.items():
            if sha(r["FileName"]) != frozen[r["FileName"]]:
                raise RuntimeError("A frozen audio changed: " + parent)
        infos = {}
        for i,parent in enumerate(sorted(parent_sets[75]),1):
            safety()
            infos[parent] = parent_options(parent, source_rows[parent], hashes)
            if i % 50 == 0:
                print(f"aligned candidate windows prepared: {i}/471",flush=True)
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        selected = []
        cumulative = [0]*6
        previous_set = set()
        for n in (25,50,75):
            parents = sorted(parent_sets[n] - previous_set)
            target_counts = quotas(counts,len(parent_sets[n]))
            current_counts = [a-b for a,b in zip(target_counts,cumulative)]
            targets = []
            for bi,q in enumerate(current_counts):
                pool=reference[bi]
                for j in range(q):
                    target=pool[min(len(pool)-1,int((j+.5)*len(pool)/q))]
                    targets.append((bi,target))
            if len(targets) != len(parents):
                raise RuntimeError("Quota allocation mismatch")
            matrix=np.zeros((len(parents),len(targets)))
            best={}
            for pi,parent in enumerate(parents):
                safety()
                for ti,(bi,target) in enumerate(targets):
                    option,cost=pick(infos[parent]["options"],target,bi)
                    best[pi,ti]=option
                    matrix[pi,ti]=cost
            rr,cc=linear_sum_assignment(matrix)
            for pi,ti in zip(rr,cc):
                parent=parents[int(pi)]
                info=infos[parent]
                option=best[int(pi),int(ti)]
                bi,target=targets[int(ti)]
                path=root/"audio"/(parent+"_min1.wav")
                actual=emit_audio(info["source"],path,option)
                warnings=["emotion_and_transcript_not_human_verified"]
                if option["boundary_type"]=="asr_token_window":
                    warnings.append("token_boundaries_may_split_linguistic_words_or_clause")
                if not option["asr_source_exact_inside"]:
                    warnings.append("asr_source_text_disagreement_inside_window")
                if info["alignment_ratio"] < .9:
                    warnings.append("source_alignment_ratio_below_0.9")
                if option["bin_index"] != bi:
                    warnings.append("duration_bin_deviation_for_boundary_quality")
                rec={
                    "segment_id":parent+"_min1","parent_id":parent,
                    "audio":path.relative_to(root).as_posix(),"audio_sha256":sha(path),
                    "a_audio":info["source"],"a_audio_sha256":hashes[info["source"]],
                    "original_parent_audio":info["parent_source"],
                    "a_parent_offset_seconds":info["a_parent_offset_seconds"],
                    "start_seconds_in_a":option["start_frame"]/actual["sr"],
                    "end_seconds_in_a":option["end_frame"]/actual["sr"],
                    "start_seconds_in_parent":info["a_parent_offset_seconds"]+option["start_frame"]/actual["sr"],
                    "end_seconds_in_parent":info["a_parent_offset_seconds"]+option["end_frame"]/actual["sr"],
                    "actual_seconds":actual["seconds"],"target_seconds":target,
                    "duration_bin":BINS[option["bin_index"]][2],"target_bin":BINS[bi][2],
                    "sample_rate":actual["sr"],"channels":actual["channels"],"bits_per_sample":actual["width"]*8,
                    "text":option["text"],"text_method":"source_text_slice_mapped_to_cached_ASR_tokens",
                    "prompt":info["a_row"]["Prompt"],"intended_emotion":"Unconfident",
                    "in_u25":parent in parent_sets[25],"in_u50":parent in parent_sets[50],"in_u75":True,
                    "boundary_type":option["boundary_type"],"alignment_ratio":info["alignment_ratio"],
                    "asr_source_exact_inside":option["asr_source_exact_inside"],
                    "alignment_cache":info["alignment_cache"],"mapped_char_coverage":option["mapped_char_coverage"],
                    "human_emotion_verified":False,"human_transcript_verified":False,
                    "qc_status":"pending_human_review","needs_review":True,
                    "review_reasons":warnings,"training_approved":False,
                }
                selected.append(rec)
                dump(root/"records"/(rec["segment_id"]+".json"),rec)
            cumulative=target_counts
            previous_set=parent_sets[n]
            print(f"audio candidates emitted through U{n}: {len(selected)}",flush=True)
        selected.sort(key=lambda r:r["parent_id"])
        summary={}
        for n in (25,50,75):
            recs=[r for r in selected if r[f"in_u{n}"]]
            hist=[sum(r["duration_bin"]==b[2] for r in recs) for b in BINS]
            tts=[]
            for r in recs:
                tts.append({"FileName":str(root/r["audio"]),"Text":r["text"],"Split_Set":"Train",
                            "Excited":0,"Unconfident":1,"Neutral_3Class":0,"Language":"ZH",
                            "Source":"TTS","ParentID":r["parent_id"],"DurationSeconds":r["actual_seconds"],
                            "DurationBin":r["duration_bin"],"Prompt":r["prompt"],
                            "QCStatus":"pending_human_review","HumanEmotionVerified":False,
                            "HumanTranscriptVerified":False,"DatasetVersion":"match_min1_candidate_20260911",
                            "TrainingApproved":False})
            write_csv(root/"manifests"/f"u{n}_CANDIDATE_NOT_APPROVED.csv",tts,BASE_FIELDS+QC_FIELDS)
            real=[r for r in arm_rows[n] if not r.get("ParentID")]
            # Preserve the real rows and their order/values; include all three real splits.
            merged=real+tts
            write_csv(root/"training_csv"/f"nnime_plus_u{n}_CANDIDATE_NOT_APPROVED.csv",merged,BASE_FIELDS+QC_FIELDS)
            dur=[r["actual_seconds"] for r in recs]
            summary[f"B{n}-min1"]={
                "n_tts":len(recs),"n_unique_parents":len({r["parent_id"] for r in recs}),
                "target_bin_counts":quotas(counts,len(recs)),"actual_bin_counts":hist,
                "total_seconds":sum(dur),"mean_seconds":statistics.mean(dur),
                "min_seconds":min(dur),"max_seconds":max(dur),
                "pending_human_review":len(recs),
                "token_boundary_review":sum(r["boundary_type"]=="asr_token_window" for r in recs),
                "text_disagreement_review":sum(not r["asr_source_exact_inside"] for r in recs),
                "target_bin_deviations":sum(r["duration_bin"]!=r["target_bin"] for r in recs),
                "training_approved":False,
            }
        write_csv(root/"metadata.csv",selected,list(selected[0]))
        make_visual(root,counts,summary)
        make_review(root,selected)
        result={
            "state":"complete_candidates_pending_human_review","created_at":utc(),
            "policy":"Match-min1; NNIME Train U >=1s conditional distribution; boundary constrained",
            "nnime_train_u_total":628,"reference_ge1_count":sum(counts),"excluded_reference_under1":excluded_under1,
            "reference_bins":[b[2] for b in BINS],"reference_counts":counts,
            "real_nnime_rows_modified":False,"tts_split":"Train only",
            "n_tts_audio":len(selected),"arms":summary,"training_approved":False,
            "input_sha256":hashes,"dataset_builder_sha256":sha(__file__),
            "source_mutation":False,"ASR_or_TTS_API_called":False,
            "audio_processing":"sample-exact PCM subset of original A child; no resample, fade, speed or normalization",
            "safety":{"memory_max_bytes":int((guard/"memory.max").read_text()),
                      "memory_events":(guard/"memory.events").read_text()},
        }
        dump(root/"build_report.json",result)
        dump(root/"status.json",{k:v for k,v in result.items() if k!="input_sha256"})
        print(json.dumps({k:v for k,v in result.items() if k!="input_sha256"},ensure_ascii=False),flush=True)
    except BaseException as exc:
        dump(root/"status.json",{"state":"stopped","error":str(exc),"training_approved":False,"updated_at":utc()})
        raise


if __name__=="__main__":
    main()
