"""Match-min1 B25/B50/B75, seed42 only, CL2/full10, provisional Train-only.

The 2026-09-12 authorization permits exploratory training, NOT a human-QC claim.
Original candidate CSVs, records, review forms and audio are never edited.
"""
import argparse
from collections import Counter
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import wave

from run_n3_phase_a_safe import (
    cgroup_path, command_for, run_child, sha, utc, verify_files,
    verify_memory_limits, write_status,
)

CRAB = Path(__file__).resolve().parents[1]
A_ROOT = CRAB / "experiments/n3_phase_a_multiseed_20260911"
DATA_ROOT = CRAB / "experiments/datasets/nnime_u_match_min1_20260911"
A_UNIT = "crab-n3-a-multiseed-20260911.service"
GROUP = "n3-phaseB-matchmin1-cl2-s42-full10-unreviewed-20260912"
DOSES = (25, 50, 75)
COUNTS = (157, 314, 471)
BINS = ("1-<2", "2-<4", "4-<6", "6-<8", "8-<10", "10-12")
TARGETS = {25: [62, 65, 17, 6, 3, 4], 50: [125, 130, 33, 12, 6, 8], 75: [188, 195, 50, 18, 9, 11]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bin_for(seconds):
    require(math.isfinite(seconds) and 1 <= seconds <= 12, "Audio outside Match-min1 [1,12] seconds")
    return BINS[next((i for i, edge in enumerate((2, 4, 6, 8, 10)) if seconds < edge), 5)]


def verify_pcm(record, path):
    require(sha(path) == record["audio_sha256"], "B audio hash mismatch: " + str(path))
    require(sha(record["a_audio"]) == record["a_audio_sha256"], "A source audio changed")
    with wave.open(str(path), "rb") as child, wave.open(record["a_audio"], "rb") as source:
        for stream in (child, source):
            require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype())
                    == (24000, 1, 2, "NONE"), "Unexpected PCM format")
        start = round(record["start_seconds_in_a"] * 24000)
        end = round(record["end_seconds_in_a"] * 24000)
        require(0 <= start < end <= source.getnframes(), "Segment outside A child")
        require(end - start == child.getnframes(), "Frame-count mismatch")
        seconds = child.getnframes() / 24000
        require(abs(seconds - record["actual_seconds"]) < 1e-9, "Duration metadata mismatch")
        require(bin_for(seconds) == record["duration_bin"], "Duration bin mismatch")
        source.setpos(start)
        require(child.readframes(child.getnframes()) == source.readframes(end - start), "Not an exact A PCM subset")


def paired_rows(a_rows, b_rows, count):
    require(len(a_rows) == len(b_rows) == 4300 + count, "Wrong combined row count")
    a_real = [row for row in a_rows if not row["ParentID"]]
    b_real = [row for row in b_rows if not row["ParentID"]]
    require(len(a_real) == len(b_real) == 4300, "Wrong real row count")
    require(all(all(row.get(k) == value for k, value in a.items()) for a, row in zip(a_real, b_real)), "Real rows changed")
    a_tts = {row["ParentID"]: row for row in a_rows if row["ParentID"]}
    b_tts = {row["ParentID"]: row for row in b_rows if row["ParentID"]}
    require(len(a_tts) == len(b_tts) == count and a_tts.keys() == b_tts.keys(), "Parent sets differ or duplicate parents")
    for row in b_tts.values():
        require(row["Split_Set"] == "Train" and row["Source"] == "TTS", "Synthetic data outside Train")
        require([float(row[k]) for k in ("Excited", "Unconfident", "Neutral_3Class")] == [0, 1, 0], "Unexpected intended label")
        require(bool(row["Text"].strip()), "Empty segment transcript")
        require(all(row[k] == "False" for k in ("HumanEmotionVerified", "HumanTranscriptVerified", "TrainingApproved")), "Original QC flags changed")
        require(row["QCStatus"] == "pending_human_review", "Unexpected candidate QC status")
    require(Counter(row["Split_Set"] for row in b_rows) == {"Train": 3054 + count, "Development": 608, "Test": 638}, "Wrong split counts")
    # B50/B75 candidates were sorted by parent; retain A's exact row order for paired shuffling.
    real_iter = iter(b_real)
    return [b_tts[row["ParentID"]] if row["ParentID"] else next(real_iter) for row in a_rows]


def audit_data(a_config):
    report = read(DATA_ROOT / "build_report.json")
    hashes = dict(report["input_sha256"])
    for path, expected in hashes.items():
        require(sha(path) == expected, "Dataset build input changed: " + path)
    records = {}
    audio_hashes = {}
    for path in sorted((DATA_ROOT / "records").glob("*.json")):
        record = read(path)
        require(record["parent_id"] not in records, "Duplicate record parent")
        require(record["human_emotion_verified"] is False and record["human_transcript_verified"] is False
                and record["training_approved"] is False, "Do not overwrite historical QC claims")
        audio = (DATA_ROOT / record["audio"]).resolve()
        require(audio.parent == DATA_ROOT / "audio", "Audio outside frozen dataset")
        verify_pcm(record, audio)
        records[record["parent_id"]] = record
        audio_hashes[str(audio)] = record["audio_sha256"]
        hashes[str(path)] = sha(path)
    require(len(records) == len(audio_hashes) == 471, "Expected all 471 unique segments")
    arms = []
    parent_sets = []
    for dose, count, a_arm in zip(DOSES, COUNTS, a_config["arms"][:3]):
        require(a_arm["seed"] == 42 and a_arm["dose"] == dose, "Wrong A baseline arm")
        b_path = DATA_ROOT / f"training_csv/nnime_plus_u{dose}_CANDIDATE_NOT_APPROVED.csv"
        fields, b_rows = read_csv(b_path)
        _, a_rows = read_csv(a_arm["training_csv"])
        ordered = paired_rows(a_rows, b_rows, count)
        parents = {row["ParentID"] for row in ordered if row["ParentID"]}
        parent_sets.append(parents)
        durations = []
        boundaries = Counter()
        a_by_parent = {row["ParentID"]: row for row in a_rows if row["ParentID"]}
        for row in ordered:
            if not row["ParentID"]:
                continue
            record = records[row["ParentID"]]
            require(Path(row["FileName"]).resolve() == (DATA_ROOT / record["audio"]).resolve(), "B filename mismatch")
            require(record["a_audio"] == a_by_parent[row["ParentID"]]["FileName"], "Wrong paired A child")
            require(row["Text"] == record["text"] and row["Prompt"] == record["prompt"], "Text/prompt mismatch")
            require(abs(float(row["DurationSeconds"]) - record["actual_seconds"]) < 1e-9, "CSV duration mismatch")
            require(row["DurationBin"] == record["duration_bin"], "CSV bin mismatch")
            require(record[f"in_u{dose}"] is True, "Wrong dose membership")
            durations.append(record["actual_seconds"])
            boundaries[record["boundary_type"]] += 1
        counts = [sum(bin_for(s) == b for s in durations) for b in BINS]
        require(counts == TARGETS[dose], "Duration quota mismatch")
        hashes[str(b_path)] = sha(b_path)
        arms.append({"dose": dose, "count": count, "fields": fields, "rows": ordered,
                     "candidate_csv": str(b_path), "row_order_changed": ordered != b_rows,
                     "total_seconds": sum(durations), "mean_seconds": sum(durations) / count,
                     "bin_counts": counts, "boundary_counts": dict(boundaries)})
    require(parent_sets[0] < parent_sets[1] < parent_sets[2] == records.keys(), "Non-nested parent sets")
    for name in ("build_report.json", "independent_audit.json", "status.json"):
        path = DATA_ROOT / name
        hashes[str(path)] = sha(path)
    return arms, hashes, audio_hashes


def verify_predecessor():
    status = read(A_ROOT / "status.json")
    require(status.get("state") == "complete_dev_only" and len(status.get("completed", [])) == 9, "A multi-seed not complete")
    require({(r["seed"], r["dose"]) for r in status["completed"]} == {(s, d) for s in (42, 43, 44) for d in DOSES}, "Incomplete A matrix")
    for result in status["completed"]:
        actual = read(A_ROOT / result["experiment_id"] / "run_result.json")
        require(actual["test_evaluated"] is False and all(result[k] == v for k, v in actual.items()), "A raw result mismatch")
    properties = subprocess.check_output(["systemctl", "--user", "show", A_UNIT, "-p", "ActiveState", "-p", "Result"], text=True)
    props = dict(line.split("=", 1) for line in properties.splitlines() if "=" in line)
    require(props.get("ActiveState") == "inactive" and props.get("Result") == "success", "A service not inactive/success")
    return status


def prepare(root):
    require(not root.exists(), "Refusing existing campaign")
    a_config = read(A_ROOT / "campaign.json")
    verify_files(a_config)
    predecessor = verify_predecessor()
    audited, hashes, audio_hashes = audit_data(a_config)
    config = copy.deepcopy(a_config)
    config.update(created_at=utc(), approval_status="approved_provisional", arms=[],
                  approval_notes="2026-09-12 user explicitly approved B25/B50/B75 seed42 only without human review; no Test or extra seeds.",
                  experiment_question="Long12 versus Match-min1 construction strategy, paired dose/seed42; not pure duration causality",
                  predecessor_campaign=str(A_ROOT), predecessor_unit=A_UNIT,
                  execution_order="B25 -> B50 -> B75; seed42 only; independent V1 starts",
                  candidate_flags_policy="Original TrainingApproved=False retained as historical candidate state; this campaign separately authorizes provisional training; human flags remain False")
    config["common_cli"]["wandb_group"] = GROUP
    config["frozen_files_sha256"].update(hashes)
    root.mkdir(parents=True)
    dump(root / "predecessor_completion.json", predecessor)
    dump(root / "tts_audio_hashes.json", audio_hashes)
    config["tts_audio_sha256_manifest"] = str(root / "tts_audio_hashes.json")
    for info, a_arm in zip(audited, a_config["arms"][:3]):
        dose = info["dose"]
        ident = f"n3b{dose}s42"
        train_csv = root / f"{ident}_train_dev_test_provisional.csv"
        with train_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=info["fields"])
            writer.writeheader()
            writer.writerows(info["rows"])
        a_result_path = Path(a_arm["output_dir"]) / "run_result.json"
        a_result = read(a_result_path)
        metadata = read(a_arm["metadata_json"])
        for key in ("multiseed_protocol", "historical_seed42_reused", "text_max_tokens_audited_original"):
            metadata.pop(key, None)
        metadata.update(experiment_id=ident, phase="B-Match-min1", audio_policy="match_nnime_train_u_ge1s",
                        dataset_version="match_min1_20260911_unreviewed_s42_v1",
                        qc_status="provisional_no_human_segment_emotion_or_transcript_validation",
                        human_emotion_verified=False, human_transcript_verified=False,
                        exploratory_training_authorized=True, training_seed=42,
                        data_sampling_policy="same_A_parent_sets_and_row_order_no_resampling",
                        tts_seconds=info["total_seconds"], mean_tts_seconds=info["mean_seconds"],
                        tts_audio_fraction=info["total_seconds"] / (info["total_seconds"] + metadata["real_effective_seconds"]),
                        duration_bins=list(BINS), duration_bin_counts=info["bin_counts"], boundary_counts=info["boundary_counts"],
                        candidate_csv=info["candidate_csv"], candidate_rows_reordered_to_match_A=info["row_order_changed"],
                        candidate_training_approved_flag_preserved_false=True,
                        source_label_mapping="TTS intended U=1 inherited from parent; segment emotion NOT human verified",
                        paired_a_wandb_url=a_result["wandb_url"], paired_a_run_name=a_result["run_name"],
                        paired_a_result_sha256=sha(a_result_path), parent_campaign_sha256=sha(A_ROOT / "campaign.json"),
                        manifest_sha256=sha(train_csv), comparison_scope="construction_strategy_not_pure_duration", created_at=utc())
        meta_path = root / f"{ident}_metadata.json"
        dump(meta_path, metadata)
        config["arms"].append(dict(experiment_id=ident, dose=dose, seed=42, tts_count=info["count"],
                                   training_csv=str(train_csv), metadata_json=str(meta_path), output_dir=str(root / ident),
                                   run_name=f"n3b{dose}-matchmin1-cl2-s42-full10-unreviewed-v1",
                                   expected_optimizer_updates=a_arm["expected_optimizer_updates"],
                                   paired_a_result=str(a_result_path)))
        for path in (train_csv, meta_path, a_result_path):
            config["frozen_files_sha256"][str(path)] = sha(path)
    audit = {"created_at": utc(), "status": "pass", "human_qc": "not_performed", "segments_checked": 471,
             "build_inputs_checked": len(read(DATA_ROOT / "build_report.json")["input_sha256"]),
             "sample_exact_A_subsets": True, "source_files_modified": False,
             "arms": [{k: v for k, v in item.items() if k not in ("fields", "rows")} for item in audited]}
    dump(root / "preflight_audit.json", audit)
    for path in (A_ROOT / "campaign.json", A_ROOT / "status.json", root / "predecessor_completion.json",
                 root / "tts_audio_hashes.json", root / "preflight_audit.json", Path(__file__).resolve()):
        config["frozen_files_sha256"][str(path)] = sha(path)
    validate_config(config, root)
    dump(root / "campaign.json", config)
    print("PREPARED: three B seed42 arms, 471 exact PCM subsets; original QC files unchanged", flush=True)


def validate_config(config, root):
    original = read(A_ROOT / "campaign.json")
    require(config["common_cli"] == dict(original["common_cli"], wandb_group=GROUP), "Unexpected CLI changes")
    require(config["common_flags"] == original["common_flags"], "Unexpected flags/Test policy")
    require(config["approval_status"] == "approved_provisional", "Missing exploratory approval")
    for key in ("python", "crab_root", "wandb_entity", "wandb_project"):
        require(config[key] == original[key], "Changed runtime identity")
    require(config["tts_audio_sha256_manifest"] == str(root / "tts_audio_hashes.json"), "Wrong audio manifest")
    require([(a["dose"], a["seed"]) for a in config["arms"]] == [(d, 42) for d in DOSES], "Only B25/B50/B75 seed42 approved")
    for arm, count, a_arm in zip(config["arms"], COUNTS, original["arms"][:3]):
        ident = f'n3b{arm["dose"]}s42'
        require(arm["experiment_id"] == ident and arm["tts_count"] == count, "Wrong B arm/count")
        require(arm["output_dir"] == str(root / ident) and arm["metadata_json"] == str(root / f"{ident}_metadata.json")
                and arm["training_csv"] == str(root / f"{ident}_train_dev_test_provisional.csv"), "Unexpected output/input path")
        require(arm["paired_a_result"] == str(Path(a_arm["output_dir"]) / "run_result.json"), "Wrong A comparator")
        require(arm["run_name"] == f'n3b{arm["dose"]}-matchmin1-cl2-s42-full10-unreviewed-v1', "Wrong run name")
        require(arm["expected_optimizer_updates"] == ((3054 + count + 15) // 16) * 10, "Wrong full10 budget")
        metadata = read(arm["metadata_json"])
        require(metadata["human_emotion_verified"] is False and metadata["human_transcript_verified"] is False
                and metadata["exploratory_training_authorized"] is True, "False QC claim or missing approval")
        require(metadata["manifest_sha256"] == sha(arm["training_csv"]), "Training CSV hash mismatch")
        _, a_rows = read_csv(a_arm["training_csv"])
        _, b_rows = read_csv(arm["training_csv"])
        require(paired_rows(a_rows, b_rows, count) == b_rows, "Training row order not paired to A")


def validate_result(result, arm):
    require(result.get("test_evaluated") is False and result.get("run_name") == arm["run_name"]
            and result.get("optimizer_updates") == arm["expected_optimizer_updates"], "Unexpected Test, run identity, or incomplete full10")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-inputs-only", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.prepare:
        prepare(root)
        return
    config = read(root / "campaign.json")
    validate_config(config, root)
    verify_files(config)
    if args.check_inputs_only:
        print("CHECK PASS: only B seed42, fixed CL2/full10, paired rows and frozen inputs", flush=True)
        return
    require(not (root / "status.json").exists(), "Already started; no retry/resume/overwrite")
    cgroup = cgroup_path()
    verify_memory_limits(cgroup)
    completed = []
    try:
        verify_predecessor()
        smoke_dir = root / "smoke"
        require(not smoke_dir.exists(), "Existing smoke output")
        write_status(root, dict(state="smoke_test", completed=completed))
        run_child(command_for(config, config["arms"][2], smoke_dir, smoke=True), config, root, cgroup, "smoke_B75")
        for arm in config["arms"]:
            verify_files(config)
            output = Path(arm["output_dir"])
            require(not output.exists(), "Existing model output")
            write_status(root, dict(state="training", active=arm["experiment_id"], active_seed=42, active_dose=arm["dose"], completed=completed))
            run_child(command_for(config, arm, output), config, root, cgroup, arm["experiment_id"])
            result = read(output / "run_result.json")
            validate_result(result, arm)
            baseline = read(arm["paired_a_result"])
            completed.append(dict(result, experiment_id=arm["experiment_id"], dose=arm["dose"], seed=42, actual_epochs=10,
                                  paired_a_wandb_url=baseline["wandb_url"], paired_a_dev_macro_f1=baseline["best_dev_macro_f1"],
                                  b_minus_a_dev_macro_f1=result["best_dev_macro_f1"] - baseline["best_dev_macro_f1"]))
            write_status(root, dict(state="between_runs", completed=completed))
        dump(root / "paired_dev_comparison.json", dict(created_at=utc(), test_evaluated=False, human_qc="not_performed", results=completed))
        write_status(root, dict(state="complete_dev_only", completed=completed, note="Only 3 seed42 B runs; unreviewed; no Test; no extra seeds"))
    except BaseException as exc:
        write_status(root, dict(state="stopped", completed=completed, error=str(exc), error_type=type(exc).__name__))
        raise


if __name__ == "__main__":
    main()
