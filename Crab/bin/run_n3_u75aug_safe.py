"""Approved U75 Augmented v1 N3 pair: augmented vs no-noise control; full10, CL2, Dev only.

Same recipe as the full10 A multi-seed campaign (seed 42). The dataset is read
only; derived CSVs and audits live in the new campaign directory. Each arm is
followed by a Dev-only evaluation from its best checkpoint on disk; the
historical n3a75s42 checkpoint is Dev-evaluated read-only as the baseline.
Refuses changed inputs, existing outputs, Test use, or execution outside the queue.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import wave

from run_n3_phase_a_safe import (
    cgroup_path, command_for, run_child, sha, utc, verify_files,
    verify_memory_limits, write_status,
)

CRAB = Path(__file__).resolve().parents[1]
GEMINI = Path("/home/brant/Project/gemini")
DATASET = GEMINI / "dataset/interview_unconfident_u75_augmented_v1"
MANIFEST = DATASET / "manifests/u75.csv"
MANIFEST_SHA256 = "174adad76de03dcf24a5deb01816f2d96b78a6b68b209975a97a10718e5f07ca"
SOURCE = CRAB / "experiments/n3_phase_a_multiseed_20260911"
A75_CSV = CRAB / "experiments/n3_phase_a_long12_20260909/n3a75_train_dev_test.csv"
BASELINE = SOURCE / "n3a75s42"
BASELINE_RUN_ID = "4oj0yndt"
DEV_EVALUATOR = CRAB / "bin/eval_n3_dev_safe.py"
GROUP = "n3-u75aug-v1-cl2-s42-full10-20261012"
CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
COLUMNS = ["FileName", "Text", "Split_Set", "Excited", "Unconfident", "Neutral_3Class",
           "Language", "Source", "ParentID", "DurationSeconds", "DurationBin", "Prompt"]
N_REAL = {"Train": 3054, "Development": 608, "Test": 638}
EXPECTED_UPDATES = ((3054 + 471 + 15) // 16) * 10
ARMS = (
    {"experiment_id": "n3u75augv1s42", "source": "TTS_U75_AUG_V1", "noise": True,
     "run_name": "n3u75augv1-cl2-s42-full10-exploratory-v1"},
    {"experiment_id": "n3u75nonoisev1s42", "source": "TTS_U75_AUG_V1_NONOISE", "noise": False,
     "run_name": "n3u75nonoisev1-cl2-s42-full10-exploratory-v1"},
)
MODEL_FILES = [
    "mode.json", "train_norm_stat.pkl", "final_ser.pt", "run_result.json",
    "text_lora_adapter/adapter_model.safetensors", "text_lora_adapter/adapter_config.json",
    "audio_lora_adapter/adapter_model.safetensors", "audio_lora_adapter/adapter_config.json",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def wav_info(path):
    with wave.open(str(path)) as w:
        return {"sr": w.getframerate(), "channels": w.getnchannels(),
                "sample_width": w.getsampwidth(), "seconds": w.getnframes() / w.getframerate()}


def audit_dataset():
    """Read-only checks of the user-approved u75.csv; returns rows with provenance."""
    require(sha(MANIFEST) == MANIFEST_SHA256, "u75.csv changed since user review")
    rows = read_csv(MANIFEST)
    require(len(rows) == 471, "Expected 471 TTS rows")
    require(all(r["Split_Set"] == "Train" for r in rows), "TTS row outside Train")
    require(all((r["Excited"], r["Unconfident"], r["Neutral_3Class"]) == ("0", "1", "0")
                for r in rows), "TTS row is not Unconfident")
    require(all(r["Language"] == "ZH" for r in rows), "Unexpected language")
    require(len({r["ParentID"] for r in rows}) == 471, "Duplicate ParentID")
    provenance = {}
    with (DATASET / "provenance.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            provenance[str(DATASET / item["audio"])] = item
    audit = []
    for row in rows:
        path = Path(row["FileName"])
        require(path.resolve().is_relative_to(DATASET / "audio") and path.is_file(),
                "Audio outside dataset/audio or missing: " + row["FileName"])
        item = provenance[row["FileName"]]
        digest = sha(path)
        require(item["sha256"] == digest and item["parent_id"] == row["ParentID"],
                "Provenance mismatch: " + row["FileName"])
        info = wav_info(path)
        require((info["sr"], info["channels"], info["sample_width"]) == (24000, 1, 2)
                and info["seconds"] <= 12.0, "Unexpected audio format: " + row["FileName"])
        source = GEMINI / item["source_audio"]
        record = {"row": row, "condition": item["condition"], "human_qc": item["human_qc"],
                  "sha256": digest, "source_audio": str(source),
                  "noise_release_status": item.get("noise_release_status")}
        if "environment" in item["condition"]:
            require(source.is_file() and source.resolve().is_relative_to(GEMINI / "dataset"),
                    "Missing pre-noise source: " + str(source))
            src_info = wav_info(source)
            require(src_info == info, "Pre-noise source format/length differs: " + str(source))
            record["source_sha256"] = sha(source)
        audit.append(record)
    require(len({a["sha256"] for a in audit}) == 471, "Duplicate TTS audio")
    return audit


def real_rows():
    rows = read_csv(A75_CSV)
    real = rows[:4300]
    require(all(r["Source"] == "NNIME" and not r["ParentID"] for r in real) and
            all(r["Source"] != "NNIME" for r in rows[4300:]), "Unexpected A75 CSV layout")
    counts = {s: sum(r["Split_Set"] == s for r in real) for s in N_REAL}
    require(counts == N_REAL, "NNIME split counts changed: " + repr(counts))
    return real


def tts_rows(audit, arm):
    out = []
    for item in audit:
        row = dict(item["row"], Source=arm["source"])
        if not arm["noise"] and "environment" in item["condition"]:
            row["FileName"] = item["source_audio"]
        out.append({c: row[c] for c in COLUMNS})
    return out


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def check_pandas_equivalence(path):
    """The trainer parses with pandas; NNIME rows must parse identically to A75."""
    import pandas as pd
    new, old = pd.read_csv(path), pd.read_csv(A75_CSV)
    require(list(new.columns) == list(old.columns), "Column mismatch vs A75 CSV")
    require(new.iloc[:4300].equals(old.iloc[:4300]), "NNIME rows parse differently from A75")
    require(len(new) == 4771 and (new.iloc[4300:]["Split_Set"] == "Train").all(),
            "TTS rows must all be Train")


def prepare(root):
    require(root.parent == CRAB / "experiments" and not root.exists(),
            "Campaign must be a new direct experiments subdirectory")
    source = read(SOURCE / "campaign.json")
    verify_files(source)
    require(read(SOURCE / "status.json")["state"] == "complete_dev_only", "A campaign incomplete")
    baseline = read(BASELINE / "run_result.json")
    require(baseline["optimizer_updates"] == EXPECTED_UPDATES and
            baseline["test_evaluated"] is False and
            baseline["wandb_url"].endswith("/" + BASELINE_RUN_ID), "Unexpected A75 baseline")
    audit = audit_dataset()
    real = real_rows()
    root.mkdir()
    conditions, qc = {}, {}
    for item in audit:
        conditions[item["condition"]] = conditions.get(item["condition"], 0) + 1
        qc[item["human_qc"]] = qc.get(item["human_qc"], 0) + 1
    a75_parents = {r["ParentID"] for r in read_csv(A75_CSV)[4300:]}
    parents = {a["row"]["ParentID"] for a in audit}
    hashes, arms, dev_targets = {}, [], []
    for spec in ARMS:
        ident = spec["experiment_id"]
        tts = tts_rows(audit, spec)
        csv_path = root / f"{ident}_train_dev_test.csv"
        write_csv(csv_path, real + tts)
        check_pandas_equivalence(csv_path)
        for row in tts:
            hashes[row["FileName"]] = sha(row["FileName"])
        seconds = sum(wav_info(r["FileName"])["seconds"] for r in tts)
        metadata = {
            "experiment_id": ident, "phase": "U75-augmented-v1-exploratory",
            "result_status": "exploratory",
            "dataset_name": "interview_unconfident_u75_augmented_v1",
            "dataset_path": str(DATASET),
            "dataset_status": "research candidate; pending human QC and release review (README 2026-10-08)",
            "dataset_manifest": str(MANIFEST), "dataset_manifest_sha256": MANIFEST_SHA256,
            "dataset_readme_sha256": sha(DATASET / "README.md"),
            "dataset_provenance_sha256": sha(DATASET / "provenance.jsonl"),
            "dataset_validation_report_sha256": sha(DATASET / "validation_report.json"),
            "dataset_modified": False,
            "arm_role": ("augmented as released" if spec["noise"] else
                         "no-noise control: 141 environment rows use their pre-mix source audio"),
            "environment_noise_rows": 141 if spec["noise"] else 0,
            "condition_counts_in_manifest": conditions,
            "human_qc_counts": qc,
            "human_qc_pending_rows_in_manifest": qc.get("pending", 0),
            "noise_source_caveat": "57 MS-SNSD per-file mappings unresolved; 84 Freesound HQ-preview pending original check",
            "confounds_vs_a75": "fillers (70 re-TTS rows, rewritten text) + 1 parent swap" +
                                (" + environment noise (141 rows)" if spec["noise"] else ""),
            "parent_swap_vs_a75": {"added": sorted(parents - a75_parents),
                                   "removed": sorted(a75_parents - parents),
                                   "note": "A75 had replaced utt_000279 (low alignment/ASR mismatch); u75.csv keeps it, user-approved 2026-10-12"},
            "tts_source_label": spec["source"],
            "tts_purpose": "train_only_never_validation_or_test",
            "n_real_train": 3054, "n_tts_train": 471, "n_unique_tts_parents": 471,
            "tts_seconds": seconds, "tts_ratio_relative_to_real_u": 0.75, "ratio_denominator": 628,
            "tts_audio_sha256_manifest": str(root / "tts_audio_hashes.json"),
            "training_manifest_sha256": sha(csv_path),
            "training_seed": 42, "epoch_policy": "full10_no_early_stop_best_dev_checkpoint",
            "initialization_semantics": "same_V1_new_optimizer_no_resume",
            "normalization_policy": "recompute_each_mixed_train_historical_algorithm",
            "recipe_source_campaign": str(SOURCE / "campaign.json"),
            "recipe_source_campaign_sha256": sha(SOURCE / "campaign.json"),
            "baseline_experiment": "n3a75s42", "baseline_wandb_run": BASELINE_RUN_ID,
            "baseline_best_dev_macro_f1": baseline["best_dev_macro_f1"],
            "infrastructure_deviations_vs_a75": {
                "launcher": ["systemd-run CPUQuota=300% Nice=10", "shared train_queue CPUQuota=200%"],
                "num_workers": [0, 0], "pin_memory": [False, False]},
            "test_evaluation": "explicitly deferred",
            "created_at": utc(),
        }
        meta_path = root / f"{ident}_metadata.json"
        dump(meta_path, metadata)
        arms.append({"experiment_id": ident, "training_csv": str(csv_path),
                     "metadata_json": str(meta_path), "run_name": spec["run_name"],
                     "output_dir": str(root / ident), "tts_count": 471, "tts_seconds": seconds,
                     "seed": 42, "noise": spec["noise"],
                     "expected_optimizer_updates": EXPECTED_UPDATES})
        dev_targets.append({"id": ident, "model_dir": str(root / ident),
                            "training_csv": str(csv_path), "wandb_sync": True})
    dev_targets.append({"id": "n3a75s42", "model_dir": str(BASELINE),
                        "training_csv": str(A75_CSV), "wandb_sync": False,
                        "best_dev_macro_f1": baseline["best_dev_macro_f1"],
                        "wandb_run_id": BASELINE_RUN_ID})
    dump(root / "tts_audio_hashes.json", hashes)
    dump(root / "data_audit.json", {
        "created_at": utc(), "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA256,
        "n_rows": 471, "split": {"Train": 471}, "label": "Unconfident", "unique_parents": 471,
        "audio_format": "24 kHz mono 16-bit, <=12 s", "condition_counts": conditions,
        "human_qc_counts": qc, "provenance_sha256_match": 471,
        "overlap_with_a75_parents": len(parents & a75_parents),
        "pre_noise_sources_same_format_and_length": 141,
        "rows": [{"file": a["row"]["FileName"], "parent": a["row"]["ParentID"],
                  "condition": a["condition"], "human_qc": a["human_qc"], "sha256": a["sha256"],
                  "pre_noise_source": a["source_audio"] if "source_sha256" in a else None,
                  "pre_noise_sha256": a.get("source_sha256")} for a in audit],
    })
    config = copy.deepcopy(source)
    config.update({
        "created_at": utc(),
        "approval_status": "approved_provisional",
        "approval_notes": "User approved on 2026-10-12: U75 augmented v1 as-is (u75.csv) plus no-noise control; A75 full10 recipe; normal-priority shared queue; Dev only, Test later.",
        "experiment_question": "Exploratory: does environment noise in U75 augmented v1 change N3 Dev (aug vs no-noise control), and how both compare with A75",
        "execution_order": "smoke -> n3u75augv1s42 -> Dev eval -> n3u75nonoisev1s42 -> Dev eval -> n3a75s42 Dev eval (read-only)",
        "arms": arms, "dev_eval_targets": dev_targets,
        "tts_audio_sha256_manifest": str(root / "tts_audio_hashes.json"),
        "recipe_source_campaign": str(SOURCE / "campaign.json"),
    })
    for key in ("predecessor_campaign", "predecessor_unit"):
        config.pop(key, None)
    config["common_cli"]["wandb_group"] = GROUP
    keep = [str(CRAB / p) for p in ("bin/train_crab_lora.py", "bin/run_n3_phase_a_safe.py",
                                    "src/data/dataset/dataset.py", "src/models/ser.py",
                                    "data/nnime_crab_format.csv", "data/nnime_class_weights.json")]
    keep += [p for p in source["frozen_files_sha256"] if "strategyA_xlsr_xlmr_lora" in p]
    frozen = {p: source["frozen_files_sha256"][p] for p in keep}
    extra = [SOURCE / "campaign.json", A75_CSV, MANIFEST, DATASET / "provenance.jsonl",
             DATASET / "README.md", DATASET / "validation_report.json",
             Path(__file__).resolve(), DEV_EVALUATOR, root / "data_audit.json"]
    extra += [Path(a[k]) for a in arms for k in ("training_csv", "metadata_json")]
    extra += [BASELINE / name for name in MODEL_FILES]
    frozen.update({str(p): sha(p) for p in extra})
    config["frozen_files_sha256"] = frozen
    validate_config(config, root)
    dump(root / "campaign.json", config)
    print(json.dumps({"prepared": str(root), "arms": [a["experiment_id"] for a in arms],
                      "tts_audio_files": len(hashes), "frozen_files": len(frozen)}, ensure_ascii=False))


def validate_config(config, root):
    source = read(SOURCE / "campaign.json")
    require(config["common_cli"] == dict(source["common_cli"], wandb_group=GROUP),
            "CLI differs from the A75 full10 recipe")
    require(config["common_flags"] == source["common_flags"], "Flags differ from A75 recipe")
    require(config["common_cli"]["seed"] == 42 and config["common_cli"]["epochs"] == 10 and
            config["common_cli"]["early_stop_patience"] == 0 and
            config["common_cli"]["contrastive_weight"] == 2, "Unexpected recipe")
    require("--no_eval_test" in config["common_flags"], "Test must stay deferred")
    for key in ("python", "crab_root", "wandb_entity", "wandb_project"):
        require(config[key] == source[key], "Unexpected configuration change: " + key)
    require([a["experiment_id"] for a in config["arms"]] == [a["experiment_id"] for a in ARMS],
            "Expected exactly the augmented and no-noise arms")
    for arm in config["arms"]:
        ident = arm["experiment_id"]
        require(arm["output_dir"] == str(root / ident) and
                arm["training_csv"] == str(root / f"{ident}_train_dev_test.csv") and
                arm["expected_optimizer_updates"] == EXPECTED_UPDATES, "Invalid arm: " + ident)
        metadata = read(arm["metadata_json"])
        require(metadata["training_manifest_sha256"] == sha(arm["training_csv"]) and
                metadata["training_seed"] == 42, "Metadata mismatch: " + ident)


def dev_comparison(results):
    by_id = {r["experiment_id"]: r for r in results}
    out = {"created_at": utc(), "split": "Development", "test_evaluated": False,
           "status": "exploratory; single seed; data QC pending", "arms": by_id}
    pairs = {"environment_noise_effect": ("n3u75augv1s42", "n3u75nonoisev1s42"),
             "fillers_plus_parent_swap_vs_a75": ("n3u75nonoisev1s42", "n3a75s42"),
             "total_vs_a75": ("n3u75augv1s42", "n3a75s42")}
    out["differences"] = {}
    for name, (a, b) in pairs.items():
        if a in by_id and b in by_id:
            out["differences"][name] = {
                "minus": f"{a} - {b}",
                "dev_macro_f1": by_id[a]["dev_macro_f1"] - by_id[b]["dev_macro_f1"],
                **{f"{c}_f1": by_id[a]["per_class"][c]["f1"] - by_id[b]["per_class"][c]["f1"]
                   for c in CLASSES},
                "U_precision": by_id[a]["per_class"]["Unconfident"]["precision"] -
                               by_id[b]["per_class"]["Unconfident"]["precision"],
                "U_recall": by_id[a]["per_class"]["Unconfident"]["recall"] -
                            by_id[b]["per_class"]["Unconfident"]["recall"]}
    return out


def run_dev_eval(config, root, cgroup, target):
    run_child([config["python"], str(DEV_EVALUATOR), "--campaign", str(root), "--target", target],
              config, root, cgroup, "dev_eval_" + target)
    return read(root / "dev_eval" / target / "dev_result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-inputs-only", action="store_true",
                        help="CPU-only validation; no GPU or training")
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.prepare:
        prepare(root)
        return
    config = read(root / "campaign.json")
    validate_config(config, root)
    verify_files(config)
    if args.check_inputs_only:
        print("CHECK PASS: recipe, metadata, frozen files and TTS audio", flush=True)
        return
    require(os.environ.get("TQ_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
            "Must run inside the shared queue on GPU 0")
    require(not (root / "status.json").exists(), "Already started; no automatic retry/overwrite")
    cgroup = cgroup_path()
    verify_memory_limits(cgroup)
    completed, dev_results, dev_errors = [], [], {}
    try:
        write_status(root, {"state": "smoke_test", "completed": completed})
        require(not (root / "smoke").exists(), "Smoke output already exists")
        run_child(command_for(config, config["arms"][0], root / "smoke", smoke=True),
                  config, root, cgroup, "smoke_" + config["arms"][0]["experiment_id"])
        for arm in config["arms"]:
            verify_files(config)
            output = Path(arm["output_dir"])
            require(not output.exists(), "Output already exists: " + str(output))
            write_status(root, {"state": "training", "active": arm["experiment_id"],
                                "completed": completed, "dev_eval": dev_results})
            run_child(command_for(config, arm, output), config, root, cgroup, arm["experiment_id"])
            result = read(output / "run_result.json")
            require(result.get("test_evaluated") is False and result.get("run_name") == arm["run_name"]
                    and result.get("optimizer_updates") == arm["expected_optimizer_updates"],
                    "Unexpected Test, run identity, or incomplete full10 training")
            completed.append(dict(result, experiment_id=arm["experiment_id"], seed=42, actual_epochs=10))
            write_status(root, {"state": "dev_eval", "active": arm["experiment_id"],
                                "completed": completed, "dev_eval": dev_results})
            try:
                dev_results.append(run_dev_eval(config, root, cgroup, arm["experiment_id"]))
            except Exception as exc:  # training result stays valid; keep the queue slot moving
                dev_errors[arm["experiment_id"]] = str(exc)
        try:
            dev_results.append(run_dev_eval(config, root, cgroup, "n3a75s42"))
        except Exception as exc:
            dev_errors["n3a75s42"] = str(exc)
        dump(root / "dev_comparison.json", dev_comparison(dev_results))
        write_status(root, {"state": "complete_dev_only" if not dev_errors else "complete_dev_eval_errors",
                            "completed": completed, "dev_eval": dev_results, "dev_eval_errors": dev_errors,
                            "note": "Two full10 arms; no Test; exploratory U75 augmented v1"})
    except BaseException as exc:
        write_status(root, {"state": "stopped", "completed": completed, "dev_eval": dev_results,
                            "dev_eval_errors": dev_errors, "error": str(exc),
                            "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
