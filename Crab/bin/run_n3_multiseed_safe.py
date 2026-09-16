"""Approved Long12 A25/A50/A75 x seeds 42/43/44; full10, CL2, Dev only.

Reuses the frozen trainer and RAM guard without editing the active CL campaign.
Refuses overlap, changed inputs, existing outputs, incomplete CL, or Test use.
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess

from run_n3_phase_a_safe import (
    cgroup_path, command_for, run_child, sha, utc, verify_files,
    verify_memory_limits, write_status,
)

CRAB = Path(__file__).resolve().parents[1]
ORIGINAL = CRAB / "experiments/n3_phase_a_long12_20260909"
PREDECESSOR = CRAB / "experiments/n3_a25_cl_ablation_20260911"
CL_UNIT = "crab-n3-a25-cl-20260911.service"
GROUP = "n3-phaseA-long12-cl2-s424344-full10-20260911"
SEEDS = (42, 43, 44)
DOSES = (25, 50, 75)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(root):
    if root.exists():
        raise RuntimeError("Refusing to overwrite campaign: " + str(root))
    original = read(ORIGINAL / "campaign.json")
    verify_files(original)
    config = copy.deepcopy(original)
    config.update({
        "created_at": utc(),
        "approval_notes": "User approved on 2026-09-11: finish CL, report, then start planned A multi-seed; provisional TTS Train-only; no Test or B.",
        "experiment_question": "Dose stability across training seeds with fixed nested TTS parent sets",
        "predecessor_campaign": str(PREDECESSOR),
        "predecessor_unit": CL_UNIT,
        "execution_order": "seed-major: s42 A25/A50/A75, s43 A25/A50/A75, s44 A25/A50/A75",
        "arms": [],
    })
    config["common_cli"].update(early_stop_patience=0, wandb_group=GROUP)
    root.mkdir(parents=True)
    for seed in SEEDS:
        for dose, source in zip(DOSES, original["arms"]):
            ident = f"n3a{dose}s{seed}"
            metadata = read(Path(source["metadata_json"]))
            metadata.update({
                "experiment_id": ident, "phase": "A-multiseed",
                "multiseed_protocol": "fixed-data_CL2_full10_s424344_v1",
                "training_seed": seed, "data_sampling_policy": "original_A_parent_sets_frozen_no_resampling",
                "epoch_policy": "full10_no_early_stop_best_dev_checkpoint",
                "historical_seed42_reused": False,
                "parent_campaign_sha256": sha(ORIGINAL / "campaign.json"),
                "created_at": utc(),
            })
            meta_path = root / f"{ident}_metadata.json"
            dump(meta_path, metadata)
            arm = copy.deepcopy(source)
            arm.update(experiment_id=ident, seed=seed, dose=dose,
                       metadata_json=str(meta_path), output_dir=str(root / ident),
                       run_name=f"n3a{dose}-long12-cl2-s{seed}-full10-provisional-v1",
                       expected_optimizer_updates=((3054 + source["tts_count"] + 15) // 16) * 10)
            config["arms"].append(arm)
            config["frozen_files_sha256"][str(meta_path)] = sha(meta_path)
    for path in (ORIGINAL / "campaign.json", PREDECESSOR / "campaign.json", Path(__file__).resolve()):
        config["frozen_files_sha256"][str(path)] = sha(path)
    validate_config(config, root)
    dump(root / "campaign.json", config)
    print("Prepared 9 new runs: fixed A data, CL2, full10, no Test; CL completion required.", flush=True)


def validate_config(config, root):
    original = read(ORIGINAL / "campaign.json")
    expected_cli = dict(original["common_cli"], early_stop_patience=0, wandb_group=GROUP)
    if config["common_cli"] != expected_cli or config["common_flags"] != original["common_flags"]:
        raise RuntimeError("Unexpected CLI/flag changes from approved protocol")
    for key in ("python", "crab_root", "wandb_entity", "wandb_project", "tts_audio_sha256_manifest"):
        if config[key] != original[key]:
            raise RuntimeError("Unexpected configuration change: " + key)
    if config["predecessor_campaign"] != str(PREDECESSOR) or config["predecessor_unit"] != CL_UNIT:
        raise RuntimeError("Unexpected predecessor")
    expected_order = [(s, d) for s in SEEDS for d in DOSES]
    if [(a["seed"], a["dose"]) for a in config["arms"]] != expected_order:
        raise RuntimeError("Expected exactly nine seed-major arms")
    for arm in config["arms"]:
        source = original["arms"][DOSES.index(arm["dose"])]
        ident = f'n3a{arm["dose"]}s{arm["seed"]}'
        if (arm["experiment_id"] != ident or arm["output_dir"] != str(root / ident)
                or arm["metadata_json"] != str(root / f"{ident}_metadata.json")
                or arm["training_csv"] != source["training_csv"]
                or arm["tts_count"] != source["tts_count"]
                or arm["expected_optimizer_updates"] != ((3054 + source["tts_count"] + 15) // 16) * 10
                or arm["run_name"] != f'n3a{arm["dose"]}-long12-cl2-s{arm["seed"]}-full10-provisional-v1'):
            raise RuntimeError("Invalid arm: " + ident)
        metadata = read(Path(arm["metadata_json"]))
        if (metadata["training_seed"] != arm["seed"]
                or metadata["manifest_sha256"] != sha(arm["training_csv"])
                or metadata["n_tts_train"] != arm["tts_count"]):
            raise RuntimeError("Metadata mismatch: " + ident)


def verify_predecessor():
    status = read(PREDECESSOR / "status.json")
    if status.get("state") != "complete_dev_only":
        raise RuntimeError("CL campaign not successfully complete; refusing to start")
    results = status.get("completed", [])
    if [r.get("contrastive_weight") for r in results] != [0, 1, 2, 3]:
        raise RuntimeError("CL completion does not contain all four arms")
    for weight, result in enumerate(results):
        actual = read(PREDECESSOR / f"n3a25cl{weight}" / "run_result.json")
        if actual.get("test_evaluated") is not False or any(result.get(k) != v for k, v in actual.items()):
            raise RuntimeError("CL result mismatch or unexpected Test")
    properties = subprocess.check_output([
        "systemctl", "--user", "show", CL_UNIT, "-p", "ActiveState", "-p", "Result"
    ], text=True)
    values = dict(line.split("=", 1) for line in properties.splitlines() if "=" in line)
    if values.get("ActiveState") != "inactive" or values.get("Result") != "success":
        raise RuntimeError("CL service not inactive/success: " + repr(values))
    verify_files(read(PREDECESSOR / "campaign.json"))
    return status


def arm_command(config, arm, output, smoke=False):
    current = copy.deepcopy(config)
    current["common_cli"]["seed"] = arm["seed"]
    return command_for(current, arm, output, smoke=smoke)


def aggregate(completed):
    if len(completed) != 9 or {(r["seed"], r["dose"]) for r in completed} != {(s, d) for s in SEEDS for d in DOSES}:
        raise RuntimeError("Aggregation requires all nine unique runs")
    values = {(r["seed"], r["dose"]): r["best_dev_macro_f1"] for r in completed}
    return {
        "created_at": utc(), "metric": "best_dev_macro_f1", "test_evaluated": False,
        "std_definition": "sample standard deviation, ddof=1, across three training seeds",
        "limitations": "3 seeds; fixed TTS sample; provisional QC; no new real-only control; no Test",
        "by_dose": {str(d): {"seeds": list(SEEDS), "values": [values[s, d] for s in SEEDS],
                             "mean": statistics.mean(values[s, d] for s in SEEDS),
                             "sample_std": statistics.stdev(values[s, d] for s in SEEDS)} for d in DOSES},
        "paired_differences": {f"A{b}_minus_A{a}": {str(s): values[s, b] - values[s, a] for s in SEEDS}
                               for a, b in ((25, 50), (25, 75), (50, 75))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-inputs-only", action="store_true", help="CPU-only validation; no GPU or service launch")
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.prepare:
        prepare(root)
        return
    config = read(root / "campaign.json")
    validate_config(config, root)
    verify_files(config)
    if args.check_inputs_only:
        print("CHECK PASS: protocol, metadata, frozen source files and TTS audio", flush=True)
        return
    if (root / "status.json").exists():
        raise RuntimeError("Already started; no automatic retry/resume/overwrite")
    cgroup = cgroup_path()
    verify_memory_limits(cgroup)
    completed = []
    try:
        predecessor = verify_predecessor()
        dump(root / "predecessor_completion.json", predecessor)
        write_status(root, {"state": "smoke_test", "completed": completed})
        smoke_output = root / "smoke"
        if smoke_output.exists():
            raise RuntimeError("Smoke output already exists")
        run_child(arm_command(config, config["arms"][2], smoke_output, smoke=True), config, root, cgroup, "smoke_A75")
        for arm in config["arms"]:
            verify_files(config)
            output = Path(arm["output_dir"])
            if output.exists():
                raise RuntimeError("Output already exists: " + str(output))
            write_status(root, {"state": "training", "active": arm["experiment_id"],
                                "active_seed": arm["seed"], "active_dose": arm["dose"], "completed": completed})
            run_child(arm_command(config, arm, output), config, root, cgroup, arm["experiment_id"])
            result = read(output / "run_result.json")
            if (result.get("test_evaluated") is not False or result.get("run_name") != arm["run_name"]
                    or result.get("optimizer_updates") != arm["expected_optimizer_updates"]):
                raise RuntimeError("Unexpected Test, run identity, or incomplete full10 training")
            completed.append(dict(result, experiment_id=arm["experiment_id"], seed=arm["seed"], dose=arm["dose"], actual_epochs=10))
            write_status(root, {"state": "between_runs", "completed": completed})
        dump(root / "aggregate_dev.json", aggregate(completed))
        write_status(root, {"state": "complete_dev_only", "completed": completed, "note": "All 9 full10; no Test; provisional Train-only TTS"})
    except BaseException as exc:
        write_status(root, {"state": "stopped", "completed": completed, "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
