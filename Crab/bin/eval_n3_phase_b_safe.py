"""Prepare only B25/B50/B75 seed42 Test; reuse the frozen historical evaluator.

No training, new seed, A evaluation, or Test-based checkpoint selection.
Preparation is CPU-only; execution requires the existing 20-GiB memory cgroup.
Original training artifacts, candidate QC flags, and completed run_result stay intact.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

from eval_n3_phase_a_safe import CLASSES, FILES, dump, test_rows, verify
from run_n3_phase_a_safe import sha, utc

CRAB = Path(__file__).resolve().parents[1]
TRAIN = CRAB / "experiments/n3_phase_b_matchmin1_s42_20260912"
HISTORICAL = CRAB / "experiments/n3_phase_a_test_20260910/evaluation.json"
EVALUATOR = CRAB / "bin/eval_n3_phase_a_safe.py"
ENTITY = "d1249119-feng-chia-university"
PROJECT = "Crab_N3_experiment"
RUN_IDS = {25: "592jz8yi", 50: "2k381ytp", 75: "8m6vrmtm"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_arm(arm, result, completed):
    dose = arm["dose"]
    require(dose in RUN_IDS and arm["seed"] == 42, "Only B doses25/50/75 seed42 authorized")
    identifier = f"n3b{dose}s42"
    require(arm["experiment_id"] == identifier, "Wrong B experiment ID")
    require(Path(arm["output_dir"]).resolve() == TRAIN / identifier, "Wrong checkpoint directory")
    require(result["run_name"] == arm["run_name"], "Training run name mismatch")
    require(result["wandb_url"] ==
            f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{RUN_IDS[dose]}", "Wrong W&B destination")
    require(result["test_evaluated"] is False, "Expected original Dev-only training result")
    require(isinstance(result["best_epoch"], int) and 0 <= result["best_epoch"] < 10,
            "Invalid best Dev epoch")
    require(math.isfinite(result["best_dev_macro_f1"]) and
            0 <= result["best_dev_macro_f1"] <= 1, "Invalid Dev score")
    require(result["optimizer_updates"] == {25: 2010, 50: 2110, 75: 2210}[dose],
            "Unexpected training update count")
    require(completed["experiment_id"] == identifier and completed["actual_epochs"] == 10,
            "B training is not complete/full10")
    require(all(completed.get(k) == v for k, v in result.items()),
            "Status differs from original training result")


def prepare(root):
    require(root.parent == CRAB / "experiments" and root != TRAIN,
            "Output must be a new direct experiments subdirectory")
    require(not root.exists(), "Refusing existing evaluation directory")
    campaign = read(TRAIN / "campaign.json")
    status = read(TRAIN / "status.json")
    require(status["state"] == "complete_dev_only" and len(status["completed"]) == 3,
            "All three B training runs must be complete")
    require([(a["dose"], a["seed"]) for a in campaign["arms"]] ==
            [(25, 42), (50, 42), (75, 42)], "Unexpected B campaign matrix")
    require((campaign["wandb_entity"], campaign["wandb_project"]) == (ENTITY, PROJECT),
            "Unexpected W&B project")
    cli = campaign["common_cli"]
    require(cli["contrastive_weight"] == 2 and cli["epochs"] == 10 and
            cli["early_stop_patience"] == 0, "Wrong B training recipe")
    unit = subprocess.check_output(
        ["systemctl", "--user", "show", "crab-n3-b-matchmin1-s42-20260912.service",
         "-p", "ActiveState", "-p", "Result"], text=True)
    props = dict(line.split("=", 1) for line in unit.splitlines() if "=" in line)
    require(props.get("ActiveState") == "inactive" and props.get("Result") == "success",
            "B training service must be inactive/success")
    historical = read(HISTORICAL)
    common = ["bin/run_n3_phase_a_safe.py", "src/data/dataset/dataset.py",
              "src/models/ser.py", "data/nnime_crab_format.csv",
              "data/nnime_class_weights.json"]
    for relative in common:
        path = str(CRAB / relative)
        require(sha(path) == campaign["frozen_files_sha256"][path] ==
                historical["frozen_sha256"][path], "Training/evaluation protocol changed: " + path)
    require(sha(EVALUATOR) == historical["frozen_sha256"][str(EVALUATOR)],
            "Historical evaluator changed")
    test_csv = CRAB / "data/nnime_crab_format.csv"
    rows, _ = test_rows(test_csv)
    paths = {Path(__file__).resolve(), EVALUATOR, HISTORICAL,
             TRAIN / "campaign.json", TRAIN / "status.json"}
    paths.update(CRAB / relative for relative in common)
    paths.update((CRAB / "src/data/dataset").glob("*.py"))
    paths.update((CRAB / "src/models").glob("*.py"))
    arms = []
    for arm in campaign["arms"]:
        model_dir = Path(arm["output_dir"])
        result_path = model_dir / "run_result.json"
        result = read(result_path)
        completed = next(r for r in status["completed"]
                         if r["experiment_id"] == arm["experiment_id"])
        validate_arm(arm, result, completed)
        csv_path = Path(arm["training_csv"])
        require(sha(csv_path) == campaign["frozen_files_sha256"][str(csv_path)],
                "B training CSV changed")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            selected = [r for r in csv.DictReader(stream) if r["Split_Set"] == "Test"]
        require([{k: r[k] for k in rows[0]} for r in selected] == rows,
                "B Test is not the original real NNIME Test")
        require(all(not r.get("ParentID") and r.get("Source") != "TTS" for r in selected),
                "Synthetic Test row detected")
        paths.update([result_path, csv_path])
        paths.update(model_dir / name for name in FILES)
        arms.append({"id": arm["experiment_id"], "model_dir": str(model_dir),
                     "best_epoch": result["best_epoch"],
                     "best_dev_macro_f1": result["best_dev_macro_f1"],
                     "wandb_run_id": RUN_IDS[arm["dose"]]})
    audio = {r["FileName"]: sha(r["FileName"]) for r in rows}
    require(audio == historical["test_audio_sha256"], "Original real Test audio changed")
    config = {
        "created_at": utc(), "crab_root": str(CRAB), "python": campaign["python"],
        "wandb_entity": ENTITY, "wandb_project": PROJECT, "arms": arms,
        "frozen_sha256": {str(p): sha(p) for p in sorted(paths)},
        "test_audio_sha256": audio, "test_csv": str(test_csv),
        "authorization": "2026-09-14 B25/B50/B75 seed42 Test only; no retraining or A Test",
        "training_campaign": str(TRAIN), "dev_selected_b_candidate": "n3b75s42",
        "protocol": {
            "split": "Test", "real_samples": 638, "tts_samples": 0,
            "class_order": CLASSES, "class_support": [181, 165, 292],
            "batch_size": 1, "num_workers": 0, "pin_memory": False,
            "audio_sr": 16000, "audio_max_seconds": 12, "text_max_len": 128,
            "precision": "float32", "seed": 42,
            "normalization": "each checkpoint saved Train statistics",
            "checkpoint_selection": "best Dev; frozen before Test",
            "training_performed": False, "provisional_tts_human_qc_pending": True,
            "wandb_logging": "append Test summary fields to original completed B training runs",
            "comparison_limit": "No new full10 A Test; no paired A/B Test difference",
            "historical_test_already_seen": True,
        },
    }
    verify(config)
    root.mkdir(exist_ok=False)
    dump(root / "evaluation.json", config)
    print(json.dumps({"prepared": str(root), "arms": [a["id"] for a in arms],
                      "test_samples": 638, "frozen_files": len(paths),
                      "manifest_sha256": sha(root / "evaluation.json")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.prepare:
        prepare(root)
        return
    config = read(root / "evaluation.json")
    require([a["id"] for a in config["arms"]] ==
            ["n3b25s42", "n3b50s42", "n3b75s42"], "Only three B Test arms authorized")
    require((config["wandb_entity"], config["wandb_project"]) == (ENTITY, PROJECT),
            "Wrong W&B project")
    verify(config)
    # The unchanged evaluator checks cgroup limits and serializes isolated workers.
    os.execv(config["python"], [config["python"], str(EVALUATOR), "--campaign", str(root)])


if __name__ == "__main__":
    main()
