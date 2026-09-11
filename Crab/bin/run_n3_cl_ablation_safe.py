"""Run the approved N3 A25 stage-2 CL ablation under the existing RAM guard.

All four arms independently warm-start V1. Only contrastive_weight differs.
New lambda=2 is a reproducibility check; original A25 is never overwritten.
"""
import argparse
import copy
import json
from pathlib import Path

from run_n3_phase_a_safe import (
    cgroup_path, command_for, run_child, sha, utc,
    verify_files, verify_memory_limits, write_status,
)

CRAB = Path(__file__).resolve().parents[1]
ORIGINAL = CRAB / "experiments/n3_phase_a_long12_20260909"


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(root):
    if root.exists():
        raise RuntimeError("Refusing to overwrite an existing campaign: " + str(root))
    original = json.loads((ORIGINAL / "campaign.json").read_text())
    verify_files(original)
    baseline = json.loads((ORIGINAL / "n3a25/run_result.json").read_text())
    config = copy.deepcopy(original)
    config["created_at"] = utc()
    config["approval_notes"] = "User approved CL weights 0/1/2/3 on unchanged A25; TTS remains provisional Train-only."
    config["experiment_question"] = "A25 V1-warmstart stage2: CE versus CE+lambda*mean(five CL terms)"
    config["original_a25_result"] = baseline
    config["original_campaign_sha256"] = sha(ORIGINAL / "campaign.json")
    config["common_cli"]["wandb_group"] = "n3a25-cl0123-s42-20260911"
    config["arms"] = []
    root.mkdir(parents=True)
    template = json.loads((ORIGINAL / "n3a25_metadata.json").read_text())
    for weight in (0, 1, 2, 3):
        ident = f"n3a25cl{weight}"
        metadata = copy.deepcopy(template)
        metadata.update({
            "experiment_id": ident, "phase": "A-CL", "cl_weight": weight,
            "cl_definition": "stage2 CE + lambda * mean(audio_frame,text_frame,audio_pooled,text_pooled,fusion CL)",
            "cl_zero_scope": "disable stage2 CL only; V1 pretraining history unchanged",
            "lambda2_is_new_reproducibility_run": weight == 2,
            "parent_a25_wandb_url": baseline["wandb_url"],
            "created_at": utc(),
        })
        meta_path = root / (ident + "_metadata.json")
        dump(meta_path, metadata)
        config["arms"].append({
            "experiment_id": ident, "contrastive_weight": weight,
            "training_csv": str(ORIGINAL / "n3a25_train_dev_test.csv"),
            "metadata_json": str(meta_path),
            "run_name": f"n3a25-cl{weight}-s42-provisional-r1",
            "output_dir": str(root / ident), "tts_count": 157,
        })
        config["frozen_files_sha256"][str(meta_path)] = sha(meta_path)
    config["frozen_files_sha256"][str(Path(__file__).resolve())] = sha(__file__)
    config["frozen_files_sha256"][str(ORIGINAL / "n3a25/run_result.json")] = sha(
        ORIGINAL / "n3a25/run_result.json")
    dump(root / "campaign.json", config)
    print("Prepared CL 0/1/2/3, identical A25 data, separate outputs, no Test.")


def arm_command(config, arm, model_dir, smoke=False):
    current = copy.deepcopy(config)
    current["common_cli"]["contrastive_weight"] = arm["contrastive_weight"]
    return command_for(current, arm, model_dir, smoke=smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.prepare:
        prepare(root)
        return
    config = json.loads((root / "campaign.json").read_text())
    cgroup = cgroup_path()
    verify_memory_limits(cgroup)
    verify_files(config)
    if args.check_only:
        print("CHECK PASS: inputs and memory isolation")
        return
    if (root / "status.json").exists():
        raise RuntimeError("Already started; no automatic retry/overwrite")
    completed = []
    try:
        write_status(root, {"state": "smoke_cl3", "completed": completed})
        run_child(arm_command(config, config["arms"][3], root / "smoke_cl3", smoke=True),
                  config, root, cgroup, "smoke_cl3")
        for arm in config["arms"]:
            verify_files(config)
            output = Path(arm["output_dir"])
            if output.exists():
                raise RuntimeError("Output already exists: " + str(output))
            write_status(root, {"state": "training", "active": arm["experiment_id"],
                                "active_cl_weight": arm["contrastive_weight"], "completed": completed})
            run_child(arm_command(config, arm, output), config, root, cgroup, arm["experiment_id"])
            result = json.loads((output / "run_result.json").read_text())
            if result["test_evaluated"]:
                raise RuntimeError("Unexpected Test evaluation")
            result.update({"experiment_id": arm["experiment_id"],
                           "contrastive_weight": arm["contrastive_weight"]})
            completed.append(result)
        write_status(root, {"state": "complete_dev_only", "completed": completed,
                            "note": "No Test; lambda=0 only removes stage2 CL. All TTS provisional."})
    except BaseException as exc:
        write_status(root, {"state": "stopped", "completed": completed,
                            "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
