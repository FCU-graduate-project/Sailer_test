"""Frozen NNIME Test evaluation for completed N3 phase-A best-Dev checkpoints.

No training/optimizer, no Test-based checkpoint selection, and no TTS evaluation.
Run --prepare once, then run --campaign inside the same memory-limited cgroup
used by run_n3_phase_a_safe.py. Each checkpoint is evaluated in its own process.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

from run_n3_phase_a_safe import (
    cgroup_path, run_child, sha, utc, verify_memory_limits, write_status,
)

CRAB = Path(__file__).resolve().parents[1]
TRAIN = CRAB / "experiments/n3_phase_a_long12_20260909"
CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
FILES = [
    "mode.json", "train_norm_stat.pkl", "final_ser.pt",
    "text_lora_adapter/adapter_model.safetensors",
    "text_lora_adapter/adapter_config.json",
    "audio_lora_adapter/adapter_model.safetensors",
    "audio_lora_adapter/adapter_config.json",
]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if r["Split_Set"] == "Test"]
    labels = []
    for row in rows:
        values = [float(row[c]) for c in CLASSES]
        if sorted(values) != [0.0, 0.0, 1.0]:
            raise RuntimeError("Unexpected Test label: " + repr(values))
        labels.append(values.index(1.0))
        if not Path(row["FileName"]).is_file():
            raise FileNotFoundError(row["FileName"])
    if len(rows) != 638 or [labels.count(i) for i in range(3)] != [181, 165, 292]:
        raise RuntimeError("NNIME Test support mismatch")
    if len({r["FileName"] for r in rows}) != 638:
        raise RuntimeError("Duplicate Test audio paths")
    return rows, labels


def prepare(root):
    if (root / "evaluation.json").exists():
        raise RuntimeError("Evaluation already prepared; refusing overwrite")
    campaign = json.loads((TRAIN / "campaign.json").read_text())
    for name in ["bin/run_n3_phase_a_safe.py", "src/data/dataset/dataset.py",
                 "src/models/ser.py", "data/nnime_crab_format.csv",
                 "data/nnime_class_weights.json"]:
        path = str(CRAB / name)
        if sha(path) != campaign["frozen_files_sha256"][path]:
            raise RuntimeError("Training protocol input changed: " + path)
    rows, _ = test_rows(CRAB / "data/nnime_crab_format.csv")
    arms = [{
        "id": "n3_reference",
        "model_dir": str(CRAB / "experiments/nnime_v1_warmstart_lora"),
        "expected_macro_f1": 0.5876700387,
        "wandb_run_id": None,
    }]
    for arm in campaign["arms"]:
        result = json.loads((Path(arm["output_dir"]) / "run_result.json").read_text())
        arms.append({
            "id": arm["experiment_id"], "model_dir": arm["output_dir"],
            "best_epoch": result["best_epoch"],
            "best_dev_macro_f1": result["best_dev_macro_f1"],
            "wandb_run_id": result["wandb_url"].rsplit("/", 1)[-1],
        })
        with Path(arm["training_csv"]).open(newline="", encoding="utf-8-sig") as stream:
            selected = [r for r in csv.DictReader(stream) if r["Split_Set"] == "Test"]
        columns = list(rows[0])
        if [{k: r[k] for k in columns} for r in selected] != rows:
            raise RuntimeError("Phase-A Test differs from original NNIME")
    paths = [Path(__file__).resolve(), CRAB / "bin/run_n3_phase_a_safe.py",
             CRAB / "src/data/dataset/dataset.py", CRAB / "src/models/ser.py",
             CRAB / "data/nnime_crab_format.csv", CRAB / "data/nnime_class_weights.json"]
    for arm in arms:
        paths.extend(Path(arm["model_dir"]) / name for name in FILES)
    frozen = {str(p): sha(p) for p in paths}
    audio = {r["FileName"]: sha(r["FileName"]) for r in rows}
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "created_at": utc(), "crab_root": str(CRAB), "python": campaign["python"],
        "wandb_entity": campaign["wandb_entity"], "wandb_project": campaign["wandb_project"],
        "arms": arms, "frozen_sha256": frozen, "test_audio_sha256": audio,
        "test_csv": str(CRAB / "data/nnime_crab_format.csv"),
        "protocol": {
            "split": "Test", "real_samples": 638, "tts_samples": 0,
            "class_order": CLASSES, "class_support": [181, 165, 292],
            "batch_size": 1, "num_workers": 0, "pin_memory": False,
            "audio_sr": 16000, "audio_max_seconds": 12, "text_max_len": 128,
            "precision": "float32", "seed": 42,
            "normalization": "each checkpoint saved Train statistics",
            "checkpoint_selection": "best Dev; frozen before Test",
            "training_performed": False, "provisional_tts_human_qc_pending": True,
            "wandb_logging": "append Test summary fields to original completed training runs",
        },
    }
    dump(root / "evaluation.json", config)
    print("Prepared 638 real Test rows; original N3 regression + 3 phase-A evaluations.")


def verify(config):
    for path, expected in {**config["frozen_sha256"], **config["test_audio_sha256"]}.items():
        if sha(path) != expected:
            raise RuntimeError("Frozen evaluation input changed: " + path)


def evaluate(root, config, arm):
    verify_memory_limits(cgroup_path())
    verify(config)
    out = root / arm["id"]
    out.mkdir(exist_ok=False)
    import random
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
        recall_score,
    )
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    sys.path.insert(0, str(CRAB))
    import src.models as net
    from src.data.dataset.dataset import LazyWavSet, TxtSet, load_norm_stat

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no unplanned CPU fallback")
    model_dir = Path(arm["model_dir"])
    mode = json.loads((model_dir / "mode.json").read_text())
    if mode["ft_mode"] != "lora" or mode["freeze_audio"] or mode["classes"] != CLASSES:
        raise RuntimeError("Unsupported checkpoint mode/classes")
    print("Loading best-Dev checkpoint: " + arm["id"], flush=True)
    ssl = PeftModel.from_pretrained(
        AutoModel.from_pretrained(mode["ssl_type"], local_files_only=True),
        model_dir / "audio_lora_adapter", is_trainable=False,
    ).to("cuda").eval()
    text = PeftModel.from_pretrained(
        AutoModel.from_pretrained(mode["text_model_path"], local_files_only=True),
        model_dir / "text_lora_adapter", is_trainable=False,
    ).to("cuda").eval()
    head = net.MultiModalEmotionClassifierDeep(
        features1_dim=ssl.config.hidden_size, features2_dim=text.config.hidden_size,
        fusion_hidden_dim=mode["fusion_hidden_dim"], num_emotions=3, dropout=0.5,
    ).to("cuda")
    head.load_state_dict(torch.load(
        model_dir / "final_ser.pt", map_location="cpu", weights_only=True), strict=True)
    head.eval()
    tokenizer = AutoTokenizer.from_pretrained(mode["text_model_path"], local_files_only=True)
    rows, labels = test_rows(Path(config["test_csv"]))
    mean, std = load_norm_stat(model_dir / "train_norm_stat.pkl")
    wavs = LazyWavSet([r["FileName"] for r in rows], wav_mean=mean, wav_std=std)
    texts = TxtSet([r.get("Text", "") or "" for r in rows], tokenizer, max_len=128)
    logits_list = []
    started = time.monotonic()
    with torch.no_grad():
        for index in range(len(rows)):
            wav, dur = wavs[index]
            if dur < 400:
                raise RuntimeError("Invalid audio length at " + rows[index]["FileName"])
            # Exactly the historical collate behavior for Test batch_size=1.
            audio = torch.zeros((1, dur), dtype=torch.float32)
            audio[0, :dur] = torch.tensor(wav[:dur])
            mask_audio = torch.ones_like(audio)
            ids, text_mask = texts[index]
            features = ssl(audio.to("cuda"), attention_mask=mask_audio.to("cuda")).last_hidden_state
            text_features = text(
                input_ids=ids.unsqueeze(0).to("cuda"),
                attention_mask=text_mask.unsqueeze(0).to("cuda"),
            ).last_hidden_state
            logits = head(features, text_features).float()
            if logits.shape != (1, 3) or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid prediction")
            logits_list.append(logits.cpu())
            if (index + 1) % 100 == 0 or index + 1 == len(rows):
                print(f'{arm["id"]}: evaluated {index + 1}/{len(rows)}', flush=True)
    logits = torch.cat(logits_list)
    predicted = logits.argmax(1).numpy()
    probabilities = logits.softmax(1).numpy()
    cm = confusion_matrix(labels, predicted, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1, 2], zero_division=0)
    weights = json.loads((CRAB / "data/nnime_class_weights.json").read_text())["class_weight"]
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor(labels), weight=torch.tensor([weights[c] for c in CLASSES])).item()
    result = {
        "experiment_id": arm["id"], "evaluated_at": utc(), "test_evaluated": True,
        "test_macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
        "test_WAR": float(accuracy_score(labels, predicted)),
        "test_UAR": float(recall_score(labels, predicted, average="macro", zero_division=0)),
        "test_loss": loss, "test_samples": len(rows),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                         "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(), "class_order": CLASSES,
        "test_N_to_U_rate": float(cm[2, 1] / cm[2].sum()),
        "test_E_to_U_rate": float(cm[0, 1] / cm[0].sum()),
        "elapsed_inference_seconds": time.monotonic() - started,
        "normalization": {"mean": float(mean), "std": float(std)},
        "model_dir": str(model_dir), "best_epoch": arm.get("best_epoch"),
        "best_dev_macro_f1": arm.get("best_dev_macro_f1"),
        "evaluation_manifest_sha256": sha(root / "evaluation.json"),
        "protocol": config["protocol"], "wandb_sync": "not_applicable",
    }
    if "expected_macro_f1" in arm:
        result["regression_passed"] = abs(
            result["test_macro_f1"] - arm["expected_macro_f1"]) < 0.0001
    with (out / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["FileName", "true_label", "predicted_label"] + ["p_" + c for c in CLASSES])
        for row, label, pred, prob in zip(rows, labels, predicted, probabilities):
            writer.writerow([row["FileName"], CLASSES[label], CLASSES[pred]] + prob.tolist())
    dump(out / "test_result.json", result)
    if result.get("regression_passed") is False:
        raise RuntimeError("Historical N3 Test failed regression; stop before phase-A evaluation")
    if arm["wandb_run_id"]:
        try:
            import wandb
            remote = wandb.Api(timeout=45).run(
                config["wandb_entity"] + "/" + config["wandb_project"] + "/" + arm["wandb_run_id"])
            old_dev = float(remote.summary["best_dev_macro_f1"])
            if abs(old_dev - arm["best_dev_macro_f1"]) > 1e-8:
                raise RuntimeError("W&B training run does not match local checkpoint provenance")
            fields = {k: v for k, v in result.items() if k.startswith("test_")}
            fields.update({
                "test_checkpoint_selection": "best_dev",
                "test_batch_size": 1, "test_tts_samples": 0,
                "test_evaluation_manifest_sha256": result["evaluation_manifest_sha256"],
                "test_confusion_matrix_counts": result["confusion_matrix"],
                "test_class_order": CLASSES, "test_evaluated_at": result["evaluated_at"],
            })
            for c in CLASSES:
                for metric, value in result["per_class"][c].items():
                    fields["test_" + c + "_" + metric] = value
            remote.summary.update(fields)
            result["wandb_sync"] = "complete"
            result["wandb_url"] = remote.url
        except Exception as exc:
            result["wandb_sync"] = "pending"
            result["wandb_sync_error"] = str(exc)
            print("W&B sync pending; local Test result is preserved: " + str(exc), flush=True)
        dump(out / "test_result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.prepare:
        prepare(root)
        return
    config = json.loads((root / "evaluation.json").read_text())
    cgroup = cgroup_path()
    verify_memory_limits(cgroup)
    if args.worker is not None:
        evaluate(root, config, config["arms"][args.worker])
        return
    verify(config)
    if (root / "status.json").exists():
        raise RuntimeError("Evaluation has already started; no automatic rerun or overwrite")
    completed = []
    try:
        for index, arm in enumerate(config["arms"]):
            write_status(root, {"state": "evaluating", "active": arm["id"], "completed": completed})
            run_child([config["python"], str(Path(__file__).resolve()),
                       "--campaign", str(root), "--worker", str(index)],
                      config, root, cgroup, arm["id"])
            result = json.loads((root / arm["id"] / "test_result.json").read_text())
            completed.append(result)
        pending = any(r["wandb_sync"] == "pending" for r in completed)
        write_status(root, {"state": "complete_sync_pending" if pending else "complete",
                            "completed": completed})
    except BaseException as exc:
        write_status(root, {"state": "stopped", "completed": completed,
                            "error": str(exc), "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
