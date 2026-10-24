"""Dev-only (NNIME Development) evaluation of a best-Dev N3 checkpoint loaded from disk.

Replicates the trainer's Dev loader exactly (pandas parse, CSV order, batch 16,
zero padding + masks, FP32) so the confusion matrix belongs to the same Dev
score that selected the checkpoint. Never reads Test rows, never trains.
Run by run_n3_u75aug_safe.py inside the queue's memory-limited cgroup.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

from run_n3_phase_a_safe import cgroup_path, sha, utc, verify_memory_limits

CRAB = Path(__file__).resolve().parents[1]
CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
DEV_SUPPORT = [183, 190, 235]
DEV_BATCH = 16  # trainer: cur_bs * 4 with batch_size 16 / accumulation 4


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dev_rows(csv_path):
    """Same parse as the trainer (pandas, Split_Set == Development, file order)."""
    import pandas as pd
    frame = pd.read_csv(csv_path)
    dev = frame[frame["Split_Set"] == "Development"]
    require(len(dev) == 608 and (dev["Source"] == "NNIME").all() and dev["ParentID"].isna().all(),
            "Dev must be the 608 original NNIME rows")
    labels = dev[CLASSES].to_numpy()
    require(((labels == 0) | (labels == 1)).all() and (labels.sum(1) == 1).all(), "Bad Dev label")
    labels = labels.argmax(1)
    require([int((labels == i).sum()) for i in range(3)] == DEV_SUPPORT, "Dev support changed")
    paths = dev["FileName"].tolist()
    require(all(Path(p).is_file() for p in paths), "Missing Dev audio")
    return paths, dev["Text"].fillna("").to_numpy(), labels


def evaluate(root, config, target):
    verify_memory_limits(cgroup_path())
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score,
    )
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    sys.path.insert(0, str(CRAB))
    import src.models as net
    from src.data.dataset.dataset import LazyWavSet, TxtSet, load_norm_stat

    require(torch.cuda.is_available(), "CUDA unavailable; no unplanned CPU fallback")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    model_dir = Path(target["model_dir"])
    run_result = json.loads((model_dir / "run_result.json").read_text())
    require(run_result["test_evaluated"] is False, "Expected a Dev-only training result")
    expected = run_result["best_dev_macro_f1"]
    if "best_dev_macro_f1" in target:
        require(abs(target["best_dev_macro_f1"] - expected) < 1e-12, "Baseline result changed")
    out = root / "dev_eval" / target["id"]
    out.mkdir(parents=True, exist_ok=False)
    frozen = {str(model_dir / n): sha(model_dir / n) for n in (
        "mode.json", "train_norm_stat.pkl", "final_ser.pt", "run_result.json",
        "text_lora_adapter/adapter_model.safetensors", "audio_lora_adapter/adapter_model.safetensors")}
    frozen[target["training_csv"]] = sha(target["training_csv"])
    mode = json.loads((model_dir / "mode.json").read_text())
    require(mode["ft_mode"] == "lora" and not mode["freeze_audio"] and mode["classes"] == CLASSES,
            "Unsupported checkpoint mode/classes")
    print("Loading best-Dev checkpoint from disk: " + target["id"], flush=True)
    ssl = PeftModel.from_pretrained(
        AutoModel.from_pretrained(mode["ssl_type"], local_files_only=True),
        model_dir / "audio_lora_adapter", is_trainable=False).to("cuda").eval()
    text = PeftModel.from_pretrained(
        AutoModel.from_pretrained(mode["text_model_path"], local_files_only=True),
        model_dir / "text_lora_adapter", is_trainable=False).to("cuda").eval()
    head = net.MultiModalEmotionClassifierDeep(
        features1_dim=ssl.config.hidden_size, features2_dim=text.config.hidden_size,
        fusion_hidden_dim=mode["fusion_hidden_dim"], num_emotions=3, dropout=0.5).to("cuda")
    head.load_state_dict(torch.load(model_dir / "final_ser.pt", map_location="cpu",
                                    weights_only=True), strict=True)
    head.eval()
    tokenizer = AutoTokenizer.from_pretrained(mode["text_model_path"], local_files_only=True)
    paths, texts, labels = dev_rows(target["training_csv"])
    mean, std = load_norm_stat(model_dir / "train_norm_stat.pkl")
    wavs = LazyWavSet(paths, wav_mean=mean, wav_std=std)
    tokens = TxtSet(texts, tokenizer, max_len=128)
    logits_list = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(paths), DEV_BATCH):
            index = range(start, min(start + DEV_BATCH, len(paths)))
            items = [wavs[i] for i in index]
            longest = max(dur for _, dur in items)
            # Same as the trainer's collate_fn_bimodal.
            audio = torch.zeros((len(items), longest))
            mask = torch.zeros((len(items), longest))
            for row, (wav, dur) in enumerate(items):
                audio[row, :dur] = torch.tensor(wav[:dur])
                mask[row, :dur] = 1
            ids = torch.stack([tokens[i][0] for i in index])
            text_mask = torch.stack([tokens[i][1] for i in index])
            features = ssl(audio.to("cuda").float(), attention_mask=mask.to("cuda").float()).last_hidden_state
            text_features = text(input_ids=ids.to("cuda"),
                                 attention_mask=text_mask.to("cuda")).last_hidden_state
            logits = head(features, text_features).float()
            require(logits.shape == (len(items), 3) and torch.isfinite(logits).all(),
                    "Invalid prediction")
            logits_list.append(logits.cpu())
    logits = torch.cat(logits_list)
    predicted = logits.argmax(1).numpy()
    probabilities = logits.softmax(1).numpy()
    cm = confusion_matrix(labels, predicted, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1, 2], zero_division=0)
    macro = float(f1_score(labels, predicted, average="macro", zero_division=0))
    u = CLASSES.index("Unconfident")
    result = {
        "experiment_id": target["id"], "evaluated_at": utc(), "split": "Development",
        "test_evaluated": False, "dev_samples": len(paths),
        "dev_macro_f1": macro,
        "dev_weighted_f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
        "dev_WAR": float(accuracy_score(labels, predicted)),
        "dev_UAR": float(recall_score(labels, predicted, average="macro", zero_division=0)),
        "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                          "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": cm.tolist(), "class_order": CLASSES,
        "confusion_rows": "true label", "confusion_columns": "predicted label",
        "U_counts": {"predicted_U": int(cm[:, u].sum()), "TP": int(cm[u, u]),
                     "FP": int(cm[:, u].sum() - cm[u, u]), "FN": int(cm[u].sum() - cm[u, u]),
                     "U_to_E": int(cm[u, 0]), "U_to_N": int(cm[u, 2]),
                     "E_to_U": int(cm[0, u]), "N_to_U": int(cm[2, u])},
        "best_epoch_0_based": run_result["best_epoch"],
        "training_best_dev_macro_f1": expected,
        "dev_reproduction_abs_diff": abs(macro - expected),
        "dev_reproduction_passed": abs(macro - expected) < 1e-4,
        "elapsed_inference_seconds": time.monotonic() - started,
        "normalization": {"mean": float(mean), "std": float(std)},
        "model_dir": str(model_dir), "frozen_inputs_sha256": frozen,
        "protocol": {"batch_size": DEV_BATCH, "order": "CSV order, no shuffle",
                     "padding": "zeros + attention mask (trainer collate)", "precision": "float32",
                     "checkpoint": "best Dev weights reloaded from disk"},
        "wandb_sync": "not_applicable",
    }
    with (out / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["FileName", "true_label", "predicted_label"] + ["p_" + c for c in CLASSES])
        for path, label, pred, prob in zip(paths, labels, predicted, probabilities):
            writer.writerow([path, CLASSES[label], CLASSES[pred]] + prob.tolist())
    dump(out / "dev_result.json", result)
    if target["wandb_sync"]:
        try:
            import wandb
            run_id = run_result["wandb_url"].rstrip("/").rsplit("/", 1)[-1]
            remote = wandb.Api(timeout=45).run(
                f'{config["wandb_entity"]}/{config["wandb_project"]}/{run_id}')
            require(abs(float(remote.summary["best_dev_macro_f1"]) - expected) < 1e-8,
                    "W&B run does not match local checkpoint provenance")
            fields = {"dev_eval_macro_f1": macro, "dev_eval_reproduction_passed": result["dev_reproduction_passed"],
                      "dev_eval_confusion_matrix_counts": result["confusion_matrix"],
                      "dev_eval_class_order": CLASSES, "dev_eval_evaluated_at": result["evaluated_at"]}
            fields.update({"dev_eval_" + k: v for k, v in result["U_counts"].items()})
            for c in CLASSES:
                for metric, value in result["per_class"][c].items():
                    fields[f"dev_eval_{c}_{metric}"] = value
            remote.summary.update(fields)
            result["wandb_sync"] = "complete"
        except Exception as exc:
            result["wandb_sync"] = "pending"
            result["wandb_sync_error"] = str(exc)
            print("W&B sync pending; local Dev result preserved: " + str(exc), flush=True)
        dump(out / "dev_result.json", result)
    print(json.dumps({k: result[k] for k in ("experiment_id", "dev_macro_f1", "U_counts",
                                             "dev_reproduction_passed")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--check-data-only", action="store_true",
                        help="CPU-only: validate the Dev rows of the target CSV")
    args = parser.parse_args()
    root = args.campaign.resolve()
    config = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    target = next((t for t in config["dev_eval_targets"] if t["id"] == args.target), None)
    require(target is not None, "Unknown Dev evaluation target")
    if args.check_data_only:
        paths, _, labels = dev_rows(target["training_csv"])
        print(f"CHECK PASS: {target['id']} Dev rows {len(paths)}, support {DEV_SUPPORT}", flush=True)
        return
    evaluate(root, config, target)


if __name__ == "__main__":
    main()
