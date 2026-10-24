"""CPU-only analysis of EXISTING full10 A/B seed42 TEST predictions.

No torch, model inference, training, queue calls, W&B writes, or transcript export.
Dev is inventoried by WAV headers only; no Dev prediction is fabricated.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import wave

CRAB = Path(__file__).resolve().parents[1]
CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
BINS = ["<1", "1-2", "2-4", "4-8", "8-12"]
SOURCES = {"a": CRAB / "experiments/n3_full10_a_test_s42_20260927",
           "b": CRAB / "experiments/n3_phase_b_test_s42_20260914"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bucket(seconds):
    require(math.isfinite(seconds) and seconds > 0, "Invalid duration")
    return BINS[next((i for i, bound in enumerate((1, 2, 4, 8)) if seconds < bound), 4)]


def metrics(truth, predicted):
    require(len(truth) == len(predicted), "Prediction count mismatch")
    cm = [[0] * 3 for _ in range(3)]
    for actual, prediction in zip(truth, predicted):
        require(actual in range(3) and prediction in range(3), "Invalid label")
        cm[actual][prediction] += 1
    per_class = {}
    for i, label in enumerate(CLASSES):
        support = sum(cm[i]); selected = sum(row[i] for row in cm); tp = cm[i][i]
        per_class[label] = {
            "support": support, "predicted_count": selected, "tp": tp,
            "fp": selected - tp, "fn": support - tp,
            "precision": tp / selected if selected else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (support + selected) if support + selected else None,
        }
    complete = all(v["support"] > 0 for v in per_class.values())
    return {"n": len(truth), "confusion_matrix": cm, "per_class": per_class,
            "macro_f1": sum(v["f1"] for v in per_class.values()) / 3 if complete else None,
            "accuracy": sum(cm[i][i] for i in range(3)) / len(truth) if truth else None}


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def percentage(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def run(out):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicitly hide GPUs for CPU diagnostics")
    require(out.parent == CRAB / "experiments" and not out.exists(), "Require a new experiments directory")
    files = {str(Path(__file__).resolve()): sha(__file__)}
    csv_path = CRAB / "data/nnime_crab_format.csv"
    files[str(csv_path)] = sha(csv_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    inventory = []
    for row in source:
        if row["Split_Set"] not in ("Development", "Test"):
            continue
        labels = [float(row[c]) for c in CLASSES]
        require(sorted(labels) == [0, 0, 1], "Invalid reference label")
        with wave.open(row["FileName"], "rb") as audio:
            seconds = audio.getnframes() / audio.getframerate()
        inventory.append({"FileName": row["FileName"], "split": row["Split_Set"],
                          "true_label": CLASSES[labels.index(1)], "original_seconds": seconds,
                          "effective_seconds": min(seconds, 12), "length_bin": bucket(min(seconds, 12)),
                          "clipped_at_12_seconds": seconds > 12})
    test = [r for r in inventory if r["split"] == "Test"]
    dev = [r for r in inventory if r["split"] == "Development"]
    require(len(test) == 638 and len(dev) == 608, "Unexpected split counts")
    require(len({r["FileName"] for r in test + dev}) == 1246, "Duplicated or overlapping split paths")
    truth = [CLASSES.index(r["true_label"]) for r in test]
    predictions = {}; by_model = {}; audio_hashes = None
    for group, root in SOURCES.items():
        config_path = root / "evaluation.json"
        config = read_json(config_path)
        files[str(config_path)] = sha(config_path)
        require(config["protocol"]["split"] == "Test", "Source is not Test")
        require(config["frozen_sha256"][str(csv_path)] == files[str(csv_path)], "Reference CSV changed")
        if audio_hashes is None:
            audio_hashes = config["test_audio_sha256"]
        require(audio_hashes == config["test_audio_sha256"], "A/B Test audio differs")
        for dose in (25, 50, 75):
            name = f"n3{group}{dose}s42"
            prediction_path = root / name / "predictions.csv"
            result_path = root / name / "test_result.json"
            for path in (prediction_path, result_path):
                files[str(path)] = sha(path)
            with prediction_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            result = read_json(result_path)
            require(len(rows) == len(test), "Missing predictions")
            pred = []
            for row, reference in zip(rows, test):
                require(row["FileName"] == reference["FileName"] and
                        row["true_label"] == reference["true_label"], "Reference alignment mismatch")
                probs = [float(row["p_" + c]) for c in CLASSES]
                require(all(math.isfinite(p) and 0 <= p <= 1 for p in probs)
                        and abs(sum(probs) - 1) < 1e-6, "Invalid probabilities")
                selected = CLASSES.index(row["predicted_label"])
                require(selected == max(range(3), key=probs.__getitem__), "Argmax mismatch")
                pred.append(selected)
            full = metrics(truth, pred)
            require(full["confusion_matrix"] == result["confusion_matrix"] and
                    abs(full["macro_f1"] - result["test_macro_f1"]) < 1e-12,
                    "Recomputed Test metrics differ")
            lengths = {}
            for label in BINS:
                indices = [i for i, row in enumerate(test) if row["length_bin"] == label]
                lengths[label] = metrics([truth[i] for i in indices], [pred[i] for i in indices])
            predictions[name] = pred
            by_model[name] = {"all": full, "by_length": lengths}
    require(set(audio_hashes) == {r["FileName"] for r in test}, "Test audio set mismatch")
    for path, expected in audio_hashes.items():
        require(sha(path) == expected, "Test audio changed: " + path)
    pairwise = {}
    for dose in (25, 50, 75):
        a = predictions[f"n3a{dose}s42"]; b = predictions[f"n3b{dose}s42"]
        groups = {"all": list(range(len(truth)))}
        groups.update({c: [i for i, t in enumerate(truth) if t == j] for j, c in enumerate(CLASSES)})
        groups.update({"U_length_" + label: [i for i, t in enumerate(truth)
                       if t == 1 and test[i]["length_bin"] == label] for label in BINS})
        pairwise[str(dose)] = {}
        for label, indices in groups.items():
            pairwise[str(dose)][label] = {
                "n": len(indices),
                "both_correct": sum(a[i] == truth[i] and b[i] == truth[i] for i in indices),
                "a_only_correct": sum(a[i] == truth[i] and b[i] != truth[i] for i in indices),
                "b_only_correct": sum(a[i] != truth[i] and b[i] == truth[i] for i in indices),
                "both_wrong": sum(a[i] != truth[i] and b[i] != truth[i] for i in indices),
            }
    support = {}
    for split, rows in (("Development", dev), ("Test", test)):
        support[split] = {label: {c: sum(r["length_bin"] == label and r["true_label"] == c
                                        for r in rows) for c in CLASSES} for label in BINS}
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "analysis_split": "Test",
               "new_inference": False, "gpu_used": False, "wandb_writes": False,
               "dev_predictions_available": False, "dev_inventory_only": True,
               "bins_seconds": "[0,1),[1,2),[2,4),[4,8),[8,12]; effective=min(original,12)",
               "class_order": CLASSES, "length_support": support, "models": by_model,
               "paired_correctness": pairwise, "inputs_sha256": files,
               "test_audio_sha256": audio_hashes,
               "limitations": ["Single seed; descriptive existing Test analysis, not new validation",
                               "No causal conclusion about segment quality or duration",
                               "No significance testing; bins may have few or zero U samples",
                               "Length uses WAV headers and 12-second cap, not VAD speech duration",
                               "Macro-F1 omitted when any true class is absent in a bin"]}
    # Check source files once more before publishing derived outputs.
    require(all(sha(p) == h for p, h in files.items()), "Input changed during analysis")
    out.mkdir()
    dump(out / "summary.json", summary)
    write_csv(out / "dev_test_length_inventory.csv", inventory)
    comparison = []
    for i, row in enumerate(test):
        entry = dict(row)
        entry.update({name: CLASSES[pred[i]] for name, pred in predictions.items()})
        comparison.append(entry)
    write_csv(out / "test_paired_predictions.csv", comparison)
    table = []
    for name, model in by_model.items():
        for label, values in model["by_length"].items():
            u = values["per_class"][CLASSES[1]]
            table.append({"run": name, "split": "Test", "length_bin": label,
                          "n": values["n"], "E_support": values["per_class"][CLASSES[0]]["support"],
                          "U_support": u["support"], "N_support": values["per_class"][CLASSES[2]]["support"],
                          "macro_f1": values["macro_f1"], "U_precision": u["precision"],
                          "U_recall": u["recall"], "U_f1": u["f1"],
                          "U_to_E": values["confusion_matrix"][1][0],
                          "U_to_N": values["confusion_matrix"][1][2]})
    write_csv(out / "test_metrics_by_length.csv", table)
    report = ["# A/B seed42 錯誤分析：CPU-only、既有 Test", "",
              "本次不占 GPU、不重跑模型、不訓練。所有錯誤分析都是已有的 638 筆 **Test** 預測；Dev 608 筆只統計長度，沒有 Dev 逐筆結果。", "",
              "## 真人 Test 時長與三類支持筆數", "",
              "區間左含右不含，最後區間含12秒。使用原音訊時長上限12秒，不是VAD發聲長度。", "",
              "| 有效秒數 | E | U | N | 合計 |", "|---|---:|---:|---:|---:|"]
    for label, counts in support["Test"].items():
        report.append(f"| {label} | {counts[CLASSES[0]]} | {counts[CLASSES[1]]} | {counts[CLASSES[2]]} | {sum(counts.values())} |")
    report += ["", "## Test U 判讀：漏判與誤報", "",
               "| 模型 | 正確U | U→E | U→N | 非U→U誤報 | U precision | U recall | U F1 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
    order = [f"n3{g}{d}s42" for d in (25, 50, 75) for g in "ab"]
    for name in order:
        full = by_model[name]["all"]; u = full["per_class"][CLASSES[1]]; cm = full["confusion_matrix"]
        report.append(f"| {name} | {u['tp']} | {cm[1][0]} | {cm[1][2]} | {u['fp']} | {percentage(u['precision'])} | {percentage(u['recall'])} | {percentage(u['f1'])} |")
    report += ["", "## Test U recall：依有效長度", "",
               "![Test U recall heatmap](test_u_recall_by_length.png)", "",
               "| 模型 | " + " | ".join(BINS) + " |", "|---|" + "---:|" * len(BINS)]
    for name in order:
        report.append("| " + name + " | " + " | ".join(percentage(by_model[name]["by_length"][label]["per_class"][CLASSES[1]]["recall"]) for label in BINS) + " |")
    report += ["", "## 同一筆真人U：A/B正確集合的差異", "",
               "| 劑量 | 兩者皆正確 | 只有A正確 | 只有B正確 | 兩者皆錯 |", "|---|---:|---:|---:|---:|"]
    for dose, groups in pairwise.items():
        x = groups[CLASSES[1]]
        report.append(f"| {dose}% | {x['both_correct']} | {x['a_only_correct']} | {x['b_only_correct']} | {x['both_wrong']} |")
    report += ["", "![Test U prediction tradeoff](test_u_prediction_tradeoff.png)", "",
               "## 使用界線與下一步", "",
               "- 這是相同Test逐筆對照；不把Test排名當新的選模或調參依据。", 
               "- 小樣本區間先看U支持筆數；沒有U標為NA，不當作0%失敗。缺任一真實類別的區間不報三類Macro-F1。",
               "- 時長、說話者、文本與情緒難度可能同時不同；不能把差異直接歸因於切段或未人工驗收。",
               "- Dev重推仍待GPU：保留原Development順序、FP32、Train正規化、原Dev batch16及padding；不可拿Test batch1結果替代歷史Dev。",
               "- 本地研究產物；不含逐字稿、不上傳W&B、不commit／push。",
               "", "[機器可讀結果](summary.json)／[分長度指標](test_metrics_by_length.csv)／[逐筆對照](test_paired_predictions.csv)／[Dev與Test長度清單](dev_test_length_inventory.csv)", ""]
    (out / "README.md").write_text("\n".join(report), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    values = [[by_model[name]["by_length"][label]["per_class"][CLASSES[1]]["recall"] for label in BINS] for name in order]
    matrix = np.array([[np.nan if x is None else x * 100 for x in row] for row in values])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colored = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(order)), [x.replace("n3", "").replace("s42", "").upper() for x in order])
    ax.set_xticks(range(len(BINS)), [f"{label}s\nU n={support['Test'][label][CLASSES[1]]}" for label in BINS])
    for i in range(len(order)):
        for j in range(len(BINS)):
            value = matrix[i, j]
            ax.text(j, i, "NA" if np.isnan(value) else f"{value:.1f}%", ha="center", va="center", color="white" if value > 65 else "black")
    ax.set_title("Existing NNIME Test: U recall by effective duration (seed42)")
    fig.colorbar(colored, ax=ax, label="U recall (%)")
    fig.tight_layout(); fig.savefig(out / "test_u_recall_by_length.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tp = [by_model[name]["all"]["per_class"][CLASSES[1]]["tp"] for name in order]
    fp = [by_model[name]["all"]["per_class"][CLASSES[1]]["fp"] for name in order]
    ax.bar(range(6), tp, label="True U predicted U", color="#3274a1")
    ax.bar(range(6), fp, bottom=tp, label="Non-U predicted U", color="#d98b46")
    for i in range(6):
        ax.text(i, tp[i] / 2, str(tp[i]), ha="center", va="center", color="white")
        ax.text(i, tp[i] + fp[i] / 2, str(fp[i]), ha="center", va="center")
    ax.set_xticks(range(6), [x.replace("n3", "").replace("s42", "").upper() for x in order])
    ax.set_ylabel("Test clips predicted as U")
    ax.set_title("U prediction tradeoff: same 165 true-U Test clips, seed42")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "test_u_prediction_tradeoff.png", dpi=160); plt.close(fig)
    print(json.dumps({"output": str(out), "test_models": len(by_model), "test_rows": 638,
                      "dev_inventory_rows": 608, "gpu_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
