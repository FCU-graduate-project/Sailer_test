"""Chapter 3 figures, redrawn compactly from saved aggregates.

Reads only existing CSV/JSON outputs (durations, t-SNE coordinates, Test
predictions). Never opens audio, never runs a model.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CRAB = HERE.parents[2]
OUT = HERE / "assets"
OUT.mkdir(exist_ok=True)
EXP = CRAB / "experiments"
TSNE = CRAB / "docs/report/tsne_20260921/coordinates"

def register_cjk_font():
    """matplotlib only reads the first face of a .ttc; extract the TC face."""
    ttc = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    otf = HERE / ".fontcache/NotoSansCJKtc-Regular.otf"
    if not otf.exists() and ttc.exists():
        from fontTools.ttLib import TTCollection
        otf.parent.mkdir(exist_ok=True)
        for face in TTCollection(str(ttc)).fonts:
            if face["name"].getDebugName(1) == "Noto Sans CJK TC":
                face.save(str(otf))
                break
    if otf.exists():
        font_manager.fontManager.addfont(str(otf))


register_cjk_font()
plt.rcParams.update({
    "font.family": ["Noto Sans CJK TC", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
})
CLS = {"E": "#D98C2B", "U": "#7B5EA7", "N": "#2F8C8C"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def duration_figure():
    a75 = pd.read_csv(EXP / "n3_phase_a_long12_20260909/n3a75_train_dev_test.csv")
    b75 = pd.read_csv(EXP / "n3_phase_b_matchmin1_s42_20260912/"
                           "n3b75s42_train_dev_test_provisional.csv")
    real_u = a75[(a75.Source == "NNIME") & (a75.Split_Set == "Train")
                 & (a75.Unconfident == 1)].DurationSeconds
    tts_a = a75[a75.Source != "NNIME"].DurationSeconds
    tts_b = b75[b75.Source != "NNIME"].DurationSeconds
    assert len(real_u) == 628 and len(tts_a) == 471 and len(tts_b) == 471
    edges = [0, 1, 2, 4, 6, 8, 10, 12.01]
    labels = ["<1", "1–2", "2–4", "4–6", "6–8", "8–10", "10–12"]
    series = [(f"NNIME 真人 U（n=628）", real_u, "#3b3b3b"),
              (f"A75 Long12（平均 {tts_a.mean():.1f} s）", tts_a, "#8fb0cc"),
              (f"B75 Match-min1（平均 {tts_b.mean():.1f} s）", tts_b, "#d9a441")]
    fig, ax = plt.subplots(figsize=(3.25, 2.15), layout="constrained")
    x = np.arange(len(labels))
    for i, (lab, d, c) in enumerate(series):
        h, _ = np.histogram(np.clip(d, 0, 12), bins=edges)
        ax.bar(x + (i - 1) * .27, 100 * h / h.sum(), .26, color=c, label=lab,
               linewidth=0)
    ax.set_xticks(x, labels)
    ax.set_xlabel("片段長度（秒）")
    ax.set_ylabel("組內比例（%）")
    ax.yaxis.grid(True, alpha=.25, linewidth=.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 64)
    ax.legend(frameon=False, loc="upper right", handlelength=1)
    save(fig, "fig_duration")
    return tts_a.mean(), tts_b.mean()


def tsne_figure():
    panels = [
        ("(a) 預訓練 XLS-R（未微調），NNIME Train", TSNE / "mapping.csv", "mapped_label"),
        ("(b) 基準模型分類前特徵，NNIME Test", TSNE / "n3_reference.csv", "true_label"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.15), layout="constrained")
    for ax, (title, path, col) in zip(axes, panels):
        df = pd.read_csv(path)
        for k in ("N", "E", "U"):
            sub = df[df[col].str.startswith(k)]
            ax.scatter(sub.tsne_1, sub.tsne_2, s=2.0 if len(df) > 1000 else 4,
                       c=CLS[k], alpha=.75, linewidths=0, label=k)
        ax.set_title(title, pad=3, fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_color("#c8c8c8")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=4, color=CLS[k])
               for k in "EUN"]
    fig.legend(handles, ["E 興奮", "U 不自信", "N 中性"], loc="outside lower center",
               ncol=3, frameon=False)
    save(fig, "fig_tsne")


def test_analysis_figure():
    """(a) baseline confusion; (b) U recall and (c) predicted-U share by length."""
    paired = pd.read_csv(EXP / "n3_ab_test_cpu_diagnostics_20260927/test_paired_predictions.csv")
    base = pd.read_csv(EXP / "n3_phase_a_test_20260910/n3_reference/predictions.csv")
    df = paired.merge(base[["FileName", "predicted_label"]].rename(
        columns={"predicted_label": "base"}), on="FileName")
    assert len(df) == 638
    order = ["Excited", "Unconfident", "Neutral_3Class"]
    m = np.array([[((df.true_label == a) & (df.base == b)).sum() for b in order] for a in order])
    assert m.tolist() == [[90, 33, 58], [13, 75, 77], [25, 38, 229]]

    bins = ["<1", "1-2", "2-4", "4-8", "8-12"]
    runs = {"base": ("基準模型", "#222222", "-", "o"),
            "n3a25s42": ("A25", "#8fb0cc", "--", "o"),
            "n3a75s42": ("A75", "#3f6f99", "-", "o"),
            "n3b25s42": ("B25", "#e2b766", "--", "s"),
            "n3b75s42": ("B75", "#b5761a", "-", "s")}
    fig = plt.figure(figsize=(6.3, 2.15), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.2, 1.2])
    ax = fig.add_subplot(gs[0])
    r = m / m.sum(1, keepdims=True)
    ax.imshow(r, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]}\n{100 * r[i, j]:.0f}%", ha="center", va="center",
                    fontsize=6.3, color="white" if r[i, j] > .55 else "#222")
    ax.set_xticks(range(3), list("EUN")); ax.set_yticks(range(3), list("EUN"))
    ax.set_xlabel("預測"); ax.set_ylabel("真實")
    ax.set_title("(a) 基準模型混淆矩陣", pad=3)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)

    u_sup = [((df.length_bin == b) & (df.true_label == "Unconfident")).sum() for b in bins]
    n_all = [(df.length_bin == b).sum() for b in bins]
    ax_r = fig.add_subplot(gs[1]); ax_s = fig.add_subplot(gs[2])
    for run, (lab, c, ls, mk) in runs.items():
        rec, share = [], []
        for b in bins:
            sub = df[df.length_bin == b]
            u = sub[sub.true_label == "Unconfident"]
            rec.append(100 * (u[run] == "Unconfident").mean())
            share.append(100 * (sub[run] == "Unconfident").mean())
        ax_r.plot(range(5), rec, ls=ls, marker=mk, ms=2.8, lw=1, color=c, label=lab)
        ax_s.plot(range(5), share, ls=ls, marker=mk, ms=2.8, lw=1, color=c, label=lab)
    truth = [100 * ((df.length_bin == b) & (df.true_label == "Unconfident")).sum() / n
             for b, n in zip(bins, n_all)]
    ax_s.plot(range(5), truth, color="#999999", lw=3, alpha=.35, label="真實 U 比例")
    ax_r.set_xticks(range(5), [f"{b}\n({n})" for b, n in zip(bins, u_sup)])
    ax_s.set_xticks(range(5), [f"{b}\n({n})" for b, n in zip(bins, n_all)])
    ax_r.set_xlabel("片段長度（秒）與真人 U 筆數"); ax_s.set_xlabel("片段長度（秒）與片段總數")
    ax_r.set_ylabel("不自信召回率（%）"); ax_s.set_ylabel("被判為不自信的比例（%）")
    ax_r.set_title("(b) 不自信召回率", pad=3); ax_s.set_title("(c) 判為不自信的比例", pad=3)
    for a_ in (ax_r, ax_s):
        a_.set_ylim(0, 100)
        a_.yaxis.grid(True, alpha=.25, linewidth=.5)
    ax_r.legend(frameon=False, loc="upper left", handlelength=1.6, fontsize=6.5)
    ax_s.legend(frameon=False, loc="upper left", handlelength=1.6, fontsize=6.5,
                handles=[ax_s.get_lines()[-1]])
    save(fig, "fig_test_analysis")


if __name__ == "__main__":
    print("TTS mean durations A75/B75:", duration_figure())
    tsne_figure()
    test_analysis_figure()
