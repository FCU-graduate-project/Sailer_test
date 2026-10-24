"""Render aggregate duration counts only; never open source audio/transcripts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/datasets/nnime_u_match_min1_20260911/build_report.json"
OUT = Path(__file__).resolve().parent / "assets"


def main():
    report = json.loads(SOURCE.read_text())
    series = [("NNIME Train U >= 1 s", report["reference_counts"])]
    assert sum(series[0][1]) == report["reference_ge1_count"] == 510
    for arm, n in [("B25-min1", 157), ("B50-min1", 314), ("B75-min1", 471)]:
        counts = report["arms"][arm]["actual_bin_counts"]
        assert sum(counts) == n
        series.append((arm.split("-")[0], counts))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.0, 3.1), layout="constrained")
    x = np.arange(6)
    colors = ["#303030", "#8293a4", "#bcc6cd", "#eeeeee"]
    hatches = ["", "//", "..", "xx"]
    for i, (label, counts) in enumerate(series):
        ax.bar(x + (i - 1.5) * .2, 100 * np.array(counts) / sum(counts),
               width=.19, color=colors[i], edgecolor="#303030", linewidth=.5,
               hatch=hatches[i], label=label)
    ax.set_xticks(x, ["[1, 2)", "[2, 4)", "[4, 6)", "[6, 8)", "[8, 10)", "[10, 12]"])
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Within-group proportion (%)")
    ax.set_ylim(0, 49)
    ax.yaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"duration_distribution.{ext}", dpi=180)
    plt.close(fig)
    print("Validated 510 reference clips and 157/314/471 synthetic subsets.")


if __name__ == "__main__":
    main()
