"""
Run B / C / D analyses on the diagnostic results saved by diagnostic_run.py.

B: Alpha sensitivity sweep — does audio encoder respond to prosody intensity?
   Per (folder, emotion): max pairwise KL across alpha levels.
   Aggregate by language. If CN sensitivity << EN sensitivity → WavLM 對中文 prosody 不敏感。

C: Cross-lingual prediction distribution KL
   Aggregate prediction distribution over all 560 EN vs all 560 CN.
   KL(EN || CN), KL(CN || EN). Big values = predictions diverge a lot per language.

D: TSNE of WavLM audio embeddings
   2D plot. Color = emotion, marker = language. If CN clips collapse → audio
   encoder maps Chinese to indistinct region (representation collapse evidence).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE

NPZ_PATH    = Path("/home/brant/Project/SAILER_test/Crab/data/diagnostic_results.npz")
TSNE_PATH   = Path("/home/brant/Project/SAILER_test/Crab/data/diagnostic_tsne.png")
REPORT_PATH = Path("/home/brant/Project/SAILER_test/Crab/data/diagnostic_report.txt")

EPS = 1e-9


def kl_div(p, q):
    """KL(p || q) for 1-D probability vectors."""
    p = np.asarray(p) + EPS
    q = np.asarray(q) + EPS
    return float(np.sum(p * np.log(p / q)))


def js_div(p, q):
    """Symmetric Jensen-Shannon divergence."""
    m = 0.5 * (np.asarray(p) + np.asarray(q))
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def analysis_B(probs, folders, emotions, alphas, langs):
    """Per (folder, emotion) max pairwise KL across alphas. Aggregate by language."""
    sensitivity_by_lang = {"en": [], "cn": []}
    n = len(probs)
    seen = set()
    for i in range(n):
        key = (folders[i], emotions[i])
        if key in seen:
            continue
        seen.add(key)
        mask = (folders == folders[i]) & (emotions == emotions[i])
        idxs = np.where(mask)[0]
        alpha_vals  = alphas[idxs]
        probs_subset = probs[idxs]
        order = np.argsort(alpha_vals)
        ps = probs_subset[order]  # [num_alpha, 3]
        max_kl = 0.0
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                max_kl = max(max_kl, js_div(ps[a], ps[b]))
        sensitivity_by_lang[langs[i]].append(max_kl)

    en_sens = np.array(sensitivity_by_lang["en"])
    cn_sens = np.array(sensitivity_by_lang["cn"])
    return en_sens, cn_sens


def analysis_C(probs, langs):
    """Aggregate pred distribution per language."""
    en_mean = probs[langs == "en"].mean(axis=0)
    cn_mean = probs[langs == "cn"].mean(axis=0)
    return en_mean, cn_mean


def analysis_D(audio_embs, langs, emotions, save_path):
    """TSNE of 1024-d audio embeddings → 2D plot."""
    print(f"   running TSNE on {audio_embs.shape}…")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(audio_embs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    emo_unique = sorted(set(emotions.tolist()))
    cmap = matplotlib.colormaps.get_cmap("tab10")
    emo_colors = {e: cmap(i / max(1, len(emo_unique) - 1)) for i, e in enumerate(emo_unique)}

    for ax, lang_filter, title in [
        (axes[0], "en", "English synthetic (560 clips)"),
        (axes[1], "cn", "Chinese synthetic (560 clips)"),
    ]:
        for emo in emo_unique:
            mask = (langs == lang_filter) & (emotions == emo)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=20, alpha=0.6, c=[emo_colors[emo]], label=emo)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, ncol=2)

    plt.suptitle("WavLM-Large audio embedding TSNE — EN vs CN by emotion", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"   saved → {save_path}")
    return coords


def cluster_separation_metric(coords, langs, emotions):
    """Quick measure: silhouette-like — for each lang, average inter-emo distance / intra-emo distance."""
    from scipy.spatial.distance import pdist, squareform
    out = {}
    for lang in ("en", "cn"):
        mask = langs == lang
        c = coords[mask]
        e = emotions[mask]
        D = squareform(pdist(c))
        same_emo, diff_emo = [], []
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                if e[i] == e[j]:
                    same_emo.append(D[i, j])
                else:
                    diff_emo.append(D[i, j])
        out[lang] = (np.mean(same_emo), np.mean(diff_emo), np.mean(diff_emo) / np.mean(same_emo))
    return out


def main():
    z = np.load(NPZ_PATH, allow_pickle=True)
    audio_embs = z["audio_embs"]
    probs      = z["probs"]
    folders    = z["folders"]
    langs      = z["langs"]
    emotions   = z["emotions"]
    alphas     = z["alphas"]
    classes    = list(z["classes"])

    print(f"Loaded {len(probs)} clips ({np.sum(langs=='en')} EN + {np.sum(langs=='cn')} CN)")
    print(f"Classes: {classes}")

    report_lines = []
    def log(s=""):
        print(s)
        report_lines.append(s)

    log("=" * 80)
    log("Crab Diagnostic Baseline — current EN-trained model on bilingual synthetic data")
    log("=" * 80)

    # ─────────────── B: Alpha sensitivity sweep ───────────────
    log("\n[B] Alpha sensitivity sweep")
    log("    Per (folder, emotion): max pairwise JS divergence across alpha 0.3–0.9")
    log("    Bigger = audio encoder responds more to prosody intensity")
    en_sens, cn_sens = analysis_B(probs, folders, emotions, alphas, langs)
    log(f"    EN: n={len(en_sens):3d}  mean JS={en_sens.mean():.4f}  median={np.median(en_sens):.4f}  std={en_sens.std():.4f}")
    log(f"    CN: n={len(cn_sens):3d}  mean JS={cn_sens.mean():.4f}  median={np.median(cn_sens):.4f}  std={cn_sens.std():.4f}")
    ratio = cn_sens.mean() / max(en_sens.mean(), EPS)
    log(f"    Sensitivity ratio CN/EN = {ratio:.3f}")
    log(f"    → CN sensitivity is {ratio*100:.1f}% of EN baseline")
    if ratio < 0.5:
        log(f"    ⚠ ratio < 0.5 → strong evidence audio encoder is insensitive to CN prosody")
    elif ratio < 0.8:
        log(f"    △ ratio 0.5–0.8 → moderate evidence of CN sensitivity gap")
    else:
        log(f"    ✓ ratio ≥ 0.8 → audio encoder reasonably sensitive to CN prosody")

    # ─────────────── C: Cross-lingual prediction KL ───────────────
    log("\n[C] Cross-lingual prediction distribution")
    en_mean, cn_mean = analysis_C(probs, langs)
    log(f"    Avg pred distribution (over 560 clips each):")
    log(f"      EN: " + "  ".join(f"{c}={en_mean[i]:.3f}" for i, c in enumerate(classes)))
    log(f"      CN: " + "  ".join(f"{c}={cn_mean[i]:.3f}" for i, c in enumerate(classes)))
    log(f"    KL(EN||CN) = {kl_div(en_mean, cn_mean):.4f}")
    log(f"    KL(CN||EN) = {kl_div(cn_mean, en_mean):.4f}")
    log(f"    JS divergence = {js_div(en_mean, cn_mean):.4f}")
    log(f"    → Bigger JS = predictions diverge more by language (independent of ground truth)")

    # ─────────────── D: TSNE plot ───────────────
    log("\n[D] TSNE on WavLM audio embeddings")
    coords = analysis_D(audio_embs, langs, emotions, TSNE_PATH)
    log(f"    TSNE saved → {TSNE_PATH}")
    sep = cluster_separation_metric(coords, langs, emotions)
    log(f"    Cluster separation (inter-emo dist / intra-emo dist):")
    for lang in ("en", "cn"):
        same, diff, sep_ratio = sep[lang]
        log(f"      {lang.upper()}: intra={same:.2f}  inter={diff:.2f}  ratio={sep_ratio:.3f}  (>1 = emotions separated)")

    sep_ratio_en = sep["en"][2]
    sep_ratio_cn = sep["cn"][2]
    log(f"    Separation ratio CN/EN = {sep_ratio_cn/sep_ratio_en:.3f}")
    if sep_ratio_cn / sep_ratio_en < 0.7:
        log(f"    ⚠ CN embeddings cluster less by emotion → representation collapse hint")

    # ─────────────── Summary ───────────────
    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    log(f"  B (alpha sensitivity): CN/EN = {ratio:.3f}")
    log(f"  C (cross-lingual JS):  {js_div(en_mean, cn_mean):.4f}")
    log(f"  D (cluster separation): CN/EN = {sep_ratio_cn/sep_ratio_en:.3f}")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    log(f"\n📄 Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
