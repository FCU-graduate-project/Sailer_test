#!/usr/bin/env python3
"""
Grid search for optimal VAD thresholds — 3-class interview emotion mapping.

Classes:
  Excited       : Arousal > 4.5 AND Dominance > 4.5 AND Valence > V_th
  Unconfident   : Dominance < D_low_th  (applied after Excited is removed)
  Neutral_3Class: everything else

Angry samples are dropped entirely before mapping.
"""

import csv
from itertools import product

DATA_PATH = '/home/brant/Project/SAILER_test/Crab/data/msp2_processed_labels.csv'

EMO_COLS = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']

# ── grid ──────────────────────────────────────────────────────────────────────
V_TH_RANGE    = [3.8, 4.0, 4.2, 4.5]          # Valence threshold for Excited
D_LOW_TH_RANGE = [3.6, 3.8, 4.0, 4.2]         # Dominance upper bound for Unconfident
UNCONF_MIN    = 25_000                          # minimum Unconfident samples required


def load_data(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                **{c: int(r[c]) for c in EMO_COLS},
                'Arousal':   float(r['EmoAct']),
                'Valence':   float(r['EmoVal']),
                'Dominance': float(r['EmoDom']),
            })
    return rows


def apply_mapping(non_angry, v_th, d_low_th):
    excited, unconfident, neutral_3 = [], [], []
    for r in non_angry:
        a, v, d = r['Arousal'], r['Valence'], r['Dominance']
        if a > 4.5 and d > 4.5 and v > v_th:
            excited.append(r)
        elif d < d_low_th:
            unconfident.append(r)
        else:
            neutral_3.append(r)
    return excited, unconfident, neutral_3


def stats(cls):
    n = len(cls)
    if n == 0:
        return {c: 0 for c in EMO_COLS} | {'total': 0}
    s = {'total': n}
    for c in EMO_COLS:
        s[c] = sum(r[c] for r in cls)
    return s


def compute_score(e_s, u_s, n_s):
    """
    Composite score (higher = better):
      35% Excited purity        = (Happy + Surprise) / Excited_total
      30% Unconfident FS ratio  = (Fear + Sad) / Unconfident_total
      10% Unconfident Happy penalty  = 1 - Happy / Unconfident_total
      15% Neutral_3 purity      = Neutral / Neutral_3_total
      10% Unconfident size bonus = min(total / 25000, 1.0)
    """
    e_pur  = (e_s['Happy'] + e_s['Surprise']) / max(e_s['total'], 1)
    u_fs   = (u_s['Fear']  + u_s['Sad'])      / max(u_s['total'], 1)
    u_hap  = u_s['Happy']                      / max(u_s['total'], 1)
    n_pur  = n_s['Neutral']                    / max(n_s['total'], 1)
    u_size = min(u_s['total'] / UNCONF_MIN, 1.0)

    return (0.35 * e_pur +
            0.30 * u_fs +
            0.10 * (1 - u_hap) +
            0.15 * n_pur +
            0.10 * u_size)


def main():
    print("Loading data …")
    rows = load_data(DATA_PATH)
    print(f"  Total rows  : {len(rows):,}")

    non_angry = [r for r in rows if r['Angry'] == 0]
    angry_cnt = len(rows) - len(non_angry)
    print(f"  Angry dropped: {angry_cnt:,}")
    print(f"  Working set  : {len(non_angry):,}")

    W = 108
    sep = '=' * W
    hdr = (f"{'V_th':>5} {'D_low':>5} | "
           f"{'Excited':>8} {'E_pur%':>7} | "
           f"{'Unconf':>8} {'U_FS%':>7} {'U_Hap%':>7} {'≥25k':>4} | "
           f"{'Neut3':>8} {'N_pur%':>7} | {'Score':>7}")
    print(f"\n{sep}\n{hdr}\n{sep}")

    results = []
    for v_th, d_low in product(V_TH_RANGE, D_LOW_TH_RANGE):
        exc, unconf, neut = apply_mapping(non_angry, v_th, d_low)
        e_s, u_s, n_s = stats(exc), stats(unconf), stats(neut)
        score = compute_score(e_s, u_s, n_s)

        e_pur = (e_s['Happy'] + e_s['Surprise']) / max(e_s['total'], 1)
        u_fs  = (u_s['Fear']  + u_s['Sad'])      / max(u_s['total'], 1)
        u_hap = u_s['Happy']                      / max(u_s['total'], 1)
        n_pur = n_s['Neutral']                    / max(n_s['total'], 1)

        ok = 'Y' if u_s['total'] >= UNCONF_MIN else 'N'
        print(f"{v_th:5.1f} {d_low:5.1f} | "
              f"{e_s['total']:8,} {e_pur*100:6.1f}% | "
              f"{u_s['total']:8,} {u_fs*100:6.1f}% {u_hap*100:6.1f}% {ok:>4} | "
              f"{n_s['total']:8,} {n_pur*100:6.1f}% | {score:7.4f}")

        results.append(dict(
            V_th=v_th, D_low=d_low, score=score,
            e_s=e_s, u_s=u_s, n_s=n_s,
            e_pur=e_pur, u_fs=u_fs, u_hap=u_hap, n_pur=n_pur,
        ))

    print(sep)

    results.sort(key=lambda x: -x['score'])

    print("\n\n" + "=" * 60)
    print("TOP 3 COMBINATIONS")
    print("=" * 60)
    for rank, r in enumerate(results[:3], 1):
        e, u, n = r['e_s'], r['u_s'], r['n_s']
        ok = 'YES' if u['total'] >= UNCONF_MIN else 'NO'
        print(f"\n#{rank}  V_th={r['V_th']}  D_low_th={r['D_low']}  Score={r['score']:.4f}")
        print(f"  Excited      : {e['total']:7,}  | Happy+Surprise: {r['e_pur']*100:.1f}%")
        print(f"  Unconfident  : {u['total']:7,}  | Fear+Sad: {r['u_fs']*100:.1f}%  "
              f"Happy: {r['u_hap']*100:.1f}%  ≥25k: {ok}")
        print(f"  Neutral_3    : {n['total']:7,}  | Neutral: {r['n_pur']*100:.1f}%")

    best = results[0]
    e, u, n = best['e_s'], best['u_s'], best['n_s']
    print("\n\n" + "=" * 60)
    print(f"BEST COMBINATION: V_th={best['V_th']}  D_low_th={best['D_low']}")
    print("=" * 60)
    print(f"\nExcited  ({e['total']:,} samples total)")
    for lbl in ['Happy', 'Surprise', 'Fear', 'Sad', 'Neutral', 'Angry', 'Disgust', 'Contempt']:
        cnt = e[lbl]
        print(f"  {lbl:<10}: {cnt:6,}  ({cnt/e['total']*100:.1f}%)")

    print(f"\nUnconfident  ({u['total']:,} samples total)")
    for lbl in ['Fear', 'Sad', 'Happy', 'Neutral', 'Angry', 'Surprise', 'Disgust', 'Contempt']:
        cnt = u[lbl]
        print(f"  {lbl:<10}: {cnt:6,}  ({cnt/u['total']*100:.1f}%)")

    print(f"\nNeutral_3Class  ({n['total']:,} samples total)")
    for lbl in ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disguust', 'Contempt']:
        lbl_fix = lbl if lbl != 'Disguust' else 'Disgust'
        cnt = n.get(lbl_fix, 0)
        print(f"  {lbl_fix:<10}: {cnt:6,}  ({cnt/n['total']*100:.1f}%)")

    total_mapped = e['total'] + u['total'] + n['total']
    print(f"\nTotal mapped samples : {total_mapped:,}  (dropped Angry: {angry_cnt:,})")


if __name__ == '__main__':
    main()
