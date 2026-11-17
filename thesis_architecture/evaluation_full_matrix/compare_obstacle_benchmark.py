#!/usr/bin/env python3
r"""
compare_obstacle_benchmark.py
=============================
Paired comparison of the active-sensing benchmark WITH obstacles against the one WITHOUT: same
heldout shapes, variants (seeds) and kernels, so every final-round result of a method can be paired
with its obstacle-free counterpart. Reads the two `metrics_all.csv` tables (no torch, no DB).

Output (--out_dir):
  obstacle_vs_free_final.csv   per kernel x method x metric: n pairs, mean without / with obstacles,
                               mean paired difference (with - without) and its 95 % bootstrap CI over
                               shapes (variants averaged within a shape first)
  obstacle_vs_free_final.png   the paired differences as dots with CI, one panel per metric x kernel

    python compare_obstacle_benchmark.py \
        --obst_csv plots/exploration_comparison_obstacles_dice6_r0.08/extended/metrics_all.csv \
        --free_csv plots/exploration_comparison_combined/metrics_all.csv
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from plot_exploration_summary import COLOR, LABEL, METHODS, MUTED, SPINE, TEXT   # noqa: E402

#: (column, title, lower is better)
METRICS = (('E_truth', 'Ergodicity vs. GT', True),
           ('swept06', 'GT mass within 0.06 of path', False),
           ('belief_rmse', 'Belief RMSE vs. GT', True),
           ('out_gt_frac', 'Path outside GT support', True))
KERNELS = ('rbf', 'ak')
KERNEL_NAME = {'rbf': 'RBF belief', 'ak': 'Attentive-Kernel belief'}


def final_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    last = max(int(r['round']) for r in rows)
    return {(r['kernel'], r['method'], r['shape'], int(r['variant'])): r
            for r in rows if int(r['round']) == last}


def num(v):
    return float('nan') if v in ('', 'None', None) else float(v)


def paired(free, obst, kernel, method, col, n_boot=4000, seed=0):
    """-> dict(n, mean_free, mean_obst, diff, lo, hi) or None (no pairs)."""
    by_shape = {}
    for key, ro in obst.items():
        if key[0] != kernel or key[1] != method or key not in free:
            continue
        a, b = num(free[key][col]), num(ro[col])
        if a == a and b == b:
            by_shape.setdefault(key[2], []).append((a, b))
    if not by_shape:
        return None
    pairs = np.array([p for v in by_shape.values() for p in v])
    shape_diff = np.array([np.mean([b - a for a, b in v]) for v in by_shape.values()])
    rng = np.random.default_rng(seed)
    boot = rng.choice(shape_diff, size=(n_boot, len(shape_diff)), replace=True).mean(1)
    return dict(n=len(pairs), mean_free=pairs[:, 0].mean(), mean_obst=pairs[:, 1].mean(),
                diff=shape_diff.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--obst_csv', required=True)
    ap.add_argument('--free_csv', required=True)
    ap.add_argument('--out_dir', default=None, help='Default: the folder of --obst_csv.')
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.obst_csv))
    free, obst = final_rows(a.free_csv), final_rows(a.obst_csv)
    methods = [m for m in METHODS if any(k[1] == m for k in obst) and any(k[1] == m for k in free)]
    res = {}
    with open(os.path.join(out_dir, 'obstacle_vs_free_final.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'metric', 'n_pairs', 'mean_without', 'mean_with',
                    'mean_diff_with_minus_without', 'ci95_lo', 'ci95_hi'])
        for k in KERNELS:
            for m in methods:
                for col, _, _ in METRICS:
                    r = paired(free, obst, k, m, col)
                    if r is None:
                        continue
                    res[(k, m, col)] = r
                    w.writerow([k, m, col, r['n'], r['mean_free'], r['mean_obst'], r['diff'],
                                r['lo'], r['hi']])

    fig, axes = plt.subplots(len(METRICS), len(KERNELS), squeeze=False, facecolor='white',
                             figsize=(5.2 * len(KERNELS), 0.42 * len(methods) * len(METRICS) + 1.5),
                             sharey=True)
    ys = np.arange(len(methods))[::-1]
    for i, (col, title, lower) in enumerate(METRICS):
        for j, k in enumerate(KERNELS):
            ax = axes[i, j]
            ax.set_facecolor('white')
            ax.axvline(0.0, color=MUTED, lw=1.0, zorder=1)
            for y, m in zip(ys, methods):
                r = res.get((k, m, col))
                if r is None:
                    continue
                ax.plot([r['lo'], r['hi']], [y, y], color=COLOR[m], lw=2, solid_capstyle='round', zorder=2)
                ax.plot(r['diff'], y, 'o', ms=8, color=COLOR[m], mec='white', mew=2, zorder=3)
            ax.set_yticks(ys)
            ax.set_yticklabels([LABEL[m] for m in methods], fontsize=8.5, color=TEXT)
            better = 'lower' if lower else 'higher'
            ax.set_title(f'{title}: with − without obstacles\n{KERNEL_NAME[k]} ({better} is better)',
                         fontsize=9.5, color=TEXT)
            ax.grid(axis='x', alpha=0.2)
            ax.tick_params(axis='x', labelsize=8, colors=MUTED)
            for s in ax.spines.values():
                s.set_color(SPINE)
    fig.suptitle('Final round, paired by shape / variant / kernel (same seeds): mean difference '
                 'with 95 % bootstrap CI over the heldout shapes', fontsize=11, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, 'obstacle_vs_free_final.png')
    fig.savefig(out, dpi=130, facecolor='white')
    print(f'  saved {out}')


if __name__ == '__main__':
    main()
