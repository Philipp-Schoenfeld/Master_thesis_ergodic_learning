#!/usr/bin/env python3
"""Report and plots for the missions of active_sensing_3d.py.

    python plot_active_sensing_3d.py --out_dir results/active_sensing_3d
writes report.md, curves.png, rounds_to_reach.png and examples.png into --out_dir.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#00C853', '#F57C00', '#8E24AA', '#00838F', '#1565C0', '#C62828']
REFERENCE = 'linear'
COND_TITLE = {'none_known': 'no prior samples', 'ten_samples': 'ten known samples'}


def style(ax):
    ax.set_facecolor('white')
    ax.grid(alpha=0.2)
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555')


def load(out_dir):
    cells = {}
    for f in sorted(glob.glob(os.path.join(out_dir, '*__*.json'))):
        base = os.path.basename(f)[:-5]
        meth, cond = base.rsplit('__', 1)
        cells[(meth, cond)] = json.load(open(f))
    return cells


def series(recs, key, rounds):
    """(n_missions, rounds): value after each round, carried forward once a mission is done."""
    out = np.full((len(recs), rounds), np.nan)
    for i, m in enumerate(recs):
        rows = m['rows']
        for r in range(rounds):
            out[i, r] = rows[min(r, len(rows) - 1)][key]
    return out


def boot(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ms(x):
    return f"{np.nanmean(x):.2e} ± {np.nanstd(x, ddof=1) / np.sqrt(len(x)):.1e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='results/active_sensing_3d')
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.out_dir, 'config.json')))['args']
    R = cfg['rounds']
    cells = load(a.out_dir)
    methods = list(dict.fromkeys(m for m, _ in cells))
    conds = list(dict.fromkeys(c for _, c in cells))
    col = {m: (COLORS[i] if m != REFERENCE else '#555555') for i, m in enumerate(methods)}
    names = {m: m.replace('_', ' ') for m in methods}

    lines = [f"# 3D active sensing: start-conditioned planners vs linear rays\n",
             f"UCB target (kappa {cfg['kappa0']} -> {cfg['kappa1']}), RBF-GP belief over the particles, "
             f"{cfg['refine_iters']} refinement iterations per round, <= {R} rounds, {cfg['n_cand']} candidates, "
             f"sensing/coverage radius {cfg['r_sense']}. E = Sinkhorn footprint error of the whole driven path "
             f"against the TRUE density (lower is better); mean ± SEM over missions; Δ = paired difference to "
             f"**linear rays** (95 % bootstrap CI).\n"]
    for cond in conds:
        lines += [f"\n## {COND_TITLE.get(cond, cond)}\n",
                  "| method | n | E round 0 | E round 1 | E round 2 | E final | Δ final vs linear | swept final | path length | reached 99 % | rounds to 99 % |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        ref = series(cells[(REFERENCE, cond)], 'e_truth', R) if (REFERENCE, cond) in cells else None
        for m in methods:
            if (m, cond) not in cells:
                continue
            recs = cells[(m, cond)]
            E = series(recs, 'e_truth', R)
            sw = series(recs, 'swept', R)[:, -1]
            ln = series(recs, 'length', R)[:, -1]
            reached = np.array([any(r['swept'] >= cfg['reach'] for r in x['rows']) for x in recs])
            rt = np.array([next((r['round'] + 1 for r in x['rows'] if r['swept'] >= cfg['reach']), R + 1) for x in recs])
            delta = '–'
            if ref is not None and m != REFERENCE:
                d, lo, hi = boot(E[:, -1] - ref[:, -1])
                delta = f"{d:+.2e} [{lo:+.2e}, {hi:+.2e}]"
            lines.append(f"| {names[m]} | {len(recs)} | " + " | ".join(ms(E[:, r]) for r in range(min(3, R)))
                         + f" | {ms(E[:, -1])} | {delta} | {sw.mean():.3f} | {ln.mean():.1f} | {reached.mean() * 100:.0f}% | {rt.mean():.2f} |")
    open(os.path.join(a.out_dir, 'report.md'), 'w').write('\n'.join(lines) + '\n')

    # curves
    fig, axes = plt.subplots(2, len(conds), figsize=(6.5 * len(conds), 9), facecolor='white', squeeze=False)
    for c, cond in enumerate(conds):
        for m in methods:
            if (m, cond) not in cells:
                continue
            recs = cells[(m, cond)]
            E = series(recs, 'e_truth', R)
            mean = E.mean(0)
            sem = E.std(0, ddof=1) / np.sqrt(len(E))
            axes[0, c].plot(range(1, R + 1), mean, '-o', color=col[m], label=names[m], lw=2)
            axes[0, c].fill_between(range(1, R + 1), mean - sem, mean + sem, color=col[m], alpha=0.15)
            L = series(recs, 'length', R).mean(0)
            axes[1, c].plot(L, series(recs, 'swept', R).mean(0), '-o', color=col[m], label=names[m], lw=2)
        axes[0, c].set_yscale('log')
        axes[0, c].set_xlabel('round'); axes[0, c].set_ylabel('E (Sinkhorn, true density)')
        axes[1, c].set_xlabel('cumulative path length'); axes[1, c].set_ylabel('swept truth mass')
        axes[0, c].set_title(COND_TITLE.get(cond, cond), color='#1A1A2E')
        for r in (0, 1):
            style(axes[r, c])
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, 'curves.png'), dpi=130); plt.close(fig)

    # rounds to reach 99 %
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 4.2), facecolor='white', squeeze=False)
    for c, cond in enumerate(conds):
        ax = axes[0, c]
        for j, m in enumerate(methods):
            if (m, cond) not in cells:
                continue
            recs = cells[(m, cond)]
            rt = np.array([next((r['round'] + 1 for r in x['rows'] if r['swept'] >= cfg['reach']), R + 1) for x in recs])
            ax.bar(j, rt.mean(), color=col[m], alpha=0.85, yerr=rt.std(ddof=1) / np.sqrt(len(rt)), ecolor='#555')
        ax.set_xticks(range(len(methods))); ax.set_xticklabels([names[m] for m in methods], rotation=25, ha='right', fontsize=8)
        ax.set_ylabel(f'rounds until {cfg["reach"] * 100:.0f} % swept (cap = {R + 1})')
        ax.set_title(COND_TITLE.get(cond, cond), color='#1A1A2E'); style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, 'rounds_to_reach.png'), dpi=130); plt.close(fig)

    # examples: mission 0 of the first condition, one panel per method
    from data_surfaces import load_surface_db
    from eval_lm3d_compare import pick_entries
    entries = pick_entries(load_surface_db(cfg['db3d'], splits=tuple(cfg['splits'])), cfg['per_split'], cfg['seed'])
    cond = conds[0]
    shades = plt.cm.viridis(np.linspace(0.1, 0.9, R))
    fig = plt.figure(figsize=(4.8 * len(methods), 9.6), facecolor='white')
    for row, mi in enumerate([0, len(entries) // 2]):
        e = entries[mi]
        P = np.asarray(e['parts'])
        for j, m in enumerate(methods):
            if (m, cond) not in cells:
                continue
            ax = fig.add_subplot(2, len(methods), row * len(methods) + j + 1, projection='3d')
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=4 + 40 * P[:, 3] / P[:, 3].max(), c='#444444', alpha=0.3)
            z = np.load(os.path.join(a.out_dir, os.path.basename(
                glob.glob(os.path.join(a.out_dir, f"{m}__{cond}_paths.npz"))[0])))
            for r in range(R):
                k = f'pos_{mi}_{r}'
                if k not in z:
                    break
                q = z[k].astype(float)
                ax.plot(q[:, 0], q[:, 1], q[:, 2], color=shades[r], lw=2.0, alpha=0.95, label=f'round {r + 1}')
            rows = cells[(m, cond)][mi]['rows']
            ax.set_title(f"{names[m]}\n{e['name']}: swept {rows[-1]['swept']:.2f} in {len(rows)} rounds",
                         fontsize=9, color='#1A1A2E')
            ax.tick_params(labelsize=6)
            if row == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, 'examples.png'), dpi=120); plt.close(fig)
    print('wrote report.md, curves.png, rounds_to_reach.png, examples.png to', a.out_dir)


if __name__ == '__main__':
    main()
