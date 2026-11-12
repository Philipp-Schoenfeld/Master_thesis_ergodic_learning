"""Intuitive summary figures for the exploration comparison (reads the DB / CSV written by
run_exploration_comparison.py, needs no torch).

  01_rank_scorecard      mean rank of every method per metric (1 = best) + overall rank
  02_winrate_matrix      head-to-head: share of paired runs in which row beats column
  03_final_metrics_dots  final-round metrics with 95 % CI, both kernels side by side
  04_per_shape_heatmap   which method wins on which shape (relative error + rank)
  05_rank_over_rounds    how the E-vs-GT ranking evolves over the five rounds
  06_efficiency          path length needed to reach 90 / 99 % coverage; error vs. path length
  07_distributions       per-run spread of the final error and belief quality
  ranking.csv            numbers behind the scorecard

Usage: python plot_exploration_summary.py [--dir plots/exploration_comparison]
"""
import argparse
import csv
import os
import sqlite3

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

# order = display order of the groups: model-free baselines, search without a prior, search with a prior
METHODS = ('rw', 'lawn', 'gpgoal', 'linray', 'linadam', 'lintsvec',
           'unet630', 'old500', 'unet0', 'unet1890', 'lm1300', 'dit630', 'selfsup')
KERNELS = ('rbf', 'ak')
LABEL = {'unet630': 'U-Net λ=630', 'dit630': 'DiT λ=630', 'rw': 'Random walk',
         'lawn': 'Lawnmower', 'linadam': 'Linear init + Adam-SVGD',
         'selfsup': 'Self-supervised + TSVEC', 'old500': 'U-Net λ=500 (older)',
         'unet0': 'U-Net, no ErgLoss', 'unet1890': 'U-Net λ=1890', 'lm1300': 'U-Net λ=1300 (600 ep.)',
         'lintsvec': 'Linear init + TSVEC (no prior)', 'gpgoal': 'Line to GP-sampled point',
         'linray': 'Best straight ray (no search)'}
SHORT = {'unet630': 'U-Net 630', 'dit630': 'DiT', 'rw': 'Random walk', 'lawn': 'Lawnmower',
         'linadam': 'Lin+Adam', 'selfsup': 'Self-sup.', 'old500': 'U-Net 500', 'unet0': 'U-Net λ0',
         'unet1890': 'U-Net 1890', 'lm1300': 'U-Net 1300', 'lintsvec': 'Lin+TSVEC',
         'gpgoal': 'GP-goal line', 'linray': 'Best ray'}
COLOR = {'unet630': '#1565C0', 'dit630': '#6A1B9A', 'rw': '#888888', 'lawn': '#E65100',
         'linadam': '#00897B', 'selfsup': '#C62828', 'old500': '#0D47A1', 'unet0': '#42A5F5',
         'unet1890': '#5C6BC0', 'lm1300': '#26C6DA', 'lintsvec': '#2E7D32', 'gpgoal': '#F9A825',
         'linray': '#8D6E63'}
KERNEL_NAME = {'rbf': 'RBF belief', 'ak': 'Attentive-Kernel belief'}
TEXT, MUTED, SPINE = '#1A1A2E', '#555555', '#CCCCCC'

# (column, label, lower is better, number format)
RANK_METRICS = [
    ('E_truth', 'Ergodicity\nvs. GT', True, '{:.1f}'),
    ('E_belief', 'Ergodicity\nvs. belief', True, '{:.1f}'),
    ('belief_rmse', 'Belief RMSE\nvs. GT', True, '{:.3f}'),
    ('belief_corr', 'Belief corr.\nwith GT', False, '{:.3f}'),
    ('swept03', 'GT swept\n(r = 0.03)', False, '{:.3f}'),
    ('out_gt_frac', 'Path outside\nGT', True, '{:.2f}'),
    ('in_hi50_frac', 'Path in\ndense GT', False, '{:.2f}'),
]
# light = worst rank, dark = best rank
RANK_CMAP = LinearSegmentedColormap.from_list('rank', ['#F1F5FB', '#9EC1E8', '#1565C0', '#0A2F63'])


# --------------------------------------------------------------------------- data
class Data:
    """arr[kernel][metric] -> float array [method, shape, variant, round]."""

    def __init__(self, directory, db_path=None, only=None):
        with open(os.path.join(directory, 'metrics_all.csv'), newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if only:
            rows = [r for r in rows if r['method'] in only]
        self.shapes = sorted({r['shape'] for r in rows})
        self.n_var = max(int(r['variant']) for r in rows) + 1
        self.n_rounds = max(int(r['round']) for r in rows) + 1
        self.methods = [m for m in METHODS if any(r['method'] == m for r in rows)]
        si = {s: i for i, s in enumerate(self.shapes)}
        mi = {m: i for i, m in enumerate(self.methods)}
        cols = [c for c in rows[0] if c not in ('round', 'shape', 'kernel', 'method', 'variant')]
        shp = (len(self.methods), len(self.shapes), self.n_var, self.n_rounds)
        self.arr = {k: {c: np.full(shp, np.nan) for c in cols} for k in KERNELS}
        for r in rows:
            idx = (mi[r['method']], si[r['shape']], int(r['variant']), int(r['round']))
            for c in cols:
                v = r[c]
                self.arr[r['kernel']][c][idx] = float(v) if v not in ('', 'None') else np.nan
        # coverage-reach columns come from the runs table
        db_path = db_path or os.path.join(directory, 'exploration_comparison.db')
        db = sqlite3.connect('file:' + os.path.abspath(db_path).replace(os.sep, '/') + '?mode=ro', uri=True)
        self.reach = {k: {c: np.full(shp[:3], np.nan) for c in
                          ('reached90', 'reached99', 'path_len_90', 'path_len_99', 'seconds')}
                      for k in KERNELS}
        for sh, k, m, v, *vals in db.execute(
                'select shape,kernel,method,variant,reached90,reached99,path_len_90,path_len_99,'
                'seconds from runs'):
            if m not in mi:
                continue
            for c, x in zip(('reached90', 'reached99', 'path_len_90', 'path_len_99', 'seconds'),
                            vals):
                self.reach[k][c][mi[m], si[sh], v] = np.nan if x is None else x
        db.close()

    def final(self, k, c):
        return self.arr[k][c][..., -1]                      # [method, shape, variant]


def ranks(x, lower):
    """Rank over axis 0 (1 = best); ties are not averaged (continuous metrics)."""
    order = np.argsort(x if lower else -x, axis=0)
    return np.argsort(order, axis=0) + 1


def style(ax, grid=True):
    ax.set_facecolor('white')
    for s in ax.spines.values():
        s.set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    if grid:
        ax.grid(alpha=0.2)


def plain_log_axis(axis, ticks):
    """Log axis with plain decimal tick labels instead of 10^n notation."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())


def save(fig, directory, name):
    path = os.path.join(directory, name)
    fig.savefig(path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('  saved', path)


def overall_order(d, k):
    """Methods sorted by mean rank over the rank metrics (best first) + that mean rank."""
    per_metric = np.array([ranks(d.final(k, c), low).mean(axis=(1, 2)) for c, _, low, _ in
                           RANK_METRICS])                   # [metric, method]
    overall = per_metric.mean(axis=0)
    return np.argsort(overall), per_metric, overall


# --------------------------------------------------------------------------- figures
def fig_scorecard(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(17, 1.5 + 0.62 * len(d.methods)), facecolor='white')
    rows_csv = [['kernel', 'method', 'metric', 'mean_rank', 'mean_value']]
    for ax, k in zip(axes, KERNELS):
        order, per_metric, overall = overall_order(d, k)
        ncol = len(RANK_METRICS) + 1
        for ri, mi in enumerate(order):
            for ci in range(ncol):
                if ci < len(RANK_METRICS):
                    c, _, low, fmt = RANK_METRICS[ci]
                    rk = per_metric[ci, mi]
                    val = np.nanmean(d.final(k, c)[mi])
                    txt = f'{rk:.2f}\n{fmt.format(val)}'
                    rows_csv.append([k, d.methods[mi], c, f'{rk:.3f}', f'{val:.5f}'])
                else:
                    rk = overall[mi]
                    txt = f'{rk:.2f}\n#{ri + 1}'
                    rows_csv.append([k, d.methods[mi], 'overall', f'{rk:.3f}', ''])
                shade = (len(d.methods) - rk) / (len(d.methods) - 1)
                x0 = ci + (0.25 if ci == ncol - 1 else 0)
                ax.add_patch(plt.Rectangle((x0, ri), 0.94, 0.94, color=RANK_CMAP(shade), lw=0))
                ax.text(x0 + 0.47, ri + 0.47, txt, ha='center', va='center', fontsize=8.5,
                        color='white' if shade > 0.5 else TEXT,
                        fontweight='bold' if ci == ncol - 1 else 'normal', linespacing=1.3)
        ax.set_xlim(-0.05, ncol + 0.3)
        ax.set_ylim(len(d.methods), -0.05)
        ax.set_xticks([i + 0.47 for i in range(len(RANK_METRICS))] + [ncol - 1 + 0.72])
        ax.set_xticklabels([m[1] for m in RANK_METRICS] + ['Overall\nmean rank'], fontsize=8,
                           color=TEXT)
        ax.xaxis.tick_top()
        ax.set_yticks([i + 0.47 for i in range(len(d.methods))])
        ax.set_yticklabels([LABEL[d.methods[mi]] for mi in order], fontsize=9.5, color=TEXT)
        for lab, mi in zip(ax.get_yticklabels(), order):
            lab.set_color(COLOR[d.methods[mi]])
            lab.set_fontweight('bold')
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(KERNEL_NAME[k], fontsize=11, color=TEXT, pad=38)
    fig.suptitle(f'Rank scorecard after round 5: mean rank among the {len(d.methods)} methods (1 = best), top '
                 'number; mean metric value below.\nRanks are formed per (shape, variant) with '
                 f'the same seed, then averaged over {len(d.shapes)} shapes × {d.n_var} variants.',
                 fontsize=10, color=MUTED, y=0.0, va='top')
    fig.tight_layout()
    save(fig, out, '01_rank_scorecard.png')
    with open(os.path.join(out, 'ranking.csv'), 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows_csv)


def fig_winrate(d, out):
    cmap = LinearSegmentedColormap.from_list('wr', ['#E65100', '#F7F7F7', '#1565C0'])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor='white')
    for ax, k in zip(axes, KERNELS):
        order = overall_order(d, k)[0]
        E = d.final(k, 'E_truth').reshape(len(d.methods), -1)
        n = len(order)
        W = np.full((n, n), np.nan)
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                if i != j:
                    W[i, j] = np.mean(E[a] < E[b])
        ax.imshow(W, cmap=cmap, norm=TwoSlopeNorm(0.5, 0, 1))
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, '–', ha='center', va='center', color=MUTED)
                else:
                    ax.text(j, i, f'{100 * W[i, j]:.0f} %', ha='center', va='center', fontsize=10,
                            color='white' if abs(W[i, j] - 0.5) > 0.3 else TEXT,
                            fontweight='bold' if W[i, j] > 0.5 else 'normal')
        ax.set_xticks(range(n))
        ax.set_xticklabels([SHORT[d.methods[m]] for m in order], fontsize=9, color=TEXT,
                           rotation=20)
        ax.set_yticks(range(n))
        ax.set_yticklabels([SHORT[d.methods[m]] for m in order], fontsize=9, color=TEXT)
        ax.set_xlabel('… beaten by column', fontsize=9, color=MUTED)
        ax.set_ylabel('Row method wins against …', fontsize=9, color=MUTED)
        ax.set_title(KERNEL_NAME[k], fontsize=11, color=TEXT)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle('Head-to-head on final ergodicity vs. GT: share of the 175 paired runs '
                 '(same shape + seed) in which the row method has the LOWER error\n'
                 '50 % = coin flip; blue = row wins, orange = row loses', fontsize=10,
                 color=MUTED, y=0.0, va='top')
    fig.tight_layout()
    save(fig, out, '02_winrate_matrix.png')


def fig_dots(d, out):
    panels = [('E_truth', 'Ergodicity vs. GT ↓ (log)', True), ('E_belief', 'Ergodicity vs. belief ↓ (log)', True),
              ('belief_rmse', 'Belief RMSE vs. GT ↓', False), ('belief_corr', 'Belief corr. with GT ↑', False),
              ('out_gt_frac', 'Path outside GT support ↓', False),
              ('in_hi50_frac', 'Path in dense GT (≥ 0.5) ↑', False)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), facecolor='white')
    order = overall_order(d, 'ak')[0]
    ypos = {m: i for i, m in enumerate(order)}
    for ax, (c, title, log) in zip(axes.flat, panels):
        style(ax)
        for k, mk, fill in (('rbf', 'o', True), ('ak', 'D', False)):
            for mi in range(len(d.methods)):
                sm = np.nanmean(d.final(k, c)[mi], axis=1)          # per-shape variant mean
                mean, ci = sm.mean(), 1.96 * sm.std() / np.sqrt(len(sm))
                col = COLOR[d.methods[mi]]
                y = ypos[mi] + (-0.17 if k == 'rbf' else 0.17)
                ax.errorbar(mean, y, xerr=ci, fmt=mk, color=col, ms=7, lw=1.6, capsize=3,
                            mfc=col if fill else 'white', mew=1.8)
        if log:
            ax.set_xscale('log')
            plain_log_axis(ax.xaxis, [4, 6, 10, 15, 25, 40])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([SHORT[d.methods[m]] for m in order], fontsize=9, color=TEXT)
        ax.set_ylim(len(order) - 0.4, -0.6)
        ax.set_title(title, fontsize=10, color=TEXT)
    h = [plt.Line2D([], [], marker='o', color=MUTED, ls='', mfc=MUTED, ms=7, label='RBF belief'),
         plt.Line2D([], [], marker='D', color=MUTED, ls='', mfc='white', mew=1.8, ms=7,
                    label='Attentive-Kernel belief')]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f'Final metrics after round 5 (mean over {len(d.shapes)} shapes, whiskers: 95 % CI over '
                 'shapes); methods ordered by overall rank under the AK belief',
                 fontsize=11, color=TEXT)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save(fig, out, '03_final_metrics_dots.png')


def fig_per_shape(d, out):
    cmap = LinearSegmentedColormap.from_list('rel', ['#0A2F63', '#1565C0', '#9EC1E8', '#F1F5FB'])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.34 * len(d.shapes) + 2.6), facecolor='white',
                             sharey=True)
    for ax, k in zip(axes, KERNELS):
        E = np.nanmean(d.final(k, 'E_truth'), axis=2)               # [method, shape]
        rel = E / E.min(axis=0, keepdims=True)
        rk = ranks(E, True)
        order = overall_order(d, k)[0]
        im = ax.imshow(rel[order].T, cmap=cmap, norm=LogNorm(1, 3), aspect='auto')
        for si in range(len(d.shapes)):
            for j, mi in enumerate(order):
                win = rk[mi, si] == 1
                ax.text(j, si, f'{rk[mi, si]}', ha='center', va='center', fontsize=8.5,
                        color='white' if rel[mi, si] < 1.5 else TEXT,
                        fontweight='bold' if win else 'normal')
                if win:
                    ax.add_patch(plt.Rectangle((j - 0.5, si - 0.5), 1, 1, fill=False, ec='#00C853',
                                               lw=2))
        wins = [(rk[mi] == 1).sum() for mi in order]
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f'{SHORT[d.methods[mi]]}\n{w} wins' for mi, w in zip(order, wins)],
                           fontsize=8.5, color=TEXT)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(d.shapes)))
        ax.set_yticklabels(d.shapes, fontsize=8, color=TEXT)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(KERNEL_NAME[k], fontsize=11, color=TEXT, pad=42)
    cb = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.025, pad=0.03, aspect=45)
    plain_log_axis(cb.ax.xaxis, [1, 1.25, 1.5, 2, 3])
    cb.set_label('final ergodicity vs. GT relative to the best method on this shape '
                 '(1 = best, ≥ 3 = capped)', fontsize=8.5, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.suptitle('Winner per shape: numbers are ranks (1 = best, green frame), mean over 7 variants',
                 fontsize=10, color=MUTED, y=1.02)
    save(fig, out, '04_per_shape_heatmap.png')


def fig_rank_rounds(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor='white', sharey=True)
    for ax, k in zip(axes, KERNELS):
        style(ax)
        E = d.arr[k]['E_truth']                                     # [m, s, v, r]
        mean_rank = np.array([ranks(E[..., r], True).mean(axis=(1, 2))
                              for r in range(d.n_rounds)])          # [round, method]
        ends = mean_rank[-1].copy()
        label_y = ends.copy()
        srt = np.argsort(ends)                                # push labels apart (min. 0.17)
        for a, b in zip(srt[:-1], srt[1:]):
            label_y[b] = max(label_y[b], label_y[a] + 0.17)
        for mi, m in enumerate(d.methods):
            ax.plot(range(1, d.n_rounds + 1), mean_rank[:, mi], color=COLOR[m], lw=2.6,
                    marker='o', ms=7, mec='white', mew=1.5)
            ax.text(d.n_rounds + 0.15, label_y[mi], f'{SHORT[m]} ({ends[mi]:.2f})', color=COLOR[m],
                    fontsize=9, va='center', fontweight='bold')
        ax.set_xlim(0.8, d.n_rounds + 1.5)
        ax.set_xticks(range(1, d.n_rounds + 1))
        ax.set_ylim(len(d.methods) + 0.3, 0.7)
        ax.set_yticks(range(1, len(d.methods) + 1))
        ax.set_xlabel('exploration round', fontsize=9, color=MUTED)
        ax.set_title(KERNEL_NAME[k], fontsize=11, color=TEXT)
    axes[0].set_ylabel('mean rank in ergodicity vs. GT (1 = best)', fontsize=9, color=MUTED)
    fig.suptitle('Ranking over the mission: ergodicity of the path driven so far vs. ground truth',
                 fontsize=11, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, out, '05_rank_over_rounds.png')


def fig_efficiency(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(16 + 0.25 * len(d.methods), 4.8), facecolor='white',
                             gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    w = 0.38
    for ax, thr in zip(axes[:2], ('90', '99')):
        style(ax)
        for mi, m in enumerate(d.methods):
            for off, k in ((-w / 2, 'rbf'), (w / 2, 'ak')):
                reach = d.reach[k]['reached' + thr][mi]
                length = d.reach[k]['path_len_' + thr][mi]            # NaN if not reached
                sm = np.nanmean(length.reshape(-1))
                ax.bar(mi + off, sm, w * 0.92, color=COLOR[m], alpha=1.0 if k == 'rbf' else 0.55,
                       hatch=None if k == 'rbf' else '//', ec='white')
                ax.text(mi + off, sm + 0.3, f'{100 * np.nanmean(reach):.0f}%', ha='center',
                        fontsize=7.5, color=MUTED)
        ax.set_xticks(range(len(d.methods)))
        ax.set_xticklabels([SHORT[m].replace('Random walk', 'Random\nwalk') for m in d.methods],
                           fontsize=8.5, color=TEXT)
        ax.set_ylim(0, 29)
        ax.set_ylabel('path length until reached (mean over runs that reach it)', fontsize=8,
                      color=MUTED)
        ax.set_title(f'Path length until {thr} % of GT mass is swept (r = 0.06)', fontsize=10,
                     color=TEXT)
    axes[0].text(0.01, 0.99, 'solid = RBF, hatched = AK; labels: share of runs reaching the level',
                 transform=axes[0].transAxes, fontsize=7.5, color=MUTED, va='top')
    ax = axes[2]
    style(ax)
    for mi, m in enumerate(d.methods):
        for k, mk, fill in (('rbf', 'o', True), ('ak', 'D', False)):
            pl = np.nanmean(d.final(k, 'path_len')[mi], axis=1)
            e = np.nanmean(d.final(k, 'E_truth')[mi], axis=1)
            ax.errorbar(pl.mean(), e.mean(), xerr=1.96 * pl.std() / np.sqrt(len(pl)),
                        yerr=1.96 * e.std() / np.sqrt(len(e)), fmt=mk, color=COLOR[m], ms=9,
                        mfc=COLOR[m] if fill else 'white', mew=1.8, capsize=3, lw=1.4)
        ax.annotate(SHORT[m], (np.nanmean(d.final('ak', 'path_len')[mi]),
                               np.nanmean(d.final('ak', 'E_truth')[mi])), xytext=(10, {'dit630': 14, 'unet630': -16, 'old500': -10}.get(m, 6)),
                    textcoords='offset points', fontsize=9, color=COLOR[m], fontweight='bold')
    ax.set_yscale('log')
    plain_log_axis(ax.yaxis, [4, 6, 10, 15, 25, 40])
    ax.set_xlabel('total path length after round 5', fontsize=9, color=MUTED)
    ax.set_ylabel('final ergodicity vs. GT ↓ (log)', fontsize=9, color=MUTED)
    ax.set_title('Quality vs. path length (bottom-left = best)', fontsize=10, color=TEXT)
    h = [plt.Line2D([], [], marker='o', color=MUTED, ls='', ms=7, label='RBF'),
         plt.Line2D([], [], marker='D', color=MUTED, ls='', mfc='white', mew=1.8, ms=7,
                    label='AK')]
    ax.legend(handles=h, frameon=False, fontsize=8.5, loc='lower right')
    fig.tight_layout()
    save(fig, out, '06_efficiency.png')


def fig_distributions(d, out):
    panels = [('E_truth', 'Final ergodicity vs. GT ↓ (log)', True),
              ('belief_rmse', 'Final belief RMSE vs. GT ↓', False)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor='white')
    rng = np.random.default_rng(0)
    order = overall_order(d, 'ak')[0]
    for i, k in enumerate(KERNELS):
        for j, (c, title, log) in enumerate(panels):
            ax = axes[i, j]
            style(ax)
            for pos, mi in enumerate(order):
                v = d.final(k, c)[mi].reshape(-1)
                col = COLOR[d.methods[mi]]
                ax.scatter(pos + rng.uniform(-0.3, 0.3, len(v)), v, s=7, color=col, alpha=0.35,
                           lw=0)
                bp = ax.boxplot(v, positions=[pos], widths=0.55, showfliers=False,
                                patch_artist=True, medianprops=dict(color=TEXT, lw=2),
                                whiskerprops=dict(color=col), capprops=dict(color=col),
                                boxprops=dict(edgecolor=col, facecolor='none', lw=1.6))
            if log:
                ax.set_yscale('log')
                plain_log_axis(ax.yaxis, [0.6, 1, 2, 4, 10, 20, 50, 100])
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([SHORT[d.methods[m]] for m in order], fontsize=9, color=TEXT)
            ax.set_title(f'{title} – {KERNEL_NAME[k]}', fontsize=10, color=TEXT)
    fig.suptitle(f'Every dot is one run ({len(d.shapes)} shapes × {d.n_var} variants); box: quartiles, line: median',
                 fontsize=11, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, out, '07_distributions.png')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'plots', 'exploration_comparison'),
                   help='Folder with metrics_all.csv (the `extended` sub-folder for the full method set).')
    p.add_argument('--db', default=None,
                   help='Comparison DB (default: exploration_comparison.db in --dir or its parent).')
    p.add_argument('--methods', default=None,
                   help='Comma-separated subset to show (default: every method found).')
    a = p.parse_args()
    out = os.path.join(a.dir, 'summary')
    os.makedirs(out, exist_ok=True)
    db = a.db
    if db is None:
        for cand in (os.path.join(a.dir, 'exploration_comparison.db'),
                     os.path.join(os.path.dirname(a.dir), 'exploration_comparison.db')):
            if os.path.isfile(cand):
                db = cand
                break
    d = Data(a.dir, db, only=a.methods.split(',') if a.methods else None)
    print(f'{len(d.methods)} methods, {len(d.shapes)} shapes, {d.n_var} variants, '
          f'{d.n_rounds} rounds')
    fig_scorecard(d, out)
    fig_winrate(d, out)
    fig_dots(d, out)
    fig_per_shape(d, out)
    fig_rank_rounds(d, out)
    fig_efficiency(d, out)
    fig_distributions(d, out)


if __name__ == '__main__':
    main()
