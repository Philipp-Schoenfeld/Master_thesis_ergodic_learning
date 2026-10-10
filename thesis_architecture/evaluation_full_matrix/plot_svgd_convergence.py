r"""
plot_svgd_convergence.py
=========================
Figures + summary table for `run_svgd_convergence.py`, read straight from its
SQLite DB (works on partial DBs -- whatever is stored is plotted).

Per method the curve is the mean ergodic error (against the TRUE density) over
the n_init initialisations at every SVGD iteration; the band is +-1 standard
deviation across those initialisations. Outputs under
`results/<out_tag>/plots/`:

  overview_<metric>.png      strategies (rows) x knowledge states (columns);
                             mean over shapes of the per-shape mean, band =
                             mean over shapes of the per-shape std
  per_shape/<cond>_<strategy>_<metric>.png
                             one panel per holdout shape (not averaged)
  convergence_summary.csv    per (knowledge, strategy, method): mean/std of the
                             error at selected iterations, averaged over shapes

If `extend_svgd_convergence.py` has added continuation states (table
`runs_ext`), they are appended automatically and the outputs get an
`_<N>iters` suffix (overview_E_total_3000iters.png, per_shape_3000iters/,
convergence_summary_3000iters.csv) so the 1000-iteration files stay untouched;
`--metric J` plots the evaluation matrix's own J = E_total + 0.02 * path_len
(`metrics_explore_exploit.add_J`), derived per iteration from the stored
E_total and path_len arrays; non-default metrics get the metric name in the
output file names (overview_J_3000iters.png, convergence_summary_J_3000iters.csv).
`--max_iter N` truncates the curves (e.g. `--max_iter 1000` rebuilds the
original figures).

Usage:
    python plot_svgd_convergence.py --out_tag svgd_convergence_YYYYMMDD
    python plot_svgd_convergence.py --out_tag smoke --metric E_explore
"""
import argparse
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_arch, 'exploration'),
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402

import svgd_convergence_db as sdb                                    # noqa: E402
from metrics_explore_exploit import LAMBDA_LEN_J                     # noqa: E402

METRICS = {'E_total': 'Ergodic error E (total)',
           'J': f'J = E + {LAMBDA_LEN_J:g} * path length',
           'E_explore': 'Ergodic error E (low-frequency band)',
           'E_exploit': 'Ergodic error E (high-frequency band)'}
AXIS_LABEL = {'E_total': 'Ergodic error E', 'E_explore': 'E (low-frequency band)',
              'E_exploit': 'E (high-frequency band)', 'J': 'J'}
METHOD_STYLE = {                       # CFM = generated trajectories (project style)
    'cfm': dict(color='#00C853', label='CFM warm start', lw=2.0),
    'selfsup': dict(color='#1565C0', label='Self-supervised (single-pass)', lw=1.8),
    'random_walk': dict(color='#E65100', label='Random walk', lw=1.6),
    'linear': dict(color='#6A1B9A', label='Linear (angled lines)', lw=1.6),
}
STRATEGY_TITLE = {'lse': 'LSE (level set)', 'ucb': 'UCB', 'eid': 'EID'}
COND_TITLE = {'ground_truth': 'Ground truth known', 'half_known': 'Half known',
              'ten_samples': 'Ten samples known', 'none_known': 'None known'}
COND_ORDER = ['ground_truth', 'half_known', 'ten_samples', 'none_known']
STRAT_ORDER = ['lse', 'ucb', 'eid']
SHARED = sdb.SHARED
INK, MUTED, SPINE = '#1A1A2E', '#555555', '#cccccc'
SUMMARY_ITERS = (0, 10, 25, 100, 250, 500, 1000, 1500, 2000, 3000)
FIRST_RUN_ITERS = 1000      # marks where the original run ended (extension starts)


def style_axes(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=8)


def broadcast_shared(out):
    """Baseline entries stored once for all knowledge states / strategies
    (key component 'all', truth-target mode) are copied into every
    (knowledge, strategy) panel. `out`: {(cond, strat, method): value}."""
    panels = {(k[0], k[1]) for k in out if SHARED not in (k[0], k[1])} or         {(c, s) for c in COND_ORDER for s in STRAT_ORDER}
    for key in [k for k in out if SHARED in (k[0], k[1])]:
        for (c, st) in panels:
            out[(c, st, key[2])] = out[key]
        del out[key]
    return out


def load_arrays(conn, metric, max_iter=None):
    """-> {(cond, strat, method): {shape: (n_init, iters+1) array}}"""
    runs = []
    cols = ('E_total', 'path_len') if metric == 'J' else (metric,)
    for r in sdb.iter_runs(conn, columns=cols, include_ext=True):
        full = (r['E_total'] + LAMBDA_LEN_J * r['path_len']) if metric == 'J' else r[metric]
        arr = full if max_iter is None else full[:max_iter + 1]
        runs.append((r['knowledge_condition'], r['strategy'], r['method'],
                     r['shape'], r['init_idx'], arr))
    # A partially extended DB has runs of different length: use the common one.
    if runs:
        lens = [len(a) for *_k, a in runs]
        common = min(lens)
        n_short = sum(1 for n in lens if n == common)
        if common != max(lens):
            print(f"[plot_svgd_convergence] WARNING: {n_short} of {len(runs)} runs have only "
                  f"{common - 1} iterations (extension incomplete); truncating all curves "
                  f"to {common - 1}.")
        runs = [(*k, a[:common]) for *k, a in runs]
    data = {}
    for c, st, m, shape, idx, arr in runs:
        data.setdefault((c, st, m), {}).setdefault(shape, {})[idx] = arr
    out = {}
    for key, per_shape in data.items():
        out[key] = {s: np.stack([d[i] for i in sorted(d)]) for s, d in per_shape.items()}
    return broadcast_shared(out)


def mark_first_run_end(ax, n_iters):
    """Faint vertical line where the original 1000-iteration run ended."""
    if n_iters > FIRST_RUN_ITERS:
        ax.axvline(FIRST_RUN_ITERS, color=MUTED, lw=0.8, ls=':', alpha=0.6)


def band(ax, x, mean, std, style):
    lo = np.maximum(mean - std, mean * 0.02)        # keep the band positive on the log axis
    ax.plot(x, mean, color=style['color'], lw=style['lw'], label=style['label'], alpha=0.95)
    ax.fill_between(x, lo, mean + std, color=style['color'], alpha=0.18, linewidth=0)


def plot_overview(data, metric, out_path, n_init_note):
    conds = [c for c in COND_ORDER if any(k[0] == c for k in data)]
    strats = [s for s in STRAT_ORDER if any(k[1] == s for k in data)]
    if not conds or not strats:
        return
    fig, axes = plt.subplots(len(strats), len(conds), figsize=(4.4 * len(conds), 3.4 * len(strats)),
                             facecolor='white', squeeze=False, sharex=True)
    for i, st in enumerate(strats):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            style_axes(ax)
            for m, sty in METHOD_STYLE.items():
                per_shape = data.get((c, st, m))
                if not per_shape:
                    continue
                means = np.stack([a.mean(axis=0) for a in per_shape.values()])
                stds = np.stack([a.std(axis=0) for a in per_shape.values()])
                x = np.arange(means.shape[1])
                band(ax, x, means.mean(axis=0), stds.mean(axis=0), sty)
            mark_first_run_end(ax, len(x) - 1)
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(COND_TITLE[c], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{STRATEGY_TITLE[st]}\n{METRICS[metric]}", fontsize=9, color=INK)
            if i == len(strats) - 1:
                ax.set_xlabel('SVGD iteration', fontsize=9, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=False,
                   fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    n_shapes = len({s for v in data.values() for s in v})
    fig.suptitle(f"{METRICS[metric]} vs. SVGD iterations -- mean over {n_shapes} holdout shapes; "
                 f"band: +-1 std across initialisations{n_init_note}",
                 fontsize=10, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_per_shape(data, metric, cond, strat, out_path):
    shapes = sorted({s for k, v in data.items() if k[0] == cond and k[1] == strat for s in v})
    if not shapes:
        return
    ncols = 4
    nrows = int(np.ceil(len(shapes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows),
                             facecolor='white', squeeze=False)
    for k, shape in enumerate(shapes):
        ax = axes[k // ncols][k % ncols]
        style_axes(ax)
        for m, sty in METHOD_STYLE.items():
            arr = data.get((cond, strat, m), {}).get(shape)
            if arr is None:
                continue
            x = np.arange(arr.shape[1])
            band(ax, x, arr.mean(axis=0), arr.std(axis=0), sty)
            mark_first_run_end(ax, len(x) - 1)
        ax.set_yscale('log')
        ax.set_title(shape, fontsize=9, color=INK)
        if k // ncols == nrows - 1:
            ax.set_xlabel('SVGD iteration', fontsize=8, color=INK)
        if k % ncols == 0:
            ax.set_ylabel(AXIS_LABEL[metric], fontsize=8, color=INK)
    for k in range(len(shapes), nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=False,
                   fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{COND_TITLE[cond]} -- {STRATEGY_TITLE[strat]} -- {METRICS[metric]} "
                 f"(mean +- 1 std across initialisations)", fontsize=10, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def write_summary(data, out_path, metric='E_total'):
    pre = 'E' if metric == 'E_total' else metric
    rows = []
    for (c, st, m), per_shape in sorted(data.items()):
        means = np.stack([a.mean(axis=0) for a in per_shape.values()])
        stds = np.stack([a.std(axis=0) for a in per_shape.values()])
        n_iters = means.shape[1] - 1
        row = {'knowledge_condition': c, 'strategy': st, 'method': m,
               'n_shapes': len(per_shape),
               'n_init': int(min(a.shape[0] for a in per_shape.values()))}
        for it in SUMMARY_ITERS:
            if it <= n_iters:
                row[f'mean_{pre}_iter{it}'] = float(means[:, it].mean())
                row[f'std_{pre}_iter{it}'] = float(stds[:, it].mean())
        rows.append(row)
    keys = ['knowledge_condition', 'strategy', 'method', 'n_shapes', 'n_init'] +         [f'{p}_{pre}_iter{it}' for it in SUMMARY_ITERS for p in ('mean', 'std')]
    keys = [k for k in keys if any(k in r for r in rows)]
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--metric', type=str, default='E_total', choices=sorted(METRICS))
    ap.add_argument('--no_per_shape', action='store_true')
    ap.add_argument('--max_iter', type=int, default=None,
                    help='Truncate the curves at this iteration (default: everything '
                         'stored, including extension states).')
    args = ap.parse_args()

    out_dir = os.path.join(_here, 'results', args.out_tag)
    conn = sdb.open_db(os.path.join(out_dir, 'svgd_convergence.db'))
    cfg = sdb.get_meta(conn, 'config', {})
    data = load_arrays(conn, args.metric, args.max_iter)
    if not data:
        print("[plot_svgd_convergence] DB holds no runs yet.")
        return
    n_iters = min(a.shape[1] for v in data.values() for a in v.values()) - 1
    sfx = f'_{n_iters}iters' if n_iters > FIRST_RUN_ITERS else ''
    plots = os.path.join(out_dir, 'plots')
    os.makedirs(plots, exist_ok=True)
    note = f" (n={cfg['n_init']})" if cfg.get('n_init') else ''
    plot_overview(data, args.metric, os.path.join(plots, f'overview_{args.metric}{sfx}.png'), note)
    if not args.no_per_shape:
        shape_dir = os.path.join(plots, f'per_shape{sfx}')
        os.makedirs(shape_dir, exist_ok=True)
        for c in COND_ORDER:
            for st in STRAT_ORDER:
                plot_per_shape(data, args.metric, c, st,
                               os.path.join(shape_dir, f'{c}_{st}_{args.metric}.png'))
    msfx = '' if args.metric == 'E_total' else f'_{args.metric}'
    write_summary(data, os.path.join(out_dir, f'convergence_summary{msfx}{sfx}.csv'),
                  args.metric)
    print(f"[plot_svgd_convergence] wrote figures to {plots} and "
          f"convergence_summary{msfx}{sfx}.csv ({n_iters} iterations)")


if __name__ == '__main__':
    main()
