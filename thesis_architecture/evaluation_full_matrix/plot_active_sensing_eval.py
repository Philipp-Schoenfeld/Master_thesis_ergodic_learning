r"""
plot_active_sensing_eval.py
===========================
Evaluation of a belief-comparison run of `run_mission_eval.py --beliefs ...`:
which active-sensing model (RBF-GP, Attentive-Kernel GP, neural process)
works best together with which initialiser (start-conditioned CFM, linear
rays, self-supervised generator)?

Reads <out_root>/<out_tag>/belief_<name>/shards/*.db and writes

  tables_active_sensing/rounds.csv         one row per (cond, belief, method, shape, round)
  tables_active_sensing/per_round.csv      mean / SEM over shapes per round
  tables_active_sensing/final.csv          mean / SEM over shapes after the last round
  plots_active_sensing/belief_quality_<cond>.png
  plots_active_sensing/mission_<cond>.png
  plots_active_sensing/final_<cond>.png

Shapes that reached the coverage threshold stop early; their last values are
carried forward to later rounds, so per-round means always cover all shapes.
Round 0 of the belief plots is the belief before any mission measurement.
`sd_mean` / `nll` / `cov95` show whether sigma is calibrated -- the beliefs'
sigma scales differ (the NP predicts a much smaller sigma), so the stored
`info_gain` (sum of sigma) is not comparable across beliefs and not plotted.

    python plot_active_sensing_eval.py --out_tag active_sensing_20261023_tsvec_svgd
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

import mission_db as mdb          # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))

BELIEFS = ('rbf', 'ak', 'np')
METHODS = ('cfm', 'linear', 'selfsup')
BELIEF_LABEL = {'rbf': 'RBF-GP', 'ak': 'Attentive-Kernel GP', 'np': 'Neural process'}
METHOD_LABEL = {'cfm': 'CFM (start-conditioned)', 'linear': 'Linear init',
                'selfsup': 'Self-supervised'}
COND_LABEL = {'none_known': 'no prior knowledge', 'ten_samples': 'ten prior samples'}
# Categorical slots 1-3 of the dataviz reference palette (validated, light mode);
# markers are the secondary encoding (aqua is below 3:1 contrast on white).
COLOR = {'rbf': '#2a78d6', 'ak': '#1baf7a', 'np': '#eb6834'}
MARKER = {'rbf': 'o', 'ak': 's', 'np': '^'}
TEXT, MUTED, SPINE = '#1A1A2E', '#555', '#ccc'

ROUND_COLS = ('shape', 'round', 'n_exec', 'E_truth', 'swept_mass', 'reached99', 'path_len')
BELIEF_COLS = ('rmse', 'corr', 'nll', 'cov95', 'sd_mean', 'update_s')

BELIEF_PANELS = (('corr', 'Belief correlation with truth (higher is better)', None),
                 ('rmse', 'Belief RMSE (lower is better)', None),
                 ('nll', 'NLL of truth under belief (lower is better)', None),
                 ('cov95', 'Truth inside 95% interval (target 0.95)', 0.95))
MISSION_PANELS = (('E_truth', 'Ergodic error vs. truth (lower is better)', None),
                  ('swept_mass', 'Swept true mass (higher is better)', 0.99),
                  ('path_len', 'Driven path length', None))
FINAL_PANELS = (('E_truth', 'Ergodic error vs. truth'),
                ('swept_mass', 'Swept true mass'),
                ('corr', 'Belief correlation'),
                ('nll', 'Belief NLL'))


def style(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=8)


# ── loading ──────────────────────────────────────────────────────────────────

def load(run_dir):
    """-> list of dicts, one per (cond, belief, method, shape, round), with the
    mission scalars and the belief scores after that round (round -1 = the
    initial belief, mission scalars NaN)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, 'belief_*', 'shards', '*.db'))):
        belief = os.path.basename(os.path.dirname(os.path.dirname(path)))[len('belief_'):]
        cond, strat, method = os.path.basename(path)[:-3].split('__')
        conn = mdb.open_db(path)
        mission = {(d['shape'], d['round']): d for d in mdb.iter_rounds(conn, columns=ROUND_COLS)}
        for bm in mdb.iter_belief_metrics(conn):
            key = (bm['shape'], bm['round'])
            if bm['round'] >= 0 and key not in mission:
                continue                       # metrics of a round whose commit never happened
            m = mission.get(key, {})
            rows.append(dict(cond=cond, strategy=strat, belief=belief, method=method,
                             shape=bm['shape'], round=bm['round'],
                             **{c: m.get(c, np.nan) for c in ROUND_COLS[2:]},
                             **{c: bm[c] for c in BELIEF_COLS}))
        conn.close()
    return rows


def curves(rows, cond, belief, method, metric, rounds):
    """(n_shapes, len(rounds)) with last-value carry-forward for finished shapes."""
    by_shape = {}
    for r in rows:
        if (r['cond'], r['belief'], r['method']) == (cond, belief, method):
            by_shape.setdefault(r['shape'], {})[r['round']] = r[metric]
    out = []
    for vals in by_shape.values():
        line, last = [], np.nan
        for rd in rounds:
            v = vals.get(rd)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                last = v
            line.append(last)
        out.append(line)
    return np.array(out, dtype=np.float64)


def mean_sem(a):
    n = np.sum(~np.isnan(a), axis=0)
    mean = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mean)
    return mean, sd / np.sqrt(np.maximum(n, 1)), n


# ── tables ───────────────────────────────────────────────────────────────────

def write_tables(rows, run_dir, conds, max_round):
    tdir = os.path.join(run_dir, 'tables_active_sensing')
    os.makedirs(tdir, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(tdir, 'rounds.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    metrics = ('E_truth', 'swept_mass', 'path_len') + BELIEF_COLS
    rounds = list(range(-1, max_round + 1))
    per_round, final = [], []
    for cond in conds:
        for belief in BELIEFS:
            for method in METHODS:
                if not any((r['cond'], r['belief'], r['method']) == (cond, belief, method)
                           for r in rows):
                    continue
                stats = {m: mean_sem(curves(rows, cond, belief, method, m, rounds))
                         for m in metrics}
                for i, rd in enumerate(rounds):
                    d = dict(cond=cond, belief=belief, method=method, round=rd)
                    for m in metrics:
                        d[f'{m}_mean'], d[f'{m}_sem'] = stats[m][0][i], stats[m][1][i]
                    per_round.append(d)
                d = dict(cond=cond, belief=belief, method=method,
                         n_shapes=int(stats['E_truth'][2][-1]))
                for m in metrics:
                    d[f'{m}_mean'], d[f'{m}_sem'] = stats[m][0][-1], stats[m][1][-1]
                reached = [r for r in rows if (r['cond'], r['belief'], r['method']) ==
                           (cond, belief, method) and r['reached99'] == 1]
                first = {}
                for r in reached:
                    first[r['shape']] = min(first.get(r['shape'], 99), r['n_exec'])
                d['frac_reached99'] = len(first) / max(d['n_shapes'], 1)
                d['rounds_to_99_mean'] = float(np.mean(list(first.values()))) if first else np.nan
                final.append(d)
    for name, table in (('per_round.csv', per_round), ('final.csv', final)):
        with open(os.path.join(tdir, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
    return final


# ── figures ──────────────────────────────────────────────────────────────────

def legend_handles():
    return [plt.Line2D([], [], color=COLOR[b], marker=MARKER[b], lw=2, ms=7,
                       label=BELIEF_LABEL[b]) for b in BELIEFS]


def line_grid(rows, cond, panels, rounds, xlabels, title, path):
    methods = [m for m in METHODS if any(r['method'] == m and r['cond'] == cond for r in rows)]
    fig, axes = plt.subplots(len(panels), len(methods), figsize=(4.2 * len(methods),
                             2.9 * len(panels)), facecolor='white', squeeze=False)
    for j, method in enumerate(methods):
        for i, (metric, ylabel, ref) in enumerate(panels):
            ax = axes[i, j]
            style(ax)
            for belief in BELIEFS:
                a = curves(rows, cond, belief, method, metric, rounds)
                if a.size == 0:
                    continue
                mean, sem, _ = mean_sem(a)
                x = np.arange(len(rounds))
                ax.plot(x, mean, color=COLOR[belief], marker=MARKER[belief], lw=2, ms=6)
                ax.fill_between(x, mean - sem, mean + sem, color=COLOR[belief], alpha=0.15, lw=0)
            if ref is not None:
                ax.axhline(ref, color=MUTED, lw=1, ls='--')
            ax.set_xticks(np.arange(len(rounds)))
            ax.set_xticklabels(xlabels)
            if i == 0:
                ax.set_title(METHOD_LABEL[method], fontsize=11, color=TEXT)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8.5, color=TEXT)
            if i == len(panels) - 1:
                ax.set_xlabel('Executed rounds', fontsize=9, color=MUTED)
    fig.suptitle(title, fontsize=13, color=TEXT)
    fig.legend(handles=legend_handles(), loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965), fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


def final_figure(final, cond, path):
    rows = [d for d in final if d['cond'] == cond]
    methods = [m for m in METHODS if any(d['method'] == m for d in rows)]
    fig, axes = plt.subplots(1, len(FINAL_PANELS), figsize=(3.6 * len(FINAL_PANELS), 3.6),
                             facecolor='white')
    offs = {'rbf': -0.22, 'ak': 0.0, 'np': 0.22}
    for ax, (metric, title) in zip(axes, FINAL_PANELS):
        style(ax)
        for d in rows:
            x = methods.index(d['method']) + offs[d['belief']]
            ax.errorbar(x, d[f'{metric}_mean'], yerr=d[f'{metric}_sem'], color=COLOR[d['belief']],
                        marker=MARKER[d['belief']], ms=7, lw=1.5, capsize=3)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABEL[m].split(' (')[0] for m in methods], fontsize=8.5)
        ax.set_title(title, fontsize=10.5, color=TEXT)
    fig.suptitle(f'After the last round, {COND_LABEL.get(cond, cond)} (mean ± SEM over shapes)',
                 fontsize=12, color=TEXT)
    fig.legend(handles=legend_handles(), loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.93), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True,
                    help='Result folder of the run, including the refiner suffix.')
    ap.add_argument('--out_root', default=os.path.join(_here, 'results'))
    a = ap.parse_args()
    run_dir = os.path.join(a.out_root, a.out_tag)
    rows = load(run_dir)
    if not rows:
        raise SystemExit(f'no belief_*/shards/*.db with belief metrics under {run_dir}')
    conds = [c for c in ('none_known', 'ten_samples') if any(r['cond'] == c for r in rows)]
    max_round = max(r['round'] for r in rows)
    final = write_tables(rows, run_dir, conds, max_round)

    pdir = os.path.join(run_dir, 'plots_active_sensing')
    os.makedirs(pdir, exist_ok=True)
    cfg_path = os.path.join(run_dir, 'config.json')
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    kap = (f", kappa {cfg['kappa0']:g} → {cfg['kappa1']:g}"
           if cfg.get('kappa0') is not None else '')
    for cond in conds:
        rb = list(range(-1, max_round + 1))
        line_grid(rows, cond, BELIEF_PANELS, rb, ['0'] + [str(r + 1) for r in rb[1:]],
                  f'Belief quality, {COND_LABEL.get(cond, cond)}{kap} (mean ± SEM over shapes)',
                  os.path.join(pdir, f'belief_quality_{cond}.png'))
        rm = list(range(0, max_round + 1))
        line_grid(rows, cond, MISSION_PANELS, rm, [str(r + 1) for r in rm],
                  f'Mission progress, {COND_LABEL.get(cond, cond)}{kap} (mean ± SEM over shapes)',
                  os.path.join(pdir, f'mission_{cond}.png'))
        final_figure(final, cond, os.path.join(pdir, f'final_{cond}.png'))
    print(f'  {len(rows)} rows -> {pdir}, {os.path.join(run_dir, "tables_active_sensing")}')


if __name__ == '__main__':
    main()
