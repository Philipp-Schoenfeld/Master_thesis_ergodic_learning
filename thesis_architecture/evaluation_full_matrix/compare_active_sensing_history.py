"""
Puts the new active-sensing runs (as_models_20261028) next to the earlier ones, using only the stored
mission results -- nothing is recomputed.

All runs use the same 25 holdout shapes, UCB target with kappa 1.0 -> 0.05, full execution, <= 5 rounds
and the TSVEC-SVGD refiner, but differ in refinement iterations, candidates per round and belief:

  setting                                   iters  cands  belief  source
  50 it, rbf (kappa_schedule_small)            50      5   rbf    mission_kappa_schedule_small_20261022_tsvec_svgd (shards)
  100 it, ak (as_models)                      100     30   ak     as_models_20261028_<label>_tsvec_svgd (rounds.csv)
  1500 it, rbf (kappa_schedule)              1500     30   rbf    mission_kappa_schedule_20261021_tsvec_svgd (shards)
  1500 it, rbf / ak / np (active_sensing)    1500     30   ...    active_sensing_20261023_tsvec_svgd (rounds.csv)

Because absolute energies depend on the belief and the budget, every planner is compared to the linear
rays of ITS OWN run (paired over shapes), at equal cumulative path length. Output:
results/<tag>/history/{report.md, summary.json, history_vs_linear.png, budget_trend.png}.

    python compare_active_sensing_history.py --tag as_models_20261028
"""

import argparse
import glob
import json
import os
import sqlite3

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_active_sensing_models import MODELS, boot, interp_at, final_rows, style, INK, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
COND = {'none_known': 'no prior samples', 'ten_samples': 'ten known samples'}
PATH_L = 15
MIN_N = 12          # cells whose equal-path value rests on fewer shapes are listed in the report but not plotted

# (setting label, iterations, candidates, belief) for the earlier runs
OLD_RUNS = [
    dict(key='k50', setting='50 it · rbf', iters=50, cands=5, belief='rbf', kind='db',
         dir='mission_kappa_schedule_small_20261022_tsvec_svgd'),
    dict(key='k1500', setting='1500 it · rbf', iters=1500, cands=30, belief='rbf', kind='db',
         dir='mission_kappa_schedule_20261021_tsvec_svgd'),
    dict(key='a1500rbf', setting='1500 it · rbf (23.10.)', iters=1500, cands=30, belief='rbf', kind='csv',
         dir='active_sensing_20261023_tsvec_svgd'),
    dict(key='a1500ak', setting='1500 it · ak', iters=1500, cands=30, belief='ak', kind='csv',
         dir='active_sensing_20261023_tsvec_svgd'),
]
# display name, colour, marker for the methods that only exist in the older runs
OTHER = {
    'cfm': ('old start model (default ckpt)', '#7FDFA3', 'o'),
    'selfsup': ('self-supervised init', '#546E7A', 'h'),
    'ebm': ('EBM init', '#8D6E63', 'p'),
    'random_walk': ('random walk', '#9E9E9E', '<'),
    'heuristic': ('heuristic', '#BDBDBD', '>'),
    'linear': ('linear rays', '#888888', 'x'),
}


def load_db(d):
    frames = []
    for f in sorted(glob.glob(os.path.join(RESULTS, d, 'shards', '*.db'))):
        con = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
        frames.append(pd.read_sql_query(
            'select knowledge_condition as cond, method, shape, round, E_truth, path_len, swept_mass from rounds', con))
        con.close()
    return pd.concat(frames, ignore_index=True)


def load_csv(d, belief):
    df = pd.read_csv(os.path.join(RESULTS, d, 'tables_active_sensing', 'rounds.csv'))
    df = df[df['round'] >= 0]
    return df[df.belief == belief] if 'belief' in df else df


def collect(tag):
    """-> list of dicts(setting, iters, cands, belief, df, methods) with df columns cond, method, shape, round, E_truth, path_len."""
    runs = []
    for r in OLD_RUNS:
        p = os.path.join(RESULTS, r['dir'])
        if not os.path.isdir(p):
            print(f"[hist] {r['dir']} missing -- skipped")
            continue
        df = load_db(r['dir']) if r['kind'] == 'db' else load_csv(r['dir'], r['belief'])
        df = df[df.cond.isin(COND)]
        runs.append(dict(r, df=df))
    # the new run: one directory per model label; keep the linear rays of the dit630 run
    parts = []
    for lab in MODELS:
        if lab == 'linear':
            continue
        p = os.path.join(RESULTS, f'{tag}_{lab}_tsvec_svgd', 'tables_active_sensing', 'rounds.csv')
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d['round'] >= 0]
        d = d.assign(method=np.where(d.method == 'linear', 'linear', lab))
        parts.append(d)
    new = pd.concat(parts, ignore_index=True)
    runs.append(dict(key='new', setting='100 it · ak (new)', iters=100, cands=30, belief='ak', kind='csv', dir=tag, df=new))
    return runs


def label_of(method, run):
    if run['key'] == 'new':
        return MODELS[method][0], MODELS[method][1], MODELS[method][2]
    return OTHER[method]


def per_method(run):
    """method -> cond -> dict(E_final, sem, d_final, d_path (paired vs linear), n)."""
    df = run['df']
    out = {}
    for cond in COND:
        d = df[df.cond == cond]
        lin = d[d.method == 'linear']
        lin_f = final_rows(lin).set_index('shape').E_truth
        lin_p = interp_at(lin.assign(cond=cond), cond, PATH_L)
        for m in sorted(set(d.method)):
            dm = d[d.method == m]
            f = final_rows(dm).set_index('shape').E_truth
            p = interp_at(dm.assign(cond=cond), cond, PATH_L)
            df_ = (f - lin_f.reindex(f.index)).dropna()
            dp_ = (p - lin_p.reindex(p.index)).dropna()
            out.setdefault(m, {})[cond] = dict(
                E_final=float(f.mean()), sem=float(f.std(ddof=1) / np.sqrt(len(f))), n=int(len(f)),
                d_final=boot(df_) if m != 'linear' and len(df_) > 2 else (0.0, 0.0, 0.0),
                d_path=boot(dp_) if m != 'linear' and len(dp_) > 2 else (0.0, 0.0, 0.0),
                n_path=int(len(dp_)), E_path=float(p.mean()) if len(p) else float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='as_models_20261028')
    a = ap.parse_args()
    out = os.path.join(RESULTS, a.tag, 'history')
    os.makedirs(out, exist_ok=True)
    runs = collect(a.tag)
    summ = {}
    L = ['# Active sensing: new models next to the earlier runs', '',
         'Same 25 shapes, UCB (kappa 1.0 -> 0.05), full execution, <= 5 rounds, TSVEC-SVGD. Every planner is compared to '
         'the linear rays of its own run (paired over shapes). E = ergodic energy against the true density (lower is better). '
         f'Equal-path column: E at cumulative path length {PATH_L}, interpolated between rounds. Δ = planner − linear rays '
         '(95 % bootstrap CI over shapes); negative = better than the rays.', '']
    for run in runs:
        res = summ[run['setting']] = per_method(run)
        L += [f"## {run['setting']}  (iterations {run['iters']}, candidates {run['cands']}, belief {run['belief']})", '',
              '| condition | method | E final | Δ final vs rays | E at L=%d | Δ at L=%d vs rays | n (equal path) |' % (PATH_L, PATH_L),
              '|---|---|---|---|---|---|---|']
        for cond in COND:
            for m in res:
                c = res[m][cond]
                nm = label_of(m, run)[0]
                dfin = '–' if m == 'linear' else '%+.2f [%+.2f, %+.2f]' % c['d_final']
                dp = '–' if m == 'linear' else '%+.2f [%+.2f, %+.2f]' % c['d_path']
                L.append(f"| {COND[cond]} | {nm} | {c['E_final']:.2f} ± {c['sem']:.2f} | {dfin} | {c['E_path']:.2f} | {dp} | {c['n_path']} |")
        L.append('')
    L += [f'Cells with n < {MIN_N} shapes in the equal-path column are not plotted (they cover too few shapes).', '']
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L))
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=float)
    print('\n'.join(L))

    # ------------------------------------------------------------ plot 1: forest, Δ at equal path length vs rays
    fig, axes = plt.subplots(1, 2, figsize=(12, 9.5), facecolor='white', sharey=True)
    rows = []                       # (run index, method)
    for ri, run in enumerate(runs):
        for m in summ[run['setting']]:
            if m != 'linear' and min(summ[run['setting']][m][c]['n_path'] for c in COND) >= MIN_N:
                rows.append((ri, m))
    ypos, y, last = {}, 0.0, None
    for ri, m in rows:
        if last is not None and ri != last:
            y += 0.9
        ypos[(ri, m)] = y
        y += 1.0
        last = ri
    for ax, cond in zip(axes, COND):
        style(ax)
        for (ri, m), yy in ypos.items():
            run = runs[ri]
            c = summ[run['setting']][m][cond]
            mean, lo, hi = c['d_path']
            _, col, mk = label_of(m, run)
            ax.errorbar(mean, yy, xerr=[[mean - lo], [hi - mean]], color=col, marker=mk, ms=7, lw=2, capsize=3,
                        mec='#444444' if col in ('#7FDFA3', '#BDBDBD') else col)
        ax.axvline(0, color='#888888', lw=1)
        ax.set_xlabel(f'Δ energy at path length {PATH_L} vs. linear rays of the same run (negative = better)', fontsize=8.5, color=MUTED)
        ax.set_title(COND[cond], fontsize=10.5, color=INK, loc='left')
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([label_of(m, runs[ri])[0] for (ri, m) in ypos], fontsize=8, color=INK)
    axes[0].invert_yaxis()
    ymin = {}
    for (ri, m), yy in ypos.items():
        ymin.setdefault(ri, [yy, yy])
        ymin[ri][1] = yy
    for ri, (y0, y1) in ymin.items():
        for ax in axes:
            ax.axhspan(y0 - 0.5, y1 + 0.5, color='#f4f4f4' if ri % 2 == 0 else 'white', zorder=0)
        axes[0].text(axes[0].get_xlim()[0], y0 - 0.62, runs[ri]['setting'] + f" · {runs[ri]['cands']} candidates", fontsize=8.5,
                     color=INK, fontweight='bold', va='bottom')
    fig.suptitle('Every planner against the linear rays of its own run (25 shapes, paired, 95 % CI)', fontsize=11.5, color=INK, y=0.995)
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'history_vs_linear.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ------------------------------------------------------------ plot 2: budget trend of the old start model
    # The same checkpoint ('cfm' in the old runs, 'old500' in the new one) under growing budget; the two
    # beliefs are separate lines because absolute energies are not comparable across beliefs.
    byk = {r['key']: r for r in runs}
    series = [('old start model, belief rbf', [('k50', 'cfm'), ('k1500', 'cfm')], '#7FDFA3', '-', 'o'),
              ('old start model, belief ak', [('new', 'old500'), ('a1500ak', 'cfm')], '#2E7D32', '--', 's'),
              ('self-supervised init, belief rbf', [('k50', 'selfsup'), ('k1500', 'selfsup')], '#546E7A', '-', 'h'),
              ('random walk, belief rbf', [('k50', 'random_walk'), ('k1500', 'random_walk')], '#9E9E9E', '-', '<')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='white', sharey=True)
    for ax, cond in zip(axes, COND):
        style(ax)
        for si, (name, pts, col, ls, mk) in enumerate(series):
            xs, ys, los, his = [], [], [], []
            for key, m in pts:
                if key not in byk:
                    continue
                c = summ[byk[key]['setting']][m][cond]
                xs.append(byk[key]['iters'] * 1.06 ** (si - 1.5))     # small horizontal offset so the points do not overlap
                ys.append(c['d_path'][0])
                los.append(c['d_path'][0] - c['d_path'][1])
                his.append(c['d_path'][2] - c['d_path'][0])
            if xs:
                ax.errorbar(xs, ys, yerr=[los, his], color=col, ls=ls, marker=mk, ms=7, lw=2, capsize=3, label=name,
                            mec='#444444' if col == '#7FDFA3' else col)
        ax.axhline(0, color='#888888', lw=1)
        ax.set_xscale('log')
        ax.set_xlabel('SVGD iterations per round', fontsize=9, color=MUTED)
        ax.set_title(COND[cond], fontsize=10.5, color=INK, loc='left')
    axes[0].set_ylabel(f'Δ energy at path length {PATH_L} vs. rays of the same run', fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#ddd', labelcolor=INK)
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'budget_trend.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[hist] done ->', out)


if __name__ == '__main__':
    main()
