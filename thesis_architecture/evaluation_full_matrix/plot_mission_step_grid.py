r"""
plot_mission_step_grid.py -- step-by-step overview of one replanning mission
=============================================================================
One shape, one (knowledge condition, strategy); rows = methods (cfm / random_walk /
linear), columns = the first N execution steps. Every tile shows

  * the planning target density of that round (what the GP belief made of the
    ground truth; WHITE_INFERNO, alpha 0.55),
  * the raw initialisation of the driven candidate (grey),
  * the best optimised full trajectory of that round (blue),
  * the path actually driven so far: previous units (thin green) and the unit
    driven in this step (thick green).

Reads the shard DBs written by `run_mission_eval.py` / `run_mission_eval_budget_matched.py`.
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, _arch, os.path.join(_arch, 'exploration')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402

import mission_db as mdb                                             # noqa: E402

INK, MUTED, SPINE = '#1A1A2E', '#555555', '#cccccc'
METHOD_LABEL = {'cfm': 'CFM warm start', 'random_walk': 'Random walk', 'linear': 'Linear'}


def white_inferno(n=256):
    base = plt.get_cmap('inferno')(np.linspace(0, 1, n))
    fade = np.linspace(0, 1, n // 4)[:, None]
    head = np.ones((n // 4, 4)) * (1 - fade) + base[: n // 4] * fade
    return LinearSegmentedColormap.from_list('white_inferno', np.vstack([head, base[n // 4:]]))


def load_mission(root, cond, strategy, method, shape, n_steps):
    conn = mdb.open_db(mdb.shard_path(root, cond, strategy, method))
    rounds = [r for r in mdb.iter_rounds(
        conn, columns=('shape', 'round', 'selected_idx', 'target', 'segment'), shape=shape)
        if r['round'] < n_steps]
    steps = []
    for r in rounds:
        cand = next(mdb.iter_candidates(conn, with_states=True, shape=shape,
                                        round=r['round'], selected=1))
        B = mdb.load_basis(conn, cand['nxi'], cand['init_curve'].shape[0])
        steps.append(dict(target=r['target'], segment=r['segment'], init=cand['init_curve'],
                          best=B @ cand['cps'][-1], n_iters=cand['n_iters']))
    conn.close()
    return steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--root', required=True, help='run directory containing shards/')
    ap.add_argument('--cond', default='none_known')
    ap.add_argument('--strategy', default='eid')
    ap.add_argument('--shape', default='organic_10')
    ap.add_argument('--n_steps', type=int, default=10)
    ap.add_argument('--methods', default='cfm,random_walk,linear')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    methods = args.methods.split(',')
    cmap = white_inferno()
    fig, axes = plt.subplots(len(methods), args.n_steps, facecolor='white', squeeze=False,
                             figsize=(2.0 * args.n_steps, 2.15 * len(methods) + 0.9))
    for i, m in enumerate(methods):
        steps = load_mission(args.root, args.cond, args.strategy, m, args.shape, args.n_steps)
        for j in range(args.n_steps):
            ax = axes[i][j]
            ax.set_facecolor('white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(SPINE)
            if i == 0:
                ax.set_title(f'step {j + 1}', fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(METHOD_LABEL.get(m, m), fontsize=9, color=INK)
            if j >= len(steps):                          # mission already finished
                ax.text(0.5, 0.5, 'finished', ha='center', va='center', fontsize=8, color=MUTED)
                continue
            s = steps[j]
            ax.imshow(s['target'], origin='lower', extent=(0, 1, 0, 1), cmap=cmap,
                      alpha=0.55, vmin=0, vmax=float(s['target'].max()) or 1.0)
            ax.plot(*s['init'].T, color='#9E9E9E', lw=1.0, alpha=0.9, zorder=2)
            ax.plot(*s['best'].T, color='#1565C0', lw=1.4, alpha=0.9, zorder=3)
            if j > 0:
                prev = np.concatenate([t['segment'] for t in steps[:j]], axis=0)
                ax.plot(*prev.T, color='#00C853', lw=1.2, alpha=0.85, zorder=4)
            ax.plot(*s['segment'].T, color='#00C853', lw=2.6, alpha=0.95, zorder=5)
            ax.text(0.03, 0.03, f"{s['n_iters']} it", transform=ax.transAxes, fontsize=6.5,
                    color=INK, bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
    handles = [Line2D([], [], color='#9E9E9E', lw=1.5, label='initialisation (selected candidate)'),
               Line2D([], [], color='#1565C0', lw=1.8, label='best optimised full trajectory'),
               Line2D([], [], color='#00C853', lw=1.5, label='driven in earlier steps'),
               Line2D([], [], color='#00C853', lw=3, label='driven in this step')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f'Mission steps -- shape {args.shape}, {args.cond}, {args.strategy} '
                 f'(background: planning target density of the round)',
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor='white')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
