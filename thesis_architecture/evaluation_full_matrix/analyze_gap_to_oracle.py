r"""
analyze_gap_to_oracle.py
==========================
"Gap-to-oracle" ("time to recovery", Philipp's definition, 2026-10-20): for
every (shape, method, strategy, refiner) run under partial knowledge
(`half_known`, `ten_samples`, `none_known`), how many replanning ROUNDS does
it take until the driven path's `E_truth` first drops to or below what a
FULLY-INFORMED agent (`ground_truth` knowledge condition) already had after
its very first round -- i.e. how fast does active exploration under
uncertainty catch up to where an oracle already started.

Purely a POST-HOC read of `run_mission_eval.py`'s existing shard DBs
(`mission_db.py` -- `rounds.E_truth` is already stored every round); no
generator-side changes, no new metric computed during the mission itself.

Needs the `ground_truth` condition to have been run alongside the 3 partial-
knowledge ones for every (method, strategy, refiner) -- it is NOT one of
`run_mission_eval.py`'s 3 default conditions, pass it explicitly via
`--conditions ground_truth,half_known,ten_samples,none_known`.

Definition, precisely: `round` is 0-indexed in the stored DB (`MissionSet
.run`'s `for r in range(a.max_rounds)`) -- "after round 1" in Philipp's own
words is `round == 0`'s stored row, i.e. the MINIMUM round number present in
the ground_truth shard for that shape. "Catches up" = the first round (of
the partial-knowledge run) whose `E_truth <= oracle_round0_E_truth`. A shape/
condition that never catches up within `max_rounds` gets the sentinel
`n_rounds_used - 1` (the run's own last round, i.e. the current
`n90_from_series`-style "never" convention used throughout this deliverable),
flagged separately in the output so it is not silently averaged in with real
catch-up rounds.

Usage
-----
    python analyze_gap_to_oracle.py --out_root results --out_tag mission_active_exploration_20261020 \
        --methods cfm,selfsup,ebm,linear,random_walk,heuristic --strategies ucb,eid \
        --refiners sun,tsvec,tsvec_svgd
"""
import argparse
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

import mission_db as mdb                                            # noqa: E402
from common.svgd_refine import run_suffix                           # noqa: E402

PARTIAL_CONDITIONS = ('half_known', 'ten_samples', 'none_known')
ORACLE_CONDITION = 'ground_truth'

METHOD_COLORS = {
    'cfm':          '#2a78d6', 'selfsup':     '#eb6834', 'ebm':         '#1baf7a',
    'linear':       '#e87ba4', 'random_walk': '#008300', 'heuristic':   '#6250d6',
}
METHOD_LABELS = {
    'cfm': 'CFM', 'selfsup': 'Self-supervised', 'ebm': 'Energy-Policy (EBM)',
    'linear': 'Linear', 'random_walk': 'Random walk', 'heuristic': 'Heuristic',
}
REFINER_LABELS = {'sun': 'Sun', 'tsvec': 'Adam-SVGD', 'tsvec_svgd': 'TSVEC-SVGD'}


def _style(ax):
    ax.set_facecolor('white')
    ax.grid(alpha=0.2, color='#ccc')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555', labelsize=8)


def oracle_round0_e_truth(root, strategy, method):
    """-> {shape: E_truth at the oracle's (ground_truth condition's) first
    stored round} for every shape in that shard."""
    path = mdb.shard_path(root, ORACLE_CONDITION, strategy, method)
    if not os.path.exists(path):
        return {}
    conn = mdb.open_db(path)
    out = {}
    for row in mdb.iter_rounds(conn, columns=('shape', 'round', 'E_truth')):
        shape = row['shape']
        if shape not in out or row['round'] < out[shape][0]:
            out[shape] = (row['round'], row['E_truth'])
    conn.close()
    return {s: e for s, (_r, e) in out.items()}


def gap_rounds_for_condition(root, cond, strategy, method, oracle_e_truth):
    """-> {shape: (gap_round:int, caught_up:bool, n_rounds_used:int)}."""
    path = mdb.shard_path(root, cond, strategy, method)
    if not os.path.exists(path):
        return {}
    conn = mdb.open_db(path)
    per_shape = {}
    for row in mdb.iter_rounds(conn, columns=('shape', 'round', 'E_truth')):
        per_shape.setdefault(row['shape'], []).append((row['round'], row['E_truth']))
    conn.close()
    out = {}
    for shape, rows in per_shape.items():
        if shape not in oracle_e_truth:
            continue
        rows.sort(key=lambda rv: rv[0])
        ref = oracle_e_truth[shape]
        n_rounds_used = len(rows)
        hit = next((r for r, e in rows if e <= ref), None)
        if hit is None:
            out[shape] = (rows[-1][0], False, n_rounds_used)
        else:
            out[shape] = (hit, True, n_rounds_used)
    return out


def collect(out_root, methods, strategies, refiners):
    """-> list of dicts: method, strategy, refiner, cond, shapes stats."""
    records = []
    for refiner in refiners:
        root = out_root + run_suffix(refiner)
        if not os.path.isdir(root):
            print(f"[gap_to_oracle] {root} missing (refiner={refiner!r}) -- skipped")
            continue
        for strategy in strategies:
            for method in methods:
                oracle = oracle_round0_e_truth(root, strategy, method)
                if not oracle:
                    continue
                for cond in PARTIAL_CONDITIONS:
                    gaps = gap_rounds_for_condition(root, cond, strategy, method, oracle)
                    if not gaps:
                        continue
                    vals = np.array([g for g, _c, _n in gaps.values()], dtype=np.float64)
                    caught = np.array([c for _g, c, _n in gaps.values()])
                    records.append(dict(
                        method=method, strategy=strategy, refiner=refiner, condition=cond,
                        n_shapes=len(gaps), gap_rounds_mean=float(vals.mean()),
                        gap_rounds_std=float(vals.std()), gap_rounds_median=float(np.median(vals)),
                        frac_caught_up=float(caught.mean()), per_shape=gaps))
    return records


def write_csv(records, out_path):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'strategy', 'refiner', 'condition', 'n_shapes',
                   'gap_rounds_mean', 'gap_rounds_std', 'gap_rounds_median', 'frac_caught_up'])
        for r in records:
            w.writerow([r['method'], r['strategy'], r['refiner'], r['condition'], r['n_shapes'],
                       r['gap_rounds_mean'], r['gap_rounds_std'], r['gap_rounds_median'],
                       r['frac_caught_up']])


def plot_gap_boxplots(records, out_path):
    """One box per (method, refiner), x-axis = rounds to catch up to the
    oracle, faceted by strategy x condition -- same box+jittered-strip style
    as `plot_init_comparison_per_metric.py` for visual consistency."""
    strategies = sorted({r['strategy'] for r in records})
    conditions = [c for c in PARTIAL_CONDITIONS if any(r['condition'] == c for r in records)]
    refiners = sorted({r['refiner'] for r in records})
    methods = sorted({r['method'] for r in records})

    fig, axes = plt.subplots(len(conditions), len(strategies),
                             figsize=(6.5 * len(strategies), 0.5 * len(methods) * len(refiners)
                                     * len(conditions) + 2),
                             facecolor='white', squeeze=False)
    rng = np.random.default_rng(0)
    for ci, cond in enumerate(conditions):
        for si, strategy in enumerate(strategies):
            ax = axes[ci][si]
            entries = []
            for method in methods:
                for refiner in refiners:
                    match = [r for r in records if r['method'] == method and r['refiner'] == refiner
                            and r['strategy'] == strategy and r['condition'] == cond]
                    if not match:
                        continue
                    r = match[0]
                    vals = np.array([g for g, _c, _n in r['per_shape'].values()], dtype=np.float64)
                    entries.append(dict(method=method, refiner=refiner, vals=vals,
                                        median=np.median(vals), frac=r['frac_caught_up']))
            entries.sort(key=lambda e: e['median'])
            n = len(entries)
            ys = np.arange(n)[::-1]
            for y, e in zip(ys, entries):
                jitter = rng.uniform(-0.28, 0.28, size=e['vals'].shape)
                ax.scatter(e['vals'], y + jitter, color=METHOD_COLORS[e['method']], s=8, alpha=0.3,
                          zorder=2, edgecolors='none')
            if n:
                bp = ax.boxplot([e['vals'] for e in entries], positions=ys, vert=False, widths=0.6,
                                patch_artist=True, showfliers=False, zorder=3,
                                medianprops=dict(color='#1A1A2E', linewidth=1.5))
                for box, e in zip(bp['boxes'], entries):
                    box.set_facecolor(METHOD_COLORS[e['method']])
                    box.set_alpha(0.85)
                    box.set_edgecolor('#1A1A2E')
                    if e['refiner'] != 'sun':
                        box.set_hatch('///' if e['refiner'] == 'tsvec' else '...')
            ax.set_yticks(ys)
            ax.set_yticklabels([f"{METHOD_LABELS[e['method']]} + {REFINER_LABELS[e['refiner']]} "
                                f"({e['frac']*100:.0f}% caught up)" for e in entries], fontsize=6.5)
            ax.set_xlabel('Rounds to reach the oracle\'s round-0 E_truth')
            ax.set_title(f"{cond} / {strategy}", fontsize=9, color='#1A1A2E')
            _style(ax)
    fig.suptitle('Gap-to-oracle ("time to recovery"): rounds of active exploration under partial '
                'knowledge to match a fully-informed agent\'s first round', fontsize=11,
                color='#1A1A2E')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_root', type=str, default=os.path.join(_here, 'results'))
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--methods', type=str,
                    default='cfm,selfsup,ebm,linear,random_walk,heuristic')
    ap.add_argument('--strategies', type=str, default='ucb,eid')
    ap.add_argument('--refiners', type=str, default='sun,tsvec,tsvec_svgd')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='Where to write gap_to_oracle.csv/.png (default: <out_root>/<out_tag>_gap_to_oracle/)')
    args = ap.parse_args()

    methods = [m for m in args.methods.split(',') if m]
    strategies = [s for s in args.strategies.split(',') if s]
    refiners = [r for r in args.refiners.split(',') if r]
    out_root = os.path.join(args.out_root, args.out_tag)
    out_dir = args.out_dir or os.path.join(args.out_root, f'{args.out_tag}_gap_to_oracle')
    os.makedirs(out_dir, exist_ok=True)

    records = collect(out_root, methods, strategies, refiners)
    if not records:
        print("[gap_to_oracle] no shards found -- nothing to analyze "
             f"(looked under {out_root}[+ run_suffix per refiner])")
        return

    write_csv(records, os.path.join(out_dir, 'gap_to_oracle.csv'))
    print(f"[gap_to_oracle] {len(records)} (method,strategy,refiner,condition) rows -> "
         f"{out_dir}/gap_to_oracle.csv")

    plot_gap_boxplots(records, os.path.join(out_dir, 'gap_to_oracle.png'))
    print(f"[gap_to_oracle] gap_to_oracle.png written -> {out_dir}")


if __name__ == '__main__':
    main()
