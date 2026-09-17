r"""
run_heuristic_linear_matrix.py
================================
Follow-up to `run_ideal_matrix.py`, per Philipp's request (2026-09-17): add
two more baseline families next to the CFM/particle/spectral comparison,
each driven by the same four tuned acquisition strategies (mass/eid/lse/ucb,
see `variant_runner.HEURISTIC_LINEAR_STRATEGIES`) that already feed the CFM
runs:

    heuristic   the GUI's own TSP+serpentine initializer
                (`init_baselines.gui_heuristic_path`, extracted from
                `interactive_sim.py` so both share one implementation),
                refined on B-spline control points at svgd_iters in
                (0, 500, 1000).
    linear      the identical initial path, refined directly on dense
                waypoints instead of B-spline control points
                (`variant_runner.linear_waypoints_variant_tuned`), at
                svgd_iters in (500, 1000).

Pure CPU, no CFM network call -- a --pilot smoke test finishes in seconds;
the full 25-shape x 4-strategy x 4-knowledge-condition matrix was timed at
roughly 3-4s per svgd_iters=500 row and 7-8s per svgd_iters=1000 row (see
session notes), i.e. on the order of 2-3 hours total. Still long enough that
CLAUDE.md's rule applies -- start manually, not from here.

Example
-------
    # Smoke test (seconds): 1 shape, 1 strategy, no SVGD
    python run_heuristic_linear_matrix.py --pilot --strategies eid \
        --heuristic_svgd_iters 0 --linear_svgd_iters 500 --out_tag smoke --no_viz

    # Full matrix (NOT started automatically -- ask before running)
    python run_heuristic_linear_matrix.py --out_tag heuristic_linear_YYYYMMDD
"""

import argparse
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
          os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from common.data import load_truth                                # noqa: E402
from common.svgd_refine import SvgdRefiner                         # noqa: E402

import variant_runner as vr                                        # noqa: E402
from metrics_explore_exploit import ExploreExploitErgodic          # noqa: E402
from run_eval_matrix import (score_and_save, summarise, plot_metric_bars,  # noqa: E402
                             plot_tradeoff)
import viz                                                          # noqa: E402

PILOT_SHAPES = ['A', 'rand_gmm_20', 'organic_20']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--shapes', type=str, default=None)
    ap.add_argument('--strategies', type=str,
                    default=','.join(vr.HEURISTIC_LINEAR_STRATEGIES.keys()),
                    help="Familien-Namen (lse,ucb,mass,eid), nicht STRATEGIES-Keys -- "
                         "siehe variant_runner.HEURISTIC_LINEAR_STRATEGIES.")
    ap.add_argument('--heuristic_svgd_iters', type=str,
                    default=','.join(str(i) for i in vr.HEURISTIC_SVGD_ITERS))
    ap.add_argument('--linear_svgd_iters', type=str,
                    default=','.join(str(i) for i in vr.LINEAR_WAYPOINTS_SVGD_ITERS))
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--truth_res', type=int, default=96)
    ap.add_argument('--no_viz', action='store_true')
    args = ap.parse_args()

    families = [s for s in args.strategies.split(',') if s]
    for f in families:
        if f not in vr.HEURISTIC_LINEAR_STRATEGIES:
            raise KeyError(f"unbekannte Familie {f!r}; bekannt: "
                           f"{sorted(vr.HEURISTIC_LINEAR_STRATEGIES)}")
    heuristic_iters = [int(x) for x in args.heuristic_svgd_iters.split(',') if x != '']
    linear_iters = [int(x) for x in args.linear_svgd_iters.split(',') if x != '']

    if args.shapes:
        shapes = [s.strip() for s in args.shapes.split(',')]
    elif args.pilot:
        shapes = PILOT_SHAPES
    else:
        shapes = None

    device = args.device
    names, truths = load_truth(labels=shapes, n=999, split='val',
                               resolution=args.truth_res, device=device)
    print(f"[heuristic_linear_matrix] {len(names)} Formen: {names}")
    print(f"[heuristic_linear_matrix] Familien: {families}  "
         f"heuristic_svgd_iters: {heuristic_iters}  linear_svgd_iters: {linear_iters}")

    refiner = SvgdRefiner(seed=args.seed)
    ee = ExploreExploitErgodic(device=device)

    out_dir = os.path.join(_here, 'results', args.out_tag)
    raw_dir = os.path.join(out_dir, 'raw')
    tables_dir = os.path.join(out_dir, 'tables')
    plots_dir = os.path.join(out_dir, 'plots')
    for d in (raw_dir, tables_dir, plots_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({
            'families': {k: vr.HEURISTIC_LINEAR_STRATEGIES[k] for k in families},
            'heuristic_svgd_iters': heuristic_iters,
            'linear_svgd_iters': linear_iters,
            'shapes_resolved': names,
            'knowledge_conditions': vr.KNOWLEDGE_CONDITIONS,
            'truth_res': args.truth_res,
        }, f, indent=2)

    rows = []
    panels = {}

    def _collect(cond, row, curve, truth_np, name):
        key = (cond, row['variant_id'])
        panels.setdefault(key, []).append((name, truth_np, curve.detach().cpu().numpy()))

    t0 = time.time()
    for i, name in enumerate(names):
        truth = truths[i]
        truth_np = truth.detach().cpu().numpy()
        phi_k_truth = ee.target_coeffs(truth)

        for cond in vr.KNOWLEDGE_CONDITIONS:
            for family in families:
                strat = vr.HEURISTIC_LINEAR_STRATEGIES[family]
                s = vr.STRATEGIES[strat]
                belief = vr.build_belief(
                    cond, truth, seed=args.seed, device=device,
                    gp_noise=s.get('gp_noise', 0.05),
                    gp_lengthscale=s.get('gp_lengthscale', 0.08))

                for n_iters in heuristic_iters:
                    curve = vr.gui_heuristic_variant_tuned(belief, strat, n_iters, refiner)
                    sub = f"{family}__{strat}"
                    row = score_and_save(curve, truth, phi_k_truth, ee, name,
                                         'heuristic_tuned', sub, n_iters, cond,
                                         raw_dir, save_files=not args.no_viz)
                    row.update(family=family, strategy=strat)
                    rows.append(row)
                    _collect(cond, row, curve, truth_np, name)

                for n_iters in linear_iters:
                    curve = vr.linear_waypoints_variant_tuned(belief, strat, n_iters, refiner)
                    sub = f"{family}__{strat}"
                    row = score_and_save(curve, truth, phi_k_truth, ee, name,
                                         'linear_waypoints_tuned', sub, n_iters, cond,
                                         raw_dir, save_files=not args.no_viz)
                    row.update(family=family, strategy=strat)
                    rows.append(row)
                    _collect(cond, row, curve, truth_np, name)

        print(f"[heuristic_linear_matrix] [{i + 1}/{len(names)}] {name} fertig, "
             f"{time.time() - t0:.1f}s seit Start, {len(rows)} Zeilen")

    summarise(rows, tables_dir)
    plot_metric_bars(rows, plots_dir)
    plot_tradeoff(rows, plots_dir)

    panel_dir = os.path.join(plots_dir, 'holdout_panels')
    for (cond, vid), items in panels.items():
        d = os.path.join(panel_dir, cond)
        os.makedirs(d, exist_ok=True)
        viz.plot_holdout_panel(items, os.path.join(d, f'{vid}.png'), title=f'{vid} | {cond}')

    print(f"[heuristic_linear_matrix] fertig: {len(rows)} Zeilen, "
         f"{time.time() - t0:.1f}s gesamt -> {out_dir}")


if __name__ == '__main__':
    main()
