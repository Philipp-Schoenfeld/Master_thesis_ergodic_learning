r"""
run_ideal_matrix.py
=====================
Follow-up to `run_eval_matrix.py`, prepared after the full-matrix run showed
the lawnmower baseline beating CFM under every knowledge condition except
`ground_truth`. Three root causes were found (see the evaluation report
artifact) and are fixed here instead of re-running the same setup:

  1. Fixed `ucb`/kappa=2.0 -> the project's own tuned acquisition settings
     (`variant_runner.STRATEGIES`, from `exploration_optimierung`'s search).
  2. Plain `ucb_density` every replanning round -> `debt_density` with
     recency-weighted visitation, so replanning stops re-covering the same
     spot (`variant_runner.cfm_ideal_replan_1_6`).
  3. Particle-conditioned checkpoint only -> adds the spectral-conditioned
     checkpoint via the new `SpectralPlanner` wrapper (`spectral_planner.py`),
     with the caveat that it was trained on a different, smaller shape set
     (see that module's docstring) and so tests generalisation as well as
     representation choice.

NOT YET EXECUTED — prepared per Philipp's request to review and start
manually. Mechanics were validated with a 1-shape / 1-strategy smoke test
(see the session notes); the full matrix has not been run.

Example
-------
    # Smoke test (seconds): 1 shape, 1 strategy, no SVGD, particles only
    python run_ideal_matrix.py --pilot --representations particles \
        --strategies niveau_svgd0 --out_tag smoke --no_viz

    # Full prepared matrix (NOT started automatically — ask before running)
    python run_ideal_matrix.py --out_tag ideal_run_YYYYMMDD
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

import apply_cfm_belief as acb                                    # noqa: E402
from common.data import load_truth                                # noqa: E402
from common.svgd_refine import SvgdRefiner                         # noqa: E402

import variant_runner as vr                                        # noqa: E402
from spectral_planner import SpectralPlanner                       # noqa: E402
from metrics_explore_exploit import ExploreExploitErgodic          # noqa: E402
from run_eval_matrix import (score_and_save, summarise, plot_metric_bars,  # noqa: E402
                             plot_tradeoff, DEFAULT_CKPT)
import viz                                                          # noqa: E402

SPECTRAL_CKPT = os.path.join(_arch, 'checkpoints', 'cond_spectral_crossattn_ep900.pt')
PILOT_SHAPES = ['A', 'rand_gmm_20', 'organic_20']


def build_particle_planner(ckpt, device):
    p = acb.CfmPlanner(ckpt=ckpt, device=device, pts=vr.PTS_RENDER, steps=100, cfg_weight=2.0)
    if not p.start_cond:
        raise RuntimeError(f"{ckpt} ist nicht startpunkt-konditioniert; "
                           "replan_1_6 braucht start=.")
    return p


def build_spectral_planner(ckpt, device):
    return SpectralPlanner(ckpt, device=device, pts=vr.PTS_RENDER, steps=100)


PLANNER_BUILDERS = {'particles': build_particle_planner, 'spectral': build_spectral_planner}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--shapes', type=str, default=None)
    ap.add_argument('--representations', type=str, default='particles,spectral')
    ap.add_argument('--strategies', type=str, default=','.join(vr.STRATEGIES.keys()))
    ap.add_argument('--replan_schemes', type=str, default='no_replan,replan_1_6')
    ap.add_argument('--particle_ckpt', type=str, default=DEFAULT_CKPT)
    ap.add_argument('--spectral_ckpt', type=str, default=SPECTRAL_CKPT)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--truth_res', type=int, default=96)
    ap.add_argument('--no_viz', action='store_true')
    ap.add_argument('--skip_baselines', action='store_true',
                    help='Maeander/Irrfahrt ueberspringen (schon aus dem ersten Lauf bekannt).')
    args = ap.parse_args()

    reps = [r for r in args.representations.split(',') if r]
    for r in reps:
        if r not in PLANNER_BUILDERS:
            raise KeyError(f"unbekannte Repraesentation {r!r}; bekannt: {sorted(PLANNER_BUILDERS)}")
    strategies = [s for s in args.strategies.split(',') if s]
    for s in strategies:
        if s not in vr.STRATEGIES:
            raise KeyError(f"unbekannte Strategie {s!r}; bekannt: {sorted(vr.STRATEGIES)}")
    schemes = [s for s in args.replan_schemes.split(',') if s]

    if args.shapes:
        shapes = [s.strip() for s in args.shapes.split(',')]
    elif args.pilot:
        shapes = PILOT_SHAPES
    else:
        shapes = None

    device = args.device
    names, truths = load_truth(labels=shapes, n=999, split='val',
                               resolution=args.truth_res, device=device)
    print(f"[ideal_matrix] {len(names)} Formen: {names}")
    print(f"[ideal_matrix] Repraesentationen: {reps}  Strategien: {strategies}  "
         f"Replan-Schemata: {schemes}")

    planners = {rep: PLANNER_BUILDERS[rep](
        args.particle_ckpt if rep == 'particles' else args.spectral_ckpt, device)
        for rep in reps}
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
            'representations': reps, 'strategies': {k: vr.STRATEGIES[k] for k in strategies},
            'replan_schemes': schemes, 'shapes_resolved': names,
            'knowledge_conditions': vr.KNOWLEDGE_CONDITIONS,
            'spectral_checkpoint_caveat':
                'trained on a different, smaller shape set than shape_library.'
                'VALIDATION_SHAPES (sigma, rand_poly_7, G, spiral_2cw, star_5, W, '
                'lissajous_1_3, heart, 5, phi) -- spectral-arm results test '
                'out-of-distribution generalisation too, not representation alone.',
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

        if not args.skip_baselines:
            fixed = {('lawnmower', 'fixed'): vr.lawnmower_variant(),
                    ('random_walk', 'fixed'): vr.random_walk_variant(seed=args.seed * 131 + i)}
            for (method, sub), curve in fixed.items():
                curve = curve.to(device)
                base_row = score_and_save(curve, truth, phi_k_truth, ee, name, method, sub,
                                          None, 'shared', raw_dir, save_files=not args.no_viz)
                for cond in vr.KNOWLEDGE_CONDITIONS:
                    row = dict(base_row, knowledge_condition=cond)
                    rows.append(row)
                    _collect(cond, row, curve, truth_np, name)

        for rep in reps:
            planner = planners[rep]
            gen = vr.REPRESENTATIONS[rep]
            for cond in vr.KNOWLEDGE_CONDITIONS:
                belief = vr.build_belief(cond, truth, seed=args.seed, device=device)
                for strat in strategies:
                    for scheme in schemes:
                        b = belief.clone()
                        if scheme == 'no_replan':
                            curve = gen['no_replan'](planner, b, strat, refiner)
                        elif scheme == 'replan_1_6':
                            curve = gen['replan_1_6'](planner, b, truth, cond, strat, refiner)
                        else:
                            raise KeyError(f"unbekanntes Replan-Schema {scheme!r}")
                        sub = f"{strat}__{scheme}"
                        row = score_and_save(curve, truth, phi_k_truth, ee, name, rep, sub,
                                             None, cond, raw_dir, save_files=not args.no_viz)
                        row.update(representation=rep, strategy=strat, replan_scheme=scheme)
                        rows.append(row)
                        _collect(cond, row, curve, truth_np, name)

        print(f"[ideal_matrix] [{i + 1}/{len(names)}] {name} fertig, "
             f"{time.time() - t0:.1f}s seit Start, {len(rows)} Zeilen")

    summarise(rows, tables_dir)
    plot_metric_bars(rows, plots_dir)
    plot_tradeoff(rows, plots_dir)

    panel_dir = os.path.join(plots_dir, 'holdout_panels')
    for (cond, vid), items in panels.items():
        d = os.path.join(panel_dir, cond)
        os.makedirs(d, exist_ok=True)
        viz.plot_holdout_panel(items, os.path.join(d, f'{vid}.png'), title=f'{vid} | {cond}')

    print(f"[ideal_matrix] fertig: {len(rows)} Zeilen, {time.time() - t0:.1f}s gesamt -> {out_dir}")


if __name__ == '__main__':
    main()
