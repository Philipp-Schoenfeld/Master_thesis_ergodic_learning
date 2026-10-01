r"""
generate_raw_pool_metrics.py -- full-metric stats over the 30 raw candidates
=================================================================================
Philipp's follow-up request (2026-10-01): extend the candidate-pool mean+-std
chart (currently coverage only, free since pre_svgd already scores all 30
raw candidates with coverage_vs_truth for selection) to E_ergodic_total and
J too. Those are NOT free -- only the single selected/refined winner gets
scored with the full metric suite today, so this generates all 30 raw
candidates per cell and scores each one directly (no SVGD refinement here,
that's a separate, orthogonal question already answered in section 19).

Same 4 paired families x 2 representations x 4 knowledge conditions x 8
shapes scope as the rest of this analysis. No SvgdRefiner needed -- this
only touches the network generation + scoring, not refinement.

Usage:
    python generate_raw_pool_metrics.py                 # full scope, 8 shapes
    python generate_raw_pool_metrics.py --shapes A       # smoke test, 1 shape
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
          os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from common.data import load_truth                                 # noqa: E402
import variant_runner as vr                                        # noqa: E402
from metrics_explore_exploit import ExploreExploitErgodic          # noqa: E402
from run_eval_matrix import compute_row, DEFAULT_CKPT               # noqa: E402
from run_ideal_matrix import PLANNER_BUILDERS, SPECTRAL_CKPT        # noqa: E402
import apply_cfm_belief as acb                                      # noqa: E402

ALL_SHAPES = ['A', 'a_lc', 'digit_5', 'greek_upper_0', 'korean_5',
             'rand_ana_poly_10', 'rand_gmm_10', 'rand_gmm_20']
FAMILIES_SVGD0 = ['niveau_svgd0', 'ucb_tuned_svgd0', 'mass_tuned_svgd0', 'eid_tuned_svgd0']
SEED = 0
TRUTH_RES = 96
N_CANDIDATES = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shapes', type=str, default=','.join(ALL_SHAPES))
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--particle_ckpt', type=str, default=DEFAULT_CKPT)
    ap.add_argument('--spectral_ckpt', type=str, default=SPECTRAL_CKPT)
    args = ap.parse_args()

    shapes = [s.strip() for s in args.shapes.split(',') if s.strip()]
    device = args.device

    planners = {
        'particles': PLANNER_BUILDERS['particles'](args.particle_ckpt, device),
        'spectral': PLANNER_BUILDERS['spectral'](args.spectral_ckpt, device),
    }
    ee = ExploreExploitErgodic(device=device)

    names, truths = load_truth(labels=shapes, n=999, split='val',
                               resolution=TRUTH_RES, device=device)
    print(f"[generate_raw_pool_metrics] {len(names)} shapes: {names}")

    rows = []
    for name, truth in zip(names, truths):
        phi_k_truth = ee.target_coeffs(truth)
        t_cell = time.perf_counter()
        for cond in vr.KNOWLEDGE_CONDITIONS:
            for rep in ['particles', 'spectral']:
                planner = planners[rep]
                for svgd0_strat in FAMILIES_SVGD0:
                    s = vr.STRATEGIES[svgd0_strat]
                    belief = vr.build_belief(
                        cond, truth, seed=SEED, device=device,
                        gp_noise=s.get('gp_noise', 0.05),
                        gp_lengthscale=s.get('gp_lengthscale', 0.08))
                    strat_args, _unused_iters, cfg_weight = vr.build_strategy_args(svgd0_strat, device)
                    planner.cfg_weight = cfg_weight
                    mu, sd = belief.posterior_grid()
                    phi = acb.zieldichte(mu, sd, strat_args.kappa, strat_args)

                    if rep == 'particles':
                        parts = acb.phi_particles(phi, strat_args.n_particles,
                                                  mode=strat_args.phi_mode,
                                                  quantile=strat_args.phi_quantile,
                                                  device=belief.device)
                        cps = planner.plan(parts, n_candidates=N_CANDIDATES)
                    else:
                        cps = planner.plan(phi, n_candidates=N_CANDIDATES)
                    curves = planner.render(cps)
                    t_gen = time.perf_counter()

                    family = svgd0_strat[:-len('_svgd0')]
                    sub = f'{family}_pool__no_replan'
                    e_totals, js, covs = [], [], []
                    for curve in curves:
                        row = compute_row(curve, truth, phi_k_truth, ee, name, rep,
                                          sub, None, cond)
                        e_totals.append(row['E_ergodic_total'])
                        js.append(row['J'])
                        covs.append(row['coverage'])
                    t_score = time.perf_counter()
                    rows.append(dict(
                        shape=name, knowledge_condition=cond, representation=rep,
                        family=family, n=len(curves),
                        E_ergodic_total_mean=float(np.mean(e_totals)),
                        E_ergodic_total_std=float(np.std(e_totals)),
                        J_mean=float(np.mean(js)), J_std=float(np.std(js)),
                        coverage_mean=float(np.mean(covs)), coverage_std=float(np.std(covs)),
                    ))
                    print(f"[generate_raw_pool_metrics]   {cond}/{rep}/{family}: "
                         f"gen={t_gen - t_cell:.2f}s score={t_score - t_gen:.2f}s", flush=True)
                    t_cell = time.perf_counter()
        print(f"[generate_raw_pool_metrics] {name} done, {len(rows)} rows so far")

    out_csv = os.path.join(_here, 'results', 'raw_pool_metrics.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[generate_raw_pool_metrics] finished: {len(rows)} rows -> {out_csv}")


if __name__ == '__main__':
    main()
