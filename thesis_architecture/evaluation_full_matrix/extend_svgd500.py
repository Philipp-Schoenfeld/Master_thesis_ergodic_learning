r"""
extend_svgd500.py -- extra 500-SVGD-iteration comparison point
==================================================================
Philipp's request (2026-09-18): for the 4 paired families (niveau, ucb,
mass, eid_tuned), which already have svgd0/svgd25 rows computed by the
--paired_svgd run, add a THIRD comparison point at 500 SVGD iterations --
without re-generating any candidates (no network inference needed at all).

How: reload the already-cached, already-selected RAW (svgd0) trajectory
from disk (raw/**/trajectory.npy), reconstruct the exact same target
density phi (deterministic: same seed/gp_noise/gp_lengthscale as the
original run -- belief.posterior_grid() depends only on those, never on
which candidates were drawn), and SVGD-refine that same raw trajectory
with 500 iterations instead of 25. Saved into the SAME out_tag folder
under a new subvariant (e.g. niveau_svgd500__no_replan), so it merges into
the existing results/<out_tag>/raw/ tree exactly like any other row.

Scope: the 8 shapes that are fully done in best_of_30_paired_20260918 (see
analysis_paired_partial_20260918/analyze.py's ONLY_FULLY_DONE_SHAPES).
particles + spectral, all 4 knowledge conditions, no_replan only (the
originally paired scheme).

Usage:
    python extend_svgd500.py                 # full scope (8 shapes)
    python extend_svgd500.py --shapes A       # smoke test, 1 shape
"""
import argparse
import csv
import os
import sys

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
from common.svgd_refine import SvgdRefiner                          # noqa: E402
import variant_runner as vr                                         # noqa: E402
from metrics_explore_exploit import ExploreExploitErgodic           # noqa: E402
from run_eval_matrix import compute_row, save_trajectory_files, DEFAULT_CKPT  # noqa: E402
from run_ideal_matrix import PLANNER_BUILDERS, SPECTRAL_CKPT         # noqa: E402
import apply_cfm_belief as acb                                      # noqa: E402

OUT_TAG = 'best_of_30_paired_20260918'
RAW_DIR = os.path.join(_here, 'results', OUT_TAG, 'raw')

ALL_SHAPES = ['A', 'a_lc', 'digit_5', 'greek_upper_0', 'korean_5',
             'rand_ana_poly_10', 'rand_gmm_10', 'rand_gmm_20']
FAMILIES_SVGD0 = ['niveau_svgd0', 'ucb_tuned_svgd0', 'mass_tuned_svgd0', 'eid_tuned_svgd0']
SVGD_ITERS_EXT = 500
SEED = 0
TRUTH_RES = 96


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
    refiner = SvgdRefiner(seed=SEED)
    ee = ExploreExploitErgodic(device=device)

    names, truths = load_truth(labels=shapes, n=999, split='val',
                               resolution=TRUTH_RES, device=device)
    print(f"[extend_svgd500] {len(names)} shapes: {names}")

    rows = []
    n_missing = 0
    for name, truth in zip(names, truths):
        phi_k_truth = ee.target_coeffs(truth)
        for cond in vr.KNOWLEDGE_CONDITIONS:
            for rep in ['particles', 'spectral']:
                planner = planners[rep]
                for svgd0_strat in FAMILIES_SVGD0:
                    sub0 = f"{svgd0_strat}__no_replan"
                    vid0 = f"{rep}_{sub0}"
                    traj_path = os.path.join(RAW_DIR, cond, rep, vid0, name, 'trajectory.npy')
                    if not os.path.isfile(traj_path):
                        print(f"[extend_svgd500] MISSING, skipping: {traj_path}")
                        n_missing += 1
                        continue
                    raw_np = np.load(traj_path)
                    raw_curve = torch.as_tensor(raw_np, dtype=torch.float32, device=device)

                    s = vr.STRATEGIES[svgd0_strat]
                    belief = vr.build_belief(
                        cond, truth, seed=SEED, device=device,
                        gp_noise=s.get('gp_noise', 0.05),
                        gp_lengthscale=s.get('gp_lengthscale', 0.08))
                    strat_args, _unused_iters, _cfg_weight = vr.build_strategy_args(svgd0_strat, device)
                    mu, sd = belief.posterior_grid()
                    phi = acb.zieldichte(mu, sd, strat_args.kappa, strat_args)

                    nxi = planner.nxi if rep == 'spectral' else None
                    refined = vr.svgd(refiner, raw_curve, phi, SVGD_ITERS_EXT, nxi=nxi)

                    family = svgd0_strat[:-len('_svgd0')]
                    strat500 = f"{family}_svgd{SVGD_ITERS_EXT}"
                    sub500 = f"{strat500}__no_replan"
                    row = compute_row(refined, truth, phi_k_truth, ee, name, rep,
                                      sub500, None, cond)
                    row.update(representation=rep, strategy=strat500, replan_scheme='no_replan')
                    save_trajectory_files(refined, row, truth, RAW_DIR)
                    rows.append(row)
        print(f"[extend_svgd500] {name} done, {len(rows)} rows so far")

    out_csv = os.path.join(_here, 'results', 'svgd500_extension.csv')
    if rows:
        keys = sorted({k for row in rows for k in row})
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"[extend_svgd500] finished: {len(rows)} rows ({n_missing} missing raw "
         f"trajectories skipped) -> {out_csv}")
    print("[extend_svgd500] new rows also saved under "
         f"results/{OUT_TAG}/raw/**/*_svgd500__no_replan/ like any other row.")


if __name__ == '__main__':
    main()
