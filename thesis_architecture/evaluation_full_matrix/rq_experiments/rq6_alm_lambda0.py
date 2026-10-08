"""
rq6_alm_lambda0.py
===================
RQ6 (warm-started Lagrange multipliers for a constrained ergodic-coverage
solver) -- the narrow, honestly-scoped question `alm_solver.py`'s docstring
describes: does warm-starting the Augmented Lagrangian multipliers (lambda
for the start-point equality, mu for the workspace/obstacle inequalities) at
all reduce the number of outer iterations to feasibility, compared to the
only thing possible without a predictor (lambda=0, mu=0)?

This is NOT a test of Flow-Opt's own method (a trained network predicting
lambda_0 self-supervised on the fixed-point residual of ITS solver) -- that
would need a new network architecture and a GPU training run, out of scope
here (see `alm_solver.py`'s module docstring for the full reasoning). Instead
the cheapest available INFORMED source of multipliers is used as the stand-in
for "a predicted lambda_0": the PREVIOUS replanning round's converged
multipliers, in a simple two-round setup (round 1 solves from scratch at a
random start; round 2 perturbs the start by a small step, simulating the
agent having moved, and is solved both cold and warm from round 1's result).
If even this weak, zero-cost warm start helps, a trained predictor -- which
could in principle do much better by learning the mapping from problem to
good multipliers directly -- is a reasonable next step; if it does not, that
is also worth knowing before investing in training one.

Both the primal variable (the B-spline control points) AND the multipliers
are warm-started in the "warm" condition (a real replanner would warm-start
both); the "cold" condition warm-starts only the primal (also the realistic
baseline -- nothing stops a non-predictor solver from reusing the previous
control points) and starts lambda=mu=0. This isolates the EFFECT OF THE
MULTIPLIERS specifically, not of warm-starting in general.

Usage:
    python rq6_alm_lambda0.py --n_trials 5 --out_tag smoke
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
_arch = os.path.dirname(_mat)
for _p in (_here, _mat, os.path.join(_arch, 'ergodic_dataset_generator')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import exp_common as C                                             # noqa: E402
import paper_style as ps                                           # noqa: E402
from alm_solver import AlmBSplineSolver, NXI, N_EVAL                # noqa: E402
from obstacles import bspline_basis_matrix                          # noqa: E402

B_BASIS = bspline_basis_matrix(NXI, N_EVAL, 5)
FIELDS = ['trial', 'round1_iters', 'round1_feasible', 'cold_iters', 'cold_feasible',
         'warm_iters', 'warm_feasible']


def _load_done(csv_path):
    """Resumability for cluster jobs (24h limit, see CLAUDE.md): trial seeds
    already present in the CSV are skipped on restart, matching the
    append-immediately pattern `generate_raw_pool_metrics.py` already uses."""
    done = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                done.add(int(row['trial']))
    return done


def run_trial(seed, args):
    rng = np.random.default_rng(seed)
    trial = C.make_trial(seed=seed)
    phi = C.density_grid(trial['pdf_fn'], res=96)
    phik = C.target_coeffs(phi)
    start1 = np.asarray(trial['x0'])

    solver1 = AlmBSplineSolver(B_BASIS, phik, C.K_IDX, start1,
                               obstacle_center=args.obstacle_center,
                               obstacle_radius=args.obstacle_radius)
    xi0 = np.clip(np.tile(start1, (NXI, 1)) + 0.01 * rng.normal(size=(NXI, 2)), 0.02, 0.98)
    res1 = solver1.solve(xi0, n_outer=args.n_outer, tol=args.tol)

    # Round 2: the start has moved a little (simulated replanning step),
    # same target density -- the setting Flow-Opt's own warm start targets
    # (consecutive, correlated problems in a replanning loop), just without
    # a trained predictor supplying lambda_0.
    start2 = np.clip(start1 + rng.normal(scale=args.start_shift, size=2), 0.03, 0.97)
    solver2 = AlmBSplineSolver(B_BASIS, phik, C.K_IDX, start2,
                               obstacle_center=args.obstacle_center,
                               obstacle_radius=args.obstacle_radius)
    xi0_2 = res1['xi'].copy()   # primal warm start: both conditions get this

    res_cold = solver2.solve(xi0_2, n_outer=args.n_outer, tol=args.tol)
    res_warm = solver2.solve(xi0_2, lam0=res1['lam'], mu0=res1['mu'],
                             n_outer=args.n_outer, tol=args.tol)
    return dict(
        trial=seed, round1_iters=res1['outer_iters'], round1_feasible=res1['feasible'],
        cold_iters=res_cold['outer_iters'], cold_feasible=res_cold['feasible'],
        warm_iters=res_warm['outer_iters'], warm_feasible=res_warm['feasible'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n_trials', type=int, default=20)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--n_outer', type=int, default=80)
    ap.add_argument('--tol', type=float, default=5e-3,
                    help='Feasibility tolerance (max constraint violation). See the '
                         'module docstring of alm_solver.py / the development log in '
                         "rq_experiments/README.md for why 1e-3 is too tight for the "
                         "obstacle constraint to always reach within n_outer iterations.")
    ap.add_argument('--start_shift', type=float, default=0.05,
                    help='Std-dev of the simulated between-round agent displacement.')
    ap.add_argument('--obstacle_center', type=float, nargs=2, default=(0.5, 0.5))
    ap.add_argument('--obstacle_radius', type=float, default=0.08)
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop starting new trials after this many hours; already-written '
                         'rows stay valid, re-run with the same --out_tag to resume.')
    args = ap.parse_args()

    out_dir = os.path.join(_mat, 'results', args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'rq6_alm_lambda0.csv')
    done = _load_done(csv_path)
    need_header = not os.path.isfile(csv_path)
    t0 = time.time()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if need_header:
            w.writeheader()
        for i in range(args.n_trials):
            if C.time_budget_exceeded(t0, args.time_budget_h):
                print(f"[rq6] time budget reached before trial {i}, stopping.", flush=True)
                break
            seed = args.seed0 + i
            if seed in done:
                continue
            r = run_trial(seed, args)
            print(f"[rq6] trial {r['trial']}: round1={r['round1_iters']} outer iters, "
                 f"cold={r['cold_iters']} ({'ok' if r['cold_feasible'] else 'NOT feasible'}), "
                 f"warm={r['warm_iters']} ({'ok' if r['warm_feasible'] else 'NOT feasible'})", flush=True)
            w.writerow(r)
            f.flush()
    print(f"[rq6] -> {csv_path}")

    import pandas as pd
    rows = pd.read_csv(csv_path).to_dict('records')
    cold = np.array([r['cold_iters'] if r['cold_feasible'] else np.nan for r in rows])
    warm = np.array([r['warm_iters'] if r['warm_feasible'] else np.nan for r in rows])
    plot_dir = os.path.join(out_dir, 'plots')
    ps.violin({'cold': cold, 'warm': warm},
             'Outer iterations to feasibility: zero vs. warm-started multipliers',
             'Outer ALM iterations to feasibility', os.path.join(plot_dir, 'rq6_cold_vs_warm.png'),
             methods=['cold', 'warm'])
    print(f"[rq6] plot -> {plot_dir}")

    ok = ~np.isnan(cold) & ~np.isnan(warm)
    if ok.sum() >= 2:
        delta = cold[ok] - warm[ok]   # positive = warm was faster
        print(f"[rq6] {ok.sum()}/{len(rows)} trials feasible both ways. "
             f"median(cold - warm) = {np.median(delta):+.1f} outer iterations "
             f"({(delta > 0).sum()} faster warm, {(delta < 0).sum()} faster cold, "
             f"{(delta == 0).sum()} tied)")
    else:
        print("[rq6] too few trials feasible under both conditions to compare.")


if __name__ == '__main__':
    main()
