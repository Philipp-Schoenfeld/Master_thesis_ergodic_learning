#!/usr/bin/env python3
r"""
eval_alm_warmstart_convergence.py
===================================
The actual Decision-Focused-Learning validation, in the style of
`rq_experiments/rq6_alm_lambda0.py`: does the DFL-trained lambda0 head
reduce the number of ALM outer iterations needed to reach feasibility
(violation <= tol), compared to the solver's own zero-initialised duals
(`alm_solver.py::init_multipliers()`)?

Unlike `eval_dfl_ergodicity_svgd.py`, this does NOT touch the production
SVGD/Sun solver (which has no Lagrange multipliers to warm-start) -- it
evaluates the ALM solver (`alm_solver_torch.py`) that the lambda0 head was
actually trained against, using the exact same self-referential phi_k_target
convention training used (spectral_batch[...,0] = c_k of the shape itself,
NOT the real target density grid -- that's `eval_dfl_ergodicity_svgd.py`'s
job). Pure PyTorch, no JAX, so this is much cheaper than the SVGD eval.

For each of the 25 holdout shapes, N_PER_SHAPE generated trajectories give
N_PER_SHAPE (lambda0, mu0) warm-start candidates; the cold baseline is the
solver's own zero-init. Both run from the same generic cold-start xi0 (all
control points at the shape's start) for N_OUTER outer ALM iterations,
tracking violation(xi) each iteration -> outer_iters-to-feasibility.

Usage:
  python eval_alm_warmstart_convergence.py --checkpoint <path> \
      --n_per_shape 30 --n_outer 30 --out eval_results/alm_warmstart.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
if _arch not in sys.path:
    sys.path.insert(0, _arch)

from flow_matching_cond_spectral_crossattn import (                  # noqa: E402
    SpectralCrossAttnFlowNetwork, generate_spectral_trajectories,
)
from flow_matching_runner_ergodic import trajectory_to_spectral, _load_shapes  # noqa: E402
from obstacles import bspline_basis_matrix                            # noqa: E402
import alm_solver_torch as alm                                        # noqa: E402


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--n_per_shape', type=int, default=30)
    p.add_argument('--n_outer', type=int, default=30, help="matches alm_solver.py::solve()'s own default")
    p.add_argument('--inner_steps', type=int, default=8, help="matches DFL training default")
    p.add_argument('--inner_lr', type=float, default=2e-3)
    p.add_argument('--rho0', type=float, default=10.0)
    p.add_argument('--rho_growth', type=float, default=1.3)
    p.add_argument('--rho_max', type=float, default=1e5)
    p.add_argument('--tol', type=float, default=1e-3, help="matches alm_solver.py::solve()'s own default")
    p.add_argument('--n_eval', type=int, default=128, help="curve evaluation points, matches alm_solver.py's N_EVAL")
    p.add_argument('--bspline_deg', type=int, default=5)
    p.add_argument('--steps', type=int, default=100, help="CFM sampling steps")
    p.add_argument('--cfg_weight', type=float, default=2.0)
    p.add_argument('--shapes_limit', type=int, default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"device={device}  n_per_shape={args.n_per_shape}  n_outer={args.n_outer}  tol={args.tol}")

    t0 = time.time()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = SpectralCrossAttnFlowNetwork(
        nxi=ckpt['nxi'], nd=ckpt['nd'], D=ckpt['D'], S=ckpt['S'],
        n_lambda=ckpt['n_lambda'], predict_lambda=ckpt['predict_lambda'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    assert ckpt['predict_lambda'], "checkpoint has no lambda0 head"
    nxi = ckpt['nxi']
    expected_n_lambda = alm.n_lambda_for_alm(nxi)
    assert ckpt['n_lambda'] == expected_n_lambda, (
        f"checkpoint n_lambda={ckpt['n_lambda']} != n_lambda_for_alm({nxi})={expected_n_lambda}"
    )
    print(f"loaded checkpoint epoch={ckpt.get('epoch')}  nxi={nxi}  n_lambda={ckpt['n_lambda']}  ({time.time()-t0:.1f}s)")

    _, holdout_shapes, _, _ = _load_shapes(nxi)
    shape_names = list(holdout_shapes.keys())
    if args.shapes_limit:
        shape_names = shape_names[:args.shapes_limit]
    print(f"{len(shape_names)} holdout shapes: {shape_names}")

    basis_np = bspline_basis_matrix(nxi, args.n_eval, args.bspline_deg).astype(np.float32)
    basis = torch.tensor(basis_np, dtype=torch.float32, device=device)
    center = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)

    all_xi0, all_start, all_phik, all_kidx = [], [], [], []
    all_lam0_warm, all_mu0_warm = [], []
    all_cond, all_shape = [], []

    for shape in shape_names:
        base_cp = holdout_shapes[shape]
        spec_np, k_idx_cond_np = trajectory_to_spectral(base_cp, S=ckpt['S'])
        spec_t = torch.tensor(spec_np, dtype=torch.float32, device=device)
        k_idx_cond_t = torch.tensor(k_idx_cond_np, dtype=torch.long, device=device)

        gen_cps, lambda0_raw = generate_spectral_trajectories(
            model, spec_t, k_idx_cond_t, num_samples=args.n_per_shape,
            nxi=nxi, nd=ckpt['nd'], steps=args.steps, device=device, cfg_weight=args.cfg_weight,
        )
        assert lambda0_raw is not None, "model did not return lambda0 -- predict_lambda mismatch"

        # Same self-referential phi_k_target convention DFL training used:
        # spectral_batch[...,0] = c_k of the shape itself, repeated per sample.
        phi_k_target = spec_t[:, 0].unsqueeze(0).expand(args.n_per_shape, -1)   # (N, S)
        k_idx_batch = k_idx_cond_t.float().unsqueeze(0).expand(args.n_per_shape, -1, -1)  # (N, S, 2)
        start = torch.tensor(base_cp[0], dtype=torch.float32, device=device).unsqueeze(0).expand(args.n_per_shape, -1)
        xi0 = alm.make_cold_start(start, nxi)                          # (N, nxi, 2)

        lam0_warm, mu0_warm = alm.split_lambda0(lambda0_raw, nxi, obstacle_enabled=True)

        for cond in ('warm', 'cold'):
            all_xi0.append(xi0)
            all_start.append(start)
            all_phik.append(phi_k_target)
            all_kidx.append(k_idx_batch)
            if cond == 'warm':
                all_lam0_warm.append(lam0_warm)
                all_mu0_warm.append(mu0_warm)
            else:
                all_lam0_warm.append(torch.zeros_like(lam0_warm))
                all_mu0_warm.append(torch.zeros_like(mu0_warm))
            all_cond += [cond] * args.n_per_shape
            all_shape += [shape] * args.n_per_shape

    xi = torch.cat(all_xi0, dim=0)
    start_all = torch.cat(all_start, dim=0)
    phik_all = torch.cat(all_phik, dim=0)
    kidx_all = torch.cat(all_kidx, dim=0)
    lam = torch.cat(all_lam0_warm, dim=0)
    mu = torch.cat(all_mu0_warm, dim=0)
    C = xi.shape[0]
    print(f"ALM batch: C={C} candidates (warm+cold), n_outer={args.n_outer} — running...")

    t1 = time.time()
    rho = args.rho0
    violation_log = torch.empty(args.n_outer, C, device=device)
    objective_log = torch.empty(args.n_outer, C, device=device)
    # NOT torch.no_grad(): _inner_minimise computes its "gradient descent" step
    # via torch.autograd.grad (real autodiff, not a hand-derived formula), so
    # grad mode must stay enabled even for this forward-only evaluation. We
    # just never call .backward() and .detach() after every outer step so the
    # unrolled graph (built fresh each inner_minimise call for the DFL
    # double-backward use case) never accumulates across outer iterations here.
    for it in range(args.n_outer):
        xi, lam, mu, rho = alm._outer_step(
            xi, lam, mu, rho, basis, phik_all, kidx_all, start_all, center, 0.0,
            args.inner_lr, args.inner_steps, args.rho_growth, args.rho_max,
            create_graph=False,
        )
        xi, lam, mu = xi.detach(), lam.detach(), mu.detach()
        with torch.no_grad():
            violation_log[it] = alm.violation(xi, start_all, center, 0.0)
            objective_log[it] = alm.objective(xi, basis, phik_all, kidx_all)
    alm_wall = time.time() - t1
    print(f"ALM solve done in {alm_wall:.1f}s ({alm_wall / max(C * args.n_outer, 1) * 1000:.3f} ms/candidate-outer-iter)")

    viol_np = violation_log.T.detach().cpu().numpy()                   # (C, n_outer)
    obj_np = objective_log.T.detach().cpu().numpy()

    # outer_iters to feasibility: first outer index with violation <= tol, else n_outer (censored).
    feasible_mask = viol_np <= args.tol
    outer_iters = np.full(C, args.n_outer, dtype=np.int32)
    for c in range(C):
        hit = np.flatnonzero(feasible_mask[c])
        if hit.size:
            outer_iters[c] = hit[0] + 1                                # 1-indexed, matches alm_solver.py's outer_iters

    results = {}
    idx = 0
    for shape in shape_names:
        results[shape] = {}
        for cond in ('warm', 'cold'):
            sl = slice(idx, idx + args.n_per_shape)
            idx += args.n_per_shape
            feasible_rate = float((outer_iters[sl] < args.n_outer).mean())
            results[shape][cond] = {
                'outer_iters_median': float(np.median(outer_iters[sl])),
                'outer_iters_mean': float(np.mean(outer_iters[sl])),
                'feasible_within_budget_rate': feasible_rate,
                'final_violation_median': float(np.median(viol_np[sl, -1])),
                'final_objective_median': float(np.median(obj_np[sl, -1])),
                'outer_iters_per_sample': outer_iters[sl].tolist(),
            }
    assert idx == C

    overall = {}
    for cond in ('warm', 'cold'):
        mask = np.array([c == cond for c in all_cond])
        overall[cond] = {
            'outer_iters_median': float(np.median(outer_iters[mask])),
            'outer_iters_mean': float(np.mean(outer_iters[mask])),
            'feasible_within_budget_rate': float((outer_iters[mask] < args.n_outer).mean()),
            'final_violation_median': float(np.median(viol_np[mask, -1])),
        }
    print("OVERALL:", json.dumps(overall, indent=2))

    mean_violation_curve = {}
    for cond in ('warm', 'cold'):
        mask = np.array([c == cond for c in all_cond])
        mean_violation_curve[cond] = viol_np[mask].mean(axis=0).tolist()

    result = dict(
        checkpoint=args.checkpoint, epoch=ckpt.get('epoch'), n_per_shape=args.n_per_shape,
        n_outer=args.n_outer, inner_steps=args.inner_steps, tol=args.tol,
        shapes=shape_names, per_shape=results, overall=overall,
        mean_violation_curve=mean_violation_curve,
        timing=dict(alm_wall_s=alm_wall, n_candidates=C,
                   ms_per_candidate_outer_iter=alm_wall / max(C * args.n_outer, 1) * 1000),
    )
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out}  (total wall {time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
