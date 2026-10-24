#!/usr/bin/env python3
r"""
eval_dfl_ergodicity_svgd.py
============================
Validates the DFL-trained checkpoint (flow_matching_runner_ergodic.py) on
the 25 holdout shapes in two ways:

(A) Ergodicity quality: for N_PER_SHAPE generated trajectories per shape,
    the ergodic Fourier error against the REAL target density
    (`shape_library.pdf_on_grid`, not the self-referential c_k(x1) used as
    training conditioning) -- `ergodic_energy_torch.py`'s solver-equivalent
    metric.

(B) SVGD/Sun warm-start convergence: the same generated trajectories as
    warm start for `BatchedSunTorch` (the batched Torch wrapper around the
    Sun et al. FM-Stein solver that also produced the training data), versus
    a random-walk cold-start baseline (`init_baselines.random_walk_path`,
    the established baseline in `run_svgd_convergence.py`). Logs the
    ergodic error at every SVGD iteration and reports iterations-to-
    convergence (plateau rule: first iteration after which the error
    improves by less than --conv_tol relative over the next --conv_window
    iterations), warm vs. cold, per shape and overall.

Usage:
  python eval_dfl_ergodicity_svgd.py --checkpoint <path> --n_per_shape 30 \
      --n_iters 400 --out results_dfl_eval.json

For a quick timing probe before sizing a full cluster run:
  python eval_dfl_ergodicity_svgd.py --checkpoint <path> --n_per_shape 2 \
      --n_iters 50 --shapes_limit 2 --out /tmp/probe.json
"""
import argparse
import json
import os
import sqlite3
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

from flow_matching_cond_spectral_crossattn import (                 # noqa: E402
    SpectralCrossAttnFlowNetwork, generate_spectral_trajectories,
)
from flow_matching_runner_ergodic import trajectory_to_spectral, _load_shapes  # noqa: E402
from obstacles import bspline_basis_matrix                           # noqa: E402
from ergodic_energy_torch import (                                   # noqa: E402
    make_k_grid, target_coeffs_from_grid, coeffs_from_points, W_ERGODIC, K_DEFAULT,
)
from shape_library import pdf_on_grid                                # noqa: E402
from svgd_batched import BatchedSunTorch                             # noqa: E402
from init_baselines import random_walk_path                          # noqa: E402

NXI = 25
T_DENSE = 128
BSPLINE_DEG = 5
TRUTH_RES = 96
_DB_PATH = os.path.join(_arch, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')


def iters_to_convergence(err, window, tol):
    """err: (C, W) numpy, W=n_iters+1. Plateau rule: first i such that
    min(err[i+1:i+1+window]) is within `tol` relative improvement of
    err[i]. -> (C,) int array; W-1 ("never converged in budget") if the
    rule never fires or there isn't enough runway to evaluate it."""
    C, W = err.shape
    out = np.full(C, W - 1, dtype=np.int32)
    if W - 1 <= window:
        return out
    fut = np.lib.stride_tricks.sliding_window_view(err[:, 1:], window_shape=window, axis=1)
    fut_min = fut.min(axis=2)                                        # (C, W-window)
    ref = err[:, :fut_min.shape[1]]
    rel_impr = (ref - fut_min) / np.maximum(ref, 1e-9)
    conv_mask = rel_impr < tol
    for c in range(C):
        hit = np.flatnonzero(conv_mask[c])
        if hit.size:
            out[c] = hit[0]
    return out


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--n_per_shape', type=int, default=30)
    p.add_argument('--n_iters', type=int, default=400)
    p.add_argument('--conv_window', type=int, default=None,
                   help="default: max(10, n_iters // 8)")
    p.add_argument('--conv_tol', type=float, default=0.05)
    p.add_argument('--shapes_limit', type=int, default=None,
                   help="only the first N holdout shapes -- for a timing probe")
    p.add_argument('--steps', type=int, default=100, help="CFM sampling steps")
    p.add_argument('--cfg_weight', type=float, default=2.0)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    if args.conv_window is None:
        args.conv_window = max(10, args.n_iters // 8)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"device={device}  n_per_shape={args.n_per_shape}  n_iters={args.n_iters}  "
          f"conv_window={args.conv_window}  conv_tol={args.conv_tol}")

    t0 = time.time()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = SpectralCrossAttnFlowNetwork(
        nxi=ckpt['nxi'], nd=ckpt['nd'], D=ckpt['D'], S=ckpt['S'],
        n_lambda=ckpt['n_lambda'], predict_lambda=ckpt['predict_lambda'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"loaded checkpoint epoch={ckpt.get('epoch')}  ({time.time()-t0:.1f}s)")

    _, holdout_shapes, _, holdout_densities = _load_shapes(ckpt['nxi'])
    shape_names = list(holdout_shapes.keys())
    if args.shapes_limit:
        shape_names = shape_names[:args.shapes_limit]
    print(f"{len(shape_names)} holdout shapes: {shape_names}")

    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT shape_name, x0 FROM ergodic_pairs WHERE split='val'")
    x0_by_name = {name: json.loads(x0) for name, x0 in cur.fetchall()}
    conn.close()

    # Fourier basis for the REAL-density ergodic metric (K_DEFAULT=10, solver-equivalent).
    k_idx_np, Lambda_np = make_k_grid(K=K_DEFAULT)
    k_idx = torch.tensor(k_idx_np, dtype=torch.float32, device=device)
    Lambda = torch.tensor(Lambda_np, dtype=torch.float32, device=device)

    B_np = bspline_basis_matrix(ckpt['nxi'], T_DENSE, BSPLINE_DEG).astype(np.float64)
    B_t = torch.tensor(B_np, dtype=torch.float32, device=device)

    all_curves, all_phis, all_starts, all_phik = [], [], [], []
    all_cond, all_shape = [], []
    gen_quality = {}    # shape -> dict of per-sample ergodic error (Part A, pre-SVGD)

    for shape in shape_names:
        base_cp = holdout_shapes[shape]
        spec_np, k_idx_cond_np = trajectory_to_spectral(base_cp, S=ckpt['S'])
        spec_t = torch.tensor(spec_np, dtype=torch.float32, device=device)
        k_idx_cond_t = torch.tensor(k_idx_cond_np, dtype=torch.long, device=device)

        gen_cps, _lambda0 = generate_spectral_trajectories(
            model, spec_t, k_idx_cond_t, num_samples=args.n_per_shape,
            nxi=ckpt['nxi'], nd=ckpt['nd'], steps=args.steps, device=device,
            cfg_weight=args.cfg_weight,
        )                                                            # (N, nxi, 2)

        dens_map, _gx, _gy = pdf_on_grid(json.loads(holdout_densities[shape]), resolution=TRUTH_RES)
        dens_t = torch.tensor(np.asarray(dens_map), dtype=torch.float32, device=device).clamp(min=0.0)
        phi_k = target_coeffs_from_grid(dens_t, k_idx)                # (100,)

        gen_curves = torch.einsum('tn,bnd->btd', B_t, gen_cps)        # (N, T, 2)
        c_k_gen = coeffs_from_points(gen_curves, k_idx)                # (N, 100)
        erg_err_gen = (0.5 * Lambda * (c_k_gen - phi_k.unsqueeze(0)).pow(2)).sum(-1) * W_ERGODIC
        gen_quality[shape] = {
            'mean': float(erg_err_gen.mean()), 'std': float(erg_err_gen.std()),
            'median': float(erg_err_gen.median()), 'per_sample': erg_err_gen.detach().cpu().tolist(),
        }
        print(f"  [{shape}] gen ergodic error: mean={gen_quality[shape]['mean']:.4g} "
              f"std={gen_quality[shape]['std']:.4g}")

        x0 = torch.tensor(x0_by_name[shape], dtype=torch.float32)
        rw_curves = torch.stack([
            random_walk_path(n_points=T_DENSE, seed=i, start=x0) for i in range(args.n_per_shape)
        ], dim=0).to(device)                                          # (N, T, 2)

        all_curves.append(gen_curves.detach().cpu().numpy())
        all_curves.append(rw_curves.detach().cpu().numpy())
        all_phis.append(np.broadcast_to(np.asarray(dens_map), (args.n_per_shape, TRUTH_RES, TRUTH_RES)))
        all_phis.append(np.broadcast_to(np.asarray(dens_map), (args.n_per_shape, TRUTH_RES, TRUTH_RES)))
        all_starts.append(np.broadcast_to(x0_by_name[shape], (args.n_per_shape, 2)))
        all_starts.append(np.broadcast_to(x0_by_name[shape], (args.n_per_shape, 2)))
        all_phik.append(phi_k.detach().cpu().numpy()[None].repeat(args.n_per_shape, axis=0))
        all_phik.append(phi_k.detach().cpu().numpy()[None].repeat(args.n_per_shape, axis=0))
        all_cond += ['warm'] * args.n_per_shape + ['cold'] * args.n_per_shape
        all_shape += [shape] * (2 * args.n_per_shape)

    curves_all = np.concatenate(all_curves, axis=0).astype(np.float64)
    phis_all = np.concatenate(all_phis, axis=0).astype(np.float64)
    starts_all = np.concatenate(all_starts, axis=0).astype(np.float64)
    phik_all = torch.tensor(np.concatenate(all_phik, axis=0), dtype=torch.float32, device=device)
    C = curves_all.shape[0]
    print(f"SVGD batch: C={C} candidates (warm+cold), n_iters={args.n_iters} — starting run_batch...")

    t1 = time.time()
    bst = BatchedSunTorch(B_np, device=device)
    out = bst.run(curves_all, phis_all, starts_all, seeds=None, n_iters=args.n_iters,
                 record=True, dynamics='pointmass')
    svgd_wall = time.time() - t1
    print(f"run_batch done in {svgd_wall:.1f}s ({svgd_wall / max(C * args.n_iters, 1) * 1000:.3f} ms/candidate-iter)")

    cps_log = out['cps'].to(device)                                   # (C, n_iters+1, nxi, 2) torch
    # Safety net: the expensive part (run_batch) is done: save the raw control-point
    # log to disk BEFORE the post-processing below, so a bug there (as happened once:
    # fourier_basis OOM'd computing all (C, W, T, 2) points at once, ~86 GiB) never
    # throws away ~3.5 min of solver compute again.
    raw_path = args.out + '.cps_log.npy'
    os.makedirs(os.path.dirname(raw_path) or '.', exist_ok=True)
    np.save(raw_path, cps_log.detach().cpu().numpy())
    print(f"Saved raw cps log -> {raw_path}  (safety net before post-processing)")

    # Chunked ergodic-error-per-iteration computation: fourier_basis's broadcast
    # (chunk, T, 1, 2) * (M, 2) must stay well under GPU memory, so flatten
    # (C, W) into one axis and process it in chunks instead of all at once.
    C_, W_ = cps_log.shape[0], cps_log.shape[1]
    cps_flat = cps_log.reshape(C_ * W_, cps_log.shape[2], 2)           # (C*W, nxi, 2)
    phik_flat = phik_all.unsqueeze(1).expand(C_, W_, -1).reshape(C_ * W_, -1)  # (C*W, 100)
    err_flat = torch.empty(C_ * W_, device=device)
    CHUNK = 20000
    for i in range(0, C_ * W_, CHUNK):
        sl = slice(i, min(i + CHUNK, C_ * W_))
        curve_chunk = torch.einsum('tn,bnd->btd', B_t, cps_flat[sl])   # (chunk, T, 2)
        c_k_chunk = coeffs_from_points(curve_chunk, k_idx)             # (chunk, 100)
        err_flat[sl] = (0.5 * Lambda * (c_k_chunk - phik_flat[sl]).pow(2)).sum(-1) * W_ERGODIC
    err_log_np = err_flat.reshape(C_, W_).detach().cpu().numpy()

    conv_iters = iters_to_convergence(err_log_np, args.conv_window, args.conv_tol)

    # ── aggregate per shape / condition ──
    svgd_results = {}
    idx = 0
    for shape in shape_names:
        svgd_results[shape] = {}
        for cond in ('warm', 'cold'):
            sl = slice(idx, idx + args.n_per_shape)
            idx += args.n_per_shape
            svgd_results[shape][cond] = {
                'conv_iters_median': float(np.median(conv_iters[sl])),
                'conv_iters_mean': float(np.mean(conv_iters[sl])),
                'final_err_median': float(np.median(err_log_np[sl, -1])),
                'final_err_mean': float(np.mean(err_log_np[sl, -1])),
                'init_err_median': float(np.median(err_log_np[sl, 0])),
                'conv_iters_per_sample': conv_iters[sl].tolist(),
            }
    assert idx == C

    overall = {}
    for cond in ('warm', 'cold'):
        mask = np.array([c == cond for c in all_cond])
        overall[cond] = {
            'conv_iters_median': float(np.median(conv_iters[mask])),
            'conv_iters_mean': float(np.mean(conv_iters[mask])),
            'final_err_median': float(np.median(err_log_np[mask, -1])),
        }
    print("OVERALL:", json.dumps(overall, indent=2))

    # Mean error-over-iteration curve per condition, for plotting later.
    mean_curve = {}
    for cond in ('warm', 'cold'):
        mask = np.array([c == cond for c in all_cond])
        mean_curve[cond] = err_log_np[mask].mean(axis=0).tolist()

    result = dict(
        checkpoint=args.checkpoint, epoch=ckpt.get('epoch'), n_per_shape=args.n_per_shape,
        n_iters=args.n_iters, conv_window=args.conv_window, conv_tol=args.conv_tol,
        shapes=shape_names, gen_quality=gen_quality, svgd_per_shape=svgd_results,
        svgd_overall=overall, mean_error_curve=mean_curve,
        timing=dict(svgd_wall_s=svgd_wall, ms_per_candidate_iter=svgd_wall / max(C * args.n_iters, 1) * 1000,
                   n_candidates=C),
    )
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {args.out}  (total wall {time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
