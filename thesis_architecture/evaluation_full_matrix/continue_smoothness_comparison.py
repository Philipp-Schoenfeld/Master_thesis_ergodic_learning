r"""
continue_smoothness_comparison.py -- extend an existing run by more iterations
================================================================================
Philipp's observation on the v6 convergence plot (2026-10-12): most variants are still
decreasing noticeably at iteration 2000, not flattened -- confirmed by inspecting
`convergence.npz` directly (e.g. `linear_bspline_svgd`'s median error falls ~35 % per decade
of iterations even between 1500 and 2000, no visible plateau). This script continues the
SAME swarms for more iterations instead of re-running from scratch, by taking each candidate's
final stored state as the new initial curve.

Exactness of the continuation, per solver family:
  TSVEC-SVGD   EXACT. The update has no hidden state besides the particle positions (a
               memoryless Gauss-Newton-preconditioned step each iteration), so resuming from
               the final particles reproduces bit-for-bit what one longer run would have done.
  Adam-SVGD    APPROXIMATE. The Adam moments (m, v, t) are not persisted in the DB and restart
               at zero on resume, so there is a brief re-warm-up transient (the per-coordinate
               step size readapts over the first ~20-50 iterations) that a single uninterrupted
               run would not have. The moments adapt quickly, so this is a minor perturbation,
               not a different trajectory family.
  Sun (both flows)  APPROXIMATE. The optimised control sequence is not persisted either; the
               final DENSE trajectory is PID-tracked back into a fresh control sequence --
               the same mechanism `sun_refine.prepare()` already uses for every fresh init --
               which reproduces the POSITIONS closely but re-derives velocity/acceleration from
               finite differences of that curve rather than continuing the actual optimised
               state.
`cfm_only` has no iterations and is left untouched (copied as-is).

Resumable: `--target_n_iters` is the ABSOLUTE iteration count each cell should reach; a cell
already at or past it is skipped, so a crashed/interrupted run can be restarted with the same
command. Writes to a NEW `--out_tag` (a copy of `--src_tag`'s DB), so the source run is kept.

Usage:
  python continue_smoothness_comparison.py --src_tag smoothness_comparison_v6_20261012 \
      --out_tag smoothness_comparison_v6cont_20261012 --target_n_iters 4000
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'), os.path.join(_root, 'bsplinax-main'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import smoothness_db as sdb                                        # noqa: E402
import run_smoothness_comparison as rsc1                           # noqa: E402
import run_smoothness_comparison_v2 as v2                          # noqa: E402
from state_codec import unpack_states                              # noqa: E402


def final_dense_curve(row, B32):
    """Decode a stored row's states and return its LAST state as an (N_POINTS, 2) dense curve,
    the representation `run_cell` expects as `init_curves`."""
    states = row['cps']                                            # (n_states, nxi, 2)
    last = states[-1]
    if row['log_space'] == 'cps':
        return np.einsum('pi,id->pd', B32, last).astype(np.float32)
    return last.astype(np.float32)                                 # already (N_POINTS, 2)


def continue_cell(conn, shape, variant, truth_np, extra_iters, metric_stride, B32, device):
    """-> True if extended, False if nothing to do (cfm_only or already skipped by caller)."""
    rows = list(sdb.iter_runs(conn, shape=shape, variant=variant, with_states=True))
    if not rows:
        return False
    old_n_iters = rows[0]['n_iters']
    init_curves = np.stack([final_dense_curve(r, B32) for r in rows])
    new_rows = v2.run_cell(variant, init_curves, truth_np, extra_iters, metric_stride, B32, device)

    out_rows = []
    for old, new in zip(rows, new_rows):
        old_states = old['cps']                                    # (old_n_states, nxi, 2)
        new_states = unpack_states(new['states'], new['n_states'], new['nxi'])
        assert old['log_space'] == new['log_space'] and old['nxi'] == new['nxi'], \
            (shape, variant, old['log_space'], new['log_space'], old['nxi'], new['nxi'])
        combined = np.concatenate([old_states[:-1], new_states], axis=0)   # drop duplicated boundary
        n_iters_new = old_n_iters + extra_iters
        n_states_new = n_iters_new + 1
        assert combined.shape[0] == n_states_new, (combined.shape, n_states_new)
        smooth = np.concatenate([old['smooth_series'][:-1], new['smooth_series']])
        plen = np.concatenate([old['path_len_series'][:-1], new['path_len_series']])
        out_rows.append(dict(
            cand_idx=old['cand_idx'], init_param=None, n_iters=n_iters_new, nxi=old['nxi'],
            n_points=old['n_points'], log_space=old['log_space'],
            smoothness_weight=old['smoothness_weight'], n_states=n_states_new,
            init_curve=old['init_curve'],                          # keep the ORIGINAL start
            states=rsc1._pack(combined), smooth_series=smooth, path_len_series=plen))
    sdb.save_runs(conn, shape, variant, out_rows)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--src_tag', required=True)
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--target_n_iters', type=int, required=True)
    ap.add_argument('--shapes', type=str, default=None)
    ap.add_argument('--variants', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--time_budget_h', type=float, default=None)
    args = ap.parse_args()

    src_db = os.path.join(_here, 'results', args.src_tag, 'smoothness_comparison.db')
    out_dir = os.path.join(_here, 'results', args.out_tag)
    out_db = os.path.join(out_dir, 'smoothness_comparison.db')
    if not os.path.exists(out_db):
        if not os.path.exists(src_db):
            raise FileNotFoundError(src_db)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(src_db, out_db)
        print(f"[continue] copied {args.src_tag} -> {args.out_tag}")
    else:
        print(f"[continue] {args.out_tag} already exists -> resuming")

    conn = sdb.open_db(out_db)
    cfg = sdb.get_meta(conn, 'config', {})
    shapes = args.shapes.split(',') if args.shapes else cfg['shapes']
    variants = args.variants.split(',') if args.variants else cfg['variants']
    metric_stride = cfg.get('metric_stride', 1)
    B32 = rsc1.basis_matrix()
    device = args.device

    t0 = time.time()
    n_done = n_skipped = 0
    try:
        for shape in shapes:
            truth_np = sdb.load_truth(conn, shape).astype(np.float32)
            for variant in variants:
                if variant == 'cfm_only':
                    continue
                if (args.time_budget_h is not None
                        and (time.time() - t0) / 3600.0 > args.time_budget_h):
                    print("[continue] time budget reached, stopping.")
                    raise StopIteration
                existing = list(sdb.iter_runs(conn, shape=shape, variant=variant))
                if existing and existing[0]['n_iters'] >= args.target_n_iters:
                    n_skipped += 1
                    continue
                extra = args.target_n_iters - (existing[0]['n_iters'] if existing else 0)
                t_cell = time.time()
                continue_cell(conn, shape, variant, truth_np, extra, metric_stride, B32, device)
                n_done += 1
                print(f"[continue] ({shape}/{variant}) +{extra} it. done in "
                     f"{time.time() - t_cell:.1f}s; {n_done} cells extended, "
                     f"{(time.time() - t0) / 60:.1f} min total", flush=True)
    except StopIteration:
        pass
    except KeyboardInterrupt:
        print("[continue] interrupted -- finished cells are already committed.")
    cfg['n_iters'] = args.target_n_iters
    sdb.set_meta(conn, 'config', cfg)
    conn.commit()
    print(f"[continue] done: {n_done} cells extended, {n_skipped} already at target, "
         f"{(time.time() - t0) / 60:.1f} min -> {out_db}")


if __name__ == '__main__':
    main()
