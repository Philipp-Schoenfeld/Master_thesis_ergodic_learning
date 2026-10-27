r"""
project_waypoint_variants.py -- "optimise waypoints, then project onto a B-spline" variants
===========================================================================================
The smoothness comparison (`run_smoothness_comparison_v2.py`) has two ways to get a smooth
trajectory out of a solver:

  * B-spline variants (`*_bspline_*`): the solver optimises the NXI = 25 control points
    directly, so every iterate is already a B-spline.
  * waypoint variants (`*_waypoint_*`, `linear_sun_*`): the solver optimises the raw
    N_POINTS = 128 waypoints; the result is rough unless a smoothness force is added.

This script adds the third option: take the waypoint solvers' own optimisation, and at every
logged iteration replace the 128-point curve by its least-squares B-spline fit with NXI control
points (control point 0 fixed to the curve's first waypoint, equal time parameter -- the same
projection as `ergodic_dataset_generator/study_waypoint_projection.py`'s `project`). The
projected state at iteration i is what you would get by stopping the waypoint solver at i and
projecting; the optimisation itself is NOT changed (no projection inside the loop).

Because the waypoint solvers' iterates are already stored in the source DB (every state logged),
nothing is re-optimised: the existing runs are only post-processed, so the new `<variant>_proj`
rows are exactly paired with their waypoint originals (same init curves, same seeds, same
iterates). Rows are stored with `log_space='cps'`, so `compute_convergence.py` and
`plot_smoothness_comparison.py` treat them like any B-spline variant.

Writes a new DB `results/<out_tag>/smoothness_comparison.db` that holds ONLY the projected
variants (+ truths, basis, config copied from the source). Resumable per (shape, variant,
candidate). Combine with the source run's results via `combine_projection_results.py`.

Usage:
  python project_waypoint_variants.py --src_tag smoothness_comparison_v6_20261012 \
      --out_tag smoothness_comparison_v6_wpproj_20261026
"""
import argparse
import os
import sqlite3
import sys
import time

import numpy as np

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
from state_codec import pack_states                                # noqa: E402

#: source waypoint variant -> projected variant
PROJ_SUFFIX = '_proj'
WAYPOINT_VARIANTS = ('linear_waypoint_svgd', 'linear_waypoint_svgd_smooth',
                     'linear_adam_waypoint', 'linear_adam_waypoint_smooth',
                     'linear_sun_pointmass', 'linear_sun_jerk')


def projection_operator(B32):
    """-> (B (T, nxi) float64, M (nxi - 1, T) float64) with
    rest_cps = M @ (Y - B[:, :1] * x0) for a waypoint curve Y (T, 2) starting at x0."""
    B = B32.astype(np.float64)
    return B, np.linalg.pinv(B[:, 1:])


def project_states(Y, B, M):
    """(S, T, 2) waypoint curves -> (S, nxi, 2) float64 control points, cp 0 = Y[:, 0]."""
    Y = Y.astype(np.float64)
    x0 = Y[:, 0, :]                                                # (S, 2)
    resid = Y - B[None, :, 0:1] * x0[:, None, :]                   # (S, T, 2)
    rest = np.einsum('kt,std->skd', M, resid)
    return np.concatenate([x0[:, None, :], rest], axis=1)


def project_row(row, B, M, B32, metric_stride):
    """One source run row (log_space='raw', with 'cps' states) -> projected run row dict."""
    if row['log_space'] != 'raw':
        raise ValueError(f"{row['shape']}/{row['variant']}/{row['cand_idx']}: expected "
                         f"log_space 'raw', got {row['log_space']!r}")
    n_states = row['n_states']
    n_iters = row['n_iters']
    proj = project_states(row['cps'], B, M).astype(np.float32)     # (n_states, nxi, 2)

    midx = list(range(0, n_iters + 1, metric_stride))
    if midx[-1] != n_iters:
        midx.append(n_iters)
    dense = rsc1.render_cps_states(B32, proj[midx])
    smooth, plen = rsc1.smoothness_and_pathlen(dense)
    full_s = np.full(n_states, np.nan, dtype=np.float32)
    full_p = np.full(n_states, np.nan, dtype=np.float32)
    full_s[midx] = smooth
    full_p[midx] = plen
    return dict(cand_idx=row['cand_idx'], init_param=row['init_param'], n_iters=n_iters,
                nxi=B.shape[1], n_points=row['n_points'], log_space='cps',
                smoothness_weight=row['smoothness_weight'], n_states=n_states,
                init_curve=row['init_curve'], states=pack_states(proj, wide_range=True),
                smooth_series=full_s, path_len_series=full_p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--src_tag', required=True, help='Result folder with the waypoint runs.')
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--variants', default=','.join(WAYPOINT_VARIANTS),
                    help='Comma-separated SOURCE waypoint variants to project.')
    ap.add_argument('--shapes', default=None, help='Comma-separated shapes (default: all in the source config).')
    args = ap.parse_args()

    src_path = os.path.join(_here, 'results', args.src_tag, 'smoothness_comparison.db')
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    src = sqlite3.connect(f'file:{src_path}?mode=ro', uri=True)
    cfg = sdb.get_meta(src, 'config')
    shapes = args.shapes.split(',') if args.shapes else cfg['shapes']
    src_variants = [v for v in args.variants.split(',') if v]
    present = {r[0] for r in src.execute('SELECT DISTINCT variant FROM runs')}
    missing = [v for v in src_variants if v not in present]
    if missing:
        raise KeyError(f'variants not in {args.src_tag}: {missing} (present: {sorted(present)})')
    stride = cfg['metric_stride']
    nxi, n_points, degree = cfg['nxi'], cfg['n_points'], cfg['degree']

    out_path = os.path.join(_here, 'results', args.out_tag, 'smoothness_comparison.db')
    conn = sdb.open_db(out_path)
    for table in ('truths', 'basis'):
        rows = src.execute(f'SELECT * FROM {table}').fetchall()
        ph = ','.join('?' * len(rows[0]))
        conn.executemany(f'INSERT OR REPLACE INTO {table} VALUES ({ph})', rows)
    out_variants = [v + PROJ_SUFFIX for v in src_variants]
    cfg_out = dict(cfg, variants=out_variants, shapes=shapes, derived_from=args.src_tag,
                   derived_by='project_waypoint_variants.py', projected_from=src_variants,
                   started=time.strftime('%Y-%m-%d %H:%M:%S'))
    sdb.set_meta(conn, 'config', cfg_out)
    conn.commit()

    B32 = rsc1.basis_matrix()
    assert B32.shape == (n_points, nxi), (B32.shape, n_points, nxi)
    B, M = projection_operator(B32)
    t0 = time.time()
    n_done = n_skipped = 0
    for si, shape in enumerate(shapes):
        for v_src, v_out in zip(src_variants, out_variants):
            have = sdb.existing_cands(conn, shape, v_out)
            rows_src = [r for r in sdb.iter_runs(src, shape=shape, variant=v_src, with_states=True)
                        if r['cand_idx'] not in have]
            if not rows_src:
                n_skipped += 1
                continue
            t_cell = time.time()
            sdb.save_runs(conn, shape, v_out,
                          [project_row(r, B, M, B32, stride) for r in rows_src])
            n_done += 1
            print(f'[proj] {shape} ({si + 1}/{len(shapes)}) {v_out}: {len(rows_src)} candidates in '
                  f'{time.time() - t_cell:.1f}s ({(time.time() - t0) / 60:.1f} min total)', flush=True)
    print(f'[proj] done: {n_done} cells projected, {n_skipped} already complete -> {out_path}')


if __name__ == '__main__':
    main()
