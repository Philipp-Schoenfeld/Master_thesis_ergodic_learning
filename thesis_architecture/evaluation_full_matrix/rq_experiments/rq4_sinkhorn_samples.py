"""
rq4_sinkhorn_samples.py
========================
RQ4 (non-smooth, sample-only targets, mirrors Sun et al. Fig. 7/9): does
particle-conditioned CFM match FM-Sinkhorn's coverage of a non-smooth target
-- Sun's own ten test icons (`sun_icons/`, point clouds, no density) -- at a
fraction of the solver time, without ever needing a score or a density grid?

Methods:
    fm_sinkhorn   Sun's Sinkhorn-divergence flow (`exp_common.timed_sinkhorn_solve`,
                  `sinkhorn_jax.py` -- `ott-jax` does not import in this
                  environment, see that module's docstring), from a straight
                  line toward the icon's centroid.
    cfm_raw       one CFM forward pass, conditioned DIRECTLY on the icon's
                  points as (x, y, weight) particles -- no density grid at
                  any point (`exp_common.cfm_raw_curve_from_particles`).
    cfm_svgd      the CFM curve refined with `SvgdRefiner` (Sun's FM-Stein
                  backend, same as the data generator) against a Gaussian-
                  splatted density grid of the icon (`exp_common.icon_density_grid`)
                  -- shown for contrast: refinement, unlike raw conditioning,
                  does need SOME grid representation here.

Metric: `exp_common.coverage_error`, the Fourier ergodic metric of Mathew and
Mezic (2011) -- the same metric Sun et al. cite for their own Q3.A/Q3.B
"coverage error" (see the thesis-doc conversation and `coverage_error`'s own
docstring for how this was confirmed).

Usage:
    python rq4_sinkhorn_samples.py --icons star,heart --n_starts 2 --out_tag smoke
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
_arch = os.path.dirname(_mat)
for _p in (_here, _mat, os.path.join(_arch, 'exploration'),
          os.path.join(_arch, 'ergodic_dataset_generator')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import exp_common as C                                             # noqa: E402
import paper_style as ps                                           # noqa: E402
from common.sun_refine import run_batch as sun_run_batch           # noqa: E402

DT, TSTEPS = 0.05, 200
METHODS = ['fm_sinkhorn', 'cfm_raw', 'cfm_svgd']
GRID_RES = 96


FIELDS = ['icon', 'start_idx', 'method', 'coverage_error', 'time_s']


def _load_done(csv_path):
    """Resumability for cluster jobs (24h limit): an (icon, start_idx) run is
    atomic in `run_one` (all methods land together), so it is "done" once ANY
    row for that key is on disk."""
    done = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                done.add((row['icon'], int(row['start_idx'])))
    return done


def random_start(rng, margin=0.08):
    return tuple(rng.uniform(margin, 1 - margin, size=2))


def run_one(icon, start, planner, device, args):
    """-> dict(method -> dict(curve, coverage_error, time_s))."""
    pts = C.load_icon(icon)
    out = {}

    x0j, u0 = C.sun_cold_x0_u0(start, DT, TSTEPS)
    res = C.timed_sinkhorn_solve(pts, x0j, u0, checkpoints=[args.sinkhorn_iters],
                                 epsilon=args.epsilon, n_sinkhorn_iters=args.sinkhorn_inner)[0]
    out['fm_sinkhorn'] = dict(curve=res['traj_xy'], time_s=res['time_s'])

    curve, t_cfm = C.cfm_raw_curve_from_particles(planner, pts, start=start, device=device)
    out['cfm_raw'] = dict(curve=curve, time_s=t_cfm)

    phi = C.icon_density_grid(icon, res=GRID_RES)
    refined = sun_run_batch(curve[None], phi, np.asarray(start)[None], args.svgd_iters, planner.nxi)
    out['cfm_svgd'] = dict(curve=refined['final_pos'][0], time_s=t_cfm)  # refine time not separately metered here

    for m in out:
        out[m]['coverage_error'] = C.coverage_error(out[m]['curve'], pts)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--icons', type=str, default=','.join(C.ICON_NAMES))
    ap.add_argument('--n_starts', type=int, default=3)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--sinkhorn_iters', type=int, default=300)
    ap.add_argument('--sinkhorn_inner', type=int, default=50,
                    help='Sinkhorn iterations per flow-matching step (inner loop).')
    ap.add_argument('--epsilon', type=float, default=0.01)
    ap.add_argument('--svgd_iters', type=int, default=100)
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop starting new trials after this many hours; already-written '
                         'rows stay valid, re-run with the same --out_tag to resume.')
    args = ap.parse_args()

    icons = [i for i in args.icons.split(',') if i]
    from run_eval_matrix import DEFAULT_CKPT
    from run_ideal_matrix import build_particle_planner
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    planner = build_particle_planner(args.ckpt or DEFAULT_CKPT, device)
    warm_pts = C.load_icon(icons[0])
    C.cfm_raw_curve_from_particles(planner, warm_pts, start=(0.5, 0.5), device=device)

    out_dir = os.path.join(_mat, 'results', args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'rq4_sinkhorn_samples.csv')

    # Starts are precomputed per icon (deterministic in `args.seed0`) rather
    # than drawn inline, so that "start 0 of icon X" is the same value
    # whether or not this run resumes a previous one -- needed below, where
    # the qualitative figure recomputes start 0 fresh regardless of what the
    # resumable data loop skipped.
    rng = np.random.default_rng(args.seed0)
    starts = {icon: [random_start(rng) for _ in range(args.n_starts)] for icon in icons}

    done = _load_done(csv_path)
    need_header = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if need_header:
            w.writeheader()
        t0 = time.time()
        for icon in icons:
            for s in range(args.n_starts):
                if C.time_budget_exceeded(t0, args.time_budget_h):
                    print(f"[rq4] time budget reached at icon={icon}, stopping.", flush=True)
                    break
                if (icon, s) in done:
                    continue
                res = run_one(icon, starts[icon][s], planner, device, args)
                for m in METHODS:
                    w.writerow(dict(icon=icon, start_idx=s, method=m,
                                    coverage_error=res[m]['coverage_error'], time_s=res[m]['time_s']))
                f.flush()
            print(f"[rq4] {icon} done", flush=True)
    print(f"[rq4] -> {csv_path}")

    # Qualitative figure: recomputed fresh for start 0 of each icon (cheap --
    # len(icons) calls -- and correct under resume, unlike reusing whatever
    # the data loop above happened to compute this run).
    qualitative_row = {icon: {m: run_one(icon, starts[icon][0], planner, device, args)[m]['curve']
                              for m in METHODS} for icon in icons}

    plot_dir = os.path.join(out_dir, 'plots')
    import pandas as pd
    df = pd.read_csv(csv_path)
    ps.box({m: df[df['method'] == m]['coverage_error'].to_numpy() for m in METHODS},
          'Coverage error on non-smooth targets (Sun icons, Mathew & Mezic metric)',
          'Coverage error (lower is better)',
          os.path.join(plot_dir, 'rq4_coverage_error_box.png'), methods=METHODS, log_y=True)

    # Fig. 7-style grid: one row per icon, one column per method, plus a
    # raw-straight-line "init" column for visual reference.
    rows_grid = []
    for icon, curves in qualitative_row.items():
        full_pts = C.load_icon(icon, n_particles=None)
        row = [dict(trajs=[curves[m]], particles=full_pts) for m in METHODS]
        rows_grid.append(row)
    ps.trajectory_panel(rows_grid, os.path.join(plot_dir, 'rq4_qualitative_grid.png'),
                        ncols=len(METHODS), col_titles=[ps.METHOD_LABELS.get(m, m) for m in METHODS],
                        row_titles=list(qualitative_row.keys()),
                        suptitle='Non-smooth targets: Sun icons (cf. Sun et al. Fig. 7)')
    print(f"[rq4] plots -> {plot_dir}")

    for m in METHODS:
        v = df[df['method'] == m]['coverage_error']
        print(f"  {m:14s} median coverage_error={v.median():.4g}  "
             f"median time={df[df['method'] == m]['time_s'].median():.2f}s")


if __name__ == '__main__':
    main()
