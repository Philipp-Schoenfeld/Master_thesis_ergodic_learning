r"""
run_smoothness_comparison_v2.py
=================================
Second round of the big smoothness comparison (Philipp's request,
2026-10-09), same DB/plot machinery as `run_smoothness_comparison.py` (see
its module docstring for the shared background: all 25 holdout shapes, no
unknown-region split, EVERY SVGD state logged, 30-trajectory
mean/covariance), but a different set of 7 variants contrasting TWO solver
families at `--n_iters` (2000) iterations each:

  sun_*        `exploration/common/sun_refine.py` -- Sun et al.'s FM-Stein
               solver, a DYNAMICS MODEL (PointMassLQR or, new in this round,
               JerkPenalizedLQR) simulated and Stein-gradient-steered; this is
               what `run_smoothness_comparison.py` already used throughout.
  *_svgd       `exploration/common/tsvec_svgd.py` -- NEW: a direct
               reimplementation of Li et al. 2026 ("Stein Variational Ergodic
               Surface Coverage with SE(3) Constraints", arXiv:2603.09458 --
               the TSVEC paper CLAUDE.md already cites as this thesis's
               theoretical downstream-solver reference), reduced to flat 2D:
               NO dynamics model -- particles (B-spline control points or raw
               waypoints) are moved directly by a preconditioned
               (Gauss-Newton) Stein-variational update. All 30 trajectories
               of a cell are now a SINGLE, mutually-interacting SVGD swarm
               (Philipp's explicit choice, confirmed 2026-10-09), not 30
               independent runs -- see `tsvec_svgd.py`'s module docstring for
               the full derivation and every equation it is checked against.

The 7 variants
---------------
  cfm_only                 Unchanged from round 1: one CFM forward pass
                           (B-spline control points), no refinement.
  cfm_bspline_svgd         The SAME 30 CFM samples as the initial particles of
                           ONE `tsvec_svgd` swarm, on NXI=25 B-spline control
                           points, with smoothness term (w_smooth=15, see
                           TSVEC_SPEC) -- "regular, as already implemented"
                           SVGD, but now with the paper's actual algorithm
                           instead of the sun/dynamics one.
  linear_bspline_svgd      Same `tsvec_svgd` swarm, linear-chord initial
                           particles instead of CFM, same NXI=25, same
                           w_smooth. Paired with `cfm_bspline_svgd`:
                           does the CFM prior help THIS solver too.
  linear_waypoint_svgd     Same swarm, linear init, raw waypoints
                           (nxi=N_POINTS, no B-spline), w_smooth=0 -- isolates
                           the "no smoothness term at all" case for this
                           solver family.
  linear_waypoint_svgd_smooth  Same as above, w_smooth=15 -- does adding the
                           smoothness term back,
                           on raw waypoints, recover it "for free" here too,
                           mirroring round 1's `linear_svgd_raw_smooth`
                           finding for the sun/dynamics solver.
  linear_sun_pointmass     `sun_refine` with `dynamics='pointmass'`, linear
                           init, raw-waypoint-projected output -- identical to
                           round 1's `linear_svgd_raw`, re-run at 2000
                           iterations for a fair comparison at the same
                           budget as the new variants.
  linear_sun_jerk          `sun_refine` with `dynamics='jerk'` (NEW:
                           `JerkPenalizedLQR`, a smoother dynamics model --
                           acceleration is itself a Q-penalised state and the
                           control is jerk, not acceleration directly, so
                           consecutive-step acceleration jumps are
                           structurally discouraged), linear init, raw
                           waypoints. Answers Philipp's "Sun_svgd ... auf
                           einer glatteren Dynamik".

Reuses `run_smoothness_comparison.py`'s helpers (CFM sampling, linear-chord
init, the smoothness/path-length metric, state packing) verbatim via import
-- only the per-cell solver call and the variant table are new.

Example
-------
    # Self-test without GPU / checkpoint (seconds)
    python run_smoothness_comparison_v2.py --dry_run --shapes A --n_init 4 \
        --n_iters 10 --out_tag smoke_v2

    # Small timed test on the cluster, informs the full run's time budget
    python run_smoothness_comparison_v2.py --out_tag smoothness_v2_test --n_shapes 2

    # Full run (NOT started automatically -- ask before running)
    python run_smoothness_comparison_v2.py --out_tag smoothness_comparison_v2_YYYYMMDD
"""
import argparse
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

import smoothness_db as sdb                                        # noqa: E402
import run_smoothness_comparison as rsc1                           # noqa: E402

NXI = rsc1.NXI
N_POINTS = rsc1.N_POINTS
DEGREE = rsc1.DEGREE
TRUTH_RES = rsc1.TRUTH_RES
SEED = rsc1.SEED

VARIANTS = ('cfm_only', 'cfm_bspline_svgd', 'linear_bspline_svgd',
           'linear_waypoint_svgd', 'linear_waypoint_svgd_smooth',
           'linear_sun_pointmass', 'linear_sun_jerk')
NEEDS_CFM = {'cfm_only', 'cfm_bspline_svgd'}
NEEDS_LINEAR = {'linear_bspline_svgd', 'linear_waypoint_svgd',
                'linear_waypoint_svgd_smooth', 'linear_sun_pointmass', 'linear_sun_jerk'}
#: Energy weights: the project's own 2D TSVEC weights (`ergodic_energy_torch.
#: W_ERGODIC/W_SMOOTH/W_BOUNDARY` = `SE3_SVGD/tsvec_2d.py`: 600 / 15 / 30),
#: `tsvec_svgd.TsvecSvgd`'s defaults; w_smooth = 0.0 for the "no smoothness"
#: ablation. NOT the paper's 3.0 / 5.0 -- at that energy scale the target
#: is nearly flat and the solver diffuses instead of covering (round-2 job
#: 164986: linear-init ergodic error stuck at ~20-26). That run also used a
#: w_smooth = 1e-4 workaround for an apparent smoothness "freeze", which
#: turned out to be a symptom of the same energy-scale problem; see
#: `tsvec_svgd.py`'s "Energy scale and update-rule fixes" for the diagnosis.
#: tau = `tsvec_svgd.TAU` (the paper's 0.1).
from ergodic_energy_torch import W_SMOOTH                          # noqa: E402

TSVEC_SPEC = {
    'cfm_bspline_svgd':            dict(nxi=NXI, w_smooth=W_SMOOTH),
    'linear_bspline_svgd':          dict(nxi=NXI, w_smooth=W_SMOOTH),
    'linear_waypoint_svgd':         dict(nxi=N_POINTS, w_smooth=0.0),
    'linear_waypoint_svgd_smooth':  dict(nxi=N_POINTS, w_smooth=W_SMOOTH),
}
SUN_SPEC = {
    'linear_sun_pointmass': dict(dynamics='pointmass'),
    'linear_sun_jerk':      dict(dynamics='jerk'),
}


def run_cell_tsvec(variant, init_curves, truth_np, n_iters, metric_stride, B32, device):
    """One `tsvec_svgd` joint-swarm run -> list of row dicts (one per
    particle), same contract as `run_smoothness_comparison.run_cell`."""
    from common.tsvec_svgd import TsvecSvgd
    from ergodic_energy_torch import target_coeffs_from_grid
    spec = TSVEC_SPEC[variant]
    nxi = spec['nxi']
    C = init_curves.shape[0]
    if nxi == NXI:
        B = B32
        P0 = np.linalg.lstsq(B32, init_curves.transpose(1, 0, 2).reshape(N_POINTS, C * 2),
                             rcond=None)[0].reshape(NXI, C, 2).transpose(1, 0, 2).astype(np.float32)
    else:
        B = None
        P0 = init_curves.astype(np.float32)
    svgd = TsvecSvgd(B=B, K=10, device=device, w_smooth=spec['w_smooth'])
    phi_k = target_coeffs_from_grid(torch.as_tensor(truth_np, dtype=torch.float32,
                                                    device=device), svgd.k_idx)
    out = svgd.run(P0, phi_k, n_iters=n_iters, record=True)
    log = out['log'].detach().cpu().numpy().transpose(1, 0, 2, 3)   # (C, n_states, nxi, 2)

    midx = list(range(0, n_iters + 1, metric_stride))
    if midx[-1] != n_iters:
        midx.append(n_iters)
    rows = []
    for c in range(C):
        states = log[c]
        states_m = states[midx]
        dense = rsc1.render_cps_states(B32, states_m) if nxi == NXI else states_m
        smooth, plen = rsc1.smoothness_and_pathlen(dense)
        full = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full_p = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full[midx] = smooth
        full_p[midx] = plen
        rows.append(dict(
            cand_idx=c, init_param=None, n_iters=n_iters, nxi=nxi, n_points=N_POINTS,
            log_space='cps' if nxi == NXI else 'raw', smoothness_weight=spec['w_smooth'],
            n_states=n_iters + 1, init_curve=init_curves[c], states=rsc1._pack(states),
            smooth_series=full, path_len_series=full_p))
    return rows


def run_cell_sun(variant, init_curves, truth_np, n_iters, metric_stride, B32, device):
    """One `sun_refine` (via `BatchedSunTorch`) run, raw-waypoint output,
    `dynamics` selecting PointMassLQR vs. JerkPenalizedLQR."""
    from svgd_batched import BatchedSunTorch
    spec = SUN_SPEC[variant]
    C = init_curves.shape[0]
    bs = BatchedSunTorch(np.eye(N_POINTS, dtype=np.float32), device=device)
    seeds = [0] * C
    out = bs.run(init_curves.astype(np.float64), truth_np, None, seeds, n_iters, record=True,
                smoothness_weight=0.0, log_space='raw', dynamics=spec['dynamics'])
    cps = out['cps'].detach().cpu().numpy() if hasattr(out['cps'], 'detach') else np.asarray(out['cps'])

    midx = list(range(0, n_iters + 1, metric_stride))
    if midx[-1] != n_iters:
        midx.append(n_iters)
    rows = []
    for c in range(C):
        states = cps[c]
        states_m = states[midx]
        smooth, plen = rsc1.smoothness_and_pathlen(states_m)
        full = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full_p = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full[midx] = smooth
        full_p[midx] = plen
        rows.append(dict(
            cand_idx=c, init_param=None, n_iters=n_iters, nxi=N_POINTS, n_points=N_POINTS,
            log_space='raw', smoothness_weight=0.0, n_states=n_iters + 1,
            init_curve=init_curves[c], states=rsc1._pack(states),
            smooth_series=full, path_len_series=full_p))
    return rows


def run_cell(variant, init_curves, truth_np, n_iters, metric_stride, B32, device):
    if variant == 'cfm_only':
        smooth, plen = rsc1.smoothness_and_pathlen(init_curves)
        rows = []
        for c in range(init_curves.shape[0]):
            states = init_curves[c][None].astype(np.float32)
            rows.append(dict(
                cand_idx=c, init_param=None, n_iters=0, nxi=N_POINTS, n_points=N_POINTS,
                log_space='dense', smoothness_weight=0.0, n_states=1,
                init_curve=init_curves[c], states=rsc1._pack(states),
                smooth_series=smooth[c:c + 1], path_len_series=plen[c:c + 1]))
        return rows
    if variant in TSVEC_SPEC:
        return run_cell_tsvec(variant, init_curves, truth_np, n_iters, metric_stride, B32, device)
    return run_cell_sun(variant, init_curves, truth_np, n_iters, metric_stride, B32, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True,
                    help='results/<out_tag>/smoothness_comparison.db is written.')
    ap.add_argument('--shapes', type=str, default=None)
    ap.add_argument('--n_shapes', type=int, default=25)
    ap.add_argument('--variants', type=str, default=','.join(VARIANTS))
    ap.add_argument('--n_init', type=int, default=30,
                    help='Particles per swarm / trajectories per distribution.')
    ap.add_argument('--n_iters', type=int, default=2000)
    ap.add_argument('--metric_stride', type=int, default=1)
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--time_budget_h', type=float, default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    variants = [v for v in args.variants.split(',') if v]
    for v in variants:
        if v not in VARIANTS:
            raise KeyError(f"unknown variant {v!r}; known: {VARIANTS}")

    device = args.device
    from common.data import load_truth
    labels = [s.strip() for s in args.shapes.split(',')] if args.shapes else None
    names, truths = load_truth(labels=labels, n=args.n_shapes, split='val',
                               resolution=TRUTH_RES, device=device)
    print(f"[smooth_v2] {len(names)} shapes: {names}", flush=True)

    out_dir = os.path.join(_here, 'results', args.out_tag)
    db_path = os.path.join(out_dir, 'smoothness_comparison.db')
    conn = sdb.open_db(db_path)
    B32 = rsc1.basis_matrix()
    sdb.save_basis(conn, B32, NXI, N_POINTS, DEGREE)
    sdb.set_meta(conn, 'config', dict(
        shapes=names, variants=variants, n_init=args.n_init, n_iters=args.n_iters,
        metric_stride=args.metric_stride, nxi=NXI, n_points=N_POINTS, degree=DEGREE,
        truth_res=TRUTH_RES, seed=SEED, dry_run=args.dry_run, round=2,
        tsvec_spec=TSVEC_SPEC, sun_spec=SUN_SPEC,
        started=time.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()

    planner = None
    if NEEDS_CFM & set(variants):
        if args.dry_run:
            planner = rsc1.DummyPlanner()
        else:
            import apply_cfm_belief as acb
            from run_eval_matrix import DEFAULT_CKPT
            planner = acb.CfmPlanner(ckpt=args.ckpt or DEFAULT_CKPT, nxi=NXI, pts=N_POINTS,
                                     device=device)
            assert planner.nxi == NXI, f"planner nxi={planner.nxi}, expected {NXI}"

    t0 = time.time()
    n_cells_done = n_cells_skipped = 0
    n_blocks = len(names) * len(variants)
    block_i = 0
    try:
        for shape, truth in zip(names, truths):
            truth_np = truth.detach().cpu().numpy().astype(np.float32)
            sdb.save_truth(conn, shape, truth_np)
            cfm_curves_cache = None
            lin_curves_cache = None
            for variant in variants:
                block_i += 1
                if (args.time_budget_h is not None
                        and (time.time() - t0) / 3600.0 > args.time_budget_h):
                    print("[smooth_v2] time budget reached, stopping submission.")
                    raise StopIteration
                have = sdb.existing_cands(conn, shape, variant)
                need_idx = [i for i in range(args.n_init) if i not in have]
                if not need_idx:
                    n_cells_skipped += 1
                    continue

                if variant in NEEDS_CFM:
                    if cfm_curves_cache is None:
                        cfm_curves_cache = rsc1.sample_cfm_curves(
                            planner, truth, args.n_init, device, rsc1.task_seed(shape, 'cfm'))
                    init_curves = cfm_curves_cache
                else:
                    if lin_curves_cache is None:
                        lin_curves_cache, _angles = rsc1.linear_curves(args.n_init)
                    init_curves = lin_curves_cache

                t_cell = time.time()
                rows = run_cell(variant, init_curves, truth_np, args.n_iters,
                                args.metric_stride, B32, device)
                sdb.save_runs(conn, shape, variant, rows)
                n_cells_done += 1
                el = time.time() - t0
                print(f"[smooth_v2] block {block_i}/{n_blocks} ({shape}/{variant}) "
                     f"done in {time.time() - t_cell:.1f}s; {n_cells_done} cells stored, "
                     f"{el / 60:.1f} min total", flush=True)
    except StopIteration:
        pass
    except KeyboardInterrupt:
        print("[smooth_v2] interrupted -- finished cells are already committed.")
    print(f"[smooth_v2] done: {n_cells_done} cells stored, {n_cells_skipped} "
         f"already complete, {(time.time() - t0) / 60:.1f} min -> {db_path}")


if __name__ == '__main__':
    main()
