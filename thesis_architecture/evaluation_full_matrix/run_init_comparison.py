r"""
run_init_comparison.py
========================
Controlled comparison of 7 trajectory-initialization methods, each refined
by 2 SVGD solver families, on all 25 holdout shapes (Philipp's request,
2026-10-19). Same per-(shape, variant) cell / SQLite-everything-logged
architecture as `run_smoothness_comparison_v2.py` (see its module docstring
for the shared background), widened along two axes instead of one:

  methods (7)  cfm            `apply_cfm_belief.CfmPlanner` (the thesis's
                               main conditional flow-matching network).
               selfsup        `selfsup_planner.SelfsupPlanner`.
               ebm            `ebm_planner.EbmPlanner` -- "Energy-Policy",
                               the Ranking-NCE energy-based second
                               generative head (`src/methods/EBM_RankingNCE`).
               dfl            `flow_matching_cond_spectral_crossattn
                               .SpectralCrossAttnFlowNetwork` trained with
                               `--dfl_weight` (Decision-Focused Learning,
                               predicts B-spline control points + lambda0;
                               only the control points are used here -- no
                               ALM/SE(3) solver in this flat-2D comparison).
               linear         `init_baselines.linear_angle_path` (n_init
                               evenly-spaced chords).
               random_walk    `init_baselines.random_walk_path` from the
                               shape's dataset start x0.
               heuristic      `ergodic_solver._generate_initial_trajectory`
                               via `generate_dataset.get_initial_trajectory`
                               -- the ACTUAL dataset-generation heuristic
                               (verified by reading `generate_dataset.py`'s
                               own init call; NOT `init_baselines
                               .heuristic_peak_path` / `gui_heuristic_path`,
                               two separate, non-identical reimplementations
                               that exist elsewhere in the repo but are not
                               what the dataset generator itself uses).
                               Deterministic given (shape_def, x0) -- a
                               small fixed jitter (same JITTER=0.01 scale
                               `SvgdRefiner` already uses internally) is
                               added per candidate so the `n_init` "samples"
                               aren't byte-identical duplicates, which would
                               degenerate the TSVEC-SVGD joint swarm (no
                               internal particle jitter of its own, unlike
                               `BatchedSvgdTorch`) to a zero-diversity kernel
                               at iteration 0.

  solvers (2)  tsvec_svgd     `exploration/common/tsvec_svgd.TsvecSvgd` --
                               paper-faithful (arXiv:2603.09458) Gauss-Newton
                               preconditioned SVGD, no dynamics model, all
                               `n_init` candidates ONE mutually-interacting
                               swarm. This is "TSVEC-SVGD" in Philipp's own
                               plot labeling (matches `run_smoothness_
                               comparison_v2.py`'s round-2 usage of the name).
               adam_svgd      `svgd_batched.BatchedSvgdTorch` -- the
                               project's older TSVEC-*like* solver: a
                               256-particle Adam-stepped SVGD swarm PER
                               candidate (`common/svgd_refine.py`'s energy
                               weights), independent across candidates. This
                               is "Adam-SVGD" (confirmed: the only solver in
                               the repo with Adam optimizer state; `tsvec_
                               svgd.py` has none).

`variant` in the DB = f"{method}__{solver}" (14 combinations/shape). Every
intermediate state of every trajectory is logged (`init_comparison_db.py`),
so mean/variance of all 6 metrics (ergodic error, coverage, smoothness,
path length, fraction of steps outside the target's support, time-to-
convergence/n90) can be recomputed at any step without re-running a solver.

Example
-------
    # Self-test without GPU / checkpoints (seconds)
    python run_init_comparison.py --dry_run --shapes A --n_init 3 \
        --n_iters 10 --out_tag smoke

    # Small timed test on the cluster, informs the full run's time budget
    python run_init_comparison.py --out_tag init_comparison_test --n_shapes 2

    # Full run (NOT started automatically -- ask before running)
    python run_init_comparison.py --out_tag init_comparison_YYYYMMDD
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
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src'),
           os.path.join(_root, 'src', 'methods', 'EBM_RankingNCE')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import init_comparison_db as idb                                   # noqa: E402
import run_smoothness_comparison as rsc1                           # noqa: E402

NXI = rsc1.NXI                     # 25
N_POINTS = rsc1.N_POINTS           # 128
DEGREE = rsc1.DEGREE               # 5
TRUTH_RES = rsc1.TRUTH_RES         # 96
SEED = 0
JITTER = 0.01                      # matches common.svgd_refine.SvgdRefiner.JITTER

METHODS = ('cfm', 'selfsup', 'ebm', 'dfl', 'linear', 'random_walk', 'heuristic')
SOLVERS = ('tsvec_svgd', 'adam_svgd')

_DATASET_DB = os.path.join(_arch, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')


def task_seed(*parts):
    return rsc1.task_seed(*parts)


# ── per-shape metadata (x0, shape_def) ───────────────────────────────────────

def load_shape_meta(names):
    """-> {shape: (x0 (2,) float32 np, shape_def dict)}."""
    conn = sqlite3.connect(_DATASET_DB)
    rows = conn.execute(
        "SELECT shape_name, x0, density_params FROM ergodic_pairs WHERE split='val'").fetchall()
    conn.close()
    meta = {}
    for name, x0_json, dens_json in rows:
        if name in names and name not in meta:
            meta[name] = (np.asarray(json.loads(x0_json), dtype=np.float32),
                          json.loads(dens_json))
    missing = [n for n in names if n not in meta]
    if missing:
        raise KeyError(f"no x0/density_params found for shapes: {missing}")
    return meta


# ── new metric: fraction of waypoints outside the target's support ──────────

def outside_fraction(dense_states, truth_np, thresh_frac=0.02):
    """dense_states: (S, T, 2) in [0,1]^2; truth_np: (R, R). -> (S,) float32,
    fraction of waypoints whose (bilinearly-interpolated) truth density is
    below `thresh_frac * truth_np.max()`."""
    R = truth_np.shape[-1]
    thresh = thresh_frac * float(truth_np.max())
    xs = np.clip(dense_states[..., 0], 0.0, 1.0) * (R - 1)
    ys = np.clip(dense_states[..., 1], 0.0, 1.0) * (R - 1)
    x0i = np.floor(xs).astype(np.int64)
    y0i = np.floor(ys).astype(np.int64)
    x1i = np.clip(x0i + 1, 0, R - 1)
    y1i = np.clip(y0i + 1, 0, R - 1)
    fx, fy = xs - x0i, ys - y0i
    v = (truth_np[y0i, x0i] * (1 - fx) * (1 - fy) + truth_np[y0i, x1i] * fx * (1 - fy) +
        truth_np[y1i, x0i] * (1 - fx) * fy + truth_np[y1i, x1i] * fx * fy)
    return (v < thresh).mean(axis=-1).astype(np.float32)


# ── per-method dense-curve (n_init, N_POINTS, 2) generation ─────────────────

class PlannerBundle:
    """Lazily-constructed, per-process-cached planner handles for the
    particle-conditioned methods (cfm/selfsup/ebm) and the DFL spectral net
    (conditioned differently -- see `dfl_curves`)."""

    def __init__(self, args, device, methods):
        self.device = device
        self.cfm = self.selfsup = self.ebm = None
        self.dfl_model = self.dfl_ckpt = self.dfl_val_shapes = None
        if 'cfm' in methods:
            if args.dry_run:
                self.cfm = rsc1.DummyPlanner()
            else:
                import apply_cfm_belief as acb
                from run_eval_matrix import DEFAULT_CKPT
                self.cfm = acb.CfmPlanner(ckpt=args.ckpt or DEFAULT_CKPT, nxi=NXI,
                                          pts=N_POINTS, device=device)
        if 'selfsup' in methods:
            if args.dry_run:
                self.selfsup = rsc1.DummyPlanner()
            else:
                from selfsup_planner import SelfsupPlanner
                self.selfsup = SelfsupPlanner(ckpt=args.selfsup_ckpt, device=device, pts=N_POINTS)
        if 'ebm' in methods:
            if args.dry_run:
                self.ebm = rsc1.DummyPlanner()
            else:
                from ebm_planner import EbmPlanner
                self.ebm = EbmPlanner(ckpt=args.ebm_ckpt, device=device, pts=N_POINTS)
        if 'dfl' in methods and not args.dry_run:
            from flow_matching_cond_spectral_crossattn import SpectralCrossAttnFlowNetwork
            from flow_matching_runner_ergodic import _load_shapes
            ck = torch.load(args.dfl_ckpt, map_location=device, weights_only=True)
            model = SpectralCrossAttnFlowNetwork(
                nxi=ck['nxi'], nd=ck['nd'], D=ck['D'], S=ck['S'],
                n_lambda=ck['n_lambda'], predict_lambda=ck['predict_lambda']).to(device)
            model.load_state_dict(ck['model_state_dict'])
            model.eval()
            self.dfl_model, self.dfl_ckpt = model, ck
            _train_shapes, val_shapes, _td, _vd = _load_shapes(ck['nxi'])
            self.dfl_val_shapes = val_shapes


def method_curves(method, bundle, shape, truth, truth_np, n_init, device, x0, shape_def, args):
    """-> (n_init, N_POINTS, 2) float32 numpy dense curves.

    cfm/selfsup go through `rsc1.sample_cfm_curves` uniformly (samples a belief
    particle cloud from the true density, calls `.plan`/`.render`) --
    deliberately WITHOUT passing `x0`/`start`, matching this script's lineage
    (`run_smoothness_comparison*.py`): none of the solvers below pin the start
    point either (`TsvecSvgd.run` has no start arg at all; `BatchedSvgdTorch
    .run` is always called with `start=None`), so feeding a method's start-
    conditioning here but not another's would make the comparison asymmetric
    rather than more realistic.

    ebm is a FORCED exception: the real checkpoint was trained with
    `use_start_cond=True`, and `EnergyNet.forward` hard-asserts `start is
    not None` -- confirmed by a cluster smoke test crash (job 174564,
    2026-10-19), not a design choice. Without `x0` the model cannot run at
    all, so it is passed here even though no other method gets it.
    """
    seed = task_seed(shape, method)
    if method == 'cfm':
        return rsc1.sample_cfm_curves(bundle.cfm, truth, n_init, device, seed)
    if method == 'selfsup':
        return rsc1.sample_cfm_curves(bundle.selfsup, truth, n_init, device, seed)
    if method == 'ebm':
        start_t = torch.as_tensor(x0, dtype=torch.float32, device=device)
        if isinstance(bundle.ebm, rsc1.DummyPlanner):
            cps = bundle.ebm.plan(None, n_candidates=n_init, start=start_t)
        else:
            import apply_cfm_belief as acb
            torch.manual_seed(seed)
            parts = acb.phi_particles(truth, rsc1.N_PARTICLES, mode='uniform', device=truth.device)
            cps = bundle.ebm.plan(parts, n_candidates=n_init, start=start_t)
        return bundle.ebm.render(cps).detach().cpu().numpy().astype(np.float32)
    if method == 'dfl':
        if args.dry_run:
            return rsc1.sample_cfm_curves(rsc1.DummyPlanner(), truth, n_init, device, seed)
        from flow_matching_cond_spectral_crossattn import generate_spectral_trajectories
        from flow_matching_runner_ergodic import trajectory_to_spectral
        ck = bundle.dfl_ckpt
        base_cp = bundle.dfl_val_shapes[shape]
        spec_np, k_idx_np = trajectory_to_spectral(base_cp, S=ck['S'])
        spec_t = torch.tensor(spec_np, dtype=torch.float32, device=device)
        k_idx_t = torch.tensor(k_idx_np, dtype=torch.long, device=device)
        torch.manual_seed(seed)
        gen_cps, _lambda0 = generate_spectral_trajectories(
            bundle.dfl_model, spec_t, k_idx_t, num_samples=n_init, nxi=ck['nxi'], nd=ck['nd'],
            steps=100, device=device, cfg_weight=2.0)
        B32 = rsc1.basis_matrix()
        return np.einsum('pi,kid->kpd', B32, gen_cps.detach().cpu().numpy()).astype(np.float32)
    if method == 'linear':
        curves, _angles = rsc1.linear_curves(n_init)
        return curves
    if method == 'random_walk':
        from init_baselines import random_walk_path
        x0_t = torch.as_tensor(x0, dtype=torch.float32)
        curves = torch.stack([random_walk_path(n_points=N_POINTS, seed=seed * 1000 + i, start=x0_t)
                              for i in range(n_init)])
        return curves.numpy().astype(np.float32)
    if method == 'heuristic':
        from generate_dataset import get_initial_trajectory
        from exploration_optimierung.mission import resample_arclength
        base = get_initial_trajectory(shape_def, x0=tuple(x0.tolist()))
        base_t = resample_arclength(torch.as_tensor(base, dtype=torch.float32), N_POINTS)
        rng = np.random.default_rng(seed)
        jitter = rng.normal(scale=JITTER, size=(n_init, N_POINTS, 2)).astype(np.float32)
        curves = base_t.numpy()[None] + jitter
        curves[:, 0] = base_t.numpy()[0]            # keep the start point exact
        return np.clip(curves, 0.0, 1.0).astype(np.float32)
    raise KeyError(method)


# ── solver cells (mirrors run_smoothness_comparison_v2.run_cell_tsvec/sun) ──

def run_cell_tsvec_svgd(init_curves, truth_np, n_iters, metric_stride, B32, device,
                        outside_thresh):
    from common.tsvec_svgd import TsvecSvgd
    from ergodic_energy_torch import target_coeffs_from_grid, W_SMOOTH
    C = init_curves.shape[0]
    P0 = np.linalg.lstsq(B32, init_curves.transpose(1, 0, 2).reshape(N_POINTS, C * 2),
                         rcond=None)[0].reshape(NXI, C, 2).transpose(1, 0, 2).astype(np.float32)
    svgd = TsvecSvgd(B=B32, K=10, device=device, w_smooth=W_SMOOTH)
    phi_k = target_coeffs_from_grid(torch.as_tensor(truth_np, dtype=torch.float32, device=device),
                                    svgd.k_idx)
    out = svgd.run(P0, phi_k, n_iters=n_iters, record=True)
    log = out['log'].detach().cpu().numpy().transpose(1, 0, 2, 3)      # (C, n_states, nxi, 2)
    return _rows_from_log(log, init_curves, n_iters, metric_stride, B32, truth_np,
                          outside_thresh, NXI, 'cps', W_SMOOTH)


def run_cell_adam_svgd(init_curves, truth_np, n_iters, metric_stride, B32, device,
                       outside_thresh, seeds):
    from svgd_batched import BatchedSvgdTorch, K_FREQ
    from ergodic_energy_torch import target_coeffs_from_grid, make_k_grid
    bs = BatchedSvgdTorch(B32, device=device)
    # NOTE: BatchedSvgdTorch's internal Fourier grid is K_FREQ=8 (SvgdRefiner.K,
    # 64 modes), NOT the project's "standard" K_DEFAULT=10 used by tsvec_svgd /
    # the ergodic-error metric elsewhere -- phi_k must be built at the SAME K
    # the solver itself uses internally, or the coefficient<->frequency mapping
    # silently mismatches (same [k1,k2] nested-loop order in both, verified
    # against `svgd_batched.BatchedSvgdTorch.__init__`, so only K differs).
    k_idx, _Lambda = make_k_grid(K=K_FREQ)
    phi_k = target_coeffs_from_grid(
        torch.as_tensor(truth_np, dtype=torch.float32, device=device),
        torch.as_tensor(k_idx, dtype=torch.float32, device=device)).detach().cpu().numpy()
    out = bs.run(init_curves.astype(np.float64), phi_k, None, seeds, n_iters, record=True)
    log = out['cps'].detach().cpu().numpy() if hasattr(out['cps'], 'detach') else np.asarray(out['cps'])
    return _rows_from_log(log, init_curves, n_iters, metric_stride, B32, truth_np,
                          outside_thresh, NXI, 'cps', 0.0)


def _rows_from_log(log, init_curves, n_iters, metric_stride, B32, truth_np, outside_thresh,
                   nxi, log_space, w_smooth):
    """log: (C, n_iters+1, nxi, 2) -> list of row dicts for `idb.save_runs`."""
    C = log.shape[0]
    midx = list(range(0, n_iters + 1, metric_stride))
    if midx[-1] != n_iters:
        midx.append(n_iters)
    rows = []
    for c in range(C):
        states_m = log[c][midx]
        dense = rsc1.render_cps_states(B32, states_m) if log_space == 'cps' else states_m
        smooth, plen = rsc1.smoothness_and_pathlen(dense)
        outside = outside_fraction(dense, truth_np, outside_thresh)
        full_s = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full_p = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full_o = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full_s[midx], full_p[midx], full_o[midx] = smooth, plen, outside
        rows.append(dict(
            cand_idx=c, init_param=None, n_iters=n_iters, nxi=nxi, n_points=N_POINTS,
            log_space=log_space, smoothness_weight=w_smooth, n_states=n_iters + 1,
            init_curve=init_curves[c], states=rsc1._pack(log[c]),
            smooth_series=full_s, path_len_series=full_p, outside_series=full_o))
    return rows


def run_cell(solver, init_curves, truth_np, n_iters, metric_stride, B32, device, outside_thresh):
    if solver == 'tsvec_svgd':
        return run_cell_tsvec_svgd(init_curves, truth_np, n_iters, metric_stride, B32, device,
                                   outside_thresh)
    if solver == 'adam_svgd':
        seeds = list(range(init_curves.shape[0]))
        return run_cell_adam_svgd(init_curves, truth_np, n_iters, metric_stride, B32, device,
                                  outside_thresh, seeds)
    raise KeyError(solver)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True,
                    help='results/<out_tag>/init_comparison.db is written.')
    ap.add_argument('--shapes', type=str, default=None)
    ap.add_argument('--n_shapes', type=int, default=25)
    ap.add_argument('--methods', type=str, default=','.join(METHODS))
    ap.add_argument('--solvers', type=str, default=','.join(SOLVERS))
    ap.add_argument('--n_init', type=int, default=10, help='Trajectories per distribution.')
    ap.add_argument('--n_iters', type=int, default=1500)
    ap.add_argument('--outside_thresh', type=float, default=0.02,
                    help="Fraction of a shape's own max density below which a "
                         'waypoint counts as "outside" the target support.')
    ap.add_argument('--metric_stride', type=int, default=1)
    ap.add_argument('--ckpt', type=str, default=None, help='CFM checkpoint override.')
    ap.add_argument('--selfsup_ckpt', type=str, default=None)
    ap.add_argument('--ebm_ckpt', type=str, default=None)
    ap.add_argument('--dfl_ckpt', type=str, default=None,
                    help='Required (no default) unless --dry_run or "dfl" not in --methods.')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--time_budget_h', type=float, default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    methods = [m for m in args.methods.split(',') if m]
    solvers = [s for s in args.solvers.split(',') if s]
    for m in methods:
        if m not in METHODS:
            raise KeyError(f"unknown method {m!r}; known: {METHODS}")
    for s in solvers:
        if s not in SOLVERS:
            raise KeyError(f"unknown solver {s!r}; known: {SOLVERS}")
    if 'dfl' in methods and not args.dry_run and not args.dfl_ckpt:
        raise ValueError("--dfl_ckpt is required when 'dfl' is in --methods (no safe default "
                         "-- see plan: cluster-only checkpoint, rsync-verified path in "
                         "run_job_init_comparison.bash).")

    device = args.device
    from common.data import load_truth
    labels = [s.strip() for s in args.shapes.split(',')] if args.shapes else None
    names, truths = load_truth(labels=labels, n=args.n_shapes, split='val',
                               resolution=TRUTH_RES, device=device)
    print(f"[init_cmp] {len(names)} shapes: {names}", flush=True)
    shape_meta = load_shape_meta(names)

    out_dir = os.path.join(_here, 'results', args.out_tag)
    db_path = os.path.join(out_dir, 'init_comparison.db')
    conn = idb.open_db(db_path)
    B32 = rsc1.basis_matrix()
    idb.save_basis(conn, B32, NXI, N_POINTS, DEGREE)
    idb.set_meta(conn, 'config', dict(
        shapes=names, methods=methods, solvers=solvers, n_init=args.n_init,
        n_iters=args.n_iters, outside_thresh=args.outside_thresh,
        metric_stride=args.metric_stride, nxi=NXI, n_points=N_POINTS, degree=DEGREE,
        truth_res=TRUTH_RES, seed=SEED, dry_run=args.dry_run,
        started=time.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()

    bundle = PlannerBundle(args, device, methods)

    t0 = time.time()
    n_cells_done = n_cells_skipped = 0
    n_blocks = len(names) * len(methods) * len(solvers)
    block_i = 0
    try:
        for shape, truth in zip(names, truths):
            truth_np = truth.detach().cpu().numpy().astype(np.float32)
            idb.save_truth(conn, shape, truth_np)
            x0, shape_def = shape_meta[shape]
            curves_cache = {}
            for method in methods:
                for solver in solvers:
                    block_i += 1
                    if (args.time_budget_h is not None
                            and (time.time() - t0) / 3600.0 > args.time_budget_h):
                        print("[init_cmp] time budget reached, stopping submission.")
                        raise StopIteration
                    variant = f"{method}__{solver}"
                    have = idb.existing_cands(conn, shape, variant)
                    need_idx = [i for i in range(args.n_init) if i not in have]
                    if not need_idx:
                        n_cells_skipped += 1
                        continue

                    if method not in curves_cache:
                        curves_cache[method] = method_curves(
                            method, bundle, shape, truth, truth_np, args.n_init, device,
                            x0, shape_def, args)
                    init_curves = curves_cache[method]

                    t_cell = time.time()
                    rows = run_cell(solver, init_curves, truth_np, args.n_iters,
                                    args.metric_stride, B32, device, args.outside_thresh)
                    idb.save_runs(conn, shape, method, solver, rows)
                    n_cells_done += 1
                    el = time.time() - t0
                    print(f"[init_cmp] block {block_i}/{n_blocks} ({shape}/{variant}) "
                         f"done in {time.time() - t_cell:.1f}s; {n_cells_done} cells stored, "
                         f"{el / 60:.1f} min total", flush=True)
    except StopIteration:
        pass
    except KeyboardInterrupt:
        print("[init_cmp] interrupted -- finished cells are already committed.")
    print(f"[init_cmp] done: {n_cells_done} cells stored, {n_cells_skipped} "
         f"already complete, {(time.time() - t0) / 60:.1f} min -> {db_path}")


if __name__ == '__main__':
    main()
