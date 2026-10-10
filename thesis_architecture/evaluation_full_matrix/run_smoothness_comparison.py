r"""
run_smoothness_comparison.py
=============================
Big smoothness comparison (Philipp's request, 2026-10-08), directly on ALL
holdout shapes, no unknown-region split (`ergodic_dataset_775.db`'s `val`
split has none -- see module docstring of `exploration/common/data.py`).
For every shape, 5 variants are each run as a DISTRIBUTION of 30 trajectories
(Philipp's own meeting note, 2026-10-01: "wir muessten ausnutzen, dass wir
ein generatives Modell haben -- sample nicht 1, sondern n moegliche Pfade als
Pfadverteilung ... vergleiche dann die Verteilungen ... immer Mean und
Standard Deviation plotten" / "wie sehen Mean und Covariance aus bei einer
Verteilung von 30 Trajektorien ... bei den verschiedensten
SVGD-Zwischenschritten"), and EVERY intermediate SVGD state of every
trajectory is written to a SQLite DB (`smoothness_db.py`, states compressed
with `state_codec.py` exactly like `mission_db.py`) so that mean/covariance of
smoothness (and, from the stored states, of the trajectories themselves) can
be recomputed at any step without re-running the solver.

The 5 variants
---------------
  cfm_only              One batched CFM forward pass (B-spline control
                         points, the network's native output), NO refinement.
                         1 state per trajectory (the network sample itself).
  cfm_svgd              The SAME 30 CFM samples, refined with the "regular,
                         as already implemented" SVGD -- the current default
                         refiner backend since 2026-10-06 (CLAUDE.md): Sun et
                         al.'s FM-Stein solver, the exact core of the dataset
                         generator (`exploration/common/sun_refine.py` /
                         `ergodic_dataset_generator/ergodic_solver.py`), on
                         NXI=25 B-spline control points, `--n_iters` (600,
                         the project's "ground truth" iteration count)
                         iterations against the TRUE density.
  linear_svgd_bspline    Same solver, SAME 30 linear-chord initialisations
                         (`init_baselines.linear_angle_path`, angles evenly
                         spaced over [0, 180) deg) instead of CFM, still on
                         NXI=25 B-spline control points. Paired with
                         `cfm_svgd` (same solver, same n_iters): answers
                         Philipp's note "bringt der Prior etwas, wie schnell
                         konvergiert SVGD mit [vs. ohne] gelerntem Prior".
  linear_svgd_raw        Same solver, same linear inits, but WITHOUT the
                         B-spline bottleneck: the solver optimises the raw
                         simulated waypoints directly (`nxi = N_POINTS`,
                         `log_space='raw'` -- a genuine extension of
                         `sun_refine.py`, see its module docstring: the
                         existing code already supported a raw-waypoint
                         FINAL curve via `nxi == T`, but the logged
                         INTERMEDIATE states always went through a B-spline
                         fit regardless of `nxi`; `log_space='raw'` logs
                         linearly-resampled raw positions instead). Answers
                         Philipp's note "Svdg direkt auf einer Trajektorie
                         ohne B-Spline [...], wie lange muss man optimieren
                         um eine vergleichbare Trajektorie zu erzeugen".
  linear_svgd_raw_smooth Same as `linear_svgd_raw`, plus an explicit
                         smoothness force in the Stein gradient
                         (`smoothness_weight`, another new opt-in extension
                         of `sun_refine.py` -- the "sun" backend has no
                         smoothness term today, unlike the older "tsvec"
                         backend's built-in `W_SMOOTH`). Answers Philipp's
                         note "fuege fuer die SVGD-Optimierung noch
                         Smoothness hinzu". `--smoothness_weight` (default
                         below) is PROVISIONAL -- the units of this solver's
                         log-score are not directly comparable to the old
                         tsvec energy's W_SMOOTH=15, so the right order of
                         magnitude is something to read off the test run,
                         not something this script can derive on its own.

All of variants 2-5 share the identical solver core, iteration count and
target density; only the initialisation distribution and the representation
(B-spline vs. raw, smoothness force on/off) differ -- a controlled comparison,
not five unrelated pipelines.

Metrics stored per logged state, per trajectory (`smoothness_db.py`):
`metrics_explore_exploit.smoothness_energy`'s own definition (arclength
resample to 128 points, then `W_SMOOTH * sum(accel^2)`) and path length.
`plot_smoothness_comparison.py` turns these per-trajectory series into the
30-trajectory distribution's mean and covariance (scalar variance of
smoothness, plus -- since every control point of every state is stored -- the
mean trajectory and its pointwise positional covariance, exactly the
"Mean und Covariance einer Verteilung von 30 Trajektorien" Philipp asked for).

Execution model: one GPU batch per (shape, variant) -- all `n_init` (30)
trajectories of that cell refine together via `svgd_batched.BatchedSunTorch`
(JAX, vmapped). Resumable: a finished (shape, variant, cand_idx) row is
skipped on re-run with the same `--out_tag`.

Example
-------
    # Self-test without GPU / checkpoint (seconds)
    python run_smoothness_comparison.py --dry_run --shapes A --n_init 2 \
        --n_iters 20 --out_tag smoke

    # Small timed test on the cluster (a handful of shapes, see
    # run_job_smoothness_comparison.bash for the self-test + timing step)
    python run_smoothness_comparison.py --out_tag smoothness_test --n_shapes 2

    # Full run (NOT started automatically -- ask before running)
    python run_smoothness_comparison.py --out_tag smoothness_comparison_YYYYMMDD
"""
import argparse
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                       # noqa: E402

import smoothness_db as sdb                                        # noqa: E402
import run_svgd_convergence as rsc                                 # noqa: E402

NXI = rsc.NXI                      # 25, B-spline control points
N_POINTS = rsc.N_POINTS            # 128, dense render resolution (also the
                                    # raw-waypoint variants' point count)
DEGREE = rsc.DEGREE
TRUTH_RES = rsc.TRUTH_RES          # 96
N_PARTICLES = 256                  # CFM conditioning cloud size (project default)
SEED = 0

VARIANTS = ('cfm_only', 'cfm_svgd', 'linear_svgd_bspline', 'linear_svgd_raw',
           'linear_svgd_raw_smooth')
#: which variants need a CFM forward pass / a linear-chord init.
NEEDS_CFM = {'cfm_only', 'cfm_svgd'}
NEEDS_LINEAR = {'linear_svgd_bspline', 'linear_svgd_raw', 'linear_svgd_raw_smooth'}
#: which variants run the "sun" FM-Stein solver at all (cfm_only does not).
NEEDS_SVGD = {'cfm_svgd', 'linear_svgd_bspline', 'linear_svgd_raw', 'linear_svgd_raw_smooth'}
#: per-variant (nxi, log_space, uses_smoothness_force).
VARIANT_SPEC = {
    'cfm_svgd':              dict(nxi=NXI, log_space='cps', smooth=False),
    'linear_svgd_bspline':   dict(nxi=NXI, log_space='cps', smooth=False),
    'linear_svgd_raw':       dict(nxi=N_POINTS, log_space='raw', smooth=False),
    'linear_svgd_raw_smooth': dict(nxi=N_POINTS, log_space='raw', smooth=True),
}
#: provisional -- see module docstring. Tune after reading the test run's
#: linear_svgd_raw vs. linear_svgd_raw_smooth smoothness numbers.
DEFAULT_SMOOTHNESS_WEIGHT = 200.0


def task_seed(*parts):
    return rsc.task_seed(*parts)


def basis_matrix():
    return rsc.basis_matrix(NXI, N_POINTS, DEGREE)


# ── CFM / linear initialisations ─────────────────────────────────────────────

class DummyPlanner:
    """Stand-in for the CFM network (`--dry_run`): random smooth control
    points. Only for self-tests without GPU / checkpoint."""
    nxi = NXI

    def __init__(self):
        self.B = torch.as_tensor(basis_matrix())

    def plan(self, _particles, n_candidates=1, **_kw):
        g = torch.Generator().manual_seed(1234 + int(torch.randint(0, 10 ** 6, (1,))))
        steps = torch.randn(n_candidates, self.nxi, 2, generator=g) * 0.12
        cps = (0.5 + steps.cumsum(dim=1) * 0.5).clamp(0.05, 0.95)
        return cps

    def render(self, cps):
        return torch.einsum('pi,kid->kpd', self.B, cps.float())


def sample_cfm_curves(planner, truth, n_init, device, seed):
    """n_init CFM samples conditioned on the TRUE density (no partial
    knowledge -- this experiment is about solver/representation comparison,
    not belief states). -> (n_init, N_POINTS, 2) float32 np."""
    import apply_cfm_belief as acb
    torch.manual_seed(seed)
    if isinstance(planner, DummyPlanner):
        parts = None
        cps = planner.plan(parts, n_candidates=n_init)
    else:
        parts = acb.phi_particles(truth, N_PARTICLES, mode='uniform', device=truth.device)
        cps = planner.plan(parts, n_candidates=n_init)
    curves = planner.render(cps).detach().cpu().numpy().astype(np.float32)
    return curves


def linear_curves(n_init, n_points=N_POINTS):
    """n_init evenly-spaced chords through the workspace centre (the
    unconstrained `linear` baseline of `run_svgd_convergence.py`).
    -> (n_init, n_points, 2) float32 np, (angle_deg,) list."""
    from init_baselines import linear_angle_path
    angles = [180.0 * i / n_init for i in range(n_init)]
    curves = np.stack([linear_angle_path(a, n_points).numpy() for a in angles]).astype(np.float32)
    return curves, angles


# ── metrics over logged states ──────────────────────────────────────────────

@torch.no_grad()
def smoothness_and_pathlen(curves_np):
    """(N, T, 2) numpy dense curves -> (smooth (N,), path_len (N,)) float32,
    `metrics_explore_exploit.smoothness_energy`'s own definition inlined and
    batched: arclength-resample every curve to SMOOTHNESS_RESAMPLE_POINTS,
    then `W_SMOOTH * sum(accel^2)` / path length of the resampled curve."""
    from exploration_optimierung.mission import resample_arclength
    from metrics_explore_exploit import SMOOTHNESS_RESAMPLE_POINTS
    from ergodic_energy_torch import smoothness_term, W_SMOOTH
    curves = torch.as_tensor(curves_np, dtype=torch.float32)
    resampled = torch.stack([resample_arclength(c, SMOOTHNESS_RESAMPLE_POINTS) for c in curves])
    smooth = smoothness_term(resampled, w=W_SMOOTH).numpy().astype(np.float32)
    path_len = (resampled[:, 1:] - resampled[:, :-1]).norm(dim=-1).sum(dim=1).numpy().astype(np.float32)
    return smooth, path_len


def render_cps_states(B32, cps):
    """(S, nxi, 2) -> (S, N_POINTS, 2), `np.einsum` batched over the state axis."""
    return np.einsum('pi,sid->spd', B32, cps)


# ── one (shape, variant) cell ───────────────────────────────────────────────

def run_cell(variant, init_curves, init_params, truth_np, n_iters, bs_cps, bs_raw,
            smoothness_weight, metric_stride, B32):
    """-> list of row dicts (one per candidate) ready for `sdb.save_runs`."""
    C = init_curves.shape[0]
    if variant == 'cfm_only':
        smooth, plen = smoothness_and_pathlen(init_curves)
        rows = []
        for c in range(C):
            states = init_curves[c][None].astype(np.float32)           # (1, N_POINTS, 2)
            rows.append(dict(
                cand_idx=c, init_param=init_params[c] if init_params else None,
                n_iters=0, nxi=N_POINTS, n_points=N_POINTS, log_space='dense',
                smoothness_weight=0.0, n_states=1, init_curve=init_curves[c],
                states=_pack(states), smooth_series=smooth[c:c + 1],
                path_len_series=plen[c:c + 1]))
        return rows

    spec = VARIANT_SPEC[variant]
    bs = bs_cps if spec['log_space'] == 'cps' else bs_raw
    w = smoothness_weight if spec['smooth'] else 0.0
    seeds = [0] * C                                  # the "sun" solver is deterministic
    out = bs.run(init_curves.astype(np.float64), truth_np, None, seeds, n_iters,
                record=True, smoothness_weight=w, log_space=spec['log_space'])
    cps = out['cps'].detach().cpu().numpy() if hasattr(out['cps'], 'detach') else np.asarray(out['cps'])
    # cps: (C, n_iters+1, nxi, 2)
    midx = list(range(0, n_iters + 1, metric_stride))
    if midx[-1] != n_iters:
        midx.append(n_iters)
    rows = []
    for c in range(C):
        states = cps[c]                                              # (n_iters+1, nxi, 2)
        states_m = states[midx]
        dense = (render_cps_states(B32, states_m) if spec['log_space'] == 'cps'
                else states_m)                                        # raw states already dense
        smooth, plen = smoothness_and_pathlen(dense)
        full = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full_p = np.full(n_iters + 1, np.nan, dtype=np.float32)
        full[midx] = smooth
        full_p[midx] = plen
        rows.append(dict(
            cand_idx=c, init_param=init_params[c] if init_params else None,
            n_iters=n_iters, nxi=spec['nxi'], n_points=N_POINTS,
            log_space=spec['log_space'], smoothness_weight=w, n_states=n_iters + 1,
            init_curve=init_curves[c], states=_pack(states),
            smooth_series=full, path_len_series=full_p))
    return rows


def _pack(states):
    from state_codec import pack_states
    return pack_states(states)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True,
                    help='results/<out_tag>/smoothness_comparison.db is written.')
    ap.add_argument('--shapes', type=str, default=None,
                    help='Comma-separated shape names (default: all val shapes).')
    ap.add_argument('--n_shapes', type=int, default=25,
                    help='Used only without --shapes (default: all 25 holdout shapes).')
    ap.add_argument('--variants', type=str, default=','.join(VARIANTS))
    ap.add_argument('--n_init', type=int, default=30,
                    help='Trajectories per distribution (Philipp: 30).')
    ap.add_argument('--n_iters', type=int, default=600,
                    help="SVGD/FM-Stein iterations ('until converges' -- the "
                         "project's ground-truth convergence horizon, CLAUDE.md).")
    ap.add_argument('--smoothness_weight', type=float, default=DEFAULT_SMOOTHNESS_WEIGHT,
                    help='linear_svgd_raw_smooth only -- PROVISIONAL, see module docstring.')
    ap.add_argument('--metric_stride', type=int, default=1,
                    help='Compute smoothness/path-length only every k-th logged state '
                         '(all states are still stored regardless -- this only thins '
                         'the metric series if CPU time becomes the bottleneck).')
    ap.add_argument('--ckpt', type=str, default=None, help='CFM checkpoint override.')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop submitting new cells after this many hours; '
                         'finished cells are kept, re-run to resume.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Self-test: random control points instead of the CFM network, '
                         'tiny n_iters recommended (no checkpoint, works on CPU).')
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
    print(f"[smooth_cmp] {len(names)} shapes: {names}", flush=True)

    out_dir = os.path.join(_here, 'results', args.out_tag)
    db_path = os.path.join(out_dir, 'smoothness_comparison.db')
    conn = sdb.open_db(db_path)
    B32 = basis_matrix()
    sdb.save_basis(conn, B32, NXI, N_POINTS, DEGREE)
    sdb.set_meta(conn, 'config', dict(
        shapes=names, variants=variants, n_init=args.n_init, n_iters=args.n_iters,
        smoothness_weight=args.smoothness_weight, metric_stride=args.metric_stride,
        nxi=NXI, n_points=N_POINTS, degree=DEGREE, truth_res=TRUTH_RES, seed=SEED,
        dry_run=args.dry_run, started=time.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()

    planner = None
    if NEEDS_CFM & set(variants):
        if args.dry_run:
            planner = DummyPlanner()
        else:
            import apply_cfm_belief as acb
            from run_eval_matrix import DEFAULT_CKPT
            planner = acb.CfmPlanner(ckpt=args.ckpt or DEFAULT_CKPT, nxi=NXI, pts=N_POINTS,
                                     device=device)
            assert planner.nxi == NXI, f"planner nxi={planner.nxi}, expected {NXI}"

    from svgd_batched import BatchedSunTorch
    bs_cps = BatchedSunTorch(B32, device=device)
    bs_raw = BatchedSunTorch(np.eye(N_POINTS, dtype=np.float32), device=device)

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
                    print("[smooth_cmp] time budget reached, stopping submission.")
                    raise StopIteration
                have = sdb.existing_cands(conn, shape, variant)
                need_idx = [i for i in range(args.n_init) if i not in have]
                if not need_idx:
                    n_cells_skipped += 1
                    continue
                if len(need_idx) != args.n_init:
                    print(f"[smooth_cmp] {shape}/{variant}: resuming, "
                         f"{args.n_init - len(need_idx)}/{args.n_init} already stored "
                         "(partial cells are recomputed whole -- cheap relative to a cell).")

                if variant in NEEDS_CFM:
                    if cfm_curves_cache is None:
                        cfm_curves_cache = sample_cfm_curves(
                            planner, truth, args.n_init, device, task_seed(shape, 'cfm'))
                    init_curves, init_params = cfm_curves_cache, None
                else:
                    if lin_curves_cache is None:
                        lin_curves_cache = linear_curves(args.n_init)
                    init_curves, init_params = lin_curves_cache

                t_cell = time.time()
                rows = run_cell(variant, init_curves, init_params, truth_np, args.n_iters,
                                bs_cps, bs_raw, args.smoothness_weight, args.metric_stride, B32)
                sdb.save_runs(conn, shape, variant, rows)
                n_cells_done += 1
                el = time.time() - t0
                print(f"[smooth_cmp] block {block_i}/{n_blocks} ({shape}/{variant}) "
                     f"done in {time.time() - t_cell:.1f}s; {n_cells_done} cells stored, "
                     f"{el / 60:.1f} min total", flush=True)
    except StopIteration:
        pass
    except KeyboardInterrupt:
        print("[smooth_cmp] interrupted -- finished cells are already committed.")
    print(f"[smooth_cmp] done: {n_cells_done} cells stored, {n_cells_skipped} "
         f"already complete, {(time.time() - t0) / 60:.1f} min -> {db_path}")


if __name__ == '__main__':
    main()
