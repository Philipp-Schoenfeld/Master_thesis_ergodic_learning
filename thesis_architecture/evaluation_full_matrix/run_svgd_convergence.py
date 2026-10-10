r"""
run_svgd_convergence.py
========================
SVGD convergence benchmark (Philipp's request, 2026-10-01): how does the
ergodic error evolve over up to 1000 SVGD iterations, depending on where the
solver was initialised?

Design
------
* Shapes: the 12 holdout shapes of the `exploration/` benches (first 12 of the
  `val` split, `--shapes` overrides).
* Knowledge states: ground_truth, half_known, ten_samples, none_known
  (`variant_runner.KNOWLEDGE_CONDITIONS`, same belief construction as every
  other bench).
* Strategies (target density + CFM setting), the better of Optuna / grid per
  acquisition family, judged by the objective J (lower = better) of
  `exploration_optimierung/results`:

      lse -> niveau_svgd25      grid  J=0.2581 (tau=0.6067)  vs. Optuna best 0.2696
      ucb -> ucb_tuned_svgd0    grid  J=0.2615 (kappa=0.4597) vs. Optuna best 0.2743
      eid -> eid_optuna_ideal_v2 Optuna J=0.2513             vs. grid best  0.2708

  (the Optuna `ucb`/`niveau` bests come from 2-5 trials only, the grid ones
  from the full sweep.)
* Four initialisation families with `--n_init` (30) members each:
    cfm          one batched forward pass of the CFM planner, n_init samples
                 (single shot, no replanning: the planner output IS the
                 warm start; the strategy's tuned kappa/tau, quantile
                 particles, cfg_weight and GP settings are used, its
                 mission-level debt/visit terms and internal SVGD budget are
                 not -- iteration 0 is the raw network sample)
    selfsup      one batched forward pass of the self-supervised single-pass
                 generator (`flow_matching_particles_selfsupervised.py`,
                 wrapped by `selfsup_planner.SelfsupPlanner`), same particle
                 conditioning as cfm but no CFG, no ODE integration and no
                 start conditioning. Known caveat: the only trained
                 checkpoint has a measured `diversity` of ~0.08 (near
                 "ignores the noise input"), so its n_init candidates are
                 markedly more similar to each other than cfm's.
    random_walk  `init_baselines.random_walk_path`, seeds 0..n_init-1
    linear       `init_baselines.linear_angle_path`, n_init angles evenly
                 spaced over [0, 180) deg, each a full chord through the
                 workspace centre
  Random walk / linear inits do not depend on shape, knowledge or strategy
  (paired design); cfm and selfsup both do (they condition on the belief's
  target density), so both run once per (shape, knowledge, strategy).
* SVGD target (`--svgd_target`):
    truth   (default) every init is refined against the TRUE density. The
            knowledge state then only shapes what the CFM planner is
            conditioned on, i.e. it measures how good the warm start is under
            partial knowledge while the solver converges to the same optimum
            for all methods. Random walk / linear do not depend on knowledge
            or strategy in this mode, so they are computed ONCE per shape and
            stored with knowledge_condition='all', strategy='all' (the plot
            script shows them in every panel; `iter_runs(expand_shared=True)`
            yields them for every condition).
    belief  every init is refined against the strategy's `zieldichte` of the
            knowledge-state belief (what a deployed solver would see);
            random walk / linear are then run per (knowledge, strategy). Note
            that for none_known this target is flat, so all methods converge
            to the same blind-coverage error.
* Every init is refined with `SvgdRefiner` on nxi=25 B-spline control points
  (the solver's own representation), `--n_iters` (1000) iterations, and the
  control points of EVERY intermediate state plus the ergodic error against
  the TRUE density are written to a SQLite DB (`svgd_convergence_db.py`).

Runs on CPU workers (SVGD is numpy), the CFM forward pass on the main process
(GPU if available). Resumable: finished (shape, knowledge, strategy, method,
init_idx) rows are skipped on re-run with the same `--out_tag`.

Example
-------
    # Self-test without GPU / checkpoint (seconds)
    python run_svgd_convergence.py --dry_run --shapes A --conditions none_known \
        --n_init 2 --n_iters 20 --workers 2 --out_tag smoke

    # Full run (NOT started automatically -- ask before running)
    python run_svgd_convergence.py --out_tag svgd_convergence_YYYYMMDD
"""

import argparse
import collections
import multiprocessing as mp
import os
import sys
import time
import traceback
import zlib

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

import svgd_convergence_db as sdb                                  # noqa: E402

#: strategy key (plot/DB label) -> `variant_runner.STRATEGIES` entry.
STRATEGY_MAP = {
    'lse': 'niveau_svgd25',
    'ucb': 'ucb_tuned_svgd0',
    'eid': 'eid_optuna_ideal_v2',
}
METHODS = ('cfm', 'random_walk', 'linear', 'selfsup')
#: Methods that need a network forward pass and condition on the belief's
#: target density -- unlike random_walk/linear, they are not shared across
#: knowledge states/strategies (see the SHARED-baseline handling below).
NETWORK_METHODS = ('cfm', 'selfsup')
SHARED = 'all'      # knowledge/strategy label of baseline rows that do not depend on either
NXI = 25
N_POINTS = 128
DEGREE = 5
SEED = 0
TRUTH_RES = 96
DEFAULT_N_SHAPES = 12


def task_seed(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7fffffff


# ── Worker side (CPU, one SVGD run per task) ────────────────────────────────

_W = {}


def _worker_init():
    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from metrics_explore_exploit import ExploreExploitErgodic
    _W['ee'] = ExploreExploitErgodic(device='cpu')
    _W['B'] = {}


def basis_matrix(nxi=NXI, n_points=N_POINTS, degree=DEGREE):
    from obstacles import bspline_basis_matrix
    return bspline_basis_matrix(nxi, n_points, degree).astype(np.float32)


def score_states(ee, curves, phi_k):
    """Batched ergodic metrics of dense curves (N,T,2) against phi_k (M,).
    Same expression as `ExploreExploitErgodic.score`, vectorised over N
    (checked against it in test_svgd_convergence.py). -> dict of (N,) arrays."""
    from ergodic_energy_torch import coeffs_from_points
    c = torch.as_tensor(curves, dtype=torch.float32, device=ee.device)
    pk = torch.as_tensor(phi_k, dtype=torch.float32, device=ee.device)
    c_k = coeffs_from_points(c, ee.k_idx)                          # (N, M)
    weighted = ee.w * 0.5 * ee.Lambda * (c_k - pk.unsqueeze(0)).pow(2)
    e_explore = weighted[:, ee.low].sum(dim=1)
    e_exploit = weighted[:, ee.high].sum(dim=1)
    path_len = (c[:, 1:] - c[:, :-1]).norm(dim=-1).sum(dim=1)
    return {'E_total': (e_explore + e_exploit).numpy(),
            'E_explore': e_explore.numpy(), 'E_exploit': e_exploit.numpy(),
            'path_len': path_len.numpy()}


def run_task(task):
    """One SVGD run with full per-iteration logging -> DB row dict."""
    from common.svgd_refine import SvgdRefiner
    ee = _W['ee']
    nxi, n_points = task['nxi'], task['init_curve'].shape[0]
    key = (nxi, n_points)
    if key not in _W['B']:
        _W['B'][key] = basis_matrix(nxi, n_points)
    B = _W['B'][key]

    # 'refiner' fehlt nur in Aufgaben aus Laeufen vor dem Flag -> bisheriger TSVEC-Refiner.
    refiner = SvgdRefiner(seed=task['seed'], backend=task.get('refiner', 'tsvec'))
    log = []
    refiner.refine(task['init_curve'].astype(np.float64),
                   task['phi'].astype(np.float64), task['n_iters'],
                   nxi=nxi, trajectory_log=log, start=task.get('start'))
    cps = np.stack(log).astype(np.float32)                        # (n_iters+1, nxi, 2)
    if cps.shape[0] != task['n_iters'] + 1:
        raise RuntimeError(f"expected {task['n_iters'] + 1} states, got {cps.shape[0]}")
    curves = np.einsum('pi,sid->spd', B, cps)
    m = score_states(ee, curves, task['phi_k_truth'])
    raw = score_states(ee, task['init_curve'][None], task['phi_k_truth'])
    return dict(shape=task['shape'], knowledge_condition=task['cond'],
                strategy=task['strategy'], method=task['method'],
                init_idx=task['init_idx'], init_param=task['init_param'],
                n_iters=task['n_iters'], nxi=nxi, init_curve=task['init_curve'],
                E_raw_init=float(raw['E_total'][0]), cps=cps, **m)


def run_task_safe(task):
    try:
        return run_task(task)
    except Exception:                                              # noqa: BLE001
        return {'error': traceback.format_exc(),
                'key': (task['shape'], task['cond'], task['strategy'],
                        task['method'], task['init_idx'])}


# ── Main-process side (belief, planner, init generation) ────────────────────

class DummyPlanner:
    """Stand-in for the CFM network (`--dry_run`): random smooth control
    points. Only for self-tests without GPU / checkpoint."""
    nxi = NXI
    cfg_weight = 1.0

    def __init__(self):
        self.B = torch.as_tensor(basis_matrix())

    def plan(self, _cond, n_candidates=1, **_kw):
        g = torch.Generator().manual_seed(1234 + int(torch.randint(0, 10**6, (1,))))
        steps = torch.randn(n_candidates, self.nxi, 2, generator=g) * 0.12
        cps = (0.5 + steps.cumsum(dim=1) * 0.5).clamp(0.05, 0.95)
        return cps

    def render(self, cps):
        return torch.einsum('pi,kid->kpd', self.B, cps.float())


def build_planner_curves(planner, representation, belief, strategy_name, n_init,
                         device, seed, start=None):
    """n_init samples from `planner` (CfmPlanner or SelfsupPlanner) for the
    belief's target density, as dense curves. Generic over any object with
    the `.plan`/`.render` interface -- used for both 'cfm' and 'selfsup'.
    `start`: (2,) tensor or None, forwarded to `planner.plan` (only takes
    effect with a start-conditioned checkpoint; one shared start for the
    whole batch, same as every other start-conditioned caller in this repo;
    `SelfsupPlanner.plan` ignores it, see `selfsup_planner.py`).
    -> (curves (n,T,2) float32 np, phi (R,R) float32 np)."""
    import apply_cfm_belief as acb
    import variant_runner as vr
    args, _svgd_iters, cfg_weight = vr.build_strategy_args(strategy_name, device)
    planner.cfg_weight = cfg_weight
    mu, sd = belief.posterior_grid()
    phi = acb.zieldichte(mu, sd, args.kappa, args)
    torch.manual_seed(seed)
    if representation == 'particles':
        parts = acb.phi_particles(phi, args.n_particles, mode=args.phi_mode,
                                  quantile=args.phi_quantile, device=belief.device)
        cps = planner.plan(parts, n_candidates=n_init, start=start)
    else:
        cps = planner.plan(phi, n_candidates=n_init, start=start)
    curves = planner.render(cps).detach().cpu().numpy().astype(np.float32)
    return curves, phi.detach().cpu().numpy().astype(np.float32)


def target_density(belief, strategy_name, device):
    import apply_cfm_belief as acb
    import variant_runner as vr
    args, _s, _c = vr.build_strategy_args(strategy_name, device)
    mu, sd = belief.posterior_grid()
    return acb.zieldichte(mu, sd, args.kappa, args).detach().cpu().numpy().astype(np.float32)


def baseline_inits(method, n_init, start=None, linear_length=None):
    """-> list of (curve (T,2) float32 np, init_param).

    `start` (x, y) tuple or None: if set, random_walk begins there
    (`random_walk_path(start=...)`) and linear becomes a ray leaving `start`
    (`linear_ray_path`, headings spread over the full circle, since a ray --
    unlike the centred chord -- is not symmetric under a 180 deg flip);
    otherwise unchanged (random_walk centred, linear a chord through the
    workspace centre).
    """
    from init_baselines import random_walk_path, linear_angle_path, linear_ray_path
    out = []
    for i in range(n_init):
        if method == 'random_walk':
            start_t = None if start is None else torch.tensor(start, dtype=torch.float32)
            out.append((random_walk_path(N_POINTS, seed=i, start=start_t)
                        .numpy().astype(np.float32), float(i)))
        elif start is not None:
            ang = 360.0 * i / n_init
            out.append((linear_ray_path(start, ang, linear_length, N_POINTS)
                        .numpy().astype(np.float32), ang))
        else:
            ang = 180.0 * i / n_init
            out.append((linear_angle_path(ang, N_POINTS).numpy().astype(np.float32), ang))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True,
                    help='results/<out_tag>/svgd_convergence.db is written '
                         '(with --refiner sun: results/<out_tag>_sun/).')
    ap.add_argument('--shapes', type=str, default=None,
                    help='Comma-separated shape names (default: first 12 val shapes).')
    ap.add_argument('--n_shapes', type=int, default=DEFAULT_N_SHAPES)
    ap.add_argument('--conditions', type=str, default=None,
                    help='Comma-separated knowledge states (default: all four).')
    ap.add_argument('--strategies', type=str, default=','.join(STRATEGY_MAP))
    ap.add_argument('--methods', type=str, default=','.join(METHODS))
    ap.add_argument('--n_init', type=int, default=30,
                    help='Initialisations per (shape, knowledge, strategy, method).')
    ap.add_argument('--n_iters', type=int, default=1000)
    ap.add_argument('--svgd_target', type=str, default='truth', choices=['truth', 'belief'],
                    help='Density the solver converges to (see module docstring).')
    ap.add_argument('--representation', type=str, default='particles',
                    choices=['particles', 'spectral'])
    ap.add_argument('--ckpt', type=str, default=None,
                    help='CFM checkpoint (default: transfer/netz2d_startpunkt.pt for '
                         'particles, the spectral checkpoint otherwise).')
    ap.add_argument('--selfsup_ckpt', type=str, default=None,
                    help='Self-supervised checkpoint (default: selfsup_planner.DEFAULT_SELFSUP_CKPT).')
    ap.add_argument('--start_pos', type=str, default=None,
                    help='"x,y": fix every init (cfm/random_walk/linear) and the SVGD '
                         'refinement itself to this start point (default: unset, the '
                         'original unconstrained behaviour -- cfm untouched, random_walk '
                         'centred, linear a chord through the workspace centre). Uses the '
                         'same start-conditioning the CFM checkpoint already supports '
                         '(`CfmPlanner.plan(start=...)`), `random_walk_path(start=...)` and '
                         '`SvgdRefiner.refine(start=...)` (pins the first control point).')
    ap.add_argument('--linear_length_units', type=float, default=2.0,
                    help='Only with --start_pos: length of the linear baseline ray, in '
                         'workspace diagonals (`init_baselines.linear_ray_path`); unused '
                         'otherwise (the centred chord is a fixed length).')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop submitting new blocks after this many hours; '
                         'finished runs are kept, re-run to resume.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Self-test: random control points instead of the CFM '
                         'network (no checkpoint, no GPU).')
    from common.svgd_refine import add_refiner_arg, run_suffix
    add_refiner_arg(ap)
    args = ap.parse_args()

    import variant_runner as vr
    from common.data import load_truth
    from metrics_explore_exploit import ExploreExploitErgodic
    from run_eval_matrix import DEFAULT_CKPT
    from run_ideal_matrix import PLANNER_BUILDERS, SPECTRAL_CKPT

    strategies = [s for s in args.strategies.split(',') if s]
    for s in strategies:
        if s not in STRATEGY_MAP:
            raise KeyError(f"unknown strategy {s!r}; known: {sorted(STRATEGY_MAP)}")
    methods = [m for m in args.methods.split(',') if m]
    for m in methods:
        if m not in METHODS:
            raise KeyError(f"unknown method {m!r}; known: {METHODS}")
    conditions = ([c for c in args.conditions.split(',') if c]
                  if args.conditions else list(vr.KNOWLEDGE_CONDITIONS))
    for c in conditions:
        if c not in vr.KNOWLEDGE_CONDITIONS:
            raise KeyError(f"unknown knowledge condition {c!r}")
    start_pos = None
    if args.start_pos:
        start_pos = tuple(float(v) for v in args.start_pos.split(','))
        if len(start_pos) != 2:
            raise ValueError('--start_pos needs "x,y"')
    linear_length = args.linear_length_units * np.sqrt(2.0)

    device = args.device
    labels = [s.strip() for s in args.shapes.split(',')] if args.shapes else None
    names, truths = load_truth(labels=labels, n=args.n_shapes, split='val',
                               resolution=TRUTH_RES, device=device)
    print(f"[svgd_conv] {len(names)} shapes: {names}")

    out_dir = os.path.join(_here, 'results', args.out_tag + run_suffix(args.refiner))
    db_path = os.path.join(out_dir, 'svgd_convergence.db')
    conn = sdb.open_db(db_path)
    sdb.save_basis(conn, basis_matrix(), NXI, N_POINTS, DEGREE)
    sdb.set_meta(conn, 'config', dict(
        strategy_map=STRATEGY_MAP, shapes=names, conditions=conditions,
        methods=methods, n_init=args.n_init, n_iters=args.n_iters,
        representation=args.representation, svgd_target=args.svgd_target, nxi=NXI, n_points=N_POINTS,
        degree=DEGREE, seed=SEED, truth_res=TRUTH_RES, dry_run=args.dry_run,
        refiner=args.refiner, start_pos=start_pos,
        linear_length=linear_length if start_pos else None,
        strategies={k: vr.STRATEGIES[v] for k, v in STRATEGY_MAP.items()},
        started=time.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()

    planners = {}
    if 'cfm' in methods:
        if args.dry_run:
            planners['cfm'] = DummyPlanner()
        else:
            ckpt = args.ckpt or (DEFAULT_CKPT if args.representation == 'particles'
                                 else SPECTRAL_CKPT)
            planners['cfm'] = PLANNER_BUILDERS[args.representation](ckpt, device)
            assert planners['cfm'].nxi == NXI, f"planner nxi={planners['cfm'].nxi}, expected {NXI}"
    if 'selfsup' in methods:
        if args.dry_run:
            planners['selfsup'] = DummyPlanner()
        else:
            from selfsup_planner import SelfsupPlanner, DEFAULT_SELFSUP_CKPT
            planners['selfsup'] = SelfsupPlanner(args.selfsup_ckpt or DEFAULT_SELFSUP_CKPT, device)
            assert planners['selfsup'].nxi == NXI, \
                f"selfsup planner nxi={planners['selfsup'].nxi}, expected {NXI}"

    ee_main = ExploreExploitErgodic(device=device)
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(args.workers, initializer=_worker_init)
    pending = collections.deque()
    max_pending = args.workers * 4
    t0 = time.time()
    n_done = n_fail = n_skipped = 0
    n_blocks = len(names) * len(conditions) * len(strategies)
    block_i = 0

    def drain(block=False):
        nonlocal n_done, n_fail
        while pending and (block or pending[0].ready()):
            res = pending.popleft().get()
            if 'error' in res:
                n_fail += 1
                print(f"[svgd_conv] FAILED {res['key']}:\n{res['error']}", flush=True)
                continue
            sdb.save_run(conn, res)
            n_done += 1
            if n_done % 25 == 0:
                conn.commit()

    def submit_block(shape, cond, strat, need, inits, svgd_phi):
        for m, idxs in need.items():
            for i in idxs:
                curve, param = inits[m][i]
                pending.append(pool.apply_async(run_task_safe, ({
                    'shape': shape, 'cond': cond, 'strategy': strat,
                    'method': m, 'init_idx': i, 'init_param': param,
                    'init_curve': curve, 'phi': svgd_phi,
                    'phi_k_truth': phi_k_truth,
                    'n_iters': args.n_iters, 'nxi': NXI, 'refiner': args.refiner,
                    'start': start_pos, 'seed': task_seed(shape, cond, strat, m, i)},)))

    try:
        for shape, truth in zip(names, truths):
            truth_np = truth.detach().cpu().numpy().astype(np.float32)
            sdb.save_truth(conn, shape, truth_np)
            phi_k_truth = ee_main.target_coeffs(truth).detach().cpu().numpy()
            if args.svgd_target == 'truth':
                shared = [m for m in methods if m not in NETWORK_METHODS]
                have = sdb.existing_keys(conn, shape, SHARED, SHARED)
                need = {m: [i for i in range(args.n_init) if (m, i) not in have]
                        for m in shared}
                inits = {m: baseline_inits(m, args.n_init, start=start_pos,
                                           linear_length=linear_length)
                         for m in shared if need[m]}
                if any(need.values()):
                    submit_block(shape, SHARED, SHARED, need, inits, truth_np)
            for cond in conditions:
                for strat in strategies:
                    block_i += 1
                    if (args.time_budget_h is not None
                            and (time.time() - t0) / 3600.0 > args.time_budget_h):
                        print("[svgd_conv] time budget reached, stopping submission.")
                        raise StopIteration
                    strat_name = STRATEGY_MAP[strat]
                    belief_mode = args.svgd_target == 'belief'
                    block_methods = methods if belief_mode else \
                        [m for m in methods if m in NETWORK_METHODS]
                    have = sdb.existing_keys(conn, shape, cond, strat)
                    need = {m: [i for i in range(args.n_init) if (m, i) not in have]
                            for m in block_methods}
                    if not any(need.values()):
                        n_skipped += 1
                        continue
                    s = vr.STRATEGIES[strat_name]
                    belief = vr.build_belief(
                        cond, truth, seed=SEED, device=device,
                        gp_noise=s.get('gp_noise', 0.05),
                        gp_lengthscale=s.get('gp_lengthscale', 0.08))
                    start_t = (None if start_pos is None
                              else torch.tensor(start_pos, dtype=torch.float32))
                    inits, phi = {}, None
                    for m in NETWORK_METHODS:
                        if m in block_methods and need[m]:
                            curves, phi = build_planner_curves(
                                planners[m], args.representation, belief, strat_name,
                                args.n_init, device, task_seed(shape, cond, strat, m),
                                start=start_t)
                            inits[m] = [(c, None) for c in curves]
                    if phi is None:
                        phi = target_density(belief, strat_name, device)
                    for m in ('random_walk', 'linear'):
                        if m in block_methods and need[m]:
                            inits[m] = baseline_inits(m, args.n_init, start=start_pos,
                                                      linear_length=linear_length)
                    sdb.save_target(conn, shape, cond, strat, phi)
                    svgd_phi = phi if belief_mode else truth_np
                    submit_block(shape, cond, strat, need, inits, svgd_phi)
                    while len(pending) > max_pending:
                        drain(block=False)
                        if len(pending) > max_pending:
                            time.sleep(0.2)
                    el = time.time() - t0
                    print(f"[svgd_conv] block {block_i}/{n_blocks} ({shape}/{cond}/{strat}) "
                          f"submitted; {n_done} runs stored, {len(pending)} pending, "
                          f"{n_fail} failed, {el / 60:.1f} min", flush=True)
                    conn.commit()
    except StopIteration:
        pass
    except KeyboardInterrupt:
        print("[svgd_conv] interrupted -- storing finished runs, dropping pending ones.")
        pool.terminate()
        pending.clear()
    finally:
        if pending:
            drain(block=True)
        conn.commit()
        pool.close()
        pool.join()
    print(f"[svgd_conv] done: {n_done} runs stored, {n_fail} failed, "
          f"{n_skipped} blocks already complete, {(time.time() - t0) / 60:.1f} min "
          f"-> {db_path}")
    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    mp.freeze_support()
    main()
