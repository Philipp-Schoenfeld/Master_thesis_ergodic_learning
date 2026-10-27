r"""
run_mission_eval.py
====================
Replanning-mission benchmark (Philipp's request, 2026-10-02): how many length
units does an agent need until its driven path has covered 99 % of the ground
truth, depending on how the candidate trajectories are initialised and on how
much of the target it knows at the start?

One MISSION = one (shape, knowledge state, target-density strategy, init method)
and runs as follows (one "length unit" = the workspace diagonal, `mission.LENGTH_UNIT`):

  1. target density Phi from the current belief (`apply_cfm_belief.debt_density`
     with the strategy's tuned hyper-parameters: lse / ucb / eid exactly as in
     `run_svgd_convergence.STRATEGY_MAP`, including coverage debt),
  2. `n_init` (30) candidate trajectories, all beginning at the current agent
     position:
         cfm          one batched forward pass of the start-conditioned CFM
                      planner (the start point is a network input, the first
                      control point is set exactly),
         selfsup      one batched forward pass of the self-supervised
                      single-pass generator (`selfsup_planner.SelfsupPlanner`);
                      it has no start conditioning, so its raw output does NOT
                      begin exactly at the agent position -- the start-point
                      force in step 3 (applied to every candidate of every
                      method regardless) pulls it there during refinement
                      instead, same as it would for any other unconditioned
                      init,
         random_walk  `init_baselines.random_walk_path(start=position)`,
         linear       `init_baselines.linear_ray_path`: straight rays from the
                      position, headings 360*i/n_init degrees,
  3. refinement (`--refiner`): 'sun' (default) = Sun et al.'s FM-Stein solver,
     the same core as the data generator (`svgd_batched.BatchedSunTorch`, target
     = the density grid Phi, the start point is the initial state); 'tsvec' =
     the previous SVGD (`svgd_batched.BatchedSvgdTorch`, a vectorised copy of `SvgdRefiner`,
     checked against it in `test_mission_eval.py`) refines EVERY candidate for
     `n_iters` (1500) iterations against Phi, with the start-point force
     (`SvgdRefiner.W_START`) pulling the first control point to the agent
     position; every SVGD state of every candidate is stored,
  4. the best candidate (lowest ergodic error E against Phi after SVGD; the
     planner never sees the ground truth -- `--select_by truth` is an oracle
     switch) is driven for exactly ONE length unit; the sensor measures along
     that unit, the GP belief is updated,
  5. back to 1., until the driven path has swept >= 99 % of the ground-truth
     mass (`metrics_explore_exploit.swept_mass_fraction`, coverage radius
     `--coverage_radius`, default = the sensor radius 0.06) or `--max_rounds`.

After every executed unit the ergodic errors of the driven path against the
ground truth (E_truth) and against the updated target density (E_target;
`J_* = E + 0.02 * path length`), the swept mass, the information gain and the
belief error are stored (`mission_db.py`) -- the plots come from
`plot_mission_eval.py`.

Execution model: all holdout shapes of a (knowledge, strategy, method) set
advance in lockstep, so one planning round is ONE batch of shapes x 30
candidates -- the CFM forward pass (sliced, ~0.23 s per candidate, the
bottleneck) and the SVGD of all candidates run on the GPU; the CPU worker pool
generates the baseline initialisations and compresses the SVGD logs while the
GPU works on the next set. Resumable per round: a finished (shape, round) is
never recomputed; the belief is replayed from the stored measurements.

Example
-------
    # self-test of the whole chain
    python test_mission_eval.py

    # full run (hours): all 25 holdout shapes x 3 knowledge states x 3 strategies x 3 methods
    python run_mission_eval.py --out_tag mission_eval_YYYYMMDD --out_root G:/mission_eval
"""

import os

# One BLAS thread per process, set BEFORE numpy is imported (spawned workers
# inherit it): the machine has 6 physical cores, and without this every worker
# opens ~70 threads and the pool crawls (measured: >10x slower).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import ast
import collections
import glob
import hashlib
import json
import multiprocessing as mp
import sqlite3
import sys
import time
import traceback
import types

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

import mission_db as mdb                                           # noqa: E402
import run_svgd_convergence as rsc                                 # noqa: E402
from state_codec import pack_states                                # noqa: E402

STRATEGY_MAP = rsc.STRATEGY_MAP
#: 'ebm' (Energy-Policy, network_candidates path) and 'heuristic' (belief-
#: density-grid GUI heuristic, baseline_inits path) added 2026-10-20 for the
#: active-exploration comparison -- extended LOCALLY (not in `rsc`/run_svgd_
#: convergence.py, which other callers still use with its original 4
#: methods unchanged). 'dfl' deliberately excluded: its conditioning needs
#: the ground-truth trajectory's own spectral coefficients
#: (`trajectory_to_spectral`), which don't exist under partial belief.
METHODS = rsc.METHODS + ('ebm', 'heuristic')
NETWORK_METHODS = rsc.NETWORK_METHODS + ('ebm',)
#: The --methods default; 'cfm_guided' (the start-conditioned CFM with GP-belief
#: guidance at inference time, `belief_guidance.py`) is opt-in only. It plans
#: with the same network and the same seeds as 'cfm', so --guide_weight 0
#: reproduces 'cfm' exactly.
DEFAULT_METHODS = METHODS
METHODS = METHODS + ('cfm_guided',)
NETWORK_METHODS = NETWORK_METHODS + ('cfm_guided',)
GUIDE_MODULES = (os.path.join(_here, 'belief_guidance.py'),
                 os.path.join(_arch, 'flow_matching_cond_particles_start.py'))
DEFAULT_CONDITIONS = ('half_known', 'ten_samples', 'none_known')
NXI, N_POINTS, DEGREE, SEED, TRUTH_RES = rsc.NXI, rsc.N_POINTS, rsc.DEGREE, rsc.SEED, rsc.TRUTH_RES
task_seed = rsc.task_seed


# ── Result fingerprint: reuse stored missions, never mix configurations ──────

#: Bump when the mission logic in THIS file changes its results (this file is
#: not part of the code digest, so harmless edits here keep resumes working).
RESULT_VERSION = 1

#: Modules whose code determines the stored results. Their digest ignores
#: comments, docstrings and formatting, so only real code changes count.
RESULT_MODULES = (
    os.path.join(_here, 'svgd_batched.py'), os.path.join(_here, 'run_svgd_convergence.py'),
    os.path.join(_here, 'variant_runner.py'), os.path.join(_here, 'init_baselines.py'),
    os.path.join(_here, 'selfsup_planner.py'), os.path.join(_here, 'ebm_planner.py'),
    os.path.join(_here, 'metrics_explore_exploit.py'), os.path.join(_here, 'run_ideal_matrix.py'),
    os.path.join(_arch, 'exploration', 'apply_cfm_belief.py'),
    *(os.path.join(_arch, 'exploration', 'common', f) for f in (
        'tsvec_svgd.py', 'belief.py', 'ak_belief.py', 'np_belief.py', 'observation.py',
        'metrics.py', 'acquisition.py', 'svgd_refine.py')),
    os.path.join(_arch, 'attentive_kernel_belief.py'), os.path.join(_arch, 'neural_process_belief.py'),
    os.path.join(_arch, 'ergodic_energy_torch.py'),
    os.path.join(_arch, 'exploration_optimierung', 'mission.py'))

#: CLI arguments that change what a mission computes (not where it is stored or
#: how fast it runs). `max_rounds` only caps a mission -- a run extended to more
#: rounds is a valid resume -- and acts on results solely through the kappa
#: schedule, which enters the fingerprint as the effective per-round list.
RESULT_ARGS = ('refiner', 'n_init', 'n_iters', 'coverage_threshold',
               'coverage_radius', 'select_by', 'start_pos', 'linear_length_units',
               'svgd_precision', 'state_stride', 'metric_stride', 'representation',
               'cfm_chunk', 'full_execution', 'dry_run')


def code_digest(paths=RESULT_MODULES):
    h = hashlib.sha256()
    for p in paths:
        if not os.path.exists(p):
            h.update(f'{os.path.basename(p)}:missing'.encode())
            continue
        tree = ast.parse(open(p, encoding='utf-8').read())
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        h.update(os.path.basename(p).encode())
        h.update(ast.dump(tree).encode())
    return h.hexdigest()


def file_id(path):
    """Portable identity of a checkpoint: name and size, not the absolute path."""
    return None if not path else [os.path.basename(path), os.path.getsize(path)]


def result_fingerprint(ctx, cond, strat, method, belief, scfg, kappas):
    """-> (sha256 hex, the dict it is computed from) of everything that
    determines the results of one mission set."""
    a = ctx.args
    d = dict(version=RESULT_VERSION, code=ctx.code_digest, shapes=list(ctx.names),
             cond=cond, strategy=strat, method=method, belief=belief, strategy_args=scfg,
             args={k: getattr(a, k, None) for k in RESULT_ARGS}, kappa_schedule=kappas,
             planner_ckpt=ctx.ckpt_ids.get(method))
    if belief == 'ak':
        d['ak_fit_iters'] = a.ak_fit_iters
    if belief == 'np':
        d['np_ckpt'] = file_id(a.np_ckpt)
    if method == 'cfm_guided':
        d['guide'] = dict(weight=a.guide_weight, beta=a.guide_beta, t_start=a.guide_t_start,
                          pts=a.guide_pts, grid=a.guide_grid, code=ctx.guide_digest)
    blob = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest(), d


def reuse_index(roots, exclude):
    """{fingerprint: (shard path, n stored rounds)} over all shards under
    `roots` (also belief_*/ sub-roots), skipping the current run `exclude`.
    Shards from before the fingerprint existed have none and are never reused."""
    index = {}
    excl = os.path.abspath(exclude)
    for root in roots:
        for path in (glob.glob(os.path.join(root, '*', 'shards', '*.db'))
                     + glob.glob(os.path.join(root, '*', 'belief_*', 'shards', '*.db'))):
            if os.path.abspath(path).startswith(excl + os.sep):
                continue
            try:
                conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
                fp = mdb.get_meta(conn, 'fingerprint')
                n = conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0] if fp else 0
                conn.close()
            except sqlite3.Error:
                continue
            if fp and n > index.get(fp, (None, 0))[1]:
                index[fp] = (path, n)
    return index


def copy_shard(src, dst):
    """Consistent snapshot of `src` into `dst` (SQLite backup API, safe while
    another process may still be writing `src`)."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    s = sqlite3.connect(f'file:{src}?mode=ro', uri=True)
    d = sqlite3.connect(dst)
    s.backup(d)
    d.close()
    s.close()


# ── Worker side (CPU): baseline initialisations, log compression ─────────────

def _worker_init():
    torch.set_num_threads(1)


#: Same symmetry-breaking jitter scale as `run_init_comparison.py`'s static-
#: comparison heuristic variant (matches `common.svgd_refine.SvgdRefiner
#: .JITTER`): `gui_heuristic_path` is deterministic given (phi, start), so
#: without this every one of a shape's `n_init` "candidates" would be a
#: byte-identical duplicate -- fine for the independent-swarm backends
#: (sun/tsvec), but degenerates `tsvec_svgd`'s mutually-interacting JOINT
#: swarm to a zero-diversity kernel at iteration 0 (no jitter of its own,
#: see `svgd_batched.BatchedTsvecSvgdTorch`).
HEURISTIC_JITTER = 0.01


def baseline_inits(method, start, init_params, linear_length, n_points=N_POINTS, phi=None):
    """(C, T, 2) float32 start curves of a baseline method, all beginning at
    `start`. `init_params`: random walk -> seeds, linear -> headings in
    degrees, heuristic -> jitter seeds. `phi`: (R, R) belief density grid
    (numpy), required only for `method == 'heuristic'` -- the GUI/
    interactive_sim heuristic conditioned on the CURRENT BELIEF, not the
    ground truth or the dataset-generator's analytic shape_def (neither of
    which exists mid-mission); see `init_baselines.gui_heuristic_path`'s own
    docstring on why this is the fair "same information level" choice here."""
    from init_baselines import random_walk_path, linear_ray_path, gui_heuristic_path
    out = []
    base = None
    if method == 'heuristic':
        base = gui_heuristic_path(torch.as_tensor(phi, dtype=torch.float32),
                                  start_pos=tuple(float(v) for v in start), n_points=n_points)
    for p in init_params:
        if method == 'random_walk':
            c = random_walk_path(n_points, seed=int(p), start=torch.tensor(start, dtype=torch.float32))
        elif method == 'heuristic':
            rng = np.random.default_rng(int(p))
            jitter = rng.normal(scale=HEURISTIC_JITTER, size=(n_points, 2)).astype(np.float32)
            c_np = base.numpy() + jitter
            c_np[0] = base.numpy()[0]                    # keep the start point exact
            out.append(np.clip(c_np, 0.0, 1.0).astype(np.float32))
            continue
        else:
            c = linear_ray_path(start, float(p), linear_length, n_points)
        out.append(c.numpy().astype(np.float32))
    return np.stack(out)


def _pool_task(t):
    if t['kind'] == 'init':
        return dict(shape=t['shape'], round=t['round'],
                    inits=baseline_inits(t['method'], t['start'], t['init_params'],
                                         t['linear_length'], phi=t.get('phi')))
    if t['kind'] == 'pack':
        # wide_range default True: candidates well short of the configured n_iters
        # (low --n_iters budgets, or any early-stopped run) are not yet pulled back
        # into [0, 1] by SVGD's boundary term and can exceed the narrow quantisation
        # range state_codec.py uses for converged (iteration ~1500) states -- caused
        # a real consistency-check failure (worst_seg up to 5.8e-1) on the
        # n_iters<=1000 budget-sweep runs. Negligible precision cost either way
        # (~1.7e-4 vs ~3.1e-5 per coordinate, both far below sensor/solver scale).
        wide_range = t.get('wide_range', True)
        return dict(shape=t['shape'], round=t['round'],
                    blobs=[pack_states(t['cps'][c], wide_range=wide_range)
                          for c in range(t['cps'].shape[0])])
    raise KeyError(t['kind'])


def pool_task_safe(t):
    try:
        return _pool_task(t)
    except Exception:                                              # noqa: BLE001
        return {'error': traceback.format_exc(), 'shape': t.get('shape'), 'round': t.get('round')}


# ── Main-process side ────────────────────────────────────────────────────────

class DummyPlanner:
    """Stand-in for the CFM network (`--dry_run`): random smooth control points
    that begin at `start`. Only for self-tests without GPU / checkpoint."""
    nxi = NXI
    cfg_weight = 1.0

    def __init__(self):
        self.B = torch.as_tensor(rsc.basis_matrix(NXI, N_POINTS, DEGREE))

    def plan(self, _particles, n_candidates=1, start=None, **_kw):
        steps = torch.randn(n_candidates, self.nxi, 2) * 0.22
        cps = (0.5 + steps.cumsum(dim=1) * 0.5).clamp(0.05, 0.95)
        if start is not None:
            s = start.detach().cpu().reshape(-1, 2).float()
            cps = cps - cps[:, :1] + s.expand(n_candidates, 2).unsqueeze(1)
            cps = cps.clamp(0.02, 0.98)
            cps[:, 0] = s.expand(n_candidates, 2)
        return cps

    def render(self, cps):
        return torch.einsum('pi,kid->kpd', self.B, cps.float())


def _truth_like(truth, mu):
    t = truth.to(device=mu.device, dtype=mu.dtype)
    if t.shape != mu.shape:
        t = torch.nn.functional.interpolate(t[None, None], size=mu.shape, mode='bilinear',
                                            align_corners=True)[0, 0]
    return t


def plan_guided(planner, particles, n, start, guide, a):
    """`CfmPlanner.plan` with belief guidance: the same call into the
    start-conditioned generator, plus the guide (`belief_guidance.py`)."""
    kw = {}
    if start is not None and planner.start_cond:
        kw['start'] = start.to(planner.device).reshape(-1)
    out = planner._gen(planner.model, particles.to(planner.device), num_samples=n,
                       nxi=planner.nxi, steps=planner.steps, device=str(planner.device),
                       cfg_weight=planner.cfg_weight, guide=guide, guide_weight=a.guide_weight,
                       guide_t_start=a.guide_t_start, **kw)
    cps = out[0] if isinstance(out, tuple) else out
    return cps.detach()


def belief_rmse_from_mu(mu, truth):
    return float(((mu - _truth_like(truth, mu)) ** 2).mean().sqrt())


def belief_quality(mu, sd, truth):
    """Active-sensing scores of a belief (mu, sd on the belief grid) against the
    ground truth: rmse, corr (Pearson, scale-free), nll (mean Gaussian NLL of
    the truth, i.e. whether sigma is calibrated), cov95, sd_mean."""
    mu, sd = mu.double(), sd.double().clamp(min=1e-3)
    t = _truth_like(truth, mu)
    err = mu - t
    a, b = mu.flatten() - mu.mean(), t.flatten() - t.mean()
    corr = float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))
    nll = float((torch.log(sd) + 0.5 * (err / sd) ** 2).mean() + 0.5 * np.log(2 * np.pi))
    return dict(rmse=float(err.pow(2).mean().sqrt()), corr=corr, nll=nll,
                cov95=float((err.abs() <= 1.96 * sd).double().mean()),
                sd_mean=float(sd.mean()))


def eval_indices(n_iters, stride):
    idx = list(range(0, n_iters + 1, stride))
    if idx[-1] != n_iters:
        idx.append(n_iters)
    return idx


def ergodic_E_batch(ctx, curves, phik, chunk=2500):
    """Total ergodic error (K=10, W=600, the project metric of
    `ExploreExploitErgodic`) of N dense curves (N, T, 2) against per-curve target
    coefficients phik (N, M), in chunks to bound the (N, T, M, 2) intermediate."""
    from ergodic_energy_torch import coeffs_from_points
    ee = ctx.ee
    out = []
    for i in range(0, curves.shape[0], chunk):
        c = coeffs_from_points(curves[i:i + chunk], ee.k_idx)
        out.append((ee.w * 0.5 * ee.Lambda * (c - phik[i:i + chunk]) ** 2).sum(-1))
    return torch.cat(out)


class ShapeState:
    """Everything one shape carries through its mission."""

    def __init__(self, name, truth, belief):
        self.name, self.truth, self.belief = name, truth, belief
        self.driven = None
        self.n_done = 0                 # completed rounds
        self.done = False               # 99 % reached
        self.mu = self.sd = self.phi = None
        self.unc = self.unc0 = None


class MissionSet:
    """One (knowledge state, strategy, method) over all shapes, in lockstep.

    `run()` is a generator: it yields `None` to hand control back to the
    scheduler (cooperative slices of GPU work) or a list of pool tasks and
    receives the list of their results via `send`.
    """

    def __init__(self, ctx, cond, strat, method, belief='rbf'):
        self.ctx, self.cond, self.strat, self.method = ctx, cond, strat, method
        self.belief = belief
        # Seeds of the guided CFM are those of 'cfm', so both see the same noise.
        self.seed_method = 'cfm' if method == 'cfm_guided' else method
        self.tag = f'{method}/{strat}/{cond}'
        # Belief comparison runs keep one result root per belief model (each with
        # the usual shards/ layout); without --beliefs it is ctx.out_dir as before.
        belief_roots = getattr(ctx, 'belief_roots', None)
        root = belief_roots[belief] if belief_roots else ctx.out_dir
        if belief_roots:
            self.tag += f'/{belief}'
        import variant_runner as vr
        self.vr = vr
        self.strat_name = STRATEGY_MAP[strat]
        self.args, _svgd, self.cfg_weight = vr.build_strategy_args(self.strat_name, ctx.device)
        self.scfg = vr.STRATEGIES[self.strat_name]
        # Explicit per-round kappa schedule (Philipp's request, 2026-10-21):
        # only 'ucb' has a kappa `debt_density` reads directly -- 'eid'/'lse'
        # are governed by their own tuned parameter (e.g. phi_tau), which a
        # kappa list says nothing about, so the schedule is a no-op for them
        # (same scoping `sample_annealed_mission.py` already documents).
        # Reuses `common.acquisition.kappa_schedule` verbatim, not a new
        # formula -- `--kappas` (explicit per-round list) takes priority over
        # `--kappa0`/`--kappa1` (that function's smooth interpolation) when
        # both are given, matching `sample_annealed_mission.py`'s own CLI.
        self.kappas = None
        a = ctx.args
        if self.strat == 'ucb' and (a.kappas or a.kappa0 is not None):
            if a.kappas:
                self.kappas = [float(v) for v in a.kappas.split(',')]
                if len(self.kappas) != a.max_rounds:
                    raise ValueError(f"--kappas has {len(self.kappas)} values, "
                                     f"need {a.max_rounds} (= --max_rounds)")
            else:
                from common.acquisition import kappa_schedule
                self.kappas = [kappa_schedule(r, a.max_rounds, a.kappa0, a.kappa1)
                              for r in range(a.max_rounds)]
        path = mdb.shard_path(root, cond, strat, method)
        self.fingerprint = None
        if getattr(ctx, 'code_digest', None) is not None:
            self.fingerprint, fp_detail = result_fingerprint(ctx, cond, strat, method, belief,
                                                             self.scfg, self.kappas)
            self._reuse(path)
        self.conn = mdb.open_db(path)
        if self.fingerprint is not None:
            self._check_fingerprint(fp_detail)
        self.t0 = time.time()

    def _reuse(self, path):
        """Start from an identical, already computed mission set of another run."""
        if os.path.exists(path):
            conn = sqlite3.connect(path)
            n = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='rounds'").fetchone()[0]
            n = conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0] if n else 0
            conn.close()
            if n:
                return
        src = getattr(self.ctx, 'reuse', {}).get(self.fingerprint)
        if src:
            copy_shard(src[0], path)
            print(f"[mission] {self.tag}: reusing {src[1]} stored rounds of an identical "
                  f"mission set from {src[0]}", flush=True)

    def _check_fingerprint(self, detail):
        n = self.conn.execute('SELECT COUNT(*) FROM rounds').fetchone()[0]
        stored = mdb.get_meta(self.conn, 'fingerprint')
        if n and stored is None:
            print(f"[mission] {self.tag}: WARNING resuming a shard without fingerprint "
                  f"(created before fingerprints existed); configuration not verified",
                  flush=True)
        elif n and stored != self.fingerprint:
            old = mdb.get_meta(self.conn, 'fingerprint_detail', {}) or {}
            diff = sorted(k for k in set(old) | set(detail) if old.get(k) != detail.get(k))
            raise RuntimeError(
                f"{self.tag}: the stored rounds in this shard were computed with a different "
                f"configuration (differs in: {diff}); refusing to mix them. Use a new --out_tag.")
        mdb.set_meta(self.conn, 'fingerprint', self.fingerprint)
        mdb.set_meta(self.conn, 'fingerprint_detail', detail)
        self.conn.commit()
        self.plan_s = self.svgd_s = 0.0

    # -- state construction / resume -----------------------------------------
    def build_states(self):
        ctx = self.ctx
        vr = self.vr
        mdb.save_basis(self.conn, rsc.basis_matrix(NXI, N_POINTS, DEGREE), NXI, N_POINTS, DEGREE)
        mdb.set_meta(self.conn, 'config', dict(ctx.config, cond=self.cond, strategy=self.strat,
                                              method=self.method, belief=self.belief,
                                              strategy_args=self.scfg))
        a = ctx.args
        states = []
        for name, truth in zip(ctx.names, ctx.truths):
            mdb.save_truth(self.conn, name, truth.detach().cpu().numpy())
            t0 = time.time()
            b = vr.build_belief(self.cond, truth, seed=SEED, device=ctx.device,
                                gp_noise=self.scfg.get('gp_noise', 0.05),
                                gp_lengthscale=self.scfg.get('gp_lengthscale', 0.08),
                                dtype=torch.float64, kernel=self.belief,
                                ak_fit_iters=getattr(a, 'ak_fit_iters', 200),
                                np_ckpt=getattr(a, 'np_ckpt', None))
            st = ShapeState(name, truth, b)
            mu0, sd0 = b.posterior_grid()
            st.unc0 = float(sd0.float().sum())   # same path as `refresh` -> consistent gains
            b._cache = None
            mdb.save_belief_metrics(self.conn, dict(
                shape=name, round=-1, belief=self.belief, update_s=time.time() - t0,
                n_obs=int(b.n_obs), **belief_quality(mu0.float(), sd0.float(), truth)))
            states.append(st)
        self.conn.commit()
        replayed = 0
        for st in states:
            for row in mdb.load_replay_rows(self.conn, st.name):
                st.belief.observe(torch.as_tensor(row['obs_pts'], device=ctx.device),
                                  torch.as_tensor(row['obs_vals'], device=ctx.device))
                seg = torch.as_tensor(row['segment'], device=ctx.device)
                st.driven = seg if st.driven is None else torch.cat([st.driven, seg], dim=0)
                st.n_done = row['round'] + 1
                st.done = row['reached99']
                replayed += 1
        if replayed:
            print(f"[mission] {self.tag}: resumed, replayed {replayed} stored rounds", flush=True)
        return states

    def refresh(self, st):
        """Belief fields, visitation and the target density of the CURRENT state."""
        ctx, args = self.ctx, self.args
        if self.kappas is not None:
            # `st.n_done` is the round ABOUT TO BE planned (0-indexed, same
            # convention as `run()`'s `for r in range(a.max_rounds)`) --
            # `self.args` is this MissionSet's OWN args instance (`build_
            # strategy_args` builds a fresh one per set), so mutating it here
            # cannot cross-talk with other concurrently-scheduled sets the
            # way mutating the shared `ctx.args` would.
            args.kappa = self.kappas[min(st.n_done, len(self.kappas) - 1)]
        mu, sd = st.belief.posterior_grid()
        st.belief._cache = None                      # free the n x n Cholesky factor
        mu, sd = mu.float(), sd.float()
        visit = self.vr._visit_field(st.driven, args, ctx.device)
        phi, _ = ctx.acb.debt_density(mu, sd, visit, args.kappa, args)
        phi = phi.float()
        if float(phi.max()) < 1e-6:                  # nothing left to attract -> uniform
            phi = torch.ones_like(phi)
            ctx.n_uniform_fallback += 1
        st.mu, st.sd, st.phi = mu, sd, phi
        st.unc = float(sd.sum())

    # -- candidates ----------------------------------------------------------
    def network_candidates(self, method, active, r, starts):
        """Slice-wise batched planning with `ctx.planners[method]` (cfm or
        selfsup) -> self._net_curves (A, n_init, T, 2) float32."""
        ctx, args = self.ctx, self.args
        planner = ctx.planners[method]
        n_init = ctx.args.n_init
        clouds = []
        for st in active:
            torch.manual_seed(task_seed(st.name, self.cond, self.strat, 'parts', r))
            parts = ctx.acb.phi_particles(st.phi, args.n_particles, mode=args.phi_mode,
                                          quantile=args.phi_quantile, device=ctx.device)
            clouds.append(parts.unsqueeze(0).expand(n_init, -1, -1))
        all_parts = torch.cat(clouds, dim=0)
        all_starts = torch.as_tensor(np.repeat(starts, n_init, axis=0), dtype=torch.float32,
                                     device=ctx.device)
        guides = None
        if method == 'cfm_guided' and not ctx.args.dry_run:
            from belief_guidance import SpectralInfoGuide, ShapeGuides
            a = ctx.args
            guides = [SpectralInfoGuide(st.belief, st.phi, ctx.ee, ctx.B32, beta=a.guide_beta,
                                        n_meas=a.guide_pts, grid=a.guide_grid) for st in active]
        total, chunk, out = all_parts.shape[0], ctx.args.cfm_chunk, []
        for c0 in range(0, total, chunk):
            t0 = time.time()
            planner.cfg_weight = self.cfg_weight
            torch.manual_seed(task_seed(self.cond, self.strat, self.seed_method, 'plan', r, c0))
            with torch.no_grad():
                n = min(chunk, total - c0)
                if guides is None:
                    cps = planner.plan(all_parts[c0:c0 + chunk], n_candidates=n,
                                       start=all_starts[c0:c0 + chunk])
                else:
                    cps = plan_guided(planner, all_parts[c0:c0 + chunk], n,
                                      all_starts[c0:c0 + chunk],
                                      ShapeGuides(guides, n_init, first=c0), ctx.args)
                out.append(planner.render(cps).detach().cpu())
            self.plan_s += time.time() - t0
            yield None
        curves = torch.cat(out, dim=0).numpy().astype(np.float32)
        self._net_curves = curves.reshape(len(active), n_init, *curves.shape[1:])

    def run_svgd(self, active, inits, starts, n_iters=None):
        """SVGD of all candidates of all active shapes in ONE GPU batch, then the
        ergodic error against each shape's planning target along the iterations.
        -> dict of host arrays (+ the stored-state log on the host).

        `n_iters`: override for `ctx.args.n_iters` (default: unchanged), so a
        caller can run a shorter (or longer) batch than the run's nominal
        budget, e.g. to cap baselines to a CFM-derived iteration count
        (`run_mission_eval_budget_matched.py`)."""
        ctx = self.ctx
        a = ctx.args
        n_iters = a.n_iters if n_iters is None else n_iters
        A, n_init = len(active), a.n_init
        r = active[0].n_done
        flat = inits.reshape(A * n_init, *inits.shape[2:]).astype(np.float64)
        if a.refiner == 'sun':
            # Sun: the density grid itself is the target (score of log Phi).
            target = np.repeat(np.stack([st.phi.detach().cpu().numpy().astype(np.float64)
                                         for st in active]), n_init, axis=0)
        elif a.refiner == 'tsvec_svgd':
            # BatchedTsvecSvgdTorch needs Fourier coefficients at THIS
            # project's K_DEFAULT=10 (100 modes) -- NOT `ctx.svgd_ref._phi_k`
            # (K=8, 64 modes, built for the 'tsvec'/Adam-SVGD backend only).
            # `ctx.ee.target_coeffs` is exactly the K=10 computation already
            # used for `phi_k10` below (the metric target), reused here.
            target = np.repeat(np.stack([ctx.ee.target_coeffs(st.phi).detach().cpu().numpy().astype(np.float64)
                                         for st in active]), n_init, axis=0)
        else:
            target = np.repeat(np.stack([ctx.svgd_ref._phi_k(st.phi.detach().cpu().numpy().astype(np.float64))
                                         for st in active]), n_init, axis=0)
        phi_k10 = torch.repeat_interleave(torch.stack([ctx.ee.target_coeffs(st.phi) for st in active]),
                                          n_init, dim=0)
        starts_rep = np.repeat(starts, n_init, axis=0)
        seeds = [task_seed(st.name, self.cond, self.strat, self.seed_method, r, i)
                 for st in active for i in range(n_init)]
        t0 = time.time()
        dep_eta = getattr(a, 'depletion_eta', 0.0)          # budget-matched runner has no such flag
        dep = ({} if dep_eta <= 0 else
               dict(depletion_eta=dep_eta, depletion_sigma=getattr(a, 'depletion_sigma', 0.06)))
        out = ctx.bs.run(flat, target, starts_rep, seeds, n_iters, record=True, **dep)
        if ctx.device != 'cpu':
            torch.cuda.synchronize()
        self.svgd_s += time.time() - t0
        cps = out['cps']                                              # (A*n, n_iters+1, nxi, 2) float32, device
        midx = eval_indices(n_iters, a.metric_stride)
        with torch.no_grad():
            curves = torch.einsum('pi,csid->cspd', ctx.B32, cps[:, midx])
            E = ergodic_E_batch(ctx, curves.reshape(-1, curves.shape[2], 2),
                                phi_k10.repeat_interleave(len(midx), dim=0)).reshape(A * n_init, len(midx))
            E_init = ergodic_E_batch(ctx, torch.as_tensor(flat, dtype=torch.float32, device=ctx.device),
                                     phi_k10)
            del curves
        sidx = eval_indices(n_iters, a.state_stride)
        cps_host = cps[:, sidx].cpu().numpy()
        del cps, out['cps']
        return dict(final_cps=out['final_cps'], E_series=E.cpu().numpy().astype(np.float32),
                    E_init=E_init.cpu().numpy(), midx=midx, sidx=sidx, cps_host=cps_host,
                    seeds=seeds)

    # -- one executed unit ---------------------------------------------------
    def execute(self, st, res, r, start, n_iters_used=None, full_execution=False):
        """Pick the winner, drive one unit, update the belief, score. -> (row, cand rows).

        `n_iters_used`: the SVGD iteration count actually behind `res` (default:
        `ctx.args.n_iters`), recorded in the stored candidate rows -- lets a
        caller execute on a truncated/capped run (budget-matched evaluation)
        while keeping the DB honest about how many iterations were spent.

        `full_execution`: drive the WHOLE refined candidate (whatever length it
        naturally is, ~2-7 length units depending on method) instead of trimming
        to exactly one length unit before replanning. Candidates are no longer
        required to be at least one length unit long to be eligible."""
        ctx, args = self.ctx, self.args
        n_iters_used = ctx.args.n_iters if n_iters_used is None else n_iters_used
        from common.metrics import coverage_vs_truth, path_length, trim_to_length
        from common.observation import measure, thin
        from exploration_optimierung.mission import (LENGTH_UNIT, PTS_PER_UNIT,
                                                     resample_arclength)
        from metrics_explore_exploit import swept_mass_fraction, LAMBDA_LEN_J
        dev = ctx.device
        n_init = ctx.args.n_init
        si = ctx.names.index(st.name)
        E_final = np.asarray(res['E_final'], dtype=np.float64)
        if ctx.args.select_by == 'truth':
            dense = np.einsum('pi,cid->cpd', ctx.B64, res['final_cps'])
            sel_score = rsc.score_states(ctx.ee, dense.astype(np.float32),
                                         ctx.phi_k_truth[si])['E_total']
        else:
            sel_score = E_final
        # A plan shorter than one length unit cannot be driven for a full unit: such
        # candidates are only eligible if no candidate is long enough (counted).
        # (full_execution: any positive-length plan is eligible -- it gets driven whole.)
        dense_all = np.einsum('pi,cid->cpd', ctx.B64, res['final_cps'])
        plan_len = np.linalg.norm(np.diff(dense_all, axis=1), axis=-1).sum(axis=-1)
        eligible = plan_len >= (1e-6 if full_execution else LENGTH_UNIT)
        if eligible.any():
            sel = int(np.argmin(np.where(eligible, sel_score, np.inf)))
        else:
            sel = int(np.argmax(plan_len))
            ctx.n_short_plans += 1

        # -- the unit (or, full_execution: the whole refined candidate) that is driven
        curve = torch.as_tensor(ctx.B64 @ res['final_cps'][sel], dtype=torch.float32, device=dev)
        curve[0] = torch.as_tensor(start, dtype=torch.float32, device=dev)      # hard snap, as SvgdRefiner.refine
        curve = curve.clamp(0.0, 1.0)
        seg = curve if full_execution else trim_to_length(curve, LENGTH_UNIT)
        n_pts = max(8, int(round(PTS_PER_UNIT * path_length(seg) / LENGTH_UNIT)))
        seg = resample_arclength(seg, n_pts)
        torch.manual_seed(task_seed(st.name, self.cond, self.strat, self.seed_method, 'meas', r))
        pts, vals = measure(seg, st.truth, noise_std=args.noise, sensor_radius=args.sensor_radius)
        pts_t, vals_t = thin(pts, vals, max_points=args.max_obs)

        plan_target, mu_plan, sd_plan, unc_before = st.phi, st.mu, st.sd, st.unc
        t_upd = time.time()
        st.belief.observe(pts_t, vals_t)
        update_s = time.time() - t_upd
        st.driven = seg if st.driven is None else torch.cat([st.driven, seg], dim=0)
        self.refresh(st)                                        # belief / target AFTER the update
        belief_row = dict(shape=st.name, round=r, belief=self.belief, update_s=update_s,
                          n_obs=int(st.belief.n_obs), **belief_quality(st.mu, st.sd, st.truth))

        # -- scores of the whole driven path
        sc_truth = ctx.ee.score(st.driven, ctx.phi_k_truth_t[si])
        e_target = ctx.ee.score(st.driven, ctx.ee.target_coeffs(st.phi))['E_ergodic_total']
        e_target_plan = ctx.ee.score(st.driven, ctx.ee.target_coeffs(plan_target))['E_ergodic_total']
        swept = swept_mass_fraction(st.driven, st.truth, sensor_radius=ctx.args.coverage_radius)
        cov = float(coverage_vs_truth(st.driven, st.truth))
        plen = path_length(st.driven)
        reached = swept >= ctx.args.coverage_threshold
        st.n_done = r + 1
        st.done = bool(reached)
        row = dict(
            shape=st.name, knowledge_condition=self.cond, strategy=self.strat,
            method=self.method, round=r, n_exec=r + 1, start_x=float(start[0]),
            start_y=float(start[1]), selected_idx=sel, start_gap=float(res['start_gap'][sel]),
            swept_mass=swept, reached99=int(reached),
            E_truth=sc_truth['E_ergodic_total'], E_truth_explore=sc_truth['E_ergodic_explore'],
            E_truth_exploit=sc_truth['E_ergodic_exploit'], E_target=e_target,
            E_target_plan=e_target_plan,
            J_truth=sc_truth['E_ergodic_total'] + LAMBDA_LEN_J * plen,
            J_target=e_target + LAMBDA_LEN_J * plen, cov=cov,
            cov_norm=cov / max(ctx.cov_blind[si], 1e-12), path_len=plen,
            seg_len=path_length(seg), info_gain=unc_before - st.unc,
            info_gain_cum=st.unc0 - st.unc, unc_before=unc_before, unc_after=st.unc,
            belief_rmse=belief_rmse_from_mu(st.mu, st.truth), n_obs=int(st.belief.n_obs),
            E_cand_best=float(E_final.min()), E_cand_median=float(np.median(E_final)),
            E_cand_worst=float(E_final.max()),
            target=plan_target.detach().cpu().numpy(), mu_plan=mu_plan.detach().cpu().numpy(),
            sd_plan=sd_plan.detach().cpu().numpy(), segment=seg.detach().cpu().numpy(),
            obs_pts=pts_t.detach().cpu().numpy(), obs_vals=vals_t.detach().cpu().numpy(),
            belief_metrics=belief_row)
        cands = [dict(cand_idx=c, selected=(c == sel), init_param=res['init_params'][c],
                      n_iters=n_iters_used, nxi=NXI, n_states=res['n_states'],
                      state_stride=ctx.args.state_stride, E_stride=ctx.args.metric_stride,
                      E_init=float(res['E_init'][c]), E_final=float(E_final[c]),
                      init_curve=res['inits'][c], states=res['blobs'][c],
                      E_series=res['E_series'][c]) for c in range(n_init)]
        return row, cands

    # -- the generator -------------------------------------------------------
    def run(self):
        ctx = self.ctx
        a = ctx.args
        states = self.build_states()
        for st in states:
            if not st.done:
                self.refresh(st)
            yield None
        for r in range(a.max_rounds):
            if all(st.done for st in states):
                break
            active = [st for st in states if not st.done and st.n_done == r]
            if not active:
                continue                     # nothing to plan in this round (already past it on resume)
            if ctx.should_stop():
                print(f"[mission] {self.tag}: time budget reached before round {r}", flush=True)
                return
            A = len(active)
            starts = np.stack([(st.driven[-1].detach().cpu().numpy().astype(np.float64)
                                if st.driven is not None else np.array(ctx.start_pos))
                               for st in active])
            # 1. candidates -------------------------------------------------
            if self.method in NETWORK_METHODS:
                yield from self.network_candidates(self.method, active, r, starts)
                inits = self._net_curves
                init_params = [[float('nan')] * a.n_init for _ in active]
            else:
                if self.method in ('random_walk', 'heuristic'):
                    init_params = [[float(task_seed(st.name, 'rw', r, i)) for i in range(a.n_init)]
                                   for st in active]
                else:
                    init_params = [[360.0 * i / a.n_init for i in range(a.n_init)] for _ in active]
                res_init = yield [dict(kind='init', shape=st.name, round=r, method=self.method,
                                       start=starts[j], init_params=init_params[j],
                                       linear_length=a.linear_length_units * ctx.length_unit,
                                       phi=(st.phi.detach().cpu().numpy()
                                            if self.method == 'heuristic' else None))
                                  for j, st in enumerate(active)]
                inits = np.stack([x['inits'] for x in res_init])
            # 2. SVGD of the whole batch on the GPU --------------------------
            t_svgd = time.time()
            sv = self.run_svgd(active, inits, starts)
            yield None
            # 3. compression of the logs in the worker pool ---------------------
            packed = yield [dict(kind='pack', shape=st.name, round=r,
                                 cps=sv['cps_host'][j * a.n_init:(j + 1) * a.n_init])
                            for j, st in enumerate(active)]
            del sv['cps_host']
            # 4. select, drive one unit, update, score, store ------------------
            for j, st in enumerate(active):
                sl = slice(j * a.n_init, (j + 1) * a.n_init)
                final = sv['final_cps'][sl]
                res = dict(final_cps=final, start_gap=np.linalg.norm(final[:, 0] - starts[j][None], axis=1),
                           E_final=sv['E_series'][sl][:, -1], E_series=sv['E_series'][sl],
                           E_init=sv['E_init'][sl], inits=inits[j], init_params=init_params[j],
                           blobs=packed[j]['blobs'], n_states=len(sv['sidx']))
                row, cands = self.execute(st, res, r, starts[j], full_execution=a.full_execution)
                mdb.save_belief_metrics(self.conn, row['belief_metrics'])
                mdb.save_round(self.conn, row, cands)
                yield None
            n_done99 = sum(1 for s in states if s.done)
            print(f"[mission] {self.tag} round {r + 1}: {A} shapes planned, "
                  f"{n_done99}/{len(states)} at >= {a.coverage_threshold:.0%}, "
                  f"SVGD batch {time.time() - t_svgd:.0f} s, "
                  f"{(time.time() - self.t0) / 60:.1f} min", flush=True)
        n99 = sum(1 for s in states if s.done)
        print(f"[mission] {self.tag} FINISHED: {n99}/{len(states)} shapes reached "
              f"{a.coverage_threshold:.0%}, {(time.time() - self.t0) / 60:.1f} min "
              f"(planning {self.plan_s / 60:.1f} min, SVGD {self.svgd_s / 60:.1f} min)", flush=True)
        self.conn.commit()


# ── Scheduler ────────────────────────────────────────────────────────────────

def schedule(sets, pool, parallel_sets, ctx):
    """Cooperative round-robin over the set generators. A generator that yields
    `None` has done a slice of work; one that yields a task list waits for the
    pool; `parallel_sets` of them are open at once so the GPU work of one set
    overlaps the CPU tasks (baseline inits, log compression) of the others."""
    waiting = collections.deque(sets)
    open_ents, rr = [], 0
    while waiting or open_ents:
        while len(open_ents) < parallel_sets and waiting and not ctx.should_stop():
            ms = waiting.popleft()
            open_ents.append(dict(ms=ms, gen=ms.run(), futs=None))
        if not open_ents:
            break
        progressed = False
        n = len(open_ents)
        for k in range(n):
            ent = open_ents[(rr + k) % n]
            send = None
            if ent['futs'] is not None:
                if not all(f.ready() for f in ent['futs']):
                    continue
                send = [f.get() for f in ent['futs']]
                for res in send:
                    if 'error' in res:
                        raise RuntimeError(f"pool task failed ({ent['ms'].tag}, "
                                           f"{res['shape']}, round {res['round']}):\n{res['error']}")
                ent['futs'] = None
            try:
                y = ent['gen'].send(send)
            except StopIteration:
                open_ents.remove(ent)
                progressed = True
                break
            if y is not None:
                ent['futs'] = [pool.apply_async(pool_task_safe, (t,)) for t in y]
            rr = (rr + k + 1) % max(len(open_ents), 1)
            progressed = True
            break
        if not progressed:
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True,
                    help='results go to <out_root>/<out_tag>/shards/*.db')
    ap.add_argument('--out_root', type=str, default=os.path.join(_here, 'results'),
                    help='Parent directory of the result folder (a full run needs ~10-15 GB; '
                         'check the free space of the drive).')
    ap.add_argument('--shapes', type=str, default=None,
                    help='Comma-separated shape names (default: all validation shapes).')
    from common.svgd_refine import add_refiner_arg
    add_refiner_arg(ap)
    ap.add_argument('--n_shapes', type=int, default=999,
                    help='Take at most this many validation shapes (default: all 25).')
    ap.add_argument('--conditions', type=str, default=','.join(DEFAULT_CONDITIONS))
    ap.add_argument('--strategies', type=str, default=','.join(STRATEGY_MAP))
    ap.add_argument('--methods', type=str, default=','.join(DEFAULT_METHODS),
                    help=f'Comma-separated init methods; known: {",".join(METHODS)}.')
    ap.add_argument('--n_init', type=int, default=30, help='Candidates per plan.')
    ap.add_argument('--n_iters', type=int, default=1500,
                    help='Refinement iterations per candidate (SVGD steps for tsvec, FM-Stein '
                         'iterations for sun).')
    ap.add_argument('--depletion_eta', type=float, default=0.0,
                    help='TRACE footprint-overlap depletion (Zheng et al., arXiv:2608.02304, Eq. 3): '
                         'per-visit discount eta in [0,1) of the target density along the candidate '
                         'path itself, phi_k = phi * prod_{j<k}(1 - eta*v_jk). 0 = off (default). '
                         "Only for --refiner sun.")
    ap.add_argument('--depletion_sigma', type=float, default=0.06,
                    help='Footprint tolerance sigma_fp of the overlap kernel (default: sensor radius).')
    ap.add_argument('--max_rounds', type=int, default=40,
                    help='Safety cap on the executed length units per mission.')
    ap.add_argument('--coverage_threshold', type=float, default=0.99,
                    help='Mission ends when this fraction of the ground-truth mass is swept.')
    ap.add_argument('--coverage_radius', type=float, default=0.06,
                    help='Agent radius for "swept" (default: the sensor radius of the '
                         'measurement model, also used by steps_to_full_coverage).')
    ap.add_argument('--select_by', type=str, default='target', choices=['target', 'truth'],
                    help="Which error picks the candidate to drive: 'target' = ergodic error "
                         "against the planning target (no access to the ground truth), "
                         "'truth' = oracle.")
    ap.add_argument('--start_pos', type=str, default='0.5,0.5',
                    help='Agent position before the first unit (x,y).')
    ap.add_argument('--linear_length_units', type=float, default=2.0,
                    help='Length of the linear ray inits in length units (the CFM plans are '
                         '~2 units long).')
    ap.add_argument('--svgd_precision', type=str, default='mixed', choices=['mixed', 'fp64'],
                    help="'mixed': optimiser state float64, energy/gradient float32 (6x faster on "
                         "consumer GPUs; after 1500 iterations control points differ from fp64 by "
                         "< 1e-4 and E by < 2e-5 relative, see test_mission_eval.py); 'fp64': the "
                         "reference precision.")
    ap.add_argument('--state_stride', type=int, default=1,
                    help='Store every k-th SVGD state (1 = every iteration).')
    ap.add_argument('--metric_stride', type=int, default=10,
                    help='Evaluate E against the plan target every k-th iteration.')
    ap.add_argument('--representation', type=str, default='particles', choices=['particles'])
    ap.add_argument('--ckpt', type=str, default=None,
                    help='Start-conditioned CFM checkpoint (default: transfer/netz2d_startpunkt.pt).')
    ap.add_argument('--selfsup_ckpt', type=str, default=None,
                    help='Self-supervised checkpoint (default: selfsup_planner.DEFAULT_SELFSUP_CKPT).')
    ap.add_argument('--ebm_ckpt', type=str, default=None,
                    help='Ranking-NCE EBM checkpoint (default: ebm_planner.DEFAULT_EBM_CKPT).')
    ap.add_argument('--workers', type=int, default=6,
                    help='CPU workers for baseline inits and log compression.')
    ap.add_argument('--parallel_sets', type=int, default=4,
                    help='Mission sets kept open at once.')
    ap.add_argument('--cfm_chunk', type=int, default=120,
                    help='Candidates per CFM forward call (~0.23 s each on an RTX 2070 Super).')
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop opening new rounds after this many hours; re-run with the same '
                         '--out_tag to resume.')
    ap.add_argument('--dry_run', action='store_true',
                    help='Self-test: random control points instead of the CFM network.')
    ap.add_argument('--full_execution', action='store_true',
                    help='Drive each refined candidate in full (whatever length it naturally is) '
                         'instead of trimming to exactly one length unit before replanning. '
                         'Default off: unchanged one-length-unit-per-round behaviour.')
    ap.add_argument('--kappas', type=str, default=None,
                    help="Comma-separated, explicit per-round kappa values for the 'ucb' "
                         'strategy (must have exactly --max_rounds entries), e.g. '
                         '"2,1,0.5,0.1,0.05" for a 5-round exploration-to-exploitation '
                         "schedule. No-op for 'eid'/'lse' (governed by their own tuned "
                         'parameter, not kappa). Takes priority over --kappa0/--kappa1.')
    ap.add_argument('--kappa0', type=float, default=None,
                    help="Starting kappa (round 0) for 'ucb' via common.acquisition"
                         '.kappa_schedule, used only when --kappas is not given.')
    ap.add_argument('--kappa1', type=float, default=0.05,
                    help="Final-round kappa for 'ucb', used only when --kappa0 is given "
                         'and --kappas is not.')
    ap.add_argument('--beliefs', type=str, default=None,
                    help="Comma-separated belief models to compare: 'rbf' (GP, stationary "
                         "kernel), 'ak' (GP, Attentive Kernel), 'np' (pretrained neural "
                         "process). Each gets its own result root <out_dir>/belief_<name>/ "
                         "with the usual shards/ layout. Default: no belief dimension, RBF "
                         "GP, results directly in <out_dir> (previous behaviour). 'ak'/'np' "
                         "only for the conditions none_known and ten_samples.")
    ap.add_argument('--ak_fit_iters', type=int, default=200,
                    help='Marginal-likelihood Adam steps per AK belief update.')
    ap.add_argument('--np_ckpt', type=str, default=None,
                    help='Neural-process checkpoint (default: newest '
                         'checkpoints/belief_np_date_*_final.pt).')
    ap.add_argument('--guide_weight', type=float, default=0.5,
                    help="cfm_guided only: guidance step at t=1 relative to the velocity RMS "
                         "(quadratic ramp from --guide_t_start). 0 reproduces 'cfm'.")
    ap.add_argument('--guide_beta', type=float, default=1.0,
                    help='cfm_guided only: weight of the GP information term (expected '
                         'reduction of the target-coefficient variance by measuring along the '
                         'candidate); 0 = pure ergodic guidance towards the planning target.')
    ap.add_argument('--guide_t_start', type=float, default=0.3,
                    help='cfm_guided only: flow time from which the guidance acts.')
    ap.add_argument('--guide_pts', type=int, default=32,
                    help='cfm_guided only: curve points treated as future measurement locations.')
    ap.add_argument('--guide_grid', type=int, default=32,
                    help='cfm_guided only: integration grid of the coefficient covariance.')
    ap.add_argument('--reuse_from', type=str, default=None,
                    help='Comma-separated result roots searched for mission sets with an '
                         'identical fingerprint (same configuration, shapes, checkpoints and '
                         'solver/belief code); those are copied instead of recomputed. '
                         "Default: --out_root. 'none' disables the search.")
    a = ap.parse_args()

    import variant_runner as vr
    from common.data import load_truth
    from common.svgd_refine import SvgdRefiner, run_suffix
    from metrics_explore_exploit import ExploreExploitErgodic
    from exploration_optimierung.mission import LENGTH_UNIT, blind_coverage
    from svgd_batched import BatchedSvgdTorch, BatchedSunTorch, BatchedTsvecSvgdTorch
    import apply_cfm_belief as acb

    strategies = [s for s in a.strategies.split(',') if s]
    for s in strategies:
        if s not in STRATEGY_MAP:
            raise KeyError(f"unknown strategy {s!r}; known: {sorted(STRATEGY_MAP)}")
    methods = [m for m in a.methods.split(',') if m]
    for m in methods:
        if m not in METHODS:
            raise KeyError(f"unknown method {m!r}; known: {METHODS}")
    conditions = [c for c in a.conditions.split(',') if c]
    for c in conditions:
        if c not in vr.KNOWLEDGE_CONDITIONS:
            raise KeyError(f"unknown knowledge condition {c!r}")
    start_pos = tuple(float(v) for v in a.start_pos.split(','))
    if len(start_pos) != 2:
        raise ValueError('--start_pos needs "x,y"')
    beliefs = [b for b in a.beliefs.split(',') if b] if a.beliefs else ['rbf']
    for b in beliefs:
        if b not in ('rbf', 'ak', 'np'):
            raise KeyError(f"unknown belief {b!r}; known: rbf, ak, np")
        if b != 'rbf':
            bad = [c for c in conditions if c not in ('none_known', 'ten_samples')]
            if bad:
                raise ValueError(f"belief {b!r} does not support conditions {bad}")
    if 'np' in beliefs:
        from common.np_belief import latest_np_ckpt
        a.np_ckpt = a.np_ckpt or latest_np_ckpt()
    if 'cfm_guided' in methods and 'np' in beliefs and a.guide_beta > 0:
        raise ValueError("cfm_guided with --guide_beta > 0 needs a GP belief (rbf/ak): the "
                         "neural process has no closed-form posterior covariance")

    device = a.device
    labels = [s.strip() for s in a.shapes.split(',')] if a.shapes else None
    names, truths = load_truth(labels=labels, n=a.n_shapes, split='val',
                               resolution=TRUTH_RES, device=device)
    print(f"[mission] {len(names)} shapes: {names}", flush=True)

    out_dir = os.path.join(a.out_root, a.out_tag + run_suffix(a.refiner))
    os.makedirs(out_dir, exist_ok=True)

    planners, ckpt_ids = {}, {}
    if 'cfm' in methods or 'cfm_guided' in methods:
        if a.dry_run:
            planners['cfm'] = DummyPlanner()
        else:
            from run_eval_matrix import DEFAULT_CKPT
            from run_ideal_matrix import PLANNER_BUILDERS
            ckpt_ids['cfm'] = file_id(a.ckpt or DEFAULT_CKPT)
            planners['cfm'] = PLANNER_BUILDERS[a.representation](a.ckpt or DEFAULT_CKPT, device)
            assert planners['cfm'].nxi == NXI, f"planner nxi={planners['cfm'].nxi}, expected {NXI}"
            assert planners['cfm'].start_cond, "the mission needs a start-conditioned checkpoint"
        planners['cfm_guided'] = planners['cfm']
        ckpt_ids['cfm_guided'] = ckpt_ids.get('cfm')
    if 'selfsup' in methods:
        if a.dry_run:
            planners['selfsup'] = DummyPlanner()
        else:
            from selfsup_planner import SelfsupPlanner, DEFAULT_SELFSUP_CKPT
            ckpt_ids['selfsup'] = file_id(a.selfsup_ckpt or DEFAULT_SELFSUP_CKPT)
            planners['selfsup'] = SelfsupPlanner(a.selfsup_ckpt or DEFAULT_SELFSUP_CKPT, device)
            assert planners['selfsup'].nxi == NXI, \
                f"selfsup planner nxi={planners['selfsup'].nxi}, expected {NXI}"
    if 'ebm' in methods:
        if a.dry_run:
            planners['ebm'] = DummyPlanner()
        else:
            from ebm_planner import EbmPlanner, DEFAULT_EBM_CKPT
            ckpt_ids['ebm'] = file_id(a.ebm_ckpt or DEFAULT_EBM_CKPT)
            planners['ebm'] = EbmPlanner(a.ebm_ckpt or DEFAULT_EBM_CKPT, device)
            assert planners['ebm'].nxi == NXI, f"ebm planner nxi={planners['ebm'].nxi}, expected {NXI}"

    ee = ExploreExploitErgodic(device=device)
    truths_t = list(truths)
    B_np = rsc.basis_matrix(NXI, N_POINTS, DEGREE)
    if a.refiner == 'sun':
        bs = BatchedSunTorch(B_np, device)
    elif a.refiner == 'tsvec_svgd':
        bs = BatchedTsvecSvgdTorch(B_np, device)
    else:
        bs = BatchedSvgdTorch(B_np, device, torch.float64,
                              compute_dtype=torch.float32 if a.svgd_precision == 'mixed' else torch.float64)
    ctx = types.SimpleNamespace(
        args=a, device=device, names=list(names), truths=truths_t, planners=planners, ee=ee,
        acb=acb, svgd_ref=SvgdRefiner(0, backend=a.refiner), start_pos=start_pos, length_unit=LENGTH_UNIT, bs=bs,
        B64=B_np.astype(np.float64), B32=torch.as_tensor(B_np, dtype=torch.float32, device=device),
        cov_blind=[blind_coverage(t) for t in truths_t], n_uniform_fallback=0, n_short_plans=0,
        phi_k_truth_t=[ee.target_coeffs(t) for t in truths_t], out_dir=out_dir,
        code_digest=code_digest(), ckpt_ids=ckpt_ids, guide_digest=code_digest(GUIDE_MODULES))
    ctx.phi_k_truth = [p.detach().cpu().numpy() for p in ctx.phi_k_truth_t]
    reuse_roots = ([] if a.reuse_from == 'none' else
                   [r for r in (a.reuse_from or a.out_root).split(',') if r])
    ctx.reuse = reuse_index(reuse_roots, out_dir) if reuse_roots else {}
    print(f"[mission] code digest {ctx.code_digest[:12]}; {len(ctx.reuse)} fingerprinted "
          f"mission sets found for reuse under {reuse_roots or 'nothing'}", flush=True)
    t_start = time.time()
    ctx.should_stop = lambda: (a.time_budget_h is not None
                               and (time.time() - t_start) / 3600.0 > a.time_budget_h)
    ctx.config = dict(vars(a), shapes_resolved=list(names), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                      strategy_map=STRATEGY_MAP, length_unit=LENGTH_UNIT, nxi=NXI,
                      n_points=N_POINTS, degree=DEGREE)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(ctx.config, f, indent=1, default=str)
    if a.beliefs:
        ctx.belief_roots = {b: os.path.join(out_dir, f'belief_{b}') for b in beliefs}
        for b, root in ctx.belief_roots.items():
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, 'config.json'), 'w') as f:
                json.dump(dict(ctx.config, belief=b), f, indent=1, default=str)

    sets = [MissionSet(ctx, c, s, m, b) for c in conditions for s in strategies
            for m in methods for b in beliefs]
    print(f"[mission] {len(sets)} mission sets x {len(names)} shapes; {a.n_init} candidates x "
          f"{a.n_iters} refinement iterations per round; refiner={a.refiner}, "
          f"svgd={a.svgd_precision}, workers={a.workers}, beliefs={','.join(beliefs)}",
          flush=True)
    mp_ctx = mp.get_context('spawn')
    pool = mp_ctx.Pool(a.workers, initializer=_worker_init)
    try:
        schedule(sets, pool, a.parallel_sets, ctx)
    except KeyboardInterrupt:
        print("[mission] interrupted -- finished rounds are stored; re-run to resume.", flush=True)
        pool.terminate()
        sys.exit(130)
    finally:
        pool.close()
        pool.join()
        for ms in sets:
            ms.conn.commit()
    print(f"[mission] done in {(time.time() - t_start) / 3600:.2f} h -> {out_dir} "
          f"(uniform-target fallbacks: {ctx.n_uniform_fallback}, rounds without a full-unit plan: "
          f"{ctx.n_short_plans})", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
