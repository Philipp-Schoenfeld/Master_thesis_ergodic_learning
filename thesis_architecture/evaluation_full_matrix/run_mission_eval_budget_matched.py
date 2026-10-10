r"""
run_mission_eval_budget_matched.py
===================================
Variant of `run_mission_eval.py`: instead of giving every initialisation
method a FIXED SVGD budget (1500 iterations for cfm, random_walk and linear
alike), the budget each replanning round is set by how many iterations the
CFM candidates themselves need to reach their own plateau, and random_walk /
linear are only allowed that many SVGD steps in that same round.

One GROUP = one (knowledge state, target-density strategy); it drives three
coupled missions (cfm, random_walk, linear) in lockstep, round by round:

  1. CFM candidates are generated and refined for up to `--cfm_run_iters`
     SVGD iterations (default: `--n_iters`, long enough that the plateau
     criterion below has room to fire with its look-ahead window).
  2. Per shape, the per-candidate "good enough" iteration is the first one
     after which the candidate's own ergodic error improves by less than
     `--conv_tol` (relative) over the next `--conv_window` iterations -- the
     same plateau rule already used for the right-axis of
     `plot_mission_continuous.py --svgd_conv`; the round's budget is the
     MEDIAN of that over the 30 CFM candidates, snapped to the nearest
     `--metric_stride` grid point.
  3. CFM executes (selects a winner, drives one unit) using the candidate
     STATES AT THAT ITERATION, not at iteration 1500 -- i.e. CFM stops once
     it judges itself converged, exactly like a deployed system would.
  4. random_walk's and linear's OWN candidates (different initial curves,
     same belief/target this round) are refined for ONLY that many
     iterations and select/execute the same way.
  5. If CFM has already finished a shape (>= coverage threshold) while
     random_walk/linear have not, the LAST budget computed for that shape is
     carried forward.

Nothing here reruns the original fixed-1500 experiment or touches its shard
DBs (`mission_eval_20261006/shards/*.db`); round 0 is recomputed from
scratch rather than copied from there, so this script is fully
self-contained and resumable through its own shard DBs, same schema as
`run_mission_eval.py` (`mission_db.py`), so `plot_mission_eval.py` and
`plot_mission_continuous.py` read the output unmodified.

Why a full new run is needed and not just post-hoc truncation of the
existing 2026-10-06 run: that run stores every SVGD iteration's control
points (`state_stride=1`) and the ergodic error every 10 iterations
(`metric_stride=10`) for all three methods, which is enough to recompute
round 0 "for free" (same seeds, same initial belief, independent of any
later execution). But from round 1 on, which candidate wins and at what
(now earlier, less-refined) state determines the driven segment, hence the
measurement, hence the belief update, hence the next round's target density
and start position for EVERY method -- so the stored round-1+ candidates
were generated against conditioning that no longer matches. Measured
plateau iterations from the 2026-10-06 run (reused here only as a sanity
expectation, not as input data): cfm ~340 median, linear ~780, random_walk
~1050 (out of 1500) -- so capping the baselines to CFM's number should make
this run cheaper than the original 1500-for-everyone run, not more
expensive.

Usage
-----
    # smoke test (no GPU / CFM checkpoint needed)
    python run_mission_eval_budget_matched.py --out_tag budget_matched_smoke \
        --dry_run --shapes A,organic_10 --conditions none_known --strategies eid \
        --max_rounds 3 --cfm_run_iters 150 --conv_window 20 --conv_tol 0.05

    # full run (needs a GPU and the CFM checkpoint, same as run_mission_eval.py)
    python run_mission_eval_budget_matched.py --out_tag mission_eval_budget_matched_YYYYMMDD \
        --out_root G:/mission_eval
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import multiprocessing as mp
import sys
import time
import types

import numpy as np

import mission_db as mdb                                           # noqa: E402
import run_mission_eval as rme                                      # noqa: E402

rsc = rme.rsc
STRATEGY_MAP = rme.STRATEGY_MAP
DEFAULT_CONDITIONS = rme.DEFAULT_CONDITIONS


def _iters_to_convergence(metric, stride, n_iters, window, tol):
    """First logged iteration after which `metric` improves by less than `tol`
    (relative) over the next `window` iterations; `n_iters` if that never
    happens. Copy of `plot_mission_continuous.iters_to_convergence` (not
    imported from there to avoid pulling in its matplotlib/torch plotting
    deps into every worker)."""
    w = max(1, int(round(window / stride)))
    m = np.asarray(metric, dtype=np.float64)
    if len(m) <= w:
        return float(n_iters)
    gain = m[:-w] - m[w:]
    ok = np.nonzero(gain < tol * np.abs(m[:-w]))[0]
    return float(ok[0] * stride) if len(ok) else float(n_iters)


def shape_plateau_iters(E_series_batch, n_init, metric_stride, n_iters, window, tol):
    """(A*n_init, n_mid) E-series -> list of A per-shape budgets: the MEDIAN
    over the shape's n_init candidates of each candidate's own plateau
    iteration (same aggregation as `plot_mission_continuous.convergence_variants`),
    snapped to the metric_stride grid and clamped to [metric_stride, n_iters]."""
    A = E_series_batch.shape[0] // n_init
    out = []
    for j in range(A):
        sl = slice(j * n_init, (j + 1) * n_init)
        vals = [_iters_to_convergence(E_series_batch[c], metric_stride, n_iters, window, tol)
                for c in range(sl.start, sl.stop)]
        u = int(round(np.median(vals) / metric_stride)) * metric_stride
        out.append(int(min(max(u, metric_stride), n_iters)))
    return out


class BudgetGroup:
    """One (knowledge_condition, strategy): couples cfm/random_walk/linear
    missions so the latter two never get more SVGD iterations in a round
    than CFM judged "good enough" for itself in that same round."""

    def __init__(self, ctx, cond, strat, conv_window, conv_tol, cfm_run_iters):
        self.ctx, self.cond, self.strat = ctx, cond, strat
        self.conv_window, self.conv_tol, self.cfm_run_iters = conv_window, conv_tol, cfm_run_iters
        self.cfm = rme.MissionSet(ctx, cond, strat, 'cfm')
        self.rw = rme.MissionSet(ctx, cond, strat, 'random_walk')
        self.lin = rme.MissionSet(ctx, cond, strat, 'linear')
        self.tag = f'budget/{strat}/{cond}'

    def _resume_budgets(self):
        """Last stored per-shape iteration count from the cfm shard, so a
        resumed run carries forward the right cap for shapes CFM has already
        finished."""
        d = {}
        for (name,) in self.cfm.conn.execute("SELECT DISTINCT shape FROM rounds"):
            row = self.cfm.conn.execute(
                "SELECT n_iters FROM candidates WHERE shape=? ORDER BY round DESC LIMIT 1",
                (name,)).fetchone()
            if row:
                d[name] = int(row[0])
        return d

    def _starts(self, active):
        ctx = self.ctx
        return np.stack([(st.driven[-1].detach().cpu().numpy().astype(np.float64)
                          if st.driven is not None else np.array(ctx.start_pos))
                         for st in active])

    def _finish_round(self, ms, active, r, starts, inits, init_params, sv, use_iters, pool):
        ctx, a = self.ctx, self.ctx.args
        n_init = a.n_init
        # wide_range: early-stopped candidates (esp. random_walk at a small cap) are not
        # yet pulled back into [0, 1] by SVGD's boundary term and can exceed the narrow
        # quantisation range state_codec.py uses for converged (iteration ~1500) states.
        pack_tasks = [dict(kind='pack', shape=st.name, round=r, wide_range=True,
                           cps=sv['cps_host'][j * n_init:(j + 1) * n_init, :use_iters[j] + 1])
                     for j, st in enumerate(active)]
        packed = (pool.map(rme.pool_task_safe, pack_tasks) if pool is not None
                  else [rme.pool_task_safe(t) for t in pack_tasks])
        for p in packed:
            if 'error' in p:
                raise RuntimeError(f"pack failed ({self.tag}, {p.get('shape')}, "
                                   f"round {p.get('round')}):\n{p['error']}")
        for j, st in enumerate(active):
            sl = slice(j * n_init, (j + 1) * n_init)
            u = use_iters[j]
            mi = u // a.metric_stride
            final = sv['cps_host'][sl, u]
            res = dict(final_cps=final,
                       start_gap=np.linalg.norm(final[:, 0] - starts[j][None], axis=1),
                       E_final=sv['E_series'][sl][:, mi], E_series=sv['E_series'][sl][:, :mi + 1],
                       E_init=sv['E_init'][sl], inits=inits[j], init_params=init_params[j],
                       blobs=packed[j]['blobs'], n_states=u + 1)
            row, cands = ms.execute(st, res, r, starts[j], n_iters_used=u)
            mdb.save_round(ms.conn, row, cands)

    def _do_cfm_round(self, active, r, pool):
        ms, ctx, a = self.cfm, self.ctx, self.ctx.args
        starts = self._starts(active)
        for _ in ms.cfm_candidates(active, r, starts):
            pass                                                      # exhaust the (single-process) generator
        inits = ms._cfm_curves
        init_params = [[float('nan')] * a.n_init for _ in active]
        sv = ms.run_svgd(active, inits, starts, n_iters=self.cfm_run_iters)
        budgets = shape_plateau_iters(sv['E_series'], a.n_init, a.metric_stride,
                                      self.cfm_run_iters, self.conv_window, self.conv_tol)
        self._finish_round(ms, active, r, starts, inits, init_params, sv, budgets, pool)
        return budgets

    def _do_capped_round(self, ms, active, r, last_budget, pool):
        ctx, a = self.ctx, self.ctx.args
        starts = self._starts(active)
        fallback = int(round(float(np.median(list(last_budget.values()))))) if last_budget else a.n_iters
        use_iters = [last_budget.get(st.name, fallback) for st in active]
        n_iters_batch = int(max(use_iters))
        if ms.method == 'random_walk':
            init_params = [[float(rsc.task_seed(st.name, 'rw', r, i)) for i in range(a.n_init)]
                           for st in active]
        else:
            init_params = [[360.0 * i / a.n_init for i in range(a.n_init)] for _ in active]
        tasks = [dict(kind='init', shape=st.name, round=r, method=ms.method, start=starts[j],
                      init_params=init_params[j], linear_length=a.linear_length_units * ctx.length_unit)
                for j, st in enumerate(active)]
        res_init = pool.map(rme.pool_task_safe, tasks) if pool is not None \
            else [rme.pool_task_safe(t) for t in tasks]
        for p in res_init:
            if 'error' in p:
                raise RuntimeError(f"init failed ({self.tag}, {p.get('shape')}, "
                                   f"round {p.get('round')}):\n{p['error']}")
        inits = np.stack([x['inits'] for x in res_init])
        sv = ms.run_svgd(active, inits, starts, n_iters=n_iters_batch)
        self._finish_round(ms, active, r, starts, inits, init_params, sv, use_iters, pool)

    def run(self, pool):
        ctx, a = self.ctx, self.ctx.args
        cfm_states = self.cfm.build_states()
        rw_states = self.rw.build_states()
        lin_states = self.lin.build_states()
        for ms, states in ((self.cfm, cfm_states), (self.rw, rw_states), (self.lin, lin_states)):
            for st in states:
                if not st.done:
                    ms.refresh(st)
        last_budget = self._resume_budgets()
        t0 = time.time()
        for r in range(a.max_rounds):
            all_states = cfm_states + rw_states + lin_states
            if all(st.done for st in all_states):
                break
            if ctx.should_stop():
                print(f"[budget] {self.tag}: time budget reached before round {r}", flush=True)
                return
            active_cfm = [st for st in cfm_states if not st.done and st.n_done == r]
            if active_cfm:
                budgets = self._do_cfm_round(active_cfm, r, pool)
                for st, u in zip(active_cfm, budgets):
                    last_budget[st.name] = u
            for ms, states in ((self.rw, rw_states), (self.lin, lin_states)):
                active = [st for st in states if not st.done and st.n_done == r]
                if active:
                    self._do_capped_round(ms, active, r, last_budget, pool)
            n_done = sum(st.done for st in all_states)
            budget_sample = sorted(set(last_budget.values()))
            print(f"[budget] {self.tag} round {r + 1}: {n_done}/{len(all_states)} shapes done, "
                  f"budgets in use: {budget_sample[:4]}{'...' if len(budget_sample) > 4 else ''}, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        for ms in (self.cfm, self.rw, self.lin):
            ms.conn.commit()
        n99 = sum(st.done for st in cfm_states + rw_states + lin_states)
        print(f"[budget] {self.tag} FINISHED: {n99}/{len(cfm_states) + len(rw_states) + len(lin_states)} "
              f"shapes reached {a.coverage_threshold:.0%}, {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--out_root', type=str, default=os.path.join(rme._here, 'results'))
    ap.add_argument('--shapes', type=str, default=None)
    from common.svgd_refine import add_refiner_arg
    add_refiner_arg(ap)
    ap.add_argument('--n_shapes', type=int, default=999)
    ap.add_argument('--conditions', type=str, default=','.join(DEFAULT_CONDITIONS))
    ap.add_argument('--strategies', type=str, default=','.join(STRATEGY_MAP))
    ap.add_argument('--n_init', type=int, default=30)
    ap.add_argument('--n_iters', type=int, default=1500,
                    help='Nominal budget label only (stored in config.json for reference); the '
                         'actual per-round budget is set by --cfm_run_iters + the plateau rule.')
    ap.add_argument('--cfm_run_iters', type=int, default=1500,
                    help="How long CFM's own candidates are refined before the plateau rule picks "
                         "the round's budget (needs enough room for --conv_window to look ahead).")
    ap.add_argument('--conv_window', type=int, default=100,
                    help='Plateau rule: look-ahead window in SVGD iterations.')
    ap.add_argument('--conv_tol', type=float, default=0.05,
                    help='Plateau rule: converged once the relative improvement over the window '
                         'drops below this.')
    ap.add_argument('--max_rounds', type=int, default=40)
    ap.add_argument('--coverage_threshold', type=float, default=0.99)
    ap.add_argument('--coverage_radius', type=float, default=0.06)
    ap.add_argument('--select_by', type=str, default='target', choices=['target'],
                    help='Oracle select_by=truth is not supported by this coupled runner.')
    ap.add_argument('--start_pos', type=str, default='0.5,0.5')
    ap.add_argument('--linear_length_units', type=float, default=2.0)
    ap.add_argument('--svgd_precision', type=str, default='mixed', choices=['mixed', 'fp64'])
    ap.add_argument('--state_stride', type=int, default=1,
                    help='Must stay 1: the budget cap reads the exact control-point state at an '
                         'arbitrary iteration, which needs every iteration logged.')
    ap.add_argument('--metric_stride', type=int, default=10)
    ap.add_argument('--representation', type=str, default='particles', choices=['particles'])
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--cfm_chunk', type=int, default=120)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--time_budget_h', type=float, default=None)
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()
    if a.state_stride != 1:
        raise ValueError('--state_stride must be 1 for this runner (see help text)')
    if a.device is None:
        import torch
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    import torch
    import variant_runner as vr
    from common.data import load_truth
    from common.svgd_refine import SvgdRefiner, run_suffix
    from metrics_explore_exploit import ExploreExploitErgodic
    from exploration_optimierung.mission import LENGTH_UNIT, blind_coverage
    from svgd_batched import BatchedSvgdTorch, BatchedSunTorch
    import apply_cfm_belief as acb

    strategies = [s for s in a.strategies.split(',') if s]
    for s in strategies:
        if s not in STRATEGY_MAP:
            raise KeyError(f"unknown strategy {s!r}; known: {sorted(STRATEGY_MAP)}")
    conditions = [c for c in a.conditions.split(',') if c]
    for c in conditions:
        if c not in vr.KNOWLEDGE_CONDITIONS:
            raise KeyError(f"unknown knowledge condition {c!r}")
    start_pos = tuple(float(v) for v in a.start_pos.split(','))
    if len(start_pos) != 2:
        raise ValueError('--start_pos needs "x,y"')

    device = a.device
    labels = [s.strip() for s in a.shapes.split(',')] if a.shapes else None
    names, truths = load_truth(labels=labels, n=a.n_shapes, split='val',
                               resolution=rme.TRUTH_RES, device=device)
    print(f"[budget] {len(names)} shapes: {names}", flush=True)

    out_dir = os.path.join(a.out_root, a.out_tag + run_suffix(a.refiner))
    os.makedirs(out_dir, exist_ok=True)

    if a.dry_run:
        planner = rme.DummyPlanner()
    else:
        from run_eval_matrix import DEFAULT_CKPT
        from run_ideal_matrix import PLANNER_BUILDERS
        planner = PLANNER_BUILDERS[a.representation](a.ckpt or DEFAULT_CKPT, device)
        assert planner.nxi == rsc.NXI, f"planner nxi={planner.nxi}, expected {rsc.NXI}"
        assert planner.start_cond, "the mission needs a start-conditioned checkpoint"

    ee = ExploreExploitErgodic(device=device)
    truths_t = list(truths)
    B_np = rsc.basis_matrix(rsc.NXI, rsc.N_POINTS, rsc.DEGREE)
    if a.refiner == 'sun':
        bs = BatchedSunTorch(B_np, device)
    else:
        bs = BatchedSvgdTorch(B_np, device, torch.float64,
                              compute_dtype=torch.float32 if a.svgd_precision == 'mixed' else torch.float64)
    ctx = types.SimpleNamespace(
        args=a, device=device, names=list(names), truths=truths_t, planner=planner, ee=ee,
        acb=acb, svgd_ref=SvgdRefiner(0, backend=a.refiner), start_pos=start_pos, length_unit=LENGTH_UNIT, bs=bs,
        B64=B_np.astype(np.float64), B32=torch.as_tensor(B_np, dtype=torch.float32, device=device),
        cov_blind=[blind_coverage(t) for t in truths_t], n_uniform_fallback=0, n_short_plans=0,
        phi_k_truth_t=[ee.target_coeffs(t) for t in truths_t], out_dir=out_dir)
    ctx.phi_k_truth = [p.detach().cpu().numpy() for p in ctx.phi_k_truth_t]
    t_start = time.time()
    ctx.should_stop = lambda: (a.time_budget_h is not None
                               and (time.time() - t_start) / 3600.0 > a.time_budget_h)
    ctx.config = dict(vars(a), shapes_resolved=list(names), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                      strategy_map=STRATEGY_MAP, length_unit=LENGTH_UNIT, nxi=rsc.NXI,
                      n_points=rsc.N_POINTS, degree=rsc.DEGREE,
                      budget_matched=True, methods='cfm,random_walk,linear')
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(ctx.config, f, indent=1, default=str)

    groups = [BudgetGroup(ctx, c, s, a.conv_window, a.conv_tol, a.cfm_run_iters)
             for c in conditions for s in strategies]
    print(f"[budget] {len(groups)} (condition, strategy) groups x {len(names)} shapes; "
          f"cfm plateau rule: window={a.conv_window}, tol={a.conv_tol}, cfm_run_iters={a.cfm_run_iters}; "
          f"refiner={a.refiner}, svgd={a.svgd_precision}, workers={a.workers}", flush=True)
    mp_ctx = mp.get_context('spawn')
    pool = mp_ctx.Pool(a.workers, initializer=rme._worker_init) if a.workers > 0 else None
    try:
        for g in groups:
            if ctx.should_stop():
                print(f"[budget] time budget reached, skipping remaining groups", flush=True)
                break
            g.run(pool)
    except KeyboardInterrupt:
        print("[budget] interrupted -- finished rounds are stored; re-run to resume.", flush=True)
        if pool is not None:
            pool.terminate()
        sys.exit(130)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        for g in groups:
            for ms in (g.cfm, g.rw, g.lin):
                ms.conn.commit()
    print(f"[budget] done in {(time.time() - t_start) / 3600:.2f} h -> {out_dir}", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
