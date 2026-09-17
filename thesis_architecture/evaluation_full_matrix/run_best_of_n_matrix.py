r"""
run_best_of_n_matrix.py
=========================
Follow-up to `run_ideal_matrix.py` / `run_heuristic_linear_matrix.py`, per
Philipp's request (2026-09-17): a new results folder with every planner
version from both of those runs, but for every variant that samples from a
*distribution* of candidates (CFM, random walk) generate `--n_candidates`
(default 30) candidates for the same test case, score them, and keep -- per
metric panel -- the best candidate's value (direction-aware, see
`run_eval_matrix.METRIC_DIRECTION`), plus a spread statistic for the overlay
marker in `plot_metric_bars(..., show_distribution=True)`.

Heuristic/linear-waypoint variants (and lawnmower) stay deterministic --
single run, no best-of-N. Decided explicitly with Philipp: only the
variants he named (CFM, random walk) get resampled; the rest keep the same
single fixed seed as everywhere else in the project.

Two CFM selection modes (`--selection`), decided 2026-09-18
-------------------------------------------------------------
`post_svgd` (the original design): `n_candidates` fully independent
rollouts, *each* refined by SVGD, score all `n_candidates` finished
trajectories and keep the best (`best_of_n_curve_and_row`). Correct but a
hard multiplier on the already-expensive CFM generation -- SVGD does not
batch across candidates, so this runs it `n_candidates` times per decision
point.

`pre_svgd` (now the default, ~2.25x faster, measured): one batched network
forward pass produces all `n_candidates` raw samples, they are scored
*before* SVGD, and only the winner is refined
(`variant_runner.REPRESENTATIONS_BEST_OF_N`). Trade-off: ranks candidates by
pre-refinement quality, which is only a good proxy for post-refinement
quality to the extent SVGD doesn't reorder them -- should hold at the small
`svgd_iters` (0/25/50) every tuned `STRATEGIES` entry uses, but wasn't
separately verified against `post_svgd`. A further consequence: `pre_svgd`
only ever produces ONE final trajectory, so there is no pool of finished
candidates to report a post-refinement mean/std over -- the overlay marker
for `pre_svgd` rows instead shows the spread of the `n_candidates` *raw*
`coverage_vs_truth` scores considered at each decision point (labelled
`coverage_mean`/`coverage_std` only, not populated for the other metrics --
see `variant_runner.py`'s module comment above
`REPRESENTATIONS_BEST_OF_N` for the full reasoning).

COST WARNING -- read before running the full matrix
-----------------------------------------------------
Even at ~2.25x faster, this is still a hard multiplier on the CFM part of
the matrix. Time a small `--pilot --n_candidates 3` (or smaller) run first
and extrapolate the real cost before starting the full matrix -- see
CLAUDE.md's cluster rules; this script does not start anything on its own,
and neither should you without asking first.

Extending a folder later (resumable by design)
-------------------------------------------------
As long as `--no_viz` was *not* passed, every row's `metrics.json` +
`trajectory.npy` is cached under `raw/`. At the end of `main()`, the summary
tables / `metric_bars` plots / holdout panels are always rebuilt from
*every* cached row under `raw_dir` (`load_cached_rows`), not just the ones
this particular invocation just generated -- so running this script again
with the same `--out_tag` but different `--representations`/`--strategies`
(e.g. first `--representations particles`, later add
`--representations spectral`) extends the same folder's tables and plots
with the union, instead of overwriting them. Re-running an *identical*
(shape, variant, knowledge_condition) combination overwrites just that one
cached row -- safe to interrupt and resume, or to redo one strategy without
discarding the rest.

Example
-------
    # Smoke test (seconds): 1 shape, 1 strategy, 3 candidates, no SVGD
    python run_best_of_n_matrix.py --pilot --n_candidates 3 \
        --strategies niveau_svgd0 --replan_schemes no_replan \
        --representations particles --skip_heuristic_linear --out_tag smoke --no_viz

    # Full matrix (NOT started automatically -- ask before running)
    python run_best_of_n_matrix.py --out_tag best_of_30_YYYYMMDD
"""

import argparse
import glob
import itertools
import json
import os
import signal
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

import torch

from common.data import load_truth                                # noqa: E402
from common.svgd_refine import SvgdRefiner                         # noqa: E402

import variant_runner as vr                                        # noqa: E402
from metrics_explore_exploit import ExploreExploitErgodic          # noqa: E402
from run_eval_matrix import (compute_row, save_trajectory_files,   # noqa: E402
                             best_of_n_metrics, summarise, variant_id,
                             plot_metric_bars, plot_tradeoff, DEFAULT_CKPT)
from run_ideal_matrix import (PLANNER_BUILDERS, SPECTRAL_CKPT)      # noqa: E402
import viz                                                          # noqa: E402

PILOT_SHAPES = ['A', 'rand_gmm_20', 'organic_20']
DEFAULT_N_CANDIDATES = 30


class TerminateInterrupt(Exception):
    """Raised by the SIGTERM handler below -- same pattern as
    `flow_matching_runner_particles.py`'s training loop
    (`--signal=SIGTERM@120` in the SLURM script warns 120s before the 24h
    hard limit). Caught around the main shape loop so a job that's about to
    be killed finalises (rebuilds tables/plots from whatever is cached) and
    exits cleanly instead of being hard-killed mid-write."""


def _sigterm_handler(signum, frame):
    raise TerminateInterrupt()


def row_done(raw_dir, method, sub, svgd_iters, knowledge_condition, shape_name):
    """True if this row's `metrics.json` is already cached on disk. Lets a
    resumed/re-submitted job (three chained `sbatch` jobs against the same
    `--out_tag`, per Philipp's request) skip every already-finished
    (shape, knowledge_condition, variant) combination and only compute what
    a previous job didn't get to -- the checkpoint/resume unit here is one
    row, not a periodic snapshot, because every row is already saved to
    disk the moment it's computed (`save_trajectory_files`, called inline in
    the loop below, not batched/buffered)."""
    vid = variant_id(method, sub, svgd_iters)
    return os.path.isfile(os.path.join(raw_dir, knowledge_condition, method, vid,
                                       shape_name, 'metrics.json'))


def best_of_n_curve_and_row(generate_fn, n_candidates, truth, phi_k_truth, ee,
                            shape_name, method, sub, svgd_iters,
                            knowledge_condition):
    """Runs `generate_fn()` (a zero-arg closure producing one full candidate
    curve) `n_candidates` times, scores each candidate with `compute_row`,
    and returns `(best_curve, row)`:

    * `best_curve` -- the candidate with the best `E_ergodic_total` (the
      project's primary scalar quality metric) -- this is the one physical
      trajectory saved to `raw/**/trajectory.npy`/`viz.png`, since only one
      curve can be stored per test case.
    * `row` -- per-metric best-of-N summary from `best_of_n_metrics`, merged
      onto that same best candidate's full metric row. The *value* shown
      for each metric panel is that metric's own best-of-N pick (which may
      come from a *different* candidate than the one saved/visualised) --
      exactly what Philipp asked for: "die entsprechende Metrik auf allen
      Kandidaten berechnen und nur ... die beste Metrik uebernehmen".
    """
    candidate_rows, candidate_curves = [], []
    for _ in range(n_candidates):
        curve = generate_fn()
        row = compute_row(curve, truth, phi_k_truth, ee, shape_name, method,
                          sub, svgd_iters, knowledge_condition)
        candidate_rows.append(row)
        candidate_curves.append(curve)
    best_idx = min(range(n_candidates),
                   key=lambda i: candidate_rows[i]['E_ergodic_total'])
    row = dict(candidate_rows[best_idx])
    row.update(best_of_n_metrics(candidate_rows))
    row['n_candidates'] = n_candidates
    row['selection'] = 'post_svgd'
    return candidate_curves[best_idx], row


def load_cached_rows(raw_dir):
    """`(row, curve_np)` for every cached `raw/**/metrics.json` under this
    out_tag folder -- see the module docstring's "Extending a folder later"
    section. `curve_np` is `None` if `trajectory.npy` is missing next to it
    (shouldn't happen for anything this script itself wrote)."""
    out = []
    for p in sorted(glob.glob(os.path.join(raw_dir, '**', 'metrics.json'), recursive=True)):
        d = os.path.dirname(p)
        with open(p) as f:
            row = json.load(f)
        traj_p = os.path.join(d, 'trajectory.npy')
        curve = np.load(traj_p) if os.path.isfile(traj_p) else None
        out.append((row, curve))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--shapes', type=str, default=None)
    ap.add_argument('--n_candidates', type=int, default=DEFAULT_N_CANDIDATES)
    ap.add_argument('--selection', type=str, default='pre_svgd',
                    choices=['pre_svgd', 'post_svgd'],
                    help='pre_svgd (default, ~2.25x faster): select the best '
                         'of n_candidates raw network samples before SVGD, '
                         'refine only that one. post_svgd (original design): '
                         'refine all n_candidates independently, keep the '
                         'best finished trajectory. See module docstring.')
    ap.add_argument('--representations', type=str, default='particles,spectral')
    ap.add_argument('--strategies', type=str, default=','.join(vr.STRATEGIES.keys()))
    ap.add_argument('--replan_schemes', type=str, default='no_replan,replan_1_6')
    ap.add_argument('--particle_ckpt', type=str, default=DEFAULT_CKPT)
    ap.add_argument('--spectral_ckpt', type=str, default=SPECTRAL_CKPT)
    ap.add_argument('--skip_heuristic_linear', action='store_true',
                    help='Heuristik-/Linear-Waypoint-Varianten (Phase 2, '
                         'deterministisch, kein Best-of-N) weglassen.')
    ap.add_argument('--heuristic_families', type=str,
                    default=','.join(vr.HEURISTIC_LINEAR_STRATEGIES.keys()))
    ap.add_argument('--heuristic_svgd_iters', type=str,
                    default=','.join(str(i) for i in vr.HEURISTIC_SVGD_ITERS))
    ap.add_argument('--linear_svgd_iters', type=str,
                    default=','.join(str(i) for i in vr.LINEAR_WAYPOINTS_SVGD_ITERS))
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--truth_res', type=int, default=96)
    ap.add_argument('--no_viz', action='store_true')
    args = ap.parse_args()

    reps = [r for r in args.representations.split(',') if r]
    for r in reps:
        if r not in PLANNER_BUILDERS:
            raise KeyError(f"unbekannte Repraesentation {r!r}")
    strategies = [s for s in args.strategies.split(',') if s]
    for s in strategies:
        if s not in vr.STRATEGIES:
            raise KeyError(f"unbekannte Strategie {s!r}")
    schemes = [s for s in args.replan_schemes.split(',') if s]
    heuristic_families = [f for f in args.heuristic_families.split(',') if f]
    heuristic_iters = [int(x) for x in args.heuristic_svgd_iters.split(',') if x != '']
    linear_iters = [int(x) for x in args.linear_svgd_iters.split(',') if x != '']

    if args.shapes:
        shapes = [s.strip() for s in args.shapes.split(',')]
    elif args.pilot:
        shapes = PILOT_SHAPES
    else:
        shapes = None

    device = args.device
    names, truths = load_truth(labels=shapes, n=999, split='val',
                               resolution=args.truth_res, device=device)
    print(f"[best_of_n_matrix] {len(names)} Formen: {names}  n_candidates={args.n_candidates}")
    print(f"[best_of_n_matrix] Repraesentationen: {reps}  Strategien: {strategies}  "
         f"Replan-Schemata: {schemes}")

    planners = {rep: PLANNER_BUILDERS[rep](
        args.particle_ckpt if rep == 'particles' else args.spectral_ckpt, device)
        for rep in reps}
    refiner = SvgdRefiner(seed=args.seed)
    ee = ExploreExploitErgodic(device=device)

    out_dir = os.path.join(_here, 'results', args.out_tag)
    raw_dir = os.path.join(out_dir, 'raw')
    tables_dir = os.path.join(out_dir, 'tables')
    plots_dir = os.path.join(out_dir, 'plots')
    for d in (raw_dir, tables_dir, plots_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({
            'n_candidates': args.n_candidates, 'selection': args.selection,
            'representations': reps,
            'strategies': strategies, 'replan_schemes': schemes,
            'heuristic_families': ([] if args.skip_heuristic_linear else heuristic_families),
            'heuristic_svgd_iters': heuristic_iters, 'linear_svgd_iters': linear_iters,
            'shapes_resolved': names, 'knowledge_conditions': vr.KNOWLEDGE_CONDITIONS,
        }, f, indent=2)

    rows = []
    panels = {}
    n_skipped = [0]

    def _collect(cond, row, curve, truth_np, name):
        key = (cond, row['variant_id'])
        panels.setdefault(key, []).append((name, truth_np, curve.detach().cpu().numpy()))

    # SLURM warns 120s before the 24h hard limit via `--signal=SIGTERM@120`
    # (see the run_job_*.bash scripts) -- catching it below turns that into a
    # clean finalize-and-exit instead of a hard kill mid-write. Every row is
    # already saved to disk the moment it's computed (see `row_done`'s
    # docstring), so nothing beyond the row in flight when the signal
    # arrives is ever at risk.
    signal.signal(signal.SIGTERM, _sigterm_handler)
    interrupted = False

    t0 = time.time()
    try:
        for i, name in enumerate(names):
            truth = truths[i]
            truth_np = truth.detach().cpu().numpy()
            phi_k_truth = ee.target_coeffs(truth)

            # -- Lawnmower (deterministic) + Irrfahrt (best-of-N) -----------
            if not args.no_viz and row_done(raw_dir, 'lawnmower', 'fixed', None,
                                            'shared', name):
                n_skipped[0] += 1
            else:
                lawn = vr.lawnmower_variant().to(device)
                base_row = compute_row(lawn, truth, phi_k_truth, ee, name,
                                       'lawnmower', 'fixed', None, 'shared')
                if not args.no_viz:
                    save_trajectory_files(lawn, base_row, truth, raw_dir)
                for cond in vr.KNOWLEDGE_CONDITIONS:
                    row = dict(base_row, knowledge_condition=cond)
                    rows.append(row)
                    _collect(cond, row, lawn, truth_np, name)

            if not args.no_viz and row_done(raw_dir, 'random_walk', 'fixed', None,
                                            'shared', name):
                n_skipped[0] += 1
            else:
                _rw_counter = itertools.count(1)

                def _rw_candidate():
                    seed = args.seed * 131 + i * 1000 + next(_rw_counter)
                    return vr.random_walk_variant(seed=seed).to(device)

                best_curve, base_row = best_of_n_curve_and_row(
                    _rw_candidate, args.n_candidates, truth, phi_k_truth, ee, name,
                    'random_walk', 'fixed', None, 'shared')
                if not args.no_viz:
                    save_trajectory_files(best_curve, base_row, truth, raw_dir)
                for cond in vr.KNOWLEDGE_CONDITIONS:
                    row = dict(base_row, knowledge_condition=cond)
                    rows.append(row)
                    _collect(cond, row, best_curve, truth_np, name)

            # -- CFM: best-of-N je (Repraesentation, Wissensstufe, Strategie,
            #    Replan-Schema) ------------------------------------------------
            for rep in reps:
                planner = planners[rep]
                gen = vr.REPRESENTATIONS[rep]
                gen_bon = vr.REPRESENTATIONS_BEST_OF_N[rep]
                for cond in vr.KNOWLEDGE_CONDITIONS:
                    for strat in strategies:
                        s = vr.STRATEGIES[strat]
                        belief0 = vr.build_belief(
                            cond, truth, seed=args.seed, device=device,
                            gp_noise=s.get('gp_noise', 0.05),
                            gp_lengthscale=s.get('gp_lengthscale', 0.08))
                        for scheme in schemes:
                            sub = f"{strat}__{scheme}"
                            if not args.no_viz and row_done(raw_dir, rep, sub, None,
                                                            cond, name):
                                n_skipped[0] += 1
                                continue

                            if args.selection == 'pre_svgd':
                                if scheme == 'no_replan':
                                    best_curve, raw_scores = gen_bon['no_replan'](
                                        planner, belief0.clone(), strat, refiner,
                                        truth, args.n_candidates)
                                elif scheme == 'replan_1_6':
                                    best_curve, raw_scores = gen_bon['replan_1_6'](
                                        planner, belief0.clone(), truth, cond, strat,
                                        refiner, args.n_candidates)
                                else:
                                    raise KeyError(f"unbekanntes Replan-Schema {scheme!r}")
                                row = compute_row(best_curve, truth, phi_k_truth, ee,
                                                  name, rep, sub, None, cond)
                                row['coverage_mean'] = float(np.mean(raw_scores))
                                row['coverage_std'] = float(np.std(raw_scores))
                                row['n_candidates'] = args.n_candidates
                                row['selection'] = 'pre_svgd'
                            else:
                                # Called synchronously within this same iteration
                                # (`best_of_n_curve_and_row` below), so the closure
                                # over `strat`/`cond`/`belief0` is safe despite the
                                # loop -- no deferred/stored-for-later invocation
                                # that Python's late-binding closures would trip up.
                                if scheme == 'no_replan':
                                    def gen_fn():
                                        return gen['no_replan'](planner, belief0.clone(), strat, refiner)
                                elif scheme == 'replan_1_6':
                                    def gen_fn():
                                        return gen['replan_1_6'](planner, belief0.clone(),
                                                                 truth, cond, strat, refiner)
                                else:
                                    raise KeyError(f"unbekanntes Replan-Schema {scheme!r}")
                                best_curve, row = best_of_n_curve_and_row(
                                    gen_fn, args.n_candidates, truth, phi_k_truth, ee,
                                    name, rep, sub, None, cond)

                            row.update(representation=rep, strategy=strat, replan_scheme=scheme)
                            if not args.no_viz:
                                save_trajectory_files(best_curve, row, truth, raw_dir)
                            rows.append(row)
                            _collect(cond, row, best_curve, truth_np, name)

            # -- Heuristik/lineare Waypoints (deterministisch, aus Phase 2) -----
            if not args.skip_heuristic_linear:
                for cond in vr.KNOWLEDGE_CONDITIONS:
                    for family in heuristic_families:
                        strat = vr.HEURISTIC_LINEAR_STRATEGIES[family]
                        s = vr.STRATEGIES[strat]
                        belief = vr.build_belief(
                            cond, truth, seed=args.seed, device=device,
                            gp_noise=s.get('gp_noise', 0.05),
                            gp_lengthscale=s.get('gp_lengthscale', 0.08))
                        sub = f"{family}__{strat}"

                        for n_iters in heuristic_iters:
                            if not args.no_viz and row_done(raw_dir, 'heuristic_tuned', sub,
                                                            n_iters, cond, name):
                                n_skipped[0] += 1
                                continue
                            curve = vr.gui_heuristic_variant_tuned(belief, strat, n_iters, refiner)
                            row = compute_row(curve, truth, phi_k_truth, ee, name,
                                              'heuristic_tuned', sub, n_iters, cond)
                            row.update(family=family, strategy=strat)
                            if not args.no_viz:
                                save_trajectory_files(curve, row, truth, raw_dir)
                            rows.append(row)
                            _collect(cond, row, curve, truth_np, name)

                        for n_iters in linear_iters:
                            if not args.no_viz and row_done(raw_dir, 'linear_waypoints_tuned',
                                                            sub, n_iters, cond, name):
                                n_skipped[0] += 1
                                continue
                            curve = vr.linear_waypoints_variant_tuned(belief, strat, n_iters, refiner)
                            row = compute_row(curve, truth, phi_k_truth, ee, name,
                                              'linear_waypoints_tuned', sub, n_iters, cond)
                            row.update(family=family, strategy=strat)
                            if not args.no_viz:
                                save_trajectory_files(curve, row, truth, raw_dir)
                            rows.append(row)
                            _collect(cond, row, curve, truth_np, name)

            print(f"[best_of_n_matrix] [{i + 1}/{len(names)}] {name} fertig, "
                 f"{time.time() - t0:.1f}s seit Start, {len(rows)} Zeilen neu, "
                 f"{n_skipped[0]} uebersprungen (schon gecacht)")
    except (KeyboardInterrupt, TerminateInterrupt):
        interrupted = True
        print(f"\n[best_of_n_matrix] [!] unterbrochen (SIGTERM/Ctrl-C) nach "
             f"{time.time() - t0:.1f}s -- baue Tabellen/Plots aus allem bisher "
             "Gecachten und beende sauber. Derselbe Befehl erneut gestartet "
             "setzt automatisch dort fort, wo hier aufgehoert wurde.")

    # -- Tabellen/Plots immer aus ALLEN gecachten Zeilen neu bauen, nicht nur
    #    aus denen dieses einzelnen Aufrufs -- siehe "Extending a folder
    #    later" im Moduldocstring. Ohne --no_viz ist raw_dir die
    #    Wahrheitsquelle; mit --no_viz gibt es dort nichts zum Zusammenfuehren.
    if args.no_viz:
        all_rows, all_panels = rows, panels
    else:
        # Knowledge-independent baselines (lawnmower/random_walk) are only
        # ever *saved* once, under the literal `knowledge_condition='shared'`
        # (see `save_trajectory_files`/`compute_row` calls above) -- the x4
        # duplication across the real knowledge conditions happens only in
        # the in-memory `rows` list within a single invocation. Rebuilding
        # from disk has to redo that expansion explicitly, or a reloaded
        # folder would silently lose 3 of every 4 shared-baseline rows.
        cached = load_cached_rows(raw_dir)
        all_rows = []
        for r, _c in cached:
            if r['knowledge_condition'] == 'shared':
                all_rows.extend(dict(r, knowledge_condition=cond)
                                for cond in vr.KNOWLEDGE_CONDITIONS)
            else:
                all_rows.append(r)

        shapes_in_cache = sorted({r['shape'] for r, _c in cached})
        cache_names, cache_truths = load_truth(
            labels=shapes_in_cache, n=999, split='val',
            resolution=args.truth_res, device='cpu')
        truth_by_shape = dict(zip(cache_names, cache_truths))
        all_panels = {}
        for row, curve_np in cached:
            if curve_np is None:
                continue
            t = truth_by_shape.get(row['shape'])
            t_np = t.detach().cpu().numpy() if t is not None else None
            conds = (vr.KNOWLEDGE_CONDITIONS if row['knowledge_condition'] == 'shared'
                    else [row['knowledge_condition']])
            for cond in conds:
                key = (cond, row['variant_id'])
                all_panels.setdefault(key, []).append((row['shape'], t_np, curve_np))

    summarise(all_rows, tables_dir)
    plot_metric_bars(all_rows, plots_dir, show_distribution=True)
    plot_tradeoff(all_rows, plots_dir)

    panel_dir = os.path.join(plots_dir, 'holdout_panels')
    for (cond, vid), items in all_panels.items():
        d = os.path.join(panel_dir, cond)
        os.makedirs(d, exist_ok=True)
        viz.plot_holdout_panel(items, os.path.join(d, f'{vid}.png'), title=f'{vid} | {cond}')

    status = 'unterbrochen (SIGTERM/Ctrl-C)' if interrupted else 'fertig'
    print(f"[best_of_n_matrix] {status}: {len(rows)} Zeilen neu in diesem Lauf, "
         f"{n_skipped[0]} uebersprungen (schon gecacht), "
         f"{len(all_rows)} Zeilen insgesamt im Ordner, "
         f"{time.time() - t0:.1f}s gesamt -> {out_dir}")
    if interrupted:
        sys.exit(0)


if __name__ == '__main__':
    main()
