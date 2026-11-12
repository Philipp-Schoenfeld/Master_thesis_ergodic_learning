#!/usr/bin/env python3
r"""
run_convergence_tests.py
========================
Systematic convergence tests of the active-sensing pipeline (replanning mission of
`sample_annealed_mission.run_mission`, 5 rounds, kappa 1 -> 0.05, as in
`run_exploration_comparison.py`). "Convergence" is looked at on four levels:

  sweep     MISSION level, the solver budget. The same missions (same shapes, seeds, belief) with
            n_iters in {0, 10, 25, 50, 100, 250, 500, 1000} SVGD iterations per round and every
            ergodic-search variant (network prior + TSVEC-SVGD, straight rays + TSVEC-SVGD, i.e.
            no prior, straight rays + Adam-SVGD). Question: how many iterations does the
            final-round ergodic error E_truth need to converge, and from which budget on does the
            learned prior stop paying off? 0 iterations = the raw candidates (B-spline fit only).
            Paired differences to the no-prior search with 95 % bootstrap CIs over shapes.
            Iteration counts that the finished comparison DB already holds (default 500) are not
            recomputed, they are read from that DB.
  trace     SOLVER level inside the mission. One long run (default 1000 iterations) per method
            with every candidate's ergodic error against the round's target recorded at ~40
            log-spaced iteration counts: E(iteration) of the best / median candidate, iterations
            to 90 % of the total improvement (n90), residual improvement of the last half
            (plateau check), iterations the prior saves to reach the error the no-prior search
            reaches after N iterations.
  analysis  no new compute, from stored results:
              variants   does the number of random restarts suffice? CI half-width of the
                         methods' mean error and of the paired difference vs. the number of
                         variants used (from the comparison DB), plus stability of the ranking
              reach      share of runs that reach 90 / 99 % GT coverage and the path they need
              belief     belief RMSE / correlation / NLL vs. round, per kernel
              shards     solver convergence of the OTHER active-sensing pipeline
                         (`run_mission_eval.py`, 30 candidates): E vs. iteration of the stored
                         candidate states of every `results/*/belief_*/shards` found

    python run_convergence_tests.py --smoke            # mechanics, seconds, separate folder
    python run_convergence_tests.py --dry_run          # count the missions still to compute
    python run_convergence_tests.py --parts analysis   # only the cheap analyses
    python run_convergence_tests.py --parts sweep --methods lintsvec     # one method (parallel jobs)

Output: ``results/convergence_tests_<tag>/`` with sweep.db (same schema as the comparison DB,
method name ``<method>@<n_iters>``), trace/*.npz, CSV tables and figures. Everything resumable.
"""
import argparse
import csv
import glob
import json
import math
import os
import sqlite3
import time
import types

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

import run_exploration_comparison as rec                                    # noqa: E402
import run_svgd_convergence as rsc                                          # noqa: E402
import sample_annealed_mission as sam                                       # noqa: E402
from common.acquisition import kappa_schedule                               # noqa: E402
from common.data import load_truth                                          # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
TEXT, MUTED, SPINE = '#1A1A2E', '#555555', '#CCCCCC'
PARTS = ('sweep', 'trace', 'analysis')
DEFAULT_ITERS = (0, 10, 25, 50, 100, 250, 500, 1000)
MAIN_DB = os.path.join(_here, 'plots', 'exploration_comparison', 'exploration_comparison.db')
REF_METHOD = 'lintsvec'          # the ergodic search without a learned prior


def default_methods():
    """The chosen best models (best_models.json from select_best_models.py) + the no-prior search
    + the Adam reference; falls back to the models of the first comparison."""
    path = os.path.join(_here, 'best_models.json')
    try:
        with open(path, encoding='utf-8') as f:
            ch = json.load(f)['chosen']
        nets = [ch['dit'], ch['unet'], ch['selfsup']]
    except Exception:
        nets = ['dit630', 'unet630', 'selfsup']
    return list(dict.fromkeys(nets + ['lintsvec', 'linadam']))


def style(ax):
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(SPINE)


def boot_ci(x, n=2000, seed=0):
    """Mean and 95 % bootstrap CI over the entries of x (NaNs dropped)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def split_name(method):
    """'lintsvec@250' -> ('lintsvec', 250); a plain method name -> (name, None)."""
    if '@' in method:
        m, it = method.split('@')
        return m, int(it)
    return method, None


# ===========================================================================
# sweep: n_iters on the mission level
# ===========================================================================

def part_sweep(a, ctx, device, db, main_db_path):
    names, truths = load_truth(labels=a.shape_names, n=a.n_shapes or 10_000, split='val',
                               resolution=rsc.TRUTH_RES, device=device)
    names = list(names)
    for n, t in zip(names, truths):
        if not a.dry_run:
            db.put_truth(n, t)
    kappas = [kappa_schedule(r, a.n_rounds, a.kappa0, a.kappa1) for r in range(a.n_rounds)]
    main = (sqlite3.connect(f'file:{main_db_path}?mode=ro', uri=True)
            if os.path.isfile(main_db_path) else None)

    def in_main(shape, kernel, method, variant, iters):
        if main is None or iters != a.main_iters:
            return False
        return main.execute('SELECT 1 FROM runs WHERE shape=? AND kernel=? AND method=? AND variant=? '
                            'AND n_iters=?', (shape, kernel, method, variant, iters)).fetchone() is not None

    t_start = time.time()
    for method in a.compute_methods:
        todo = [(k, i, n, v, it) for v in range(a.n_variants) for i, n in enumerate(names)
                for k in a.kernels for it in a.iters
                if not db.has_run(n, k, f'{method}@{it}', v) and not in_main(n, k, method, v, it)]
        print(f'\n=== sweep {method}: {len(todo)} missions to compute ===')
        if not todo or a.dry_run:
            continue
        planner = rec.build_planner(method, a, device)
        bs = ctx.refiner_for(method)
        for k, i, n, v, it in todo:
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                print('  time budget reached, stopping cleanly (resume later)')
                return
            cfg = types.SimpleNamespace(**vars(a))
            cfg.n_iters = it
            cfg.conv_iters = None
            res = rec.run_one(method, k, n, truths[i].detach().cpu(), planner, bs, ctx.B32, ctx.ee,
                              ctx.base_args, cfg, kappas, device, i * 100 + v)
            db.put_run(n, k, f'{method}@{it}', v, res, cfg)
            last = res['rows'][-1]
            print(f'  [{k}/{method}@{it}] {n} v{v}: {res["seconds"]:.0f}s  E_truth={last["E_truth"]:.2f}  '
                  f'swept={last["swept06"]:.0%}')
        del planner
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def load_sweep_table(db, main_db_path, methods, kernels, main_iters, max_variant):
    """-> (dict (kernel, method, iters) -> {(shape, variant): final-round metrics + t_mission}, n_rounds),
    merging the sweep DB and (for `main_iters`, variants < max_variant) the comparison DB."""
    out = {}
    cols = ['E_truth', 'E_belief', 'swept06', 'path_len', 'belief_rmse', 'E_ref_best', 'E_init_best',
            'out_gt_frac']

    def pull(conn, only_iters=None):
        q = (f'SELECT u.shape,u.kernel,u.method,u.variant,u.n_iters,r.round,'
             f'{",".join("r." + c for c in cols)},r.t_round FROM rounds r JOIN runs u ON u.id=r.run_id')
        rows = conn.execute(q).fetchall()
        n_rounds = max(r[5] for r in rows) + 1 if rows else 0
        secs = {}
        for rec_ in rows:
            shape, kernel, method, variant, n_it, rnd = rec_[:6]
            base, it = split_name(method)
            it = n_it if it is None else it
            if (kernel not in kernels or base not in methods or variant >= max_variant
                    or (only_iters is not None and it not in only_iters)):
                continue
            key = (kernel, base, it)
            if rec_[-1] is not None:
                secs[key + (shape, variant)] = secs.get(key + (shape, variant), 0.0) + rec_[-1]
            if rnd == n_rounds - 1:
                out.setdefault(key, {})[(shape, variant)] = dict(zip(cols, rec_[6:-1]))
        for k, t in secs.items():
            cell = out.get(k[:3], {}).get(k[3:])
            if cell is not None:
                cell['t_mission'] = t
        return n_rounds

    n_rounds = pull(db.conn)
    if os.path.isfile(main_db_path):
        main = sqlite3.connect(f'file:{main_db_path}?mode=ro', uri=True)
        n_rounds = max(n_rounds, pull(main, only_iters={main_iters}))
    return out, n_rounds


def analyse_sweep(a, db, main_db_path, out_dir):
    methods = a.methods
    kernels = a.kernels
    table, n_rounds = load_sweep_table(db, main_db_path, methods, kernels, a.main_iters, a.n_variants)
    if not table:
        print('  sweep: nothing stored yet')
        return
    iters = sorted({it for (_, _, it) in table})
    rows = []
    for kernel in kernels:
        ref = table.get((kernel, a.ref_method))
        for method in methods:
            for it in iters:
                cell = table.get((kernel, method, it))
                if not cell:
                    continue
                shapes = sorted({s for s, _ in cell})
                per_shape = {s: np.mean([cell[(s, v)]['E_truth'] for (s2, v) in cell if s2 == s]) for s in shapes}
                E_mean, lo, hi = boot_ci(list(per_shape.values()))
                row = dict(kernel=kernel, method=method, n_iters=it, n_shapes=len(shapes),
                           n_runs=len(cell), E_truth=E_mean, E_lo=lo, E_hi=hi)
                for c in ('swept06', 'E_belief', 'path_len', 'out_gt_frac', 'E_ref_best', 'E_init_best'):
                    vals = [np.mean([cell[(s, v)][c] for (s2, v) in cell if s2 == s and cell[(s, v)][c] is not None])
                            for s in shapes if any(s2 == s and cell[(s, v)][c] is not None for (s2, v) in cell)]
                    row[c] = float(np.mean(vals)) if vals else float('nan')
                ref_cell = table.get((kernel, a.ref_method, it))
                if ref_cell and method != a.ref_method:
                    ref_shapes = {s: np.mean([ref_cell[(s, v)]['E_truth'] for (s2, v) in ref_cell if s2 == s])
                                  for s in {s for s, _ in ref_cell}}
                    d = [per_shape[s] - ref_shapes[s] for s in shapes if s in ref_shapes]
                    row['dE_vs_ref'], row['dE_vs_ref_lo'], row['dE_vs_ref_hi'] = boot_ci(d)
                rows.append(row)
    keys = ['kernel', 'method', 'n_iters', 'n_shapes', 'n_runs', 'E_truth', 'E_lo', 'E_hi', 'dE_vs_ref',
            'dE_vs_ref_lo', 'dE_vs_ref_hi', 'swept06', 'E_belief', 'path_len', 'out_gt_frac', 'E_init_best',
            'E_ref_best']
    with open(os.path.join(out_dir, 'sweep_by_iters.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})

    # saturation: smallest budget within tol of the largest budget's error (mean curve)
    with open(os.path.join(out_dir, 'sweep_saturation.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'E_at_max_iters', 'max_iters', 'tol', 'iters_to_within_tol'])
        for kernel in kernels:
            for method in methods:
                curve = sorted((r['n_iters'], r['E_truth']) for r in rows
                               if r['kernel'] == kernel and r['method'] == method)
                if len(curve) < 2:
                    continue
                e_max = curve[-1][1]
                for tol in (0.05, 0.10):
                    n_ok = next((it for it, e in curve if e <= e_max * (1 + tol)), curve[-1][0])
                    w.writerow([kernel, method, f'{e_max:.3f}', curve[-1][0], tol, n_ok])

    # figure
    fig, axes = plt.subplots(len(kernels), 3, figsize=(17, 4.6 * len(kernels)), facecolor='white', squeeze=False)
    for ki, kernel in enumerate(kernels):
        for pi, (col, title, delta) in enumerate((('E_truth', 'final E vs. ground truth ↓', False),
                                                  ('dE_vs_ref', f'paired Δ vs. {rec.METHOD_LABEL.get(a.ref_method, a.ref_method)} '
                                                                '(<0: better) ↓', True),
                                                  ('E_ref_best', 'best candidate vs. its round target ↓', False))):
            ax = axes[ki, pi]
            for method in methods:
                rs = sorted((r for r in rows if r['kernel'] == kernel and r['method'] == method
                             and not (isinstance(r.get(col), float) and math.isnan(r[col])) and col in r),
                            key=lambda r: r['n_iters'])
                if not rs:
                    continue
                x = np.array([max(r['n_iters'], 1) for r in rs], dtype=float)     # 0 plotted at 1 (log axis)
                y = np.array([r[col] for r in rs], dtype=float)
                color = rec.METHOD_COLOR.get(method)
                ax.plot(x, y, marker='o', lw=2, color=color, label=rec.METHOD_LABEL.get(method, method))
                if delta:
                    lo = np.array([r['dE_vs_ref_lo'] for r in rs]); hi = np.array([r['dE_vs_ref_hi'] for r in rs])
                    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
                elif col == 'E_truth':
                    ax.fill_between(x, [r['E_lo'] for r in rs], [r['E_hi'] for r in rs], color=color, alpha=0.12, lw=0)
            if delta:
                ax.axhline(0, color='#999', lw=1)
            ax.set_xscale('log')
            ax.set_xlabel('SVGD iterations per round (0 plotted at 1: raw candidates)', fontsize=8, color=MUTED)
            ax.set_title(f'{title}\n{kernel.upper()} belief', fontsize=9.5, color=TEXT)
            style(ax)
            if pi == 0:
                ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle('Convergence in the solver budget: final-round error of the 5-round mission '
                 '(mean over heldout shapes, 95 % bootstrap CI over shapes)', fontsize=12, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, 'sweep_convergence.png'), dpi=120, facecolor='white')
    plt.close(fig)

    # cost-benefit: seconds per mission vs. E_truth (needs the t_* columns of this DB)
    fig, axes = plt.subplots(1, len(kernels), figsize=(7.5 * len(kernels), 5), facecolor='white', squeeze=False)
    any_t = False
    for ki, kernel in enumerate(kernels):
        ax = axes[0, ki]
        for method in methods:
            pts = []
            for it in iters:
                cell = table.get((kernel, method, it))
                if not cell:
                    continue
                tv = [c['t_mission'] for c in cell.values() if c.get('t_mission') is not None]
                if not tv:
                    continue
                pts.append((float(np.mean(tv)),
                            float(np.mean([c['E_truth'] for c in cell.values()])), it))
            if pts:
                any_t = True
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', lw=2,
                        color=rec.METHOD_COLOR.get(method), label=rec.METHOD_LABEL.get(method, method))
                for t_, e_, it in pts:
                    ax.annotate(str(it), (t_, e_), fontsize=6.5, color=MUTED, xytext=(3, 3), textcoords='offset points')
        ax.set_xscale('log')
        ax.set_xlabel('seconds per mission (sum of the rounds; main-DB runs: timed since 2026-11-11 only)', fontsize=8, color=MUTED)
        ax.set_ylabel('final E vs. ground truth', fontsize=8, color=MUTED)
        ax.set_title(f'Cost vs. quality, {kernel.upper()} belief (labels: iterations)', fontsize=9.5, color=TEXT)
        style(ax)
        ax.legend(fontsize=7.5, frameon=False)
    if any_t:
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'sweep_cost_vs_quality.png'), dpi=120, facecolor='white')
    plt.close(fig)
    print(f'  sweep tables + figures in {out_dir}')


# ===========================================================================
# trace: E(iteration) inside the mission
# ===========================================================================

def trace_iters(n_iters, n_points=40):
    its = np.unique(np.round(np.geomspace(1, n_iters, n_points - 1)).astype(int))
    return np.concatenate([[0], its]).astype(int)


def part_trace(a, ctx, device, out_dir):
    names, truths = load_truth(labels=a.shape_names, n=a.n_shapes or 10_000, split='val',
                               resolution=rsc.TRUTH_RES, device=device)
    names = list(names)
    kappas = [kappa_schedule(r, a.n_rounds, a.kappa0, a.kappa1) for r in range(a.n_rounds)]
    tdir = os.path.join(out_dir, 'trace')
    os.makedirs(tdir, exist_ok=True)
    its = trace_iters(a.trace_iters)
    t_start = time.time()
    for method in a.compute_methods:
        for kernel in a.trace_kernels:
            todo = [(i, n) for i, n in enumerate(names)
                    if not os.path.isfile(os.path.join(tdir, f'{method}__{kernel}__{n}.npz'))]
            print(f'\n=== trace {method} ({kernel}): {len(todo)} missions to compute ===')
            if not todo or a.dry_run:
                continue
            planner = rec.build_planner(method, a, device)
            bs = ctx.refiner_for(method)
            for i, n in todo:
                if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                    print('  time budget reached, stopping cleanly (resume later)')
                    return
                cfg = types.SimpleNamespace(**vars(a))
                cfg.n_iters, cfg.conv_iters = a.trace_iters, its
                res = rec.run_one(method, kernel, n, truths[i].detach().cpu(), planner, bs, ctx.B32, ctx.ee,
                                  ctx.base_args, cfg, kappas, device, i * 100)
                conv = np.stack([c for c in res['convs'] if c is not None]).astype(np.float32)
                np.savez_compressed(os.path.join(tdir, f'{method}__{kernel}__{n}.npz'), iters=its, conv=conv,
                                    E_truth=np.array([r['E_truth'] for r in res['rows']]),
                                    seconds=res['seconds'])
                print(f'  [{kernel}/{method}] {n}: {res["seconds"]:.0f}s  '
                      f'E_best {conv[0, 0].min():.2f} -> {conv[-1, -1].min():.2f} (last round)')
            del planner
            if device.startswith('cuda'):
                torch.cuda.empty_cache()


def n_to_fraction(its, curve, frac=0.9):
    """First iteration at which `frac` of the total improvement (curve[0] - curve[-1]) is reached."""
    total = curve[0] - curve[-1]
    if total <= 1e-9:
        return float('nan')
    target = curve[0] - frac * total
    idx = np.flatnonzero(curve <= target)
    return float(its[idx[0]]) if len(idx) else float('nan')


def first_below(its, curve, level):
    idx = np.flatnonzero(curve <= level)
    return float(its[idx[0]]) if len(idx) else float('nan')


def analyse_trace(a, out_dir):
    tdir = os.path.join(out_dir, 'trace')
    files = sorted(glob.glob(os.path.join(tdir, '*.npz')))
    if not files:
        print('  trace: nothing stored yet')
        return
    data = {}       # (kernel, method) -> list of (shape, iters, conv[round, iter, cand])
    for f in files:
        method, kernel, shape = os.path.basename(f)[:-4].split('__')
        z = np.load(f)
        data.setdefault((kernel, method), []).append((shape, z['iters'], z['conv']))
    rows = []
    for (kernel, method), items in sorted(data.items()):
        its = items[0][1]
        n_rounds = items[0][2].shape[0]
        for r in range(n_rounds):
            best = np.stack([c[r].min(axis=-1) for _, _, c in items])        # (shapes, iters)
            med = np.stack([np.median(c[r], axis=-1) for _, _, c in items])
            n90 = [n_to_fraction(its, b, 0.9) for b in best]
            n99 = [n_to_fraction(its, b, 0.99) for b in best]
            half = int(np.searchsorted(its, its[-1] // 2))
            late = [(b[half] - b[-1]) / max(b[0], 1e-9) for b in best]       # share of the initial error still gained in the last half
            rows.append(dict(kernel=kernel, method=method, round=r, n_shapes=len(items),
                             E_best_0=best[:, 0].mean(), E_best_end=best[:, -1].mean(),
                             E_med_0=med[:, 0].mean(), E_med_end=med[:, -1].mean(),
                             n90_median=float(np.nanmedian(n90)), n99_median=float(np.nanmedian(n99)),
                             late_gain_share=float(np.mean(late))))
    keys = list(rows[0])
    with open(os.path.join(out_dir, 'trace_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # iterations a prior saves: iteration at which the method's best candidate first reaches the error the
    # no-prior search has after `ref_iters` iterations (per shape and round, then the median)
    with open(os.path.join(out_dir, 'trace_prior_savings.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'round', 'ref_method', 'ref_iters', 'iters_to_reach_median',
                    'share_reached', 'n_shapes'])
        for kernel in a.trace_kernels:
            ref = {s: (its_, c) for s, its_, c in data.get((kernel, a.ref_method), [])}
            for method in a.methods:
                if method == a.ref_method:
                    continue
                for r in range(n_rounds):
                    for ref_it in (10, 50, 100, 250):
                        if ref_it > a.trace_iters:
                            continue
                        reach = []
                        for s, its_, c in data.get((kernel, method), []):
                            if s not in ref:
                                continue
                            ref_curve = ref[s][1][r].min(axis=-1)
                            level = ref_curve[min(int(np.searchsorted(ref[s][0], ref_it)), len(ref_curve) - 1)]
                            reach.append(first_below(its_, c[r].min(axis=-1), level))
                        if reach:
                            arr = np.array(reach)
                            w.writerow([kernel, method, r, a.ref_method, ref_it,
                                        f'{np.nanmedian(arr):.1f}' if (~np.isnan(arr)).any() else 'nan',
                                        f'{np.mean(~np.isnan(arr)):.2f}', len(arr)])

    # figure: E(best candidate)/E(iteration 0) vs iterations, one column per round
    kernel = a.trace_kernels[0]
    fig, axes = plt.subplots(2, n_rounds, figsize=(3.6 * n_rounds, 7.2), facecolor='white', squeeze=False)
    for r in range(n_rounds):
        for ri, (what, ylabel) in enumerate((('abs', 'best candidate E vs. round target'),
                                             ('rel', 'median candidate, share of initial E'))):
            ax = axes[ri, r]
            for method in a.methods:
                items = data.get((kernel, method))
                if not items:
                    continue
                its = items[0][1]
                if what == 'abs':
                    y = np.stack([c[r].min(axis=-1) for _, _, c in items])
                else:
                    y = np.stack([np.median(c[r], axis=-1) / max(np.median(c[r][0]), 1e-9) for _, _, c in items])
                m, sem = y.mean(0), y.std(0) / math.sqrt(len(y))
                x = np.maximum(its, 1)
                ax.plot(x, m, lw=2, color=rec.METHOD_COLOR.get(method), label=rec.METHOD_LABEL.get(method, method))
                ax.fill_between(x, m - sem, m + sem, color=rec.METHOD_COLOR.get(method), alpha=0.15, lw=0)
            ax.set_xscale('log'); ax.set_yscale('log')
            if ri == 1:
                ax.set_xlabel('SVGD iterations', fontsize=8, color=MUTED)
            if r == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
            if ri == 0:
                ax.set_title(f'round {r}', fontsize=10, color=TEXT)
            style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'Solver convergence inside the mission ({kernel.upper()} belief; mean ± s.e.m. over shapes, '
                 'iteration 0 = B-spline fit of the raw candidates)', fontsize=12, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, 'trace_convergence.png'), dpi=120, facecolor='white')
    plt.close(fig)
    print(f'  trace tables + figure in {out_dir}')


# ===========================================================================
# analysis: from stored results
# ===========================================================================

def final_matrix(conn, kernel, method, col='E_truth'):
    """-> (shapes, array[shape, variant]) of the final round's `col`."""
    q = ('SELECT u.shape,u.variant,r.' + col + ' FROM rounds r JOIN runs u ON u.id=r.run_id '
         'WHERE u.kernel=? AND u.method=? AND r.round=(SELECT MAX(round) FROM rounds WHERE run_id=u.id)')
    rows = conn.execute(q, (kernel, method)).fetchall()
    if not rows:
        return [], np.zeros((0, 0))
    shapes = sorted({r[0] for r in rows})
    nv = max(r[1] for r in rows) + 1
    arr = np.full((len(shapes), nv), np.nan)
    for s, v, e in rows:
        arr[shapes.index(s), v] = np.nan if e is None else e
    return shapes, arr


def analyse_variants(conn, a, out_dir):
    methods = [m for (m,) in conn.execute('SELECT DISTINCT method FROM runs ORDER BY method')]
    rng = np.random.default_rng(0)
    rows = []
    for kernel in a.analysis_kernels:
        mats = {}
        for m in methods:
            shapes, arr = final_matrix(conn, kernel, m)
            if arr.size and not np.isnan(arr).any():
                mats[m] = (shapes, arr)
        if len(mats) < 2:
            continue
        common = sorted(set.intersection(*[set(s) for s, _ in mats.values()]))
        nv = min(arr.shape[1] for _, arr in mats.values())
        mats = {m: arr[[shapes.index(s) for s in common], :nv] for m, (shapes, arr) in mats.items()}
        ref = REF_METHOD if REF_METHOD in mats else 'unet630'
        full_order = np.argsort(np.argsort([mats[m].mean() for m in mats]))
        for n in range(1, nv + 1):
            subsets = [tuple(rng.choice(nv, size=n, replace=False)) for _ in range(60 if n < nv else 1)]
            for m in mats:
                means, deltas = [], []
                for sub in subsets:
                    per_shape = mats[m][:, list(sub)].mean(axis=1)
                    means.append(per_shape.mean())
                    if m != ref and ref in mats:
                        deltas.append((per_shape - mats[ref][:, list(sub)].mean(axis=1)))
                half = [(lambda c: (c[2] - c[1]) / 2)(boot_ci(mats[m][:, list(sub)].mean(axis=1), n=400))
                        for sub in subsets[:10]]
                row = dict(kernel=kernel, method=m, n_variants=n, E_mean=float(np.mean(means)),
                           E_mean_sd_over_subsets=float(np.std(means)), ci_halfwidth_mean=float(np.mean(half)))
                if deltas:
                    hw = [(lambda c: (c[2] - c[1]) / 2)(boot_ci(d, n=400)) for d in deltas[:10]]
                    row.update(ref_method=ref, delta_vs_ref=float(np.mean([d.mean() for d in deltas])),
                               delta_ci_halfwidth=float(np.mean(hw)))
                rows.append(row)
            # ranking stability: Spearman between the ranking with n variants and with all variants
            rho = []
            for sub in subsets:
                order = np.argsort(np.argsort([mats[m][:, list(sub)].mean() for m in mats]))
                rho.append(np.corrcoef(order, full_order)[0, 1])
            for r_ in rows:
                if r_['kernel'] == kernel and r_['n_variants'] == n:
                    r_['rank_rho_vs_full'] = float(np.mean(rho))
    if not rows:
        return
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('kernel', 'method', 'n_variants'), k))
    with open(os.path.join(out_dir, 'variant_convergence.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fig, axes = plt.subplots(len(a.analysis_kernels), 3, figsize=(16, 4.4 * len(a.analysis_kernels)), facecolor='white', squeeze=False)
    for ki, kernel in enumerate(a.analysis_kernels):
        sel = [r for r in rows if r['kernel'] == kernel]
        for pi, (col, title) in enumerate((('ci_halfwidth_mean', 'CI half-width of the mean E (over shapes)'),
                                           ('delta_ci_halfwidth', 'CI half-width of the paired Δ vs. reference'),
                                           ('rank_rho_vs_full', 'ranking stability (Spearman ρ vs. all variants)'))):
            ax = axes[ki, pi]
            for m in sorted({r['method'] for r in sel}):
                rs = sorted((r for r in sel if r['method'] == m and col in r), key=lambda r: r['n_variants'])
                if rs:
                    ax.plot([r['n_variants'] for r in rs], [r[col] for r in rs], marker='o', lw=1.8,
                            color=rec.METHOD_COLOR.get(m), label=rec.METHOD_LABEL.get(m, m))
            ax.set_xlabel('variants used per shape', fontsize=8, color=MUTED)
            ax.set_title(f'{title}\n{kernel.upper()} belief', fontsize=9.5, color=TEXT)
            style(ax)
            if pi == 0:
                ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle('Do the random restarts suffice? Statistical convergence of the comparison', fontsize=12, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, 'variant_convergence.png'), dpi=120, facecolor='white')
    plt.close(fig)


def analyse_reach(conn, a, out_dir):
    q = ('SELECT kernel,method,COUNT(*),AVG(reached90),AVG(reached99),AVG(path_len_90),AVG(path_len_99),'
         'AVG(seconds) FROM runs GROUP BY kernel,method ORDER BY kernel,method')
    with open(os.path.join(out_dir, 'coverage_reach.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'n_runs', 'share_reaching_90', 'share_reaching_99',
                    'mean_path_len_at_90_if_reached', 'mean_path_len_at_99_if_reached', 'mean_seconds_per_mission'])
        for rec_ in conn.execute(q):
            w.writerow([rec_[0], rec_[1], rec_[2]] + [f'{x:.4f}' if x is not None else '' for x in rec_[3:]])


def analyse_belief(conn, a, out_dir):
    cols = ['belief_rmse', 'belief_corr', 'belief_nll', 'belief_cov95', 'n_obs']
    q = (f'SELECT u.kernel,u.method,r.round,{",".join("AVG(r." + c + ")" for c in cols)} '
         'FROM rounds r JOIN runs u ON u.id=r.run_id GROUP BY u.kernel,u.method,r.round ORDER BY 1,2,3')
    rows = conn.execute(q).fetchall()
    with open(os.path.join(out_dir, 'belief_convergence.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'round'] + cols)
        w.writerows(rows)
    fig, axes = plt.subplots(len(a.analysis_kernels), 3, figsize=(15, 4.2 * len(a.analysis_kernels)), facecolor='white', squeeze=False)
    for ki, kernel in enumerate(a.analysis_kernels):
        for pi, (c, title) in enumerate((('belief_rmse', 'belief RMSE vs. GT ↓'), ('belief_corr', 'belief correlation ↑'),
                                         ('belief_nll', 'belief NLL ↓'))):
            ax = axes[ki, pi]
            ci = 3 + cols.index(c)
            for m in sorted({r[1] for r in rows if r[0] == kernel}):
                rs = [r for r in rows if r[0] == kernel and r[1] == m]
                ax.plot([r[2] for r in rs], [r[ci] for r in rs], marker='o', lw=1.8, color=rec.METHOD_COLOR.get(m),
                        label=rec.METHOD_LABEL.get(m, m))
            ax.set_xlabel('round', fontsize=8, color=MUTED)
            ax.set_title(f'{title}\n{kernel.upper()} belief', fontsize=9.5, color=TEXT)
            style(ax)
            if pi == 0:
                ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.suptitle('Belief convergence over the rounds (mean over shapes and variants)', fontsize=12, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, 'belief_convergence.png'), dpi=120, facecolor='white')
    plt.close(fig)


def analyse_shards(a, out_dir):
    """Solver convergence of the run_mission_eval pipeline from its stored candidate E series:
    best-candidate E(iteration) / E(0), mean over shapes and rounds, per run directory and method."""
    shard_files = sorted(glob.glob(os.path.join(_here, 'results', a.shard_glob, 'belief_*', 'shards', '*.db')))
    if not shard_files:
        print('  shards: none found (pull results/*/belief_*/shards from the cluster to analyse them)')
        return
    series = {}
    for path in shard_files:
        run = os.path.relpath(path, os.path.join(_here, 'results')).split(os.sep)[0]
        belief = os.path.basename(os.path.dirname(os.path.dirname(path)))
        cond, _, method = os.path.basename(path)[:-3].split('__')
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        per = {}
        for shape, rnd, stride, n_it, blob in conn.execute(
                'SELECT shape,round,E_stride,n_iters,E_series FROM candidates'):
            e = np.frombuffer(blob, dtype=np.float32)
            its = np.concatenate([np.arange(0, n_it, stride), [n_it]]) if n_it % stride else np.arange(0, n_it + 1, stride)
            if len(its) != len(e):
                continue
            per.setdefault((shape, rnd), []).append(e)
        if not per:
            continue
        curves = np.stack([np.stack(v).min(axis=0) for v in per.values()])     # best candidate per (shape, round)
        series[(run, belief, cond, method)] = (its, curves)
        conn.close()
    rows = []
    for key, (its, curves) in sorted(series.items()):
        rel = curves / np.maximum(curves[:, :1], 1e-9)
        m = np.nanmean(rel, axis=0)
        n90 = float(its[np.flatnonzero(m <= 1 - 0.9 * (1 - m[-1]))[0]])
        rows.append(dict(run=key[0], belief=key[1], condition=key[2], method=key[3], n_series=len(curves),
                         iters=int(its[-1]), E0=float(curves[:, 0].mean()), E_end=float(curves[:, -1].mean()),
                         rel_end=float(m[-1]), n90_of_mean_rel=n90))
    with open(os.path.join(out_dir, 'shards_solver_convergence.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='white')
    for key, (its, curves) in sorted(series.items()):
        if key[2] != 'none_known':
            continue
        m = np.nanmean(curves / np.maximum(curves[:, :1], 1e-9), axis=0)
        ax.plot(np.maximum(its, 1), m, lw=1.6, label=f'{key[0].replace("_tsvec_svgd", "")} / {key[3]}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('SVGD iterations', fontsize=9, color=MUTED)
    ax.set_ylabel('best candidate E / E(iteration 0)', fontsize=9, color=MUTED)
    ax.set_title('run_mission_eval pipeline: solver convergence of the stored candidates (none_known)',
                 fontsize=10, color=TEXT)
    ax.legend(fontsize=6.5, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'shards_solver_convergence.png'), dpi=120, facecolor='white')
    plt.close(fig)
    print(f'  shards: {len(series)} candidate series analysed')


def part_analysis(a, out_dir, main_db_path):
    if os.path.isfile(main_db_path):
        conn = sqlite3.connect(f'file:{main_db_path}?mode=ro', uri=True)
        print('  analysis: variants / reach / belief from', main_db_path)
        analyse_variants(conn, a, out_dir)
        analyse_reach(conn, a, out_dir)
        analyse_belief(conn, a, out_dir)
    else:
        print(f'  analysis: comparison DB {main_db_path} not found, skipping variants / reach / belief')
    analyse_shards(a, out_dir)


# ===========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parts', type=str, default=','.join(PARTS))
    p.add_argument('--tag', type=str, default='main')
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--main_db', type=str, default=MAIN_DB, help='Finished comparison DB (read only).')
    p.add_argument('--main_iters', type=int, default=500,
                   help='n_iters of the comparison DB; its runs are reused for this sweep level.')
    p.add_argument('--methods', type=str, default=None,
                   help='Default: best models of best_models.json + lintsvec + linadam.')
    p.add_argument('--ref_method', type=str, default=REF_METHOD, help='Reference of the paired differences.')
    p.add_argument('--kernels', type=str, default='ak', help='Belief kernels of the sweep / analysis.')
    p.add_argument('--trace_kernels', type=str, default='ak')
    p.add_argument('--analysis_kernels', type=str, default='rbf,ak',
                   help='Belief kernels of the variants / belief analyses of the comparison DB.')
    p.add_argument('--iters', type=str, default=','.join(map(str, DEFAULT_ITERS)),
                   help='n_iters levels of the sweep.')
    p.add_argument('--trace_iters', type=int, default=1000, help='Length of the recorded runs of `trace`.')
    p.add_argument('--n_shapes', type=int, default=None)
    p.add_argument('--shape_names', type=str, default=None)
    p.add_argument('--n_variants', type=int, default=3, help='Variants (seeds) per shape in the sweep.')
    p.add_argument('--n_rounds', type=int, default=5)
    p.add_argument('--n_init', type=int, default=8)
    p.add_argument('--kappa0', type=float, default=1.0)
    p.add_argument('--kappa1', type=float, default=0.05)
    p.add_argument('--walk_len', type=float, default=6.1)
    p.add_argument('--ak_fit_iters', type=int, default=200)
    p.add_argument('--selfsup_ckpt', type=str, default=None)
    p.add_argument('--gp_goal_source', type=str, default='mean')
    p.add_argument('--shard_glob', type=str, default='as_models_*',
                   help='Run directories (under results/) whose belief_*/shards are analysed.')
    p.add_argument('--max_minutes', type=float, default=None, help='Stop cleanly after this wall time.')
    p.add_argument('--smoke', action='store_true', help='Tiny sizes, separate folder (mechanics only).')
    p.add_argument('--no_tables', action='store_true',
                   help='Compute only; skip tables/figures (for per-method cluster jobs, whose partial '
                        'results cannot be analysed alone; analyse once after merging).')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.smoke:
        a.tag, a.iters, a.trace_iters, a.n_shapes, a.n_variants = 'smoke', '0,5,10', 20, 2, 2
        a.n_rounds, a.n_init, a.ak_fit_iters, a.main_iters = 2, 2, 20, 10
        a.methods = a.methods or 'lintsvec,linadam'
    a.methods = (a.methods.split(',') if a.methods else default_methods())
    for m in a.methods + [a.ref_method]:
        if m not in rec.METHODS:
            raise SystemExit(f'unknown method {m}; known: {rec.METHODS}')
    a.parts = [x for x in a.parts.split(',') if x]
    for x in a.parts:
        if x not in PARTS:
            raise SystemExit(f'unknown part {x}; known: {PARTS}')
    a.kernels = a.kernels.split(',')
    a.trace_kernels = a.trace_kernels.split(',')
    a.analysis_kernels = a.analysis_kernels.split(',')
    a.iters = [int(x) for x in a.iters.split(',')]
    a.shape_names = a.shape_names.split(',') if a.shape_names else None
    # only the requested methods are COMPUTED; the reference method is appended for the analysis
    # (paired differences) but is not recomputed by every per-method job
    a.compute_methods = list(a.methods)
    if a.ref_method not in a.methods:
        a.methods = a.methods + [a.ref_method]
    out_dir = a.out_dir or os.path.join(_here, 'results', f'convergence_tests_{a.tag}')
    os.makedirs(out_dir, exist_ok=True)
    device = a.device
    print(f'convergence tests -> {out_dir}\n  parts={a.parts} methods={a.methods} kernels={a.kernels} '
          f'iters={a.iters} variants={a.n_variants}  device={device}')
    db_path = os.path.join(out_dir, 'sweep.db')
    db = rec.ResultDB(db_path)
    if {'sweep', 'trace'} & set(a.parts):
        ctx = rec.make_context(device)
    if 'sweep' in a.parts:
        part_sweep(a, ctx, device, db, a.main_db)
    if 'trace' in a.parts:
        part_trace(a, ctx, device, out_dir)
    if a.dry_run:
        return
    if a.no_tables:
        print('\n--no_tables: skipping tables/figures (run them once on the merged results).')
        return
    print('\n=== tables + figures ===')
    if 'sweep' in a.parts:
        analyse_sweep(a, db, a.main_db, out_dir)
    if 'trace' in a.parts:
        analyse_trace(a, out_dir)
    if 'analysis' in a.parts:
        part_analysis(a, out_dir, a.main_db)


if __name__ == '__main__':
    main()
