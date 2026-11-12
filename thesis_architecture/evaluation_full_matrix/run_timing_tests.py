#!/usr/bin/env python3
r"""
run_timing_tests.py
===================
Systematic compute-time tests of every variant of the active-sensing pipeline (the replanning
mission of `sample_annealed_mission.run_mission` as used by `run_exploration_comparison.py`):
which stage costs how much, how does it scale, and what does one full mission cost per method.

Stages of one planning round (names as in the `t_*` columns of the comparison DB):
  t_target   belief posterior on the grid + target density (UCB with coverage debt)
  t_init     candidate source: network forward + rendering, straight rays, or the baseline path
  t_refine   SVGD refinement of all candidates + candidate selection (0 without refiner)
  t_belief   measurements along the driven segment + belief update (AK: kernel fit, NP: forward)

Parts (``--parts``, default all):
  planner    t_init of every network vs. the number of candidates `n_init`
  init       t_init of the network-free candidate sources (rays, random walk, lawnmower, GP-goal
             line, best ray) vs. `n_init`
  refiner    SVGD refinement time vs. `n_init` and `n_iters` -> seconds per SVGD iteration, for the
             TSVEC-SVGD (networks, lintsvec) and the Adam-SVGD (linadam); `sun` with --refiners
  belief     belief update (observe + posterior on the grid) of rbf / ak / np vs. number of stored
             measurements
  missions   a full mission per method x belief kernel on a few shapes: per-round stage times,
             total seconds, plus the quality reached (E_truth), so cost and benefit sit together

Protocol: GPU-synchronised wall clock (`torch.cuda.synchronize` before and after), `--warmup`
discarded calls, `--reps` timed repetitions, median and inter-quartile range reported. Everything
is recorded together with the hardware (`hardware.json`) because times are only comparable on the
same machine: run the whole script in ONE job on ONE GPU type.

    python run_timing_tests.py --smoke              # tiny sizes, seconds (mechanics only)
    python run_timing_tests.py --dry_run            # list what would be timed, time nothing
    python run_timing_tests.py                      # everything (cluster: run_job_timing_tests.bash)

Output (``--out_dir`` default ``results/timing_tests_<tag>``): timing_components.csv,
timing_missions_rounds.csv, timing_missions_summary.csv, hardware.json, timing_overview.png.
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import types

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

import sample_annealed_mission as sam                                       # noqa: E402
import sample_annealed_mission_overview as ov                               # noqa: E402
import apply_cfm_belief as acb                                              # noqa: E402
import run_exploration_comparison as rec                                    # noqa: E402
import run_svgd_convergence as rsc                                          # noqa: E402
import variant_runner as vr                                                 # noqa: E402
import mission_baselines as mb                                              # noqa: E402
from common.acquisition import kappa_schedule                               # noqa: E402
from common.data import load_truth                                          # noqa: E402
from common.observation import measure, thin                                # noqa: E402
from init_baselines import linear_ray_path, random_walk_path                # noqa: E402
from ergodic_energy_torch import coeffs_from_points                         # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
TEXT, MUTED, SPINE = '#1A1A2E', '#555555', '#CCCCCC'
PARTS = ('planner', 'init', 'refiner', 'belief', 'missions')
TIMING_SHAPES = ('A', 'rand_gmm_20', 'organic_20')     # PILOT_SHAPES of run_ideal_matrix


# ===========================================================================
# Helpers
# ===========================================================================

def sync(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_call(fn, device, reps, warmup):
    """Seconds of `reps` calls of fn() after `warmup` discarded calls (GPU-synchronised)."""
    for _ in range(warmup):
        fn()
    sync(device)
    out = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.append(time.perf_counter() - t0)
    return out


def quartiles(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))


def hardware_info(device):
    info = dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), host=platform.node(),
                platform=platform.platform(), processor=platform.processor(),
                cpu_count=os.cpu_count(), python=sys.version.split()[0], torch=torch.__version__,
                numpy=np.__version__, device=str(device))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_mem_gb=round(props.total_memory / 2 ** 30, 1),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    try:
        info['git_rev'] = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=_here, stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        info['git_rev'] = None
    return info


class Writer:
    """Appends rows to timing_components.csv (one flat table: part, variant, x-name, x, stats)."""
    FIELDS = ['part', 'variant', 'param', 'value', 'median_s', 'q25_s', 'q75_s', 'n', 'extra']

    def __init__(self, path):
        self.f = open(path, 'w', newline='', encoding='utf-8')
        self.w = csv.writer(self.f)
        self.w.writerow(self.FIELDS)
        self.rows = []

    def add(self, part, variant, param, value, times, extra=''):
        med, q1, q3 = quartiles(times)
        row = [part, variant, param, value, f'{med:.6g}', f'{q1:.6g}', f'{q3:.6g}', len(times), extra]
        self.w.writerow(row)
        self.f.flush()
        self.rows.append(dict(zip(self.FIELDS, row)))
        print(f'  [{part}] {variant:<22} {param}={value:<6} {med * 1e3:9.2f} ms '
              f'[{q1 * 1e3:.2f}, {q3 * 1e3:.2f}]  {extra}')

    def close(self):
        self.f.close()


# ===========================================================================
# Parts
# ===========================================================================

def part_planner(wr, a, ctx, truth, device):
    """t_init of every network vs. n_init."""
    phi = truth.to(device).float()
    args = types.SimpleNamespace(**vars(ctx.base_args))
    start = torch.tensor([0.5, 0.5], device=device)
    for method in a.net_methods:
        try:
            planner = rec.build_planner(method, a, device)
        except (SystemExit, FileNotFoundError) as e:
            print(f'  [planner] {method}: skipped ({str(e).splitlines()[0]})')
            continue
        parts = acb.phi_particles(phi, args.n_particles, mode=args.phi_mode,
                                  quantile=args.phi_quantile, device=device)
        planner.cfg_weight = 2.0
        for n_init in a.n_init_sweep:
            def call():
                with torch.no_grad():
                    cps = planner.plan(parts, n_candidates=n_init, start=start)
                    planner.render(cps)
            if device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats()
            times = time_call(call, device, a.reps, a.warmup)
            mem = (f'peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB'
                   if device.startswith('cuda') else '')
            wr.add('planner', method, 'n_init', n_init, times, mem)
        del planner
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def part_init(wr, a, ctx, truth, device):
    """t_init of the network-free candidate sources."""
    phi = truth.to(device).float()
    start = torch.tensor([0.5, 0.5])
    ee = ctx.ee
    walk_len = a.walk_len
    gp_fn = mb.gp_goal_path_fn(walk_len, 0)
    lawn_fn = ov.lawnmower_round_fn(walk_len, a.n_rounds)
    mu = phi.cpu()
    for n_init in a.n_init_sweep:
        def rays():
            torch.stack([linear_ray_path(start.numpy(), 360.0 * i / n_init, walk_len, rsc.N_POINTS)
                         for i in range(n_init)])

        def walks():
            for i in range(n_init):
                random_walk_path(n_points=2048, seed=i, start=start)

        def best_ray():
            mb.best_ray_path_fn(ee, n_init, walk_len, rsc.N_POINTS)(0, start, phi)
        wr.add('init', 'straight rays (lintsvec / linadam)', 'n_init', n_init,
               time_call(rays, device, a.reps, a.warmup))
        wr.add('init', 'random-walk candidates', 'n_init', n_init,
               time_call(walks, device, a.reps, a.warmup))
        wr.add('init', 'best ray (linray, incl. E)', 'n_init', n_init,
               time_call(best_ray, device, a.reps, a.warmup))
    # single-path baselines do not depend on n_init
    wr.add('init', 'random walk (rw)', 'n_init', 1, time_call(
        lambda: ov.baseline_path_fn(types.SimpleNamespace(net='rw'), a, 0)(0, start, phi),
        device, a.reps, a.warmup))
    wr.add('init', 'lawnmower round (lawn)', 'n_init', 1, time_call(
        lambda: lawn_fn(0, start, phi), device, a.reps, a.warmup))
    wr.add('init', 'line to GP-sampled point (gpgoal)', 'n_init', 1, time_call(
        lambda: gp_fn(0, start, phi, mu=mu), device, a.reps, a.warmup))


def part_refiner(wr, a, ctx, truth, device):
    """SVGD refinement time vs. n_init and n_iters; reported per candidate-iteration too."""
    phi = truth.to(device).float()
    ee = ctx.ee
    start = np.array([0.5, 0.5])
    backends = {'tsvec_svgd': (ctx.bs_tsvec, lambda p: ee.target_coeffs(p).cpu().numpy().astype(np.float64)),
                'adam_svgd': (ctx.bs_adam, lambda p: ee.target_coeffs(p).cpu().numpy().astype(np.float64))}
    if 'sun' in a.refiners:
        try:
            from svgd_batched import BatchedSunTorch
            backends['sun'] = (BatchedSunTorch(ctx.B_np, device), lambda p: p.cpu().numpy())
        except Exception as e:      # JAX missing etc.
            print(f'  [refiner] sun: skipped ({e})')
    for name, (bs, target_fn) in backends.items():
        if name not in a.refiners:
            continue
        for n_init in a.n_init_sweep:
            curves = np.stack([linear_ray_path(start, 360.0 * i / n_init, a.walk_len, rsc.N_POINTS).numpy()
                               for i in range(n_init)]).astype(np.float64)
            target = target_fn(phi)
            starts = np.tile(start, (n_init, 1))
            seeds = list(range(n_init))
            for n_iters in a.iters_sweep:
                call = lambda: bs.run(curves, target, starts, seeds, n_iters, record=False)   # noqa: E731
                times = time_call(call, device, a.reps_slow, a.warmup_slow)
                per_it = np.array(times) / n_iters
                wr.add('refiner', name, f'n_init={n_init},n_iters', n_iters, times,
                       f'{np.median(per_it) * 1e3:.3f} ms/iter, '
                       f'{np.median(per_it) / n_init * 1e6:.1f} us/cand-iter')


def _make_belief(kernel, truth, device, np_ckpt):
    return vr.build_belief(sam.CONDITION, truth, seed=0, device=device, gp_noise=0.05,
                           gp_lengthscale=0.08, dtype=torch.float64, kernel=kernel,
                           ak_fit_iters=a_global.ak_fit_iters, np_ckpt=np_ckpt)


a_global = None


def part_belief(wr, a, ctx, truth, device):
    """One belief update (observe a new segment) + posterior on the grid, vs. stored measurements."""
    seg_pts = torch.rand(120, 2).mul(0.9).add(0.05)
    for kernel in a.belief_kernels:
        for n_prev in a.nobs_sweep:
            def setup():
                b = _make_belief(kernel, truth.to(device), device, a.np_ckpt)
                if n_prev > 0:
                    p, v = measure(torch.rand(max(2, n_prev), 2, device=device).mul(0.9).add(0.05),
                                   truth.to(device), noise_std=0.05, sensor_radius=0.06)
                    b.observe(*thin(p, v, max_points=n_prev))
                return b

            times_obs, times_post = [], []
            try:
                for rep in range(a.reps + a.warmup):
                    b = setup()
                    pts, vals = measure(seg_pts.to(device), truth.to(device), noise_std=0.05,
                                        sensor_radius=0.06)
                    pts, vals = thin(pts, vals, max_points=64)
                    sync(device)
                    t0 = time.perf_counter()
                    b.observe(pts, vals)
                    sync(device)
                    t1 = time.perf_counter()
                    b.posterior_grid()
                    sync(device)
                    t2 = time.perf_counter()
                    b._cache = None
                    if rep >= a.warmup:
                        times_obs.append(t1 - t0)
                        times_post.append(t2 - t1)
            except Exception as e:
                print(f'  [belief] {kernel}: skipped ({type(e).__name__}: {str(e)[:100]})')
                break
            wr.add('belief', f'{kernel} observe(64 pts)', 'n_obs_before', n_prev, times_obs)
            wr.add('belief', f'{kernel} posterior grid', 'n_obs_before', n_prev, times_post)


def part_missions(wr_dir, a, ctx, device):
    """One full mission per method x kernel on a few shapes: stage times per round + quality."""
    names, truths = load_truth(labels=list(a.shapes), n=len(a.shapes), split='val',
                               resolution=rsc.TRUTH_RES, device=device)
    kappas = [kappa_schedule(r, a.n_rounds, a.kappa0, a.kappa1) for r in range(a.n_rounds)]
    rows_path = os.path.join(wr_dir, 'timing_missions_rounds.csv')
    cols = ['method', 'kernel', 'shape', 'rep', 'round', 't_target', 't_init', 't_refine', 't_belief',
            't_round', 'E_truth', 'swept06', 'path_len']
    rows = []
    for method in a.methods:
        try:
            planner = rec.build_planner(method, a, device)
        except (SystemExit, FileNotFoundError) as e:
            print(f'  [missions] {method}: skipped ({str(e).splitlines()[0]})')
            continue
        bs = ctx.refiner_for(method)
        # one discarded warm-up mission (CUDA context, cudnn autotune, first allocations)
        warm = types.SimpleNamespace(**vars(a))
        warm.n_rounds, warm.n_iters = 1, min(a.n_iters, 20)
        rec.run_one(method, a.kernels[0], names[0], truths[0].detach().cpu(), planner, bs, ctx.B32,
                    ctx.ee, ctx.base_args, warm, kappas[:1], device, 0)
        for kernel in a.kernels:
            for si, (name, truth) in enumerate(zip(names, truths)):
                for rep in range(a.mission_reps):
                    res = rec.run_one(method, kernel, name, truth.detach().cpu(), planner, bs, ctx.B32,
                                      ctx.ee, ctx.base_args, a, kappas, device, si * 100 + rep)
                    for r, row in enumerate(res['rows']):
                        rows.append(dict(method=method, kernel=kernel, shape=name, rep=rep, round=r,
                                         **{c: row.get(c) for c in cols[5:]}))
                    tot = sum(rw['t_round'] for rw in res['rows'])
                    print(f'  [missions] {method:<9} {kernel:<3} {name:<14} rep{rep}: '
                          f'{tot:6.1f} s ({tot / len(res["rows"]):.2f} s/round)  '
                          f'E_truth={res["rows"][-1]["E_truth"]:.2f}')
        del planner
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    with open(rows_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    # per method x kernel summary: seconds per mission (sum over rounds) and per stage
    sum_path = os.path.join(wr_dir, 'timing_missions_summary.csv')
    keys = sorted({(r['method'], r['kernel']) for r in rows})
    with open(sum_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'kernel', 'n_missions', 's_per_mission_median', 's_per_mission_q25',
                    's_per_mission_q75', 's_target', 's_init', 's_refine', 's_belief',
                    'E_truth_final_mean', 'swept06_final_mean'])
        for m, k in keys:
            sel = [r for r in rows if r['method'] == m and r['kernel'] == k]
            runs = sorted({(r['shape'], r['rep']) for r in sel})
            per = []
            for sh, rp in runs:
                per.append(sum(r['t_round'] for r in sel if r['shape'] == sh and r['rep'] == rp))
            med, q1, q3 = quartiles(per)
            stage = {c: float(np.mean([sum(r[c] for r in sel if r['shape'] == sh and r['rep'] == rp)
                                       for sh, rp in runs])) for c in
                     ('t_target', 't_init', 't_refine', 't_belief')}
            last = [r for r in sel if r['round'] == a.n_rounds - 1]
            w.writerow([m, k, len(runs), f'{med:.3f}', f'{q1:.3f}', f'{q3:.3f}'] +
                       [f'{stage[c]:.3f}' for c in ('t_target', 't_init', 't_refine', 't_belief')] +
                       [f'{np.mean([r["E_truth"] for r in last]):.3f}',
                        f'{np.mean([r["swept06"] for r in last]):.4f}'])
    return rows


# ===========================================================================
# Plot
# ===========================================================================

def style(ax):
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(SPINE)


def plot_overview(out_dir, comp_rows, mission_rows, hw):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), facecolor='white')
    ax = axes[0, 0]
    for v in sorted({r['variant'] for r in comp_rows if r['part'] == 'planner'}):
        pts = [(float(r['value']), float(r['median_s'])) for r in comp_rows
               if r['part'] == 'planner' and r['variant'] == v]
        col = rec.METHOD_COLOR.get(v, None)
        ax.plot(*zip(*pts), marker='o', lw=2, color=col, label=rec.METHOD_LABEL.get(v, v))
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('candidates per plan (n_init)', color=MUTED); ax.set_ylabel('seconds per plan()', color=MUTED)
    ax.set_title('Network forward pass (t_init)', fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False); style(ax)

    ax = axes[0, 1]
    for v in sorted({r['variant'] for r in comp_rows if r['part'] == 'init'}):
        pts = [(float(r['value']), float(r['median_s'])) for r in comp_rows
               if r['part'] == 'init' and r['variant'] == v]
        ax.plot(*zip(*pts), marker='o', lw=2, label=v) if len(pts) > 1 else ax.axhline(
            pts[0][1], lw=1.5, ls='--', label=v)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('candidates per plan (n_init)', color=MUTED); ax.set_ylabel('seconds', color=MUTED)
    ax.set_title('Network-free candidate sources (t_init)', fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False); style(ax)

    ax = axes[0, 2]
    for v in sorted({r['variant'] for r in comp_rows if r['part'] == 'refiner'}):
        for n_init in sorted({r['param'] for r in comp_rows if r['part'] == 'refiner' and r['variant'] == v}):
            pts = [(float(r['value']), float(r['median_s'])) for r in comp_rows
                   if r['part'] == 'refiner' and r['variant'] == v and r['param'] == n_init]
            ax.plot(*zip(*pts), marker='o', lw=1.8, label=f'{v}, {n_init.split(",")[0]}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('SVGD iterations', color=MUTED); ax.set_ylabel('seconds per refinement call', color=MUTED)
    ax.set_title('SVGD refinement (t_refine)', fontsize=10, color=TEXT)
    ax.legend(fontsize=6.5, frameon=False, ncol=2); style(ax)

    ax = axes[1, 0]
    for v in sorted({r['variant'] for r in comp_rows if r['part'] == 'belief'}):
        pts = [(float(r['value']), float(r['median_s'])) for r in comp_rows
               if r['part'] == 'belief' and r['variant'] == v]
        ax.plot(*zip(*pts), marker='o', lw=1.8, ls='--' if 'posterior' in v else '-', label=v)
    ax.set_yscale('log')
    ax.set_xlabel('measurements stored before the update', color=MUTED); ax.set_ylabel('seconds', color=MUTED)
    ax.set_title('Belief update (t_belief) and posterior grid (t_target)', fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False); style(ax)

    ax = axes[1, 1]
    if mission_rows:
        keys = sorted({(r['method'], r['kernel']) for r in mission_rows})
        stages = [('t_target', '#B0BEC5'), ('t_init', '#1565C0'), ('t_refine', '#E65100'), ('t_belief', '#00897B')]
        for ki, kernel in enumerate(sorted({k for _, k in keys})):
            ms = [m for m, k in keys if k == kernel]
            x = np.arange(len(ms)) + ki * (len(ms) + 1)
            bottom = np.zeros(len(ms))
            for st, col in stages:
                vals = np.array([np.mean([sum(r[st] for r in mission_rows if r['method'] == m and r['kernel'] == kernel
                                              and r['shape'] == sh and r['rep'] == rp)
                                          for sh, rp in {(r['shape'], r['rep']) for r in mission_rows
                                                         if r['method'] == m and r['kernel'] == kernel}]) for m in ms])
                ax.bar(x, vals, bottom=bottom, color=col, label=st if ki == 0 else None, width=0.8)
                bottom += vals
            ax.set_xticks(x)
            ax.set_xticklabels([f'{m}\n({kernel})' for m in ms], rotation=60, fontsize=6.5, ha='right')
    ax.set_ylabel('seconds per mission (all rounds)', color=MUTED)
    ax.set_title('Full mission: where the time goes', fontsize=10, color=TEXT)
    ax.legend(fontsize=7, frameon=False); style(ax)

    ax = axes[1, 2]
    ax.axis('off')
    txt = '\n'.join(f'{k}: {v}' for k, v in hw.items() if k in
                    ('gpu', 'gpu_mem_gb', 'cuda', 'torch', 'processor', 'cpu_count', 'host', 'date', 'git_rev'))
    ax.text(0.0, 1.0, 'Hardware / software\n\n' + txt, va='top', fontsize=9, color=TEXT, family='monospace')
    fig.suptitle('Compute time of the active-sensing pipeline (median of repetitions)', fontsize=13, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, 'timing_overview.png')
    fig.savefig(path, dpi=120, facecolor='white')
    plt.close(fig)
    print(f'  figure: {path}')


# ===========================================================================

def main():
    global a_global
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--parts', type=str, default=','.join(PARTS))
    p.add_argument('--tag', type=str, default=time.strftime('%Y%m%d'))
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--shapes', type=str, default=','.join(TIMING_SHAPES),
                   help='Shapes of the mission timing (and the density used by the component tests: first one).')
    p.add_argument('--net_methods', type=str, default='unet630,dit630,old500,unet0,unet1890,lm1300,selfsup',
                   help='Networks of the planner part; a network whose checkpoint is missing is skipped.')
    p.add_argument('--methods', type=str, default=','.join(rec.METHODS),
                   help='Methods of the mission part.')
    p.add_argument('--kernels', type=str, default='rbf,ak')
    p.add_argument('--belief_kernels', type=str, default='rbf,ak,np',
                   help='Belief models of the belief part (np needs its checkpoint).')
    p.add_argument('--refiners', type=str, default='tsvec_svgd,adam_svgd',
                   help="Add 'sun' to time the FM-Stein refiner as well (needs JAX).")
    p.add_argument('--n_init_sweep', type=str, default='1,2,4,8,16,30,60')
    p.add_argument('--iters_sweep', type=str, default='10,50,200')
    p.add_argument('--nobs_sweep', type=str, default='0,64,128,256,512')
    p.add_argument('--reps', type=int, default=15)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--reps_slow', type=int, default=3, help='Repetitions of the SVGD calls.')
    p.add_argument('--warmup_slow', type=int, default=1)
    p.add_argument('--mission_reps', type=int, default=2, help='Repetitions per (method, kernel, shape).')
    p.add_argument('--n_rounds', type=int, default=5)
    p.add_argument('--n_init', type=int, default=8)
    p.add_argument('--n_iters', type=int, default=500)
    p.add_argument('--kappa0', type=float, default=1.0)
    p.add_argument('--kappa1', type=float, default=0.05)
    p.add_argument('--walk_len', type=float, default=6.1)
    p.add_argument('--ak_fit_iters', type=int, default=200)
    p.add_argument('--selfsup_ckpt', type=str, default=None)
    p.add_argument('--np_ckpt', type=str, default=None)
    p.add_argument('--gp_goal_source', type=str, default='mean', choices=['mean', 'phi'])
    p.add_argument('--smoke', action='store_true', help='Tiny sizes, separate folder (mechanics only).')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.smoke:
        a.parts = 'planner,init,refiner,belief,missions'
        a.n_init_sweep, a.iters_sweep, a.nobs_sweep = '1,4', '5', '0,64'
        a.reps, a.warmup, a.reps_slow, a.warmup_slow, a.mission_reps = 2, 1, 1, 0, 1
        a.n_rounds, a.n_init, a.n_iters, a.ak_fit_iters = 2, 2, 10, 20
        a.shapes = TIMING_SHAPES[0]
        a.kernels = 'rbf'
        a.tag = 'smoke'
    for k in ('parts', 'net_methods', 'methods', 'kernels', 'belief_kernels', 'refiners', 'shapes'):
        setattr(a, k, [x for x in getattr(a, k).split(',') if x])
    for k in ('n_init_sweep', 'iters_sweep', 'nobs_sweep'):
        setattr(a, k, [int(x) for x in getattr(a, k).split(',') if x])
    for part in a.parts:
        if part not in PARTS:
            raise SystemExit(f'unknown part {part}; known: {PARTS}')
    for m in a.methods + a.net_methods:
        if m not in rec.METHODS:
            raise SystemExit(f'unknown method {m}; known: {rec.METHODS}')
    a_global = a
    out_dir = a.out_dir or os.path.join(_here, 'results', f'timing_tests_{a.tag}')
    device = a.device

    print(f'timing tests -> {out_dir}\n  parts={a.parts}  device={device}\n'
          f'  planner: {len(a.net_methods)} nets x n_init {a.n_init_sweep}\n'
          f'  refiner: {a.refiners} x n_init {a.n_init_sweep} x n_iters {a.iters_sweep}\n'
          f'  belief:  {a.belief_kernels} x n_obs {a.nobs_sweep}\n'
          f'  missions: {len(a.methods)} methods x {len(a.kernels)} kernels x {len(a.shapes)} shapes x '
          f'{a.mission_reps} reps, {a.n_rounds} rounds, n_init={a.n_init}, n_iters={a.n_iters}')
    if a.dry_run:
        return
    os.makedirs(out_dir, exist_ok=True)
    hw = hardware_info(device)
    with open(os.path.join(out_dir, 'hardware.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(hardware=hw, args={k: v for k, v in vars(a).items()}), f, indent=1)
    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = False      # fixed algorithm choice, comparable timings

    names, truths = load_truth(labels=a.shapes[:1], n=1, split='val', resolution=rsc.TRUTH_RES, device=device)
    truth = truths[0].detach().cpu()
    ctx = rec.make_context(device)
    wr = Writer(os.path.join(out_dir, 'timing_components.csv'))
    mission_rows = []
    t0 = time.time()
    if 'planner' in a.parts:
        print('\n=== planner forward pass ===')
        part_planner(wr, a, ctx, truth, device)
    if 'init' in a.parts:
        print('\n=== network-free candidate sources ===')
        part_init(wr, a, ctx, truth, device)
    if 'refiner' in a.parts:
        print('\n=== SVGD refinement ===')
        part_refiner(wr, a, ctx, truth, device)
    if 'belief' in a.parts:
        print('\n=== belief update ===')
        part_belief(wr, a, ctx, truth, device)
    wr.close()
    if 'missions' in a.parts:
        print('\n=== full missions ===')
        mission_rows = part_missions(out_dir, a, ctx, device)
    plot_overview(out_dir, wr.rows, mission_rows, hw)
    print(f'\ndone in {(time.time() - t0) / 60:.1f} min -> {out_dir}')


if __name__ == '__main__':
    main()
