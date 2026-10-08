r"""
rq5_dynamics.py
================
RQ5 (Uebertragbarkeit auf andere Dynamiken, spiegelt Sun et al. Fig. 11): ein
Vorwaertsdurchlauf des CFM-Netzes ist dynamik-agnostisch (eine xy-Kurve), aber
jede der sechs Dynamiken (siehe `dynamics_zoo.py`) braucht ihre eigene
Steuerfolge. Zwei Produkte:

1. Ein qualitatives Raster wie Fig. 11, hier mit DREI Zeilen statt einer:
   Sun kalt / CFM roh (per Dynamik simuliert) / CFM + FM-Stein-Warm-Start --
   fuer EIN festes Trial (Suns eigenes Protokoll fuer Fig. 11: festes GMM,
   fester Start).
2. Ein quantitativer Balken (Iterationen bis zum Ziel-Niveau je Dynamik,
   Sun kalt vs. CFM-Warm-Start), gemittelt ueber `--n_trials` zufaellige
   Trials (Suns Q1-Protokoll).

Usage:
    python rq5_dynamics.py --n_trials 3 --seed0 0 --out_tag smoke
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
for _p in (_here, _mat):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dynamics_zoo as dz                                          # noqa: E402
import exp_common as C                                             # noqa: E402
import paper_style as ps                                           # noqa: E402

DT, TSTEPS = 0.05, 200
CHECKPOINTS = [0, 10, 25, 50, 100, 200, 400]
SWEEP_FIELDS = ['trial', 'dynamics', 'method', 'iters', 'E']


def _load_done_trials(csv_path):
    """Resumability for cluster jobs (24h limit): one trial = one block of
    rows (all dynamics x both methods x all checkpoints), written together
    below, so a trial index already on disk is skipped on restart."""
    done = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                done.add(int(row['trial']))
    return done


def qualitative_grid(trial, planner, device, out_path):
    """Fig.-11-Stil, 3 Zeilen x 6 Dynamiken, EIN Trial."""
    phi = C.density_grid(trial['pdf_fn'], res=96)
    curve, _ = C.cfm_raw_curve(planner, phi, start=trial['x0'], device=device)

    names = list(dz.DYNAMICS)
    row_sun, row_raw, row_warm = [], [], []
    for name in names:
        cls = dz.DYNAMICS[name]
        pm, lin, solve = dz.build_for_curve(cls, curve, DT, TSTEPS)

        x0c, u0c = dz.cold_x0_u0(pm, trial['x0'], DT, TSTEPS)
        sun_cold = C.timed_solve(pm, lin, solve, trial['score_fn'], x0c, u0c,
                                 checkpoints=[TSTEPS], warmup=False)[0]['traj_xy']

        x0r, u0r = pm.flat_map(curve, DT, TSTEPS)
        cfm_raw = pm.positions(pm.traj_sim(x0r, u0r))

        cfm_warm = C.timed_solve(pm, lin, solve, trial['score_fn'], x0r, u0r,
                                 checkpoints=[100], warmup=False)[0]['traj_xy']

        title = name + ('*' if name in dz.FROM_SUN_REPO else '')
        row_sun.append(dict(phi=phi, trajs=[sun_cold], start=trial['x0'], title=title))
        row_raw.append(dict(phi=phi, trajs=[cfm_raw], start=trial['x0']))
        row_warm.append(dict(phi=phi, trajs=[cfm_warm], start=trial['x0']))

    ps.trajectory_panel(
        [row_sun, row_raw, row_warm], out_path, ncols=len(names),
        row_titles=['Sun (cold)', 'CFM (raw)', 'CFM + FM-Stein (100 it.)'],
        suptitle='Six dynamics, one target distribution (cf. Sun et al. Fig. 11; '
                 '* = dynamics already in the Sun repo)')
    return out_path


def iters_to_eps(pm, lin, solve, trial, x0, u0, phik, eps, checkpoints):
    for r in C.timed_solve(pm, lin, solve, trial['score_fn'], x0, u0, checkpoints, warmup=False):
        if C.ergodic_error(r['traj_xy'], phik) <= eps:
            return r['iters']
    return None   # censored: nicht erreicht


def quantitative_sweep(n_trials, seed0, planner, device, out_dir, eps_quantile=0.5,
                       time_budget_h=None):
    """Iterationen bis Epsilon je Dynamik, Sun kalt vs. CFM-Warm-Start,
    gemittelt ueber `n_trials` zufaellige Trials. Epsilon je Dynamik einzeln
    (relativ zur Ground Truth, wie in RQ1): Median von Sun-kalt bei
    `max(CHECKPOINTS)` ueber alle Trials dieser Dynamik."""
    names = list(dz.DYNAMICS)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'rq5_dynamics_sweep.csv')
    done = _load_done_trials(csv_path)
    need_header = not os.path.isfile(csv_path)
    t0 = time.time()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        if need_header:
            w.writeheader()
        for i in range(n_trials):
            if C.time_budget_exceeded(t0, time_budget_h):
                print(f"[rq5] time budget reached before trial {i}, stopping.", flush=True)
                break
            if i in done:
                continue
            seed = seed0 + i
            trial = C.make_trial(seed=seed)
            phi = C.density_grid(trial['pdf_fn'], res=96)
            phik = C.target_coeffs(phi)
            curve, _ = C.cfm_raw_curve(planner, phi, start=trial['x0'], device=device)
            for name in names:
                cls = dz.DYNAMICS[name]
                pm, lin, solve = dz.build_for_curve(cls, curve, DT, TSTEPS)
                x0c, u0c = dz.cold_x0_u0(pm, trial['x0'], DT, TSTEPS)
                x0r, u0r = pm.flat_map(curve, DT, TSTEPS)
                res_cold = C.timed_solve(pm, lin, solve, trial['score_fn'], x0c, u0c,
                                         checkpoints=CHECKPOINTS, warmup=False)
                res_warm = C.timed_solve(pm, lin, solve, trial['score_fn'], x0r, u0r,
                                         checkpoints=CHECKPOINTS, warmup=False)
                for r in res_cold:
                    w.writerow(dict(trial=i, dynamics=name, method='sun_cold',
                                    iters=r['iters'], E=C.ergodic_error(r['traj_xy'], phik)))
                for r in res_warm:
                    w.writerow(dict(trial=i, dynamics=name, method='cfm_warm',
                                    iters=r['iters'], E=C.ergodic_error(r['traj_xy'], phik)))
            f.flush()
            print(f"[rq5] trial {i + 1}/{n_trials} done", flush=True)

    import pandas as pd
    df = pd.read_csv(csv_path)
    bars = np.full((len(names), 2), np.nan)
    for j, name in enumerate(names):
        sub = df[df['dynamics'] == name]
        eps = sub[(sub['method'] == 'sun_cold') & (sub['iters'] == max(CHECKPOINTS))]['E'].quantile(eps_quantile)
        for k, m in enumerate(['sun_cold', 'cfm_warm']):
            vals = []
            for _, g in sub[sub['method'] == m].groupby('trial'):
                g = g.sort_values('iters')
                below = g[g['E'] <= eps]
                vals.append(float(below['iters'].iloc[0]) if len(below) else np.nan)
            bars[j, k] = np.nanmedian(vals) if not all(np.isnan(vals)) else np.nan

    plot_path = os.path.join(out_dir, 'plots', 'rq5_iters_to_eps.png')
    ps.grouped_bars(bars, 'Iterations to reach a per-dynamics target ergodic error',
                    'Median iterations to target (NaN = not reached)', plot_path,
                    group_labels=names, series_labels=['Sun (cold)', 'CFM (warm start)'],
                    series_colors=[ps.METHOD_COLORS['sun_cold'], ps.METHOD_COLORS['cfm_fmstein']])
    return csv_path, plot_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n_trials', type=int, default=5,
                    help='Fuer den quantitativen Balken (Iterationen bis Epsilon).')
    ap.add_argument('--seed0', type=int, default=0,
                    help='Trial 0 des quantitativen Sweeps = Trial des qualitativen Rasters.')
    ap.add_argument('--skip_quant', action='store_true',
                    help='Nur das qualitative Raster (schneller Smoke-Test).')
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop starting new trials after this many hours; already-written '
                         'rows stay valid, re-run with the same --out_tag to resume.')
    args = ap.parse_args()

    import torch
    from run_eval_matrix import DEFAULT_CKPT
    from run_ideal_matrix import build_particle_planner
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    planner = build_particle_planner(args.ckpt or DEFAULT_CKPT, device)
    warm_phi = C.density_grid(C.make_trial(seed=2 ** 31 - 1)['pdf_fn'], res=96)
    C.cfm_raw_curve(planner, warm_phi, start=(0.5, 0.5), device=device)

    out_dir = os.path.join(_mat, 'results', args.out_tag)
    trial0 = C.make_trial(seed=args.seed0)
    grid_path = qualitative_grid(trial0, planner, device,
                                os.path.join(out_dir, 'plots', 'rq5_qualitative_grid.png'))
    print(f"[rq5] qualitative grid -> {grid_path}")

    if not args.skip_quant:
        csv_path, plot_path = quantitative_sweep(args.n_trials, args.seed0, planner, device, out_dir,
                                                 time_budget_h=args.time_budget_h)
        print(f"[rq5] sweep -> {csv_path}\n[rq5] bars -> {plot_path}")


if __name__ == '__main__':
    main()
