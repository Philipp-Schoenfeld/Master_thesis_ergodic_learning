"""
rq3_gp_targets.py
==================
RQ3 (unknown targets as a GP belief, mirrors Sun et al. Fig. 6): a
replanning mission under partial knowledge (`half_known`/`ten_samples`/
`none_known`), comparing Sun's FM-Stein solver against CFM as the per-round
candidate generator, both driven through the SAME belief-update loop.

This is new infrastructure, not a reuse of `run_mission_eval.py`: that
runner is CFM-only and batches whole shape sets on the GPU for throughput.
Here the loop is per-trial and much simpler (no batching across shapes), so
that Sun's FM-Stein solver -- for which no batched implementation exists --
can sit in the same slot as CFM's forward pass. Shared building blocks are
reused: `variant_runner.build_belief`/`phi_from_belief`/`needs_belief_update`
(the project's own knowledge-condition machinery), `exploration_optimierung
.mission.LENGTH_UNIT`/`resample_arclength`, `common.observation.measure`/
`thin`, `common.metrics.trim_to_length`/`path_length`,
`metrics_explore_exploit.swept_mass_fraction` (the natural analogue of
Sun's "collected positive life signals" for a continuous target density
instead of a binary signal grid -- see the thesis doc's RQ3 section for why
this substitution is made explicit rather than silently reused), and
`common.sun_refine.make_grid_score` (new, see its docstring) for turning the
belief's density grid into a JAX score FM-Stein can descend.

Per round:
    1. phi = current belief's target density (`variant_runner.phi_from_belief`)
    2. a candidate curve from the agent's current position:
         sun       straight-line start toward the density centroid, refined
                   with FM-Stein against `make_grid_score(phi)` for
                   `--sun_iters` iterations
         cfm_raw   one CFM forward pass (start-conditioned)
         cfm_svgd  the CFM curve refined with `SvgdRefiner` (Sun's solver on
                   control points, same backend as the data generator)
    3. drive exactly one length unit, observe, update the belief
    4. repeat until `swept_mass_fraction(driven_path, truth) >= 0.99` or
       `--max_rounds`

Usage:
    python rq3_gp_targets.py --n_trials 3 --conditions half_known --out_tag smoke
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
_arch = os.path.dirname(_mat)
for _p in (_here, _mat, os.path.join(_arch, 'exploration'),
          os.path.join(_arch, 'ergodic_dataset_generator')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import ergodic_solver as es                                        # noqa: E402
from common.sun_refine import make_grid_score, run_batch as sun_run_batch  # noqa: E402
import exp_common as C                                             # noqa: E402
import paper_style as ps                                           # noqa: E402

DT, TSTEPS = 0.05, 200
SUN_RES = 64   # belief grid resolution the FM-Stein score is built on


FIELDS = ['trial', 'condition', 'method', 'round', 'swept_mass', 'path_len', 'reached99']


def _load_done(csv_path):
    """Resumability for cluster jobs (24h limit): a (trial, condition, method)
    mission is atomic in `run_trial` (all its rounds land in one call), so it
    is "done" once ANY row for that key is on disk -- same granularity as
    `rq1_rq2_runtime.py`'s own resume check."""
    done = set()
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                done.add((int(row['trial']), row['condition'], row['method']))
    return done


def _centroid(phi):
    R = phi.shape[0]
    xs = np.linspace(0, 1, R)
    X, Y = np.meshgrid(xs, xs)
    w = np.clip(phi, 0.0, None)
    s = w.sum()
    if s < 1e-9:
        return np.array([0.5, 0.5])
    return np.array([(X * w).sum() / s, (Y * w).sum() / s])


def sun_round_curve(pos, phi_np, n_iters):
    """A straight line from `pos` toward the belief's density centroid,
    refined with FM-Stein against `phi_np`'s score. Deliberately simple (no
    heuristic sweep like `ergodic_solver`'s GMM path): this is a per-round
    LOCAL plan, not the single long global trajectory the data generator
    builds, so there is no natural multi-component shape to route through."""
    import jax.numpy as jnp
    target = _centroid(phi_np)
    score_fn = make_grid_score(phi_np)
    pm, linearize_dyn, solve_lqr = es._build_lqr(DT)
    pm.positions = lambda x_traj: np.array(x_traj)[:, :2]
    # Initial velocity aimed at the density centroid, Sun's own "cold" recipe
    # (`ergodic_solver.run_ergodic_coverage`'s default branch) adapted to an
    # arbitrary target point instead of the fixed centre (0.5, 0.5).
    T = DT * TSTEPS
    x0 = jnp.array([pos[0], pos[1], 2.0 * (target[0] - pos[0]) / T,
                   2.0 * (target[1] - pos[1]) / T])
    u0 = jnp.zeros((TSTEPS, 2))
    res = C.timed_solve(pm, linearize_dyn, solve_lqr, score_fn, x0, u0,
                        checkpoints=[n_iters], warmup=False)[0]
    return torch.as_tensor(res['traj_xy'], dtype=torch.float32).clamp(0.0, 1.0)


def run_trial(trial_seed, condition, method, planner, device, args):
    """One mission (one shape/belief, one method) -> list of per-round rows."""
    import variant_runner as vr
    from common.metrics import trim_to_length, path_length
    from common.observation import measure, thin
    from exploration_optimierung.mission import LENGTH_UNIT, PTS_PER_UNIT, resample_arclength
    from metrics_explore_exploit import swept_mass_fraction

    trial = C.make_trial(seed=trial_seed)
    truth_np = C.density_grid(trial['pdf_fn'], res=SUN_RES)
    truth_np = truth_np / max(truth_np.max(), 1e-12)
    truth = torch.as_tensor(truth_np, dtype=torch.float32, device=device)

    belief = vr.build_belief(condition, truth, seed=trial_seed, device=device)
    pos = np.array(trial['x0'], dtype=np.float64)
    driven = None
    rows = []
    for r in range(args.max_rounds):
        phi_t = vr.phi_from_belief(belief, norm='max')
        phi_np = phi_t.detach().cpu().numpy().astype(np.float64)

        if method == 'sun':
            curve = sun_round_curve(pos, phi_np, args.sun_iters)
        elif method == 'cfm_raw':
            c_np, _ = C.cfm_raw_curve(planner, phi_np, start=pos, device=device)
            curve = torch.as_tensor(c_np, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        elif method == 'cfm_svgd':
            c_np, _ = C.cfm_raw_curve(planner, phi_np, start=pos, device=device)
            refined = sun_run_batch(c_np[None], phi_np, np.asarray(pos)[None],
                                    args.svgd_iters, planner.nxi)
            curve = torch.as_tensor(refined['final_pos'][0], dtype=torch.float32,
                                    device=device).clamp(0.0, 1.0)
        else:
            raise ValueError(method)

        curve[0] = torch.as_tensor(pos, dtype=torch.float32, device=device)
        seg = trim_to_length(curve, LENGTH_UNIT)
        n_pts = max(8, int(round(PTS_PER_UNIT * path_length(seg) / LENGTH_UNIT)))
        seg = resample_arclength(seg, n_pts)

        torch.manual_seed(trial_seed * 1000 + r)
        pts, vals = measure(seg, truth, noise_std=args.noise, sensor_radius=args.sensor_radius)
        pts_t, vals_t = thin(pts, vals, max_points=64)
        if vr.needs_belief_update(condition):
            belief.observe(pts_t, vals_t)

        driven = seg if driven is None else torch.cat([driven, seg], dim=0)
        swept = swept_mass_fraction(driven, truth, sensor_radius=args.coverage_radius)
        plen = path_length(driven)
        pos = seg[-1].detach().cpu().numpy().astype(np.float64)

        rows.append(dict(trial=trial_seed, condition=condition, method=method,
                         round=r, swept_mass=swept, path_len=plen,
                         reached99=int(swept >= 0.99)))
        if swept >= 0.99:
            break
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n_trials', type=int, default=5)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--conditions', type=str, default='half_known,ten_samples,none_known')
    ap.add_argument('--methods', type=str, default='sun,cfm_raw,cfm_svgd')
    ap.add_argument('--sun_iters', type=int, default=60,
                    help='FM-Stein iterations per round for the "sun" method.')
    ap.add_argument('--svgd_iters', type=int, default=25,
                    help='SvgdRefiner iterations per round for "cfm_svgd".')
    ap.add_argument('--max_rounds', type=int, default=15)
    ap.add_argument('--noise', type=float, default=0.05)
    ap.add_argument('--sensor_radius', type=float, default=0.06)
    ap.add_argument('--coverage_radius', type=float, default=0.06)
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop starting new trials after this many hours; already-written '
                         'rows stay valid, re-run with the same --out_tag to resume.')
    args = ap.parse_args()

    from run_eval_matrix import DEFAULT_CKPT
    from run_ideal_matrix import build_particle_planner
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    planner = None
    methods = [m for m in args.methods.split(',') if m]
    if any(m.startswith('cfm') for m in methods):
        planner = build_particle_planner(args.ckpt or DEFAULT_CKPT, device)
        warm_phi = C.density_grid(C.make_trial(seed=2 ** 31 - 1)['pdf_fn'], res=SUN_RES)
        C.cfm_raw_curve(planner, warm_phi, start=(0.5, 0.5), device=device)

    out_dir = os.path.join(_mat, 'results', args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'rq3_gp_targets.csv')
    conditions = [c for c in args.conditions.split(',') if c]

    done = _load_done(csv_path)
    need_header = not os.path.isfile(csv_path)
    t0 = time.time()
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if need_header:
            w.writeheader()
        for i in range(args.n_trials):
            if C.time_budget_exceeded(t0, args.time_budget_h):
                print(f"[rq3] time budget reached before trial {i}, stopping.", flush=True)
                break
            seed = args.seed0 + i
            for cond in conditions:
                for m in methods:
                    if (seed, cond, m) in done:
                        continue
                    for row in run_trial(seed, cond, m, planner, device, args):
                        w.writerow(row)
                    f.flush()
            print(f"[rq3] trial {i + 1}/{args.n_trials} done", flush=True)
    print(f"[rq3] -> {csv_path}")

    import pandas as pd
    df = pd.read_csv(csv_path)
    plot_dir = os.path.join(out_dir, 'plots')
    for cond in conditions:
        sub = df[df['condition'] == cond]
        rounds_to_99 = {}
        for m in methods:
            vals = []
            for _, g in sub[sub['method'] == m].groupby('trial'):
                hit = g[g['reached99'] == 1]
                vals.append(float(hit['round'].iloc[0]) + 1 if len(hit) else np.nan)
            rounds_to_99[m] = np.array(vals)
        ps.violin(rounds_to_99, f"Rounds to sweep 99% of the true target mass ({cond})",
                 'Rounds (length units driven)',
                 os.path.join(plot_dir, f'rq3_rounds_to_99_{cond}.png'),
                 methods=methods, log_y=False)
        print(f"[rq3] {cond}: " + ', '.join(
            f"{m} median={np.nanmedian(rounds_to_99[m]):.1f}" for m in methods))

        ps.anytime_curve(sub.rename(columns={'round': 'x_round'}),
                         f"Swept true mass over driven length units ({cond})",
                         os.path.join(plot_dir, f'rq3_anytime_{cond}.png'),
                         methods=methods, x='x_round', y='swept_mass', log_x=False, log_y=False,
                         xlabel='Executed length units (rounds)', ylabel='Swept true target mass')
    print(f"[rq3] plots -> {plot_dir}")


if __name__ == '__main__':
    main()
