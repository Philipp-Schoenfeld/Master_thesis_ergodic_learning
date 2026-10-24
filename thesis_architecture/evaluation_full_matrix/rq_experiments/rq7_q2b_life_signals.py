r"""
rq7_q2b_life_signals.py
========================
Q2.B of Sun et al. ("Flow Matching Ergodic Coverage", RSS 2025), Fig. 6: search
and rescue with an UNNORMALIZED target. Two hidden regions emit positive life
signals; a Gaussian process fitted to sparse prior (2 positive, 18 negative)
measurements gives the likelihood of measuring a positive signal, which is NOT
a normalized density. A second-order differential-drive robot plans one
trajectory (300 points) per method; the score is the number of trajectory
points at which a positive signal is measured.

Methods (same dynamics, start state and GP for all):
    fm_stein      Sun's FM-Stein flow, cold start (score = grad log p is
                  invariant to the missing normalization constant)
    cfm_raw       one CFM forward pass, tracked by the dynamics (`flat_map`)
    cfm_fmstein   CFM curve as warm start, then the same FM-Stein budget

What is taken from the paper's Fig. 6 (extracted from its vector graphics
`sns_trajs.svg`, arXiv 2504.17872v1): 2 positive + 18 negative prior samples,
300 trajectory points, grayscale GP-likelihood contours (colorbar 0 .. ~0.9),
and -- for the example panels (`--figure_world`, default on) -- the EXACT prior
sample positions and the exact start position of the paper's panel.
What is NOT in the paper or in MurpheyLab/lqr-flow-matching (so chosen here,
marked `ASSUMPTION`): the world generator of the 100 randomized trials, GP
kernel/link, horizon, iteration budget, and the counting rule.

Reused from this repo: `dynamics_zoo.DiffDrive2LQR`, `exp_common` (FM-Stein
loop, CFM forward pass), `exploration/common/belief.GPBelief`,
`sun_refine.make_grid_score`.

Usage:
    python rq7_q2b_life_signals.py --n_trials 3 --out_tag smoke_q2b
    python rq7_q2b_life_signals.py --plot_only --out_tag smoke_q2b
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
_arch = os.path.dirname(_mat)
for _p in (_here, _mat, os.path.join(_arch, 'exploration'),
          os.path.join(_arch, 'ergodic_dataset_generator')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import dynamics_zoo as dz                                          # noqa: E402
import exp_common as C                                             # noqa: E402

DT, TSTEPS = 0.05, 300            # 300 trajectory points as in the paper's panels
GP_RES = 64
FIELDS = ['trial', 'method', 'collected', 'n_pts', 'path_len', 'oob_frac', 'E_norm', 'time_s']
ALL_METHODS = ['fm_stein', 'cfm_raw', 'cfm_fmstein']
DEFAULT_METHODS = ','.join(ALL_METHODS)
COLORS = {'fm_stein': '#ff7f0e', 'cfm_raw': '#90CAF9', 'cfm_fmstein': '#1565C0'}
LABELS = {'fm_stein': 'FM-Stein', 'cfm_raw': 'CFM (raw)', 'cfm_fmstein': 'CFM + FM-Stein'}

# Prior samples and start of the paper's Fig. 6 (FM-Stein panel), read from the
# SVG marker coordinates, in unit-square coordinates (x right, y up).
PAPER_POS = [(0.1702, 0.5955), (0.7565, 0.6752)]
PAPER_NEG = [(0.2227, 0.1545), (0.7425, 0.7478), (0.8211, 0.3461), (0.7022, 0.2579),
             (0.7495, 0.3084), (0.1144, 0.3019), (0.0690, 0.6394), (0.7270, 0.2872),
             (0.1422, 0.0894), (0.4537, 0.4300), (0.0928, 0.7521), (0.4722, 0.8076),
             (0.5755, 0.8494), (0.6216, 0.7594), (0.4950, 0.6933), (0.2941, 0.9147),
             (0.7587, 0.3429), (0.8743, 0.2324)]
PAPER_X0 = (0.7640, 0.5000)


# ── World: hidden emitter regions, prior measurements, GP likelihood ─────────

def make_world(seed, args):
    """ASSUMPTION (the paper gives no generator): two hidden disc-shaped
    emitter regions (min. centre distance 0.4); one positive prior sample in
    each, `n_neg` negative samples uniform outside the regions; random start."""
    rng = np.random.default_rng(10_000 + seed)
    while True:
        centers = rng.uniform(0.15, 0.85, size=(2, 2))
        if np.linalg.norm(centers[0] - centers[1]) >= 0.4:
            break
    radii = np.full(2, args.region_radius)
    pos = []
    for c in centers:
        while True:
            p = c + rng.uniform(-1, 1, size=2) * args.region_radius * 0.5
            if np.linalg.norm(p - c) <= args.region_radius * 0.5:
                pos.append(p)
                break
    neg = []
    while len(neg) < args.n_neg:
        p = rng.uniform(0.05, 0.95, size=2)
        if (np.linalg.norm(p[None] - centers, axis=1) > radii + 0.03).all():
            neg.append(p)
    prior_x = np.vstack([np.asarray(pos), np.asarray(neg)])
    prior_y = np.concatenate([np.ones(2), -np.ones(len(neg))])
    return dict(centers=centers, radii=radii, prior_x=prior_x, prior_y=prior_y,
                x0=np.asarray(C.random_x0(rng), dtype=np.float64))


def figure_world(args):
    """The paper's own Fig. 6 prior samples and start; the emitter regions are
    centred on its two positive samples (ASSUMPTION: radius `region_radius`)."""
    prior_x = np.vstack([np.asarray(PAPER_POS), np.asarray(PAPER_NEG)])
    prior_y = np.concatenate([np.ones(2), -np.ones(len(PAPER_NEG))])
    return dict(centers=np.asarray(PAPER_POS), radii=np.full(2, args.region_radius),
                prior_x=prior_x, prior_y=prior_y, x0=np.asarray(PAPER_X0, dtype=np.float64))


def gp_likelihood(world, args):
    """Likelihood of a positive signal on a (GP_RES, GP_RES) grid (row = y):
    GP regression on the +-1 labels (`GPBelief`, the project's own GP), then
    sigmoid(gain * mu - bias). ASSUMPTION: the link and its constants were
    tuned by eye against the paper's contour plot (background ~0.1, peaks
    ~0.9, white holes around negatives). Not normalized -- the point of Q2.B."""
    import torch
    from common.belief import GPBelief
    b = GPBelief(grid_res=GP_RES, lengthscale=args.gp_lengthscale, variance=1.0,
                 noise=args.gp_noise)
    b.observe(torch.as_tensor(world['prior_x'], dtype=torch.float32),
              torch.as_tensor(world['prior_y'], dtype=torch.float32))
    mu = b.posterior_grid()[0].detach().cpu().numpy().astype(np.float64)
    return 1.0 / (1.0 + np.exp(-(args.lik_gain * mu - args.lik_bias)))


def sun_cold_init(x0_xy):
    """Cold start analogous to Sun's recipe for the point mass (`u = 0`, initial
    velocity towards the centre, magnitude 2 * distance / T): the diff-drive
    heads for (0.5, 0.5) at that speed. With v = 0 (`dz.cold_x0_u0`) the
    stationary robot collapsed onto the nearest peak in every test."""
    import jax.numpy as jnp
    d = np.array([0.5, 0.5]) - np.asarray(x0_xy)
    v = 2.0 * np.linalg.norm(d) / (DT * TSTEPS)
    x0 = jnp.array([x0_xy[0], x0_xy[1], np.arctan2(d[1], d[0]), v, 0.0])
    return x0, jnp.zeros((TSTEPS, 2))


def count_collected(traj_xy, world):
    """Trajectory points inside a hidden emitter region = positive signals
    measured (ASSUMPTION: the paper does not state its counting rule; its
    violins reach ~80 for 300 points, i.e. a per-point count)."""
    d = np.linalg.norm(np.asarray(traj_xy)[:, None, :] - world['centers'][None], axis=-1)
    return int((d <= world['radii'][None]).any(axis=1).sum())


# ── One trial ─────────────────────────────────────────────────────────────────

def run_trial(label, world, methods, planner, device, pm, lin, solve, args):
    from common.sun_refine import make_grid_score
    phi = gp_likelihood(world, args)
    phi_unit = phi / max(float(phi.max()), 1e-9)      # CFM conditioning + score only
    x0_xy = world['x0']
    x0c, u0c = sun_cold_init(x0_xy)
    score_fn = make_grid_score(phi_unit)              # grad log p: scale-free
    phik_norm = C.target_coeffs(phi)                  # normalized reference metric
    out, trajs = [], {}

    t_cfm = 0.0
    if any(m.startswith('cfm') for m in methods):
        curve, t_cfm = C.cfm_raw_curve(planner, phi_unit, start=x0_xy, device=device)
        x0w, u0w = pm.flat_map(curve, DT, TSTEPS)

    for m in methods:
        t0 = time.perf_counter()
        if m == 'fm_stein':
            traj = C.timed_solve(pm, lin, solve, score_fn, x0c, u0c, [args.iters],
                                 warmup=False)[0]['traj_xy']
        elif m == 'cfm_raw':
            traj = pm.positions(pm.traj_sim(x0w, u0w))
        elif m == 'cfm_fmstein':
            traj = C.timed_solve(pm, lin, solve, score_fn, x0w, u0w, [args.iters],
                                 warmup=False)[0]['traj_xy']
        else:
            raise ValueError(m)
        dt_s = time.perf_counter() - t0 + (t_cfm if m.startswith('cfm') else 0.0)
        traj = np.asarray(traj, dtype=np.float64)
        if not np.isfinite(traj).all():
            traj = np.tile(np.asarray(x0_xy)[None], (TSTEPS, 1))
        out.append(dict(trial=label, method=m, collected=count_collected(traj, world),
                        n_pts=len(traj),
                        path_len=float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()),
                        oob_frac=float(((traj < 0) | (traj > 1)).any(axis=1).mean()),
                        E_norm=C.ergodic_error(traj, phik_norm), time_s=dt_s))
        trajs[m] = traj
    return out, phi, trajs


def save_trial(out_dir, name, world, phi, trajs):
    d = os.path.join(out_dir, 'trials')
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, name + '.npz'), phi=phi, centers=world['centers'],
                        radii=world['radii'], prior_x=world['prior_x'],
                        prior_y=world['prior_y'], x0=world['x0'],
                        **{f'traj_{m}': t for m, t in trajs.items()})


# ── Fig. 6 ────────────────────────────────────────────────────────────────────

def _read_rows(csv_path):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['trial'] = int(r['trial'])
        for k in ('collected', 'n_pts', 'path_len', 'oob_frac', 'E_norm', 'time_s'):
            r[k] = float(r[k])
    return rows


def _per_method(rows, methods, key='collected'):
    out = {m: {} for m in methods}
    for r in rows:
        if r['method'] in out and r['trial'] >= 0:
            out[r['method']][r['trial']] = r[key]
    return out


def _pick_example(rows, methods):
    """Trial whose per-method counts are closest to the per-method medians."""
    pm_ = _per_method(rows, methods)
    trials = sorted(set.intersection(*[set(v) for v in pm_.values()]))
    med = {m: np.median(list(pm_[m].values())) for m in methods}
    return min(trials, key=lambda t: sum((pm_[m][t] - med[m]) ** 2 for m in methods))


def _violin(ax, vals_by_method, methods):
    ax.set_facecolor('white')
    ax.grid(axis='y', ls='--', lw=1.4, color='#b0b0b0', alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    for i, m in enumerate(methods):
        v = np.asarray(vals_by_method[m], dtype=float)
        vp = ax.violinplot([v], positions=[i], widths=0.85, showextrema=False)
        for b in vp['bodies']:
            b.set_facecolor(COLORS[m])
            b.set_edgecolor('#444444')
            b.set_linewidth(1.6)
            b.set_alpha(0.9)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        lo, hi = v[v >= q1 - 1.5 * (q3 - q1)].min(), v[v <= q3 + 1.5 * (q3 - q1)].max()
        ax.vlines(i, lo, hi, color='#444444', lw=2.5, zorder=3)
        ax.vlines(i, q1, q3, color='#444444', lw=8, zorder=3)
        ax.plot([i], [med], marker='_', color='white', ms=7, mew=2, zorder=4)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=10, color='#1A1A2E')
    ax.set_title('Collected Positive Signals (↑ higher is better)', fontsize=11,
                 color='#1A1A2E')
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def _panel(ax, z, m, collected=None):
    ax.set_facecolor('#fafafa')
    cf = ax.contourf(np.linspace(0, 1, z['phi'].shape[1]), np.linspace(0, 1, z['phi'].shape[0]),
                     z['phi'], levels=np.arange(0.0, 1.05, 0.08), cmap='gray_r', vmin=0, vmax=1.0,
                     zorder=0)
    tr = z[f'traj_{m}']
    ax.plot(tr[:, 0], tr[:, 1], color='black', lw=1.5, zorder=2)
    ax.scatter(tr[:, 0], tr[:, 1], s=26, color=COLORS[m], edgecolors='black', linewidths=0.8,
               zorder=3)
    ax.scatter(*z['x0'], s=260, color=COLORS[m], edgecolors='black', linewidths=1.0, zorder=5)
    neg, pos = z['prior_y'] < 0, z['prior_y'] > 0
    ax.scatter(z['prior_x'][neg, 0], z['prior_x'][neg, 1], s=170, marker='X', color='white',
               edgecolors='black', linewidths=1.0, zorder=4)
    ax.scatter(z['prior_x'][pos, 0], z['prior_x'][pos, 1], s=200, marker='X', color='#2ca02c',
               edgecolors='black', linewidths=1.0, zorder=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    t = LABELS[m] + (f'  ({collected}/{len(tr)})' if collected is not None else '')
    ax.set_title(t, fontsize=13, color='#1A1A2E', fontweight='bold' if m == 'fm_stein' else None)
    return cf


def plot_fig6(out_dir, methods, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    rows = _read_rows(os.path.join(out_dir, 'rq7_q2b_life_signals.csv'))
    methods = [m for m in methods if m in {r['method'] for r in rows}]
    fig_path = os.path.join(out_dir, 'trials', 'figure_world.npz')
    if args.figure_world and os.path.isfile(fig_path):
        z, ex = np.load(fig_path), 'paper prior samples'
    else:
        ex_id = args.example_trial if args.example_trial is not None else _pick_example(rows, methods)
        z, ex = np.load(os.path.join(out_dir, 'trials', f'trial_{ex_id:04d}.npz')), f'trial {ex_id}'

    n = len(methods)
    fig = plt.figure(figsize=(5.2 + 4.0 * n, 4.6), facecolor='white')
    gs = GridSpec(1, n + 2, figure=fig, width_ratios=[1.35] + [1.0] * n + [0.05], wspace=0.12)
    _violin(fig.add_subplot(gs[0]), {m: list(_per_method(rows, [m])[m].values()) for m in methods},
            methods)
    cf = None
    for i, m in enumerate(methods):
        ax = fig.add_subplot(gs[1 + i])
        d = np.linalg.norm(z[f'traj_{m}'][:, None, :] - z['centers'][None], axis=-1)
        cf = _panel(ax, z, m, int((d <= z['radii'][None]).any(axis=1).sum()))
        if i == 0:
            ax.legend(handles=[
                Line2D([], [], marker='X', ls='', mfc='#2ca02c', mec='black', ms=11, label='Positive Signal'),
                Line2D([], [], marker='X', ls='', mfc='white', mec='black', ms=11, label='Negative Signal')],
                loc='lower right', fontsize=10, framealpha=1.0, fancybox=True, shadow=True)
    cb = fig.colorbar(cf, cax=fig.add_subplot(gs[n + 1]))
    cb.set_ticks(np.arange(0.0, 0.81, 0.16))
    n_tr = len({r['trial'] for r in rows if r['trial'] >= 0})
    fig.suptitle(f'Q2.B replica of Sun et al. Fig. 6 (violins: {n_tr} randomized trials; '
                 f'panels: {ex})', color='#1A1A2E', fontsize=12, y=1.02)
    path = os.path.join(out_dir, 'plots', 'fig6_q2b_life_signals.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    print(f"{'method':14s}{'collected med/mean':>22s}{'oob_frac mean':>16s}{'E_norm med':>12s}{'time_s med':>12s}")
    for m in methods:
        g = {k: np.asarray(list(_per_method(rows, [m], k)[m].values()))
             for k in ('collected', 'oob_frac', 'E_norm', 'time_s')}
        print(f"{m:14s}{np.median(g['collected']):>12.1f}/{g['collected'].mean():<8.1f}"
              f"{g['oob_frac'].mean():>16.3f}{np.median(g['E_norm']):>12.4f}"
              f"{np.median(g['time_s']):>12.2f}")
    return path


def _load_done(csv_path, methods):
    done = {}
    if os.path.isfile(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                done.setdefault(int(row['trial']), set()).add(row['method'])
    return {t for t, ms in done.items() if set(methods) <= ms}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n_trials', type=int, default=100, help='Paper: 100 randomized trials.')
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--methods', type=str, default=DEFAULT_METHODS)
    ap.add_argument('--iters', type=int, default=300,
                    help='ASSUMPTION: FM-Stein iterations (same for FM-Stein and CFM+FM-Stein).')
    ap.add_argument('--n_neg', type=int, default=18)
    ap.add_argument('--region_radius', type=float, default=0.10)
    ap.add_argument('--gp_lengthscale', type=float, default=0.17)
    ap.add_argument('--gp_noise', type=float, default=0.3)
    ap.add_argument('--lik_gain', type=float, default=4.25)
    ap.add_argument('--lik_bias', type=float, default=1.7)
    ap.add_argument('--figure_world', action=argparse.BooleanOptionalAction, default=True,
                    help="Example panels use the paper's exact prior samples and start.")
    ap.add_argument('--example_trial', type=int, default=None)
    ap.add_argument('--plot_only', action='store_true')
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop starting new trials after this many hours; re-run with the '
                         'same --out_tag to resume.')
    args = ap.parse_args()

    methods = [m for m in args.methods.split(',') if m]
    for m in methods:
        if m not in ALL_METHODS:
            raise SystemExit(f'unknown method {m!r}; choose from {ALL_METHODS}')
    out_dir = os.path.join(_mat, 'results', args.out_tag)
    csv_path = os.path.join(out_dir, 'rq7_q2b_life_signals.csv')

    if not args.plot_only:
        import torch
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        planner = None
        if any(m.startswith('cfm') for m in methods):
            from run_eval_matrix import DEFAULT_CKPT
            from run_ideal_matrix import build_particle_planner
            planner = build_particle_planner(args.ckpt or DEFAULT_CKPT, device)
            warm = C.density_grid(C.make_trial(seed=2 ** 31 - 1)['pdf_fn'], res=GP_RES)
            C.cfm_raw_curve(planner, warm, start=(0.5, 0.5), device=device)

        pm, lin, solve = dz.build(dz.DiffDrive2LQR, dt=DT)
        pm.positions = lambda x_traj: np.asarray(x_traj)[:, :2]
        os.makedirs(out_dir, exist_ok=True)

        if args.figure_world:
            fw = figure_world(args)
            rows, phi, trajs = run_trial(-1, fw, methods, planner, device, pm, lin, solve, args)
            save_trial(out_dir, 'figure_world', fw, phi, trajs)
            print('[rq7] figure world: ' + ', '.join(f"{r['method']}={r['collected']}" for r in rows),
                  flush=True)

        done = _load_done(csv_path, methods)
        need_header = not os.path.isfile(csv_path)
        t0 = time.time()
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if need_header:
                w.writeheader()
            for i in range(args.n_trials):
                if C.time_budget_exceeded(t0, args.time_budget_h):
                    print(f'[rq7] time budget reached before trial {i}, stopping.', flush=True)
                    break
                seed = args.seed0 + i
                if seed in done:
                    continue
                world = make_world(seed, args)
                rows, phi, trajs = run_trial(seed, world, methods, planner, device, pm, lin,
                                             solve, args)
                save_trial(out_dir, f'trial_{seed:04d}', world, phi, trajs)
                w.writerows(rows)
                f.flush()
                print(f'[rq7] trial {i + 1}/{args.n_trials}: ' +
                      ', '.join(f"{r['method']}={r['collected']}" for r in rows), flush=True)
        print(f'[rq7] -> {csv_path}')

    print(f'[rq7] Fig. 6 replica -> {plot_fig6(out_dir, methods, args)}')


if __name__ == '__main__':
    main()
