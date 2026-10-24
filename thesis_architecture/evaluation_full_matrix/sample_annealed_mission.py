#!/usr/bin/env python3
r"""
sample_annealed_mission.py
===========================
A small, local, single-process spot check of `run_mission_eval.py`'s own
replanning-mission pipeline -- reused as-is, not reimplemented:

  * init: `apply_cfm_belief.CfmPlanner` with the start-conditioned checkpoint
    (default: `transfer/netz2d_startpunkt.pt`, same as `run_mission_eval.py`'s
    own default),
  * refinement: `svgd_batched.BatchedTsvecSvgdTorch` (`--refiner tsvec_svgd`
    in the real pipeline) -- the paper-faithful TSVEC-SVGD
    (`common/tsvec_svgd.py`, Li et al. arXiv:2603.09458), run for `--n_iters`
    (500 here) iterations per round,
  * belief: `common.belief.GPBelief` via `variant_runner.build_belief`,
    knowledge condition `'none_known'` (zero prior measurements),
  * target density: `apply_cfm_belief.debt_density`, `variant_runner
    .STRATEGIES['ucb_tuned_svgd0']` ('ucb' family) for everything except the
    one deliberate change below,
  * candidate selection: lowest ergodic error (K=10, `ergodic_energy_torch
    .coeffs_from_points`) against the round's own target -- `--select_by
    target`, the real pipeline's default (no look at the ground truth).

The one deliberate deviation from the real pipeline, and the actual point of
this script: `variant_runner.STRATEGIES['ucb_tuned_svgd0']['param']` is a
SINGLE tuned kappa (0.4597), fixed for the whole mission. Here `kappa` is
instead annealed every round from `--kappa0` to `--kappa1` via
`common.acquisition.kappa_schedule` -- the same exploration-to-exploitation
schedule `exploration/variant_e_two_stage/run_e.py`'s annealed mode and
`exploration/variant_c_receding/run_c.py` already use. This is a sample
check of THAT, not a reproduction of the tuned baseline.

Also hardcoded on: "full execution" (drive the WHOLE refined candidate every
round, not one length-unit slice before replanning) -- `run_mission_eval.py
--full_execution` does the same on the full matrix; here it is not optional,
since the previous round of questions was specifically about seeing the
complete planned trajectory.

Scoped down from the real matrix run for a local, GPU-less machine:
single process (no multiprocessing pool, no results database), `--n_init`
defaults to 8 instead of the real pipeline's 30 candidates per round, and
only the `'ucb_tuned_svgd0'` strategy (the only one `kappa` governs directly
-- `debt_density`/`zieldichte` need a kappa-less strategy's own parameter,
e.g. `phi_tau` for 'lse', to be annealed the same way, which this script does
not attempt).

    python sample_annealed_mission.py --n_rounds 5          # shape 'A' only
    python sample_annealed_mission.py --shape_names "" --shapes 2
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors   # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
import torch                          # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
          os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import apply_cfm_belief as acb                                    # noqa: E402
import run_svgd_convergence as rsc                                # noqa: E402
import variant_runner as vr                                        # noqa: E402
from common.acquisition import kappa_schedule                      # noqa: E402
from common.data import load_truth                                 # noqa: E402
from common.metrics import path_length                             # noqa: E402
from common.observation import measure, thin                       # noqa: E402
from ergodic_energy_torch import coeffs_from_points                # noqa: E402
from exploration_optimierung.mission import (LENGTH_UNIT, PTS_PER_UNIT,  # noqa: E402
                                             resample_arclength)
from metrics_explore_exploit import ExploreExploitErgodic          # noqa: E402
from run_eval_matrix import DEFAULT_CKPT                           # noqa: E402
from run_ideal_matrix import PLANNER_BUILDERS                      # noqa: E402
from svgd_batched import BatchedTsvecSvgdTorch                     # noqa: E402

STRATEGY_NAME = 'ucb_tuned_svgd0'   # the only family `kappa` governs directly
CONDITION = 'none_known'            # zero prior measurements, as requested


# ===========================================================================
# Style (project convention, see CLAUDE.md "Visualization Aesthetics") --
# same helpers as exploration/variant_e_two_stage/plot_annealed.py, kept
# local rather than imported across the exploration/evaluation_full_matrix
# boundary.
# ===========================================================================

def white_inferno():
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mcolors.LinearSegmentedColormap.from_list('white_inferno', inf)


def style(ax, xlim, ylim):
    ax.set_facecolor('white')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')


def full_view(curve, pad_frac=0.08):
    pts = curve.detach().cpu().numpy() if torch.is_tensor(curve) else curve
    x0, x1 = min(0.0, pts[:, 0].min()), max(1.0, pts[:, 0].max())
    y0, y1 = min(0.0, pts[:, 1].min()), max(1.0, pts[:, 1].max())
    px, py = pad_frac * (x1 - x0), pad_frac * (y1 - y0)
    return (x0 - px, x1 + px), (y0 - py, y1 + py)


ROUND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'explore_to_exploit', ['#5E35B1', '#1E88E5', '#00ACC1', '#00C853'])


def round_colors(n_rounds):
    if n_rounds == 1:
        return ['#00C853']
    return [mcolors.to_hex(ROUND_CMAP(t)) for t in np.linspace(0.0, 1.0, n_rounds)]


def info_gain_cutoff(belief, seg, unc0, threshold):
    """Smallest prefix length k of `seg` whose predicted information gain
    (unc0 - belief.uncertainty_after(seg[:k])) reaches `threshold`.

    `uncertainty_after` depends only on measurement *locations*, not values
    (see its docstring), so this can be evaluated before anything is driven
    or measured. Gain is monotone non-decreasing in k (more candidate
    measurement locations cannot raise GP posterior variance), so a plain
    binary search finds the crossing point in O(log n) lookaheads. Falls
    back to the full curve (k = n) if even that does not reach `threshold`.
    """
    n = seg.shape[0]
    with torch.no_grad():
        full_gain = float(unc0 - belief.uncertainty_after(seg))
        if full_gain < threshold:
            return n, full_gain
        lo, hi = 2, n
        while lo < hi:
            mid = (lo + hi) // 2
            gain = float(unc0 - belief.uncertainty_after(seg[:mid]))
            if gain >= threshold:
                hi = mid
            else:
                lo = mid + 1
        gain = float(unc0 - belief.uncertainty_after(seg[:lo]))
    return lo, gain


# ===========================================================================
# One mission: CFM init -> TSVEC-SVGD refine -> execute (full candidate, or
# only until --ig_threshold predicted information gain is reached) ->
# observe -> repeat, kappa annealed every round.
# ===========================================================================

def run_mission(name, truth, planner, bs, ee, B32, args, device,
                n_rounds, n_init, n_iters, kappas, start_pos, seed,
                ig_threshold=None):
    belief = vr.build_belief(CONDITION, truth, seed=seed, device=device,
                             gp_noise=args.gp_noise, gp_lengthscale=0.08,
                             dtype=torch.float64, kernel=args.belief_kernel,
                             ak_fit_iters=args.ak_fit_iters, np_ckpt=args.np_ckpt)
    start = torch.as_tensor(start_pos, dtype=torch.float32, device=device)

    segments, phis, infos = [], [], []
    for r in range(n_rounds):
        torch.manual_seed(seed * 1000 + r)
        mu, sd = belief.posterior_grid()
        belief._cache = None
        mu, sd = mu.float(), sd.float()
        visit = vr._visit_field(torch.cat(segments, dim=0).to(device) if segments else None,
                                args, device)
        kap = kappas[r]
        args.kappa = kap
        phi, _ = acb.debt_density(mu, sd, visit, kap, args)
        phi = phi.float()
        if float(phi.max()) < 1e-6:
            phi = torch.ones_like(phi)

        parts = acb.phi_particles(phi, args.n_particles, mode=args.phi_mode,
                                  quantile=args.phi_quantile, device=device)
        planner.cfg_weight = 2.0
        with torch.no_grad():
            cps = planner.plan(parts, n_candidates=n_init, start=start)
            curves = planner.render(cps).detach()                   # (n_init, T, 2)

        # -- TSVEC-SVGD refinement (common/tsvec_svgd.py via its batched
        #    adapter), K=10 Fourier target to match `ee`/`BatchedTsvecSvgdTorch`
        #    (NOT the K=8 `SvgdRefiner`/'tsvec' backend uses -- see
        #    `run_mission_eval.py`'s own `run_svgd` target-construction branch).
        phi_k10 = ee.target_coeffs(phi)
        target_np = phi_k10.detach().cpu().numpy().astype(np.float64)
        curves_np = curves.cpu().numpy().astype(np.float64)
        starts_np = np.tile(start.detach().cpu().numpy(), (n_init, 1))
        seeds = [seed * 100000 + r * 1000 + i for i in range(n_init)]
        out = bs.run(curves_np, target_np, starts_np, seeds, n_iters, record=False)
        final_cps = torch.as_tensor(out['final_cps'], device=device)
        final_curves = torch.einsum('pi,cid->cpd', B32, final_cps)   # (n_init, T, 2)

        # -- select by lowest ergodic error against THIS round's target
        #    (`--select_by target`, the real pipeline's default -- no look at
        #    the ground truth).
        c = coeffs_from_points(final_curves, ee.k_idx)
        E = (ee.w * 0.5 * ee.Lambda * (c - phi_k10.unsqueeze(0)) ** 2).sum(-1)
        sel = int(torch.argmin(E))
        winner = final_curves[sel].clone()
        winner[0] = start                                             # hard snap, as `execute()`
        winner = winner.clamp(0.0, 1.0)

        # -- execute: the WHOLE refined candidate by default ("full
        #    execution"), or, if `ig_threshold` is set, only the prefix
        #    needed to reach that much predicted information gain.
        n_pts = max(8, int(round(PTS_PER_UNIT * path_length(winner) / LENGTH_UNIT)))
        seg = resample_arclength(winner, n_pts)

        ig_gain = None
        if ig_threshold is not None:
            unc0 = belief.total_uncertainty()
            k, ig_gain = info_gain_cutoff(belief, seg, unc0, ig_threshold)
            seg = seg[:k]

        pts, vals = measure(seg, truth, noise_std=args.noise,
                            sensor_radius=args.sensor_radius)
        pts_t, vals_t = thin(pts, vals, max_points=args.max_obs)
        belief.observe(pts_t, vals_t)

        segments.append(seg.detach().cpu())
        phis.append(phi.detach().cpu())
        infos.append({'gain': ig_gain, 'n_used': seg.shape[0], 'n_planned': n_pts})
        start = seg[-1].detach()
        extra = (f"  dH={ig_gain:.3f} (target {ig_threshold:.3f}, "
                f"{seg.shape[0]}/{n_pts} pts driven)" if ig_gain is not None else "")
        print(f"    [{name}] round {r}: kappa={kap:.3f}  "
              f"path_len={path_length(seg):.2f}  n_obs={belief.n_obs}{extra}")

    return segments, phis, infos


# ===========================================================================
# Plot: one row per shape, one column per round.
# ===========================================================================

def plot_round(ax, phi, seg, kap, col, row_label=None, col_title=None, cmap=None,
               info=None):
    ax.imshow(phi.numpy(), origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
             vmin=0, vmax=1.0, alpha=0.55, zorder=1)
    pts = seg.numpy()
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.2, alpha=0.95, zorder=3)
    ax.plot(pts[0, 0], pts[0, 1], 'o', color=col, ms=5, zorder=4)
    style(ax, *full_view(seg))
    if col_title:
        ax.set_title(col_title, fontsize=10, color='#1A1A2E')
    if row_label:
        ax.set_ylabel(row_label, fontsize=10.5, color='#1A1A2E')
    ax.text(0.03, 0.97, f"κ = {kap:.3g}", transform=ax.transAxes, fontsize=9,
            color='#1A1A2E', ha='left', va='top', zorder=5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#ccc', alpha=0.85))
    if info is not None and info['gain'] is not None:
        ax.text(0.03, 0.03, f"dH={info['gain']:.2f}  "
               f"{info['n_used']}/{info['n_planned']} pts",
               transform=ax.transAxes, fontsize=7.5, color='#555',
               ha='left', va='bottom')


def plot_summary(ax, truth, segments, colors, kappas, row_label=None, col_title=None,
                 cmap=None):
    """All rounds of one mission overlaid on the GROUND TRUTH density (not a
    round's belief-based Phi) -- the final-result view, driven path vs. what
    was actually there, which the mission itself never got to see."""
    ax.imshow(truth.numpy(), origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
             vmin=0, vmax=float(truth.max()) or 1.0, alpha=0.55, zorder=1)
    all_pts = torch.cat(segments, dim=0)
    for r, (seg, col, kap) in enumerate(zip(segments, colors, kappas)):
        pts = seg.numpy()
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.0, alpha=0.9, zorder=3,
                label=f"Round {r}: κ = {kap:.3g}")
    style(ax, *full_view(all_pts))
    ax.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white',
              edgecolor='#ddd', framealpha=0.85, handlelength=1.5)
    if col_title:
        ax.set_title(col_title, fontsize=10, color='#1A1A2E')
    if row_label:
        ax.set_ylabel(row_label, fontsize=10.5, color='#1A1A2E')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=2,
                   help='Number of heldout (val-split) shapes to sample, in '
                        'DB order -- only used with --shape_names "".')
    p.add_argument('--shape_names', type=str, default='A',
                   help='Comma-separated, explicit heldout shape names, e.g. '
                        '"A,korean_5" -- forwarded to '
                        'common.data.load_truth(labels=...). Pass "" to use '
                        'the first --shapes val-split shapes instead.')
    p.add_argument('--n_rounds', type=int, default=5)
    p.add_argument('--n_init', type=int, default=8,
                   help='Candidates per round (real pipeline default: 30; '
                        'kept low here for a CPU-feasible sample).')
    p.add_argument('--n_iters', type=int, default=500,
                   help='TSVEC-SVGD iterations per candidate per round.')
    p.add_argument('--kappa0', type=float, default=1.0,
                   help='Starting kappa (round 0) -- high means strongly '
                        'exploration-dominated (sigma term swamps mu). Ignored '
                        'if --kappas is given.')
    p.add_argument('--kappa1', type=float, default=0.05,
                   help='Final-round kappa -- low means strongly '
                        'exploitation-dominated (mu term dominates). Ignored '
                        'if --kappas is given.')
    p.add_argument('--kappas', type=str, default=None,
                   help='Comma-separated, explicit per-round kappa values '
                        '(length must equal --n_rounds), e.g. '
                        '"3,1.5,0.75,0.35,0.1" -- overrides the linear '
                        '--kappa0/--kappa1 schedule when a non-linear decay '
                        'is wanted.')
    p.add_argument('--ig_threshold', type=float, default=None,
                   help='If set, replan once the predicted information gain '
                        '(reduction in GPBelief.total_uncertainty(), looked '
                        'ahead via GPBelief.uncertainty_after -- '
                        'value-independent, so computable before driving or '
                        'measuring anything) reaches this threshold, instead '
                        'of always driving the whole refined candidate every '
                        'round. Falls back to full execution for a round '
                        'whose entire candidate does not reach the threshold. '
                        'Default: None (unchanged full-execution behaviour).')
    p.add_argument('--ckpt', type=str, default=None,
                   help='Start-conditioned CFM checkpoint (default: '
                        'transfer/netz2d_startpunkt.pt, same as '
                        'run_mission_eval.py).')
    p.add_argument('--belief_kernel', type=str, default='rbf', choices=['rbf', 'ak', 'np'],
                   help="Belief model: 'rbf' (GP, stationary, lengthscale 0.08, "
                        "previous behaviour), 'ak' (GP with Attentive Kernel, "
                        "common/ak_belief.py, refitted after every round's "
                        "measurements) or 'np' (pretrained neural process, "
                        "common/np_belief.py). Default output gets an '_ak' / "
                        "'_np' suffix.")
    p.add_argument('--ak_fit_iters', type=int, default=200,
                   help='Marginal-likelihood Adam steps per belief update (AK only).')
    p.add_argument('--np_ckpt', type=str, default=None,
                   help='Neural-process checkpoint (NP only; default: newest '
                        'checkpoints/belief_np_date_*_final.pt).')
    p.add_argument('--start_pos', type=str, default='0.5,0.5')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default=None,
                   help="Default: plots/sample_annealed_mission[_ak]_<kappa "
                        "schedule>.png, e.g. ..._k1.2-0.05.png, so runs with "
                        "different schedules or kernels never overwrite each other.")
    a = p.parse_args()

    device = a.device
    labels = a.shape_names.split(',') if a.shape_names else None
    names, truths = load_truth(labels=labels, n=a.shapes, split='val',
                               resolution=rsc.TRUTH_RES, device=device)
    if a.kappas:
        kappas = [float(v) for v in a.kappas.split(',')]
        if len(kappas) != a.n_rounds:
            raise ValueError(f"--kappas has {len(kappas)} values, need {a.n_rounds} "
                             f"(= --n_rounds)")
    else:
        kappas = [kappa_schedule(r, a.n_rounds, a.kappa0, a.kappa1)
                  for r in range(a.n_rounds)]
    if a.belief_kernel == 'np' and a.ig_threshold is not None:
        raise ValueError('--ig_threshold needs a look-ahead of the uncertainty, '
                         'which the NP belief cannot provide')
    if a.out is None:
        kernel_tag = '' if a.belief_kernel == 'rbf' else f'_{a.belief_kernel}'
        kappa_tag = ('kappas' + '-'.join(f'{k:g}' for k in kappas) if a.kappas
                     else f'k{a.kappa0:g}-{a.kappa1:g}')
        a.out = os.path.join(_here, 'plots',
                             f'sample_annealed_mission{kernel_tag}_{kappa_tag}.png')

    print(f"\n=== Sample replanning mission: CFM init + TSVEC-SVGD refine, "
          f"{a.n_rounds} rounds, kappa " + ' -> '.join(f'{k:g}' for k in kappas) +
          f", condition='{CONDITION}' ===")
    print(f"  Shapes: {', '.join(names)}  |  device={device}  |  "
          f"n_init={a.n_init}  n_iters={a.n_iters}\n")

    args, _svgd_iters, cfg_weight = vr.build_strategy_args(STRATEGY_NAME, device)
    args.belief_kernel, args.ak_fit_iters = a.belief_kernel, a.ak_fit_iters
    args.np_ckpt = a.np_ckpt
    planner = PLANNER_BUILDERS['particles'](a.ckpt or DEFAULT_CKPT, device)
    if not planner.start_cond:
        raise RuntimeError('checkpoint is not start-conditioned; this mission needs start=.')
    ee = ExploreExploitErgodic(device=device)
    B_np = rsc.basis_matrix(rsc.NXI, rsc.N_POINTS, rsc.DEGREE)
    bs = BatchedTsvecSvgdTorch(B_np, device)
    B32 = torch.as_tensor(B_np, dtype=torch.float32, device=device)
    start_pos = tuple(float(v) for v in a.start_pos.split(','))

    cmap = white_inferno()
    colors = round_colors(a.n_rounds)
    n_cols = a.n_rounds + 1                   # + 1 summary panel (ground truth)
    fig, axes = plt.subplots(len(names), n_cols,
                             figsize=(3.4 * n_cols, 3.7 * len(names)),
                             facecolor='white', squeeze=False)

    for i, (name, truth) in enumerate(zip(names, truths)):
        print(f"  [{i + 1}/{len(names)}] {name} ...")
        segments, phis, infos = run_mission(name, truth, planner, bs, ee, B32, args,
                                            device, a.n_rounds, a.n_init, a.n_iters,
                                            kappas, start_pos, a.seed + i,
                                            ig_threshold=a.ig_threshold)
        for r in range(a.n_rounds):
            kap = kappas[r]
            plot_round(axes[i, r], phis[r], segments[r], kap, colors[r],
                      row_label=name if r == 0 else None,
                      col_title=f'Round {r}' if i == 0 else None,
                      cmap=cmap, info=infos[r])
        plot_summary(axes[i, a.n_rounds], truth.detach().cpu(), segments, colors, kappas,
                    col_title='All rounds (ground truth)' if i == 0 else None,
                    cmap=cmap)

    exec_note = (f'replan at dH>={a.ig_threshold:g} info gain' if a.ig_threshold
                is not None else 'full execution per round')
    kernel_note = {'rbf': '', 'ak': ', Attentive-Kernel belief',
                   'np': ', neural-process belief'}[a.belief_kernel]
    fig.suptitle('CFM init + TSVEC-SVGD refine, kappa annealed '
                 + ' → '.join(f'{k:g}' for k in kappas) + f' per round, {exec_note}'
                 f'{kernel_note} (sample, starting from zero prior knowledge)',
                 fontsize=12.5, color='#1A1A2E')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor='white', bbox_inches='tight')
    print(f"\n  Saved: {a.out}\n")


if __name__ == '__main__':
    main()
