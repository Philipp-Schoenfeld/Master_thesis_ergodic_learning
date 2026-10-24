#!/usr/bin/env python3
r"""
plot_annealed.py
=================
Static visualization of the annealed, multi-round Variant E mission
(`run_mission_annealed` in `run_e.py`) -- a lightweight alternative to
`interactive_sim.py` for machines where the live GUI (matplotlib widgets +
MuJoCo + CFM net, redrawn on a timer) is too heavy.

Runs `n_rounds` complete replanning rounds per shape, starting from a belief
with no prior measurements at all ("none known": `n_prior=0`, so mu=0 and
sigma is uniform everywhere before round 0). Each round plans and drives one
full trajectory against `Phi = mu + kappa*sigma`, with kappa falling from
`--kappa0` to `--kappa1` across the rounds (`kappa_schedule`, the same
exploration-to-exploitation schedule used throughout `exploration/`), then
folds that round's measurements into the belief before the next round is
planned.

One figure (one PNG) per run, laid out as a grid: one column per round, one
row per shape. Each panel shows exactly the round it belongs to -- the
COMPLETE trajectory driven that round (not cropped to the [0, 1] domain: the
gradient planner is not constrained to stay inside it, so a panel's axes
extend to whatever the driven curve actually reaches) against the belief-
based target density Phi that round was planned against, not the ground
truth. Round 0's panel is therefore genuinely flat -- with zero prior
measurements mu=0 and sigma is uniform everywhere, so Phi carries no
structure yet (the degenerate case `common.acquisition.ucb_density`
documents); each following panel shows how Phi sharpens as the belief from
the previous rounds' measurements comes in, and kappa falling makes mu
dominate over sigma.

Round colors are an ordered progression from cool violet (round 0,
exploration-heavy) to the project's canonical neon green (last round,
exploitation-heavy) -- the rounds form a sequence, not independent
categories, so the color encodes that order rather than cycling through an
arbitrary categorical palette.

    python plot_annealed.py --shapes 2 --n_rounds 4
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors   # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..')))

from common.data import load_truth                                   # noqa: E402
from common.planner import GradientPlanner                            # noqa: E402

from run_e import run_mission_annealed                                # noqa: E402


# ===========================================================================
# Style (project convention, see CLAUDE.md "Visualization Aesthetics")
# ===========================================================================

def white_inferno():
    """White fading into standard inferno -- the project's density colormap."""
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
    """Axis limits covering the [0, 1] domain AND the whole driven curve.

    The gradient planner has no domain constraint, so a round's curve can
    leave [0, 1] entirely (visible as the sprawling early rounds when
    starting from zero prior knowledge). Clipping the view to [0, 1], as
    `common/belief.py`'s own convention does for the density grid, would
    silently cut off the part of the trajectory that left the domain --
    exactly the "only part of the execution" complaint this view avoids.
    """
    pts = curve.detach().cpu().numpy()
    x0, x1 = min(0.0, pts[:, 0].min()), max(1.0, pts[:, 0].max())
    y0, y1 = min(0.0, pts[:, 1].min()), max(1.0, pts[:, 1].max())
    px, py = pad_frac * (x1 - x0), pad_frac * (y1 - y0)
    return (x0 - px, x1 + px), (y0 - py, y1 + py)


# Ordered round colors: cool violet (pure exploration) -> the project's
# canonical neon green (pure exploitation). An ordered progression, not an
# arbitrary categorical cycle, because the rounds themselves are ordered by
# falling kappa.
ROUND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    'explore_to_exploit', ['#5E35B1', '#1E88E5', '#00ACC1', '#00C853'])


def round_colors(n_rounds):
    if n_rounds == 1:
        return ['#00C853']
    return [mcolors.to_hex(ROUND_CMAP(t))
            for t in np.linspace(0.0, 1.0, n_rounds)]


# ===========================================================================
# Mission + plot
# ===========================================================================

def plot_round(ax, phi, seg, kap, col, row_label=None, col_title=None, cmap=None):
    """One panel: the belief-based Phi this round was planned against, and
    the complete trajectory driven that round -- nothing cropped, nothing
    from other rounds overlaid."""
    im = phi.numpy()
    ax.imshow(im, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
             vmin=0, vmax=1.0, alpha=0.55, zorder=1)

    pts = seg.detach().cpu().numpy()
    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2.2, alpha=0.95, zorder=3)
    ax.plot(pts[0, 0], pts[0, 1], 'o', color=col, ms=5, zorder=4)

    style(ax, *full_view(seg))
    if col_title:
        ax.set_title(col_title, fontsize=10, color='#1A1A2E')
    if row_label:
        ax.set_ylabel(row_label, fontsize=10.5, color='#1A1A2E')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=2,
                   help='How many holdout shapes to render, one panel each.')
    p.add_argument('--n_rounds', type=int, default=4)
    p.add_argument('--kappa0', type=float, default=4.0)
    p.add_argument('--kappa1', type=float, default=0.5)
    p.add_argument('--grid_res', type=int, default=48)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--steps', type=int, default=120,
                   help='Gradient-planner steps per round. Kept lower than '
                        'the 150-step default elsewhere so this stays fast '
                        'on a weak machine.')
    p.add_argument('--sensor_radius', type=float, default=0.03)
    p.add_argument('--noise', type=float, default=0.02)
    p.add_argument('--gp_noise', type=float, default=0.05)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'plots',
        'annealed_rounds.png'))
    a = p.parse_args()

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    print(f"\n=== Variant E, annealed: {a.n_rounds} rounds, "
          f"kappa {a.kappa0} -> {a.kappa1}, starting from zero prior ===")
    print(f"  Shapes: {', '.join(names)}\n")

    def mk(seed):
        return GradientPlanner(metric=a.metric, steps=a.steps, seed=seed)

    cmap = white_inferno()
    colors = round_colors(a.n_rounds)
    fig, axes = plt.subplots(len(names), a.n_rounds,
                             figsize=(3.4 * a.n_rounds, 3.7 * len(names)),
                             facecolor='white', squeeze=False)

    for i, (name, truth) in enumerate(zip(names, T)):
        print(f"  [{i + 1}/{len(names)}] {name} ...")
        _, row, segments, phis = run_mission_annealed(
            truth, mk, n_rounds=a.n_rounds, n_prior=0, grid_res=a.grid_res,
            n_particles=a.n_particles, noise=a.noise,
            sensor_radius=a.sensor_radius, kappa0=a.kappa0, kappa1=a.kappa1,
            seed=a.seed + i, gp_noise=a.gp_noise, label=f"E-R{a.n_rounds} {name}")
        print(f"        coverage={row['coverage']:.4f}  "
              f"belief_rmse={row['belief_rmse']:.4f}  "
              f"path_len={row['path_len']:.2f}")
        for r in range(a.n_rounds):
            kap = a.kappa0 + (a.kappa1 - a.kappa0) * (r / max(1, a.n_rounds - 1))
            plot_round(axes[i, r], phis[r], segments[r], kap, colors[r],
                      row_label=name if r == 0 else None,
                      col_title=f'Round {r}  (κ={kap:.2f})' if i == 0 else None,
                      cmap=cmap)

    fig.suptitle('Variant E, annealed -- each round against the belief\'s '
                 'own target density, starting from zero prior knowledge',
                 fontsize=12.5, color='#1A1A2E')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor='white', bbox_inches='tight')
    print(f"\n  Saved: {a.out}\n")


if __name__ == '__main__':
    main()
