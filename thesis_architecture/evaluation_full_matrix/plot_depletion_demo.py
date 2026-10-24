r"""
plot_depletion_demo.py -- effect of the TRACE footprint-overlap depletion on one refinement
==========================================================================================
Same start path (a loop driven twice) and the same target density, refined with
`sun_refine.run_batch` at several depletion strengths eta. The last panel shows the depleted
density phi_k(x) = phi(x) * prod_{j<k}(1 - eta * v(x, x_j)) as seen by the LAST state of the
eta-path (Zheng et al., arXiv:2608.02304, Eq. 3-4).
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, _arch, os.path.join(_arch, 'exploration')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402

from common import sun_refine as sr                                  # noqa: E402
from plot_mission_step_grid import white_inferno, INK, MUTED, SPINE  # noqa: E402


def depleted_density(phi, pos, eta, sigma):
    """phi_k on the grid for k = last state (same gap/stride rule as the solver)."""
    R = phi.shape[0]
    ys, xs = np.mgrid[0:R, 0:R] / (R - 1)
    k = len(pos) - 1
    js = [j for j in range(k) if j < k - sr.DEPLETION_GAP and j % sr.DEPLETION_STRIDE == 0]
    D = np.ones_like(phi)
    for j in js:
        v = np.exp(-((xs - pos[j, 0]) ** 2 + (ys - pos[j, 1]) ** 2) / (2 * sigma ** 2))
        D *= 1.0 - eta * v
    return phi * D


def style(ax, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(alpha=0.2)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(SPINE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--etas', default='0,0.5,0.9')
    ap.add_argument('--sigma', type=float, default=0.06)
    ap.add_argument('--n_iters', type=int, default=400)
    ap.add_argument('--out', default=os.path.join(_here, 'results', 'depletion_demo.png'))
    a = ap.parse_args()
    etas = [float(e) for e in a.etas.split(',')]

    t = np.linspace(0, 1, 100)
    curve = np.stack([0.5 + 0.3 * np.cos(4 * np.pi * t), 0.5 + 0.3 * np.sin(4 * np.pi * t)], 1)
    yy, xx = np.mgrid[0:64, 0:64] / 63
    phi = (np.exp(-((xx - .5) ** 2 + (yy - .8) ** 2) / (2 * .08 ** 2))
           + 0.7 * np.exp(-((xx - .25) ** 2 + (yy - .3) ** 2) / (2 * .08 ** 2)))
    start = np.array([[.8, .5]])
    cmap = white_inferno()

    fig, axes = plt.subplots(1, len(etas) + 1, facecolor='white', figsize=(3.2 * (len(etas) + 1), 3.6))
    paths = {}
    for ax, eta in zip(axes, etas):
        pos = sr.run_batch(curve[None], phi, start, a.n_iters, 25,
                           depletion_eta=eta, depletion_sigma=a.sigma)['final_pos'][0]
        paths[eta] = pos
        ax.imshow(phi, origin='lower', extent=(0, 1, 0, 1), cmap=cmap, alpha=0.55)
        ax.plot(curve[:, 0], curve[:, 1], color='#999999', lw=1.0, ls='--', zorder=2)
        ax.plot(pos[:, 0], pos[:, 1], color='#00C853', lw=2.2, alpha=0.95, zorder=3)
        ax.plot(*pos[0], 'o', color=INK, ms=4, zorder=4)
        style(ax, f'eta = {eta:g}' + (' (off)' if eta == 0 else ''))
    eta_d = max(e for e in etas)
    eta_d = 0.5 if 0.5 in etas else eta_d
    ax = axes[-1]
    phi_k = depleted_density(phi, paths[eta_d], eta_d, a.sigma)
    ax.imshow(phi_k, origin='lower', extent=(0, 1, 0, 1), cmap=cmap, alpha=0.55)
    ax.plot(paths[eta_d][:, 0], paths[eta_d][:, 1], color='#00C853', lw=1.2, alpha=0.5)
    ax.plot(*paths[eta_d][-1], 's', color=INK, ms=4, zorder=4)
    style(ax, f'depleted phi_k at last state (eta = {eta_d:g})')
    fig.suptitle('TRACE footprint-overlap depletion in the FM-Stein refiner: loop driven twice, '
                 'bimodal target', fontsize=11, color=INK)
    fig.text(0.5, 0.005, 'dashed grey: start path   green: refined path   dot: start state',
             ha='center', fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=170, facecolor='white')
    print(a.out)


if __name__ == '__main__':
    main()
