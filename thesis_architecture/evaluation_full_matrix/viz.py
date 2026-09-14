r"""
viz.py
======
Plotting nach der verbindlichen Optik aus `CLAUDE.md`: weisser Hintergrund,
`WHITE_INFERNO`-Dichte bei alpha=0.55, generierte Bahn in Neongruen
(`#00C853`). Eigenstaendig statt aus `apply_cfm_belief._white_inferno`
importiert — die dortigen Namen sind mit fuehrendem Unterstrich
modul-privat, und die Konvention steht ohnehin verbindlich in `CLAUDE.md`,
nicht nur in dieser einen Datei.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def white_inferno():
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mcolors.LinearSegmentedColormap.from_list('white_inferno', inf)


CMAP = white_inferno()


def style_axes(ax):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(alpha=0.2, color='#ccc')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')


def draw_density(ax, truth):
    ax.imshow(np.asarray(truth), origin='lower', extent=[0, 1, 0, 1],
              cmap=CMAP, alpha=0.55, vmin=0.0)


def draw_curve(ax, curve, first=True, color='#00C853'):
    c = np.asarray(curve)
    ax.plot(c[:, 0], c[:, 1], color=color,
            lw=2.2 if first else 1.4, alpha=0.95 if first else 0.3)


def plot_single_trajectory(truth, curve, out_path, title='', particles=None):
    """Eine Bahn ueber ihrer Zieldichte — die Pro-Lauf-Visualisierung."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2), facecolor='white')
    style_axes(ax)
    draw_density(ax, truth)
    if particles is not None and len(particles):
        p = np.asarray(particles)
        ax.scatter(p[:, 0], p[:, 1], s=6, alpha=0.3, color='#444444')
    draw_curve(ax, curve)
    ax.set_title(title, color='#1A1A2E', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)


def plot_holdout_panel(items, out_path, ncols=6, title=''):
    """Gitter ueber mehrere Formen — `items`: Liste von (name, truth, curve).

    Eine Zeile fuer alle 24 Holdout-Formen einer (Methode, Variante,
    Wissensstufe): der von der Auswertungs-Vorgabe geforderte Ueberblick ueber
    die volle Holdout-Menge, nicht nur eine Einzelform.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(items)
    ncols = min(ncols, max(n, 1))
    nrows = max(1, -(-n // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.3 * nrows),
                             facecolor='white', squeeze=False)
    for i, (name, truth, curve) in enumerate(items):
        ax = axes[i // ncols][i % ncols]
        style_axes(ax)
        draw_density(ax, truth)
        if curve is not None:
            draw_curve(ax, curve)
        ax.set_title(name, color='#555', fontsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')
    fig.suptitle(title, color='#1A1A2E', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, facecolor='white')
    plt.close(fig)
