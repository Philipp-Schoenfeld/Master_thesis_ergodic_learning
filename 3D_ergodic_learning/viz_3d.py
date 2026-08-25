r"""
viz_3d.py
=========
3D counterpart of the plotting helpers in the 2D runners.

The project's visual conventions are kept exactly (white ground, WHITE_INFERNO
density, deep blue ground truth #1565C0, neon green generation #00C853, dark
particles #444444, grey spines). What changes is that a heatmap cannot be
`imshow`n onto a 3D axis, so the density is rendered two ways at once:

* as a **voxel scatter** of the cells above a threshold, sized and alpha'd by
  density — this shows where the mass actually is in space;
* as a **shadow projection** onto the z = 0 floor, which is what makes a planar
  target readable at a glance and lets a 3D figure be compared against the 2D
  reference figure directly.

The second one is the reason this file exists rather than a few inline calls:
for planar data the floor projection *is* the 2D picture, so the port can be
eyeballed against the old results without extra tooling.
"""

import matplotlib
import matplotlib.colors as _mcolors
import matplotlib.pyplot as plt
import numpy as np

# Same construction as the 2D runners: inferno with the low end faded to white.
_inferno_colors = plt.colormaps['inferno'](np.linspace(0.0, 1.0, 256))
_n_white = 40
for _i in range(_n_white):
    _t = _i / _n_white
    _inferno_colors[_i] = (1 - _t) * np.array([1, 1, 1, 1]) + _t * _inferno_colors[_n_white]
WHITE_INFERNO = _mcolors.LinearSegmentedColormap.from_list('white_inferno', _inferno_colors)

GT_BLUE = '#1565C0'
GEN_GREEN = '#00C853'
PARTICLE_GREY = '#444444'
TEXT_DARK = '#1A1A2E'
TEXT_MID = '#555'
SPINE = '#ccc'


def cp_to_bspline(cps, pts=512, deg=5):
    """Control points -> dense curve. Works for any coordinate dimension."""
    from obstacles import bspline_basis_matrix
    B = bspline_basis_matrix(cps.shape[0], pts, deg)
    return B @ cps


def style_axes3d(ax, title=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel('x', fontsize=7, color=TEXT_MID, labelpad=-6)
    ax.set_ylabel('y', fontsize=7, color=TEXT_MID, labelpad=-6)
    ax.set_zlabel('z', fontsize=7, color=TEXT_MID, labelpad=-6)
    ax.tick_params(labelsize=5.5, colors=TEXT_MID, pad=-2)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor('white')
        pane.pane.set_edgecolor(SPINE)
        pane.pane.set_alpha(1.0)
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')
    if title:
        ax.set_title(title, fontsize=9, color=TEXT_DARK, pad=2)


def draw_density(ax, volume, max_points=4000, floor_shadow=True, seed=0):
    """Voxel scatter of the density plus an optional floor projection.

    `max_points` caps the scatter: a 64^3 volume has 262k cells and matplotlib
    will not render that usefully. Cells are sampled with probability
    proportional to density, so the visible subset still reflects the mass.
    """
    if volume is None:
        return
    R = volume.shape[-1]
    v = np.asarray(volume, dtype=np.float64)
    if v.max() > 0:
        v = v / v.max()

    if floor_shadow:
        # Max over z = the silhouette. For planar data this reproduces the 2D
        # heatmap exactly, which is what makes the two figure sets comparable.
        shadow = v.max(axis=0)
        ax.contourf(np.linspace(0, 1, R), np.linspace(0, 1, R), shadow,
                    zdir='z', offset=0.0, levels=16, cmap=WHITE_INFERNO,
                    vmin=0, vmax=1, alpha=0.45, zorder=0)

    occ = np.argwhere(v > 0.05)
    if len(occ) == 0:
        return
    w = v[occ[:, 0], occ[:, 1], occ[:, 2]]
    if len(occ) > max_points:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(occ), size=max_points, replace=False, p=w / w.sum())
        occ, w = occ[pick], w[pick]

    zc = occ[:, 0] / (R - 1)
    yc = occ[:, 1] / (R - 1)
    xc = occ[:, 2] / (R - 1)
    ax.scatter(xc, yc, zc, c=w, cmap=WHITE_INFERNO, vmin=0, vmax=1,
               s=3.0, alpha=0.30, linewidths=0, depthshade=False, zorder=1)


def draw_particles(ax, particles, max_points=400, seed=0):
    """Conditioning particles as small dark dots."""
    if particles is None:
        return
    p = np.asarray(particles)
    if len(p) > max_points:
        rng = np.random.default_rng(seed)
        p = p[rng.choice(len(p), size=max_points, replace=False)]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=PARTICLE_GREY, s=4, alpha=0.28,
               linewidths=0, depthshade=False, zorder=2)


def draw_trajectories(ax, base, gen_cps, bspline_pts=512, bspline_deg=5,
                      floor_shadow=True):
    """Ground truth (blue) and generated (green) curves, project-standard style."""
    if base is not None and len(base) >= 6:
        c = cp_to_bspline(np.asarray(base), bspline_pts, bspline_deg)
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=GT_BLUE, lw=2.0, alpha=0.9,
                label='Ground Truth', zorder=4)
        if floor_shadow:
            ax.plot(c[:, 0], c[:, 1], zs=0.0, zdir='z', color=GT_BLUE,
                    lw=0.9, alpha=0.25, zorder=3)

    for i, cp in enumerate(gen_cps if gen_cps is not None else []):
        a = 0.95 if i == 0 else 0.3
        cp = np.asarray(cp)
        if len(cp) >= 6:
            c = cp_to_bspline(cp, bspline_pts, bspline_deg)
            ax.plot(c[:, 0], c[:, 1], c[:, 2], color=GEN_GREEN, lw=2.2, alpha=a,
                    label='Generated' if i == 0 else '', zorder=5)
            if floor_shadow and i == 0:
                ax.plot(c[:, 0], c[:, 1], zs=0.0, zdir='z', color=GEN_GREEN,
                        lw=0.9, alpha=0.25, zorder=3)
        ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], color=GEN_GREEN, s=7,
                   alpha=max(0.1, a * 0.6), linewidths=0, depthshade=False,
                   zorder=5)


SENSOR_ORANGE = '#EF6C00'


def draw_frames(ax, curve, R, n_arrows=14, length=0.09, axis=2):
    """Sensor axis as arrows along the curve.

    Only the optical axis is drawn, not the full triad: three arrows per sample
    turns a 25-panel grid into noise, and the optical axis is the one the
    pointing objective is about. Arrows are subsampled so spacing stays readable
    regardless of how densely the curve was rendered.
    """
    if R is None:
        return
    curve = np.asarray(curve)
    R = np.asarray(R)
    T = len(curve)
    idx = np.linspace(0, T - 1, min(n_arrows, T)).astype(int)
    p = curve[idx]
    a = R[idx][:, :, axis]
    ax.quiver(p[:, 0], p[:, 1], p[:, 2],
              a[:, 0], a[:, 1], a[:, 2],
              length=length, normalize=True, color=SENSOR_ORANGE,
              linewidth=1.1, alpha=0.85, arrow_length_ratio=0.35, zorder=6)


def panel(ax, base, gen_cps, particles, volume, title, obstacle=None,
          bspline_pts=512, bspline_deg=5, elev=22, azim=-58, floor_shadow=True,
          gen_R=None, base_R=None):
    """One complete holdout panel.

    `gen_R` / `base_R` are (T, 3, 3) frames along the *rendered* curve; pass
    None to keep the position-only picture.
    """
    ax.set_facecolor('white')
    ax.view_init(elev=elev, azim=azim)
    draw_density(ax, volume, floor_shadow=floor_shadow)
    draw_particles(ax, particles)
    if obstacle is not None:
        obstacle.draw(ax)
    draw_trajectories(ax, base, gen_cps, bspline_pts, bspline_deg, floor_shadow)

    if gen_R is not None and gen_cps is not None and len(gen_cps):
        draw_frames(ax, cp_to_bspline(np.asarray(gen_cps[0]), bspline_pts,
                                      bspline_deg), gen_R)
    if base_R is not None and base is not None:
        draw_frames(ax, cp_to_bspline(np.asarray(base), bspline_pts, bspline_deg),
                    base_R, length=0.07)

    style_axes3d(ax, title)


def save_grid(panels, save_path, suptitle, max_cols=5, figscale=4.6, dpi=150):
    """Render a grid of panels. `panels` is a list of dicts of `panel` kwargs."""
    n = len(panels)
    if n == 0:
        return
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(figscale * n_cols, figscale * 1.05 * n_rows),
                     facecolor='white')
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', color=TEXT_DARK, y=0.995)

    for i, kw in enumerate(panels):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        ax.set_facecolor('white')
        panel(ax, **kw)
        if i == 0:
            ax.legend(frameon=True, fontsize=6.5, loc='upper right',
                      facecolor='white', edgecolor='#ddd', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved -> {save_path}")
