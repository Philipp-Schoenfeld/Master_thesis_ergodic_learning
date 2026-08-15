"""
shape_rasterizer.py
===================
Utilities to generate continuous 2D density functions from shape descriptions.

Core idea: sample many points densely from a shape boundary or interior,
then create a small-bandwidth GMM (KDE-style) so the components merge into a
visually continuous density — NOT scattered blobs.

sigma ≈ 0.010-0.015 → looks like a solid stroke
sigma ≈ 0.020-0.030 → looks like a filled region
"""

import io
import numpy as np
import os
import matplotlib
if 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle
from matplotlib.collections import PatchCollection


# ─────────────────────────────────────────────────────────────────────────────
#  Point-cloud generators
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_mask(fig, threshold=200):
    """Render a matplotlib figure to a binary mask (dark pixels = shape)."""
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)
    gray = buf[..., :3].mean(axis=2)
    plt.close(fig)
    return gray < threshold   # True where shape is


def render_text(char, n_points=250, font_size=72, dpi=100, fontproperties=None):
    """
    Render a text character and return uniformly sampled interior points.
    Points are in [0.05, 0.95]^2.
    """
    fig, ax = plt.subplots(figsize=(1, 1), dpi=dpi, facecolor='white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    if fontproperties is not None:
        ax.text(0.5, 0.48, char, ha='center', va='center',
                color='black', fontproperties=fontproperties)
    else:
        ax.text(0.5, 0.48, char, ha='center', va='center',
                fontsize=font_size, fontweight='bold', color='black',
                fontfamily='DejaVu Sans')
    fig.tight_layout(pad=0)
    mask = _fig_to_mask(fig)
    H, W = mask.shape
    ys, xs = np.where(mask)
    pts = np.column_stack([xs / W, 1.0 - ys / H])
    pts = pts[(pts[:, 0] > 0.02) & (pts[:, 0] < 0.98) &
              (pts[:, 1] > 0.02) & (pts[:, 1] < 0.98)]
    if len(pts) < 10:
        return np.array([[0.5, 0.5]])
    
    # Even stride sampling to prevent clustering
    step = max(1, len(pts) // n_points)
    pts = pts[::step][:n_points]
    # Add tiny noise to break grid moire patterns
    pts += np.random.default_rng(42).normal(0, 0.003, pts.shape)
    return pts


def render_filled_polygon(vertices, n_points=250, dpi=100):
    """
    Render a filled polygon and return sampled interior points.
    vertices : (N, 2) array in [0,1]^2
    """
    fig, ax = plt.subplots(figsize=(1, 1), dpi=dpi, facecolor='white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    poly = MplPolygon(vertices, closed=True, fc='black', ec='black')
    ax.add_patch(poly)
    fig.tight_layout(pad=0)
    mask = _fig_to_mask(fig)
    H, W = mask.shape
    ys, xs = np.where(mask)
    pts = np.column_stack([xs / W, 1.0 - ys / H])
    if len(pts) < 10:
        return np.array([[0.5, 0.5]])
        
    # Even stride sampling
    step = max(1, len(pts) // n_points)
    pts = pts[::step][:n_points]
    pts += np.random.default_rng(42).normal(0, 0.003, pts.shape)
    return pts


def sample_boundary(vertices, n_points=250, closed=True):
    """
    Sample points uniformly along a polyline boundary.
    Produces outline/stroke shapes rather than filled ones.
    """
    verts = np.array(vertices)
    if closed:
        verts = np.vstack([verts, verts[0]])
    segs = np.diff(verts, axis=0)
    lengths = np.linalg.norm(segs, axis=1)
    total = lengths.sum()
    cum = np.concatenate([[0], np.cumsum(lengths)])
    ts = np.linspace(0, total, n_points)
    pts = []
    for t in ts:
        i = np.searchsorted(cum, t, side='right') - 1
        i = min(i, len(segs) - 1)
        frac = (t - cum[i]) / (lengths[i] + 1e-12)
        pts.append(verts[i] + frac * segs[i])
    return np.array(pts)


def star_polygon(n_tips, r_outer=0.32, r_inner=0.14,
                 cx=0.5, cy=0.5, start_deg=90):
    """Generate star polygon vertices (alternating outer/inner tips)."""
    angles = []
    for k in range(2 * n_tips):
        deg = start_deg + k * 180.0 / n_tips
        angles.append(np.deg2rad(deg))
    verts = []
    for k, a in enumerate(angles):
        r = r_outer if k % 2 == 0 else r_inner
        verts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.array(verts)


def heart_curve(n=300):
    """Parametric heart curve sampled uniformly in [0,1]^2."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    x = (x - x.min()) / (x.max() - x.min()) * 0.72 + 0.14
    y = (y - y.min()) / (y.max() - y.min()) * 0.72 + 0.14
    return np.column_stack([x, y])


def circle_boundary(r=0.30, cx=0.5, cy=0.5, n=250):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def ellipse_boundary(rx=0.35, ry=0.22, cx=0.5, cy=0.5, n=250):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])


def lissajous_curve(a=3, b=2, delta=np.pi/4, n=300):
    t = np.linspace(0, 2 * np.pi, n)
    x = 0.5 + 0.4 * np.sin(a * t + delta)
    y = 0.5 + 0.4 * np.sin(b * t)
    return np.column_stack([x, y])


def figure_eight(n=300):
    t = np.linspace(0, 2 * np.pi, n)
    x = 0.5 + 0.38 * np.sin(t)
    y = 0.5 + 0.28 * np.sin(2 * t)
    return np.column_stack([x, y])


def spiral_curve(turns=2.5, n=300):
    t = np.linspace(0, turns * 2 * np.pi, n)
    r = 0.06 + 0.36 * t / t.max()
    return np.column_stack([0.5 + r * np.cos(t), 0.5 + r * np.sin(t)])


def random_smooth_polygon(n_verts=7, rng=None, r_mean=0.30, r_var=0.08,
                          n_points=250):
    """Organic random closed polygon via smoothed random radii."""
    if rng is None:
        rng = np.random.default_rng()
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_verts))
    radii  = np.clip(rng.normal(r_mean, r_var, n_verts), 0.08, 0.44)
    # close and interpolate smoothly
    angles = np.append(angles, angles[0] + 2 * np.pi)
    radii  = np.append(radii, radii[0])
    t_fine = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    r_fine = np.interp(t_fine, np.mod(angles, 2 * np.pi + 1e-9), radii)
    return np.column_stack([0.5 + r_fine * np.cos(t_fine),
                            0.5 + r_fine * np.sin(t_fine)])


# ─────────────────────────────────────────────────────────────────────────────
#  GMM factory
# ─────────────────────────────────────────────────────────────────────────────

def points_to_gmm(points, sigma=0.012):
    """
    Convert a point cloud to a dense-KDE GMM shape definition.

    With sigma ≈ 0.010-0.015 the components overlap visually, producing a
    smooth, continuous density along the shape rather than discrete blobs.

    Parameters
    ----------
    points : (N, 2) array in [0,1]^2
    sigma  : Gaussian std (smaller = sharper stroke)

    Returns
    -------
    dict with keys 'means', 'covs', 'weights' compatible with make_pdf_and_score()
    """
    pts = np.array(points, dtype=float)
    # Clip to safe interior to avoid boundary artefacts
    pts = np.clip(pts, 0.02, 0.98)
    s2 = float(sigma) ** 2
    cov = [[s2, 0.0], [0.0, s2]]
    return {
        'means':   pts.tolist(),
        'covs':    [cov] * len(pts),
        'weights': [1.0] * len(pts),
    }
