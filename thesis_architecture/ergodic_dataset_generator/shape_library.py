"""
shape_library.py  (v2 — continuous densities)
=============================================
Each shape is described by a dense GMM where many small-bandwidth Gaussians
(sigma ≈ 0.010-0.015) are placed along the shape boundary or interior.
With enough components they merge into a visually continuous density.

Usage:
    from shape_library import get_shape, PREVIEW_SHAPES, VALIDATION_SHAPES
    shape_def = get_shape('N')          # lazy: builds on first call
    pdf_fn, score_fn = make_pdf_and_score(shape_def)
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

try:
    from .shape_rasterizer import (
        render_text, render_filled_polygon, sample_boundary,
        star_polygon, heart_curve, circle_boundary, ellipse_boundary,
        lissajous_curve, figure_eight, spiral_curve,
        random_smooth_polygon, points_to_gmm,
    )
except ImportError:
    from shape_rasterizer import (
        render_text, render_filled_polygon, sample_boundary,
        star_polygon, heart_curve, circle_boundary, ellipse_boundary,
        lissajous_curve, figure_eight, spiral_curve,
        random_smooth_polygon, points_to_gmm,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Core: GMM → JAX pdf + score
# ─────────────────────────────────────────────────────────────────────────────

def make_pdf_and_score(shape_def):
    """
    Build JAX pdf and score (∇ log p) from a shape_def dict.
    Supports both GMM (means, covs, weights) and analytical segments.
    """
    if shape_def.get('type') == 'analytical':
        base = _make_analytical_pdf_and_score(shape_def['segments'],
                                              shape_def.get('sigma', 0.025))
        return _vielleicht_sockel(base, shape_def)

    w = np.array(shape_def['weights'], dtype=np.float32)
    w = w / w.sum()

    means_j   = jnp.array(shape_def['means'],   dtype=jnp.float32)  # (K, 2)
    covs_j    = jnp.array(shape_def['covs'],    dtype=jnp.float32)  # (K, 2, 2)
    weights_j = jnp.array(w,                    dtype=jnp.float32)  # (K,)

    def _comp(x2d, mean, cov, weight):
        return weight * mvn.pdf(x2d, mean, cov)

    def pdf_fn(x):
        x2d  = x[:2]
        vals = jax.vmap(_comp, in_axes=(None, 0, 0, 0))(
            x2d, means_j, covs_j, weights_j
        )
        return jnp.sum(vals) + 1e-10

    def log_pdf(x):
        return jnp.log(pdf_fn(x))

    score_fn = jax.grad(log_pdf)
    return _vielleicht_sockel((pdf_fn, score_fn), shape_def)


# ---------------------------------------------------------------------------
#  Sockel: eine breite, fast gleichverteilte Komponente unter der Form
# ---------------------------------------------------------------------------
#  Warum das existiert: die Zieldichten, die im Einsatz aus einem Glauben
#  entstehen (mu + kappa*sigma, Niveaumenge, Informationsdichte), haben einen
#  Traeger ueber dem *ganzen* Quadrat und halten in den obersten 30 % ihrer
#  Zellen nur 0,34 bis 0,48 der Masse. Die bisherigen Trainingsdichten kommen
#  auf 0,658 beziehungsweise 0,851 — das Netz sieht im Training also nie
#  etwas, das einer Glaubensdichte auch nur aehnelt.
#
#  Ein Sockel schliesst genau diese Luecke, ohne den ueberwachten Aufbau
#  anzutasten: die Dichte bleibt analytisch, `jax.grad` liefert den Score
#  weiterhin von selbst, und der SVGD-Solver laeuft unveraendert.
#
#      p'(x) = (1-a) * p(x)/Z_p  +  a * g(x)/Z_g
#
#  Beide Anteile werden ueber dem Einheitsquadrat auf Masse 1 normiert. Die
#  zwei Konstanten werden einmal beim Bauen numerisch bestimmt und im
#  shape_def mitgefuehrt, damit die Dichte nach dem Umweg ueber die Datenbank
#  exakt dieselbe ist.

def _gitter_mittel(fn, res=96):
    xs = np.linspace(0.0, 1.0, res)
    gx, gy = np.meshgrid(xs, xs)
    z = np.zeros(res * res, dtype=np.float32)
    pts = jnp.array(np.stack([gx.ravel(), gy.ravel(), z, z], axis=1))
    return float(np.mean(np.array(jax.vmap(fn)(pts))))


def _gauss_mittel(center, sigma, res=96):
    xs = np.linspace(0.0, 1.0, res)
    gx, gy = np.meshgrid(xs, xs)
    d2 = (gx - center[0]) ** 2 + (gy - center[1]) ** 2
    return float(np.mean(np.exp(-d2 / (2.0 * sigma ** 2))))


def _vielleicht_sockel(base, shape_def):
    ped = shape_def.get('pedestal')
    if not ped:
        return base
    base_pdf, _ = base
    a  = float(ped['weight'])
    sg = float(ped['sigma'])
    c  = jnp.array(ped['center'], dtype=jnp.float32)
    zb = float(ped['z_base'])
    zp = float(ped['z_ped'])

    def pdf_fn(x):
        x2d = x[:2]
        g = jnp.exp(-jnp.sum((x2d - c) ** 2) / (2.0 * sg ** 2))
        return (1.0 - a) * base_pdf(x) / zb + a * g / zp + 1e-10

    def log_pdf(x):
        return jnp.log(pdf_fn(x))

    return jax.jit(pdf_fn), jax.jit(jax.grad(log_pdf))


def mit_sockel(base_def, weight, seed):
    """Kopie von `base_def` mit einem Sockel vom Gewicht `weight`."""
    rng = np.random.default_rng(seed)
    sigma  = float(rng.uniform(0.40, 0.95))
    center = [float(v) for v in rng.uniform(0.35, 0.65, size=2)]
    pdf_base, _ = make_pdf_and_score(base_def)
    d = dict(base_def)
    d['pedestal'] = {'weight': float(weight), 'sigma': sigma, 'center': center,
                     'z_base': _gitter_mittel(pdf_base),
                     'z_ped':  _gauss_mittel(center, sigma)}
    return d


@jax.jit
def _jax_dist_to_segment_sq(p, a, b):
    dx = b - a
    l2 = jnp.sum(dx**2)
    l2 = jnp.where(l2 == 0, 1e-8, l2)
    t = jnp.clip(jnp.dot(p - a, dx) / l2, 0.0, 1.0)
    proj = a + t * dx
    return jnp.sum((p - proj)**2)

def _make_analytical_pdf_and_score(segments, sigma=0.025):
    segments_j = jnp.array(segments, dtype=jnp.float32)
    
    def _comp(x2d, seg):
        d2 = _jax_dist_to_segment_sq(x2d, seg[0], seg[1])
        return jnp.exp(-d2 / (2 * sigma**2))
        
    def pdf_fn(x):
        x2d = x[:2]
        vals = jax.vmap(_comp, in_axes=(None, 0))(x2d, segments_j)
        # Using max ensures uniform density without doubling at segment intersections
        return jnp.max(vals) + 1e-10
        
    def log_pdf(x):
        return jnp.log(pdf_fn(x))
        
    score_fn = jax.grad(log_pdf)
    return jax.jit(pdf_fn), jax.jit(score_fn)


def pdf_on_grid(shape_def, resolution=100):
    """Evaluate density on a [0,1]² grid for visualisation → (R,R), gx, gy."""
    pdf_fn, _ = make_pdf_and_score(shape_def)
    xs = np.linspace(0.0, 1.0, resolution)
    ys = np.linspace(0.0, 1.0, resolution)
    gx, gy = np.meshgrid(xs, ys)
    zeros  = np.zeros(resolution ** 2, dtype=np.float32)
    pts    = jnp.array(np.stack([gx.ravel(), gy.ravel(), zeros, zeros], axis=1))
    vals   = jax.vmap(pdf_fn)(pts)
    return np.array(vals).reshape(resolution, resolution), gx, gy


# ─────────────────────────────────────────────────────────────────────────────
#  Shape builders (all return shape_def dicts)
# ─────────────────────────────────────────────────────────────────────────────

_SIGMA_STROKE = 0.011   # thin stroke (letters, outlines)
_SIGMA_FILL   = 0.020   # filled/wide shapes


def _letter(char, filled=True, n=250, sigma=None, fontproperties=None):
    pts = render_text(char, n_points=n, fontproperties=fontproperties)
    if sigma is None:
        sigma = _SIGMA_FILL if filled else _SIGMA_STROKE
    return points_to_gmm(pts, sigma=sigma)


def _outline(pts, n=250, sigma=None):
    if sigma is None:
        sigma = _SIGMA_STROKE
    pts2 = sample_boundary(pts, n_points=n) if pts.shape[0] < n else pts
    return points_to_gmm(pts2, sigma=sigma)


def _filled(pts, n=250):
    return points_to_gmm(pts, sigma=_SIGMA_FILL)


# ── Geometric outlines ────────────────────────────────────────────────────────

def _star_outline(n_tips, n=250):
    verts = star_polygon(n_tips)
    pts   = sample_boundary(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _star_filled(n_tips, n=250):
    verts = star_polygon(n_tips)
    pts   = render_filled_polygon(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_FILL)


def _ring(n_blobs_on_ring=200, r=0.30):
    pts = circle_boundary(r=r, n=n_blobs_on_ring)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _circle_filled(r=0.30, n=300):
    # Random uniform disk
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii  = r * np.sqrt(rng.uniform(0, 1, n))
    pts    = np.column_stack([0.5 + radii * np.cos(angles),
                              0.5 + radii * np.sin(angles)])
    return points_to_gmm(pts, sigma=_SIGMA_FILL)


def _ellipse(rx=0.35, ry=0.22, n=250):
    pts = ellipse_boundary(rx=rx, ry=ry, n=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _rect(x0=0.2, y0=0.2, x1=0.8, y1=0.8, n=250):
    verts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    pts   = sample_boundary(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _triangle(n=250, kind='equilateral'):
    if kind == 'equilateral':
        verts = np.array([[0.5, 0.82], [0.17, 0.18], [0.83, 0.18]])
    elif kind == 'right':
        verts = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85]])
    else:
        verts = np.array([[0.5, 0.85], [0.15, 0.15], [0.85, 0.35]])
    pts = sample_boundary(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _diamond_outline(n=250):
    verts = np.array([[0.5, 0.85], [0.85, 0.5], [0.5, 0.15], [0.15, 0.5]])
    pts   = sample_boundary(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _cross(arm_w=0.12, n=300):
    verts = np.array([
        [0.5 - arm_w, 0.85], [0.5 + arm_w, 0.85],
        [0.5 + arm_w, 0.5 + arm_w], [0.85, 0.5 + arm_w],
        [0.85, 0.5 - arm_w], [0.5 + arm_w, 0.5 - arm_w],
        [0.5 + arm_w, 0.15], [0.5 - arm_w, 0.15],
        [0.5 - arm_w, 0.5 - arm_w], [0.15, 0.5 - arm_w],
        [0.15, 0.5 + arm_w], [0.5 - arm_w, 0.5 + arm_w],
    ])
    pts = render_filled_polygon(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_FILL)


def _heart(n=250):
    pts = heart_curve(n=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _figure_eight(n=250):
    pts = figure_eight(n=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _lissajous(a=3, b=2, delta=np.pi/4, n=250):
    pts = lissajous_curve(a=a, b=b, delta=delta, n=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _spiral(turns=2.5, n=300):
    pts = spiral_curve(turns=turns, n=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _arrow(n=250):
    verts = np.array([
        [0.12, 0.42], [0.60, 0.42], [0.60, 0.25],
        [0.88, 0.50], [0.60, 0.75], [0.60, 0.58], [0.12, 0.58],
    ])
    pts = render_filled_polygon(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_FILL)


def _moon(n=250):
    # Crescent = big circle minus smaller offset circle, approximated as boundary
    t   = np.linspace(0, 2 * np.pi, n // 2, endpoint=False)
    outer = np.column_stack([0.5 + 0.32 * np.cos(t), 0.5 + 0.32 * np.sin(t)])
    # inner arc (visible part of crescent)
    t2 = np.linspace(np.pi * 0.15, np.pi * 1.85, n // 2)
    inner = np.column_stack([0.62 + 0.28 * np.cos(t2), 0.5 + 0.28 * np.sin(t2)])
    pts = np.vstack([outer, inner])
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _hexagon(n=250):
    angles = np.deg2rad(np.arange(0, 360, 60) + 30)
    verts  = np.column_stack([0.5 + 0.36 * np.cos(angles),
                               0.5 + 0.36 * np.sin(angles)])
    pts = sample_boundary(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _pentagon(n=250):
    angles = np.deg2rad(np.arange(0, 360, 72) + 90)
    verts  = np.column_stack([0.5 + 0.36 * np.cos(angles),
                               0.5 + 0.36 * np.sin(angles)])
    pts = sample_boundary(verts, n_points=n)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _zigzag(peaks=5, n=250):
    xs = np.linspace(0.10, 0.90, peaks * 2 - 1)
    ys = np.array([0.75 if i % 2 == 0 else 0.25 for i in range(len(xs))])
    verts = np.column_stack([xs, ys])
    pts   = sample_boundary(verts, n_points=n, closed=False)
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _wave(n=250, cycles=2):
    t   = np.linspace(0, cycles * 2 * np.pi, n)
    pts = np.column_stack([np.linspace(0.05, 0.95, n),
                           0.5 + 0.30 * np.sin(t)])
    return points_to_gmm(pts, sigma=_SIGMA_STROKE)


def _bimodal(n=250):
    # Two separated blobs — simple but valid continuous density
    rng = np.random.default_rng(7)
    a   = rng.normal([0.30, 0.50], 0.06, (n // 2, 2))
    b   = rng.normal([0.70, 0.50], 0.06, (n // 2, 2))
    pts = np.clip(np.vstack([a, b]), 0.02, 0.98)
    return points_to_gmm(pts, sigma=0.025)


def _rand_gmm(seed, min_comp=3, max_comp=12):
    rng = np.random.default_rng(seed)
    n_comp = rng.integers(min_comp, max_comp + 1)
    means = rng.uniform(0.15, 0.85, size=(n_comp, 2)).tolist()
    covs = []
    weights = []
    for _ in range(n_comp):
        a = rng.uniform(0.001, 0.05)
        b = rng.uniform(0.001, 0.05)
        theta = rng.uniform(0, 2*np.pi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        R = np.array([[c_t, -s_t], [s_t, c_t]])
        D = np.array([[a, 0], [0, b]])
        cov = (R @ D @ R.T).tolist()
        covs.append(cov)
        weights.append(float(rng.uniform(0.1, 1.0)))
    weights = (np.array(weights) / sum(weights)).tolist()
    return {'means': means, 'covs': covs, 'weights': weights}

def _rand_ana_poly(seed, min_verts=3, max_verts=8):
    rng = np.random.default_rng(seed)
    n_verts = rng.integers(min_verts, max_verts + 1)
    pts = rng.uniform(0.1, 0.9, (n_verts, 2))
    segments = []
    path = rng.permutation(n_verts)
    for i in range(n_verts - 1):
        segments.append((pts[path[i]].tolist(), pts[path[i+1]].tolist()))
    extra = rng.integers(0, 3)
    for _ in range(extra):
        i, j = rng.choice(n_verts, 2, replace=False)
        segments.append((pts[i].tolist(), pts[j].tolist()))
    return {'type': 'analytical', 'sigma': 0.025, 'segments': segments}


def _test_gmm_1():
    # The 3-component Gaussian mixture from the user's notebook
    means = [[0.3, 0.5], [0.5, 0.5], [0.7, 0.5]]
    covs = [
        [[0.002, 0.0], [0.0, 0.04]],
        [[0.02, -0.018], [-0.018, 0.02]],
        [[0.002, 0.0], [0.0, 0.04]]
    ]
    weights = [0.34, 0.34, 0.33]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_2():
    # 4-component cross-shaped mixture
    means = [[0.5, 0.2], [0.5, 0.8], [0.2, 0.5], [0.8, 0.5]]
    covs = [
        [[0.01, 0.0], [0.0, 0.03]],
        [[0.01, 0.0], [0.0, 0.03]],
        [[0.03, 0.0], [0.0, 0.01]],
        [[0.03, 0.0], [0.0, 0.01]]
    ]
    weights = [0.25, 0.25, 0.25, 0.25]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_3():
    # Dense bimodal distribution with rotated covariances
    means = [[0.3, 0.3], [0.7, 0.7]]
    covs = [
        [[0.03, 0.02], [0.02, 0.03]],
        [[0.03, -0.02], [-0.02, 0.03]]
    ]
    weights = [0.5, 0.5]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_4():
    # Asymmetric diagonal distribution
    means = [[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]
    covs = [
        [[0.02, 0.015], [0.015, 0.02]],
        [[0.01, 0.0], [0.0, 0.01]],
        [[0.05, 0.0], [0.0, 0.005]]
    ]
    weights = [0.4, 0.2, 0.4]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_5():
    # "ring-like" sparse distribution using 5 wide Gaussians
    means = [
        [0.5, 0.8], [0.2, 0.6], [0.3, 0.2], [0.7, 0.2], [0.8, 0.6]
    ]
    covs = [
        [[0.02, 0.0], [0.0, 0.01]],
        [[0.015, -0.01], [-0.01, 0.015]],
        [[0.015, 0.01], [0.01, 0.015]],
        [[0.015, -0.01], [-0.01, 0.015]],
        [[0.015, 0.01], [0.01, 0.015]],
    ]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_6():
    # 3x3 Grid of Gaussians
    means = [[0.2, 0.2], [0.5, 0.2], [0.8, 0.2],
             [0.2, 0.5], [0.5, 0.5], [0.8, 0.5],
             [0.2, 0.8], [0.5, 0.8], [0.8, 0.8]]
    covs = [[[0.005, 0.0], [0.0, 0.005]]] * 9
    weights = [0.111] * 9
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_7():
    # Two parallel diagonal lines using 6 small Gaussians
    means = [[0.2, 0.3], [0.5, 0.6], [0.8, 0.9],
             [0.3, 0.2], [0.6, 0.5], [0.9, 0.8]]
    cov = [[0.015, 0.01], [0.01, 0.015]]
    covs = [cov] * 6
    weights = [0.166] * 6
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_8():
    # A "U" shape using 3 wide Gaussians
    means = [[0.2, 0.7], [0.5, 0.3], [0.8, 0.7]]
    covs = [
        [[0.01, 0.0], [0.0, 0.05]],
        [[0.05, 0.0], [0.0, 0.01]],
        [[0.01, 0.0], [0.0, 0.05]]
    ]
    weights = [0.33, 0.34, 0.33]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_9():
    # Corner checkerboard pattern with varying scales
    means = [[0.15, 0.15], [0.85, 0.85], [0.15, 0.85], [0.85, 0.15]]
    covs = [
        [[0.02, 0.0], [0.0, 0.02]],
        [[0.02, 0.0], [0.0, 0.02]],
        [[0.005, 0.0], [0.0, 0.005]],
        [[0.005, 0.0], [0.0, 0.005]]
    ]
    weights = [0.35, 0.35, 0.15, 0.15]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_10():
    # Large central blob surrounded by 4 smaller orbiting blobs
    means = [[0.5, 0.5], [0.5, 0.8], [0.5, 0.2], [0.2, 0.5], [0.8, 0.5]]
    covs = [
        [[0.03, 0.0], [0.0, 0.03]],
        [[0.003, 0.0], [0.0, 0.003]],
        [[0.003, 0.0], [0.0, 0.003]],
        [[0.003, 0.0], [0.0, 0.003]],
        [[0.003, 0.0], [0.0, 0.003]]
    ]
    weights = [0.6, 0.1, 0.1, 0.1, 0.1]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_11():
    # Curvy S-like shape using 5 overlapping components
    means = [[0.7, 0.8], [0.5, 0.65], [0.5, 0.5], [0.5, 0.35], [0.3, 0.2]]
    covs = [
        [[0.02, 0.0], [0.0, 0.01]],
        [[0.01, -0.01], [-0.01, 0.02]],
        [[0.02, 0.0], [0.0, 0.01]],
        [[0.01, -0.01], [-0.01, 0.02]],
        [[0.02, 0.0], [0.0, 0.01]]
    ]
    weights = [0.2] * 5
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_12():
    # Spiral of 6 shrinking components
    import math
    means = []
    covs = []
    weights = []
    for i in range(6):
        theta = i * math.pi / 2.5
        r = 0.4 * (1.0 - i * 0.15)
        means.append([0.5 + r * math.cos(theta), 0.5 + r * math.sin(theta)])
        scale = 0.015 * (1.0 - i * 0.1)
        covs.append([[scale, 0.0], [0.0, scale]])
        weights.append(1.0)
    weights = [w / sum(weights) for w in weights]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_13():
    # 3 concentric horizontal bands
    means = [[0.5, 0.2], [0.5, 0.5], [0.5, 0.8]]
    covs = [
        [[0.08, 0.0], [0.0, 0.003]],
        [[0.08, 0.0], [0.0, 0.003]],
        [[0.08, 0.0], [0.0, 0.003]]
    ]
    weights = [0.33, 0.34, 0.33]
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_14():
    # "X" shape using 5 components (1 center, 4 arms)
    means = [[0.5, 0.5], [0.25, 0.75], [0.75, 0.75], [0.25, 0.25], [0.75, 0.25]]
    covs = [
        [[0.01, 0.0], [0.0, 0.01]],
        [[0.01, 0.008], [0.008, 0.01]],
        [[0.01, -0.008], [-0.008, 0.01]],
        [[0.01, -0.008], [-0.008, 0.01]],
        [[0.01, 0.008], [0.008, 0.01]]
    ]
    weights = [0.2] * 5
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_15():
    # Highly scattered random field (8 components)
    rng = np.random.default_rng(42)
    means = rng.uniform(0.1, 0.9, size=(8, 2)).tolist()
    covs = []
    for _ in range(8):
        v = rng.uniform(0.002, 0.015)
        covs.append([[v, 0.0], [0.0, v]])
    weights = [0.125] * 8
    return {'means': means, 'covs': covs, 'weights': weights}


def _test_gmm_16():
    # Dense 4x4 Grid (16 components)
    means = []
    covs = []
    for x in np.linspace(0.2, 0.8, 4):
        for y in np.linspace(0.2, 0.8, 4):
            means.append([x, y])
            covs.append([[0.002, 0.0], [0.0, 0.002]])
    weights = [1.0/16] * 16
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_17():
    # Concentric Rings (12 outer, 4 inner)
    means = []
    covs = []
    for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
        means.append([0.5 + 0.35 * np.cos(theta), 0.5 + 0.35 * np.sin(theta)])
        covs.append([[0.002, 0.0], [0.0, 0.002]])
    for theta in np.linspace(0, 2*np.pi, 4, endpoint=False):
        means.append([0.5 + 0.15 * np.cos(theta), 0.5 + 0.15 * np.sin(theta)])
        covs.append([[0.002, 0.0], [0.0, 0.002]])
    weights = [1.0/16] * 16
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_18():
    # Logarithmic Spiral (20 modes)
    means = []
    covs = []
    for i in range(20):
        t = 0.5 * (i + 1)
        r = 0.05 * np.exp(0.15 * t)
        means.append([0.5 + r * np.cos(t), 0.5 + r * np.sin(t)])
        v = max(0.001, 0.01 - 0.0004 * i)
        covs.append([[v, 0.0], [0.0, v]])
    weights = [1.0/20] * 20
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_19():
    # Fractal Cross (13 modes)
    means = [[0.5, 0.5]]
    covs = [[[0.01, 0.0], [0.0, 0.01]]]
    weights = [1.0]
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for dx, dy in dirs:
        means.append([0.5 + 0.2*dx, 0.5 + 0.2*dy])
        covs.append([[0.002, 0.0], [0.0, 0.002]])
        weights.append(0.5)
        means.append([0.5 + 0.3*dx + 0.1*dy, 0.5 + 0.3*dy + 0.1*dx])
        covs.append([[0.001, 0.0], [0.0, 0.001]])
        weights.append(0.25)
        means.append([0.5 + 0.3*dx - 0.1*dy, 0.5 + 0.3*dy - 0.1*dx])
        covs.append([[0.001, 0.0], [0.0, 0.001]])
        weights.append(0.25)
    weights = [w / sum(weights) for w in weights]
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_20():
    # Parallel Thin Lines (20 modes)
    means = []
    covs = []
    for y in np.linspace(0.1, 0.9, 10):
        means.append([0.3, y])
        covs.append([[0.0005, 0.0], [0.0, 0.005]])
        means.append([0.7, y])
        covs.append([[0.0005, 0.0], [0.0, 0.005]])
    weights = [1.0/20] * 20
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_21():
    # High-Frequency Sine Wave (15 modes)
    means = []
    covs = []
    xs = np.linspace(0.1, 0.9, 15)
    ys = 0.5 + 0.3 * np.sin(xs * 4 * np.pi)
    for x, y in zip(xs, ys):
        means.append([x, y])
        covs.append([[0.002, 0.0], [0.0, 0.002]])
    weights = [1.0/15] * 15
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_22():
    # Weighted X-Shape (9 modes)
    means = [[0.5, 0.5]]
    covs = [[[0.03, 0.0], [0.0, 0.03]]]
    weights = [0.05]
    for d in np.linspace(0.15, 0.4, 2):
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            means.append([0.5 + d*dx, 0.5 + d*dy])
            if d > 0.3:
                covs.append([[0.002, 0.0], [0.0, 0.002]])
                weights.append(0.3)
            else:
                covs.append([[0.01, 0.0], [0.0, 0.01]])
                weights.append(0.1)
    weights = [w / sum(weights) for w in weights]
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_23():
    # Sparse "Dust" Field (25 modes)
    rng = np.random.default_rng(99)
    means = rng.uniform(0.05, 0.95, size=(25, 2)).tolist()
    covs = [[[0.001, 0.0], [0.0, 0.001]] for _ in range(25)]
    weights = [1.0/25] * 25
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_24():
    # Nested Triangles (6 modes)
    means = []
    covs = []
    for theta in [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]:
        means.append([0.5 + 0.35 * np.cos(theta), 0.5 + 0.35 * np.sin(theta)])
        covs.append([[0.005, 0.0], [0.0, 0.005]])
    for theta in [-np.pi/2, -np.pi/2 + 2*np.pi/3, -np.pi/2 + 4*np.pi/3]:
        means.append([0.5 + 0.15 * np.cos(theta), 0.5 + 0.15 * np.sin(theta)])
        covs.append([[0.002, 0.0], [0.0, 0.002]])
    weights = [1.0/6] * 6
    return {'means': means, 'covs': covs, 'weights': weights}

def _test_gmm_25():
    # The "Obstacle Course" (11 modes)
    means = []
    covs = []
    weights = []
    for y in np.linspace(0.1, 0.9, 10):
        means.append([0.2, y])
        covs.append([[0.001, 0.0], [0.0, 0.008]])
        weights.append(0.05)
    means.append([0.85, 0.5])
    covs.append([[0.0005, 0.0], [0.0, 0.0005]])
    weights.append(0.5)
    weights = [w / sum(weights) for w in weights]
    return {'means': means, 'covs': covs, 'weights': weights}

def _rand_gmm_complex(seed, min_comps=5, max_comps=40):
    rng = np.random.default_rng(seed)
    n_comps = rng.integers(min_comps, max_comps + 1)
    means = []
    covs = []
    weights = []
    for _ in range(n_comps):
        means.append(rng.uniform(0.1, 0.9, 2).tolist())
        a = rng.uniform(0.0001, 0.04)
        b = rng.uniform(0.0001, 0.04)
        theta = rng.uniform(0, 2*np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        D = np.array([[a, 0], [0, b]])
        covs.append((R @ D @ R.T).tolist())
        weights.append(float(rng.uniform(0.01, 1.0)))
    weights = (np.array(weights) / sum(weights)).tolist()
    return {'means': means, 'covs': covs, 'weights': weights}

def _rand_ana_poly_complex(seed, min_verts=5, max_verts=15):
    rng = np.random.default_rng(seed)
    n_verts = rng.integers(min_verts, max_verts + 1)
    
    if rng.random() > 0.5:
        # Star
        pts = []
        r_inner = rng.uniform(0.05, 0.2)
        r_outer = rng.uniform(0.3, 0.45)
        cx, cy = rng.uniform(0.4, 0.6), rng.uniform(0.4, 0.6)
        angles = np.linspace(0, 2*np.pi, n_verts * 2, endpoint=False)
        for i, a in enumerate(angles):
            r = r_outer if i % 2 == 0 else r_inner
            r += rng.uniform(-0.02, 0.02)
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    else:
        # Concave polygon
        pts_raw = rng.uniform(0.1, 0.9, (n_verts, 2))
        cx, cy = pts_raw.mean(axis=0)
        angles = np.arctan2(pts_raw[:,1] - cy, pts_raw[:,0] - cx)
        sort_idx = np.argsort(angles)
        pts = pts_raw[sort_idx].tolist()
        for i in range(1, n_verts, 2):
            pts[i] = [cx + (pts[i][0] - cx) * 0.2, cy + (pts[i][1] - cy) * 0.2]
            
    segments = []
    for i in range(len(pts)):
        segments.append((pts[i], pts[(i+1) % len(pts)]))
    
    extra = rng.integers(0, 4)
    for _ in range(extra):
        i, j = rng.choice(len(pts), 2, replace=False)
        segments.append((pts[i], pts[j]))
        
    return {'type': 'analytical', 'sigma': float(rng.uniform(0.015, 0.035)), 'segments': segments}

def _rand_organic(seed, pts=100):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, pts + 1)[:-1]
    kind = rng.choice(['amoeba', 'cloud', 'leaf', 'starfish'])
    
    if kind == 'amoeba':
        r = rng.uniform(0.2, 0.3)
        for _ in range(rng.integers(3, 7)):
            k = rng.integers(2, 6)
            phase = rng.uniform(0, 2*np.pi)
            amp = rng.uniform(0.02, 0.08)
            r += amp * np.sin(k * t + phase)
        x = rng.uniform(0.4, 0.6) + r * np.cos(t)
        y = rng.uniform(0.4, 0.6) + r * np.sin(t)
    elif kind == 'cloud':
        r = rng.uniform(0.2, 0.25)
        for _ in range(rng.integers(4, 9)):
            k = rng.integers(3, 9)
            phase = rng.uniform(0, 2*np.pi)
            amp = rng.uniform(0.03, 0.07)
            r += amp * np.abs(np.sin(k * t / 2 + phase))
        x = rng.uniform(0.4, 0.6) + r * np.cos(t)
        y = rng.uniform(0.4, 0.6) + r * np.sin(t)
    elif kind == 'leaf':
        t_leaf = t - np.pi/2
        base = rng.uniform(0.1, 0.2)
        freq = rng.integers(4, 12)
        r = base * (1 + np.sin(t_leaf)) * (1 + rng.uniform(0.2, 0.4) * np.cos(freq * t_leaf))
        r += rng.uniform(0, 0.02, pts)
        x = rng.uniform(0.4, 0.6) + r * np.cos(t)
        y = rng.uniform(0.4, 0.6) + r * np.sin(t)
    elif kind == 'starfish':
        k = rng.integers(4, 7)
        r = rng.uniform(0.2, 0.3) + rng.uniform(0.1, 0.2) * np.cos(k * t) + rng.uniform(0.02, 0.08) * np.cos(k * 2 * t)
        x = rng.uniform(0.4, 0.6) + r * np.cos(t)
        y = rng.uniform(0.4, 0.6) + r * np.sin(t)
        
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    
    segments = []
    for i in range(pts):
        segments.append(([float(x[i]), float(y[i])], [float(x[(i+1)%pts]), float(y[(i+1)%pts])]))
        
    return {'type': 'analytical', 'sigma': float(rng.uniform(0.02, 0.04)), 'segments': segments}

# ─────────────────────────────────────────────────────────────────────────────
#  Named catalogue  (built lazily via get_shape())
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.font_manager as fm


def _finde_font(dateiname):
    """Erste vorhandene Fassung einer Schrift, sonst None.

    Die alte Fassung stand fest auf /usr/share/fonts und war in ein
    try/except gehuellt — das aber nie ausloeste: `FontProperties(fname=...)`
    prueft die Datei beim Anlegen nicht, sie scheitert erst beim Rendern.
    Auf dem Cluster fielen deshalb sieben von acht Array-Aufgaben nach
    wenigen Minuten mit `FileNotFoundError` aus, mitten in der Erzeugung.

    Gesucht wird deshalb in dieser Reihenfolge, und die Datei muss wirklich
    da sein:

      1. $ERGODIC_FONT_DIR          (ausdrueckliche Vorgabe)
      2. <Projektwurzel>/fonts/     (mit dem Repo mitgeliefert)
      3. ~/Master_thesis/fonts/
      4. die Systempfade
    """
    orte = []
    if os.environ.get('ERGODIC_FONT_DIR'):
        orte.append(os.path.join(os.environ['ERGODIC_FONT_DIR'], dateiname))
    _wurzel = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    orte.append(os.path.join(_wurzel, 'fonts', dateiname))
    orte.append(os.path.expanduser(os.path.join('~', 'Master_thesis', 'fonts', dateiname)))
    orte.append(os.path.join('/usr/share/fonts/opentype/noto', dateiname))
    orte.append(os.path.join('/usr/share/fonts/truetype/noto', dateiname))
    for o in orte:
        if os.path.isfile(o):
            return fm.FontProperties(fname=o, size=72)
    return None


_KOREAN_FONT = _finde_font('NotoSansCJK-Bold.ttc')
_CJK_FONT = _finde_font('NotoSerifCJK-Bold.ttc')
if _KOREAN_FONT is None or _CJK_FONT is None:
    import warnings
    warnings.warn(
        'CJK-Schriften nicht gefunden. Die Formen korean_* und cjk_* koennen '
        'nicht erzeugt werden. Erwartet unter <Projektwurzel>/fonts/ oder '
        '$ERGODIC_FONT_DIR.')

GREEK_UPPER = 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'
GREEK_LOWER = 'αβγδεζηθικλμνξοπρστυφχψω'
KOREAN_CHARS = ['가', '나', '다', '라', '마', '바', '사', '아', '자', '차', '카', '타', '파', '하', 
                '고', '노', '도', '로', '모', '보', '소', '오', '조', '초', '코', '토', '포', '호',
                '구', '누']
CJK_CHARS = [
    '龜', '龍', '龘', '鬱', '愛', '夢', '櫻', '華', '雪', '風', 
    '月', '花', '鳥', '星', '海', '雲', '雷', '雨', '山', '川',
    '天', '地', '日', '水', '火', '木', '金', '土', '劍', '心',
    '靈', '魂', '神', '鬼', '魔', '道', '佛', '禪', '氣', '血',
    '骨', '肉', '生', '死', '光', '影', '黑', '白', '赤', '青',
    '黃', '綠', '紫', '銀', '鐵', '銅', '石', '玉', '珠', '寶',
    '鏡', '鐘', '笛', '琴', '書'
]

# Map name → builder lambda (deferred so import is fast)
_BUILDERS = {
    # ── Letters (uppercase) ──────────────────────────────────────────────────
    **{c: (lambda ch=c: _letter(ch)) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
    # ── Letters (lowercase) ──────────────────────────────────────────────────
    **{c.lower() + '_lc': (lambda ch=c: _letter(ch.lower())) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
    # ── Digits ───────────────────────────────────────────────────────────────
    **{f'digit_{d}': (lambda dd=d: _letter(str(dd))) for d in range(10)},
    # ── Greek ────────────────────────────────────────────────────────────────
    **{f'greek_upper_{i}': (lambda ch=c: _letter(ch)) for i, c in enumerate(GREEK_UPPER)},
    **{f'greek_lower_{i}': (lambda ch=c: _letter(ch)) for i, c in enumerate(GREEK_LOWER)},
    # ── Korean ───────────────────────────────────────────────────────────────
    **{f'korean_{i}': (lambda ch=c: _letter(ch, fontproperties=_KOREAN_FONT)) for i, c in enumerate(KOREAN_CHARS)},
    
    # ── Random GMM Shapes (185) ──────────────────────────────────────────────
    **{f'rand_gmm_{i}': (lambda s=i: _rand_gmm(seed=s)) for i in range(185)},
    # ── Random Analytical polygons (185) ─────────────────────────────────────
    **{f'rand_ana_poly_{i}': (lambda s=i: _rand_ana_poly(seed=s+1000)) for i in range(185)},
    
    # ── Random GMM Complex (75) ──────────────────────────────────────────────
    **{f'rand_gmm_complex_{i}': (lambda s=i: _rand_gmm_complex(seed=s+5000)) for i in range(75)},
    # ── Random Analytical Polygon Complex (75) ───────────────────────────────
    **{f'rand_ana_poly_complex_{i}': (lambda s=i: _rand_ana_poly_complex(seed=s+6000)) for i in range(75)},
    # ── Organic Shapes (50) ──────────────────────────────────────────────────
    **{f'organic_{i}': (lambda s=i: _rand_organic(seed=s+7000)) for i in range(50)},
    # ── CJK Characters (65) ──────────────────────────────────────────────────
    **{f'cjk_{i}': (lambda ch=c: _letter(ch, fontproperties=_CJK_FONT)) for i, c in enumerate(CJK_CHARS)},
}


_CACHE = {}   # name → shape_def (so each shape is built only once)

def get_shape(name):
    """Lazily build and cache a shape definition by name."""
    if name not in _CACHE:
        if name not in _BUILDERS:
            raise KeyError(f"Unknown shape '{name}'. Available: {list(_BUILDERS.keys())[:10]}…")
        _CACHE[name] = _BUILDERS[name]()
    return _CACHE[name]

def all_shape_names():
    return list(_BUILDERS.keys())

# ─────────────────────────────────────────────────────────────────────────────
#  Fixed splits
# ─────────────────────────────────────────────────────────────────────────────

def all_dataset_shapes():
    chars = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'] + \
            [c.lower() + '_lc' for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'] + \
            [f'digit_{d}' for d in range(10)] + \
            [f'greek_upper_{i}' for i in range(24)] + \
            [f'greek_lower_{i}' for i in range(24)] + \
            [f'korean_{i}' for i in range(30)] + \
            [f'cjk_{i}' for i in range(65)]
    gmms = [f'rand_gmm_{i}' for i in range(185)] + [f'rand_gmm_complex_{i}' for i in range(75)]
    anas = [f'rand_ana_poly_{i}' for i in range(185)] + [f'rand_ana_poly_complex_{i}' for i in range(75)]
    orgs = [f'organic_{i}' for i in range(50)]
    return chars + gmms + anas + orgs

PREVIEW_SHAPES = ['A', 'greek_upper_5', 'korean_0', 'rand_gmm_0', 'rand_ana_poly_0']

VALIDATION_SHAPES = [
    'A', 'greek_upper_0', 'korean_5', 'digit_5', 'a_lc',
    'rand_gmm_10', 'rand_gmm_20',
    'rand_ana_poly_10', 'rand_ana_poly_20', 'rand_ana_poly_30',
    'rand_gmm_complex_0', 'rand_gmm_complex_10', 'rand_gmm_complex_20', 'rand_gmm_complex_30',
    'rand_ana_poly_complex_0', 'rand_ana_poly_complex_10', 'rand_ana_poly_complex_20', 'rand_ana_poly_complex_30',
    'organic_0', 'organic_10', 'organic_20', 'organic_30',
    'cjk_0', 'cjk_10', 'cjk_20'
]

def train_shape_names(total=750):
    val_set = set(VALIDATION_SHAPES)
    candidates = [n for n in all_dataset_shapes() if n not in val_set]
    import random
    rng = random.Random(42)
    rng.shuffle(candidates)
    return candidates[:total]


# ═════════════════════════════════════════════════════════════════════════════
#  Flache Formen — die Trainingsverteilung an Glaubensdichten heranfuehren
# ═════════════════════════════════════════════════════════════════════════════
#  Gemessen ueber die zwoelf Holdout-Formen (Anteil der Masse in den obersten
#  30 % der Zellen, und Traegeranteil):
#
#      wahre Dichten          Traeger 0,658   Konzentration 0,851
#      Phi (UCB/Niveaumenge)  Traeger 1,000   Konzentration 0,337-0,481
#
#  Die vier Familien hier erzeugen Dichten im zweiten Band, bleiben dabei aber
#  analytisch: `make_pdf_and_score` liefert den Score wie bisher ueber
#  `jax.grad`, der SVGD-Solver laeuft unveraendert, und es entstehen echte
#  ueberwachte Labels. Die Architektur CFM+ErgLoss bleibt unangetastet — es
#  aendert sich nur, was sie zu sehen bekommt.
#
#  Wichtig: `all_dataset_shapes()` und `train_shape_names()` werden bewusst
#  NICHT angefasst. Deren Reihenfolge haengt an einem Shuffle mit Keim 42;
#  jede Ergaenzung dort wuerde die bestehenden 750 Trainingsformen austauschen
#  und alle frueheren Laeufe unvergleichbar machen.

def _weichzeichnen(base_def, s):
    """Form verbreitern: GMM ueber die Kovarianzen, Segmente ueber sigma."""
    d = dict(base_def)
    if d.get('type') == 'analytical':
        d['sigma'] = float(min(d.get('sigma', 0.025) + s, 0.090))
    else:
        covs = np.array(d['covs'], dtype=np.float64)
        covs[:, 0, 0] += s * s
        covs[:, 1, 1] += s * s
        d['covs'] = covs.tolist()
    return d


def _breite_moden(seed):
    """Zwei bis vier breite Gauss-Moden — so sieht ein Phi nach wenigen
    Messungen tatsaechlich aus."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))
    means = rng.uniform(0.25, 0.75, size=(n, 2)).tolist()
    covs, w = [], []
    for _ in range(n):
        a = float(rng.uniform(0.010, 0.055))
        b = float(rng.uniform(0.010, 0.055))
        t = float(rng.uniform(0, 2 * np.pi))
        c_t, s_t = np.cos(t), np.sin(t)
        R = np.array([[c_t, -s_t], [s_t, c_t]])
        covs.append((R @ np.diag([a, b]) @ R.T).tolist())
        w.append(float(rng.uniform(0.4, 1.0)))
    w = (np.array(w) / sum(w)).tolist()
    return {'means': means, 'covs': covs, 'weights': w}


def _ring_kontur(seed):
    """Geschlossene, glatte Kontur. Das ist die Gestalt, die eine
    Niveaumengen- oder Informationsdichte erzeugt: Masse auf dem Rand einer
    Menge statt in ihrem Inneren."""
    rng = np.random.default_rng(seed)
    r0 = float(rng.uniform(0.22, 0.34))
    cx, cy = rng.uniform(0.40, 0.60, size=2)
    am = rng.uniform(-0.18, 0.18, size=3)
    ph = rng.uniform(0, 2 * np.pi, size=3)
    th = np.linspace(0, 2 * np.pi, 65)
    r = r0 * (1 + sum(am[k] * np.cos((k + 2) * th + ph[k]) for k in range(3)))
    r = np.clip(r, 0.08, 0.44)
    x = np.clip(cx + r * np.cos(th), 0.03, 0.97)
    y = np.clip(cy + r * np.sin(th), 0.03, 0.97)
    segs = [([float(x[i]), float(y[i])], [float(x[i + 1]), float(y[i + 1])])
            for i in range(len(th) - 1)]
    return {'type': 'analytical', 'sigma': float(rng.uniform(0.030, 0.055)),
            'segments': segs}


_FLAT_POOL = None


def _flat_basis(i):
    """Grundform aus dem bestehenden Trainingsvorrat — nie aus VALIDATION_SHAPES,
    damit kein Holdout ueber die Hintertuer ins Training gelangt."""
    global _FLAT_POOL
    if _FLAT_POOL is None:
        _FLAT_POOL = train_shape_names(750)
    return get_shape(_FLAT_POOL[(i * 7 + 3) % len(_FLAT_POOL)])


def _flat_ped(i, seed):
    rng = np.random.default_rng(seed)
    return mit_sockel(_flat_basis(i), float(rng.uniform(0.50, 0.95)), seed)


def _flat_blur(i, seed):
    rng = np.random.default_rng(seed)
    d = _weichzeichnen(_flat_basis(i), float(rng.uniform(0.04, 0.10)))
    return mit_sockel(d, float(rng.uniform(0.45, 0.93)), seed)


def _flat_broad(seed):
    rng = np.random.default_rng(seed)
    return mit_sockel(_breite_moden(seed), float(rng.uniform(0.30, 0.85)), seed)


def _flat_ring(seed):
    rng = np.random.default_rng(seed)
    return mit_sockel(_ring_kontur(seed), float(rng.uniform(0.45, 0.92)), seed)


N_FLAT_PED, N_FLAT_BLUR, N_FLAT_BROAD, N_FLAT_RING = 180, 80, 80, 60

_BUILDERS.update({
    **{f'flat_ped_{i}':   (lambda k=i: _flat_ped(k,   20000 + k)) for i in range(N_FLAT_PED)},
    **{f'flat_blur_{i}':  (lambda k=i: _flat_blur(k,  30000 + k)) for i in range(N_FLAT_BLUR)},
    **{f'flat_broad_{i}': (lambda k=i: _flat_broad(   40000 + k)) for i in range(N_FLAT_BROAD)},
    **{f'flat_ring_{i}':  (lambda k=i: _flat_ring(    50000 + k)) for i in range(N_FLAT_RING)},
    # Zwoelf flache Holdout-Formen, drei je Familie. Sie kommen in einen
    # eigenen Split `val_flat`, damit der bestehende `val`-Split — und damit
    # jede frueher gemessene Holdout-Zahl — unveraendert bleibt.
    **{f'flat_val_ped_{i}':   (lambda k=i: _flat_ped(900 + k, 60000 + k)) for i in range(3)},
    **{f'flat_val_blur_{i}':  (lambda k=i: _flat_blur(910 + k, 61000 + k)) for i in range(3)},
    **{f'flat_val_broad_{i}': (lambda k=i: _flat_broad(     62000 + k)) for i in range(3)},
    **{f'flat_val_ring_{i}':  (lambda k=i: _flat_ring(      63000 + k)) for i in range(3)},
})


def flat_shape_names(split='train'):
    """Namen der flachen Formen. `split='train'` gibt 400, `'val'` gibt 12."""
    if split == 'val':
        return ([f'flat_val_ped_{i}' for i in range(3)]
                + [f'flat_val_blur_{i}' for i in range(3)]
                + [f'flat_val_broad_{i}' for i in range(3)]
                + [f'flat_val_ring_{i}' for i in range(3)])
    return ([f'flat_ped_{i}' for i in range(N_FLAT_PED)]
            + [f'flat_blur_{i}' for i in range(N_FLAT_BLUR)]
            + [f'flat_broad_{i}' for i in range(N_FLAT_BROAD)]
            + [f'flat_ring_{i}' for i in range(N_FLAT_RING)])
