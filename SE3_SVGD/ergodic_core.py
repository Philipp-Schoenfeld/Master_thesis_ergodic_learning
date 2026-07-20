#!/usr/bin/env python3
"""
Dimension-Agnostic Ergodic Core
================================
Shared Fourier-based ergodic metric and target distribution utilities
that work for any spatial dimension (2D, 3D, ...).

Used by both B-spline SVGD and regular SVGD in 2D and 3D.
"""

import numpy as np
import jax.numpy as jnp
from itertools import product as iter_product


# ============================================================================
# 1. Target Distribution — Segment-Based Gaussian Mixture
# ============================================================================

# Predefined 2D target shapes (line segments in [0,1]^2)
SHAPE_SEGMENTS_2D = {
    'N': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.25, 0.85], [0.75, 0.15]),
        ([0.75, 0.15], [0.75, 0.85]),
    ],
    'H': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
        ([0.25, 0.50], [0.75, 0.50]),
    ],
    'II': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
    ],
}

def project_2d_to_3d(u, v, projection_type="plane"):
    if projection_type == "plane":
        return np.array([u, v, 0.5 * u + 0.5 * v])
    elif projection_type == "sphere":
        lon = (u - 0.5) * np.pi
        lat = (v - 0.5) * np.pi * 0.7
        r = 0.45
        x = 0.5 + r * np.cos(lat) * np.sin(lon)
        y = 0.5 + r * np.cos(lat) * np.cos(lon)
        z = 0.5 + r * np.sin(lat)
        return np.array([x, y, z])
    elif projection_type == "cube":
        lon = (u - 0.5) * np.pi
        lat = (v - 0.5) * np.pi * 0.7
        sx = np.cos(lat) * np.sin(lon)
        sy = np.cos(lat) * np.cos(lon)
        sz = np.sin(lat)
        
        max_val = max(abs(sx), abs(sy), abs(sz))
        r_cube = 0.4
        cx = sx * (r_cube / max_val)
        cy = sy * (r_cube / max_val)
        cz = sz * (r_cube / max_val)
        return np.array([0.5 + cx, 0.5 + cy, 0.5 + cz])
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")

def get_3d_segments(shape_name, projection_type="plane", num_points=20):
    """
    Builds 3D line segments by projecting the 2D shape onto a surface.
    For non-linear surfaces (sphere, cube), discretizes the segments.
    """
    segs_2d = SHAPE_SEGMENTS_2D[shape_name]
    segs_3d = []

    for (a2, b2) in segs_2d:
        if projection_type == "plane":
            a3 = project_2d_to_3d(a2[0], a2[1], projection_type).tolist()
            b3 = project_2d_to_3d(b2[0], b2[1], projection_type).tolist()
            segs_3d.append((a3, b3))
        else:
            u_vals = np.linspace(a2[0], b2[0], num_points)
            v_vals = np.linspace(a2[1], b2[1], num_points)
            pts_3d = [project_2d_to_3d(u, v, projection_type).tolist() for u, v in zip(u_vals, v_vals)]
            for i in range(len(pts_3d) - 1):
                segs_3d.append((pts_3d[i], pts_3d[i+1]))

    return segs_3d


def dist_to_segment_nd(point, seg_start, seg_end):
    """
    Shortest Euclidean distance from a point to a line segment.
    Works for any dimension (2D, 3D, ...).

    Args:
        point: (..., dim) array of query points
        seg_start: (dim,) segment start
        seg_end: (dim,) segment end

    Returns:
        (...,) array of distances
    """
    seg_start = np.asarray(seg_start)
    seg_end = np.asarray(seg_end)
    d = seg_end - seg_start
    len_sq = np.dot(d, d)
    # Project point onto line, clamp to [0, 1]
    t = np.clip(np.sum((point - seg_start) * d, axis=-1) / (len_sq + 1e-12), 0, 1)
    # Closest point on segment
    proj = seg_start + t[..., None] * d
    return np.sqrt(np.sum((point - proj) ** 2, axis=-1))


def build_target_distribution_2d(shape_name, stroke_width=0.045, grid_res=200):
    """
    Builds a 2D target probability field over [0,1]^2.

    Returns:
        Xg, Yg: (grid_res, grid_res) meshgrid arrays
        Zg: (grid_res, grid_res) target density values
    """
    segments = SHAPE_SEGMENTS_2D[shape_name]
    xs = np.linspace(0, 1, grid_res)
    ys = np.linspace(0, 1, grid_res)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.stack([Xg, Yg], axis=-1)  # (grid_res, grid_res, 2)

    d_min = np.full(Xg.shape, 1e10)
    for seg_start, seg_end in segments:
        d = dist_to_segment_nd(pts, seg_start, seg_end)
        d_min = np.minimum(d_min, d)

    Zg = np.exp(-d_min ** 2 / (2 * stroke_width ** 2))
    return Xg, Yg, Zg


def build_target_distribution_3d(shape_name, stroke_width=0.045, grid_res=50, projection_type="plane"):
    """
    Builds a 3D target probability field over [0,1]^3.

    Returns:
        grid_axes: tuple of (xs, ys, zs) 1D arrays
        grid_pts: (N_total, 3) flattened grid points
        Wg: (N_total,) target density values (flattened)
        grid_shape: (grid_res, grid_res, grid_res)
    """
    segments = get_3d_segments(shape_name, projection_type)
    xs = np.linspace(0, 1, grid_res)
    ys = np.linspace(0, 1, grid_res)
    zs = np.linspace(0, 1, grid_res)
    Xg, Yg, Zg_grid = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([Xg, Yg, Zg_grid], axis=-1)  # (R, R, R, 3)

    d_min = np.full(Xg.shape, 1e10)
    for seg_start, seg_end in segments:
        d = dist_to_segment_nd(pts, seg_start, seg_end)
        d_min = np.minimum(d_min, d)

    Wg = np.exp(-d_min ** 2 / (2 * stroke_width ** 2))
    grid_pts = pts.reshape(-1, 3)
    return (xs, ys, zs), grid_pts, Wg.ravel(), Xg.shape


# ============================================================================
# 2. Fourier Decomposition (Dimension-Agnostic)
# ============================================================================

def build_fourier_indices(K, dim):
    """
    Generates all combinations of wave numbers (k_1, ..., k_dim) up to K.

    Args:
        K: maximum wave number per dimension
        dim: spatial dimension (2 or 3)

    Returns:
        k_indices: (K^dim, dim) array of integer wave number tuples
    """
    ranges = [range(K)] * dim
    return np.array(list(iter_product(*ranges)))


def compute_lambda_k(k_indices):
    """
    Spectral decay weights: Λ_k = (1 + ||k||^2)^{-3/2}

    Strongly penalizes low-frequency mismatches.
    """
    return (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)


def fourier_basis_nd(pts, k_indices):
    """
    Evaluates Fourier cosine basis functions at given points.
    Works for any dimension.

    F_k(x) = prod_d cos(π * k_d * x_d)

    Args:
        pts: (N, dim) array of spatial points
        k_indices: (M, dim) array of wave number tuples

    Returns:
        (N, M) array of basis function evaluations
    """
    # pts[:, None, :] has shape (N, 1, dim)
    # k_indices[None, :, :] has shape (1, M, dim)
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)


def fourier_basis_nd_jax(pts, k_indices):
    """
    JAX version of fourier_basis_nd for use inside JIT-compiled functions.
    """
    args = jnp.pi * pts[:, None, :] * k_indices[None, :, :]
    return jnp.prod(jnp.cos(args), axis=-1)


def fourier_basis_grad_nd(pts, k_indices):
    """
    Gradient of the Fourier cosine basis w.r.t. spatial coordinates.
    Works for any dimension.

    ∂F_k/∂x_d = -π k_d sin(π k_d x_d) * ∏_{d'≠d} cos(π k_{d'} x_{d'})

    Args:
        pts: (N, dim) array
        k_indices: (M, dim) array

    Returns:
        (N, M, dim) gradient array
    """
    dim = pts.shape[1]
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]  # (N, M, dim)
    c = np.cos(args)  # (N, M, dim)
    s = np.sin(args)  # (N, M, dim)

    # For each dimension d, the gradient is:
    #   -π * k_d * sin(args_d) * prod_{d' != d} cos(args_{d'})
    grads = np.zeros((*pts.shape[:1], k_indices.shape[0], dim))
    for d in range(dim):
        # Product of cosines over all dimensions except d
        cos_prod = np.ones(c.shape[:2])
        for d2 in range(dim):
            if d2 != d:
                cos_prod *= c[:, :, d2]
        grads[:, :, d] = -np.pi * k_indices[None, :, d] * s[:, :, d] * cos_prod

    return grads


def compute_target_fourier_coeffs(grid_pts, grid_weights, k_indices):
    """
    Computes the Fourier coefficients φ_k of a target distribution.

    φ_k = Σ_i w_i F_k(x_i)   where w_i are normalized weights.

    Args:
        grid_pts: (N, dim) grid points
        grid_weights: (N,) unnormalized density values
        k_indices: (M, dim) wave numbers

    Returns:
        phi_k: (M,) Fourier coefficients
    """
    w = grid_weights / grid_weights.sum()
    F = fourier_basis_nd(grid_pts, k_indices)  # (N, M)
    return np.sum(w[:, None] * F, axis=0)


def compute_ergodic_metric_jax(X, k_indices_jnp, Lambda_k_jnp, phi_k_jnp):
    """
    JAX-compatible ergodic metric computation.

    Args:
        X: (T, dim) trajectory positions
        k_indices_jnp: (M, dim) JAX array
        Lambda_k_jnp: (M,) JAX array
        phi_k_jnp: (M,) JAX array

    Returns:
        ergodic_metric: scalar
    """
    Fk = fourier_basis_nd_jax(X, k_indices_jnp)  # (T, M)
    c_k = jnp.mean(Fk, axis=0)  # (M,)
    diff_k = c_k - phi_k_jnp
    return 0.5 * jnp.sum(Lambda_k_jnp * diff_k ** 2)


def compute_ergodic_metric_numpy(X, k_indices, Lambda_k, phi_k):
    """
    NumPy version of the ergodic metric for final evaluation.

    Args:
        X: (T, dim) trajectory positions
        k_indices: (M, dim) array
        Lambda_k: (M,) array
        phi_k: (M,) array

    Returns:
        ergodic_metric: scalar
    """
    Fk = fourier_basis_nd(X, k_indices)  # (T, M)
    c_k = np.mean(Fk, axis=0)
    diff_k = c_k - phi_k
    return 0.5 * np.sum(Lambda_k * diff_k ** 2)
