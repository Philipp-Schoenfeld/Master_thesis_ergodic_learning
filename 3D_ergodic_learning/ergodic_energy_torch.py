r"""
ergodic_energy_torch.py  —  3D port
===================================
Differentiable PyTorch port of the TSVEC solver energy, lifted to three
dimensions.

    E = W_SMOOTH   * sum(accel^2)
      + W_ERGODIC  * 0.5 * sum(Lambda_k * (c_k - phi_k)^2)
      + W_BOUNDARY * 0.5 * (sum(lo^2) + sum(hi^2))
      + W_OBSTACLE * 0.5 * sum(max(r - dist, 0)^2)

Three terms carry over unchanged because they were already written with
``sum(-1)`` over the coordinate axis: smoothness, boundary, obstacle. The
ergodic term is the one that genuinely changes.

What changed, and why
---------------------
**Basis.** F_k(x) = cos(pi k1 x) cos(pi k2 y) cos(pi k3 z). The implementation
in `fourier_basis` was already dimension-agnostic (an elementwise cos followed
by a product over the last axis), so only the index grid grows: K^3 modes
instead of K^2. At the solver's K = 10 that is **1000 modes instead of 100**.

**Lambda exponent.** The 2D solver uses Lambda_k = (1 + |k|^2)^(-3/2). That
exponent is not a magic constant: it is -(d+1)/2 for d = 2, the Sobolev weight
from Mathew & Mezic that makes the ergodic metric a norm on H^-s. Carrying -3/2
into 3D would silently change which function space the metric lives in, so the
default here is **-(d+1)/2 = -2.0**. `LAMBDA_EXPONENT` is a module constant and
a constructor argument, so the 2D value can be restored for an ablation.

**Memory.** `target_coeffs_from_grid` used to build the full (H*W, M) basis
matrix. In 3D that is (R^3, K^3) — at R = 64, K = 10 it would be 262144 x 1000
floats, about 1 GB per call. It is therefore evaluated in chunks over the grid
points; the result is identical, the peak allocation is bounded by `chunk`.

Conventions that must match the 2D version, and do:
  * no normalisation on the basis
  * K^3 ordering is k1 outer, k2 middle, k3 inner
  * c_k is the *mean* over the T points (the time average)
  * smoothness has no 0.5 factor; the other three do
"""

import math
import numpy as np
import torch
import torch.nn as nn

# ── Constants, taken verbatim from SE3_SVGD/tsvec_2d.py ──────────────────────
# Deliberately NOT re-tuned for 3D: keeping them identical is what makes a
# 3D run on planar data comparable to the 2D reference.
W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0
W_OBSTACLE = 50000.0

BOUNDARY_MARGIN = 0.03
K_DEFAULT = 10                  # -> 1000 modes in 3D
ND = 3
LAMBDA_EXPONENT = -(ND + 1) / 2.0       # -2.0 in 3D; the 2D solver used -1.5
OBSTACLE_CENTER = (0.5, 0.5, 0.5)
OBSTACLE_RADIUS = 0.12


def make_k_grid(K=K_DEFAULT, nd=ND, exponent=LAMBDA_EXPONENT):
    """(K^nd, nd) index grid and (K^nd,) weights.

    Ordering is odometer-style with the first axis slowest, matching the 2D
    version's ``[[k1, k2] for k1 in range(K) for k2 in range(K)]``.

    Computed in float64: Lambda_k in float32 costs ~1e-7 relative precision,
    which is enough to break an exact comparison against a NumPy reference.
    Callers cast down to float32 for training.
    """
    axes = np.meshgrid(*([np.arange(K, dtype=np.float64)] * nd), indexing='ij')
    k_idx = np.stack([a.reshape(-1) for a in axes], axis=-1)       # (K^nd, nd)
    Lambda = (1.0 + (k_idx ** 2).sum(axis=1)) ** exponent
    return k_idx, Lambda


def fourier_basis(pts, k_idx):
    """F_k at each point. pts: (..., T, nd), k_idx: (M, nd) -> (..., T, M).

    Unchanged from 2D — the product over the trailing axis already generalises.
    """
    args = math.pi * pts.unsqueeze(-2) * k_idx
    return torch.cos(args).prod(dim=-1)


def coeffs_from_points(pts, k_idx):
    """Time-averaged ergodic coefficients c_k. (..., T, nd) -> (..., M)."""
    return fourier_basis(pts, k_idx).mean(dim=-2)


def target_coeffs_from_grid(density, k_idx, eps=1e-12, chunk=8192):
    """Target coefficients phi_k from a density volume on [0,1]^3.

    Normalises the density to sum 1 over the volume and takes the weighted
    average of the basis — same as 2D, but evaluated in chunks over grid points
    so the (R^3, K^3) basis is never materialised in full.

    density: (R, R, R) or (B, R, R, R)   ->   (M,) or (B, M)
    """
    single = density.dim() == 3
    d = density.unsqueeze(0) if single else density
    Bsz, Dz, Hy, Wx = d.shape

    dev, dt = d.device, d.dtype
    zs = torch.linspace(0, 1, Dz, device=dev, dtype=dt)
    ys = torch.linspace(0, 1, Hy, device=dev, dtype=dt)
    xs = torch.linspace(0, 1, Wx, device=dev, dtype=dt)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    w = d.reshape(Bsz, -1)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=eps)

    kt = k_idx.to(device=dev, dtype=dt)
    phi = torch.zeros(Bsz, kt.shape[0], device=dev, dtype=dt)
    for s in range(0, grid.shape[0], chunk):
        g = grid[s:s + chunk]                              # (c, 3)
        Fk = fourier_basis(g, kt)                          # (c, M)
        phi += w[:, s:s + chunk] @ Fk
    return phi.squeeze(0) if single else phi


# ── Individual terms ─────────────────────────────────────────────────────────
def smoothness_term(X, w=W_SMOOTH):
    """w * sum(accel^2), per sample. X: (B, T, nd) -> (B,). Dimension-agnostic."""
    accel = X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
    return w * accel.pow(2).sum(dim=(1, 2))


def ergodic_term(X, k_idx, Lambda, phi_k, w=W_ERGODIC):
    """w * 0.5 * sum(Lambda_k * (c_k - phi_k)^2). X: (B, T, nd) -> (B,)"""
    c_k = coeffs_from_points(X, k_idx)                # (B, M)
    if phi_k.dim() == 1:
        phi_k = phi_k.unsqueeze(0)
    diff = c_k - phi_k
    return w * 0.5 * (Lambda * diff.pow(2)).sum(dim=-1)


def boundary_term(X, margin=BOUNDARY_MARGIN, w=W_BOUNDARY):
    """w * 0.5 * (sum(lo^2) + sum(hi^2)). X: (B, T, nd) -> (B,). Unchanged."""
    lo = (X - margin).clamp(max=0.0)
    hi = (X - (1.0 - margin)).clamp(min=0.0)
    return w * 0.5 * (lo.pow(2).sum(dim=(1, 2)) + hi.pow(2).sum(dim=(1, 2)))


def obstacle_term(X, center=OBSTACLE_CENTER, radius=OBSTACLE_RADIUS, w=W_OBSTACLE):
    """w * 0.5 * sum(max(r - dist, 0)^2). X: (B, T, nd) -> (B,). Unchanged."""
    c = torch.as_tensor(center, dtype=X.dtype, device=X.device)
    dist = ((X - c) ** 2).sum(-1).add(1e-12).sqrt()
    violation = (radius - dist).clamp(min=0.0)
    return w * 0.5 * violation.pow(2).sum(dim=1)


# ── Assembled energy ─────────────────────────────────────────────────────────
class ErgodicEnergy(nn.Module):
    """Solver energy as a differentiable module.

    Args:
        K:        frequency grid is K x K x K, so K^3 coefficients.
        basis:    optional (T, nxi) B-spline basis. When given, inputs are read
                  as control points and rendered to a T-point curve before the
                  energy is evaluated — the solver-equivalent mode.
        use_obstacle: the solver defaults to False; enable to train paths that
                  avoid the obstacle instead of only steering around it at
                  inference time.
        lambda_exponent: -2.0 in 3D. Pass -1.5 to reproduce the 2D weighting
                  for an ablation.
    """

    def __init__(self, K=K_DEFAULT, basis=None, use_obstacle=False,
                 obstacle_center=OBSTACLE_CENTER, obstacle_radius=OBSTACLE_RADIUS,
                 boundary_margin=BOUNDARY_MARGIN,
                 w_ergodic=W_ERGODIC, w_smooth=W_SMOOTH,
                 w_boundary=W_BOUNDARY, w_obstacle=W_OBSTACLE,
                 nd=ND, lambda_exponent=LAMBDA_EXPONENT):
        super().__init__()
        k_idx, Lambda = make_k_grid(K, nd=nd, exponent=lambda_exponent)
        self.register_buffer('k_idx', torch.from_numpy(k_idx).float())
        self.register_buffer('Lambda', torch.from_numpy(Lambda).float())
        if basis is not None:
            b = basis if torch.is_tensor(basis) else torch.as_tensor(basis)
            self.register_buffer('basis', b.float())
        else:
            self.basis = None
        self.K, self.nd = K, nd
        self.lambda_exponent = lambda_exponent
        self.use_obstacle = use_obstacle
        self.obstacle_center = obstacle_center
        self.obstacle_radius = obstacle_radius
        self.boundary_margin = boundary_margin
        self.w_ergodic, self.w_smooth = w_ergodic, w_smooth
        self.w_boundary, self.w_obstacle = w_boundary, w_obstacle

    def extra_repr(self):
        return (f"K={self.K} ({self.K ** self.nd} modes), nd={self.nd}, "
                f"lambda_exponent={self.lambda_exponent}")

    def render(self, X):
        """Control points -> curve, if a basis was supplied."""
        if self.basis is None:
            return X
        return torch.einsum('ti,bid->btd', self.basis, X)

    def forward(self, X, phi_k, return_terms=False):
        """X: (B, nxi, 3) control points (or (B, T, 3) points if basis is None).
        phi_k: (M,) or (B, M) target coefficients.
        Returns (B,) energy, or (energy, dict) when return_terms=True.
        """
        P = self.render(X)
        terms = {
            'smooth':   smoothness_term(P, self.w_smooth),
            'ergodic':  ergodic_term(P, self.k_idx, self.Lambda, phi_k,
                                     self.w_ergodic),
            'boundary': boundary_term(P, self.boundary_margin, self.w_boundary),
        }
        if self.use_obstacle:
            terms['obstacle'] = obstacle_term(P, self.obstacle_center,
                                              self.obstacle_radius, self.w_obstacle)
        total = sum(terms.values())
        return (total, terms) if return_terms else total


def energy_from_control_points(cps, phi_k, basis, K=K_DEFAULT, use_obstacle=False):
    """Convenience one-shot call; prefer the module when used in a training loop."""
    return ErgodicEnergy(K=K, basis=basis, use_obstacle=use_obstacle).to(cps.device)(
        cps, phi_k)


# ── Quality metrics ──────────────────────────────────────────────────────────
def coverage_distance(curves, density, chunk=4096):
    """Density-weighted mean distance from the target mass to the curve.

    For every occupied voxel, the distance to the nearest point on the curve,
    averaged with the normalised density as weight. Uses no Fourier basis, so it
    cross-checks the spectral ergodic metric instead of restating it. Lower is
    better; the unit is domain widths.

    The voxel list is chunked because a 64^3 volume against a 100-point curve is
    26M pairwise distances per sample — fine in slices, wasteful in one block.

    curves: (B, T, 3), density: (R, R, R) -> (B,)
    """
    dev = curves.device
    Dz, Hy, Wx = density.shape
    zs = torch.linspace(0, 1, Dz, device=dev, dtype=curves.dtype)
    ys = torch.linspace(0, 1, Hy, device=dev, dtype=curves.dtype)
    xs = torch.linspace(0, 1, Wx, device=dev, dtype=curves.dtype)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing='ij')
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)

    w = density.reshape(-1).clamp(min=0).to(curves.dtype)
    keep = w > w.max() * 1e-3            # skip the empty background
    pts, w = pts[keep], w[keep]
    w = w / w.sum().clamp(min=1e-12)

    out = []
    for c in curves:
        acc = torch.zeros((), device=dev, dtype=curves.dtype)
        for s in range(0, pts.shape[0], chunk):
            p, ww = pts[s:s + chunk], w[s:s + chunk]
            acc = acc + (ww * torch.cdist(p, c).min(dim=1).values).sum()
        out.append(acc)
    return torch.stack(out)


def path_length(curves):
    """Arc length. (B, T, nd) -> (B,). Guards against 'good only by wandering'."""
    return (curves[:, 1:] - curves[:, :-1]).norm(dim=-1).sum(dim=1)


def planarity(curves):
    """RMS distance of the curve from its own best-fit plane. (B, T, 3) -> (B,).

    Not part of the 2D code — added because the whole point of this port is that
    the data is 3D but planar. A generated curve that stays planar has a value
    near 0; one that genuinely uses the third dimension does not. It is the
    cheapest check that the 3D pipeline has not silently collapsed to 2D, and
    equally the cheapest check that it *has* reproduced the planar reference.
    """
    c = curves - curves.mean(dim=1, keepdim=True)
    # Smallest singular value of the centred point set = spread along the normal.
    s = torch.linalg.svdvals(c)                       # (B, 3), descending
    return s[:, -1] / math.sqrt(curves.shape[1])


def sample_diversity(cps):
    """Mean pairwise RMS distance between samples for one target.

    (n, nxi, nd) -> float. Near zero means the model ignores its noise input.
    """
    n = cps.shape[0]
    if n < 2:
        return float('nan')
    flat = cps.reshape(n, -1)
    d = torch.cdist(flat, flat)
    iu = torch.triu_indices(n, n, offset=1)
    return (d[iu[0], iu[1]].mean() / (flat.shape[1] ** 0.5)).item()


# ── Diversity ────────────────────────────────────────────────────────────────
def diversity_reward(X, h=None):
    """Mean pairwise distance between the K candidates for one target.

    RBF kernel with median bandwidth, returned as a *reward* (higher = more
    diverse), so the training loss subtracts it. Dimension-agnostic.

    X: (K, nxi, nd) -> scalar
    """
    Kn = X.shape[0]
    if Kn < 2:
        return X.new_zeros(())
    flat = X.reshape(Kn, -1)
    sq = torch.cdist(flat, flat).pow(2)
    if h is None:
        off = sq[~torch.eye(Kn, dtype=torch.bool, device=X.device)]
        med = off.median().clamp(min=1e-8)
        h = (med / math.log(Kn + 1)).clamp(min=0.1)
    Kmat = torch.exp(-sq / h)
    off_mean = (Kmat.sum() - Kmat.diagonal().sum()) / (Kn * (Kn - 1))
    return 1.0 - off_mean


def diversity_reward_batched(X, h=None):
    """Same reward for B targets at once. X: (B, K, nxi, nd) -> scalar."""
    Bsz, Kn = X.shape[0], X.shape[1]
    if Kn < 2:
        return X.new_zeros(())
    flat = X.reshape(Bsz, Kn, -1)
    sq = torch.cdist(flat, flat).pow(2)                      # (B, K, K)

    eye = torch.eye(Kn, dtype=torch.bool, device=X.device).expand(Bsz, -1, -1)
    if h is None:
        off = sq[~eye].view(Bsz, Kn * (Kn - 1))
        med = off.median(dim=1).values.clamp(min=1e-8)
        h = (med / math.log(Kn + 1)).clamp(min=0.1).view(Bsz, 1, 1)
    Kmat = torch.exp(-sq / h)
    diag = Kmat.diagonal(dim1=1, dim2=2).sum(dim=1)
    off_mean = (Kmat.sum(dim=(1, 2)) - diag) / (Kn * (Kn - 1))
    return (1.0 - off_mean).mean()
