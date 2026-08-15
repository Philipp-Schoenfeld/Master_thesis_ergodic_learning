r"""
ergodic_energy_torch.py
=======================
Differentiable PyTorch port of the TSVEC solver energy
(`SE3_SVGD/tsvec_2d.py :: compute_energy_and_grad`).

Four terms, all batched over ``(B, T, 2)`` point sets:

    E = W_SMOOTH   * sum(accel^2)
      + W_ERGODIC  * 0.5 * sum(Lambda_k * (c_k - phi_k)^2)
      + W_BOUNDARY * 0.5 * (sum(lo^2) + sum(hi^2))
      + W_OBSTACLE * 0.5 * sum(max(r - dist, 0)^2)

The weights are taken verbatim from the solver and deliberately **not** re-tuned,
so a model trained against this objective stays comparable to the classical
solver. Gradients come from autograd — the hand-derived gradient formulas in the
original (lines 151-153, 160-162, 169, 180-181) are unnecessary here.

Conventions that must match the solver exactly, and do:
  * Fourier basis  F_k(x) = cos(pi*k1*x) * cos(pi*k2*y), no normalisation.
  * Lambda_k = (1 + k1^2 + k2^2)^(-3/2)
  * K = 10  ->  100 modes, ordered k1 outer, k2 inner.
  * c_k is the *mean* over the T points (the time average).
  * Smoothness has no 0.5 factor; the other three do.

Note on where to evaluate: the solver optimises T=100 dense trajectory points.
A network emitting nxi=25 B-spline control points is not the same thing — the
time average over 25 control points differs from the average over the executed
path. Use `energy_from_control_points` (or pass `basis=` to `ErgodicEnergy`) to
render the curve first; evaluating directly on control points is available but
is not solver-equivalent.
"""

import math
import numpy as np
import torch
import torch.nn as nn

# ── Constants, verbatim from SE3_SVGD/tsvec_2d.py ────────────────────────────
W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0
W_OBSTACLE = 50000.0

BOUNDARY_MARGIN = 0.03          # tsvec_2d.py:165
K_DEFAULT = 10                  # tsvec_2d.py:80
OBSTACLE_CENTER = (0.5, 0.5)
OBSTACLE_RADIUS = 0.12


def make_k_grid(K=K_DEFAULT):
    """(K^2, 2) index grid and (K^2,) weights — same ordering as the solver.

    Computed in float64: Lambda_k = (1+k^2)^(-3/2) in float32 costs ~1e-7
    relative precision, which is enough to break an exact comparison against the
    NumPy solver. Callers cast down to float32 for training.
    """
    k_idx = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)],
                     dtype=np.float64)
    Lambda = (1.0 + (k_idx ** 2).sum(axis=1)) ** (-1.5)
    return k_idx, Lambda


def fourier_basis(pts, k_idx):
    """F_k at each point. pts: (..., T, 2), k_idx: (M, 2) -> (..., T, M)."""
    args = math.pi * pts.unsqueeze(-2) * k_idx
    return torch.cos(args).prod(dim=-1)


def coeffs_from_points(pts, k_idx):
    """Time-averaged ergodic coefficients c_k. (..., T, 2) -> (..., M)."""
    return fourier_basis(pts, k_idx).mean(dim=-2)


def target_coeffs_from_grid(density, k_idx, eps=1e-12):
    """Target coefficients phi_k from a density map on [0,1]^2.

    Mirrors tsvec_2d.py:98-101 — normalise the density to sum 1 over the grid
    and take the weighted average of the basis.

    density: (H, W) or (B, H, W)   ->   (M,) or (B, M)
    """
    single = density.dim() == 2
    d = density.unsqueeze(0) if single else density
    Bsz, H, W = d.shape

    ys = torch.linspace(0, 1, H, device=d.device, dtype=d.dtype)
    xs = torch.linspace(0, 1, W, device=d.device, dtype=d.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)     # (H*W, 2)

    Fk = fourier_basis(grid, k_idx.to(d.dtype))                      # (H*W, M)
    w = d.reshape(Bsz, -1)
    w = w / w.sum(dim=1, keepdim=True).clamp(min=eps)
    phi = w @ Fk                                                     # (B, M)
    return phi.squeeze(0) if single else phi


# ── Individual terms ─────────────────────────────────────────────────────────
def smoothness_term(X, w=W_SMOOTH):
    """w * sum(accel^2), per sample. X: (B, T, 2) -> (B,)"""
    accel = X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
    return w * accel.pow(2).sum(dim=(1, 2))


def ergodic_term(X, k_idx, Lambda, phi_k, w=W_ERGODIC):
    """w * 0.5 * sum(Lambda_k * (c_k - phi_k)^2). X: (B, T, 2) -> (B,)"""
    c_k = coeffs_from_points(X, k_idx)                # (B, M)
    if phi_k.dim() == 1:
        phi_k = phi_k.unsqueeze(0)
    diff = c_k - phi_k
    return w * 0.5 * (Lambda * diff.pow(2)).sum(dim=-1)


def boundary_term(X, margin=BOUNDARY_MARGIN, w=W_BOUNDARY):
    """w * 0.5 * (sum(lo^2) + sum(hi^2)). X: (B, T, 2) -> (B,)"""
    lo = (X - margin).clamp(max=0.0)
    hi = (X - (1.0 - margin)).clamp(min=0.0)
    return w * 0.5 * (lo.pow(2).sum(dim=(1, 2)) + hi.pow(2).sum(dim=(1, 2)))


def obstacle_term(X, center=OBSTACLE_CENTER, radius=OBSTACLE_RADIUS, w=W_OBSTACLE):
    """w * 0.5 * sum(max(r - dist, 0)^2). X: (B, T, 2) -> (B,)"""
    c = torch.as_tensor(center, dtype=X.dtype, device=X.device)
    dist = ((X - c) ** 2).sum(-1).add(1e-12).sqrt()
    violation = (radius - dist).clamp(min=0.0)
    return w * 0.5 * violation.pow(2).sum(dim=1)


# ── Assembled energy ─────────────────────────────────────────────────────────
class ErgodicEnergy(nn.Module):
    """Solver energy as a differentiable module.

    Args:
        K:        frequency grid is K x K (solver uses 10).
        basis:    optional (T, nxi) B-spline basis. When given, inputs are read
                  as control points and rendered to a T-point curve before the
                  energy is evaluated — this is the solver-equivalent mode.
        use_obstacle: the solver defaults to False; enable to train paths that
                  avoid the obstacle instead of only steering around it at
                  inference time.
    """

    def __init__(self, K=K_DEFAULT, basis=None, use_obstacle=False,
                 obstacle_center=OBSTACLE_CENTER, obstacle_radius=OBSTACLE_RADIUS,
                 boundary_margin=BOUNDARY_MARGIN,
                 w_ergodic=W_ERGODIC, w_smooth=W_SMOOTH,
                 w_boundary=W_BOUNDARY, w_obstacle=W_OBSTACLE):
        super().__init__()
        k_idx, Lambda = make_k_grid(K)
        self.register_buffer('k_idx', torch.from_numpy(k_idx).float())
        self.register_buffer('Lambda', torch.from_numpy(Lambda).float())
        if basis is not None:
            b = basis if torch.is_tensor(basis) else torch.as_tensor(basis)
            self.register_buffer('basis', b.float())
        else:
            self.basis = None
        self.use_obstacle = use_obstacle
        self.obstacle_center = obstacle_center
        self.obstacle_radius = obstacle_radius
        self.boundary_margin = boundary_margin
        self.w_ergodic, self.w_smooth = w_ergodic, w_smooth
        self.w_boundary, self.w_obstacle = w_boundary, w_obstacle

    def render(self, X):
        """Control points -> curve, if a basis was supplied."""
        if self.basis is None:
            return X
        return torch.einsum('ti,bid->btd', self.basis, X)

    def forward(self, X, phi_k, return_terms=False):
        """X: (B, nxi, 2) control points (or (B, T, 2) points if basis is None).
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
def coverage_distance(curves, density):
    """Density-weighted mean distance from the target mass to the curve.

    For every grid cell, the distance to the nearest point on the curve,
    averaged with the normalised density as weight. Answers "if I draw a point
    from the target distribution, how far is it from the path?" — it uses no
    Fourier basis at all, so it cross-checks the spectral ergodic metric instead
    of restating it. Lower is better; the unit is domain widths.

    curves: (B, T, 2), density: (H, W) -> (B,)
    """
    dev = curves.device
    H, W = density.shape
    ys = torch.linspace(0, 1, H, device=dev, dtype=curves.dtype)
    xs = torch.linspace(0, 1, W, device=dev, dtype=curves.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    w = density.reshape(-1).clamp(min=0).to(curves.dtype)
    keep = w > w.max() * 1e-3            # skip the empty background
    pts, w = pts[keep], w[keep]
    w = w / w.sum().clamp(min=1e-12)

    return torch.stack([(w * torch.cdist(pts, c).min(dim=1).values).sum()
                        for c in curves])


def path_length(curves):
    """Arc length. (B, T, 2) -> (B,). Guards against 'good only by wandering'."""
    return (curves[:, 1:] - curves[:, :-1]).norm(dim=-1).sum(dim=1)


def sample_diversity(cps):
    """Mean pairwise RMS distance between samples for one target.

    (n, nxi, 2) -> float. Near zero means the model ignores its noise input.
    """
    n = cps.shape[0]
    if n < 2:
        return float('nan')
    flat = cps.reshape(n, -1)
    d = _torch_cdist(flat)
    iu = torch.triu_indices(n, n, offset=1)
    return (d[iu[0], iu[1]].mean() / (flat.shape[1] ** 0.5)).item()


def _torch_cdist(flat):
    return torch.cdist(flat, flat)


# ── Diversity ────────────────────────────────────────────────────────────────
def diversity_reward(X, h=None):
    """Mean pairwise distance between the K candidates for one target.

    Reused directly from what keeps the SVGD particle population apart
    (`tsvec_2d.py:210-216`): an RBF kernel with median bandwidth. Returned as a
    *reward* (higher = more diverse), so the training loss subtracts it.

    X: (K, nxi, 2) -> scalar
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
    # Mean off-diagonal kernel value is high when candidates collapse; the
    # reward is its complement so that maximising it spreads them out.
    Kmat = torch.exp(-sq / h)
    off_mean = (Kmat.sum() - Kmat.diagonal().sum()) / (Kn * (Kn - 1))
    return 1.0 - off_mean


def diversity_reward_batched(X, h=None):
    """Same reward, computed for B targets at once. X: (B, K, nxi, 2) -> scalar.

    The bandwidth is taken per target (as in the single-target version), so a
    target whose candidates are globally far apart does not desensitise one
    whose candidates sit close together.
    """
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
