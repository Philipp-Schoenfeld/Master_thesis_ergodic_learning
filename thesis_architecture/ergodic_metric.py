r"""
ergodic_metric.py
=================
Differentiable ergodic coverage metric for use inside the training loss.

The metric is the standard weighted spectral distance between the trajectory's
time-averaged Fourier statistics and the target distribution's coefficients:

    E = sum_k  Lambda_k * (c_k - phi_k)^2
    c_k   = (1/T) sum_t F_k(x_t)          time average along the trajectory
    phi_k = E_{x ~ p}[ F_k(x) ]           target distribution coefficients
    F_k(x, y) = cos(pi k1 x) * cos(pi k2 y)
    Lambda_k  = (1 + |k|^2)^(-3/2)

Frequency grid and Lambda weighting match `_make_ergodic_k_grid` in
`flow_matching_runner_ergodic.py`, so the number stays comparable to the
spectral conditioning branch and to the SVGD / OT-CFM baselines.

Two things make this usable as a *training* loss rather than an evaluation
metric:

1. Flow matching predicts a velocity, not a trajectory, so the endpoint is
   estimated in one step as `x1_hat = x_t + (1 - t) * v_t` and the metric is
   applied to that. The estimate is poor at small t, so the term is weighted by
   `t^power`, which concentrates it where the estimate is meaningful.
2. `phi_k` is estimated from the *conditioning particles*, not from the density
   grid. The training batch is geometrically augmented (rotation, scale,
   translation, flip) and the particles are augmented in lockstep with the
   trajectory, so particle-derived coefficients stay consistent with the
   augmented target while grid-derived ones would not.
"""

import numpy as np
import torch
import torch.nn as nn

# Cached B-spline basis (curve = B @ control_points); shared with obstacles.py
# so the two never drift apart.
from obstacles import bspline_basis_matrix


def make_k_grid(K):
    """K^2 frequency indices and their Lambda weights.

    Same convention as `_make_ergodic_k_grid` in flow_matching_runner_ergodic.py.
    """
    k_idx = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)],
                     dtype=np.int64)
    Lambda = (1.0 + np.sum(k_idx ** 2, axis=1)) ** (-1.5)
    return k_idx, Lambda


def fourier_basis(pts, k_idx):
    """F_k evaluated at points.

    pts:   (B, T, 2)
    k_idx: (M, 2)
    -> (B, T, M)
    """
    args = np.pi * pts[:, :, None, :] * k_idx[None, None, :, :]
    return torch.cos(args).prod(dim=-1)


def trajectory_coeffs(curve, k_idx):
    """Time-averaged coefficients c_k. curve: (B, T, 2) -> (B, M)."""
    return fourier_basis(curve, k_idx).mean(dim=1)


def target_coeffs_from_particles(particles, k_idx, weighted=True):
    """Target coefficients phi_k from the conditioning particle cloud.

    particles: (B, N, 3) with (x, y, mu)

    With `sample_mode='uniform'` the particles are spread uniformly over the
    support rather than drawn from the density, so each one carries its density
    value mu as an importance weight and phi_k is the mu-weighted mean. With
    density-proportional sampling the particles already follow p, and the plain
    mean is the unbiased estimate — pass weighted=False there.
    """
    Fk = fourier_basis(particles[..., :2], k_idx)          # (B, N, M)
    if not weighted:
        return Fk.mean(dim=1)
    w = particles[..., 2:3].clamp(min=0.0)                 # (B, N, 1)
    return (Fk * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-8)


class ErgodicLoss(nn.Module):
    """Weighted spectral coverage error, ready to add to the flow-matching loss.

    Args:
        nxi:     number of control points
        K:       frequency grid is K x K, so K^2 coefficients
        pts:     points at which the B-spline curve is sampled for the time
                 average. Evaluating on the rendered curve rather than on the
                 raw control points measures coverage of the path that is
                 actually executed.
        deg:     B-spline degree
        weight:  lambda multiplying this term in the total loss
        t_power: exponent of the t-ramp; 0 disables the ramp
        weighted_target: see `target_coeffs_from_particles`
    """

    def __init__(self, nxi=25, K=8, pts=128, deg=5, weight=1.0, t_power=2.0,
                 weighted_target=True):
        super().__init__()
        self.K, self.weight, self.t_power = K, weight, t_power
        self.weighted_target = weighted_target

        k_idx, Lambda = make_k_grid(K)
        self.register_buffer('k_idx', torch.tensor(k_idx, dtype=torch.float32))
        self.register_buffer('Lambda', torch.tensor(Lambda, dtype=torch.float32))
        self.register_buffer('B', torch.from_numpy(
            bspline_basis_matrix(nxi, pts, deg)))

    def extra_repr(self):
        return (f"K={self.K} ({self.K ** 2} modes), weight={self.weight}, "
                f"t_power={self.t_power}")

    def coverage_error(self, cps, particles):
        """Per-sample ergodic error, no t-weighting. (B, nxi, 2), (B, N, 3) -> (B,)"""
        # Run in float32: the batch is trained under bfloat16 autocast, and the
        # cosine basis averaged over 128 points loses too much precision there.
        cps = cps.float()
        curve = torch.einsum('pi,bid->bpd', self.B, cps)
        c = trajectory_coeffs(curve, self.k_idx)
        phi = target_coeffs_from_particles(particles.float(), self.k_idx,
                                           self.weighted_target)
        return (self.Lambda * (c - phi) ** 2).sum(dim=-1)

    def forward(self, x1_hat, particles, t):
        """Scalar loss term. x1_hat: (B, nxi, 2), particles: (B, N, 3), t: (B,)"""
        err = self.coverage_error(x1_hat, particles)
        if self.t_power > 0:
            err = err * t.float() ** self.t_power
        return err.mean()
