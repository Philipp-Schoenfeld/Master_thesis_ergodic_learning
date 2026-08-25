r"""
ergodic_metric.py  —  3D port
=============================
Differentiable ergodic coverage metric for use inside the training loss.

    E = sum_k  Lambda_k * (c_k - phi_k)^2
    c_k   = (1/T) sum_t F_k(x_t)          time average along the trajectory
    phi_k = E_{x ~ p}[ F_k(x) ]           target distribution coefficients
    F_k(x, y, z) = cos(pi k1 x) cos(pi k2 y) cos(pi k3 z)
    Lambda_k     = (1 + |k|^2)^(-(d+1)/2) = (1 + |k|^2)^-2 in 3D

Frequency grid and Lambda weighting are shared with `ergodic_energy_torch.py`
so the loss term and the evaluation metric can never drift apart.

The two things that make this usable as a *training* loss carry over unchanged
from the 2D version:

1. Flow matching predicts a velocity, not a trajectory, so the endpoint is
   estimated in one step as `x1_hat = x_t + (1 - t) * v_t` and the metric is
   applied to that. The estimate is poor at small t, so the term is weighted by
   `t^power`, which concentrates it where the estimate is meaningful.
2. `phi_k` is estimated from the *conditioning particles*, not from the density
   volume. The training batch is geometrically augmented and the particles are
   augmented in lockstep with the trajectory, so particle-derived coefficients
   stay consistent with the augmented target while grid-derived ones would not.

Cost note: at K = 8 this is 512 modes in 3D rather than 64 in 2D. The basis is
evaluated on `pts` curve samples per trajectory, so the term costs 8x more than
its 2D counterpart at the same K — worth knowing when picking `--erg_K`.
"""

import numpy as np
import torch
import torch.nn as nn

# Cached B-spline basis (curve = B @ control_points); shared with obstacles.py
# so the two never drift apart.
from obstacles import bspline_basis_matrix
from ergodic_energy_torch import make_k_grid, ND, LAMBDA_EXPONENT


def fourier_basis(pts, k_idx):
    """F_k evaluated at points.

    pts:   (B, T, nd)
    k_idx: (M, nd)
    -> (B, T, M)
    """
    args = np.pi * pts[:, :, None, :] * k_idx[None, None, :, :]
    return torch.cos(args).prod(dim=-1)


def trajectory_coeffs(curve, k_idx):
    """Time-averaged coefficients c_k. curve: (B, T, nd) -> (B, M)."""
    return fourier_basis(curve, k_idx).mean(dim=1)


def target_coeffs_from_particles(particles, k_idx, weighted=True, nd=ND):
    """Target coefficients phi_k from the conditioning particle cloud.

    particles: (B, N, nd + 1) with (x, y, z, mu) in 3D

    With `sample_mode='uniform'` the particles are spread uniformly over the
    support rather than drawn from the density, so each one carries its density
    value mu as an importance weight and phi_k is the mu-weighted mean. With
    density-proportional sampling the particles already follow p, and the plain
    mean is the unbiased estimate — pass weighted=False there.
    """
    Fk = fourier_basis(particles[..., :nd], k_idx)          # (B, N, M)
    if not weighted:
        return Fk.mean(dim=1)
    w = particles[..., nd:nd + 1].clamp(min=0.0)            # (B, N, 1)
    return (Fk * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-8)


class ErgodicLoss(nn.Module):
    """Weighted spectral coverage error, ready to add to the flow-matching loss.

    Args:
        nxi:     number of control points
        K:       frequency grid is K x K x K, so K^3 coefficients
        pts:     points at which the B-spline curve is sampled for the time
                 average. Evaluating on the rendered curve rather than on the
                 raw control points measures coverage of the path that is
                 actually executed.
        deg:     B-spline degree
        weight:  lambda multiplying this term in the total loss
        t_power: exponent of the t-ramp; 0 disables the ramp
        weighted_target: see `target_coeffs_from_particles`
        nd:      coordinate dimension (3 here)
    """

    def __init__(self, nxi=25, K=8, pts=128, deg=5, weight=1.0, t_power=2.0,
                 weighted_target=True, nd=ND, lambda_exponent=LAMBDA_EXPONENT,
                 ergodic_on='position', mu_thresh=0.5, sensor_axis_index=2):
        super().__init__()
        if ergodic_on not in ('position', 'footprint'):
            raise ValueError(f"unknown ergodic_on: {ergodic_on}")
        self.K, self.weight, self.t_power = K, weight, t_power
        self.weighted_target = weighted_target
        self.nd = nd
        self.ergodic_on = ergodic_on
        self.mu_thresh = mu_thresh
        self.sensor_axis_index = sensor_axis_index

        k_idx, Lambda = make_k_grid(K, nd=nd, exponent=lambda_exponent)
        self.register_buffer('k_idx', torch.tensor(k_idx, dtype=torch.float32))
        self.register_buffer('Lambda', torch.tensor(Lambda, dtype=torch.float32))
        self.register_buffer('B', torch.from_numpy(
            bspline_basis_matrix(nxi, pts, deg)))

    def extra_repr(self):
        return (f"K={self.K} ({self.K ** self.nd} modes), nd={self.nd}, "
                f"on={self.ergodic_on}, "
                f"weight={self.weight}, t_power={self.t_power}")

    def coverage_error(self, cps, particles, rot6d=None, surface=None):
        """Per-sample ergodic error, no t-weighting.

        (B, nxi, nd), (B, N, nd+1) -> (B,)

        With `rot6d` given and `ergodic_on='footprint'`, coverage is scored
        where the sensor beam lands instead of where the robot is. That is what
        makes orientation part of the task rather than a decoration: scored on
        the position alone, the ergodic term is completely blind to where the
        sensor looks, and any orientation objective merely runs alongside it.
        With a standoff the robot is deliberately *not* on the surface, so its
        own position is the wrong thing to score in the first place.
        """
        # Run in float32: the batch is trained under bfloat16 autocast, and the
        # cosine basis averaged over many points loses too much precision there.
        cps = cps.float()
        curve = torch.einsum('pi,bid->bpd', self.B, cps)

        if self.ergodic_on == 'footprint':
            if rot6d is None:
                raise ValueError("ergodic_on='footprint' needs rot6d")
            from orientation import rot6d_to_matrix, sensor_axis
            from orientation_energy import ParticleSurface
            r6 = torch.einsum('pi,bic->bpc', self.B, rot6d.float())
            a = sensor_axis(rot6d_to_matrix(r6), self.sensor_axis_index)
            surf = surface or ParticleSurface(particles.float(), self.mu_thresh)
            curve = surf.footprint(curve, a)

        c = trajectory_coeffs(curve, self.k_idx)
        phi = target_coeffs_from_particles(particles.float(), self.k_idx,
                                           self.weighted_target, self.nd)
        return (self.Lambda * (c - phi) ** 2).sum(dim=-1)

    def forward(self, x1_hat, particles, t, rot6d=None, surface=None):
        """Scalar loss term. x1_hat: (B, nxi, nd), particles: (B, N, nd+1), t: (B,)"""
        err = self.coverage_error(x1_hat, particles, rot6d, surface)
        if self.t_power > 0:
            err = err * t.float() ** self.t_power
        return err.mean()
