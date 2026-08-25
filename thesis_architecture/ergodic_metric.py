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

Since August 2026 a second discrepancy measure is available behind
`metric='sinkhorn'`: the debiased Sinkhorn divergence, which measures the same
thing without a basis truncation. See `ErgodicLoss` for why that matters and
for the reference (Sun, Pinosky & Murphey 2025, arXiv:2504.17872). The default
stays `fourier`, so nothing changes unless the flag is set.

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


def sinkhorn_error(curve, particles, loss_fn, weighted=True, eps=1e-8):
    """Debiased Sinkhorn divergence between the path and the target cloud.

    curve:     (B, T, 2) rendered trajectory points
    particles: (B, N, 3) conditioning cloud, (x, y, mu)
    loss_fn:   a `geomloss.SamplesLoss`
    -> (B,)

    Both sides are passed as *weighted* point clouds. The trajectory carries
    uniform weights 1/T, which is exactly the empirical measure the ergodic
    metric time-averages over. The target carries mu-normalised weights for the
    same reason `target_coeffs_from_particles` does: with `sample_mode='uniform'`
    the particles are spread over the support rather than drawn from the density,
    so mu is an importance weight, not a decoration. Passing them unweighted
    would silently optimise coverage of the *support* instead of the density.
    """
    B, T, _ = curve.shape
    N = particles.shape[1]
    a = curve.new_full((B, T), 1.0 / T)

    if weighted:
        w = particles[..., 2].clamp(min=0.0)
        b = w / w.sum(dim=1, keepdim=True).clamp(min=eps)
    else:
        b = curve.new_full((B, N), 1.0 / N)

    # geomloss reshapes its inputs with .view(), which fails on the
    # non-contiguous tensor that einsum returns. Making both sides contiguous
    # here is cheaper than debugging it at the call site every time.
    return loss_fn(a.contiguous(), curve.contiguous(),
                   b.contiguous(), particles[..., :2].contiguous())


class ErgodicLoss(nn.Module):
    """Coverage error, ready to add to the flow-matching loss.

    Two discrepancy measures are available between the trajectory's empirical
    distribution and the target:

    * ``fourier`` — the Sobolev distance on a truncated cosine basis, i.e. the
      metric the SVGD/TSVEC solvers minimise. Default, unchanged behaviour.
    * ``sinkhorn`` — the debiased Sinkhorn divergence, an entropic-optimal-
      transport metric with no basis truncation at all.

    Why the second one exists: the K x K truncation caps what the Fourier metric
    can even see. The ablation over lambda from 2 to 400 drove E_ergodic from
    4.25 down to 2.14 while the basis-free coverage distance sat at 0.049-0.052
    throughout and never beat the solver — the signature of optimising a
    truncated proxy rather than coverage itself. Sun, Pinosky & Murphey (2025,
    arXiv:2504.17872) make the same argument and use exactly this replacement.
    It is also much closer to the `coverage_distance` used at evaluation time,
    which is itself an OT-flavoured quantity.

    Args:
        nxi:     number of control points
        K:       frequency grid is K x K, so K^2 coefficients (fourier only)
        pts:     points at which the B-spline curve is sampled. Evaluating on the
                 rendered curve rather than on the raw control points measures
                 coverage of the path that is actually executed.
        deg:     B-spline degree
        weight:  lambda multiplying this term in the total loss
        t_power: exponent of the t-ramp; 0 disables the ramp
        weighted_target: see `target_coeffs_from_particles`
        metric:  'fourier' | 'sinkhorn' | 'both'
        sinkhorn_blur: entropic length scale, in domain widths. 0.05 is 5 % of
                 the unit square — small enough to resolve a letter stroke,
                 large enough to stay numerically calm.
        sinkhorn_scaling: density of the epsilon-annealing schedule. geomloss
                 defaults to 0.9 for an accurate *value*; 0.5 is 6x cheaper and
                 measurably just as good for the only thing training uses, the
                 gradient (0.16 % apart on real batches, against 4 % on the
                 value). See test_sinkhorn_metric.py for the measurement.
        sinkhorn_ratio: with metric='both', the factor on the Sinkhorn part
                 before the two are summed. The two measures differ by roughly
                 an order of magnitude on real batches, so this is not cosmetic.
    """

    def __init__(self, nxi=25, K=8, pts=128, deg=5, weight=1.0, t_power=2.0,
                 weighted_target=True, metric='fourier',
                 sinkhorn_blur=0.05, sinkhorn_p=2, sinkhorn_scaling=0.5,
                 sinkhorn_ratio=1.0):
        super().__init__()
        if metric not in ('fourier', 'sinkhorn', 'both'):
            raise ValueError(f"unknown metric: {metric}")
        self.K, self.weight, self.t_power = K, weight, t_power
        self.weighted_target = weighted_target
        self.metric = metric
        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_scaling = sinkhorn_scaling
        self.sinkhorn_ratio = sinkhorn_ratio

        k_idx, Lambda = make_k_grid(K)
        self.register_buffer('k_idx', torch.tensor(k_idx, dtype=torch.float32))
        self.register_buffer('Lambda', torch.tensor(Lambda, dtype=torch.float32))
        self.register_buffer('B', torch.from_numpy(
            bspline_basis_matrix(nxi, pts, deg)))

        self._sinkhorn = None
        if metric in ('sinkhorn', 'both'):
            # Imported lazily so a pure-Fourier run needs no extra dependency.
            try:
                from geomloss import SamplesLoss
            except ImportError as e:
                raise ImportError(
                    "metric='sinkhorn' needs geomloss (pip install geomloss). "
                    "This is the package Sun et al. use for the same purpose."
                ) from e
            self._sinkhorn = SamplesLoss(
                loss='sinkhorn', p=sinkhorn_p, blur=sinkhorn_blur,
                scaling=sinkhorn_scaling, debias=True, backend='tensorized')

    def extra_repr(self):
        bits = [f"metric={self.metric}", f"weight={self.weight}",
                f"t_power={self.t_power}"]
        if self.metric in ('fourier', 'both'):
            bits.insert(1, f"K={self.K} ({self.K ** 2} modes)")
        if self.metric in ('sinkhorn', 'both'):
            bits.insert(-1, f"blur={self.sinkhorn_blur}, "
                            f"scaling={self.sinkhorn_scaling}")
        if self.metric == 'both':
            bits.append(f"sinkhorn_ratio={self.sinkhorn_ratio}")
        return ', '.join(bits)

    def render(self, cps):
        """Control points -> curve, in float32."""
        return torch.einsum('pi,bid->bpd', self.B, cps.float())

    def coverage_error(self, cps, particles, return_parts=False):
        """Per-sample coverage error, no t-weighting.

        (B, nxi, 2), (B, N, 3) -> (B,), or ((B,), dict) when return_parts.
        """
        # Run in float32: the batch is trained under bfloat16 autocast, and both
        # the cosine basis averaged over 128 points and the Sinkhorn iterations
        # lose too much precision there.
        curve = self.render(cps)
        particles = particles.float()
        parts = {}

        if self.metric in ('fourier', 'both'):
            c = trajectory_coeffs(curve, self.k_idx)
            phi = target_coeffs_from_particles(particles, self.k_idx,
                                               self.weighted_target)
            parts['fourier'] = (self.Lambda * (c - phi) ** 2).sum(dim=-1)

        if self.metric in ('sinkhorn', 'both'):
            parts['sinkhorn'] = sinkhorn_error(curve, particles, self._sinkhorn,
                                               self.weighted_target)

        if self.metric == 'fourier':
            err = parts['fourier']
        elif self.metric == 'sinkhorn':
            err = parts['sinkhorn']
        else:
            err = parts['fourier'] + self.sinkhorn_ratio * parts['sinkhorn']

        return (err, parts) if return_parts else err

    def forward(self, x1_hat, particles, t, return_parts=False):
        """Scalar loss term. x1_hat: (B, nxi, 2), particles: (B, N, 3), t: (B,)"""
        out = self.coverage_error(x1_hat, particles, return_parts=return_parts)
        err, parts = out if return_parts else (out, None)
        if self.t_power > 0:
            err = err * t.float() ** self.t_power
        if return_parts:
            return err.mean(), {k: v.mean().detach() for k, v in parts.items()}
        return err.mean()
