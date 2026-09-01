r"""
curvature.py
============
Constraint 3: **maximum curvature** -- a hard turn-radius bound, i.e. the
kinematic feasibility a non-holonomic robot actually needs.

The B-spline already guarantees `C^k` smoothness, but smooth is not the same
as *drivable*: a degree-5 spline can still fold into a hairpin far tighter
than any minimum turn radius. This constraint bounds it explicitly:

    kappa = |c' x c''| / |c'|^3          (discrete, central differences)
    E_k   = 0.5 * mean( relu(kappa/kappa_max - 1)^2 )

Three deliberate choices:

* The derivatives come from finite differences over neighbouring dense-curve
  samples, the same way the `MPDLayer`'s `kernel_size=3` convolutions obtain
  velocity/acceleration implicitly rather than from an explicit velocity
  channel in the state.

* The hinge is on the *relative* overshoot `kappa/kappa_max`, not on `kappa`
  itself. Curvature is 1/radius, so raw values run into the hundreds wherever
  the path nearly cusps, and an absolute hinge would hand the ODE a gradient
  orders of magnitude larger than the model's own velocity. The relative form
  is dimensionless, which keeps one guidance weight usable across shapes.

* **A length guard is mandatory, not cosmetic.** Curvature is not scale
  invariant: inflating the whole curve lowers kappa everywhere, so a pure
  curvature penalty is unbounded below and descent happily takes that exit --
  measured here, an unguarded run met the bound by stretching a length-4.55
  path to length 240 inside the unit square. Real solvers never see this
  because the ergodic objective and the endpoint constraints pin the scale;
  standalone, the degenerate direction has to be closed by hand:

    E = E_k + w_L * 0.5 * relu(L/L_ref - 1)^2

  The guard is one-sided, so the curve may still *shorten* by rounding off
  hairpins -- which is exactly the intended repair -- but it cannot buy
  curvature by growing.

Both terms couple neighbouring curve points, so the gradient goes through
`common.curve_energy_grad` (autograd), not the pointwise
`curve_repulsion_grad` shortcut.
"""

import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import curvature, arc_length


class MaxCurvature:
    """Bound the discrete curvature of the rendered curve by `kappa_max`.

    `length_ref` activates the anti-inflation guard described in the module
    docstring; pass the unguided curve's own length. Without it the constraint
    is satisfiable by blow-up and the result is meaningless.
    """

    def __init__(self, kappa_max=30.0, length_ref=None, length_weight=1.0):
        self.kappa_max = float(kappa_max)
        self.length_ref = None if length_ref is None else float(length_ref)
        self.length_weight = float(length_weight)

    def __repr__(self):
        return (f"MaxCurvature(kappa_max={self.kappa_max:.2f}, "
                f"min_radius={self.min_radius:.4f}, length_ref={self.length_ref})")

    @property
    def min_radius(self):
        return 1.0 / self.kappa_max

    def energy(self, curve):
        """Scalar penalty for `common.curve_energy_grad`."""
        k = curvature(curve)
        e = 0.5 * (torch.clamp(k / self.kappa_max - 1.0, min=0.0) ** 2).mean()
        if self.length_ref is not None:
            over = torch.clamp(arc_length(curve) / self.length_ref - 1.0, min=0.0)
            e = e + self.length_weight * 0.5 * (over ** 2).sum()
        return e

    def violation(self, curve):
        """(B, pts-2) absolute curvature overshoot, 0 where the bound holds."""
        return torch.clamp(curvature(curve) - self.kappa_max, min=0.0)

    def report(self, curve):
        """(max kappa, 99th-percentile kappa, fraction of samples over the bound).

        The peak and the p99 are the meaningful numbers. The *fraction* over the
        bound is reported for completeness but must not be read as a success
        measure when `kappa_max` is itself a quantile of the unguided curve: it
        then starts at exactly 1 - q by construction, and satisfying the bound
        pushes the former outliers down *onto* the threshold, where they keep
        straddling it. A run can crush the peak from 26773 to 40 while that
        fraction barely moves.
        """
        k = curvature(curve)
        return (k.max().item(),
                torch.quantile(k.flatten(), 0.99).item(),
                (k > self.kappa_max).float().mean().item())
