r"""
alignment.py
============
Constraint 5: **SE(3) tangent alignment on a lifted surface** -- keep the
curve's heading inside the surface's tangent plane, not just its points on the
surface.

This is the constraint that the 3D-lift experiment (`experiment_3d_lift.py`)
visibly needed. Lifting only pulls *points* onto the zero level set. Between
two control points the B-spline is free to take a chord straight through the
solid, which is why the lifted path on the sharp-edged pyramid kept a residual
|SDF| several times larger than on the smooth sphere: a smooth spline with 64
control points cannot hug a crease by position penalties alone.

Adding a first-order (heading) term fixes the mechanism rather than the
symptom:

    E = w_surf * 0.5 * mean( SDF(C)^2 )        (stay on the surface)
      + w_align * 0.5 * mean( (t_hat . n_hat)^2 )   (travel *along* it)

with `t_hat` the unit tangent from central differences (the `MPDLayer`'s
finite-difference philosophy) and `n_hat = grad SDF / |grad SDF|` the unit
surface normal. A curve that is on the surface *and* whose tangent is
perpendicular to the normal everywhere is, to first order, a curve that stays
on the surface between samples too.

`n_hat` is deliberately treated as a frozen direction field (computed on a
detached copy): the constraint asks "is the heading tangential *here*", not
"how does the normal move if the point moves". That keeps the gradient a
clean first-order term and avoids second derivatives of the SDF, which for the
max-of-planes pyramid do not exist at the creases anyway.

This is the inference-time cousin of the SE(3) constraints the TSVEC solver
carries (Li et al. 2026) and of Flow-Opt's idea (Idoko et al. 2026) of having
the network hand the solver an already constraint-aware initialisation: the
same quantity that would otherwise need an active Lagrange multiplier is here
enforced directly on the generated warm start.
"""

import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import tangents


class SurfaceTangentAlignment:
    """Stay on `shape`'s surface *and* keep the heading in its tangent plane.

    With `w_align=0` this reduces exactly to the position-only attraction the
    3D lift already used, which is what the experiment uses as its baseline.
    """

    def __init__(self, shape, w_surface=1.0, w_align=1.0):
        self.shape = shape
        self.w_surface = float(w_surface)
        self.w_align = float(w_align)

    def __repr__(self):
        return (f"SurfaceTangentAlignment({type(self.shape).__name__}, "
                f"w_surface={self.w_surface:g}, w_align={self.w_align:g})")

    def normals(self, curve):
        """Unit surface normals at the curve points, as a constant field."""
        with torch.enable_grad():
            P = curve.detach().requires_grad_(True)
            (n,) = torch.autograd.grad(self.shape.sdf(P).sum(), P)
        return n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def cos_tn(self, curve):
        """|cos| between unit tangent and unit normal, (B, pts). 0 = tangential."""
        t = tangents(curve)
        t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return (t * self.normals(curve)).sum(dim=-1).abs()

    def energy(self, curve):
        """Scalar penalty for `common.curve_energy_grad`."""
        e = self.w_surface * 0.5 * (self.shape.sdf(curve) ** 2).mean()
        if self.w_align > 0:
            t = tangents(curve)
            t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            cos = (t * self.normals(curve)).sum(dim=-1)
            e = e + self.w_align * 0.5 * (cos ** 2).mean()
        return e

    def report(self, curve):
        """(max |SDF|, mean |cos(t, n)|) -- surface adherence and tangency."""
        return (self.shape.sdf(curve).abs().max().item(),
                self.cos_tn(curve).mean().item())
