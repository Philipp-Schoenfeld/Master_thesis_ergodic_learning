r"""
keepin.py
=========
Constraint 1: **keep-in region** -- the mirror image of
`obstacles.CircleObstacle`.

An obstacle penalises being *inside* a forbidden set; a keep-in region
penalises being *outside* an allowed set. Both are one-sided hinge-squared
penalties that vanish where the constraint already holds, so a trajectory that
is comfortably inside the workspace is never pushed around:

    E(P) = 0.5 * sum( outside_distance(P)^2 )

Both classes follow the obstacles.py convention (`violation`, `penalty`,
`grad_penalty`, `draw`) so they plug straight into `curve_repulsion_grad`,
which needs nothing but `grad_penalty`.
"""

import numpy as np
import torch


class KeepInBox:
    """Axis-aligned workspace box the curve must not leave.

    `margin` shrinks the effective box so the curve settles just inside the
    boundary rather than grazing it -- the same cushion role `margin` plays for
    the obstacles.
    """

    def __init__(self, lo=(0.15, 0.15), hi=(0.85, 0.85), margin=0.01):
        self.lo = tuple(float(v) for v in lo)
        self.hi = tuple(float(v) for v in hi)
        self.margin = float(margin)

    def __repr__(self):
        return f"KeepInBox(lo={self.lo}, hi={self.hi}, margin={self.margin})"

    def _bounds(self, P):
        lo = torch.as_tensor(self.lo, dtype=P.dtype, device=P.device) + self.margin
        hi = torch.as_tensor(self.hi, dtype=P.dtype, device=P.device) - self.margin
        return lo, hi

    def violation(self, P):
        """(..., nd) -> (...,). Euclidean distance to the allowed box, 0 inside."""
        lo, hi = self._bounds(P)
        out = torch.clamp(lo - P, min=0.0) + torch.clamp(P - hi, min=0.0)
        return out.norm(dim=-1)

    def penalty(self, P):
        return 0.5 * (self.violation(P) ** 2).sum()

    def grad_penalty(self, P):
        """Analytic: componentwise, the signed overshoot past each face.

        Because the squared distance to a box separates per axis, the gradient
        needs no autograd and is exactly zero for any point already inside.
        """
        lo, hi = self._bounds(P)
        return -torch.clamp(lo - P, min=0.0) + torch.clamp(P - hi, min=0.0)

    def max_violation_raw(self, curve):
        """Deepest excursion past the *raw* box (margin excluded): the margin is
        a guidance cushion, the reported number should be the physical one."""
        lo = torch.as_tensor(self.lo, dtype=curve.dtype, device=curve.device)
        hi = torch.as_tensor(self.hi, dtype=curve.dtype, device=curve.device)
        out = torch.clamp(lo - curve, min=0.0) + torch.clamp(curve - hi, min=0.0)
        return out.norm(dim=-1).max().item()

    def draw(self, ax, zorder=1.5):
        import matplotlib.pyplot as plt
        w, h = self.hi[0] - self.lo[0], self.hi[1] - self.lo[1]
        ax.add_patch(plt.Rectangle(self.lo, w, h, facecolor='#00C853', alpha=0.05,
                                    edgecolor='#424242', lw=1.5, zorder=zorder))
        if self.margin > 0:
            ax.add_patch(plt.Rectangle(
                (self.lo[0] + self.margin, self.lo[1] + self.margin),
                w - 2 * self.margin, h - 2 * self.margin, facecolor='none',
                edgecolor='#424242', lw=0.8, ls=':', alpha=0.6, zorder=zorder))


class KeepInCircle:
    """Circular keep-in region (a disc corridor the curve must stay within)."""

    def __init__(self, center=(0.5, 0.5), radius=0.35, margin=0.01):
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)
        self.margin = float(margin)

    @property
    def effective_radius(self):
        return self.radius - self.margin

    def _delta(self, P):
        c = torch.as_tensor(self.center, dtype=P.dtype, device=P.device)
        return P - c

    def violation(self, P):
        d = self._delta(P).norm(dim=-1)
        return torch.clamp(d - self.effective_radius, min=0.0)

    def penalty(self, P):
        return 0.5 * (self.violation(P) ** 2).sum()

    def grad_penalty(self, P):
        d = self._delta(P)
        dist = d.norm(dim=-1)
        v = torch.clamp(dist - self.effective_radius, min=0.0)
        return (v / dist.clamp(min=1e-12))[..., None] * d

    def max_violation_raw(self, curve):
        d = self._delta(curve).norm(dim=-1)
        return torch.clamp(d - self.radius, min=0.0).max().item()

    def draw(self, ax, zorder=1.5):
        import matplotlib.pyplot as plt
        ax.add_patch(plt.Circle(self.center, self.radius, facecolor='#00C853',
                                 alpha=0.05, edgecolor='#424242', lw=1.5, zorder=zorder))
