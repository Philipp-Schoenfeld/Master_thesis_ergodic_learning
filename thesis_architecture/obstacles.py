r"""
obstacles.py
============
Obstacle definitions and repulsion gradients for inference-time guidance.

The obstacle geometry and the hinge-squared penalty match the convention already
used by the other thesis baselines (``OT_CFM/ot_cfm_2d.py`` and
``SE3_SVGD/svgd_bspline_2d.py``), so results stay directly comparable across
methods:

    E(P) = 0.5 * sum( max(radius - |P - center|, 0)^2 )

All functions accept either numpy arrays or torch tensors and return the same
type, so the same obstacle object can be used for plotting (numpy) and for
guidance inside the ODE solver (torch).
"""

import numpy as np

try:
    import torch as _torch
except ImportError:      # plotting-only usage does not need torch
    _torch = None


# ── Shared convention with OT_CFM / SE3_SVGD baselines ────────────────────────
OBSTACLE_CENTER = (0.5, 0.5)
OBSTACLE_RADIUS = 0.12
OBSTACLE_MARGIN = 0.01      # extra clearance so the curve does not graze the rim


def _is_torch(a):
    return _torch is not None and isinstance(a, _torch.Tensor)


def _relu(a):
    return _torch.clamp(a, min=0.0) if _is_torch(a) else np.maximum(a, 0.0)


class CircleObstacle:
    """Circular no-fly zone in the unit square with an analytic penalty gradient."""

    def __init__(self, center=OBSTACLE_CENTER, radius=OBSTACLE_RADIUS,
                 margin=OBSTACLE_MARGIN):
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)
        self.margin = float(margin)

    @property
    def effective_radius(self):
        return self.radius + self.margin

    def __repr__(self):
        return (f"CircleObstacle(center={self.center}, radius={self.radius}, "
                f"margin={self.margin})")

    # ── Geometry ─────────────────────────────────────────────────────────────
    def _delta(self, P):
        """P: (..., 2) -> offset from centre, same backend as P."""
        if _is_torch(P):
            c = _torch.as_tensor(self.center, dtype=P.dtype, device=P.device)
        else:
            c = np.asarray(self.center, dtype=P.dtype)
        return P - c

    def signed_distance(self, P):
        """(..., 2) -> (...,). Negative inside the (inflated) obstacle."""
        d = self._delta(P)
        dist = (d ** 2).sum(-1) ** 0.5
        return dist - self.effective_radius

    def violation(self, P):
        """(..., 2) -> (...,). Penetration depth, 0 outside the obstacle."""
        return _relu(-self.signed_distance(P))

    # ── Penalty and its gradient ─────────────────────────────────────────────
    def penalty(self, P):
        """Scalar hinge-squared energy, summed over all points."""
        return 0.5 * (self.violation(P) ** 2).sum()

    def grad_penalty(self, P):
        """dE/dP, shape (..., 2). Analytic: -(r - dist)_+ * (P - c) / dist.

        Exactly zero outside the obstacle, so points that already clear it are
        never pushed around.
        """
        d = self._delta(P)
        dist = (d ** 2).sum(-1) ** 0.5
        # Guard the singularity at the centre; the direction there is arbitrary.
        safe = dist + 1e-12
        v = _relu(self.effective_radius - dist)
        return -(v / safe)[..., None] * d

    # ── Helpers for statistics and plotting ──────────────────────────────────
    def mask(self, X, Y, inflated=False):
        """Boolean grid: True where (X, Y) lies inside the obstacle."""
        r = self.effective_radius if inflated else self.radius
        return ((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2) <= r ** 2

    def draw(self, ax, zorder=1.5, show_margin=True):
        """Draw the obstacle in the project's clean white style.

        Drawn above the density heatmap and particles but *below* the
        trajectories, so a path that cuts through stays visible instead of
        being hidden by the patch.
        """
        import matplotlib.pyplot as plt

        if show_margin and self.margin > 0:
            ax.add_patch(plt.Circle(self.center, self.effective_radius,
                                    facecolor='none', edgecolor='#424242',
                                    lw=0.8, ls=':', alpha=0.6, zorder=zorder))
        ax.add_patch(plt.Circle(self.center, self.radius,
                                facecolor='#9E9E9E', alpha=0.75,
                                edgecolor='#424242', lw=1.5, zorder=zorder))


class CompositeObstacle:
    """A collection of obstacles (e.g. drawn via UI)."""
    def __init__(self, obstacles=None):
        self.obstacles = list(obstacles) if obstacles else []

    def mask(self, X, Y, inflated=False):
        m = np.zeros_like(X, dtype=bool)
        for o in self.obstacles:
            m |= o.mask(X, Y, inflated=inflated)
        return m

    def penalty(self, P):
        if not self.obstacles:
            return 0.0 if not _is_torch(P) else _torch.zeros_like(P[..., 0]).sum()
        return sum(o.penalty(P) for o in self.obstacles)

    def grad_penalty(self, P):
        if not self.obstacles:
            return np.zeros_like(P) if not _is_torch(P) else _torch.zeros_like(P)
        return sum(o.grad_penalty(P) for o in self.obstacles)

    def violation(self, P):
        if not self.obstacles:
            return np.zeros_like(P[..., 0]) if not _is_torch(P) else _torch.zeros_like(P[..., 0])
        if _is_torch(P):
            return _torch.stack([o.violation(P) for o in self.obstacles], dim=0).amax(dim=0)
        else:
            return np.max([o.violation(P) for o in self.obstacles], axis=0)

    def draw(self, ax, zorder=1.5, show_margin=True):
        for o in self.obstacles:
            o.draw(ax, zorder=zorder, show_margin=show_margin)


# ── B-Spline coupling ─────────────────────────────────────────────────────────
_BASIS_CACHE = {}


def bspline_basis_matrix(nxi, pts=256, deg=5):
    """Clamped B-spline basis B with curve = B @ control_points. Cached, numpy."""
    key = (nxi, pts, deg)
    if key not in _BASIS_CACHE:
        from bsplinax.bspline import BsplineBasisClamped
        _BASIS_CACHE[key] = np.array(BsplineBasisClamped(
            degree=deg, num_control_points=nxi,
            num_phase_points=pts, compute_derivatives=False).B, dtype=np.float32)
    return _BASIS_CACHE[key]


def basis_torch(nxi, pts, deg, device, dtype=None):
    """Basis matrix as a torch tensor on `device`.

    Casting here explicitly avoids the 'Expected all tensors to be on the same
    device' error that bites whenever a host-side helper array meets GPU data.
    """
    B = _torch.from_numpy(bspline_basis_matrix(nxi, pts, deg))
    return B.to(device=device, dtype=dtype or _torch.float32)


def curve_repulsion_grad(cps, obstacle, B, normalize=True):
    """Repulsion gradient w.r.t. control points, evaluated on the *curve*.

    The convex-hull property means obstacle-free control points do NOT imply an
    obstacle-free curve, so the penalty must be evaluated on the dense curve.
    Since the curve is a *linear* map ``C = B @ cps``, the chain rule is exact
    and needs no autograd:

        dE/d(cps) = B^T @ grad_penalty(B @ cps)

    Args:
        cps:  (B, nxi, nd) control points
        B:    (pts, nxi) basis matrix on the same device
        normalize: divide each control point's gradient by its total basis
            weight ``B.sum(0)``. This turns the sum over curve points into a
            weighted average, making the magnitude independent of `pts` and
            roughly equal to the penetration depth — which is what makes the
            guidance weight interpretable. Set False to get the exact gradient.

    Returns:
        (B, nxi, nd)
    """
    curve = _torch.einsum('pi,bid->bpd', B, cps)      # (B, pts, nd)
    g = obstacle.grad_penalty(curve)                  # (B, pts, nd)
    grad_cp = _torch.einsum('pi,bpd->bid', B, g)      # (B, nxi, nd)
    if normalize:
        grad_cp = grad_cp / B.sum(0).clamp(min=1e-8)[None, :, None]
    return grad_cp


def violation_per_sample(cps, obstacle, B):
    """(B, nxi, nd) -> (B,) deepest penetration into the raw obstacle per sample."""
    curve = _torch.einsum('pi,bid->bpd', B, cps)
    d = curve - _torch.as_tensor(obstacle.center, dtype=curve.dtype,
                                 device=curve.device)
    dist = (d ** 2).sum(-1) ** 0.5
    return _torch.clamp(obstacle.radius - dist, min=0.0).amax(dim=1)


def polish_out_of_obstacle(cps, obstacle, B, max_iters=250, tol=1e-7,
                           stall_patience=15, stall_rtol=1e-3):
    """Pure descent on the penalty until the curve clears the obstacle.

    A radially symmetric potential has one degenerate configuration: a curve
    running exactly through the obstacle centre is pushed *along itself*, never
    sideways, so plain descent stalls. That case is measure-zero for generated
    trajectories but not impossible, so a stall is detected and broken with a
    deterministic nudge perpendicular to the curve tangent at the deepest point.
    Normal descent then restores the shape.
    """
    if hasattr(obstacle, 'obstacles'):
        for obs in obstacle.obstacles:
            cps = polish_out_of_obstacle(cps, obs, B, max_iters, tol, stall_patience, stall_rtol)
        return cps

    cps = cps.clone()
    last_v, stalled_for = None, 0

    for _ in range(max_iters):
        g = curve_repulsion_grad(cps, obstacle, B)
        if g.abs().max() < tol:
            break
        cps = cps - g

        v = violation_per_sample(cps, obstacle, B)
        if last_v is not None:
            improved = (last_v - v) > stall_rtol * last_v.clamp(min=1e-12)
            stalled_for = 0 if bool(improved.any()) else stalled_for + 1
            if stalled_for >= stall_patience:
                cps = _break_symmetry(cps, obstacle, B, active=(v > tol))
                stalled_for = 0
        last_v = v

    return cps


def _break_symmetry(cps, obstacle, B, active, scale=0.05):
    """Nudge stalled samples perpendicular to the curve at the deepest point."""
    curve = _torch.einsum('pi,bid->bpd', B, cps)
    d = curve - _torch.as_tensor(obstacle.center, dtype=curve.dtype,
                                 device=curve.device)
    deep = _torch.clamp(obstacle.radius - (d ** 2).sum(-1) ** 0.5,
                        min=0.0).argmax(dim=1)                      # (B,)

    idx = _torch.arange(curve.shape[0], device=curve.device)
    lo = (deep - 1).clamp(min=0)
    hi = (deep + 1).clamp(max=curve.shape[1] - 1)
    tang = curve[idx, hi] - curve[idx, lo]                           # (B, 2)
    tang = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    perp = _torch.stack([-tang[..., 1], tang[..., 0]], dim=-1)       # (B, 2)

    step = scale * obstacle.radius * perp * active[:, None].to(curve.dtype)
    return cps + step[:, None, :]


def max_violation(cps, obstacle, B):
    """Deepest penetration of the rendered curve into the *raw* obstacle.

    Measured against `radius`, not `effective_radius`: the margin is a guidance
    cushion, the reported violation should be the physical one.
    """
    if hasattr(obstacle, 'obstacles'):
        return max([max_violation(cps, o, B) for o in obstacle.obstacles] + [0.0])

    curve = _torch.einsum('pi,bid->bpd', B, cps)
    d = curve - _torch.as_tensor(obstacle.center, dtype=curve.dtype,
                                 device=curve.device)
    dist = (d ** 2).sum(-1) ** 0.5
    return _torch.clamp(obstacle.radius - dist, min=0.0).max().item()
