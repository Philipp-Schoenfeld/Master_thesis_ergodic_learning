r"""
obstacles.py  —  3D port
========================
Obstacle definitions and repulsion gradients for inference-time guidance.

3D counterpart of ``thesis_architecture/obstacles.py``. The circle becomes a
sphere; the hinge-squared penalty and its analytic gradient are unchanged in
form, because both are written in terms of ``|P - c|`` and therefore do not
care how many coordinates P has:

    E(P) = 0.5 * sum( max(radius - |P - center|, 0)^2 )

Two things genuinely had to change rather than just widen:

* ``mask`` takes three grids instead of two, and ``draw`` renders a translucent
  sphere on an ``Axes3D`` instead of a patch on a 2D axis.
* ``_break_symmetry`` had a closed form in 2D (rotate the tangent by 90°). In
  3D the perpendicular direction is not unique, so the nudge is taken along the
  component of the centre-offset that is orthogonal to the tangent — the
  steepest escape direction that a purely radial potential cannot supply. If
  the curve runs exactly through the centre that component vanishes too, and
  the fallback is a cross product with whichever axis is least aligned with the
  tangent.

All functions accept numpy arrays or torch tensors and return the same type, so
one obstacle object serves both plotting (numpy) and guidance (torch).
"""

import os
import sys

import numpy as np

try:
    import torch as _torch
except ImportError:      # plotting-only usage does not need torch
    _torch = None

# bsplinax lives next to the repo root, not on the default path. Doing this here
# rather than in each runner means the module can also be imported straight from
# a test or a notebook.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_here, '..', 'bsplinax-main'),
           os.path.join(_here, '..', 'src')):
    _p = os.path.normpath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


# ── Same convention as the 2D baselines, lifted to the unit cube ──────────────
OBSTACLE_CENTER = (0.5, 0.5, 0.5)
OBSTACLE_RADIUS = 0.12
OBSTACLE_MARGIN = 0.01      # extra clearance so the curve does not graze the rim


def _is_torch(a):
    return _torch is not None and isinstance(a, _torch.Tensor)


def _relu(a):
    return _torch.clamp(a, min=0.0) if _is_torch(a) else np.maximum(a, 0.0)


class SphereObstacle:
    """Spherical no-fly zone in the unit cube with an analytic penalty gradient."""

    def __init__(self, center=OBSTACLE_CENTER, radius=OBSTACLE_RADIUS,
                 margin=OBSTACLE_MARGIN):
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)
        self.margin = float(margin)
        if len(self.center) != 3:
            raise ValueError(f"3D obstacle needs a 3-vector centre, got {self.center}")

    @property
    def effective_radius(self):
        return self.radius + self.margin

    def __repr__(self):
        return (f"SphereObstacle(center={self.center}, radius={self.radius}, "
                f"margin={self.margin})")

    # ── Geometry ─────────────────────────────────────────────────────────────
    def _delta(self, P):
        """P: (..., 3) -> offset from centre, same backend as P."""
        if _is_torch(P):
            c = _torch.as_tensor(self.center, dtype=P.dtype, device=P.device)
        else:
            c = np.asarray(self.center, dtype=P.dtype)
        return P - c

    def signed_distance(self, P):
        """(..., 3) -> (...,). Negative inside the (inflated) obstacle."""
        d = self._delta(P)
        dist = (d ** 2).sum(-1) ** 0.5
        return dist - self.effective_radius

    def violation(self, P):
        """(..., 3) -> (...,). Penetration depth, 0 outside the obstacle."""
        return _relu(-self.signed_distance(P))

    # ── Penalty and its gradient ─────────────────────────────────────────────
    def penalty(self, P):
        """Scalar hinge-squared energy, summed over all points."""
        return 0.5 * (self.violation(P) ** 2).sum()

    def grad_penalty(self, P):
        """dE/dP, shape (..., 3). Analytic: -(r - dist)_+ * (P - c) / dist.

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
    def mask(self, X, Y, Z, inflated=False):
        """Boolean grid: True where (X, Y, Z) lies inside the obstacle."""
        r = self.effective_radius if inflated else self.radius
        return ((X - self.center[0]) ** 2
                + (Y - self.center[1]) ** 2
                + (Z - self.center[2]) ** 2) <= r ** 2

    def draw(self, ax, zorder=1.5, show_margin=True, n_mesh=24):
        """Draw a translucent sphere on a 3D axis, in the project's grey.

        Drawn below the trajectories so a path that cuts through stays visible.
        `plot_surface` ignores zorder reliably on Axes3D, so the argument is
        accepted for signature parity with the 2D version but not forwarded.
        """
        u = np.linspace(0, 2 * np.pi, n_mesh)
        v = np.linspace(0, np.pi, n_mesh // 2)
        cx, cy, cz = self.center

        def _sphere(r):
            x = cx + r * np.outer(np.cos(u), np.sin(v))
            y = cy + r * np.outer(np.sin(u), np.sin(v))
            z = cz + r * np.outer(np.ones_like(u), np.cos(v))
            return x, y, z

        x, y, z = _sphere(self.radius)
        ax.plot_surface(x, y, z, color='#9E9E9E', alpha=0.30,
                        linewidth=0, antialiased=True, shade=True)
        if show_margin and self.margin > 0:
            xm, ym, zm = _sphere(self.effective_radius)
            ax.plot_wireframe(xm, ym, zm, color='#424242', alpha=0.18,
                              linewidth=0.4, rstride=3, cstride=3)


# Alias so downstream code can stay dimension-agnostic when it only needs
# "the obstacle of this project".
CircleObstacle = SphereObstacle


# ── B-Spline coupling ─────────────────────────────────────────────────────────
# Unchanged from the 2D version on purpose: the basis matrix B maps control
# points to curve points along the *parameter* axis and knows nothing about the
# coordinate dimension, so `curve = B @ cps` works for nd = 2 and nd = 3 alike.
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
    """Basis matrix as a torch tensor on `device`."""
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
        cps:  (B, nxi, 3) control points
        B:    (pts, nxi) basis matrix on the same device
        normalize: divide each control point's gradient by its total basis
            weight ``B.sum(0)``, so the magnitude is independent of `pts` and
            roughly equal to the penetration depth.

    Returns:
        (B, nxi, 3)
    """
    curve = _torch.einsum('pi,bid->bpd', B, cps)      # (B, pts, 3)
    g = obstacle.grad_penalty(curve)                  # (B, pts, 3)
    grad_cp = _torch.einsum('pi,bpd->bid', B, g)      # (B, nxi, 3)
    if normalize:
        grad_cp = grad_cp / B.sum(0).clamp(min=1e-8)[None, :, None]
    return grad_cp


def violation_per_sample(cps, obstacle, B):
    """(B, nxi, 3) -> (B,) deepest penetration into the raw obstacle per sample."""
    curve = _torch.einsum('pi,bid->bpd', B, cps)
    d = curve - _torch.as_tensor(obstacle.center, dtype=curve.dtype,
                                 device=curve.device)
    dist = (d ** 2).sum(-1) ** 0.5
    return _torch.clamp(obstacle.radius - dist, min=0.0).amax(dim=1)


def polish_out_of_obstacle(cps, obstacle, B, max_iters=250, tol=1e-7,
                           stall_patience=15, stall_rtol=1e-3):
    """Pure descent on the penalty until the curve clears the obstacle.

    A radially symmetric potential has a degenerate configuration: a curve
    running exactly through the obstacle centre is pushed *along itself*, never
    sideways, so plain descent stalls. In 3D the degenerate set is a line rather
    than a single diameter, but the failure mode is identical. A stall is
    detected and broken with a deterministic nudge orthogonal to the tangent.
    """
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
    """Nudge stalled samples orthogonally to the curve at the deepest point.

    In 2D the perpendicular of a tangent is unique up to sign. In 3D it spans a
    plane, so a direction has to be chosen. The one taken here is the component
    of the centre-offset orthogonal to the tangent: it points straight out of
    the sphere in the only way a radial gradient cannot express. When the curve
    passes exactly through the centre that component is zero as well, and the
    fallback is the cross product with whichever coordinate axis is least
    aligned with the tangent — always well-conditioned.
    """
    curve = _torch.einsum('pi,bid->bpd', B, cps)                     # (B, pts, 3)
    c = _torch.as_tensor(obstacle.center, dtype=curve.dtype, device=curve.device)
    d = curve - c
    deep = _torch.clamp(obstacle.radius - (d ** 2).sum(-1) ** 0.5,
                        min=0.0).argmax(dim=1)                       # (B,)

    idx = _torch.arange(curve.shape[0], device=curve.device)
    lo = (deep - 1).clamp(min=0)
    hi = (deep + 1).clamp(max=curve.shape[1] - 1)
    tang = curve[idx, hi] - curve[idx, lo]                           # (B, 3)
    tang = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # Radial offset at the deepest point, projected off the tangent.
    r = d[idx, deep]                                                 # (B, 3)
    perp = r - (r * tang).sum(-1, keepdim=True) * tang
    nrm = perp.norm(dim=-1, keepdim=True)

    # Fallback where the offset is (anti)parallel to the tangent: cross with the
    # least-aligned unit axis, which is guaranteed not to be parallel.
    eye = _torch.eye(3, dtype=curve.dtype, device=curve.device)      # (3, 3)
    least = tang.abs().argmin(dim=-1)                                # (B,)
    alt = _torch.cross(tang, eye[least], dim=-1)
    alt = alt / alt.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    use_alt = (nrm < 1e-9)
    perp = _torch.where(use_alt, alt, perp / nrm.clamp(min=1e-12))

    step = scale * obstacle.radius * perp * active[:, None].to(curve.dtype)
    return cps + step[:, None, :]


def max_violation(cps, obstacle, B):
    """Deepest penetration of the rendered curve into the *raw* obstacle.

    Measured against `radius`, not `effective_radius`: the margin is a guidance
    cushion, the reported violation should be the physical one.
    """
    curve = _torch.einsum('pi,bid->bpd', B, cps)
    d = curve - _torch.as_tensor(obstacle.center, dtype=curve.dtype,
                                 device=curve.device)
    dist = (d ** 2).sum(-1) ** 0.5
    return _torch.clamp(obstacle.radius - dist, min=0.0).max().item()
