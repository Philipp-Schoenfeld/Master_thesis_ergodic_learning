r"""
shapes_3d.py
============
Signed-distance 3D shapes and their attraction gradients for inference-time
guidance -- the mirror image of obstacles.py's CircleObstacle repulsion.
Where a CircleObstacle pushes a curve *out of* a forbidden disc, an SDFShape
pulls a curve *onto* the zero level set of its signed-distance field:

    E(P) = 0.5 * sum( SDF(P)^2 )

Two-sided by construction (not hinge-squared), so a point that starts either
inside or outside the shape is pulled toward the surface, not just away from
a violation. `grad_penalty` uses autograd on the scalar penalty rather than a
hand-derived formula -- a box or torus SDF's analytic gradient is easy to get
subtly wrong at edges/corners, autograd is exact everywhere it is defined.

Reuses `curve_repulsion_grad`/`basis_torch` from obstacles.py unchanged: that
function only ever calls `obstacle.grad_penalty(curve)`, so it is already
agnostic to both the shape and the number of spatial dimensions (nd=3 here
vs. nd=2 for the planar obstacles).
"""

import torch

from obstacles import curve_repulsion_grad, basis_torch  # noqa: F401  (re-exported)


class SDFShape:
    """Base class: subclasses implement sdf(P) -> (...,), a differentiable
    torch expression. grad_penalty is derived automatically via autograd."""

    def sdf(self, P):
        raise NotImplementedError

    def penalty(self, P):
        return 0.5 * (self.sdf(P) ** 2).sum()

    def grad_penalty(self, P):
        """dE/dP for E = 0.5*SDF(P)^2. Safe to call from inside a no_grad()
        block -- grad tracking is switched on locally just for this scalar."""
        with torch.enable_grad():
            P_ = P.detach().requires_grad_(True)
            e = self.penalty(P_)
            (g,) = torch.autograd.grad(e, P_)
        return g

    def violation(self, P):
        """(..., 3) -> (...,). Absolute distance to the surface."""
        return self.sdf(P).abs()


class SphereShape(SDFShape):
    def __init__(self, center=(0.5, 0.5, 0.5), radius=0.42):
        self.center = tuple(float(c) for c in center)
        self.radius = float(radius)

    def sdf(self, P):
        c = torch.as_tensor(self.center, dtype=P.dtype, device=P.device)
        return (P - c).norm(dim=-1) - self.radius

    def draw(self, ax, color='#9E9E9E', alpha=0.25):
        u = torch.linspace(0, 2 * torch.pi, 36)
        v = torch.linspace(0, torch.pi, 24)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        cx, cy, cz = self.center
        x = cx + self.radius * torch.cos(uu) * torch.sin(vv)
        y = cy + self.radius * torch.sin(uu) * torch.sin(vv)
        z = cz + self.radius * torch.cos(vv)
        ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), color=color,
                         alpha=alpha, linewidth=0, shade=True, zorder=1)


class CubeShape(SDFShape):
    def __init__(self, center=(0.5, 0.5, 0.5), half_extent=0.35):
        self.center = tuple(float(c) for c in center)
        he = (half_extent,) * 3 if isinstance(half_extent, (int, float)) \
            else tuple(float(h) for h in half_extent)
        self.half = he

    def sdf(self, P):
        c = torch.as_tensor(self.center, dtype=P.dtype, device=P.device)
        h = torch.as_tensor(self.half, dtype=P.dtype, device=P.device)
        q = (P - c).abs() - h
        outside = torch.clamp(q, min=0.0).norm(dim=-1)
        inside = torch.clamp(q.amax(dim=-1), max=0.0)
        return outside + inside

    def draw(self, ax, color='#9E9E9E', alpha=0.25):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        cx, cy, cz = self.center
        hx, hy, hz = self.half
        lo = (cx - hx, cy - hy, cz - hz)
        hi = (cx + hx, cy + hy, cz + hz)

        def c(xb, yb, zb):
            return (hi[0] if xb else lo[0], hi[1] if yb else lo[1], hi[2] if zb else lo[2])

        faces = [
            [c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0)],  # z-
            [c(0, 0, 1), c(1, 0, 1), c(1, 1, 1), c(0, 1, 1)],  # z+
            [c(0, 0, 0), c(1, 0, 0), c(1, 0, 1), c(0, 0, 1)],  # y-
            [c(0, 1, 0), c(1, 1, 0), c(1, 1, 1), c(0, 1, 1)],  # y+
            [c(0, 0, 0), c(0, 1, 0), c(0, 1, 1), c(0, 0, 1)],  # x-
            [c(1, 0, 0), c(1, 1, 0), c(1, 1, 1), c(1, 0, 1)],  # x+
        ]
        pc = Poly3DCollection(faces, facecolor=color, alpha=alpha,
                               edgecolor='#616161', linewidths=0.5)
        ax.add_collection3d(pc)


class PyramidShape(SDFShape):
    """Right square pyramid: base at z=z0, apex at (cx, cy, z0 + height).

    The exact SDF of a pyramid needs a nearest-point-on-triangle computation
    near the edges; here the pyramid is instead treated as the intersection
    of 5 half-spaces (4 tilted side faces + the base plane) and the SDF is
    approximated as the max over their individual (unit-normalized) plane
    distances. That is exact for interior points and for any point whose
    nearest surface feature is a face -- it only underestimates the true
    distance right at an edge or the apex, a measure-zero set that autograd's
    gradient still points usefully away from.
    """

    def __init__(self, center=(0.5, 0.5), half_base=0.35, z0=0.12, height=0.68):
        self.cx, self.cy = float(center[0]), float(center[1])
        self.a = float(half_base)
        self.z0 = float(z0)
        self.h = float(height)

    def sdf(self, P):
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        a, h, z0 = self.a, self.h, self.z0
        norm = (1.0 + (a / h) ** 2) ** 0.5
        taper = a * (1.0 - (z - z0) / h)
        f_base = z0 - z
        f_px = ((x - self.cx) - taper) / norm
        f_mx = ((self.cx - x) - taper) / norm
        f_py = ((y - self.cy) - taper) / norm
        f_my = ((self.cy - y) - taper) / norm
        return torch.stack([f_base, f_px, f_mx, f_py, f_my], dim=-1).amax(dim=-1)

    def draw(self, ax, color='#9E9E9E', alpha=0.25):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        a, cx, cy, z0 = self.a, self.cx, self.cy, self.z0
        apex = (cx, cy, z0 + self.h)
        base = [(cx - a, cy - a, z0), (cx + a, cy - a, z0),
                (cx + a, cy + a, z0), (cx - a, cy + a, z0)]
        faces = [base] + [[base[i], base[(i + 1) % 4], apex] for i in range(4)]
        pc = Poly3DCollection(faces, facecolor=color, alpha=alpha,
                               edgecolor='#616161', linewidths=0.5)
        ax.add_collection3d(pc)


class TorusShape(SDFShape):
    def __init__(self, center=(0.5, 0.5, 0.5), R=0.32, r=0.13):
        self.center = tuple(float(c) for c in center)
        self.R = float(R)
        self.r = float(r)

    def sdf(self, P):
        c = torch.as_tensor(self.center, dtype=P.dtype, device=P.device)
        d = P - c
        qxy = torch.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2 + 1e-12) - self.R
        q = torch.stack([qxy, d[..., 2]], dim=-1)
        return q.norm(dim=-1) - self.r

    def draw(self, ax, color='#9E9E9E', alpha=0.25):
        u = torch.linspace(0, 2 * torch.pi, 48)
        v = torch.linspace(0, 2 * torch.pi, 24)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        cx, cy, cz = self.center
        x = cx + (self.R + self.r * torch.cos(vv)) * torch.cos(uu)
        y = cy + (self.R + self.r * torch.cos(vv)) * torch.sin(uu)
        z = cz + self.r * torch.sin(vv)
        ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), color=color,
                         alpha=alpha, linewidth=0, shade=True, zorder=1)


def _lift_generic(P2d, grad_fn, z_iters, z_lr, polish_iters, polish_lr, z_init):
    """Shared two-phase (z-only, then full 3D) descent behind both
    `lift_curve_to_shape` and `lift_points_to_shape`.

    The nearest-point projection this force performs cannot "drape" a point
    up and over a shape's far side -- the closest surface point to anywhere
    below a sphere's centre is always on its lower half, never the crown, no
    matter how the shape is sized or positioned; pure distance-descent has no
    notion of approaching from a direction. `z_init` is what actually decides
    which side of the shape gets used: starting the 3rd coordinate *above*
    the shape (e.g. z_init=1.0, over shapes that live in the lower half of
    the unit cube) makes the nearest surface point the upper/angled one for
    essentially every (x, y), which is what makes the lift land on a pyramid's
    sloped faces, a dome's crown, or a torus's top instead of pooling on the
    underside closest to a z=0 starting sheet.

    Phase 1 (z-only): a z-channel is added to the 2D points (initialised at
    `z_init`) and driven purely by the shape's attraction force restricted to
    its z-component -- the direct analogue of the obstacle repulsion term
    added to the velocity at inference time, just for a 3rd coordinate that
    never existed before. Converges to a clean lift whenever `shape` is
    roughly a height field over the (x, y) footprint of the points as seen
    from the `z_init` side.

    Phase 2 (polish): a few full 3D descent steps let (x, y) move too. This
    is what actually drives the remaining SDF violation to (near) zero for
    shapes that are not single-valued height fields from that side (e.g. a
    point whose (x, y) footprint sits outside a pyramid's base, or over a
    torus's hole) -- the same role `polish_out_of_obstacle` plays for hard
    obstacle clearance in obstacles.py.

    `grad_fn(P3) -> dE/dP3` abstracts over whether P3 are raw points (density
    texture) or B-spline control points (a curve, via the basis pullback).
    """
    P3 = torch.cat([P2d, torch.full_like(P2d[..., :1], z_init)], dim=-1)

    for _ in range(z_iters):
        g = grad_fn(P3)
        P3 = P3 - z_lr * torch.cat(
            [torch.zeros_like(g[..., :2]), g[..., 2:3]], dim=-1)

    for _ in range(polish_iters):
        g = grad_fn(P3)
        if g.abs().max() < 1e-7:
            break
        P3 = P3 - polish_lr * g

    return P3


def lift_curve_to_shape(cps2d, shape, B, z_iters=150, z_lr=0.6,
                         polish_iters=150, polish_lr=1.0, z_init=0.0):
    """Give a 2D B-spline curve (control points) a 3rd dimension and pull the
    rendered curve onto `shape`'s surface. See `_lift_generic` for the
    two-phase force it uses and what `z_init` controls."""
    return _lift_generic(cps2d, lambda P: curve_repulsion_grad(P, shape, B),
                          z_iters, z_lr, polish_iters, polish_lr, z_init)


def lift_points_to_shape(P2d, shape, z_iters=150, z_lr=0.6,
                          polish_iters=150, polish_lr=1.0, z_init=0.0):
    """Give a raw 2D point cloud (e.g. a target-density grid) a 3rd dimension
    and pull it onto `shape`'s surface with the identical force used for
    `lift_curve_to_shape` -- useful to render what the same attraction field
    does to the target density itself, for a direct visual correspondence
    with the lifted path."""
    return _lift_generic(P2d, shape.grad_penalty,
                          z_iters, z_lr, polish_iters, polish_lr, z_init)
