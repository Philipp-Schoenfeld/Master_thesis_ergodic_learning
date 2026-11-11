"""
surface3d.py
============
Fast projection of planned trajectories onto the lit side of a figure.

The network's positions are close to, but not exactly on, the surface, and its
learned sensor axis disagrees with the surface normal on thin shapes (words,
digits) where front and back are a few millimetres apart.  For the robot the
trajectory is therefore snapped to the side the density was projected onto,
and the tool axis is the inward surface normal there.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


class LitSurface:
    """Dense samples of the *facing* side of a figure with a KD-tree."""

    def __init__(self, figure, n_dense=60000, seed=123, min_facing=0.05):
        import trimesh
        mesh = figure.mesh
        pts, fid = trimesh.sample.sample_surface(mesh, n_dense, seed=seed)
        nrm = np.asarray(mesh.face_normals)[fid]
        facing = (nrm @ (-figure.view)) > min_facing
        if facing.sum() < 100:                      # degenerate view: use everything
            facing[:] = True
        self.pts = np.asarray(pts, dtype=np.float64)[facing]
        self.nrm = np.asarray(nrm, dtype=np.float64)[facing]
        self.tree = cKDTree(self.pts)
        self._obs_key = None
        self._free_idx, self._free_tree = None, None

    def set_obstacle(self, obstacle):
        """Make the part of the surface inside the (inflated) obstacle unreachable.

        ``project`` then returns the nearest *free* surface point, so a path that was pushed
        out of the obstacle cannot be pulled back into it by the snapping.  ``None`` clears
        it.  The free-space KD-tree is rebuilt only when the geometry changes."""
        empty = obstacle is None or not len(getattr(obstacle, 'obstacles', [0]))
        key = None if empty else obstacle.signature()
        if key == self._obs_key:
            return
        self._obs_key = key
        self._free_idx, self._free_tree = None, None
        if key is None:
            return
        free = np.flatnonzero(~obstacle.contains(self.pts, inflated=True))
        if 0 < len(free) < len(self.pts):               # everything blocked: ignore the obstacle
            self._free_idx = free
            self._free_tree = cKDTree(self.pts[free])

    def _nearest(self, q):
        q = np.asarray(q, dtype=np.float64)
        if self._free_tree is None:
            return self.tree.query(q)[1]
        return self._free_idx[self._free_tree.query(q)[1]]

    def project(self, curve):
        """(T, 3) -> points on the lit surface and their outward normals."""
        idx = self._nearest(curve)
        return self.pts[idx], self.nrm[idx]

    def project_all(self, curve):
        """Like ``project`` but ignoring the obstacle cut-out: the on-surface curve *under* the
        obstacle, on which the surface-aware repulsion is evaluated."""
        idx = self.tree.query(np.asarray(curve, dtype=np.float64))[1]
        return self.pts[idx], self.nrm[idx]

    def snap_index(self, p):
        return int(self._nearest(p))

    def snap_point(self, p):
        return self.pts[self.snap_index(p)]


def make_lit(figure):
    """The reachable space of a target: the lit side of a surface figure, or the
    whole cube of a volume figure."""
    if getattr(figure, 'is_volume', False):
        from volume3d import CubeSpace
        return CubeSpace(figure)
    return LitSurface(figure)


def _smooth(x, k=5):
    """Moving average along axis 0 with edge padding (keeps the ends fixed)."""
    if k <= 1:
        return x
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], axis=0)
    ker = np.ones(k) / k
    return np.stack([np.convolve(xp[:, d], ker, mode='valid') for d in range(x.shape[1])], axis=1)


def _resample_arclength(curve):
    """(T, 3) polyline -> (T, 3) points evenly spaced in arc length (ends kept)."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return curve
    u = np.linspace(0.0, cum[-1], len(curve))
    return np.stack([np.interp(u, cum, curve[:, d]) for d in range(curve.shape[1])], axis=1)


#: Samples of the spline on which the obstacle repulsion is evaluated.
DENSE_PTS = 512
#: Repel on the surface-projected curve (tangentially) instead of in free 3D space.
SURFACE_AWARE = True


def _repel_controls(c, B, obstacle, s0=None, max_iters=250):
    """Refinement step: push the control points out of the obstacle.

    Pure descent on the hinge-squared penalty of the dense curve (the same repulsion
    gradient the flow integration uses), numpy in / numpy out.  The pinned start stays put."""
    from obstacles import polish_out_of_obstacle, bspline_basis_matrix
    cps = torch.from_numpy(c).float().unsqueeze(0)
    # A dense basis, not the 128-point one of the planner: a fast segment of the spline can cross
    # the sphere between two coarse samples without any sample being inside it.
    B_dense = torch.from_numpy(bspline_basis_matrix(B.shape[1], DENSE_PTS, 5)).float()
    cps = polish_out_of_obstacle(cps, obstacle, B_dense, max_iters=max_iters)
    out = cps[0].numpy().astype(np.float64)
    if s0 is not None:
        out[0] = s0
    return out


def _repel_on_surface(c, lit, obstacle, s0=None, max_iters=300, tol=1e-4, stall_patience=8):
    """Obstacle repulsion for a path that must stay on a surface.

    The 3D repulsion on the raw spline is wrong here: the network flies at a standoff above the
    surface, so the cheapest way out of the sphere is *up*, and the projection back onto the
    surface then lands inside the cut-out and jumps to its rim.  So the penalty is evaluated on
    the spline projected onto the surface (``lit.project_all``, ignoring the cut-out) and the
    gradient is applied tangentially to the surface only; the descent moves the control points,
    exactly like ``obstacles.polish_out_of_obstacle``.  A path through the sphere centre has no
    sideways gradient, so a stall is broken with a nudge along surface-normal x tangent."""
    from obstacles import bspline_basis_matrix
    Bd = bspline_basis_matrix(c.shape[0], DENSE_PTS, 5).astype(np.float64)
    wsum = np.maximum(Bd.sum(0), 1e-8)[:, None]
    r_eff = max(o.effective_radius for o in getattr(obstacle, 'obstacles', [obstacle]))
    c = c.copy()
    last, stalled = np.inf, 0
    for _ in range(max_iters):
        P, nrm = lit.project_all(Bd @ c)
        depth = obstacle.violation(P)
        worst = float(depth.max())
        if worst < tol:
            break
        g = obstacle.grad_penalty(P)
        g = g - (g * nrm).sum(-1, keepdims=True) * nrm
        c = c - (Bd.T @ g) / wsum
        stalled = stalled + 1 if worst > 0.99 * last else 0
        last = worst
        if stalled >= stall_patience:
            d = int(depth.argmax())
            t = P[min(d + 1, len(P) - 1)] - P[max(d - 1, 0)]
            side = np.cross(nrm[d], t)
            c = c + 0.1 * r_eff * side / max(np.linalg.norm(side), 1e-12)
            stalled = 0
        if s0 is not None:
            c[0] = s0
    return c


def snap_plan(cps, B, lit, start=None, iters=3, smooth_k=5, obstacle=None):
    """Project a planned B-spline onto the lit surface.

    cps (nxi, 3) torch, B (T, nxi) torch -> (cps_new (nxi, 3), curve (T, 3),
    inward normals as tool axes (T, 3)), all numpy except cps_new (torch).
    The first control point is pinned to ``start`` (snapped onto the surface),
    so a round continues exactly where the previous one ended.

    With an ``obstacle`` every refinement iteration first repels the control points from
    it, and ``lit`` (see ``set_obstacle``) only offers surface points outside it for the
    projection; the returned curve therefore never enters the inflated obstacle.
    """
    Bn = B.detach().cpu().numpy().astype(np.float64)
    c = cps.detach().cpu().numpy().astype(np.float64)
    if obstacle is not None:
        lit.set_obstacle(obstacle)
    s0 = None
    if start is not None:
        s0 = lit.snap_point(start)
        c[0] = s0
    surface = SURFACE_AWARE and hasattr(lit, 'project_all')   # a real surface (not the cube volume)

    def repel(c):
        if surface:
            return _repel_on_surface(c, lit, obstacle, s0)
        return _repel_controls(c, B, obstacle, s0)

    for _ in range(iters):
        if obstacle is not None:
            c = repel(c)
        curve, _ = lit.project(Bn @ c)
        if s0 is not None:
            curve[0] = s0
        if smooth_k > 1:
            curve = _smooth(curve, smooth_k)
            curve, _ = lit.project(curve)
            if s0 is not None:
                curve[0] = s0
        if s0 is None:
            c = np.linalg.lstsq(Bn, curve, rcond=None)[0]
        else:
            # first control point fixed (clamped basis: curve[0] == c[0]), fit the rest
            rest = np.linalg.lstsq(Bn[:, 1:], curve - Bn[:, :1] * s0[None], rcond=None)[0]
            c = np.vstack([s0[None], rest])
    if obstacle is not None:
        # The last refit can dip back into the cut-out; without this the final projection
        # would set those points onto different spots of its rim, i.e. jump across it.
        c = repel(c)
    curve, nrm = lit.project(Bn @ c)
    if s0 is not None:
        curve[0] = s0
    if obstacle is not None:
        # The detour around an obstacle speeds the spline up, which leaves a few widely spaced
        # samples; resample by arc length (and project again) so the path is evenly sampled.
        curve, nrm = lit.project(_resample_arclength(curve))
        if s0 is not None:
            curve[0] = s0
    if s0 is not None:
        nrm[0] = lit.project(s0[None])[1][0]
    axes = _smooth(-nrm, smooth_k)
    axes = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-9)
    return torch.from_numpy(c).float(), curve, axes
