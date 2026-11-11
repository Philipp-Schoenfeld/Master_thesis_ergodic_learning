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
        key = None if obstacle is None or not len(getattr(obstacle, 'obstacles', [0]))             else obstacle.signature()
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


def _repel_controls(c, B, obstacle, s0=None, max_iters=250):
    """Refinement step: push the control points out of the obstacle.

    Pure descent on the hinge-squared penalty of the dense curve (the same repulsion
    gradient the flow integration uses), numpy in / numpy out.  The pinned start stays put."""
    from obstacles import polish_out_of_obstacle
    cps = torch.from_numpy(c).float().unsqueeze(0)
    cps = polish_out_of_obstacle(cps, obstacle, B.detach().cpu().float(), max_iters=max_iters)
    out = cps[0].numpy().astype(np.float64)
    if s0 is not None:
        out[0] = s0
    return out


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
    for _ in range(iters):
        if obstacle is not None:
            c = _repel_controls(c, B, obstacle, s0)
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
    curve, nrm = lit.project(Bn @ c)
    if s0 is not None:
        curve[0] = s0
        nrm[0] = lit.project(s0[None])[1][0]
    axes = _smooth(-nrm, smooth_k)
    axes = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-9)
    return torch.from_numpy(c).float(), curve, axes
