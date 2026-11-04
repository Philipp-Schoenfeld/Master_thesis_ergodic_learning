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

    def project(self, curve):
        """(T, 3) -> points on the lit surface and their outward normals."""
        _, idx = self.tree.query(np.asarray(curve, dtype=np.float64))
        return self.pts[idx], self.nrm[idx]

    def snap_index(self, p):
        return int(self.tree.query(np.asarray(p, dtype=np.float64))[1])

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


def snap_plan(cps, B, lit, start=None, iters=3, smooth_k=5):
    """Project a planned B-spline onto the lit surface.

    cps (nxi, 3) torch, B (T, nxi) torch -> (cps_new (nxi, 3), curve (T, 3),
    inward normals as tool axes (T, 3)), all numpy except cps_new (torch).
    The first control point is pinned to ``start`` (snapped onto the surface),
    so a round continues exactly where the previous one ended.
    """
    Bn = B.detach().cpu().numpy().astype(np.float64)
    c = cps.detach().cpu().numpy().astype(np.float64)
    s0 = None
    if start is not None:
        s0 = lit.snap_point(start)
        c[0] = s0
    for _ in range(iters):
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
