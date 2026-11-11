"""
test_obstacle3d.py
==================
Checks of the obstacle handling on a surface (no network, no GUI, a few seconds on the CPU):

    python test_obstacle3d.py

A flat face with a sphere centred on it and a path running straight through the sphere centre
(the degenerate case: the radial gradient has no sideways component).  After ``snap_plan`` the
path must lie on the face, keep clear of the sphere along every *segment* (not just at the
samples), and be evenly sampled.
"""

import numpy as np
import torch
import trimesh

import paths3d

paths3d.setup()
from obstacles import SphereObstacle, CompositeObstacle, bspline_basis_matrix  # noqa: E402
from surface3d import LitSurface, snap_plan                                     # noqa: E402

FAILURES = []


def check(name, ok, detail=''):
    print('%s %-58s %s' % ('[ok]  ' if ok else '[FAIL]', name, detail))
    if not ok:
        FAILURES.append(name)


class _Fig:
    """Minimal stand-in for a figure: a flat face (z = 0.5) seen from +z."""

    def __init__(self):
        self.mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.1), transform=trimesh.transformations.translation_matrix((0.5, 0.5, 0.45)))
        self.view = np.array([0.0, 0.0, -1.0])          # lit side: outward normals ~ -view = +z


def seg_clearance(c, centre, radius):
    a, b = c[:-1], c[1:]
    ab = b - a
    t = np.clip(((centre - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[:, None] * ab - centre, axis=1).min() - radius


def main():
    lit = LitSurface(_Fig(), n_dense=40000)
    nxi, T = 25, 128
    B = torch.from_numpy(bspline_basis_matrix(nxi, T, 5)).float()
    sphere = SphereObstacle(center=(0.5, 0.5, 0.5), radius=0.08, margin=0.04)
    obs = CompositeObstacle([sphere])
    s = np.linspace(0, 1, nxi)[:, None]
    line = (np.array([0.1, 0.5, 0.56]) * (1 - s) + np.array([0.9, 0.5, 0.56]) * s)   # 6 cm above the face
    cps = torch.from_numpy(line).float()

    _, free, _ = snap_plan(cps, B, lit, start=line[0])
    check('without obstacle the path crosses the sphere centre',
          np.linalg.norm(free - np.array(sphere.center), axis=1).min() < 0.03)

    _, curve, axes = snap_plan(cps, B, lit, start=line[0], obstacle=obs)
    centre = np.array(sphere.center)
    check('path stays on the face', np.abs(curve[:, 2] - 0.5).max() < 2e-3,
          'max |z - 0.5| = %.4f' % np.abs(curve[:, 2] - 0.5).max())
    check('every sample clears the inflated sphere',
          (np.linalg.norm(curve - centre, axis=1) >= sphere.effective_radius - 1e-3).all(),
          'min %.3f vs %.3f' % (np.linalg.norm(curve - centre, axis=1).min(), sphere.effective_radius))
    check('every segment clears the raw sphere', seg_clearance(curve, centre, sphere.radius) > 0.0,
          '%+.3f' % seg_clearance(curve, centre, sphere.radius))
    step = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    check('evenly sampled (max step / mean step < 2)', step.max() / step.mean() < 2.0,
          '%.2f' % (step.max() / step.mean()))
    check('start pinned (on the face), tool axes are unit vectors',
          np.linalg.norm(curve[0] - np.array([0.1, 0.5, 0.5])) < 0.02
          and np.allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-5))

    # the same obstacle is a no-op for a path that stays away from it
    far = (np.array([0.1, 0.8, 0.56]) * (1 - s) + np.array([0.9, 0.8, 0.56]) * s)
    _, a0, _ = snap_plan(torch.from_numpy(far).float(), B, lit, start=far[0])
    _, a1, _ = snap_plan(torch.from_numpy(far).float(), B, lit, start=far[0], obstacle=obs)
    from scipy.spatial import cKDTree
    dense = lambda c: np.stack([np.interp(np.linspace(0, 1, 2000), np.linspace(0, 1, len(c)), c[:, d])
                                for d in range(3)], axis=1)                  # noqa: E731
    gap = cKDTree(dense(a0)).query(a1)[0].max()                              # geometry, not samples
    check('a path far from the obstacle keeps its geometry (< 1 cm)', gap < 1e-2, '%.4f' % gap)

    print('\nfailures:', FAILURES or 'none')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
