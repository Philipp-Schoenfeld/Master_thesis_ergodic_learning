"""
volume3d.py
===========
The "explore the space itself" variant of the 3D simulation.  Instead of a
density painted on a surface, the unknown quantity is a scalar field in a cube
shaped workspace: a handful of 3D Gaussian blobs the agent has to find by
flying through the cube.  A :class:`VolumeFigure` offers the same attributes as
:class:`figures3d.Figure` (``pts``, ``nrm``, ``w``, ``lit``, ``start``, ``view``),
so the belief, the solvers, the planner and the MuJoCo side work unchanged:

* ``pts``   uniform random points in the cube (the belief lives on them),
* ``nrm``   the same constant vector everywhere (there is no surface), so the
            normal part of the belief kernel drops out and the "surface
            normal" tool axis becomes one fixed direction,
* ``lit``   all True (every point can be measured).

:class:`CubeSpace` replaces :class:`surface3d.LitSurface`: "projecting" a
curve means clamping it into the cube.
"""

import numpy as np
from scipy.spatial import cKDTree

#: Workspace cube in the unit cube the network was trained in.
CUBE_LO, CUBE_HI = 0.1, 0.9
N_POINTS = 10000
#: Direction the "surface normal" points to (towards the robot, see scene3d.FACE_DIR).
FIXED_NORMAL = np.array([-1.0, 0.0, 0.0])
VOLUME_SEEDS = tuple(range(1, 9))
BLOB_COUNTS = (2, 3, 5)
#: Standard target: a solid cube centred in the workspace; selectable edge lengths
#: (all smaller than the workspace edge CUBE_HI - CUBE_LO).
BOX_SIDES = (0.30, 0.20, 0.40)
BOX_EDGE_WIDTH = 0.012                    # softness of the box faces
#: The agent starts on the robot-facing side of the workspace, outside a centred box.
BOX_START = (0.2, 0.5, 0.5)
CAMERA_DIR = np.array([-0.7, -0.55, 0.45])


def list_volume_entries():
    """Selectable volumes, the solid target cube first (shape 'Cube', surface = edge
    length), then 8 random blob layouts x 2/3/5 blobs (shape = layout, surface =
    blob count), mirroring the form / surface selectors."""
    boxes = [dict(shape='Cube', surface='side %.2f' % side, view_id=0, view=[1.0, 0.0, 0.0],
                  hit_frac=1.0, split='volume', kind='box', side=side, placement_key='cube')
             for side in BOX_SIDES]
    return boxes + [dict(shape='Volume %d' % s, surface='%d blobs' % n, view_id=0, view=[1.0, 0.0, 0.0],
                 hit_frac=1.0, split='volume', kind='blobs', seed=s, n_blobs=n, placement_key='cube')
            for s in VOLUME_SEEDS for n in BLOB_COUNTS]


def blob_field(pts, centres, sigmas, weights):
    """Sum of axis-aligned 3D Gaussians evaluated at ``pts`` (P, 3)."""
    z = ((pts[:, None, :] - centres[None]) / sigmas[None]) ** 2
    return (weights[None] * np.exp(-0.5 * z.sum(axis=2))).sum(axis=1)


def box_field(pts, side, centre=0.5 * (CUBE_LO + CUBE_HI), width=BOX_EDGE_WIDTH):
    """Solid cube of edge ``side`` around ``centre`` with soft faces (product of
    per-axis sigmoids)."""
    d = side / 2.0 - np.abs(np.asarray(pts, dtype=np.float64) - centre)
    return np.prod(1.0 / (1.0 + np.exp(-d / width)), axis=1)


def _draw_blobs(rng, n_blobs, margin=0.12, min_dist=0.3):
    lo, hi = CUBE_LO + margin, CUBE_HI - margin
    centres = []
    tries = 0
    while len(centres) < n_blobs:
        c = rng.uniform(lo, hi, size=3)
        tries += 1
        if all(np.linalg.norm(c - o) >= min_dist for o in centres) or tries > 500:
            centres.append(c)
    sigmas = rng.uniform(0.05, 0.10, size=(n_blobs, 3))
    weights = rng.uniform(0.6, 1.0, size=n_blobs)
    return np.array(centres), sigmas, weights


class VolumeFigure:
    is_volume = True
    mesh = None
    camera_dir = CAMERA_DIR / np.linalg.norm(CAMERA_DIR)

    def __init__(self, entry, n_points=N_POINTS):
        self.entry = dict(entry)
        self.shape = entry['shape']
        self.surface_key = entry['surface']
        self.label = entry['surface']
        rng_pts = np.random.default_rng(7)                    # the same points for every volume
        self.pts = rng_pts.uniform(CUBE_LO, CUBE_HI, size=(n_points, 3)).astype(np.float32)
        self.is_box = entry.get('kind') == 'box'
        if self.is_box:
            w = box_field(self.pts, float(entry['side']))
        else:
            rng = np.random.default_rng(1000 * int(entry['seed']) + int(entry['n_blobs']))
            self.blobs = _draw_blobs(rng, int(entry['n_blobs']))
            w = blob_field(self.pts.astype(np.float64), *self.blobs)
        self.w = (w / max(float(w.max()), 1e-12)).astype(np.float32)
        self.nrm = np.tile(FIXED_NORMAL.astype(np.float32), (n_points, 1))
        self.view = -FIXED_NORMAL                             # figure frame == world frame orientation
        self.facing = np.ones(n_points, dtype=bool)
        self.lit = self.facing
        self.start = np.asarray(BOX_START if self.is_box else (0.5, 0.5, 0.5), dtype=np.float32)

    @property
    def title(self):
        if self.is_box:
            return 'Target cube (%s) in the workspace' % self.label
        return '%s (%s) - explore the cube' % (self.shape, self.label)

    @property
    def corners(self):
        return np.array([[x, y, z] for x in (CUBE_LO, CUBE_HI) for y in (CUBE_LO, CUBE_HI)
                         for z in (CUBE_LO, CUBE_HI)])

    @property
    def edges(self):
        """The 12 cube edges as (start, end) point pairs."""
        c = self.corners
        return [(c[i], c[j]) for i in range(8) for j in range(i + 1, 8)
                if int((c[i] != c[j]).sum()) == 1]


class CubeSpace:
    """Stand-in for ``LitSurface``: the whole cube is reachable, so projecting
    a curve onto it is clamping, and the 'normal' is one constant direction."""

    def __init__(self, figure):
        self.pts = np.asarray(figure.pts, dtype=np.float64)
        self.nrm = np.tile(FIXED_NORMAL, (len(self.pts), 1))
        self.tree = cKDTree(self.pts)
        self.obstacle = None

    def set_obstacle(self, obstacle):
        """Free space = the cube minus the (inflated) obstacle; ``project`` clamps and pushes out."""
        self.obstacle = obstacle if obstacle is not None and len(getattr(obstacle, 'obstacles', [0]))             else None

    def _free(self, p):
        shape = np.shape(p)
        p = np.clip(np.atleast_2d(np.asarray(p, dtype=np.float64)), CUBE_LO, CUBE_HI)
        if self.obstacle is not None:
            for _ in range(3):                    # pushing out may leave the cube: clamp again
                p = np.clip(self.obstacle.project_out(p, inflated=True), CUBE_LO, CUBE_HI)
                if not self.obstacle.contains(p, inflated=True).any():
                    break
        return p.reshape(shape)

    def project(self, curve):
        c = self._free(curve)
        return c, np.tile(FIXED_NORMAL, (len(c), 1))

    def snap_point(self, p):
        return self._free(p)

    def snap_index(self, p):
        return int(self.tree.query(np.asarray(p, dtype=np.float64))[1])
