"""
figures3d.py
============
The 3D targets of the simulation: a held-out *entry* of the projected dataset is
a 2D form (``shape``) thrown like a slide projector onto a triangle mesh
(``surface``) from a viewing direction (``view``).  A :class:`Figure` holds the
mesh and, for M points sampled on it, the projected ground-truth density that
the robot can measure.  Everything lives in the unit cube, exactly as the 3D
network sees it; only the MuJoCo side maps it into the robot workspace.
"""

import json
import os
import sqlite3

import numpy as np

import paths3d

paths3d.setup()
import surfaces  # noqa: E402  (3D_ergodic_learning/surfaces.py)

CACHE_DIR = os.path.join(paths3d.HERE, 'cache')
#: Surface listed first for each form: the standard first target is the cube.
FIRST_SURFACE = 'wuerfel'
#: Surfaces shown smaller than in the dataset: edge factor about the centre of the unit cube.
#: The projected density is carried along unchanged, only the geometry shrinks.
SHRINK = {'wuerfel': 0.5}
CUBE_CENTRE = 0.5


# -- Entries ------------------------------------------------------------------
def list_entries(split='val_beides'):
    """Held-out entries of ``split`` as dicts ``shape, surface, view_id, view,
    hit_frac, split``, sorted by shape then surface.  The database is 1.5 GB, so
    the list is cached after the first read.  Surfaces made from downloaded
    meshes (``extern_*``) are left out: they only exist after
    ``surfaces.externe_aufnehmen()``.  The pseudo-split ``'volume'`` lists the
    cube-shaped workspaces of :mod:`volume3d`."""
    if split == 'volume':
        import volume3d
        return volume3d.list_volume_entries()
    entries = [e for e in _read_entries(split) if not e['surface'].startswith('extern_')]
    # the cube surface comes first for every form (stable: the order is otherwise unchanged)
    return sorted(entries, key=lambda e: e['surface'] != FIRST_SURFACE)


def _read_entries(split):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, 'entries_%s.json' % split)
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    con = sqlite3.connect('file:' + paths3d.DB + '?mode=ro', uri=True)
    rows = con.execute(
        "SELECT shape_name, surface, view_id, AVG(view_x), AVG(view_y), AVG(view_z), "
        "AVG(hit_frac) FROM ergodic_pairs_3d WHERE split=? "
        "GROUP BY shape_name, surface, view_id ORDER BY shape_name, surface, view_id",
        (split,)).fetchall()
    con.close()
    out = [dict(shape=r[0], surface=r[1], view_id=int(r[2]), view=[r[3], r[4], r[5]],
                hit_frac=float(r[6]), split=split) for r in rows]
    with open(cache, 'w') as f:
        json.dump(out, f)
    return out


def density_spec(shape_name):
    """2D density definition (``gmm`` or ``analytical``) of a form."""
    con = sqlite3.connect('file:' + paths3d.DB + '?mode=ro', uri=True)
    row = con.execute('SELECT density_params FROM dichten WHERE shape_name=?',
                      (shape_name,)).fetchone()
    con.close()
    if row is None:
        raise KeyError('unknown form %r' % shape_name)
    return json.loads(row[0])


def _gmm_density(pts, spec):
    means = np.asarray(spec['means'], dtype=np.float64)
    covs = np.asarray(spec['covs'], dtype=np.float64)
    weights = np.asarray(spec['weights'], dtype=np.float64)
    weights = weights / weights.sum()
    dets = np.linalg.det(covs)
    inv = np.linalg.inv(covs)
    diff = pts[:, None, :] - means[None, :, :]
    maha = np.einsum('pki,kij,pkj->pk', diff, inv, diff)
    norm = weights / (2.0 * np.pi * np.sqrt(np.maximum(dets, 1e-30)))
    return (norm[None, :] * np.exp(-0.5 * maha)).sum(axis=1)


def _analytical_density(pts, spec):
    segments = np.asarray(spec['segments'], dtype=np.float64)
    sigma = float(spec.get('sigma', 0.025))
    a, b = segments[:, 0, :], segments[:, 1, :]
    ab = b - a
    denom = np.maximum((ab * ab).sum(axis=1), 1e-8)
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(axis=2) / denom[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d2 = ((pts[:, None, :] - proj) ** 2).sum(axis=2)
    return np.exp(-d2 / (2.0 * sigma ** 2)).max(axis=1)


def _grid_mean(fn, res=96):
    axis = np.linspace(0.0, 1.0, res)
    gx, gy = np.meshgrid(axis, axis)
    return float(np.mean(fn(np.stack([gx.ravel(), gy.ravel()], axis=1))))


def density_values(pts, spec):
    """Density of a form at ``pts`` (P, 2) in [0,1]^2: the base GMM / polyline
    density, optionally mixed with a broad pedestal and an 'unknown region'
    exactly like ``shape_library._vielleicht_sockel`` / ``_vielleicht_unknown``."""
    def base_fn(p):
        return (_analytical_density(p, spec) if spec.get('type') == 'analytical'
                else _gmm_density(p, spec)) + 1e-10

    vals = base_fn(pts)
    ped = spec.get('pedestal')
    if ped:
        a = float(ped['weight'])
        c = np.asarray(ped['center'], dtype=np.float64)
        g = np.exp(-((pts - c) ** 2).sum(axis=1) / (2.0 * float(ped['sigma']) ** 2))
        vals = (1.0 - a) * vals / float(ped['z_base']) + a * g / float(ped['z_ped']) + 1e-10
    unk = spec.get('unknown_region')
    if unk:
        a = float(unk['a'])
        region = _gmm_density(pts, unk) + 1e-10
        vals = (1.0 - a) * vals / float(unk['z_base']) + a * region / float(unk['z_region']) + 1e-10
    return vals


def density_grid(spec, resolution=128):
    """Max-normalised 2D density, indexed ``[y, x]`` like ``shape_library.pdf_on_grid``."""
    axis = np.linspace(0.0, 1.0, resolution)
    gx, gy = np.meshgrid(axis, axis)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid = density_values(pts, spec).reshape(resolution, resolution)
    return grid / max(float(grid.max()), 1e-12)


# -- Figure -------------------------------------------------------------------
_SURFACE_CACHE = {}


def _surface(key):
    if key not in _SURFACE_CACHE:
        _SURFACE_CACHE[key] = surfaces.build(key)
    return _SURFACE_CACHE[key]


class Figure:
    """One selected (form, surface) target.

    Attributes
    ----------
    mesh      trimesh.Trimesh in the unit cube
    pts, nrm  (M, 3) points on the surface and their outward unit normals
    w         (M,) ground-truth density at ``pts`` (0 on the side facing away
              from the projector), max-normalised
    lit       (M,) bool, the part of the surface the projector hits
    start     (3,) default start point of the agent (lit point nearest to the
              centroid of the lit region)
    """

    def __init__(self, entry, n_points=8000, seed=0, dens_res=128):
        self.entry = dict(entry)
        self.shape = entry['shape']
        self.surface_key = entry['surface']
        surf = _surface(entry['surface'])
        self.label = surf.label
        v = np.asarray(entry['view'], dtype=np.float64)
        old_view = surf.view
        surf.view = v / max(np.linalg.norm(v), 1e-12)
        try:
            self.dens2d = density_grid(density_spec(entry['shape']), dens_res)
            pts, nrm, w = surfaces.project(surf, self.dens2d, n_points=n_points, seed=seed)
        finally:
            surf.view = old_view
        self.shrink = float(SHRINK.get(entry['surface'], 1.0))
        self.mesh = surf.mesh
        if self.shrink != 1.0:
            pts = CUBE_CENTRE + self.shrink * (np.asarray(pts, dtype=np.float64) - CUBE_CENTRE)
            self.mesh = surf.mesh.copy()                  # the cached dataset mesh stays untouched
            self.mesh.vertices = CUBE_CENTRE + self.shrink * (self.mesh.vertices - CUBE_CENTRE)
        self.view = np.asarray(entry['view'], dtype=np.float64)
        self.view = self.view / max(np.linalg.norm(self.view), 1e-12)
        self.pts = np.asarray(pts, dtype=np.float32)
        self.nrm = np.asarray(nrm, dtype=np.float32)
        m = float(w.max())
        self.w = np.asarray(w / m if m > 0 else w, dtype=np.float32)
        self.facing = (self.nrm @ (-self.view.astype(np.float32))) > 1e-6
        self.lit = self.facing
        centre = self.pts[self.lit].mean(axis=0) if self.lit.any() else self.pts.mean(axis=0)
        cand = np.flatnonzero(self.lit) if self.lit.any() else np.arange(len(self.pts))
        self.start = self.pts[cand[np.argmin(np.linalg.norm(self.pts[cand] - centre, axis=1))]]

    @property
    def title(self):
        return '%s on %s' % (self.shape, self.label)


def build_figure(entry, n_points=8000, seed=0):
    if entry.get('split') == 'volume':
        import volume3d
        return volume3d.VolumeFigure(entry)
    return Figure(entry, n_points=n_points, seed=seed)
