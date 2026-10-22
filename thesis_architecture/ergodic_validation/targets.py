r"""
targets.py -- canonical target densities with known answers
============================================================
Every target is a density grid (R, R) on [0, 1]^2, indexed [iy, ix], values at the nodes
linspace(0, 1, R), normalised to max 1 -- the same convention as `common.data.load_truth`, so
the grids go through the smoothness pipeline (`run_smoothness_comparison_v2.run_cell`) unchanged.

  sun_gmm3     the 3-component GMM of Sun et al.'s official tutorial
               (MurpheyLab/lqr-flow-matching, tutorials/stein_flow_coverage.ipynb, cell 2)
  blobs_75_25  two separated Gaussians with masses 0.75 / 0.25: an ergodic trajectory must spend
               ~75 % of its time in the left half (x < 0.51)
  uniform      uniform density: every 4 x 4 cell must get ~1/16 of the time
  ring         annulus around the centre: the hole must (almost) not be visited
  A, organic_30, korean_5   holdout shapes from ergodic_dataset_775.db (letters / GMM mixtures)
"""
import numpy as np

R_DEFAULT = 96


def _grid(R):
    xs = np.linspace(0.0, 1.0, R)
    gx, gy = np.meshgrid(xs, xs)            # gx[iy, ix] = x, gy[iy, ix] = y
    return gx, gy


def _gauss(gx, gy, mean, cov):
    cov = np.asarray(cov, dtype=np.float64)
    inv = np.linalg.inv(cov)
    dx, dy = gx - mean[0], gy - mean[1]
    q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))


def sun_gmm3(R=R_DEFAULT):
    gx, gy = _grid(R)
    d = (0.34 * _gauss(gx, gy, (0.3, 0.5), [[0.002, 0.0], [0.0, 0.04]])
         + 0.34 * _gauss(gx, gy, (0.5, 0.5), [[0.02, -0.018], [-0.018, 0.02]])
         + 0.33 * _gauss(gx, gy, (0.7, 0.5), [[0.002, 0.0], [0.0, 0.04]]))
    return d / d.max()


def blobs_75_25(R=R_DEFAULT):
    gx, gy = _grid(R)
    d = 0.75 * _gauss(gx, gy, (0.3, 0.5), np.eye(2) * 0.006) + 0.25 * _gauss(gx, gy, (0.74, 0.5), np.eye(2) * 0.006)
    return d / d.max()


def uniform(R=R_DEFAULT):
    return np.ones((R, R))


def ring(R=R_DEFAULT):
    gx, gy = _grid(R)
    r = np.sqrt((gx - 0.5) ** 2 + (gy - 0.5) ** 2)
    d = np.exp(-0.5 * ((r - 0.3) / 0.045) ** 2)
    return d / d.max()


SYNTHETIC = {'sun_gmm3': sun_gmm3, 'blobs_75_25': blobs_75_25, 'uniform': uniform, 'ring': ring}

#: region checks with an exact expected time fraction: name -> (mask(x, y) on points, expected)
REGION_CHECKS = {
    'blobs_75_25': [('left half (x < 0.51)', lambda x, y: x < 0.51, None)],
    'ring': [('hole (r < 0.15)', lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.15 ** 2, None)],
}


def region_mass(density, mask):
    """Target mass inside `mask` (evaluated on the density nodes)."""
    R = density.shape[0]
    gx, gy = _grid(R)
    d = np.clip(np.asarray(density, dtype=np.float64), 0, None)
    return float(d[mask(gx, gy)].sum() / d.sum())


def load(names, R=R_DEFAULT):
    """-> list of (name, grid) for synthetic names and/or holdout shape labels."""
    out, shapes = [], [n for n in names if n not in SYNTHETIC]
    grids = {}
    if shapes:
        from common.data import load_truth
        got, truths = load_truth(labels=shapes, n=len(shapes), split='val', resolution=R, device='cpu')
        grids.update({n: t.numpy().astype(np.float64) for n, t in zip(got, truths)})
        missing = set(shapes) - set(got)
        if missing:
            raise KeyError(f'holdout shapes not found: {sorted(missing)}')
    for n in names:
        out.append((n, SYNTHETIC[n](R) if n in SYNTHETIC else grids[n]))
    return out
