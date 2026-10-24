r"""
ergodicity_metrics.py -- does a trajectory actually spend its time where the target mass is?
=============================================================================================

A trajectory x(t) is ergodic w.r.t. a target density p if its time average equals the spatial
average: for every region A, (fraction of time spent in A) = (target mass in A). The solvers in
this project optimise a truncated Fourier version of that statement (Mathew & Mezic 2011,
"Spectral Multiscale Coverage"; 64-100 low-frequency cosine modes). A solver can drive that
truncated metric to ~0 and still not be ergodic in any visible sense, because the low modes
cannot see sharp structure (letters, holes). The metrics here check the definition directly,
independently of the Fourier truncation the solvers use:

  hellinger      Hellinger distance between the time-averaged visitation density (histogram of
                 the time-uniform trajectory samples, Gaussian-blurred) and the target density
                 blurred with the SAME kernel. 0 = identical, 1 = disjoint supports.
  tv_c           Total variation between time fraction and mass fraction over a c x c partition
                 of the square (c = 2, 4, 8): the region-wise form of the ergodic definition.
  low_excess     time fraction spent where the target is ~0 (< 2 % of its max), minus the target
                 mass there (time wasted outside the support).
  coverage       target mass within `COVER_RADIUS` of the path (sensor-footprint coverage).
  spec_rel_K     relative truncated spectral error sum_k L_k (c_k - phi_k)^2 / sum_k L_k phi_k^2
                 for an independent mode count K (default 20, i.e. 400 modes).
  ergodic_k10    the project's reported number (ergodic_energy_torch.ergodic_term, K = 10, w = 600).

All inputs are time-uniform samples (N, T, 2) on [0, 1]^2 and a density grid (R, R) indexed
[iy, ix] (the convention of shape_library.pdf_on_grid / ergodic_energy_torch).
"""
import numpy as np
import torch
from scipy import ndimage

GRID = 64                 # evaluation resolution of the visitation histogram
BLUR = 0.03               # Gaussian blur (domain units) for the Hellinger comparison
LOW_REL = 0.02            # "outside the support" = target below 2 % of its maximum
COVER_RADIUS = 0.05       # footprint radius for the coverage metric
CELLS = (2, 4, 8)         # partitions for the region-wise total variation


def resample_density(density, R=GRID):
    """(H, W) node grid (values at linspace(0, 1, H)) -> (R, R) cell masses at the cell centres
    (i + 0.5) / R of the histogram bins used by `visitation`, normalised to sum 1."""
    d = np.asarray(density, dtype=np.float64)
    H, W = d.shape
    c = (np.arange(R) + 0.5) / R
    yy, xx = np.meshgrid(c * (H - 1), c * (W - 1), indexing='ij')
    v = ndimage.map_coordinates(d, [yy, xx], order=1, mode='nearest')
    v = np.clip(v, 0.0, None)
    return v / max(v.sum(), 1e-300)


def visitation(X, R=GRID):
    """(T, 2) time-uniform samples -> (R, R) time-fraction histogram, sum 1."""
    X = np.clip(np.asarray(X, dtype=np.float64), 0.0, 1.0 - 1e-9)
    h, _, _ = np.histogram2d(X[:, 1], X[:, 0], bins=R, range=[[0, 1], [0, 1]])
    return h / max(h.sum(), 1e-300)


def _blur(p, R=GRID, sigma=BLUR):
    q = ndimage.gaussian_filter(p, sigma * R, mode='constant')
    return q / max(q.sum(), 1e-300)


def hellinger(p, q):
    return float(np.sqrt(max(0.0, 1.0 - np.sqrt(p * q).sum())))


def region_tv(p, q, c, R=GRID):
    """Total variation of two (R, R) distributions after summing into c x c blocks."""
    b = R // c
    pp = p.reshape(c, b, c, b).sum(axis=(1, 3))
    qq = q.reshape(c, b, c, b).sum(axis=(1, 3))
    return float(0.5 * np.abs(pp - qq).sum())


def _densify(X, factor=4):
    """Linear interpolation between consecutive samples (for footprint coverage only)."""
    X = np.asarray(X, dtype=np.float64)
    t = np.linspace(0, len(X) - 1, (len(X) - 1) * factor + 1)
    i = np.minimum(np.floor(t).astype(int), len(X) - 2)
    a = (t - i)[:, None]
    return (1 - a) * X[i] + a * X[i + 1]


def coverage(X, target, R=GRID, radius=COVER_RADIUS):
    """Target mass within `radius` of the (densified) path."""
    occ = np.zeros((R, R), dtype=bool)
    Y = np.clip(_densify(X), 0.0, 1.0 - 1e-9)
    occ[(Y[:, 1] * R).astype(int), (Y[:, 0] * R).astype(int)] = True
    dist = ndimage.distance_transform_edt(~occ) / R
    return float(target[dist <= radius].sum())


def spectral_rel(X, density, K=20):
    """Relative truncated spectral error with K x K modes (independent of the solvers' K)."""
    from ergodic_energy_torch import make_k_grid, coeffs_from_points, target_coeffs_from_grid
    k_idx, Lam = (torch.as_tensor(a, dtype=torch.float64) for a in make_k_grid(K))
    phi = target_coeffs_from_grid(torch.as_tensor(np.asarray(density), dtype=torch.float64), k_idx)
    c = coeffs_from_points(torch.as_tensor(np.asarray(X), dtype=torch.float64), k_idx)
    num = (Lam * (c - phi) ** 2).sum(-1)
    den = (Lam * phi ** 2).sum()           # includes k = 0 (phi_0 = 1): well defined for uniform
    return (num / den).numpy()


def ergodic_k10(X, density):
    """The project's reported ergodic error (K = 10, w = 600)."""
    from ergodic_energy_torch import ergodic_term, make_k_grid, target_coeffs_from_grid
    k_idx, Lam = (torch.as_tensor(a, dtype=torch.float32) for a in make_k_grid(10))
    phi = target_coeffs_from_grid(torch.as_tensor(np.asarray(density), dtype=torch.float32), k_idx)
    return ergodic_term(torch.as_tensor(np.asarray(X), dtype=torch.float32), k_idx, Lam, phi).numpy()


def evaluate(X, density, R=GRID):
    """All metrics for a batch of trajectories. X: (N, T, 2) -> dict of (N,) arrays."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[None]
    tgt = resample_density(density, R)
    tgt_b = _blur(tgt, R)
    low = tgt < LOW_REL * tgt.max()
    m_low = float(tgt[low].sum())
    out = {k: [] for k in ['hellinger', 'low_excess', 'coverage'] + [f'tv_{c}' for c in CELLS]}
    for x in X:
        v = visitation(x, R)
        out['hellinger'].append(hellinger(_blur(v, R), tgt_b))
        for c in CELLS:
            out[f'tv_{c}'].append(region_tv(v, tgt, c, R))
        out['low_excess'].append(float(v[low].sum()) - m_low)
        out['coverage'].append(coverage(x, tgt, R))
    out = {k: np.asarray(v) for k, v in out.items()}
    out['spec_rel_20'] = spectral_rel(X, density, K=20)
    out['ergodic_k10'] = ergodic_k10(X, density)
    seg = np.linalg.norm(np.diff(X, axis=1), axis=-1)
    out['path_len'] = seg.sum(1)
    acc = X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
    out['accel'] = (acc ** 2).sum(axis=(1, 2))
    return out


def oracle_samples(density, n_traj, T, seed=0):
    """i.i.d. samples from the target, shaped like trajectories (N, T, 2): the best any
    time-average with T samples can do (finite-sample floor of the metrics above)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(density, dtype=np.float64)
    p = np.clip(d, 0, None).ravel()
    p /= p.sum()
    R = d.shape[0]
    idx = rng.choice(p.size, size=(n_traj, T), p=p)
    iy, ix = np.divmod(idx, d.shape[1])
    # density values sit on linspace(0, 1, R) nodes; jitter within half a node spacing
    h = 1.0 / (R - 1)
    x = ix * h + rng.uniform(-0.5, 0.5, idx.shape) * h
    y = iy * h + rng.uniform(-0.5, 0.5, idx.shape) * h
    return np.clip(np.stack([x, y], -1), 0, 1)
