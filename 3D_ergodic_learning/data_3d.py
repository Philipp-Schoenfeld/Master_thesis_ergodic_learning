r"""
data_3d.py
==========
Dataset access for the 3D pipeline, including the bridge from the existing 2D
database.

The working assumption for this port is: **the database holds 3D distributions
and 3D trajectories, but they lie on a plane.** Two consequences drive the
design here.

1. `load_pairs` reads whatever the blob contains. A stored trajectory of shape
   (T, 3) is used as-is. A stored (T, 2) trajectory is *lifted* onto the plane
   z = `Z_PLANE` — so the current 2D database can drive the 3D code today, and a
   genuinely 3D database drops in later without touching the callers.

2. A perfectly flat target is not representable by a truncated cosine basis: a
   Dirac sheet in z has energy at every frequency. The density volume therefore
   places the 2D density in a Gaussian *slab* of width `Z_SIGMA` around the
   plane. The default 0.05 is chosen to be resolvable: at K = 10 the shortest
   representable wavelength is 2/(K-1) ~ 0.22, so a slab narrower than about
   0.05 would alias rather than be described. Making the slab thinner is not
   "more accurate" — it is asking the basis for something it cannot express.

Everything else (particle sampling, augmentation) is the 2D logic widened by one
axis, with one deliberate default: augmentation rotates about the plane normal
only, so planar data stays planar and a 3D run can be checked against the 2D
reference. `--rot_full` switches to general SO(3).
"""

import json
import math
import os
import sqlite3
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (os.path.join(_root, 'bsplinax-main'),
           os.path.join(_root, 'src'),
           os.path.join(_root, 'thesis_architecture', 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The 2D density rasteriser is reused as-is; only the extrusion is new.
from shape_library import pdf_on_grid          # noqa: E402

DEFAULT_DB = os.path.join(_root, 'thesis_architecture', 'ergodic_dataset_generator',
                          'ergodic_dataset_775.db')

Z_PLANE = 0.5      # where a lifted 2D trajectory is placed
Z_SIGMA = 0.05     # slab half-width of the density around that plane


# ── Database ─────────────────────────────────────────────────────────────────
def load_pairs(nxi, db_path=DEFAULT_DB, splits=('train', 'val')):
    """Read (trajectory, density_params) pairs, lifting 2D content to 3D.

    Returns two dicts keyed by a unique shape label:
        trajectories[label] -> (nxi, 3) float32
        shape_defs[label]   -> dict, the density definition
        splits_map[label]   -> 'train' | 'val'
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    q = ("SELECT trajectory, shape_name, split, density_params FROM ergodic_pairs "
         f"WHERE split IN ({','.join('?' * len(splits))}) ORDER BY id ASC")
    rows = cur.execute(q, tuple(splits)).fetchall()
    conn.close()

    trajectories, shape_defs, splits_map = {}, {}, {}
    n_lifted = 0
    for blob, label, split, params_str in rows:
        xy = np.frombuffer(blob, dtype=np.float32)
        if xy.size % 3 == 0 and _looks_3d(xy):
            pts = xy.reshape(-1, 3)
        else:
            pts = _lift_to_plane(xy.reshape(-1, 2))
            n_lifted += 1

        idx = np.linspace(0, len(pts) - 1, nxi).astype(int)
        key = label if label not in trajectories else f"{label}#{len(trajectories)}"
        trajectories[key] = pts[idx].astype(np.float32)
        shape_defs[key] = json.loads(params_str)
        splits_map[key] = split

    if n_lifted:
        print(f"  [data_3d] {n_lifted} trajectories were 2D and were lifted to "
              f"the plane z={Z_PLANE}")
    return trajectories, shape_defs, splits_map


def _looks_3d(flat):
    """Heuristic: a (T,3) blob divides by 3 and not plausibly by 2.

    The stored blobs carry no dimension tag, so ambiguity is real: a length
    divisible by 6 fits both readings. Preferring the 2D reading in that case is
    the safe default for the current database, and a genuinely 3D database
    should carry an explicit column — see the note in README.md.
    """
    return flat.size % 3 == 0 and flat.size % 2 != 0


def _lift_to_plane(xy, z=Z_PLANE):
    """(T, 2) -> (T, 3) by appending a constant height."""
    return np.concatenate([xy, np.full((len(xy), 1), z, dtype=xy.dtype)], axis=1)


# ── Density volumes ──────────────────────────────────────────────────────────
def density_volume(shape_def, resolution=64, z_plane=Z_PLANE, z_sigma=Z_SIGMA):
    """3D density volume (R, R, R) on [0,1]^3, normalised to max 1.

    Built by extruding the 2D density into a Gaussian slab around `z_plane`.
    Indexing is [z, y, x] to match `target_coeffs_from_grid`.
    """
    d2, _, _ = pdf_on_grid(shape_def, resolution=resolution)
    d2 = np.asarray(d2, dtype=np.float64)
    if d2.max() > 0:
        d2 = d2 / d2.max()

    zs = np.linspace(0.0, 1.0, resolution)
    prof = np.exp(-0.5 * ((zs - z_plane) / z_sigma) ** 2)     # (R,)
    vol = prof[:, None, None] * d2[None, :, :]                # (R, R, R)
    if vol.max() > 0:
        vol = vol / vol.max()
    return vol.astype(np.float32)


# ── Particle sampling ────────────────────────────────────────────────────────
def sample_particles(volumes, shape_indices, N, device, threshold=1e-5,
                     mode='uniform'):
    """Sample N particles per batch element from 3D density volumes.

    volumes:       (S, R, R, R) stack, indexed [z, y, x]
    shape_indices: (B,) indices into that stack
    Returns:       (B, N, 4) = (x, y, z, mu)

    `mode='uniform'` draws uniformly over the *support* and attaches the density
    value as a feature — the reason `target_coeffs_from_particles` has to use a
    mu-weighted mean. `mode='density'` draws proportional to the density.
    """
    B = shape_indices.shape[0]
    R = volumes.shape[-1]
    Dz, Hy, Wx = volumes.shape[1], volumes.shape[2], volumes.shape[3]

    batch = volumes[shape_indices].to(device)                 # (B, Dz, Hy, Wx)
    flat = batch.reshape(B, -1)

    weights = (flat > threshold).float() if mode == 'uniform' else flat + 1e-7
    # A volume with empty support would make multinomial fail; fall back to
    # uniform over all voxels so one bad shape cannot kill a run.
    empty = weights.sum(dim=1) <= 0
    if bool(empty.any()):
        weights[empty] = 1.0

    idx = torch.multinomial(weights, num_samples=N, replacement=True)

    iz = idx // (Hy * Wx)
    rem = idx % (Hy * Wx)
    iy = rem // Wx
    ix = rem % Wx

    # Sub-voxel jitter, same convention as the 2D version.
    jx = (torch.rand(B, N, device=device) - 0.5) / max(Wx - 1, 1)
    jy = (torch.rand(B, N, device=device) - 0.5) / max(Hy - 1, 1)
    jz = (torch.rand(B, N, device=device) - 0.5) / max(Dz - 1, 1)

    px = torch.clamp(ix.float() / max(Wx - 1, 1) + jx, 0.0, 1.0)
    py = torch.clamp(iy.float() / max(Hy - 1, 1) + jy, 0.0, 1.0)
    pz = torch.clamp(iz.float() / max(Dz - 1, 1) + jz, 0.0, 1.0)

    pmu = torch.gather(flat, 1, idx)
    return torch.stack([px, py, pz, pmu], dim=-1)


# ── Augmentation ─────────────────────────────────────────────────────────────
def _rot_z(angles):
    """Batch of rotations about the z axis. angles: (B,) -> (B, 3, 3)."""
    c, s = torch.cos(angles), torch.sin(angles)
    o, l = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([
        torch.stack([c, -s, o], dim=-1),
        torch.stack([s,  c, o], dim=-1),
        torch.stack([o,  o, l], dim=-1),
    ], dim=1)


def _rot_axis_angle(axis, angles):
    """Rodrigues rotation. axis: (B, 3) unit, angles: (B,) -> (B, 3, 3)."""
    B = axis.shape[0]
    c, s = torch.cos(angles)[:, None, None], torch.sin(angles)[:, None, None]
    K = torch.zeros(B, 3, 3, device=axis.device, dtype=axis.dtype)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(B, 3, 3)
    return I + s * K + (1 - c) * (K @ K)


def augment_batch(x, particles, p_flip=0.0, rot_range=20.0,
                  scale_range=(0.75, 1.25), trans_range=0.08, noise_std=0.01,
                  rot_full=False):
    """Synchronised geometric augmentation of trajectory and particle cloud.

    x:         (B, nxi, 3) positions, or (B, nxi, 9) = positions + 6D rotation
    particles: (B, N, 4) where [..., :3] is (x,y,z) and [..., 3] is mu

    When x carries a 6D rotation block it is rotated *with* the positions:
    R_new = R_aug @ R_old, applied to the two stored columns. Forgetting this is
    the classic silent bug of adding orientation to an existing pipeline — the
    positions move, the frames do not, and the network learns an inconsistency
    between where the robot is and where it looks.

    Rotation defaults to the plane normal (z) only. That is the right default
    for planar data: it keeps the plane a plane, which is what makes a 3D run
    comparable to the 2D reference. `rot_full=True` draws a random axis instead
    and tilts the data out of its plane — useful once the database is genuinely
    3D, misleading before that.
    """
    B = x.shape[0]
    device = x.device
    has_rot = x.shape[-1] == 9
    pos, rot6d = (x[..., :3], x[..., 3:]) if has_rot else (x, None)

    centroids = pos.clone().mean(dim=1, keepdim=True)        # (B, 1, 3)

    angles = (torch.rand(B, device=device) * 2 - 1) * rot_range * (math.pi / 180.0)
    if rot_full:
        axis = torch.randn(B, 3, device=device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        R = _rot_axis_angle(axis, angles)
    else:
        R = _rot_z(angles)

    scales = torch.empty(B, 1, 1, device=device).uniform_(*scale_range)
    trans = torch.empty(B, 1, 3, device=device).uniform_(-trans_range, trans_range)
    flip_mask = torch.rand(B, device=device) < p_flip

    def apply(pts):
        out = torch.bmm(pts - centroids, R.transpose(1, 2)) + centroids
        out = (out - centroids) * scales + centroids
        out = out + trans
        out[flip_mask, :, 0] = 1.0 - out[flip_mask, :, 0]
        return out

    x_out = apply(pos)
    x_out = torch.clamp(x_out + torch.randn_like(x_out) * noise_std, 0.0, 1.0)

    if has_rot:
        # Rotate the two stored basis columns by the same R. Translation and
        # scale do not act on a rotation; the mirror does, but flipping x turns
        # a right-handed frame into a left-handed one, so the third column is
        # rebuilt by Gram-Schmidt downstream rather than mirrored here.
        c1 = torch.bmm(rot6d[..., :3], R.transpose(1, 2))
        c2 = torch.bmm(rot6d[..., 3:], R.transpose(1, 2))
        x_out = torch.cat([x_out, c1, c2], dim=-1)

    p_xyz = apply(particles[..., :3])
    p_xyz = torch.clamp(p_xyz + torch.randn_like(p_xyz) * noise_std, 0.0, 1.0)
    p_out = torch.cat([p_xyz, particles[..., 3:4]], dim=-1)

    return x_out, p_out


# ── Target coefficient cache ─────────────────────────────────────────────────
def prepare_targets(labels, shape_defs, resolution, K, cache_dir=None,
                    z_plane=Z_PLANE, z_sigma=Z_SIGMA, use_cache=True):
    """Density volumes and phi_k for a list of labels.

    phi_k is constant during training and the volumes are expensive to build
    (the 2D rasteriser runs through JAX on the CPU), so both are cached to an
    .npz keyed by the label set and the grid parameters.

    Returns (labels_used, volumes (S,R,R,R) float32, phi (S,M) float32).
    """
    from ergodic_energy_torch import make_k_grid, target_coeffs_from_grid

    cache_dir = cache_dir or os.path.join(_here, 'cache')
    key = f"{len(labels)}_{resolution}_{K}_{z_plane:g}_{z_sigma:g}_" \
          f"{abs(hash(tuple(sorted(labels)))) % (10 ** 10)}"
    path = os.path.join(cache_dir, f"targets3d_{key}.npz")

    if use_cache and os.path.isfile(path):
        z = np.load(path)
        used = [str(n) for n in z['labels']]
        print(f"  Targets loaded from cache: {os.path.basename(path)} "
              f"({len(used)} shapes)")
        return used, z['volumes'], z['phi']

    k_idx_np, _ = make_k_grid(K)
    k_idx = torch.tensor(k_idx_np, dtype=torch.float64)

    used, vols, phis, skipped = [], [], [], []
    print(f"  Building {len(labels)} density volumes "
          f"(grid {resolution}^3, K={K} -> {K ** 3} modes)...")
    from tqdm import tqdm
    for lbl in tqdm(labels, unit='shape'):
        try:
            vol = density_volume(shape_defs[lbl], resolution, z_plane, z_sigma)
        except Exception as e:
            skipped.append((lbl, type(e).__name__))
            continue
        phi = target_coeffs_from_grid(torch.tensor(vol, dtype=torch.float64), k_idx)
        used.append(lbl)
        vols.append(vol)
        phis.append(phi.numpy().astype(np.float32))

    if skipped:
        print(f"  [WARN] Skipped {len(skipped)} shape(s), e.g. {skipped[:3]}")
    if not used:
        raise RuntimeError("No target shapes could be built.")

    volumes, phi = np.stack(vols), np.stack(phis)
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, volumes=volumes, phi=phi,
                            labels=np.array(used, dtype='U64'))
        print(f"  Targets cached -> {os.path.basename(path)}")
    return used, volumes, phi
