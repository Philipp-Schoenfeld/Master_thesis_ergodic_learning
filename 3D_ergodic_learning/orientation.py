r"""
orientation.py
==============
Everything needed to give a trajectory an orientation as well as a position.

Three independent pieces live here:

1. **Rotation representations.** The 6D representation of Zhou et al. (2019):
   predict two 3-vectors, Gram-Schmidt them into a rotation matrix. The reason
   this matters is not convenience — it is that *any* representation of SO(3)
   with fewer than five dimensions is necessarily discontinuous, so a network
   regressing Euler angles or quaternions has to learn around a discontinuity it
   can never fit. 6D is the smallest continuous choice.

   A second consequence is what makes this port cheap: R^6 is a **vector
   space**. Conditional flow matching interpolates linearly and takes MSE on the
   velocity, both of which are only valid in a vector space. Predicting 6D and
   projecting to SO(3) *afterwards* keeps the whole existing CFM machinery valid;
   predicting quaternions or matrices directly would not. The same argument
   applies to the B-spline: the basis is applied to the 6D control values, and
   the projection happens per curve point.

   That is a deliberate trade-off, and it is worth naming: the resulting R(t) is
   smooth in the ambient 6D space, not a constant-speed geodesic on SO(3). If
   intrinsic geodesic smoothness turns out to matter, the principled upgrade is
   a cumulative B-spline on the Lie group (Sommer et al. 2020) — a much larger
   change, deliberately not taken here.

2. **Frames derived from a curve (Stufe 0).** Two constructions, because they
   answer different questions:
     * `lookat_frames`  — the sensor axis points at the target, the remaining
       degree of freedom follows the direction of travel. Task-aware; this is
       the one to use when there is something to look at.
     * `rmf_frames`     — rotation-minimising frame by double reflection
       (Wang et al. 2008). Follows the curve with the least possible twist, no
       target needed. Frenet-Serret is deliberately *not* the default: it is
       undefined at zero curvature and flips through inflection points.

3. **A surface field.** From a density volume, the distance to the occupied set
   and the direction towards it, both trilinearly interpolated so they stay
   differentiable in the query point. This is what the pointing and standoff
   energies need, and it is computed once per shape.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


# ── Rotation representations ─────────────────────────────────────────────────
def rot6d_to_matrix(x):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt. Zhou et al. (2019).

    The two input vectors need not be orthogonal or unit; the result is always a
    proper rotation (det = +1) as long as they are linearly independent.
    """
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)          # columns are the axes


def matrix_to_rot6d(R):
    """(..., 3, 3) -> (..., 6). Inverse of the above up to the discarded column."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


def identity_rot6d(shape, device=None, dtype=torch.float32):
    """6D encoding of the identity rotation, broadcast to `shape` + (6,)."""
    e = torch.zeros(*shape, 6, device=device, dtype=dtype)
    e[..., 0] = 1.0
    e[..., 4] = 1.0
    return e


def sensor_axis(R, axis=2):
    """The body axis a sensor looks along. (..., 3, 3) -> (..., 3).

    Column `axis` of R, i.e. the world-frame direction of the body basis vector.
    Default 2 = body z, the usual optical axis convention.
    """
    return R[..., axis]


# ── SO(3) helpers ────────────────────────────────────────────────────────────
def geodesic_angle(Ra, Rb, eps=1e-12):
    """Rotation angle between two rotations, in radians. (..., 3, 3) -> (...,).

    Computed as ``atan2(|w|, (tr - 1) / 2)`` with w the vector part of the
    skew-symmetric component of the relative rotation, since |w| = sin(theta)
    and (tr-1)/2 = cos(theta).

    The textbook form ``arccos((tr - 1) / 2)`` is *not* used, and the reason is
    not cosmetic. arccos has infinite derivative at +-1, so it must be clamped
    away from the endpoints to avoid NaN — and any such clamp puts a hard floor
    under the result: with eps = 1e-7 two *identical* rotations report
    4.5e-4 rad rather than zero. Summed over a 100-point curve that is 0.045 rad
    of pure artefact in `rot_path_length`, and it makes the angular smoothness
    term of a perfectly constant frame nonzero. atan2 has neither problem and is
    equally accurate near theta = pi.

    The `eps` inside the square root is not optional. |w| = 0 exactly whenever
    two consecutive rotations are identical, and d/dx sqrt(x) is infinite at
    x = 0, so the *backward* pass returns NaN there even though the forward
    value is a correct 0. That state is not exotic: the orientation head is
    zero-initialised, so at the first training step every frame along the curve
    is identical and every consecutive pair hits it at once. Without the eps
    that first optimiser step poisons the whole network — observed as every
    weight turning NaN after exactly one step, while the forward pass had just
    reported a perfectly healthy `angsmooth = 0`.

    The fix is a masked branch, not `sqrt(sq + eps)`. An eps under the root
    would floor every angle at sqrt(eps), and those floors accumulate: summed
    over a 60-point straight line they turn the rotation-minimising frame's
    exactly-zero twist into 3e-5 rad of pure artefact — the very failure mode
    the paragraph above rejects arccos for. Branching keeps the forward value
    exactly 0 below the threshold and returns a finite 0 gradient there.
    """
    rel = Ra.transpose(-1, -2) @ Rb
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    wx = rel[..., 2, 1] - rel[..., 1, 2]
    wy = rel[..., 0, 2] - rel[..., 2, 0]
    wz = rel[..., 1, 0] - rel[..., 0, 1]
    sq = wx * wx + wy * wy + wz * wz                   # = (2 sin(theta))^2
    # Masked branch rather than `sqrt(sq + eps)`: this keeps the forward value
    # exactly 0 (an eps floor would accumulate along a curve and break the
    # rotation-minimising-frame property, which is exact by construction) while
    # still handing back a finite 0 gradient at sq = 0.
    nz = sq > eps
    sin2 = torch.where(nz, torch.sqrt(torch.where(nz, sq, torch.ones_like(sq))),
                       torch.zeros_like(sq))
    return torch.atan2(sin2, tr - 1.0)


def rot_path_length(R):
    """Total geodesic length of a rotation sequence. (B, T, 3, 3) -> (B,)."""
    return geodesic_angle(R[:, :-1], R[:, 1:]).sum(dim=1)


def frame_from_axes(z_dir, hint, eps=1e-8):
    """Build a rotation whose body-z is `z_dir` and body-x follows `hint`.

    z_dir: (..., 3) desired optical axis (need not be unit)
    hint:  (..., 3) preferred body-x direction, projected orthogonal to z_dir

    Where `hint` is (anti)parallel to z_dir the construction is degenerate; the
    fallback is whichever coordinate axis is least aligned with z_dir, which is
    always well conditioned.
    """
    z = F.normalize(z_dir, dim=-1, eps=eps)
    x = hint - (z * hint).sum(-1, keepdim=True) * z
    n = x.norm(dim=-1, keepdim=True)

    eye = torch.eye(3, device=z.device, dtype=z.dtype)
    alt_idx = z.abs().argmin(dim=-1)                              # (...,)
    alt = eye[alt_idx]                                            # (..., 3)
    alt = alt - (z * alt).sum(-1, keepdim=True) * z
    alt = F.normalize(alt, dim=-1, eps=eps)

    x = torch.where(n < 1e-6, alt, x / n.clamp(min=eps))
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def curve_tangents(curve, eps=1e-8):
    """Unit tangents by central differences. (B, T, 3) -> (B, T, 3)."""
    t = torch.zeros_like(curve)
    t[:, 1:-1] = curve[:, 2:] - curve[:, :-2]
    t[:, 0] = curve[:, 1] - curve[:, 0]
    t[:, -1] = curve[:, -1] - curve[:, -2]
    return F.normalize(t, dim=-1, eps=eps)


# ── Stufe 0: frames derived from the curve ───────────────────────────────────
def lookat_frames(curve, target_dir):
    """Task-aware frame: body-z points at the target, body-x follows motion.

    curve:      (B, T, 3)
    target_dir: (B, T, 3) direction from each curve point towards the target
    -> (B, T, 3, 3)
    """
    return frame_from_axes(target_dir, curve_tangents(curve))


def rmf_frames(curve, initial_normal=None, eps=1e-8):
    """Rotation-minimising frame by double reflection (Wang et al. 2008).

    Body-z is the unit tangent; the normal plane is propagated with as little
    twist as possible. Unlike Frenet-Serret this is defined on straight segments
    and does not flip at inflection points, which is exactly why it is the
    default here — generated trajectories routinely contain near-straight runs.

    curve: (B, T, 3) -> (B, T, 3, 3)
    """
    B, T, _ = curve.shape
    tang = curve_tangents(curve, eps)

    if initial_normal is None:
        seed = torch.zeros_like(tang[:, 0])
        seed[..., 2] = 1.0
        # If the first tangent is nearly vertical, seed with x instead.
        bad = tang[:, 0, 2].abs() > 0.9
        seed[bad] = torch.tensor([1.0, 0.0, 0.0], device=curve.device,
                                 dtype=curve.dtype)
        n0 = seed - (tang[:, 0] * seed).sum(-1, keepdim=True) * tang[:, 0]
        initial_normal = F.normalize(n0, dim=-1, eps=eps)

    normals = [initial_normal]
    for i in range(T - 1):
        x_i, x_j = curve[:, i], curve[:, i + 1]
        t_i, t_j = tang[:, i], tang[:, i + 1]
        n_i = normals[-1]

        # Reflection 1: across the plane bisecting the two points.
        v1 = x_j - x_i
        c1 = (v1 * v1).sum(-1, keepdim=True).clamp(min=eps)
        n_l = n_i - (2.0 / c1) * (v1 * n_i).sum(-1, keepdim=True) * v1
        t_l = t_i - (2.0 / c1) * (v1 * t_i).sum(-1, keepdim=True) * v1

        # Reflection 2: across the plane bisecting the reflected and next tangent.
        v2 = t_j - t_l
        c2 = (v2 * v2).sum(-1, keepdim=True).clamp(min=eps)
        n_j = n_l - (2.0 / c2) * (v2 * n_l).sum(-1, keepdim=True) * v2
        normals.append(F.normalize(n_j, dim=-1, eps=eps))

    N = torch.stack(normals, dim=1)                    # (B, T, 3)
    Bn = torch.cross(tang, N, dim=-1)
    return torch.stack([N, Bn, tang], dim=-1)          # body-z is the tangent


def frenet_frames(curve, eps=1e-8):
    """Frenet-Serret frame. Provided for comparison only — see `rmf_frames`."""
    t = curve_tangents(curve, eps)
    dt = torch.zeros_like(t)
    dt[:, 1:-1] = t[:, 2:] - t[:, :-2]
    dt[:, 0] = t[:, 1] - t[:, 0]
    dt[:, -1] = t[:, -1] - t[:, -2]
    n = F.normalize(dt, dim=-1, eps=eps)
    b = torch.cross(t, n, dim=-1)
    return torch.stack([n, b, t], dim=-1)


# ── Surface field ────────────────────────────────────────────────────────────
class SurfaceField:
    """Distance to the target's occupied set and the direction towards it.

    Built once per shape from a density volume with a Euclidean distance
    transform; queried with trilinear interpolation so both quantities stay
    differentiable in the query point — which is what lets the pointing and
    standoff energies push the *trajectory*, not just score it.

    Convention: `distance` is 0 inside the occupied set and grows outside, in
    units of domain widths. `direction` is a unit vector from the query point
    towards the nearest occupied voxel; inside the set it degenerates and is
    replaced by the stored fallback (the plane normal for planar targets).
    """

    def __init__(self, volume, threshold=0.05, device='cpu', fallback=(0., 0., -1.)):
        from scipy.ndimage import distance_transform_edt

        vol = np.asarray(volume)
        if vol.max() > 0:
            vol = vol / vol.max()
        occ = vol > threshold                                   # (R, R, R) [z,y,x]
        self.shape = occ.shape
        R = occ.shape[-1]

        if not occ.any():
            raise ValueError("SurfaceField: the volume has no occupied voxels")

        # Distance from every voxel to the nearest occupied one, plus the index
        # of that voxel — one pass gives both.
        dist_vox, idx = distance_transform_edt(~occ, return_indices=True)
        spacing = 1.0 / max(R - 1, 1)
        dist = dist_vox * spacing                               # domain widths

        zz, yy, xx = np.meshgrid(*[np.arange(s) for s in occ.shape], indexing='ij')
        dvec = np.stack([(idx[2] - xx), (idx[1] - yy), (idx[0] - zz)],
                        axis=0).astype(np.float32) * spacing    # (3, R, R, R) xyz
        norm = np.linalg.norm(dvec, axis=0, keepdims=True)
        inside = norm[0] < 1e-9
        fb = np.asarray(fallback, dtype=np.float32)
        fb = fb / max(np.linalg.norm(fb), 1e-12)
        dvec = np.where(norm < 1e-9, fb[:, None, None, None], dvec / np.maximum(norm, 1e-9))

        self.device = device
        # grid_sample wants (N, C, D, H, W) with D=z, H=y, W=x.
        self._dist = torch.tensor(dist, dtype=torch.float32,
                                  device=device)[None, None]
        self._dir = torch.tensor(dvec, dtype=torch.float32, device=device)[None]
        self._inside = torch.tensor(inside, dtype=torch.float32,
                                    device=device)[None, None]

    def to(self, device):
        self.device = device
        self._dist = self._dist.to(device)
        self._dir = self._dir.to(device)
        self._inside = self._inside.to(device)
        return self

    def _sample(self, field, pts):
        """Trilinear sample of `field` (1, C, R, R, R) at pts (B, T, 3) in [0,1]."""
        B, T, _ = pts.shape
        C = field.shape[1]
        # [0,1] -> [-1,1]; grid_sample expects the last axis ordered (x, y, z).
        g = (pts * 2.0 - 1.0).view(B, 1, 1, T, 3)
        out = F.grid_sample(field.expand(B, C, *field.shape[2:]), g,
                            mode='bilinear', padding_mode='border',
                            align_corners=True)                  # (B, C, 1, 1, T)
        return out.view(B, C, T).permute(0, 2, 1)                # (B, T, C)

    def distance(self, pts):
        """(B, T, 3) -> (B, T). Distance to the occupied set, 0 inside."""
        return self._sample(self._dist, pts)[..., 0]

    def direction(self, pts, eps=1e-8):
        """(B, T, 3) -> (B, T, 3). Unit direction towards the occupied set."""
        d = self._sample(self._dir, pts)
        return F.normalize(d, dim=-1, eps=eps)

    def inside(self, pts):
        """(B, T, 3) -> (B, T) in [0,1]. Soft indicator of being inside."""
        return self._sample(self._inside, pts)[..., 0]

    def footprint(self, pts, axis):
        """Where the sensor beam lands: pts + distance(pts) * axis.

        An approximation of ray-casting that keeps the dependence on *both* the
        position and the orientation, and is differentiable in both. Pointing
        straight at the surface puts the footprint on it; tilting away slides it
        along. That coupling is the entire reason orientation can influence
        coverage rather than merely decorate it.
        """
        return pts + self.distance(pts).unsqueeze(-1) * axis


def frames_for_curve(curve, field=None, mode='lookat'):
    """Stufe 0 entry point: an orientation for a curve, without any model.

    mode='lookat' needs a field (something to point at); mode='rmf' does not.
    """
    if mode == 'rmf':
        return rmf_frames(curve)
    if mode == 'frenet':
        return frenet_frames(curve)
    if mode == 'lookat':
        if field is None:
            raise ValueError("mode='lookat' needs a SurfaceField to point at")
        return lookat_frames(curve, field.direction(curve))
    raise ValueError(f"unknown frame mode: {mode}")
