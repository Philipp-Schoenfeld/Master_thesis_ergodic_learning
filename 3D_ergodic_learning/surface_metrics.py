"""Standoff and pointing measured against a local surface plane.

`ParticleSurface` (orientation_energy.py) takes distance and direction from the
*nearest occupied particle*. That is fine as a training signal at a standoff of
0.12, but as an evaluation metric it is biased near the surface: a point lying
on the surface has its nearest particle ~0.03 away *sideways* (particle
spacing), so a path that sits exactly on the surface and looks exactly along
the normal scores a standoff of ~0.03 and a pointing angle of ~90 deg. The DB
teacher (standoff 0, axis = -normal by construction) scored 0.085 / 86 deg in
the first lambda-630 comparison for exactly that reason.

Here the surface around each query point is a plane fitted (PCA) to its `k`
nearest particles -- all particles, not only the occupied ones, since the
particles sample the whole surface and mu only carries the density. Distance
is measured to that plane, pointing is the angle between the sensor axis and
the plane normal line (sign-free: the fitted normal has no orientation).
"""
import torch


def local_planes(q, particles, k=16):
    """q (B, T, 3), particles (B, N, >=3) -> (distance (B, T), normal (B, T, 3))."""
    P = particles[..., :3].float()
    q = q.float()
    B, T, _ = q.shape
    idx = torch.cdist(q, P).topk(k, dim=-1, largest=False).indices      # (B, T, k)
    nb = torch.gather(P, 1, idx.reshape(B, -1, 1).expand(-1, -1, 3)).reshape(B, T, k, 3)
    c = nb.mean(dim=2)
    X = nb - c.unsqueeze(2)
    # eigh in float64: the smallest eigenvalue of a nearly flat patch is tiny.
    _, evecs = torch.linalg.eigh((X.transpose(-1, -2) @ X).double())
    n = evecs[..., 0].float()                                            # smallest
    dist = ((q - c) * n).sum(-1).abs()
    return dist, n


def standoff_pointing(curve, axis, particles, k=16):
    """curve (B, T, 3), sensor axis (B, T, 3) -> (distance (B, T), angle_deg (B, T))."""
    dist, n = local_planes(curve, particles, k)
    a = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    cos = (a * n).sum(-1).abs().clamp(max=1.0)
    return dist, torch.rad2deg(torch.acos(cos))
