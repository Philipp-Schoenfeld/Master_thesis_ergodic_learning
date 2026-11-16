r"""
obstacle_cond.py
================
Obstacle conditioning for the 3D flow-matching networks (opt-in, ``--obstacle_cond``).

The conditioning is one spherical no-fly zone per sample, or none. Four pieces:

* ``ObstacleEmbedding``  (B, 4) = (cx, cy, cz, r) -> (B, D), added to the time
  conditioning exactly like the start embedding. Fourier features + MLP, last
  linear layer zero-initialised, so an obstacle-conditioned net starts out
  identical to the unconditioned one. "No obstacle" is a learned null token.
* ``sample_training_obstacles``  draws, per sample, either no obstacle or a
  sphere placed relative to the (already augmented) ground-truth path. Most
  spheres cut the path, a minority sit close by without touching it, so the
  network also learns *not* to deviate when it does not have to.
* ``repel_targets``  pushes the ground-truth control points out of the sphere.
  The stored teacher paths know nothing about the obstacle; without this step
  the CFM imitation term would pull a colliding sample straight into the
  sphere while the obstacle loss pushes it out.
* ``ObstacleLoss``  the loss on the one-step endpoint estimate ``x1_hat`` (the
  same curve the ergodic loss scores), with two hinge terms:
    - sphere term: penetration of the inflated sphere, only for samples that
      have an obstacle;
    - surface term: the target surface is an obstacle as well. Any curve point
      closer to the surface than ``surf_clearance`` is penalised, for every
      sample, with or without a sphere.
"""

import math

import torch
import torch.nn as nn


class ObstacleEmbedding(nn.Module):
    """Obstacle (B, 4) = (cx, cy, cz, r) -> (B, D); zero output at initialisation."""

    def __init__(self, D: int = 128, n_freq: int = 6, r_scale: float = 10.0):
        super().__init__()
        self.r_scale = r_scale
        self.register_buffer('freqs', 2.0 ** torch.arange(n_freq).float() * math.pi)
        self.net = nn.Sequential(
            nn.Linear(4 * n_freq * 2 + 4, D), nn.SiLU(), nn.Linear(D, D))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.cat([obs[:, :3], obs[:, 3:4] * self.r_scale], dim=-1)   # (B, 4)
        a = z.unsqueeze(-1) * self.freqs                                   # (B, 4, F)
        feat = torch.cat([a.sin(), a.cos()], dim=-1).flatten(1)
        return self.net(torch.cat([feat, z], dim=-1))


class ObstacleCondMixin:
    """Shared forward helper: `self.obstacle_emb`, `self.null_obstacle_token`."""

    def _obstacle_vector(self, ref: torch.Tensor, obstacle, obstacle_present):
        B = ref.shape[0]
        null = self.null_obstacle_token.to(ref.dtype).expand(B, -1)
        if obstacle is None:
            return null
        e = self.obstacle_emb(obstacle.to(ref.dtype))
        if obstacle_present is None:
            return e
        m = obstacle_present.view(-1, 1).to(e.dtype)
        return e * m + null * (1.0 - m)


# ===========================================================================
# Training-time sampling and target repair
# ===========================================================================

def sample_training_obstacles(x1, basis, p_obstacle=0.5, r_min=0.04, r_max=0.10,
                              hit_frac=0.7, margin=0.02, start=None):
    """-> obstacle (B, 4), present (B,) bool.

    x1: (B, nxi, >=3) augmented ground-truth control points; basis (P, nxi).
    A fraction `hit_frac` of the spheres is centred within 0.5 r of the path
    (so it penetrates it by at least half a radius); the rest sits 1.5-3 r away,
    near the path but clear of it. A sphere that would contain the start point
    is dropped (the start is held fixed and cannot be moved out of it).
    """
    b, dev = x1.shape[0], x1.device
    curve = torch.einsum('pi,bid->bpd', basis, x1[..., :3].float())
    P = curve.shape[1]
    k = torch.randint(int(0.2 * P), P, (b,), device=dev)
    pt = curve[torch.arange(b, device=dev), k]
    r = r_min + (r_max - r_min) * torch.rand(b, device=dev)
    dirn = torch.randn(b, 3, device=dev)
    dirn = dirn / dirn.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    hit = torch.rand(b, device=dev) < hit_frac
    off = torch.where(hit, 0.5 * r * torch.rand(b, device=dev),
                      r * (1.5 + 1.5 * torch.rand(b, device=dev)))
    c = pt + dirn * off[:, None]
    present = torch.rand(b, device=dev) < p_obstacle
    if start is not None:
        far = (start[:, :3].float() - c).norm(dim=-1) > (r + margin + 0.02)
        present = present & far
    obs = torch.cat([c, r[:, None]], dim=-1)
    return obs, present


def deterministic_obstacle(x1_single, basis, radius=0.08, frac=0.55, lift=0.3):
    """One sphere sitting on the path of a single sample (for plots/evaluation).

    x1_single: (nxi, >=3) control points. Returns (4,) tensor on the same device.
    """
    curve = torch.einsum('pi,id->pd', basis, x1_single[..., :3].float())
    pt = curve[int(frac * (curve.shape[0] - 1))]
    tang = curve[min(int(frac * curve.shape[0]) + 3, curve.shape[0] - 1)] \
        - curve[max(int(frac * curve.shape[0]) - 3, 0)]
    tang = tang / tang.norm().clamp(min=1e-9)
    ref = torch.tensor([0., 0., 1.], device=curve.device)
    side = torch.cross(tang, ref, dim=0)
    if side.norm() < 1e-3:
        side = torch.cross(tang, torch.tensor([1., 0., 0.], device=curve.device), dim=0)
    side = side / side.norm().clamp(min=1e-9)
    return torch.cat([pt + lift * radius * side, torch.tensor([radius], device=curve.device)])


@torch.no_grad()
def repel_targets(x1, obstacle, present, basis, margin=0.02, extra=0.01,
                  iters=60, step=1.5):
    """Push ground-truth control points out of their sphere (control point 0 held).

    Same gradient as ``obstacles.curve_repulsion_grad`` (hinge-squared on the
    rendered curve, mapped back with B^T and normalised by the basis weight),
    batched over per-sample centres/radii. Only the positions change.
    """
    pos = x1[..., :3].clone().float()
    c = obstacle[:, None, :3]
    r = (obstacle[:, 3] + margin + extra)[:, None]
    m = present[:, None].float()
    wsum = basis.sum(0).clamp(min=1e-8)[None, :, None]
    for _ in range(iters):
        curve = torch.einsum('pi,bid->bpd', basis, pos)
        d = curve - c
        dist = d.norm(dim=-1)
        v = torch.relu(r - dist) * m
        g = -(v / (dist + 1e-9)).unsqueeze(-1) * d
        g_cp = torch.einsum('pi,bpd->bid', basis, g) / wsum
        g_cp[:, 0] = 0.0
        pos = pos - step * g_cp
    out = x1.clone()
    out[..., :3] = pos.to(x1.dtype)
    return out


def sphere_penetration(curve, obstacle, margin=0.0):
    """(B, P, 3), (B, 4) -> (B, P) penetration depth into the sphere (>= 0)."""
    dist = (curve - obstacle[:, None, :3]).norm(dim=-1)
    return torch.relu(obstacle[:, None, 3] + margin - dist)


# ===========================================================================
# Loss
# ===========================================================================

class ObstacleLoss(nn.Module):
    """Sphere + surface hinge terms on the endpoint estimate x1_hat.

    sphere  = 0.5 * mean_p relu(r + margin - |P - c|)^2      (present samples)
    surface = 0.5 * mean_p relu(clearance - dist(P, surface))^2   (all samples)
    loss    = (lambda_obs * sphere + lambda_surf * surface) * t^t_power

    The ``t^t_power`` ramp matches ErgodicLoss/OrientationLoss: x1_hat is
    meaningless at small t. The surface distance comes from the same
    ``ParticleSurface`` the orientation/footprint terms use, so it lives in the
    augmented frame of the batch.
    """

    def __init__(self, nxi=25, pts=128, deg=5, lambda_obs=10000.0, lambda_surf=10000.0,
                 margin=0.02, surf_clearance=0.07, t_power=2.0, mu_thresh=0.5):
        super().__init__()
        from obstacles import bspline_basis_matrix
        self.lambda_obs, self.lambda_surf = lambda_obs, lambda_surf
        self.margin, self.surf_clearance = margin, surf_clearance
        self.t_power, self.mu_thresh = t_power, mu_thresh
        self.register_buffer('B', torch.from_numpy(bspline_basis_matrix(nxi, pts, deg)))
        # A weight of zero must still log the raw terms, so `weight` is only
        # an "is anything on" flag for the loss function.
        self.weight = max(lambda_obs, lambda_surf)

    def extra_repr(self):
        return (f"lambda_obs={self.lambda_obs}, lambda_surf={self.lambda_surf}, "
                f"margin={self.margin}, surf_clearance={self.surf_clearance}, "
                f"t_power={self.t_power}")

    def forward(self, x1_hat, particles, t, obstacle, present, surface=None):
        P = torch.einsum('pi,bid->bpd', self.B, x1_hat[..., :3].float())
        sphere = 0.5 * sphere_penetration(P, obstacle.float(), self.margin) \
            .pow(2).mean(dim=1) * present.float()
        from orientation_energy import ParticleSurface
        surf = surface or ParticleSurface(particles.float(), self.mu_thresh)
        dist = surf.distance(P)
        surface_term = 0.5 * torch.relu(self.surf_clearance - dist).pow(2).mean(dim=1)
        err = self.lambda_obs * sphere + self.lambda_surf * surface_term
        if self.t_power > 0:
            err = err * t.float() ** self.t_power
        parts = {'obs_sphere': sphere.mean().detach(),
                 'obs_surface': surface_term.mean().detach()}
        return err.mean(), parts


# ===========================================================================
# Evaluation: does the generated path respect the sphere and the surface?
# ===========================================================================

@torch.no_grad()
def evaluate_compliance(model, items, device, nxi=25, nd=3, n_gen=4, steps=100,
                        cfg_weight=2.0, radius=0.08, surf_clearance=0.07,
                        bspline_pts=256, bspline_deg=5, hit_tol=1e-3):
    """Generate with a sphere on the ground-truth path and without conditioning.

    items: list of dicts with x1 (nxi, >=3), parts (N, 4), start (3,) -- tensors.
    For every item a sphere is placed on the stored path (``deterministic_obstacle``);
    the model generates n_gen paths (a) conditioned on that sphere and (b) with the
    null obstacle token. (b) is the baseline: how often the unconditioned model runs
    through the very same sphere. Penetration is measured against the raw radius.

    Also reports the surface clearance of the generated paths and of the stored
    ground truth, so the surface term has a reference.
    """
    from flow_matching_cond_particles_crossattn import generate_particle_trajectories
    from obstacles import basis_torch
    from orientation_energy import ParticleSurface

    B = basis_torch(nxi, bspline_pts, bspline_deg, device)
    acc = {k: [] for k in ('hit_cond', 'hit_null', 'pen_cond', 'pen_null',
                           'smin_cond', 'smin_null', 'smin_gt',
                           'surf_viol_cond', 'surf_viol_null')}
    for it in items:
        x1 = it['x1'].to(device).float()
        parts = it['parts'].to(device).float()
        start = it['start'].to(device).float() if it.get('start') is not None else None
        obs = deterministic_obstacle(x1, B, radius=radius)
        surf = ParticleSurface(parts[None].expand(n_gen, -1, -1).contiguous(), 0.5)
        gt_curve = torch.einsum('pi,id->pd', B, x1[..., :3])
        acc['smin_gt'].append(ParticleSurface(parts[None], 0.5)
                              .distance(gt_curve[None]).amin().item())
        for tag, o in (('cond', obs), ('null', None)):
            gen, _ = generate_particle_trajectories(
                model, parts, num_samples=n_gen, nxi=nxi, nd=nd, steps=steps,
                device=str(device), cfg_weight=cfg_weight,
                start=start if getattr(model, 'start_cond', False) else None,
                obs_cond=o)
            curve = torch.einsum('pi,bid->bpd', B, gen[..., :3].float())
            pen = sphere_penetration(curve, obs[None].expand(n_gen, -1)).amax(dim=1)
            smin = surf.distance(curve).amin(dim=1)
            acc[f'hit_{tag}'].append((pen > hit_tol).float().mean().item())
            acc[f'pen_{tag}'].append(pen.mean().item())
            acc[f'smin_{tag}'].append(smin.mean().item())
            acc[f'surf_viol_{tag}'].append((smin < surf_clearance).float().mean().item())
    return {k: float(sum(v) / max(len(v), 1)) for k, v in acc.items()}
