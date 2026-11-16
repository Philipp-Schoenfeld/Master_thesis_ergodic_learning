#!/usr/bin/env python3
"""Checks for the obstacle conditioning (obstacle_cond.py). CPU only, a few seconds.

    python test_obstacle_cond.py
"""
import sys

import torch

from obstacle_cond import (ObstacleLoss, deterministic_obstacle, repel_targets,
                           sample_training_obstacles, sphere_penetration)
from flow_matching_cond_particles_crossattn import (
    ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss,
    generate_particle_trajectories)
from flow_matching_cond_particles_dit import ParticleDiTFlowNetwork
from obstacles import basis_torch
from ergodic_metric import ErgodicLoss

FAILED = []


def check(name, ok, info=''):
    print(f"[{'ok' if ok else 'FAIL'}]   {name:<64} {info}")
    if not ok:
        FAILED.append(name)


def _path(B=64, nxi=25, seed=0):
    """Smooth curves inside the cube with rot6d columns, plus particles."""
    g = torch.Generator().manual_seed(seed)
    s = torch.linspace(0, 1, nxi)[None, :, None]
    a = torch.rand(B, 1, 3, generator=g) * 0.3 + 0.2
    b = torch.rand(B, 1, 3, generator=g) * 0.3 + 0.5
    pos = a + (b - a) * s + 0.05 * torch.sin(6 * s) * torch.rand(B, 1, 3, generator=g)
    rot = torch.tensor([1., 0., 0., 0., 1., 0.]).expand(B, nxi, 6)
    parts = torch.rand(B, 96, 4, generator=g)
    return torch.cat([pos, rot], dim=-1), parts


def test_sampling_and_repair():
    print("\n--- sampling and teacher repair ---\n")
    torch.manual_seed(1)
    B = basis_torch(25, 128, 5, 'cpu')
    x1, _ = _path(B=2000)
    obs, present = sample_training_obstacles(x1, B, p_obstacle=0.5, hit_frac=0.7,
                                             start=x1[:, 0, :3])
    check("share of samples with an obstacle ~ p_obstacle",
          abs(present.float().mean().item() - 0.5) < 0.06,
          f"{present.float().mean().item():.3f}")
    pen0 = sphere_penetration(torch.einsum('pi,bid->bpd', B, x1[..., :3]), obs).amax(1)
    hit = (pen0 > 0) & present
    check("~70 % of the obstacles cut the stored path",
          abs(hit.float().sum().item() / present.float().sum().item() - 0.7) < 0.08,
          f"{hit.float().sum().item() / present.float().sum().item():.3f}")
    r = obs[:, 3]
    check("radii inside [r_min, r_max]", bool((r >= 0.04 - 1e-6).all() and (r <= 0.10 + 1e-6).all()))

    fixed = repel_targets(x1, obs, present, B, margin=0.02)
    curve = torch.einsum('pi,bid->bpd', B, fixed[..., :3])
    pen = sphere_penetration(curve, obs, margin=0.0).amax(1) * present.float()
    check("repaired teacher clears the raw sphere (max penetration)",
          pen.max().item() < 1e-3, f"max {pen.max().item():.2e}, "
          f"{(pen > 1e-4).sum().item()} of {int(present.sum())} above 1e-4")
    check("control point 0 (start) unchanged", torch.equal(fixed[:, 0], x1[:, 0]))
    check("samples without obstacle unchanged",
          torch.equal(fixed[~present], x1[~present]))
    check("rotation channels unchanged", torch.equal(fixed[..., 3:], x1[..., 3:]))
    moved = (fixed[..., :3] - x1[..., :3]).norm(dim=-1).amax(1)
    check("repair moves the path only moderately (< 0.2)",
          moved[present].max().item() < 0.2, f"max {moved[present].max().item():.3f}")


def test_loss():
    print("\n--- ObstacleLoss ---\n")
    B = basis_torch(25, 128, 5, 'cpu')
    x1, parts = _path(B=8)
    t = torch.full((8,), 0.9)
    loss = ObstacleLoss(nxi=25, pts=128, lambda_obs=1000., lambda_surf=1000.,
                        surf_clearance=0.07)
    far = torch.tensor([5., 5., 5., 0.05]).expand(8, -1)
    pres = torch.ones(8, dtype=torch.bool)
    # push particles far away from the path so the surface term is off
    parts_far = parts.clone(); parts_far[..., :3] += 10.
    l0, p0 = loss(x1, parts_far, t, far, pres)
    check("zero loss for a clear path (far sphere, far surface)", l0.item() == 0.0,
          f"{l0.item():.2e}")
    on = torch.stack([deterministic_obstacle(x1[i], B, radius=0.08) for i in range(8)])
    xg = x1.clone().requires_grad_(True)
    l1, p1 = loss(xg, parts_far, t, on, pres)
    l1.backward()
    check("sphere term > 0 and finite gradient for a path through the sphere",
          l1.item() > 0 and torch.isfinite(xg.grad).all().item(), f"{l1.item():.3e}")
    l2, p2 = loss(x1, parts_far, t, on, torch.zeros(8, dtype=torch.bool))
    check("absent obstacle -> sphere term off", p2['obs_sphere'].item() == 0.0)
    # surface term: particles right on the path
    parts_on = parts.clone(); parts_on[..., :3] = torch.einsum('pi,bid->bpd', B, x1[..., :3])[:, ::1][:, :96] \
        if False else x1[:, :1, :3].expand(-1, 96, -1)
    l3, p3 = loss(x1, parts_on, t, far, torch.zeros(8, dtype=torch.bool))
    check("surface term active without any obstacle", p3['obs_surface'].item() > 0,
          f"{p3['obs_surface'].item():.3e}")


def test_models():
    print("\n--- conditioning in U-Net and DiT ---\n")
    x1, parts = _path(B=4)
    obs = torch.tensor([[0.5, 0.5, 0.5, 0.08]]).expand(4, -1).contiguous()
    pres = torch.tensor([True, False, True, False])
    t = torch.rand(4)
    for name, mk in (
        ('unet', lambda **k: ParticleCrossAttnFlowNetwork(nxi=25, D=32, n_heads=4,
                                                         predict_orientation=True,
                                                         start_cond=True, **k)),
        ('dit', lambda **k: ParticleDiTFlowNetwork(nxi=25, D=32, depth=2, n_heads=4,
                                                  predict_orientation=True,
                                                  start_cond=True, **k)),
    ):
        torch.manual_seed(0)
        base = mk()
        torch.manual_seed(0)
        cond = mk(obstacle_cond=True)
        res = cond.load_state_dict(base.state_dict(), strict=False)
        check(f"{name}: only obstacle keys are missing after loading a plain net",
              all(k.startswith(('obstacle_emb.', 'null_obstacle_token'))
                  for k in res.missing_keys) and not res.unexpected_keys,
              f"{len(res.missing_keys)} new keys")
        base.eval(); cond.eval()
        with torch.no_grad():
            vb, rb = base(x1, t, parts, start=x1[:, 0, :3])
            vc, rc = cond(x1, t, parts, start=x1[:, 0, :3], obstacle=obs,
                          obstacle_present=pres)
        check(f"{name}: obstacle-conditioned net == plain net at initialisation",
              torch.equal(vb, vc) and torch.equal(rb, rc))

        # now make the embedding non-trivial and check that it has an effect
        with torch.no_grad():
            # Output layers are zero-initialised (adaLN-Zero etc.), so a freshly
            # built net outputs 0 whatever it is conditioned on; perturb every
            # still-zero parameter to emulate a trained net.
            for p in cond.parameters():
                if p.abs().sum() == 0:
                    p.normal_(0, 0.1)
            v1, _ = cond(x1, t, parts, start=x1[:, 0, :3], obstacle=obs,
                         obstacle_present=pres)
            v2, _ = cond(x1, t, parts, start=x1[:, 0, :3], obstacle=obs * 0.5,
                         obstacle_present=pres)
        d = (v1 - v2).abs().amax(dim=(1, 2))
        check(f"{name}: output depends on the obstacle only where present",
              bool(d[pres].min() > 0) and bool(d[~pres].max() == 0),
              f"present {d[pres].min():.2e}, absent {d[~pres].max():.2e}")

        # training step through the loss with all terms
        cond.train()
        erg = ErgodicLoss(nxi=25, nd=3, metric='sinkhorn', sinkhorn_blur=0.05,
                          weight=10., ergodic_on='footprint')
        from orientation_energy import OrientationLoss
        ori = OrientationLoss(nxi=25, pts=64, weight=0.012)
        oloss = ObstacleLoss(nxi=25, pts=64)
        loss, parts_d = compute_particle_cfm_loss(
            cond, x1, parts, p_drop=0.1, ergodic=erg, orientation=ori,
            w_cfm_rot=0.5, obstacle=obs, obstacle_present=pres, obstacle_loss=oloss)
        loss.backward()
        got = all(k in parts_d for k in ('obs', 'obs_sphere', 'obs_surface'))
        gn = cond.obstacle_emb.net[-1].weight.grad
        check(f"{name}: loss with obstacle terms backprops into the embedding",
              got and gn is not None and gn.abs().sum().item() > 0
              and torch.isfinite(loss).item())

        cond.eval()
        gen, rot = generate_particle_trajectories(
            cond, parts[0], num_samples=3, nxi=25, steps=5, device='cpu',
            start=x1[0, 0, :3], obs_cond=obs[0])
        gen0, _ = generate_particle_trajectories(
            cond, parts[0], num_samples=3, nxi=25, steps=5, device='cpu',
            start=x1[0, 0, :3])
        check(f"{name}: generation with / without obstacle runs, shapes ok",
              gen.shape == (3, 25, 3) and gen0.shape == (3, 25, 3) and rot.shape == (3, 25, 6))


if __name__ == '__main__':
    test_sampling_and_repair()
    test_loss()
    test_models()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print("all obstacle-conditioning checks passed")
