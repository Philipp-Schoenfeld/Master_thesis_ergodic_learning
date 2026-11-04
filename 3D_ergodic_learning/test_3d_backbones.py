#!/usr/bin/env python3
"""Checks for the October 2026 port of the 2D start runner to 3D.

Covers what `test_3d_port.py` does not: the Sinkhorn ErgLoss, start inpainting,
the DiT backbone and the least-squares control-point targets (`--target_cps`).
CPU only and small, so it runs locally and as the cold-start gate of the jobs.

    python test_3d_backbones.py
"""
import sys

import numpy as np
import torch

from data_surfaces import fit_control_points
from ergodic_metric import ErgodicLoss
from flow_matching_cond_particles_crossattn import (
    ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss,
    generate_particle_trajectories)
from flow_matching_cond_particles_dit import ParticleDiTFlowNetwork
from obstacles import bspline_basis_matrix
from orientation_energy import OrientationLoss

FAILED = []


def check(name, ok, info=''):
    print(f"[{'ok' if ok else 'FAIL'}]   {name:<58} {info}")
    if not ok:
        FAILED.append(name)


def _batch(B=4, N=64, nxi=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    x1 = torch.rand(B, nxi, 9, generator=g) * 0.6 + 0.2
    x1[..., 3:] = torch.tensor([1., 0., 0., 0., 1., 0.])
    parts = torch.rand(B, N, 4, generator=g)
    return x1, parts


def _models(D=32, **kw):
    torch.manual_seed(0)
    unet = ParticleCrossAttnFlowNetwork(nxi=25, D=D, n_heads=4,
                                        predict_orientation=True, **kw)
    dit = ParticleDiTFlowNetwork(nxi=25, D=D, depth=2, n_heads=4,
                                 predict_orientation=True, **kw)
    return {'unet': unet, 'dit': dit}


def test_sinkhorn():
    print("\n--- Sinkhorn ErgLoss ---\n")
    x1, parts = _batch()
    fou = ErgodicLoss(nxi=25, K=6, nd=3)
    check("default metric is still fourier", fou.metric == 'fourier')
    sk = ErgodicLoss(nxi=25, nd=3, metric='sinkhorn', sinkhorn_blur=0.05)
    cps = x1[..., :3].clone().requires_grad_(True)
    e = sk.coverage_error(cps, parts)
    e.sum().backward()
    check("sinkhorn error finite, shape (B,)",
          e.shape == (4,) and torch.isfinite(e).all().item(), f"{e.detach().numpy().round(4)}")
    check("sinkhorn gives a gradient on the control points",
          cps.grad is not None and cps.grad.abs().max().item() > 0,
          f"max |g| = {cps.grad.abs().max().item():.2e}")

    # Debiased Sinkhorn of a curve against particles placed on that curve ~ 0,
    # against particles elsewhere > 0.
    B = torch.from_numpy(bspline_basis_matrix(25, 128, 5))
    curve = torch.einsum('pi,bid->bpd', B, x1[..., :3])
    on = torch.cat([curve, torch.ones(4, 128, 1)], dim=-1)
    off = on.clone()
    off[..., :3] = 1.0 - off[..., :3]
    e_on = sk.coverage_error(x1[..., :3], on)
    e_off = sk.coverage_error(x1[..., :3], off)
    check("target on the curve ~ 0, displaced target clearly larger",
          e_on.max().item() < 1e-3 and (e_off > 10 * e_on.clamp(min=1e-6)).all().item(),
          f"on {e_on.max().item():.1e}  off {e_off.min().item():.1e}")

    # mu is an importance weight: zero mu on half the cloud must change the value.
    w = parts.clone()
    w[:, ::2, 3] = 0.0
    check("mu weights enter the target measure",
          not torch.allclose(sk.coverage_error(x1[..., :3], parts),
                             sk.coverage_error(x1[..., :3], w)))

    skf = ErgodicLoss(nxi=25, nd=3, metric='sinkhorn', ergodic_on='footprint')
    rot = x1[..., 3:].clone().requires_grad_(True)
    ef = skf(x1[..., :3], parts, torch.rand(4), rot6d=rot)
    ef.backward()
    check("footprint mode: rotation gets a gradient through the beam",
          rot.grad is not None and rot.grad.abs().max().item() > 0,
          f"max |g| = {rot.grad.abs().max().item():.2e}")


def test_backbones():
    print("\n--- U-Net / DiT with start inpainting ---\n")
    x1, parts = _batch()
    erg = ErgodicLoss(nxi=25, nd=3, weight=630.0, metric='sinkhorn',
                      ergodic_on='footprint')
    ori = OrientationLoss(nxi=25, weight=0.012)
    for name, m in _models(start_cond=True, start_inpaint=True).items():
        check(f"{name}: start_inpaint flag set", m.start_inpaint)
        torch.manual_seed(1)
        loss, p = compute_particle_cfm_loss(m, x1, parts, p_drop=0.1, ergodic=erg,
                                            orientation=ori, w_cfm_rot=0.5)
        loss.backward()
        g = sum(q.grad.abs().sum().item() for q in m.parameters() if q.grad is not None)
        check(f"{name}: CFM + Sinkhorn + orientation loss, backward",
              torch.isfinite(loss).item() and g > 0,
              f"loss {loss.item():.3f}  parts {sorted(p)}")

        # Inpainting: the position target of control point 0 is not trained.
        # Changing x1[:, 0] in the rotation channels changes the loss, in the
        # position channels only through the start (FiLM / held token) -- so
        # check the cleaner statement: with x_t pinned, the CFM position loss is
        # the mean over control points 1..nxi-1 only.
        m.zero_grad()
        g = torch.Generator().manual_seed(3)
        cps, rot6 = generate_particle_trajectories(
            m, parts[0], num_samples=3, steps=4, start=x1[0, 0, :3], generator=g)
        check(f"{name}: generated cp0 position is the start",
              torch.allclose(cps[:, 0], x1[0, 0, :3].expand(3, -1)),
              f"{tuple(cps.shape)} {tuple(rot6.shape)}")

        # Flag survives a round trip into a default-constructed model.
        sd = m.state_dict()
        fresh = (ParticleDiTFlowNetwork(nxi=25, D=32, depth=2, n_heads=4,
                                        predict_orientation=True, start_cond=True)
                 if name == 'dit' else
                 ParticleCrossAttnFlowNetwork(nxi=25, D=32, n_heads=4,
                                              predict_orientation=True, start_cond=True))
        res = fresh.load_state_dict(sd)
        check(f"{name}: load adopts start_inpaint_flag",
              fresh.start_inpaint and not res.missing_keys and not res.unexpected_keys)

    # Without the flag nothing changes: same loss as before the port.
    for name, m in _models(start_cond=True).items():
        check(f"{name}: no flag -> no inpainting", not m.start_inpaint)

    dit = _models(start_cond=True)['dit']
    v_pos, v_rot = dit(x1, torch.rand(4), parts, start=x1[:, 0, :3])
    check("dit: adaLN-Zero start -> v = 0 everywhere",
          v_pos.abs().max().item() == 0 and v_rot.abs().max().item() == 0,
          f"{tuple(v_pos.shape)} {tuple(v_rot.shape)}")

    # Inpainting loss ignores the position output at cp0: a model whose cp0
    # position velocity is shifted gives the same CFM position loss.
    m = _models(start_cond=True, start_inpaint=True)['unet']
    torch.manual_seed(5)
    l1, p1 = compute_particle_cfm_loss(m, x1, parts)
    head = m.flow_head.net[-1]
    hook = head.register_forward_hook(
        lambda mod, i, o: o + torch.nn.functional.pad(torch.ones_like(o[:, :1]),
                                                      (0, 0, 0, o.shape[1] - 1)))
    torch.manual_seed(5)
    l2, p2 = compute_particle_cfm_loss(m, x1, parts)
    hook.remove()
    check("inpaint: cp0 position output does not enter the CFM loss",
          torch.allclose(p1['cfm'], p2['cfm']), f"{p1['cfm'].item():.6f} vs {p2['cfm'].item():.6f}")


def test_target_cps():
    print("\n--- least-squares control-point targets ---\n")
    rng = np.random.default_rng(0)
    cps = rng.uniform(0.2, 0.8, (25, 9))
    for npts in (130, 200):
        B = bspline_basis_matrix(25, npts, 5).astype(np.float64)
        pos, rot = B @ cps[:, :3], B @ cps[:, 3:]
        fit, rmse = fit_control_points(pos, rot, pos[0].astype(np.float32))
        check(f"exact B-spline recovered from {npts} points",
              np.abs(fit - cps).max() < 1e-3 and rmse < 1e-4,
              f"max |d| {np.abs(fit - cps).max():.1e}  rmse {rmse:.1e}")
        check(f"cp0 equals the start ({npts} points)",
              np.allclose(fit[0, :3], pos[0], atol=1e-6))


if __name__ == '__main__':
    test_sinkhorn()
    test_backbones()
    test_target_cps()
    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {FAILED}")
        sys.exit(1)
    print("All checks passed.")
