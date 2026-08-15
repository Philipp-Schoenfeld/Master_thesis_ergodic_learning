#!/usr/bin/env python3
"""
test_obstacle_grad.py
=====================
Correctness checks for the obstacle repulsion. Run this before spending GPU
time on a guided generation run — a wrong gradient sign or a broken chain rule
is invisible in the plots until it has wasted a full sweep.

Usage:
  python -u test_obstacle_grad.py
"""

import os, sys
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from obstacles import (CircleObstacle, basis_torch, curve_repulsion_grad,
                       max_violation, polish_out_of_obstacle)

torch.manual_seed(0)
FAILED = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def main():
    obs = CircleObstacle(margin=0.01)
    print(f"  {obs}\n")

    # ── 1. grad_penalty vs. finite differences ────────────────────────────────
    P = torch.rand(200, 2, dtype=torch.float64)
    g = obs.grad_penalty(P)
    eps = 1e-6
    fd = torch.zeros_like(P)
    for k in range(2):
        Pp, Pm = P.clone(), P.clone()
        Pp[:, k] += eps
        Pm[:, k] -= eps
        # penalty() sums, so differentiate per point via the squared violation.
        fp = 0.5 * obs.violation(Pp) ** 2
        fm = 0.5 * obs.violation(Pm) ** 2
        fd[:, k] = (fp - fm) / (2 * eps)
    inside = obs.violation(P) > 1e-3          # FD is unreliable at the kink
    err = (g[inside] - fd[inside]).abs().max().item()
    scale = max(fd[inside].abs().max().item(), 1e-12)
    check('grad_penalty matches finite differences',
          err / scale < 1e-4, f'rel err {err / scale:.2e}, {int(inside.sum())} pts inside')

    # ── 2. exactly zero outside ───────────────────────────────────────────────
    outside = obs.violation(P) == 0
    check('gradient is exactly zero outside the obstacle',
          bool((g[outside] == 0).all()), f'{int(outside.sum())} pts outside')

    # ── 3. gradient points inward (descent pushes outward) ────────────────────
    d = P[inside] - torch.tensor(obs.center, dtype=P.dtype)
    radial = (g[inside] * d).sum(-1)
    check('descent direction pushes points away from the centre',
          bool((radial <= 0).all()), 'grad . (P-c) <= 0 everywhere inside')

    # ── 4. chain rule B^T g vs. autograd ──────────────────────────────────────
    nxi, pts, deg = 25, 256, 5
    B = basis_torch(nxi, pts, deg, device='cpu', dtype=torch.float64)
    cps = (torch.rand(4, nxi, 2, dtype=torch.float64) * 0.6 + 0.2).requires_grad_(True)

    E = obs.penalty(torch.einsum('pi,bid->bpd', B, cps))
    E.backward()
    auto = cps.grad.detach()
    manual = curve_repulsion_grad(cps.detach(), obs, B, normalize=False)
    err = (auto - manual).abs().max().item()
    scale = max(auto.abs().max().item(), 1e-12)
    check('B^T @ grad_penalty equals autograd through the B-spline',
          err / scale < 1e-8, f'rel err {err / scale:.2e}')

    # ── 5. normalized gradient has coordinate-sized magnitude ─────────────────
    norm_g = curve_repulsion_grad(cps.detach(), obs, B, normalize=True)
    mag = norm_g.abs().max().item()
    check('normalized gradient is on the scale of the penetration depth',
          mag <= obs.effective_radius * 1.5, f'max |g| = {mag:.4f}')

    # ── 6. polish clears the obstacle, including the degenerate case ──────────
    # A curve through the exact centre is the one configuration a radial
    # potential cannot resolve on its own; the stall-breaker must handle it.
    t = torch.linspace(0.05, 0.95, nxi, dtype=torch.float64)
    cases = {
        'diagonal exactly through centre': torch.stack([t, t], -1),
        'diagonal offset by 0.02':         torch.stack([t, t + 0.02], -1),
        'horizontal through centre':       torch.stack([t, torch.full_like(t, 0.5)], -1),
        'vertical through centre':         torch.stack([torch.full_like(t, 0.5), t], -1),
    }
    for name, line in cases.items():
        x0 = line[None].clone()
        x = polish_out_of_obstacle(x0, obs, B)
        before, after = max_violation(x0, obs, B), max_violation(x, obs, B)
        check(f'polish clears: {name}', after < 1e-6,
              f'violation {before:.4f} -> {after:.2e}')

    # Batched: all samples must clear simultaneously.
    batch = torch.stack([c for c in cases.values()], dim=0)
    xb = polish_out_of_obstacle(batch.clone(), obs, B)
    check('polish clears all four cases in one batch',
          max_violation(xb, obs, B) < 1e-6,
          f'worst {max_violation(xb, obs, B):.2e}')

    # ── 7. guidance is a no-op far from the obstacle ──────────────────────────
    far = torch.full((2, nxi, 2), 0.9, dtype=torch.float64)
    check('no force on control points far from the obstacle',
          bool((curve_repulsion_grad(far, obs, B) == 0).all()))

    print()
    if FAILED:
        print(f"  {len(FAILED)} check(s) FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("  All checks passed.")


if __name__ == '__main__':
    main()
