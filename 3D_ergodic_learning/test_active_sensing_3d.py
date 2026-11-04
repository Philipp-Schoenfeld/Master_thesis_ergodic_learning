#!/usr/bin/env python3
"""CPU checks of the building blocks of active_sensing_3d.py.

    python test_active_sensing_3d.py
"""
import sys

import numpy as np
import torch

from active_sensing_3d import (GPBelief, fibonacci_sphere, kappa_schedule, linear_candidates,
                               reflect, rot6_from_axis, ucb_density)
from orientation import rot6d_to_matrix

FAILED = []


def check(name, ok, info=''):
    print(f"[{'ok' if ok else 'FAIL'}]   {name:<58} {info}")
    if not ok:
        FAILED.append(name)


rng = np.random.default_rng(0)
xyz = rng.uniform(0.1, 0.9, (200, 3)).astype(np.float32)
truth = np.exp(-((xyz - 0.5) ** 2).sum(-1) / 0.05).astype(np.float32)

gp = GPBelief(xyz)
mu, sd = gp.posterior()
check("prior: mean 0, sd = signal", np.allclose(mu, 0) and np.allclose(sd, 1))
idx = np.arange(0, 200, 4)
gp.observe(idx, truth[idx])
mu, sd = gp.posterior()
check("posterior sd shrinks at observed particles", sd[idx].max() < 0.2 and sd.min() >= 0, f"{sd[idx].max():.3f}")
check("posterior mean close to truth at observed particles", np.abs(mu[idx] - truth[idx]).max() < 0.1)
check("sd smaller than prior everywhere", (sd < 1.0).all())

check("kappa schedule endpoints", kappa_schedule(0, 5, 1.0, 0.05) == 1.0
      and abs(kappa_schedule(4, 5, 1.0, 0.05) - 0.05) < 1e-9)
phi = ucb_density(mu, sd, np.zeros(200, np.float32), 0.5)
check("ucb density peak 1, positive", abs(phi.max() - 1) < 1e-6 and phi.min() > 0)
check("visited particles are discounted", ucb_density(mu, sd, np.ones(200, np.float32), 0.5).min() >= 0.99e-4
      and ucb_density(mu, sd, np.r_[np.ones(100), np.zeros(100)].astype(np.float32), 0.5)[:100].max() < 0.01)

z = torch.nn.functional.normalize(torch.randn(50, 3), dim=-1)
R = rot6d_to_matrix(rot6_from_axis(z))
check("rot6_from_axis: column 2 is the axis, proper rotation",
      torch.allclose(R[..., 2], z, atol=1e-5) and torch.allclose(torch.det(R), torch.ones(50), atol=1e-4))
zz = torch.tensor([[0., 0., 1.], [0., 0., -1.]])
check("rot6_from_axis: axis along z handled",
      torch.allclose(rot6d_to_matrix(rot6_from_axis(zz))[..., 2], zz, atol=1e-5))

x = reflect(rng.uniform(-3, 4, 1000))
check("reflect stays in the box", x.min() >= 0.02 - 1e-9 and x.max() <= 0.98 + 1e-9)
check("fibonacci directions are unit", np.allclose(np.linalg.norm(fibonacci_sphere(30), axis=-1), 1))
start = np.array([0.5, 0.5, 0.5], np.float32)
c = linear_candidates(start, xyz, 30, 6.5, rng)
check("linear candidates: shape, start held, in box",
      tuple(c.shape) == (30, 25, 9) and torch.allclose(c[:, 0, :3], torch.from_numpy(start).expand(30, 3))
      and c[..., :3].min() >= 0.02 - 1e-6 and c[..., :3].max() <= 0.98 + 1e-6)
d = (c[:, 1:, :3] - c[:, :-1, :3]).norm(dim=-1).sum(-1)
check("linear candidates have different headings", (c[:, 3, :3] - c[:, 0, :3]).norm(dim=-1).std() > 0.01, f"len~{d.mean():.1f}")

print()
if FAILED:
    print(f"{len(FAILED)} check(s) FAILED: {FAILED}")
    sys.exit(1)
print("All checks passed.")
