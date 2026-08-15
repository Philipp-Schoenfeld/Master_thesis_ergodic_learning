#!/usr/bin/env python3
"""
test_ergodic_energy_port.py
===========================
Blocker check: the PyTorch energy port must agree with the original NumPy
solver energy (`SE3_SVGD/tsvec_2d.py :: compute_energy_and_grad`) to rounding
precision. Training against a subtly wrong objective would be invisible for
weeks, so this runs before anything else.

The original module is imported unmodified. It pulls in `init_strategies` from
a machine-specific path that no longer exists, so that import is stubbed here
rather than editing the solver.

Usage:
  python -u test_ergodic_energy_port.py
"""

import os, sys, types
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'bsplinax-main')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Stub the missing helper so the unmodified solver module imports cleanly.
if 'init_strategies' not in sys.modules:
    _stub = types.ModuleType('init_strategies')
    _stub.get_initialization = lambda *a, **k: None
    sys.modules['init_strategies'] = _stub

import tsvec_2d as ref
import ergodic_energy_torch as port

FAILED = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    print(f"  Reference: K={ref.K}, {len(ref.k_indices)} modes, T={ref.T}, "
          f"USE_OBSTACLE={ref.USE_OBSTACLE}")
    print(f"  Weights:   erg={ref.W_ERGODIC} smooth={ref.W_SMOOTH} "
          f"bound={ref.W_BOUNDARY} obst={ref.W_OBSTACLE}\n")

    # ── Constants must have been copied, not re-tuned ────────────────────────
    check('weights match the solver',
          (port.W_ERGODIC, port.W_SMOOTH, port.W_BOUNDARY, port.W_OBSTACLE) ==
          (ref.W_ERGODIC, ref.W_SMOOTH, ref.W_BOUNDARY, ref.W_OBSTACLE))

    k_idx_np, Lambda_np = port.make_k_grid(ref.K)
    check('frequency grid matches (order and values)',
          np.array_equal(k_idx_np.astype(np.int64), ref.k_indices) and
          np.allclose(Lambda_np, ref.Lambda_k, atol=1e-6))

    k_idx = torch.tensor(k_idx_np, dtype=torch.float64)
    Lambda = torch.tensor(Lambda_np, dtype=torch.float64)

    # ── phi_k from the same target density the solver uses ───────────────────
    dens = torch.tensor(ref.Zg, dtype=torch.float64)          # (200, 200)
    phi = port.target_coeffs_from_grid(dens, k_idx)
    check('phi_k matches the solver target coefficients',
          np.allclose(phi.numpy(), ref.phi_k, atol=1e-10),
          f'max |dphi| = {np.abs(phi.numpy() - ref.phi_k).max():.3e}')

    # ── Full energy on several trajectories ──────────────────────────────────
    rng = np.random.default_rng(0)
    T = ref.T
    trajs = {
        'random walk':   np.clip(np.cumsum(rng.normal(0, .03, (T, 2)), 0) + .5, 0, 1),
        'uniform noise': rng.random((T, 2)),
        'diagonal line': np.stack([np.linspace(.1, .9, T)] * 2, -1),
        'tight circle':  np.stack([.5 + .2 * np.cos(np.linspace(0, 2 * np.pi, T)),
                                   .5 + .2 * np.sin(np.linspace(0, 2 * np.pi, T))], -1),
        'out of bounds': np.stack([np.linspace(-.2, 1.2, T),
                                   np.full(T, .5)], -1),   # triggers boundary term
    }

    for name, X in trajs.items():
        e_ref, _ = ref.compute_energy_and_grad(X.ravel().copy(), T)
        Xt = torch.tensor(X, dtype=torch.float64).unsqueeze(0)
        e_port = (port.smoothness_term(Xt)
                  + port.ergodic_term(Xt, k_idx, Lambda, phi)
                  + port.boundary_term(Xt)).item()
        check(f'energy matches: {name}', rel(e_port, e_ref) < 1e-10,
              f'{e_port:.6f} vs {e_ref:.6f}  (rel {rel(e_port, e_ref):.2e})')

    # ── Obstacle term against the solver's own branch ────────────────────────
    ref.USE_OBSTACLE = True
    X = trajs['diagonal line']
    e_ref, _ = ref.compute_energy_and_grad(X.ravel().copy(), T)
    Xt = torch.tensor(X, dtype=torch.float64).unsqueeze(0)
    e_port = (port.smoothness_term(Xt)
              + port.ergodic_term(Xt, k_idx, Lambda, phi)
              + port.boundary_term(Xt)
              + port.obstacle_term(Xt, ref.OBSTACLE_CENTER, ref.OBSTACLE_RADIUS)).item()
    check('energy matches with obstacle enabled', rel(e_port, e_ref) < 1e-10,
          f'{e_port:.4f} vs {e_ref:.4f}  (rel {rel(e_port, e_ref):.2e})')
    ref.USE_OBSTACLE = False

    # ── Gradient: autograd vs the solver's hand-derived formula ──────────────
    X = trajs['random walk']
    _, g_ref = ref.compute_energy_and_grad(X.ravel().copy(), T)
    Xt = torch.tensor(X, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    (port.smoothness_term(Xt)
     + port.ergodic_term(Xt, k_idx, Lambda, phi)
     + port.boundary_term(Xt)).backward()
    g_port = Xt.grad[0].numpy().ravel()
    err = np.abs(g_port - g_ref).max() / max(np.abs(g_ref).max(), 1e-12)
    check('autograd gradient matches the hand-derived one', err < 1e-8,
          f'rel err {err:.2e}')

    # ── Batching must not change per-sample values ───────────────────────────
    batch = torch.tensor(np.stack(list(trajs.values())), dtype=torch.float64)
    e_batch = (port.smoothness_term(batch)
               + port.ergodic_term(batch, k_idx, Lambda, phi)
               + port.boundary_term(batch))
    singles = torch.stack([
        port.smoothness_term(batch[i:i + 1])
        + port.ergodic_term(batch[i:i + 1], k_idx, Lambda, phi)
        + port.boundary_term(batch[i:i + 1])
        for i in range(batch.shape[0])]).squeeze(-1)
    check('batched energy equals per-sample energy',
          torch.allclose(e_batch, singles, atol=1e-12))

    # ── Diversity reward behaves sanely ──────────────────────────────────────
    same = torch.rand(6, 25, 2, dtype=torch.float64).mean(0, keepdim=True).repeat(6, 1, 1)
    spread = torch.rand(6, 25, 2, dtype=torch.float64)
    d_same = port.diversity_reward(same).item()
    d_spread = port.diversity_reward(spread).item()
    check('diversity reward is ~0 for collapsed candidates', abs(d_same) < 1e-9,
          f'{d_same:.2e}')
    check('diversity reward is higher for spread candidates', d_spread > d_same,
          f'{d_spread:.4f} > {d_same:.2e}')

    print()
    if FAILED:
        print(f"  {len(FAILED)} check(s) FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("  All checks passed — the Torch port reproduces the solver energy.")


if __name__ == '__main__':
    main()
