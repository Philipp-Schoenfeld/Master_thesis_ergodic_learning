"""Standalone diagnostic for the NaN found in compute_spectral_cfm_loss_dfl at
production scale (S=256, nxi=25, real bsplinax basis). Not a training run:
loads the real DB data and the real B-spline basis, then runs the ALM outer
loop for several lambda0 initialisations (zero, like alm_solver.py's own
default, vs. random at increasing scale, like an untrained LambdaHead would
actually produce) with per-iteration prints to find exactly where/why it
first turns non-finite. CPU-only, no GPU needed."""
import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from obstacles import basis_torch
import alm_solver_torch as alm

NXI = 25
S = 256
K = int(np.ceil(np.sqrt(S)))
B_SAMPLES = 4

_here = os.path.dirname(os.path.abspath(__file__))
_db = os.path.join(_here, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')
conn = sqlite3.connect(_db)
cur = conn.cursor()
cur.execute("SELECT trajectory, shape_name FROM ergodic_pairs WHERE split='train' ORDER BY id ASC LIMIT ?", (B_SAMPLES,))
rows = cur.fetchall()
conn.close()

trajs = []
for blob, name in rows:
    arr = np.frombuffer(blob, dtype=np.float32)
    xy = arr.reshape(-1, 2)
    idx = np.linspace(0, len(xy) - 1, NXI).astype(int)
    trajs.append(xy[idx].astype(np.float32))
    print(f"  loaded {name}: raw len={len(xy)}  range=({xy.min():.3f},{xy.max():.3f})")

x1 = torch.tensor(np.stack(trajs, axis=0))   # (B, NXI, 2)
print("x1 range:", x1.min().item(), x1.max().item())

k1, k2 = np.meshgrid(np.arange(K), np.arange(K), indexing='ij')
k_idx_full = np.stack([k1.ravel(), k2.ravel()], axis=-1).astype(np.float32)
k_idx = torch.tensor(k_idx_full[:S]).unsqueeze(0).expand(B_SAMPLES, -1, -1).float()

args_pi = torch.pi * x1.unsqueeze(2) * torch.tensor(k_idx_full[:S]).unsqueeze(0).unsqueeze(0)
c_k = torch.cos(args_pi).prod(dim=-1).mean(dim=1)
print("c_k range:", c_k.min().item(), c_k.max().item())

basis = basis_torch(NXI, pts=128, deg=5, device='cpu')
print("basis row sums (should be ~1):", basis.sum(dim=1).min().item(), basis.sum(dim=1).max().item())

start = x1[:, 0, :]
xi0 = alm.make_cold_start(start, NXI)
n_lambda = alm.n_lambda_for_alm(NXI)
center = torch.tensor([0.5, 0.5])


def run_outer_loop(lambda0, n_outer=5, inner_steps=8, verbose_inner=False):
    lam, mu = alm.split_lambda0(lambda0, NXI, obstacle_enabled=True)
    print(f"  initial mu range: {mu.min().item():.4g} .. {mu.max().item():.4g}  "
          f"lam range: {lam.min().item():.4g} .. {lam.max().item():.4g}")
    xi, rho = xi0, 10.0
    for outer in range(n_outer):
        if verbose_inner:
            x = xi.clone().requires_grad_(True)
            m = torch.zeros_like(x); v = torch.zeros_like(x)
            b1, b2, eps = 0.9, 0.999, 1e-8
            for t in range(inner_steps):
                L = alm._lagrangian(x, lam, mu, rho, basis, c_k, k_idx, start, center, 0.0).sum()
                g, = torch.autograd.grad(L, x, create_graph=True)
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                m_hat = m / (1 - b1 ** (t + 1))
                v_hat = v / (1 - b2 ** (t + 1))
                x = x - 2e-3 * m_hat / (torch.sqrt(v_hat) + eps)
                print(f"    outer {outer} inner {t}: g_absmax={g.abs().max().item():.4g}  "
                      f"x_absmax={x.abs().max().item():.4g}  x_nan={torch.isnan(x).any().item()}  "
                      f"L={L.item():.4g}")
                if torch.isnan(x).any():
                    print("    !!! NaN inside inner Adam loop !!!")
                    return
            xi = x
            h = alm._h_eq(xi, start)
            g_c = torch.cat([alm._g_box(xi), alm._g_obstacle(xi, center, 0.0)], dim=-1)
            lam = lam + rho * h
            mu = torch.clamp(mu + rho * g_c, min=0.0)
            rho = min(rho * 1.3, 1e5)
        else:
            xi, lam, mu, rho = alm._outer_step(xi, lam, mu, rho, basis, c_k, k_idx, start,
                                               center, 0.0, 2e-3, inner_steps, 1.3, 1e5)
        v_ = alm.violation(xi, start, center, 0.0)
        o_ = alm.objective(xi, basis, c_k, k_idx)
        print(f"  outer {outer}: rho={rho:.3f}  violation={v_.tolist()}  objective={o_.tolist()}  "
              f"xi_nan={torch.isnan(xi).any().item()}  mu_max={mu.max().item():.4g}")
        if torch.isnan(xi).any() or torch.isnan(o_).any():
            print("  !!! NaN appeared at this outer iteration !!!")
            return


for mode, scale in [('zero', 0.0), ('random', 0.1), ('random', 0.5), ('random', 1.0), ('random', 3.0)]:
    print(f"\n=== lambda0 mode={mode} scale={scale} ===")
    if mode == 'zero':
        lambda0 = torch.zeros(B_SAMPLES, n_lambda)
    else:
        gen = torch.Generator().manual_seed(1)
        lambda0 = torch.randn(B_SAMPLES, n_lambda, generator=gen) * scale
    run_outer_loop(lambda0, verbose_inner=(mode == 'random' and scale == 1.0))
