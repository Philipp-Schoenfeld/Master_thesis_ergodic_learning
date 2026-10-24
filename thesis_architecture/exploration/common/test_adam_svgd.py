r"""
test_adam_svgd.py -- correctness checks for adam_svgd.py
=========================================================
 1. The SVGD direction (Liu & Wang, median heuristic) samples a known 2D Gaussian: 300 particles
    started far away end with the target mean and covariance (the standard SVGD sanity check of
    the DartML reference code, there on a 1D GMM).
 2. The kernel repulsion term equals the autograd gradient sum_j grad_{x_j} k(x_j, x_i).
 3. grad_energy equals autograd of the same energy (TsvecSvgd residual) for B-spline and raw.
 4. On a toy ergodic target the ergodic error falls by > 80 % and the swarm does not collapse
    (pairwise distance between candidates stays > 0).
Run: python test_adam_svgd.py   (exit 0 = all passed, CPU, < 1 min)
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_expl = os.path.dirname(_here)
_arch = os.path.dirname(_expl)
for _p in (_expl, _arch):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from common.adam_svgd import AdamSvgd
from common.tsvec_svgd import _residual


def test_gaussian_sampling():
    torch.manual_seed(0)
    mu = torch.tensor([1.0, -2.0], dtype=torch.float64)
    S = torch.tensor([[1.0, 0.6], [0.6, 0.8]], dtype=torch.float64)
    Si = torch.linalg.inv(S)
    x = torch.randn(300, 2, dtype=torch.float64) * 0.3 + torch.tensor([-4.0, 4.0], dtype=torch.float64)
    for _ in range(3000):
        score = -(x - mu) @ Si
        x = x + 0.05 * AdamSvgd.svgd_direction(x, score)
    m, C = x.mean(0), torch.cov(x.T)
    assert torch.allclose(m, mu, atol=0.1), m
    assert torch.allclose(C, S, atol=0.15), C
    print(f"[test] SVGD samples N(mu, S): mean {m.numpy().round(3)}, cov {C.numpy().round(3).tolist()}, OK")


def test_repulsion_matches_autograd():
    torch.manual_seed(1)
    P = torch.rand(7, 5, dtype=torch.float64)
    zero = torch.zeros_like(P)
    rep = AdamSvgd.svgd_direction(P, zero) * P.shape[0]
    sq = torch.cdist(P, P) ** 2
    iu = torch.triu_indices(7, 7, 1)
    h = (torch.median(sq[iu[0], iu[1]]) / np.log(8.0)).item()
    Pj = P.clone().requires_grad_(True)
    tot = torch.zeros_like(P)
    for i in range(7):
        k = torch.exp(-((Pj - P[i].detach()) ** 2).sum(-1) / h).sum()   # sum_j k(x_j, x_i)
        g, = torch.autograd.grad(k, Pj)
        tot[i] = g.sum(0)                                                 # sum over j of grad wrt x_j
    assert torch.allclose(rep, tot, atol=1e-10), (rep - tot).abs().max()
    print(f"[test] repulsion term == autograd sum_j grad_xj k(x_j, x_i): max|diff| {(rep - tot).abs().max():.1e}, OK")


def test_energy_gradient():
    from ergodic_energy_torch import target_coeffs_from_grid
    torch.manual_seed(2)
    xs = torch.linspace(0, 1, 48)
    gy, gx = torch.meshgrid(xs, xs, indexing='ij')
    dens = torch.exp(-((gx - 0.4) ** 2 + (gy - 0.6) ** 2) / 0.02)
    B = torch.linalg.qr(torch.rand(40, 12, dtype=torch.float64))[0] * 0.5
    for name, Bm, D in (('B-spline-like', B, 12), ('raw waypoints', None, 40)):
        s = AdamSvgd(B=Bm, K=6, dtype=torch.float64, beta=3.0)
        phi = target_coeffs_from_grid(dens.double(), s.k_idx)
        P = (torch.rand(3, D, 2, dtype=torch.float64) * 0.6 + 0.2).requires_grad_(True)
        V, g = s.grad_energy(P.detach(), phi)
        e = s.energy
        tot = sum(0.5 * (_residual(s.render(P[i:i + 1])[0], e.k_idx, e.Lambda, phi, e.w_smooth, e.w_ergodic,
                                   e.w_boundary) ** 2).sum() for i in range(3))
        ga, = torch.autograd.grad(tot, P)
        err = (g - ga.reshape(3, -1)).abs().max() / ga.abs().max()
        assert err < 1e-8, (name, err)
        print(f"[test] grad_energy == autograd ({name}): rel err {err:.1e}, OK")


def test_ergodic_descent_no_collapse():
    from ergodic_energy_torch import ergodic_term, target_coeffs_from_grid
    xs = torch.linspace(0, 1, 48)
    gy, gx = torch.meshgrid(xs, xs, indexing='ij')
    dens = torch.exp(-((gx - 0.35) ** 2 + (gy - 0.6) ** 2) / 0.01) + 0.6 * torch.exp(-((gx - 0.7) ** 2 + (gy - 0.3) ** 2) / 0.008)
    N, T = 12, 48
    t = torch.linspace(0, 1, T).view(1, T, 1)
    ang = torch.linspace(0, np.pi, N + 1)[:-1].view(N, 1, 1)
    P0 = 0.5 + (t - 0.5) * 0.9 * torch.cat([torch.cos(ang), torch.sin(ang)], -1)
    s = AdamSvgd(B=None, K=8, w_smooth=15.0, lr=5e-3)
    phi = target_coeffs_from_grid(dens, s.k_idx)
    e0 = ergodic_term(P0, s.k_idx, s.energy.Lambda, phi).mean().item()
    out = s.run(P0, phi, 600, record=False)
    e1 = ergodic_term(out['final'], s.k_idx, s.energy.Lambda, phi).mean().item()
    spread = torch.cdist(out['final'].reshape(N, -1), out['final'].reshape(N, -1)).mean().item()
    assert e1 < 0.2 * e0, (e0, e1)
    assert spread > 0.1, spread
    print(f"[test] ergodic error {e0:.2f} -> {e1:.3f}, mean pairwise distance {spread:.2f} (no collapse), OK")


if __name__ == '__main__':
    test_gaussian_sampling()
    test_repulsion_matches_autograd()
    test_energy_gradient()
    test_ergodic_descent_no_collapse()
    print('ALL TESTS PASSED')
