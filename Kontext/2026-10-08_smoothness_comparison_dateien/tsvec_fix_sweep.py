"""Diagnostic sweep for tsvec_svgd.py's update rule (not project code)."""
import os
import sys
import time

_eval = '/media/philipp/storage/Dokumente/Uni/Master_thesis/thesis_architecture/evaluation_full_matrix'
_arch = os.path.dirname(_eval)
_root = os.path.dirname(_arch)
for _p in (_eval, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from common.data import load_truth
import common.tsvec_svgd as tv
from ergodic_energy_torch import target_coeffs_from_grid, ergodic_term
from init_baselines import linear_angle_path


def step(svgd, P, phi_k, tau, cfg):
    N = P.shape[0]
    D_in = P.shape[1]
    D = D_in * 2
    X = svgd.render(P)
    T = X.shape[1]
    r, dr_dX = svgd._residual_and_dX(X, phi_k)
    BK = svgd._dX_dP(T, D_in)
    J = torch.matmul(dr_dX, BK.unsqueeze(0)) * cfg['beta'] ** 0.5
    r = r * cfg['beta'] ** 0.5
    H = torch.bmm(J.transpose(1, 2), J)
    score = -torch.bmm(J.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)
    diff = X.unsqueeze(1) - X.unsqueeze(0)
    sq = (diff ** 2).sum(-1)
    w = torch.exp(-sq / tv.KERNEL_LENGTH)
    norm = T if cfg['kernel_mean'] else 1.0
    k = w.sum(-1) / norm
    dk = (-2.0 / tv.KERNEL_LENGTH) * w.unsqueeze(-1) * diff / norm
    g = torch.matmul(dk.reshape(N, N, T * 2), BK.unsqueeze(0))
    phi_star = (torch.einsum('ij,id->jd', k, score) + g.sum(0)) / N
    lhs = torch.einsum('ij,iab->jab', k ** 2, H) + torch.bmm(g.permute(1, 2, 0), g.permute(1, 0, 2))
    if cfg['lhs_avg']:
        lhs = lhs / N
    diag_mean = lhs.diagonal(dim1=1, dim2=2).mean(-1).view(N, 1, 1)
    lhs = lhs + (cfg['lm'] * diag_mean + 1e-8) * torch.eye(D)
    alpha = torch.linalg.solve(lhs, phi_star.unsqueeze(-1)).squeeze(-1)
    return (P.reshape(N, D) + tau * alpha).clamp(*tv.PARAM_CLIP).reshape(P.shape), X


def run(shape, cfg, n_iters, bspline):
    names, truths = load_truth(labels=[shape], n=1, split='val', resolution=96, device='cpu')
    curves = np.stack([linear_angle_path(180.0 * i / 30, 128).numpy() for i in range(30)]).astype(np.float32)
    if bspline:
        import sqlite3
        db = os.path.join(_eval, 'results', 'smoothness_comparison_v2_20261009', 'smoothness_comparison.db')
        blob, = sqlite3.connect(db).execute(
            "SELECT matrix FROM basis WHERE nxi=25 AND n_points=128").fetchone()
        B = np.frombuffer(blob, dtype=np.float32).reshape(128, 25).copy()
        P0 = np.linalg.lstsq(B, curves.transpose(1, 0, 2).reshape(128, 60), rcond=None)[0] \
            .reshape(25, 30, 2).transpose(1, 0, 2).astype(np.float32)
    else:
        B, P0 = None, curves
    svgd = tv.TsvecSvgd(B=B, K=10, w_smooth=cfg.get('w_smooth', 0.0),
                        w_ergodic=cfg.get('w_ergodic', tv.W_ERGODIC_PAPER),
                        w_boundary=cfg.get('w_boundary', tv.W_BOUNDARY))
    phi_k = target_coeffs_from_grid(truths[0], svgd.k_idx)
    P = torch.as_tensor(P0)
    out = []
    t0 = time.time()
    for it in range(n_iters + 1):
        X = svgd.render(P)
        if it % (n_iters // 4) == 0:
            e = ergodic_term(X, svgd.k_idx, svgd.Lambda, phi_k).mean().item()
            pl = np.linalg.norm(np.diff(X.numpy(), axis=1), axis=-1).sum(1).mean()
            a = X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
            sm = 15.0 * (a ** 2).sum(dim=(1, 2)).mean().item()
            out.append(f'{it}:E={e:.2f},L={pl:.2f},S={sm:.3f}')
        if it < n_iters:
            P, _ = step(svgd, P, phi_k, cfg['tau'], cfg)
    return '  '.join(out) + f'  ({time.time() - t0:.0f}s)'


if __name__ == '__main__':
    base = dict(kernel_mean=True, lhs_avg=True, lm=0.1, beta=1.0, tau=0.1,
                w_ergodic=600.0, w_boundary=30.0)
    configs = [
        ('project weights, w_s=0', dict(base, w_smooth=0.0)),
        ('project weights, w_s=15', dict(base, w_smooth=15.0)),
        ('project weights, w_s=1.5', dict(base, w_smooth=1.5)),
    ]
    which = sys.argv[1] if len(sys.argv) > 1 else 'raw'
    for name, cfg in configs:
        print(f'[{which}] {name}: {run("A", cfg, 400, which == "bspline")}', flush=True)
