r"""
svgd_refine.py
==============
SVGD-Nachverfeinerung einer geplanten Bahn gegen eine *aktuelle* Zieldichte.

Bis hierher stand diese Klasse in `interactive_sim.py`. Sie ist von dort
hierher gezogen worden, weil ausser der GUI inzwischen auch die Batch-Suche in
`thesis_architecture/exploration_optimierung/` sie braucht — und zwar in
Arbeitsprozessen, die den Matplotlib-/MuJoCo-Rumpf der GUI nicht mitladen
duerfen. Zwei Fassungen nebeneinander waeren genau der Fall, vor dem der
Projektgrundsatz warnt: die gefahrenen Zahlen der GUI und die der Suche wuerden
auseinanderdriften, ohne dass es jemandem auffiele. `interactive_sim.py`
importiert die Klasse deshalb von hier und verhaelt sich unveraendert.

Der Inhalt ist gegenueber der GUI-Fassung unveraendert.
"""

import contextlib
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_expl = os.path.dirname(_here)
_arch = os.path.dirname(_expl)
_root = os.path.dirname(_arch)
for _p in (_arch, os.path.join(_root, 'SE3_SVGD')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import ergodic_core as ergo          # noqa: E402
import svgd_engine as svgde          # noqa: E402


class SvgdRefiner:
    """Verfeinert eine geplante Bahn mit dem bestehenden SVGD-Solver
    (SE3_SVGD/svgd_engine.py + ergodic_core.py) — Ziel ist die aktuelle
    Zieldichte Phi der jeweiligen Runde, nicht ein fest verdrahtetes Ziel wie
    in `SE3_SVGD/tsvec_2d.py`.

    Ablauf pro Aufruf: `N_PARTICLES` Kopien der vom Netz gelieferten Bahn,
    leicht verrauscht, als Partikelschwarm; SVGD zieht sie ueber `n_iters`
    Adam-Schritte auf die Fourier-Koeffizienten von Phi; zurueckgegeben wird
    der Partikel mit der niedrigsten Endenergie. `n_iters=0` ist ein reiner
    Durchreicher (kein SVGD) -- das ist die 0-Stellung des Reglers in der GUI.

    Energie- und Adam-Gewichte identisch zu `SE3_SVGD/tsvec_2d.py`
    (W_ERGODIC=600, W_SMOOTH=15, W_BOUNDARY=30); K=8 statt dessen K=10, damit
    dieselbe Fourier-Basis wie die "Ergodizitaet vs. Phi"-Live-Metrik benutzt
    wird und deren Anzeige die SVGD-Verfeinerung direkt sichtbar macht.
    """

    K = 8
    DIM = 2
    N_PARTICLES = 8
    JITTER = 0.01
    W_ERGODIC, W_SMOOTH, W_BOUNDARY = 600.0, 15.0, 30.0

    def __init__(self, seed=0):
        self.k_idx = ergo.build_fourier_indices(self.K, self.DIM)
        self.Lambda_k = ergo.compute_lambda_k(self.k_idx)
        self.rng = np.random.default_rng(seed)

    def _phi_k(self, phi_grid):
        R = phi_grid.shape[-1]
        xs = np.linspace(0, 1, R)
        Xg, Yg = np.meshgrid(xs, xs)
        grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
        grid_w = np.clip(phi_grid.ravel(), 0.0, None)
        return ergo.compute_target_fourier_coeffs(grid_pts, grid_w, self.k_idx)

    def _energy_and_grad(self, phi_k, obstacle=None, obstacle_weight=20.0):
        k_idx, Lambda_k = self.k_idx, self.Lambda_k
        W_E, W_S, W_B = self.W_ERGODIC, self.W_SMOOTH, self.W_BOUNDARY

        def fn(X_flat, T):
            X = X_flat.reshape(T, 2)
            F = ergo.fourier_basis_nd(X, k_idx)
            c_k = F.mean(axis=0)
            dF = ergo.fourier_basis_grad_nd(X, k_idx)
            diff = c_k - phi_k
            erg_cost = 0.5 * np.sum(Lambda_k * diff ** 2)
            erg_grad = np.einsum('m,tmd->td', Lambda_k * diff, dF) / T

            sm_cost = svgde.compute_smoothness_cost_numpy(X)
            sm_grad = svgde.compute_smoothness_grad_numpy(X)
            bd_cost = svgde.compute_boundary_cost_numpy(X)
            bd_grad = svgde.compute_boundary_grad_numpy(X)

            energy = W_E * erg_cost + W_S * sm_cost + W_B * bd_cost
            grad = W_E * erg_grad + W_S * sm_grad + W_B * bd_grad
            
            if obstacle is not None:
                energy += obstacle_weight * obstacle.penalty(X)
                grad += obstacle_weight * obstacle.grad_penalty(X)

            return energy, grad.ravel()
        return fn

    def refine(self, curve_np, phi_np, n_iters, nxi=None, obstacle=None, obstacle_weight=20.0):
        """curve_np: (T,2) Startbahn (vom Netz). phi_np: (R,R) Zieldichte,
        Werte in [0,1]. -> (T,2) verfeinerte Bahn."""
        if n_iters <= 0:
            return curve_np
        T = curve_np.shape[0]
        phi_k = self._phi_k(phi_np)
        energy_fn_orig = self._energy_and_grad(phi_k, obstacle=obstacle, obstacle_weight=obstacle_weight)

        if nxi is not None and nxi != T:
            from obstacles import bspline_basis_matrix
            B = bspline_basis_matrix(nxi, T, 5)
            P, _, _, _ = np.linalg.lstsq(B, curve_np, rcond=None)
            
            def energy_fn(P_flat, N_pts):
                P_mat = P_flat.reshape(nxi, 2)
                X = B @ P_mat
                energy, grad_X = energy_fn_orig(X.ravel(), T)
                grad_X_mat = grad_X.reshape(T, 2)
                grad_P_mat = B.T @ grad_X_mat
                return energy, grad_P_mat.ravel()

            jitter = self.rng.normal(scale=self.JITTER, size=(self.N_PARTICLES, nxi, 2))
            init = P[None] + jitter
            particles = init.reshape(self.N_PARTICLES, nxi * 2)
            
            with open(os.devnull, 'w') as devnull, \
                    contextlib.redirect_stderr(devnull):
                particles, _ = svgde.run_svgd_numpy(
                    particles, nxi, int(n_iters), energy_fn, dim=self.DIM,
                    label='SVGD-Verfeinerung')
                    
            particles = particles.reshape(self.N_PARTICLES, nxi, 2)
            scores = [energy_fn(p.ravel(), nxi)[0] for p in particles]
            best_P = particles[int(np.argmin(scores))]
            return B @ best_P
        else:
            jitter = self.rng.normal(scale=self.JITTER, size=(self.N_PARTICLES, T, 2))
            init = np.clip(curve_np[None] + jitter, 0.02, 0.98)
            particles = init.reshape(self.N_PARTICLES, T * 2)

            with open(os.devnull, 'w') as devnull, \
                    contextlib.redirect_stderr(devnull):
                particles, _ = svgde.run_svgd_numpy(
                    particles, T, int(n_iters), energy_fn_orig, dim=self.DIM,
                    label='SVGD-Verfeinerung')

            particles = particles.reshape(self.N_PARTICLES, T, 2)
            scores = [energy_fn_orig(p.ravel(), T)[0] for p in particles]
            return particles[int(np.argmin(scores))]
