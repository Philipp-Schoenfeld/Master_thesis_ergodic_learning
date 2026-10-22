r"""
adam_svgd.py -- reference SVGD with Adam on whole trajectories (joint swarm)
=============================================================================
Re-implementation for the smoothness comparison (2026-10-11), following two published
reference implementations instead of the GUI refiner (`svgd_batched.BatchedSvgdTorch`, kept
unchanged for the GUI):

* Liu & Wang, "Stein Variational Gradient Descent", NeurIPS 2016, reference code
  DartML/Stein-Variational-Gradient-Descent (python/svgd.py):
    phi*(x_i) = 1/N sum_j [ k(x_j, x_i) grad log p(x_j) + grad_{x_j} k(x_j, x_i) ],
    RBF kernel k = exp(-||x - x'||^2 / h) on the full parameter vector,
    median heuristic h = med(||x_i - x_j||^2) / log(N + 1),
    per-coordinate adaptive step (there AdaGrad with momentum; here Adam, Kingma & Ba 2015,
    with bias correction -- same normalisation family).
* Lee, Lerch, Ramos, Abraham, "Stein Variational Ergodic Search", RSS 2024: SVGD over a JOINT
  set of trajectories, target p(x) ~ exp(-lambda * L(x)) with L = ergodic metric + smoothness
  + boundary penalties, whole-trajectory RBF kernel with the median heuristic.

Differences to the GUI refiner it replaces in the pipeline: one joint swarm of all candidates
(instead of 8 jittered copies per candidate with best-of selection, which logged a different
particle every iteration), no jitter, the analytic energy/Jacobian of `tsvec_svgd.TsvecSvgd`
(finite-difference checked in test_tsvec_svgd.py), a temperature `beta` and a selectable number
of Fourier modes K.

The energy is TsvecSvgd's V = 0.5 ||r||^2 (ergodic + smoothness + boundary, weights x beta);
grad V = J^T r. The parameterisation is the same as there: B-spline control points (X = B P) or
raw waypoints (B = None).
"""
import numpy as np
import torch

from .tsvec_svgd import PARAM_CLIP, TsvecSvgd

ADAM_LR = 2e-3
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


class AdamSvgd:
    def __init__(self, B, K=10, device='cpu', dtype=torch.float32, w_smooth=None, beta=1.0,
                 lr=ADAM_LR, use_kernel=True):
        kw = {} if w_smooth is None else dict(w_smooth=w_smooth)
        self.energy = TsvecSvgd(B=B, K=K, device=device, dtype=dtype, beta=beta, **kw)
        self.device, self.dtype = self.energy.device, dtype
        self.lr = lr
        self.use_kernel = use_kernel
        self.k_idx = self.energy.k_idx

    def render(self, P):
        return self.energy.render(P)

    def grad_energy(self, P, phi_k):
        """P: (N, D_in, 2) -> (V (N,), grad V (N, D_in*2)) via the analytic Jacobian."""
        N, D_in = P.shape[0], P.shape[1]
        X = self.energy.render(P)
        r, dr_dX = self.energy._residual_and_dX(X, phi_k)
        BK = self.energy._dX_dP(X.shape[1], D_in)
        J = torch.matmul(dr_dX, BK.unsqueeze(0))
        g = torch.bmm(J.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)
        return 0.5 * (r ** 2).sum(-1), g

    @staticmethod
    def svgd_direction(Pf, score, use_kernel=True):
        """Liu & Wang's phi* with the median-heuristic RBF kernel. Pf, score: (N, D)."""
        N = Pf.shape[0]
        if not use_kernel or N == 1:
            return score
        sq = torch.cdist(Pf, Pf) ** 2
        iu = torch.triu_indices(N, N, 1, device=Pf.device)
        med = torch.median(sq[iu[0], iu[1]])
        h = torch.clamp(med / np.log(N + 1.0), min=1e-8)
        Kx = torch.exp(-sq / h)
        # sum_j grad_{x_j} k(x_j, x_i) = 2/h * sum_j k_ij (x_i - x_j)
        rep = (2.0 / h) * (Kx.sum(1, keepdim=True) * Pf - Kx @ Pf)
        return (Kx @ score + rep) / N

    def run(self, init_P, phi_k, n_iters, record=True):
        """init_P: (N, D_in, 2) -> dict(final, log (n_iters+1, N, D_in, 2) or None, energy)."""
        P = torch.as_tensor(np.asarray(init_P) if not torch.is_tensor(init_P) else init_P,
                            dtype=self.dtype).to(self.device).clone()
        phi_k = (phi_k if torch.is_tensor(phi_k) else torch.as_tensor(np.asarray(phi_k))).to(self.device, self.dtype)
        shape = P.shape
        Pf = P.reshape(shape[0], -1)
        m = torch.zeros_like(Pf)
        v = torch.zeros_like(Pf)
        log = None
        if record:
            log = torch.empty((n_iters + 1,) + tuple(shape), dtype=self.dtype, device=self.device)
            log[0] = P
        for it in range(n_iters):
            _, g = self.grad_energy(Pf.reshape(shape), phi_k)
            d = self.svgd_direction(Pf, -g, self.use_kernel)
            t = it + 1
            m = ADAM_B1 * m + (1 - ADAM_B1) * d
            v = ADAM_B2 * v + (1 - ADAM_B2) * d * d
            step = self.lr * (m / (1 - ADAM_B1 ** t)) / (torch.sqrt(v / (1 - ADAM_B2 ** t)) + ADAM_EPS)
            Pf = (Pf + step).clamp(*PARAM_CLIP)
            if record:
                log[t] = Pf.reshape(shape)
        V, _ = self.grad_energy(Pf.reshape(shape), phi_k)
        return dict(final=Pf.reshape(shape), log=log, energy=V)
