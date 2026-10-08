r"""
svgd_batched.py
===============
Vectorised re-implementation of

    SvgdRefiner(seed).refine(curve, phi, n_iters, nxi=25, start=start,
                             trajectory_log=log)

for C candidates at once (every candidate keeps its own N-particle swarm and
its own RNG stream, so candidate c reproduces a fresh `SvgdRefiner(seeds[c])`
call). The mathematics is unchanged -- same energy (W_ERGODIC / W_SMOOTH /
W_BOUNDARY / W_START terms), same median-heuristic RBF kernel, same Adam
update, same update clamp at 200 and the same clip of the control points to
[0.02, 0.98]; all constants are read from `SvgdRefiner`, so they cannot drift.

Why a second implementation: the reference loops over particles in Python and
evaluates the generic dim-agnostic Fourier basis (64 modes x 2 coordinates of
cos/sin per point). The basis is separable,

    F_(k1,k2)(x, y) = cos(pi k1 x) cos(pi k2 y),

so the coefficients are one small matrix product of the two 1-D cosine tables,
and the 1-D tables for k = 0..K-1 come from the angle-addition recurrence
(2 transcendental calls per coordinate instead of 2*K). That is what makes
~260 000 SVGD runs of 1500 iterations affordable. `test_mission_eval.py`
checks the result against the reference refiner (max |diff| of the logged
control points, with and without the start pin).

Only NumPy is needed in the worker processes.
"""

import os
import sys
import zlib

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from common.svgd_refine import SvgdRefiner                     # noqa: E402

#: Solver constants, read from the reference implementation.
K_FREQ = SvgdRefiner.K
N_PARTICLES = SvgdRefiner.N_PARTICLES
JITTER = SvgdRefiner.JITTER
W_ERGODIC = SvgdRefiner.W_ERGODIC
W_SMOOTH = SvgdRefiner.W_SMOOTH
W_BOUNDARY = SvgdRefiner.W_BOUNDARY
W_START = SvgdRefiner.W_START
ADAM_LR, ADAM_B1, ADAM_B2, ADAM_EPS = 2e-3, 0.9, 0.999, 1e-8
BOUNDARY_MARGIN = 0.03
CP_CLIP = (0.02, 0.98)
UPDATE_CLAMP = 200.0


def cos_sin_multiples(theta, K):
    """cos(k theta), sin(k theta) for k = 0..K-1 via angle addition.

    theta: (...,) -> two arrays (K, ...) (frequency axis first, so every
    slice is contiguous). Two transcendental calls in total; the recurrence is
    accurate to ~1e-15 for K <= 10.
    """
    c1, s1 = np.cos(theta), np.sin(theta)
    cs = np.empty((K,) + theta.shape, dtype=theta.dtype)
    sn = np.empty_like(cs)
    cs[0] = 1.0
    sn[0] = 0.0
    if K > 1:
        cs[1] = c1
        sn[1] = s1
    for k in range(1, K - 1):
        cs[k + 1] = c1 * cs[k] - s1 * sn[k]
        sn[k + 1] = s1 * cs[k] + c1 * sn[k]
    return cs, sn


class BatchedSvgd:
    """Batched SVGD refinement on B-spline control points.

    Args:
        B: (T, nxi) clamped B-spline basis (cast to float64 exactly like the
           reference does when it mixes the float32 basis with float64 data).
    """

    def __init__(self, B):
        self.B = np.asarray(B, dtype=np.float32).astype(np.float64)
        self.T, self.nxi = self.B.shape
        K = K_FREQ
        k_idx = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)], dtype=np.float64)
        self.Lambda = (1.0 + (k_idx ** 2).sum(axis=1)) ** (-1.5)
        self.kvec = np.arange(K, dtype=np.float64)
        self.iu = np.triu_indices(N_PARTICLES, k=1)

    # -- energy and gradient w.r.t. the control points -----------------------
    def energy_grad(self, P, phik, start, want_grad=True):
        """P: (C, N, nxi, 2); phik: (C, K*K); start: (C, 2) or None.
        -> energy (C, N), grad (C, N, nxi, 2) (None if not requested)."""
        C, N, nxi, _ = P.shape
        T, K = self.T, K_FREQ
        CN = C * N
        Pm = P.reshape(CN, nxi, 2).transpose(1, 0, 2).reshape(nxi, CN * 2)
        X = (self.B @ Pm).reshape(T, CN, 2).transpose(1, 0, 2)          # (CN, T, 2)

        kv = self.kvec[:, None, None]
        cx, sx = cos_sin_multiples(np.pi * X[..., 0], K)               # (K, CN, T)
        cy, sy = cos_sin_multiples(np.pi * X[..., 1], K)
        cx = np.ascontiguousarray(cx.transpose(1, 0, 2))                # (CN, K, T)
        cy = np.ascontiguousarray(cy.transpose(1, 0, 2))
        c = np.matmul(cx, cy.transpose(0, 2, 1)) / T                    # (CN, K, K)
        diff = c.reshape(C, N, K * K) - phik[:, None, :]
        erg = 0.5 * (self.Lambda * diff ** 2).sum(-1)                   # (C, N)

        accel = X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2]
        sm = (accel ** 2).sum(axis=(1, 2)).reshape(C, N)
        lo = np.minimum(X - (0.0 + BOUNDARY_MARGIN), 0.0)
        hi = np.maximum(X - (1.0 - BOUNDARY_MARGIN), 0.0)
        bd = 0.5 * ((lo ** 2).sum(axis=(1, 2)) + (hi ** 2).sum(axis=(1, 2))).reshape(C, N)
        energy = W_ERGODIC * erg + W_SMOOTH * sm + W_BOUNDARY * bd
        d0 = None
        if start is not None:
            d0 = X[:, 0, :] - np.repeat(start, N, axis=0)               # (CN, 2)
            energy = energy + W_START * 0.5 * (d0 ** 2).sum(-1).reshape(C, N)
        if not want_grad:
            return energy, None

        G = (self.Lambda * diff).reshape(CN, K, K)
        skx = np.ascontiguousarray((sx * kv).transpose(1, 0, 2))        # k * sin(k pi x)
        sky = np.ascontiguousarray((sy * kv).transpose(1, 0, 2))
        A = np.matmul(G, cy)                                            # (CN, K, T)
        A2 = np.matmul(G.transpose(0, 2, 1), cx)
        gex = -(np.pi / T) * np.einsum('nkt,nkt->nt', skx, A)           # (CN, T)
        gey = -(np.pi / T) * np.einsum('nkt,nkt->nt', sky, A2)
        gX = W_ERGODIC * np.stack([gex, gey], axis=-1)
        gs = np.zeros_like(X)
        gs[:, :-2] += 2.0 * accel
        gs[:, 1:-1] -= 4.0 * accel
        gs[:, 2:] += 2.0 * accel
        gX += W_SMOOTH * gs + W_BOUNDARY * (lo + hi)
        if d0 is not None:
            gX[:, 0, :] += W_START * d0
        gXm = gX.transpose(1, 0, 2).reshape(T, CN * 2)
        gP = (self.B.T @ gXm).reshape(nxi, CN, 2).transpose(1, 0, 2)
        return energy, gP.reshape(C, N, nxi, 2)

    # -- one SVGD update direction -------------------------------------------
    def _svgd_delta(self, Pf, scores):
        """Pf, scores: (C, N, D) -> update (C, N, D); mirrors `svgd_step_numpy`."""
        C, N, D = Pf.shape
        diff = Pf[:, :, None, :] - Pf[:, None, :, :]
        sq = (diff ** 2).sum(-1)                                        # (C, N, N)
        tri = sq[:, self.iu[0], self.iu[1]]                             # (C, N(N-1)/2)
        # median over the positive entries of the full matrix == median over
        # the upper triangle (every pair appears twice)
        if (tri > 0).all():
            med = np.median(tri, axis=1)
        else:                                                           # degenerate: replicate the reference
            med = np.empty(C)
            for i in range(C):
                pos = sq[i][sq[i] > 0]
                med[i] = np.median(pos) if len(pos) > 0 else 1.0
        h = np.maximum(med / np.log(N + 1), 0.1)
        Km = np.exp(-sq / h[:, None, None])
        term1 = Km @ scores
        term2 = (Km @ Pf) - Km.sum(-1, keepdims=True) * Pf
        return (term1 + (-2.0 / h)[:, None, None] * term2) / N

    # -- the full run --------------------------------------------------------
    def run(self, init_curves, phi_k, start, seeds, n_iters, record=True):
        """Refine C candidates.

        init_curves: (C, T, 2) dense start curves; phi_k: (K*K,) or (C, K*K)
        target coefficients (K = SvgdRefiner.K); start: (2,) / (C, 2) / None;
        seeds: (C,) ints (stream of candidate c == `SvgdRefiner(seeds[c])`).

        Returns dict:
          cps        (C, n_iters+1, nxi, 2) float32 (if `record`): entry 0 = the
                     B-spline fit of the init, entry i = best particle (lowest
                     energy) after iteration i -- same convention as
                     `SvgdRefiner.refine(trajectory_log=...)`
          final_cps  (C, nxi, 2) float64, best particle of the final swarm
          final_energy (C,)
        """
        C = init_curves.shape[0]
        N, nxi, T = N_PARTICLES, self.nxi, self.T
        curves = np.asarray(init_curves, dtype=np.float64)
        rhs = curves.transpose(1, 0, 2).reshape(T, C * 2)
        P0 = np.linalg.lstsq(self.B, rhs, rcond=None)[0].reshape(nxi, C, 2).transpose(1, 0, 2)
        phik = np.asarray(phi_k, dtype=np.float64)
        if phik.ndim == 1:
            phik = np.broadcast_to(phik, (C, phik.shape[0]))
        st = None
        if start is not None:
            st = np.asarray(start, dtype=np.float64)
            st = np.broadcast_to(st, (C, 2)) if st.ndim == 1 else st

        jitter = np.stack([np.random.default_rng(int(s)).normal(scale=JITTER, size=(N, nxi, 2))
                           for s in seeds])
        P = (P0[:, None] + jitter).reshape(C, N, nxi * 2)
        m = np.zeros_like(P)
        v = np.zeros_like(P)
        ar = np.arange(C)
        log = None
        if record:
            log = np.empty((C, n_iters + 1, nxi, 2), dtype=np.float32)
            log[:, 0] = P0

        for it in range(n_iters):
            e, g = self.energy_grad(P.reshape(C, N, nxi, 2), phik, st)
            if record and it >= 1:                       # state after iteration `it`
                best = e.argmin(axis=1)
                log[:, it] = P[ar, best].reshape(C, nxi, 2)
            delta = self._svgd_delta(P, -g.reshape(C, N, nxi * 2))
            mx = np.abs(delta).max(axis=(1, 2))
            scale = np.where(mx > UPDATE_CLAMP, UPDATE_CLAMP / np.maximum(mx, 1e-300), 1.0)
            delta = delta * scale[:, None, None]
            t = it + 1
            m = ADAM_B1 * m + (1 - ADAM_B1) * delta
            v = ADAM_B2 * v + (1 - ADAM_B2) * delta ** 2
            m_hat = m / (1 - ADAM_B1 ** t)
            v_hat = v / (1 - ADAM_B2 ** t)
            P = np.clip(P + ADAM_LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS), *CP_CLIP)

        e, _ = self.energy_grad(P.reshape(C, N, nxi, 2), phik, st, want_grad=False)
        best = e.argmin(axis=1)
        final = P[ar, best].reshape(C, nxi, 2)
        if record and n_iters >= 1:
            log[:, n_iters] = final
        return dict(cps=log, final_cps=final, final_energy=e[ar, best])


class BatchedSvgdTorch:
    """The same solver as `BatchedSvgd`, written with torch ops so that all
    candidates of a planning round (shapes x candidates) run as ONE batch on the
    GPU. Same math, same float64 default; `test_mission_eval.py` compares it with
    `BatchedSvgd` and with the reference `SvgdRefiner`."""

    def __init__(self, B, device='cuda', dtype=None, compute_dtype=None):
        """`dtype`: dtype of the optimiser state (particles, Adam moments,
        kernel); `compute_dtype`: dtype of the energy / gradient evaluation
        (the memory-bound bulk of the work; consumer GPUs run float64 at
        1/32 of their float32 rate). Default: both float64 = the reference."""
        import torch
        self.torch = torch
        self.dev = torch.device(device)
        self.dt = dtype or torch.float64
        self.cdt = compute_dtype or self.dt
        self.B = torch.as_tensor(np.asarray(B, dtype=np.float32).astype(np.float64),
                                 device=self.dev, dtype=self.cdt)
        self.T, self.nxi = self.B.shape
        K = K_FREQ
        k_idx = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)], dtype=np.float64)
        self.Lambda = torch.as_tensor((1.0 + (k_idx ** 2).sum(axis=1)) ** (-1.5),
                                      device=self.dev, dtype=self.cdt)
        self.kv = torch.arange(K, device=self.dev, dtype=self.cdt).view(1, K, 1)
        iu = np.triu_indices(N_PARTICLES, k=1)
        self.iu = (torch.as_tensor(iu[0], device=self.dev), torch.as_tensor(iu[1], device=self.dev))

    def _cos_sin(self, theta):
        """(CN, T) -> cos, sin of k*theta, k = 0..K-1, each (CN, K, T)."""
        torch = self.torch
        K = K_FREQ
        c1, s1 = torch.cos(theta), torch.sin(theta)
        cs = [torch.ones_like(theta), c1]
        sn = [torch.zeros_like(theta), s1]
        for k in range(1, K - 1):
            cs.append(c1 * cs[k] - s1 * sn[k])
            sn.append(s1 * cs[k] + c1 * sn[k])
        return torch.stack(cs, dim=1), torch.stack(sn, dim=1)

    def energy_grad(self, P, phik, start, want_grad=True):
        """P: (C, N, nxi, 2); phik: (C, K*K); start: (C, 2) or None."""
        torch = self.torch
        C, N, nxi, _ = P.shape
        T, K = self.T, K_FREQ
        CN = C * N
        out_dt = P.dtype
        P = P.to(self.cdt)
        phik = phik.to(self.cdt)
        if start is not None:
            start = start.to(self.cdt)
        X = torch.matmul(self.B, P.reshape(CN, nxi, 2))                 # (CN, T, 2)
        cx, sx = self._cos_sin(np.pi * X[..., 0])                       # (CN, K, T)
        cy, sy = self._cos_sin(np.pi * X[..., 1])
        c = torch.matmul(cx, cy.transpose(1, 2)) / T                    # (CN, K, K)
        diff = c.reshape(C, N, K * K) - phik[:, None, :]
        erg = 0.5 * (self.Lambda * diff ** 2).sum(-1)
        accel = X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2]
        sm = (accel ** 2).sum(dim=(1, 2)).reshape(C, N)
        lo = torch.clamp(X - BOUNDARY_MARGIN, max=0.0)
        hi = torch.clamp(X - (1.0 - BOUNDARY_MARGIN), min=0.0)
        bd = 0.5 * ((lo ** 2).sum(dim=(1, 2)) + (hi ** 2).sum(dim=(1, 2))).reshape(C, N)
        energy = W_ERGODIC * erg + W_SMOOTH * sm + W_BOUNDARY * bd
        d0 = None
        if start is not None:
            d0 = X[:, 0, :] - start.repeat_interleave(N, dim=0)         # (CN, 2)
            energy = energy + W_START * 0.5 * (d0 ** 2).sum(-1).reshape(C, N)
        if not want_grad:
            return energy.to(out_dt), None
        G = (self.Lambda * diff).reshape(CN, K, K)
        A = torch.matmul(G, cy)                                         # (CN, K, T)
        A2 = torch.matmul(G.transpose(1, 2), cx)
        gex = -(np.pi / T) * ((sx * self.kv) * A).sum(dim=1)            # (CN, T)
        gey = -(np.pi / T) * ((sy * self.kv) * A2).sum(dim=1)
        gX = W_ERGODIC * torch.stack([gex, gey], dim=-1)
        gs = torch.zeros_like(X)
        gs[:, :-2] += 2.0 * accel
        gs[:, 1:-1] -= 4.0 * accel
        gs[:, 2:] += 2.0 * accel
        gX = gX + W_SMOOTH * gs + W_BOUNDARY * (lo + hi)
        if d0 is not None:
            gX[:, 0, :] += W_START * d0
        gP = torch.matmul(self.B.T, gX)                                 # (CN, nxi, 2)
        return energy.to(out_dt), gP.reshape(C, N, nxi, 2).to(out_dt)

    def _svgd_delta(self, Pf, scores):
        torch = self.torch
        C, N, D = Pf.shape
        sq = ((Pf[:, :, None, :] - Pf[:, None, :, :]) ** 2).sum(-1)
        tri = sq[:, self.iu[0], self.iu[1]]
        med = torch.quantile(tri, 0.5, dim=1)           # linear interpolation == numpy median
        degenerate = (tri <= 0).any(dim=1)
        if bool(degenerate.any()):                      # identical particles: replicate the reference
            for i in torch.nonzero(degenerate).flatten().tolist():
                pos = sq[i][sq[i] > 0]
                med[i] = pos.median() if pos.numel() else 1.0
        h = torch.clamp(med / np.log(N + 1), min=0.1)
        Km = torch.exp(-sq / h[:, None, None])
        term1 = torch.matmul(Km, scores)
        term2 = torch.matmul(Km, Pf) - Km.sum(-1, keepdim=True) * Pf
        return (term1 + (-2.0 / h)[:, None, None] * term2) / N

    def run(self, init_curves, phi_k, start, seeds, n_iters, record=True):
        """Same contract as `BatchedSvgd.run` (numpy in, numpy out), except that
        `record=True` returns `cps` as a float32 TORCH tensor on the device (the
        log of a whole round is large; callers move slices to the host)."""
        torch = self.torch
        C = init_curves.shape[0]
        N, nxi, T = N_PARTICLES, self.nxi, self.T
        B64 = self.B.detach().cpu().numpy()
        curves = np.asarray(init_curves, dtype=np.float64)
        rhs = curves.transpose(1, 0, 2).reshape(T, C * 2)
        P0 = np.linalg.lstsq(B64, rhs, rcond=None)[0].reshape(nxi, C, 2).transpose(1, 0, 2)
        phik = np.asarray(phi_k, dtype=np.float64)
        if phik.ndim == 1:
            phik = np.broadcast_to(phik, (C, phik.shape[0]))
        jitter = np.stack([np.random.default_rng(int(s)).normal(scale=JITTER, size=(N, nxi, 2))
                           for s in seeds])
        dev, dt = self.dev, self.dt
        phik_t = torch.as_tensor(np.ascontiguousarray(phik), device=dev, dtype=dt)
        st = None
        if start is not None:
            sa = np.asarray(start, dtype=np.float64)
            sa = np.broadcast_to(sa, (C, 2)) if sa.ndim == 1 else sa
            st = torch.as_tensor(np.ascontiguousarray(sa), device=dev, dtype=dt)
        P0_t = torch.as_tensor(P0, device=dev, dtype=dt)
        P = (P0_t[:, None] + torch.as_tensor(jitter, device=dev, dtype=dt)).reshape(C, N, nxi * 2)
        m = torch.zeros_like(P)
        v = torch.zeros_like(P)
        ar = torch.arange(C, device=dev)
        log = None
        if record:
            log = torch.empty((C, n_iters + 1, nxi, 2), dtype=torch.float32, device=dev)
            log[:, 0] = P0_t.float()
        for it in range(n_iters):
            e, g = self.energy_grad(P.reshape(C, N, nxi, 2), phik_t, st)
            if record and it >= 1:
                best = e.argmin(dim=1)
                log[:, it] = P[ar, best].reshape(C, nxi, 2).float()
            delta = self._svgd_delta(P, -g.reshape(C, N, nxi * 2))
            mx = delta.abs().amax(dim=(1, 2))
            scale = torch.where(mx > UPDATE_CLAMP, UPDATE_CLAMP / torch.clamp(mx, min=1e-300),
                                torch.ones_like(mx))
            delta = delta * scale[:, None, None]
            t = it + 1
            m = ADAM_B1 * m + (1 - ADAM_B1) * delta
            v = ADAM_B2 * v + (1 - ADAM_B2) * delta ** 2
            m_hat = m / (1 - ADAM_B1 ** t)
            v_hat = v / (1 - ADAM_B2 ** t)
            P = torch.clamp(P + ADAM_LR * m_hat / (torch.sqrt(v_hat) + ADAM_EPS), *CP_CLIP)
        e, _ = self.energy_grad(P.reshape(C, N, nxi, 2), phik_t, st, want_grad=False)
        best = e.argmin(dim=1)
        final = P[ar, best].reshape(C, nxi, 2)
        if record and n_iters >= 1:
            log[:, n_iters] = final.float()
        return dict(cps=log, final_cps=final.detach().cpu().numpy(),
                    final_energy=e[ar, best].detach().cpu().numpy())


class BatchedSunTorch:
    """Batch counterpart of `SvgdRefiner(backend='sun')`: Sun et al.'s FM-Stein
    solver (Stein variational flow + LQ flow matching, the core of the data
    generator `ergodic_solver.py`), one vmapped JAX call for all candidates via
    `common.sun_refine.run_batch` -- the single-trajectory refiner goes through
    the same function, so candidate c is exactly a lone `refine` call.

    Same `run` contract as `BatchedSvgdTorch`, except that the target is the
    density GRID phi (C, R, R) (or (R, R)) instead of Fourier coefficients, and
    `seeds` are accepted but unused (the solver is deterministic)."""

    def __init__(self, B, device='cuda'):
        import torch
        self.torch = torch
        self.dev = torch.device(device)
        self.T, self.nxi = np.asarray(B).shape

    def run(self, init_curves, phis, start, seeds, n_iters, record=True):
        from common.sun_refine import run_batch
        torch = self.torch
        curves = np.asarray(init_curves, dtype=np.float64)
        if curves.shape[1] != self.T:
            raise ValueError(f"init curves have {curves.shape[1]} points, basis expects {self.T}")
        out = run_batch(curves, phis, start, int(n_iters), self.nxi, record=record)
        log = None
        if record:
            C = curves.shape[0]
            log = torch.empty((C, n_iters + 1, self.nxi, 2), dtype=torch.float32, device=self.dev)
            log[:, 0] = torch.as_tensor(out['init_cps'], dtype=torch.float32, device=self.dev)
            if n_iters >= 1:
                log[:, 1:] = torch.as_tensor(np.array(out['log']), dtype=torch.float32, device=self.dev)
        return dict(cps=log, final_cps=out['final_cps'], final_energy=None)


# -- compact storage of the logged states: see state_codec.py (numpy-only, so the
#    worker processes do not have to import the solver stack) --------------------
from state_codec import pack_states, unpack_states, Q_LO, Q_HI      # noqa: E402,F401
