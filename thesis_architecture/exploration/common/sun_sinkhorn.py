r"""
sun_sinkhorn.py -- Sun et al.'s flow-matching ergodic coverage with the Sinkhorn-divergence flow
================================================================================================
Reference: Sun, Pinosky, Murphey, "Flow Matching Ergodic Coverage", RSS 2025, official tutorial
MurpheyLab/lqr-flow-matching, tutorials/sinkhorn_flow_coverage.ipynb. Same LQ flow matching as
the Stein variant (`sun_refine.py`): the trajectory's time samples are the particles, a descent
direction dx on them is turned into a control update by the LQR (`lqrax`), u <- u + step * v. Only
the direction differs:

  Stein flow (sun_refine.py):     dx_i = mean_j [ k(x_j, x_i) grad log p(x_j) + grad_j k(x_j, x_i) ]
  Sinkhorn flow (this module):    dx = -SCALE * grad_x S_eps(mu_x, nu),
      mu_x = uniform measure on the trajectory samples, nu = target, S the debiased Sinkhorn
      divergence S = OT_eps(mu, nu) - 1/2 OT_eps(mu, mu) - 1/2 OT_eps(nu, nu).

Why it is needed (validation 2026-10-11, ergodic_validation/): the Stein flow cannot move time
between well-separated modes -- inside a mode grad log p does not depend on the mode's weight --
so on a 75/25 two-blob target it ended at 61/39. Optimal transport matches the total mass, which
fixes exactly that.

Official settings, kept: cost |x - y|_1 (ott `PNormP(p=1)`), eps = 0.01 (absolute), divergence
scaled by 1e3, step 0.005, PointMassLQR Q = diag(1, 1, 1e-3, 1e-3), R = 0.01 I, dt 0.05, 200 steps.
Implementation differences (ott-jax is not installed; installing it would move the JAX version
of the working environment):
  * log-domain Sinkhorn written out here; the gradient uses the envelope theorem at the fixed
    point, grad_{x_i} OT_eps = sum_j pi_ij grad_x c(x_i, y_j) -- the same quantity ott's implicit
    differentiation returns (checked against autodiff of the unrolled solver in
    test_sun_sinkhorn.py).
  * the potentials are warm-started from the previous outer iteration, N_INNER Sinkhorn updates
    per outer iteration (consecutive trajectories differ by one small step; the test checks the
    warm-started gradient against a fully converged one).
  * the target is a density GRID, represented as a weighted point cloud on a TARGET_RES^2 node
    grid (nodes below 1e-4 of the maximum weight dropped) instead of target samples.
  * a soft boundary term, the same construction as in our Stein port (`sun_refine._log_target`,
    W_BOUNDARY, margin 0.03): outside the margin each sample is pushed back with
    W_BOUNDARY_FLOW * violation. Without it the weakly controllable END of the jerk-dynamics
    trajectories escaped the square (coordinates up to 1.5 for 38 % of the candidates on the
    ring target, half of the time outside the support); inside the margin it is exactly zero.
    It did NOT fix that (nor did smaller steps 0.0025 / 0.00125, a lower jerk penalty -- unstable --
    or smoothing the direction along time): with the jerk-penalised dynamics the Sinkhorn flow lets
    the trajectory end leave the square on ring / uniform, cause not resolved. The pipeline
    therefore keeps the Stein flow for the jerk variant (run_smoothness_comparison_v2.SUN_SPEC).
    For the point mass the Sinkhorn flow passes every validation target.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

from . import sun_refine as sr

EPS = 0.01
SCALE = 1e3
STEP_SIZE = 0.005
N_INNER = 10
N_INNER_FIRST = 300
TARGET_RES = 40
BOUNDARY_MARGIN = 0.03
#: push per unit violation; at 0.025 outside the margin it equals the largest OT push (SCALE / N = 5)
W_BOUNDARY_FLOW = 200.0


def target_cloud(phi, res=TARGET_RES, rel_min=1e-4):
    """(R, R) density grid on linspace nodes, [iy, ix] -> (Y (M, 2), log_b (M,))."""
    d = np.clip(np.asarray(phi, dtype=np.float64), 0, None)
    R = d.shape[0]
    c = np.linspace(0.0, 1.0, res)
    yy, xx = np.meshgrid(c * (R - 1), c * (R - 1), indexing='ij')
    v = ndimage.map_coordinates(d, [yy, xx], order=1, mode='nearest')
    keep = v > rel_min * v.max()
    gx, gy = np.meshgrid(c, c)
    Y = np.stack([gx[keep], gy[keep]], -1)
    w = v[keep] / v[keep].sum()
    return Y.astype(np.float32), np.log(w).astype(np.float32)


def _cost(x, y):
    """L1 cost matrix (ott PNormP(p=1)). x: (N, 2), y: (M, 2) -> (N, M)."""
    return jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)


def _lse(a, axis):
    return jax.scipy.special.logsumexp(a, axis=axis)


def sinkhorn_potentials(C, log_a, log_b, f, g, n_iter, eps=EPS):
    """Log-domain Sinkhorn, n_iter alternating updates from (f, g)."""
    def body(_, fg):
        f, g = fg
        f = -eps * _lse((g[None, :] - C) / eps + log_b[None, :], axis=1)
        g = -eps * _lse((f[:, None] - C) / eps + log_a[:, None], axis=0)
        return f, g
    return jax.lax.fori_loop(0, n_iter, body, (f, g))


def symmetric_potential(C, log_a, f, n_iter, eps=EPS):
    """OT_eps(a, a): averaged symmetric updates (as ott's symmetric Sinkhorn)."""
    def body(_, f):
        return 0.5 * (f - eps * _lse((f[None, :] - C) / eps + log_a[None, :], axis=1))
    return jax.lax.fori_loop(0, n_iter, body, f)


def divergence_grad(x, Y, log_b, f, g, fxx, n_iter, eps=EPS):
    """grad_x S_eps(mu_x, nu) for the uniform measure on x (N, 2). Returns (grad (N, 2), f, g, fxx)."""
    N = x.shape[0]
    log_a = jnp.full((N,), -jnp.log(N))
    Cxy = _cost(x, Y)
    f, g = sinkhorn_potentials(Cxy, log_a, log_b, f, g, n_iter, eps)
    pxy = jnp.exp((f[:, None] + g[None, :] - Cxy) / eps + log_a[:, None] + log_b[None, :])
    gxy = jnp.einsum('ij,ijd->id', pxy, jnp.sign(x[:, None, :] - Y[None, :, :]))
    Cxx = _cost(x, x)
    fxx = symmetric_potential(Cxx, log_a, fxx, n_iter, eps)
    pxx = jnp.exp((fxx[:, None] + fxx[None, :] - Cxx) / eps + log_a[:, None] + log_a[None, :])
    gxx = jnp.einsum('ij,ijd->id', pxx, jnp.sign(x[:, None, :] - x[None, :, :]))
    return gxy - gxx, f, g, fxx


def boundary_push(x, margin=BOUNDARY_MARGIN, w=W_BOUNDARY_FLOW):
    """Inward push for samples outside [margin, 1 - margin]^2 (works for JAX and torch arrays)."""
    relu = lambda a: 0.5 * (a + abs(a))
    return w * (relu(margin - x) - relu(x - (1.0 - margin)))


def _make_iteration(pm):
    def iteration(carry, x0, Y, log_b, n_inner, step):
        u, f, g, fxx = carry
        x_traj, A, B = pm.linearize_dyn(x0, u)
        grad, f, g, fxx = divergence_grad(x_traj[:, :2], Y, log_b, f, g, fxx, n_inner)
        dx = jnp.zeros_like(x_traj).at[:, :2].set(-SCALE * grad + boundary_push(x_traj[:, :2]))
        v, _ = pm.solve(jnp.zeros(x_traj.shape[1]), A, B, dx)
        return (u + step * v, f, g, fxx)
    return iteration


_IT = {'pointmass': _make_iteration(sr._PM)}
_POS = {'pointmass': sr._positions}
if hasattr(sr, '_PM2'):
    _IT['jerk'] = _make_iteration(sr._PM2)
    _POS['jerk'] = sr._positions2


@partial(jax.jit, static_argnums=(4, 5, 6))
def _run(u0, x0, Y, log_b, n_iters, dynamics, record, fit, step):
    it = jax.vmap(_IT[dynamics], in_axes=((0, 0, 0, 0), 0, None, None, None, None))
    pos = jax.vmap(_POS[dynamics])
    C, N = u0.shape[0], u0.shape[1]
    M = Y.shape[0]
    carry = (u0, jnp.zeros((C, N)), jnp.zeros((C, M)), jnp.zeros((C, N)))
    carry = it(carry, x0, Y, log_b, N_INNER_FIRST, 0.0)        # converge the potentials once, no move

    def body(carry, _):
        carry = it(carry, x0, Y, log_b, N_INNER, step)
        out = jnp.einsum('ij,cjd->cid', fit, pos(carry[0], x0)) if record else 0.0
        return carry, out

    carry, log = jax.lax.scan(body, carry, None, length=n_iters)
    return carry[0], log


def divergence_grad_torch(x, Y, log_b, f, g, fxx, n_iter, eps=EPS):
    """Batched torch twin of `divergence_grad` (same formulas) for the GPU:
    x (C, N, 2), Y (M, 2), log_b (M,), f/fxx (C, N), g (C, M). JAX has no CUDA in this
    environment, and the (C, N, M) Sinkhorn kernels dominate the run time on the CPU."""
    import torch
    C, N, _ = x.shape
    log_a = torch.full((C, N), -float(np.log(N)), dtype=x.dtype, device=x.device)
    Cxy = torch.cdist(x, Y.expand(C, -1, -1), p=1)
    Cxx = torch.cdist(x, x, p=1)
    lb = log_b.view(1, 1, -1)
    for _ in range(n_iter):
        f = -eps * torch.logsumexp((g[:, None, :] - Cxy) / eps + lb, dim=2)
        g = -eps * torch.logsumexp((f[:, :, None] - Cxy) / eps + log_a[:, :, None], dim=1)
        fxx = 0.5 * (fxx - eps * torch.logsumexp((fxx[:, None, :] - Cxx) / eps + log_a[:, None, :], dim=2))
    pxy = torch.exp((f[:, :, None] + g[:, None, :] - Cxy) / eps + log_a[:, :, None] + lb)
    pxx = torch.exp((fxx[:, :, None] + fxx[:, None, :] - Cxx) / eps + log_a[:, :, None] + log_a[:, None, :])
    # sum_j p_ij sign(x_i - y_j), per coordinate, without materialising (C, N, M, 2)
    gxy = torch.stack([(pxy * torch.sign(x[:, :, None, d] - Y[None, None, :, d])).sum(-1) for d in range(2)], -1)
    gxx = torch.stack([(pxx * torch.sign(x[:, :, None, d] - x[:, None, :, d])).sum(-1) for d in range(2)], -1)
    return gxy - gxx, f, g, fxx


_LQR_FNS = {}


def _lqr_fns(dynamics, pm=None):
    """jit-compiled (linearised positions, LQ flow-matching update, positions) for the dynamics;
    `pm` overrides the LQR instance (e.g. other Q / R for a sweep)."""
    key = (dynamics, id(pm))
    if key not in _LQR_FNS:
        pm = pm if pm is not None else (sr._PM if dynamics == 'pointmass' else sr._PM2)
        lin = jax.jit(jax.vmap(lambda x0, u: pm.linearize_dyn(x0, u)[0][..., :2]))

        def _update(x0, u, dx_pos, step):
            # linearisation recomputed inside the same jit: passing (A, B) in from a separate call
            # made the vmapped Riccati solve ~27x slower (165 ms vs 6 ms per iteration, 30 cand.)
            x_traj, A, B = pm.linearize_dyn(x0, u)
            dx = jnp.zeros_like(x_traj).at[:, :2].set(dx_pos)
            return u + step * pm.solve(jnp.zeros(x_traj.shape[1]), A, B, dx)[0]

        upd = jax.jit(jax.vmap(_update, in_axes=(0, 0, 0, None)))
        _LQR_FNS[key] = (lin, upd, jax.jit(jax.vmap(_POS[dynamics])))
    return _LQR_FNS[key]


def run_batch(curves, phi, n_iters, nxi, record=False, log_space='raw', dynamics='pointmass', step=STEP_SIZE,
              device=None, lqr=None):
    """Same contract as `sun_refine.run_batch` (one shared target grid `phi` (R, R)):
    -> dict(final_cps, final_pos (C, TSTEPS+1, 2), init_cps, log (C, n_iters, nxi, 2) or None).
    `device`: torch device for the Sinkhorn part (default: cuda if available). With
    device='jax' everything runs in the jit-compiled JAX scan (`_run`, the tested reference).
    `lqr`: optional LQR instance overriding the default Q / R of `dynamics` (torch path only)."""
    import torch
    curves = np.asarray(curves, dtype=np.float64)
    C, T = curves.shape[:2]
    u0, x0 = sr.prepare(curves, None, dynamics=dynamics)
    Y, log_b = target_cloud(phi)
    fit = sr._fit_matrix(nxi)
    log_fit = fit if log_space == 'cps' else sr._linear_resample_matrix(nxi, sr.TSTEPS + 1)
    if log_space == 'raw':
        init_cps = sr._resample_all(curves, None, nxi)
    else:
        B_T = sr._basis(nxi, T)
        init_cps = np.linalg.lstsq(B_T, curves.transpose(1, 0, 2).reshape(T, C * 2),
                                   rcond=None)[0].reshape(nxi, C, 2).transpose(1, 0, 2)
    f32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.float32))
    if device == 'jax':
        u, log = _run(f32(u0), f32(x0), f32(Y), f32(log_b), int(n_iters), dynamics, bool(record),
                      f32(log_fit), jnp.float32(step))
        log = np.asarray(log).transpose(1, 0, 2, 3) if record else None
    else:
        dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        lin, upd, posf = _lqr_fns(dynamics, lqr)
        Yt = torch.as_tensor(Y, device=dev)
        lbt = torch.as_tensor(log_b, device=dev)
        u, x0j = f32(u0), f32(x0)
        N = u0.shape[1]
        f = torch.zeros((C, N), device=dev)
        g = torch.zeros((C, len(Y)), device=dev)
        fxx = torch.zeros((C, N), device=dev)
        logs = [] if record else None
        for it in range(n_iters + 1):
            xt = torch.as_tensor(np.array(lin(x0j, u)), device=dev)
            grad, f, g, fxx = divergence_grad_torch(xt, Yt, lbt, f, g, fxx, N_INNER_FIRST if it == 0 else N_INNER)
            if it == 0:                                    # converge the potentials once, no move
                continue
            u = upd(x0j, u, jnp.asarray((-SCALE * grad + boundary_push(xt)).cpu().numpy()), jnp.float32(step))
            if record:
                logs.append(np.einsum('ij,cjd->cid', log_fit, np.asarray(posf(u, x0j))))
        log = np.stack(logs, axis=1) if record else None
    pos = np.asarray(jax.vmap(_POS[dynamics])(u, f32(x0)), dtype=np.float64)
    final_cps = np.einsum('ij,cjd->cid', fit, pos)
    return dict(final_cps=final_cps, final_pos=pos, init_cps=init_cps, log=log)
