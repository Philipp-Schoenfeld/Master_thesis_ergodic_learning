r"""
test_tsvec_svgd.py -- correctness checks for tsvec_svgd.py
=============================================================
No reference implementation exists elsewhere in the repo for this algorithm
(it is a fresh reduction of arXiv:2603.09458 to flat 2D), so this test leans
on finite-difference gradient checks rather than comparison against existing
code.   Run: python test_tsvec_svgd.py   (exit 0 = all passed, CPU, ~1 min)

 1. The analytic Jacobian (`TsvecSvgd._residual_and_dX` chained through
    `_dX_dP`, the path `step()` actually uses) matches finite differences on
    the full P -> r pipeline, both for raw waypoints and a B-spline basis.
 2. The kernel-gradient `g` used in `step` matches finite differences.
 3. B = identity (T, T) numpy matrix behaves exactly like B = None (raw
    waypoints) -- X = I @ P = P, so both code paths must agree bit-for-bit
    (up to float rounding).
 4. A few SVGD steps monotonically-ish decrease the mean particle energy
    V(P) = 0.5||r(P)||^2 on a toy target (sanity: the preconditioned update
    direction is a DESCENT direction, not a sign error).
 5. `w_smooth=0` vs `w_smooth>0`: the smoothness-weighted run ends up with a
    lower smoothness-energy (3-point acceleration) curve on a toy problem,
    the same sanity check `sun_refine.py`'s test already applies to its own
    smoothness force.
 6. Timing: wall-clock per iteration at the real scale (N=30, T=128,
    nxi=25 B-spline / nxi=128 raw) -- printed, not asserted (informs the
    cluster time estimate, this machine has no GPU).
 7. Regression: `TsvecSvgd.run(init_P, phi_k, ...)` accepts `phi_k` already
    as a torch tensor (not just numpy) -- caught on the cluster's real GPU
    run (2026-10-09, job 164971): `target_coeffs_from_grid(..., device=
    'cuda')` returns a CUDA tensor, and `run()` used to call `np.asarray()`
    on it unconditionally, which raises for any GPU tensor. Not reproducible
    on this CPU-only machine without simulating the "caller already built a
    tensor" case directly.
"""
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(os.path.dirname(_here))
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from tsvec_svgd import TsvecSvgd, _kernel_and_grad, make_k_grid_torch  # noqa: E402

torch.manual_seed(0)


def _toy_phi_k(k_idx, Lambda, device='cpu'):
    """A random but fixed target coefficient vector, same scale as a real
    `target_coeffs_from_grid` output (bounded, smooth density)."""
    rng = np.random.default_rng(0)
    T = 64
    xs = np.linspace(0, 1, T)
    X, Y = np.meshgrid(xs, xs)
    phi = np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / 0.02) + 0.5 * np.exp(
        -((X - 0.75) ** 2 + (Y - 0.3) ** 2) / 0.01)
    phi = torch.as_tensor(phi / phi.max(), dtype=torch.float32, device=device)
    from ergodic_energy_torch import target_coeffs_from_grid
    return target_coeffs_from_grid(phi, k_idx)


def test_residual_jacobian():
    k_idx, Lambda = make_k_grid_torch(6)
    phi_k = _toy_phi_k(k_idx, Lambda)
    T = 16

    for name, B in (('raw waypoints', None), ('B-spline (random orthonormal)', None)):
        nxi_or_T = T
        if name.startswith('B-spline'):
            rng = np.random.default_rng(7)
            nxi = 9
            Q, _ = np.linalg.qr(rng.normal(size=(T, nxi)))
            B = Q.astype(np.float32)
            nxi_or_T = nxi
        svgd = TsvecSvgd(B=B, K=6)
        P = (torch.rand(1, nxi_or_T, 2, dtype=torch.float32) * 0.6 + 0.2)
        X = svgd.render(P)
        r0, dr_dX = svgd._residual_and_dX(X, phi_k)
        BK = svgd._dX_dP(T, nxi_or_T)
        J_analytic = torch.matmul(dr_dX, BK.unsqueeze(0))[0]            # (R, D)

        f = svgd._residual_fn(phi_k)
        P_flat = P.reshape(-1)
        r0_f = f(P_flat)
        assert (r0[0] - r0_f).abs().max() < 1e-5, "residual forward mismatch"
        eps = 1e-4
        J_fd = torch.zeros_like(J_analytic)
        for d in range(P_flat.shape[0]):
            pp = P_flat.clone()
            pp[d] += eps
            J_fd[:, d] = (f(pp) - r0_f) / eps
        err = (J_analytic - J_fd).abs().max().item() / J_fd.abs().max().item()
        assert err < 1e-2, f"[{name}] Jacobian mismatch: max rel |diff|={err}"
        print(f"[test] analytic Jacobian vs. finite differences ({name}): "
             f"max rel |diff|={err:.2e}, OK")


def test_kernel_gradient():
    N, T = 4, 16
    X = (torch.rand(N, T, 2, dtype=torch.float64) * 0.6 + 0.2)
    k0, dk = _kernel_and_grad(X)
    assert float(k0.max()) <= 1.0 + 1e-12, "kernel must be averaged over t (k in [0,1])"
    eps = 1e-6
    worst = 0.0
    for i, j, t, d in ((1, 2, 3, 0), (0, 3, 15, 1), (2, 1, 7, 1)):
        Xp = X.clone()
        Xp[i, t, d] += eps
        fd = (_kernel_and_grad(Xp)[0][i, j] - k0[i, j]) / eps
        worst = max(worst, abs(float(dk[i, j, t, d]) - float(fd)))
    assert worst < 1e-5, f"kernel grad mismatch: max |diff|={worst}"
    print(f"[test] kernel gradient (analytic) vs. finite differences: max|diff|={worst:.1e}, OK")


def test_ergodic_error_actually_drops():
    """Regression guard for the 2026-10-10 fix (see tsvec_svgd.py's "Energy
    scale and update-rule fixes"): the first version left ergodic error
    essentially unchanged over 2000 iterations. With the default weights and
    update rule it must fall by well over half within 100 iterations here."""
    from ergodic_energy_torch import ergodic_term
    k_idx, Lambda = make_k_grid_torch(10)
    phi_k = _toy_phi_k(k_idx, Lambda)
    T, N = 48, 10
    P0 = np.stack([np.linspace([0.1, 0.1 + 0.08 * i], [0.9, 0.9 - 0.08 * i], T)
                   for i in range(N)]).astype(np.float32)
    svgd = TsvecSvgd(B=None, K=10)
    e0 = ergodic_term(torch.as_tensor(P0), svgd.k_idx, svgd.Lambda, phi_k).mean().item()
    out = svgd.run(P0, phi_k, n_iters=100, record=False)
    e1 = ergodic_term(out['final'], svgd.k_idx, svgd.Lambda, phi_k).mean().item()
    assert e1 < 0.5 * e0, f"ergodic error should drop by >50%: {e0:.3f} -> {e1:.3f}"
    print(f"[test] ergodic error drops with default weights: {e0:.3f} -> {e1:.3f}, OK")


def test_identity_basis_matches_raw():
    k_idx, Lambda = make_k_grid_torch(6)
    phi_k = _toy_phi_k(k_idx, Lambda)
    T = 20
    N = 3
    rng = np.random.default_rng(1)
    P0 = (np.linspace([0.1, 0.1], [0.9, 0.9], T)[None].repeat(N, 0)
         + 0.03 * rng.normal(size=(N, T, 2))).astype(np.float32)

    svgd_raw = TsvecSvgd(B=None, K=6)
    svgd_eye = TsvecSvgd(B=np.eye(T, dtype=np.float32), K=6)

    out_raw = svgd_raw.run(P0, phi_k, n_iters=3, tau=0.05, record=False)
    out_eye = svgd_eye.run(P0, phi_k, n_iters=3, tau=0.05, record=False)
    err = (out_raw['final'] - out_eye['final']).abs().max().item()
    assert err < 1e-3, f"B=eye should match B=None: max|diff|={err}"
    print(f"[test] B=identity matches raw waypoints: max|diff|={err:.2e}, OK")


def test_descent_and_smoothness_force():
    k_idx, Lambda = make_k_grid_torch(8)
    phi_k = _toy_phi_k(k_idx, Lambda)
    T, N = 32, 6
    rng = np.random.default_rng(2)
    P0 = (np.linspace([0.1, 0.1], [0.9, 0.9], T)[None].repeat(N, 0)
         + 0.02 * rng.normal(size=(N, T, 2))).astype(np.float32)

    def energy(P, svgd):
        f = svgd._residual_fn(phi_k)
        r = torch.vmap(f)(P.reshape(P.shape[0], -1))
        return 0.5 * (r ** 2).sum(dim=-1).mean().item()

    svgd0 = TsvecSvgd(B=None, K=8, w_smooth=0.0)
    out0 = svgd0.run(P0, phi_k, n_iters=15, tau=0.05, record=False)
    e_before = energy(torch.as_tensor(P0), svgd0)
    e_after = energy(out0['final'], svgd0)
    assert e_after < e_before, f"energy should decrease: {e_before} -> {e_after}"
    print(f"[test] mean particle energy decreases over 15 steps: {e_before:.4g} -> {e_after:.4g}, OK")

    def accel_cost(P):
        a = P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]
        return float((a ** 2).sum(dim=(1, 2)).mean())

    svgd_smooth = TsvecSvgd(B=None, K=8, w_smooth=50.0)
    out_smooth = svgd_smooth.run(P0, phi_k, n_iters=15, tau=0.05, record=False)
    c0, c_plain, c_smooth = (accel_cost(torch.as_tensor(P0)), accel_cost(out0['final']),
                             accel_cost(out_smooth['final']))
    assert c_smooth < c_plain, f"w_smooth=50 should reduce accel cost: {c_plain} vs {c_smooth}"
    print(f"[test] w_smooth=50 reduces acceleration cost: init={c0:.4g}, "
         f"w_smooth=0 -> {c_plain:.4g}, w_smooth=50 -> {c_smooth:.4g}, OK")


def test_timing_real_scale():
    k_idx, Lambda = make_k_grid_torch(10)
    phi_k = _toy_phi_k(k_idx, Lambda)
    T, N, nxi = 128, 30, 25
    rng = np.random.default_rng(3)
    P0_raw = (np.linspace([0.1, 0.1], [0.9, 0.9], T)[None].repeat(N, 0)
             + 0.02 * rng.normal(size=(N, T, 2))).astype(np.float32)
    try:
        from obstacles import bspline_basis_matrix
        B = bspline_basis_matrix(nxi, T, 5).astype(np.float32)
    except ModuleNotFoundError:
        # `bsplinax` isn't installed in this local ad hoc environment (it is
        # in the `thesis` conda env, see CLAUDE.md) -- a smooth-ish stand-in
        # basis is fine here since this sub-test only measures wall-clock
        # time, not numerical correctness (covered by the tests above).
        print("[timing] bsplinax not installed locally -- using a random "
             "orthonormal stand-in basis for the bspline timing case only")
        Q, _ = np.linalg.qr(rng.normal(size=(T, nxi)))
        B = Q.astype(np.float32)
    P0_bs = np.linalg.lstsq(B, P0_raw.transpose(1, 0, 2).reshape(T, N * 2),
                            rcond=None)[0].reshape(nxi, N, 2).transpose(1, 0, 2).astype(np.float32)

    for name, svgd, P0 in (('bspline (nxi=25)', TsvecSvgd(B=B, K=10), P0_bs),
                           ('raw waypoints (T=128)', TsvecSvgd(B=None, K=10), P0_raw)):
        t0 = time.time()
        n = 5
        svgd.run(P0, phi_k, n_iters=n, tau=0.05, record=False)
        dt = (time.time() - t0) / n
        print(f"[timing] {name}: {dt * 1000:.1f} ms/iteration (CPU, N={N}) "
             f"-> {dt * 2000:.1f} s for 2000 iterations")


def test_tensor_inputs_accepted():
    """`run()` must accept init_P/phi_k already as torch tensors, not just
    numpy -- see point 7 above."""
    k_idx, Lambda = make_k_grid_torch(6)
    phi_k_tensor = torch.rand(k_idx.shape[0], dtype=torch.float32)  # NOT a numpy array
    svgd = TsvecSvgd(B=None, K=6)
    P0_tensor = torch.rand(4, 16, 2, dtype=torch.float32) * 0.6 + 0.2  # also NOT numpy
    out = svgd.run(P0_tensor, phi_k_tensor, n_iters=3, tau=0.05, record=True)
    assert torch.isfinite(out['final']).all()
    assert torch.isfinite(out['log']).all()
    print("[test] TsvecSvgd.run() accepts torch-tensor init_P/phi_k (not just numpy): OK")


if __name__ == '__main__':
    test_residual_jacobian()
    test_kernel_gradient()
    test_ergodic_error_actually_drops()
    test_identity_basis_matches_raw()
    test_descent_and_smoothness_force()
    test_tensor_inputs_accepted()
    test_timing_real_scale()
    print("ALL TESTS PASSED")
