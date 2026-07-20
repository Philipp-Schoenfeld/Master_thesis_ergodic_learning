#!/usr/bin/env python3
"""
Signature Kernel SVGD on B-Spline Control Points  (Component 4b)
==================================================================
SVGD update loop operating on B-spline control points, using the
Signature Kernel for inter-particle repulsion.

Three-step update per iteration:
  1. Score:     ∇_w ergodic_cost(w_i)  via Log-Surrogate MMD  (analytic)
  2. Kernel:    Signature Kernel Gram matrix  K^sig(traj_i, traj_j)
  3. SVGD:     Δw_i = (1/N) Σ_j [ K(j,i) · score_j + ∇_{w_j} K(j,i) ]

The kernel gradient flows through the B-spline reconstruction:
  ∇_w k^sig = ∂k/∂traj · ∂traj/∂w  ≈  SPSA(k^sig) projected through Bᵀ.

Adapted from:
  - SigKernel_CMA/sv_cma_es_2d.py  (PDE sig kernel + SPSA gradient)
  - SE3_SVGD/tsvec_2d.py           (Adam optimizer)
"""

import numpy as np
from bspline_trajectory import BSplineTrajectoryAdapter
from log_surrogate_mmd import FourierErgodicMetric


# ============================================================================
# 1. Signature Kernel — PDE Finite-Difference Solver
#    (reused from SigKernel_CMA/sv_cma_es_2d.py)
# ============================================================================

def signature_kernel_pde(x, y):
    """
    Compute the signature kernel k^sig(x, y) between two discrete paths
    using the PDE finite-difference scheme.

    The signature kernel satisfies:
        d²K / (ds dt) = <dx(s), dy(t)> · K(s, t)
    with boundary conditions  K(0, t) = 1,  K(s, 0) = 1.

    Parameters
    ----------
    x : ndarray (T, 2) — first path
    y : ndarray (T, 2) — second path

    Returns
    -------
    k_sig : float — scalar signature kernel value
    """
    Tx, Ty = len(x), len(y)
    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)
    M = np.ones((Tx, Ty))
    D = dx @ dy.T
    for i in range(Tx - 1):
        A = M[i, 1:] + M[i, :-1] * (D[i] - 1.0)
        M[i + 1, 1:] = M[i + 1, 0] + np.cumsum(A)
    return M[-1, -1]


def signature_kernel_matrix(paths):
    """
    Compute the N×N Gram matrix of signature kernels.

    Parameters
    ----------
    paths : list of ndarray, each (T, 2)

    Returns
    -------
    K : ndarray (N, N)
    """
    N = len(paths)
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = signature_kernel_pde(paths[i], paths[i])
        for j in range(i + 1, N):
            K[i, j] = signature_kernel_pde(paths[i], paths[j])
            K[j, i] = K[i, j]
    return K


def signature_kernel_grad_spsa(x_j, x_i, eps=1e-3):
    """
    Approximate ∇_{x_j} k^sig(x_j, x_i) using SPSA.

    Returns an array of shape (T, 2).
    """
    T_len, dim = x_j.shape
    delta = np.sign(np.random.randn(T_len, dim))
    delta[delta == 0] = 1.0
    k_plus = signature_kernel_pde(x_j + eps * delta, x_i)
    k_minus = signature_kernel_pde(x_j - eps * delta, x_i)
    return (k_plus - k_minus) / (2.0 * eps) * (1.0 / delta)


# ============================================================================
# 2. SVGD on B-Spline Control Points
# ============================================================================

def svgd_step_bspline(
    control_points,
    adapter,
    metric,
    w_ergodic=600.0,
    w_smooth=15.0,
    w_boundary=30.0,
    w_obstacle=50000.0,
    use_obstacle=False,
    obstacle_center=(0.5, 0.5),
    obstacle_radius=0.12,
    use_log_surrogate=True,
    use_sig_kernel=True,
    rbf_bandwidth=None,
    spsa_eps=1e-3,
):
    """
    Perform one SVGD step on B-spline control points.

    Parameters
    ----------
    control_points : ndarray (N, n_ctrl, 2) — current control points
    adapter        : BSplineTrajectoryAdapter
    metric         : FourierErgodicMetric
    (weights, obstacle params, etc.)

    Returns
    -------
    updates    : ndarray (N, n_ctrl, 2) — SVGD update direction
    energies   : ndarray (N,) — current energies
    """
    N, n_ctrl, D = control_points.shape
    B_np = adapter.B_np
    dB_ds_np = adapter.dB_ds_np
    d2B_ds2_np = adapter.d2B_ds2_np

    # ── Phase 1: Compute scores (negative cost gradients) ──────────
    scores = np.zeros_like(control_points)    # (N, n_ctrl, 2)
    energies = np.zeros(N)

    for i in range(N):
        cost, grad_w = metric.full_cost_and_grad_cp(
            control_points[i], B_np, dB_ds_np, d2B_ds2_np,
            w_ergodic=w_ergodic, w_smooth=w_smooth, w_boundary=w_boundary,
            w_obstacle=w_obstacle, use_obstacle=use_obstacle,
            obstacle_center=obstacle_center, obstacle_radius=obstacle_radius,
            use_log_surrogate=use_log_surrogate,
        )
        scores[i] = -grad_w    # Score = negative gradient (descent direction)
        energies[i] = cost

    # ── Phase 2: Compute kernel matrix ────────────────────────────
    # Reconstruct dense trajectories for kernel evaluation
    trajectories = adapter.control_points_to_trajectory(control_points)  # (N, T, 2)

    if use_sig_kernel:
        paths = [trajectories[i] for i in range(N)]
        K_mat = signature_kernel_matrix(paths)        # (N, N)
    else:
        # Fallback: RBF kernel on flattened control points
        cp_flat = control_points.reshape(N, -1)
        from scipy.spatial.distance import pdist, squareform
        sq = squareform(pdist(cp_flat, 'sqeuclidean'))
        pos = sq[sq > 0]
        med = np.median(pos) if len(pos) > 0 else 1.0
        h = max(med / np.log(N + 1), 0.1) if rbf_bandwidth is None else rbf_bandwidth
        K_mat = np.exp(-sq / h)

    # ── Phase 3: Kernel gradients ─────────────────────────────────
    #   ∇_{w_j} k^sig(traj(w_j), traj(w_i))
    #   ≈ Bᵀ @ SPSA_grad(traj_j, traj_i)
    # For the RBF kernel, the gradient is analytic.

    updates = np.zeros_like(control_points)   # (N, n_ctrl, 2)

    if use_sig_kernel:
        for i in range(N):
            for j in range(N):
                # Attractive: score_j weighted by kernel
                updates[i] += K_mat[j, i] * scores[j]

                # Repulsive: kernel gradient w.r.t. w_j
                # ∇_{traj_j} k^sig  via SPSA
                traj_grad = signature_kernel_grad_spsa(
                    trajectories[j], trajectories[i], eps=spsa_eps
                )
                # Project through B-spline: ∂w = Bᵀ @ ∂traj
                cp_grad = B_np.T @ traj_grad   # (n_ctrl, 2)
                updates[i] += cp_grad

            updates[i] /= N
    else:
        # RBF kernel: analytic gradient
        cp_flat = control_points.reshape(N, -1)
        h_val = max(np.median(pdist(cp_flat, 'sqeuclidean')) / np.log(N + 1), 0.1)
        for i in range(N):
            for j in range(N):
                updates[i] += K_mat[j, i] * scores[j]
                # RBF gradient: K * (-2/h) * (w_j - w_i)
                diff = (control_points[j] - control_points[i])
                updates[i] += K_mat[j, i] * (-2.0 / h_val) * diff
            updates[i] /= N

    return updates, energies


# ============================================================================
# 3. Full SVGD Optimization Loop
# ============================================================================

def run_svgd_bspline(
    initial_control_points,
    adapter,
    metric,
    n_iters=150,
    lr=0.005,
    w_ergodic=600.0,
    w_smooth=15.0,
    w_boundary=30.0,
    w_obstacle=50000.0,
    use_obstacle=False,
    obstacle_center=(0.5, 0.5),
    obstacle_radius=0.12,
    use_log_surrogate=True,
    use_sig_kernel=True,
    verbose=True,
):
    """
    Run the full SVGD optimization loop on B-spline control points.

    Parameters
    ----------
    initial_control_points : ndarray (N, n_ctrl, 2)
    adapter    : BSplineTrajectoryAdapter
    metric     : FourierErgodicMetric
    n_iters    : int
    lr         : float — Adam learning rate
    (weights, obstacle params, etc.)

    Returns
    -------
    control_points : ndarray (N, n_ctrl, 2) — optimized
    energy_log     : list of float — mean energy per iteration
    """
    control_points = initial_control_points.copy()
    N, n_ctrl, D = control_points.shape
    energy_log = []

    # Adam state
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    m = np.zeros_like(control_points)
    v = np.zeros_like(control_points)

    for it in range(n_iters):
        updates, energies = svgd_step_bspline(
            control_points, adapter, metric,
            w_ergodic=w_ergodic, w_smooth=w_smooth, w_boundary=w_boundary,
            w_obstacle=w_obstacle, use_obstacle=use_obstacle,
            obstacle_center=obstacle_center, obstacle_radius=obstacle_radius,
            use_log_surrogate=use_log_surrogate,
            use_sig_kernel=use_sig_kernel,
        )

        # Gradient clipping
        mx = np.max(np.abs(updates))
        if mx > 200:
            updates *= 200.0 / mx

        # Adam step
        t_adam = it + 1
        m = beta1 * m + (1 - beta1) * updates
        v = beta2 * v + (1 - beta2) * updates ** 2
        m_hat = m / (1 - beta1 ** t_adam)
        v_hat = v / (1 - beta2 ** t_adam)

        control_points += lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Clamp control points to [margin, 1-margin]
        control_points = np.clip(control_points, 0.02, 0.98)

        energy_log.append(float(np.mean(energies)))

        if verbose and ((it + 1) % 10 == 0 or it == 0):
            print(f"    SVGD iter {it+1:4d}/{n_iters}  "
                  f"mean_E={np.mean(energies):.3f}  "
                  f"best_E={np.min(energies):.3f}")

    return control_points, energy_log


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/Unified_Pipeline")

    print("Signature Kernel SVGD on B-Splines — self-test")

    adapter = BSplineTrajectoryAdapter(
        degree=4, num_control_points=16, num_phase_points=100
    )
    metric = FourierErgodicMetric(target_shape='N', K=10)

    # Create 3 identical initial trajectories (should diversify)
    np.random.seed(42)
    N = 3
    traj_base = np.cumsum(np.random.randn(100, 2) * 0.01, axis=0)
    traj_base = (traj_base - traj_base.min(0)) / (traj_base.max(0) - traj_base.min(0) + 1e-8)
    traj_base = traj_base * 0.6 + 0.2

    w_base = adapter.trajectory_to_control_points(traj_base)
    initial_cps = np.stack([w_base + np.random.randn(*w_base.shape) * 0.01
                            for _ in range(N)])

    # Run a few SVGD iterations with RBF kernel (faster for testing)
    final_cps, e_log = run_svgd_bspline(
        initial_cps, adapter, metric,
        n_iters=15, lr=0.003,
        use_sig_kernel=False,   # Use RBF for quick test
        verbose=True,
    )

    # Check diversification
    dists_init = [np.linalg.norm(initial_cps[0] - initial_cps[j])
                  for j in range(1, N)]
    dists_final = [np.linalg.norm(final_cps[0] - final_cps[j])
                   for j in range(1, N)]
    print(f"  Initial pairwise dists: {[f'{d:.4f}' for d in dists_init]}")
    print(f"  Final pairwise dists:   {[f'{d:.4f}' for d in dists_final]}")
    print(f"  Energy log: {e_log[0]:.1f} → {e_log[-1]:.1f}")
    print("  ✓ Self-test passed.")
