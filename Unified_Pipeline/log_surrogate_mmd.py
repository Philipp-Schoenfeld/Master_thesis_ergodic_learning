#!/usr/bin/env python3
"""
Log-Surrogate MMD Ergodic Metric  (Component 4a)
==================================================
Implements the Log-Surrogate MMD ergodic cost and its analytic gradient
with respect to B-spline control points.

Standard Fourier metric:
    E = Σ_k Λ_k (c_k - φ_k)²

Log-Surrogate variant (better gradient landscape for multimodal targets):
    E_log = log( Σ_k Λ_k exp((c_k - φ_k)²) )

The gradient propagates through the chain:
    control points w → dense trajectory (B @ w) → Fourier coefficients → cost

Adapted from SE3_SVGD/tsvec_2d.py  Fourier basis functions.
"""

import numpy as np


# ============================================================================
# 1. Target Distribution & Fourier Reference  (reused across methods)
# ============================================================================

SEGMENT_DEFS = {
    'N': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.25, 0.85], [0.75, 0.15]),
        ([0.75, 0.15], [0.75, 0.85]),
    ],
    'H': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
        ([0.25, 0.50], [0.75, 0.50]),
    ],
    'II': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
    ],
}
STROKE_WIDTH = 0.045


def _dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (len_sq + 1e-12), 0, 1)
    return np.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)


def target_distribution(x, y, segments):
    d_min = np.full_like(x, 1e10)
    for (ax, ay), (bx, by) in segments:
        d_min = np.minimum(d_min, _dist_to_segment(x, y, ax, ay, bx, by))
    return np.exp(-d_min ** 2 / (2 * STROKE_WIDTH ** 2))


class FourierErgodicMetric:
    """
    Precomputes the Fourier reference spectrum φ_k for a given target
    distribution on [0,1]², and provides cost + gradient functions.
    """

    def __init__(self, target_shape: str = 'N', K: int = 10, grid_res: int = 200):
        """
        Parameters
        ----------
        target_shape : str  — one of 'N', 'H', 'II'
        K            : int  — number of Fourier modes per dimension
        grid_res     : int  — grid resolution for numerical integration
        """
        self.K = K
        self.K_sq = K * K
        self.target_shape = target_shape

        # Build wavenumber indices  (K², 2)
        self.k_indices = np.array(
            [[k1, k2] for k1 in range(K) for k2 in range(K)],
            dtype=np.float64
        )

        # Spectral weights  Λ_k = (1 + ||k||²)^{-3/2}
        self.Lambda_k = (1.0 + np.sum(self.k_indices ** 2, axis=1)) ** (-1.5)

        # Target reference coefficients  φ_k
        xs = np.linspace(0, 1, grid_res)
        Xg, Yg = np.meshgrid(xs, xs)
        segments = SEGMENT_DEFS[target_shape]
        Zg = target_distribution(Xg, Yg, segments)

        grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
        grid_w = Zg.ravel()
        grid_w = grid_w / grid_w.sum()

        self.phi_k = np.sum(grid_w[:, None] * self._basis(grid_pts), axis=0)

        # Store grid for visualization
        self.Xg, self.Yg, self.Zg = Xg, Yg, Zg

    # ──────────────────────────────────────────────────────────────────
    #  Fourier basis functions
    # ──────────────────────────────────────────────────────────────────

    def _basis(self, pts):
        """Evaluate cosine basis at pts.  pts: (M, 2) → (M, K²)."""
        args = np.pi * pts[:, None, :] * self.k_indices[None, :, :]
        return np.prod(np.cos(args), axis=-1)

    def _basis_grad(self, pts):
        """Gradient of cosine basis.  pts: (M, 2) → (M, K², 2)."""
        args = np.pi * pts[:, None, :] * self.k_indices[None, :, :]
        c, s = np.cos(args), np.sin(args)
        gx = -np.pi * self.k_indices[None, :, 0] * s[:, :, 0] * c[:, :, 1]
        gy = -np.pi * self.k_indices[None, :, 1] * c[:, :, 0] * s[:, :, 1]
        return np.stack([gx, gy], axis=-1)

    # ──────────────────────────────────────────────────────────────────
    #  Standard Fourier Ergodic Cost  (on raw trajectory)
    # ──────────────────────────────────────────────────────────────────

    def ergodic_cost(self, traj):
        """
        Standard Fourier ergodic cost for a trajectory.

        Parameters
        ----------
        traj : ndarray (T, 2)

        Returns
        -------
        cost : float
        """
        Fk = self._basis(traj)           # (T, K²)
        c_k = np.mean(Fk, axis=0)        # time-averaged coefficients
        diff_k = c_k - self.phi_k
        return 0.5 * np.sum(self.Lambda_k * diff_k ** 2)

    def ergodic_cost_and_grad(self, traj):
        """
        Standard Fourier ergodic cost + gradient w.r.t. trajectory.

        Parameters
        ----------
        traj : ndarray (T, 2)

        Returns
        -------
        cost : float
        grad : ndarray (T, 2)
        """
        T = len(traj)
        Fk = self._basis(traj)
        c_k = np.mean(Fk, axis=0)
        diff_k = c_k - self.phi_k
        cost = 0.5 * np.sum(self.Lambda_k * diff_k ** 2)

        Fk_g = self._basis_grad(traj)       # (T, K², 2)
        w_diff = self.Lambda_k * diff_k      # (K²,)
        grad = (1.0 / T) * np.einsum('k,tkd->td', w_diff, Fk_g)

        return cost, grad

    # ──────────────────────────────────────────────────────────────────
    #  Log-Surrogate MMD Cost  (on raw trajectory)
    # ──────────────────────────────────────────────────────────────────

    def log_surrogate_cost(self, traj):
        """
        Log-Surrogate MMD:
            E_log = log( Σ_k Λ_k exp((c_k - φ_k)²) )

        This puts more gradient pressure on the worst-covered modes.

        Parameters
        ----------
        traj : ndarray (T, 2)

        Returns
        -------
        cost : float
        """
        Fk = self._basis(traj)
        c_k = np.mean(Fk, axis=0)
        diff_k = c_k - self.phi_k
        sq = diff_k ** 2
        return np.log(np.sum(self.Lambda_k * np.exp(sq)) + 1e-30)

    def log_surrogate_cost_and_grad(self, traj):
        """
        Log-Surrogate MMD cost + gradient w.r.t. trajectory.

        ∂E_log/∂traj = Σ_k  w_k * 2(c_k - φ_k) / T * ∂F_k/∂traj
        where  w_k = Λ_k exp((c_k - φ_k)²) / Σ_j Λ_j exp((c_j - φ_j)²)
        (softmax attention over modes)

        Parameters
        ----------
        traj : ndarray (T, 2)

        Returns
        -------
        cost : float
        grad : ndarray (T, 2)
        """
        T = len(traj)
        Fk = self._basis(traj)
        c_k = np.mean(Fk, axis=0)
        diff_k = c_k - self.phi_k
        sq = diff_k ** 2

        # Softmax attention weights over modes
        exp_sq = self.Lambda_k * np.exp(sq)
        Z = np.sum(exp_sq) + 1e-30
        cost = np.log(Z)

        # Gradient: weighted combination (modes with larger error get more weight)
        attn = exp_sq / Z                          # (K²,)
        w_diff = attn * 2.0 * diff_k               # (K²,)

        Fk_g = self._basis_grad(traj)               # (T, K², 2)
        grad = (1.0 / T) * np.einsum('k,tkd->td', w_diff, Fk_g)

        return cost, grad

    # ──────────────────────────────────────────────────────────────────
    #  Control-point-level cost  (with smoothness + boundary)
    # ──────────────────────────────────────────────────────────────────

    def full_cost_and_grad_cp(
        self,
        w,
        B_np,
        dB_ds_np,
        d2B_ds2_np,
        w_ergodic=600.0,
        w_smooth=15.0,
        w_boundary=30.0,
        w_obstacle=50000.0,
        use_obstacle=False,
        obstacle_center=(0.5, 0.5),
        obstacle_radius=0.12,
        use_log_surrogate=True,
    ):
        """
        Compute the full trajectory cost and its gradient w.r.t. B-spline
        control points  w ∈ ℝ^{n_ctrl × 2}.

        The gradient chain:
            w → traj = B @ w → cost
            ∂cost/∂w = Bᵀ @ ∂cost/∂traj

        Parameters
        ----------
        w             : ndarray (n_ctrl, 2)
        B_np          : ndarray (T, n_ctrl) — position basis
        dB_ds_np      : ndarray (T, n_ctrl) — velocity basis
        d2B_ds2_np    : ndarray (T, n_ctrl) — acceleration basis
        w_ergodic     : float — ergodic cost weight
        w_smooth      : float — smoothness (acceleration) weight
        w_boundary    : float — boundary penalty weight
        w_obstacle    : float — obstacle penalty weight
        use_obstacle  : bool
        obstacle_center : tuple
        obstacle_radius : float
        use_log_surrogate : bool — use log-surrogate MMD instead of standard

        Returns
        -------
        cost : float
        grad_w : ndarray (n_ctrl, 2)
        """
        # Reconstruct dense trajectory
        traj = B_np @ w                    # (T, 2)
        T = len(traj)

        cost = 0.0
        grad_traj = np.zeros_like(traj)    # (T, 2)

        # ---- Ergodic cost ----
        if use_log_surrogate:
            e_cost, e_grad = self.log_surrogate_cost_and_grad(traj)
        else:
            e_cost, e_grad = self.ergodic_cost_and_grad(traj)

        cost += w_ergodic * e_cost
        grad_traj += w_ergodic * e_grad

        # ---- Smoothness (acceleration penalty on dense trajectory) ----
        accel = traj[2:] - 2 * traj[1:-1] + traj[:-2]
        cost += w_smooth * np.sum(accel ** 2)
        g_smooth = np.zeros_like(traj)
        g_smooth[:-2] += 2 * w_smooth * accel
        g_smooth[1:-1] -= 4 * w_smooth * accel
        g_smooth[2:] += 2 * w_smooth * accel
        grad_traj += g_smooth

        # ---- Boundary penalty ----
        margin = 0.03
        lo = np.minimum(traj - margin, 0.0)
        hi = np.maximum(traj - (1.0 - margin), 0.0)
        cost += w_boundary * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))
        grad_traj += w_boundary * (lo + hi)

        # ---- Obstacle penalty ----
        if use_obstacle:
            dx = traj[:, 0] - obstacle_center[0]
            dy = traj[:, 1] - obstacle_center[1]
            dist = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)
            violation = np.maximum(obstacle_radius - dist, 0.0)
            cost += w_obstacle * 0.5 * np.sum(violation ** 2)
            grad_traj[:, 0] += w_obstacle * violation * (-dx / dist)
            grad_traj[:, 1] += w_obstacle * violation * (-dy / dist)

        # ---- Chain rule: ∂cost/∂w = Bᵀ @ ∂cost/∂traj ----
        grad_w = B_np.T @ grad_traj        # (n_ctrl, T) @ (T, 2)

        return cost, grad_w


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/Unified_Pipeline")
    from bspline_trajectory import BSplineTrajectoryAdapter

    print("Log-Surrogate MMD — self-test")

    metric = FourierErgodicMetric(target_shape='N', K=10)
    adapter = BSplineTrajectoryAdapter(degree=4, num_control_points=16, num_phase_points=100)

    # Generate a test trajectory
    np.random.seed(0)
    traj = np.cumsum(np.random.randn(100, 2) * 0.01, axis=0)
    traj = (traj - traj.min(0)) / (traj.max(0) - traj.min(0) + 1e-8)
    traj = traj * 0.8 + 0.1

    # Standard ergodic cost
    cost_std = metric.ergodic_cost(traj)
    print(f"  Standard ergodic cost: {cost_std:.6f}")

    # Log-surrogate cost
    cost_log = metric.log_surrogate_cost(traj)
    print(f"  Log-surrogate cost:    {cost_log:.6f}")

    # Gradient test: finite-difference vs analytic
    cost_a, grad_a = metric.log_surrogate_cost_and_grad(traj)
    eps = 1e-5
    grad_fd = np.zeros_like(traj)
    for i in range(min(5, len(traj))):  # Check first 5 points
        for d in range(2):
            traj_p = traj.copy(); traj_p[i, d] += eps
            traj_m = traj.copy(); traj_m[i, d] -= eps
            grad_fd[i, d] = (metric.log_surrogate_cost(traj_p) -
                             metric.log_surrogate_cost(traj_m)) / (2 * eps)

    rel_err = np.linalg.norm(grad_a[:5] - grad_fd[:5]) / (np.linalg.norm(grad_fd[:5]) + 1e-12)
    print(f"  Gradient relative error (first 5 pts): {rel_err:.6e}")
    assert rel_err < 0.01, f"Gradient check failed! rel_err={rel_err}"

    # Control-point-level cost
    w = adapter.trajectory_to_control_points(traj)
    cost_cp, grad_cp = metric.full_cost_and_grad_cp(
        w, adapter.B_np, adapter.dB_ds_np, adapter.d2B_ds2_np,
        use_log_surrogate=True
    )
    print(f"  CP-level cost: {cost_cp:.4f}")
    print(f"  CP-level grad shape: {grad_cp.shape}")

    # Gradient check on control points (finite difference)
    eps = 1e-5
    grad_fd_cp = np.zeros_like(w)
    for i in range(min(4, len(w))):
        for d in range(2):
            w_p = w.copy(); w_p[i, d] += eps
            w_m = w.copy(); w_m[i, d] -= eps
            c_p = metric.full_cost_and_grad_cp(
                w_p, adapter.B_np, adapter.dB_ds_np, adapter.d2B_ds2_np,
                use_log_surrogate=True
            )[0]
            c_m = metric.full_cost_and_grad_cp(
                w_m, adapter.B_np, adapter.dB_ds_np, adapter.d2B_ds2_np,
                use_log_surrogate=True
            )[0]
            grad_fd_cp[i, d] = (c_p - c_m) / (2 * eps)

    rel_err_cp = np.linalg.norm(grad_cp[:4] - grad_fd_cp[:4]) / (np.linalg.norm(grad_fd_cp[:4]) + 1e-12)
    print(f"  CP-gradient relative error (first 4 CPs): {rel_err_cp:.6e}")
    assert rel_err_cp < 0.01, f"CP gradient check failed! rel_err={rel_err_cp}"

    print("  ✓ All self-tests passed.")
