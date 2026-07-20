#!/usr/bin/env python3
"""
3D Stein Variational Ergodic Coverage (TSVEC) — Regular SVGD (no B-splines)

=============================================================================
This is the 3D extension of tsvec_2d.py. It uses the same shared modules
(ergodic_core, svgd_engine) to maximize code reuse.

Key differences from 2D:
  - Spatial dimension: 3 (workspace [0,1]^3)
  - Trajectory points: (T, 3) instead of (T, 2)
  - Fourier indices: K^3 combinations (k1, k2, k3)
  - Target distribution: extruded 2D shapes along z-axis
  - Visualization: 3D Axes3D plots
  - Analytic gradients generalized to 3D via shared modules
=============================================================================
"""

import time
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# Add parent paths for imports
sys.path.append("/home/philipp/Documents/Uni/Master_thesis/src")
sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/SE3_SVGD")

from init_strategies_nd import get_initialization
from ergodic_core import (
    build_target_distribution_3d,
    build_fourier_indices,
    compute_lambda_k,
    compute_target_fourier_coeffs,
    compute_ergodic_metric_numpy,
    fourier_basis_nd,
    fourier_basis_grad_nd,
    get_3d_segments
)
from svgd_engine import (
    compute_smoothness_cost_numpy,
    compute_smoothness_grad_numpy,
    compute_boundary_cost_numpy,
    compute_boundary_grad_numpy,
    compute_obstacle_cost_numpy,
    compute_obstacle_grad_numpy,
    svgd_step_numpy,
    run_svgd_numpy,
)

np.random.seed(42)

# ============================================================================
# 1. Configuration
# ============================================================================

DIM = 3
TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
TARGET_PROJECTION = os.environ.get('TARGET_PROJECTION', 'plane')

# Build 3D target distribution
_grid_axes, _grid_pts, _grid_weights, _grid_shape = build_target_distribution_3d(
    TARGET_SHAPE, stroke_width=0.045, grid_res=50, projection_type=TARGET_PROJECTION
)

# Fourier decomposition
K_FOURIER = 5
k_indices = build_fourier_indices(K_FOURIER, DIM)
Lambda_k = compute_lambda_k(k_indices)
phi_k = compute_target_fourier_coeffs(_grid_pts, _grid_weights, k_indices)

# ============================================================================
# 2. Hyperparameters
# ============================================================================

T = 100
N_PARTICLES = 10
N_ITERS = int(os.environ.get("N_ITERS", 200))

# Adam parameters
ADAM_LR = 2e-3
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Energy weights
W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0

# Obstacle
USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

INIT_NOISE_STD = 0.02

# ============================================================================
# 3. Energy Function & Analytic Gradient
# ============================================================================

def compute_energy_and_grad(X_flat, T_steps):
    """
    Computes total energy and its analytic gradient for a single 3D trajectory.
    Uses shared dim-agnostic cost components from svgd_engine and ergodic_core.

    Args:
        X_flat: (T*3,) flattened trajectory
        T_steps: number of time steps

    Returns:
        energy: scalar total energy
        grad: (T*3,) gradient
    """
    X = X_flat.reshape(T_steps, DIM)
    grad = np.zeros_like(X)
    energy = 0.0

    # ---- Smoothness (acceleration penalty) ----
    energy += W_SMOOTH * compute_smoothness_cost_numpy(X)
    grad += W_SMOOTH * compute_smoothness_grad_numpy(X)

    # ---- Ergodic cost ----
    Fk = fourier_basis_nd(X, k_indices)  # (T, M)
    c_k = np.mean(Fk, axis=0)
    diff_k = c_k - phi_k
    energy += W_ERGODIC * 0.5 * np.sum(Lambda_k * diff_k ** 2)

    Fk_g = fourier_basis_grad_nd(X, k_indices)  # (T, M, dim)
    w_diff = Lambda_k * diff_k
    grad += W_ERGODIC * (1.0 / T_steps) * np.einsum('k,tkd->td', w_diff, Fk_g)

    # ---- Boundary penalty ----
    energy += W_BOUNDARY * compute_boundary_cost_numpy(X)
    grad += W_BOUNDARY * compute_boundary_grad_numpy(X)

    # ---- Obstacle penalty ----
    if USE_OBSTACLE:
        energy += W_OBSTACLE * compute_obstacle_cost_numpy(X, OBSTACLE_CENTER, OBSTACLE_RADIUS)
        grad += W_OBSTACLE * compute_obstacle_grad_numpy(X, OBSTACLE_CENTER, OBSTACLE_RADIUS)

    return energy, grad.ravel()


# ============================================================================
# 4. Master Benchmark Function
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle

    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape"]
    results = {}
    benchmark_data = {}

    print(f"3D TSVEC  |  {N_PARTICLES} particles, T={T}, K_Fourier={K_FOURIER}, {N_ITERS} iters")
    print(f"Fourier modes: {len(k_indices)}")
    print(f"Weights:  ergodic={W_ERGODIC}, smooth={W_SMOOTH}, boundary={W_BOUNDARY}, obstacle={W_OBSTACLE}")
    print(f"Adam:     lr={ADAM_LR}, β1={ADAM_BETA1}, β2={ADAM_BETA2}")
    print("-" * 65)

    for strat in strategies:
        print(f"Running strategy: {strat}")
        t_start = time.time()

        init_p, base_t = get_initialization(strat, N_PARTICLES, T, dim=DIM, noise_std=INIT_NOISE_STD)

        final_p, e_log = run_svgd_numpy(
            init_p, T, N_ITERS, compute_energy_and_grad, DIM,
            adam_lr=ADAM_LR, adam_beta1=ADAM_BETA1, adam_beta2=ADAM_BETA2, adam_eps=ADAM_EPS,
            n_particles=N_PARTICLES, label=f"SVGD ({strat})"
        )

        final_E = np.array([compute_energy_and_grad(final_p[i], T)[0] for i in range(N_PARTICLES)])
        best = int(np.argmin(final_E))
        elapsed = time.time() - t_start
        print(f"  -> Best energy: {final_E[best]:.3f} (Time: {elapsed:.2f}s)\n")

        results[strat] = {
            'initial': init_p,
            'base_traj': base_t,
            'final': final_p,
            'energy_log': e_log,
            'best_idx': best
        }

        benchmark_data[strat] = {
            'mean_cost': float(np.mean(final_E)),
            'best_cost': float(final_E[best]),
            'time_s': float(elapsed)
        }

        if save_npy:
            np.save(os.path.join(out_dir, f"{strat}_trajs.npy"), final_p)

    # ============================================================================
    # 5. 3D Visualization
    # ============================================================================

    colors = plt.cm.rainbow(np.linspace(0, 1, N_PARTICLES))

    def plot_particles_3d(ax, parts, title, highlight_best=False, best_idx=-1, base_traj=None):
        if base_traj is not None:
            ax.plot(base_traj[:, 0], base_traj[:, 1], base_traj[:, 2],
                    '--', color='black', lw=2.0, label='Base Trajectory', zorder=9)

        for i in range(N_PARTICLES):
            tr = parts[i].reshape(T, DIM)
            lw = 2.5 if (highlight_best and i == best_idx) else 0.6
            al = 1.0 if (highlight_best and i == best_idx) else 0.3
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], '-', color=colors[i], lw=lw, alpha=al)
            ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], color=colors[i], s=10, depthshade=True)

        # Plot the target shape skeleton
        try:
            target_segs = get_3d_segments(TARGET_SHAPE, TARGET_PROJECTION)
            for (a, b) in target_segs:
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], 'k--', lw=1.5, alpha=0.3)
        except Exception:
            pass

        if highlight_best and best_idx >= 0:
            best_tr = parts[best_idx].reshape(T, DIM)
            ax.plot(best_tr[:, 0], best_tr[:, 1], best_tr[:, 2], '-', color=colors[best_idx],
                    lw=2.5, label=f'Best (#{best_idx})', zorder=10)
            ax.legend(loc='lower right', fontsize=8)

        if USE_OBSTACLE:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            ox = OBSTACLE_CENTER[0] + OBSTACLE_RADIUS * np.outer(np.cos(u), np.sin(v))
            oy = OBSTACLE_CENTER[1] + OBSTACLE_RADIUS * np.outer(np.sin(u), np.sin(v))
            oz = OBSTACLE_CENTER[2] + OBSTACLE_RADIUS * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(ox, oy, oz, color='gray', alpha=0.3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title, fontsize=11, fontweight='bold')

    fig = plt.figure(figsize=(14, 5 * len(strategies)))
    fig.suptitle(f"Regular SVGD 3D (No B-Spline) | Shape: {TARGET_SHAPE} | Iters: {N_ITERS}\n"
                 f"W_erg: {W_ERGODIC}, W_smooth: {W_SMOOTH}, W_bnd: {W_BOUNDARY}, LR: {ADAM_LR}", fontsize=14, fontweight='bold')
    
    for row, strat in enumerate(strategies):
        res = results[strat]

        ax1 = fig.add_subplot(len(strategies), 2, 2 * row + 1, projection='3d')
        plot_particles_3d(ax1, res['initial'], f'[{strat}] Initial', base_traj=res['base_traj'])

        ax2 = fig.add_subplot(len(strategies), 2, 2 * row + 2, projection='3d')
        plot_particles_3d(ax2, res['final'], f'[{strat}] Final',
                         highlight_best=True, best_idx=res['best_idx'])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    timestamp = datetime.now().strftime("%H-%M_%d-%m")
    out_path = os.path.join(out_dir, f'tsvec_3d_{TARGET_SHAPE}_comparison_{timestamp}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T': T,
            'DIM': DIM,
            'N_PARTICLES': N_PARTICLES,
            'N_ITERS': N_ITERS,
            'K_FOURIER': K_FOURIER,
            'W_ERGODIC': W_ERGODIC,
            'W_SMOOTH': W_SMOOTH,
            'W_BOUNDARY': W_BOUNDARY
        }, f, indent=4)

    return benchmark_data


if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    TARGET_PROJECTION = os.environ.get('TARGET_PROJECTION', 'plane')
    out_dir = f'/home/philipp/Documents/Uni/Master_thesis/results/SE3_SVGD_3D_{TARGET_SHAPE}_{TARGET_PROJECTION}'
    run_benchmark(out_dir=out_dir)
