#!/usr/bin/env python3
"""
3D Stein Variational Gradient Descent on B-Splines (Ergodic Coverage)

=============================================================================
This is the 3D extension of svgd_bspline_2d.py. It uses the same shared
modules (ergodic_core, svgd_engine) to maximize code reuse.

Key differences from 2D:
  - Spatial dimension: 3 (workspace [0,1]^3)
  - Control points C: shape (3, K) instead of (2, K)
  - State vector: [x, y, z, vx, vy, vz] (6D instead of 4D)
  - Fourier indices: K^3 combinations (k1, k2, k3)
  - Target distribution: extruded 2D shapes along z-axis
  - Visualization: 3D Axes3D plots
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
sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/bsplinax-main")

from init_strategies_nd import get_initialization
from ergodic_core import (
    build_target_distribution_3d,
    build_fourier_indices,
    compute_lambda_k,
    compute_target_fourier_coeffs,
    compute_ergodic_metric_jax,
    compute_ergodic_metric_numpy,
    get_3d_segments
)
from svgd_engine import (
    compute_smoothness_cost_jax,
    compute_boundary_cost_jax,
    compute_obstacle_cost_jax,
    compute_control_regularization_jax,
    compute_smoothness_cost_numpy,
    compute_boundary_cost_numpy,
    compute_obstacle_cost_numpy,
    forward_sim_nd,
    build_bspline_svgd_step,
    build_adam_optimizer_jax,
    init_bspline_from_positions_nd,
)

import jax
import jax.numpy as jnp
from jax import jit, vmap

from bsplinax.bspline import BsplineBasisClamped

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
K_FOURIER = 5  # Reduced from 10 in 2D because K^3 = 125 vs K^2 = 100
k_indices = build_fourier_indices(K_FOURIER, DIM)
Lambda_k = compute_lambda_k(k_indices)
phi_k = compute_target_fourier_coeffs(_grid_pts, _grid_weights, k_indices)

# Convert to JAX
k_indices_jnp = jnp.array(k_indices)
Lambda_k_jnp = jnp.array(Lambda_k)
phi_k_jnp = jnp.array(phi_k)

# ============================================================================
# 2. Hyperparameters
# ============================================================================

T = 100
N_PARTICLES = 30
N_ITERS = int(os.environ.get("N_ITERS", 100000))

# B-Spline setup
NUM_CONTROL_POINTS = 29  # Optimized via Optuna
DEGREE = 3
dt = 0.05

# Loss weights
W_ERGODIC = 1705.2602796178128
W_SMOOTH = 11.17039814415494
W_BOUNDARY = 18.573679572888505
W_CONTROL = 0.01

# Obstacle
USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

INIT_NOISE_STD = 0.02

# Adam optimizer
ADAM_LR = 0.014424807735718683
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Hardware
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
except Exception:
    gpu = cpu

# ============================================================================
# 3. B-Spline Basis Setup
# ============================================================================

basis_generator = BsplineBasisClamped(
    degree=DEGREE,
    num_control_points=NUM_CONTROL_POINTS,
    num_phase_points=T,
    compute_derivatives=False
)
B_mat = jnp.array(basis_generator.B)
B_outer = jnp.einsum('tk,tl->kl', B_mat, B_mat) * dt

# ============================================================================
# 4. Energy Function (JAX)
# ============================================================================

def compute_energy_jax(C, s0):
    """
    Total objective for a single 3D trajectory.
    C: (3, K), s0: (6,)
    """
    s_traj = forward_sim_nd(C, s0, B_mat, dt, DIM)
    X = s_traj[:, :DIM]  # (T, 3) positions

    energy = W_SMOOTH * compute_smoothness_cost_jax(X)
    energy += W_ERGODIC * compute_ergodic_metric_jax(X, k_indices_jnp, Lambda_k_jnp, phi_k_jnp)
    energy += W_BOUNDARY * compute_boundary_cost_jax(X)

    if USE_OBSTACLE:
        energy += W_OBSTACLE * compute_obstacle_cost_jax(X, OBSTACLE_CENTER, OBSTACLE_RADIUS)

    energy += W_CONTROL * compute_control_regularization_jax(C, B_outer)

    return energy

grad_energy_jax = jax.grad(compute_energy_jax, argnums=0)

# ============================================================================
# 5. Build SVGD + Adam (using shared engine)
# ============================================================================

svgd_step_fn = build_bspline_svgd_step(compute_energy_jax, grad_energy_jax, B_outer)
optimize_C_all = build_adam_optimizer_jax(
    svgd_step_fn, N_ITERS, ADAM_LR, ADAM_BETA1, ADAM_BETA2, ADAM_EPS,
    chunk_size=1000, label="B-Spline SVGD 3D"
)

# ============================================================================
# 6. NumPy Energy for Final Assessment
# ============================================================================

def compute_energy_and_grad(X_flat, T_steps):
    """
    NumPy energy evaluation for final metrics (no gradient needed for B-spline).
    """
    X = X_flat.reshape(T_steps, DIM)
    energy = 0.0

    energy += W_SMOOTH * compute_smoothness_cost_numpy(X)

    ergodic_metric = compute_ergodic_metric_numpy(X, k_indices, Lambda_k, phi_k)
    energy += W_ERGODIC * ergodic_metric

    energy += W_BOUNDARY * compute_boundary_cost_numpy(X)

    if USE_OBSTACLE:
        energy += W_OBSTACLE * compute_obstacle_cost_numpy(X, OBSTACLE_CENTER, OBSTACLE_RADIUS)

    grad = np.zeros_like(X).ravel()  # Placeholder (not used in B-spline mode)
    return energy, ergodic_metric, grad

# ============================================================================
# 7. Master Benchmark Function
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle

    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape"]
    results = {}
    benchmark_data = {}

    print(f"3D SVGD (B-Spline) | {N_PARTICLES} particles, T={T}, K={NUM_CONTROL_POINTS}, {N_ITERS} iters")
    print(f"Fourier: K={K_FOURIER}, total modes={len(k_indices)}")
    print(f"Weights: ergodic={W_ERGODIC}, smooth={W_SMOOTH}, boundary={W_BOUNDARY}, obstacle={W_OBSTACLE}")
    print(f"Adam: lr={ADAM_LR}, beta1={ADAM_BETA1}, beta2={ADAM_BETA2}")
    print("-" * 65)

    sim_fn = jax.jit(vmap(lambda C, s0: forward_sim_nd(C, s0, B_mat, dt, DIM), in_axes=(0, 0)))

    for strat in strategies:
        print(f"Running strategy: {strat}")
        t_start = time.time()

        # 1. Generate initial positions (3D)
        init_p, base_t = get_initialization(strat, N_PARTICLES, T, dim=DIM, noise_std=INIT_NOISE_STD)
        pos_trajs = init_p.reshape(N_PARTICLES, T, DIM)

        # 2. Extract B-spline control points
        C_init, x0_init = init_bspline_from_positions_nd(pos_trajs, dt, B_mat, DIM)

        C_all = jnp.array(C_init)
        x0_all = jnp.array(x0_init)

        # 3. Simulate initial trajectories
        initial_x_trajs = np.array(sim_fn(C_all, x0_all))
        initial_pos = initial_x_trajs[:, :, :DIM].reshape(N_PARTICLES, -1)

        # 4. Optimize
        C_all_opt, energy_log = optimize_C_all(C_all, x0_all, label_override=f"B-Spline SVGD 3D ({strat})")

        # 5. Simulate final trajectories
        final_x_trajs = np.array(sim_fn(C_all_opt, x0_all))
        final_pos = final_x_trajs[:, :, :DIM].reshape(N_PARTICLES, -1)

        # 6. Evaluate
        final_metrics = [compute_energy_and_grad(final_pos[i], T) for i in range(N_PARTICLES)]
        final_E = np.array([m[0] for m in final_metrics])
        final_erg = np.array([m[1] for m in final_metrics])

        best = int(np.argmin(final_erg))
        elapsed = time.time() - t_start

        print(f"  -> Best total energy: {final_E[best]:.3f} | Best pure ergodic: {final_erg[best]:.5f} (Time: {elapsed:.2f}s)\n")

        results[strat] = {
            'initial': initial_pos,
            'base_traj': base_t,
            'final': final_pos,
            'best_idx': best,
            'energy_log': np.array(energy_log).tolist()
        }

        benchmark_data[strat] = {
            'mean_cost': float(np.mean(final_E)),
            'best_cost': float(final_E[best]),
            'time_s': float(elapsed)
        }

        if save_npy:
            np.save(os.path.join(out_dir, f"{strat}_trajs.npy"), final_pos)

    # ============================================================================
    # 8. 3D Visualization
    # ============================================================================

    colors = plt.cm.rainbow(np.linspace(0, 1, N_PARTICLES))

    def plot_particles_3d(ax, parts, title, highlight_best=False, best_idx=-1, base_traj=None):
        """Plot 3D trajectories on an Axes3D."""
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
    fig.suptitle(f"B-Spline SVGD 3D | Shape: {TARGET_SHAPE} | Iters: {N_ITERS}\n"
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
    out_path = os.path.join(out_dir, f'svgd_bspline_3d_{TARGET_SHAPE}_comparison_{N_ITERS}_{timestamp}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T': T,
            'DIM': DIM,
            'N_PARTICLES': N_PARTICLES,
            'N_ITERS': N_ITERS,
            'NUM_CONTROL_POINTS': NUM_CONTROL_POINTS,
            'K_FOURIER': K_FOURIER,
            'W_ERGODIC': W_ERGODIC,
            'W_SMOOTH': W_SMOOTH,
            'W_BOUNDARY': W_BOUNDARY
        }, f, indent=4)

    return benchmark_data


if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    TARGET_PROJECTION = os.environ.get('TARGET_PROJECTION', 'plane')
    out_name = f'/home/philipp/Documents/Uni/Master_thesis/results/SE3_SVGD_BSpline_3D_{TARGET_SHAPE}_{TARGET_PROJECTION}'
    run_benchmark(save_npy=True, out_dir=out_name)
