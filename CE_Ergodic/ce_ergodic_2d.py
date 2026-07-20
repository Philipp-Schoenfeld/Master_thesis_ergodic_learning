#!/usr/bin/env python3
"""
2D CE Ergodic Coverage — Letter "N" target

Aligned mathematically with the STOEC repository:
- D_KL(xi || Gamma) (Mode-covering reverse KL Divergence)
- Fast identity sensor footprint (accumarray/2D histogram)
- m=10 parameterized Dubins motion primitives
- Full CE covariance matrix update
"""

import time
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home/philipp/Documents/Uni/Master_thesis")
from init_strategies import get_initialization

np.random.seed(42)

# ============================================================================
# 1. Target Distribution
# ============================================================================

TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')

if TARGET_SHAPE == 'N':
    N_SEGMENTS = [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.25, 0.85], [0.75, 0.15]),
        ([0.75, 0.15], [0.75, 0.85]),
    ]
elif TARGET_SHAPE == 'H':
    N_SEGMENTS = [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
        ([0.25, 0.50], [0.75, 0.50]),
    ]
elif TARGET_SHAPE == 'II':
    N_SEGMENTS = [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
    ]
else:
    raise ValueError(f"Unknown target shape: {TARGET_SHAPE}")
STROKE_WIDTH = 0.045

def _dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (len_sq + 1e-12), 0, 1)
    return np.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)

def target_distribution(x, y):
    d_min = np.full_like(x, 1e10)
    for (ax, ay), (bx, by) in N_SEGMENTS:
        d_min = np.minimum(d_min, _dist_to_segment(x, y, ax, ay, bx, by))
    return np.exp(-d_min ** 2 / (2 * STROKE_WIDTH ** 2))

grid_res = 50
_xs = np.linspace(0, 1, grid_res)
_ys = np.linspace(0, 1, grid_res)
Xg, Yg = np.meshgrid(_xs, _ys)

Zg = target_distribution(Xg, Yg)
_grid_w = Zg.ravel() / (np.sum(Zg) + 1e-12)

# ============================================================================
# 1.5. Ergodic Metric — Fourier Decomposition
# ============================================================================

K_WAVES = 10
k_indices = np.array([[k1, k2] for k1 in range(K_WAVES) for k2 in range(K_WAVES)])
Lambda_k = (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)

def fourier_basis(pts):
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)

_grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
phi_k = np.sum(_grid_w[:, None] * fourier_basis(_grid_pts), axis=0)

# ============================================================================
# 2. Hyperparameters
# ============================================================================

T_WAYPOINTS = 100       # Trajectory length
DT = 0.02               # Time step duration
M_PRIMITIVES = 10       # Number of motion primitives
STEPS_PER_PRIMITIVE = T_WAYPOINTS // M_PRIMITIVES

N_INDEPENDENT_RUNS = 10

USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

# CE Method Hyperparameters
CE_ITERS = 20
CE_SAMPLES = 100
CE_ELITE_FRAC = 0.1
K_ELITE = int(CE_SAMPLES * CE_ELITE_FRAC)
CE_ALPHA = 0.8          # Smoothing factor for updates to mu, sigma

# Robot Dynamics
V_MAX = 5.0
W_MAX = 10.0

# ============================================================================
# 3. Dubins Dynamics & Sensor Footprint
# ============================================================================

def simulate_dubins(x0, y0, theta0, Z):
    """
    Simulate Dubins dynamics using m discrete primitives.
    Z: shape (M_SAMPLES, 2 * M_PRIMITIVES)
    """
    M = Z.shape[0]
    V = Z[:, :M_PRIMITIVES]
    W = Z[:, M_PRIMITIVES:]

    trajs = np.zeros((M, T_WAYPOINTS, 2))
    
    theta = np.full(M, theta0)
    x = np.full(M, x0)
    y = np.full(M, y0)

    step_idx = 0
    for i in range(M_PRIMITIVES):
        for _ in range(STEPS_PER_PRIMITIVE):
            theta += W[:, i] * DT
            x += V[:, i] * np.cos(theta) * DT
            y += V[:, i] * np.sin(theta) * DT
            if step_idx < T_WAYPOINTS:
                trajs[:, step_idx, 0] = x
                trajs[:, step_idx, 1] = y
                step_idx += 1

    return trajs

def compute_costs(trajs):
    """
    Compute KL Divergence costs using the exact STOEC methodology.
    """
    M = trajs.shape[0]
    costs = np.zeros(M)
    
    for i in range(M):
        traj = trajs[i]
        
        # Continuous boundary penalty for safety
        lo = np.minimum(traj, 0.0)
        hi = np.maximum(traj - 1.0, 0.0)
        boundary_violations = np.sum(lo**2 + hi**2)
        if boundary_violations > 0:
            costs[i] = 1000.0 + 1000.0 * boundary_violations
            continue
            
        # Ergodic cost (Spectral Decomposition)
        Fk = fourier_basis(traj)
        c_k = np.mean(Fk, axis=0)
        diff_k = c_k - phi_k
        
        # Weight by 600.0 to match other testbenches' cost scales
        ergodic_cost = 600.0 * 0.5 * np.sum(Lambda_k * diff_k ** 2)
        
        # Obstacle penalty
        obstacle_cost = 0.0
        if USE_OBSTACLE:
            dx = traj[:, 0] - OBSTACLE_CENTER[0]
            dy = traj[:, 1] - OBSTACLE_CENTER[1]
            dist = np.sqrt(dx**2 + dy**2 + 1e-12)
            violation = np.maximum(OBSTACLE_RADIUS - dist, 0.0)
            obstacle_cost = W_OBSTACLE * 0.5 * np.sum(violation**2)
        
        costs[i] = ergodic_cost + obstacle_cost

    return costs

# ============================================================================
# 4. CE Optimization Loop
# ============================================================================

def init_controls_from_path(path):
    """
    Extracts initial (v, w) controls from a positional path.
    """
    v_all = np.zeros(M_PRIMITIVES)
    w_all = np.zeros(M_PRIMITIVES)
    
    dx = np.diff(path[:, 0])
    dy = np.diff(path[:, 1])
    theta = np.unwrap(np.arctan2(dy, dx))
    
    for i in range(M_PRIMITIVES):
        start = i * STEPS_PER_PRIMITIVE
        end = min((i + 1) * STEPS_PER_PRIMITIVE, len(theta) - 1)
        
        if end > start:
            v_all[i] = np.mean(np.sqrt(dx[start:end]**2 + dy[start:end]**2) / DT)
            w_all[i] = (theta[end] - theta[start]) / ((end - start) * DT)
            
    return np.concatenate([v_all, w_all])

def run_ce_optimization(initial_path):
    """
    Runs CE optimization for a SINGLE robot trajectory using full covariance matrix.
    """
    mu = init_controls_from_path(initial_path)
    d = len(mu)
    
    # Initialize full covariance matrix
    C = np.eye(d)
    
    x0, y0 = initial_path[0, 0], initial_path[0, 1]
    
    dx0 = initial_path[1, 0] - initial_path[0, 0]
    dy0 = initial_path[1, 1] - initial_path[0, 1]
    theta0 = np.arctan2(dy0, dx0)
    
    best_traj = None
    best_cost = float('inf')
    
    for it in range(CE_ITERS):
        # 2. Sample (multivariate)
        Z = np.random.multivariate_normal(mu, C, size=CE_SAMPLES)
        
        # Apply strict physical bounds to Z
        V_part = np.clip(Z[:, :M_PRIMITIVES], -V_MAX, V_MAX)
        W_part = np.clip(Z[:, M_PRIMITIVES:], -W_MAX, W_MAX)
        Z = np.concatenate([V_part, W_part], axis=1)
        
        # 3. Simulate and evaluate
        trajs = simulate_dubins(x0, y0, theta0, Z)
        costs = compute_costs(trajs)
        
        # 4. Select Elites
        elite_indices = np.argsort(costs)[:K_ELITE]
        elite_Z = Z[elite_indices]
        
        if costs[elite_indices[0]] < best_cost:
            best_cost = costs[elite_indices[0]]
            best_traj = trajs[elite_indices[0]].copy()
            
        # 5. Update distribution (mean & FULL covariance)
        new_mu = np.mean(elite_Z, axis=0)
        new_C = np.cov(elite_Z, rowvar=False) + np.eye(d) * 1e-4
        
        mu = CE_ALPHA * new_mu + (1 - CE_ALPHA) * mu
        C = CE_ALPHA * new_C + (1 - CE_ALPHA) * C
        
        if (it + 1) % 10 == 0:
            print(f"      CE Iter {it+1}/{CE_ITERS} | Best Cost: {best_cost:.3f}")

    return best_traj, best_cost

# ============================================================================
# 5. Master Benchmark Function
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle
    
    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape", "polynomial", "rrt"]
    results = {}
    benchmark_data = {}

    print(f"2D CE-Ergodic | {N_INDEPENDENT_RUNS} independent runs per strategy")
    print(f"CE Method:    samples={CE_SAMPLES}, elites={K_ELITE}, iters={CE_ITERS}, Full Covariance")
    print(f"Dynamics:     T={T_WAYPOINTS}, m={M_PRIMITIVES}, dt={DT}")
    print("-" * 65)

    for strat in strategies:
        print(f"\nRunning strategy: {strat}")
        t_start = time.time()
        
        init_paths, base_t = get_initialization(strat, N_INDEPENDENT_RUNS, T_WAYPOINTS, noise_std=0.04)
        init_paths = init_paths.reshape(N_INDEPENDENT_RUNS, T_WAYPOINTS, 2)
        
        final_trajs = np.zeros_like(init_paths)
        run_costs = np.zeros(N_INDEPENDENT_RUNS)
        
        for i in range(N_INDEPENDENT_RUNS):
            opt_traj, best_cost = run_ce_optimization(init_paths[i])
            final_trajs[i] = opt_traj
            run_costs[i] = best_cost
            
        elapsed = time.time() - t_start
        print(f"  [DONE] Mean Cost: {np.mean(run_costs):.3f} (Time: {elapsed:.2f}s)")
        
        results[strat] = {
            'initial': init_paths.reshape(N_INDEPENDENT_RUNS, -1),
            'base_traj': base_t,
            'final': final_trajs.reshape(N_INDEPENDENT_RUNS, -1),
            'best_idx': int(np.argmin(run_costs))
        }
        
        benchmark_data[strat] = {
            'mean_cost': float(np.mean(run_costs)),
            'best_cost': float(np.min(run_costs)),
            'time_s': float(elapsed)
        }
        
        if save_npy:
            np.save(os.path.join(out_dir, f"{strat}_trajs.npy"), final_trajs)

    # ============================================================================
    # 6. Comparison-Grid Visualisation
    # ============================================================================

    Xg_vis, Yg_vis = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    Zg_vis = target_distribution(Xg_vis, Yg_vis)

    fig, axes = plt.subplots(len(strategies), 2, figsize=(12, 5 * len(strategies)))
    cmap = 'YlOrRd'
    colors = plt.cm.rainbow(np.linspace(0, 1, N_INDEPENDENT_RUNS))

    def plot_particles(ax, parts, title, highlight_best=False, best_idx=-1, base_traj=None):
        ax.contourf(Xg_vis, Yg_vis, Zg_vis, levels=30, cmap=cmap, alpha=0.6)
        ax.contour(Xg_vis, Yg_vis, Zg_vis, levels=6, colors='k', linewidths=0.3, alpha=0.3)
        
        if USE_OBSTACLE:
            circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='gray', alpha=0.8, zorder=5)
            ax.add_patch(circle)
            
        if base_traj is not None:
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--', color='white', lw=3.0, zorder=8)
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--', color='black', lw=1.5, zorder=9, label='Base Trajectory')
            
        for i in range(N_INDEPENDENT_RUNS):
            tr = parts[i].reshape(T_WAYPOINTS, 2)
            lw = 2.5 if (highlight_best and i == best_idx) else 0.8
            al = 1.0 if (highlight_best and i == best_idx) else 0.4
            ax.plot(tr[:, 0], tr[:, 1], '-', color=colors[i], lw=lw, alpha=al)
            ax.plot(tr[0, 0], tr[0, 1], 'o', color=colors[i], ms=4)
            
        if highlight_best and best_idx >= 0:
            best_tr = parts[best_idx].reshape(T_WAYPOINTS, 2)
            ax.plot(best_tr[:, 0], best_tr[:, 1], '-', color=colors[best_idx],
                    lw=2.5, label=f'Best (#{best_idx})', zorder=10)
            ax.legend(loc='lower right', fontsize=9)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    for row, strat in enumerate(strategies):
        res = results[strat]
        ax = axes[row, 0]
        plot_particles(ax, res['initial'], f'[{strat}] Initial (10 independent seeds)', base_traj=res['base_traj'])
        if res['base_traj'] is not None:
            ax.legend(loc='lower right', fontsize=9)
        
        ax = axes[row, 1]
        plot_particles(ax, res['final'], f'[{strat}] Final CE trajectories', highlight_best=True, best_idx=res['best_idx'])

    plt.tight_layout()
    out = os.path.join(out_dir, f'ce_ergodic_2d_{os.environ.get("TARGET_SHAPE", "N")}_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T_WAYPOINTS': T_WAYPOINTS,
            'DT': DT,
            'M_PRIMITIVES': M_PRIMITIVES,
            'CE_ITERS': CE_ITERS,
            'CE_SAMPLES': CE_SAMPLES
        }, f, indent=4)
        
    return benchmark_data

if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    run_benchmark(out_dir=f'/home/philipp/Documents/Uni/Master_thesis/results/CE_Ergodic_{TARGET_SHAPE}')
