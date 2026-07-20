"""
2D Stein Variational Ergodic Coverage (TSVEC) — Letter "N" target

Optimizes trajectory particles via SVGD so their time-averaged spatial
statistics match a target distribution shaped like the letter "N".

Key design choices:
  - Diverse initialization (diagonal, circle, zigzag, etc.) to avoid
    symmetric local minima.
  - Adam optimizer inside SVGD for adaptive step sizes.
  - Low smoothness weight so trajectories can fold and sweep.
  - Long trajectories (T=300) for dense coverage.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import sys
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


grid_res = 200
_xs = np.linspace(0, 1, grid_res)
_ys = np.linspace(0, 1, grid_res)
Xg, Yg = np.meshgrid(_xs, _ys)
Zg = target_distribution(Xg, Yg)

# ============================================================================
# 2. Ergodic Metric — Fourier Decomposition
# ============================================================================

K = 10
k_indices = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)])
Lambda_k = (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)


def fourier_basis(pts):
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)


def fourier_basis_grad(pts):
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    c, s = np.cos(args), np.sin(args)
    gx = -np.pi * k_indices[None, :, 0] * s[:, :, 0] * c[:, :, 1]
    gy = -np.pi * k_indices[None, :, 1] * c[:, :, 0] * s[:, :, 1]
    return np.stack([gx, gy], axis=-1)


_grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
_grid_w = Zg.ravel()
_grid_w = _grid_w / _grid_w.sum()
phi_k = np.sum(_grid_w[:, None] * fourier_basis(_grid_pts), axis=0)

# ============================================================================
# 3. Hyperparameters
# ============================================================================

T = 100           # long enough to sweep and fill the N
N_PARTICLES = 10
N_ITERS = 600

# Adam parameters
ADAM_LR = 2e-3
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Energy weights — low smoothness so trajectories can fold freely
W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0

# Obstacle settings
USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

# Initialization settings
INIT_NOISE_STD = 0.02

# ============================================================================
# 4. Diverse Initialization
# ============================================================================




# ============================================================================
# 5. Energy Function & Analytic Gradient
# ============================================================================


def compute_energy_and_grad(X_flat, T):
    X = X_flat.reshape(T, 2)
    grad = np.zeros_like(X)
    energy = 0.0

    # ---- Smoothness (acceleration penalty) ----
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    energy += W_SMOOTH * np.sum(accel ** 2)
    grad[:-2] += 2 * W_SMOOTH * accel
    grad[1:-1] -= 4 * W_SMOOTH * accel
    grad[2:] += 2 * W_SMOOTH * accel

    # ---- Ergodic cost ----
    Fk = fourier_basis(X)
    c_k = np.mean(Fk, axis=0)
    diff_k = c_k - phi_k
    energy += W_ERGODIC * 0.5 * np.sum(Lambda_k * diff_k ** 2)
    Fk_g = fourier_basis_grad(X)
    w_diff = Lambda_k * diff_k
    grad += W_ERGODIC * (1.0 / T) * np.einsum('k,tkd->td', w_diff, Fk_g)

    # ---- Boundary penalty ----
    margin = 0.03
    lo = np.minimum(X - margin, 0.0)
    hi = np.maximum(X - (1.0 - margin), 0.0)
    energy += W_BOUNDARY * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))
    grad += W_BOUNDARY * (lo + hi)

    # ---- Obstacle penalty ----
    if USE_OBSTACLE:
        dx = X[:, 0] - OBSTACLE_CENTER[0]
        dy = X[:, 1] - OBSTACLE_CENTER[1]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq + 1e-12)
        violation = np.maximum(OBSTACLE_RADIUS - dist, 0.0)
        
        energy += W_OBSTACLE * 0.5 * np.sum(violation**2)
        grad[:, 0] += W_OBSTACLE * violation * (-dx / dist)
        grad[:, 1] += W_OBSTACLE * violation * (-dy / dist)

    return energy, grad.ravel()


# ============================================================================
# 6. SVGD with Adam
# ============================================================================


def svgd_step(particles, T):
    Np, D = particles.shape

    # Pairwise kernel with median bandwidth + floor
    sq = squareform(pdist(particles, 'sqeuclidean'))
    pos = sq[sq > 0]
    med = np.median(pos) if len(pos) > 0 else 1.0
    h = max(med / np.log(Np + 1), 0.1)

    K_mat = np.exp(-sq / h)

    # Scores
    scores = np.zeros_like(particles)
    energies = np.zeros(Np)
    for i in range(Np):
        E, g = compute_energy_and_grad(particles[i], T)
        scores[i] = -g
        energies[i] = E

    # SVGD update: attractive + repulsive
    update = np.zeros_like(particles)
    for i in range(Np):
        for j in range(Np):
            update[i] += K_mat[j, i] * scores[j]
            update[i] += K_mat[j, i] * (-2.0 / h) * (particles[j] - particles[i])
    update /= Np

    return update, energies


# ============================================================================
# 7. Run Optimization
# ============================================================================

def run_svgd(initial_particles_flat, T, N_ITERS):
    particles = initial_particles_flat.copy()
    m = np.zeros_like(particles)
    v = np.zeros_like(particles)
    energy_log = []
    
    for it in range(N_ITERS):
        delta, energies = svgd_step(particles, T)
        mx = np.max(np.abs(delta))
        if mx > 200:
            delta *= 200.0 / mx

        t_adam = it + 1
        m = ADAM_BETA1 * m + (1 - ADAM_BETA1) * delta
        v = ADAM_BETA2 * v + (1 - ADAM_BETA2) * delta ** 2
        m_hat = m / (1 - ADAM_BETA1 ** t_adam)
        v_hat = v / (1 - ADAM_BETA2 ** t_adam)
        particles += ADAM_LR * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

        particles = particles.reshape(N_PARTICLES, T, 2)
        particles = np.clip(particles, 0.02, 0.98)
        particles = particles.reshape(N_PARTICLES, -1)
        energy_log.append(np.mean(energies))
        
    return particles, energy_log

# ============================================================================
# 7. Master Benchmark Function
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle
    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape", "polynomial", "rrt"]
    results = {}
    benchmark_data = {}

    print(f"2D TSVEC  |  {N_PARTICLES} particles, T={T}, K={K}, {N_ITERS} iters")
    print(f"Weights:  ergodic={W_ERGODIC}, smooth={W_SMOOTH}, boundary={W_BOUNDARY}, obstacle={W_OBSTACLE}")
    print(f"Adam:     lr={ADAM_LR}, β1={ADAM_BETA1}, β2={ADAM_BETA2}")
    print("-" * 65)

    for strat in strategies:
        print(f"Running strategy: {strat}")
        t_start = time.time()
        
        init_p, base_t = get_initialization(strat, N_PARTICLES, T, noise_std=INIT_NOISE_STD)
        final_p, e_log = run_svgd(init_p, T, N_ITERS)
        
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
    # 8. Visualization
    # ============================================================================

    fig, axes = plt.subplots(len(strategies), 2, figsize=(12, 5 * len(strategies)))
    cmap = 'YlOrRd'
    colors = plt.cm.rainbow(np.linspace(0, 1, N_PARTICLES))

    def plot_particles(ax, parts, title, highlight_best=False, best_idx=-1, base_traj=None):
        ax.contourf(Xg, Yg, Zg, levels=30, cmap=cmap, alpha=0.6)
        ax.contour(Xg, Yg, Zg, levels=6, colors='k', linewidths=0.3, alpha=0.3)
        if USE_OBSTACLE:
            circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='gray', alpha=0.8, zorder=5)
            ax.add_patch(circle)
            
        if base_traj is not None:
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--', color='white', lw=3.0, zorder=8)
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--', color='black', lw=1.5, zorder=9, label='Base Trajectory')
            
        for i in range(N_PARTICLES):
            tr = parts[i].reshape(T, 2)
            lw = 2.5 if (highlight_best and i == best_idx) else 0.8
            al = 1.0 if (highlight_best and i == best_idx) else 0.4
            ax.plot(tr[:, 0], tr[:, 1], '-', color=colors[i], lw=lw, alpha=al)
            ax.plot(tr[0, 0], tr[0, 1], 'o', color=colors[i], ms=4)
        if highlight_best and best_idx >= 0:
            best_tr = parts[best_idx].reshape(T, 2)
            ax.plot(best_tr[:, 0], best_tr[:, 1], '-', color=colors[best_idx],
                    lw=2.5, label=f'Best (#{best_idx})', zorder=10)
            ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('y')

    for row, strat in enumerate(strategies):
        res = results[strat]
        ax = axes[row, 0]
        plot_particles(ax, res['initial'], f'[{strat}] Initial', base_traj=res['base_traj'])
        if res['base_traj'] is not None:
            ax.legend(loc='lower right', fontsize=9)
        
        ax = axes[row, 1]
        plot_particles(ax, res['final'], f'[{strat}] Final', highlight_best=True, best_idx=res['best_idx'])

    plt.tight_layout()
    out = os.path.join(out_dir, f'tsvec_2d_{os.environ.get("TARGET_SHAPE", "N")}_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T': T,
            'N_PARTICLES': N_PARTICLES,
            'N_ITERS': N_ITERS,
            'W_ERGODIC': W_ERGODIC,
            'W_SMOOTH': W_SMOOTH,
            'W_BOUNDARY': W_BOUNDARY
        }, f, indent=4)
        
    return benchmark_data

if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    run_benchmark(out_dir=f'/home/philipp/Documents/Uni/Master_thesis/results/SE3_SVGD_{TARGET_SHAPE}')
