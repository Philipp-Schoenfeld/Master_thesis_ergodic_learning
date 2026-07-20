#!/usr/bin/env python3
"""
2D Signature Kernel SV-CMA-ES — Letter "N" target

Implements "Diversifying Parallel Ergodic Search: A Signature Kernel
Evolution Strategy" on a 2D domain.

Three core components:
  1. Spectral ergodic cost  — Fourier-based ergodicity metric.
  2. Signature kernel (PDE) — path-aware repulsive kernel via a finite-
     difference solver for the hyperbolic PDE formulation.
  3. SV-CMA-ES update       — Stein Variational CMA-ES that uses the
     signature kernel for inter-particle repulsion and CMA-ES sampling
     as a gradient-free score surrogate.

Uses the same five initialization strategies and comparison-grid
visualisation as the SE3_SVGD, Stein_Flow_matching, HEDAC, and
LB_Ergodic testbenches.

Dependencies: NumPy only (+ matplotlib for plots).
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


# Visualisation grid
grid_res = 200
_xs_vis = np.linspace(0, 1, grid_res)
_ys_vis = np.linspace(0, 1, grid_res)
Xg, Yg = np.meshgrid(_xs_vis, _ys_vis)
Zg = target_distribution(Xg, Yg)

# ============================================================================
# 2. Ergodic Metric — Fourier Decomposition  (same as the other methods)
# ============================================================================

K = 10
k_indices = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)])
Lambda_k = (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)


def fourier_basis(pts):
    """Evaluate cosine Fourier basis  F_k(s) = prod_i cos(s_i k_i pi / L_i).
    pts: (M, 2),  returns (M, K^2)."""
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)


def fourier_basis_grad(pts):
    """Gradient of Fourier basis w.r.t. pts.  Returns (M, K^2, 2)."""
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    c, s = np.cos(args), np.sin(args)
    gx = -np.pi * k_indices[None, :, 0] * s[:, :, 0] * c[:, :, 1]
    gy = -np.pi * k_indices[None, :, 1] * c[:, :, 0] * s[:, :, 1]
    return np.stack([gx, gy], axis=-1)


# Target Fourier coefficients  mu_k = int F_k(s) d pi(s)
_grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
_grid_w = Zg.ravel()
_grid_w = _grid_w / _grid_w.sum()
phi_k = np.sum(_grid_w[:, None] * fourier_basis(_grid_pts), axis=0)

# ============================================================================
# 3. Hyperparameters
# ============================================================================

N_PARTICLES = 10      # number of trajectory particles
T = 100               # trajectory time-steps (waypoints)

# --- SV-CMA-ES ---
N_OUTER_ITERS = 150   # Stein variational outer iterations
M_SAMPLES = 20        # CMA-ES sub-population size per particle
ELITE_FRAC = 0.5      # top fraction used as elite
ALPHA_X = 0.005       # Stein step size for trajectory update (Adam LR)
SIGMA_INIT = 0.03     # initial CMA-ES step size
SIGMA_MIN = 1e-4      # minimum step size clamp
SIGMA_MAX = 0.05      # maximum step size clamp

# --- SPSA for signature kernel gradient ---
SPSA_EPS = 1e-3       # perturbation size for finite-difference kernel grad

# --- Energy weights (same as the other methods for comparability) ---
W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0

# --- Initialization ---
INIT_NOISE_STD = 0.02

# --- Obstacle ---
USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

# ============================================================================
# 4. Ergodic Cost Function
# ============================================================================


def ergodic_cost(X_flat, T):
    """
    Total trajectory cost = ergodic + smoothness + boundary.

    Parameters
    ----------
    X_flat : ndarray (T*2,)   — flattened trajectory
    T      : int              — number of time-steps

    Returns
    -------
    cost : float
    """
    X = X_flat.reshape(T, 2)
    cost = 0.0

    # ---- Smoothness (acceleration penalty) ----
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    cost += W_SMOOTH * np.sum(accel ** 2)

    # ---- Ergodic cost ----
    Fk = fourier_basis(X)                  # (T, K^2)
    c_k = np.mean(Fk, axis=0)             # time-averaged Fourier coefficients
    diff_k = c_k - phi_k
    cost += W_ERGODIC * 0.5 * np.sum(Lambda_k * diff_k ** 2)

    # ---- Boundary penalty ----
    margin = 0.03
    lo = np.minimum(X - margin, 0.0)
    hi = np.maximum(X - (1.0 - margin), 0.0)
    cost += W_BOUNDARY * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))
    
    # ---- Obstacle penalty ----
    if USE_OBSTACLE:
        dx = X[:, 0] - OBSTACLE_CENTER[0]
        dy = X[:, 1] - OBSTACLE_CENTER[1]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        violation = np.maximum(OBSTACLE_RADIUS - dist, 0.0)
        cost += W_OBSTACLE * 0.5 * np.sum(violation**2)

    return cost


def compute_energy_and_grad(X_flat, T):
    """Full energy + analytic gradient (for post-hoc evaluation & comparison)."""
    X = X_flat.reshape(T, 2)
    grad = np.zeros_like(X)
    energy = 0.0

    # Smoothness
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    energy += W_SMOOTH * np.sum(accel ** 2)
    grad[:-2] += 2 * W_SMOOTH * accel
    grad[1:-1] -= 4 * W_SMOOTH * accel
    grad[2:] += 2 * W_SMOOTH * accel

    # Ergodic
    Fk = fourier_basis(X)
    c_k = np.mean(Fk, axis=0)
    diff_k = c_k - phi_k
    energy += W_ERGODIC * 0.5 * np.sum(Lambda_k * diff_k ** 2)
    Fk_g = fourier_basis_grad(X)
    w_diff = Lambda_k * diff_k
    grad += W_ERGODIC * (1.0 / T) * np.einsum('k,tkd->td', w_diff, Fk_g)

    # Boundary
    margin = 0.03
    lo = np.minimum(X - margin, 0.0)
    hi = np.maximum(X - (1.0 - margin), 0.0)
    energy += W_BOUNDARY * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))
    grad += W_BOUNDARY * (lo + hi)

    return energy, grad.ravel()


# ============================================================================
# 5. Signature Kernel — PDE Finite-Difference Solver
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

    # Discrete velocity vectors
    dx = np.diff(x, axis=0)   # (Tx-1, 2)
    dy = np.diff(y, axis=0)   # (Ty-1, 2)

    # Kernel matrix: K[i, j] for step i of x, step j of y
    # Size is Tx × Ty  (including the "start" boundary)
    M = np.ones((Tx, Ty))

    # Fill using the discrete PDE recursion:
    #   M[i+1, j+1] = M[i, j+1] + M[i+1, j] + M[i, j] * (<dx_i, dy_j> - 1)
    # This can be vectorized over j using np.cumsum
    D = dx @ dy.T  # (Tx-1, Ty-1)
    for i in range(Tx - 1):
        A = M[i, 1:] + M[i, :-1] * (D[i] - 1.0)
        M[i + 1, 1:] = M[i + 1, 0] + np.cumsum(A)

    return M[-1, -1]


def signature_kernel_matrix(particles_list):
    """
    Compute the N×N Gram matrix of signature kernels.

    Parameters
    ----------
    particles_list : list of ndarray, each (T, 2)

    Returns
    -------
    K : ndarray (N, N) — Gram matrix
    """
    N = len(particles_list)
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = signature_kernel_pde(particles_list[i], particles_list[i])
        for j in range(i + 1, N):
            K[i, j] = signature_kernel_pde(particles_list[i], particles_list[j])
            K[j, i] = K[i, j]
    return K


def signature_kernel_grad_spsa(x_j, x_i, eps=SPSA_EPS):
    """
    Approximate  ∇_{x_j} k^sig(x_j, x_i)  using SPSA (Simultaneous
    Perturbation Stochastic Approximation).

    Returns an array of shape (T, 2) — the gradient w.r.t. x_j.
    """
    T_len, dim = x_j.shape
    # Random perturbation direction (Rademacher)
    delta = np.sign(np.random.randn(T_len, dim))
    delta[delta == 0] = 1.0

    k_plus = signature_kernel_pde(x_j + eps * delta, x_i)
    k_minus = signature_kernel_pde(x_j - eps * delta, x_i)

    grad = (k_plus - k_minus) / (2.0 * eps) * (1.0 / delta)
    return grad


# ============================================================================
# 6. SV-CMA-ES Step
# ============================================================================


def sv_cma_es_step(particles, sigmas, T):
    """
    Perform one iteration of the Stein Variational CMA-ES update.

    For each particle (trajectory):
      1. Sample m perturbations from N(x_i, sigma_i^2 I).
      2. Evaluate fitness, rank, compute CMA-ES mean shift (Δ_i).
      3. Update step-size sigma_i.
    Then apply the Stein variational update using the signature kernel.

    Parameters
    ----------
    particles : ndarray (N, T*2)  — current trajectory particles (flat)
    sigmas    : ndarray (N,)      — CMA-ES step sizes
    T         : int               — time-steps

    Returns
    -------
    new_particles : ndarray (N, T*2)
    new_sigmas    : ndarray (N,)
    mean_energy   : float
    """
    N, D = particles.shape
    m = M_SAMPLES
    n_elite = max(1, int(ELITE_FRAC * m))

    # --- Logarithmic recombination weights for the elite ---
    raw_w = np.log(n_elite + 0.5) - np.log(np.arange(1, n_elite + 1))
    raw_w = raw_w / raw_w.sum()

    # ---- Phase 1: CMA-ES sampling & mean-shift estimation ----
    deltas = np.zeros((N, D))
    all_energies = np.zeros(N)

    for i in range(N):
        # Sample m offspring around particle i
        noise = np.random.randn(m, D)
        samples = particles[i][None, :] + sigmas[i] * noise   # (m, D)

        # Clip samples to domain
        samples_clipped = np.clip(samples.reshape(m, T, 2), 0.01, 0.99)
        samples = samples_clipped.reshape(m, D)

        # Evaluate fitness for each sample
        costs = np.array([ergodic_cost(samples[j], T) for j in range(m)])

        # Rank and select elite
        order = np.argsort(costs)
        elite_idx = order[:n_elite]
        all_energies[i] = costs[order[0]]  # best sample energy

        # Weighted mean shift  Δ_i = Σ w_k (ξ_k - x_i)
        elite_offsets = samples[elite_idx] - particles[i]  # (n_elite, D)
        deltas[i] = raw_w @ elite_offsets

        # ---- CMA-ES step-size adaptation (1/5th success rule) ----
        # Simple: adjust sigma based on fraction of elite that improved
        f_parent = ergodic_cost(particles[i], T)
        n_improved = np.sum(costs[elite_idx] < f_parent)
        p_succ = n_improved / n_elite

        # Multiplicative update
        if p_succ > 0.2:
            sigmas[i] *= 1.1   # expand
        else:
            sigmas[i] *= 0.85  # shrink
        sigmas[i] = np.clip(sigmas[i], SIGMA_MIN, SIGMA_MAX)

    # ---- Phase 2: Stein variational update with signature kernel ----
    # Reshape particles to list of (T, 2) paths
    paths = [particles[i].reshape(T, 2) for i in range(N)]

    # Compute signature kernel Gram matrix
    K_sig = signature_kernel_matrix(paths)

    # Compute kernel gradients via SPSA
    # ∇_{x_j} k^sig(x_j, x_i)  for all (j, i) pairs
    kernel_grads = np.zeros((N, N, T, 2))  # [j, i, :, :]
    for j in range(N):
        for i in range(N):
            if i != j:
                kernel_grads[j, i] = signature_kernel_grad_spsa(paths[j], paths[i])
            else:
                # Self-gradient via SPSA as well
                kernel_grads[j, i] = signature_kernel_grad_spsa(paths[j], paths[i])

    # Stein update direction for each particle i:
    #   update_i = (1 / N) * sum_j [ Delta_j * k(x_j, x_i) + nabla_{x_j} k(x_j, x_i) ]
    updates = np.zeros_like(particles)
    for i in range(N):
        for j in range(N):
            # Attractive: score-surrogate (delta) weighted by kernel
            updates[i] += deltas[j] * K_sig[j, i]
            # Repulsive: kernel gradient
            updates[i] += kernel_grads[j, i].ravel()
        updates[i] /= N

    return updates, sigmas, np.mean(all_energies)


# ============================================================================
# 7. Run Optimisation
# ============================================================================


def run_sv_cma_es(initial_particles_flat, T, n_iters):
    """
    Run the full SV-CMA-ES optimisation loop.

    Parameters
    ----------
    initial_particles_flat : ndarray (N, T*2)
    T                      : int
    n_iters                : int

    Returns
    -------
    particles : ndarray (N, T*2)
    energy_log : list of float
    """
    particles = initial_particles_flat.copy()
    N, D = particles.shape
    sigmas = np.full(N, SIGMA_INIT)
    energy_log = []
    
    # Adam parameters
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros_like(particles)
    v = np.zeros_like(particles)

    for it in range(n_iters):
        updates, sigmas, mean_e = sv_cma_es_step(particles, sigmas, T)
        
        # Clip updates to prevent explosion
        mx = np.max(np.abs(updates))
        if mx > 200:
            updates *= 200.0 / mx
            
        # Adam step
        t_adam = it + 1
        m = beta1 * m + (1 - beta1) * updates
        v = beta2 * v + (1 - beta2) * updates**2
        m_hat = m / (1 - beta1**t_adam)
        v_hat = v / (1 - beta2**t_adam)
        
        particles += ALPHA_X * m_hat / (np.sqrt(v_hat) + eps)
        
        # Clip to domain
        particles = particles.reshape(N, T, 2)
        particles = np.clip(particles, 0.01, 0.99)
        particles = particles.reshape(N, D)
        
        energy_log.append(mean_e)

        if (it + 1) % 10 == 0 or it == 0:
            print(f"    iter {it+1:4d}/{n_iters}  "
                  f"mean_E={mean_e:.3f}  "
                  f"σ=[{sigmas.min():.4f}, {sigmas.max():.4f}]")

    return particles, energy_log


# ============================================================================
# 8. Run Over Initialisation Strategies
# ============================================================================

# ============================================================================
# 8. Master Benchmark Function
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle
    
    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape", "polynomial", "rrt"]
    results = {}
    benchmark_data = {}

    print(f"2D SV-CMA-ES  |  {N_PARTICLES} particles, T={T}, K={K}, "
          f"{N_OUTER_ITERS} outer iters")
    print(f"CMA-ES:       m={M_SAMPLES}, elite_frac={ELITE_FRAC}, "
          f"σ_init={SIGMA_INIT}")
    print(f"Stein:        α_x={ALPHA_X}, SPSA_eps={SPSA_EPS}")
    print(f"Weights:      ergodic={W_ERGODIC}, smooth={W_SMOOTH}, "
          f"boundary={W_BOUNDARY}")
    print("-" * 65)

    for strat in strategies:
        print(f"\nRunning strategy: {strat}")
        t_start = time.time()

        # Get initialisation
        init_p, base_t = get_initialization(strat, N_PARTICLES, T,
                                            noise_std=INIT_NOISE_STD)

        # Run SV-CMA-ES
        final_p, e_log = run_sv_cma_es(init_p, T, N_OUTER_ITERS)

        # Evaluate final energies (using the full energy+grad for comparability)
        final_E = np.array([compute_energy_and_grad(final_p[i], T)[0]
                            for i in range(N_PARTICLES)])
        best = int(np.argmin(final_E))
        elapsed = time.time() - t_start

        print(f"  -> Best energy: {final_E[best]:.3f}  "
              f"Mean: {final_E.mean():.3f}  (Time: {elapsed:.2f}s)")

        results[strat] = {
            'initial': init_p,
            'base_traj': base_t,
            'final': final_p,
            'energy_log': e_log,
            'best_idx': best,
        }
        
        benchmark_data[strat] = {
            'mean_cost': float(np.mean(final_E)),
            'best_cost': float(final_E[best]),
            'time_s': float(elapsed)
        }
        
        if save_npy:
            np.save(os.path.join(out_dir, f"{strat}_trajs.npy"), final_p)

    # ============================================================================
    # 9. Comparison-Grid Visualisation
    # ============================================================================

    fig, axes = plt.subplots(len(strategies), 2, figsize=(12, 5 * len(strategies)))
    cmap = 'YlOrRd'
    colors = plt.cm.rainbow(np.linspace(0, 1, N_PARTICLES))

    def plot_particles(ax, parts, title,
                       highlight_best=False, best_idx=-1, base_traj=None):
        ax.contourf(Xg, Yg, Zg, levels=30, cmap=cmap, alpha=0.6)
        ax.contour(Xg, Yg, Zg, levels=6, colors='k', linewidths=0.3, alpha=0.3)

        if USE_OBSTACLE:
            circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='gray', alpha=0.8, zorder=5)
            ax.add_patch(circle)

        if base_traj is not None:
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--',
                    color='white', lw=3.0, zorder=8)
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--',
                    color='black', lw=1.5, zorder=9, label='Base Trajectory')

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

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    for row, strat in enumerate(strategies):
        res = results[strat]

        ax = axes[row, 0]
        plot_particles(ax, res['initial'], f'[{strat}] Initial',
                       base_traj=res['base_traj'])
        if res['base_traj'] is not None:
            ax.legend(loc='lower right', fontsize=9)

        ax = axes[row, 1]
        plot_particles(ax, res['final'], f'[{strat}] SV-CMA-ES Result',
                       highlight_best=True, best_idx=res['best_idx'])

    plt.tight_layout()
    out = os.path.join(out_dir, f'sv_cma_es_2d_{os.environ.get("TARGET_SHAPE", "N")}_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T': T,
            'N_PARTICLES': N_PARTICLES,
            'N_OUTER_ITERS': N_OUTER_ITERS,
            'M_SAMPLES': M_SAMPLES,
            'W_ERGODIC': W_ERGODIC
        }, f, indent=4)
        
    return benchmark_data

if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    run_benchmark(out_dir=f'/home/philipp/Documents/Uni/Master_thesis/results/SigKernel_CMA_{TARGET_SHAPE}')
