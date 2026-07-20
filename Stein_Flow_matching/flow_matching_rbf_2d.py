#!/usr/bin/env python3
"""
2D Flow Matching Ergodic Coverage Testbench with RBFs (Alternative to CNF)

Refactors the discrete Euler LQR flow matching into a Continuous Normalizing Flow 
with a finite-dimensional RBF parameterization of the control inputs.
Solves a dense algebraic system for the optimal control points flow.
"""

import time
import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add Master_thesis parent folder to path to import initialization strategies
sys.path.append("/home/philipp/Documents/Uni/Master_thesis/src")
from init_strategies import get_initialization

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax


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

K_FOURIER = 10
k_indices = np.array([[k1, k2] for k1 in range(K_FOURIER) for k2 in range(K_FOURIER)])
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

T = 100           # trajectory steps
N_PARTICLES = 10  # number of trajectory particles
N_ITERS = int(os.environ.get("N_ITERS", 800000))   # flow matching iterations
step_size = 0.01  # flow matching step size

# RBF Specific
NUM_RBF_CENTERS = 144
DEGREE = 3

W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0
USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

INIT_NOISE_STD = 0.02
dt = 0.05

Q_mat = jnp.diag(jnp.array([1.0, 1.0, 0.001, 0.001]))
R_mat = jnp.diag(jnp.array([0.01, 0.01]))

# ============================================================================
# 4. Energy Function for Final Trajectory Assessment
# ============================================================================

def compute_energy_and_grad(X_flat, T_steps):
    X = X_flat.reshape(T_steps, 2)
    grad = np.zeros_like(X)
    energy = 0.0

    # ---- Smoothness (acceleration penalty) ----
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    energy += W_SMOOTH * np.sum(accel ** 2)

    # ---- Ergodic cost ----
    Fk = fourier_basis(X)
    c_k = np.mean(Fk, axis=0)
    diff_k = c_k - phi_k
    energy += W_ERGODIC * 0.5 * np.sum(Lambda_k * diff_k ** 2)

    # ---- Boundary penalty ----
    margin = 0.03
    lo = np.minimum(X - margin, 0.0)
    hi = np.maximum(X - (1.0 - margin), 0.0)
    energy += W_BOUNDARY * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))

    # ---- Obstacle penalty ----
    if USE_OBSTACLE:
        dx = X[:, 0] - OBSTACLE_CENTER[0]
        dy = X[:, 1] - OBSTACLE_CENTER[1]
        dist_sq = dx**2 + dy**2
        dist = np.sqrt(dist_sq + 1e-12)
        violation = np.maximum(OBSTACLE_RADIUS - dist, 0.0)
        energy += W_OBSTACLE * 0.5 * np.sum(violation**2)

    return energy, grad.ravel()

# ============================================================================
# 5. JAX-differentiable Target Distribution & Stein Gradient
# ============================================================================

if TARGET_SHAPE == 'N':
    N_SEGMENTS_JNP = [(jnp.array(a), jnp.array(b)) for a, b in N_SEGMENTS]
elif TARGET_SHAPE == 'H':
    N_SEGMENTS_JNP = [(jnp.array(a), jnp.array(b)) for a, b in N_SEGMENTS]
elif TARGET_SHAPE == 'II':
    N_SEGMENTS_JNP = [(jnp.array(a), jnp.array(b)) for a, b in N_SEGMENTS]

def jax_dist_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    len_sq = jnp.sum(ab**2)
    t = jnp.clip(jnp.sum(ap * ab) / (len_sq + 1e-12), 0.0, 1.0)
    closest = a + t * ab
    return jnp.sqrt(jnp.sum((p - closest)**2) + 1e-12)

def pdf(x):
    p = x[:2]
    dists = [jax_dist_to_segment(p, a, b) for a, b in N_SEGMENTS_JNP]
    d_min = jnp.min(jnp.stack(dists))
    return jnp.exp(-d_min**2 / (2 * STROKE_WIDTH**2))

def log_pdf(x):
    lpdf = jnp.log(pdf(x) + 1e-12)
    if USE_OBSTACLE:
        dx = x[0] - OBSTACLE_CENTER[0]
        dy = x[1] - OBSTACLE_CENTER[1]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-12)
        violation = jnp.maximum(OBSTACLE_RADIUS - dist, 0.0)
        lpdf -= W_OBSTACLE * 0.5 * violation**2
    return lpdf

score_pdf = jax.grad(log_pdf)

def kernel(x1, x2, h):
    return jnp.exp(-1.0 * jnp.sum(jnp.square(x1[:2]-x2[:2])) / h)

d_kernel = jax.grad(kernel, argnums=(0))

def stein_grad_unit(x1, x2, h):
    return kernel(x2, x1, h) * score_pdf(x2) + d_kernel(x2, x1, h)

def stein_grad_state(x, x_traj, h):
    vals = jax.vmap(stein_grad_unit, in_axes=(None, 0, None))(x, x_traj, h)
    return jnp.mean(vals, axis=0)

def stein_grad(traj, h):
    return jax.vmap(stein_grad_state, in_axes=(0, None, None))(traj, traj, h)

# Setup JAX device
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
except Exception:
    gpu = cpu

# ============================================================================
# 6. RBF Flow Matching Formulation
# ============================================================================

# Precompute RBF basis
t_vals = np.linspace(0, 1, T)
c_vals = np.linspace(0, 1, NUM_RBF_CENTERS)
sigma = 1.5 / NUM_RBF_CENTERS

B_np = np.zeros((T, NUM_RBF_CENTERS))
for k in range(NUM_RBF_CENTERS):
    B_np[:, k] = np.exp(-0.5 * ((t_vals - c_vals[k]) / sigma) ** 2)

B_mat = jnp.array(B_np) # Shape: (T, NUM_RBF_CENTERS)
B_outer = jnp.einsum('tk,tl->kl', B_mat, B_mat) * dt
G_R = jnp.kron(R_mat, B_outer)

def forward_sim(C, s0):
    """
    Simulate the double integrator system.
    C: (2, K) control points
    s0: (4,) initial state
    """
    # u(t) = C * B(t)
    u_traj = jnp.einsum('dk,tk->td', C, B_mat)
    
    def step(s, u):
        s_next = s + dt * jnp.array([s[2], s[3], u[0], u[1]])
        return s_next, s_next
    
    _, s_traj = jax.lax.scan(step, s0, u_traj)
    return s_traj

def compute_G_and_d(C, s0, h_traj):
    """
    Assemble the quadrature integration of G and d.
    """
    # M(t) = d s(t) / d C
    # Shape: (T, 4, 2, K)
    M = jax.jacfwd(forward_sim)(C, s0)
    
    # Reshape to (T, 4, 2K) to match vec(V)
    M_flat = M.reshape(T, 4, 2 * NUM_RBF_CENTERS)
    
    # G_Q = int M^T Q M dt
    G_Q = jnp.einsum('tfi,fg,tgj->ij', M_flat, Q_mat, M_flat) * dt
    
    G = G_Q + G_R
    
    # d = int M^T Q h(t) dt
    d = jnp.einsum('tfi,fg,tg->i', M_flat, Q_mat, h_traj) * dt
    
    return G, d

def solve_v(C, s0, h_traj):
    """
    Solve the dense LQR equivalent system for the RBF control update V.
    """
    G, d = compute_G_and_d(C, s0, h_traj)
    # Solve G * vec(V) = d
    V_flat = jnp.linalg.solve(G, d)
    # Reshape back to (2, K)
    V = V_flat.reshape(2, NUM_RBF_CENTERS)
    return V

stein_grad_jit = None
flow_matching_scan_step = None

def compile_jax_functions():
    global stein_grad_jit, flow_matching_scan_step
    stein_grad_jit = jax.jit(stein_grad, device=gpu)

    @jit
    def flow_matching_scan_step_inner(carry, _):
        C_all, x0_all = carry
        
        # 1. Simulate state trajectories
        x_trajs = vmap(forward_sim, in_axes=(0, 0))(C_all, x0_all)
        
        # 2. Compute collective Stein gradient
        all_x = x_trajs.reshape(-1, 4)
        stein_dx_all = stein_grad_jit(all_x, h=0.01)
        stein_dx_trajs = stein_dx_all.reshape(N_PARTICLES, T, 4)
        
        # 3. Solve LQR for optimal RBF velocity flow V
        V_trajs = vmap(solve_v, in_axes=(0, 0, 0))(C_all, x0_all, stein_dx_trajs)
        
        # 4. Update control points
        C_all_new = C_all + step_size * V_trajs
        return (C_all_new, x0_all), None
        
    flow_matching_scan_step = flow_matching_scan_step_inner

# ============================================================================
# 7. Finite Differences to Initialise RBF Control Points
# ============================================================================

def init_rbf_from_positions(pos_trajs, dt):
    """
    Convert (N, T, 2) positions to (N, 2, K) RBF control points.
    """
    N, T_len, _ = pos_trajs.shape
    
    # Velocity: v_t = (pos_{t+1} - pos_t) / dt
    v = np.zeros_like(pos_trajs)
    v[:, :-1, :] = (pos_trajs[:, 1:, :] - pos_trajs[:, :-1, :]) / dt
    v[:, -1, :] = v[:, -2, :]
    
    # Control input (acceleration): u_t = (v_{t+1} - v_t) / dt
    u = np.zeros_like(pos_trajs)
    u[:, :-1, :] = (v[:, 1:, :] - v[:, :-1, :]) / dt
    u[:, -1, :] = u[:, -2, :]
    
    # Fit RBF to u
    B_np = np.array(B_mat)
    B_pinv = np.linalg.pinv(B_np) # (K, T)
    
    # u is (N, T, 2) -> we want C shape (N, 2, K)
    # C_T = B_pinv @ u -> (N, K, 2)
    C_init_T = np.einsum('kt,ntd->nkd', B_pinv, u)
    C_init = np.transpose(C_init_T, (0, 2, 1)) # (N, 2, K)
    
    # Initial state: x0 = [x_0, y_0, vx_0, vy_0]
    x0 = np.zeros((N, 4))
    x0[:, :2] = pos_trajs[:, 0, :]
    x0[:, 2:] = v[:, 0, :]
    
    return C_init, x0

# ============================================================================
# 8. Master Benchmark Function
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle
    compile_jax_functions()
    
    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape", "polynomial", "rrt"]
    results = {}
    benchmark_data = {}

    print(f"2D Flow Matching (RBF) | {N_PARTICLES} particles, T={T}, K={NUM_RBF_CENTERS}, {N_ITERS} iters")
    print(f"Weights: ergodic={W_ERGODIC}, smooth={W_SMOOTH}, boundary={W_BOUNDARY}, obstacle={W_OBSTACLE}")
    print("-" * 65)

    # Use JIT to compile the forward simulation to convert control points to physical state trajectories for final evaluation
    sim_fn = jax.jit(vmap(forward_sim, in_axes=(0, 0)))

    for strat in strategies:
        print(f"Running strategy: {strat}")
        t_start = time.time()
        
        # 1. Generate noisy initial positions from strategies
        init_p, base_t = get_initialization(strat, N_PARTICLES, T, noise_std=INIT_NOISE_STD)
        pos_trajs = init_p.reshape(N_PARTICLES, T, 2)
        
        # 2. Extract matching control trajectory & initial states
        C_init, x0_init = init_rbf_from_positions(pos_trajs, dt)
        
        # Convert to JAX arrays
        C_all = jnp.array(C_init)
        x0_all = jnp.array(x0_init)
        
        # 3. Simulate initial physical trajectories
        initial_x_trajs = np.array(sim_fn(C_all, x0_all))
        initial_pos = initial_x_trajs[:, :, :2].reshape(N_PARTICLES, -1)
        
        # 4. JIT compilation & scan optimization loop
        (C_all_opt, _), _ = lax.scan(flow_matching_scan_step, (C_all, x0_all), None, length=N_ITERS)
        
        # 5. Simulate final physical trajectories
        final_x_trajs = np.array(sim_fn(C_all_opt, x0_all))
        final_pos = final_x_trajs[:, :, :2].reshape(N_PARTICLES, -1)
        
        # 6. Calculate energies for assessment
        final_E = np.array([compute_energy_and_grad(final_pos[i], T)[0] for i in range(N_PARTICLES)])
        best = int(np.argmin(final_E))
        elapsed = time.time() - t_start
        
        print(f"  -> Best energy: {final_E[best]:.3f} (Time: {elapsed:.2f}s)\n")
        
        results[strat] = {
            'initial': initial_pos,
            'base_traj': base_t,
            'final': final_pos,
            'best_idx': best
        }
        
        benchmark_data[strat] = {
            'mean_cost': float(np.mean(final_E)),
            'best_cost': float(final_E[best]),
            'time_s': float(elapsed)
        }
        
        if save_npy:
            np.save(os.path.join(out_dir, f"{strat}_trajs.npy"), final_pos)

    # ============================================================================
    # 9. Visualization
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
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    for row, strat in enumerate(strategies):
        res = results[strat]
        
        # Initial column
        ax = axes[row, 0]
        plot_particles(ax, res['initial'], f'[{strat}] Initial', base_traj=res['base_traj'])
        if res['base_traj'] is not None:
            ax.legend(loc='lower right', fontsize=9)
        
        # Final column
        ax = axes[row, 1]
        plot_particles(ax, res['final'], f'[{strat}] Final', highlight_best=True, best_idx=res['best_idx'])

    plt.tight_layout()
    out = os.path.join(out_dir, f'flow_matching_rbf_2d_{os.environ.get("TARGET_SHAPE", "N")}_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T': T,
            'N_PARTICLES': N_PARTICLES,
            'N_ITERS': N_ITERS,
            'NUM_RBF_CENTERS': NUM_RBF_CENTERS,
            'W_ERGODIC': W_ERGODIC
        }, f, indent=4)
        
    return benchmark_data

if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    run_benchmark(out_dir=f'/home/philipp/Documents/Uni/Master_thesis/results/Stein_Flow_matching_RBF_{TARGET_SHAPE}')
