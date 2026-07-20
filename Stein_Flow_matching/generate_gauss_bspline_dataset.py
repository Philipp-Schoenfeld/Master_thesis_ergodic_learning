#!/usr/bin/env python3
"""
Pipeline to generate a dataset of 100 B-Spline Ergodic Trajectories on a Simple Gaussian Distribution.
Based on Flow Matching B-Spline method.
"""

import time
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add Master_thesis parent folder to path to import initialization strategies
sys.path.append("/home/philipp/Documents/Uni/Master_thesis/src")
from init_strategies import get_initialization

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

# B-Spline library
sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/bsplinax-main")
from bsplinax.bspline import BsplineBasisClamped

np.random.seed(42)

# ============================================================================
# 1. Target Distribution (Simple Gaussian)
# ============================================================================

GAUSS_MU = np.array([0.5, 0.5])
GAUSS_SIGMA = 0.15

def target_distribution(x, y):
    return np.exp(-((x - GAUSS_MU[0])**2 + (y - GAUSS_MU[1])**2) / (2 * GAUSS_SIGMA**2))

grid_res = 200
_xs = np.linspace(0, 1, grid_res)
_ys = np.linspace(0, 1, grid_res)
Xg, Yg = np.meshgrid(_xs, _ys)
Zg = target_distribution(Xg, Yg)

def calculate_optimal_control_points(Zg_target, T_steps):
    # A simple Gaussian requires fewer control points than complex shapes
    return 30 

# ============================================================================
# 2. Ergodic Metric — Fourier Decomposition
# ============================================================================

K_FOURIER = 10
k_indices = np.array([[k1, k2] for k1 in range(K_FOURIER) for k2 in range(K_FOURIER)])
Lambda_k = (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)

def fourier_basis(pts):
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)

_grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
_grid_w = Zg.ravel()
_grid_w = _grid_w / _grid_w.sum()
phi_k = np.sum(_grid_w[:, None] * fourier_basis(_grid_pts), axis=0)

# ============================================================================
# 3. Hyperparameters
# ============================================================================

T = 100           
N_PARTICLES = 100  # Generate 100 dataset
N_ITERS = 15000    # Ensure convergence for random initialization
step_size = 0.05  

NUM_CONTROL_POINTS = calculate_optimal_control_points(Zg, T)
DEGREE = 3

W_ERGODIC = 1200.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0

INIT_NOISE_STD = 0.05
dt = 0.05

Q_mat = jnp.diag(jnp.array([1.0, 1.0, 0.001, 0.001]))
R_mat = jnp.diag(jnp.array([0.01, 0.01]))

# ============================================================================
# 4. Energy Function
# ============================================================================

def compute_energy_and_grad(X_flat, T_steps):
    X = X_flat.reshape(T_steps, 2)
    energy = 0.0

    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    energy += W_SMOOTH * np.sum(accel ** 2)

    Fk = fourier_basis(X)
    c_k = np.mean(Fk, axis=0)
    diff_k = c_k - phi_k
    ergodic_metric = 0.5 * np.sum(Lambda_k * diff_k ** 2)
    energy += W_ERGODIC * ergodic_metric

    margin = 0.03
    lo = np.minimum(X - margin, 0.0)
    hi = np.maximum(X - (1.0 - margin), 0.0)
    energy += W_BOUNDARY * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))

    return energy, ergodic_metric

# ============================================================================
# 5. JAX-differentiable Target Distribution & Stein Gradient
# ============================================================================

def log_pdf(x):
    # Log pdf of Simple Gaussian
    mu = jnp.array(GAUSS_MU)
    sigma = GAUSS_SIGMA
    return -jnp.sum((x[:2] - mu)**2) / (2 * sigma**2)

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

try:
    gpu = jax.devices("cuda")[0]
except Exception:
    gpu = jax.devices("cpu")[0]

# ============================================================================
# 6. B-Spline Flow Matching Formulation
# ============================================================================

basis_generator = BsplineBasisClamped(
    degree=DEGREE, 
    num_control_points=NUM_CONTROL_POINTS, 
    num_phase_points=T,
    compute_derivatives=False
)
B_mat = jnp.array(basis_generator.B)
B_outer = jnp.einsum('tk,tl->kl', B_mat, B_mat) * dt
G_R = jnp.kron(R_mat, B_outer)

def forward_sim(C, s0):
    u_traj = jnp.einsum('dk,tk->td', C, B_mat)
    
    def step(s, u):
        s_next = s + dt * jnp.array([s[2], s[3], u[0], u[1]])
        return s_next, s_next
    
    _, s_traj = jax.lax.scan(step, s0, u_traj)
    return s_traj

def compute_G_and_d(C, s0, h_traj):
    M = jax.jacfwd(forward_sim)(C, s0)
    M_flat = M.reshape(T, 4, 2 * NUM_CONTROL_POINTS)
    G_Q = jnp.einsum('tfi,fg,tgj->ij', M_flat, Q_mat, M_flat) * dt
    G = G_Q + G_R
    d = jnp.einsum('tfi,fg,tg->i', M_flat, Q_mat, h_traj) * dt
    return G, d

def solve_v(C, s0, h_traj):
    G, d = compute_G_and_d(C, s0, h_traj)
    V_flat = jnp.linalg.solve(G, d)
    V = V_flat.reshape(2, NUM_CONTROL_POINTS)
    return V

stein_grad_jit = None
flow_matching_scan_step = None

def compile_jax_functions():
    global stein_grad_jit, flow_matching_scan_step
    stein_grad_jit = jax.jit(stein_grad, device=gpu)

    @jit
    def flow_matching_scan_step_inner(carry, _):
        C_all, x0_all = carry
        
        x_trajs = vmap(forward_sim, in_axes=(0, 0))(C_all, x0_all)
        
        all_x = x_trajs.reshape(-1, 4)
        stein_dx_all = stein_grad_jit(all_x, h=0.01)
        stein_dx_trajs = stein_dx_all.reshape(N_PARTICLES, T, 4)
        
        V_trajs = vmap(solve_v, in_axes=(0, 0, 0))(C_all, x0_all, stein_dx_trajs)
        
        C_all_new = C_all + step_size * V_trajs
        return (C_all_new, x0_all), None
        
    flow_matching_scan_step = flow_matching_scan_step_inner

# ============================================================================
# 7. Finite Differences to Initialise B-Spline Control Points
# ============================================================================

def init_bspline_from_positions(pos_trajs, dt):
    N, T_len, _ = pos_trajs.shape
    v = np.zeros_like(pos_trajs)
    v[:, :-1, :] = (pos_trajs[:, 1:, :] - pos_trajs[:, :-1, :]) / dt
    v[:, -1, :] = v[:, -2, :]
    u = np.zeros_like(pos_trajs)
    u[:, :-1, :] = (v[:, 1:, :] - v[:, :-1, :]) / dt
    u[:, -1, :] = u[:, -2, :]
    B_np = np.array(B_mat)
    B_pinv = np.linalg.pinv(B_np)
    C_init_T = np.einsum('kt,ntd->nkd', B_pinv, u)
    C_init = np.transpose(C_init_T, (0, 2, 1))
    x0 = np.zeros((N, 4))
    x0[:, :2] = pos_trajs[:, 0, :]
    x0[:, 2:] = v[:, 0, :]
    return C_init, x0

# ============================================================================
# 8. Generation Method
# ============================================================================

def generate_dataset(out_dir: str):
    compile_jax_functions()
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating 100 Gaussian B-Spline trajectories...")
    t_start = time.time()
    
    # Let's use 'random' init to get a diverse set of trajectories
    # Get initial positions using random strategy
    init_p, base_t = get_initialization('random', N_PARTICLES, T, noise_std=INIT_NOISE_STD)
    pos_trajs = init_p.reshape(N_PARTICLES, T, 2)
    
    C_init, x0_init = init_bspline_from_positions(pos_trajs, dt)
    C_all = jnp.array(C_init)
    x0_all = jnp.array(x0_init)
    
    print(f"Starting optimization loop ({N_ITERS} iterations)...")
    CHUNK_SIZE = 100
    n_chunks = N_ITERS // CHUNK_SIZE
    
    C_opt, x0_opt = C_all, x0_all
    for _ in tqdm(range(n_chunks), desc="Optimizing Dataset"):
        (C_opt, x0_opt), _ = lax.scan(flow_matching_scan_step, (C_opt, x0_opt), None, length=CHUNK_SIZE)
    C_all_opt = C_opt
    
    sim_fn = jax.jit(vmap(forward_sim, in_axes=(0, 0)))
    final_x_trajs = np.array(sim_fn(C_all_opt, x0_all))
    final_pos = final_x_trajs[:, :, :2].reshape(N_PARTICLES, -1)
    
    elapsed = time.time() - t_start
    print(f"Generation completed in {elapsed:.2f}s")
    
    final_metrics = [compute_energy_and_grad(final_pos[i], T) for i in range(N_PARTICLES)]
    final_E = np.array([m[0] for m in final_metrics])
    print(f"Mean Energy: {np.mean(final_E):.3f}")
    
    # Save dataset
    out_file = os.path.join(out_dir, "gauss_bspline_dataset_100.npy")
    np.save(out_file, final_pos)
    print(f"Dataset saved to {out_file} with shape {final_pos.shape} (N_PARTICLES, T*2)")
    
    # Visualize 5 random trajectories
    plt.figure(figsize=(10, 5))
    
    sample_indices = np.random.choice(N_PARTICLES, size=min(5, N_PARTICLES), replace=False)
    
    plt.subplot(1, 2, 1)
    plt.contourf(Xg, Yg, Zg, levels=30, cmap='YlOrRd', alpha=0.6)
    for i in sample_indices:  
        tr = pos_trajs[i].reshape(T, 2)
        plt.plot(tr[:, 0], tr[:, 1], '-', alpha=0.4)
        plt.plot(tr[0, 0], tr[0, 1], 'ko', ms=2)
    plt.title('Initial Random Particles')
    plt.xlim(0, 1); plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.contourf(Xg, Yg, Zg, levels=30, cmap='YlOrRd', alpha=0.6)
    for i in sample_indices:  
        tr = final_pos[i].reshape(T, 2)
        plt.plot(tr[:, 0], tr[:, 1], '-', alpha=0.7)
        plt.plot(tr[0, 0], tr[0, 1], 'ko', ms=2)
    plt.title(f'Optimized Dataset Sample (n={len(sample_indices)})')
    plt.xlim(0, 1); plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gauss_bspline_samples.png"))
    print("Sample visualization saved to gauss_bspline_samples.png")

if __name__ == "__main__":
    out_dir = '/home/philipp/Documents/Uni/Master_thesis/results/Gauss_Dataset'
    generate_dataset(out_dir)
