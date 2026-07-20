#!/usr/bin/env python3
"""
2D Stein Variational Gradient Descent on B-Splines (Ergodic Coverage)

=============================================================================
GRAND OVERALL IDEA
=============================================================================
This script solves the "Ergodic Coverage" problem for a robotic system. The 
goal is to generate a set of continuous trajectories (an ensemble of particles) 
such that the time-averaged spatial statistics of these trajectories closely match
a target probability distribution (e.g., shapes 'N', 'H', 'II').

To achieve this, we use a hybrid approach:
1. B-Spline Parameterization: Instead of discretely optimizing a trajectory 
   point-by-point, we parameterize the control inputs using finite-dimensional 
   B-Splines. This naturally enforces continuous and smooth physical trajectories.
2. SVGD (Stein Variational Gradient Descent): We use SVGD to jointly optimize 
   multiple trajectories (particles). SVGD pushes particles towards regions of 
   high target probability while maintaining diversity through a repulsive 
   kernel force, preventing all trajectories from collapsing into a single path.

=============================================================================
SPECIAL LIBRARIES & USAGE
=============================================================================
- JAX (`jax`, `jax.numpy`): A high-performance numerical computing library. 
  It provides automatic differentiation (`jax.grad`) which we use to take 
  derivatives of our physical simulation and energy function with respect to 
  the B-spline control points. It also provides `jax.jit` for Just-In-Time 
  compilation (making loops extremely fast via XLA), `jax.vmap` for automatic 
  vectorization across our trajectory ensemble, and `jax.lax.scan` for fast 
  compiled loop execution.
- BSPLINAX (`bsplinax.bspline`): A JAX-compatible B-Spline library. Used to 
  generate the B-spline basis matrix mapping control points to time steps.

=============================================================================
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

# Import the B-Spline library (ensure this is in your Python path)
sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/bsplinax-main")
from bsplinax.bspline import BsplineBasisClamped

np.random.seed(42)

# ============================================================================
# 1. Target Distribution
# ============================================================================
# We define a spatial target distribution based on line segments. The distribution
# is constructed as a Gaussian mixture over the shortest distances to these segments.

TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')

# Coordinates for the target shapes drawn in a [0, 1] x [0, 1] workspace
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
    """Calculates the shortest Euclidean distance from point (px, py) to a line segment."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (len_sq + 1e-12), 0, 1)
    return np.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)

def target_distribution(x, y):
    """Creates the target probability field over the 2D workspace."""
    d_min = np.full_like(x, 1e10)
    for (ax, ay), (bx, by) in N_SEGMENTS:
        d_min = np.minimum(d_min, _dist_to_segment(x, y, ax, ay, bx, by))
    return np.exp(-d_min ** 2 / (2 * STROKE_WIDTH ** 2))

# Generate a high-resolution grid for plotting and Fourier coefficient calculation
grid_res = 200
_xs = np.linspace(0, 1, grid_res)
_ys = np.linspace(0, 1, grid_res)
Xg, Yg = np.meshgrid(_xs, _ys)
Zg = target_distribution(Xg, Yg)

def calculate_optimal_control_points(Zg_target, T_steps):
    """
    Calculates the spatial complexity of the target distribution using Shannon entropy
    and maps it to an optimal number of B-Spline control points.
    """
    P = Zg_target.ravel()
    P = P / P.sum()
    entropy = -np.sum(P[P > 1e-10] * np.log(P[P > 1e-10]))
    
    # Linear mapping from entropy to control point count
    # Entropy ~9.47 ('II') -> ~40 points, Entropy ~9.70 ('N') -> ~60 points
    K = int(40 + (entropy - 9.47) * 87)
    
    # Ensure K is within safe bounds (at least 20, max T-10)
    return int(np.clip(K, 20, T_steps - 10))

# ============================================================================
# 2. Ergodic Metric — Fourier Decomposition
# ============================================================================
# Ergodicity measures how well a trajectory's time-averaged statistics match a 
# spatial distribution. We compute this efficiently by projecting both the 
# trajectory and the target distribution into the Fourier domain.

K_FOURIER = 10
# Generate all combinations of wave numbers (k1, k2) up to K_FOURIER
k_indices = np.array([[k1, k2] for k1 in range(K_FOURIER) for k2 in range(K_FOURIER)])
# Lambda_k is a spectral decay weight that strongly penalizes low-frequency mismatches
Lambda_k = (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)

def fourier_basis(pts):
    """Evaluates the Fourier cosine basis functions at given 2D points."""
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)

# Precompute the Fourier coefficients (phi_k) of the target distribution
_grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
_grid_w = Zg.ravel()
_grid_w = _grid_w / _grid_w.sum() # Normalize so it integrates to 1
phi_k = np.sum(_grid_w[:, None] * fourier_basis(_grid_pts), axis=0)

# Convert numpy arrays to JAX arrays so they can be used inside JIT-compiled functions
k_indices_jnp = jnp.array(k_indices)
Lambda_k_jnp = jnp.array(Lambda_k)
phi_k_jnp = jnp.array(phi_k)

# ============================================================================
# 3. Hyperparameters
# ============================================================================

T = 100           # Total number of discrete time steps in the trajectory simulation
N_PARTICLES = 30  # Number of SVGD particles (trajectories in the ensemble)
N_ITERS = int(os.environ.get("N_ITERS", 30000))   # Number of Adam optimization steps for SVGD

# B-Spline specific hyperparams
NUM_CONTROL_POINTS = calculate_optimal_control_points(Zg, T)
DEGREE = 3                # Cubic B-Splines

# Loss function weights
W_ERGODIC = 1200.0   # Weight for matching the target distribution
W_SMOOTH = 15.0      # Weight for trajectory smoothness (acceleration penalty)
W_BOUNDARY = 30.0    # Penalty for leaving the [0, 1] workspace
W_CONTROL = 0.01     # Tikhonov regularization on the control effort to prevent jitter

# Obstacle avoidance parameters
USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

INIT_NOISE_STD = 0.02
dt = 0.05 # Simulation time step

# Adam Optimizer parameters
ADAM_LR = 0.005
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# Hardware allocation (defaults to GPU if available)
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
except Exception:
    gpu = cpu

# ============================================================================
# 4. B-Spline Setup and Cost Function in JAX
# ============================================================================

# Initialize the B-Spline basis generator
# compute_derivatives=False prevents JAX from hanging during compilation when K is large
basis_generator = BsplineBasisClamped(
    degree=DEGREE, 
    num_control_points=NUM_CONTROL_POINTS, 
    num_phase_points=T,
    compute_derivatives=False
)
# B_mat has shape (T, K). It linearly maps K control points to T trajectory phase points.
B_mat = jnp.array(basis_generator.B)

# B_outer represents the continuous inner product integral: \int B(t)^T B(t) dt.
# It acts as a Gram Matrix, allowing us to map Euclidean distances in the control 
# space to true L2 functional distances in the continuous trajectory space.
B_outer = jnp.einsum('tk,tl->kl', B_mat, B_mat) * dt

def forward_sim(C, s0):
    """
    Simulate the robot's kinematics (a double integrator system).
    Args:
      C: (2, K) matrix of B-Spline control points (acceleration commands).
      s0: (4,) initial state vector [x, y, vx, vy].
    """
    # Evaluate the continuous control signal u(t) at T discrete time steps.
    # u(t) = sum_k C_k * B_k(t)
    u_traj = jnp.einsum('dk,tk->td', C, B_mat)
    
    def step(s, u):
        # Euler integration step: next_state = current_state + dt * derivative
        s_next = s + dt * jnp.array([s[2], s[3], u[0], u[1]])
        # jax.lax.scan expects (carry, output). We output the full state.
        return s_next, s_next
    
    # jax.lax.scan is a highly optimized unrolled loop for JAX.
    _, s_traj = jax.lax.scan(step, s0, u_traj)
    return s_traj

def compute_energy_jax(C, s0):
    """
    Calculates the total objective function (energy) for a single trajectory.
    This function must be purely differentiable for JAX to compute gradients.
    """
    s_traj = forward_sim(C, s0)
    X = s_traj[:, :2] # Extract purely spatial coordinates (x, y)
    
    # 1. Smoothness (Discrete Acceleration Penalty)
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    energy = W_SMOOTH * jnp.sum(accel ** 2)
    
    # 2. Ergodic Cost
    # Compute the trajectory's Fourier coefficients (c_k) and compare to target (phi_k)
    args = jnp.pi * X[:, None, :] * k_indices_jnp[None, :, :]
    Fk = jnp.prod(jnp.cos(args), axis=-1)
    c_k = jnp.mean(Fk, axis=0)
    diff_k = c_k - phi_k_jnp
    ergodic_metric = 0.5 * jnp.sum(Lambda_k_jnp * diff_k ** 2)
    energy += W_ERGODIC * ergodic_metric
    
    # 3. Boundary Cost (Keep trajectory inside [0, 1] workspace)
    margin = 0.03
    lo = jnp.minimum(X - margin, 0.0)
    hi = jnp.maximum(X - (1.0 - margin), 0.0)
    energy += W_BOUNDARY * 0.5 * (jnp.sum(lo ** 2) + jnp.sum(hi ** 2))
    
    # 4. Obstacle Avoidance Cost
    if USE_OBSTACLE:
        dx = X[:, 0] - OBSTACLE_CENTER[0]
        dy = X[:, 1] - OBSTACLE_CENTER[1]
        dist_sq = dx**2 + dy**2
        dist = jnp.sqrt(dist_sq + 1e-12)
        violation = jnp.maximum(OBSTACLE_RADIUS - dist, 0.0)
        energy += W_OBSTACLE * 0.5 * jnp.sum(violation**2)
        
    # 5. Control Regularization (L2 Norm of continuous control effort)
    # This term penalizes \int ||u(t)||^2 dt exactly, preventing
    # high-frequency jitter in the un-evaluated "null space" of the discrete points.
    energy += W_CONTROL * jnp.einsum('dk,kl,dl->', C, B_outer, C)
        
    return energy

# jax.grad automatically generates a function that computes the exact analytical 
# derivative of the energy with respect to the control points (C).
grad_energy_jax = jax.grad(compute_energy_jax, argnums=0)

# ============================================================================
# 5. SVGD on Control Points
# ============================================================================

def rbf_kernel(C1, C2, h):
    """
    Computes the Radial Basis Function (RBF) kernel between two sets of control points.
    Instead of using standard Euclidean distance which is susceptible to control noise, 
    we use the B_outer Mahalanobis distance. This guarantees the distance perfectly 
    reflects the true continuous L2 difference between the physical trajectories.
    """
    diff = C1 - C2
    sq_dist = jnp.einsum('dk,kl,dl->', diff, B_outer, diff)
    return jnp.exp(-sq_dist / h)

grad_rbf_kernel = jax.grad(rbf_kernel, argnums=0)

@jit
def svgd_step_jax(C_all, x0_all):
    """
    Executes a single vectorised Stein Variational Gradient Descent (SVGD) update step.
    The @jit decorator compiles this entire block into optimized GPU/CPU machine code.
    """
    N = C_all.shape[0]
    
    # jax.vmap essentially maps a function designed for a single input over an array of 
    # inputs (N particles) in parallel without needing a slow Python for-loop.
    energies = jax.vmap(compute_energy_jax, in_axes=(0, 0))(C_all, x0_all)
    grads = jax.vmap(grad_energy_jax, in_axes=(0, 0))(C_all, x0_all)
    
    # In SVGD, we want to maximize a probability density exp(-Energy), 
    # so the score function is the negative gradient of the energy.
    scores = -grads
    
    # Compute pairwise squared functional distances between all trajectories
    def compute_sq_dist(c1, c2):
        diff = c1 - c2
        return jnp.einsum('dk,kl,dl->', diff, B_outer, diff)
        
    # Double vmap computes a full (N, N) distance matrix efficiently
    sq_dists = jax.vmap(jax.vmap(compute_sq_dist, in_axes=(None, 0)), in_axes=(0, None))(C_all, C_all)
    
    # Median heuristic dynamically scales the RBF bandwidth (h) to the current spread of particles
    h = jnp.maximum(jnp.median(sq_dists) / jnp.log(N + 1.0), 0.1)
    
    def get_K(c1, c2):
        return rbf_kernel(c1, c2, h)
    
    def get_grad_K(c1, c2):
        return grad_rbf_kernel(c1, c2, h)
    
    vmap_j_K = jax.vmap(get_K, in_axes=(0, None))
    vmap_j_grad_K = jax.vmap(get_grad_K, in_axes=(0, None))
    
    # Compute the full (N, N) Kernel matrix and its gradient matrix
    K_mat = jax.vmap(vmap_j_K, in_axes=(None, 0))(C_all, C_all) # shape (N, N)
    grad_K_mat = jax.vmap(vmap_j_grad_K, in_axes=(None, 0))(C_all, C_all) # shape (N, N, 2, K)
    
    # ========================================================================
    # The Core SVGD Update Equation
    # update_i = (1/N) * sum_j [ Kernel(j, i) * score_j + Gradient_Kernel(j, i) ]
    # 
    # - The first term (Kernel * score) pulls particle 'i' towards high probability 
    #   regions discovered by other particles 'j'.
    # - The second term (Gradient_Kernel) acts as a repulsive force, pushing 
    #   particles away from each other to prevent collapse and ensure diversity.
    # ========================================================================
    updates = jnp.sum(K_mat[:, :, None, None] * scores[None, :, :, :] + grad_K_mat, axis=1) / N
    
    return updates, energies

@jit
def optimize_C_all(C_all_init, x0_all):
    """
    Runs the entire Adam optimization loop entirely within compiled JAX code.
    """
    def scan_step(carry, _):
        C_all, m, v, it = carry
        
        updates, energies = svgd_step_jax(C_all, x0_all)
        
        # Gradient clipping to ensure numerical stability during early iterations
        max_norm = jnp.max(jnp.abs(updates))
        updates = jax.lax.cond(
            max_norm > 200.0,
            lambda u: u * (200.0 / max_norm),
            lambda u: u,
            updates
        )

        # Standard Adam Optimizer update rules
        t_adam = it + 1
        m_new = ADAM_BETA1 * m + (1 - ADAM_BETA1) * updates
        v_new = ADAM_BETA2 * v + (1 - ADAM_BETA2) * (updates ** 2)
        m_hat = m_new / (1 - ADAM_BETA1 ** t_adam)
        v_hat = v_new / (1 - ADAM_BETA2 ** t_adam)
        
        # Move control points along the SVGD optimal direction
        C_all_new = C_all + ADAM_LR * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)
        
        return (C_all_new, m_new, v_new, t_adam), jnp.mean(energies)

    # Initialize Adam moments
    m0 = jnp.zeros_like(C_all_init)
    v0 = jnp.zeros_like(C_all_init)
    
    # lax.scan runs the `scan_step` function N_ITERS times sequentially.
    # It is substantially faster than a python `for` loop because the loop unrolling 
    # happens inside the compiled C++/CUDA backend.
    (C_all_final, _, _, _), energy_log = jax.lax.scan(
        scan_step, 
        (C_all_init, m0, v0, 0), 
        None, 
        length=N_ITERS
    )
    
    return C_all_final, energy_log


# ============================================================================
# 6. Finite Differences to Initialise B-Spline Control Points
# ============================================================================

def init_bspline_from_positions(pos_trajs, dt_val):
    """
    Given a rough initial spatial trajectory generated by standard geometric strategies 
    (like a noisy line or RRT), this computes the starting B-spline control points.
    
    It extracts velocities and accelerations using finite differences, and then 
    solves a pseudo-inverse least-squares problem to find the optimal B-spline 
    control points that map closely to those accelerations.
    """
    N, T_len, _ = pos_trajs.shape
    
    # Velocity: v_t = (pos_{t+1} - pos_t) / dt
    v = np.zeros_like(pos_trajs)
    v[:, :-1, :] = (pos_trajs[:, 1:, :] - pos_trajs[:, :-1, :]) / dt_val
    v[:, -1, :] = v[:, -2, :]
    
    # Control input (acceleration): u_t = (v_{t+1} - v_t) / dt
    u = np.zeros_like(pos_trajs)
    u[:, :-1, :] = (v[:, 1:, :] - v[:, :-1, :]) / dt_val
    u[:, -1, :] = u[:, -2, :]
    
    # Fit B-Spline to u using the Moore-Penrose Pseudo-inverse
    B_np = np.array(B_mat)
    B_pinv = np.linalg.pinv(B_np) # Shape: (K, T)
    
    # Matrix multiplication: C = B_pinv @ u
    C_init_T = np.einsum('kt,ntd->nkd', B_pinv, u)
    C_init = np.transpose(C_init_T, (0, 2, 1)) # Final shape: (N, 2, K)
    
    # Set the rigid initial state vector x0 = [x_0, y_0, vx_0, vy_0]
    x0 = np.zeros((N, 4))
    x0[:, :2] = pos_trajs[:, 0, :]
    x0[:, 2:] = v[:, 0, :]
    
    return C_init, x0

# ============================================================================
# 7. Energy Function for Final Assessment (Numpy)
# ============================================================================

def compute_energy_and_grad(X_flat, T_steps):
    """
    A NumPy equivalent of the energy function used exclusively at the end of the 
    script to precisely evaluate and log the isolated pure ergodic metric.
    """
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
    
    # Isolate the pure ergodic metric so we can tune algorithms explicitly on coverage success
    ergodic_metric = 0.5 * np.sum(Lambda_k * diff_k ** 2)
    energy += W_ERGODIC * ergodic_metric

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

    return energy, ergodic_metric, grad.ravel()

# ============================================================================
# 8. Master Benchmark Function
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    """
    The main driver loop that runs the SVGD generation across multiple different 
    starting configurations ("strategies") to see which converges to the best coverage.
    """
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle
    
    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape", "polynomial", "rrt"]
    results = {}
    benchmark_data = {}

    print(f"2D SVGD (B-Spline) | {N_PARTICLES} particles, T={T}, K={NUM_CONTROL_POINTS}, {N_ITERS} iters")
    print(f"Weights: ergodic={W_ERGODIC}, smooth={W_SMOOTH}, boundary={W_BOUNDARY}, obstacle={W_OBSTACLE}")
    print(f"Adam: lr={ADAM_LR}, beta1={ADAM_BETA1}, beta2={ADAM_BETA2}")
    print("-" * 65)

    # Compile the forward simulator for fast bulk inference at the end
    sim_fn = jax.jit(vmap(forward_sim, in_axes=(0, 0)))

    for strat in strategies:
        print(f"Running strategy: {strat}")
        t_start = time.time()
        
        # 1. Generate noisy initial geometric positions
        init_p, base_t = get_initialization(strat, N_PARTICLES, T, noise_std=INIT_NOISE_STD)
        pos_trajs = init_p.reshape(N_PARTICLES, T, 2)
        
        # 2. Extract matching control trajectory & initial states
        C_init, x0_init = init_bspline_from_positions(pos_trajs, dt)
        
        # Convert to JAX arrays
        C_all = jnp.array(C_init)
        x0_all = jnp.array(x0_init)
        
        # 3. Simulate initial physical trajectories
        initial_x_trajs = np.array(sim_fn(C_all, x0_all))
        initial_pos = initial_x_trajs[:, :, :2].reshape(N_PARTICLES, -1)
        
        # 4. Execute the fully JIT-compiled optimization loop
        C_all_opt, energy_log = optimize_C_all(C_all, x0_all)
        
        # 5. Simulate final physical trajectories utilizing the optimal B-Spline control points
        final_x_trajs = np.array(sim_fn(C_all_opt, x0_all))
        final_pos = final_x_trajs[:, :, :2].reshape(N_PARTICLES, -1)
        
        # 6. Calculate objective energies for assessment
        final_metrics = [compute_energy_and_grad(final_pos[i], T) for i in range(N_PARTICLES)]
        final_E = np.array([m[0] for m in final_metrics])
        final_erg = np.array([m[1] for m in final_metrics])
        
        # We exclusively use the pure ergodic coverage metric to define our "Best" trajectory
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
    # 9. Visualization
    # ============================================================================
    # Renders the high-quality comparison plots to display SVGD evolution

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
    out = os.path.join(out_dir, f'svgd_bspline_2d_{os.environ.get("TARGET_SHAPE", "N")}_comparison30000.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T': T,
            'N_PARTICLES': N_PARTICLES,
            'N_ITERS': N_ITERS,
            'NUM_CONTROL_POINTS': NUM_CONTROL_POINTS,
            'W_ERGODIC': W_ERGODIC,
            'W_SMOOTH': W_SMOOTH,
            'W_BOUNDARY': W_BOUNDARY
        }, f, indent=4)
        
    return benchmark_data

if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    run_benchmark(out_dir=f'/home/philipp/Documents/Uni/Master_thesis/results/SE3_SVGD_BSpline_{TARGET_SHAPE}')
