#!/usr/bin/env python3
"""
Dimension-Agnostic SVGD Engine
================================
Shared SVGD update rules, Adam optimizers, forward simulation, and cost
function building blocks that work for any spatial dimension (2D, 3D, ...).

Contains both:
- JAX-based components for B-spline SVGD (JIT-compiled, GPU-accelerated)
- NumPy-based components for regular SVGD (CPU, with analytic gradients)
"""

import os
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import jit, vmap
from scipy.spatial.distance import pdist, squareform

from ergodic_core import (
    compute_ergodic_metric_jax,
    compute_ergodic_metric_numpy,
    fourier_basis_nd,
    fourier_basis_nd_jax,
    fourier_basis_grad_nd,
)


# ============================================================================
# 1. Shared Cost Components (JAX)
# ============================================================================

def compute_smoothness_cost_jax(X):
    """
    Discrete acceleration penalty: ||x_{t+2} - 2x_{t+1} + x_t||^2
    Works for any spatial dimension.

    Args:
        X: (T, dim) trajectory positions (JAX array)
    Returns:
        scalar smoothness cost
    """
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    return jnp.sum(accel ** 2)


def compute_boundary_cost_jax(X, margin=0.03, lo_bound=0.0, hi_bound=1.0):
    """
    Penalty for leaving the [lo_bound, hi_bound]^dim workspace.
    Works for any dimension.

    Args:
        X: (T, dim) trajectory positions (JAX array)
    Returns:
        scalar boundary cost
    """
    lo = jnp.minimum(X - (lo_bound + margin), 0.0)
    hi = jnp.maximum(X - (hi_bound - margin), 0.0)
    return 0.5 * (jnp.sum(lo ** 2) + jnp.sum(hi ** 2))


def compute_obstacle_cost_jax(X, center, radius):
    """
    Sphere obstacle avoidance (works for any dimension).

    Args:
        X: (T, dim) trajectory positions (JAX array)
        center: (dim,) obstacle center
        radius: scalar obstacle radius
    Returns:
        scalar obstacle cost
    """
    center = jnp.array(center)
    diff = X - center[None, :]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)
    violation = jnp.maximum(radius - dist, 0.0)
    return 0.5 * jnp.sum(violation ** 2)


def compute_control_regularization_jax(C, B_outer):
    """
    L2 norm of continuous control effort: ∫ ||u(t)||^2 dt
    Computed exactly via the B-spline Gram matrix.

    Args:
        C: (dim, K) control point matrix
        B_outer: (K, K) Gram matrix
    Returns:
        scalar regularization cost
    """
    return jnp.einsum('dk,kl,dl->', C, B_outer, C)


# ============================================================================
# 2. Shared Cost Components (NumPy)
# ============================================================================

def compute_smoothness_cost_numpy(X):
    """NumPy version of acceleration penalty."""
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    return np.sum(accel ** 2)


def compute_smoothness_grad_numpy(X):
    """Gradient of acceleration penalty w.r.t. X."""
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    grad = np.zeros_like(X)
    grad[:-2] += 2 * accel
    grad[1:-1] -= 4 * accel
    grad[2:] += 2 * accel
    return grad


def compute_boundary_cost_numpy(X, margin=0.03, lo_bound=0.0, hi_bound=1.0):
    """NumPy version of boundary penalty."""
    lo = np.minimum(X - (lo_bound + margin), 0.0)
    hi = np.maximum(X - (hi_bound - margin), 0.0)
    return 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))


def compute_boundary_grad_numpy(X, margin=0.03, lo_bound=0.0, hi_bound=1.0):
    """Gradient of boundary penalty w.r.t. X."""
    lo = np.minimum(X - (lo_bound + margin), 0.0)
    hi = np.maximum(X - (hi_bound - margin), 0.0)
    return lo + hi


def compute_obstacle_cost_numpy(X, center, radius):
    """NumPy version of sphere obstacle penalty."""
    center = np.asarray(center)
    diff = X - center[None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    violation = np.maximum(radius - dist, 0.0)
    return 0.5 * np.sum(violation ** 2)


def compute_obstacle_grad_numpy(X, center, radius):
    """Gradient of obstacle penalty w.r.t. X."""
    center = np.asarray(center)
    diff = X - center[None, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    violation = np.maximum(radius - dist, 0.0)
    grad = np.zeros_like(X)
    for d in range(X.shape[1]):
        grad[:, d] = violation * (-diff[:, d] / dist)
    return grad


# ============================================================================
# 3. Forward Simulation (JAX, dimension-agnostic)
# ============================================================================

def forward_sim_nd(C, s0, B_mat, dt, dim):
    """
    Simulate double-integrator kinematics for any spatial dimension.

    State vector: [x_1, ..., x_dim, v_1, ..., v_dim]

    Args:
        C: (dim, K) B-spline control points (acceleration commands)
        s0: (2*dim,) initial state [positions, velocities]
        B_mat: (T, K) B-spline basis matrix
        dt: time step
        dim: spatial dimension

    Returns:
        s_traj: (T, 2*dim) state trajectory
    """
    # Evaluate control signal: u(t) = C @ B(t)^T => (T, dim)
    u_traj = jnp.einsum('dk,tk->td', C, B_mat)

    def step(s, u):
        # s = [pos(dim), vel(dim)]
        pos = s[:dim]
        vel = s[dim:]
        s_next = jnp.concatenate([pos + dt * vel, vel + dt * u])
        return s_next, s_next

    _, s_traj = jax.lax.scan(step, s0, u_traj)
    return s_traj


# ============================================================================
# 4. B-Spline SVGD Components (JAX)
# ============================================================================

def rbf_kernel_mahalanobis(C1, C2, B_outer, h):
    """
    RBF kernel using the B-spline Gram matrix for functional distance.
    Works for any dimension (C shape is (dim, K)).

    Args:
        C1, C2: (dim, K) control point matrices
        B_outer: (K, K) Gram matrix
        h: bandwidth scalar
    Returns:
        scalar kernel value
    """
    diff = C1 - C2
    sq_dist = jnp.einsum('dk,kl,dl->', diff, B_outer, diff)
    return jnp.exp(-sq_dist / h)


def build_bspline_svgd_step(compute_energy_fn, grad_energy_fn, B_outer):
    """
    Factory function that builds a JIT-compiled SVGD step for B-spline SVGD.
    Dimension-agnostic: works for any C shape (dim, K).

    Args:
        compute_energy_fn: function(C, s0) -> scalar
        grad_energy_fn: function(C, s0) -> (dim, K) gradient
        B_outer: (K, K) Gram matrix

    Returns:
        svgd_step_fn: JIT-compiled function(C_all, x0_all) -> (updates, energies)
    """
    grad_rbf = jax.grad(rbf_kernel_mahalanobis, argnums=0)

    @jit
    def svgd_step(C_all, x0_all):
        N = C_all.shape[0]

        energies = vmap(compute_energy_fn, in_axes=(0, 0))(C_all, x0_all)
        grads = vmap(grad_energy_fn, in_axes=(0, 0))(C_all, x0_all)
        scores = -grads

        # Pairwise functional distances
        def compute_sq_dist(c1, c2):
            diff = c1 - c2
            return jnp.einsum('dk,kl,dl->', diff, B_outer, diff)

        sq_dists = vmap(vmap(compute_sq_dist, in_axes=(None, 0)), in_axes=(0, None))(C_all, C_all)

        # Median heuristic for bandwidth
        h = jnp.maximum(jnp.median(sq_dists) / jnp.log(N + 1.0), 0.1)

        def get_K(c1, c2):
            return rbf_kernel_mahalanobis(c1, c2, B_outer, h)

        def get_grad_K(c1, c2):
            return grad_rbf(c1, c2, B_outer, h)

        vmap_j_K = vmap(get_K, in_axes=(0, None))
        vmap_j_grad_K = vmap(get_grad_K, in_axes=(0, None))

        K_mat = vmap(vmap_j_K, in_axes=(None, 0))(C_all, C_all)
        grad_K_mat = vmap(vmap_j_grad_K, in_axes=(None, 0))(C_all, C_all)

        # SVGD update: attractive (kernel * score) + repulsive (grad kernel)
        updates = jnp.sum(
            K_mat[:, :, None, None] * scores[None, :, :, :] + grad_K_mat,
            axis=1
        ) / N

        return updates, energies

    return svgd_step


def build_adam_optimizer_jax(svgd_step_fn, n_iters, adam_lr, adam_beta1, adam_beta2, adam_eps,
                            chunk_size=1000, label="SVGD"):
    """
    Factory that builds an Adam optimization loop using chunked lax.scan
    with a tqdm progress bar between chunks.

    Args:
        svgd_step_fn: JIT-compiled SVGD step function
        n_iters: total number of iterations
        adam_lr, adam_beta1, adam_beta2, adam_eps: Adam hyperparameters
        chunk_size: number of iterations per compiled chunk (default 1000)
        label: label for the progress bar

    Returns:
        optimize_fn: function(C_all_init, x0_all) -> (C_all_final, energy_log)
    """
    def _make_chunk_runner(chunk_len):
        """Build a JIT-compiled scan runner for a fixed chunk length."""
        @jit
        def _run_chunk(carry, x0_all):
            def scan_step(carry, _):
                C_all, m, v, it = carry

                updates, energies = svgd_step_fn(C_all, x0_all)

                # Gradient clipping
                max_norm = jnp.max(jnp.abs(updates))
                updates = jax.lax.cond(
                    max_norm > 200.0,
                    lambda u: u * (200.0 / max_norm),
                    lambda u: u,
                    updates
                )

                # Adam update
                t_adam = it + 1
                m_new = adam_beta1 * m + (1 - adam_beta1) * updates
                v_new = adam_beta2 * v + (1 - adam_beta2) * (updates ** 2)
                m_hat = m_new / (1 - adam_beta1 ** t_adam)
                v_hat = v_new / (1 - adam_beta2 ** t_adam)

                C_all_new = C_all + adam_lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)

                return (C_all_new, m_new, v_new, t_adam), jnp.mean(energies)

            return jax.lax.scan(scan_step, carry, None, length=chunk_len)
        return _run_chunk

    # Pre-build runners for the main chunk size and the remainder
    remainder = n_iters % chunk_size
    _run_main = _make_chunk_runner(chunk_size)
    _run_remainder = _make_chunk_runner(remainder) if remainder > 0 else None
    n_full_chunks = n_iters // chunk_size

    def optimize(C_all_init, x0_all, label_override=None):
        m0 = jnp.zeros_like(C_all_init)
        v0 = jnp.zeros_like(C_all_init)
        carry = (C_all_init, m0, v0, 0)

        all_energies = []
        desc = label_override if label_override is not None else label
        pbar = tqdm(total=n_iters, desc=desc, unit="it", position=int(os.environ.get("TQDM_POSITION", 0)))

        for _ in range(n_full_chunks):
            carry, chunk_energies = _run_main(carry, x0_all)
            jax.block_until_ready(carry[0])
            all_energies.append(np.array(chunk_energies))
            pbar.update(chunk_size)
            pbar.set_postfix(energy=f"{float(chunk_energies[-1]):.3f}")

        if _run_remainder is not None:
            carry, chunk_energies = _run_remainder(carry, x0_all)
            jax.block_until_ready(carry[0])
            all_energies.append(np.array(chunk_energies))
            pbar.update(remainder)
            pbar.set_postfix(energy=f"{float(chunk_energies[-1]):.3f}")

        pbar.close()
        C_all_final = carry[0]
        energy_log = np.concatenate(all_energies)
        return C_all_final, jnp.array(energy_log)

    return optimize


# ============================================================================
# 5. B-Spline Initialization (Dimension-Agnostic)
# ============================================================================

def init_bspline_from_positions_nd(pos_trajs, dt_val, B_mat, dim):
    """
    Given initial spatial trajectories, computes starting B-spline control points
    via finite-difference acceleration extraction and pseudo-inverse fitting.

    Args:
        pos_trajs: (N, T, dim) position trajectories
        dt_val: time step
        B_mat: (T, K) B-spline basis matrix (as JAX or numpy array)
        dim: spatial dimension

    Returns:
        C_init: (N, dim, K) initial control points
        x0: (N, 2*dim) initial state vectors [pos, vel]
    """
    N, T_len, _ = pos_trajs.shape

    # Velocity via finite differences
    v = np.zeros_like(pos_trajs)
    v[:, :-1, :] = (pos_trajs[:, 1:, :] - pos_trajs[:, :-1, :]) / dt_val
    v[:, -1, :] = v[:, -2, :]

    # Acceleration via finite differences
    u = np.zeros_like(pos_trajs)
    u[:, :-1, :] = (v[:, 1:, :] - v[:, :-1, :]) / dt_val
    u[:, -1, :] = u[:, -2, :]

    # Fit B-spline via pseudo-inverse
    B_np = np.array(B_mat)
    B_pinv = np.linalg.pinv(B_np)  # (K, T)

    # C_init_T: (N, K, dim) -> transpose to (N, dim, K)
    C_init_T = np.einsum('kt,ntd->nkd', B_pinv, u)
    C_init = np.transpose(C_init_T, (0, 2, 1))

    # Initial state: [pos_0, vel_0]
    x0 = np.zeros((N, 2 * dim))
    x0[:, :dim] = pos_trajs[:, 0, :]
    x0[:, dim:] = v[:, 0, :]

    return C_init, x0


# ============================================================================
# 6. Regular SVGD Components (NumPy)
# ============================================================================

def svgd_step_numpy(particles, compute_energy_and_grad_fn, T, dim):
    """
    Single SVGD step using NumPy (for regular, non-B-spline SVGD).
    Dimension-agnostic.

    Args:
        particles: (N, T*dim) flattened particle trajectories
        compute_energy_and_grad_fn: function(X_flat, T) -> (energy, grad_flat)
        T: number of time steps
        dim: spatial dimension

    Returns:
        update: (N, T*dim) SVGD update
        energies: (N,) energy values
    """
    Np, D = particles.shape

    # Pairwise RBF kernel with median bandwidth
    sq = squareform(pdist(particles, 'sqeuclidean'))
    pos = sq[sq > 0]
    med = np.median(pos) if len(pos) > 0 else 1.0
    h = max(med / np.log(Np + 1), 0.1)

    K_mat = np.exp(-sq / h)

    # Compute scores (negative energy gradients)
    scores = np.zeros_like(particles)
    energies = np.zeros(Np)
    for i in range(Np):
        E, g = compute_energy_and_grad_fn(particles[i], T)
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


def run_svgd_numpy(particles, T, n_iters, compute_energy_and_grad_fn, dim,
                   adam_lr=2e-3, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
                   n_particles=None, label="SVGD"):
    """
    Full SVGD optimization loop with Adam optimizer (NumPy).

    Args:
        particles: (N, T*dim) initial flattened trajectories
        T: number of time steps
        n_iters: number of optimization iterations
        compute_energy_and_grad_fn: function(X_flat, T) -> (energy, grad_flat)
        dim: spatial dimension
        adam_lr, adam_beta1, adam_beta2, adam_eps: Adam hyperparameters
        n_particles: number of particles (inferred from particles if None)
        label: description for tqdm progress bar

    Returns:
        particles: (N, T*dim) optimized trajectories
        energy_log: list of mean energies per iteration
    """
    if n_particles is None:
        n_particles = particles.shape[0]

    particles = particles.copy()
    m = np.zeros_like(particles)
    v = np.zeros_like(particles)
    energy_log = []

    pbar = tqdm(range(n_iters), desc=label, unit="it", position=int(os.environ.get("TQDM_POSITION", 0)))
    for it in pbar:
        delta, energies = svgd_step_numpy(particles, compute_energy_and_grad_fn, T, dim)
        mx = np.max(np.abs(delta))
        if mx > 200:
            delta *= 200.0 / mx

        t_adam = it + 1
        m = adam_beta1 * m + (1 - adam_beta1) * delta
        v = adam_beta2 * v + (1 - adam_beta2) * delta ** 2
        m_hat = m / (1 - adam_beta1 ** t_adam)
        v_hat = v / (1 - adam_beta2 ** t_adam)
        particles += adam_lr * m_hat / (np.sqrt(v_hat) + adam_eps)

        particles = particles.reshape(n_particles, T, dim)
        particles = np.clip(particles, 0.02, 0.98)
        particles = particles.reshape(n_particles, -1)
        mean_e = np.mean(energies)
        energy_log.append(mean_e)
        pbar.set_postfix(energy=f"{mean_e:.3f}")

    return particles, energy_log
