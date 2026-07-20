#!/usr/bin/env python
# coding: utf-8

import time
from tqdm import tqdm
import jax.numpy as jnp
import os
import numpy as np
import sqlite3

from jax import jit, grad, vmap
from jax.scipy.stats import multivariate_normal as mvn
import jax

# Initialize JAX devices
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
except:
    gpu = cpu
jnp.set_printoptions(precision=4)

from lqrax import LQR

# --- Target Distribution ---
mean_simple = jnp.array([0.5, 0.5])
cov_simple = jnp.array([
    [0.02, 0.0],
    [0.0, 0.02]
])

mean1 = jnp.array([0.3, 0.5])
cov1 = jnp.array([
    [0.002, 0.0],
    [0.0, 0.04]
])

mean2 = jnp.array([0.5, 0.5])
cov2 = jnp.array([
    [0.02, -0.018],
    [-0.018, 0.02]
])

mean3 = jnp.array([0.7, 0.5])
cov3 = jnp.array([
    [0.002, 0.0],
    [0.0, 0.04]
])

w1, w2, w3 = 0.34, 0.34, 0.33

def pdf(x):
    # only evaluate the first two dimensions (2D position)
    val1 = w1 * mvn.pdf(x[:2], mean_simple, cov_simple)
    return val1 

def log_pdf(x):
    return jnp.log(pdf(x))

score_pdf = grad(log_pdf)

# --- Dynamics ---
dt = 0.05
tsteps = 200
T = dt * tsteps

class PointMassLQR(LQR):
    def __init__(self, dt, x_dim, u_dim, Q, R):
        super().__init__(dt, x_dim, u_dim, Q, R)

    def dyn(self, xt, ut):
        return jnp.array([xt[2], xt[3], ut[0], ut[1]])

Q = jnp.diag(jnp.array([1.0, 1.0, 0.001, 0.001]))
R = jnp.diag(jnp.array([0.01, 0.01]))
pointmass_lqr = PointMassLQR(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)

linearize_dyn = jit(pointmass_lqr.linearize_dyn, device=cpu)
solve_lqr = jit(pointmass_lqr.solve, device=cpu)

# --- Stein Variational Gradient ---
def kernel(x1, x2, h):
    return jnp.exp(-1.0 * jnp.sum(jnp.square(x1[:2]-x2[:2])) / h)

d_kernel = jax.grad(kernel, argnums=(0))

def stein_grad_unit(x1, x2, h):
    val = kernel(x2, x1, h) * score_pdf(x2) + d_kernel(x2, x1, h)
    return val

def stein_grad_state(x, x_traj, h):
    vals = jax.vmap(stein_grad_unit, in_axes=(None, 0, None))(x, x_traj, h)
    return jnp.mean(vals, axis=0)

def stein_grad(traj, h):
    return jax.vmap(stein_grad_state, in_axes=(0, None, None))(traj, traj, h)

stein_grad = jax.jit(stein_grad, device=gpu)

# --- Main Generation Loop ---
def main():
    # --- Database Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "stein_coverage_results.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            x0_x REAL,
            x0_y REAL,
            trajectory BLOB,
            shape TEXT
        )
    ''')
    conn.commit()
    
    num_runs = 500
    step_size = 0.01
    num_iters = 200
    
    print(f"Starting generation of {num_runs} trajectories. Saving to {db_path}...")
    
    # Ensure JAX functions are compiled before starting the timing
    print("Compiling JAX functions...")
    _test_x0 = jnp.array([0.1, 0.2, 0.0, 0.0])
    _test_u_traj = jnp.zeros((tsteps, 2))
    _test_x_traj, _test_A, _test_B = linearize_dyn(_test_x0, _test_u_traj)
    _test_stein = stein_grad(_test_x_traj, h=0.01)
    solve_lqr(jnp.zeros(4), _test_A, _test_B, _test_stein)
    print("Compilation finished.")
    
    for run in tqdm(range(num_runs)):
        # Vary the initialization slightly
        # Base is [0.1, 0.2], vary uniformly by +/- 0.1
        _x0_np = np.array([0.1, 0.2]) + np.random.uniform(-0.1, 0.1, size=2)
        _x0 = jnp.array(_x0_np)
        
        x0 = jnp.array([
            _x0[0],
            _x0[1],
            2.0 * (0.5-_x0[0]) / T,
            2.0 * (0.5-_x0[1]) / T,
        ])
        u_traj = jnp.zeros((tsteps, 2))
        z0 = jnp.zeros(4)
        
        # Run optimization
        for i in range(num_iters):
            x_traj, A_traj, B_traj = linearize_dyn(x0, u_traj)
            stein_dx_traj = stein_grad(x_traj, h=0.01)
            v_traj, z_traj = solve_lqr(z0, A_traj, B_traj, stein_dx_traj)
            u_traj += step_size * v_traj 
            
        final_x_traj = pointmass_lqr.traj_sim(x0, u_traj)
        final_x_traj_np = np.array(final_x_traj)
        
        # Save to database
        shape_str = f"{final_x_traj_np.shape[0]},{final_x_traj_np.shape[1]}"
        cursor.execute('''
            INSERT INTO runs (x0_x, x0_y, trajectory, shape)
            VALUES (?, ?, ?, ?)
        ''', (float(_x0_np[0]), float(_x0_np[1]), final_x_traj_np.tobytes(), shape_str))
        
        if (run + 1) % 10 == 0:
            conn.commit()
    
    conn.commit()
    conn.close()
    print(f"Finished saving {num_runs} trajectories to {db_path}")

if __name__ == "__main__":
    main()
