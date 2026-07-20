#!/usr/bin/env python3
import optuna
import os
import json
import time
import argparse
import sys
import numpy as np
from datetime import datetime

import jax
import jax.numpy as jnp
from jax import vmap

# Add parent paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bsplinax-main")))

from init_strategies_nd import get_initialization
from ergodic_core import (
    build_target_distribution_3d,
    build_fourier_indices,
    compute_lambda_k,
    compute_target_fourier_coeffs,
    compute_ergodic_metric_jax,
    compute_ergodic_metric_numpy
)
from svgd_engine import (
    compute_smoothness_cost_jax,
    compute_boundary_cost_jax,
    compute_control_regularization_jax,
    forward_sim_nd,
    build_bspline_svgd_step,
    build_adam_optimizer_jax,
    init_bspline_from_positions_nd,
)

from bsplinax.bspline import BsplineBasisClamped

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for 3D B-Spline SVGD")
    parser.add_argument('--shape', type=str, default="N", choices=["N", "H", "II"], help="Target shape")
    parser.add_argument('--projection', type=str, default="plane", choices=["plane", "sphere", "cube"], help="Target shape projection")
    parser.add_argument('--trials', type=int, default=50, help="Number of Optuna trials")
    parser.add_argument('--strategy', type=str, default="n_shape", help="Initialization strategy to tune on")
    args = parser.parse_args()

    # Fixed config
    DIM = 3
    T = 100
    N_PARTICLES = 10  # Use fewer particles for faster tuning
    K_FOURIER = 5
    DEGREE = 3
    dt = 0.05
    W_CONTROL = 0.01

    print(f"Setting up target distribution for shape '{args.shape}' with projection '{args.projection}'...")
    _grid_axes, _grid_pts, _grid_weights, _grid_shape = build_target_distribution_3d(
        args.shape, stroke_width=0.045, grid_res=50, projection_type=args.projection
    )
    
    k_indices = build_fourier_indices(K_FOURIER, DIM)
    Lambda_k = compute_lambda_k(k_indices)
    phi_k = compute_target_fourier_coeffs(_grid_pts, _grid_weights, k_indices)

    k_indices_jnp = jnp.array(k_indices)
    Lambda_k_jnp = jnp.array(Lambda_k)
    phi_k_jnp = jnp.array(phi_k)

    def objective(trial):
        # 1. Sample hyperparameters
        N_ITERS = trial.suggest_int('n_iters', 1000, 10000, step=1000)
        NUM_CONTROL_POINTS = trial.suggest_int('num_control_points', 10, 40)
        W_ERGODIC = trial.suggest_float('w_ergodic', 100.0, 5000.0, log=True)
        W_SMOOTH = trial.suggest_float('w_smooth', 1.0, 100.0, log=True)
        W_BOUNDARY = trial.suggest_float('w_boundary', 10.0, 500.0, log=True)
        ADAM_LR = trial.suggest_float('adam_lr', 1e-4, 5e-2, log=True)

        # 2. Setup B-Spline Basis
        basis_generator = BsplineBasisClamped(
            degree=DEGREE,
            num_control_points=NUM_CONTROL_POINTS,
            num_phase_points=T,
            compute_derivatives=False
        )
        B_mat = jnp.array(basis_generator.B)
        B_outer = jnp.einsum('tk,tl->kl', B_mat, B_mat) * dt

        # 3. Energy Function definition (capturing sampled weights)
        def compute_energy_jax(C, s0):
            s_traj = forward_sim_nd(C, s0, B_mat, dt, DIM)
            X = s_traj[:, :DIM]
            
            energy = W_SMOOTH * compute_smoothness_cost_jax(X)
            energy += W_ERGODIC * compute_ergodic_metric_jax(X, k_indices_jnp, Lambda_k_jnp, phi_k_jnp)
            energy += W_BOUNDARY * compute_boundary_cost_jax(X)
            energy += W_CONTROL * compute_control_regularization_jax(C, B_outer)
            return energy

        grad_energy_jax = jax.grad(compute_energy_jax, argnums=0)

        # 4. Build SVGD Step & Optimizer
        svgd_step_fn = build_bspline_svgd_step(compute_energy_jax, grad_energy_jax, B_outer)
        optimize_C_all = build_adam_optimizer_jax(
            svgd_step_fn, N_ITERS, ADAM_LR, 
            adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
            chunk_size=1000, label=f"Trial {trial.number}"
        )

        sim_fn = jax.jit(vmap(lambda C, s0: forward_sim_nd(C, s0, B_mat, dt, DIM), in_axes=(0, 0)))

        # 5. Initialization
        init_p, _ = get_initialization(args.strategy, N_PARTICLES, T, dim=DIM, noise_std=0.02)
        pos_trajs = init_p.reshape(N_PARTICLES, T, DIM)
        C_init, x0_init = init_bspline_from_positions_nd(pos_trajs, dt, np.array(B_mat), DIM)

        C_all = jnp.array(C_init)
        x0_all = jnp.array(x0_init)

        # 6. Run optimization
        C_all_opt, energy_log = optimize_C_all(C_all, x0_all, label_override=f"Trial {trial.number}")

        # 7. Evaluate
        final_x_trajs = np.array(sim_fn(C_all_opt, x0_all))
        final_pos = final_x_trajs[:, :, :DIM].reshape(N_PARTICLES, -1)

        final_ergs = []
        for i in range(N_PARTICLES):
            X_i = final_pos[i].reshape(T, DIM)
            erg = compute_ergodic_metric_numpy(X_i, k_indices, Lambda_k, phi_k)
            final_ergs.append(erg)

        # We return the best pure ergodic metric from the ensemble
        best_erg = float(np.min(final_ergs))
        return best_erg

    # Create Optuna study
    study = optuna.create_study(direction='minimize', study_name=f"3D_BSpline_SVGD_{args.shape}_{args.projection}")
    print(f"Starting tuning for {args.trials} trials...")
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "="*50)
    print("Tuning Completed!")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best Ergodic Metric: {study.best_value:.5f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    timestamp = datetime.now().strftime("%H-%M_%d-%m")
    out_file = f"best_hyperparams_{args.shape}_{args.projection}_{timestamp}.json"
    
    with open(out_file, "w") as f:
        json.dump({
            "best_metric": study.best_value,
            "best_params": study.best_params,
            "shape": args.shape,
            "projection": args.projection,
            "strategy": args.strategy
        }, f, indent=4)
    print(f"\nSaved best hyperparameters to {out_file}")


if __name__ == "__main__":
    main()
