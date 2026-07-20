#!/usr/bin/env python3
"""
Unified B-Spline + Flow Matching + SVGD Ergodic Pipeline  (Component 5)
=========================================================================
Three-phase pipeline runner for ergodic trajectory design on a 2D domain.

Phase 1 (Offline):   Train Spline-CFM  →  pushforward map
Phase 2 (Interface): Generate N diverse B-spline control point ensembles
Phase 3 (Online):    SVGD refinement  →  final trajectories

Provides a `run_benchmark()` function matching the interface of all other
methods for integration with master_benchmark.py.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis")

from bspline_trajectory import BSplineTrajectoryAdapter
from spline_cfm_trainer import (
    SplineVelocityNet,
    train_spline_cfm,
    sample_annulus,
)
from ensemble_generator import generate_ensemble
from log_surrogate_mmd import FourierErgodicMetric
from signature_kernel_svgd import run_svgd_bspline


# ============================================================================
# Configuration
# ============================================================================

TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')

# B-spline parameters
DEGREE = 4
NUM_CONTROL_POINTS = 16
T = 100                    # dense trajectory waypoints

# Phase 1: CFM training
CFM_EPOCHS = 1500
CFM_BATCH_SIZE = 256
CFM_LR = 2e-3
CFM_HIDDEN_DIM = 256
CFM_N_LAYERS = 4

# Phase 2: Ensemble generation
N_PARTICLES = 10

# Phase 3: SVGD refinement
SVGD_ITERS = 150
SVGD_LR = 0.005

# Energy weights (matching other methods for comparability)
W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0
W_OBSTACLE = 50000.0

# Obstacle
USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5]
OBSTACLE_RADIUS = 0.12

# Ergodic metric
USE_LOG_SURROGATE = True
USE_SIG_KERNEL = True       # Use signature kernel (True) or RBF (False)
K_FOURIER = 10


# ============================================================================
# Evaluation: Standard Fourier Energy (for cross-method comparability)
# ============================================================================

def compute_fourier_energy(traj_flat, T_steps, metric, w_erg, w_sm, w_bnd,
                           use_obstacle=False, obs_center=(0.5, 0.5),
                           obs_radius=0.12, w_obs=50000.0):
    """Compute the same Fourier energy metric used by all other methods."""
    X = traj_flat.reshape(T_steps, 2)
    energy = 0.0

    # Smoothness (acceleration penalty)
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    energy += w_sm * np.sum(accel ** 2)

    # Ergodic cost (standard Fourier, not log-surrogate)
    energy += w_erg * metric.ergodic_cost(X)

    # Boundary penalty
    margin = 0.03
    lo = np.minimum(X - margin, 0.0)
    hi = np.maximum(X - (1.0 - margin), 0.0)
    energy += w_bnd * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))

    # Obstacle
    if use_obstacle:
        dx = X[:, 0] - obs_center[0]
        dy = X[:, 1] - obs_center[1]
        dist = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)
        violation = np.maximum(obs_radius - dist, 0.0)
        energy += w_obs * 0.5 * np.sum(violation ** 2)

    return energy


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_plot(
    initial_trajs, final_trajs, metric, best_idx,
    use_obstacle, out_path, n_particles, T_steps,
    phase1_loss=None, phase3_energy_log=None,
    initial_cps=None, final_cps=None,
):
    """Create a rich comparison-grid visualization."""
    Xg, Yg, Zg = metric.Xg, metric.Yg, metric.Zg
    cmap = 'YlOrRd'
    colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))

    # Layout: 2×2 grid
    #   [0,0] Initial ensemble          [0,1] Final ensemble
    #   [1,0] Control polygon overlay   [1,1] Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def plot_trajs(ax, trajs, title, highlight_best=False, bidx=-1, cps=None):
        ax.contourf(Xg, Yg, Zg, levels=30, cmap=cmap, alpha=0.6)
        ax.contour(Xg, Yg, Zg, levels=6, colors='k', linewidths=0.3, alpha=0.3)
        if use_obstacle:
            circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                                color='gray', alpha=0.8, zorder=5)
            ax.add_patch(circle)
        for i in range(n_particles):
            tr = trajs[i].reshape(T_steps, 2)
            lw = 2.5 if (highlight_best and i == bidx) else 0.8
            al = 1.0 if (highlight_best and i == bidx) else 0.4
            ax.plot(tr[:, 0], tr[:, 1], '-', color=colors[i], lw=lw, alpha=al)
            ax.plot(tr[0, 0], tr[0, 1], 'o', color=colors[i], ms=4)
            if cps is not None:
                cp = cps[i]
                ax.plot(cp[:, 0], cp[:, 1], '--s', color=colors[i], lw=0.5, ms=2, alpha=0.3)
                
        if highlight_best and bidx >= 0:
            best_tr = trajs[bidx].reshape(T_steps, 2)
            ax.plot(best_tr[:, 0], best_tr[:, 1], '-', color=colors[bidx],
                    lw=2.5, label=f'Best (#{bidx})', zorder=10)
            if cps is not None:
                best_cp = cps[bidx]
                ax.plot(best_cp[:, 0], best_cp[:, 1], '--s', color=colors[bidx], lw=1.5, ms=4, label=f'Best CP', zorder=11)
            ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x'); ax.set_ylabel('y')

    # [0,0] Initial
    plot_trajs(axes[0, 0], initial_trajs, 'Phase 2: CFM Initial Ensemble (NN Output)', cps=initial_cps)

    # [0,1] Final
    plot_trajs(axes[0, 1], final_trajs, 'Phase 3: SVGD Refined',
               highlight_best=True, bidx=best_idx, cps=final_cps)

    # [1,0] Control polygon overlay
    ax = axes[1, 0]
    ax.contourf(Xg, Yg, Zg, levels=30, cmap=cmap, alpha=0.4)
    if use_obstacle:
        circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                            color='gray', alpha=0.8, zorder=5)
        ax.add_patch(circle)
    # Show best trajectory with control polygon (initial vs final)
    if best_idx >= 0:
        best_tr = final_trajs[best_idx].reshape(T_steps, 2)
        ax.plot(best_tr[:, 0], best_tr[:, 1], '-', color=colors[best_idx],
                lw=2.0, alpha=0.8, label='Best Final Trajectory')
        if final_cps is not None:
            best_fcp = final_cps[best_idx]
            ax.plot(best_fcp[:, 0], best_fcp[:, 1], '--s', color=colors[best_idx], lw=1.5, ms=5, label='Final Control Polygon')
        if initial_cps is not None:
            best_icp = initial_cps[best_idx]
            init_tr = initial_trajs[best_idx].reshape(T_steps, 2)
            ax.plot(init_tr[:, 0], init_tr[:, 1], ':', color='gray', lw=1.5, alpha=0.6, label='Initial Trajectory')
            ax.plot(best_icp[:, 0], best_icp[:, 1], '--^', color='gray', lw=1.0, ms=4, alpha=0.6, label='Initial Control Polygon (NN)')
            
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title('Best Trajectory Control Polygons: Initial vs Final', fontsize=13, fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(loc='lower right', fontsize=9)

    # [1,1] Training curves
    ax = axes[1, 1]
    if phase1_loss is not None:
        ax.plot(phase1_loss, 'b-', alpha=0.7, label='Phase 1: CFM loss')
    ax.set_xlabel('Epoch / Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if phase3_energy_log is not None:
        ax2 = ax.twinx()
        ax2.plot(phase3_energy_log, 'r-', alpha=0.7, label='Phase 3: mean energy')
        ax2.set_ylabel('Mean Energy', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right', fontsize=9)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()



# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
    target_shape='N',
    use_obstacle=False,
    verbose=True,
):
    """
    Execute the full three-phase pipeline.

    Returns
    -------
    result : dict with keys:
        'initial_trajs'  : (N, T*2)
        'final_trajs'    : (N, T*2)
        'control_points' : (N, n_ctrl, 2)
        'best_idx'       : int
        'phase1_loss'    : list
        'phase3_elog'    : list
        'time_s'         : float
    """
    t_start = time.time()

    # ── Initialize components ──────────────────────────────────────
    adapter = BSplineTrajectoryAdapter(
        degree=DEGREE,
        num_control_points=NUM_CONTROL_POINTS,
        num_phase_points=T,
        spatial_dim=2,
    )
    metric = FourierErgodicMetric(target_shape=target_shape, K=K_FOURIER)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Unified B-Spline + CFM + SVGD Pipeline")
        print(f"  Target: {target_shape}  |  B-spline: deg={DEGREE}, "
              f"n_ctrl={NUM_CONTROL_POINTS}, T={T}")
        print(f"  Particles: {N_PARTICLES}  |  SVGD iters: {SVGD_ITERS}")
        print(f"  Weights: erg={W_ERGODIC}, smooth={W_SMOOTH}, bnd={W_BOUNDARY}")
        print(f"  Log-Surrogate: {USE_LOG_SURROGATE}  |  "
              f"SigKernel: {USE_SIG_KERNEL}")
        print(f"{'='*65}")

    # ── Phase 1: Train Spline-CFM ─────────────────────────────────
    if verbose:
        print("\n── Phase 1: Training Spline-CFM pushforward map ──")
    model = SplineVelocityNet(
        n_ctrl=NUM_CONTROL_POINTS,
        spatial_dim=2,
        hidden_dim=CFM_HIDDEN_DIM,
        n_layers=CFM_N_LAYERS,
    )
    phase1_loss = train_spline_cfm(
        model, adapter,
        target_shape=target_shape,
        epochs=CFM_EPOCHS,
        batch_size=CFM_BATCH_SIZE,
        lr=CFM_LR,
        verbose=verbose,
    )
    t_phase1 = time.time() - t_start

    # ── Phase 2: Generate diverse ensemble ────────────────────────
    if verbose:
        print(f"\n── Phase 2: Generating {N_PARTICLES} diverse B-spline "
              f"priors ({t_phase1:.1f}s elapsed) ──")
    init_cps, init_trajs, z_latent = generate_ensemble(
        model, adapter,
        n_particles=N_PARTICLES,
        strategy='spread',
        n_ode_steps=30,
    )
    initial_trajs_flat = init_trajs.reshape(N_PARTICLES, -1)
    t_phase2 = time.time() - t_start

    # ── Phase 3: SVGD refinement ──────────────────────────────────
    if verbose:
        print(f"\n── Phase 3: SVGD refinement ({t_phase2:.1f}s elapsed) ──")
    final_cps, phase3_elog = run_svgd_bspline(
        init_cps, adapter, metric,
        n_iters=SVGD_ITERS,
        lr=SVGD_LR,
        w_ergodic=W_ERGODIC,
        w_smooth=W_SMOOTH,
        w_boundary=W_BOUNDARY,
        w_obstacle=W_OBSTACLE,
        use_obstacle=use_obstacle,
        obstacle_center=OBSTACLE_CENTER,
        obstacle_radius=OBSTACLE_RADIUS,
        use_log_surrogate=USE_LOG_SURROGATE,
        use_sig_kernel=USE_SIG_KERNEL,
        verbose=verbose,
    )
    elapsed = time.time() - t_start

    # Reconstruct final dense trajectories
    final_trajs = adapter.control_points_to_trajectory(final_cps)
    final_trajs = np.clip(final_trajs, 0.01, 0.99)
    final_trajs_flat = final_trajs.reshape(N_PARTICLES, -1)

    # Evaluate final energies
    final_E = np.array([
        compute_fourier_energy(
            final_trajs_flat[i], T, metric,
            W_ERGODIC, W_SMOOTH, W_BOUNDARY,
            use_obstacle=use_obstacle,
            obs_center=OBSTACLE_CENTER,
            obs_radius=OBSTACLE_RADIUS,
            w_obs=W_OBSTACLE,
        )
        for i in range(N_PARTICLES)
    ])
    best_idx = int(np.argmin(final_E))

    if verbose:
        print(f"\n── Results ({elapsed:.1f}s total) ──")
        print(f"  Best energy:  {final_E[best_idx]:.3f}")
        print(f"  Mean energy:  {np.mean(final_E):.3f}")
        print(f"  Phase timing: P1={t_phase1:.1f}s  P2={t_phase2-t_phase1:.1f}s  "
              f"P3={elapsed-t_phase2:.1f}s")

    return {
        'initial_trajs': initial_trajs_flat,
        'final_trajs': final_trajs_flat,
        'initial_cps': init_cps,
        'control_points': final_cps,
        'best_idx': best_idx,
        'final_energies': final_E,
        'phase1_loss': phase1_loss,
        'phase3_elog': phase3_elog,
        'time_s': elapsed,
    }


# ============================================================================
# Benchmark Interface  (matching other methods)
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    """
    Run the unified pipeline and produce results in the same format as
    all other benchmark methods.

    Parameters
    ----------
    out_dir      : str — output directory
    save_npy     : bool — save trajectory arrays
    use_obstacle : bool — enable obstacle avoidance

    Returns
    -------
    benchmark_data : dict — per-strategy results (single strategy: 'unified')
    """
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle
    os.makedirs(out_dir, exist_ok=True)

    target_shape = os.environ.get('TARGET_SHAPE', 'N')

    # Run the pipeline
    result = run_pipeline(
        target_shape=target_shape,
        use_obstacle=use_obstacle,
        verbose=True,
    )

    # Build benchmark data (single strategy: 'unified')
    benchmark_data = {
        'unified': {
            'mean_cost': float(np.mean(result['final_energies'])),
            'best_cost': float(result['final_energies'][result['best_idx']]),
            'time_s': float(result['time_s']),
        }
    }

    # Create metric for plotting
    metric = FourierErgodicMetric(target_shape=target_shape, K=K_FOURIER)

    # Save visualization
    out_path = os.path.join(
        out_dir,
        f'unified_pipeline_{target_shape}_results.png'
    )
    create_comparison_plot(
        result['initial_trajs'],
        result['final_trajs'],
        metric,
        result['best_idx'],
        use_obstacle,
        out_path,
        N_PARTICLES, T,
        phase1_loss=result['phase1_loss'],
        phase3_energy_log=result['phase3_elog'],
        initial_cps=result['initial_cps'],
        final_cps=result['control_points'],
    )

    # Save settings
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'target_shape': target_shape,
            'degree': DEGREE,
            'num_control_points': NUM_CONTROL_POINTS,
            'T': T,
            'N_PARTICLES': N_PARTICLES,
            'CFM_EPOCHS': CFM_EPOCHS,
            'SVGD_ITERS': SVGD_ITERS,
            'W_ERGODIC': W_ERGODIC,
            'W_SMOOTH': W_SMOOTH,
            'W_BOUNDARY': W_BOUNDARY,
            'USE_LOG_SURROGATE': USE_LOG_SURROGATE,
            'USE_SIG_KERNEL': USE_SIG_KERNEL,
            'use_obstacle': use_obstacle,
        }, f, indent=4)

    if save_npy:
        np.save(os.path.join(out_dir, 'initial_trajs.npy'), result['initial_trajs'])
        np.save(os.path.join(out_dir, 'final_trajs.npy'), result['final_trajs'])
        np.save(os.path.join(out_dir, 'control_points.npy'), result['control_points'])

    print(f"\nResults saved to {out_dir}")
    return benchmark_data


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    run_benchmark(
        out_dir=f'/home/philipp/Documents/Uni/Master_thesis/results/Unified_Pipeline_{TARGET_SHAPE}',
    )
