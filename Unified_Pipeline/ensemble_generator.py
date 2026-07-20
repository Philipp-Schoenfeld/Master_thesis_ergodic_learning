#!/usr/bin/env python3
"""
Ensemble Generator  (Component 3)
==================================
Given a trained Spline-CFM model, generate N diverse B-spline trajectory
priors by sampling latent points on the annulus and pushing them through
the learned ODE.

Interface phase between offline training (Phase 1) and online SVGD
refinement (Phase 3).
"""

import numpy as np
import torch

from bspline_trajectory import BSplineTrajectoryAdapter
from spline_cfm_trainer import (
    SplineVelocityNet,
    pushforward_spline,
    sample_annulus,
)


def generate_ensemble(
    model: SplineVelocityNet,
    adapter: BSplineTrajectoryAdapter,
    n_particles: int = 10,
    delta: float = 0.05,
    n_ode_steps: int = 30,
    strategy: str = 'uniform',
) -> tuple:
    """
    Generate N diverse B-spline control point ensembles from the trained
    pushforward model.

    Parameters
    ----------
    model       : trained SplineVelocityNet
    adapter     : BSplineTrajectoryAdapter
    n_particles : int — number of trajectory particles to generate
    delta       : float — annulus inner radius
    n_ode_steps : int — RK4 integration steps
    strategy    : str — latent sampling strategy:
                  'uniform'  — uniform random on the annulus
                  'spread'   — evenly spaced angles for maximum diversity

    Returns
    -------
    control_points : ndarray (N, n_ctrl, D) — B-spline control points
    trajectories   : ndarray (N, T, D)      — reconstructed dense trajectories
    z_latent       : ndarray (N, 2)         — latent positions used
    """
    n_ctrl = adapter.num_control_points
    D = adapter.spatial_dim

    # Sample latent points
    if strategy == 'spread':
        # Evenly spaced angles on the annulus for maximum initial diversity
        angles = np.linspace(0, 2 * np.pi, n_particles, endpoint=False)
        # Randomize radii within the annulus
        r = np.sqrt(delta ** 2 + (1.0 - delta ** 2) * np.random.rand(n_particles))
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        z_latent = torch.tensor(np.column_stack([x, y]), dtype=torch.float32)
    else:
        z_latent = sample_annulus(n_particles, delta)

    # Push through the learned ODE
    w_torch = pushforward_spline(
        model, z_latent,
        n_ctrl=n_ctrl, spatial_dim=D, n_steps=n_ode_steps
    )
    control_points = w_torch.numpy()  # (N, n_ctrl, D)

    # Clamp control points to stay within [0, 1]²
    control_points = np.clip(control_points, 0.02, 0.98)

    # Reconstruct dense trajectories
    trajectories = adapter.control_points_to_trajectory(control_points)  # (N, T, D)

    # Also clamp the trajectories
    trajectories = np.clip(trajectories, 0.01, 0.99)

    return control_points, trajectories, z_latent.numpy()


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/Unified_Pipeline")

    print("Ensemble Generator — self-test")

    adapter = BSplineTrajectoryAdapter(
        degree=4, num_control_points=16, num_phase_points=100
    )
    model = SplineVelocityNet(n_ctrl=16, spatial_dim=2, hidden_dim=128, n_layers=3)

    # Use untrained model for smoke test
    cps, trajs, z_lat = generate_ensemble(
        model, adapter, n_particles=5, strategy='spread'
    )
    print(f"  Control points shape: {cps.shape}")
    print(f"  Trajectories shape:   {trajs.shape}")
    print(f"  Latent points shape:  {z_lat.shape}")
    print(f"  Traj range: [{trajs.min():.3f}, {trajs.max():.3f}]")
    print("  ✓ Self-test passed.")
