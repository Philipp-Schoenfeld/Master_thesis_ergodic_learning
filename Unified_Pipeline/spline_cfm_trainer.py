#!/usr/bin/env python3
"""
Spline-Based OT-CFM Trainer  (Component 2)
===========================================
Trains a pushforward map from a latent annulus to the target density,
operating on **B-spline control points** instead of raw spatial
coordinates.

The VelocityNet maps  (s, z) → ℝ^{n_ctrl × 2}  (flattened control
point positions).  Training uses the same OT-CFM framework as
pf_ergodic_core.py but with B-spline parameterization.

Adapted from:
  - experiments/PF_Ergodic/pf_ergodic_core.py  (OT-CFM training)
  - SE3_SVGD/tsvec_2d.py                       (target distribution)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

# Local imports
sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis")
sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/Unified_Pipeline")
from bspline_trajectory import BSplineTrajectoryAdapter


# ============================================================================
# 1. Target Distribution — Letter Shapes  (on [0,1]²)
# ============================================================================

SEGMENT_DEFS = {
    'N': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.25, 0.85], [0.75, 0.15]),
        ([0.75, 0.15], [0.75, 0.85]),
    ],
    'H': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
        ([0.25, 0.50], [0.75, 0.50]),
    ],
    'II': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
    ],
}
STROKE_WIDTH = 0.045


def _dist_to_segment_np(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (len_sq + 1e-12), 0, 1)
    return np.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)


def sample_target_trajectories(n_trajs: int,
                                T: int,
                                target_shape: str = 'N',
                                noise_std: float = 0.01) -> np.ndarray:
    """
    Generate target trajectories that trace the letter shape.

    Each trajectory is a random walk along the segments of the target shape,
    with small Gaussian noise for diversity.

    Parameters
    ----------
    n_trajs : int       — number of trajectories to generate
    T       : int       — number of waypoints per trajectory
    target_shape : str  — one of 'N', 'H', 'II'
    noise_std : float   — Gaussian noise standard deviation

    Returns
    -------
    trajs : ndarray (n_trajs, T, 2)
    """
    segments = SEGMENT_DEFS[target_shape]
    n_segs = len(segments)
    trajs = np.zeros((n_trajs, T, 2))

    for i in range(n_trajs):
        # Build a concatenated path along all segments
        pts_per_seg = T // n_segs
        remainder = T - pts_per_seg * n_segs
        path = []
        for si, ((ax, ay), (bx, by)) in enumerate(segments):
            n_pts = pts_per_seg + (1 if si < remainder else 0)
            ts = np.linspace(0, 1, n_pts, endpoint=(si == n_segs - 1))
            seg_pts = np.column_stack([
                ax + ts * (bx - ax),
                ay + ts * (by - ay),
            ])
            path.append(seg_pts)
        path = np.concatenate(path, axis=0)[:T]

        # Randomly reverse with 50% probability
        if np.random.rand() > 0.5:
            path = path[::-1].copy()

        # Random cyclic shift for diversity
        shift = np.random.randint(0, T)
        path = np.roll(path, shift, axis=0)

        # Add small noise
        path += np.random.randn(T, 2) * noise_std
        path = np.clip(path, 0.02, 0.98)

        trajs[i] = path

    return trajs


# ============================================================================
# 2. Source Distribution — Latent Annulus
# ============================================================================

def sample_annulus(n: int, delta: float = 0.05) -> torch.Tensor:
    """
    Sample uniformly from the annular latent domain D_delta.
    Returns shape (n, 2).
    """
    s = torch.rand(n)
    r = torch.sqrt(delta ** 2 + (1.0 - delta ** 2) * s)
    theta = 2.0 * torch.pi * torch.rand(n)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)


# ============================================================================
# 3. Spline-Adapted Velocity Network
# ============================================================================

class SplineVelocityNet(nn.Module):
    """
    Time-conditioned velocity field  v_θ(s, z) → ℝ^{n_ctrl * D}.

    Input:  s ∈ [0,1]  (flow time) + z ∈ ℝ²  (latent position) → dim 3
    Output: flattened control point velocity ∈ ℝ^{n_ctrl * D}.

    Zero-initialization on the last layer so training starts from an
    identity-like map (zero displacement).
    """

    def __init__(self,
                 n_ctrl: int = 16,
                 spatial_dim: int = 2,
                 hidden_dim: int = 256,
                 n_layers: int = 4):
        super().__init__()
        self.n_ctrl = n_ctrl
        self.spatial_dim = spatial_dim
        out_dim = n_ctrl * spatial_dim

        layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Zero-init last layer for identity-like start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s : (B,) or (B,1)  flow time
            z : (B,2)          latent position
        Returns:
            (B, n_ctrl * D)    flattened control point velocity
        """
        if s.dim() == 1:
            s = s.unsqueeze(1)
        return self.net(torch.cat([s, z], dim=1))


# ============================================================================
# 4. OT-CFM Training on B-Spline Control Points
# ============================================================================

def train_spline_cfm(
    model: SplineVelocityNet,
    adapter: BSplineTrajectoryAdapter,
    target_shape: str = 'N',
    epochs: int = 2000,
    batch_size: int = 256,
    delta: float = 0.05,
    lr: float = 2e-3,
    w_accel: float = 0.1,
    w_boundary: float = 1.0,
    verbose: bool = True,
) -> list:
    """
    Train the pushforward map G_θ using Optimal-Transport Conditional
    Flow Matching on B-spline control points.

    The network learns to map latent annulus points to control point
    positions that, when reconstructed via B @ w, trace the target
    distribution.

    Loss:
        L = E[ || v_θ(s, y_s) - (x1 - z0) ||² ]
              + w_accel * accel_penalty
              + w_boundary * boundary_penalty

    where (z0, x1) are coupled via Hungarian OT to minimise transport cost.

    Parameters
    ----------
    model        : SplineVelocityNet
    adapter      : BSplineTrajectoryAdapter
    target_shape : str — letter target shape
    epochs       : int
    batch_size   : int
    delta        : float — annulus inner radius
    lr           : float — learning rate
    w_accel      : float — acceleration regularization weight
    w_boundary   : float — boundary penalty weight
    verbose      : bool

    Returns
    -------
    loss_log : list of per-epoch scalar losses
    """
    n_ctrl = adapter.num_control_points
    D = adapter.spatial_dim
    T = adapter.num_phase_points
    torch_B = adapter.torch_B       # (T, n_ctrl)
    torch_d2B = adapter.torch_d2B   # (T, n_ctrl)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_log = []

    # Pre-generate target trajectories and fit to control points
    n_target_pool = max(batch_size * 4, 2000)
    target_trajs = sample_target_trajectories(
        n_target_pool, T, target_shape=target_shape, noise_std=0.015
    )
    # Fit each target trajectory to B-spline control points
    target_cps = adapter.trajectory_to_control_points(target_trajs)  # (pool, n_ctrl, D)
    target_cps_flat = target_cps.reshape(n_target_pool, n_ctrl * D)
    target_cps_torch = torch.from_numpy(target_cps_flat).float()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- Sample source (latent annulus) ---
        z0 = sample_annulus(batch_size, delta)               # (B, 2)

        # --- Sample target control points from the pre-generated pool ---
        idx = torch.randint(0, n_target_pool, (batch_size,))
        x1 = target_cps_torch[idx]                           # (B, n_ctrl*D)

        # --- Mini-batch OT coupling (Hungarian) ---
        # Use latent source positions vs centroid of target control points
        target_centroids = x1.reshape(batch_size, n_ctrl, D).mean(dim=1)  # (B, 2)
        C = torch.cdist(z0, target_centroids) ** 2
        _, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
        x1_coupled = x1[col_ind]                             # (B, n_ctrl*D)

        # --- Straight-line interpolant ---
        s = torch.rand(batch_size, 1)
        # Embed z0 into control-point space:  each latent point → constant CP
        # z0_cp[i,j,:] = z0[i,:] for all j → flattened
        z0_cp = z0.unsqueeze(1).expand(-1, n_ctrl, -1).reshape(batch_size, n_ctrl * D)
        y_s = (1.0 - s) * z0_cp + s * x1_coupled

        # Target velocity: straight-line displacement
        v_target = x1_coupled - z0_cp                        # (B, n_ctrl*D)

        # --- CFM loss ---
        v_pred = model(s.squeeze(1), y_s[:, :2])  # Use first 2 dims as spatial
        # Actually, the network input should be the latent position, not
        # the interpolated control points.  Let's use the current interpolated
        # spatial centroid as input position (same structure as original CFM).
        y_s_reshaped = y_s.reshape(batch_size, n_ctrl, D)
        y_s_centroid = y_s_reshaped.mean(dim=1)              # (B, 2)
        v_pred = model(s.squeeze(1), y_s_centroid)

        loss_cfm = torch.mean((v_pred - v_target) ** 2)

        # --- Acceleration regularization ---
        # Reconstruct dense trajectory from current interpolated control points
        # and penalize excessive acceleration
        with torch.no_grad():
            w_current = y_s.reshape(batch_size, n_ctrl, D)
        w_for_accel = y_s.reshape(batch_size, n_ctrl, D)
        # traj = B @ w:  (T, n_ctrl) @ (B, n_ctrl, D) → (B, T, D)
        traj_recon = torch.einsum('tp,bpd->btd', torch_B, w_for_accel)
        accel = traj_recon[:, 2:] - 2 * traj_recon[:, 1:-1] + traj_recon[:, :-2]
        loss_accel = w_accel * torch.mean(accel ** 2)

        # --- Boundary penalty on reconstructed trajectory ---
        margin = 0.03
        lo = torch.clamp(margin - traj_recon, min=0.0)
        hi = torch.clamp(traj_recon - (1.0 - margin), min=0.0)
        loss_boundary = w_boundary * torch.mean(lo ** 2 + hi ** 2)

        loss = loss_cfm + loss_accel + loss_boundary
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_log.append(loss.item())

        if verbose and epoch % 300 == 0:
            print(f"  [Spline-CFM] Epoch {epoch:04d}/{epochs}  "
                  f"loss={loss.item():.4f}  "
                  f"(cfm={loss_cfm.item():.4f}, "
                  f"accel={loss_accel.item():.4f}, "
                  f"bnd={loss_boundary.item():.4f})")

    model.eval()
    return loss_log


# ============================================================================
# 5. RK4 ODE Integrator + Pushforward  (on control point space)
# ============================================================================

def _rk4_step(model, s, y, ds, n_ctrl, D):
    """Single RK4 step integrating dy/ds = v_θ(s, y) in CP space."""
    # y is (N, n_ctrl*D) — flattened control points
    # Extract centroid as spatial position for network input
    y_reshaped = y.reshape(-1, n_ctrl, D)
    centroid = y_reshaped.mean(dim=1)    # (N, 2)

    def _vel(s_val, y_val):
        y_r = y_val.reshape(-1, n_ctrl, D)
        c = y_r.mean(dim=1)
        return model(s_val, c)

    k1 = _vel(s, y)
    k2 = _vel(s + ds / 2, y + k1 * ds / 2)
    k3 = _vel(s + ds / 2, y + k2 * ds / 2)
    k4 = _vel(s + ds, y + k3 * ds)
    return y + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def pushforward_spline(model: SplineVelocityNet,
                       z_latent: torch.Tensor,
                       n_ctrl: int,
                       spatial_dim: int = 2,
                       n_steps: int = 30) -> torch.Tensor:
    """
    Integrate the learned velocity field from s=0 to s=1 to transform
    latent positions into B-spline control points.

    Parameters
    ----------
    model     : trained SplineVelocityNet
    z_latent  : (N, 2) latent positions on the annulus
    n_ctrl    : number of control points
    spatial_dim : spatial dimensionality
    n_steps   : RK4 integration steps

    Returns
    -------
    w : (N, n_ctrl, D) — pushed-forward control points in target space
    """
    model.eval()
    N = z_latent.shape[0]
    D = spatial_dim

    # Initialize: embed latent position as constant control points
    y = z_latent.unsqueeze(1).expand(-1, n_ctrl, -1).reshape(N, n_ctrl * D)

    ds = 1.0 / n_steps
    for i in range(n_steps):
        s = torch.full((N,), i * ds)
        y = _rk4_step(model, s, y, ds, n_ctrl, D)

    return y.reshape(N, n_ctrl, D)


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    print("Spline-CFM Trainer — self-test")

    adapter = BSplineTrajectoryAdapter(
        degree=4, num_control_points=16, num_phase_points=100
    )
    model = SplineVelocityNet(
        n_ctrl=16, spatial_dim=2, hidden_dim=128, n_layers=3
    )

    # Quick training (50 epochs for smoke test)
    loss_log = train_spline_cfm(
        model, adapter,
        target_shape='N', epochs=50, batch_size=64,
        verbose=True
    )
    print(f"  Final loss: {loss_log[-1]:.4f}")

    # Pushforward test
    z = sample_annulus(5, delta=0.05)
    w = pushforward_spline(model, z, n_ctrl=16, spatial_dim=2, n_steps=10)
    print(f"  Pushforward output shape: {w.shape}")

    trajs = adapter.control_points_to_trajectory(w.numpy())
    print(f"  Reconstructed trajectories shape: {trajs.shape}")
    print("  ✓ Self-test passed.")
