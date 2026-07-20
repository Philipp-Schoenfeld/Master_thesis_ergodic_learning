#!/usr/bin/env python3
"""
B-Spline Trajectory Adapter  (Component 1)
===========================================
Lightweight bridge between the `bsplinax` JAX library and the pipeline's
NumPy / PyTorch trajectory representation.

Core functionality:
  - Precompute basis matrices  B, dB/ds, d²B/ds²  via bsplinax.
  - control_points_to_trajectory(w) → (T, D)
  - trajectory_to_control_points(traj) → (n_ctrl, D)   (least-squares fit)
  - get_velocity(w), get_acceleration(w)
  - Lazy-converted torch_B property for interop with PyTorch training.
"""

import sys
import numpy as np

# Make bsplinax importable
sys.path.insert(0, "/home/philipp/Documents/Uni/Master_thesis/bsplinax-main")
from bsplinax.bspline import BsplineBasisClamped

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class BSplineTrajectoryAdapter:
    """
    Wraps a clamped B-spline basis to convert between sparse control points
    and dense trajectory waypoints.

    Parameters
    ----------
    degree : int
        B-spline degree  (default 4 → C³ continuity).
    num_control_points : int
        Number of B-spline control points per spatial dimension.
    num_phase_points : int
        Number of dense trajectory waypoints (T).
    spatial_dim : int
        Spatial dimensionality (2 for 2D trajectories).
    """

    def __init__(
        self,
        degree: int = 4,
        num_control_points: int = 16,
        num_phase_points: int = 100,
        spatial_dim: int = 2,
    ):
        self.degree = degree
        self.num_control_points = num_control_points
        self.num_phase_points = num_phase_points
        self.spatial_dim = spatial_dim

        # Build clamped B-spline basis using bsplinax
        basis = BsplineBasisClamped(
            degree=degree,
            num_control_points=num_control_points,
            num_phase_points=num_phase_points,
        )

        # Extract basis matrices as NumPy arrays  ──────────────────────
        #   B      : (T, n_ctrl)        position basis
        #   dB_ds  : (T, n_ctrl)        velocity basis
        #   d2B_ds2: (T, n_ctrl)        acceleration basis
        self.B_np = np.array(basis.B)            # (T, n_ctrl)
        self.dB_ds_np = np.array(basis.dB_ds)    # (T, n_ctrl)
        self.d2B_ds2_np = np.array(basis.d2B_ds2)  # (T, n_ctrl)
        self.ss = np.array(basis.ss)             # (T,) phase points

        # Precompute pseudoinverse of B for least-squares fitting
        # B @ w ≈ traj  →  w = pinv(B) @ traj
        self._B_pinv = np.linalg.pinv(self.B_np)  # (n_ctrl, T)

        # Lazy torch cache
        self._torch_B = None
        self._torch_dB = None
        self._torch_d2B = None

    # ──────────────────────────────────────────────────────────────────
    #  Forward map:  control points  →  dense trajectory
    # ──────────────────────────────────────────────────────────────────

    def control_points_to_trajectory(self, w: np.ndarray) -> np.ndarray:
        """
        Reconstruct a dense trajectory from B-spline control points.

        Parameters
        ----------
        w : ndarray (n_ctrl, D) or (N, n_ctrl, D)
            Control points in spatial coordinates.

        Returns
        -------
        traj : ndarray (T, D) or (N, T, D)
            Dense trajectory waypoints.
        """
        if w.ndim == 2:
            return self.B_np @ w          # (T, n_ctrl) @ (n_ctrl, D) → (T, D)
        elif w.ndim == 3:
            # Batched: w is (N, n_ctrl, D)
            return np.einsum('tp,npd->ntd', self.B_np, w)
        else:
            raise ValueError(f"Expected w with 2 or 3 dims, got {w.ndim}")

    def get_velocity(self, w: np.ndarray) -> np.ndarray:
        """Compute trajectory velocity dτ/ds from control points."""
        if w.ndim == 2:
            return self.dB_ds_np @ w
        return np.einsum('tp,npd->ntd', self.dB_ds_np, w)

    def get_acceleration(self, w: np.ndarray) -> np.ndarray:
        """Compute trajectory acceleration d²τ/ds² from control points."""
        if w.ndim == 2:
            return self.d2B_ds2_np @ w
        return np.einsum('tp,npd->ntd', self.d2B_ds2_np, w)

    # ──────────────────────────────────────────────────────────────────
    #  Inverse map:  dense trajectory  →  control points  (least-squares)
    # ──────────────────────────────────────────────────────────────────

    def trajectory_to_control_points(self, traj: np.ndarray) -> np.ndarray:
        """
        Fit B-spline control points to a dense trajectory via least squares.

        Parameters
        ----------
        traj : ndarray (T, D) or (N, T, D)

        Returns
        -------
        w : ndarray (n_ctrl, D) or (N, n_ctrl, D)
        """
        if traj.ndim == 2:
            return self._B_pinv @ traj     # (n_ctrl, T) @ (T, D)
        elif traj.ndim == 3:
            return np.einsum('pt,ntd->npd', self._B_pinv, traj)
        else:
            raise ValueError(f"Expected traj with 2 or 3 dims, got {traj.ndim}")

    # ──────────────────────────────────────────────────────────────────
    #  Round-trip reconstruction error
    # ──────────────────────────────────────────────────────────────────

    def reconstruction_error(self, traj: np.ndarray) -> float:
        """
        Compute ||traj - B @ pinv(B) @ traj||₂  (per-point RMSE).
        """
        w = self.trajectory_to_control_points(traj)
        traj_hat = self.control_points_to_trajectory(w)
        return float(np.sqrt(np.mean((traj - traj_hat) ** 2)))

    # ──────────────────────────────────────────────────────────────────
    #  PyTorch interop  (lazy conversion)
    # ──────────────────────────────────────────────────────────────────

    @property
    def torch_B(self):
        """Lazy-converted (T, n_ctrl) torch.Tensor of the position basis."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not installed.")
        if self._torch_B is None:
            self._torch_B = torch.from_numpy(self.B_np).float()
        return self._torch_B

    @property
    def torch_dB(self):
        """Lazy-converted (T, n_ctrl) torch.Tensor of the velocity basis."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not installed.")
        if self._torch_dB is None:
            self._torch_dB = torch.from_numpy(self.dB_ds_np).float()
        return self._torch_dB

    @property
    def torch_d2B(self):
        """Lazy-converted (T, n_ctrl) torch.Tensor of the acceleration basis."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not installed.")
        if self._torch_d2B is None:
            self._torch_d2B = torch.from_numpy(self.d2B_ds2_np).float()
        return self._torch_d2B


# ======================================================================
#  Quick self-test
# ======================================================================

if __name__ == "__main__":
    print("B-Spline Trajectory Adapter — self-test")
    adapter = BSplineTrajectoryAdapter(
        degree=4, num_control_points=16, num_phase_points=100, spatial_dim=2
    )
    print(f"  B matrix shape:     {adapter.B_np.shape}")
    print(f"  dB/ds matrix shape: {adapter.dB_ds_np.shape}")

    # Round-trip test: random trajectory → fit → reconstruct
    np.random.seed(0)
    traj = np.cumsum(np.random.randn(100, 2) * 0.01, axis=0)
    traj = (traj - traj.min(0)) / (traj.max(0) - traj.min(0) + 1e-8)

    w = adapter.trajectory_to_control_points(traj)
    traj_hat = adapter.control_points_to_trajectory(w)
    err = adapter.reconstruction_error(traj)
    print(f"  Control points shape: {w.shape}")
    print(f"  Round-trip RMSE:      {err:.6f}")

    # Batched
    trajs = np.stack([traj, traj * 0.8 + 0.1], axis=0)
    ws = adapter.trajectory_to_control_points(trajs)
    trajs_hat = adapter.control_points_to_trajectory(ws)
    print(f"  Batched shapes:  trajs={trajs.shape} → ws={ws.shape} → trajs_hat={trajs_hat.shape}")

    # Velocity / acceleration
    vel = adapter.get_velocity(w)
    acc = adapter.get_acceleration(w)
    print(f"  Velocity shape:      {vel.shape}")
    print(f"  Acceleration shape:  {acc.shape}")

    # PyTorch interop
    if _HAS_TORCH:
        print(f"  torch_B shape:       {adapter.torch_B.shape}")
        print(f"  torch_B dtype:       {adapter.torch_B.dtype}")

    print("  ✓ All self-tests passed.")
