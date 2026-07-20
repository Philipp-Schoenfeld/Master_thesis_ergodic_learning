#!/usr/bin/env python3
"""
PF_Ergodic Core Modules
=======================
Reusable building blocks for the Pushforward Ergodic Coverage pipeline.

Pipeline:
  1. Sample latent source distribution (annulus D_delta)
  2. Train OT-CFM pushforward map  G_theta : annulus -> multi-modal GMM
  3. Run ergodic trajectory search in the latent annulus space
       Option A: SVGD kernel repulsion  (fast, from Push_Foreward_Map/test.py)
       Option B: Fourier ergodic cost   (rigorous, gradient-descent on coverage metric)
  4. Push latent ergodic path through G_theta -> ergodic w.r.t. GMM target

Most code is adapted directly from:
  - Push_Foreward_Map/test.py          (OT-CFM map + SVGD latent search)
  - Stein_Flow_matching/flow_matching_2d.py  (Fourier ergodic metric)
  - OT_CFM/ot_cfm_core.py             (RK4 integrator)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

# ============================================================================
# 1.  Source / Target Distributions
# ============================================================================

def sample_annulus(n: int, delta: float = 0.05) -> torch.Tensor:
    """
    Sample uniformly from the annular latent domain  D_delta.
    Exact area-uniform sampling via the CDF inversion:
        r = sqrt(delta^2 + (1 - delta^2) * s),   s ~ U[0,1]
        theta ~ U[0, 2*pi)
    Returns shape (n, 2).
    """
    s = torch.rand(n)
    r = torch.sqrt(delta ** 2 + (1.0 - delta ** 2) * s)
    theta = 2.0 * torch.pi * torch.rand(n)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)


def sample_target_gmm(n: int,
                      modes=None,
                      std: float = 0.15) -> torch.Tensor:
    """
    Sample from the multi-modal Gaussian mixture target distribution.
    Default modes: [-0.5, 0]  and  [+0.5, 0].
    Returns shape (n, 2).
    """
    if modes is None:
        modes = torch.tensor([[-0.5, 0.0], [0.5, 0.0]])
    choices = torch.randint(0, len(modes), (n,))
    noise = torch.randn(n, 2) * std
    return modes[choices] + noise


# ============================================================================
# 2.  Velocity Field MLP  (v_theta : R^3 -> R^2)
# ============================================================================

class VelocityNet(nn.Module):
    """
    Time-conditioned velocity field  v_theta(s, y).
    Input:  s in [0,1]  (flow time, scalar)  + y in R^2  -> dim 3
    Output: velocity in R^2.

    Zero-initialization on the last layer so training starts from an
    identity-like map (zero displacement).
    """

    def __init__(self, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

        # Start with identity-like map
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (B,) or (B,1)  flow time
            y: (B,2)          spatial position
        Returns:
            (B,2) velocity
        """
        if s.dim() == 1:
            s = s.unsqueeze(1)
        return self.net(torch.cat([s, y], dim=1))


# ============================================================================
# 3.  OT-CFM Training  (Hungarian mini-batch coupling)
# ============================================================================

def train_cfm(model: VelocityNet,
              epochs: int = 1500,
              batch_size: int = 512,
              delta: float = 0.05,
              lr: float = 2e-3,
              gmm_modes=None,
              gmm_std: float = 0.15,
              latent_type: str = 'annulus',
              latent_modes=None,
              latent_std: float = 0.3,
              verbose: bool = True) -> list:
    """
    Train the pushforward map G_theta using Optimal-Transport Conditional
    Flow Matching (OT-CFM).

    Loss:  E[ || v_theta(s, y_s) - (x1 - z0) ||^2 ]
    where  y_s = (1-s)*z0 + s*x1  is the straight-line interpolant,
    and (z0, x1) are coupled via Hungarian OT to minimize transport cost.

    Adapted from Push_Foreward_Map/test.py::train_cfm_fast().

    Returns:
        loss_log: list of per-epoch scalar losses
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_log = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- Sample source and target (GMM) ---
        if latent_type == 'annulus':
            z0 = sample_annulus(batch_size, delta)
        elif latent_type == 'gaussian':
            if latent_modes is None:
                latent_modes = torch.tensor([[0.0, 0.0]])
            z0 = sample_target_gmm(batch_size, latent_modes, latent_std)
        else:
            raise ValueError(f"Unknown latent_type: {latent_type}")
            
        x1 = sample_target_gmm(batch_size, gmm_modes, gmm_std)

        # --- Mini-batch Optimal Transport coupling ---
        C = torch.cdist(z0, x1) ** 2           # squared Euclidean cost  (B, B)
        _, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
        x1_coupled = x1[col_ind]               # re-order targets to minimise cost

        # --- Uniform flow time ---
        s = torch.rand(batch_size, 1)

        # --- Straight-line interpolant and target velocity ---
        y_s = (1.0 - s) * z0 + s * x1_coupled
        v_target = x1_coupled - z0             # constant velocity for straight paths

        # --- CFM loss ---
        v_pred = model(s, y_s)
        loss = torch.mean((v_pred - v_target) ** 2)

        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())

        if verbose and epoch % 300 == 0:
            print(f"  [CFM] Epoch {epoch:04d}/{epochs}  loss={loss.item():.4f}")

    model.eval()
    return loss_log


# ============================================================================
# 4.  RK4 ODE Integrator  +  Pushforward
# ============================================================================

def _rk4_step(model: VelocityNet,
              s: torch.Tensor,
              y: torch.Tensor,
              ds: float) -> torch.Tensor:
    """Single RK4 step integrating  dy/ds = v_theta(s, y)."""
    k1 = model(s,        y)
    k2 = model(s + ds / 2, y + k1 * ds / 2)
    k3 = model(s + ds / 2, y + k2 * ds / 2)
    k4 = model(s + ds,   y + k3 * ds)
    return y + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def pushforward(model: VelocityNet,
                z_path: torch.Tensor,
                n_steps: int = 30) -> torch.Tensor:
    """
    Integrate the learned velocity field from s=0 to s=1 along the path.

    Args:
        model:   trained VelocityNet
        z_path:  (N, 2) latent waypoints
        n_steps: RK4 integration steps

    Returns:
        x_path:  (N, 2) pushed-forward waypoints in target space
    """
    model.eval()
    y = z_path.clone()
    ds = 1.0 / n_steps
    for i in range(n_steps):
        s = torch.full((y.size(0), 1), i * ds)
        y = _rk4_step(model, s, y, ds)
    return y


# ============================================================================
# 5a.  Latent Ergodic Search — SVGD Kernel Repulsion
#      Adapted from Push_Foreward_Map/test.py::optimize_latent_sves_path()
# ============================================================================

def svgd_ergodic_search(n_waypoints: int = 150,
                        delta: float = 0.05,
                        n_steps: int = 300,
                        bandwidth: float = 0.2,
                        lr: float = 0.5,
                        momentum: float = 0.8,
                        verbose: bool = True) -> torch.Tensor:
    """
    Spread N waypoints uniformly over the annulus via SVGD kernel repulsion,
    then sequence them into a path with nearest-neighbour TSP.

    This achieves ergodic coverage of the uniform annulus distribution because
    maximising pairwise RBF-kernel distances is equivalent to minimising KL
    divergence to the uniform distribution (in the kernel-density sense).

    Returns:
        z_path:  (N, 2) sequenced latent trajectory
    """
    # Initialise in the annulus
    z = sample_annulus(n_waypoints, delta).clone().requires_grad_(True)
    optimizer = optim.SGD([z], lr=lr, momentum=momentum)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Pairwise differences and squared distances
        diffs   = z.unsqueeze(1) - z.unsqueeze(0)   # (N, N, 2)
        sq_dist = torch.sum(diffs ** 2, dim=-1)      # (N, N)

        # RBF kernel
        K = torch.exp(-sq_dist / bandwidth)          # (N, N)

        # SVGD repulsive update: mean of  -dK/dz  across all neighbours
        repulsion = -(2.0 / bandwidth) * diffs * K.unsqueeze(-1)  # (N, N, 2)
        svgd_update = repulsion.mean(dim=1)                        # (N, 2)

        # Gradient ascent on repulsion (maximise spread)
        loss = torch.sum(-svgd_update.detach() * z)
        loss.backward()
        optimizer.step()

        # Project back onto the annulus
        with torch.no_grad():
            r = torch.norm(z, dim=1)
            r_clamped = torch.clamp(r, min=delta, max=1.0)
            z.data = z.data * (r_clamped / r).unsqueeze(1)

        if verbose and step % 75 == 0:
            print(f"  [SVGD] Step {step:03d}/{n_steps}  "
                  f"max_force={svgd_update.abs().max().item():.4f}")

    # Nearest-neighbour sequencing (greedy TSP)
    z_final = z.detach()
    order = [0]
    unvisited = set(range(1, n_waypoints))
    cur = 0
    while unvisited:
        uv_list = list(unvisited)
        d = torch.norm(z_final[uv_list] - z_final[cur], dim=1)
        nxt = uv_list[torch.argmin(d).item()]
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    if verbose:
        print("  [SVGD] Waypoints sequenced via nearest-neighbour TSP.")

    return z_final[order]   # (N, 2)


# ============================================================================
# 5b.  Latent Ergodic Search — Fourier Ergodic Cost
#      Analogous to Stein_Flow_matching/flow_matching_2d.py ergodic metric
#      but operating purely in the annulus latent space.
# ============================================================================

def _build_fourier_reference(n_grid: int = 300,
                             delta: float = 0.05,
                             K: int = 8) -> tuple:
    """
    Pre-compute the Fourier spectral coefficients phi_k of the uniform
    annulus distribution, for use in the ergodic cost function.

    The annulus is embedded in the square [-1, 1]^2 and evaluated on a
    uniform grid (points outside the annulus get zero weight).

    Returns:
        k_indices:  (K^2, 2) array of wavenumber pairs
        Lambda_k:   (K^2,)   spectral weights  (1 + ||k||^2)^{-3/2}
        phi_k:      (K^2,)   reference Fourier coefficients
        cos_factor: normalisation factor for the cosine basis on [-1,1]^2
    """
    # Cosine basis on [-1, 1]^2:  phi_k(x) = prod_d cos(k_d * pi * (x_d+1)/2)
    # (shifted so the domain spans [0, 2] for cos)
    k_idx = np.array([[k1, k2]
                      for k1 in range(K)
                      for k2 in range(K)], dtype=np.float32)
    Lambda = (1.0 + np.sum(k_idx ** 2, axis=1)) ** (-1.5)

    # Grid over [-1, 1]^2
    xs = np.linspace(-1.0, 1.0, n_grid)
    Xg, Yg = np.meshgrid(xs, xs)
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)   # (n_grid^2, 2)
    r_pts = np.linalg.norm(pts, axis=1)

    # Annulus mask
    mask = (r_pts >= delta) & (r_pts <= 1.0)
    w = mask.astype(np.float32)
    if w.sum() == 0:
        raise RuntimeError("No grid points inside the annulus – check delta.")
    w /= w.sum()    # normalise to a probability measure

    # Cosine basis evaluated at grid points (domain [-1,1] -> argument [0,1])
    # Use the normalised argument: (x+1)/2 maps [-1,1] -> [0,1]
    pts_norm = (pts + 1.0) / 2.0                                  # (n^2, 2)
    args = np.pi * pts_norm[:, None, :] * k_idx[None, :, :]       # (n^2, K^2, 2)
    basis = np.prod(np.cos(args), axis=-1)                         # (n^2, K^2)

    phi_k = np.sum(w[:, None] * basis, axis=0)                    # (K^2,)

    return k_idx, Lambda, phi_k


def fourier_ergodic_search(n_waypoints: int = 150,
                           delta: float = 0.05,
                           n_steps: int = 500,
                           lr: float = 1e-2,
                           K: int = 8,
                           w_smooth: float = 0.5,
                           w_boundary: float = 5.0,
                           verbose: bool = True) -> torch.Tensor:
    """
    Optimise a latent trajectory for ergodic coverage of the uniform
    annulus distribution, using the Fourier spectral ergodic metric.

    Cost:
        E = sum_k Lambda_k * (c_k(z) - phi_k)^2
          + w_smooth  * sum_t ||z_{t+1} - z_t||^2   (path length penalty)
          + w_boundary * barrier keeping z inside annulus

    Waypoints are optimised jointly, then sequenced via nearest-neighbour TSP.

    Returns:
        z_path:  (N, 2) sequenced latent trajectory
    """
    k_idx, Lambda, phi_k = _build_fourier_reference(delta=delta, K=K)
    K_sq = k_idx.shape[0]

    # Convert to torch
    k_t      = torch.tensor(k_idx,  dtype=torch.float32)   # (K^2, 2)
    Lambda_t = torch.tensor(Lambda, dtype=torch.float32)   # (K^2,)
    phi_t    = torch.tensor(phi_k,  dtype=torch.float32)   # (K^2,)

    # Initialise waypoints in the annulus
    z = sample_annulus(n_waypoints, delta).clone().requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()

        # --- Fourier coefficients of the current trajectory ---
        # Map z in [-1,1] -> [0,1] for the cosine basis
        z_norm = (z + 1.0) / 2.0                              # (N, 2)
        args   = torch.pi * z_norm[:, None, :] * k_t[None, :, :]  # (N, K^2, 2)
        basis  = torch.prod(torch.cos(args), dim=-1)           # (N, K^2)
        c_k    = basis.mean(dim=0)                             # (K^2,)

        # --- Ergodic cost ---
        diff    = c_k - phi_t                                  # (K^2,)
        E_ergod = torch.sum(Lambda_t * diff ** 2)

        # --- Smoothness penalty (consecutive waypoint distance) ---
        E_smooth = w_smooth * torch.sum((z[1:] - z[:-1]) ** 2)

        # --- Soft annulus boundary: log-barrier ---
        r = torch.norm(z, dim=1)                               # (N,)
        # Keep r in [delta, 1.0]
        inner_viol = torch.clamp(delta - r, min=0.0)          # push out of hole
        outer_viol = torch.clamp(r - 1.0,  min=0.0)          # push inside disk
        E_boundary = w_boundary * (torch.sum(inner_viol ** 2) +
                                   torch.sum(outer_viol ** 2))

        loss = E_ergod + E_smooth + E_boundary
        loss.backward()
        optimizer.step()

        # Hard project onto annulus after each step
        with torch.no_grad():
            r = torch.norm(z, dim=1)
            r_clamped = torch.clamp(r, min=delta + 1e-4, max=1.0 - 1e-4)
            z.data = z.data * (r_clamped / r).unsqueeze(1)

        if verbose and step % 100 == 0:
            print(f"  [Fourier] Step {step:04d}/{n_steps}  "
                  f"E_ergod={E_ergod.item():.4f}  "
                  f"E_smooth={E_smooth.item():.4f}  "
                  f"E_bound={E_boundary.item():.4f}")

    # Nearest-neighbour sequencing
    z_final = z.detach()
    order = [0]
    unvisited = set(range(1, n_waypoints))
    cur = 0
    while unvisited:
        uv_list = list(unvisited)
        d = torch.norm(z_final[uv_list] - z_final[cur], dim=1)
        nxt = uv_list[torch.argmin(d).item()]
        order.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    if verbose:
        print("  [Fourier] Waypoints sequenced via nearest-neighbour TSP.")

    return z_final[order]   # (N, 2)


# ============================================================================
# 6.  Ergodic Error Metric  (Fourier spectral metric in target space)
# ============================================================================

def compute_ergodic_error(traj: np.ndarray,
                          target_samples: np.ndarray,
                          K: int = 10,
                          domain: tuple = (-1.0, 1.0)) -> float:
    """
    Compute the Fourier spectral ergodic error between a trajectory and
    a target distribution.

    E_erg = sum_k Lambda_k * (c_k(traj) - phi_k(target))^2

    Both trajectory and target samples are expected in the domain range.
    The cosine basis is normalised to [0,1] internally.

    Args:
        traj:           (T, 2)  trajectory waypoints
        target_samples: (M, 2)  samples from the target distribution
        K:              number of Fourier modes per dimension
        domain:         (lo, hi) spatial domain bounds (same for both axes)

    Returns:
        scalar ergodic error
    """
    lo, hi = domain
    k_idx = np.array([[k1, k2]
                      for k1 in range(K)
                      for k2 in range(K)], dtype=np.float32)
    Lambda = (1.0 + np.sum(k_idx ** 2, axis=1)) ** (-1.5)

    def _basis(pts):
        pts_n = (pts - lo) / (hi - lo)               # -> [0, 1]
        args = np.pi * pts_n[:, None, :] * k_idx[None, :, :]
        return np.prod(np.cos(args), axis=-1)         # (N, K^2)

    c_k   = _basis(traj).mean(axis=0)
    phi_k = _basis(target_samples).mean(axis=0)
    diff  = c_k - phi_k
    return float(np.sum(Lambda * diff ** 2))
