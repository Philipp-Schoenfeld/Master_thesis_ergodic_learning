#!/usr/bin/env python3
"""
Sequential Ergodic Coverage via Pushforward Maps
=================================================

For each target scenario (1–5 non-overlapping Gaussians):
  1. Train an OT-CFM pushforward map  G_θ : annulus → single Gaussian  (per mode)
  2. Generate a star-shaped latent trajectory on the annulus
  3. Push through G_θ → warm-start trajectory already ≈ ergodic on that mode
  4. Refine with SVGD ergodic search (Fourier spectral cost, SE3_SVGD style)
  5. Concatenate per-mode sub-trajectories into a full sequential path

Visualization mirrors Push_Foreward_Map/test.py:
  • Row per scenario, 3 panels:
    Panel 1: target GMM scatter (purple)
    Panel 2: star latent paths in annulus (blue + boundary circles)
    Panel 3: sequential trajectory over faint target (orange + red)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

from pf_ergodic_core import (
    VelocityNet,
    sample_annulus,
    sample_target_gmm,
    train_cfm,
    pushforward,
    compute_ergodic_error,
)

# ============================================================================
# Target Scenarios  (1–5 non-overlapping Gaussians)
# ============================================================================

SCENARIOS = [
    {
        "name": "1 Gaussian",
        "modes": torch.tensor([[0.0, 0.0]]),
        "std": 0.18,
    },
    {
        "name": "2 Gaussians – Horizontal",
        "modes": torch.tensor([[-0.5, 0.0], [0.5, 0.0]]),
        "std": 0.15,
    },
    {
        "name": "3 Gaussians – Triangle",
        "modes": torch.tensor([[-0.6, -0.35], [0.6, -0.35], [0.0, 0.55]]),
        "std": 0.13,
    },
    {
        "name": "4 Gaussians – Ring",
        "modes": torch.tensor([[-0.55, 0.0], [0.55, 0.0],
                                [0.0, -0.55], [0.0, 0.55]]),
        "std": 0.12,
    },
    {
        "name": "5 Gaussians – Pentagon",
        "modes": torch.tensor([
            [-0.6, -0.4], [0.6, -0.4],
            [-0.3,  0.5], [0.3,  0.5],
            [0.0,   0.0],
        ]),
        "std": 0.10,
    },
]

# ============================================================================
# Hyper-parameters
# ============================================================================

# OT-CFM training
CFM_EPOCHS   = 1500
CFM_BATCH    = 512
CFM_LR       = 2e-3
CFM_HIDDEN   = 128
CFM_N_LAYERS = 3
DELTA        = 0.05

# Star-shaped latent trajectory
N_SPOKES       = 30       # radial spokes per mode
PTS_PER_SPOKE  = 5        # points per half-spoke (out or in)
# → total waypoints per mode = N_SPOKES * 2 * PTS_PER_SPOKE = 300

# SVGD ergodic refinement
REFINE_STEPS  = 200
REFINE_LR     = 2e-3
REFINE_K      = 8          # Fourier modes per dim
W_ERGODIC_REF = 600.0
W_SMOOTH_REF  = 15.0
W_BOUNDARY_REF = 30.0

# Pushforward ODE
RK4_STEPS = 30

# Transition between modes
TRANSITION_PTS = 10

# Visualisation
N_VIS = 2000
SEED  = 42
XLIM = YLIM = (-1.2, 1.2)


# ============================================================================
# 1.  Star-shaped latent trajectory on the annulus
# ============================================================================

def generate_star_trajectory(n_spokes: int = 30,
                             pts_per_spoke: int = 5,
                             delta: float = 0.05) -> torch.Tensor:
    """
    Generate a star pattern on the annulus: radial back-and-forth spokes.

    Each spoke sweeps from inner radius δ to outer radius 1.0 at a
    uniformly spaced angle, then returns.  This produces an ergodic
    (area-filling) trajectory on the annulus.

    Returns:
        z_star: (N_total, 2) latent trajectory
    """
    segments = []
    for k in range(n_spokes):
        theta = 2.0 * np.pi * k / n_spokes

        # Outward leg: δ → 1.0
        s_vals = np.linspace(0, 1, pts_per_spoke)
        r_out = np.sqrt(delta ** 2 + (1.0 - delta ** 2) * s_vals)
        x_out = r_out * np.cos(theta)
        y_out = r_out * np.sin(theta)
        leg_out = np.stack([x_out, y_out], axis=1)

        # Return leg: 1.0 → δ
        leg_in = leg_out[::-1].copy()

        segments.append(leg_out)
        segments.append(leg_in)

    z_star = np.concatenate(segments, axis=0)
    return torch.tensor(z_star, dtype=torch.float32)


# ============================================================================
# 2.  Fourier ergodic cost for SVGD refinement  (single Gaussian target)
# ============================================================================

def _build_single_gaussian_phi(mode: np.ndarray,
                                std: float,
                                K: int = 8,
                                n_grid: int = 200,
                                domain: tuple = (-1.2, 1.2)):
    """
    Pre-compute Fourier reference coefficients φ_k for a single Gaussian.
    """
    lo, hi = domain
    k_idx = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)],
                     dtype=np.float32)
    Lambda = (1.0 + np.sum(k_idx ** 2, axis=1)) ** (-1.5)

    # Evaluate Gaussian density on a grid
    xs = np.linspace(lo, hi, n_grid)
    Xg, Yg = np.meshgrid(xs, xs)
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)

    dx = pts[:, 0] - mode[0]
    dy = pts[:, 1] - mode[1]
    w = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * std ** 2))
    w /= w.sum() + 1e-12

    # Cosine basis
    pts_n = (pts - lo) / (hi - lo)
    args = np.pi * pts_n[:, None, :] * k_idx[None, :, :]
    basis = np.prod(np.cos(args), axis=-1)
    phi_k = np.sum(w[:, None] * basis, axis=0)

    return k_idx, Lambda, phi_k


def svgd_ergodic_refine(traj_init: np.ndarray,
                        mode: np.ndarray,
                        std: float,
                        n_steps: int = 200,
                        lr: float = 2e-3,
                        K: int = 8,
                        w_ergodic: float = 600.0,
                        w_smooth: float = 15.0,
                        w_boundary: float = 30.0,
                        domain: tuple = (-1.2, 1.2),
                        verbose: bool = True) -> np.ndarray:
    """
    Refine a trajectory for ergodic coverage of a single Gaussian mode
    using Fourier spectral cost + smoothness + boundary, optimised with Adam.

    Adapted from SE3_SVGD/tsvec_2d.py energy function.

    Args:
        traj_init: (T, 2) initial trajectory in target space
        mode:      (2,)   Gaussian centre
        std:       scalar Gaussian std
    Returns:
        traj_refined: (T, 2) refined trajectory
    """
    lo, hi = domain
    k_idx, Lambda, phi_k_np = _build_single_gaussian_phi(mode, std, K,
                                                          domain=domain)
    k_t = torch.tensor(k_idx, dtype=torch.float32)
    Lambda_t = torch.tensor(Lambda, dtype=torch.float32)
    phi_t = torch.tensor(phi_k_np, dtype=torch.float32)

    z = torch.tensor(traj_init, dtype=torch.float32).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    T = z.shape[0]

    for step in range(n_steps):
        optimizer.zero_grad()

        # --- Fourier coefficients of current trajectory ---
        z_norm = (z - lo) / (hi - lo)                             # → [0, 1]
        args = torch.pi * z_norm[:, None, :] * k_t[None, :, :]   # (T, K², 2)
        basis = torch.prod(torch.cos(args), dim=-1)               # (T, K²)
        c_k = basis.mean(dim=0)                                   # (K²,)

        # Ergodic cost
        diff = c_k - phi_t
        E_ergod = w_ergodic * 0.5 * torch.sum(Lambda_t * diff ** 2)

        # Smoothness (acceleration penalty)
        accel = z[2:] - 2.0 * z[1:-1] + z[:-2]
        E_smooth = w_smooth * torch.sum(accel ** 2)

        # Boundary penalty  (keep inside domain)
        margin = 0.03
        lo_viol = torch.clamp(lo + margin - z, min=0.0)
        hi_viol = torch.clamp(z - (hi - margin), min=0.0)
        E_boundary = w_boundary * 0.5 * (torch.sum(lo_viol ** 2) +
                                          torch.sum(hi_viol ** 2))

        loss = E_ergod + E_smooth + E_boundary
        loss.backward()
        optimizer.step()

        if verbose and step % 50 == 0:
            print(f"    [Refine] Step {step:03d}/{n_steps}  "
                  f"E_erg={E_ergod.item():.4f}  "
                  f"E_sm={E_smooth.item():.4f}  "
                  f"E_bd={E_boundary.item():.4f}")

    return z.detach().numpy()


# ============================================================================
# 3.  Per-mode pipeline: train → star → push → refine
# ============================================================================

def process_single_mode(mode: torch.Tensor,
                        std: float,
                        mode_idx: int,
                        n_modes: int,
                        verbose: bool = True):
    """
    Full pipeline for one Gaussian mode.
    Returns: (z_star, x_push, x_refined) — all numpy arrays (T, 2).
    """
    mode_np = mode.numpy()

    # ---- 1. Train OT-CFM for this single mode ----
    if verbose:
        print(f"\n  [Mode {mode_idx+1}/{n_modes}] Training OT-CFM  "
              f"(centre={mode_np.tolist()}, σ={std:.2f}) …")
    model = VelocityNet(hidden_dim=CFM_HIDDEN, n_layers=CFM_N_LAYERS)
    loss_log = train_cfm(
        model,
        epochs=CFM_EPOCHS,
        batch_size=CFM_BATCH,
        delta=DELTA,
        lr=CFM_LR,
        gmm_modes=mode.unsqueeze(0),   # single mode  (1, 2)
        gmm_std=std,
        verbose=verbose,
    )
    if verbose:
        print(f"  [Mode {mode_idx+1}] Final CFM loss: {loss_log[-1]:.5f}")

    # ---- 2. Star-shaped latent trajectory ----
    z_star = generate_star_trajectory(N_SPOKES, PTS_PER_SPOKE, DELTA)

    # ---- 3. Pushforward ----
    x_push = pushforward(model, z_star, n_steps=RK4_STEPS)
    x_push_np = x_push.numpy()

    # ---- 4. SVGD ergodic refinement ----
    if verbose:
        print(f"  [Mode {mode_idx+1}] Refining with SVGD ergodic search …")
    x_refined = svgd_ergodic_refine(
        x_push_np, mode_np, std,
        n_steps=REFINE_STEPS,
        lr=REFINE_LR,
        K=REFINE_K,
        w_ergodic=W_ERGODIC_REF,
        w_smooth=W_SMOOTH_REF,
        w_boundary=W_BOUNDARY_REF,
        verbose=verbose,
    )

    return z_star.numpy(), x_push_np, x_refined


# ============================================================================
# 4.  Sequential concatenation with linear transitions
# ============================================================================

def concatenate_trajectories(segments: list[np.ndarray],
                             n_transition: int = 10) -> np.ndarray:
    """
    Concatenate per-mode trajectories with short linear transitions.
    """
    if len(segments) == 1:
        return segments[0]

    parts = [segments[0]]
    for i in range(1, len(segments)):
        # Linear interpolation from end of previous to start of next
        start = segments[i - 1][-1]
        end   = segments[i][0]
        t = np.linspace(0, 1, n_transition + 2)[1:-1]  # exclude endpoints
        transition = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]
        parts.append(transition)
        parts.append(segments[i])

    return np.concatenate(parts, axis=0)


# ============================================================================
# 5.  Visualisation  (Push_Foreward_Map/test.py style)
# ============================================================================

def _panel_target(ax, target_pts, title):
    ax.scatter(target_pts[:, 0], target_pts[:, 1],
               s=2, alpha=0.5, c='purple')
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)


def _panel_latent(ax, z_segments, title):
    """Plot all per-mode star trajectories in the annulus."""
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(z_segments), 1)))
    for i, z_np in enumerate(z_segments):
        c = colors[i % len(colors)]
        ax.plot(z_np[:, 0], z_np[:, 1], '-', color=c,
                linewidth=0.6, alpha=0.5)
        ax.scatter(z_np[:, 0], z_np[:, 1], s=4, color=c, zorder=5)

    inner = mpatches.Circle((0, 0), DELTA, color='black',
                            fill=False, linewidth=1)
    outer = mpatches.Circle((0, 0), 1.0, color='black',
                            fill=False, linewidth=1, linestyle='--')
    ax.add_patch(inner)
    ax.add_patch(outer)

    ax.set_title(title, fontsize=10)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)


def _panel_sequential(ax, full_traj, target_pts, title, erg_err):
    ax.scatter(target_pts[:, 0], target_pts[:, 1],
               s=2, alpha=0.1, c='purple')
    ax.plot(full_traj[:, 0], full_traj[:, 1], 'orange', linewidth=1.2)
    ax.scatter(full_traj[:, 0], full_traj[:, 1], s=6, c='red', zorder=5)

    ax.set_title(f"{title}\n(erg. err = {erg_err:.4f})", fontsize=10)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)


# ============================================================================
# 6.  Main
# ============================================================================

def main(save_dir: str = _HERE):
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n_scenarios = len(SCENARIOS)
    fig, axes = plt.subplots(n_scenarios, 3,
                              figsize=(15, 5 * n_scenarios),
                              facecolor='white')
    if n_scenarios == 1:
        axes = axes[np.newaxis, :]

    for sc_idx, scenario in enumerate(SCENARIOS):
        sc_name = scenario["name"]
        modes   = scenario["modes"]
        std     = scenario["std"]
        n_modes = len(modes)

        print("\n" + "=" * 62)
        print(f"Scenario {sc_idx+1}/{n_scenarios}: {sc_name}  "
              f"({n_modes} mode(s), σ={std})")
        print("=" * 62)

        z_segments = []       # latent star per mode
        x_push_segments = []  # pushforward per mode (before refine)
        x_refined_segments = []  # refined per mode

        for m_idx in range(n_modes):
            torch.manual_seed(SEED + m_idx)
            np.random.seed(SEED + m_idx)

            z_star, x_push, x_refined = process_single_mode(
                modes[m_idx], std, m_idx, n_modes, verbose=True)

            z_segments.append(z_star)
            x_push_segments.append(x_push)
            x_refined_segments.append(x_refined)

        # ---- Concatenate sequential trajectory ----
        full_traj_push = concatenate_trajectories(x_push_segments,
                                                   TRANSITION_PTS)
        full_traj_refined = concatenate_trajectories(x_refined_segments,
                                                      TRANSITION_PTS)

        # ---- Ergodic error (full trajectory vs full GMM) ----
        target_vis = sample_target_gmm(N_VIS, modes, std).numpy()

        err_push = compute_ergodic_error(full_traj_push, target_vis,
                                         K=10, domain=(-1.2, 1.2))
        err_refined = compute_ergodic_error(full_traj_refined, target_vis,
                                            K=10, domain=(-1.2, 1.2))

        print(f"\n  Ergodic error (pushforward only):  {err_push:.5f}")
        print(f"  Ergodic error (after refinement):  {err_refined:.5f}")

        # ---- Plot ----
        ax0, ax1, ax2 = axes[sc_idx, 0], axes[sc_idx, 1], axes[sc_idx, 2]

        _panel_target(ax0, target_vis,
                      f"{sc_name}\nTarget Density (GMM)")
        _panel_latent(ax1, z_segments,
                      f"Star latent paths ({n_modes} mode(s))")
        _panel_sequential(ax2, full_traj_refined, target_vis,
                          f"Sequential trajectory", err_refined)

        ax0.set_ylabel(f"Scenario {sc_idx+1}", fontsize=10, labelpad=8)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "sequential_ergodic_results.png")
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll scenarios done. Figure saved → {out_path}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential Ergodic Coverage via Pushforward Maps"
    )
    parser.add_argument("--epochs",    type=int, default=CFM_EPOCHS)
    parser.add_argument("--spokes",    type=int, default=N_SPOKES)
    parser.add_argument("--refine-steps", type=int, default=REFINE_STEPS)
    parser.add_argument("--out-dir",   type=str, default=_HERE)
    args = parser.parse_args()

    CFM_EPOCHS   = args.epochs
    N_SPOKES     = args.spokes
    REFINE_STEPS = args.refine_steps

    main(save_dir=args.out_dir)