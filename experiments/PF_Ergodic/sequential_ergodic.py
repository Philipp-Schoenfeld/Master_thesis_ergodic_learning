#!/usr/bin/env python3
"""
Sector-Aligned Ergodic Coverage via Pushforward Maps
====================================================

For each target scenario (1–5 non-overlapping Gaussians):
  1. Train a global OT-CFM pushforward map  G_θ : annulus → multi-modal GMM
  2. Discover angular sector partitions in the latent annulus
  3. Generate a sector-piecewise star trajectory (one sector per mode)
  4. Push through G_θ → warm-start trajectory with minimal cross-mode jumps
  5. Refine with SVGD ergodic search (Fourier spectral cost)

Visualization:
  • Row per scenario, 3 panels:
    Panel 1: target GMM scatter (purple)
    Panel 2: sector-coloured latent paths in annulus + boundary lines
    Panel 3: global trajectory over faint target (orange + red)
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
CFM_EPOCHS   = 3000
CFM_BATCH    = 1024
CFM_LR       = 2e-3
CFM_HIDDEN   = 256
CFM_N_LAYERS = 4
DELTA        = 0.05

# Star-shaped latent trajectory
N_SPOKES       = 30       # radial spokes per sector
PTS_PER_SPOKE  = 5        # points per half-spoke (out or in)
# → total waypoints per sector = N_SPOKES * 2 * PTS_PER_SPOKE = 300

# Sector discovery
SECTOR_PROBE_N = 1000     # probe points for sector discovery

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
# 1a.  Latent Sector Discovery
# ============================================================================

@torch.no_grad()
def discover_sectors(model: 'VelocityNet',
                    modes: torch.Tensor,
                    delta: float = 0.05,
                    n_probe: int = 1000,
                    verbose: bool = True):
    """
    Discover the angular partition of the latent annulus induced by the
    trained pushforward map.  Each target mode occupies a contiguous
    "pie-slice" sector.

    Returns:
        sector_bounds: list of (theta_lo, theta_hi) in [0, 2π) for each mode,
                       ordered by ascending theta_lo.
        sector_order:  list of mode indices in the order they appear angularly.
    """
    from pf_ergodic_core import sample_annulus, pushforward

    # --- 1. Sample probe points uniformly on the annulus ---
    z_probe = sample_annulus(n_probe, delta)
    x_probe = pushforward(model, z_probe, n_steps=RK4_STEPS)

    # --- 2. Assign each probe to its nearest target mode ---
    # (n_probe, n_modes)
    dists = torch.cdist(x_probe, modes)           # (N, M)
    assignments = torch.argmin(dists, dim=1).numpy()  # (N,)

    # --- 3. Compute angles of latent probes ---
    z_np = z_probe.numpy()
    thetas = np.arctan2(z_np[:, 1], z_np[:, 0])   # in [-π, π]
    thetas = thetas % (2.0 * np.pi)                # shift to [0, 2π)

    # --- 4. For each mode, find the angular extent ---
    n_modes = len(modes)
    mode_median_angles = []
    for m in range(n_modes):
        mask = assignments == m
        if mask.sum() == 0:
            mode_median_angles.append(0.0)
            continue
        # Circular median: average unit vectors then take atan2
        angles_m = thetas[mask]
        cx = np.cos(angles_m).mean()
        cy = np.sin(angles_m).mean()
        mode_median_angles.append(np.arctan2(cy, cx) % (2.0 * np.pi))

    # --- 5. Sort modes by their median angle ---
    sector_order = list(np.argsort(mode_median_angles))
    sorted_medians = [mode_median_angles[i] for i in sector_order]

    # --- 6. Compute sector boundaries as midpoints between neighbours ---
    sector_bounds = []
    for i in range(n_modes):
        prev_med = sorted_medians[(i - 1) % n_modes]
        curr_med = sorted_medians[i]
        next_med = sorted_medians[(i + 1) % n_modes]

        # Angular midpoint (handle wrapping)
        def _ang_mid(a, b):
            diff = (b - a) % (2.0 * np.pi)
            return (a + diff / 2.0) % (2.0 * np.pi)

        lo = _ang_mid(prev_med, curr_med)
        hi = _ang_mid(curr_med, next_med)
        sector_bounds.append((lo, hi))

    if verbose:
        for i, m_idx in enumerate(sector_order):
            lo, hi = sector_bounds[i]
            print(f"    Sector {i}: mode {m_idx}  "
                  f"θ ∈ [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°)  "
                  f"(median {np.degrees(sorted_medians[i]):.1f}°)")

    return sector_bounds, sector_order


# ============================================================================
# 1b.  Sector-piecewise star trajectory on the annulus
# ============================================================================

def generate_sector_star_trajectory(sector_bounds: list,
                                    sector_order: list,
                                    n_spokes: int = 30,
                                    pts_per_spoke: int = 5,
                                    delta: float = 0.05,
                                    n_transition: int = 10) -> torch.Tensor:
    """
    Generate a star trajectory that explores one angular sector at a time,
    with smooth transition arcs between consecutive sectors.

    Args:
        sector_bounds: list of (theta_lo, theta_hi) per sector (sorted).
        sector_order:  mode indices in angular order (for labelling only).
        n_spokes:      radial spokes *per sector*.
        pts_per_spoke: points per half-spoke.
        delta:         inner radius.
        n_transition:  points in the arc linking successive sectors.

    Returns:
        z_traj: (N_total, 2) latent trajectory tensor.
    """
    all_segments = []
    n_sectors = len(sector_bounds)

    for sec_i in range(n_sectors):
        theta_lo, theta_hi = sector_bounds[sec_i]

        # Handle wrap-around (lo > hi means the sector crosses 0/2π)
        arc = (theta_hi - theta_lo) % (2.0 * np.pi)
        if arc < 1e-6:
            arc = 2.0 * np.pi  # full circle fallback for 1-mode

        # --- Dense star pattern inside this sector ---
        for k in range(n_spokes):
            frac = (k + 0.5) / n_spokes       # avoid exact boundary
            theta = (theta_lo + arc * frac) % (2.0 * np.pi)

            # Outward leg: δ → 1.0
            s_vals = np.linspace(0, 1, pts_per_spoke)
            r_out = np.sqrt(delta ** 2 + (1.0 - delta ** 2) * s_vals)
            x_out = r_out * np.cos(theta)
            y_out = r_out * np.sin(theta)
            leg_out = np.stack([x_out, y_out], axis=1)

            # Return leg: 1.0 → δ
            leg_in = leg_out[::-1].copy()

            all_segments.append(leg_out)
            all_segments.append(leg_in)

        # --- Smooth transition arc to the next sector ---
        if n_sectors > 1:
            next_i = (sec_i + 1) % n_sectors
            next_lo, next_hi = sector_bounds[next_i]
            next_arc = (next_hi - next_lo) % (2.0 * np.pi)
            if next_arc < 1e-6:
                next_arc = 2.0 * np.pi

            # End angle of current sector (last spoke tip at inner radius)
            end_theta = (theta_lo + arc * (n_spokes - 0.5) / n_spokes) % (2.0 * np.pi)
            # Start angle of next sector (first spoke)
            start_theta = (next_lo + next_arc * 0.5 / n_spokes) % (2.0 * np.pi)

            # Interpolate along the inner radius (δ) as a smooth arc
            ang_diff = (start_theta - end_theta) % (2.0 * np.pi)
            # Take the shorter arc direction
            if ang_diff > np.pi:
                ang_diff -= 2.0 * np.pi
            t_vals = np.linspace(0, 1, n_transition + 2)[1:-1]
            r_trans = delta
            trans_angles = end_theta + ang_diff * t_vals
            x_trans = r_trans * np.cos(trans_angles)
            y_trans = r_trans * np.sin(trans_angles)
            transition = np.stack([x_trans, y_trans], axis=1)
            all_segments.append(transition)

    z_traj = np.concatenate(all_segments, axis=0)
    return torch.tensor(z_traj, dtype=torch.float32)


def generate_star_trajectory(n_spokes: int = 30,
                             pts_per_spoke: int = 5,
                             delta: float = 0.05) -> torch.Tensor:
    """
    Fallback: full-circle star pattern (used for single-mode scenarios
    where sector discovery is unnecessary).
    """
    segments = []
    for k in range(n_spokes):
        theta = 2.0 * np.pi * k / n_spokes
        s_vals = np.linspace(0, 1, pts_per_spoke)
        r_out = np.sqrt(delta ** 2 + (1.0 - delta ** 2) * s_vals)
        x_out = r_out * np.cos(theta)
        y_out = r_out * np.sin(theta)
        leg_out = np.stack([x_out, y_out], axis=1)
        leg_in = leg_out[::-1].copy()
        segments.append(leg_out)
        segments.append(leg_in)
    z_star = np.concatenate(segments, axis=0)
    return torch.tensor(z_star, dtype=torch.float32)


# ============================================================================
# 2.  Fourier ergodic cost for SVGD refinement  (single Gaussian target)
# ============================================================================

def _build_gmm_phi(modes: np.ndarray,
                   std: float,
                   K: int = 8,
                   n_grid: int = 200,
                   domain: tuple = (-1.2, 1.2)):
    """
    Pre-compute Fourier reference coefficients φ_k for a GMM.
    """
    lo, hi = domain
    k_idx = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)],
                     dtype=np.float32)
    Lambda = (1.0 + np.sum(k_idx ** 2, axis=1)) ** (-1.5)

    # Evaluate Gaussian density on a grid
    xs = np.linspace(lo, hi, n_grid)
    Xg, Yg = np.meshgrid(xs, xs)
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1)

    w = np.zeros(pts.shape[0])
    for mode in modes:
        dx = pts[:, 0] - mode[0]
        dy = pts[:, 1] - mode[1]
        w += np.exp(-(dx ** 2 + dy ** 2) / (2.0 * std ** 2))
    w /= w.sum() + 1e-12

    # Cosine basis
    pts_n = (pts - lo) / (hi - lo)
    args = np.pi * pts_n[:, None, :] * k_idx[None, :, :]
    basis = np.prod(np.cos(args), axis=-1)
    phi_k = np.sum(w[:, None] * basis, axis=0)

    return k_idx, Lambda, phi_k


def svgd_ergodic_refine(traj_init: np.ndarray,
                        modes: np.ndarray,
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
    Refine a trajectory for ergodic coverage of a multi-modal GMM
    using Fourier spectral cost + smoothness + boundary, optimised with Adam.

    Adapted from SE3_SVGD/tsvec_2d.py energy function.

    Args:
        traj_init: (T, 2) initial trajectory in target space
        modes:     (M, 2) Gaussian centres
        std:       scalar Gaussian std
    Returns:
        traj_refined: (T, 2) refined trajectory
    """
    lo, hi = domain
    k_idx, Lambda, phi_k_np = _build_gmm_phi(modes, std, K,
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
# 3.  Global pipeline: train → sector discovery → star → push → refine
# ============================================================================

def process_global_scenario(modes: torch.Tensor,
                            std: float,
                            n_modes: int,
                            verbose: bool = True):
    """
    Full global pipeline for a multi-modal GMM with sector-aligned latent
    trajectory.

    Returns:
        (z_star, x_push, x_refined, sector_bounds, sector_order)
        z_star / x_push / x_refined are numpy arrays (T, 2).
        sector_bounds / sector_order describe the discovered partitions.
    """
    modes_np = modes.numpy()

    # ---- 1. Train OT-CFM for the entire GMM target ----
    if verbose:
        print(f"\n  [Global] Training OT-CFM for {n_modes} modes "
              f"(σ={std:.2f}) …")
    model = VelocityNet(hidden_dim=CFM_HIDDEN, n_layers=CFM_N_LAYERS)
    loss_log = train_cfm(
        model,
        epochs=CFM_EPOCHS,
        batch_size=CFM_BATCH,
        delta=DELTA,
        lr=CFM_LR,
        gmm_modes=modes,
        gmm_std=std,
        verbose=verbose,
    )
    if verbose:
        print(f"  [Global] Final CFM loss: {loss_log[-1]:.5f}")

    # ---- 2. Discover angular sector partitions ----
    if n_modes > 1:
        if verbose:
            print(f"  [Global] Discovering latent sectors …")
        sector_bounds, sector_order = discover_sectors(
            model, modes, delta=DELTA, n_probe=SECTOR_PROBE_N,
            verbose=verbose)
        z_star = generate_sector_star_trajectory(
            sector_bounds, sector_order,
            n_spokes=N_SPOKES, pts_per_spoke=PTS_PER_SPOKE,
            delta=DELTA, n_transition=TRANSITION_PTS)
    else:
        # Single mode: full-circle star, no sectors needed
        sector_bounds = [(0.0, 2.0 * np.pi)]
        sector_order = [0]
        z_star = generate_star_trajectory(N_SPOKES, PTS_PER_SPOKE, DELTA)

    # ---- 3. Pushforward ----
    x_push = pushforward(model, z_star, n_steps=RK4_STEPS)
    x_push_np = x_push.numpy()

    # ---- 4. SVGD ergodic refinement ----
    if verbose:
        print(f"  [Global] Refining with SVGD ergodic search …")
    x_refined = svgd_ergodic_refine(
        x_push_np, modes_np, std,
        n_steps=REFINE_STEPS,
        lr=REFINE_LR,
        K=REFINE_K,
        w_ergodic=W_ERGODIC_REF,
        w_smooth=W_SMOOTH_REF,
        w_boundary=W_BOUNDARY_REF,
        verbose=verbose,
    )

    return z_star.numpy(), x_push_np, x_refined, sector_bounds, sector_order

# ============================================================================
# 4.  Visualisation  (Push_Foreward_Map/test.py style)
# ============================================================================

def _panel_target(ax, target_pts, title):
    ax.scatter(target_pts[:, 0], target_pts[:, 1],
               s=2, alpha=0.5, c='purple')
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)


def _panel_latent(ax, z_star, sector_bounds, sector_order, title):
    """Plot sector-coloured star trajectory in the annulus."""
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sector_bounds), 1)))
    ax.plot(z_star[:, 0], z_star[:, 1], '-', color='blue',
            linewidth=0.5, alpha=0.3)
    ax.scatter(z_star[:, 0], z_star[:, 1], s=3, color='blue', zorder=5)

    # Draw sector boundary lines
    for i, (lo, _hi) in enumerate(sector_bounds):
        bx = [0, 1.15 * np.cos(lo)]
        by = [0, 1.15 * np.sin(lo)]
        ax.plot(bx, by, '--', color=colors[i], linewidth=1.2, alpha=0.8)
        # Label
        lbl_r = 1.08
        mid = lo + 0.15
        ax.text(lbl_r * np.cos(mid), lbl_r * np.sin(mid),
                f"M{sector_order[i]}", fontsize=7, color=colors[i],
                ha='center', va='center', fontweight='bold')

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
# 5.  Main
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

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        z_star, x_push, x_refined, sec_bounds, sec_order = \
            process_global_scenario(modes, std, n_modes, verbose=True)

        # ---- Ergodic error (full trajectory vs full GMM) ----
        target_vis = sample_target_gmm(N_VIS, modes, std).numpy()

        err_push = compute_ergodic_error(x_push, target_vis,
                                         K=10, domain=(-1.2, 1.2))
        err_refined = compute_ergodic_error(x_refined, target_vis,
                                            K=10, domain=(-1.2, 1.2))

        print(f"\n  Ergodic error (pushforward only):  {err_push:.5f}")
        print(f"  Ergodic error (after refinement):  {err_refined:.5f}")

        # ---- Plot ----
        ax0, ax1, ax2 = axes[sc_idx, 0], axes[sc_idx, 1], axes[sc_idx, 2]

        _panel_target(ax0, target_vis,
                      f"{sc_name}\nTarget Density (GMM)")
        _panel_latent(ax1, z_star, sec_bounds, sec_order,
                      f"Sector-aligned latent ({n_modes} sector(s))")
        _panel_sequential(ax2, x_refined, target_vis,
                          f"Global trajectory", err_refined)

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