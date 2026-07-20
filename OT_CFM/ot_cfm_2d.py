#!/usr/bin/env python3
"""
2D OT-CFM Ergodic Coverage Testbench — Letter "N" target

Implements "Ergodic Trajectory Design by Learned Pushforward Maps:
Provable Coverage via Conditional Flow Matching" on a 2D domain.

Two-stage pipeline:
  1. Deterministic latent ergodic trajectory on annular domain
  2. OT-CFM network learns pushforward map to match target density

Uses the same five initialization strategies and comparison-grid
visualization as the other testbenches.
"""

import time
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

sys.path.append("/home/philipp/Documents/Uni/Master_thesis")
from init_strategies import get_initialization
from OT_CFM.ot_cfm_core import (
    LatentTrajectory, VelocityFieldMLP,
    sinkhorn_coupling, rk4_integrate,
    cfm_loss, nfz_penalty, acceleration_penalty,
    zeng_power_penalty,
)

np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# 1. Target Distribution: Letter "N" (identical to other methods)
# ============================================================================

N_SEGMENTS = [
    ([0.25, 0.15], [0.25, 0.85]),
    ([0.25, 0.85], [0.75, 0.15]),
    ([0.75, 0.15], [0.75, 0.85]),
]
STROKE_WIDTH = 0.045


def _dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (len_sq + 1e-12), 0, 1)
    return np.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)


def target_distribution(x, y):
    d_min = np.full_like(x, 1e10)
    for (ax, ay), (bx, by) in N_SEGMENTS:
        d_min = np.minimum(d_min, _dist_to_segment(x, y, ax, ay, bx, by))
    return np.exp(-d_min ** 2 / (2 * STROKE_WIDTH ** 2))


grid_res = 200
_xs_vis = np.linspace(0, 1, grid_res)
_ys_vis = np.linspace(0, 1, grid_res)
Xg, Yg = np.meshgrid(_xs_vis, _ys_vis)
Zg = target_distribution(Xg, Yg)

# ============================================================================
# 2. Fourier Ergodic Metric (for post-hoc evaluation, same as other methods)
# ============================================================================

K_FOURIER = 10
k_indices = np.array([[k1, k2] for k1 in range(K_FOURIER)
                       for k2 in range(K_FOURIER)])
Lambda_k = (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)


def fourier_basis(pts):
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)


_grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
_grid_w = Zg.ravel()
_grid_w = _grid_w / _grid_w.sum()
phi_k = np.sum(_grid_w[:, None] * fourier_basis(_grid_pts), axis=0)

W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0


def compute_fourier_energy(traj_flat, T):
    """Fourier energy metric for comparability with other methods."""
    X = traj_flat.reshape(T, 2)
    energy = 0.0
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    energy += W_SMOOTH * np.sum(accel ** 2)
    Fk = fourier_basis(X)
    c_k = np.mean(Fk, axis=0)
    diff_k = c_k - phi_k
    energy += W_ERGODIC * 0.5 * np.sum(Lambda_k * diff_k ** 2)
    margin = 0.03
    lo = np.minimum(X - margin, 0.0)
    hi = np.maximum(X - (1.0 - margin), 0.0)
    energy += W_BOUNDARY * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))
    if USE_OBSTACLE:
        dx = X[:, 0] - OBSTACLE_CENTER[0]
        dy = X[:, 1] - OBSTACLE_CENTER[1]
        dist = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)
        violation = np.maximum(OBSTACLE_RADIUS - dist, 0.0)
        energy += W_OBSTACLE * 0.5 * np.sum(violation ** 2)
    return energy


def compute_zeng_power_traj(traj, dt=0.02, physical_scale=8.0):
    """Compute the mean Zeng power (Watts) for a single trajectory.
    traj: shape (T, 2)
    """
    disp = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    V = physical_scale * disp / dt
    
    # Cap velocity at 30 m/s as per paper physical realizability
    V = np.clip(V, 0.0, 30.0)
    
    P0 = 79.85
    Pi = 88.63
    U_tip = 120.0
    v0 = 4.03
    d0 = 0.6
    rho = 1.225
    s = 0.05
    A = 0.503
    
    profile = P0 * (1.0 + 3.0 * (V ** 2) / (U_tip ** 2))
    
    term2_inner = np.sqrt(1.0 + (V ** 4) / (4.0 * (v0 ** 4))) - (V ** 2) / (2.0 * (v0 ** 2))
    term2_inner = np.maximum(term2_inner, 0.0)
    induced = Pi * np.sqrt(term2_inner)
    
    parasite = 0.5 * d0 * rho * s * A * (V ** 3)
    
    total_power = profile + induced + parasite
    return np.mean(total_power)


# ============================================================================
# 3. Hyperparameters
# ============================================================================

N_PARTICLES = 10
T = 100
INIT_NOISE_STD = 0.02

# OT-CFM Training
BATCH_SIZE = 1024
N_EPOCHS = 1000
LR = 2e-3
HIDDEN_DIM = 256
N_LAYERS = 4
SINKHORN_EPS = 0.05
RK4_TRAIN_STEPS = 8
RK4_EVAL_STEPS = 50
DELTA = 0.05

# Penalty weights
LAMBDA_NFZ = 100.0
LAMBDA_ACC = 0.01
ACC_PENALTY_FREQ = 10  # compute expensive Hessian penalty every N epochs
ACC_PENALTY_BATCH = 64  # smaller batch for Hessian finite-diff (9 RK4 fwd passes)

# Obstacle
USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-generated sample pool size (generated once, resampled per epoch)
TARGET_POOL_SIZE = 50000
ANNULUS_POOL_SIZE = 50000


def _build_target_pool(n_pool, device='cpu'):
    """Pre-generate a large pool of target density samples via rejection."""
    samples = []
    while len(samples) < n_pool:
        candidates = np.random.uniform(0, 1, size=(n_pool * 4, 2))
        probs = target_distribution(candidates[:, 0], candidates[:, 1])
        probs /= probs.max()
        accept = np.random.uniform(0, 1, size=len(candidates)) < probs
        samples.extend(candidates[accept].tolist())
    arr = np.array(samples[:n_pool])
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _build_annulus_pool(n_pool, delta=0.05, device='cpu'):
    """Pre-generate a large pool of annulus samples via rejection."""
    samples = []
    while len(samples) < n_pool:
        z = np.random.uniform(-1, 1, size=(n_pool * 2, 2))
        r = np.linalg.norm(z, axis=1)
        mask = (r >= delta) & (r <= 1.0)
        samples.extend(z[mask].tolist())
    arr = np.array(samples[:n_pool])
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _sample_from_pool(pool, n_samples):
    """Draw a random batch from a pre-generated pool."""
    idx = torch.randint(0, pool.shape[0], (n_samples,))
    return pool[idx]


# ============================================================================
# 5. Training Loop
# ============================================================================


def train_ot_cfm(use_obstacle=False):
    """Train the OT-CFM velocity field network.

    Returns:
        model: trained VelocityFieldMLP
        loss_log: list of per-epoch losses
    """
    latent = LatentTrajectory(delta=DELTA)
    model = VelocityFieldMLP(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    # Dynamic penalty configuration based on Table 5 of the paper
    if use_obstacle:
        lambda_nfz = 500.0
        lambda_energy = 1e-3
        lambda_acc = 1e-1
    else:
        lambda_nfz = 0.0
        lambda_energy = 1e-2
        lambda_acc = 1e-3

    # Obstacle as NFZ
    nfz_center = torch.tensor(OBSTACLE_CENTER, dtype=torch.float32, device=DEVICE)
    nfz_radius = OBSTACLE_RADIUS

    loss_log = []
    t_train_start = time.time()
    print(f"  Training OT-CFM on {DEVICE} | {N_EPOCHS} epochs, batch={BATCH_SIZE}")
    print(f"  MLP: {HIDDEN_DIM}×{N_LAYERS} layers, LR={LR}, Sinkhorn ε={SINKHORN_EPS}")
    print(f"  Weights: NFZ={lambda_nfz}, Energy={lambda_energy}, Acc={lambda_acc}")

    # --- Pre-generate sample pools (avoids per-epoch rejection sampling) ---
    print(f"  Pre-generating sample pools: {TARGET_POOL_SIZE} target, {ANNULUS_POOL_SIZE} annulus...")
    target_pool = _build_target_pool(TARGET_POOL_SIZE, device=DEVICE)
    annulus_pool = _build_annulus_pool(ANNULUS_POOL_SIZE, delta=DELTA, device=DEVICE)
    print(f"  Pools ready.")

    for epoch in range(N_EPOCHS):
        model.train()

        # --- Sample source z0 ~ pi_0^delta and target x1 ~ f_target ---
        z0 = _sample_from_pool(annulus_pool, BATCH_SIZE)
        x1 = _sample_from_pool(target_pool, BATCH_SIZE)

        # --- Sinkhorn OT coupling ---
        with torch.no_grad():
            indices = sinkhorn_coupling(z0, x1, eps=SINKHORN_EPS)
        x1_paired = x1[indices]

        # --- Sample flow time s ~ Uniform[0, 1] ---
        s = torch.rand(BATCH_SIZE, device=DEVICE)

        # --- CFM loss ---
        loss_cfm = cfm_loss(model, z0, x1_paired, s)
        loss_total = loss_cfm

        # --- NFZ penalty (only if obstacle is active) ---
        loss_nfz_val = 0.0
        if lambda_nfz > 0.0:
            z0_pen = _sample_from_pool(annulus_pool, min(256, BATCH_SIZE))
            loss_nfz = nfz_penalty(model, z0_pen, nfz_center, nfz_radius,
                                   n_steps=RK4_TRAIN_STEPS)
            loss_total = loss_total + lambda_nfz * loss_nfz
            loss_nfz_val = loss_nfz.item()

        # --- Zeng power (energy) penalty ---
        loss_energy_val = 0.0
        if lambda_energy > 0.0:
            # Generate a latent cycle of 200 points (period tau=2, dt=0.01)
            cycle = latent.generate_cycle(n_points=200)
            # Pick a random starting point for a chunk of length 64
            start_idx = np.random.randint(0, 200 - 64)
            chunk = cycle[start_idx : start_idx + 64]
            latent_chunk = torch.tensor(chunk, dtype=torch.float32, device=DEVICE)
            loss_energy = zeng_power_penalty(model, latent_chunk, dt=0.01,
                                             n_steps=RK4_TRAIN_STEPS, physical_scale=8.0)
            loss_total = loss_total + lambda_energy * loss_energy
            loss_energy_val = loss_energy.item()

        # --- Acceleration penalty (computed every ACC_PENALTY_FREQ epochs) ---
        loss_acc_val = 0.0
        if lambda_acc > 0.0 and epoch % ACC_PENALTY_FREQ == 0:
            z0_acc = _sample_from_pool(annulus_pool, ACC_PENALTY_BATCH)
            loss_acc = acceleration_penalty(model, z0_acc, n_steps=RK4_TRAIN_STEPS)
            loss_total = loss_total + lambda_acc * loss_acc
            loss_acc_val = loss_acc.item()

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_log.append(loss_total.item())
        if (epoch + 1) % 100 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_train_start
            eta = elapsed / (epoch + 1) * (N_EPOCHS - epoch - 1)
            print(f"    Epoch {epoch+1:4d}/{N_EPOCHS}  L_total={loss_total.item():.4f}  "
                  f"L_cfm={loss_cfm.item():.4f}  "
                  f"L_nfz={loss_nfz_val:.4f}  "
                  f"L_energy={loss_energy_val:.2f}  "
                  f"L_acc={loss_acc_val:.4f}  LR={lr_now:.2e}  "
                  f"({elapsed:.0f}s / ETA {eta:.0f}s)")

    model.eval()
    return model, loss_log


# ============================================================================
# 6. Inference: Generate Trajectories via Pushforward
# ============================================================================


def generate_trajectories(model, init_positions, n_points=100):
    """Push initial positions through the trained flow map G_theta.

    For each trajectory particle, we create a latent path on the annulus
    and integrate it through the learned velocity field.

    Args:
        model: trained VelocityFieldMLP
        init_positions: (N, T, 2) initial waypoint positions
        n_points: trajectory length (T)

    Returns:
        trajectories: (N, T, 2) pushed-forward trajectories
    """
    N = init_positions.shape[0]
    latent = LatentTrajectory(delta=DELTA)
    trajectories = np.zeros((N, n_points, 2))

    model.eval()
    with torch.no_grad():
        for i in range(N):
            # Generate a latent cycle for this particle
            cycle = latent.generate_cycle(n_points)  # (T, 2)
            z0 = torch.tensor(cycle, dtype=torch.float32, device=DEVICE)

            # Push through the learned map via RK4 integration
            y1 = rk4_integrate(model, z0, n_steps=RK4_EVAL_STEPS)
            traj = y1.cpu().numpy()

            # Clip to domain [0, 1]
            traj = np.clip(traj, 0.0, 1.0)
            trajectories[i] = traj

    return trajectories


# ============================================================================
# 7. Master Benchmark Function
# ============================================================================


def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False, n_particles: int = None):
    global USE_OBSTACLE, N_PARTICLES
    USE_OBSTACLE = use_obstacle
    if n_particles is not None:
        N_PARTICLES = n_particles

    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape", "polynomial", "rrt"]
    results = {}
    benchmark_data = {}

    print(f"2D OT-CFM  |  {N_PARTICLES} particles, T={T}")
    print(f"Training:  epochs={N_EPOCHS}, batch={BATCH_SIZE}, LR={LR}")
    print(f"MLP:       {HIDDEN_DIM}×{N_LAYERS}, RK4 train={RK4_TRAIN_STEPS}, eval={RK4_EVAL_STEPS}")
    print("-" * 65)

    # --- Stage 1: Train the OT-CFM network (once for all strategies) ---
    print("\n[Stage 1] Training OT-CFM pushforward map...")
    t_train_start = time.time()
    model, loss_log = train_ot_cfm(use_obstacle=use_obstacle)
    train_time = time.time() - t_train_start
    print(f"  Training completed in {train_time:.1f}s\n")

    # Save model
    torch.save(model.state_dict(), os.path.join(out_dir, 'ot_cfm_model.pt'))

    # --- Stage 2: Generate trajectories per init strategy ---
    print("[Stage 2] Generating trajectories per initialization strategy...")

    for strat in strategies:
        print(f"\n  Strategy: {strat}")
        t_start = time.time()

        # Get initialization (for visualization and starting positions)
        init_p, base_t = get_initialization(strat, N_PARTICLES, T,
                                            noise_std=INIT_NOISE_STD)
        pos_trajs = init_p.reshape(N_PARTICLES, T, 2)

        # Generate pushed-forward trajectories
        final_trajs = generate_trajectories(model, pos_trajs, n_points=T)
        final_flat = final_trajs.reshape(N_PARTICLES, -1)

        # Evaluate with shared Fourier energy metric
        energies = np.array([compute_fourier_energy(final_flat[j], T)
                             for j in range(N_PARTICLES)])
        best = int(np.argmin(energies))
        
        # Evaluate Zeng power (period tau=2.0 in normalized units, dt = 2.0 / T)
        zeng_powers = np.array([compute_zeng_power_traj(final_trajs[j], dt=2.0/T, physical_scale=8.0)
                                for j in range(N_PARTICLES)])
        
        elapsed = time.time() - t_start

        print(f"    -> Best energy: {energies[best]:.3f}  "
              f"Mean: {energies.mean():.3f}  "
              f"Mean Zeng Power: {zeng_powers.mean():.1f} W  (Time: {elapsed:.2f}s)")

        results[strat] = {
            'initial': init_p,
            'base_traj': base_t,
            'final': final_flat,
            'best_idx': best,
        }

        benchmark_data[strat] = {
            'mean_cost': float(np.mean(energies)),
            'best_cost': float(energies[best]),
            'mean_zeng_power_W': float(np.mean(zeng_powers)),
            'time_s': float(elapsed + train_time / len(strategies)),
        }

        if save_npy:
            np.save(os.path.join(out_dir, f"{strat}_trajs.npy"), final_flat)

    # ============================================================================
    # 8. Comparison-Grid Visualization
    # ============================================================================

    fig, axes = plt.subplots(len(strategies), 2, figsize=(12, 5 * len(strategies)))
    cmap = 'YlOrRd'
    colors = plt.cm.rainbow(np.linspace(0, 1, N_PARTICLES))

    def plot_particles(ax, parts, title,
                       highlight_best=False, best_idx=-1, base_traj=None):
        ax.contourf(Xg, Yg, Zg, levels=30, cmap=cmap, alpha=0.6)
        ax.contour(Xg, Yg, Zg, levels=6, colors='k', linewidths=0.3, alpha=0.3)

        if USE_OBSTACLE:
            circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                                color='gray', alpha=0.8, zorder=5)
            ax.add_patch(circle)

        if base_traj is not None:
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--',
                    color='white', lw=3.0, zorder=8)
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--',
                    color='black', lw=1.5, zorder=9, label='Base Trajectory')

        for i in range(N_PARTICLES):
            tr = parts[i].reshape(T, 2)
            lw = 2.5 if (highlight_best and i == best_idx) else 0.8
            al = 1.0 if (highlight_best and i == best_idx) else 0.4
            ax.plot(tr[:, 0], tr[:, 1], '-', color=colors[i], lw=lw, alpha=al)
            ax.plot(tr[0, 0], tr[0, 1], 'o', color=colors[i], ms=4)

        if highlight_best and best_idx >= 0:
            best_tr = parts[best_idx].reshape(T, 2)
            ax.plot(best_tr[:, 0], best_tr[:, 1], '-', color=colors[best_idx],
                    lw=2.5, label=f'Best (#{best_idx})', zorder=10)
            ax.legend(loc='lower right', fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    for row, strat in enumerate(strategies):
        res = results[strat]

        ax = axes[row, 0]
        plot_particles(ax, res['initial'], f'[{strat}] Initial',
                       base_traj=res['base_traj'])
        if res['base_traj'] is not None:
            ax.legend(loc='lower right', fontsize=9)

        ax = axes[row, 1]
        plot_particles(ax, res['final'], f'[{strat}] OT-CFM Result',
                       highlight_best=True, best_idx=res['best_idx'])

    plt.tight_layout()
    out_fig = os.path.join(out_dir, 'ot_cfm_2d_comparison.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()

    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T': T,
            'N_PARTICLES': N_PARTICLES,
            'N_EPOCHS': N_EPOCHS,
            'BATCH_SIZE': BATCH_SIZE,
            'LR': LR,
            'HIDDEN_DIM': HIDDEN_DIM,
            'N_LAYERS': N_LAYERS,
            'DELTA': DELTA,
            'SINKHORN_EPS': SINKHORN_EPS,
            'LAMBDA_NFZ': LAMBDA_NFZ,
            'LAMBDA_ACC': LAMBDA_ACC,
            'train_time_s': train_time,
        }, f, indent=4)

    return benchmark_data


if __name__ == "__main__":
    run_benchmark(out_dir='/home/philipp/Documents/Uni/Master_thesis/OT_CFM/test_run')
