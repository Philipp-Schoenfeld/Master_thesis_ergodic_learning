#!/usr/bin/env python3
"""
flow_matching_runner_spectral.py
================================
Flow-matching trainer for character/shape trajectories with SPECTRAL conditioning.

Uses SpectralTokenizer + Cross-Attention instead of ShapeEncoderMPD + Global Pooling.
Computes dummy spectral coefficients from trajectory FFT until real Laplace-Beltrami
coefficients from the TSVEC solver are available.

Usage:
  python flow_matching_runner_spectral.py                                                    # train
  python flow_matching_runner_spectral.py --load_model checkpoints/cond_spectral_crossattn.pt  # viz
"""

import argparse, os, random, sqlite3, sys
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bsplinax.bspline import BsplineBasisClamped
from flow_matching_cond_spectral_crossattn import (
    SpectralCrossAttnFlowNetwork, compute_spectral_cfm_loss,
    generate_spectral_trajectories,
)

_DB_PATH = os.path.join(_here, 'Trajectory_data_generator', 'character_trajectories.db')

# Shapes held out from training to evaluate generalisation
HOLDOUT_LABELS = {
    'G', 'W', '5', 'sigma', 'phi',            # 5 characters
    'star_5', 'spiral_2cw', 'lissajous_1_3',   # 3 procedural
    'heart', 'rand_poly_7',                     # 2 more procedural
}

# ===========================================================================
# Helpers
# ===========================================================================

def cp_to_bspline(cps, pts=512, deg=5):
    nxi = cps.shape[0]
    B = np.array(BsplineBasisClamped(
        degree=deg, num_control_points=nxi,
        num_phase_points=pts, compute_derivatives=False).B)
    return B @ cps


# ===========================================================================
# Spectral coefficient extraction (DUMMY — replace with real TSVEC later)
# ===========================================================================

def trajectory_to_spectral(traj, S=50):
    """
    Extract S leading spectral (Fourier) coefficients from a 2D trajectory.

    This is a PLACEHOLDER that computes the FFT of the trajectory coordinates
    and returns the S leading amplitude coefficients. In the final pipeline,
    this should be replaced by the Laplace-Beltrami spectral decomposition
    of the target ergodic distribution.

    traj: (nxi, 2) numpy array
    Returns: 
        amplitudes: (S,) numpy array of spectral amplitudes
        k_indices: (S, 2) numpy array of integer frequencies
    """
    # Treat trajectory as complex signal: x + iy
    signal = traj[:, 0] + 1j * traj[:, 1]

    # Zero-pad to at least 2*S points for frequency resolution
    n_pad = max(len(signal), 2 * S)
    padded = np.zeros(n_pad, dtype=np.complex128)
    padded[:len(signal)] = signal

    # FFT and take leading S amplitudes
    fft_coeffs = np.fft.fft(padded)
    amplitudes = np.abs(fft_coeffs[:S]).astype(np.float32)

    # Normalize to [0, 1] for stable training
    max_val = amplitudes.max()
    if max_val > 1e-8:
        amplitudes = amplitudes / max_val

    # Generate dummy 2D indices (k1, k2) for the spectral coefficients
    K = int(np.ceil(np.sqrt(S)))
    k_indices = np.zeros((S, 2), dtype=np.int64)
    for i in range(S):
        k_indices[i, 0] = i % K
        k_indices[i, 1] = i // K

    return amplitudes, k_indices


# ===========================================================================
# On-the-fly augmentation
# ===========================================================================

def augment_batch(x, p_flip=0.2, rot_range=20, scale_range=(0.75, 1.25),
                  trans_range=0.08, noise_std=0.01):
    """
    Apply random geometric augmentations to a batch of trajectories.
    x : (B, nxi, 2) numpy array
    Returns augmented copy (same shape).
    """
    B, nxi, _ = x.shape
    out = x.copy()

    for i in range(B):
        traj = out[i]
        centroid = traj.mean(axis=0)

        # Random rotation
        angle = np.random.uniform(-rot_range, rot_range) * np.pi / 180
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        traj = (traj - centroid) @ R.T + centroid

        # Random scale
        scale = np.random.uniform(*scale_range)
        traj = (traj - centroid) * scale + centroid

        # Random translation
        tx = np.random.uniform(-trans_range, trans_range)
        ty = np.random.uniform(-trans_range, trans_range)
        traj = traj + np.array([tx, ty])

        # Random horizontal flip
        if np.random.rand() < p_flip:
            traj[:, 0] = 1.0 - traj[:, 0]

        # Gaussian noise
        traj = traj + np.random.normal(0, noise_std, traj.shape)

        out[i] = np.clip(traj, 0.0, 1.0)

    return out


# ===========================================================================
# Dataset loader
# ===========================================================================

def _load_shapes(nxi):
    conn = sqlite3.connect(_DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT trajectory, shape, label FROM runs ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    shapes = {}
    for blob, sh, label in rows:
        dims = tuple(map(int, sh.split(',')))
        xy = np.frombuffer(blob, dtype=np.float32).reshape(dims)[:, :2]
        idx = np.linspace(0, len(xy) - 1, nxi).astype(int)
        shapes[label] = xy[idx]
    return shapes


def _build_dataset(shapes, holdout_labels, copies_per_char, S, device):
    """
    Build training tensors with spectral coefficients as condition.
    Stores CLEAN copies — augmentation happens on-the-fly.
    """
    train_shapes, holdout_shapes = {}, {}
    for lbl, base in shapes.items():
        if lbl in holdout_labels:
            holdout_shapes[lbl] = base
        else:
            train_shapes[lbl] = base

    all_x1, all_spec, all_k_idx = [], [], []
    for lbl, base in train_shapes.items():
        # Compute spectral coefficients ONCE per base shape (condition = clean)
        spec, k_idx = trajectory_to_spectral(base, S=S)

        tiled = np.tile(base[None], (copies_per_char, 1, 1))
        spec_tiled = np.tile(spec[None], (copies_per_char, 1))
        k_idx_tiled = np.tile(k_idx[None], (copies_per_char, 1, 1))

        all_x1.append(tiled.copy())
        all_spec.append(spec_tiled.copy())
        all_k_idx.append(k_idx_tiled.copy())

    x1_np   = np.concatenate(all_x1, axis=0)
    spec_np = np.concatenate(all_spec, axis=0)
    k_idx_np = np.concatenate(all_k_idx, axis=0)

    perm = np.random.permutation(len(x1_np))
    x1   = torch.tensor(x1_np[perm],   dtype=torch.float32).to(device)
    spec = torch.tensor(spec_np[perm], dtype=torch.float32).to(device)
    k_idx = torch.tensor(k_idx_np[perm], dtype=torch.long).to(device)

    # Also precompute spectral coefficients for holdout shapes
    holdout_spec = {}
    for lbl, base in holdout_shapes.items():
        holdout_spec[lbl] = trajectory_to_spectral(base, S=S)

    return x1, spec, k_idx, train_shapes, holdout_shapes, holdout_spec


# ===========================================================================
# Training
# ===========================================================================

def train(model, x1_clean, spec_cond, k_idx_cond, loss_fn, args):
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                            eta_min=1e-5)
    model.train()
    use_cuda = x1_clean.device.type == 'cuda'
    N = x1_clean.shape[0]

    from tqdm import tqdm
    pbar = tqdm(range(args.epochs), desc="Training", unit="ep")

    for ep in pbar:
        # On-the-fly augmentation: fresh random transforms every epoch
        x1_np_aug = augment_batch(
            x1_clean.cpu().numpy(), noise_std=args.noise_std
        )
        x1 = torch.tensor(x1_np_aug, dtype=torch.float32, device=x1_clean.device)

        perm = torch.randperm(N, device=x1.device)
        ep_loss, nb = 0.0, 0

        for i in range(0, N, args.mini_batch):
            idx = perm[i : i + args.mini_batch]
            opt.zero_grad()
            with torch.autocast(
                device_type='cuda' if use_cuda else 'cpu',
                dtype=torch.bfloat16,
            ):
                loss = loss_fn(model, x1[idx], spec_cond[idx], k_idx_cond[idx], p_drop=args.p_drop)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()
        avg = ep_loss / max(nb, 1)
        if ep % 10 == 0 or ep == args.epochs - 1:
            pbar.set_postfix(loss=f"{avg:.5f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")


# ===========================================================================
# Visualisation
# ===========================================================================

def _draw_traj(ax, base, gen_cps, title, bspline_pts=512, bspline_deg=5):
    ax.set_facecolor('white')
    if len(base) >= 6:
        ax.plot(*cp_to_bspline(base, bspline_pts, bspline_deg).T,
                color='#1565C0', lw=2.5, label='Ground Truth', zorder=2)
        ax.scatter(base[:, 0], base[:, 1],
                   color='#1565C0', s=12, alpha=0.5, zorder=2)
    for i, cp in enumerate(gen_cps):
        main_alpha = 0.85 if i == 0 else 0.2
        scatter_alpha = 0.4 if i == 0 else 0.1
        if len(cp) >= 6:
            ax.plot(*cp_to_bspline(cp, bspline_pts, bspline_deg).T,
                    color='#EF5350', lw=1.8, alpha=main_alpha,
                    label='Generated' if i == 0 else '', zorder=3)
        ax.scatter(cp[:, 0], cp[:, 1],
                   color='#EF5350', s=8, alpha=scatter_alpha, zorder=3)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.10, lw=0.5)
    ax.set_title(title, fontsize=10, color='#1A1A2E', pad=6)
    ax.legend(frameon=False, fontsize=7, loc='upper left')


def visualise_set(model, shapes_dict, spectral_dict, title_prefix, save_path,
                  args, device, max_cols=5):
    """Generate & plot trajectories for a set of shapes (spectral conditioned)."""
    labels = list(shapes_dict.keys())
    n = len(labels)
    if n == 0:
        return
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 5 * n_rows),
                              facecolor='white', squeeze=False)
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold',
                 color='#1A1A2E', y=1.01)

    for idx, lbl in enumerate(labels):
        ax = axes[idx // n_cols][idx % n_cols]
        base = shapes_dict[lbl]
        spec, k_idx = spectral_dict[lbl]
        spec_t = torch.tensor(spec, dtype=torch.float32)
        k_idx_t = torch.tensor(k_idx, dtype=torch.long)

        gen, lam = generate_spectral_trajectories(
            model, spec_t, k_idx_t,
            num_samples=args.n_gen, nxi=args.nxi, nd=args.nd,
            steps=args.steps, device=str(device),
            cfg_weight=args.cfg_weight,
        )
        gen = gen.cpu().numpy()
        _draw_traj(ax, base, gen, f"'{lbl}'",
                   args.bspline_pts, args.bspline_deg)

        # Show lambda if predicted
        if lam is not None:
            lam_str = ", ".join(f"{v:.2f}" for v in lam[0].cpu().numpy())
            ax.text(0.02, 0.02, f"lam=[{lam_str}]",
                    transform=ax.transAxes, fontsize=5, alpha=0.5,
                    verticalalignment='bottom')

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")


# ===========================================================================
# Main
# ===========================================================================

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"\n{'=' * 70}")
    print(f"  Spectral Cross-Attention Flow Matching")
    print(f"  device={device}  nxi={args.nxi}  D={args.D}  S={args.S}")
    print(f"  epochs={args.epochs}  predict_lambda={args.predict_lambda}")
    print(f"  CFG: p_drop={args.p_drop}  cfg_weight={args.cfg_weight}")
    print(f"{'=' * 70}")

    # ── data ──
    all_shapes = _load_shapes(args.nxi)
    print(f"  Loaded {len(all_shapes)} shapes from DB")

    x1, spec_cond, k_idx_cond, train_shapes, holdout_shapes, holdout_spec = _build_dataset(
        all_shapes, HOLDOUT_LABELS,
        copies_per_char=args.copies_per_char,
        S=args.S, device=device,
    )

    # Precompute spectral coefficients for train shapes (for visualization)
    train_spec = {lbl: trajectory_to_spectral(base, S=args.S)
                  for lbl, base in train_shapes.items()}

    print(f"  Training: {len(train_shapes)} shapes, {x1.shape[0]} samples")
    print(f"  Spectral dim: S={args.S}")
    print(f"  Held out: {len(holdout_shapes)} shapes "
          f"({', '.join(sorted(holdout_shapes.keys()))})")

    # ── model ──
    model = SpectralCrossAttnFlowNetwork(
        nxi=args.nxi, nd=args.nd, D=args.D, S=args.S,
        n_lambda=args.n_lambda, predict_lambda=args.predict_lambda,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {params:,}\n")

    # ── train or load ──
    if args.load_model and os.path.isfile(args.load_model):
        ckpt = torch.load(args.load_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint from {args.load_model}")
    else:
        train(model, x1, spec_cond, k_idx_cond, compute_spectral_cfm_loss, args)
        print("  Training complete!")

        save_path = args.save_model
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'nxi': args.nxi, 'nd': args.nd, 'D': args.D, 'S': args.S,
            'n_lambda': args.n_lambda, 'predict_lambda': args.predict_lambda,
            'epochs': args.epochs, 'lr': args.lr,
            'p_drop': args.p_drop, 'cfg_weight': args.cfg_weight,
            'holdout_labels': list(HOLDOUT_LABELS),
        }, save_path)
        print(f"  Checkpoint saved -> {save_path}")

    # ── visualise training shapes (5 random) ──
    viz_train_keys = random.sample(list(train_shapes.keys()),
                                    min(5, len(train_shapes)))
    viz_train = {k: train_shapes[k] for k in viz_train_keys}
    viz_train_spec = {k: train_spec[k] for k in viz_train_keys}

    out_dir = os.path.join(_here, 'Trajectory_data_generator')
    visualise_set(
        model, viz_train, viz_train_spec,
        'Training Shapes (Spectral Cross-Attn)',
        os.path.join(out_dir, 'char_train_generation_spectral.png'),
        args, device,
    )

    # ── visualise held-out shapes (ALL) ──
    visualise_set(
        model, holdout_shapes, holdout_spec,
        'HELD-OUT Shapes (Spectral Cross-Attn)',
        os.path.join(out_dir, 'char_holdout_generation_spectral.png'),
        args, device,
    )


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # architecture
    p.add_argument('--nxi', type=int, default=20)
    p.add_argument('--nd',  type=int, default=2)
    p.add_argument('--D',   type=int, default=256)
    p.add_argument('--S',   type=int, default=50,
                   help="Number of spectral coefficients")
    # lambda head
    p.add_argument('--n_lambda', type=int, default=6,
                   help="Dimension of Lagrange multiplier output")
    p.add_argument('--predict_lambda', action='store_true', default=False,
                   help="Enable lambda prediction head")
    p.add_argument('--no_predict_lambda', dest='predict_lambda',
                   action='store_false',
                   help="Disable lambda prediction head")
    # training
    p.add_argument('--epochs',          type=int,   default=1000)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--mini_batch',      type=int,   default=512)
    p.add_argument('--copies_per_char', type=int,   default=50)
    p.add_argument('--noise_std',       type=float, default=0.015)
    # CFG
    p.add_argument('--p_drop',     type=float, default=0.0)
    p.add_argument('--cfg_weight', type=float, default=1.0)
    # generation / visualisation
    p.add_argument('--n_gen',       type=int, default=3)
    p.add_argument('--steps',       type=int, default=100)
    p.add_argument('--bspline_pts', type=int, default=512)
    p.add_argument('--bspline_deg', type=int, default=5)
    # persistence
    p.add_argument('--save_model', type=str,
                   default=os.path.join(_here, 'checkpoints',
                                        'cond_spectral_crossattn_ep1000.pt'))
    p.add_argument('--load_model', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
