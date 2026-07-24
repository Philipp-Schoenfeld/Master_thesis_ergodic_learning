#!/usr/bin/env python3
"""
flow_matching_runner_char.py — Flow-matching trainer for character/shape trajectories.

Conditions on reference trajectories (not class IDs), so the model can
generalise to unseen shapes at inference.

Usage:
  python flow_matching_runner_char.py                                  # train
  python flow_matching_runner_char.py --load_model checkpoints/cond_mpd_char_traj.pt  # viz only
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
from flow_matching_cond_mpd_unet_char import (
    CondMpdUNetFlowNetwork, compute_cond_cfm_loss, generate_cond_trajectories,
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
        traj = out[i]  # (nxi, 2)
        centroid = traj.mean(axis=0)

        # 1. Random rotation around centroid
        angle = np.random.uniform(-rot_range, rot_range) * np.pi / 180
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        traj = (traj - centroid) @ R.T + centroid

        # 2. Random scale around centroid
        scale = np.random.uniform(*scale_range)
        traj = (traj - centroid) * scale + centroid

        # 3. Random translation
        tx = np.random.uniform(-trans_range, trans_range)
        ty = np.random.uniform(-trans_range, trans_range)
        traj = traj + np.array([tx, ty])

        # 4. Random horizontal flip
        if np.random.rand() < p_flip:
            traj[:, 0] = 1.0 - traj[:, 0]

        # 5. Gaussian noise
        traj = traj + np.random.normal(0, noise_std, traj.shape)

        # 6. Clip to valid range
        out[i] = np.clip(traj, 0.0, 1.0)

    return out


# ===========================================================================
# Dataset loader
# ===========================================================================

def _load_shapes(nxi):
    """
    Load all shapes from the database.

    Returns dict: label → (nxi, 2) numpy array (clean reference trajectory).
    """
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
        shapes[label] = xy[idx]          # (nxi, 2)

    return shapes


def _build_dataset(shapes, holdout_labels, copies_per_char, noise_std, device):
    """
    Build training tensors, excluding held-out shapes.

    Returns
    -------
    x1       : (N, nxi, 2) float32 — augmented target trajectories
    ref_cps  : (N, nxi, 2) float32 — clean reference trajectories
    train_shapes : dict  label → base array  (training only)
    holdout_shapes : dict  label → base array
    """
    train_shapes, holdout_shapes = {}, {}
    for lbl, base in shapes.items():
        if lbl in holdout_labels:
            holdout_shapes[lbl] = base
        else:
            train_shapes[lbl] = base

    # Build arrays
    all_x1, all_ref = [], []
    for lbl, base in train_shapes.items():
        tiled = np.tile(base[None], (copies_per_char, 1, 1))
        aug   = augment_batch(tiled, noise_std=noise_std)
        all_x1.append(aug)
        all_ref.append(tiled)                                  # clean condition

    x1_np  = np.concatenate(all_x1,  axis=0)
    ref_np = np.concatenate(all_ref, axis=0)

    perm = np.random.permutation(len(x1_np))
    x1  = torch.tensor(x1_np[perm],  dtype=torch.float32).to(device)
    ref = torch.tensor(ref_np[perm], dtype=torch.float32).to(device)
    return x1, ref, train_shapes, holdout_shapes


# ===========================================================================
# Training
# ===========================================================================

def train(model, x1, ref_cps, loss_fn, epochs, lr, mini_batch):
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                            eta_min=1e-5)
    model.train()
    use_cuda = x1.device.type == 'cuda'
    N = x1.shape[0]

    from tqdm import tqdm
    pbar = tqdm(range(epochs), desc="Training", unit="ep")

    for ep in pbar:
        perm = torch.randperm(N, device=x1.device)
        ep_loss, nb = 0.0, 0

        for i in range(0, N, mini_batch):
            idx = perm[i : i + mini_batch]
            opt.zero_grad()
            with torch.autocast(
                device_type='cuda' if use_cuda else 'cpu',
                dtype=torch.bfloat16,
            ):
                loss = loss_fn(model, x1[idx], ref_cps[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()
        avg = ep_loss / max(nb, 1)
        if ep % 10 == 0 or ep == epochs - 1:
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
        if len(cp) >= 6:
            ax.plot(*cp_to_bspline(cp, bspline_pts, bspline_deg).T,
                    color='#EF5350', lw=1.8, alpha=0.85,
                    label='Generated' if i == 0 else '', zorder=3)
        ax.scatter(cp[:, 0], cp[:, 1],
                   color='#EF5350', s=8, alpha=0.4, zorder=3)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.10, lw=0.5)
    ax.set_title(title, fontsize=10, color='#1A1A2E', pad=6)
    ax.legend(frameon=False, fontsize=7, loc='upper left')


def visualise_set(model, shapes_dict, title_prefix, save_path,
                  args, device, max_cols=5):
    """Generate & plot trajectories for a set of shapes."""
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
        ref_t = torch.tensor(base, dtype=torch.float32)
        gen = generate_cond_trajectories(
            model, ref_t,
            num_samples=args.n_gen, nxi=args.nxi, nd=args.nd,
            steps=args.steps, device=str(device),
        ).cpu().numpy()
        _draw_traj(ax, base, gen, f"'{lbl}'",
                   args.bspline_pts, args.bspline_deg)

    # hide unused axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ===========================================================================
# Main
# ===========================================================================

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"\n{'=' * 65}")
    print(f"  Character Flow Matching (trajectory conditioning)")
    print(f"  device={device}  nxi={args.nxi}  D={args.D}  epochs={args.epochs}")
    print(f"{'=' * 65}")

    # ── data ──
    all_shapes = _load_shapes(args.nxi)
    print(f"  Loaded {len(all_shapes)} shapes from DB")

    x1, ref_cps, train_shapes, holdout_shapes = _build_dataset(
        all_shapes, HOLDOUT_LABELS,
        copies_per_char=args.copies_per_char,
        noise_std=args.noise_std, device=device,
    )
    print(f"  Training: {len(train_shapes)} shapes, "
          f"{x1.shape[0]} samples")
    print(f"  Held out: {len(holdout_shapes)} shapes "
          f"({', '.join(sorted(holdout_shapes.keys()))})")

    # ── model ──
    model = CondMpdUNetFlowNetwork(
        nxi=args.nxi, nd=args.nd, D=args.D,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {params:,}\n")

    # ── train or load ──
    if args.load_model and os.path.isfile(args.load_model):
        ckpt = torch.load(args.load_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint from {args.load_model}")
    else:
        train(model, x1, ref_cps, compute_cond_cfm_loss,
              args.epochs, args.lr, args.mini_batch)
        print("  Training complete!")

        save_path = args.save_model
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'nxi': args.nxi, 'nd': args.nd, 'D': args.D,
            'epochs': args.epochs, 'lr': args.lr,
            'holdout_labels': list(HOLDOUT_LABELS),
        }, save_path)
        print(f"  Checkpoint saved → {save_path}")

    # ── visualise training shapes (5 random) ──
    viz_train = dict(random.sample(list(train_shapes.items()),
                                    min(5, len(train_shapes))))
    out_dir = os.path.join(_here, 'Trajectory_data_generator')
    visualise_set(
        model, viz_train,
        'Training Shapes — Ground Truth vs Generated',
        os.path.join(out_dir, 'char_train_generation.png'),
        args, device,
    )

    # ── visualise held-out shapes (ALL) ──
    visualise_set(
        model, holdout_shapes,
        'HELD-OUT Shapes (never seen during training) — Ground Truth vs Generated',
        os.path.join(out_dir, 'char_holdout_generation.png'),
        args, device,
    )


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # architecture
    p.add_argument('--nxi', type=int, default=20)
    p.add_argument('--nd',  type=int, default=2)
    p.add_argument('--D',   type=int, default=128)
    # training
    p.add_argument('--epochs',          type=int,   default=500)
    p.add_argument('--lr',              type=float, default=3e-4)
    p.add_argument('--mini_batch',      type=int,   default=256)
    p.add_argument('--copies_per_char', type=int,   default=100)
    p.add_argument('--noise_std',       type=float, default=0.015)
    # generation / visualisation
    p.add_argument('--n_gen',       type=int, default=3)
    p.add_argument('--steps',       type=int, default=100)
    p.add_argument('--bspline_pts', type=int, default=512)
    p.add_argument('--bspline_deg', type=int, default=5)
    # persistence
    p.add_argument('--save_model', type=str,
                   default=os.path.join(_here, 'checkpoints',
                                        'cond_mpd_char_traj.pt'))
    p.add_argument('--load_model', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
