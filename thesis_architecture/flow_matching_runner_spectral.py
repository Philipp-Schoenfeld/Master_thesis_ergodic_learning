#!/usr/bin/env python3
"""
flow_matching_runner_spectral.py
================================
Flow-matching trainer for character/shape trajectories with SPECTRAL conditioning.

Uses SpectralTokenizer + Cross-Attention instead of ShapeEncoderMPD + Global Pooling.
Conditions on ergodic Fourier coefficients c_k = (1/T) Σ_t cos(π k1 xₜ) cos(π k2 yₜ),
the same spectral representation used by the SVGD and flow-matching ergodic solvers.

Usage:
  python flow_matching_runner_spectral.py                                                    # train
  python flow_matching_runner_spectral.py --load_model checkpoints/cond_spectral_crossattn.pt  # viz
"""

import argparse, os, random, sqlite3, sys, math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

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
    'star_5_ir4', 'spiral_2cw', 'lissajous_1_3_d2',   # 3 procedural
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
# Ergodic Fourier Spectral Conditioning
# ===========================================================================
#
# All ergodic coverage solvers in this codebase (tsvec_2d.py, flow_matching_2d.py,
# svgd_bspline_2d.py, etc.) share the same spectral representation:
#
#   k_indices : {(k1, k2) : k1,k2 in 0..K-1}  — 2D cosine frequency pairs
#   Basis     : F_k(x) = cos(π k1 x) · cos(π k2 y)
#   Lambda_k  : (1 + |k|²)^(-3/2)             — spectral discount (smoother → higher weight)
#   c_k       : (1/T) Σ_t F_k(x_t)            — time-averaged trajectory statistics
#   phi_k     : ∫ μ(x) F_k(x) dx              — target distribution statistics
#
# The ergodic metric is: E = Σ_k Lambda_k · (c_k - phi_k)²
# We use c_k as the spectral condition for the flow network. The k_indices
# directly index the cosine Fourier basis — no LB eigenvectors needed.

def _make_ergodic_k_grid(K):
    """Return the K² ergodic frequency index grid used by all SVGD solvers."""
    k_indices = np.array([[k1, k2] for k1 in range(K) for k2 in range(K)],
                         dtype=np.int64)          # (K², 2)
    Lambda_k = (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)  # (K²,)
    return k_indices, Lambda_k


def _ergodic_fourier_basis(pts, k_indices):
    """
    Evaluate 2D cosine Fourier basis at trajectory points.
    pts      : (T, 2) float
    k_indices: (M, 2) int
    Returns  : (T, M) float — F_k(x_t) = cos(π k1 x) · cos(π k2 y)
    """
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]   # (T, M, 2)
    return np.prod(np.cos(args), axis=-1)                     # (T, M)


def _ergodic_fourier_basis_batch(pts, k_indices):
    """
    Vectorized over batch of trajectories.
    pts      : (B, T, 2) float
    k_indices: (M, 2) int
    Returns  : (B, T, M) float
    """
    args = np.pi * pts[:, :, None, :] * k_indices[None, None, :, :]  # (B, T, M, 2)
    return np.prod(np.cos(args), axis=-1)                            # (B, T, M)


def trajectory_to_spectral(traj, S=50):
    """
    Compute ergodic Fourier coefficients c_k for a 2D trajectory.

    Uses the identical 2D cosine basis and k_indices grid as all ergodic
    coverage solvers in this codebase (tsvec_2d, flow_matching_2d, etc.).

    c_k = (1/T) Σ_t cos(π k1 · x_t) · cos(π k2 · y_t)

    These coefficients encode HOW WELL the trajectory covers the space in
    each spatial frequency — a physically grounded description that is
    distinct for different shapes, traversal directions, and start points.

    Two channels are returned per frequency:
      - c_k:      the time-averaged statistic (coverage fingerprint)
      - Lambda_k: the spectral discount weight (informative of frequency rank)

    traj: (nxi, 2) numpy array of B-spline control points in [0,1]²
    S:    number of spectral coefficients to use. K = ceil(sqrt(S)) yields
          a K×K grid; only the first S entries of the K² grid are kept.

    Returns:
        spec:     (S, 2) float32 — [c_k, Lambda_k] per frequency
        k_indices: (S, 2) int64 — (k1, k2) cosine frequency pairs
    """
    K = int(np.ceil(np.sqrt(S)))
    k_idx_full, Lambda_full = _make_ergodic_k_grid(K)  # (K², 2), (K²,)

    # Time-average cosine basis over trajectory
    Fk = _ergodic_fourier_basis(traj, k_idx_full)       # (nxi, K²)
    c_k = np.mean(Fk, axis=0)                           # (K²,)  in [-1, 1]

    # Keep first S entries
    c_k       = c_k[:S].astype(np.float32)
    Lambda_k  = Lambda_full[:S].astype(np.float32)
    k_indices = k_idx_full[:S]

    # Stack as two channels: coverage statistic + spectral weight
    spec = np.stack([c_k, Lambda_k], axis=-1)           # (S, 2)

    return spec, k_indices


def trajectory_to_spectral_batch_torch(traj, k_idx_full, Lambda_full, S=50):
    """
    Vectorized computation of ergodic Fourier coefficients for a batch (PyTorch GPU).
    traj: (B, nxi, 2)
    k_idx_full: (K2, 2) float32
    Lambda_full: (K2,) float32
    """
    B = traj.shape[0]
    # traj: (B, nxi, 1, 2) | k_idx: (1, 1, K2, 2)
    args = math.pi * traj.unsqueeze(2) * k_idx_full.unsqueeze(0).unsqueeze(0)
    
    # cos & prod: (B, nxi, K2) -> mean: (B, K2)
    c_k = torch.cos(args).prod(dim=-1).mean(dim=1)
    
    c_k = c_k[:, :S]
    Lambda_k_batch = Lambda_full[:S].unsqueeze(0).expand(B, -1)
    
    return torch.stack([c_k, Lambda_k_batch], dim=-1)  # (B, S, 2)


# ===========================================================================
# On-the-fly augmentation (GPU)
# ===========================================================================

def augment_batch_torch(x, p_flip=0.2, rot_range=20.0, scale_range=(0.75, 1.25),
                        trans_range=0.08, noise_std=0.01):
    """
    Vectorized geometric augmentation over a batch (PyTorch GPU).
    x : (B, nxi, 2)
    """
    B, nxi, _ = x.shape
    device = x.device
    out = x.clone()
    centroids = out.mean(dim=1, keepdim=True)
    
    # Rotation
    angles = (torch.rand(B, device=device) * 2 - 1) * rot_range * (math.pi / 180.0)
    c, s = torch.cos(angles), torch.sin(angles)
    R = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1)
    ], dim=1)  # (B, 2, 2)
    
    out = torch.bmm((out - centroids), R.transpose(1, 2)) + centroids
    
    # Scale
    scales = torch.empty(B, 1, 1, device=device).uniform_(*scale_range)
    out = (out - centroids) * scales + centroids
    
    # Translation
    trans = torch.empty(B, 1, 2, device=device).uniform_(-trans_range, trans_range)
    out = out + trans
    
    # Flip
    flip_mask = torch.rand(B, device=device) < p_flip
    out[flip_mask, :, 0] = 1.0 - out[flip_mask, :, 0]
    
    # Noise
    out = out + torch.randn_like(out) * noise_std
    
    return torch.clamp(out, 0.0, 1.0)


# ===========================================================================
# Dataset loader
# ===========================================================================

def _load_shapes(nxi):
    conn = sqlite3.connect(_DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT trajectory, shape, label, id FROM runs ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    shapes = {}
    for blob, sh, label, row_id in rows:
        dims = tuple(map(int, sh.split(',')))
        xy = np.frombuffer(blob, dtype=np.float32).reshape(dims)[:, :2]
        idx = np.linspace(0, len(xy) - 1, nxi).astype(int)
        key = label if label not in shapes else f"{label}#{row_id}"
        shapes[key] = xy[idx]
    return shapes


def _build_dataset(shapes, holdout_labels, copies_per_char, S, device):
    """
    Build training tensors with spectral coefficients as condition.
    Stores CLEAN copies — augmentation happens on-the-fly.
    """
    train_shapes, holdout_shapes = {}, {}
    for lbl, base in shapes.items():
        base_lbl = lbl.split('#')[0]
        if base_lbl in holdout_labels:
            holdout_shapes[lbl] = base
        else:
            train_shapes[lbl] = base

    all_x1 = []
    for lbl, base in train_shapes.items():
        tiled = np.tile(base[None], (copies_per_char, 1, 1))         # (N, nxi, 2)
        all_x1.append(tiled)

    x1_np = np.concatenate(all_x1, axis=0)
    perm  = np.random.permutation(len(x1_np))
    x1    = torch.tensor(x1_np[perm], dtype=torch.float32).to(device)

    # Precompute spectral coefficients for holdout shapes
    holdout_spec = {}
    for lbl, base in holdout_shapes.items():
        holdout_spec[lbl] = trajectory_to_spectral(base, S=S)

    return x1, train_shapes, holdout_shapes, holdout_spec


# ===========================================================================
# Training
# ===========================================================================

def _save_checkpoint(model, optimizer, scheduler, epoch, loss, args):
    """Save a resumable mid-training checkpoint."""
    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
    stem = os.path.splitext(args.save_model)[0]
    path = f"{stem}_{args.run_str}_ep{epoch+1:04d}.pt"
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss':  loss,
        'nxi':   args.nxi, 'nd': args.nd, 'D': args.D, 'S': args.S,
        'n_lambda': args.n_lambda, 'predict_lambda': args.predict_lambda,
        'p_drop': args.p_drop, 'cfg_weight': args.cfg_weight,
    }, path)
    return path


def _save_viz(model, holdout_shapes, holdout_spec, title, ep_num, args, device, tag=None):
    """Helper to save a holdout visualisation during or after training."""
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectory_data_generator')
    os.makedirs(out_dir, exist_ok=True)
    run_str  = f"S{args.S}_nxi{args.nxi}_D{args.D}_flip{args.p_flip}"
    tag_str  = f"_{tag}" if tag else ""
    viz_path = os.path.join(out_dir, f'viz_holdout_{run_str}{tag_str}_ep{ep_num:04d}.png')
    model.eval()
    with torch.no_grad():
        visualise_set(model, holdout_shapes, holdout_spec, title, viz_path, args, device, max_cols=5)
    model.train()
    return viz_path


def train(model, x1_clean, loss_fn, args, holdout_shapes=None, holdout_spec=None):
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    model.train()
    use_cuda    = x1_clean.device.type == 'cuda'
    N           = x1_clean.shape[0]
    start_epoch = 0

    # -- Resume from checkpoint if available --
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=x1_clean.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed from epoch {start_epoch} (loss={ckpt['loss']:.5f})")

    # -- Precompute spectral grid on GPU once; reused every mini-batch --
    K = int(np.ceil(np.sqrt(args.S)))
    k_idx_full_np, Lambda_full_np = _make_ergodic_k_grid(K)
    k_idx_full  = torch.tensor(k_idx_full_np,  dtype=torch.float32, device=x1_clean.device)
    Lambda_full = torch.tensor(Lambda_full_np, dtype=torch.float32, device=x1_clean.device)
    k_idx_cond  = torch.tensor(k_idx_full_np[:args.S], dtype=torch.long, device=x1_clean.device)
    # Pre-expand to full mini_batch size; slice [:actual_bs] for the last smaller batch
    k_idx_cond_batch = k_idx_cond.unsqueeze(0).expand(args.mini_batch, -1, -1).contiguous()

    # -- Catch SIGTERM (SLURM timeout) so we can emergency-save --
    import signal
    class TerminateInterrupt(Exception): pass
    def _sigterm_handler(signum, frame): raise TerminateInterrupt()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    from tqdm import tqdm
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="ep")
    ep  = start_epoch
    avg = 0.0

    try:
        for ep in pbar:
            perm        = torch.randperm(N, device=x1_clean.device)
            ep_loss, nb = 0.0, 0

            for i in range(0, N, args.mini_batch):
                idx         = perm[i : i + args.mini_batch]
                batch_clean = x1_clean[idx]
                actual_bs   = batch_clean.shape[0]

                # On-the-fly augmentation entirely on GPU (no CPU round-trip)
                batch_aug = augment_batch_torch(
                    batch_clean, noise_std=args.noise_std, p_flip=args.p_flip,
                    rot_range=args.rot_range, scale_range=args.scale_range,
                    trans_range=args.trans_range,
                )

                # Spectral coefficients recomputed on GPU after augmentation
                spec_cond = trajectory_to_spectral_batch_torch(
                    batch_aug, k_idx_full, Lambda_full, S=args.S,
                )

                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type='cuda' if use_cuda else 'cpu',
                    dtype=torch.bfloat16,
                ):
                    loss = loss_fn(
                        model, batch_aug, spec_cond,
                        k_idx_cond_batch[:actual_bs], p_drop=args.p_drop,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
                nb      += 1

            scheduler.step()
            avg    = ep_loss / max(nb, 1)
            lr_now = scheduler.get_last_lr()[0]
            if ep % 10 == 0 or ep == args.epochs - 1:
                pbar.set_postfix(loss=f"{avg:.5f}", lr=f"{lr_now:.2e}")

            # W&B logging
            if getattr(args, 'use_wandb', False) and _WANDB_OK:
                wandb.log({'train/loss': avg, 'train/lr': lr_now, 'epoch': ep + 1})

            # Periodic checkpoint
            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                _save_checkpoint(model, opt, scheduler, ep, avg, args)
                print(f"  [ep {ep+1}] checkpoint saved (loss={avg:.5f})")

            # Periodic holdout visualisation
            if (getattr(args, 'viz_every', 0) > 0
                    and (ep + 1) % args.viz_every == 0
                    and holdout_shapes is not None):
                _save_viz(model, holdout_shapes, holdout_spec,
                          f"Holdout Shapes - Epoch {ep+1}", ep + 1, args, x1_clean.device)

    except (KeyboardInterrupt, TerminateInterrupt):
        print(f"\n  [!] Training interrupted at epoch {ep+1}. Saving emergency state...")
        ckpt_path = _save_checkpoint(model, opt, scheduler, ep, avg, args)
        print(f"  [!] Emergency checkpoint -> {ckpt_path}")
        if holdout_shapes is not None:
            _save_viz(model, holdout_shapes, holdout_spec,
                      f"Emergency Holdout - Epoch {ep+1}", ep + 1,
                      args, x1_clean.device, tag="emergency")
        print("  [!] Emergency save complete. Exiting safely.")
        sys.exit(0)



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
    if not hasattr(args, "run_str"):
        args.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")
    args.run_str = f"flow_matching_spectral_outline_{args.timestamp}_S{args.S}_nxi{args.nxi}_D{args.D}_C{args.copies_per_char}_flip{args.p_flip}"
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

    x1, train_shapes, holdout_shapes, holdout_spec = _build_dataset(
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
        if args.use_wandb:
            if not _WANDB_OK:
                print("  Warning: wandb not installed — run 'pip install wandb'. Skipping.")
            else:
                wandb.init(
                    project=args.wandb_project,
                    name=args.run_name if args.run_name else args.run_str,
                    config=vars(args),
                )
        train(model, x1, compute_spectral_cfm_loss, args,
              holdout_shapes=holdout_shapes, holdout_spec=holdout_spec)
        print("  Training complete!")
        if args.use_wandb and _WANDB_OK:
            wandb.finish()

        save_path = args.save_model
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        stem = os.path.splitext(save_path)[0]
        final_save_path = f"{stem}_{args.run_str}_final.pt"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'nxi': args.nxi, 'nd': args.nd, 'D': args.D, 'S': args.S,
            'n_lambda': args.n_lambda, 'predict_lambda': args.predict_lambda,
            'epochs': args.epochs, 'lr': args.lr,
            'p_drop': args.p_drop, 'cfg_weight': args.cfg_weight,
            'holdout_labels': list(HOLDOUT_LABELS),
        }, final_save_path)
        print(f"  Checkpoint saved -> {final_save_path}")

    # ── visualise training shapes (5 random) ──
    viz_train_keys = random.sample(list(train_shapes.keys()),
                                    min(5, len(train_shapes)))
    viz_train = {k: train_shapes[k] for k in viz_train_keys}
    viz_train_spec = {k: train_spec[k] for k in viz_train_keys}

    out_dir = os.path.join(_here, 'Trajectory_data_generator')
    visualise_set(
        model, viz_train, viz_train_spec,
        'Training Shapes (Spectral Cross-Attn)',
        os.path.join(out_dir, f"{args.run_str}_train.png"),
        args, device,
    )

    # ── visualise held-out shapes (ALL) ──
    visualise_set(
        model, holdout_shapes, holdout_spec,
        'HELD-OUT Shapes (Spectral Cross-Attn)',
        os.path.join(out_dir, f"{args.run_str}_holdout.png"),
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
    p.add_argument('--epochs',          type=int,   default=750,
                   help="Total epochs (750 epochs ≈ 36h on cluster)")
    p.add_argument('--lr',              type=float, default=1e-4,
                   help="AdamW learning rate. 1e-4 is safer than 3e-4 for 38M-param "
                        "cross-attention model; cosine LR decays to 1e-5.")
    p.add_argument('--mini_batch',      type=int,   default=512)
    p.add_argument('--copies_per_char', type=int,   default=100,
                   help="Clean copies per shape before on-the-fly augmentation. "
                        "100 → 29k samples / 56 batches per epoch.")
    p.add_argument('--noise_std',       type=float, default=0.015)
    # Augmentations
    p.add_argument('--p_flip',      type=float, default=0.2)
    p.add_argument('--rot_range',   type=float, default=20.0, help="Rotation in degrees")
    p.add_argument('--scale_range', type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument('--trans_range', type=float, default=0.08)
    # CFG: p_drop > 0 is required to train the null branch; cfg_weight > 1.0 to
    # extrapolate away from the null at inference. Both must be non-trivial for
    # CFG to do anything useful.
    p.add_argument('--p_drop',     type=float, default=0.1,
                   help="Dropout probability for CFG null-branch training (0 = CFG disabled)")
    p.add_argument('--cfg_weight', type=float, default=2.0,
                   help="CFG extrapolation strength at inference (1.0 = conditioned only)")
    # generation / visualisation
    p.add_argument('--n_gen',       type=int, default=5,
                   help="Number of generated samples per shape in output plots")
    p.add_argument('--steps',       type=int, default=100)
    p.add_argument('--bspline_pts', type=int, default=512)
    p.add_argument('--bspline_deg', type=int, default=5)
    # persistence
    p.add_argument('--save_model', type=str,
                   default=os.path.join(_here, 'checkpoints',
                                        'cond_spectral_crossattn.pt'))
    p.add_argument('--load_model', type=str, default=None,
                   help="Load a final checkpoint for visualization only")
    p.add_argument('--resume', type=str, default=None,
                   help="Path to a mid-training checkpoint to resume from "
                        "(saved by --save_every). Training continues from that epoch.")
    p.add_argument('--save_every', type=int, default=100,
                   help="Save a resumable checkpoint every N epochs (0 = disabled)")
    p.add_argument('--viz_every', type=int, default=250,
                   help="Generate and save holdout shapes visualization every N epochs")
    # W&B
    p.add_argument('--use_wandb',      action='store_true', default=False,
                   help="Enable Weights & Biases logging")
    p.add_argument('--wandb_project',  type=str, default='flow-matching',
                   help="W&B project name")
    p.add_argument('--run_name',       type=str, default=None,
                   help="W&B run name (auto-generated if omitted)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
