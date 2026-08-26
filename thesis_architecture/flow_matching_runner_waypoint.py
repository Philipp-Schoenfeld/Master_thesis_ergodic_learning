#!/usr/bin/env python3
"""
flow_matching_runner_waypoint.py
================================
Flow-matching trainer for character/shape trajectories with WAYPOINT conditioning.

Uses WaypointTokenizer + Cross-Attention instead of SpectralTokenizer.
Conditions directly on the trajectory coordinates (waypoints).

Usage:
  python flow_matching_runner_waypoint.py                                                    # train
  python flow_matching_runner_waypoint.py --load_model checkpoints/cond_waypoint_crossattn.pt  # viz
"""

import argparse, os, random, sqlite3, sys, math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from checkpoint_rotation import nach_zwischenstand, nach_endstand

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
from flow_matching_cond_waypoint_crossattn import (
    WaypointCrossAttnFlowNetwork, compute_waypoint_cfm_loss,
    generate_waypoint_trajectories,
)

_DB_PATH = os.path.join(_here, 'Trajectory_data_generator', 'character_trajectories.db')

HOLDOUT_LABELS = {
    'G', 'W', '5', 'sigma', 'phi',            
    'star_5_ir4', 'spiral_2cw', 'lissajous_1_3_d2',   
    'heart', 'rand_poly_7',                     
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
# On-the-fly augmentation (GPU)
# ===========================================================================

def augment_batch_torch(x, p_flip=0.2, rot_range=20.0, scale_range=(0.75, 1.25),
                        trans_range=0.08, noise_std=0.01):
    B, nxi, _ = x.shape
    device = x.device
    out = x.clone()
    centroids = out.mean(dim=1, keepdim=True)
    
    angles = (torch.rand(B, device=device) * 2 - 1) * rot_range * (math.pi / 180.0)
    c, s = torch.cos(angles), torch.sin(angles)
    R = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1)
    ], dim=1)
    
    out = torch.bmm((out - centroids), R.transpose(1, 2)) + centroids
    
    scales = torch.empty(B, 1, 1, device=device).uniform_(*scale_range)
    out = (out - centroids) * scales + centroids
    
    trans = torch.empty(B, 1, 2, device=device).uniform_(-trans_range, trans_range)
    out = out + trans
    
    flip_mask = torch.rand(B, device=device) < p_flip
    out[flip_mask, :, 0] = 1.0 - out[flip_mask, :, 0]
    
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


def _build_dataset(shapes, holdout_labels, copies_per_char, device):
    train_shapes, holdout_shapes = {}, {}
    for lbl, base in shapes.items():
        base_lbl = lbl.split('#')[0]
        if base_lbl in holdout_labels:
            holdout_shapes[lbl] = base
        else:
            train_shapes[lbl] = base

    all_x1 = []
    for lbl, base in train_shapes.items():
        tiled = np.tile(base[None], (copies_per_char, 1, 1))         
        all_x1.append(tiled)

    x1_np = np.concatenate(all_x1, axis=0)
    perm  = np.random.permutation(len(x1_np))
    x1    = torch.tensor(x1_np[perm], dtype=torch.float32).to(device)

    return x1, train_shapes, holdout_shapes


# ===========================================================================
# Training
# ===========================================================================

def _save_checkpoint(model, optimizer, scheduler, epoch, loss, args):
    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
    stem = os.path.splitext(args.save_model)[0]
    path = f"{stem}_{args.run_str}_ep{epoch+1:04d}.pt"
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss':  loss,
        'nxi':   args.nxi, 'nd': args.nd, 'D': args.D,
        'n_lambda': args.n_lambda, 'predict_lambda': args.predict_lambda,
        'p_drop': args.p_drop, 'cfg_weight': args.cfg_weight,
    }, path)
    return path


def _save_viz(model, holdout_shapes, title, ep_num, args, device, tag=None):
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectory_data_generator')
    os.makedirs(out_dir, exist_ok=True)
    run_str  = f"nxi{args.nxi}_D{args.D}_flip{args.p_flip}"
    tag_str  = f"_{tag}" if tag else ""
    viz_path = os.path.join(out_dir, f'viz_holdout_waypoint_{run_str}{tag_str}_ep{ep_num:04d}.png')
    model.eval()
    with torch.no_grad():
        visualise_set(model, holdout_shapes, title, viz_path, args, device, max_cols=5)
    model.train()
    return viz_path


def train(model, x1_clean, loss_fn, args, holdout_shapes=None):
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    model.train()
    use_cuda    = x1_clean.device.type == 'cuda'
    N           = x1_clean.shape[0]
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=x1_clean.device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed from epoch {start_epoch} (loss={ckpt['loss']:.5f})")

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

                batch_aug = augment_batch_torch(
                    batch_clean, noise_std=args.noise_std, p_flip=args.p_flip,
                    rot_range=args.rot_range, scale_range=args.scale_range,
                    trans_range=args.trans_range,
                )

                waypoint_cond = batch_aug

                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type='cuda' if use_cuda else 'cpu',
                    dtype=torch.bfloat16,
                ):
                    loss = loss_fn(
                        model, batch_aug, waypoint_cond,
                        p_drop=args.p_drop,
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

            if getattr(args, 'use_wandb', False) and _WANDB_OK:
                wandb.log({'train/loss': avg, 'train/lr': lr_now, 'epoch': ep + 1})

            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                _save_checkpoint(model, opt, scheduler, ep, avg, args)
                print(f"  [ep {ep+1}] checkpoint saved (loss={avg:.5f})")

            if (getattr(args, 'viz_every', 0) > 0
                    and (ep + 1) % args.viz_every == 0
                    and holdout_shapes is not None):
                _save_viz(model, holdout_shapes, 
                          f"Holdout Shapes - Epoch {ep+1}", ep + 1, args, x1_clean.device)

    except (KeyboardInterrupt, TerminateInterrupt):
        print(f"\n  [!] Training interrupted at epoch {ep+1}. Saving emergency state...")
        ckpt_path = _save_checkpoint(model, opt, scheduler, ep, avg, args)
        print(f"  [!] Emergency checkpoint -> {ckpt_path}")
        if holdout_shapes is not None:
            _save_viz(model, holdout_shapes,
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


def visualise_set(model, shapes_dict, title_prefix, save_path,
                  args, device, max_cols=5):
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
        
        cond_t = torch.tensor(base, dtype=torch.float32)

        gen, lam = generate_waypoint_trajectories(
            model, cond_t,
            num_samples=args.n_gen, nxi=args.nxi, nd=args.nd,
            steps=args.steps, device=str(device),
            cfg_weight=args.cfg_weight,
        )
        gen = gen.cpu().numpy()
        _draw_traj(ax, base, gen, f"'{lbl}'",
                   args.bspline_pts, args.bspline_deg)

        if lam is not None:
            lam_str = ", ".join(f"{v:.2f}" for v in lam[0].cpu().numpy())
            ax.text(0.02, 0.02, f"lam=[{lam_str}]",
                    transform=ax.transAxes, fontsize=5, alpha=0.5,
                    verticalalignment='bottom')

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
    args.run_str = f"flow_matching_waypoint_{args.timestamp}_nxi{args.nxi}_D{args.D}_flip{args.p_flip}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"\n{'=' * 70}")
    print(f"  Waypoint Cross-Attention Flow Matching")
    print(f"  device={device}  nxi={args.nxi}  D={args.D}")
    print(f"  epochs={args.epochs}  predict_lambda={args.predict_lambda}")
    print(f"  CFG: p_drop={args.p_drop}  cfg_weight={args.cfg_weight}")
    print(f"{'=' * 70}")

    all_shapes = _load_shapes(args.nxi)
    print(f"  Loaded {len(all_shapes)} shapes from DB")

    x1, train_shapes, holdout_shapes = _build_dataset(
        all_shapes, HOLDOUT_LABELS,
        copies_per_char=args.copies_per_char,
        device=device,
    )

    print(f"  Training: {len(train_shapes)} shapes, {x1.shape[0]} samples")
    print(f"  Held out: {len(holdout_shapes)} shapes "
          f"({', '.join(sorted(holdout_shapes.keys()))})")

    model = WaypointCrossAttnFlowNetwork(
        nxi=args.nxi, nd=args.nd, D=args.D,
        n_lambda=args.n_lambda, predict_lambda=args.predict_lambda,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {params:,}\n")

    if args.load_model and os.path.isfile(args.load_model):
        ckpt = torch.load(args.load_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint from {args.load_model}")
    else:
        if args.use_wandb:
            if not _WANDB_OK:
                print("  Warning: wandb not installed. Skipping.")
            else:
                wandb.init(
                    project=args.wandb_project,
                    name=args.run_name if args.run_name else args.run_str,
                    config=vars(args),
                )
        train(model, x1, compute_waypoint_cfm_loss, args,
              holdout_shapes=holdout_shapes)
        print("  Training complete!")
        if args.use_wandb and _WANDB_OK:
            wandb.finish()

        save_path = args.save_model
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        stem = os.path.splitext(save_path)[0]
        final_save_path = f"{stem}_{args.run_str}_final.pt"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'nxi': args.nxi, 'nd': args.nd, 'D': args.D,
            'n_lambda': args.n_lambda, 'predict_lambda': args.predict_lambda,
            'epochs': args.epochs, 'lr': args.lr,
            'p_drop': args.p_drop, 'cfg_weight': args.cfg_weight,
            'holdout_labels': list(HOLDOUT_LABELS),
        }, final_save_path)
        print(f"  Checkpoint saved -> {final_save_path}")
        # Nach dem Endstand bleibt nur das `_final`: die Zwischenstaende
        # desselben Laufs werden entfernt. Bricht der Job vorher ab,
        # bleibt statt dessen der letzte Zwischenstand liegen.
        nach_endstand(stem, args.run_str, final_save_path)

    viz_train_keys = random.sample(list(train_shapes.keys()),
                                    min(5, len(train_shapes)))
    viz_train = {k: train_shapes[k] for k in viz_train_keys}

    out_dir = os.path.join(_here, 'Trajectory_data_generator')
    visualise_set(
        model, viz_train,
        'Training Shapes (Waypoint Cross-Attn)',
        os.path.join(out_dir, f"{args.run_str}_train.png"),
        args, device,
    )

    visualise_set(
        model, holdout_shapes,
        'HELD-OUT Shapes (Waypoint Cross-Attn)',
        os.path.join(out_dir, f"{args.run_str}_holdout.png"),
        args, device,
    )


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--nxi', type=int, default=20)
    p.add_argument('--nd',  type=int, default=2)
    p.add_argument('--D',   type=int, default=256)
    
    p.add_argument('--n_lambda', type=int, default=6)
    p.add_argument('--predict_lambda', action='store_true', default=False)
    p.add_argument('--no_predict_lambda', dest='predict_lambda', action='store_false')
    
    p.add_argument('--epochs',          type=int,   default=750)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--mini_batch',      type=int,   default=512)
    p.add_argument('--copies_per_char', type=int,   default=20)
    p.add_argument('--noise_std',       type=float, default=0.015)
    
    p.add_argument('--p_flip',      type=float, default=0.0)
    p.add_argument('--rot_range',   type=float, default=20.0)
    p.add_argument('--scale_range', type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument('--trans_range', type=float, default=0.08)
    
    p.add_argument('--p_drop',     type=float, default=0.1)
    p.add_argument('--cfg_weight', type=float, default=2.0)
    
    p.add_argument('--n_gen',       type=int, default=5)
    p.add_argument('--steps',       type=int, default=100)
    p.add_argument('--bspline_pts', type=int, default=512)
    p.add_argument('--bspline_deg', type=int, default=5)
    
    p.add_argument('--keep_checkpoints', type=int, default=1,
    
                   help='Wie viele Zwischenstaende eines Laufs behalten werden. '
    
                        'Vorgabe 1: beim Speichern eines neuen Standes werden '
    
                        'aeltere entfernt, und nach dem Endstand alle.')
    
    p.add_argument('--save_model', type=str,
                   default=os.path.join(_here, 'checkpoints',
                                        'cond_waypoint_crossattn.pt'))
    p.add_argument('--load_model', type=str, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save_every', type=int, default=100)
    p.add_argument('--viz_every', type=int, default=250)
    
    p.add_argument('--use_wandb',      action='store_true', default=False)
    p.add_argument('--wandb_project',  type=str, default='flow-matching')
    p.add_argument('--run_name',       type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
