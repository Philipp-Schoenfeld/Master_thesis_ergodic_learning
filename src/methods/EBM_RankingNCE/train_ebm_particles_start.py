#!/usr/bin/env python3
"""
train_ebm_particles_start.py
=============================
Trainer for the Ranking-NCE energy-based policy (`ebm_ranking_nce_core.py`)
with particle-cloud conditioning plus a robot start point, on the real
ergodic-coverage dataset -- the second generative head in the methods
comparison (src/methods/), analogous to the particle-conditioned CFM runner
`thesis_architecture/flow_matching_runner_start.py`.

Dataset: `ergodic_dataset_start_plus.db` by default (override with --db). This
is the plain ground-truth-density variant (no injected "unknown region"): the
"unknown region" generator (`ergodic_dataset_generator/generate_dataset_unknown.py`)
has only ever been run against the non-start-conditioned `_improved`/`_improved_final`
lineage (`ergodic_dataset_improved_final.db` has 2344 '#unk' rows; none of the
'_start'/'_start_plus' DBs do), so a start-point-conditioned "randomly
occluded unknowns" dataset does not currently exist on disk -- using one
would mean running an unexecuted generator first and changes the task itself
(the SVGD ground truth gets a lawnmower sweep over the unknown region baked
in), which would no longer be the same task CFM is trained/compared on.

Reuses `sample_particles`/`augment_batch_torch`/`_load_shapes` verbatim from
`flow_matching_runner_start.py` (same project convention: each runner keeps
its own copy rather than importing another runner module with its own
argparse/wandb side effects) and the shared `checkpoint_rotation` /
`shape_library` utility modules.

Start-point convention (matches flow_matching_cond_particles_start.py): the
start point fed to the model is the first control point of the *augmented*
trajectory, not the DB's raw `x0` column -- the stored x0 would be stale
after random rotation/scale/translation.
"""

import argparse, json, math, os, random, signal, sqlite3, sys
from datetime import datetime

import numpy as np
import torch

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

_here = os.path.dirname(os.path.abspath(__file__))
_thesis_arch = os.path.join(_here, '..', '..', '..', 'thesis_architecture')
for _p in (_here, _thesis_arch, os.path.join(_thesis_arch, 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ebm_ranking_nce_core import EnergyNet, ranking_nce_loss
from shape_library import pdf_on_grid
from checkpoint_rotation import nach_zwischenstand, nach_endstand

_DB_PATH = os.path.join(_thesis_arch, 'ergodic_dataset_generator', 'ergodic_dataset_start_plus.db')


# ----------------------------------------------------------------------------
# Sampling and augmentation (verbatim from flow_matching_runner_start.py, so
# the EBM sees the identical particle-cloud / trajectory distribution a CFM
# run on the same DB would)
# ----------------------------------------------------------------------------

def augment_batch_torch(x, particles, p_flip=0.2, rot_range=20.0, scale_range=(0.75, 1.25),
                         trans_range=0.08, noise_std=0.01):
    """Synchronized vectorized geometric augmentation for trajectory AND particle cloud."""
    B = x.shape[0]
    device = x.device

    centroids = x.clone().mean(dim=1, keepdim=True)

    angles = (torch.rand(B, device=device) * 2 - 1) * rot_range * (math.pi / 180.0)
    c, s = torch.cos(angles), torch.sin(angles)
    R = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1)
    ], dim=1)

    scales = torch.empty(B, 1, 1, device=device).uniform_(*scale_range)
    trans = torch.empty(B, 1, 2, device=device).uniform_(-trans_range, trans_range)
    flip_mask = torch.rand(B, device=device) < p_flip

    def apply_transform(pts):
        out = pts.clone()
        out = torch.bmm((out - centroids), R.transpose(1, 2)) + centroids
        out = (out - centroids) * scales + centroids
        out = out + trans
        out[flip_mask, :, 0] = 1.0 - out[flip_mask, :, 0]
        return out

    x_out = apply_transform(x)
    x_out = x_out + torch.randn_like(x_out) * noise_std
    x_out = torch.clamp(x_out, 0.0, 1.0)

    part_coords = apply_transform(particles[:, :, :2])
    part_coords = part_coords + torch.randn_like(part_coords) * noise_std
    part_coords = torch.clamp(part_coords, 0.0, 1.0)

    part_out = torch.cat([part_coords, particles[:, :, 2:3]], dim=-1)
    return x_out, part_out


def sample_particles(density_grids, shape_indices, N, device, threshold=1e-5, mode='uniform'):
    """Samples N particles [x, y, mu] per batch item, vectorized over the batch."""
    B = shape_indices.shape[0]
    H, W = density_grids.shape[1], density_grids.shape[2]

    batch_grids = density_grids[shape_indices].to(device)
    batch_grids_flat = batch_grids.view(B, -1)

    if mode == 'uniform':
        weights = (batch_grids_flat > threshold).float()
    else:
        weights = batch_grids_flat + 1e-7

    idx = torch.multinomial(weights, num_samples=N, replacement=True)

    sy = idx // W
    sx = idx % W

    nx = (torch.rand(B, N, device=device) - 0.5) / (W - 1)
    ny = (torch.rand(B, N, device=device) - 0.5) / (H - 1)

    px = torch.clamp(sx.float() / (W - 1) + nx, 0.0, 1.0)
    py = torch.clamp(sy.float() / (H - 1) + ny, 0.0, 1.0)

    pmu = torch.gather(batch_grids_flat, 1, idx)

    return torch.stack([px, py, pmu], dim=-1)


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------

def _load_shapes(db_path, nxi, grid_res=128, max_rows=None):
    """max_rows truncates the query for local debugging only (e.g. on a
    memory-constrained laptop without a GPU); leave it at None for a real
    run so every row in the DB is used."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    query = ("SELECT trajectory, shape_name, split, density_params FROM ergodic_pairs "
              "WHERE split IN ('train', 'val') ORDER BY id ASC")
    if max_rows:
        query += f" LIMIT {int(max_rows)}"
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    train_shapes, val_shapes = {}, {}
    train_densities, val_densities = {}, {}

    print(f"  Precomputing {grid_res}x{grid_res} density grids for {len(rows)} rows...")
    for blob, label, split, density_params_str in rows:
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
        idx = np.linspace(0, len(xy) - 1, nxi).astype(int)

        params = json.loads(density_params_str)
        d_map, _, _ = pdf_on_grid(params, resolution=grid_res)
        if d_map.max() > 0:
            d_map /= d_map.max()

        if split == 'val':
            key = label if label not in val_shapes else f"{label}#{len(val_shapes)}"
            val_shapes[key] = xy[idx]
            val_densities[key] = d_map
        else:
            key = label if label not in train_shapes else f"{label}#{len(train_shapes)}"
            train_shapes[key] = xy[idx]
            train_densities[key] = d_map

    return train_shapes, val_shapes, train_densities, val_densities


def _build_dataset(shapes, densities, copies_per_char, device):
    all_x1, all_indices = [], []
    shape_keys = list(shapes.keys())
    density_grid_tensors = []

    for i, lbl in enumerate(shape_keys):
        density_grid_tensors.append(torch.tensor(densities[lbl], dtype=torch.float32))
        tiled = np.tile(shapes[lbl][None], (copies_per_char, 1, 1))
        all_x1.append(tiled)
        all_indices.extend([i] * copies_per_char)

    x1_np = np.concatenate(all_x1, axis=0)
    all_indices = np.array(all_indices)

    perm = np.random.permutation(len(x1_np))
    x1 = torch.tensor(x1_np[perm], dtype=torch.float32).to(device)
    shape_indices = torch.tensor(all_indices[perm], dtype=torch.long).to(device)
    density_grids_stack = torch.stack(density_grid_tensors).to(device)

    return x1, shape_indices, density_grids_stack


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------

def _save_checkpoint(model, optimizer, scheduler, epoch, loss, args, final=False):
    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
    stem = os.path.splitext(args.save_model)[0]
    tag = 'final' if final else f'ep{epoch + 1:04d}'
    path = f"{stem}_{args.run_str}_{tag}.pt"
    ckpt = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch, 'loss': loss,
        'nxi': args.nxi, 'nd': args.nd, 'D': args.D, 'n_particles': args.n_particles,
        'n_negatives': args.n_negatives, 'sigma_min': args.sigma_min, 'sigma_max': args.sigma_max,
        'energy_reg': args.energy_reg, 'sample_mode': args.sample_mode,
        'use_start_cond': True, 'db': args.db,
    }
    if not final:
        ckpt['optimizer_state_dict'] = optimizer.state_dict()
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if getattr(args, 'use_wandb', False) and _WANDB_OK and wandb.run is not None:
        ckpt['wandb_id'] = wandb.run.id
    torch.save(ckpt, path)
    if final:
        nach_endstand(stem, args.run_str, path)
    else:
        nach_zwischenstand(stem, args.run_str, path, behalten=getattr(args, 'keep_checkpoints', 1))
    return path


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train(model, x1_clean, shape_indices, density_grids_stack, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    model.train()

    device = x1_clean.device
    N = x1_clean.shape[0]
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed from epoch {start_epoch} (loss={ckpt['loss']:.5f})")

    class TerminateInterrupt(Exception):
        pass

    def _sigterm_handler(signum, frame):
        raise TerminateInterrupt()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    from tqdm import tqdm
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="ep")
    ep, avg = start_epoch, 0.0
    epoch_times = []

    try:
        for ep in pbar:
            t_ep0 = datetime.now()
            perm = torch.randperm(N, device=device)
            ep_loss, ep_acc, nb = 0.0, 0.0, 0

            for i in range(0, N, args.mini_batch):
                idx = perm[i:i + args.mini_batch]
                batch_clean = x1_clean[idx]
                batch_indices = shape_indices[idx]

                particles_clean = sample_particles(density_grids_stack, batch_indices,
                                                    args.n_particles, device, mode=args.sample_mode)
                batch_aug, particles_aug = augment_batch_torch(
                    batch_clean, particles_clean, noise_std=args.noise_std, p_flip=args.p_flip,
                    rot_range=args.rot_range, scale_range=args.scale_range,
                    trans_range=args.trans_range,
                )
                start = batch_aug[:, 0, :].detach()

                opt.zero_grad(set_to_none=True)
                loss, metrics = ranking_nce_loss(
                    model, particles_aug, batch_aug, k_indices=None,
                    n_negatives=args.n_negatives, sigma_min=args.sigma_min,
                    sigma_max=args.sigma_max, energy_reg=args.energy_reg, start=start,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                ep_loss += loss.item()
                ep_acc += metrics['acc']
                nb += 1

            scheduler.step()
            avg = ep_loss / max(nb, 1)
            avg_acc = ep_acc / max(nb, 1)
            lr_now = scheduler.get_last_lr()[0]
            epoch_times.append((datetime.now() - t_ep0).total_seconds())

            pbar.set_postfix(loss=f"{avg:.4f}", acc=f"{avg_acc:.3f}", lr=f"{lr_now:.2e}",
                              s_per_ep=f"{epoch_times[-1]:.1f}")

            if getattr(args, 'use_wandb', False) and _WANDB_OK:
                wandb.log({'train/loss': avg, 'train/acc': avg_acc, 'train/lr': lr_now,
                           'train/s_per_epoch': epoch_times[-1], 'epoch': ep + 1})

            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                _save_checkpoint(model, opt, scheduler, ep, avg, args)

    except (KeyboardInterrupt, TerminateInterrupt):
        print(f"\n  [!] Training interrupted at epoch {ep + 1}. Saving emergency checkpoint...")
        _save_checkpoint(model, opt, scheduler, ep, avg, args)
        sys.exit(0)

    if epoch_times:
        print(f"  Mean wall time/epoch: {sum(epoch_times) / len(epoch_times):.2f}s "
              f"(first={epoch_times[0]:.2f}s, last={epoch_times[-1]:.2f}s)")

    return ep, avg


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(args):
    if not hasattr(args, 'timestamp'):
        args.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")
    args.run_str = (f"ebm_rankingnce_particles_start_{args.timestamp}_nxi{args.nxi}_D{args.D}"
                     f"_N{args.n_particles}_C{args.copies_per_char}")
    if getattr(args, 'tag', None):
        args.run_str += f"_{args.tag}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"\n{'=' * 70}")
    print(f"  EBM Ranking-NCE (particles + start point)")
    print(f"  device={device}  nxi={args.nxi}  D={args.D}  n_particles={args.n_particles}")
    print(f"  db={args.db}")
    print(f"  epochs={args.epochs}  mini_batch={args.mini_batch}  n_negatives={args.n_negatives}")
    print(f"{'=' * 70}")

    train_shapes, val_shapes, train_densities, val_densities = _load_shapes(
        args.db, args.nxi, grid_res=128, max_rows=getattr(args, 'max_shapes', None))
    print(f"  Loaded {len(train_shapes)} train shapes and {len(val_shapes)} val shapes from DB")

    x1, shape_indices, density_grids_stack = _build_dataset(
        train_shapes, train_densities, copies_per_char=args.copies_per_char, device=device)
    print(f"  Training: {len(train_shapes)} shapes, {x1.shape[0]} samples/epoch, "
          f"{math.ceil(x1.shape[0] / args.mini_batch)} steps/epoch")

    model = EnergyNet(context_mode='particles', nxi=args.nxi, nd=args.nd, D=args.D,
                       use_sigma_cond=True, use_start_cond=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}\n")

    if args.load_model and os.path.isfile(args.load_model):
        ckpt = torch.load(args.load_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint from {args.load_model}")

    if args.use_wandb:
        if not _WANDB_OK:
            print("  Warning: wandb not installed. Skipping.")
        else:
            resume_id = None
            if args.resume and os.path.isfile(args.resume):
                try:
                    resume_id = torch.load(args.resume, map_location='cpu', weights_only=True).get('wandb_id')
                except Exception:
                    pass
            wandb.init(project=args.wandb_project, name=args.run_name or args.run_str,
                       config=vars(args), id=resume_id, resume="allow" if resume_id else None)

    last_epoch, last_loss = train(model, x1, shape_indices, density_grids_stack, args)

    print("  Training complete!")
    if args.use_wandb and _WANDB_OK:
        wandb.finish()

    final_path = _save_checkpoint(model, None, None, last_epoch, last_loss, args, final=True)
    print(f"  Checkpoint saved -> {final_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--db', type=str, default=_DB_PATH,
                   help="Path to the ergodic_pairs SQLite DB. Default is the 'full' "
                        "start-point DB (x0 sampled anywhere, plain ground-truth density).")
    p.add_argument('--max_shapes', type=int, default=None,
                   help="Debug only: truncate the DB query to this many rows "
                        "(e.g. for a memory-constrained local smoke test). "
                        "Leave unset for a real run.")
    p.add_argument('--nxi', type=int, default=25)
    p.add_argument('--nd', type=int, default=2)
    p.add_argument('--D', type=int, default=256)
    p.add_argument('--n_particles', type=int, default=256)
    p.add_argument('--sample_mode', type=str, default='uniform', choices=['uniform', 'density'])

    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--mini_batch', type=int, default=256)
    p.add_argument('--copies_per_char', type=int, default=100)

    p.add_argument('--n_negatives', type=int, default=8)
    p.add_argument('--sigma_min', type=float, default=0.01)
    p.add_argument('--sigma_max', type=float, default=0.5)
    p.add_argument('--energy_reg', type=float, default=1e-3)

    p.add_argument('--noise_std', type=float, default=0.015)
    p.add_argument('--p_flip', type=float, default=0.0)
    p.add_argument('--rot_range', type=float, default=20.0)
    p.add_argument('--scale_range', type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument('--trans_range', type=float, default=0.08)

    p.add_argument('--save_model', type=str, default=os.path.join(_here, 'checkpoints', 'ebm.pt'))
    p.add_argument('--load_model', type=str, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--keep_checkpoints', type=int, default=1)

    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--wandb_project', type=str, default='flow-matching')
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--tag', type=str, default=None,
                   help="Distinct marker for concurrent runs so checkpoints never collide.")

    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
