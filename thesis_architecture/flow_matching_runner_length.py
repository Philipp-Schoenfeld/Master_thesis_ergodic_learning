#!/usr/bin/env python3
"""
flow_matching_runner_particles.py
=================================
Flow-matching trainer for character/shape trajectories with PARTICLE conditioning.

Uses ParticleTokenizer + Cross-Attention.
Conditions on a set of N points sampled uniformly from the target density's support,
with their exact density value mu as a 3rd feature [x, y, mu].
"""

import argparse, os, random, sqlite3, sys, math, json
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
from flow_matching_cond_particles_length import (
    ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss,
    generate_particle_trajectories,
)

sys.path.append(os.path.join(_here, 'ergodic_dataset_generator'))
from shape_library import pdf_on_grid
from checkpoint_rotation import nach_zwischenstand, nach_endstand

# Der Startpunkt-Zweig hat eine eigene Datenbank: dort sind die Startpunkte
# gleichverteilt ueber die Flaeche gezogen statt in der linken unteren Ecke,
# und sie enthaelt zusaetzlich die flachen Formen (Sockel, weichgezeichnet,
# breite Moden, Konturen). Ueber --db laesst sich jede andere waehlen.
_DB_PATH = os.path.join(_here, 'ergodic_dataset_generator', 'ergodic_dataset_length.db')


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
# Sampling and Augmentation (GPU)
# ===========================================================================

def augment_batch_torch(x, particles, p_flip=0.2, rot_range=20.0, scale_range=(0.75, 1.25),
                        trans_range=0.08, noise_std=0.01):
    """
    Synchronized vectorized geometric augmentation for trajectory AND particle cloud.
    x: (B, nxi, 2)
    particles: (B, N, 3) where [:,:,:2] is (x,y) and [:,:,2] is mu
    """
    B = x.shape[0]
    device = x.device
    
    centroids = x.clone().mean(dim=1, keepdim=True)
    
    # Random parameters
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
    
    # Re-attach mu
    part_out = torch.cat([part_coords, particles[:, :, 2:3]], dim=-1)
    
    return x_out, part_out


def sample_particles(density_grids, shape_indices, N, device, threshold=1e-5, mode='uniform'):
    """
    Samples N particles vectorized over the batch.
    density_grids: (num_total_shapes, H, W)
    shape_indices: (B,) batch indices mapping to the density_grids
    """
    B = shape_indices.shape[0]
    H, W = density_grids.shape[1], density_grids.shape[2]
    
    batch_grids = density_grids[shape_indices].to(device) # (B, H, W)
    batch_grids_flat = batch_grids.view(B, -1)
    
    if mode == 'uniform':
        weights = (batch_grids_flat > threshold).float()
    else:
        weights = batch_grids_flat + 1e-7 # small epsilon
        
    idx = torch.multinomial(weights, num_samples=N, replacement=True)
    
    sy = idx // W
    sx = idx % W
    
    nx = (torch.rand(B, N, device=device) - 0.5) / (W - 1)
    ny = (torch.rand(B, N, device=device) - 0.5) / (H - 1)
    
    px = torch.clamp(sx.float() / (W - 1) + nx, 0.0, 1.0)
    py = torch.clamp(sy.float() / (H - 1) + ny, 0.0, 1.0)
    
    pmu = torch.gather(batch_grids_flat, 1, idx)
    
    particles = torch.stack([px, py, pmu], dim=-1)
    return particles


# ===========================================================================
# Dataset loader
# ===========================================================================

def _load_shapes(nxi, grid_res=128):
    conn = sqlite3.connect(_DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT trajectory, shape_name, split, density_params, length,"
                " n_iters FROM ergodic_pairs WHERE split IN ('train', 'val')"
                " ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    train_shapes = {}
    val_shapes = {}
    train_densities = {}
    val_densities = {}
    
    print(f"  Precomputing {grid_res}x{grid_res} density grids for all shapes (this takes a moment)...")
    train_lengths, val_lengths = {}, {}
    for blob, label, split, density_params_str, laenge, n_iters in rows:
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
        idx = np.linspace(0, len(xy) - 1, nxi).astype(int)
        
        params = json.loads(density_params_str)
        # We don't need gx, gy for this
        d_map, _, _ = pdf_on_grid(params, resolution=grid_res)
        # Normalize strictly to max=1.0 for consistency
        if d_map.max() > 0:
            d_map /= d_map.max()
            
        # Jede Form kommt mehrfach vor, einmal je Iterationszahl. Der
        # Schluessel muss die Varianten unterscheiden, sonst ueberschreiben
        # sie sich gegenseitig und uebrig bliebe je Form eine.
        #
        # Geschluesselt wird ueber `n_iters`, nicht ueber die gerundete
        # Laenge: zwei Varianten koennen auf zwei Nachkommastellen dieselbe
        # Laenge haben — bei neunzehn Checkpoints und einem Zuwachs von
        # zuletzt 8 % ist das keine Spitzfindigkeit. Der Verlust waere still:
        # eine Variante fiele weg, und niemand saehe es.
        if split == 'val':
            key = f"{label}#i{n_iters}"
            val_shapes[key] = xy[idx]
            val_densities[key] = d_map
            val_lengths[key] = float(laenge)
        else:
            key = f"{label}#i{n_iters}"
            train_shapes[key] = xy[idx]
            train_densities[key] = d_map
            train_lengths[key] = float(laenge)

    return (train_shapes, val_shapes, train_densities, val_densities,
            train_lengths, val_lengths)


def _build_dataset(train_shapes, train_densities, holdout_shapes, holdout_densities, copies_per_char, n_particles, device, sample_mode='uniform', train_lengths=None):
    all_x1 = []
    all_indices = []
    all_lengths = []
    
    shape_keys = list(train_shapes.keys())
    density_grid_tensors = []
    
    for i, lbl in enumerate(shape_keys):
        base = train_shapes[lbl]
        d_map = train_densities[lbl]
        density_grid_tensors.append(torch.tensor(d_map, dtype=torch.float32))
        
        tiled = np.tile(base[None], (copies_per_char, 1, 1))
        all_x1.append(tiled)
        all_indices.extend([i] * copies_per_char)
        L = (train_lengths or {}).get(lbl, 0.0)
        all_lengths.extend([L] * copies_per_char)

    x1_np = np.concatenate(all_x1, axis=0)
    all_indices = np.array(all_indices)
    
    perm = np.random.permutation(len(x1_np))
    x1 = torch.tensor(x1_np[perm], dtype=torch.float32).to(device)
    shape_indices = torch.tensor(all_indices[perm], dtype=torch.long).to(device)
    lengths = torch.tensor(np.array(all_lengths)[perm], dtype=torch.float32).to(device)
    
    density_grids_stack = torch.stack(density_grid_tensors).to(device) # (num_train_shapes, 128, 128)
    
    # Precompute sampled particles for holdout shapes for visualization
    holdout_particles = {}
    for lbl, d_map in holdout_densities.items():
        grid_t = torch.tensor(d_map, dtype=torch.float32).unsqueeze(0).to(device)
        idx_t = torch.tensor([0], dtype=torch.long).to(device)
        parts = sample_particles(grid_t, idx_t, n_particles, device=device, mode=sample_mode)
        holdout_particles[lbl] = parts[0]  # (N, 3)

    return x1, shape_indices, density_grids_stack, holdout_particles, lengths


# ===========================================================================
# Training
# ===========================================================================

def _save_checkpoint(model, optimizer, scheduler, epoch, loss, args):
    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
    stem = os.path.splitext(args.save_model)[0]
    path = f"{stem}_{args.run_str}_ep{epoch+1:04d}.pt"
    ckpt_dict = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss':  loss,
        'nxi':   args.nxi, 'nd': args.nd, 'D': args.D, 'n_particles': args.n_particles,
        'p_drop': args.p_drop, 'cfg_weight': args.cfg_weight,
        'sample_mode': args.sample_mode,
        'lambda_erg': getattr(args, 'lambda_erg', 0.0),
        'erg_K': getattr(args, 'erg_K', 0),
        'erg_pts': getattr(args, 'erg_pts', 0),
        'erg_t_power': getattr(args, 'erg_t_power', 0.0),
    }
    ckpt_dict['db'] = _DB_PATH
    ckpt_dict['length_cond'] = True
    ckpt_dict['log_ref'] = args.log_ref
    ckpt_dict['log_scale'] = args.log_scale
    ckpt_dict['p_drop_length'] = args.p_drop_length
    ckpt_dict['n_flat'] = getattr(args, 'n_flat', 0)
    ckpt_dict['start_cond'] = True
    if getattr(args, 'use_wandb', False) and _WANDB_OK and wandb.run is not None:
        ckpt_dict['wandb_id'] = wandb.run.id
    torch.save(ckpt_dict, path)
    nach_zwischenstand(stem, args.run_str, path,
                       behalten=getattr(args, "keep_checkpoints", 1))
    return path


def _alte_staende_entfernen(stem, args, neu):
    """Aeltere Checkpoints desselben Laufs loeschen.

    Regelmaessig speichern, aber den Speicher nicht mit Zwischenstaenden
    zumuellen: nach Abschluss oder Abbruch eines Laufs soll genau der
    aktuellste Stand uebrig sein.

    Die Reihenfolge ist der Punkt: der neue Stand ist zu diesem Zeitpunkt
    bereits geschrieben und wird geprueft, *bevor* irgendetwas geloescht wird.
    Andersherum waere ein fehlgeschlagener Schreibvorgang der Verlust des
    ganzen Trainings.

    Geloescht wird ausschliesslich, was zum selben Lauf gehoert — gleicher
    `run_str`, gleiches `_ep####.pt`-Muster. Checkpoints anderer Laeufe im
    selben Verzeichnis bleiben unangetastet, auch aeltere.
    """
    keep = max(1, int(getattr(args, 'keep_checkpoints', 1)))
    if not os.path.isfile(neu) or os.path.getsize(neu) < 1024:
        print(f"  [!] Neuer Checkpoint fehlt oder ist leer ({neu}) — "
              f"alte Staende bleiben stehen.")
        return
    import glob, re
    staende = []
    for f in glob.glob(f"{stem}_{args.run_str}_ep*.pt"):
        m = re.search(r'_ep(\d+)\.pt$', f)
        if m:
            staende.append((int(m.group(1)), f))
    staende.sort()
    for _, alt in staende[:-keep]:
        if os.path.abspath(alt) == os.path.abspath(neu):
            continue
        try:
            os.remove(alt)
            print(f"  alter Stand entfernt: {os.path.basename(alt)}")
        except OSError as e:
            print(f"  [!] konnte {alt} nicht entfernen: {e}")


def _save_viz(model, holdout_shapes, holdout_particles, holdout_densities, title, ep_num, args, device, tag=None):
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectory_data_generator')
    os.makedirs(out_dir, exist_ok=True)
    tag_str  = f"_{tag}" if tag else ""
    viz_path = os.path.join(out_dir, f"{args.run_str}{tag_str}_ep{ep_num:04d}.png")
    model.eval()
    with torch.no_grad():
        visualise_set(model, holdout_shapes, holdout_particles, holdout_densities, title, viz_path, args, device, max_cols=5)
    model.train()
    return viz_path


def train(model, x1_clean, shape_indices, density_grids_stack, loss_fn, args, lengths=None, holdout_shapes=None, holdout_particles=None, holdout_densities=None):
    opt       = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    model.train()

    ergodic = None
    if getattr(args, 'lambda_erg', 0.0) > 0.0:
        from ergodic_metric import ErgodicLoss
        ergodic = ErgodicLoss(
            nxi=args.nxi, K=args.erg_K, pts=args.erg_pts, deg=args.bspline_deg,
            weight=args.lambda_erg, t_power=args.erg_t_power,
            weighted_target=(args.sample_mode == 'uniform'),
            metric=args.erg_metric, sinkhorn_blur=args.sinkhorn_blur,
            sinkhorn_scaling=args.sinkhorn_scaling,
            sinkhorn_ratio=args.sinkhorn_ratio,
        ).to(x1_clean.device)
        print(f"  Ergodic loss term active: {ergodic.extra_repr()}")

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
            ep_parts    = {}

            for i in range(0, N, args.mini_batch):
                idx         = perm[i : i + args.mini_batch]
                batch_clean = x1_clean[idx]
                batch_indices = shape_indices[idx]

                # 1. Sample fresh particles from the base grid!
                particles_clean = sample_particles(density_grids_stack, batch_indices, args.n_particles, x1_clean.device, mode=args.sample_mode)

                # 2. Augment BOTH trajectory and particles synchronously
                batch_aug, particles_aug = augment_batch_torch(
                    batch_clean, particles_clean, noise_std=args.noise_std, p_flip=args.p_flip,
                    rot_range=args.rot_range, scale_range=args.scale_range,
                    trans_range=args.trans_range,
                )

                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type='cuda' if use_cuda else 'cpu',
                    dtype=torch.bfloat16,
                ):
                    loss, parts = loss_fn(
                        model, batch_aug, particles_aug,
                        length_batch=(None if lengths is None else lengths[idx]),
                        p_drop_length=args.p_drop_length,
                        p_drop=args.p_drop, ergodic=ergodic,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
                for k, v in parts.items():
                    ep_parts[k] = ep_parts.get(k, 0.0) + v.item()
                nb      += 1

            scheduler.step()
            avg    = ep_loss / max(nb, 1)
            avg_parts = {k: v / max(nb, 1) for k, v in ep_parts.items()}
            lr_now = scheduler.get_last_lr()[0]
            if ep % 10 == 0 or ep == args.epochs - 1:
                post = {'loss': f"{avg:.5f}", 'lr': f"{lr_now:.2e}"}
                if 'erg' in avg_parts:
                    post['cfm'] = f"{avg_parts['cfm']:.5f}"
                    post['erg'] = f"{avg_parts['erg']:.2e}"
                pbar.set_postfix(**post)

            if getattr(args, 'use_wandb', False) and _WANDB_OK:
                log = {'train/loss': avg, 'train/lr': lr_now, 'epoch': ep + 1}
                for k, v in avg_parts.items():
                    log[f'train/{k}'] = v
                if 'erg' in avg_parts:
                    # Contribution actually added to the total, for tuning lambda.
                    log['train/erg_weighted'] = ergodic.weight * avg_parts['erg']
                wandb.log(log)

            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                _save_checkpoint(model, opt, scheduler, ep, avg, args)

            if (getattr(args, 'viz_every', 0) > 0
                    and (ep + 1) % args.viz_every == 0
                    and holdout_shapes is not None):
                _save_viz(model, holdout_shapes, holdout_particles, holdout_densities,
                          f"Holdout Shapes - Epoch {ep+1}", ep + 1, args, x1_clean.device)

    except (KeyboardInterrupt, TerminateInterrupt):
        print(f"\n  [!] Training interrupted at epoch {ep+1}. Saving emergency state...")
        _save_checkpoint(model, opt, scheduler, ep, avg, args)
        if holdout_shapes is not None:
            _save_viz(model, holdout_shapes, holdout_particles, holdout_densities,
                      f"Emergency Holdout - Epoch {ep+1}", ep + 1,
                      args, x1_clean.device, tag="emergency")
        sys.exit(0)


# ===========================================================================
# Visualisation
# ===========================================================================

# ── Shared white-inferno colormap (module-level, built once) ─────────────
import matplotlib.colors as _mcolors

_inferno_colors = plt.colormaps['inferno'](np.linspace(0.0, 1.0, 256))
_n_white = 40
for _i in range(_n_white):
    _t = _i / _n_white
    _inferno_colors[_i] = (1 - _t) * np.array([1, 1, 1, 1]) + _t * _inferno_colors[_n_white]
WHITE_INFERNO = _mcolors.LinearSegmentedColormap.from_list('white_inferno', _inferno_colors)


def _draw_traj(ax, base, gen_cps, particles, density_grid, title, bspline_pts=512, bspline_deg=5):
    ax.set_facecolor('white')

    if density_grid is not None:
        d = density_grid.copy()
        if d.max() > 0:
            d /= d.max()
        ax.imshow(d, extent=[0, 1, 0, 1], origin='lower',
                  cmap=WHITE_INFERNO, vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)

    # Particle scatter — small dark dots
    if particles is not None:
        ax.scatter(particles[:, 0], particles[:, 1],
                   c='#444444', s=6, alpha=0.3, zorder=1, edgecolors='none')

    if len(base) >= 6:
        ax.plot(*cp_to_bspline(base, bspline_pts, bspline_deg).T,
                color='#1565C0', lw=2.5, label='Ground Truth', zorder=2)
        ax.scatter(base[:, 0], base[:, 1],
                   color='#1565C0', s=12, alpha=0.5, zorder=2)

    for i, cp in enumerate(gen_cps):
        main_alpha = 0.95 if i == 0 else 0.3
        if len(cp) >= 6:
            ax.plot(*cp_to_bspline(cp, bspline_pts, bspline_deg).T,
                    color='#00C853', lw=2.2, alpha=main_alpha,
                    label='Generated' if i == 0 else '', zorder=3)
        ax.scatter(cp[:, 0], cp[:, 1],
                   color='#00C853', s=8, alpha=max(0.1, main_alpha * 0.65), zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, color='#1A1A2E', pad=4)
    ax.set_xlabel('x', fontsize=7, color='#555')
    ax.set_ylabel('y', fontsize=7, color='#555')
    ax.tick_params(labelsize=6, colors='#555')
    for spine in ax.spines.values():
        spine.set_color('#ccc')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')


def visualise_set(model, shapes_dict, particles_dict, densities_dict, title_prefix, save_path,
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
        parts = particles_dict[lbl].cpu().numpy()
        d_map = densities_dict[lbl] if densities_dict else None

        # particles_dict[lbl] is (N, 3) — generate_particle_trajectories
        # handles the batch expansion internally
        cond_t = particles_dict[lbl]  # (N, 3)

        gen, lam = generate_particle_trajectories(
            model, cond_t,
            num_samples=args.n_gen, nxi=args.nxi, nd=args.nd,
            steps=args.steps, device=str(device),
            cfg_weight=args.cfg_weight,
        )
        gen = gen.cpu().numpy()
        _draw_traj(ax, base, gen, parts, d_map, f"'{lbl}'",
                   args.bspline_pts, args.bspline_deg)

        if lam is not None:
            lam_str = ", ".join(f"{v:.2f}" for v in lam[0].cpu().numpy())
            ax.text(0.02, 0.02, f"lam=[{lam_str}]",
                    transform=ax.transAxes, fontsize=5, alpha=0.5,
                    verticalalignment='bottom')

        if idx == 0:
            ax.legend(frameon=True, fontsize=7, loc='upper right',
                      facecolor='white', edgecolor='#ddd', framealpha=0.9)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved -> {save_path}")


# ===========================================================================
# Main
# ===========================================================================

def run(args):
    # Always generate a fresh timestamp; run_str is always fully rebuilt
    if not hasattr(args, 'timestamp'):
        args.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")
    args.run_str = f"flow_matching_particle_ergodic_{args.timestamp}_nxi{args.nxi}_D{args.D}_N{args.n_particles}_C{args.copies_per_char}_flip{args.p_flip}_START"
    # Wie viele der flachen Formen (Sockel, weichgezeichnet, breite Moden,
    # Konturen) im Datensatz stecken. Sie fuehren die Trainingsverteilung an
    # Glaubensdichten heran; ohne sie bleibt der Name unveraendert, damit
    # aeltere Laeufe weiter zum bisherigen Schema passen.
    try:
        _c = sqlite3.connect(_DB_PATH)
        args.n_flat = _c.execute(
            "SELECT count(*) FROM ergodic_pairs "
            "WHERE split IN ('train','val') AND shape_name LIKE 'flat_%'"
        ).fetchone()[0]
        _c.close()
    except Exception:
        args.n_flat = 0
    if args.n_flat > 0:
        args.run_str += f"_FLAT{args.n_flat}"
    # Eigener Marker je Lauf. Ohne ihn greifen zwei gleichzeitig laufende Jobs
    # mit demselben Glob auf die Checkpoints des jeweils anderen zu — der lange
    # Lauf wuerde vom Stand des kurzen fortsetzen.
    args.run_str += f"_LEN-pd{args.p_drop_length:g}"
    if getattr(args, 'tag', None):
        args.run_str += f"_{args.tag}"
    # Distinct marker for runs that use the ergodic coverage term in the loss, so
    # their checkpoints and figures never get mixed up with the pure-CFM runs.
    # Without the term the name is byte-identical to before.
    if getattr(args, 'lambda_erg', 0.0) > 0.0:
        if args.erg_metric == 'fourier':
            # Unchanged name, so older runs keep matching the old scheme.
            args.run_str += f"_ERGLOSS-w{args.lambda_erg:g}-K{args.erg_K}-tp{args.erg_t_power:g}"
        elif args.erg_metric == 'sinkhorn':
            args.run_str += (f"_ERGLOSS-SINKHORN-w{args.lambda_erg:g}"
                             f"-blur{args.sinkhorn_blur:g}-tp{args.erg_t_power:g}")
        else:
            args.run_str += (f"_ERGLOSS-BOTH-w{args.lambda_erg:g}-K{args.erg_K}"
                             f"-blur{args.sinkhorn_blur:g}-r{args.sinkhorn_ratio:g}"
                             f"-tp{args.erg_t_power:g}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"\n{'=' * 70}")
    print(f"  Particle Cross-Attention Flow Matching")
    print(f"  device={device}  nxi={args.nxi}  D={args.D}  n_particles={args.n_particles}")
    print(f"  epochs={args.epochs}  sample_mode={args.sample_mode}")
    print(f"  CFG: p_drop={args.p_drop}  cfg_weight={args.cfg_weight}")
    print(f"{'=' * 70}")

    (train_shapes, holdout_shapes, train_densities, holdout_densities,
     train_lengths, holdout_lengths) = _load_shapes(args.nxi, grid_res=128)
    # Normierung der Laenge aus dem Datensatz. Sie wandert ins Netz und in den
    # Checkpoint, damit die Inferenz spaeter dieselbe benutzt.
    _L = np.array(list(train_lengths.values()), dtype=np.float64)
    if args.log_ref is None:
        args.log_ref = float(np.median(_L)) if len(_L) else 5.0
    if args.log_scale is None:
        _u = np.log1p(_L) - np.log1p(args.log_ref)
        args.log_scale = float(max(np.std(_u), 1e-3)) if len(_L) else 1.0
    print(f"  Laenge: {_L.min():.2f} bis {_L.max():.2f}  "
          f"log_ref={args.log_ref:.2f}  log_scale={args.log_scale:.3f}")
    print(f"  Loaded {len(train_shapes)} train shapes and {len(holdout_shapes)} val shapes from DB")

    x1, shape_indices, density_grids_stack, holdout_particles, lengths = _build_dataset(
        train_shapes, train_densities, holdout_shapes, holdout_densities,
        copies_per_char=args.copies_per_char, n_particles=args.n_particles,
        device=device, sample_mode=args.sample_mode,
    )

    print(f"  Training: {len(train_shapes)} shapes, {x1.shape[0]} samples")

    model = ParticleCrossAttnFlowNetwork(
        nxi=args.nxi, nd=args.nd, D=args.D,
        log_ref=args.log_ref, log_scale=args.log_scale,
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
                resume_id = None
                if args.resume and os.path.isfile(args.resume):
                    try:
                        ckpt_meta = torch.load(args.resume, map_location='cpu', weights_only=True)
                        resume_id = ckpt_meta.get('wandb_id', None)
                    except Exception:
                        pass

                wandb.init(
                    project=args.wandb_project,
                    name=args.run_name if args.run_name else args.run_str,
                    config=vars(args),
                    id=resume_id,
                    resume="allow" if resume_id else None
                )
        train(model, x1, shape_indices, density_grids_stack, compute_particle_cfm_loss, args,
              lengths=lengths,
              holdout_shapes=holdout_shapes, holdout_particles=holdout_particles,
              holdout_densities=holdout_densities)
        print("  Training complete!")
        if args.use_wandb and _WANDB_OK:
            wandb.finish()

        save_path = args.save_model
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        stem = os.path.splitext(save_path)[0]
        final_save_path = f"{stem}_{args.run_str}_final.pt"
        
        # Die Normierung der Laenge MUSS mit. Ohne `log_ref` und `log_scale`
        # kann die Inferenz nicht rekonstruieren, welche Zahl im Training
        # welcher Bahnlaenge entsprach — das Netz bekaeme dann eine Vorgabe
        # auf einer anderen Skala und ignorierte sie stillschweigend.
        torch.save({
            'model_state_dict': model.state_dict(),
            'nxi': args.nxi, 'nd': args.nd, 'D': args.D, 'n_particles': args.n_particles,
            'epochs': args.epochs, 'lr': args.lr, 'sample_mode': args.sample_mode,
            'epoch': args.epochs - 1,
            'length_cond': True,
            'log_ref': args.log_ref, 'log_scale': args.log_scale,
            'p_drop_length': args.p_drop_length,
            'cfg_weight': args.cfg_weight, 'p_drop': args.p_drop,
            'db': _DB_PATH,
        }, final_save_path)
        print(f"  Checkpoint saved -> {final_save_path}")
        # Nach dem Endstand bleibt nur das `_final`: die Zwischenstaende
        # desselben Laufs werden entfernt. Bricht der Job vorher ab,
        # bleibt statt dessen der letzte Zwischenstand liegen.
        nach_endstand(stem, args.run_str, final_save_path)

    # Visualise Training sample
    viz_train_keys = random.sample(list(train_shapes.keys()), min(5, len(train_shapes)))
    viz_train = {k: train_shapes[k] for k in viz_train_keys}
    viz_train_particles = {}
    for lbl in viz_train_keys:
        d_map = train_densities[lbl]
        grid_t = torch.tensor(d_map, dtype=torch.float32).unsqueeze(0).to(device)
        idx_t = torch.tensor([0], dtype=torch.long).to(device)
        parts = sample_particles(grid_t, idx_t, args.n_particles, device=device, mode=args.sample_mode)
        viz_train_particles[lbl] = parts[0]  # (N, 3)
        
    out_dir = os.path.join(_here, 'Trajectory_data_generator')
    visualise_set(
        model, viz_train, viz_train_particles, train_densities,
        f'Training Shapes ({args.n_particles} Particles)',
        os.path.join(out_dir, f"{args.run_str}_train.png"),
        args, device,
    )

    visualise_set(
        model, holdout_shapes, holdout_particles, holdout_densities,
        f'HELD-OUT Shapes ({args.n_particles} Particles)',
        os.path.join(out_dir, f"{args.run_str}_holdout.png"),
        args, device,
    )


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--nxi', type=int, default=64,
                   help='Kontrollpunkte. Hoeher als die 25 des Startpunkt-Laufs: die langen Pfade aus 10000 Iterationen verlieren sonst ihre Faltungen.')
    p.add_argument('--nd',  type=int, default=2)
    p.add_argument('--D',   type=int, default=256)
    p.add_argument('--n_particles', type=int, default=256)
    
    p.add_argument('--sample_mode', type=str, default='uniform', choices=['uniform', 'density'])
    
    p.add_argument('--epochs',          type=int,   default=500)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--mini_batch',      type=int,   default=64)
    p.add_argument('--copies_per_char', type=int,   default=100)
    p.add_argument('--noise_std',       type=float, default=0.015)
    
    p.add_argument('--p_flip',      type=float, default=0.0)
    p.add_argument('--rot_range',   type=float, default=20.0)
    p.add_argument('--scale_range', type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument('--trans_range', type=float, default=0.08)
    
    p.add_argument('--p_drop',     type=float, default=0.1)
    p.add_argument('--cfg_weight', type=float, default=2.0)

    # Ergodic coverage term in the loss. Default 0.0 = off, i.e. exactly the
    # pure flow-matching behaviour of previous runs.
    p.add_argument('--lambda_erg',  type=float, default=0.0,
                   help='Weight of the ergodic coverage term. 0 disables it.')
    p.add_argument('--erg_K',       type=int,   default=8,
                   help='Frequency grid is erg_K x erg_K (K^2 modes).')
    p.add_argument('--erg_pts',     type=int,   default=128,
                   help='B-spline samples used for the trajectory time average.')
    p.add_argument('--erg_t_power', type=float, default=2.0,
                   help='Ramp exponent t^p; the endpoint estimate is poor at small t.')
    # Which discrepancy measure the ergodic term uses. 'fourier' is the previous
    # behaviour byte for byte; 'sinkhorn' drops the K x K basis truncation.
    p.add_argument('--erg_metric', type=str, default='fourier',
                   choices=['fourier', 'sinkhorn', 'both'],
                   help='Discrepancy measure for the coverage term.')
    p.add_argument('--sinkhorn_blur', type=float, default=0.05,
                   help='Entropic length scale in domain widths (sinkhorn only).')
    p.add_argument('--sinkhorn_scaling', type=float, default=0.5,
                   help='Epsilon-annealing density; 0.5 is 6x cheaper than '
                        'geomloss default 0.9 at the same gradient.')
    p.add_argument('--sinkhorn_ratio', type=float, default=1.0,
                   help="Factor on the Sinkhorn part when --erg_metric both.")
    
    p.add_argument('--n_gen',       type=int, default=5)
    p.add_argument('--steps',       type=int, default=100)
    p.add_argument('--bspline_pts', type=int, default=512)
    p.add_argument('--bspline_deg', type=int, default=5)
    
    p.add_argument('--save_model', type=str,
                   default=os.path.join(_here, 'checkpoints', 'cond_particles_crossattn.pt'))
    p.add_argument('--load_model', type=str, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save_every', type=int, default=100)
    p.add_argument('--viz_every', type=int, default=250)
    
    p.add_argument('--use_wandb',      action='store_true', default=False)
    p.add_argument('--wandb_project',  type=str, default='flow-matching')
    p.add_argument('--run_name',       type=str, default=None)
    p.add_argument('--p_drop_length', type=float, default=0.1,
                   help='Anteil der Trainingsbeispiele, bei denen die Laenge '
                        'durch den Null-Token ersetzt wird. Erst dadurch '
                        'entsteht der unkonditionierte Zweig, den die '
                        'klassifikatorfreie Fuehrung zur Inferenz braucht.')
    p.add_argument('--log_ref', type=float, default=None,
                   help='Bezugslaenge der Log-Normierung. Ohne Angabe der '
                        'Median des Datensatzes.')
    p.add_argument('--log_scale', type=float, default=None,
                   help='Skala der Log-Normierung. Ohne Angabe die '
                        'Standardabweichung der normierten Laengen.')
    p.add_argument('--tag', type=str, default=None,
                   help='Kurzer Marker, der an den Laufnamen angehaengt wird, '
                        'damit sich parallele Laeufe nicht gegenseitig '
                        'fortsetzen (etwa K200 oder L500).')
    p.add_argument('--keep_checkpoints', type=int, default=1,
                   help='Wie viele Checkpoints eines Laufs behalten werden. '
                        'Vorgabe 1: beim Speichern eines neuen Standes werden '
                        'aeltere desselben Laufs entfernt.')
    p.add_argument('--db', type=str, default=None,
                   help='Pfad zur Datensatz-Datenbank. Vorgabe ist '
                        'ergodic_dataset_start.db im Generator-Verzeichnis.')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if getattr(args, 'db', None):
        _DB_PATH = args.db if os.path.isabs(args.db) else os.path.join(
            _here, 'ergodic_dataset_generator', args.db)
    if not os.path.exists(_DB_PATH):
        raise SystemExit(f'Datenbank nicht gefunden: {_DB_PATH}')
    print(f'  Datensatz: {_DB_PATH}')
    run(args)
