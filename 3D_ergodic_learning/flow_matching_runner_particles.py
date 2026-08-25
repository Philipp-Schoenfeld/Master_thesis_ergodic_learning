#!/usr/bin/env python3
r"""
flow_matching_runner_particles.py  —  3D port
=============================================
Flow-matching trainer for 3D shape trajectories with PARTICLE conditioning.

Same structure as the 2D runner: AdamW + CosineAnnealingLR, bfloat16 autocast,
CFG dropout, resume from the latest checkpoint, SIGTERM handler for the cluster
time limit, W&B logging, periodic holdout visualisation.

Differences that are not just "one more axis":

* Density is a **volume** (R^3), so the default grid resolution drops from 128
  to 64. At 128^3 a single float32 volume is 8 MB and 750 of them are 6 GB —
  the stack simply does not fit. 64^3 is 1 MB per shape.
* The volumes are therefore kept on the **CPU** and the per-batch slice is moved
  to the GPU inside `sample_particles`, instead of parking the whole stack in
  VRAM as the 2D runner does.
* Augmentation rotates about the plane normal by default (`--rot_full` for
  general SO(3)) so planar data stays planar and results stay comparable to 2D.
* `--erg_K` defaults to 6 rather than 8: the ergodic loss term costs K^3 modes
  here instead of K^2, so K=8 would mean 512 modes evaluated on every curve
  sample of every batch.
"""

import argparse, os, random, sys, math
from datetime import datetime
import numpy as np
import torch

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_3d import (load_pairs, prepare_targets, sample_particles,
                     augment_batch, DEFAULT_DB, Z_PLANE, Z_SIGMA)
from flow_matching_cond_particles_crossattn import (
    ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss,
    generate_particle_trajectories, ND,
)
from orientation_energy import (W_POINT, W_STANDOFF, W_ANGSMOOTH,
                               STANDOFF_TARGET, STANDOFF_BAND)
from orientation import (SurfaceField, frames_for_curve, matrix_to_rot6d,
                         rot6d_to_matrix)
import viz_3d


def orientation_targets(traj, labels, volumes_np, index, mode, device):
    """Stufe 0 supplies the labels that Stufe 1 needs.

    There is no orientation ground truth anywhere in the database, so the
    supervised branch has to manufacture it. The look-at frame derived from the
    stored trajectory and the target's surface field is exactly that: the best
    orientation obtainable *without* learning. A network trained on it can at
    most match Stufe 0 — which is why the self-supervised branch, where the
    orientation is learned from the objective instead, is the more interesting
    of the two. This exists to make the comparison possible.

    Returns {label: (nxi, 6) float32}.
    """
    out = {}
    for lbl in labels:
        cps = torch.tensor(traj[lbl], dtype=torch.float32, device=device)[None]
        field = SurfaceField(volumes_np[index[lbl]], device=device)
        R = frames_for_curve(cps, field, mode=mode)
        out[lbl] = matrix_to_rot6d(R)[0].cpu().numpy()
    return out


# ===========================================================================
# Checkpointing and visualisation
# ===========================================================================

def _save_checkpoint(model, optimizer, scheduler, epoch, loss, args):
    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
    stem = os.path.splitext(args.save_model)[0]
    path = f"{stem}_{args.run_str}_ep{epoch+1:04d}.pt"
    ckpt = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss':  loss,
        'nxi': args.nxi, 'nd': args.nd, 'D': args.D,
        'n_particles': args.n_particles,
        'p_drop': args.p_drop, 'cfg_weight': args.cfg_weight,
        'sample_mode': args.sample_mode,
        'grid_res': args.grid_res, 'z_plane': args.z_plane, 'z_sigma': args.z_sigma,
        'rot_full': args.rot_full,
        'orientation': args.orientation, 'frame_mode': args.frame_mode,
        'lambda_erg': args.lambda_erg, 'erg_K': args.erg_K,
        'erg_on': args.erg_on, 'lambda_ori': args.lambda_ori,
        'w_cfm_rot': args.w_cfm_rot, 'w_point': args.w_point,
        'w_standoff': args.w_standoff, 'w_angsmooth': args.w_angsmooth,
        'standoff_target': args.standoff_target,
        'standoff_band': args.standoff_band, 'mu_thresh': args.mu_thresh,
        'erg_pts': args.erg_pts, 'erg_t_power': args.erg_t_power,
        'dim': 3,
    }
    if getattr(args, 'use_wandb', False) and _WANDB_OK and wandb.run is not None:
        ckpt['wandb_id'] = wandb.run.id
    ckpt['db3d'] = getattr(args, 'db3d', None)
    ckpt['surfaces'] = getattr(args, 'surfaces', None)
    torch.save(ckpt, path)
    _alte_staende_entfernen(stem, args, path)
    return path


def _alte_staende_entfernen(stem, args, neu):
    """Aeltere Checkpoints desselben Laufs loeschen.

    Ein 3D-Checkpoint dieses Projekts wiegt rund ein Gigabyte; ein Lauf ueber
    500 Epochen mit einem Speicherabstand von 20 hinterliesse sonst 25 GB. Nach
    Abschluss oder Abbruch soll genau der aktuellste Stand uebrig sein.

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
    muster = f"{stem}_{args.run_str}_ep*.pt"
    staende = []
    for f in glob.glob(muster):
        m = re.search(r'_ep(\d+)\.pt$', f)
        if m:
            staende.append((int(m.group(1)), f))
    staende.sort()
    for _, alt in staende[:-keep]:
        if os.path.abspath(alt) == os.path.abspath(neu):
            continue
        try:
            os.remove(alt)
        except OSError as e:
            print(f"  [!] konnte {alt} nicht entfernen: {e}")


@torch.no_grad()
def visualise_set(model, labels, trajectories, particles_map, volumes_map,
                  title, save_path, args, device, max_cols=5):
    model.eval()
    panels = []
    from obstacles import basis_torch
    viz_basis = basis_torch(args.nxi, args.bspline_pts, args.bspline_deg, device)
    for lbl in labels:
        cond = particles_map[lbl]
        gen, rot6d = generate_particle_trajectories(
            model, cond, num_samples=args.n_gen, nxi=args.nxi, nd=args.nd,
            steps=args.steps, device=str(device), cfg_weight=args.cfg_weight,
        )
        gen_R = None
        if rot6d is not None:
            r = torch.einsum('ti,ic->tc', viz_basis, rot6d[0])
            gen_R = rot6d_to_matrix(r).cpu().numpy()
        panels.append(dict(
            base=trajectories[lbl],
            gen_cps=gen.cpu().numpy(),
            particles=cond.cpu().numpy(),
            volume=volumes_map.get(lbl),
            title=f"'{lbl}'",
            bspline_pts=args.bspline_pts,
            bspline_deg=args.bspline_deg,
            gen_R=gen_R,
        ))
    viz_3d.save_grid(panels, save_path, title, max_cols=max_cols)
    model.train()


def _save_viz(model, labels, trajectories, particles_map, volumes_map,
              title, ep_num, args, device, tag=None):
    out_dir = os.path.join(_here, 'viz')
    os.makedirs(out_dir, exist_ok=True)
    tag_str = f"_{tag}" if tag else ""
    path = os.path.join(out_dir, f"{args.run_str}{tag_str}_ep{ep_num:04d}.png")
    visualise_set(model, labels, trajectories, particles_map, volumes_map,
                  title, path, args, device)
    return path


# ===========================================================================
# Training
# ===========================================================================

def train(model, x1_clean, shape_indices, volumes, loss_fn, args,
          holdout=None, particle_stack=None):
    """`particle_stack` (E, N, 4) ersetzt das Ziehen aus Volumen.

    Beim Volumenpfad wurde die Partikelwolke in jedem Schritt neu gezogen, was
    nebenbei als leichte Augmentierung der Konditionierung wirkte. Die
    projizierte Datenbank liefert stattdessen feste Wolken — dafuer sind es
    genau die, auf denen der Zielpfad erzeugt wurde. Beides ist vertretbar,
    aber es ist nicht dasselbe, und der Unterschied gehoert benannt.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5)
    model.train()

    ergodic = None
    if args.lambda_erg > 0.0:
        from ergodic_metric import ErgodicLoss
        ergodic = ErgodicLoss(
            nxi=args.nxi, K=args.erg_K, pts=args.erg_pts, deg=args.bspline_deg,
            weight=args.lambda_erg, t_power=args.erg_t_power,
            weighted_target=(args.sample_mode == 'uniform'), nd=args.nd,
            ergodic_on=args.erg_on, mu_thresh=args.mu_thresh,
        ).to(x1_clean.device)
        print(f"  Ergodic loss term active: {ergodic.extra_repr()}")

    orientation_loss = None
    if args.lambda_ori > 0.0:
        if not args.orientation:
            raise SystemExit("--lambda_ori needs --orientation (the model has "
                             "no rotation head otherwise)")
        from orientation_energy import OrientationLoss
        orientation_loss = OrientationLoss(
            nxi=args.nxi, pts=args.erg_pts, deg=args.bspline_deg,
            weight=args.lambda_ori, t_power=args.erg_t_power,
            w_point=args.w_point, w_standoff=args.w_standoff,
            w_angsmooth=args.w_angsmooth,
            standoff_target=args.standoff_target,
            standoff_band=args.standoff_band, mu_thresh=args.mu_thresh,
        ).to(x1_clean.device)
        print(f"  Orientation loss term active: {orientation_loss.extra_repr()}")
        if args.erg_on == 'position' and args.w_standoff > 0.0:
            # Worth stopping over rather than warning past: the database is
            # lifted 2D, so the stored curve lies *in* the target plane and its
            # distance to the surface is ~0. A standoff of 0.12 therefore asks
            # the trajectory to leave the plane, while an ergodic term scored on
            # the position asks it to stay. The two cancel, the run looks like it
            # trains, and neither objective is met. Scoring coverage at the
            # footprint resolves it: the robot stands off, the beam lands back on
            # the surface, and both terms want the same thing.
            raise SystemExit(
                "--lambda_ori with a standoff needs --erg_on footprint.\n"
                "  With --erg_on position the standoff pushes the trajectory off\n"
                "  the target plane and the coverage term pulls it back, so the\n"
                "  two fight to a draw. Pass --erg_on footprint, or set\n"
                "  --w_standoff 0 if you deliberately want an on-surface run.")

    use_cuda = x1_clean.device.type == 'cuda'
    N = x1_clean.shape[0]
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
    def _sigterm(signum, frame): raise TerminateInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)

    from tqdm import tqdm
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="ep")
    ep, avg = start_epoch, 0.0

    try:
        for ep in pbar:
            perm = torch.randperm(N, device=x1_clean.device)
            ep_loss, nb, ep_parts = 0.0, 0, {}

            for i in range(0, N, args.mini_batch):
                idx = perm[i:i + args.mini_batch]
                batch_clean = x1_clean[idx]
                batch_idx = shape_indices[idx]

                if particle_stack is not None:
                    parts_clean = particle_stack[batch_idx]
                else:
                    parts_clean = sample_particles(volumes, batch_idx.cpu(),
                                                   args.n_particles,
                                                   x1_clean.device,
                                                   mode=args.sample_mode)
                batch_aug, parts_aug = augment_batch(
                    batch_clean, parts_clean, p_flip=args.p_flip,
                    rot_range=args.rot_range, scale_range=args.scale_range,
                    trans_range=args.trans_range, noise_std=args.noise_std,
                    rot_full=args.rot_full,
                )

                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda' if use_cuda else 'cpu',
                                    dtype=torch.bfloat16):
                    loss, parts = loss_fn(model, batch_aug, parts_aug,
                                          p_drop=args.p_drop, ergodic=ergodic,
                                          orientation=orientation_loss,
                                          w_cfm_rot=args.w_cfm_rot)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                ep_loss += loss.item()
                for k, v in parts.items():
                    ep_parts[k] = ep_parts.get(k, 0.0) + v.item()
                nb += 1

            scheduler.step()
            avg = ep_loss / max(nb, 1)
            avg_parts = {k: v / max(nb, 1) for k, v in ep_parts.items()}
            lr_now = scheduler.get_last_lr()[0]

            if ep % 10 == 0 or ep == args.epochs - 1:
                post = {'loss': f"{avg:.5f}", 'lr': f"{lr_now:.2e}"}
                if 'erg' in avg_parts:
                    post['cfm'] = f"{avg_parts['cfm']:.5f}"
                    post['erg'] = f"{avg_parts['erg']:.2e}"
                if 'ori' in avg_parts:
                    # Pointing and standoff separately: they are the pair whose
                    # balance actually needs watching, and a combined number
                    # hides which of the two is still moving.
                    post['pt'] = f"{avg_parts['ori_point']:.2f}"
                    post['so'] = f"{avg_parts['ori_standoff']:.2f}"
                pbar.set_postfix(**post)

            if args.use_wandb and _WANDB_OK:
                log = {'train/loss': avg, 'train/lr': lr_now, 'epoch': ep + 1}
                for k, v in avg_parts.items():
                    log[f'train/{k}'] = v
                if 'erg' in avg_parts:
                    log['train/erg_weighted'] = ergodic.weight * avg_parts['erg']
                if 'ori' in avg_parts:
                    log['train/ori_weighted'] = (orientation_loss.weight
                                                 * avg_parts['ori'])
                wandb.log(log)

            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                _save_checkpoint(model, opt, scheduler, ep, avg, args)

            if args.viz_every > 0 and (ep + 1) % args.viz_every == 0 and holdout:
                _save_viz(model, *holdout, f"Holdout 3D - Epoch {ep+1}",
                          ep + 1, args, x1_clean.device)

        # Nach der letzten Epoche immer sichern, unabhaengig von --save_every.
        # Sonst haengt der Endstand daran, ob die Epochenzahl zufaellig ein
        # Vielfaches des Speicherabstands ist.
        _save_checkpoint(model, opt, scheduler, ep, avg, args)

    except (KeyboardInterrupt, TerminateInterrupt):
        print(f"\n  [!] Training interrupted at epoch {ep+1}. Saving emergency state...")
        _save_checkpoint(model, opt, scheduler, ep, avg, args)
        if holdout:
            _save_viz(model, *holdout, f"Emergency Holdout 3D - Epoch {ep+1}",
                      ep + 1, args, x1_clean.device, tag="emergency")
        sys.exit(0)


# ===========================================================================
# Main
# ===========================================================================


def run_surfaces(args, device):
    """Trainingslauf auf der projizierten 3D-Datenbank.

    Der Aufbau ist derselbe wie im Volumenpfad; was wegfaellt, ist die
    Zwischenstufe ueber Dichtevolumen und die abgeleiteten Stufe-0-Rahmen. Der
    Zielpfad und sein Rahmen stehen in der Tabelle, die Partikel ebenso.
    """
    from data_surfaces import load_surface_db, volume_from_particles

    if not args.orientation:
        raise SystemExit("--db3d braucht --orientation: die Datenbank liefert "
                         "SE(3)-Bahnen, und ohne Rotationskopf koennte das Netz "
                         "sechs der neun Zustandsdimensionen gar nicht lernen.")

    eintraege = load_surface_db(args.db3d, nxi=args.nxi, surfaces=args.surfaces,
                                max_jump=args.max_jump, max_miss=args.max_miss,
                                n_train_shapes=args.n_train_shapes)
    train_e = [e for e in eintraege if e['split'] == 'train']
    hold_e = [e for e in eintraege if e['split'] == 'val']
    if not train_e:
        raise SystemExit("keine Trainingseintraege — Filter zu streng?")

    flaechen = sorted({e['surface'] for e in eintraege})
    print(f"  {len(train_e)} Trainings- und {len(hold_e)} Holdout-Eintraege")
    print(f"  Oberflaechen: {', '.join(flaechen)}")
    if args.max_jump or args.max_miss:
        print(f"  Filter aktiv: max_jump={args.max_jump} max_miss={args.max_miss}")

    x1_np = np.stack([e['x1'] for e in train_e])
    pa_np = np.stack([e['parts'] for e in train_e])
    if args.copies_per_char > 1:
        # Wiederholungen erzeugen mehr Augmentierungen je Epoche, ohne die
        # Partikel zu duplizieren: der Index zeigt auf dieselbe Wolke.
        idx_np = np.tile(np.arange(len(train_e)), args.copies_per_char)
        x1_np = np.tile(x1_np, (args.copies_per_char, 1, 1))
    else:
        idx_np = np.arange(len(train_e))

    perm = np.random.permutation(len(x1_np))
    x1 = torch.tensor(x1_np[perm], dtype=torch.float32, device=device)
    shape_indices = torch.tensor(idx_np[perm], dtype=torch.long, device=device)
    particle_stack = torch.tensor(pa_np, dtype=torch.float32, device=device)
    print(f"  Trainingstensor: {tuple(x1.shape)}  "
          f"Partikelstapel: {tuple(particle_stack.shape)} "
          f"({particle_stack.numel() * 4 / 1e6:.0f} MB auf der GPU)")

    # Holdout: je Oberflaeche zwei Beispiele, damit das Bild alle sieben zeigt.
    viz_e, gesehen = [], {}
    for e in hold_e:
        k = e['surface']
        if gesehen.get(k, 0) < 2:
            gesehen[k] = gesehen.get(k, 0) + 1
            viz_e.append(e)
    hold_lbls = [f"{e['name']}|{e['surface']}" for e in viz_e]
    traj_map = {l: e['x1'][:, :3] for l, e in zip(hold_lbls, viz_e)}
    part_map = {l: torch.tensor(e['parts'], device=device)
                for l, e in zip(hold_lbls, viz_e)}
    vol_map = {l: volume_from_particles(e['parts'])
               for l, e in zip(hold_lbls, viz_e)}

    model = ParticleCrossAttnFlowNetwork(
        nxi=args.nxi, nd=args.nd, D=args.D, predict_orientation=True).to(device)
    print(f"  Modellparameter: {sum(p.numel() for p in model.parameters()):,}\n")

    if args.load_model and os.path.isfile(args.load_model):
        ckpt = torch.load(args.load_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Gewichte geladen: {args.load_model}")

    if args.use_wandb and _WANDB_OK:
        wandb.init(project="flow3d-surfaces", name=args.run_str,
                   config=vars(args))

    train(model, x1, shape_indices, None, compute_particle_cfm_loss, args,
          holdout=(hold_lbls, traj_map, part_map, vol_map),
          particle_stack=particle_stack)

    visualise_set(model, hold_lbls, traj_map, part_map, vol_map,
                  "Holdout - Oberflaechen", os.path.join(
                      _here, 'viz', f"{args.run_str}_holdout.png"),
                  args, device)
    print("\nFertig.")


def run(args):
    if getattr(args, 'run_tag', None):
        args.timestamp = args.run_tag
    if not hasattr(args, 'timestamp'):
        args.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")
    args.run_str = (f"flow3d_particle_ergodic_{args.timestamp}_nxi{args.nxi}"
                    f"_D{args.D}_N{args.n_particles}_R{args.grid_res}"
                    f"_C{args.copies_per_char}_flip{args.p_flip}")
    if args.db3d:
        args.run_str += "_SURF"
        if args.surfaces:
            args.run_str += "-" + "+".join(s[:4] for s in args.surfaces)
    if args.rot_full:
        args.run_str += "_SO3"
    if args.orientation:
        args.run_str += f"_SE3-{args.frame_mode}"
    if args.lambda_erg > 0.0:
        args.run_str += f"_ERGLOSS-w{args.lambda_erg:g}-K{args.erg_K}-tp{args.erg_t_power:g}"
        if args.erg_on != 'position':
            args.run_str += f"-on{args.erg_on}"
    if args.lambda_ori > 0.0:
        args.run_str += (f"_ORILOSS-w{args.lambda_ori:g}"
                         f"-pt{args.w_point:g}-so{args.standoff_target:g}"
                         f"-cfmrot{args.w_cfm_rot:g}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"\n{'=' * 74}")
    print(f"  3D Particle Cross-Attention Flow Matching")
    print(f"  device={device}  nxi={args.nxi}  nd={args.nd}  D={args.D}")
    print(f"  particles={args.n_particles}  grid={args.grid_res}^3")
    print(f"  plane z={args.z_plane}  slab sigma={args.z_sigma}")
    print(f"  rotation={'SO(3)' if args.rot_full else 'about z (planar-preserving)'}")
    print(f"  CFG: p_drop={args.p_drop}  cfg_weight={args.cfg_weight}")
    print(f"{'=' * 74}")

    if args.db3d:
        return run_surfaces(args, device)

    traj, shape_defs, splits = load_pairs(args.nxi, db_path=args.db)
    train_lbls = [l for l in traj if splits[l] == 'train'][:args.n_train_shapes]
    hold_lbls = [l for l in traj if splits[l] == 'val']
    print(f"  Loaded {len(train_lbls)} train and {len(hold_lbls)} holdout shapes")

    all_lbls = train_lbls + hold_lbls
    used, volumes_np, _ = prepare_targets(
        all_lbls, shape_defs, args.grid_res, args.erg_K if args.lambda_erg > 0 else 4,
        z_plane=args.z_plane, z_sigma=args.z_sigma, use_cache=not args.no_cache)
    vol_index = {l: i for i, l in enumerate(used)}
    train_lbls = [l for l in train_lbls if l in vol_index]
    hold_lbls = [l for l in hold_lbls if l in vol_index]

    # Volumes stay on the CPU; only the per-batch slice is moved to the GPU.
    volumes = torch.from_numpy(volumes_np)
    print(f"  Volume stack: {tuple(volumes.shape)} "
          f"({volumes.numel() * 4 / 1e6:.0f} MB, kept on CPU)")

    ori_targets = None
    if args.orientation:
        print(f"  Deriving Stufe-0 orientation labels ({args.frame_mode}) "
              f"for {len(train_lbls)} shapes...")
        ori_targets = orientation_targets(traj, train_lbls, volumes_np,
                                          vol_index, args.frame_mode, device)

    x1_list, idx_list = [], []
    for lbl in train_lbls:
        block = traj[lbl]
        if ori_targets is not None:
            block = np.concatenate([block, ori_targets[lbl]], axis=-1)
        x1_list.append(np.tile(block[None], (args.copies_per_char, 1, 1)))
        idx_list.extend([vol_index[lbl]] * args.copies_per_char)
    x1_np = np.concatenate(x1_list, axis=0)
    perm = np.random.permutation(len(x1_np))
    x1 = torch.tensor(x1_np[perm], dtype=torch.float32, device=device)
    shape_indices = torch.tensor(np.array(idx_list)[perm], dtype=torch.long,
                                 device=device)
    print(f"  Training tensor: {tuple(x1.shape)}")

    hold_particles, hold_volumes = {}, {}
    for lbl in hold_lbls:
        i = torch.tensor([vol_index[lbl]], dtype=torch.long)
        hold_particles[lbl] = sample_particles(volumes, i, args.n_particles,
                                               device, mode=args.sample_mode)[0]
        hold_volumes[lbl] = volumes_np[vol_index[lbl]]

    model = ParticleCrossAttnFlowNetwork(
        nxi=args.nxi, nd=args.nd, D=args.D,
        predict_orientation=args.orientation).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}"
          f"{'  (with orientation head)' if args.orientation else ''}\n")

    if args.load_model and os.path.isfile(args.load_model):
        ckpt = torch.load(args.load_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint from {args.load_model}")
    else:
        if args.use_wandb and _WANDB_OK:
            resume_id = None
            if args.resume and os.path.isfile(args.resume):
                try:
                    resume_id = torch.load(args.resume, map_location='cpu',
                                           weights_only=True).get('wandb_id')
                except Exception:
                    pass
            wandb.init(project=args.wandb_project,
                       name=args.run_name or args.run_str,
                       config=vars(args), id=resume_id,
                       resume="allow" if resume_id else None)

        holdout = (hold_lbls, traj, hold_particles, hold_volumes)
        train(model, x1, shape_indices, volumes, compute_particle_cfm_loss,
              args, holdout=holdout)
        print("  Training complete!")
        if args.use_wandb and _WANDB_OK:
            wandb.finish()

        stem = os.path.splitext(args.save_model)[0]
        os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
        final = f"{stem}_{args.run_str}_final.pt"
        torch.save({'model_state_dict': model.state_dict(),
                    'nxi': args.nxi, 'nd': args.nd, 'D': args.D,
                    'n_particles': args.n_particles, 'dim': 3,
                    'orientation': args.orientation,
                    'frame_mode': args.frame_mode,
                    'grid_res': args.grid_res}, final)
        print(f"  Checkpoint saved -> {final}")

    out_dir = os.path.join(_here, 'viz')
    os.makedirs(out_dir, exist_ok=True)
    viz_train = random.sample(train_lbls, min(5, len(train_lbls)))
    tp = {}
    for lbl in viz_train:
        i = torch.tensor([vol_index[lbl]], dtype=torch.long)
        tp[lbl] = sample_particles(volumes, i, args.n_particles, device,
                                   mode=args.sample_mode)[0]
    visualise_set(model, viz_train, traj, tp,
                  {l: volumes_np[vol_index[l]] for l in viz_train},
                  f'Training Shapes 3D ({args.n_particles} Particles)',
                  os.path.join(out_dir, f"{args.run_str}_train.png"), args, device)
    visualise_set(model, hold_lbls, traj, hold_particles, hold_volumes,
                  f'HELD-OUT Shapes 3D ({args.n_particles} Particles)',
                  os.path.join(out_dir, f"{args.run_str}_holdout.png"), args, device)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--nxi', type=int, default=25)
    p.add_argument('--nd',  type=int, default=ND)
    p.add_argument('--D',   type=int, default=256)
    p.add_argument('--n_particles', type=int, default=512,
                   help='More than the 2D default: a volume needs more samples '
                        'than an area for the same coverage of the support.')
    p.add_argument('--sample_mode', type=str, default='uniform',
                   choices=['uniform', 'density'])

    p.add_argument('--db', type=str, default=DEFAULT_DB)
    p.add_argument('--db3d', type=str, default=None,
                   help='Auf der projizierten 3D-Datenbank trainieren statt auf '
                        'der gehobenen 2D-Datenbank. Bahn, Rahmen und Partikel '
                        'kommen dann fertig aus der Tabelle.')
    p.add_argument('--surfaces', type=str, nargs='+', default=None,
                   help='Nur diese Oberflaechen laden. Ohne Angabe alle sieben.')
    p.add_argument('--max_jump', type=float, default=None,
                   help='Eintraege mit groesserem Sprung zwischen zwei '
                        'Bahnpunkten weglassen. Sprünge entstehen an Kanten.')
    p.add_argument('--max_miss', type=float, default=None,
                   help='Eintraege mit hoeherem Fehlschussanteil weglassen.')
    p.add_argument('--n_train_shapes', type=int, default=750)
    p.add_argument('--grid_res', type=int, default=64,
                   help='Volume is grid_res^3; 128 would be 6 GB over 750 shapes.')
    p.add_argument('--z_plane', type=float, default=Z_PLANE)
    p.add_argument('--z_sigma', type=float, default=Z_SIGMA)
    p.add_argument('--no_cache', action='store_true', default=False)

    p.add_argument('--epochs',          type=int,   default=500)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--mini_batch',      type=int,   default=64)
    p.add_argument('--copies_per_char', type=int,   default=15)
    p.add_argument('--noise_std',       type=float, default=0.015)

    p.add_argument('--p_flip',      type=float, default=0.0)
    p.add_argument('--rot_range',   type=float, default=20.0)
    p.add_argument('--scale_range', type=float, nargs=2, default=[0.75, 1.25])
    p.add_argument('--trans_range', type=float, default=0.08)
    p.add_argument('--rot_full', action='store_true', default=False,
                   help='General SO(3) rotation instead of about the plane '
                        'normal. Tilts planar data out of its plane.')

    # ── Stufe 1: orientation ─────────────────────────────────────────────────
    # The labels come from Stufe 0 (frames derived from the stored trajectory),
    # because the database has no orientation of its own.
    p.add_argument('--orientation', action='store_true', default=False,
                   help='Also learn a 6D rotation per control point, supervised '
                        'by Stufe-0 frames derived from the ground-truth curve.')
    p.add_argument('--frame_mode', type=str, default='lookat',
                   choices=['lookat', 'rmf', 'frenet'],
                   help="How the Stufe-0 labels are built. 'lookat' points the "
                        "sensor at the surface; 'rmf' follows the curve with "
                        "minimal twist; 'frenet' is for comparison only.")

    p.add_argument('--p_drop',     type=float, default=0.1)
    p.add_argument('--cfg_weight', type=float, default=2.0)

    p.add_argument('--lambda_erg',  type=float, default=0.0,
                   help='Weight of the ergodic coverage term. 0 disables it.')
    p.add_argument('--erg_K',       type=int,   default=6,
                   help='Frequency grid is erg_K^3 modes (216 at K=6).')
    p.add_argument('--erg_pts',     type=int,   default=128)
    p.add_argument('--erg_t_power', type=float, default=2.0)
    p.add_argument('--erg_on', type=str, default='position',
                   choices=['position', 'footprint'],
                   help="Where coverage is scored. 'footprint' follows the "
                        "sensor beam and is the only setting under which "
                        "orientation affects coverage at all.")

    # ── Stufe 3: orientation as an objective rather than an imitation target ──
    p.add_argument('--lambda_ori', type=float, default=0.0,
                   help='Weight of the orientation objective (pointing + '
                        'standoff + angular smoothness). 0 disables it, which '
                        'is the previous behaviour exactly.')
    p.add_argument('--w_cfm_rot', type=float, default=1.0,
                   help='Weight of the CFM imitation term on the rotation '
                        'channels. Set 0 to learn orientation purely from the '
                        'objective, with the Stufe-0 frames unused.')
    p.add_argument('--w_point', type=float, default=W_POINT)
    p.add_argument('--w_standoff', type=float, default=W_STANDOFF)
    p.add_argument('--w_angsmooth', type=float, default=W_ANGSMOOTH)
    p.add_argument('--standoff_target', type=float, default=STANDOFF_TARGET)
    p.add_argument('--standoff_band', type=float, default=STANDOFF_BAND)
    p.add_argument('--mu_thresh', type=float, default=0.5,
                   help='Occupancy threshold on mu, relative to the per-sample '
                        'peak, for the particle-derived surface.')

    p.add_argument('--n_gen',       type=int, default=5)
    p.add_argument('--steps',       type=int, default=100)
    p.add_argument('--bspline_pts', type=int, default=512)
    p.add_argument('--bspline_deg', type=int, default=5)

    p.add_argument('--run_tag', type=str, default=None,
                   help='Feste Laufkennung statt des Zeitstempels. Fuer '
                        'verkettete Jobs noetig: nur mit gleicher Kennung '
                        'schreiben beide Teile unter denselben Namen, sodass '
                        'die Checkpoint-Rotation den Stand des ersten Teils '
                        'auch wirklich ersetzt statt ihn liegen zu lassen.')
    p.add_argument('--keep_checkpoints', type=int, default=1,
                   help='So viele Checkpoints desselben Laufs behalten. Beim '
                        'Speichern eines neuen werden aeltere entfernt — erst '
                        'nachdem der neue geschrieben und geprueft ist.')
    p.add_argument('--save_model', type=str,
                   default=os.path.join(_here, 'checkpoints', 'cond_particles_3d.pt'))
    p.add_argument('--load_model', type=str, default=None)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--viz_every', type=int, default=20)

    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--wandb_project', type=str, default='flow-matching-3d')
    p.add_argument('--run_name', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
