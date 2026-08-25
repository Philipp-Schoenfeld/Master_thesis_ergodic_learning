#!/usr/bin/env python3
r"""
flow_matching_runner_particles_selfsupervised.py  —  3D port
============================================================
Self-supervised training against the solver energy in 3D.

The model never sees a solved trajectory. It is trained directly on
E(B @ cps, phi_k), so the only thing the database is needed for is the *density
definitions* — the stored trajectories are used purely as a visual reference and
as a baseline energy, never as a target.

That property is worth stating explicitly here, because it is exactly what
makes a 3D extension cheap: moving the supervised pipeline to 3D would require
re-running the SVGD solver on 3D targets to produce new ground truth, which does
not exist. The self-supervised pipeline needs none of that — the energy is
defined analytically in any dimension.

Cost note: the energy uses K^3 modes. At the solver's K = 10 that is 1000 modes
per curve evaluation instead of 100, which is the dominant new cost in 3D. The
default here is K = 8 (512 modes) as a compromise; use --erg_K 10 for exact
solver parity.
"""

import argparse, os, sys
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
                     DEFAULT_DB, Z_PLANE, Z_SIGMA)
from ergodic_energy_torch import (ErgodicEnergy, K_DEFAULT, W_ERGODIC, W_SMOOTH,
                                  W_BOUNDARY, W_OBSTACLE, planarity)
from flow_matching_particles_selfsupervised import (
    SelfSupervisedParticleGenerator, compute_selfsupervised_loss,
)
from obstacles import bspline_basis_matrix
from orientation import SurfaceField, rot6d_to_matrix, rot_path_length
from orientation_energy import (SE3Energy, W_POINT, W_STANDOFF, W_ANGSMOOTH,
                                STANDOFF_TARGET, STANDOFF_BAND,
                                pointing_error_deg, incidence_ok_fraction)
import viz_3d


def _basis(nxi, T, deg, device):
    return torch.from_numpy(bspline_basis_matrix(nxi, T, deg)).to(device)


def solver_reference_energy(labels, traj, phi, energy, device, nxi, fields=None):
    """Energy of the stored solver trajectories — a free yardstick.

    Not supervision: the trajectories are only scored, never learned from.

    With orientation enabled the reference gets its frames from Stufe 0
    (look-at from the surface field), so the comparison is like-for-like: the
    solver path judged with the best orientation that can be *derived* from it,
    against a model that *learns* one.
    """
    from orientation import frames_for_curve, matrix_to_rot6d
    vals = []
    for i, lbl in enumerate(labels):
        cps = torch.tensor(traj[lbl], dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            if fields is None:
                vals.append(energy(cps, phi[i:i + 1]).item())
            else:
                R = frames_for_curve(cps, fields[i], mode='lookat')
                vals.append(energy(cps, phi[i:i + 1],
                                   rot6d=matrix_to_rot6d(R),
                                   field=fields[i]).item())
    return float(np.mean(vals)) if vals else float('nan')


def _save_checkpoint(model, opt, sched, epoch, loss, args):
    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
    stem = os.path.splitext(args.save_model)[0]
    path = f"{stem}_{args.run_str}_ep{epoch+1:04d}.pt"
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'epoch': epoch, 'loss': loss,
        'nxi': args.nxi, 'nd': args.nd, 'D': args.D,
        'n_particles': args.n_particles,
        'n_candidates': args.n_candidates,
        'diversity_weight': args.diversity_weight,
        'erg_K': args.erg_K, 'solver_T': args.solver_T,
        'grid_res': args.grid_res, 'z_plane': args.z_plane, 'z_sigma': args.z_sigma,
        'use_obstacle': args.use_obstacle,
        'w_ergodic': args.w_ergodic, 'w_smooth': args.w_smooth,
        'w_boundary': args.w_boundary, 'w_obstacle': args.w_obstacle,
        'selfsupervised': True, 'dim': 3,
    }
    if args.use_wandb and _WANDB_OK and wandb.run is not None:
        ckpt['wandb_id'] = wandb.run.id
    torch.save(ckpt, path)
    return path


@torch.no_grad()
def visualise_set(model, labels, traj, particles_map, volumes_map, title,
                  save_path, args, device, max_cols=5, basis=None):
    model.eval()
    panels = []
    # The frames must be sampled at the same resolution as the drawn curve,
    # otherwise the arrows and the line do not line up.
    viz_basis = _basis(args.nxi, args.bspline_pts, args.bspline_deg, device)
    for lbl in labels:
        cond = particles_map[lbl]
        gen, rot6d = model.generate(cond, num_samples=args.n_gen, device=str(device))
        gen_R = None
        if rot6d is not None:
            r = torch.einsum('ti,ic->tc', viz_basis, rot6d[0])
            gen_R = rot6d_to_matrix(r).cpu().numpy()
        panels.append(dict(
            base=traj.get(lbl),
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


def train(model, volumes, phi_t, energy, args, device, holdout=None,
          fields=None):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-5)
    model.train()
    S = phi_t.shape[0]
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        sched.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed from epoch {start_epoch} (loss={ckpt['loss']:.5f})")

    import signal
    class TerminateInterrupt(Exception): pass
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(TerminateInterrupt()))

    from tqdm import tqdm
    pbar = tqdm(range(start_epoch, args.epochs), desc="Selfsup 3D", unit="ep")
    ep, avg = start_epoch, 0.0
    first_avg = None

    try:
        for ep in pbar:
            perm = torch.randperm(S)
            ep_loss, nb, ep_parts = 0.0, 0, {}

            for i in range(0, S, args.mini_batch):
                idx = perm[i:i + args.mini_batch]
                parts = sample_particles(volumes, idx, args.n_particles,
                                         device, mode=args.sample_mode)
                phi = phi_t[idx.to(phi_t.device)].to(device)

                batch_fields = ([fields[int(j)] for j in idx]
                                if fields is not None else None)

                opt.zero_grad(set_to_none=True)
                loss, comp = compute_selfsupervised_loss(
                    model, parts, phi, energy,
                    n_candidates=args.n_candidates,
                    diversity_weight=args.diversity_weight,
                    fields=batch_fields)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step()

                ep_loss += loss.item()
                for k, v in comp.items():
                    ep_parts[k] = ep_parts.get(k, 0.0) + v.item()
                nb += 1

            sched.step()
            avg = ep_loss / max(nb, 1)
            avg_parts = {k: v / max(nb, 1) for k, v in ep_parts.items()}
            if first_avg is None:
                first_avg = avg_parts.get('energy', avg)

            if ep % 5 == 0 or ep == args.epochs - 1:
                post = {'loss': f"{avg:.3f}"}
                if 'energy' in avg_parts:
                    post['E'] = f"{avg_parts['energy']:.3f}"
                if 'point' in avg_parts:
                    post['pt'] = f"{avg_parts['point']:.2f}"
                    post['so'] = f"{avg_parts['standoff']:.2f}"
                if 'diversity' in avg_parts:
                    post['div'] = f"{avg_parts['diversity']:.3f}"
                pbar.set_postfix(**post)

            if args.use_wandb and _WANDB_OK:
                log = {'train/loss': avg, 'train/lr': sched.get_last_lr()[0],
                       'epoch': ep + 1}
                for k, v in avg_parts.items():
                    log[f'loss/{k}'] = v
                wandb.log(log)

            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                _save_checkpoint(model, opt, sched, ep, avg, args)
            if args.viz_every > 0 and (ep + 1) % args.viz_every == 0 and holdout:
                out = os.path.join(_here, 'viz',
                                   f"{args.run_str}_ep{ep+1:04d}.png")
                visualise_set(model, *holdout, f"Holdout 3D - Epoch {ep+1}",
                              out, args, device)

            # Feasibility gate: if the energy has not fallen by the requested
            # ratio after the probe window, the design is wrong and 20 GPU hours
            # should not be spent finding that out slowly.
            if (args.assert_energy_drops and ep + 1 == args.gate_epochs
                    and 'energy' in avg_parts):
                ratio = avg_parts['energy'] / max(first_avg, 1e-12)
                print(f"\n  [gate] energy {first_avg:.3f} -> "
                      f"{avg_parts['energy']:.3f}  (ratio {ratio:.3f})")
                if ratio > args.min_drop_ratio:
                    print("  [gate] FAILED — energy did not fall enough. Aborting.")
                    sys.exit(1)
                print("  [gate] passed\n")

    except (KeyboardInterrupt, TerminateInterrupt):
        print(f"\n  [!] Interrupted at epoch {ep+1}. Saving emergency state...")
        _save_checkpoint(model, opt, sched, ep, avg, args)
        if holdout:
            out = os.path.join(_here, 'viz', f"{args.run_str}_emergency_ep{ep+1:04d}.png")
            visualise_set(model, *holdout, f"Emergency 3D - Epoch {ep+1}",
                          out, args, device)
        sys.exit(0)


def run(args):
    if not hasattr(args, 'timestamp'):
        args.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")
    args.run_str = (f"selfsup3d_{args.timestamp}_nxi{args.nxi}_D{args.D}"
                    f"_N{args.n_particles}_R{args.grid_res}_K{args.n_candidates}"
                    f"_div{args.diversity_weight:g}_ergK{args.erg_K}")
    if args.use_obstacle:
        args.run_str += "_OBST"
    if args.orientation:
        # The marker carries the settings that change what is being optimised,
        # so an ls over checkpoints/ shows which run is which.
        args.run_str += (f"_SE3-{args.ergodic_on}-pt{args.w_point:g}"
                         f"-so{args.standoff_target:g}")

    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.seed)

    print(f"\n{'=' * 74}")
    print(f"  3D Self-Supervised Ergodic Generator")
    print(f"  device={device}  nxi={args.nxi}  D={args.D}  N={args.n_particles}")
    print(f"  energy: K={args.erg_K} ({args.erg_K ** 3} modes), T={args.solver_T}")
    print(f"  candidates K={args.n_candidates}  diversity_weight={args.diversity_weight}")
    print(f"{'=' * 74}")

    traj, shape_defs, splits = load_pairs(args.nxi, db_path=args.db)
    train_lbls = [l for l in traj if splits[l] == 'train'][:args.n_train_shapes]
    hold_lbls = [l for l in traj if splits[l] == 'val']
    if args.shapes:
        want = set(args.shapes.split(','))
        train_lbls = [l for l in traj if l.split('#')[0] in want] or train_lbls[:1]
        hold_lbls = train_lbls

    all_lbls = list(dict.fromkeys(train_lbls + hold_lbls))
    used, volumes_np, phi_np = prepare_targets(
        all_lbls, shape_defs, args.grid_res, args.erg_K,
        z_plane=args.z_plane, z_sigma=args.z_sigma, use_cache=not args.no_cache)
    index = {l: i for i, l in enumerate(used)}
    train_lbls = [l for l in train_lbls if l in index]
    hold_lbls = [l for l in hold_lbls if l in index]

    volumes = torch.from_numpy(volumes_np)                    # CPU
    phi_all = torch.from_numpy(phi_np)                        # CPU
    train_ids = torch.tensor([index[l] for l in train_lbls], dtype=torch.long)
    phi_train = phi_all[train_ids]
    vol_train = volumes[train_ids]
    print(f"  Volumes {tuple(volumes.shape)}  phi {tuple(phi_all.shape)}")

    basis = _basis(args.nxi, args.solver_T, args.bspline_deg, device)
    base_energy = ErgodicEnergy(K=args.erg_K, basis=basis,
                                use_obstacle=args.use_obstacle,
                                w_ergodic=args.w_ergodic, w_smooth=args.w_smooth,
                                w_boundary=args.w_boundary,
                                w_obstacle=args.w_obstacle).to(device)

    train_fields = hold_fields = None
    if args.orientation:
        energy = SE3Energy(base=base_energy, w_point=args.w_point,
                           w_standoff=args.w_standoff,
                           w_angsmooth=args.w_angsmooth,
                           standoff_target=args.standoff_target,
                           standoff_band=args.standoff_band,
                           ergodic_on=args.ergodic_on).to(device)
        print(f"  Energy: {base_energy.extra_repr()}")
        print(f"  SE(3):  {energy.extra_repr()}")
        print(f"  Building surface fields for {len(used)} shapes...")
        all_fields = [SurfaceField(volumes_np[i], device=device)
                      for i in range(len(used))]
        train_fields = [all_fields[index[l]] for l in train_lbls]
        hold_fields = [all_fields[index[l]] for l in hold_lbls]
    else:
        energy = base_energy
        print(f"  Energy: {energy.extra_repr()}")

    ref = solver_reference_energy(
        hold_lbls, traj,
        phi_all[torch.tensor([index[l] for l in hold_lbls])].to(device),
        energy, device, args.nxi, fields=hold_fields)
    print(f"  Solver reference energy on holdout: {ref:.4f}"
          f"{'  (frames from Stufe 0 look-at)' if args.orientation else ''}\n")

    model = SelfSupervisedParticleGenerator(
        nxi=args.nxi, nd=args.nd, D=args.D,
        predict_orientation=args.orientation).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}"
          f"{'  (with orientation head)' if args.orientation else ''}\n")

    hold_particles, hold_volumes = {}, {}
    for lbl in hold_lbls:
        i = torch.tensor([index[lbl]], dtype=torch.long)
        hold_particles[lbl] = sample_particles(volumes, i, args.n_particles,
                                               device, mode=args.sample_mode)[0]
        hold_volumes[lbl] = volumes_np[index[lbl]]

    if args.use_wandb and _WANDB_OK:
        wandb.init(project=args.wandb_project, name=args.run_name or args.run_str,
                   config=vars(args))

    os.makedirs(os.path.join(_here, 'viz'), exist_ok=True)
    holdout = (hold_lbls, traj, hold_particles, hold_volumes)
    train(model, vol_train, phi_train, energy, args, device, holdout=holdout,
          fields=train_fields)
    print("  Training complete!")

    if args.use_wandb and _WANDB_OK:
        wandb.finish()

    stem = os.path.splitext(args.save_model)[0]
    os.makedirs(os.path.dirname(args.save_model) or '.', exist_ok=True)
    final = f"{stem}_{args.run_str}_final.pt"
    torch.save({'model_state_dict': model.state_dict(),
                'nxi': args.nxi, 'nd': args.nd, 'D': args.D,
                'n_particles': args.n_particles,
                'n_candidates': args.n_candidates,
                'diversity_weight': args.diversity_weight,
                'selfsupervised': True, 'dim': 3,
                'orientation': args.orientation,
                'ergodic_on': args.ergodic_on,
                'standoff_target': args.standoff_target,
                'standoff_band': args.standoff_band,
                'w_point': args.w_point, 'w_standoff': args.w_standoff,
                'w_angsmooth': args.w_angsmooth,
                'erg_K': args.erg_K, 'grid_res': args.grid_res}, final)
    print(f"  Checkpoint saved -> {final}")

    visualise_set(model, hold_lbls, traj, hold_particles, hold_volumes,
                  'HELD-OUT 3D (self-supervised)',
                  os.path.join(_here, 'viz', f"{args.run_str}_holdout.png"),
                  args, device)

    # Diagnostics on the holdout set.
    with torch.no_grad():
        cps, rot6d = model.generate(hold_particles[hold_lbls[0]], num_samples=8,
                                    device=str(device))
        curve = torch.einsum('ti,bid->btd', basis, cps)
        pl = planarity(curve)
        print(f"  Planarity of generated curves (RMS off best-fit plane): "
              f"mean {pl.mean():.4f}, max {pl.max():.4f}")

        if rot6d is not None:
            r = torch.einsum('ti,bic->btc', basis, rot6d)
            R = rot6d_to_matrix(r)
            f0 = hold_fields[0]
            d = f0.direction(curve)
            err = pointing_error_deg(R, d)
            ok = incidence_ok_fraction(R, d, max_deg=30.0)
            dist = f0.distance(curve)
            print(f"  Pointing error:  mean {err.mean():.1f} deg, "
                  f"within 30 deg for {ok.mean() * 100:.0f} % of the path")
            print(f"  Standoff:        mean {dist.mean():.3f} "
                  f"(target {args.standoff_target:g} +- {args.standoff_band:g})")
            print(f"  Rotational path: {rot_path_length(R).mean():.2f} rad")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--nxi', type=int, default=25)
    p.add_argument('--nd',  type=int, default=3)
    p.add_argument('--D',   type=int, default=384)
    p.add_argument('--n_particles', type=int, default=512)
    p.add_argument('--bspline_deg', type=int, default=5)
    p.add_argument('--bspline_pts', type=int, default=512)
    p.add_argument('--sample_mode', type=str, default='uniform',
                   choices=['uniform', 'density'])

    p.add_argument('--db', type=str, default=DEFAULT_DB)
    p.add_argument('--shapes', type=str, default=None,
                   help='Comma-separated shape names for a quick probe run.')
    p.add_argument('--n_train_shapes', type=int, default=750)
    p.add_argument('--grid_res', type=int, default=64)
    p.add_argument('--z_plane', type=float, default=Z_PLANE)
    p.add_argument('--z_sigma', type=float, default=Z_SIGMA)
    p.add_argument('--no_cache', action='store_true', default=False)

    # ── Stufe 1+2: orientation ───────────────────────────────────────────────
    # Opt-in. Without --orientation the run is byte-identical to a position-only
    # one: no head is built, no field is computed, no term is added.
    p.add_argument('--orientation', action='store_true', default=False,
                   help='Predict a 6D rotation per control point and enable the '
                        'pointing / standoff / angular-smoothness terms.')
    p.add_argument('--ergodic_on', type=str, default='footprint',
                   choices=['position', 'footprint'],
                   help="Where coverage is scored. 'footprint' is where the "
                        "sensor beam lands and is the meaningful choice with a "
                        "standoff; 'position' scores the robot itself.")
    p.add_argument('--w_point',     type=float, default=W_POINT)
    p.add_argument('--w_standoff',  type=float, default=W_STANDOFF)
    p.add_argument('--w_angsmooth', type=float, default=W_ANGSMOOTH)
    p.add_argument('--standoff_target', type=float, default=STANDOFF_TARGET,
                   help='Desired distance from the surface, in domain widths.')
    p.add_argument('--standoff_band', type=float, default=STANDOFF_BAND,
                   help='Free band around the target; outside it costs '
                        'quadratically.')

    p.add_argument('--n_candidates', type=int, default=8)
    p.add_argument('--diversity_weight', type=float, default=0.0)
    p.add_argument('--use_obstacle', action='store_true', default=False)
    p.add_argument('--w_ergodic',  type=float, default=W_ERGODIC)
    p.add_argument('--w_smooth',   type=float, default=W_SMOOTH)
    p.add_argument('--w_boundary', type=float, default=W_BOUNDARY)
    p.add_argument('--w_obstacle', type=float, default=W_OBSTACLE)
    p.add_argument('--erg_K',    type=int, default=8,
                   help='K^3 modes. 10 matches the solver exactly (1000 modes).')
    p.add_argument('--solver_T', type=int, default=100,
                   help='Curve samples; must stay 100 for the solver weights to '
                        'transfer, since W_SMOOTH was calibrated on that spacing.')

    p.add_argument('--epochs',     type=int,   default=500)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--mini_batch', type=int,   default=32)
    p.add_argument('--clip_grad',  type=float, default=1.0)
    p.add_argument('--seed',       type=int,   default=0)

    p.add_argument('--assert_energy_drops', action='store_true', default=False)
    p.add_argument('--gate_epochs',    type=int,   default=20)
    p.add_argument('--min_drop_ratio', type=float, default=0.9)

    p.add_argument('--save_model', type=str,
                   default=os.path.join(_here, 'checkpoints', 'selfsup_3d.pt'))
    p.add_argument('--resume',     type=str, default=None)
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--viz_every',  type=int, default=20)
    p.add_argument('--n_gen',      type=int, default=5)
    p.add_argument('--device',     type=str, default=None)

    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--wandb_project', type=str, default='flow-matching-3d')
    p.add_argument('--run_name',   type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
