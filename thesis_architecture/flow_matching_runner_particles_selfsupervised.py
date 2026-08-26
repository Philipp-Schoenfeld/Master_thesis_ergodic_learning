#!/usr/bin/env python3
"""
flow_matching_runner_particles_selfsupervised.py
================================================
Self-supervised trainer: the generator is optimised directly against the SVGD
solver's energy, never against a solved trajectory.

Consequence for the data pipeline: only the *target distributions* are needed
(`shape_library`), not the solved trajectories in `ergodic_pairs.db`. The 775
offline SVGD runs are not required for this training at all. The database is
opened once, read-only, purely to compute a reference energy bar from the
solver's own output — as a yardstick, never as supervision.

The supervised pipeline (`flow_matching_runner_particles.py`,
`flow_matching_cond_particles_crossattn.py`) is imported from, never modified.

Usage:
  # Feasibility: one shape, one candidate, no diversity
  python -u flow_matching_runner_particles_selfsupervised.py \
      --shapes A --n_candidates 1 --diversity_weight 0.0 --epochs 200

  # Full run
  python -u flow_matching_runner_particles_selfsupervised.py \
      --n_candidates 8 --diversity_weight 100 --epochs 500 --use_wandb
"""

import argparse, os, sys, json, math, random, sqlite3, hashlib
from datetime import datetime
import numpy as np
import torch

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(os.path.join(_here, 'ergodic_dataset_generator'))

from bsplinax.bspline import BsplineBasisClamped
from shape_library import pdf_on_grid, get_shape, train_shape_names, VALIDATION_SHAPES

from ergodic_energy_torch import (
    ErgodicEnergy, make_k_grid, target_coeffs_from_grid,
    W_ERGODIC, W_SMOOTH, W_BOUNDARY, W_OBSTACLE, K_DEFAULT,
)
from flow_matching_particles_selfsupervised import (
    SelfSupervisedParticleGenerator, compute_selfsupervised_loss,
)
# Particle sampling and the plotting style are reused verbatim.
from flow_matching_runner_particles import sample_particles
from visualize_checkpoint import draw_panel, save_grid
from checkpoint_rotation import nach_zwischenstand, nach_endstand

_DB_PATH = os.path.join(_here, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')
_CACHE_DIR = os.path.join(_here, 'cache')

# The solver optimises T=100 dense points; rendering to exactly that many keeps
# its weights meaningful (W_SMOOTH was calibrated on this spacing).
SOLVER_T = 100


# ===========================================================================
# Target preparation (densities and phi_k), cached
# ===========================================================================

def _cache_path(names, grid_res, K, source):
    key = hashlib.md5(('|'.join(names) + f'|{grid_res}|{K}|{source}').encode()).hexdigest()[:12]
    return os.path.join(_CACHE_DIR, f'targets_{len(names)}shapes_r{grid_res}_K{K}_{key}.npz')


def load_shape_defs_from_db():
    """shape_name -> density parameters, plus the train/val split, from the DB.

    Only the `density_params` column is read — the target *distributions*. The
    `trajectory` column (the solver's answers) is deliberately left untouched;
    this trainer never sees a solved trajectory.

    Preferring these over rebuilding from `shape_library` matters twice over:
    the letter shapes are rasterised from system fonts that exist locally but
    not on every compute node (a missing NotoSerifCJK killed a full run), and
    reading the stored definitions guarantees the exact same targets the
    supervised model was trained on, which is what makes the comparison fair.
    """
    if not os.path.isfile(_DB_PATH):
        return {}, {}
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT shape_name, density_params, split FROM ergodic_pairs ORDER BY id ASC")
    defs, splits = {}, {}
    for name, params, split in cur.fetchall():
        defs[name] = json.loads(params)
        splits.setdefault(split, []).append(name)
    conn.close()
    return defs, splits


def prepare_targets(names, grid_res, K, use_cache=True, shape_defs=None):
    """Density grids and target coefficients phi_k for a list of shape names.

    phi_k is constant throughout training, and `pdf_on_grid` runs through JAX on
    the CPU, so this is computed once and cached — otherwise every job restart
    (and this job auto-resumes) would pay for it again.

    Shapes whose definition is in `shape_defs` are taken from there; anything
    else falls back to building it from `shape_library`, which may need a system
    font. A shape that cannot be built is skipped with a warning rather than
    killing the run.

    Returns (names_used, densities (S,R,R) float32, phi (S,M) float32).
    """
    source = 'db' if shape_defs else 'lib'
    path = _cache_path(names, grid_res, K, source)
    if use_cache and os.path.isfile(path):
        z = np.load(path)
        used = [str(n) for n in z['names']]
        print(f"  Targets loaded from cache: {os.path.basename(path)} ({len(used)} shapes)")
        return used, z['densities'], z['phi']

    k_idx_np, _ = make_k_grid(K)
    k_idx = torch.tensor(k_idx_np, dtype=torch.float64)

    used, dens_list, phi_list, skipped = [], [], [], []
    print(f"  Computing {len(names)} target densities (grid {grid_res}², source={source})...")
    from tqdm import tqdm
    for name in tqdm(names, unit='shape'):
        try:
            shape_def = shape_defs[name] if shape_defs and name in shape_defs \
                else get_shape(name)
            d_map, _, _ = pdf_on_grid(shape_def, resolution=grid_res)
        except Exception as e:
            skipped.append((name, type(e).__name__))
            continue
        d_map = np.asarray(d_map, dtype=np.float64)
        if d_map.max() > 0:
            d_map = d_map / d_map.max()
        phi = target_coeffs_from_grid(torch.tensor(d_map), k_idx)
        used.append(name)
        dens_list.append(d_map.astype(np.float32))
        phi_list.append(phi.numpy().astype(np.float32))

    if skipped:
        print(f"  [WARN] Skipped {len(skipped)} shape(s) that could not be built, "
              f"e.g. {skipped[:3]}")
    if not used:
        raise RuntimeError("No target shapes could be built.")

    densities, phi = np.stack(dens_list), np.stack(phi_list)
    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        # Plain unicode array, not object dtype — keeps the cache pickle-free.
        np.savez_compressed(path, densities=densities, phi=phi,
                            names=np.array(used, dtype='U64'))
        print(f"  Targets cached -> {os.path.basename(path)}")
    return used, densities, phi


def solver_reference_energy(names, phi, energy, device, nxi):
    """Energy of the solver's own trajectories, as a yardstick.

    Not supervision — the trajectories are only read to answer 'how good is the
    classical solver on these targets?', so the generator's numbers have a scale
    to be judged against.
    """
    if not os.path.isfile(_DB_PATH):
        return None
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    out = {}
    for i, name in enumerate(names):
        cur.execute("SELECT trajectory FROM ergodic_pairs WHERE shape_name=? LIMIT 1", (name,))
        row = cur.fetchone()
        if not row:
            continue
        xy = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 2)
        idx = np.linspace(0, len(xy) - 1, nxi).astype(int)
        cps = torch.tensor(xy[idx], dtype=torch.float32, device=device).unsqueeze(0)
        phi_t = torch.tensor(phi[i], dtype=torch.float32, device=device).unsqueeze(0)
        out[name] = energy(cps, phi_t).item()
    conn.close()
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
        'epoch': epoch, 'loss': loss,
        'nxi': args.nxi, 'nd': args.nd, 'D': args.D,
        'n_particles': args.n_particles, 'sample_mode': args.sample_mode,
        # Self-supervised settings, so a later resume or visualisation can tell
        # under which configuration this run was produced.
        'selfsupervised':   True,
        'n_candidates':     args.n_candidates,
        'diversity_weight': args.diversity_weight,
        'use_obstacle':     args.use_obstacle,
        'erg_K':            args.erg_K,
        'solver_T':         args.solver_T,
    }
    if getattr(args, 'use_wandb', False) and _WANDB_OK and wandb.run is not None:
        ckpt['wandb_id'] = wandb.run.id
    torch.save(ckpt, path)
    # Je Lauf bleibt genau ein Zwischenstand liegen.
    nach_zwischenstand(stem, args.run_str, path,
                       behalten=getattr(args, "keep_checkpoints", 1))
    return path


def visualise_set(model, names, densities, particles_map, title, save_path,
                  args, device, gt_map=None):
    """Holdout grid in the project style, reusing draw_panel/save_grid."""
    model.eval()
    panels = []
    with torch.no_grad():
        for i, name in enumerate(names):
            gen = model.generate(particles_map[name], num_samples=args.n_gen,
                                 device=str(device))
            panels.append(dict(
                label=f"'{name}'",
                base_cp=gt_map.get(name) if gt_map else None,
                gen_cps=gen.cpu().numpy(),
                density_grid=densities[i],
                particles=particles_map[name].cpu().numpy(),
            ))
    save_grid(panels, title, save_path, max_cols=5)
    model.train()


def _load_gt_for_reference(names, nxi):
    """Solver trajectories for the holdout shapes — drawn as a visual reference."""
    if not os.path.isfile(_DB_PATH):
        return {}
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    out = {}
    for name in names:
        cur.execute("SELECT trajectory FROM ergodic_pairs WHERE shape_name=? LIMIT 1", (name,))
        row = cur.fetchone()
        if row:
            xy = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 2)
            out[name] = xy[np.linspace(0, len(xy) - 1, nxi).astype(int)]
    conn.close()
    return out


# ===========================================================================
# Training
# ===========================================================================

def train(model, densities_t, phi_t, energy, args, device,
          holdout=None):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)
    model.train()

    S = densities_t.shape[0]
    start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed from epoch {start_epoch} (loss={ckpt['loss']:.4f})")

    import signal
    class TerminateInterrupt(Exception): pass
    def _sigterm_handler(signum, frame): raise TerminateInterrupt()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    from tqdm import tqdm
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="ep")
    ep, avg = start_epoch, 0.0
    first_energy, last_energy = None, None

    try:
        for ep in pbar:
            perm = torch.randperm(S, device=device)
            ep_loss, ep_parts, nb = 0.0, {}, 0

            for i in range(0, S, args.mini_batch):
                idx = perm[i:i + args.mini_batch]
                parts_cond = sample_particles(densities_t, idx, args.n_particles,
                                              device, mode=args.sample_mode)
                phi_batch = phi_t[idx]

                opt.zero_grad(set_to_none=True)
                loss, parts = compute_selfsupervised_loss(
                    model, parts_cond, phi_batch, energy,
                    n_candidates=args.n_candidates,
                    diversity_weight=args.diversity_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step()

                ep_loss += loss.item()
                for k, v in parts.items():
                    ep_parts[k] = ep_parts.get(k, 0.0) + v.item()
                nb += 1

            scheduler.step()
            avg = ep_loss / max(nb, 1)
            avg_parts = {k: v / max(nb, 1) for k, v in ep_parts.items()}
            lr_now = scheduler.get_last_lr()[0]

            if first_energy is None:
                first_energy = avg_parts.get('energy', avg)
            last_energy = avg_parts.get('energy', avg)

            post = {'loss': f"{avg:.2f}", 'E': f"{avg_parts.get('energy', 0):.2f}",
                    'lr': f"{lr_now:.2e}"}
            if 'diversity' in avg_parts:
                post['div'] = f"{avg_parts['diversity']:.3f}"
            pbar.set_postfix(**post)

            if getattr(args, 'use_wandb', False) and _WANDB_OK:
                log = {'train/loss': avg, 'train/lr': lr_now, 'epoch': ep + 1}
                for k, v in avg_parts.items():
                    log[f'loss/{k}'] = v
                if 'diversity' in avg_parts:
                    # Weighted contribution, so the ratio against the energy
                    # scale is visible while tuning diversity_weight.
                    log['loss/diversity_weighted'] = args.diversity_weight * avg_parts['diversity']
                wandb.log(log)

            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                _save_checkpoint(model, opt, scheduler, ep, avg, args)

            if (args.viz_every > 0 and (ep + 1) % args.viz_every == 0
                    and holdout is not None):
                out_dir = os.path.join(_here, 'Trajectory_data_generator')
                os.makedirs(out_dir, exist_ok=True)
                visualise_set(
                    model, holdout['names'], holdout['densities'],
                    holdout['particles'],
                    f"Self-supervised — Epoch {ep+1}",
                    os.path.join(out_dir, f"{args.run_str}_ep{ep+1:04d}.png"),
                    args, device, gt_map=holdout.get('gt'))

    except (KeyboardInterrupt, TerminateInterrupt):
        print(f"\n  [!] Interrupted at epoch {ep+1}. Saving emergency checkpoint...")
        print(f"  -> {_save_checkpoint(model, opt, scheduler, ep, avg, args)}")
        raise

    return first_energy, last_energy


# ===========================================================================
# Main
# ===========================================================================

def run(args):
    if not hasattr(args, 'timestamp'):
        args.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")
    args.run_str = (f"selfsup_particles_{args.timestamp}_nxi{args.nxi}_D{args.D}"
                    f"_N{args.n_particles}_K{args.n_candidates}"
                    f"_div{args.diversity_weight:g}")
    if args.use_obstacle:
        args.run_str += "_OBST"

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # ── Shapes ────────────────────────────────────────────────────────────────
    shape_defs, splits = load_shape_defs_from_db()
    if shape_defs:
        print(f"  Target definitions from DB: {len(shape_defs)} shapes "
              f"({', '.join(f'{k}={len(v)}' for k, v in splits.items())})")

    if args.shapes:
        train_names = [s.strip() for s in args.shapes.split(',') if s.strip()]
        # For an explicit shape list (the feasibility milestone) visualise those
        # very shapes — otherwise the run finishes without showing what it learnt.
        holdout_names = list(train_names)
    elif splits.get('train'):
        train_names = splits['train'][:args.n_train_shapes]
        holdout_names = list(splits.get('val', VALIDATION_SHAPES))
    else:
        train_names = train_shape_names(args.n_train_shapes)
        holdout_names = list(VALIDATION_SHAPES)

    print(f"\n{'=' * 70}")
    print(f"  Self-supervised particle generator (solver energy objective)")
    print(f"  device={device}  nxi={args.nxi}  D={args.D}  N={args.n_particles}")
    print(f"  shapes={len(train_names)}  K={args.n_candidates}  "
          f"div_weight={args.diversity_weight}  obstacle={args.use_obstacle}")
    print(f"  weights: erg={args.w_ergodic} smooth={args.w_smooth} "
          f"bound={args.w_boundary} obst={args.w_obstacle}")
    print(f"{'=' * 70}")

    train_names, dens_np, phi_np = prepare_targets(
        train_names, args.grid_res, args.erg_K,
        use_cache=not args.no_cache, shape_defs=shape_defs)
    densities_t = torch.tensor(dens_np, dtype=torch.float32, device=device)
    phi_t = torch.tensor(phi_np, dtype=torch.float32, device=device)

    # ── Energy ────────────────────────────────────────────────────────────────
    basis = np.array(BsplineBasisClamped(
        degree=args.bspline_deg, num_control_points=args.nxi,
        num_phase_points=args.solver_T, compute_derivatives=False).B)
    energy = ErgodicEnergy(
        K=args.erg_K, basis=torch.from_numpy(basis).float(),
        use_obstacle=args.use_obstacle,
        w_ergodic=args.w_ergodic, w_smooth=args.w_smooth,
        w_boundary=args.w_boundary, w_obstacle=args.w_obstacle,
    ).to(device)

    ref = solver_reference_energy(train_names[:5], phi_np, energy, device, args.nxi)
    if ref:
        print("  Solver reference energy (its own trajectories, as a yardstick):")
        for k, v in ref.items():
            print(f"    {k:<24} {v:10.2f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SelfSupervisedParticleGenerator(
        nxi=args.nxi, nd=args.nd, D=args.D).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Holdout for visualisation ─────────────────────────────────────────────
    holdout = None
    if holdout_names:
        holdout_names, h_dens, h_phi = prepare_targets(
            holdout_names, args.grid_res, args.erg_K,
            use_cache=not args.no_cache, shape_defs=shape_defs)
        h_dens_t = torch.tensor(h_dens, dtype=torch.float32, device=device)
        h_particles = {}
        for i, name in enumerate(holdout_names):
            idx_t = torch.tensor([i], dtype=torch.long, device=device)
            h_particles[name] = sample_particles(h_dens_t, idx_t, args.n_particles,
                                                 device, mode=args.sample_mode)[0]
        holdout = {'names': holdout_names, 'densities': h_dens,
                   'particles': h_particles,
                   'gt': _load_gt_for_reference(holdout_names, args.nxi)}

    if args.use_wandb and _WANDB_OK:
        resume_id = None
        if args.resume and os.path.isfile(args.resume):
            try:
                resume_id = torch.load(args.resume, map_location='cpu',
                                       weights_only=True).get('wandb_id')
            except Exception:
                pass
        wandb.init(project=args.wandb_project,
                   name=args.run_name or args.run_str, config=vars(args),
                   id=resume_id, resume="allow" if resume_id else None)
    elif args.use_wandb:
        print("  Warning: wandb not installed. Skipping.")

    e0, e1 = train(model, densities_t, phi_t, energy, args, device, holdout=holdout)

    if args.use_wandb and _WANDB_OK:
        wandb.finish()

    stem = os.path.splitext(args.save_model)[0]
    final_path = f"{stem}_{args.run_str}_final.pt"
    os.makedirs(os.path.dirname(final_path) or '.', exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(),
                'nxi': args.nxi, 'nd': args.nd, 'D': args.D,
                'n_particles': args.n_particles, 'selfsupervised': True,
                'n_candidates': args.n_candidates,
                'diversity_weight': args.diversity_weight,
                'use_obstacle': args.use_obstacle, 'erg_K': args.erg_K,
                'solver_T': args.solver_T}, final_path)
    print(f"\n  Training complete. Checkpoint -> {final_path}")

    if holdout is not None:
        out_dir = os.path.join(_here, 'Trajectory_data_generator')
        visualise_set(model, holdout['names'], holdout['densities'],
                      holdout['particles'], 'HELD-OUT Shapes (self-supervised)',
                      os.path.join(out_dir, f"{args.run_str}_holdout.png"),
                      args, device, gt_map=holdout.get('gt'))

    # ── Feasibility gate ──────────────────────────────────────────────────────
    print(f"\n  Energy: {e0:.3f} (first epoch) -> {e1:.3f} (last epoch)")
    if args.assert_energy_drops:
        if not (e1 < e0 * args.min_drop_ratio):
            print(f"  [FAIL] Energy did not drop below {args.min_drop_ratio:g}x "
                  f"its starting value — the objective is not being minimised.")
            return 1
        print(f"  [PASS] Energy dropped to {e1 / max(e0, 1e-9):.1%} of its start.")
    return 0


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Architecture
    p.add_argument('--nxi', type=int, default=25)
    p.add_argument('--nd',  type=int, default=2)
    p.add_argument('--D',   type=int, default=384)
    p.add_argument('--n_particles', type=int, default=256)
    p.add_argument('--bspline_deg', type=int, default=5)

    # Targets
    p.add_argument('--shapes', type=str, default=None,
                   help='Comma-separated shape names. Default: the 750-shape train split.')
    p.add_argument('--n_train_shapes', type=int, default=750)
    p.add_argument('--grid_res', type=int, default=128)
    p.add_argument('--sample_mode', type=str, default='uniform',
                   choices=['uniform', 'density'])
    p.add_argument('--no_cache', action='store_true',
                   help='Recompute target densities instead of using the .npz cache.')

    # Self-supervised objective
    p.add_argument('--n_candidates', type=int, default=8,
                   help='K trajectories generated per target from different noise.')
    p.add_argument('--diversity_weight', type=float, default=0.0,
                   help='Weight of the repulsion reward. Must be on the order of '
                        'the energy scale (tens to hundreds) to have any effect.')
    p.add_argument('--use_obstacle', action='store_true',
                   help='Add the obstacle term to the energy (solver default: off).')
    p.add_argument('--erg_K', type=int, default=K_DEFAULT,
                   help='Frequency grid is erg_K x erg_K (solver uses 10).')
    p.add_argument('--solver_T', type=int, default=SOLVER_T,
                   help='Points the B-spline is rendered to before the energy is '
                        'evaluated. The solver optimises 100.')
    p.add_argument('--w_ergodic',  type=float, default=W_ERGODIC)
    p.add_argument('--w_smooth',   type=float, default=W_SMOOTH)
    p.add_argument('--w_boundary', type=float, default=W_BOUNDARY)
    p.add_argument('--w_obstacle', type=float, default=W_OBSTACLE)

    # Optimisation
    p.add_argument('--epochs',     type=int,   default=500)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--mini_batch', type=int,   default=32,
                   help='Targets per step; effective batch is this times K.')
    p.add_argument('--clip_grad',  type=float, default=1.0)
    p.add_argument('--seed',       type=int,   default=0)

    # Bookkeeping
    p.add_argument('--keep_checkpoints', type=int, default=1,
                   help='Wie viele Zwischenstaende eines Laufs behalten werden. '
                        'Vorgabe 1: beim Speichern eines neuen Standes werden '
                        'aeltere entfernt, und nach dem Endstand alle.')
    p.add_argument('--save_model', type=str, default='checkpoints/selfsup.pt')
    p.add_argument('--resume',     type=str, default=None)
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--viz_every',  type=int, default=20)
    p.add_argument('--n_gen',      type=int, default=5)
    p.add_argument('--device',     type=str, default=None)
    p.add_argument('--use_wandb',  action='store_true')
    p.add_argument('--wandb_project', type=str, default='flow-matching')
    p.add_argument('--run_name',   type=str, default=None)

    # Feasibility gate
    p.add_argument('--assert_energy_drops', action='store_true',
                   help='Exit non-zero if the energy did not fall — used to gate '
                        'the full run behind a short feasibility check.')
    p.add_argument('--min_drop_ratio', type=float, default=0.9,
                   help='Final energy must be below this fraction of the first.')
    return p.parse_args()


if __name__ == '__main__':
    sys.exit(run(parse_args()))
