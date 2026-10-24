#!/usr/bin/env python3
"""
evaluate_holdout.py
====================
Standalone holdout evaluation for a trained Ranking-NCE energy-based policy
checkpoint (`train_ebm_particles_start.py`), analogous in spirit to the
project's `visualize_checkpoint.py` convention: regenerate holdout metrics
and plots from any `.pt` file without re-running training.

The EBM is not itself a trajectory generator (unlike CFM), so "holdout
evaluation" here means three things, all computed only on the DB's held-out
`split='val'` shapes (never seen during training):

1. Ranking accuracy: does the energy still rank the true trajectory below
   Gaussian noise negatives on *unseen* target densities, or did it just
   memorize the training shapes?
2. Refinement quality: starting from a corrupted ground truth (a stand-in for
   an imperfect/out-of-distribution CFM sample) at two noise scales, does
   gradient-based (Langevin) and derivative-free (CEM) refinement actually
   move the trajectory closer to the true one and lower its energy?
3. A qualitative figure in the project's house style (WHITE_INFERNO density,
   deep-blue ground truth, neon-green refined trajectory) for a handful of
   holdout shapes.

Usage:
  python3 -u evaluate_holdout.py --checkpoint checkpoints/<run>_final.pt
"""

import argparse, json, os, sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..', '..', '..')
_thesis_arch = os.path.join(_root, 'thesis_architecture')
for _p in (_here, _thesis_arch, os.path.join(_root, 'bsplinax-main')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bsplinax.bspline import BsplineBasisClamped
from ebm_ranking_nce_core import EnergyNet, ranking_nce_loss, langevin_refine, cem_optimize
from train_ebm_particles_start import _load_shapes, sample_particles, _DB_PATH

# ── Shared white-inferno colormap and trajectory colors, matching the
#    project-wide visualization style guide exactly ──────────────────────
_inferno_colors = plt.colormaps['inferno'](np.linspace(0.0, 1.0, 256))
_n_white = 40
for _i in range(_n_white):
    _t = _i / _n_white
    _inferno_colors[_i] = (1 - _t) * np.array([1, 1, 1, 1]) + _t * _inferno_colors[_n_white]
WHITE_INFERNO = mcolors.LinearSegmentedColormap.from_list('white_inferno', _inferno_colors)


def cp_to_bspline(cps, pts=512, deg=5):
    nxi = cps.shape[0]
    B = np.array(BsplineBasisClamped(
        degree=deg, num_control_points=nxi,
        num_phase_points=pts, compute_derivatives=False).B)
    return B @ cps


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = EnergyNet(context_mode='particles', nxi=ckpt['nxi'], nd=ckpt['nd'], D=ckpt['D'],
                       use_sigma_cond=True, use_start_cond=ckpt.get('use_start_cond', True)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


# ----------------------------------------------------------------------------
# Quantitative metrics
# ----------------------------------------------------------------------------

@torch.no_grad()
def eval_ranking_accuracy(model, particles, gt, start, n_negatives, sigma_min, sigma_max, n_resamples=10):
    accs = []
    for _ in range(n_resamples):
        _, metrics = ranking_nce_loss(model, particles, gt, k_indices=None,
                                       n_negatives=n_negatives, sigma_min=sigma_min,
                                       sigma_max=sigma_max, start=start)
        accs.append(metrics['acc'])
    return float(np.mean(accs))


def eval_refinement(model, particles, gt, start, corrupt_sigma, n_steps, step_size, seed):
    torch.manual_seed(seed)
    noise = torch.randn_like(gt)
    corrupted = gt + corrupt_sigma * noise

    with torch.no_grad():
        ctx_emb = model.encode_context(particles)
        e_before = model(ctx_emb, corrupted, start=start)
    refined, history = langevin_refine(model, particles, corrupted, k_indices=None,
                                        n_steps=n_steps, step_size=step_size,
                                        noise_scale=0.0, start=start)

    dist_before = (corrupted - gt).pow(2).sum(dim=(-2, -1)).sqrt()
    dist_after = (refined - gt).pow(2).sum(dim=(-2, -1)).sqrt()
    return {
        'corrupted': corrupted, 'refined': refined,
        'dist_before': dist_before, 'dist_after': dist_after,
        'energy_before': e_before, 'energy_after': history[-1],
        'frac_dist_improved': (dist_after < dist_before).float().mean().item(),
        'frac_energy_improved': (history[-1] < history[0]).float().mean().item(),
    }


def eval_cem(model, particles, gt, start, corrupt_sigma, n_iters, n_samples, init_std, seed):
    torch.manual_seed(seed)
    noise = torch.randn_like(gt)
    corrupted = gt + corrupt_sigma * noise
    B, nxi, nd = gt.shape

    with torch.no_grad():
        ctx_emb = model.encode_context(particles)
        e_before = model(ctx_emb, corrupted, start=start)
    best_action, best_energy = cem_optimize(model, particles, nxi, nd, k_indices=None,
                                             n_iters=n_iters, n_samples=n_samples, elite_frac=0.1,
                                             init_mean=corrupted.reshape(B, -1), init_std=init_std,
                                             start=start)
    dist_before = (corrupted - gt).pow(2).sum(dim=(-2, -1)).sqrt()
    dist_after = (best_action - gt).pow(2).sum(dim=(-2, -1)).sqrt()
    return {
        'dist_before': dist_before, 'dist_after': dist_after,
        'energy_before': e_before, 'energy_after': best_energy,
        'frac_dist_improved': (dist_after < dist_before).float().mean().item(),
    }


# ----------------------------------------------------------------------------
# Visualization (house style: white_inferno density, deep-blue ground truth,
# neon-green refined trajectory, dark particle dots)
# ----------------------------------------------------------------------------

def draw_shape(ax, gt_cp, refined_cp, corrupted_cp, particles, density_grid, title,
               bspline_pts=512, bspline_deg=5):
    ax.set_facecolor('white')

    d = density_grid.copy()
    if d.max() > 0:
        d /= d.max()
    ax.imshow(d, extent=[0, 1, 0, 1], origin='lower',
              cmap=WHITE_INFERNO, vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)

    ax.scatter(particles[:, 0], particles[:, 1], c='#444444', s=6, alpha=0.3,
               zorder=1, edgecolors='none')

    if corrupted_cp is not None and len(corrupted_cp) >= 6:
        ax.plot(*cp_to_bspline(corrupted_cp, bspline_pts, bspline_deg).T,
                color='#999999', lw=1.5, ls='--', alpha=0.8,
                label='Corrupted warm start', zorder=2)

    if len(gt_cp) >= 6:
        ax.plot(*cp_to_bspline(gt_cp, bspline_pts, bspline_deg).T,
                color='#1565C0', lw=2.5, alpha=0.9, label='Ground truth', zorder=3)

    if refined_cp is not None and len(refined_cp) >= 6:
        ax.plot(*cp_to_bspline(refined_cp, bspline_pts, bspline_deg).T,
                color='#00C853', lw=2.2, alpha=0.95, label='EBM-refined', zorder=4)

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


def make_figure(labels, gt_dict, refined_dict, corrupted_dict, particles_dict, density_dict,
                save_path, max_cols=3):
    n = len(labels)
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5 * n_rows),
                              facecolor='white', squeeze=False)
    fig.suptitle('EBM Ranking-NCE: Holdout Refinement (corrupted warm start -> refined)',
                 fontsize=13, fontweight='bold', color='#1A1A2E', y=1.01)

    for idx, lbl in enumerate(labels):
        ax = axes[idx // n_cols][idx % n_cols]
        draw_shape(ax, gt_dict[lbl], refined_dict[lbl], corrupted_dict[lbl],
                  particles_dict[lbl], density_dict[lbl], f"'{lbl}'")
        if idx == 0:
            ax.legend(frameon=True, fontsize=7, loc='upper right',
                      facecolor='white', edgecolor='#ddd', framealpha=0.9)

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = load_model(args.checkpoint, device)
    print(f"  Loaded checkpoint: {args.checkpoint}")
    print(f"  nxi={ckpt['nxi']} nd={ckpt['nd']} D={ckpt['D']} n_particles={ckpt['n_particles']}")

    _, val_shapes, _, val_densities = _load_shapes(args.db, ckpt['nxi'], grid_res=128)
    print(f"  Loaded {len(val_shapes)} holdout ('val') shapes from {args.db}")

    labels = list(val_shapes.keys())
    density_grids = torch.stack([torch.tensor(val_densities[l], dtype=torch.float32) for l in labels]).to(device)
    gt = torch.tensor(np.stack([val_shapes[l] for l in labels]), dtype=torch.float32).to(device)
    shape_indices = torch.arange(len(labels), device=device)

    torch.manual_seed(args.seed)
    particles = sample_particles(density_grids, shape_indices, ckpt['n_particles'], device,
                                 mode=ckpt['sample_mode'])
    start = gt[:, 0, :]

    # ── 1. Ranking accuracy on unseen shapes ────────────────────────────────
    acc = eval_ranking_accuracy(model, particles, gt, start, ckpt['n_negatives'],
                                ckpt['sigma_min'], ckpt['sigma_max'])
    print(f"\n  [1] Ranking accuracy on {len(labels)} holdout shapes: {acc:.3f}")

    # ── 2. Refinement quality at two corruption scales ──────────────────────
    results = {'ranking_accuracy_holdout': acc}
    for tag, sigma in (('mild', args.corrupt_sigma_mild), ('severe', args.corrupt_sigma_severe)):
        r = eval_refinement(model, particles, gt, start, sigma,
                            n_steps=args.langevin_steps, step_size=args.langevin_step_size, seed=args.seed)
        print(f"\n  [2.{tag}] Langevin refinement, corruption sigma={sigma}")
        print(f"      dist   : {r['dist_before'].mean():.4f} -> {r['dist_after'].mean():.4f} "
              f"(improved in {r['frac_dist_improved']*100:.0f}% of shapes)")
        print(f"      energy : {r['energy_before'].mean():.3f} -> {r['energy_after'].mean():.3f} "
              f"(improved in {r['frac_energy_improved']*100:.0f}% of shapes)")
        results[f'langevin_{tag}'] = {
            'corrupt_sigma': sigma,
            'dist_before': r['dist_before'].mean().item(), 'dist_after': r['dist_after'].mean().item(),
            'energy_before': r['energy_before'].mean().item(), 'energy_after': r['energy_after'].mean().item(),
            'frac_dist_improved': r['frac_dist_improved'], 'frac_energy_improved': r['frac_energy_improved'],
        }
        if tag == 'mild':
            refined_mild, corrupted_mild = r['refined'], r['corrupted']

    cem_r = eval_cem(model, particles, gt, start, args.corrupt_sigma_mild,
                     n_iters=args.cem_iters, n_samples=args.cem_samples,
                     init_std=args.cem_init_std, seed=args.seed)
    print(f"\n  [2.cem] CEM local search, corruption sigma={args.corrupt_sigma_mild}")
    print(f"      dist   : {cem_r['dist_before'].mean():.4f} -> {cem_r['dist_after'].mean():.4f} "
          f"(improved in {cem_r['frac_dist_improved']*100:.0f}% of shapes)")
    results['cem_mild'] = {
        'dist_before': cem_r['dist_before'].mean().item(), 'dist_after': cem_r['dist_after'].mean().item(),
        'frac_dist_improved': cem_r['frac_dist_improved'],
    }

    # ── 3. Qualitative figure for a handful of holdout shapes ───────────────
    n_viz = min(args.n_viz, len(labels))
    viz_labels = labels[:n_viz]
    gt_np = {l: val_shapes[l] for l in viz_labels}
    refined_np = {l: refined_mild[i].detach().cpu().numpy() for i, l in enumerate(labels) if l in viz_labels}
    corrupted_np = {l: corrupted_mild[i].detach().cpu().numpy() for i, l in enumerate(labels) if l in viz_labels}
    particles_np = {l: particles[i].detach().cpu().numpy() for i, l in enumerate(labels) if l in viz_labels}
    density_np = {l: val_densities[l] for l in viz_labels}

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    fig_path = os.path.join(args.out_dir, f"holdout_{stem}.png")
    make_figure(viz_labels, gt_np, refined_np, corrupted_np, particles_np, density_np, fig_path)
    print(f"\n  Figure saved -> {fig_path}")

    summary_path = os.path.join(args.out_dir, f"holdout_{stem}.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Summary saved -> {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--db', type=str, default=_DB_PATH)
    p.add_argument('--out_dir', type=str, default=os.path.join(_here, 'results'))
    p.add_argument('--n_viz', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--corrupt_sigma_mild', type=float, default=0.3,
                   help="Stand-in for a decent-but-imperfect CFM sample.")
    p.add_argument('--corrupt_sigma_severe', type=float, default=0.8,
                   help="Stand-in for a poor/out-of-distribution CFM sample.")
    p.add_argument('--langevin_steps', type=int, default=30)
    p.add_argument('--langevin_step_size', type=float, default=5e-3)
    p.add_argument('--cem_iters', type=int, default=10)
    p.add_argument('--cem_samples', type=int, default=256)
    p.add_argument('--cem_init_std', type=float, default=0.4)

    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
