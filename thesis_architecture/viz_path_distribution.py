#!/usr/bin/env python3
"""
viz_path_distribution.py
=========================
For ONE holdout datapoint, sample K trajectories from a trained flow-matching
generator (same conditioning, K different noise seeds) and overlay them to
show the *learned distribution* over paths -- as opposed to
visualize_checkpoint.py's default best-of-n grid over all holdout shapes,
which shows only the single winning sample per shape.

Reuses model_zoo.load_model/generate, visualize_checkpoint.load_holdout_shapes/
draw_panel, flow_matching_runner_particles.sample_particles, and
ergodic_energy_torch.ErgodicEnergy -- nothing new is implemented here beyond a
pairwise-curve-distance diversity metric, since nothing in the repo currently
quantifies sample-to-sample spread (only sample-vs-target quality).

Usage:
  python viz_path_distribution.py --checkpoint exploration/modelle_und_Datenbank/cond_particles_crossattn_..._final.pt --shape A --n_samples 14
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, 'ergodic_dataset_generator'),
           os.path.join(os.path.dirname(_here), 'bsplinax-main'),
           os.path.join(os.path.dirname(_here), 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_zoo import load_model, generate, describe
from visualize_checkpoint import load_holdout_shapes, draw_panel
from flow_matching_runner_particles import sample_particles
from ergodic_energy_torch import (
    ErgodicEnergy, make_k_grid, target_coeffs_from_grid,
    coverage_distance, path_length, K_DEFAULT,
)
from bsplinax.bspline import BsplineBasisClamped


def render_path_distribution(ckpt_path, shape_name, n_samples=14, steps=100,
                             seed=0, grid_res=96, length_cfg_weight=2.0,
                             out_dir=None, device=None):
    device = torch.device(device if device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, kind, meta = load_model(ckpt_path, device)
    nxi = meta['nxi']
    print(f"Loaded {os.path.basename(ckpt_path)}  kind={kind}  nxi={nxi}  "
          f"D={meta['D']}  length_cond={meta['length_cond']}  epoch={meta['epoch']}")

    shapes, densities = load_holdout_shapes(nxi, grid_res=grid_res)
    if shape_name not in shapes:
        raise SystemExit(f"'{shape_name}' not in holdout split. Available: {sorted(shapes.keys())}")
    d_map = densities[shape_name]
    base_cp = shapes[shape_name]

    dens_t = torch.tensor(d_map, dtype=torch.float32, device=device)
    idx_t = torch.tensor([0], dtype=torch.long, device=device)
    particles = sample_particles(dens_t.unsqueeze(0), idx_t, meta['n_particles'], device, mode='uniform')[0]

    # Ground-truth arclength for this holdout shape, so a length-conditioned
    # checkpoint is asked for the same length its supervised counterpart would
    # have seen -- otherwise its CFG length term pulls samples toward an
    # unrelated default length instead of the one appropriate for this shape.
    gt_len = float(path_length(torch.tensor(base_cp, dtype=torch.float32, device=device).unsqueeze(0)).item())

    cps, secs = generate(model, kind, particles, n_samples, meta, steps, device, seed,
                         length=gt_len if meta.get('length_cond') else None,
                         length_cfg_weight=length_cfg_weight if meta.get('length_cond') else 0.0)
    print(f"Sampled {n_samples} trajectories in {secs:.2f}s")

    # ── Per-sample scoring (spread quantified, not just eyeballed) ──────────
    basis = torch.tensor(np.array(BsplineBasisClamped(
        degree=5, num_control_points=nxi, num_phase_points=100,
        compute_derivatives=False).B), dtype=torch.float32, device=device)
    energy = ErgodicEnergy(K=K_DEFAULT, basis=basis).to(device)
    k_idx = torch.tensor(make_k_grid(K_DEFAULT)[0], dtype=torch.float64)
    phi = torch.tensor(target_coeffs_from_grid(torch.tensor(d_map, dtype=torch.float64), k_idx).numpy(),
                       dtype=torch.float32, device=device).unsqueeze(0)
    curves = torch.einsum('ti,bid->btd', basis, cps)
    _, terms = energy(cps, phi.expand(cps.shape[0], -1), return_terms=True)
    E_erg = terms['ergodic'].detach().cpu().numpy()
    cov = coverage_distance(curves, dens_t).detach().cpu().numpy()
    plen = path_length(curves).detach().cpu().numpy()

    # Pairwise spread between the K sampled curves themselves (mean, over
    # sample pairs, of the mean pointwise distance at matched arclength
    # parametrisation) -- the actual "width" of the learned distribution in
    # path-space: it goes to zero under mode collapse.
    curves_np = curves.detach().cpu().numpy()
    pairwise = np.linalg.norm(curves_np[:, None] - curves_np[None, :], axis=-1).mean(-1)
    off_diag = pairwise[~np.eye(n_samples, dtype=bool)]

    print(f"E_ergodic:  mean={E_erg.mean():.3f}  std={E_erg.std():.3f}  min={E_erg.min():.3f}  max={E_erg.max():.3f}")
    print(f"coverage:   mean={cov.mean():.4f}  std={cov.std():.4f}")
    print(f"path_len:   mean={plen.mean():.2f}  std={plen.std():.2f}  (GT={gt_len:.2f})")
    print(f"pairwise curve distance (diversity): mean={off_diag.mean():.4f}  std={off_diag.std():.4f}")

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    draw_panel(ax, base_cp=base_cp, gen_cps=cps.cpu().numpy(), density_grid=d_map,
              particles=particles.cpu().numpy(),
              title=(f"Learned path distribution — shape '{shape_name}'  "
                     f"({n_samples} samples, {describe(kind, meta)})\n"
                     f"E_erg={E_erg.mean():.2f}±{E_erg.std():.2f}   "
                     f"cov={cov.mean():.3f}±{cov.std():.3f}   "
                     f"len={plen.mean():.1f}±{plen.std():.1f} (GT {gt_len:.1f})   "
                     f"diversity={off_diag.mean():.3f}"))
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    fig.tight_layout()

    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = out_dir or os.path.join(_here, 'Trajectory_data_generator', 'viz_rerun')
    os.makedirs(out_dir, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime('%Y%m%d_%Hh%Mmin')
    out_path = os.path.join(out_dir, f"viz_{stem}_pathdistribution_shape{shape_name}_K{n_samples}_{stamp}.png")
    fig.savefig(out_path, dpi=180, facecolor='white')
    print(f"Saved: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description='Overlay K sampled trajectories for one holdout shape.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--shape', required=True, help="Holdout shape name, e.g. 'A'.")
    p.add_argument('--n_samples', type=int, default=14)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--grid_res', type=int, default=96)
    p.add_argument('--length_cfg_weight', type=float, default=2.0)
    p.add_argument('--out_dir', default=None)
    p.add_argument('--device', default=None)
    args = p.parse_args()
    render_path_distribution(
        args.checkpoint, args.shape, n_samples=args.n_samples, steps=args.steps,
        seed=args.seed, grid_res=args.grid_res, length_cfg_weight=args.length_cfg_weight,
        out_dir=args.out_dir, device=args.device)


if __name__ == '__main__':
    main()
