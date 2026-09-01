#!/usr/bin/env python3
r"""
experiment_3d_lift.py
======================
Prototype: take a 2D ergodic B-spline path from the trained particle-
conditioned flow-matching model and lift it, together with its target
density, onto a 3D shape (sphere / pyramid / torus) with an inference-time
attraction force -- see shapes_3d.py. The model itself is untouched and never
sees the shape, exactly like the existing obstacle-repulsion guidance in
obstacles.py, only pulling instead of pushing.

The density texture is lifted with the identical force as the path (just
applied to raw points instead of B-spline control points), so the rendered
panel gives a direct visual check of whether the green generated path
actually tracks the high-density (bright) region of the target shape once
both live on the same 3D surface.

Usage:
    python experiment_3d_lift.py [--shape A] [--checkpoint path/to/ckpt.pt]
"""
import argparse
import importlib
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src'), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(os.path.join(_here, 'ergodic_dataset_generator'))

from shape_library import get_shape, pdf_on_grid
from model_zoo import load_model
from flow_matching_runner_particles import sample_particles
from obstacles import basis_torch, bspline_basis_matrix
from shapes_3d import SphereShape, PyramidShape, TorusShape, lift_curve_to_shape, lift_points_to_shape
from visualize_checkpoint import WHITE_INFERNO

_DEFAULT_CKPT = os.path.join(
    _here, 'exploration', 'modelle_und_Datenbank',
    'cond_particles_crossattn_flow_matching_particle_ergodic_date_08_28_09h13min_'
    'nxi64_D384_N256_C2_flip0.0_START_FLAT7540_LEN-pd0.1_LINEARFREQ_LR1E4_'
    'ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt')


def cp_to_curve(cps, pts=512, deg=5):
    """cps: (nxi, nd) numpy -> (pts, nd) numpy, dense B-spline curve."""
    B = bspline_basis_matrix(cps.shape[0], pts, deg)
    return B @ cps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=_DEFAULT_CKPT)
    p.add_argument('--shape', default='A', help='shape_library density name.')
    p.add_argument('--grid_res', type=int, default=64)
    p.add_argument('--tex_res', type=int, default=110,
                   help='Resolution of the density grid lifted as a surface texture.')
    p.add_argument('--tex_threshold', type=float, default=0.04,
                   help='Drop texture points below this normalised density (keeps the plot legible).')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default=os.path.join(_here, 'viz_3d_lift'))
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, kind, meta = load_model(args.checkpoint, device)
    assert kind == 'flow', f"expected a flow checkpoint, got kind={kind!r}"
    nxi = meta['nxi']
    print(f"Loaded checkpoint: nxi={nxi} D={meta['D']} n_particles={meta['n_particles']} "
          f"epoch={meta['epoch']} length_cond={meta['length_cond']}")

    d_map, _, _ = pdf_on_grid(get_shape(args.shape), resolution=args.grid_res)
    if d_map.max() > 0:
        d_map = d_map / d_map.max()
    dens_t = torch.tensor(d_map, dtype=torch.float32, device=device).unsqueeze(0)
    idx_t = torch.tensor([0], dtype=torch.long, device=device)
    particles = sample_particles(dens_t, idx_t, meta['n_particles'], device, mode='uniform')[0]

    gen_mod = importlib.import_module(meta['_module'])
    generate_particle_trajectories = gen_mod.generate_particle_trajectories

    g = torch.Generator(device=device).manual_seed(args.seed)
    cps2d, _ = generate_particle_trajectories(
        model, particles, num_samples=1, nxi=nxi, nd=meta['nd'], steps=args.steps,
        device=str(device), cfg_weight=meta['cfg_weight'], generator=g)
    print(f"Generated 2D path: {tuple(cps2d.shape)}  "
          f"x=[{cps2d[..., 0].min():.2f},{cps2d[..., 0].max():.2f}]  "
          f"y=[{cps2d[..., 1].min():.2f},{cps2d[..., 1].max():.2f}]")

    B_basis = basis_torch(nxi, 256, 5, device=device)

    # Finer density grid for the surface texture, thresholded so the plot
    # shows the (thin) target-density stroke rather than thousands of
    # near-zero background points.
    d_tex, gx, gy = pdf_on_grid(get_shape(args.shape), resolution=args.tex_res)
    d_tex_n = d_tex / d_tex.max() if d_tex.max() > 0 else d_tex
    mask = d_tex_n > args.tex_threshold
    tex_xy = np.stack([gx[mask], gy[mask]], axis=-1)
    tex_vals = d_tex_n[mask]
    P2d_tex = torch.tensor(tex_xy, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"Density texture: {tex_xy.shape[0]} points above threshold "
          f"{args.tex_threshold} (of {d_tex_n.size} grid cells)")

    # Nearest-point projection can only ever land on the side of the shape
    # closest to where the points started -- it has no notion of "draping
    # over the top". Starting the lift's z at 1.0, above every shape (all
    # sized to sit within roughly z in [0.05, 0.85]), makes the *upper* /
    # angled surface the nearest one for essentially every (x, y), instead of
    # collapsing onto whichever patch happens to be closest to a z=0 start.
    shapes = {
        'Sphere':  SphereShape(center=(0.5, 0.5, 0.45), radius=0.40),
        'Pyramid': PyramidShape(center=(0.5, 0.5), half_base=0.32, z0=0.08, height=0.65),
        'Torus':   TorusShape(center=(0.5, 0.5, 0.4), R=0.32, r=0.15),
    }
    LIFT_KWARGS = dict(z_iters=20, z_lr=0.6, polish_iters=350, polish_lr=1.0, z_init=1.0)

    fig = plt.figure(figsize=(16, 5.5), facecolor='white')
    for i, (name, shape) in enumerate(shapes.items()):
        cps3 = lift_curve_to_shape(cps2d, shape, B_basis, **LIFT_KWARGS)
        curve3 = torch.einsum('pi,bid->bpd', B_basis, cps3)
        final_sdf = shape.sdf(curve3).abs().max().item()

        tex3 = lift_points_to_shape(P2d_tex, shape, **LIFT_KWARGS)
        tex_sdf = shape.sdf(tex3).abs().max().item()
        print(f"  [{name}] max |SDF| after lift -- path: {final_sdf:.4f}  "
              f"density texture: {tex_sdf:.4f}")

        curve_np = cp_to_curve(cps3[0].detach().cpu().numpy())
        tex3_np = tex3[0].detach().cpu().numpy()

        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.set_facecolor('white')
        shape.draw(ax)
        ax.scatter(tex3_np[:, 0], tex3_np[:, 1], tex3_np[:, 2],
                   c=tex_vals, cmap=WHITE_INFERNO, vmin=0, vmax=1,
                   s=6, alpha=0.6, edgecolors='none', zorder=3)
        ax.plot(curve_np[:, 0], curve_np[:, 1], curve_np[:, 2],
                color='#00C853', lw=2.2, alpha=0.95, zorder=5)
        cps3_np = cps3[0].detach().cpu().numpy()
        ax.scatter(cps3_np[:, 0], cps3_np[:, 1], cps3_np[:, 2],
                   color='#00C853', s=10, alpha=0.7, zorder=6)
        ax.set_title(f"{name}  (max|SDF| path={final_sdf:.3f}, density={tex_sdf:.3f})",
                    fontsize=10, color='#1A1A2E')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(labelsize=6, colors='#555')
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor('white')

    fig.suptitle(f"Target density '{args.shape}' + generated path lifted onto 3D shapes "
                f"via inference-time attraction force", fontsize=12, color='#1A1A2E')
    plt.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%Hh%Mmin')
    out_path = os.path.join(args.out_dir, f'lift3d_{args.shape}_{stamp}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {out_path}")

    # Flat top-down sanity check: same density + path, no lift at all -- isolates
    # whether a path/density mismatch comes from the 3D lift or already exists in
    # the raw 2D generation (phase 1 of the lift only ever moves z, so it cannot
    # explain an (x, y) mismatch either way).
    curve2d_np = cp_to_curve(cps2d[0].detach().cpu().numpy())
    fig2, ax2 = plt.subplots(figsize=(5.5, 5.5), facecolor='white')
    ax2.set_facecolor('white')
    ax2.imshow(d_tex_n, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO,
              vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)
    ax2.plot(curve2d_np[:, 0], curve2d_np[:, 1], color='#00C853', lw=2.2, alpha=0.95, zorder=3)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect('equal')
    ax2.set_title(f"Raw 2D generation, no lift  (shape '{args.shape}')", fontsize=10, color='#1A1A2E')
    for spine in ax2.spines.values():
        spine.set_color('#ccc')
    ax2.grid(True, alpha=0.2, lw=0.4, color='gray')
    out_path_2d = os.path.join(args.out_dir, f'flat2d_{args.shape}_{stamp}.png')
    plt.savefig(out_path_2d, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved -> {out_path_2d}")


if __name__ == '__main__':
    main()
