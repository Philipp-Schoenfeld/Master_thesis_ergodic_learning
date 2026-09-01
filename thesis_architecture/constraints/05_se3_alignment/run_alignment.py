#!/usr/bin/env python3
r"""
run_alignment.py
================
Constraint 5 across the whole holdout split, on all three solids.

For every holdout density the 2D path is generated once, lifted onto each
solid twice -- position-only (the baseline the 3D-lift experiment used) and
with the tangent-alignment term added -- and both are scored on

    * surface adherence   max |SDF| along the *dense* curve,
    * tangency            mean |cos(t, n)|,
    * footprint coverage  coverage_distance of the (x, y) projection against
      the original 2D target density, i.e. did the extra constraint cost us
      the ergodic footprint the model was asked for.

    python run_alignment.py [--shapes A digit_5] [--solids Sphere Pyramid Torus]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (pick_device, load_generator, density_and_particles, basis_torch,
                    guided_generate, curve_energy_grad, curve_of, cp_to_curve_np,
                    polish, arc_length, save, write_metrics, summarise,
                    results_dir, HOLDOUT_SHAPES, C_DARK, C_GEN, C_MARK)
from shapes_3d import SphereShape, PyramidShape, TorusShape, lift_curve_to_shape
from alignment import SurfaceTangentAlignment
from ergodic_energy_torch import coverage_distance

SOLIDS = {
    'Sphere':  lambda: SphereShape(center=(0.5, 0.5, 0.45), radius=0.40),
    'Pyramid': lambda: PyramidShape(center=(0.5, 0.5), half_base=0.32, z0=0.08, height=0.65),
    'Torus':   lambda: TorusShape(center=(0.5, 0.5, 0.4), R=0.32, r=0.15),
}
Z_INIT = 1.0          # drape from above; see shapes_3d._lift_generic
COS_TOL = 0.05        # |cos(t, n)| above this counts as a visible tangency miss


def refine_with_alignment(cps3_base, con, B, iters, lr, max_force):
    """Refine an already-lifted curve on the combined surface + alignment
    energy.

    Deliberately started from the *baseline lift* rather than from a fresh
    z_init: running the two variants through different descent schedules would
    confound the added term with the optimiser, and the question here is only
    what the alignment term itself contributes. The energy couples neighbouring
    points, so it takes the autograd route rather than the pointwise shortcut.
    """
    force = lambda c: curve_energy_grad(c, con.energy, B)      # noqa: E731
    return polish(cps3_base, force, iters=iters, lr=lr, max_force=max_force)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--solids', nargs='*', default=list(SOLIDS))
    p.add_argument('--w_align', type=float, default=1.0)
    p.add_argument('--w_surface', type=float, default=50.0)
    p.add_argument('--iters', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.2)
    p.add_argument('--max_force', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()

    shapes = args.shapes if args.shapes else HOLDOUT_SHAPES
    device = pick_device()
    model, meta = load_generator(device)
    B = basis_torch(meta['nxi'], 256, 5, device=device)
    out = results_dir(__file__)

    rows = []
    print(f"\nConstraint 5 — SE(3)-Tangentenausrichtung  --  "
          f"{len(shapes)} Formen x {len(args.solids)} Koerper")

    panels = {s: [] for s in args.solids}
    for i, shape_name in enumerate(shapes):
        d_map, particles = density_and_particles(shape_name, meta, device, seed=args.seed)
        dens_t = torch.tensor(d_map, dtype=torch.float32, device=device)
        cps2d = guided_generate(model, meta, particles, force=None, steps=args.steps,
                                device=device, seed=args.seed)

        for sname in args.solids:
            solid = SOLIDS[sname]()
            base_con = SurfaceTangentAlignment(solid, w_surface=args.w_surface, w_align=0.0)
            con = SurfaceTangentAlignment(solid, w_surface=args.w_surface,
                                          w_align=args.w_align)

            base = lift_curve_to_shape(cps2d, solid, B, z_iters=20, z_lr=0.6,
                                       polish_iters=350, polish_lr=1.0, z_init=Z_INIT)
            algn = refine_with_alignment(base, con, B, args.iters, args.lr,
                                         args.max_force)

            c_base, c_algn = curve_of(base, B), curve_of(algn, B)
            sdf_b, cos_b = base_con.report(c_base)
            sdf_a, cos_a = base_con.report(c_algn)
            rows.append({
                'shape': shape_name, 'koerper': sname,
                'sdf_max_basis': sdf_b, 'sdf_max_align': sdf_a,
                'cos_tn_basis': cos_b, 'cos_tn_align': cos_a,
                'coverage_xy_basis': coverage_distance(c_base[..., :2], dens_t)[0].item(),
                'coverage_xy_align': coverage_distance(c_algn[..., :2], dens_t)[0].item(),
                'laenge_basis': arc_length(c_base)[0].item(),
                'laenge_align': arc_length(c_algn)[0].item(),
            })
            # Per-point tangency, resampled onto the plotted (denser) curve, so
            # the figure can mark *where* the heading still cuts into the
            # surface -- the two curves differ only to first order and are
            # visually near-identical otherwise.
            panels[sname].append((shape_name, solid,
                                  cp_to_curve_np(base[0].cpu().numpy()),
                                  cp_to_curve_np(algn[0].cpu().numpy()),
                                  base_con.cos_tn(c_base)[0].cpu().numpy(),
                                  base_con.cos_tn(c_algn)[0].cpu().numpy(),
                                  sdf_b, sdf_a, cos_b, cos_a))
        r = rows[-1]
        print(f"  [{i + 1:2d}/{len(shapes)}] {shape_name:<24} "
              f"|SDF| {r['sdf_max_basis']:.4f}->{r['sdf_max_align']:.4f}   "
              f"|cos| {r['cos_tn_basis']:.3f}->{r['cos_tn_align']:.3f}")

    write_metrics(rows, out, 'alignment_metrics.csv')
    summarise(rows, label='Mittel ueber alle Formen x Koerper')
    for sname in args.solids:
        sub = [r for r in rows if r['koerper'] == sname]
        summarise(sub, keys=['sdf_max_basis', 'sdf_max_align', 'cos_tn_basis',
                             'cos_tn_align', 'coverage_xy_basis', 'coverage_xy_align'],
                  label=f'Koerper: {sname}')

    # One 3D grid per solid: baseline lift pale, aligned lift solid.
    for sname, plist in panels.items():
        n = len(plist)
        cols = min(5, n)
        nrows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(3.7 * cols, 3.5 * nrows), facecolor='white')
        for j, (nm, solid, c_base, c_algn, k_base, k_algn, sb, sa, cb, ca) in enumerate(plist):
            ax = fig.add_subplot(nrows, cols, j + 1, projection='3d')
            ax.set_facecolor('white')
            solid.draw(ax)
            ax.plot(c_base[:, 0], c_base[:, 1], c_base[:, 2], color=C_GEN,
                    lw=1.4, alpha=0.35, ls='--')
            ax.plot(c_algn[:, 0], c_algn[:, 1], c_algn[:, 2], color=C_GEN,
                    lw=2.0, alpha=0.95)
            # Mark the points whose heading still leaves the tangent plane,
            # on both curves: that is the quantity this constraint moves.
            for cnp, kv, col, sz, al in ((c_base, k_base, '#B0BEC5', 8, 0.55),
                                         (c_algn, k_algn, C_MARK, 12, 0.9)):
                m = np.interp(np.linspace(0, 1, len(cnp)),
                              np.linspace(0, 1, len(kv)), kv) > COS_TOL
                if m.any():
                    ax.scatter(cnp[m, 0], cnp[m, 1], cnp[m, 2], s=sz, c=col,
                               alpha=al, edgecolors='none', depthshade=False)
            ax.set_title(f"'{nm}'\n|cos| {cb:.2f}→{ca:.2f}  |SDF| {sb:.3f}→{sa:.3f}",
                         fontsize=7, color=C_DARK)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
            ax.set_box_aspect((1, 1, 1))
            ax.tick_params(labelsize=5, colors='#555')
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.set_facecolor('white')
        sub = [r for r in rows if r['koerper'] == sname]
        fig.suptitle(f"Constraint 5 — SE(3)-Tangentenausrichtung auf {sname}   "
                     f"(gestrichelt: nur Position, durchgezogen: + Ausrichtung)\n"
                     f"|cos(t,n)| {np.mean([r['cos_tn_basis'] for r in sub]):.3f} → "
                     f"{np.mean([r['cos_tn_align'] for r in sub]):.3f}",
                     fontsize=12, color=C_DARK, y=1.004)
        fig.tight_layout()
        save(fig, out, f'alignment_{sname.lower()}_holdout.png')
        plt.close(fig)


if __name__ == '__main__':
    main()
