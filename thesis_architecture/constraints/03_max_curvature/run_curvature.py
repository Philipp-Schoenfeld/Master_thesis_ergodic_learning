#!/usr/bin/env python3
r"""
run_curvature.py
================
Constraint 3 across the whole holdout split: an explicit turn-radius bound
enforced at inference time.

The bound is calibrated *per shape* to a quantile of that shape's own unguided
curvature, so every shape starts with a comparable share of samples in
violation instead of the bound being vacuous for calm paths and hopeless for
wild ones.

Two numerical points, both learned the hard way and both load-bearing (see
curvature.py): the anti-inflation length guard, and force clipping -- the raw
curvature gradient peaks around 3e2 while control points live at scale ~1, so
unclipped explicit Euler diverges rather than steers.

    python run_curvature.py [--quantile 0.8] [--shapes A digit_5]
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
from common import (run_over_shapes, energy_force, results_dir, pick_device,
                    load_generator, arc_length, curvature, save, C_DARK, C_GREY,
                    C_GEN, C_MARK)
from curvature import MaxCurvature


def make_build(quantile, weight, t_start, max_force, length_weight,
               polish_steps, polish_lr):
    def build(ctx):
        k_free = curvature(ctx['free_curve'])[0]
        kappa_max = float(torch.quantile(k_free, quantile))
        con = MaxCurvature(kappa_max=kappa_max,
                           length_ref=float(arc_length(ctx['free_curve'])[0]),
                           length_weight=length_weight)
        return dict(constraint=con,
                    force=energy_force(con.energy, ctx['B']),
                    gen=dict(force_weight=weight, force_t_start=t_start,
                             max_force=max_force, polish_steps=polish_steps,
                             polish_lr=polish_lr))
    return build


def metrics(con, free_curve, guided_curve):
    m_free, p_free, f_free = con.report(free_curve)
    m_con, p_con, f_con = con.report(guided_curve)
    return {
        'kappa_max_schranke': con.kappa_max,
        'kappa_peak_frei': m_free,
        'kappa_peak_constr': m_con,
        'kappa_p99_frei': p_free,
        'kappa_p99_constr': p_con,
        'anteil_ueber_frei': f_free,
        'anteil_ueber_constr': f_con,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--quantile', type=float, default=0.80)
    p.add_argument('--weight', type=float, default=30.0)
    p.add_argument('--max_force', type=float, default=0.5)
    p.add_argument('--length_weight', type=float, default=1.0)
    p.add_argument('--polish_steps', type=int, default=400)
    p.add_argument('--polish_lr', type=float, default=0.05)
    p.add_argument('--t_start', type=float, default=0.4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()

    device = pick_device()
    model_meta = load_generator(device)
    rows = run_over_shapes(
        title=f"Constraint 3 — Maximalkrümmung (κ_max = {args.quantile:.0%}-Quantil je Form)",
        build=make_build(args.quantile, args.weight, args.t_start, args.max_force,
                         args.length_weight, args.polish_steps, args.polish_lr),
        metrics_fn=metrics,
        shapes=args.shapes, seed=args.seed, steps=args.steps, device=device,
        out_dir=results_dir(__file__), tag='curvature',
        panel_title=lambda r: (f"'{r['shape']}'\nκ_peak {r['kappa_peak_frei']:.0f} → "
                               f"{r['kappa_peak_constr']:.0f}   "
                               f"κ_p99 {r['kappa_p99_frei']:.0f} → {r['kappa_p99_constr']:.0f}"),
        model_meta=model_meta)

    # Peak curvature per shape, log scale: the bound spans three orders of
    # magnitude below the unguided peaks, so a linear axis would show nothing.
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor='white')
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, [r['kappa_peak_frei'] for r in rows], width=0.4,
           color=C_GEN, alpha=0.35, label='κ_peak ohne Constraint')
    ax.bar(idx + 0.2, [r['kappa_peak_constr'] for r in rows], width=0.4,
           color=C_GEN, alpha=0.95, label='κ_peak mit Constraint')
    ax.plot(idx, [r['kappa_max_schranke'] for r in rows], ls='none', marker='_',
            ms=12, color=C_MARK, label='κ_max (Schranke je Form)')
    ax.set_yscale('log')
    ax.set_xticks(idx)
    ax.set_xticklabels([r['shape'] for r in rows], rotation=90, fontsize=6)
    ax.set_ylabel('Spitzenkrümmung κ (log)', fontsize=8, color=C_GREY)
    ax.set_title('Constraint 3 — Spitzenkrümmung je Holdout-Form',
                 fontsize=11, color=C_DARK)
    ax.tick_params(labelsize=6, colors=C_GREY)
    for spine in ax.spines.values():
        spine.set_color('#ccc')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray', axis='y')
    ax.legend(frameon=True, fontsize=7, facecolor='white', edgecolor='#ddd')
    fig.tight_layout()
    save(fig, results_dir(__file__), 'curvature_violation_per_shape.png')


if __name__ == '__main__':
    main()
