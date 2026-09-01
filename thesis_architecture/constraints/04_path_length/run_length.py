#!/usr/bin/env python3
r"""
run_length.py
=============
Constraint 4 across the whole holdout split: a target arc length enforced at
inference time.

The target is calibrated *per shape* as `--ratio` times that shape's own
unguided length, so "shorten by 30%" means the same thing for a compact letter
and a sprawling organic blob.

    python run_length.py [--ratio 0.7] [--mode exact|cap] [--shapes A digit_5]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (run_over_shapes, energy_force, results_dir, pick_device,
                    load_generator, arc_length, save, C_DARK, C_GREY, C_GEN, C_MARK)
from length import TargetLength


def make_build(ratio, mode, weight, t_start, polish_steps, polish_lr, max_force):
    def build(ctx):
        L_free = float(arc_length(ctx['free_curve'])[0])
        con = TargetLength(target=ratio * L_free, mode=mode)
        return dict(constraint=con,
                    force=energy_force(con.energy, ctx['B']),
                    gen=dict(force_weight=weight, force_t_start=t_start,
                             max_force=max_force, polish_steps=polish_steps,
                             polish_lr=polish_lr))
    return build


def metrics(con, free_curve, guided_curve):
    return {
        'ziel_laenge': con.target,
        'laenge_frei_c': con.length(free_curve),
        'laenge_constr_c': con.length(guided_curve),
        'rel_fehler_frei': con.rel_error(free_curve),
        'rel_fehler_constr': con.rel_error(guided_curve),
        'abs_rel_fehler_constr': abs(con.rel_error(guided_curve)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--ratio', type=float, default=0.7,
                   help='Target length as a fraction of the free curve length.')
    p.add_argument('--mode', choices=['exact', 'cap'], default='exact')
    p.add_argument('--weight', type=float, default=30.0)
    p.add_argument('--max_force', type=float, default=0.5)
    p.add_argument('--polish_steps', type=int, default=400)
    p.add_argument('--polish_lr', type=float, default=0.05)
    p.add_argument('--t_start', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()

    device = pick_device()
    model_meta = load_generator(device)
    rows = run_over_shapes(
        title=f"Constraint 4 — Ziel-Pfadlänge ({args.ratio:g}× freie Länge, mode={args.mode})",
        build=make_build(args.ratio, args.mode, args.weight, args.t_start,
                         args.polish_steps, args.polish_lr, args.max_force),
        metrics_fn=metrics,
        shapes=args.shapes, seed=args.seed, steps=args.steps, device=device,
        out_dir=results_dir(__file__), tag=f'length_{args.mode}_{args.ratio:g}',
        panel_title=lambda r: (f"'{r['shape']}'\nL {r['laenge_frei_c']:.2f} → "
                               f"{r['laenge_constr_c']:.2f}  (Ziel {r['ziel_laenge']:.2f})"),
        model_meta=model_meta)

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor='white')
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, [r['laenge_frei_c'] for r in rows], width=0.4,
           color=C_GEN, alpha=0.35, label='ohne Constraint')
    ax.bar(idx + 0.2, [r['laenge_constr_c'] for r in rows], width=0.4,
           color=C_GEN, alpha=0.95, label='mit Constraint')
    ax.plot(idx, [r['ziel_laenge'] for r in rows], ls='none', marker='_',
            ms=14, color=C_MARK, label='Ziel')
    ax.set_xticks(idx)
    ax.set_xticklabels([r['shape'] for r in rows], rotation=90, fontsize=6)
    ax.set_ylabel('Bogenlänge', fontsize=8, color=C_GREY)
    ax.set_title(f'Constraint 4 — Länge je Holdout-Form (Ziel = {args.ratio:g}× frei)',
                 fontsize=11, color=C_DARK)
    ax.tick_params(labelsize=6, colors=C_GREY)
    for spine in ax.spines.values():
        spine.set_color('#ccc')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray', axis='y')
    ax.legend(frameon=True, fontsize=7, facecolor='white', edgecolor='#ddd')
    fig.tight_layout()
    save(fig, results_dir(__file__), f'length_{args.mode}_{args.ratio:g}_per_shape.png')


if __name__ == '__main__':
    main()
