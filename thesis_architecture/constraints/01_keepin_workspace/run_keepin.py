#!/usr/bin/env python3
r"""
run_keepin.py
=============
Constraint 1 across the whole holdout split: the same trained model, once
unguided and once with a keep-in region enforced at inference time.

The region is calibrated *per shape* to the unguided curve's own bounding box
(shrunk by `--shrink`), so every shape poses a real containment problem rather
than a trivially satisfied one.

    python run_keepin.py [--region box|circle] [--shapes A digit_5] [--shrink 0.7]
"""
import argparse

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (run_over_shapes, pointwise_grad, results_dir, pick_device,
                    load_generator, arc_length)
from keepin import KeepInBox, KeepInCircle


def make_build(region_kind, shrink, weight, t_start):
    def build(ctx):
        c = ctx['free_curve'][0]
        lo = c.min(dim=0).values
        hi = c.max(dim=0).values
        mid = (lo + hi) / 2
        half = (hi - lo) / 2 * shrink
        if region_kind == 'box':
            region = KeepInBox(lo=tuple((mid - half).tolist()),
                               hi=tuple((mid + half).tolist()))
        else:
            region = KeepInCircle(center=tuple(mid.tolist()),
                                  radius=float(half.max()))
        return dict(constraint=region,
                    force=pointwise_grad(region, ctx['B']),
                    gen=dict(force_weight=weight, force_t_start=t_start,
                             polish_steps=300))
    return build


def metrics(region, free_curve, guided_curve):
    return {
        'austritt_frei': region.max_violation_raw(free_curve),
        'austritt_constr': region.max_violation_raw(guided_curve),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--region', choices=['box', 'circle'], default='box')
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--shrink', type=float, default=0.7,
                   help='Region size as a fraction of the free curve extent.')
    p.add_argument('--weight', type=float, default=25.0)
    p.add_argument('--t_start', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()

    device = pick_device()
    model_meta = load_generator(device)
    run_over_shapes(
        title=f"Constraint 1 — Keep-in-Region ({args.region}, {args.shrink:g}× freie Ausdehnung)",
        build=make_build(args.region, args.shrink, args.weight, args.t_start),
        metrics_fn=metrics,
        decorate=lambda ax, region: region.draw(ax),
        shapes=args.shapes, seed=args.seed, steps=args.steps, device=device,
        out_dir=results_dir(__file__), tag=f'keepin_{args.region}',
        panel_title=lambda r: (f"'{r['shape']}'\nAustritt {r['austritt_frei']:.3f} → "
                               f"{r['austritt_constr']:.3f}"),
        model_meta=model_meta)


if __name__ == '__main__':
    main()
