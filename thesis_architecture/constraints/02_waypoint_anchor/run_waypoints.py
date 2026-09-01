#!/usr/bin/env python3
r"""
run_waypoints.py
================
Constraint 2 across the whole holdout split: the same trained model forced
through fixed start, via and end poses at inference time.

The pins are calibrated *per shape* from the target density's own support
(bottom-left, top-centre, bottom-right of its bounding box), so every shape
poses a genuine traverse-the-figure task instead of pins that happen to sit
where the free curve already goes.

    python run_waypoints.py [--pins start_end|start_via_end] [--shapes A digit_5]
"""
import argparse

import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (run_over_shapes, energy_force, results_dir, pick_device,
                    load_generator)
from waypoints import WaypointPins


def _support_box(d_map, thresh=0.15):
    """Bounding box of the density's support in [0,1]^2 coordinates."""
    ys, xs = np.nonzero(d_map > thresh)
    h, w = d_map.shape
    if len(xs) == 0:
        return 0.2, 0.2, 0.8, 0.8
    return (xs.min() / (w - 1), ys.min() / (h - 1),
            xs.max() / (w - 1), ys.max() / (h - 1))


def make_build(mode, weight, t_start):
    def build(ctx):
        x0, y0, x1, y1 = _support_box(ctx['d_map'])
        dx, dy = x1 - x0, y1 - y0
        start = (x0 + 0.12 * dx, y0 + 0.12 * dy)
        end = (x1 - 0.12 * dx, y0 + 0.12 * dy)
        via = ((x0 + x1) / 2, y1 - 0.08 * dy)
        pins = ([(0.0, start), (1.0, end)] if mode == 'start_end'
                else [(0.0, start), (0.5, via), (1.0, end)])
        con = WaypointPins(pins)
        return dict(constraint=con,
                    force=energy_force(con.energy, ctx['B']),
                    gen=dict(force_weight=weight, force_t_start=t_start,
                             polish_steps=300))
    return build


def metrics(pins, free_curve, guided_curve):
    return {
        'pin_fehler_frei': pins.max_error(free_curve),
        'pin_fehler_constr': pins.max_error(guided_curve),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pins', choices=['start_end', 'start_via_end'],
                   default='start_via_end')
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--weight', type=float, default=25.0)
    p.add_argument('--t_start', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()

    device = pick_device()
    model_meta = load_generator(device)
    run_over_shapes(
        title=f"Constraint 2 — Waypoint-Pinning ({args.pins.replace('_', ' + ')})",
        build=make_build(args.pins, args.weight, args.t_start),
        metrics_fn=metrics,
        decorate=lambda ax, pins: pins.draw(ax),
        shapes=args.shapes, seed=args.seed, steps=args.steps, device=device,
        out_dir=results_dir(__file__), tag=f'waypoints_{args.pins}',
        panel_title=lambda r: (f"'{r['shape']}'\nPin-Fehler {r['pin_fehler_frei']:.3f} → "
                               f"{r['pin_fehler_constr']:.3f}"),
        model_meta=model_meta)


if __name__ == '__main__':
    main()
