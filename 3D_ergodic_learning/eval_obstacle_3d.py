#!/usr/bin/env python3
"""Obstacle-compliance check of a trained obstacle-conditioned checkpoint.

Places a sphere of each given radius on the stored path of holdout entries and
generates (a) conditioned on it and (b) with the null obstacle token. Reports
how often / how deep the paths penetrate the sphere and how close they come to
the target surface.

    python eval_obstacle_3d.py --ckpt checkpoints/obs3d_*.pt --db3d ergodic_dataset_3d_lm.db
"""
import argparse
import json

import numpy as np
import torch

from data_surfaces import load_surface_db
from eval_lm3d_compare import load_model
from obstacle_cond import evaluate_compliance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--db3d', default='ergodic_dataset_3d_lm.db')
    p.add_argument('--n_items', type=int, default=48)
    p.add_argument('--radii', type=float, nargs='+', default=[0.05, 0.08, 0.10])
    p.add_argument('--n_gen', type=int, default=4)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--surf_clearance', type=float, default=0.07)
    p.add_argument('--out', default=None, help='optional JSON output')
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, info = load_model(a.ckpt, dev)
    if not getattr(model, 'obstacle_cond', False):
        raise SystemExit('checkpoint has no obstacle conditioning')
    print(f"model: {info}")
    entries = load_surface_db(a.db3d, nxi=25, splits=('val_form', 'val_flaeche', 'val_beides'),
                              target_cps=True, deg=5)
    step = max(len(entries) // a.n_items, 1)
    items = [dict(x1=torch.tensor(e['x1']), parts=torch.tensor(e['parts']),
                  start=torch.tensor(e['start'])) for e in entries[::step][:a.n_items]]
    out = {}
    for r in a.radii:
        res = evaluate_compliance(model, items, dev, n_gen=a.n_gen, steps=a.steps,
                                  radius=r, surf_clearance=a.surf_clearance)
        out[str(r)] = res
        print(f"\nradius {r}:")
        for k, v in res.items():
            print(f"  {k:<16} {v:.4f}")
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
