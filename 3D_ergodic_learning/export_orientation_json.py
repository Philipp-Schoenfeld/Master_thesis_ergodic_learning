#!/usr/bin/env python3
r"""
export_orientation_json.py
==========================
Exportiert Bahn, Sensorachsen, Fussabdruck und Zieldichte als JSON, damit sich
das Ergebnis interaktiv drehen laesst.

Warum ueberhaupt interaktiv: die statischen 3D-Bilder verschlucken genau die
Groesse, um die es geht. Aus der Schraegansicht laesst sich nicht ablesen, ob
die Bahn ueber der Flaeche schwebt oder auf ihr liegt — und das ist der
Unterschied zwischen einem funktionierenden Standoff und einem gescheiterten.
Erst wenn man frei drehen kann, sieht man es.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.normpath(os.path.join(_here, '..', 'bsplinax-main')),
           os.path.normpath(os.path.join(_here, '..', 'thesis_architecture',
                                         'ergodic_dataset_generator'))):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from data_3d import density_volume, load_pairs, sample_particles   # noqa: E402
from model_zoo import generate, load_model                         # noqa: E402
from obstacles import basis_torch                                  # noqa: E402
from orientation import (SurfaceField, rot6d_to_matrix,            # noqa: E402
                         sensor_axis)


def r3(x, n=4):
    return [round(float(v), n) for v in np.asarray(x).reshape(-1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--labels', nargs='+',
                   default=['greek_upper_0', 'A', 'korean_5'])
    p.add_argument('--grid_res', type=int, default=48)
    p.add_argument('--curve_pts', type=int, default=160)
    p.add_argument('--n_arrows', type=int, default=28)
    p.add_argument('--floor_res', type=int, default=44)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--out', default='viz/orientation_export.json')
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, kind, meta = load_model(a.checkpoint, dev)
    traj, shape_defs, splits = load_pairs(meta['nxi'])

    vols = np.stack([density_volume(shape_defs[l], resolution=a.grid_res)
                     for l in a.labels])
    parts = sample_particles(torch.tensor(vols), torch.arange(len(a.labels)),
                             meta['n_particles'], dev)
    basis = basis_torch(meta['nxi'], a.curve_pts, 5, dev)

    out = {'checkpoint': os.path.basename(a.checkpoint),
           'epoch': int(meta.get('epoch', -1)) + 1,
           'standoff_target': 0.12, 'standoff_band': 0.03,
           'shapes': []}

    for i, lbl in enumerate(a.labels):
        cps, rot6d, _ = generate(model, kind, parts[i:i + 1], 1, meta,
                                 a.steps, dev, seed=0)
        curve = torch.einsum('ti,ic->tc', basis, cps[0].to(dev))
        r6 = torch.einsum('ti,ic->tc', basis, rot6d[0].to(dev))
        axis = sensor_axis(rot6d_to_matrix(r6).unsqueeze(0))[0]

        field = SurfaceField(vols[i], device=str(dev))
        foot = field.footprint(curve.unsqueeze(0), axis.unsqueeze(0))[0]
        d = field.direction(curve.unsqueeze(0))[0]
        dist = field.distance(curve.unsqueeze(0))[0]
        ang = torch.rad2deg(torch.arccos((axis * d).sum(-1).clamp(-1, 1)))

        ai = np.linspace(0, curve.shape[0] - 1, a.n_arrows).astype(int)

        # Bodenprojektion der Dichte, heruntergerechnet
        proj = vols[i].sum(axis=0)
        proj = proj / max(proj.max(), 1e-12)
        f = a.floor_res
        idx = np.linspace(0, proj.shape[0] - 1, f).astype(int)
        proj = proj[np.ix_(idx, idx)]

        gt = np.asarray(traj[lbl], dtype=np.float32)

        out['shapes'].append({
            'label': lbl,
            'curve': r3(curve.detach().cpu().numpy()),
            'foot': r3(foot.detach().cpu().numpy()),
            'gt': r3(gt),
            'arrow_p': r3(curve.detach().cpu().numpy()[ai]),
            'arrow_a': r3(axis.detach().cpu().numpy()[ai], 3),
            'target_d': r3(d.detach().cpu().numpy()[ai], 3),
            'floor': [round(float(v), 3) for v in proj.reshape(-1)],
            'floor_res': f,
            'pointing_deg': round(float(ang.mean()), 2),
            'within30': round(float((ang < 30).float().mean()), 3),
            'standoff': round(float(dist.mean()), 4),
            'standoff_sd': round(float(dist.std()), 4),
            'dir_spread': round(float((d - d.mean(0)).norm(dim=-1).mean()), 3),
            'z_min': round(float(curve[:, 2].min()), 3),
            'z_max': round(float(curve[:, 2].max()), 3),
        })
        print(f"  {lbl:16s} pointing {out['shapes'][-1]['pointing_deg']:5.1f} deg  "
              f"standoff {out['shapes'][-1]['standoff']:.4f}  "
              f"dir_spread {out['shapes'][-1]['dir_spread']:.3f}")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    print(f"\n  -> {a.out}  ({os.path.getsize(a.out)/1024:.0f} KB)")


if __name__ == '__main__':
    main()
