#!/usr/bin/env python3
r"""
export_artifact_data.py
=======================
Dump everything the results artifact needs into one JSON: per-shape curves
(free and constrained, for every constraint variant), the 3D lifts on all
three solids, the target-density thumbnails, and the metric rows.

The free curve depends only on (shape, seed), so it is generated once per
shape and reused across all seven variants -- that alone removes ~150
redundant ODE integrations compared with running the seven runners back to
back.

    python export_artifact_data.py [--shapes A digit_5] [--out artifact_data.json]
"""
import argparse
import base64
import importlib.util
import io
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from common import (pick_device, load_generator, density_and_particles, basis_torch,
                    guided_generate, curve_of, cp_to_curve_np, curvature,
                    arc_length, ErgodicMetrics, HOLDOUT_SHAPES, WHITE_INFERNO,
                    curve_energy_grad, polish)
from shapes_3d import SphereShape, PyramidShape, TorusShape, lift_curve_to_shape
from ergodic_energy_torch import coverage_distance

CURVE_PTS = 220        # plotted resolution; 3 decimals is < 1 screen pixel
ROUND = 3
# Lifted curves need far more precision than drawing does: the page derives
# tangents from them by finite differences over a ~0.02 spacing, so rounding to
# 1e-3 injects a ~0.025 rad direction error -- the same order as the |cos(t,n)|
# being measured. At 1e-5 that noise drops to ~2.5e-4 and the page reproduces
# the Python metric instead of a rounded caricature of it.
ROUND_LIFT = 5
Z_INIT = 1.0

SOLIDS = {
    'Sphere':  dict(make=lambda: SphereShape(center=(0.5, 0.5, 0.45), radius=0.40),
                    kind='sphere', params=dict(cx=0.5, cy=0.5, cz=0.45, r=0.40)),
    'Pyramid': dict(make=lambda: PyramidShape(center=(0.5, 0.5), half_base=0.32,
                                              z0=0.08, height=0.65),
                    kind='pyramid', params=dict(cx=0.5, cy=0.5, a=0.32, z0=0.08, h=0.65)),
    'Torus':   dict(make=lambda: TorusShape(center=(0.5, 0.5, 0.4), R=0.32, r=0.15),
                    kind='torus', params=dict(cx=0.5, cy=0.5, cz=0.4, R=0.32, r=0.15)),
}


def _load(folder, name):
    """Import a runner module by path so its build/metrics stay the source of
    truth instead of being duplicated here."""
    path = os.path.join(_here, folder, name)
    spec = importlib.util.spec_from_file_location(f'_rt_{folder}_{name[:-3]}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def curve_json(curve_np):
    """(pts, nd) -> flat rounded list, resampled to CURVE_PTS."""
    idx = np.linspace(0, len(curve_np) - 1, CURVE_PTS).astype(int)
    return [round(float(v), ROUND) for v in curve_np[idx].ravel()]


def curve_json_native(curve_t):
    """Flat rounded list at the *exact* discretisation the metrics were taken on.

    The lifted curves carry orientation, and the page recomputes tangents and
    surface normals from them client-side. Resampling first would change the
    finite differences at every sharp turn, so the page's per-point |cos(t,n)|
    would quietly disagree with the aggregate reported next to it. Exporting
    the same 256 samples `curve_of` produced keeps the two identical.
    """
    return [round(float(v), ROUND_LIFT) for v in curve_t[0].detach().cpu().numpy().ravel()]


def density_png(d_map):
    """Colour-mapped target density as a base64 PNG (small, 64x64 source)."""
    rgba = WHITE_INFERNO(np.clip(d_map, 0, 1))
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format='png')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--out', default=os.path.join(_here, 'results', 'artifact_data.json'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    args = p.parse_args()

    shapes = args.shapes if args.shapes else HOLDOUT_SHAPES
    device = pick_device()
    model, meta = load_generator(device)
    B = basis_torch(meta['nxi'], 256, 5, device=device)
    erg = ErgodicMetrics(meta['nxi'], device)

    rk = _load('01_keepin_workspace', 'run_keepin.py')
    rw = _load('02_waypoint_anchor', 'run_waypoints.py')
    rc = _load('03_max_curvature', 'run_curvature.py')
    rl = _load('04_path_length', 'run_length.py')
    al = _load('05_se3_alignment', 'run_alignment.py')
    from alignment import SurfaceTangentAlignment

    # (key, label, build, metrics_fn, decoration extractor)
    VARIANTS = [
        ('keepin_box', 'Keep-in Box', rk.make_build('box', 0.7, 25.0, 0.3), rk.metrics,
         lambda c: dict(kind='box', lo=list(c.lo), hi=list(c.hi))),
        ('keepin_circle', 'Keep-in Kreis', rk.make_build('circle', 0.7, 25.0, 0.3), rk.metrics,
         lambda c: dict(kind='circle', center=list(c.center), r=c.radius)),
        ('waypoints', 'Waypoints', rw.make_build('start_via_end', 25.0, 0.3), rw.metrics,
         lambda c: dict(kind='pins', points=[list(q) for q in c.points], phases=c.phases)),
        ('curvature', 'Maximalkrümmung', rc.make_build(0.80, 30.0, 0.4, 0.5, 1.0, 400, 0.05),
         rc.metrics, lambda c: dict(kind='none')),
        ('length_07', 'Länge 0.7×', rl.make_build(0.7, 'exact', 30.0, 0.3, 400, 0.05, 0.5),
         rl.metrics, lambda c: dict(kind='none')),
        ('length_13', 'Länge 1.3×', rl.make_build(1.3, 'exact', 30.0, 0.3, 400, 0.05, 0.5),
         rl.metrics, lambda c: dict(kind='none')),
    ]

    out = {'shapes': [], 'variants': [{'key': k, 'label': l} for k, l, *_ in VARIANTS],
           'solids': {k: dict(kind=v['kind'], params=v['params']) for k, v in SOLIDS.items()},
           'meta': dict(nxi=meta['nxi'], D=meta['D'], n_particles=meta['n_particles'],
                        epoch=meta['epoch'], seed=args.seed, steps=args.steps,
                        n_shapes=len(shapes))}

    for i, name in enumerate(shapes):
        d_map, particles = density_and_particles(name, meta, device, seed=args.seed)
        dens_t = torch.tensor(d_map, dtype=torch.float32, device=device)
        phi = erg.phi_for(d_map)

        free = guided_generate(model, meta, particles, force=None, steps=args.steps,
                               device=device, seed=args.seed)
        free_curve = curve_of(free, B)
        e_free = erg.score(free, phi, dens_t)

        entry = {
            'name': name,
            'density': density_png(d_map),
            'free': curve_json(cp_to_curve_np(free[0].cpu().numpy())),
            'free_metrics': dict(E_erg=round(e_free[0], 4), coverage=round(e_free[1], 5),
                                 laenge=round(e_free[2], 3),
                                 kappa_peak=round(float(curvature(free_curve).max()), 1)),
            'variants': {}, 'lifts': {},
        }

        ctx = dict(shape=name, free_cps=free, free_curve=free_curve, B=B,
                   meta=meta, device=device, d_map=d_map)
        for key, label, build, metrics_fn, deco in VARIANTS:
            spec = build(ctx)
            cps = guided_generate(model, meta, particles, force=spec['force'],
                                  steps=args.steps, device=device, seed=args.seed,
                                  **spec.get('gen', {}))
            g_curve = curve_of(cps, B)
            e_con = erg.score(cps, phi, dens_t)
            m = {k: (round(v, 5) if isinstance(v, float) else v)
                 for k, v in metrics_fn(spec['constraint'], free_curve, g_curve).items()}
            m.update(E_erg=round(e_con[0], 4), coverage=round(e_con[1], 5),
                     laenge=round(e_con[2], 3))
            entry['variants'][key] = {
                'curve': curve_json(cp_to_curve_np(cps[0].cpu().numpy())),
                'metrics': m, 'deco': deco(spec['constraint']),
            }

        for sname, sdef in SOLIDS.items():
            solid = sdef['make']()
            base_con = SurfaceTangentAlignment(solid, w_surface=50.0, w_align=0.0)
            con = SurfaceTangentAlignment(solid, w_surface=50.0, w_align=1.0)
            base = lift_curve_to_shape(free, solid, B, z_iters=20, z_lr=0.6,
                                       polish_iters=350, polish_lr=1.0, z_init=Z_INIT)
            algn = polish(base, lambda c: curve_energy_grad(c, con.energy, B),
                          iters=300, lr=0.2, max_force=0.5)
            cb, ca = curve_of(base, B), curve_of(algn, B)
            sb, kb = base_con.report(cb)
            sa, ka = base_con.report(ca)
            entry['lifts'][sname] = {
                'base': curve_json_native(cb),
                'align': curve_json_native(ca),
                'metrics': dict(
                    sdf_base=round(sb, 5), sdf_align=round(sa, 5),
                    cos_base=round(kb, 5), cos_align=round(ka, 5),
                    cov_base=round(coverage_distance(cb[..., :2], dens_t)[0].item(), 5),
                    cov_align=round(coverage_distance(ca[..., :2], dens_t)[0].item(), 5),
                    len_base=round(arc_length(cb)[0].item(), 3),
                    len_align=round(arc_length(ca)[0].item(), 3)),
            }

        out['shapes'].append(entry)
        print(f"  [{i + 1:2d}/{len(shapes)}] {name}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, separators=(',', ':'))
    mb = os.path.getsize(args.out) / 1e6
    print(f"  Saved -> {args.out}  ({mb:.2f} MB)")


if __name__ == '__main__':
    main()
