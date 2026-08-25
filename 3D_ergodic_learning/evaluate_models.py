#!/usr/bin/env python3
r"""
evaluate_models.py  —  3D port
==============================
Metrics for 3D checkpoints on the holdout shapes, plus the stored solver
trajectories as a reference row.

Reported per model, averaged over the holdout set:
    E_ergodic, E_smooth, E_boundary, E_total   the solver energy and its terms
    coverage                                   density-weighted distance to the
                                               curve; uses no Fourier basis, so
                                               it cross-checks the ergodic term
                                               instead of restating it
    path_len                                   arc length
    planarity                                  RMS deviation from the best-fit
                                               plane — 3D-only, and the direct
                                               check that a planar target is
                                               answered with a planar path
    diversity                                  spread of the n samples
    t_infer_ms                                 wall clock per trajectory

The CSV writer uses `csv.writer` rather than manual joins: labels routinely
contain commas ("w=2, ep492") and an unquoted join silently shifts every column
after the label. results.json carries the same content and is the safer source.
"""

import argparse, csv, json, os, sys
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_3d import load_pairs, prepare_targets, sample_particles, DEFAULT_DB
from ergodic_energy_torch import (ErgodicEnergy, coverage_distance, path_length,
                                  planarity, sample_diversity)
from model_zoo import load_model, describe, generate
from obstacles import bspline_basis_matrix, SphereObstacle
from orientation import SurfaceField, rot6d_to_matrix, rot_path_length
from orientation_energy import (pointing_error_deg, incidence_ok_fraction,
                                standoff_error)
import viz_3d

# Orientation columns are only meaningful for checkpoints that have one; they
# are reported as NaN otherwise so the table still lines up.
ORI_KEYS = ('point_err_deg', 'incidence_ok', 'standoff_err', 'rot_path')


def score(cps, energy, phi, volume, basis, rot6d=None, field=None,
          standoff_target=0.12):
    """All metrics for one (n, nxi, 3) batch against one target."""
    total, terms = energy(cps, phi, return_terms=True)
    curve = torch.einsum('ti,bid->btd', basis, cps)
    out = {
        'E_ergodic':  terms['ergodic'],
        'E_smooth':   terms['smooth'],
        'E_boundary': terms['boundary'],
        'E_total':    total,
        'coverage':   coverage_distance(curve, volume),
        'path_len':   path_length(curve),
        'planarity':  planarity(curve),
    }
    n = cps.shape[0]
    if rot6d is None or field is None:
        nan = torch.full((n,), float('nan'), device=cps.device)
        out.update({k: nan for k in ORI_KEYS})
    else:
        R = rot6d_to_matrix(torch.einsum('ti,bic->btc', basis, rot6d))
        d = field.direction(curve)
        out['point_err_deg'] = pointing_error_deg(R, d)
        out['incidence_ok'] = incidence_ok_fraction(R, d, max_deg=30.0)
        out['standoff_err'] = standoff_error(field.distance(curve),
                                             standoff_target)
        out['rot_path'] = rot_path_length(R)
    return out


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--db', type=str, default=DEFAULT_DB)
    ap.add_argument('--n_samples', type=int, default=8)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--grid_res', type=int, default=64)
    ap.add_argument('--erg_K', type=int, default=8)
    ap.add_argument('--solver_T', type=int, default=100)
    ap.add_argument('--bspline_deg', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--out', type=str, default='metrics_3d')
    ap.add_argument('--obstacle', action='store_true', default=False,
                    help='Apply inference-time sphere guidance.')
    ap.add_argument('--orientation_ref', action='store_true', default=False,
                    help='Also give the solver reference row Stufe-0 frames, so '
                         'its orientation columns are filled in too.')
    ap.add_argument('--visualize', action='store_true', default=False)
    ap.add_argument('--viz_dir', type=str, default=None)
    ap.add_argument('--viz_n_gen', type=int, default=5)
    args = ap.parse_args()

    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"  device={device}")

    traj, shape_defs, splits = load_pairs(25, db_path=args.db)
    names = [l for l in traj if splits[l] == 'val']
    used, volumes_np, phi_np = prepare_targets(names, shape_defs, args.grid_res,
                                               args.erg_K)
    names = used
    volumes = torch.from_numpy(volumes_np)
    phi_t = torch.from_numpy(phi_np).to(device)
    print(f"  {len(names)} holdout shapes, grid {args.grid_res}^3, "
          f"K={args.erg_K} ({args.erg_K ** 3} modes)")

    obstacle = SphereObstacle() if args.obstacle else None

    # Surface fields are only built when something needs them.
    need_fields = args.orientation_ref or any(
        torch.load(c, map_location='cpu', weights_only=True).get('orientation', False)
        for c in args.checkpoints if os.path.isfile(c))
    fields = ([SurfaceField(volumes_np[i], device=device) for i in range(len(names))]
              if need_fields else [None] * len(names))
    if need_fields:
        print(f"  Built {len(names)} surface fields for the orientation metrics")

    _cache = {}

    def machinery(nxi):
        if nxi not in _cache:
            b = torch.tensor(bspline_basis_matrix(nxi, args.solver_T, args.bspline_deg),
                             dtype=torch.float32, device=device)
            _cache[nxi] = (b, ErgodicEnergy(K=args.erg_K, basis=b).to(device))
        return _cache[nxi]

    labels = (args.labels if args.labels and len(args.labels) == len(args.checkpoints)
              else [os.path.basename(c)[:44] for c in args.checkpoints])

    entries = []
    for lbl, ck in zip(labels, args.checkpoints):
        if not os.path.isfile(ck):
            print(f"  [SKIP] not found: {ck}")
            continue
        model, kind, meta = load_model(ck, device)
        entries.append(dict(label=lbl, ckpt=ck, model=model, kind=kind, meta=meta))
        print(f"  Loaded [{kind:7s}] {lbl}  ({describe(kind, meta)})")

    nxi_ref = entries[0]['meta']['nxi'] if entries else 25
    entries.append(dict(label='SVGD solver (Referenz)', ckpt=None, model=None,
                        kind='solver', meta=dict(nxi=nxi_ref, n_particles=512)))

    rows, per_shape, figures = [], [], {}
    keys = ('E_ergodic', 'E_smooth', 'E_boundary', 'E_total',
            'coverage', 'path_len', 'planarity') + ORI_KEYS

    for e in entries:
        agg = {k: [] for k in keys}
        divs, times, panels = [], [], []
        basis, energy = machinery(e['meta']['nxi'])

        for i, name in enumerate(names):
            gseed = args.seed + i
            torch.manual_seed(gseed)
            idx_t = torch.tensor([i], dtype=torch.long)
            vol_dev = volumes[i].to(device)

            wants_ori = e['meta'].get('orientation', False)
            field = fields[i] if (wants_ori or args.orientation_ref) else None

            if e['kind'] == 'solver':
                cps = torch.tensor(traj[name], dtype=torch.float32,
                                   device=device).unsqueeze(0)
                parts, rot6d = None, None
                if args.orientation_ref and field is not None:
                    # The solver has no orientation of its own, so it is judged
                    # with the Stufe-0 frames derived from its own curve — the
                    # fair yardstick for a model that learns one.
                    from orientation import frames_for_curve, matrix_to_rot6d
                    rot6d = matrix_to_rot6d(
                        frames_for_curve(cps, field, mode='lookat'))
            else:
                parts = sample_particles(volumes, idx_t, e['meta']['n_particles'],
                                         device, mode='uniform')[0]
                cps, rot6d, dt = generate(e['model'], e['kind'], parts,
                                          args.n_samples, e['meta'], args.steps,
                                          device, gseed, obstacle=obstacle)
                times.append(dt / args.n_samples * 1000.0)
                divs.append(sample_diversity(cps))

            phi = phi_t[i:i + 1].expand(cps.shape[0], -1)
            s = score(cps, energy, phi, vol_dev, basis, rot6d=rot6d, field=field,
                      standoff_target=e['meta'].get('standoff_target', 0.12))
            for k, v in s.items():
                agg[k].append(v.mean().item())
            per_shape.append(dict(model=e['label'], shape=name,
                                  **{k: float(v.mean().item()) for k, v in s.items()}))

            if args.visualize and parts is not None:
                gen_R = None
                if rot6d is not None:
                    vb = torch.tensor(
                        bspline_basis_matrix(e['meta']['nxi'], 512,
                                             args.bspline_deg),
                        dtype=torch.float32, device=device)
                    gen_R = rot6d_to_matrix(
                        torch.einsum('ti,ic->tc', vb, rot6d[0])).cpu().numpy()
                panels.append(dict(base=traj[name],
                                   gen_cps=cps[:args.viz_n_gen].cpu().numpy(),
                                   particles=parts.cpu().numpy(),
                                   volume=volumes_np[i],
                                   title=f"'{name}'\nE_erg={s['E_ergodic'].mean():.2f}",
                                   obstacle=obstacle, gen_R=gen_R))

        row = dict(model=e['label'], kind=e['kind'])
        for k, v in agg.items():
            row[k] = float(np.mean(v)) if v else float('nan')
        row['diversity'] = float(np.nanmean(divs)) if divs else float('nan')
        row['t_infer_ms'] = float(np.mean(times)) if times else float('nan')
        rows.append(row)

        if args.visualize and panels:
            vdir = args.viz_dir or os.path.join(_here, 'viz', 'eval')
            os.makedirs(vdir, exist_ok=True)
            safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in e['label'])
            fp = os.path.join(vdir, f"eval3d_{safe}.png")
            viz_3d.save_grid(panels, fp,
                             f"{e['label']} — {'mit' if obstacle else 'ohne'} Hindernis")
            figures[e['label']] = fp

    hdr = ('model', 'E_ergodic', 'E_total', 'coverage', 'path_len',
           'planarity', 'point_err_deg', 'incidence_ok', 'standoff_err',
           'rot_path', 'diversity', 't_infer_ms')
    print(f"\n{'=' * 112}")
    print(f"  Mittelwerte ueber {len(names)} Holdout-Formen "
          f"(niedriger ist besser, ausser diversity)")
    print('=' * 112)
    print(f"  {'Modell':<34}{'E_ergodic':>11}{'E_total':>10}{'coverage':>10}"
          f"{'planarity':>10}{'Zeigefhl':>10}{'ok<30d':>8}{'Standoff':>10}"
          f"{'diversity':>11}{'t/traj[ms]':>12}")
    print('  ' + '-' * 108)
    best = min(r['E_ergodic'] for r in rows)

    def f(v, fmt, dash='—'):
        return dash if (v is None or np.isnan(v)) else format(v, fmt)

    for r in rows:
        mark = ' *' if r['E_ergodic'] == best else '  '
        print(f"{mark}{r['model'][:34]:<34}{r['E_ergodic']:>11.4f}"
              f"{r['E_total']:>10.3f}{r['coverage']:>10.4f}"
              f"{r['planarity']:>10.5f}"
              f"{f(r['point_err_deg'], '.1f'):>10}"
              f"{f(r['incidence_ok'], '.2f'):>8}"
              f"{f(r['standoff_err'], '.3f'):>10}"
              f"{f(r['diversity'], '.4f'):>11}"
              f"{f(r['t_infer_ms'], '.1f'):>12}")
    print('  ' + '-' * 108)
    print("  * bester ergodischer Wert. planarity ~ 0: die Bahn liegt in einer "
          "Ebene, wie das Ziel.")
    print("  Zeigefhl = mittlerer Winkel zur Sollrichtung in Grad; ok<30d = "
          "Anteil des Pfades innerhalb 30 Grad.")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in rows:
            w.writerow([r[k] for k in hdr])
    pkeys = ('model', 'shape') + keys
    with open(os.path.join(args.out, 'per_shape.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(pkeys)
        for r in per_shape:
            w.writerow([r[k] for k in pkeys])
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(dict(summary=rows, per_shape=per_shape, shapes=names,
                       figures=figures, config=vars(args)), f, indent=1)
    print(f"\n  Geschrieben nach {args.out}/: summary.csv, per_shape.csv, results.json")


if __name__ == '__main__':
    main()
