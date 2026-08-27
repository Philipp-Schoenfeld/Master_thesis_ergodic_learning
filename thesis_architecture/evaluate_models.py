#!/usr/bin/env python3
"""
evaluate_models.py
==================
Quantitative comparison of trained generators on the 25 holdout shapes.

Replaces eyeballing the holdout grids with numbers. Every model is scored with
the *same* metrics, on the *same* targets, against the classical SVGD solver as
a baseline — the solver's own trajectories are read from the database and pushed
through the identical metric code, so "better than the solver" is a measured
statement rather than an impression.

Metrics per generated trajectory (all on the rendered B-spline curve at T=100
points, which is what the solver optimises):

  E_ergodic   the ergodic metric itself: 0.5 * sum(Lambda_k (c_k - phi_k)^2),
              weighted by W_ERGODIC. The headline number.
  E_smooth    W_SMOOTH * sum(accel^2) — kinematic cost.
  E_boundary  W_BOUNDARY * hinge penalty for leaving [0,1]^2.
  E_total     their sum, i.e. the solver's own objective.
  coverage    density-weighted mean distance from the target mass to the curve.
              Independent of the Fourier basis, so it cross-checks E_ergodic
              instead of restating it. Lower is better; units are domain widths.
  path_len    curve length — a trajectory that scores well only by being
              enormously long is not useful, so this guards against that.
  diversity   mean pairwise RMS distance between the samples drawn for one
              target. Detects mode collapse.
  t_infer     milliseconds per trajectory. The whole point of amortisation, so
              it belongs in the table next to the quality numbers.

Model types are detected from the checkpoint: `selfsupervised: True` selects the
single-pass generator, otherwise the flow-matching net with Euler integration
and CFG.

Usage:
  python -u evaluate_models.py --checkpoints checkpoints/a.pt checkpoints/b.pt
  python -u evaluate_models.py --checkpoints ... --n_samples 8 --out metrics/
"""

import argparse, os, sys, json, time, sqlite3
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.append(os.path.join(_here, 'ergodic_dataset_generator'))

from bsplinax.bspline import BsplineBasisClamped
from ergodic_energy_torch import (
    ErgodicEnergy, make_k_grid, K_DEFAULT,
    coverage_distance, path_length, sample_diversity,
)
from model_zoo import load_model, generate, describe
from flow_matching_runner_particles import sample_particles
# Target definitions come from the DB (font-free, and identical to what the
# supervised models were trained on).
from flow_matching_runner_particles_selfsupervised import (
    load_shape_defs_from_db, prepare_targets,
)

_DB_PATH = os.path.join(_here, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')
SOLVER_T = 100


def solver_trajectories(names, nxi):
    """The solver's own answers, as the baseline row."""
    if not os.path.isfile(_DB_PATH):
        return {}
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    out = {}
    for name in names:
        cur.execute("SELECT trajectory FROM ergodic_pairs WHERE shape_name=? LIMIT 1", (name,))
        row = cur.fetchone()
        if row:
            xy = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 2)
            out[name] = xy[np.linspace(0, len(xy) - 1, nxi).astype(int)]
    conn.close()
    return out


# ===========================================================================
# Evaluation
# ===========================================================================

def score(cps, energy, phi, density_t, basis):
    """All per-sample metrics for one target. cps: (n, nxi, 2), phi: (M,)"""
    _, terms = energy(cps, phi, return_terms=True)
    curves = torch.einsum('ti,bid->btd', basis, cps)
    return dict(
        E_ergodic=terms['ergodic'], E_smooth=terms['smooth'],
        E_boundary=terms['boundary'],
        E_total=sum(terms.values()),
        coverage=coverage_distance(curves, density_t),
        path_len=path_length(curves),
    )


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--labels', nargs='*', default=None,
                   help='Optional short names, same order as --checkpoints.')
    p.add_argument('--n_samples', type=int, default=8,
                   help='Trajectories generated per holdout shape.')
    p.add_argument('--steps', type=int, default=100, help='ODE steps (flow models).')
    p.add_argument('--grid_res', type=int, default=64)
    p.add_argument('--erg_K', type=int, default=K_DEFAULT)
    p.add_argument('--solver_T', type=int, default=SOLVER_T)
    p.add_argument('--bspline_deg', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--out', default='metrics',
                   help='Directory for the CSV/JSON results.')
    # Visualisation, so one command yields both the numbers and the pictures.
    p.add_argument('--visualize', action='store_true',
                   help='Also render holdout figures for every checkpoint.')
    p.add_argument('--viz_dir', default=None,
                   help='Where the figures go (default: Trajectory_data_generator/viz_rerun).')
    p.add_argument('--viz_n_gen', type=int, default=5,
                   help='Candidates per shape before the best one is picked.')
    p.add_argument('--viz_obstacle_mode', choices=['both', 'off', 'on'], default='both',
                   help='Figures per checkpoint: with and without obstacle, separately.')
    p.add_argument('--viz_all_candidates', action='store_true',
                   help='Draw every candidate instead of only the best.')
    p.add_argument('--target_length', type=float, default=None,
                   help='Target path length given to the network as FiLM '
                        'conditioning (only effective for a length_cond '
                        'checkpoint, e.g. netz2d_laenge.pt; ignored otherwise). '
                        'Recorded path_len in the output table is the achieved '
                        'length, so it directly shows how well the network '
                        'tracked the request.')
    p.add_argument('--target_length_cfg', type=float, default=0.0,
                   help='Classifier-free guidance strength for --target_length '
                        '(0 = off, i.e. the raw conditioned prediction).')
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.seed)
    print(f"\n{'=' * 78}\n  Model evaluation on the holdout split\n"
          f"  device={device}  n_samples={args.n_samples}  ODE steps={args.steps}\n{'=' * 78}")

    # ── Targets ───────────────────────────────────────────────────────────────
    shape_defs, splits = load_shape_defs_from_db()
    names = splits.get('val') or []
    if not names:
        print('[ERROR] No val split found in the database.')
        return 1
    names, dens_np, phi_np = prepare_targets(names, args.grid_res, args.erg_K,
                                             shape_defs=shape_defs)
    print(f"  Holdout shapes: {len(names)}")

    dens_t = torch.tensor(dens_np, dtype=torch.float32, device=device)
    phi_t = torch.tensor(phi_np, dtype=torch.float32, device=device)

    # ── Metric machinery ──────────────────────────────────────────────────────
    # One basis (and one energy module) per distinct nxi, so checkpoints with
    # different control-point counts stay comparable instead of silently being
    # scored against the wrong basis.
    _cache = {}

    def machinery(nxi):
        if nxi not in _cache:
            b = torch.tensor(np.array(BsplineBasisClamped(
                degree=args.bspline_deg, num_control_points=nxi,
                num_phase_points=args.solver_T, compute_derivatives=False).B),
                dtype=torch.float32, device=device)
            _cache[nxi] = (b, ErgodicEnergy(K=args.erg_K, basis=b).to(device))
        return _cache[nxi]

    nxi_ref = torch.load(args.checkpoints[0], map_location='cpu',
                         weights_only=True).get('nxi', 25)

    # ── Models, plus the solver baseline ──────────────────────────────────────
    labels = args.labels if args.labels and len(args.labels) == len(args.checkpoints) \
        else [os.path.basename(c)[:44] for c in args.checkpoints]

    entries = []
    for lbl, ck in zip(labels, args.checkpoints):
        if not os.path.isfile(ck):
            print(f"  [SKIP] not found: {ck}")
            continue
        model, kind, meta = load_model(ck, device)
        entries.append(dict(label=lbl, ckpt=ck, model=model, kind=kind, meta=meta))
        extra = f"  lambda_erg={meta['lambda_erg']}" if meta['lambda_erg'] else ''
        extra += f"  K={meta['n_candidates']} div={meta['diversity_weight']}" \
            if meta['n_candidates'] else ''
        print(f"  Loaded [{kind:7s}] {lbl}  (D={meta['D']}, N={meta['n_particles']}){extra}")
    entries.append(dict(label='SVGD solver (Referenz)', ckpt=None, model=None,
                        kind='solver', meta=dict(nxi=nxi_ref, n_particles=256)))
    solver_cps = solver_trajectories(names, nxi_ref)

    # ── Run ───────────────────────────────────────────────────────────────────
    rows, per_shape = [], []
    for e in entries:
        agg = {k: [] for k in ('E_ergodic', 'E_smooth', 'E_boundary', 'E_total',
                               'coverage', 'path_len')}
        divs, times = [], []

        basis, energy = machinery(e['meta']['nxi'])

        for i, name in enumerate(names):
            gseed = args.seed + i
            torch.manual_seed(gseed)
            idx_t = torch.tensor([i], dtype=torch.long, device=device)

            if e['kind'] == 'solver':
                if name not in solver_cps:
                    continue
                cps = torch.tensor(solver_cps[name], dtype=torch.float32,
                                   device=device).unsqueeze(0)
            else:
                # Same particles for every model on a given shape (same seed),
                # so differences come from the model, not from the conditioning.
                parts = sample_particles(dens_t, idx_t, e['meta']['n_particles'],
                                         device, mode='uniform')[0]
                cps, dt = generate(e['model'], e['kind'], parts, args.n_samples,
                                   e['meta'], args.steps, device, gseed,
                                   length=args.target_length,
                                   length_cfg_weight=args.target_length_cfg)
                times.append(dt / args.n_samples * 1000.0)
                divs.append(sample_diversity(cps))

            phi = phi_t[i:i + 1].expand(cps.shape[0], -1)
            s = score(cps, energy, phi, dens_t[i], basis)
            for k, v in s.items():
                agg[k].append(v.mean().item())
            per_shape.append(dict(model=e['label'], shape=name,
                                  **{k: float(v.mean().item()) for k, v in s.items()}))

        row = dict(model=e['label'], kind=e['kind'])
        for k, v in agg.items():
            row[k] = float(np.mean(v)) if v else float('nan')
        row['diversity'] = float(np.nanmean(divs)) if divs else float('nan')
        row['t_infer_ms'] = float(np.mean(times)) if times else float('nan')
        rows.append(row)

    # ── Report ────────────────────────────────────────────────────────────────
    hdr = ('model', 'E_ergodic', 'E_total', 'coverage', 'path_len', 'diversity', 't_infer_ms')
    print(f"\n{'=' * 110}\n  Mittelwerte über {len(names)} Holdout-Formen "
          f"(niedriger ist besser, außer diversity)\n{'=' * 110}")
    print(f"  {'Modell':<46}{'E_ergodic':>11}{'E_total':>10}{'coverage':>10}"
          f"{'path_len':>10}{'diversity':>11}{'t/traj [ms]':>12}")
    print('  ' + '-' * 106)
    best_erg = min(r['E_ergodic'] for r in rows)
    for r in rows:
        mark = ' *' if r['E_ergodic'] == best_erg else '  '
        d = '—' if np.isnan(r['diversity']) else f"{r['diversity']:.4f}"
        t = '—' if np.isnan(r['t_infer_ms']) else f"{r['t_infer_ms']:.1f}"
        print(f"{mark}{r['model'][:46]:<46}{r['E_ergodic']:>11.4f}{r['E_total']:>10.3f}"
              f"{r['coverage']:>10.4f}{r['path_len']:>10.3f}{d:>11}{t:>12}")
    print('  ' + '-' * 106)
    print("  * bester ergodischer Wert. E_total ist die vollständige Solver-Zielfunktion.")

    # ── Persist ───────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'summary.csv'), 'w') as f:
        f.write(','.join(hdr) + '\n')
        for r in rows:
            f.write(','.join(str(r[k]) for k in hdr) + '\n')
    with open(os.path.join(args.out, 'per_shape.csv'), 'w') as f:
        keys = ['model', 'shape', 'E_ergodic', 'E_smooth', 'E_boundary',
                'E_total', 'coverage', 'path_len']
        f.write(','.join(keys) + '\n')
        for r in per_shape:
            f.write(','.join(str(r[k]) for k in keys) + '\n')
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(dict(summary=rows, per_shape=per_shape, shapes=names,
                       config=vars(args)), f, indent=1)
    print(f"\n  Geschrieben nach {args.out}/: summary.csv, per_shape.csv, results.json")

    # ── Figures ───────────────────────────────────────────────────────────────
    if args.visualize:
        from visualize_checkpoint import render_checkpoint
        print(f"\n{'=' * 110}\n  Visualisierung "
              f"({'alle Kandidaten' if args.viz_all_candidates else 'beste von %d' % args.viz_n_gen}"
              f", Hindernis: {args.viz_obstacle_mode})\n{'=' * 110}")
        figures = {}
        for e in entries:
            if e['kind'] == 'solver':
                continue
            print(f"\n  {e['label']}")
            figures[e['label']] = render_checkpoint(
                e['ckpt'], out_dir=args.viz_dir, n_gen=args.viz_n_gen,
                steps=args.steps, select_best=not args.viz_all_candidates,
                obstacle_mode=args.viz_obstacle_mode, device=str(device),
                grid_res=args.grid_res, erg_K=args.erg_K,
                solver_T=args.solver_T, bspline_deg=args.bspline_deg,
                seed=args.seed, length=args.target_length,
                length_cfg_weight=args.target_length_cfg)
        with open(os.path.join(args.out, 'figures.json'), 'w') as f:
            json.dump(figures, f, indent=1)

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
