r"""
study_nxi_adam_svgd.py
=======================
How many B-spline control points for the Adam-SVGD training DB
(`generate_dataset_adam_svgd.py`)? Few control points make the curve smooth
but cannot follow the density; many follow it and start to wiggle.

Runs exactly the generator's solver (linear init, hard start pin, plateau
stop, max 10000 iterations) on a fixed random subset of densities -- half
base shapes, half the same shapes' `#unk0` large-area variants -- once per
nxi, and records per density:

  erg_k8     the solver's own ergodic error (K=8, the quantity it minimises)
  erg_k10    ergodic error at K=10 (the project's standard metric,
             `ergodic_energy_torch`), on a 512-point render -- a check that a
             lower erg_k8 is not just fitting the solver's own basis
  bending    int kappa^2 ds (parameterisation-free smoothness), kappa_p95
  path_len, accel_sq (solver smoothness term), n_iters, converged

Writes <out>/nxi_study.json (all rows), <out>/nxi_study_summary.csv
(medians/IQR per nxi and variant), <out>/nxi_study_overview.png and
<out>/nxi_study_examples.png.

    python study_nxi_adam_svgd.py --out results_nxi_study --n_shapes 24 \
        --nxi 12 16 20 25 32 40 50
"""
import argparse
import csv
import json
import os
import sys
import time
import zlib

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import generate_dataset_adam_svgd as gen                           # noqa: E402
import torch                                                        # noqa: E402

COL_BASE, COL_UNK = '#2a78d6', '#eb6834'


def pick_subset(rows, n_shapes, seed):
    base = [r for r in rows if '#' not in r['name'] and r['split'] == 'train']
    by_name = {r['name']: r for r in rows}
    rng = np.random.default_rng(seed)
    names = [base[i]['name'] for i in rng.choice(len(base), n_shapes, replace=False)]
    out = []
    for n in names:
        out.append(by_name[n])
        if f'{n}#unk0' in by_name:
            out.append(by_name[f'{n}#unk0'])
    return out


def erg_k10(cps_list, grids):
    from ergodic_energy_torch import coeffs_from_points, make_k_grid, target_coeffs_from_grid
    from obstacles import bspline_basis_matrix
    k_idx, Lam = make_k_grid(10)
    k_idx, Lam = torch.as_tensor(k_idx), torch.as_tensor(Lam)
    phi = target_coeffs_from_grid(torch.as_tensor(grids), k_idx)
    nxi = cps_list[0].shape[0]
    B = torch.as_tensor(bspline_basis_matrix(nxi, 512, gen.DEGREE), dtype=torch.float64)
    X = torch.matmul(B, torch.as_tensor(np.stack(cps_list)))
    c = coeffs_from_points(X, k_idx)
    return (0.5 * (Lam * (c - phi) ** 2).sum(-1)).numpy()


def summarise(rows, nxis):
    out = []
    for nxi in nxis:
        for var in ('base', 'unk0'):
            sel = [r for r in rows if r['nxi'] == nxi and r['variant'] == var]
            if not sel:
                continue
            d = dict(nxi=nxi, variant=var, n=len(sel),
                     converged=float(np.mean([r['converged'] for r in sel])))
            for k in ('erg_k8', 'erg_k10', 'bending', 'kappa_p95', 'path_len', 'accel_sq', 'n_iters'):
                v = np.array([r[k] for r in sel], dtype=np.float64)
                d[f'{k}_median'] = float(np.median(v))
                d[f'{k}_q25'] = float(np.percentile(v, 25))
                d[f'{k}_q75'] = float(np.percentile(v, 75))
            d['sec_per_density'] = float(np.mean([r['sec'] for r in sel]))
            out.append(d)
    return out


def _style(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(labelsize=8, colors='#555')


def plot_overview(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('erg_k10', 'Ergodic error (K=10)', True),
              ('bending', r'Bending energy $\int\kappa^2\,ds$', True),
              ('kappa_p95', 'Curvature, 95th percentile', True),
              ('n_iters', 'Solver iterations until stop', False)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), facecolor='white')
    for ax, (key, title, logy) in zip(axes, panels):
        _style(ax)
        for var, col, lab in (('base', COL_BASE, 'Base shape'),
                              ('unk0', COL_UNK, 'Base + large area')):
            s = [d for d in summary if d['variant'] == var]
            x = [d['nxi'] for d in s]
            ax.fill_between(x, [d[f'{key}_q25'] for d in s], [d[f'{key}_q75'] for d in s],
                            color=col, alpha=0.15, lw=0)
            ax.plot(x, [d[f'{key}_median'] for d in s], color=col, lw=2, marker='o', ms=5,
                    label=lab)
        if logy:
            ax.set_yscale('log')
        ax.set_title(title, fontsize=10, color='#1A1A2E')
        ax.set_xlabel('Control points (nxi)', fontsize=9, color='#555')
    axes[0].legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#ddd')
    fig.suptitle('Adam-SVGD with linear init: median and interquartile range per nxi',
                 fontsize=11, color='#1A1A2E', y=1.03)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_examples(rows, grids_by_name, nxis, names, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(_here))
    from obstacles import bspline_basis_matrix
    from flow_matching_runner_start import WHITE_INFERNO
    fig, axes = plt.subplots(len(names), len(nxis), figsize=(2.6 * len(nxis), 2.7 * len(names)),
                             facecolor='white', squeeze=False)
    for i, name in enumerate(names):
        g = grids_by_name[name]
        for j, nxi in enumerate(nxis):
            ax = axes[i][j]
            ax.set_facecolor('white')
            ax.imshow(g / g.max(), extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO,
                      vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)
            r = next(r for r in rows if r['name'] == name and r['nxi'] == nxi)
            cps = np.asarray(r['cps'])
            X = bspline_basis_matrix(nxi, 512, gen.DEGREE) @ cps
            ax.plot(X[:, 0], X[:, 1], color='#1565C0', lw=1.6, alpha=0.9, zorder=2)
            ax.scatter(cps[:, 0], cps[:, 1], color='#1565C0', s=6, alpha=0.5, zorder=2)
            ax.scatter([cps[0, 0]], [cps[0, 1]], color='#1565C0', marker='*', s=90,
                       edgecolors='white', linewidths=0.6, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('#ccc')
            ax.set_title(f"nxi={nxi}  E10={r['erg_k10']:.1e}\nbend={r['bending']:.0f}",
                         fontsize=7, color='#1A1A2E')
            if j == 0:
                ax.set_ylabel(name, fontsize=7, color='#555')
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src_db', default=gen.SRC_DB)
    p.add_argument('--out', default='results_nxi_study')
    p.add_argument('--n_shapes', type=int, default=24,
                   help='Base shapes; each comes with its #unk0 variant (2x densities).')
    p.add_argument('--nxi', type=int, nargs='+', default=[12, 16, 20, 25, 32, 40, 50])
    p.add_argument('--max_iters', type=int, default=10000)
    p.add_argument('--check_every', type=int, default=50)
    p.add_argument('--patience', type=int, default=1000)
    p.add_argument('--tol', type=float, default=0.01)
    p.add_argument('--min_iters', type=int, default=500)
    p.add_argument('--compute_dtype', default='float64', choices=['float64', 'float32'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=20261024)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from svgd_batched import K_FREQ
    sub = pick_subset(gen.load_source_rows(a.src_db), a.n_shapes, a.seed)
    t0 = time.perf_counter()
    grids = np.stack([gen.density_grid(r['density_params'], 128) for r in sub])
    phik = gen.target_coeffs(grids, K_FREQ)
    print(f"  {len(sub)} densities ({a.n_shapes} base + #unk0), grids in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    x0s = np.stack([r['x0'] for r in sub])
    seeds = np.array([zlib.crc32(f"{a.seed}:{r['name']}".encode()) & 0x7FFFFFFF for r in sub])

    rows = []
    for nxi in a.nxi:
        bs, B32 = gen.make_solver(nxi, a.device, a.compute_dtype)
        P0 = gen.init_cps_for(B32, x0s)
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
        t = time.perf_counter()
        res = gen.solve_pinned(bs, P0, phik, x0s, seeds, max_iters=a.max_iters,
                               check_every=a.check_every, patience=a.patience, tol=a.tol,
                               min_iters=a.min_iters)
        if a.device.startswith('cuda'):
            torch.cuda.synchronize()
        sec = time.perf_counter() - t
        e10 = erg_k10([r['cps'] for r in res], grids)
        for r, s, e in zip(res, sub, e10):
            m = gen.curve_metrics(r['cps'])
            rows.append(dict(name=s['name'], variant='unk0' if '#unk0' in s['name'] else 'base',
                             nxi=nxi, erg_k8=r['erg'], erg_k10=float(e), n_iters=r['n_iters'],
                             converged=r['converged'], sec=sec / len(sub),
                             cps=r['cps'].tolist(), **m))
        cur = [r for r in rows if r['nxi'] == nxi]
        print(f"  nxi={nxi:3d}  {sec:6.1f}s  erg_k10 median {np.median([r['erg_k10'] for r in cur]):.2e}"
              f"  bending median {np.median([r['bending'] for r in cur]):8.0f}"
              f"  kappa_p95 median {np.median([r['kappa_p95'] for r in cur]):6.1f}"
              f"  iters mean {np.mean([r['n_iters'] for r in cur]):.0f}"
              f"  converged {np.mean([r['converged'] for r in cur]):.0%}", flush=True)
        with open(os.path.join(a.out, 'nxi_study.json'), 'w') as f:
            json.dump(dict(config=vars(a), rows=rows), f)

    summary = summarise(rows, a.nxi)
    with open(os.path.join(a.out, 'nxi_study_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    plot_overview(summary, os.path.join(a.out, 'nxi_study_overview.png'))
    ex = [sub[0]['name'], sub[1]['name'], sub[2]['name'], sub[3]['name']]
    plot_examples(rows, {s['name']: g for s, g in zip(sub, grids)}, a.nxi, ex,
                  os.path.join(a.out, 'nxi_study_examples.png'))
    print(f"  wrote {a.out}/nxi_study.json, nxi_study_summary.csv, *.png")


if __name__ == '__main__':
    main()
