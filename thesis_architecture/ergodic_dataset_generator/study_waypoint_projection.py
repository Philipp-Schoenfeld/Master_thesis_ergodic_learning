r"""
study_waypoint_projection.py
=============================
Alternative to optimising B-spline control points directly
(`generate_dataset_adam_svgd.py`): solve with Adam-SVGD on many raw waypoints,
then project the waypoint curve onto a B-spline with nxi control points
(least squares at equal time parameter, control point 0 fixed to x0) and use
the projection as the (smoothed) ground truth. Is that smoother / more
ergodic than the direct B-spline solution at the same nxi?

Same 48 densities, start points, init, start pin and plateau stop as
`study_nxi_adam_svgd.py`; the direct results are read from its JSON
(`--direct`) and re-scored with the same metric code, so all numbers are
comparable.

Waypoint solve: `BatchedSvgdTorch` with B = identity (T x T). For T = 128
the energy is exactly the one the B-spline solve evaluates on its 128-point
render. For other T the smoothness weight is scaled by (T/128)^3 (the sum of
squared second differences scales with T^-3), i.e. the same continuous
objective.

    python study_waypoint_projection.py --direct results_adam_svgd_test/nxi_f64/nxi_study.json \
        --T 128 256 --nxi 20 25 32 40 50 --out results_waypoint_projection
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import generate_dataset_adam_svgd as gen                           # noqa: E402
import study_nxi_adam_svgd as study                                 # noqa: E402
import torch                                                        # noqa: E402


def shape_metrics(cps, n=4096):
    """Smoothness of the B-spline of `cps`: `gen.curve_metrics` plus total
    turning int|kappa|ds and the number of near-stops (speed < 5% of the mean,
    i.e. cusp-like hairpins)."""
    from obstacles import bspline_basis_matrix
    m = gen.curve_metrics(cps)
    X = bspline_basis_matrix(cps.shape[0], n, gen.DEGREE).astype(np.float64) @ cps
    h = 1.0 / (n - 1)
    d1 = np.gradient(X, h, axis=0)
    d2 = np.gradient(d1, h, axis=0)
    sp = np.linalg.norm(d1, axis=1)
    k = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(sp, 1e-9) ** 3
    slow = sp < 0.05 * sp.mean()
    m['turning'] = float((k[~slow] * sp[~slow] * h).sum())
    m['cusps'] = int(np.sum(np.diff(slow.astype(int)) == 1) + slow[0])
    return m


def project(Y, x0, nxi):
    """Least-squares B-spline fit of the waypoint curve Y (T, 2) at equal time
    parameter, control point 0 fixed to x0."""
    from obstacles import bspline_basis_matrix
    B = bspline_basis_matrix(nxi, Y.shape[0], gen.DEGREE).astype(np.float64)
    rest = np.linalg.lstsq(B[:, 1:], Y - B[:, :1] @ x0[None, :], rcond=None)[0]
    return np.vstack([x0[None, :], rest])


def score(cps_list, phik, grids, bs_by_nxi):
    """Solver ergodic error (K=8, 128-point render) and K=10 error for B-spline control points."""
    nxi = cps_list[0].shape[0]
    bs, B32 = bs_by_nxi(nxi)
    X = torch.matmul(torch.as_tensor(B32, dtype=torch.float64),
                     torch.as_tensor(np.stack(cps_list)))
    e8 = gen.ergodic_error(bs, X, torch.as_tensor(phik)).numpy()
    e10 = study.erg_k10(cps_list, grids)
    return e8, e10


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--direct', required=True, help='nxi_study.json of study_nxi_adam_svgd.py')
    p.add_argument('--src_db', default=gen.SRC_DB)
    p.add_argument('--T', type=int, nargs='+', default=[128, 256])
    p.add_argument('--nxi', type=int, nargs='+', default=[20, 25, 32, 40, 50])
    p.add_argument('--out', default='results_waypoint_projection')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    direct = json.load(open(a.direct))
    dcfg = direct['config']
    sub = study.pick_subset(gen.load_source_rows(a.src_db), dcfg['n_shapes'], dcfg['seed'])
    names = [s['name'] for s in sub]
    from svgd_batched import BatchedSvgdTorch, K_FREQ, W_SMOOTH
    grids = np.stack([gen.density_grid(r['density_params'], 128) for r in sub])
    phik = gen.target_coeffs(grids, K_FREQ)
    x0s = np.stack([r['x0'] for r in sub])
    seeds = np.array([zlib.crc32(f"{dcfg['seed']}:{r['name']}".encode()) & 0x7FFFFFFF for r in sub])

    cache = {}

    def bs_by_nxi(nxi):
        if nxi not in cache:
            cache[nxi] = gen.make_solver(nxi, 'cpu', 'float64')
        return cache[nxi]

    rows = []
    # direct B-spline optimisation, re-scored
    for nxi in a.nxi:
        d = {r['name']: r for r in direct['rows'] if r['nxi'] == nxi}
        if not d:
            continue
        cps = [np.asarray(d[n]['cps']) for n in names]
        e8, e10 = score(cps, phik, grids, bs_by_nxi)
        for n, c, x8, x10 in zip(names, cps, e8, e10):
            rows.append(dict(method='direct', T=0, nxi=nxi, name=n, erg_k8=float(x8),
                             erg_k10=float(x10), n_iters=d[n]['n_iters'], cps=c.tolist(),
                             **shape_metrics(c)))

    for T in a.T:
        w = W_SMOOTH * (T / 128.0) ** 3
        bs = BatchedSvgdTorch(np.eye(T, dtype=np.float32), device=a.device,
                              dtype=torch.float64, compute_dtype=torch.float64,
                              smoothness_weight=w)
        P0 = np.stack([gen.linear_init_curve(x0, n_points=T) for x0 in x0s])
        t0 = time.perf_counter()
        res = gen.solve_pinned(bs, P0, phik, x0s, seeds, max_iters=dcfg['max_iters'],
                               check_every=dcfg['check_every'], patience=dcfg['patience'],
                               tol=dcfg['tol'], min_iters=dcfg['min_iters'])
        sec = time.perf_counter() - t0
        Y = [r['cps'] for r in res]                                      # (T, 2) waypoints
        e_raw = np.array([r['erg'] for r in res])
        print(f"  waypoints T={T} (w_smooth {w:g}): {sec:.0f}s, iters mean "
              f"{np.mean([r['n_iters'] for r in res]):.0f}, raw erg_k8 median "
              f"{np.median(e_raw):.2e}", flush=True)
        for n, y, r in zip(names, Y, res):
            rows.append(dict(method='waypoints_raw', T=T, nxi=T, name=n, erg_k8=r['erg'],
                             erg_k10=float('nan'), n_iters=r['n_iters'],
                             accel_sq=float((np.diff(y[::max(1, T // 128)], 2, axis=0) ** 2).sum())))
        from obstacles import bspline_basis_matrix
        for nxi in a.nxi:
            BT = bspline_basis_matrix(nxi, T, gen.DEGREE).astype(np.float64)
            cps = [np.clip(project(y, x0, nxi), 0.0, 1.0) for y, x0 in zip(Y, x0s)]
            e8, e10 = score(cps, phik, grids, bs_by_nxi)
            for n, c, y, x8, x10, r in zip(names, cps, Y, e8, e10, res):
                rows.append(dict(method='projected', T=T, nxi=nxi, name=n, erg_k8=float(x8),
                                 erg_k10=float(x10), n_iters=r['n_iters'],
                                 fit_rmse=float(np.linalg.norm(BT @ c - y, axis=1).mean()),
                                 cps=c.tolist(), **shape_metrics(c)))
        with open(os.path.join(a.out, 'waypoint_projection.json'), 'w') as f:
            json.dump(rows, f)

    keys = ('erg_k8', 'erg_k10', 'bending', 'kappa_p95', 'turning', 'accel_sq', 'path_len')
    print(f"\n  {'method':<16}{'T':>4}{'nxi':>5} " + ''.join(f"{k:>11}" for k in keys) + "   cusp-curves")
    summary = []
    for meth, T in [('direct', 0)] + [(m, T) for T in a.T for m in ('projected',)]:
        for nxi in a.nxi:
            sel = [r for r in rows if r['method'] == meth and r['T'] == T and r['nxi'] == nxi]
            if not sel:
                continue
            med = {k: float(np.median([r[k] for r in sel])) for k in keys}
            cus = float(np.mean([r['cusps'] > 0 for r in sel]))
            summary.append(dict(method=meth, T=T, nxi=nxi, cusp_share=cus, **med))
            print(f"  {meth:<16}{T:>4}{nxi:>5} " + ''.join(
                f"{med[k]:>11.2e}" if k.startswith('erg') or k == 'accel_sq' else f"{med[k]:>11.1f}"
                for k in keys) + f"   {cus:6.0%}")
    with open(os.path.join(a.out, 'waypoint_projection_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    plot(summary, a)


def plot(summary, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('erg_k10', 'Ergodic error (K=10)'), ('bending', r'Bending energy $\int\kappa^2\,ds$'),
              ('kappa_p95', 'Curvature, 95th percentile'), ('cusp_share', 'Curves with a cusp')]
    series = [('direct', 0, 'Direct B-spline solve', '#2a78d6', '-')]
    cols = ['#eb6834', '#1baf7a']
    for i, T in enumerate(a.T):
        series.append(('projected', T, f'Waypoints T={T}, projected', cols[i % 2], '--'))
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), facecolor='white')
    for ax, (key, title) in zip(axes, panels):
        study._style(ax)
        for meth, T, lab, col, ls in series:
            s = [d for d in summary if d['method'] == meth and d['T'] == T]
            ax.plot([d['nxi'] for d in s], [d[key] for d in s], color=col, lw=2, ls=ls,
                    marker='o', ms=5, label=lab)
        if key != 'cusp_share':
            ax.set_yscale('log')
        ax.set_title(title, fontsize=10, color='#1A1A2E')
        ax.set_xlabel('Control points (nxi)', fontsize=9, color='#555')
    axes[0].legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#ddd')
    fig.suptitle('Direct B-spline optimisation vs. waypoint solve + B-spline projection (medians, 48 densities)',
                 fontsize=11, color='#1A1A2E', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, 'waypoint_projection_overview.png'), dpi=140,
                bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    main()
