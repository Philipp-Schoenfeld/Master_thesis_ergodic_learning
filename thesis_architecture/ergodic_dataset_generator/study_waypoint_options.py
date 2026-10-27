r"""
study_waypoint_options.py
==========================
Can the waypoint solve of `generate_dataset_adam_svgd.py` (Adam-SVGD on 128
waypoints, projected onto an nxi-control-point B-spline) be made smoother
without losing ergodicity? Same 48 densities, starts, seeds and stop rule as
`study_nxi_adam_svgd.py` / `study_waypoint_projection.py`; every variant is
projected onto `--nxi` control points and scored with the same metrics.

Variants (all relative to the current setting `ref`):

  w_smooth x s     smoothness (acceleration) weight scaled by s
  jerk r           extra jerk force w_jerk * sum ||third difference||^2, with
                   w_jerk set so that its energy is r x the acceleration
                   energy of the `ref` solutions (median)
  polish           after convergence 1000 more iterations with 4x smoothness
                   weight, no swarm jitter (fresh Adam state)
  lr x             Adam learning rate (default 2e-3)
  no_svgd          jitter 0: the 8 particles stay identical, the kernel
                   repulsion is zero -> plain Adam on the energy
  ... decay        step size halves every `half_life` iterations of a density
                   (solve_pinned's lr_half_life), floor 1e-4
  (--round 2: combinations around lr 4e-3, the best single change;
   --round 3: around lr 8e-3)

    python study_waypoint_options.py --direct results_adam_svgd_test/nxi_f64/nxi_study.json \
        --out results_waypoint_options
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
import study_waypoint_projection as swp                             # noqa: E402
import svgd_batched as sb                                           # noqa: E402
import torch                                                        # noqa: E402

T = 128
ROUND = {**{v: 2 for v in ('lr 0.008', 'lr 0.004 w x0.5', 'lr 0.004 no_svgd', 'lr 0.004 decay',
                            'lr 0.008 decay', 'lr 0.004 decay w x2')},
         **{v: 3 for v in ('lr 0.008 w x0.5', 'lr 0.016', 'lr 0.008 w x2')}}


class JerkSvgd(sb.BatchedSvgdTorch):
    """`BatchedSvgdTorch` on raw waypoints (B = identity) plus an optional
    jerk term w_jerk * sum ||X[t+3] - 3X[t+2] + 3X[t+1] - X[t]||^2. With B = I
    the control points are the curve, so the term is added to energy and
    gradient directly."""

    def __init__(self, w_smooth, w_jerk=0.0, jitter=None, device='cpu'):
        super().__init__(np.eye(T, dtype=np.float32), device=device, dtype=torch.float64,
                         compute_dtype=torch.float64, smoothness_weight=w_smooth,
                         jitter_scale=jitter)
        self.w_jerk = float(w_jerk)

    def energy_grad(self, P, phik, start, want_grad=True):
        e, g = super().energy_grad(P, phik, start, want_grad)
        if self.w_jerk > 0.0:
            j = P[..., 3:, :] - 3 * P[..., 2:-1, :] + 3 * P[..., 1:-2, :] - P[..., :-3, :]
            e = e + self.w_jerk * (j ** 2).sum(dim=(-1, -2))
            if want_grad:
                gj = torch.zeros_like(P)
                gj[..., 3:, :] += 2 * j
                gj[..., 2:-1, :] -= 6 * j
                gj[..., 1:-2, :] += 6 * j
                gj[..., :-3, :] -= 2 * j
                g = g + self.w_jerk * gj
        return e, g


def jerk_sq(Y):
    j = Y[3:] - 3 * Y[2:-1] + 3 * Y[1:-2] - Y[:-3]
    return float((j ** 2).sum())


def accel_sq(Y):
    a = Y[2:] - 2 * Y[1:-1] + Y[:-2]
    return float((a ** 2).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--direct', required=True)
    p.add_argument('--src_db', default=gen.SRC_DB)
    p.add_argument('--nxi', type=int, default=40)
    p.add_argument('--out', default='results_waypoint_options')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--round', type=int, default=1, choices=[1, 2, 3],
                   help='1: single changes (incl. polish); 2: combinations around lr 4e-3; '
                        '3: around lr 8e-3.')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dcfg = json.load(open(a.direct))['config']
    sub = study.pick_subset(gen.load_source_rows(a.src_db), dcfg['n_shapes'], dcfg['seed'])
    names = [s['name'] for s in sub]
    grids = np.stack([gen.density_grid(r['density_params'], 128) for r in sub])
    phik = gen.target_coeffs(grids, sb.K_FREQ)
    x0s = np.stack([r['x0'] for r in sub])
    seeds = np.array([zlib.crc32(f"{dcfg['seed']}:{r['name']}".encode()) & 0x7FFFFFFF for r in sub])
    P0 = np.stack([gen.linear_init_curve(x0, n_points=T) for x0 in x0s])
    stop = dict(max_iters=dcfg['max_iters'], check_every=dcfg['check_every'],
                patience=dcfg['patience'], tol=dcfg['tol'], min_iters=dcfg['min_iters'])

    from obstacles import bspline_basis_matrix
    BT = bspline_basis_matrix(a.nxi, T, gen.DEGREE).astype(np.float64)
    cache = {}

    def bs_by_nxi(nxi):
        if nxi not in cache:
            cache[nxi] = gen.make_solver(nxi, 'cpu', 'float64')
        return cache[nxi]

    W = sb.W_SMOOTH
    LR = sb.ADAM_LR

    def solve(w_smooth=W, w_jerk=0.0, jitter=None, lr=LR, half_life=None, init=None,
              fixed_iters=None):
        bs = JerkSvgd(w_smooth, w_jerk, jitter, device=a.device)
        kw = dict(stop, lr=lr, lr_half_life=half_life)
        if fixed_iters:
            kw.update(max_iters=fixed_iters, min_iters=fixed_iters, tol=-1.0)
        return gen.solve_pinned(bs, P0 if init is None else init, phik, x0s, seeds, **kw)

    variants = {}
    results = {}

    def run(name, **kw):
        if name != 'ref' and ROUND.get(name, 1) != a.round:
            return
        t0 = time.perf_counter()
        res = solve(**kw)
        variants[name] = dict(kw, sec=time.perf_counter() - t0)
        results[name] = res
        print(f"  {name:<14} {variants[name]['sec']:6.0f}s  iters mean "
              f"{np.mean([r['n_iters'] for r in res]):.0f}", flush=True)

    run('ref')
    Yref = [r['cps'] for r in results['ref']]
    acc_ref = np.median([accel_sq(y) for y in Yref])
    jerk_ref = np.median([jerk_sq(y) for y in Yref])
    print(f"  ref: median accel_sq {acc_ref:.3e}, jerk_sq {jerk_ref:.3e}")
    for s in (0.5, 2.0, 4.0, 8.0):
        run(f'w_smooth x{s:g}', w_smooth=W * s)
    for r in (0.5, 2.0):
        run(f'jerk {r:g}', w_jerk=r * W * acc_ref / jerk_ref)
    for lr in (1e-3, 4e-3):
        run(f'lr {lr:g}', lr=lr)
    run('no_svgd', jitter=0.0)
    # round 2: around the best single change (lr 4e-3)
    run('lr 0.008', lr=8e-3)
    run('lr 0.004 w x0.5', lr=4e-3, w_smooth=W * 0.5)
    run('lr 0.004 no_svgd', lr=4e-3, jitter=0.0)
    run('lr 0.004 decay', lr=4e-3, half_life=2000)
    run('lr 0.008 decay', lr=8e-3, half_life=1500)
    run('lr 0.004 decay w x2', lr=4e-3, half_life=2000, w_smooth=W * 2)
    # round 3
    run('lr 0.008 w x0.5', lr=8e-3, w_smooth=W * 0.5)
    run('lr 0.016', lr=1.6e-2)
    run('lr 0.008 w x2', lr=8e-3, w_smooth=W * 2)
    if a.round == 1:
        t0 = time.perf_counter()
        init = np.stack(Yref)
        res = solve(w_smooth=4 * W, jitter=0.0, init=init, fixed_iters=1000)
        for r, rr in zip(res, results['ref']):
            r['n_iters'] = rr['n_iters'] + 1000
        variants['polish'] = dict(w_smooth=4 * W, sec=time.perf_counter() - t0)
        results['polish'] = res
        print(f"  {'polish':<14} {variants['polish']['sec']:6.0f}s", flush=True)

    rows, summary = [], []
    keys = ('erg_k8', 'erg_k10', 'bending', 'kappa_p95', 'turning', 'path_len')
    print(f"\n  {'variant':<14}" + ''.join(f"{k:>11}" for k in keys) + "      cusps    iters")
    for name, res in results.items():
        cps = [gen.project_to_bspline(r['cps'], x0, BT) for r, x0 in zip(res, x0s)]
        e8, e10 = swp.score(cps, phik, grids, bs_by_nxi)
        for n, c, x8, x10, r in zip(names, cps, e8, e10, res):
            rows.append(dict(variant=name, name=n, erg_k8=float(x8), erg_k10=float(x10),
                             n_iters=r['n_iters'], cps=c.tolist(), **swp.shape_metrics(c)))
        sel = [r for r in rows if r['variant'] == name]
        med = {k: float(np.median([r[k] for r in sel])) for k in keys}
        cus = float(np.mean([r['cusps'] > 0 for r in sel]))
        it = float(np.mean([r['n_iters'] for r in sel]))
        ref_rows = {r['name']: r for r in rows if r['variant'] == 'ref'}
        win_e = float(np.mean([r['erg_k10'] < ref_rows[r['name']]['erg_k10'] for r in sel]))
        win_b = float(np.mean([r['bending'] < ref_rows[r['name']]['bending'] for r in sel]))
        summary.append(dict(variant=name, cusp_share=cus, iters=it, win_erg=win_e,
                            win_bend=win_b, **med))
        print(f"  {name:<14}" + ''.join(f"{med[k]:>11.2e}" if k.startswith('erg') else f"{med[k]:>11.1f}"
                                        for k in keys) + f"   {cus:6.0%}  {it:7.0f}"
              f"   wins vs ref: erg {win_e:4.0%} bend {win_b:4.0%}")
    with open(os.path.join(a.out, f'waypoint_options_round{a.round}.json'), 'w') as f:
        json.dump(dict(variants=variants, rows=rows, summary=summary), f)
    plot(summary, a)


def plot(summary, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor='white')
    for ax, (yk, ylab) in zip(axes, [('bending', r'Bending energy $\int\kappa^2\,ds$ (median)'),
                                     ('kappa_p95', 'Curvature, 95th percentile (median)')]):
        study._style(ax)
        for d in summary:
            ref = d['variant'] == 'ref'
            ax.scatter(d['erg_k10'], d[yk], s=60 if ref else 36,
                       color='#eb6834' if ref else '#2a78d6', zorder=3,
                       edgecolors='white', linewidths=0.8)
            ax.annotate(d['variant'], (d['erg_k10'], d[yk]), fontsize=7, color='#555',
                        xytext=(4, 3), textcoords='offset points')
        ax.set_xlabel('Ergodic error K=10 (median), lower is better', fontsize=9, color='#555')
        ax.set_ylabel(ylab, fontsize=9, color='#555')
    fig.suptitle(f'Waypoint solve options, projected onto nxi={a.nxi} (48 densities; '
                 f'orange = current setting)', fontsize=11, color='#1A1A2E', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, f'waypoint_options_tradeoff_round{a.round}.png'), dpi=140,
                bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    main()
