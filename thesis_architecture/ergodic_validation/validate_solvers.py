r"""
validate_solvers.py -- test bench: do the solver variants of the smoothness pipeline produce
ergodic trajectories?
=========================================================================================
Runs every variant of `evaluation_full_matrix/run_smoothness_comparison_v2.py` through the SAME
per-cell function the pipeline uses (`run_cell`), on canonical targets with known answers
(`targets.py`), and judges the logged states with the independent metrics of
`ergodicity_metrics.py` -- not with the truncated Fourier metric the solvers optimise.

Pass criteria per (target, variant), on the MEDIAN over the 30 candidates at the final iteration
(fixed a priori from the i.i.d. finite-sample floor, see test_ergodicity_metrics.py):

  C1 hellinger <= floor + 0.20       (floor = i.i.d. target samples with the same T)
  C2 tv_4      <= 0.25               (time vs. mass fraction on a 4 x 4 partition)
  C3 low_excess <= 0.10              (extra time spent where the target is ~0)
  C4 hellinger_final <= hellinger_init + 0.02   (the solver must not make it worse)
  C5 hellinger_final <= min over checkpoints + 0.05   (no late degradation)
  C6 region checks (blobs: 75 % left; ring: hole) |time fraction - mass| <= 0.10
  coverage (target mass within 0.05 of the path) is reported but NOT a pass criterion: a path of
  length L covers at most ~2 * 0.05 * L of the square, so on the uniform target (L ~ 5-8) full
  footprint coverage is a length limit, not an ergodicity property (the first bench run had C7 >= 0.75
  and failed only the uniform target with it, for all solver families alike).

Modes
  run     python validate_solvers.py run --out <dir> --targets sun_gmm3,A --variants ...
          writes <dir>/<target>__<variant>.npz (skipped if present: resumable, parallelisable)
  report  python validate_solvers.py report --out <dir>
          writes report.json, report.md (pass/fail table) and the figures; exit code 1 if any
          check fails
"""
import argparse
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
_eval = os.path.join(_arch, 'evaluation_full_matrix')
for _p in (_here, _eval, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'), os.path.join(_root, 'bsplinax-main'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import ergodicity_metrics as em
import targets as tg

CHECKPOINTS = (0, 25, 100, 300, 1000, 2000)
DEFAULT_TARGETS = ('sun_gmm3', 'blobs_75_25', 'uniform', 'ring', 'A', 'organic_30', 'korean_5')
#: display order: solver family, then initialisation / parameterisation
ORDER = ('cfm_only', 'linear_sun_pointmass', 'linear_sun_jerk',
         'cfm_adam_bspline', 'linear_adam_bspline', 'linear_adam_waypoint', 'linear_adam_waypoint_smooth',
         'cfm_bspline_svgd', 'linear_bspline_svgd', 'linear_waypoint_svgd', 'linear_waypoint_svgd_smooth')
THRESH = dict(hell_margin=0.20, tv_4=0.25, low_excess=0.10, worse_than_init=0.02,
              late_degradation=0.05, region=0.10)


def _dense(row, B32):
    from state_codec import unpack_states
    st = unpack_states(row['states'], row['n_states'], row['nxi'])
    return np.einsum('pi,sid->spd', B32, st) if row['log_space'] == 'cps' else st


def run(args):
    import run_smoothness_comparison as rsc1
    import run_smoothness_comparison_v2 as v2
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    B32 = rsc1.basis_matrix()
    variants = args.variants.split(',')
    planner = None
    for name, grid in tg.load(args.targets.split(',')):
        truth_np = grid.astype(np.float32)
        lin = None
        cfm = None
        for variant in variants:
            path = os.path.join(args.out, f'{name}__{variant}.npz')
            if os.path.exists(path):
                continue
            if variant in v2.NEEDS_CFM:
                if cfm is None:
                    if planner is None:
                        import apply_cfm_belief as acb
                        from run_eval_matrix import DEFAULT_CKPT
                        planner = acb.CfmPlanner(ckpt=DEFAULT_CKPT, nxi=v2.NXI, pts=v2.N_POINTS, device=device)
                    cfm = rsc1.sample_cfm_curves(planner, torch.as_tensor(truth_np, device=device),
                                                 args.n_init, device, rsc1.task_seed(name, 'cfm'))
                init = cfm
            else:
                if lin is None:
                    lin, _ = rsc1.linear_curves(args.n_init)
                init = lin
            t0 = time.time()
            rows = v2.run_cell(variant, init, truth_np, args.n_iters, args.n_iters, B32, device)
            dense = np.stack([_dense(r, B32) for r in rows])            # (C, S, T, 2)
            S = dense.shape[1]
            ck = [c for c in CHECKPOINTS if c < S] + ([S - 1] if S - 1 not in CHECKPOINTS else [])
            ck = sorted(set(min(c, S - 1) for c in ck))
            np.savez_compressed(path, checkpoints=np.array(ck), states=dense[:, ck].astype(np.float32),
                                grid=grid, n_iters=args.n_iters, seconds=time.time() - t0)
            print(f'[validate] {name:12s} {variant:30s} {time.time() - t0:6.1f}s', flush=True)


def _region_errors(name, grid, X):
    out = []
    for label, mask, _ in tg.REGION_CHECKS.get(name, []):
        mass = tg.region_mass(grid, mask)
        frac = np.array([mask(x[:, 0], x[:, 1]).mean() for x in X])
        out.append((label, mass, frac))
    return out


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import glob
    files = sorted(glob.glob(os.path.join(args.out, '*__*.npz')))
    res, floors = {}, {}
    for f in files:
        name, variant = os.path.basename(f)[:-4].split('__')
        z = np.load(f)
        grid, states, ck = z['grid'], z['states'], z['checkpoints']
        if name not in floors:
            floors[name] = float(np.median(em.evaluate(em.oracle_samples(grid, 30, states.shape[2], seed=7), grid)['hellinger']))
        per_ck = [em.evaluate(states[:, i], grid) for i in range(len(ck))]
        fin = per_ck[-1]
        med = {k: float(np.median(v)) for k, v in fin.items()}
        hell_trace = [float(np.median(m['hellinger'])) for m in per_ck]
        regions = [(lab, mass, float(np.median(fr))) for lab, mass, fr in _region_errors(name, grid, states[:, -1])]
        checks = {
            'C1 hellinger': med['hellinger'] <= floors[name] + THRESH['hell_margin'],
            'C2 tv_4': med['tv_4'] <= THRESH['tv_4'],
            'C3 low_excess': med['low_excess'] <= THRESH['low_excess'],
            'C4 not worse than init': hell_trace[-1] <= hell_trace[0] + THRESH['worse_than_init'],
            'C5 no late degradation': hell_trace[-1] <= min(hell_trace) + THRESH['late_degradation'],
            'C6 regions': all(abs(fr - mass) <= THRESH['region'] for _, mass, fr in regions),
        }
        if variant == 'cfm_only':                      # no iterations: C4/C5 are vacuous
            checks.pop('C4 not worse than init'), checks.pop('C5 no late degradation')
        res.setdefault(name, {})[variant] = dict(
            median=med, hell_trace=hell_trace, checkpoints=ck.tolist(), regions=regions,
            checks=checks, passed=all(checks.values()), seconds=float(z['seconds']),
            hell_floor=floors[name])
    with open(os.path.join(args.out, 'report.json'), 'w') as fh:
        json.dump(res, fh, indent=1, default=float)
    _write_markdown(res, args.out)
    _figures(res, args.out)
    n_fail = sum(not v['passed'] for t in res.values() for v in t.values())
    n_all = sum(len(t) for t in res.values())
    print(f'[validate] {n_all - n_fail}/{n_all} (target, variant) cells pass -> {args.out}/report.md')
    return 1 if n_fail else 0


def _ordered(variants):
    known = [v for v in ORDER if v in variants]
    return known + sorted(v for v in variants if v not in ORDER)


def _write_markdown(res, out):
    variants = _ordered({v for t in res.values() for v in t})
    lines = ['# Ergodicity validation', '',
             'Median over 30 candidates at the final iteration. hell = Hellinger distance of the '
             'time-averaged visitation density to the target (floor = i.i.d. samples), tv4 = time vs. '
             'mass fraction on a 4x4 partition, low = extra time outside the support, cov = target mass '
             'within 0.05 of the path. FAIL lists the failed checks.', '']
    for name, rv in res.items():
        lines += [f'## {name} (hellinger floor {next(iter(rv.values()))["hell_floor"]:.2f})', '',
                  '| variant | hell init -> final | tv4 | low | cov | E10 | path len | result |',
                  '|---|---|---|---|---|---|---|---|']
        for v in variants:
            if v not in rv:
                continue
            r = rv[v]
            m, tr = r['median'], r['hell_trace']
            bad = [k for k, ok in r['checks'].items() if not ok]
            lines.append(f"| {v} | {tr[0]:.2f} -> {tr[-1]:.2f} | {m['tv_4']:.2f} | {m['low_excess']:.2f} | "
                         f"{m['coverage']:.2f} | {m['ergodic_k10']:.2f} | {m['path_len']:.1f} | "
                         f"{'PASS' if r['passed'] else 'FAIL: ' + ', '.join(bad)} |")
        lines.append('')
    with open(os.path.join(out, 'report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


def _figures(res, out):
    """Per target: for every variant the MEDIAN-quality candidate (by Hellinger, not the best --
    an honest representative), its time-averaged visitation density next to the target, and the
    Hellinger trace over the iterations."""
    import glob
    import matplotlib.pyplot as plt
    from scipy import ndimage
    sys.path.insert(0, _eval)
    import viz
    _summary_figure(res, out)
    for name, rv in res.items():
        variants = _ordered(rv)
        fig, axes = plt.subplots(3, len(variants), figsize=(2.5 * len(variants), 7.6), facecolor='white',
                                 squeeze=False)
        for j, v in enumerate(variants):
            z = np.load(os.path.join(out, f'{name}__{v}.npz'))
            grid, X = z['grid'], z['states'][:, -1]
            h = em.evaluate(X, grid)['hellinger']
            i_med = int(np.argsort(h)[len(h) // 2])
            ax = axes[0, j]
            viz.style_axes(ax)
            viz.draw_density(ax, grid)
            for k in range(len(X)):
                ax.plot(X[k, :, 0], X[k, :, 1], color='#00C853', lw=0.6, alpha=0.18)
            ax.plot(X[i_med, :, 0], X[i_med, :, 1], color='#00C853', lw=2.0, alpha=0.95)
            ax.set_title(f"{v}\n{'PASS' if rv[v]['passed'] else 'FAIL'}  hell {np.median(h):.2f}",
                         fontsize=7, color='#1A1A2E' if rv[v]['passed'] else '#C62828')
            ax = axes[1, j]
            vis = ndimage.gaussian_filter(em.visitation(X[i_med]), em.BLUR * em.GRID)
            ax.imshow(vis, origin='lower', extent=[0, 1, 0, 1], cmap=viz.CMAP)
            ax.contour(np.linspace(0, 1, grid.shape[0]), np.linspace(0, 1, grid.shape[0]), grid,
                       levels=[0.1], colors='#1565C0', linewidths=0.7)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f'time spent (median cand.), hell {h[i_med]:.2f}', fontsize=7, color='#1A1A2E')
            ax = axes[2, j]
            ax.plot(rv[v]['checkpoints'], rv[v]['hell_trace'], color='#1565C0', marker='o', ms=3)
            ax.axhline(rv[v]['hell_floor'], color='#888', ls='--', lw=0.8)
            ax.set_xscale('symlog', linthresh=25)
            ax.set_xlim(0, max(rv[v]['checkpoints'][-1], 1) * 1.1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=6)
            ax.set_title('Hellinger vs. iteration (dashed: iid floor)', fontsize=7, color='#1A1A2E')
        fig.suptitle(f'{name}: top = all 30 final trajectories (bold: median candidate), middle = its '
                     'time-averaged visitation density (blue contour: target support), bottom = convergence',
                     fontsize=9, color='#1A1A2E')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out, f'fig_{name}.png'), dpi=110, facecolor='white')
        plt.close(fig)


def _summary_figure(res, out):
    """One glance: Hellinger (median over candidates, final iteration) per variant x target, minus
    the i.i.d. floor of that target; failed cells are hatched and list their failed checks."""
    import matplotlib.pyplot as plt
    targets = list(res)
    variants = _ordered({v for t in res.values() for v in t})
    M = np.full((len(variants), len(targets)), np.nan)
    for j, t in enumerate(targets):
        for i, v in enumerate(variants):
            if v in res[t]:
                M[i, j] = res[t][v]['median']['hellinger'] - res[t][v]['hell_floor']
    fig, ax = plt.subplots(figsize=(1.35 * len(targets) + 3.2, 0.42 * len(variants) + 1.6), facecolor='white')
    im = ax.imshow(M, cmap='RdYlGn_r', vmin=0.0, vmax=0.5, aspect='auto')
    for j, t in enumerate(targets):
        for i, v in enumerate(variants):
            if v not in res[t]:
                continue
            r = res[t][v]
            bad = [k.split()[0] for k, ok in r['checks'].items() if not ok]
            txt = f"{r['median']['hellinger']:.2f}" + ('' if r['passed'] else '\n' + ','.join(bad))
            dark = M[i, j] < 0.12                      # dark green cells: light text
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='white' if dark else '#1A1A2E', fontweight='normal' if r['passed'] else 'bold')
            if not r['passed']:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch='///',
                                           edgecolor='#C62828', lw=0))
    ax.set_xticks(range(len(targets)), targets, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(variants)), variants, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label('Hellinger above the i.i.d. floor', fontsize=8)
    ax.set_title('Ergodicity check: number = Hellinger of the time-averaged visitation density '
                 '(median over 30 candidates);\nhatched = FAIL with the failed checks (C1..C6, see report.md)',
                 fontsize=8.5, color='#1A1A2E', loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'summary.png'), dpi=130, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('mode', choices=['run', 'report'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--targets', default=','.join(DEFAULT_TARGETS))
    ap.add_argument('--variants', default=None)
    ap.add_argument('--n_init', type=int, default=30)
    ap.add_argument('--n_iters', type=int, default=2000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if args.mode == 'run':
        if args.variants is None:
            import run_smoothness_comparison_v2 as v2
            args.variants = ','.join(v2.VARIANTS)
        run(args)
    else:
        sys.exit(report(args))


if __name__ == '__main__':
    main()
