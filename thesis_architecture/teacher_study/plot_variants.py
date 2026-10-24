r"""
plot_variants.py -- trajectories of several variants on several holdout shapes, side by side
=============================================================================================
Rows = shapes, columns = variants (first column: the DB teacher). Every panel shows the B-spline
trajectory (25 control points) of ONE candidate per (shape, variant), chosen by the study's rule
(smoothest candidate with Hellinger <= the variant's best on that shape + eps) -- no hand picking.
Caption of each panel: Hellinger | smoothness energy | acceleration RMS. The `--best` column gets a
green frame.

    python plot_variants.py --out_tag full_20261024 --best lm_K16_cont_o2_w15
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import matplotlib.pyplot as plt                                            # noqa: E402

import analyze_teacher_study as an                                         # noqa: E402
from analyze_teacher_study import C_DARK, C_GEN, C_GREY, C_GT, WHITE_INFERNO   # noqa: E402
from plot_best import pick                                                 # noqa: E402

VARIANTS = ['teacher_db', 'gui_lr8e-3_n3000', 'adam_o2_w15', 'tsvec_o2_w15', 'sun_sinkhorn_o2_w3000',
            'sun_sinkhorn_o3_w3', 'lm_o2_w15', 'lm_K16_o2_w15', 'lm_cont_o2_w15', 'lm_K16_cont_o2_w15']
SHAPES = 'A,korean_5,digit_5,organic_0,rand_ana_poly_30,rand_gmm_complex_10'
TITLES = {'teacher_db': 'DB teacher', 'gui_lr8e-3_n3000': 'GUI Adam\n(reference)', 'adam_o2_w15': 'Adam-SVGD\nw=15',
          'tsvec_o2_w15': 'TSVEC-SVGD\nw=15', 'sun_sinkhorn_o2_w3000': 'Sun Sinkhorn\naccel w=3000',
          'sun_sinkhorn_o3_w3': 'Sun Sinkhorn\njerk w=3', 'lm_o2_w15': 'LM\nw=15', 'lm_K16_o2_w15': 'LM, K=16\nw=15',
          'lm_cont_o2_w15': 'LM, continuation\n150>50>15', 'lm_K16_cont_o2_w15': 'LM, K=16,\ncontinuation'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--best', default='lm_K16_cont_o2_w15')
    ap.add_argument('--shapes', default=SHAPES)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--stage', default='bs25')
    ap.add_argument('--name', default='variants_grid.png')
    args = ap.parse_args()

    import pipeline as pl
    run_dir = os.path.join(_here, 'results', args.out_tag)
    d = an.apply_boundary_filter(an.load(os.path.join(run_dir, 'metrics.csv')), 0.0)
    shapes = [s.strip() for s in args.shapes.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]
    sds = {s.name: s for s in pl.load_shapes(shapes)}

    fig, axes = plt.subplots(len(shapes), len(variants), figsize=(2.6 * len(variants), 2.9 * len(shapes)),
                             facecolor='white', squeeze=False)
    for r, shape in enumerate(shapes):
        sd = sds[shape]
        for c, cfg in enumerate(variants):
            ax = axes[r][c]
            ax.imshow(sd.density, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO, vmin=0, vmax=1,
                      alpha=0.55, aspect='equal', zorder=0)
            if cfg == 'teacher_db':
                j = np.flatnonzero((d['shape'] == shape) & (d['config'] == cfg) & (d['stage'] == args.stage))[0]
                color, lw = C_GT, 2.5
            else:
                j = pick(d, shape, cfg, args.stage, args.eps)
                color, lw = C_GEN, 2.2
            f = np.load(os.path.join(run_dir, 'finals', f'{shape}__{cfg}.npz'))
            X = an._render_stage(f[args.stage][int(d['cand'][j])], args.stage)
            ax.plot(X[:, 0], X[:, 1], color=color, lw=lw, alpha=0.95, zorder=3)
            ax.plot(*sd.x0, 'o', color=C_DARK, ms=4, zorder=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            frame = cfg == args.best
            for sp in ax.spines.values():
                sp.set_color(C_GEN if frame else '#cccccc')
                sp.set_linewidth(3.5 if frame else 0.8)
            ax.set_xlabel(f'H {d["hellinger"][j]:.3f} | sm {d["smooth"][j]:.2f} | acc {d["acc_rms"][j]:.0f}',
                          fontsize=8, color=C_GREY)
            if r == 0:
                ax.set_title(TITLES.get(cfg, cfg), fontsize=9, color='#007A33' if frame else C_DARK,
                             fontweight='bold' if frame else 'normal')
            if c == 0:
                ax.set_ylabel(shape, fontsize=10, color=C_DARK)
    fig.suptitle('Teacher candidates per variant (B-spline, 25 control points; one candidate per cell chosen by '
                 'the selection rule). H = Hellinger (lower = more ergodic), sm = acceleration energy, '
                 'acc = acceleration RMS (lower = smoother)', color=C_DARK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(run_dir, 'plots', args.name)
    fig.savefig(out, dpi=100, facecolor='white')
    print(f'[plot_variants] wrote {out}')


if __name__ == '__main__':
    main()
