r"""
analyze_teacher_study.py -- tables, selection and figures for a teacher-study run
==================================================================================
Reads `results/<tag>/metrics.csv` (+ `finals/`) and writes into the same folder

  summary_config_stage.csv   per (config, stage): candidate means over all shapes, the hit rate and the
                             rule-based best-of-N means (see "Selection rule")
  selection.csv              per shape: the chosen (config, stage, candidate) next to the DB teacher
  report.md                  ranking + headline numbers
  plots/pareto.png           ergodicity vs smoothness, waypoints -> B-spline projection -> polish
  plots/force_sweeps.png     Hellinger and smoothness against the smoothness weight, per solver / order
  plots/projection_effect.png  what the B-spline projection does, per configuration
  plots/selection_gallery.png  chosen trajectory (green) vs DB teacher (blue) on every holdout density

Selection rule (Philipp's choice, 2026-10-24): per shape, among all non-diverged candidates of the
pool stages (default `--pool db`: the 25-control-point B-spline and its polish stages, i.e. what
can go into the DB), take the
SMOOTHEST one whose Hellinger distance is at most `--eps` (0.05) above the best Hellinger of that
shape. "Smoothest" = lowest `--smooth_metric` (default `smooth`, the project's accel energy).
The hit rate of a configuration is the fraction of shapes where it has at least one eligible candidate.
"""
import argparse
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, _arch):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.colors as mcolors                                        # noqa: E402
import matplotlib.pyplot as plt                                            # noqa: E402

C_GT, C_GEN, C_DARK, C_GREY = '#1565C0', '#00C853', '#1A1A2E', '#555555'
_ink = plt.colormaps['inferno'](np.linspace(0.0, 1.0, 256))
for _i in range(40):
    _ink[_i] = (1 - _i / 40) * np.array([1, 1, 1, 1]) + (_i / 40) * _ink[40]
WHITE_INFERNO = mcolors.LinearSegmentedColormap.from_list('white_inferno', _ink)

SOLVER_COLOR = {'tsvec': '#8E24AA', 'adam': '#EF6C00', 'lm': '#00838F', 'sun_sinkhorn': '#6D4C41',
                'sun_stein': '#C62828', 'db': C_GT}
METRICS = ['hellinger', 'tv_4', 'ergodic_k10', 'smooth', 'path_len_arc', 'acc_rms', 'jerk_rms',
           'curv_p95', 'low_excess', 'outside']


def style_axes(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2)
    for sp in ax.spines.values():
        sp.set_color('#cccccc')
    ax.tick_params(colors=C_GREY)
    ax.xaxis.label.set_color(C_DARK)
    ax.yaxis.label.set_color(C_DARK)
    ax.title.set_color(C_DARK)


def load(path):
    """metrics.csv, or metrics.csv.gz when only the compressed copy is checked out."""
    import gzip
    opener = gzip.open(path + '.gz', 'rt', newline='') if (not os.path.exists(path) and os.path.exists(path + '.gz'))         else open(path, newline='')
    with opener as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        if k in ('shape', 'config', 'solver', 'stage'):
            cols[k] = np.array([r[k] for r in rows])
        else:
            cols[k] = np.array([float(r[k]) for r in rows])
    cols['cand'] = cols['cand'].astype(int)
    return cols


def apply_boundary_filter(d, max_outside):
    """Mark every non-DB candidate whose curve leaves [0.03, 0.97] on more than `max_outside` of its points
    as unusable (`diverged = 1`), so selection, summaries and figures skip it. max_outside=None: no filter.
    The DB teacher rows stay untouched (they are the reference, not a candidate)."""
    if max_outside is None:
        return d
    bad = (d['outside'] > max_outside) & (d['config'] != 'teacher_db')
    d['diverged'] = np.where(bad, 1.0, d['diverged'])
    return d


def parse_cfg_name(name):
    """-> (solver family key, order, weight or None) from names like 'lm_o3_w0.1'."""
    parts = name.split('_')
    order = next((int(p[1:]) for p in parts if len(p) == 2 and p[0] == 'o' and p[1:].isdigit()), None)
    w = next((float(p[1:]) for p in parts if p.startswith('w') and p[1:].replace('.', '').isdigit()), None)
    return order, w


def select(d, pool_stages, eps, smooth_metric):
    """-> dict shape -> (row index of the chosen candidate, h_best)."""
    out = {}
    ok = (d['diverged'] == 0) & (d['config'] != 'teacher_db') & np.isin(d['stage'], pool_stages)
    for shape in np.unique(d['shape']):
        m = ok & (d['shape'] == shape)
        if not m.any():
            continue
        h_best = d['hellinger'][m].min()
        elig = m & (d['hellinger'] <= h_best + eps)
        idx = np.flatnonzero(elig)
        out[shape] = (idx[np.argmin(d[smooth_metric][idx])], h_best)
    return out


def summary(d, shapes, hbest, eps, smooth_metric):
    rows = []
    keys = sorted({(c, s) for c, s in zip(d['config'], d['stage'])},
                  key=lambda t: (t[0] != 'teacher_db', t))
    for cfg, stage in keys:
        m0 = (d['config'] == cfg) & (d['stage'] == stage) & (d['diverged'] == 0)
        if not m0.any():
            continue
        solver = d['solver'][m0][0]
        rec = {'config': cfg, 'solver': solver, 'stage': stage,
               'n_cand_valid': int(m0.sum()), 'n_shapes': len(np.unique(d['shape'][m0]))}
        for k in METRICS:
            rec[f'{k}_cand_mean'] = float(d[k][m0].mean())
        hits, best_vals = 0, {k: [] for k in METRICS}
        for sh in shapes:
            m = m0 & (d['shape'] == sh)
            if not m.any():
                continue
            idx = np.flatnonzero(m & (d['hellinger'] <= hbest[sh] + eps))
            if idx.size == 0:
                continue
            hits += 1
            j = idx[np.argmin(d[smooth_metric][idx])]
            for k in METRICS:
                best_vals[k].append(d[k][j])
        rec['hit_rate'] = hits / max(len(shapes), 1)
        for k in METRICS:
            rec[f'{k}_best'] = float(np.mean(best_vals[k])) if best_vals[k] else float('nan')
        rows.append(rec)
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------

def plot_pareto(rows, path, smooth_metric):
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor='white')
    style_axes(ax)
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r['config'], {})[r['stage']] = r
    mk = {'wp': 'o', 'bs25': 's', 'bs25_pol300': '^'}
    for cfg, st in by_cfg.items():
        if cfg == 'teacher_db':
            continue
        col = SOLVER_COLOR.get(next(iter(st.values()))['solver'], C_GREY)
        pts = [(st[s][f'hellinger_cand_mean'], st[s][f'{smooth_metric}_cand_mean'], s) for s in ('wp', 'bs25', 'bs25_pol300') if s in st]
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', color=col, alpha=0.25, lw=1)
        for h, s, stg in pts:
            ax.scatter(h, s, c=col, marker=mk[stg], s=26, alpha=0.8, edgecolors='none')
    for stg, r in by_cfg.get('teacher_db', {}).items():
        if stg in mk:
            ax.scatter(r['hellinger_cand_mean'], r[f'{smooth_metric}_cand_mean'], c=C_GT, marker=mk[stg],
                       s=120, edgecolors=C_DARK, zorder=5)
    ax.set_yscale('log')
    ax.set_xlabel('Hellinger distance to target (mean over candidates and shapes, lower = more ergodic)')
    ax.set_ylabel(f'{smooth_metric} (acceleration energy, log scale, lower = smoother)')
    ax.set_title('Ergodicity vs smoothness: waypoints (o) -> B-spline projection (s) -> polish (^)')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=k) for k, c in SOLVER_COLOR.items()]
    handles += [plt.Line2D([], [], marker=m, ls='', color=C_GREY, label=l)
                for m, l in (('o', 'waypoints'), ('s', 'B-spline (25 CP)'), ('^', 'B-spline + 300 polish steps'))]
    ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2, labelcolor=C_DARK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)


def plot_force_sweeps(rows, path, smooth_metric):
    fams = [('tsvec', 'TSVEC-SVGD'), ('adam', 'Adam-SVGD'), ('lm', 'Levenberg-Marquardt'),
            ('sun_sinkhorn', 'Sun (Sinkhorn)')]
    orders = [(1, 'velocity'), (2, 'acceleration'), (3, 'jerk')]
    fig, axes = plt.subplots(len(fams), len(orders), figsize=(12, 10), facecolor='white', squeeze=False)
    for i, (fam, ftitle) in enumerate(fams):
        for j, (order, otitle) in enumerate(orders):
            ax = axes[i][j]
            style_axes(ax)
            pts = []
            for r in rows:
                if r['solver'] != fam or r['stage'] != 'bs25' or '_teacher_' in r['config'] \
                        or '_K16_' in r['config'] or '_cont_' in r['config'] or '_bs25_' in r['config']:
                    continue
                o, w = parse_cfg_name(r['config'])
                if o == order and w is not None:
                    pts.append((w, r['hellinger_cand_mean'], r[f'{smooth_metric}_cand_mean']))
            if not pts:
                ax.set_axis_off()
                continue
            pts.sort()
            w = np.array([p[0] for p in pts])
            xs = np.where(w > 0, w, w[w > 0].min() / 3 if (w > 0).any() else 1e-3)
            ax.set_xscale('log')
            ax.plot(xs, [p[1] for p in pts], 'o-', color=SOLVER_COLOR[fam], label='Hellinger')
            ax.set_ylabel('Hellinger')
            ax2 = ax.twinx()
            ax2.plot(xs, [p[2] for p in pts], 's--', color=C_GREY, label=smooth_metric)
            ax2.set_yscale('log')
            ax2.tick_params(colors=C_GREY)
            ax.set_title(f'{ftitle}, {otitle} force', fontsize=10)
            ax.set_xlabel('weight (w = 0 drawn at the left edge)')
    fig.suptitle('Force sweeps after the B-spline projection (25 control points)', color=C_DARK)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor='white')
    plt.close(fig)


def plot_projection_effect(rows, path, smooth_metric):
    by = {}
    for r in rows:
        by.setdefault(r['config'], {})[r['stage']] = r
    names, dh, ds = [], [], []
    for cfg, st in by.items():
        if 'wp' in st and 'bs25' in st and cfg != 'teacher_db':
            names.append(cfg)
            dh.append(st['bs25']['hellinger_cand_mean'] - st['wp']['hellinger_cand_mean'])
            ds.append(st['bs25'][f'{smooth_metric}_cand_mean'] / max(st['wp'][f'{smooth_metric}_cand_mean'], 1e-12))
    if not names:
        return
    order = np.argsort(ds)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.22 * len(names))), facecolor='white', sharey=True)
    y = np.arange(len(names))
    axes[0].barh(y, np.array(dh)[order], color=C_GREY)
    axes[0].set_xlabel('change of Hellinger by the projection (positive = less ergodic)')
    axes[1].barh(y, np.array(ds)[order], color=C_GEN)
    axes[1].axvline(1.0, color=C_DARK, lw=0.8)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(f'{smooth_metric} after / before the projection (< 1 = projection smooths)')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(np.array(names)[order], fontsize=7)
    for ax in axes:
        style_axes(ax)
    fig.suptitle('Effect of the B-spline projection (25 control points, degree 5)', color=C_DARK)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor='white')
    plt.close(fig)


def _render_stage(arr, stage):
    import pipeline as pl
    if stage == 'wp':
        return np.asarray(arr, dtype=np.float64)
    deg = int(stage.split('d')[1].split('_')[0]) if 'd' in stage.split('_')[0] else pl.DEGREE
    return pl.render(np.asarray(arr)[None], pl.N_POINTS, deg)[0]


def plot_gallery(d, sel, run_dir, path, smooth_metric):
    import pipeline as pl
    shapes = list(sel.keys())
    sd_by = {s.name: s for s in pl.load_shapes(shapes)}
    n = len(shapes)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.3 * nrow), facecolor='white', squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, shape in enumerate(shapes):
        ax = axes[k // ncol][k % ncol]
        ax.set_axis_on()
        j = sel[shape][0]
        sd = sd_by[shape]
        ax.imshow(sd.density, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO, vmin=0, vmax=1,
                  alpha=0.55, aspect='equal', zorder=0)
        ref = np.load(os.path.join(run_dir, 'finals', f'{shape}__teacher_db.npz'))
        t = _render_stage(ref['bs25'][0], 'bs25')
        ax.plot(t[:, 0], t[:, 1], color=C_GT, lw=2.5, alpha=0.9, zorder=2)
        f = np.load(os.path.join(run_dir, 'finals', f'{shape}__{d["config"][j]}.npz'))
        g = _render_stage(f[d['stage'][j]][d['cand'][j]], d['stage'][j])
        ax.plot(g[:, 0], g[:, 1], color=C_GEN, lw=2.2, alpha=0.95, zorder=3)
        ax.plot(*sd.x0, 'o', color=C_DARK, ms=4, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#cccccc')
        ax.set_title(f'{shape}\n{d["config"][j]} / {d["stage"][j]}', fontsize=7, color=C_DARK)
    fig.suptitle('Selected trajectory (green) vs. current DB teacher (blue), B-spline, start point marked',
                 color=C_DARK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True, help='results/<out_tag>')
    ap.add_argument('--eps', type=float, default=0.05, help='Hellinger tolerance above the per-shape best')
    ap.add_argument('--smooth_metric', default='smooth', help='column ranked for "smoothest" (smooth, acc_rms, jerk_rms)')
    ap.add_argument('--pool', default='db',
                    help="stages the teacher may come from: 'db' = the dataset's B-spline (bs25 and its polish "
                         "stages, default), 'bs' = every projection variant, 'wp' = raw waypoints, or a "
                         "comma-separated list of stage names")
    ap.add_argument('--max_outside', type=float, default=0.0,
                    help='exclude candidates with more than this fraction of points outside [0.03, 0.97] '
                         '(default 0 = none allowed; negative = no filter)')
    ap.add_argument('--no_gallery', action='store_true')
    args = ap.parse_args()

    run_dir = os.path.join(_here, 'results', args.out_tag)
    d = load(os.path.join(run_dir, 'metrics.csv'))
    d = apply_boundary_filter(d, None if args.max_outside < 0 else args.max_outside)
    stages = np.unique(d['stage'])
    if args.pool == 'db':
        pool = [str(s) for s in stages if s == 'bs25' or s.startswith('bs25_pol')]
    elif args.pool == 'bs':
        pool = [str(s) for s in stages if s.startswith('bs')]
    else:
        pool = [s for s in args.pool.split(',') if s in stages]
    if not pool:
        raise SystemExit(f'no stage matches --pool {args.pool!r}; stages present: {list(stages)}')
    shapes = sorted(np.unique(d['shape']))
    os.makedirs(os.path.join(run_dir, 'plots'), exist_ok=True)

    sel = select(d, pool, args.eps, args.smooth_metric)
    hbest = {s: v[1] for s, v in sel.items()}
    rows = summary(d, shapes, hbest, args.eps, args.smooth_metric)
    write_csv(os.path.join(run_dir, 'summary_config_stage.csv'), rows)

    # selection vs. DB teacher
    sel_rows, ratios = [], {'smooth': [], 'acc_rms': [], 'jerk_rms': [], 'dh': []}
    for shape, (j, hb) in sel.items():
        t = np.flatnonzero((d['shape'] == shape) & (d['config'] == 'teacher_db') & (d['stage'] == 'bs25'))
        rec = {'shape': shape, 'config': d['config'][j], 'stage': d['stage'][j], 'cand': int(d['cand'][j]),
               'h_best_pool': float(hb)}
        for k in ('hellinger', 'tv_4', 'ergodic_k10', 'smooth', 'path_len_arc', 'acc_rms', 'jerk_rms'):
            rec[k] = float(d[k][j])
            if t.size:
                rec[f'teacher_{k}'] = float(d[k][t[0]])
        if t.size:
            ratios['smooth'].append(rec['smooth'] / rec['teacher_smooth'])
            ratios['acc_rms'].append(rec['acc_rms'] / rec['teacher_acc_rms'])
            ratios['jerk_rms'].append(rec['jerk_rms'] / rec['teacher_jerk_rms'])
            ratios['dh'].append(rec['hellinger'] - rec['teacher_hellinger'])
        sel_rows.append(rec)
    write_csv(os.path.join(run_dir, 'selection.csv'), sel_rows)

    # report
    cand = [r for r in rows if r['config'] != 'teacher_db' and r['stage'] in pool and r['hit_rate'] > 0]
    cand.sort(key=lambda r: (-r['hit_rate'], r[f'{args.smooth_metric}_best']))
    lines = [f'# Teacher study `{args.out_tag}`', '',
             f'{len(shapes)} shapes; selection rule: smoothest candidate with Hellinger <= best + {args.eps} '
             f'(per shape, pool stages {pool}); smoothness = `{args.smooth_metric}`; candidates with more than '
             f'{args.max_outside} of their points outside [0.03, 0.97] excluded.', '']
    if ratios['smooth']:
        lines += ['## Selected teacher vs. current DB teacher (B-spline, 25 CP)', '',
                  f'* Hellinger change (selected - DB): mean {np.mean(ratios["dh"]):+.3f}',
                  f'* smoothness energy ratio: median {np.median(ratios["smooth"]):.2f} (mean {np.mean(ratios["smooth"]):.2f})',
                  f'* acceleration RMS ratio: median {np.median(ratios["acc_rms"]):.2f}',
                  f'* jerk RMS ratio: median {np.median(ratios["jerk_rms"]):.2f}', '']
    t_rows = [r for r in rows if r['config'] == 'teacher_db' and r['stage'] == 'bs25']
    if t_rows:
        r = t_rows[0]
        lines += [f'DB teacher (bs25): Hellinger {r["hellinger_cand_mean"]:.3f}, smooth {r["smooth_cand_mean"]:.3f}, '
                  f'acc RMS {r["acc_rms_cand_mean"]:.1f}, jerk RMS {r["jerk_rms_cand_mean"]:.0f}', '']
    lines += ['## Ranking (hit rate, then smoothness of the best-of-N pick)', '',
              '| config | stage | hit rate | Hellinger | smooth | acc RMS | jerk RMS | path |', '|---|---|---|---|---|---|---|---|']
    for r in cand[:25]:
        lines.append(f'| {r["config"]} | {r["stage"]} | {r["hit_rate"]:.2f} | {r["hellinger_best"]:.3f} | '
                     f'{r["smooth_best"]:.3f} | {r["acc_rms_best"]:.1f} | {r["jerk_rms_best"]:.0f} | {r["path_len_arc_best"]:.2f} |')
    wins = {}
    for rec in sel_rows:
        wins[(rec['config'], rec['stage'])] = wins.get((rec['config'], rec['stage']), 0) + 1
    lines += ['', '## Which configuration produced the chosen teacher', '']
    for (c, s), n in sorted(wins.items(), key=lambda kv: -kv[1]):
        lines.append(f'* {c} / {s}: {n} shapes')
    with open(os.path.join(run_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    plot_pareto(rows, os.path.join(run_dir, 'plots', 'pareto.png'), args.smooth_metric)
    plot_force_sweeps(rows, os.path.join(run_dir, 'plots', 'force_sweeps.png'), args.smooth_metric)
    plot_projection_effect(rows, os.path.join(run_dir, 'plots', 'projection_effect.png'), args.smooth_metric)
    if not args.no_gallery:
        plot_gallery(d, sel, run_dir, os.path.join(run_dir, 'plots', 'selection_gallery.png'), args.smooth_metric)
    print(f'[analyze] wrote tables and plots into {run_dir}')


if __name__ == '__main__':
    main()
