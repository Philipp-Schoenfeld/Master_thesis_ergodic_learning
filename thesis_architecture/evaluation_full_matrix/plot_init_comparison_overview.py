r"""
plot_init_comparison_overview.py
===================================
Reproduces the VISUAL STYLE of Philipp's three reference plots (`overview_
E_truth_continuous_*_svgd_conv_plateau.png`, `overview_E_truth.png`, and the
attached smoothness-comparison dashboard mockup) on top of `run_init_
comparison.py`'s static 7-method x 2-solver comparison -- NOT the replanning
/GP-belief machinery those plots originally came from (out of scope for this
run, confirmed with Philipp 2026-10-19; see the plan).

Two figures:

  overview_dashboard.png   4-panel dashboard matching the reference mockup:
                            (top-left) ergodic error vs. smoothness scatter,
                            dot=median/cross=IQR over the 25 holdout shapes,
                            one marker per (method, solver); (top-right)
                            3-column per-solver-family breakdown (ergodic
                            error / smoothness / path length), one dot+bar
                            per method, marker size = n90, bold = best in
                            that column; (bottom) ergodic-error-vs-SVGD-
                            -iteration convergence curves with n90 markers
                            ("svgd_conv_plateau" style, iteration axis
                            instead of distance-driven since there is no
                            replanning here).
  overview_E_truth.png     simpler summary: final ergodic error per
                            (method, solver), mean +- std over shapes,
                            sorted -- the plain-bar analogue of the
                            replanning harness's own `overview_E_truth.png`.

Reuses `plot_init_comparison.py`'s metric loaders/caches (no reimplementation
of the ergodic-error-from-stored-states / n90 logic).

Usage
-----
    python plot_init_comparison_overview.py --out_tag init_comparison_YYYYMMDD
"""
import argparse
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402

import init_comparison_db as idb                                    # noqa: E402
from plot_init_comparison import (                                  # noqa: E402
    METHODS, SOLVERS, METHOD_COLORS, METHOD_LABELS, SOLVER_LABELS, SOLVER_LINESTYLES,
    ErgodicSeriesCache, n90_from_series, _style,
)

SOLVER_MARKERS = {'tsvec_svgd': 'o', 'adam_svgd': 's'}


def _per_shape_final(ergo, conn, shapes, variant, series_kind):
    """-> (n_shapes,) float, per-shape mean-over-candidates final value of
    `series_kind` in {'ergodic', 'coverage', 'smooth', 'path_len', 'outside'}."""
    out = []
    for shape in shapes:
        if series_kind == 'ergodic':
            _s, emat = ergo.load(shape, variant)
            if emat is None:
                continue
            last = np.where(~np.all(np.isnan(emat), axis=0))[0]
            if last.size == 0:
                continue
            out.append(float(np.nanmean(emat[:, last[-1]])))
        elif series_kind == 'coverage':
            cov = ergo.coverage_final(shape, variant)
            if cov.size == 0:
                continue
            out.append(float(cov.mean()))
        else:
            col = {'smooth': 'smooth_series', 'path_len': 'path_len_series',
                  'outside': 'outside_series'}[series_kind]
            rows = list(idb.iter_runs(conn, shape=shape, variant=variant))
            if not rows:
                continue
            vals = [r[col][np.isfinite(r[col])][-1] for r in rows if np.isfinite(r[col]).any()]
            if vals:
                out.append(float(np.mean(vals)))
    return np.asarray(out, dtype=np.float64)


def _n90_per_shape(ergo, shapes, variant):
    out = []
    for shape in shapes:
        _s, emat = ergo.load(shape, variant)
        if emat is None:
            continue
        out.append(float(np.mean(n90_from_series(emat))))
    return np.asarray(out, dtype=np.float64)


def _median_iqr(x):
    if x.size == 0:
        return np.nan, np.nan, np.nan
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return q50, q25, q75


def plot_scatter(ax, ergo, conn, shapes, variants):
    for method in METHODS:
        for solver in SOLVERS:
            variant = f"{method}__{solver}"
            if variant not in variants:
                continue
            ex = _per_shape_final(ergo, conn, shapes, variant, 'ergodic')
            sm = _per_shape_final(ergo, conn, shapes, variant, 'smooth')
            if ex.size == 0 or sm.size == 0:
                continue
            ex_med, ex_lo, ex_hi = _median_iqr(ex)
            sm_med, sm_lo, sm_hi = _median_iqr(sm)
            color = METHOD_COLORS[method]
            ax.plot([ex_lo, ex_hi], [sm_med, sm_med], color=color, lw=1.0, alpha=0.5)
            ax.plot([ex_med, ex_med], [sm_lo, sm_hi], color=color, lw=1.0, alpha=0.5)
            ax.scatter([ex_med], [sm_med], color=color, marker=SOLVER_MARKERS[solver],
                      s=70, edgecolors='#1A1A2E', linewidths=0.6, zorder=3)
            bold = solver == 'tsvec_svgd'
            ax.annotate(f"{METHOD_LABELS[method]}\n{SOLVER_LABELS[solver]}", (ex_med, sm_med),
                       fontsize=5.5, color='#1A1A2E', fontweight='bold' if bold else 'normal',
                       xytext=(4, 4), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Ergodic error (lower = better coverage)')
    ax.set_ylabel('Smoothness cost (lower = smoother)')
    ax.set_title('Coverage vs. smoothness (median over shapes; cross = IQR)', fontsize=9,
                color='#1A1A2E')
    _style(ax)


def plot_breakdown(fig, gs_row, ergo, conn, shapes, variants):
    panels = (('ergodic', 'Ergodic error'), ('smooth', 'Smoothness cost'),
             ('path_len', 'Path length'))
    axes = [fig.add_subplot(gs_row[0, i]) for i in range(3)]
    for ax, (kind, title) in zip(axes, panels):
        y = 0
        yticks, yticklabels = [], []
        for solver in SOLVERS:
            values = {}
            for method in METHODS:
                variant = f"{method}__{solver}"
                if variant not in variants:
                    continue
                vals = _per_shape_final(ergo, conn, shapes, variant, kind)
                n90 = _n90_per_shape(ergo, shapes, variant)
                if vals.size == 0:
                    continue
                values[method] = (float(np.median(vals)), float(np.percentile(vals, 25)),
                                  float(np.percentile(vals, 75)), float(np.mean(n90)) if n90.size else 0.0)
            if not values:
                continue
            best_method = min(values, key=lambda m: values[m][0])
            for method, (med, lo, hi, n90_mean) in values.items():
                color = METHOD_COLORS[method]
                ax.plot([lo, hi], [y, y], color=color, lw=1.2, alpha=0.6)
                size = 20 + 6 * np.sqrt(max(n90_mean, 1.0))
                ax.scatter([med], [y], color=color, s=size, zorder=3,
                          edgecolors='#1A1A2E', linewidths=0.5)
                weight = 'bold' if method == best_method else 'normal'
                ax.annotate(f"{med:.2g}", (med, y), fontsize=6, color='#1A1A2E',
                           fontweight=weight, xytext=(5, -2), textcoords='offset points')
                yticks.append(y)
                yticklabels.append(f"{METHOD_LABELS[method]}" + (' **' if method == best_method else ''))
                y -= 1
            y -= 0.6
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=6)
        if kind != 'path_len':
            # ergodic error / smoothness span orders of magnitude across
            # methods; path length does not (narrow range, ~1-5 domain
            # widths) -- log-scale minor ticks on that narrow a range render
            # as clutter that can spill past the subplot's own bounding box.
            ax.set_xscale('log')
        ax.margins(x=0.25)
        ax.set_title(title, fontsize=9, color='#1A1A2E')
        _style(ax)
    axes[-1].text(1.02, 0.5, 'marker size = mean n90', transform=axes[-1].transAxes,
                  fontsize=6, color='#555', rotation=90, va='center')


#: Fraction of the iteration axis where the "zoom" panel starts (Philipp's
#: request, 2026-10-19: the full-range log panel compresses the converged/
#: plateau region -- where the 14 variants actually differ -- into a sliver
#: near the bottom, since early iterations span ~2 orders of magnitude more).
ZOOM_START_FRAC = 0.3


def _convergence_curves(ergo, shapes, variants):
    """-> dict variant -> (steps (n,), mean_curve (n,)), the mean-over-shapes
    ergodic-error-vs-iteration series each panel below plots."""
    curves = {}
    for method in METHODS:
        for solver in SOLVERS:
            variant = f"{method}__{solver}"
            if variant not in variants:
                continue
            per_shape_means, steps_ref = [], None
            for shape in shapes:
                steps, mat = ergo.load(shape, variant)
                if steps is None:
                    continue
                valid = ~np.all(np.isnan(mat), axis=0)
                per_shape_means.append(np.nanmean(mat[:, valid], axis=0))
                steps_ref = steps[valid]
            if not per_shape_means:
                continue
            n = min(len(m) for m in per_shape_means)
            mean_curve = np.stack([m[:n] for m in per_shape_means]).mean(axis=0)
            curves[variant] = (steps_ref[:n], mean_curve)
    return curves


def _plot_convergence_panel(ax, curves, ergo, shapes, x_lo=None, y_range=None):
    """One convergence panel (shared by the full-range and zoomed views) --
    plots every curve's slice within [x_lo, end], with n90 markers that fall
    in that window."""
    for method in METHODS:
        for solver in SOLVERS:
            variant = f"{method}__{solver}"
            if variant not in curves:
                continue
            s, mean_curve = curves[variant]
            mask = s >= x_lo if x_lo is not None else np.ones_like(s, dtype=bool)
            if not mask.any():
                continue
            color = METHOD_COLORS[method]
            ax.plot(s[mask], mean_curve[mask], color=color, lw=1.6, ls=SOLVER_LINESTYLES[solver],
                   label=f"{METHOD_LABELS[method]} + {SOLVER_LABELS[solver]}")
            n90_shapes = _n90_per_shape(ergo, shapes, variant)
            if n90_shapes.size:
                n90_i = min(int(round(float(n90_shapes.mean()))), len(s) - 1)
                if s[n90_i] >= (x_lo if x_lo is not None else -np.inf):
                    ax.scatter([s[n90_i]], [mean_curve[n90_i]], color=color, s=40, zorder=4,
                              marker='o', edgecolors='#1A1A2E', linewidths=0.6)
    ax.set_xlabel('SVGD iteration')
    ax.set_yscale('log')
    if y_range is not None:
        ax.set_ylim(*y_range)
    _style(ax)


def plot_convergence(ax_full, ax_zoom, ergo, shapes, variants):
    curves = _convergence_curves(ergo, shapes, variants)
    if not curves:
        return
    max_iter = max(s[-1] for s, _c in curves.values())
    zoom_x0 = ZOOM_START_FRAC * max_iter

    _plot_convergence_panel(ax_full, curves, ergo, shapes)
    ax_full.axvspan(zoom_x0, max_iter, color='#1A1A2E', alpha=0.05, zorder=0)
    ax_full.set_ylabel('Ergodic error (mean over shapes)')
    ax_full.set_title('Convergence, full range (shaded = zoomed panel)', fontsize=9,
                      color='#1A1A2E')
    ax_full.legend(fontsize=5, loc='upper right', ncol=2, frameon=False)

    # Zoom y-range: 2nd-98th percentile of every curve's values within the
    # zoomed x-window, not the full curve -- a couple of slow-to-converge
    # outlier variants should not re-compress the panel we just zoomed to
    # fix that exact problem.
    zoom_vals = []
    for s, mean_curve in curves.values():
        mask = s >= zoom_x0
        if mask.any():
            zoom_vals.append(mean_curve[mask])
    zoom_vals = np.concatenate(zoom_vals) if zoom_vals else np.array([1.0])
    lo, hi = np.percentile(zoom_vals, [2, 98])
    y_range = (lo * 0.85, hi * 1.15)

    _plot_convergence_panel(ax_zoom, curves, ergo, shapes, x_lo=zoom_x0, y_range=y_range)
    ax_zoom.set_title(f'Convergence, zoomed to iteration >= {zoom_x0:.0f} '
                      '(marker = n90: 90% of final improvement reached)', fontsize=9,
                      color='#1A1A2E')


def plot_dashboard(conn, shapes, variants, out_path, ergo):
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], width_ratios=[1.0, 1.4],
                          hspace=0.35, wspace=0.3)
    ax_scatter = fig.add_subplot(gs[0, 0])
    plot_scatter(ax_scatter, ergo, conn, shapes, variants)

    gs_breakdown = gs[0, 1].subgridspec(1, 3, wspace=0.6)
    plot_breakdown(fig, gs_breakdown, ergo, conn, shapes, variants)

    gs_conv = gs[1, :].subgridspec(1, 2, wspace=0.15)
    ax_conv_full = fig.add_subplot(gs_conv[0, 0])
    ax_conv_zoom = fig.add_subplot(gs_conv[0, 1])
    plot_convergence(ax_conv_full, ax_conv_zoom, ergo, shapes, variants)

    fig.suptitle(f'Init-method x solver comparison: {len(METHODS)} methods, '
                f'{len(SOLVERS)} solvers, {len(shapes)} holdout shapes, final iteration.',
                fontsize=12, color='#1A1A2E', y=0.995)
    fig.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_overview_e_truth(conn, shapes, variants, out_path, ergo):
    """Simpler summary bar chart, the plain analogue of the replanning
    harness's `overview_E_truth.png`: final ergodic error per (method,
    solver), mean +- std over shapes, sorted best-to-worst."""
    entries = []
    for method in METHODS:
        for solver in SOLVERS:
            variant = f"{method}__{solver}"
            if variant not in variants:
                continue
            vals = _per_shape_final(ergo, conn, shapes, variant, 'ergodic')
            if vals.size == 0:
                continue
            entries.append((method, solver, float(vals.mean()), float(vals.std())))
    entries.sort(key=lambda e: e[2])

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(entries) + 1.5), facecolor='white')
    ys = np.arange(len(entries))[::-1]
    for y, (method, solver, mean, std) in zip(ys, entries):
        color = METHOD_COLORS[method]
        hatch = None if solver == 'tsvec_svgd' else '///'
        ax.barh(y, mean, xerr=std, color=color, alpha=0.85, hatch=hatch,
               edgecolor='#1A1A2E', linewidth=0.5, error_kw=dict(ecolor='#1A1A2E', lw=1))
        ax.text(mean + std + 0.01 * max(mean, 1e-6), y,
               f"{METHOD_LABELS[method]} + {SOLVER_LABELS[solver]}", va='center', fontsize=7,
               color='#1A1A2E')
    ax.set_yticks([])
    ax.set_xlabel('Ergodic error vs. true density (mean +- std over holdout shapes)')
    ax.set_title(f'Final ergodic error -- {len(shapes)} holdout shapes', fontsize=10,
                color='#1A1A2E')
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)


METRIC_DEFS = (
    ('ergodic', 'Ergodic error'), ('coverage', 'Coverage'), ('smooth', 'Smoothness'),
    ('path_len', 'Path length'), ('outside', 'Outside fraction'), ('n90', 'n90 (convergence)'),
)


def compute_rankings(ergo, conn, shapes, variants):
    """-> list of dicts (sorted best-to-worst by avg_rank): method, solver,
    metrics {key: median-over-shapes value}, ranks {key: int, 1=best},
    avg_rank (mean rank across all 6 metrics -- ALL are "lower is better"
    here, so a plain rank average needs no sign-flipping per metric)."""
    entries = []
    for method in METHODS:
        for solver in SOLVERS:
            variant = f"{method}__{solver}"
            if variant not in variants:
                continue
            metrics = {}
            for key, _label in METRIC_DEFS:
                vals = (_n90_per_shape(ergo, shapes, variant) if key == 'n90'
                       else _per_shape_final(ergo, conn, shapes, variant, key))
                metrics[key] = float(np.median(vals)) if vals.size else np.nan
            entries.append(dict(method=method, solver=solver, metrics=metrics))
    for key, _label in METRIC_DEFS:
        vals = np.array([e['metrics'][key] for e in entries])
        order = np.argsort(vals)                       # NaN sorts last = worst rank
        ranks = np.empty(len(entries))
        ranks[order] = np.arange(1, len(entries) + 1)
        for e, r in zip(entries, ranks):
            e.setdefault('ranks', {})[key] = r
    for e in entries:
        e['avg_rank'] = float(np.mean(list(e['ranks'].values())))
    entries.sort(key=lambda e: e['avg_rank'])
    return entries


def plot_ranking_heatmap(ax, entries):
    """Per-metric rank, 1 (best) -> n (worst): a 'compare magnitude on a
    grid' job, so per the dataviz skill this gets SEQUENTIAL one-hue color
    (darker = better), not the methods' categorical identity colors -- those
    belong to the bar chart below, where series identity IS the job."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#0d366b'])
    n, m = len(entries), len(METRIC_DEFS)
    rank_matrix = np.array([[e['ranks'][k] for k, _l in METRIC_DEFS] + [e['avg_rank']]
                            for e in entries])
    norm = np.zeros_like(rank_matrix)
    for j in range(m + 1):
        col = rank_matrix[:, j]
        lo, hi = col.min(), col.max()
        norm[:, j] = 1 - (col - lo) / max(hi - lo, 1e-9)
    ax.imshow(norm, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    col_labels = [lbl for _k, lbl in METRIC_DEFS] + ['Avg rank']
    ax.set_xticks(range(m + 1))
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=7.5)
    row_labels = [f"{METHOD_LABELS[e['method']]} + {SOLVER_LABELS[e['solver']]}" for e in entries]
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    for i in range(n):
        for j in range(m + 1):
            txt_color = '#1A1A2E' if norm[i, j] < 0.55 else 'white'
            weight = 'bold' if j == m else 'normal'
            fmt = f"{rank_matrix[i, j]:.0f}" if j < m else f"{rank_matrix[i, j]:.1f}"
            ax.text(j, i, fmt, ha='center', va='center', fontsize=7, color=txt_color,
                   fontweight=weight)
    # 2px surface ring around the best row (dataviz skill mark spec for an
    # emphasized/overlapping mark) -- the #1 overall row.
    ax.add_patch(plt.Rectangle((-0.5, -0.5), m + 1, 1, fill=False, edgecolor='#1A1A2E', lw=2))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(m + 2) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    ax.set_title('Rank per metric across all 14 (method, solver) combinations '
                 '(1 = best; darker = better)', fontsize=9, color='#1A1A2E')


def plot_ranking_bar(ax, entries):
    """The headline chart: composite average rank, sorted -- 'which method
    is best overall' answered in one glance. Categorical color (method
    identity IS the job here), hatch for solver, per the dataviz skill's
    'tell distinct series apart' form."""
    ys = np.arange(len(entries))[::-1]
    for y, e in zip(ys, entries):
        color = METHOD_COLORS[e['method']]
        hatch = None if e['solver'] == 'tsvec_svgd' else '///'
        ax.barh(y, e['avg_rank'], color=color, alpha=0.9, hatch=hatch,
               edgecolor='#1A1A2E', linewidth=0.8 if y == ys[0] else 0.5)
        weight = 'bold' if y == ys[0] else 'normal'
        ax.text(e['avg_rank'] + 0.15, y,
               f"{METHOD_LABELS[e['method']]} + {SOLVER_LABELS[e['solver']]}  "
               f"(avg rank {e['avg_rank']:.1f})", va='center', fontsize=7.5, color='#1A1A2E',
               fontweight=weight)
    ax.set_yticks([])
    ax.set_xlabel('Average rank across ergodic error, coverage, smoothness, path length, '
                 'outside-fraction, n90 (lower = better)')
    ax.set_title('Best overall method', fontsize=11, color='#1A1A2E', fontweight='bold')
    _style(ax)


def plot_best_method_ranking(conn, shapes, variants, out_path, ergo):
    entries = compute_rankings(ergo, conn, shapes, variants)
    if not entries:
        return
    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=(10, 0.42 * len(entries) + 1.2 + 0.32 * len(entries) + 1.5),
        facecolor='white', gridspec_kw=dict(height_ratios=[0.42 * len(entries) + 1.2,
                                                            0.32 * len(entries) + 1.5]))
    plot_ranking_bar(ax_bar, entries)
    plot_ranking_heatmap(ax_heat, entries)
    fig.suptitle(f'Which init method + solver is best? -- {len(shapes)} holdout shapes, '
                f'final iteration, ranked by average rank across 6 metrics',
                fontsize=12, color='#1A1A2E')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.join(_here, 'results', args.out_tag)
    db_path = os.path.join(out_dir, 'init_comparison.db')
    conn = idb.open_db(db_path)
    cfg = idb.get_meta(conn, 'config', {})
    shapes = cfg.get('shapes') or sorted({r[0] for r in conn.execute("SELECT DISTINCT shape FROM runs")})
    variants = sorted({r[0] for r in conn.execute("SELECT DISTINCT variant FROM runs")})
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    ergo = ErgodicSeriesCache(conn, cache_path=os.path.join(out_dir, 'metric_cache.pkl'))

    plot_dashboard(conn, shapes, variants, os.path.join(plot_dir, 'overview_dashboard.png'), ergo)
    print("[plot_overview] overview_dashboard.png written")

    plot_overview_e_truth(conn, shapes, variants, os.path.join(plot_dir, 'overview_E_truth.png'), ergo)
    print("[plot_overview] overview_E_truth.png written")

    plot_best_method_ranking(conn, shapes, variants, os.path.join(plot_dir, 'best_method_ranking.png'), ergo)
    print("[plot_overview] best_method_ranking.png written")

    ergo.save()


if __name__ == '__main__':
    main()
