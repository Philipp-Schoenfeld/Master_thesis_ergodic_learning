r"""
plot_mission_eval.py
=====================
Figures and tables for `run_mission_eval.py`, read straight from its shard DBs
(works on partial runs -- whatever is stored is plotted).

x axis of every curve = number of executed length units (one unit = the
workspace diagonal). Curves are the mean over the holdout shapes, bands are
+-1 std ACROSS shapes. A mission stops as soon as 99 % of the ground-truth mass
is swept; for the cumulative / state metrics a finished mission keeps its last
value (so the mean is always over the same set of shapes), per-execution
metrics (information gain) are averaged over the missions still running.

Outputs under `<out_root>/<out_tag>/plots/` and `.../tables/`:

  overview_E_truth.png          ergodic error of the whole driven path vs. the GROUND TRUTH
  overview_E_target.png         ... vs. the target density (known + unknown part, after the update)
  overview_J_truth.png          J = E + 0.02 * path length, ground truth
  overview_J_target.png         J, target density
  overview_swept_mass.png       fraction of the true mass within the coverage radius of the path
  overview_info_gain.png        information gain of every single executed unit
  overview_info_gain_cum_frac.png   fraction of the initial uncertainty resolved so far
  overview_belief_rmse.png      RMSE of the belief mean vs. the ground truth
  executions_to_99_cdf.png      fraction of shapes at >= 99 % after n executions
  executions_to_99_box.png      distribution of the number of executions until 99 %
  svgd_convergence_round<k>.png E vs. SVGD iteration inside planning round k (mean over shapes x candidates)
  per_shape/<cond>_<strategy>_<metric>.png   one panel per holdout shape
  paths/<cond>_<strategy>_<method>.png       (--paths) driven paths over the ground truth
  tables/missions.csv           one row per mission
  tables/summary.csv            per (knowledge, strategy, method)

Usage:
    python plot_mission_eval.py --out_tag mission_eval_YYYYMMDD
    python plot_mission_eval.py --out_tag mission_eval_YYYYMMDD --paths
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_arch, 'exploration'),
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib                                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402

import mission_db as mdb                                             # noqa: E402

METHOD_STYLE = {                       # same colours as plot_svgd_convergence.py
    'cfm': dict(color='#00C853', label='CFM warm start', lw=2.0),
    'selfsup': dict(color='#1565C0', label='Self-supervised (single-pass)', lw=1.8),
    'random_walk': dict(color='#E65100', label='Random walk', lw=1.6),
    'linear': dict(color='#6A1B9A', label='Linear (angled lines)', lw=1.6),
}
STRATEGY_TITLE = {'lse': 'LSE (level set)', 'ucb': 'UCB', 'eid': 'EID'}
COND_TITLE = {'ground_truth': 'Ground truth known', 'half_known': 'Half known',
              'ten_samples': 'Ten samples known', 'none_known': 'None known'}
COND_ORDER = ['ground_truth', 'half_known', 'ten_samples', 'none_known']
STRAT_ORDER = ['lse', 'ucb', 'eid']
INK, MUTED, SPINE = '#1A1A2E', '#555555', '#cccccc'

#: metric -> (axis label, log y?, carry the last value after the mission ended?)
METRICS = {
    'E_truth': ('Ergodic error E of the driven path vs. ground truth', True, True),
    'E_target': ('Ergodic error E of the driven path vs. target density', True, True),
    'J_truth': ('J = E + 0.02 * path length (ground truth)', True, True),
    'J_target': ('J = E + 0.02 * path length (target density)', True, True),
    'swept_mass': ('Ground-truth mass swept by the path', False, True),
    'info_gain': ('Information gain per executed unit (removed sum of sigma)', False, False),
    'info_gain_cum_frac': ('Fraction of the initial uncertainty resolved', False, True),
    'belief_rmse': ('Belief RMSE vs. ground truth', False, True),
}
SCALAR_COLS = ('shape', 'round', 'n_exec', 'swept_mass', 'reached99', 'E_truth', 'E_target',
               'J_truth', 'J_target', 'info_gain', 'info_gain_cum', 'unc_before',
               'belief_rmse', 'path_len', 'cov_norm')


def style_axes(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2)
    for sp in ax.spines.values():
        sp.set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=8)


# ── loading ──────────────────────────────────────────────────────────────────

def load_all(root):
    """-> {(cond, strat, method): {shape: {metric: array over rounds}}}, config."""
    cfg = {}
    cfg_path = os.path.join(root, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    data = {}
    for path in mdb.list_shards(root):
        cond, strat, method = os.path.basename(path)[:-3].split('__')
        conn = mdb.open_db(path)
        per_shape = {}
        for d in mdb.iter_rounds(conn, columns=SCALAR_COLS):
            per_shape.setdefault(d['shape'], []).append(d)
        conn.close()
        out = {}
        for shape, rows in per_shape.items():
            rows.sort(key=lambda r: r['round'])
            arr = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in SCALAR_COLS
                   if k != 'shape'}
            unc0 = max(float(rows[0]['unc_before']), 1e-12)
            arr['info_gain_cum_frac'] = arr['info_gain_cum'] / unc0
            out[shape] = arr
        if out:
            data[(cond, strat, method)] = out
    return data, cfg


def mission_matrix(per_shape, metric, n_max, carry):
    """-> (n_shapes, n_max) array; NaN where a mission has no value (or, with
    `carry`, the last value held after it ended)."""
    rows = []
    for shape in sorted(per_shape):
        v = per_shape[shape][metric]
        full = np.full(n_max, np.nan)
        full[:len(v)] = v
        if carry and len(v) < n_max:
            full[len(v):] = v[-1]
        rows.append(full)
    return np.stack(rows)


def executions_to_99(arr):
    """Number of executed units when the mission reached the threshold, else NaN."""
    hit = np.nonzero(arr['reached99'] > 0.5)[0]
    return float(arr['n_exec'][hit[0]]) if len(hit) else float('nan')


def executions_to_threshold(arr, threshold):
    """Executed units until the swept ground-truth mass first reaches `threshold`
    (0.99 -> same as `executions_to_99`), else NaN."""
    if threshold >= 0.99:
        return executions_to_99(arr)
    hit = np.nonzero(arr['swept_mass'] >= threshold)[0]
    return float(arr['n_exec'][hit[0]]) if len(hit) else float('nan')


# ── figures ──────────────────────────────────────────────────────────────────

def grid_axes(data, figsize_unit=(4.4, 3.4), sharey=False):
    conds = [c for c in COND_ORDER if any(k[0] == c for k in data)]
    strats = [s for s in STRAT_ORDER if any(k[1] == s for k in data)]
    fig, axes = plt.subplots(len(strats), len(conds),
                             figsize=(figsize_unit[0] * len(conds), figsize_unit[1] * len(strats)),
                             facecolor='white', squeeze=False, sharex=True, sharey=sharey)
    return fig, axes, conds, strats


def finish_grid(fig, axes, conds, strats, ylabel, title, ncols_legend=3, xlabel=True):
    for i, st in enumerate(strats):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            if i == 0:
                ax.set_title(COND_TITLE[c], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{STRATEGY_TITLE[st]}\n{ylabel}", fontsize=8, color=INK)
            if i == len(strats) - 1 and xlabel:
                ax.set_xlabel('Executed length units', fontsize=9, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=ncols_legend, frameon=False,
                   fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, fontsize=10, color=INK, y=1.03)
    fig.tight_layout()


def plot_overview(data, metric, out_path, n_max, n_shapes):
    label, logy, carry = METRICS[metric]
    fig, axes, conds, strats = grid_axes(data)
    for i, st in enumerate(strats):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            style_axes(ax)
            for m, sty in METHOD_STYLE.items():
                per = data.get((c, st, m))
                if not per:
                    continue
                M = mission_matrix(per, metric, n_max, carry)
                with np.errstate(all='ignore'):
                    mean, std = np.nanmean(M, axis=0), np.nanstd(M, axis=0)
                x = np.arange(1, n_max + 1)
                ok = np.isfinite(mean)
                lo = mean - std
                if logy:
                    lo = np.maximum(lo, mean * 0.05)
                ax.plot(x[ok], mean[ok], color=sty['color'], lw=sty['lw'], label=sty['label'],
                        alpha=0.95)
                ax.fill_between(x[ok], lo[ok], (mean + std)[ok], color=sty['color'], alpha=0.16,
                                linewidth=0)
            if logy:
                ax.set_yscale('log')
            if metric == 'swept_mass':
                ax.axhline(0.99, color=MUTED, lw=0.8, ls=':')
    note = ("finished missions keep their last value" if carry
            else "mean over the missions still running")
    finish_grid(fig, axes, conds, strats, label.split(' vs.')[0] if len(label) > 34 else label,
                f"{label} -- mean over {n_shapes} holdout shapes; band: +-1 std across shapes; {note}")
    fig.savefig(out_path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_cdf(data, out_path, n_max, n_shapes):
    fig, axes, conds, strats = grid_axes(data, sharey=True)
    for i, st in enumerate(strats):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            style_axes(ax)
            for m, sty in METHOD_STYLE.items():
                per = data.get((c, st, m))
                if not per:
                    continue
                n99 = np.array([executions_to_99(per[s]) for s in sorted(per)])
                x = np.arange(0, n_max + 1)
                frac = np.array([(n99 <= k).sum() / len(n99) for k in x])
                ax.plot(x, frac, drawstyle='steps-post', color=sty['color'], lw=sty['lw'],
                        label=sty['label'], alpha=0.95)
            ax.set_ylim(-0.02, 1.02)
            ax.axhline(1.0, color=MUTED, lw=0.6, ls=':')
    finish_grid(fig, axes, conds, strats, 'Fraction of shapes at >= 99 %',
                f"Executed length units until 99 % of the ground-truth mass is swept -- "
                f"cumulative share of the {n_shapes} holdout shapes")
    fig.savefig(out_path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_box(data, out_path, n_max, n_shapes, threshold=0.99):
    fig, axes, conds, strats = grid_axes(data, sharey=True)
    rng = np.random.default_rng(0)
    methods = list(METHOD_STYLE)
    for i, st in enumerate(strats):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            style_axes(ax)
            for k, m in enumerate(methods):
                per = data.get((c, st, m))
                if not per:
                    continue
                sty = METHOD_STYLE[m]
                n99 = np.array([executions_to_threshold(per[s], threshold) for s in sorted(per)])
                got = n99[np.isfinite(n99)]
                if len(got):
                    bp = ax.boxplot([got], positions=[k], widths=0.5, showfliers=False,
                                    patch_artist=True, manage_ticks=False)
                    for b in bp['boxes']:
                        b.set(facecolor=sty['color'], alpha=0.25, edgecolor=sty['color'])
                    for key in ('whiskers', 'caps', 'medians'):
                        for ln in bp[key]:
                            ln.set(color=sty['color'], lw=1.2)
                    ax.scatter(k + rng.uniform(-0.15, 0.15, len(got)), got, s=9,
                               color=sty['color'], alpha=0.8, zorder=3)
                miss = ~np.isfinite(n99)
                if miss.any():
                    ax.scatter(k + rng.uniform(-0.15, 0.15, miss.sum()),
                               np.full(miss.sum(), n_max + 1), s=22, marker='x',
                               color=sty['color'], alpha=0.9, zorder=3)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([METHOD_STYLE[m]['label'] for m in methods], fontsize=7,
                               rotation=12)
            ax.axhline(n_max + 0.5, color=MUTED, lw=0.6, ls=':')
            ax.set_ylim(0, n_max + 2)
    pct = f"{threshold * 100:g} %"
    finish_grid(fig, axes, conds, strats, f'Executed units until {pct}',
                f"Executed length units until {pct} of the ground-truth mass is swept -- {n_shapes} holdout shapes "
                f"(x above the dotted line: not reached within the {n_max}-unit cap)",
                xlabel=False)
    fig.savefig(out_path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_per_shape(data, metric, cond, strat, out_path):
    label, logy, carry = METRICS[metric]
    shapes = sorted({s for k, v in data.items() if k[0] == cond and k[1] == strat for s in v})
    if not shapes:
        return
    ncols = 5
    nrows = int(np.ceil(len(shapes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.6 * nrows),
                             facecolor='white', squeeze=False)
    for k, shape in enumerate(shapes):
        ax = axes[k // ncols][k % ncols]
        style_axes(ax)
        for m, sty in METHOD_STYLE.items():
            arr = data.get((cond, strat, m), {}).get(shape)
            if arr is None:
                continue
            ax.plot(arr['n_exec'], arr[metric], color=sty['color'], lw=sty['lw'] * 0.9,
                    label=sty['label'], alpha=0.95)
        if logy:
            ax.set_yscale('log')
        if metric == 'swept_mass':
            ax.axhline(0.99, color=MUTED, lw=0.7, ls=':')
        ax.set_title(shape, fontsize=9, color=INK)
        if k // ncols == nrows - 1:
            ax.set_xlabel('Executed length units', fontsize=8, color=INK)
    for k in range(len(shapes), nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9,
                   labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{COND_TITLE[cond]} -- {STRATEGY_TITLE[strat]} -- {label}", fontsize=10,
                 color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_svgd_rounds(root, data, rnd, out_path):
    """E against the planning target along the SVGD iterations of round `rnd`:
    mean over shapes x candidates, band +-1 std (log axis)."""
    series = {}
    for path in mdb.list_shards(root):
        cond, strat, method = os.path.basename(path)[:-3].split('__')
        conn = mdb.open_db(path)
        rows = conn.execute("SELECT n_iters, E_stride, E_series FROM candidates WHERE round=?",
                            (rnd,)).fetchall()
        conn.close()
        if not rows:
            continue
        n_iters, stride = rows[0][0], rows[0][1]
        E = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        series[(cond, strat, method)] = (np.array(sorted(set(range(0, n_iters + 1, stride)) | {n_iters})), E)
    if not series:
        return False
    fig, axes, conds, strats = grid_axes(series)
    for i, st in enumerate(strats):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            style_axes(ax)
            for m, sty in METHOD_STYLE.items():
                if (c, st, m) not in series:
                    continue
                x, E = series[(c, st, m)]
                mean, std = E.mean(axis=0), E.std(axis=0)
                ax.plot(x, mean, color=sty['color'], lw=sty['lw'], label=sty['label'], alpha=0.95)
                ax.fill_between(x, np.maximum(mean - std, mean * 0.05), mean + std,
                                color=sty['color'], alpha=0.16, linewidth=0)
            ax.set_yscale('log')
    for i in range(len(strats)):
        for j in range(len(conds)):
            axes[i][j].set_xlabel('SVGD iteration', fontsize=9, color=INK) if i == len(strats) - 1 else None
    finish_grid(fig, axes, conds, strats, 'Ergodic error E vs. target',
                f"Ergodic error vs. the planning target along SVGD -- planning round {rnd + 1} "
                f"(mean over shapes x candidates; band: +-1 std)", xlabel=False)
    fig.savefig(out_path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return True


def plot_paths(root, cond, strat, method, out_path, cmap):
    path = mdb.shard_path(root, cond, strat, method)
    if not os.path.exists(path):
        return
    conn = mdb.open_db(path)
    segs = {}
    for r in mdb.iter_rounds(conn, columns=('shape', 'round', 'segment', 'reached99')):
        segs.setdefault(r['shape'], []).append(r)
    shapes = sorted(segs)
    if not shapes:
        return
    ncols = 5
    nrows = int(np.ceil(len(shapes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows),
                             facecolor='white', squeeze=False)
    for k, shape in enumerate(shapes):
        ax = axes[k // ncols][k % ncols]
        ax.set_facecolor('white')
        truth = mdb.load_truth(conn, shape)
        ax.imshow(truth, origin='lower', extent=(0, 1, 0, 1), cmap=cmap, alpha=0.55, vmin=0, vmax=1)
        pts = np.concatenate([r['segment'] for r in segs[shape]], axis=0)
        ax.plot(pts[:, 0], pts[:, 1], color='#00C853', lw=1.6, alpha=0.95)
        ax.scatter([pts[0, 0]], [pts[0, 1]], s=14, color=INK, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(SPINE)
        n = len(segs[shape])
        ok = bool(segs[shape][-1]['reached99'])
        ax.set_title(f"{shape}: {n} units" + ("" if ok else " (cap)"), fontsize=8, color=INK)
    for k in range(len(shapes), nrows * ncols):
        axes[k // ncols][k % ncols].axis('off')
    fig.suptitle(f"Driven paths -- {METHOD_STYLE[method]['label']}, {STRATEGY_TITLE[strat]}, "
                 f"{COND_TITLE[cond]} (dot: start)", fontsize=10, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    conn.close()


# ── tables ───────────────────────────────────────────────────────────────────

def write_tables(data, out_dir, n_max):
    os.makedirs(out_dir, exist_ok=True)
    mission_rows, summary_rows = [], []
    for (c, st, m), per in sorted(data.items()):
        n99s = []
        for shape in sorted(per):
            a = per[shape]
            n99 = executions_to_99(a)
            n99s.append(n99)
            mission_rows.append(dict(
                knowledge_condition=c, strategy=st, method=m, shape=shape,
                rounds_run=len(a['n_exec']), reached_99=int(np.isfinite(n99)),
                executions_to_99=n99, final_swept_mass=a['swept_mass'][-1],
                final_E_truth=a['E_truth'][-1], final_E_target=a['E_target'][-1],
                final_J_truth=a['J_truth'][-1], final_path_len=a['path_len'][-1],
                info_gain_first_unit=a['info_gain'][0],
                info_gain_cum_frac_final=a['info_gain_cum_frac'][-1],
                final_belief_rmse=a['belief_rmse'][-1]))
        n99s = np.array(n99s)
        got = n99s[np.isfinite(n99s)]
        row = dict(knowledge_condition=c, strategy=st, method=m, n_shapes=len(per),
                   n_reached_99=len(got),
                   mean_exec_to_99_reached=float(got.mean()) if len(got) else float('nan'),
                   median_exec_to_99_reached=float(np.median(got)) if len(got) else float('nan'),
                   # unreached missions counted at the cap: a lower bound of the true mean
                   mean_exec_to_99_capped=float(np.where(np.isfinite(n99s), n99s, n_max).mean()))
        for k in (1, 3, 5, 10):
            for key in ('E_truth', 'swept_mass'):
                vals = [per[s][key][min(k, len(per[s][key])) - 1] for s in per]
                row[f'mean_{key}_after_{k}'] = float(np.mean(vals))
        row['mean_info_gain_cum_frac_final'] = float(np.mean(
            [per[s]['info_gain_cum_frac'][-1] for s in per]))
        summary_rows.append(row)
    for name, rows in (('missions.csv', mission_rows), ('summary.csv', summary_rows)):
        if not rows:
            continue
        with open(os.path.join(out_dir, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return summary_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--out_root', type=str, default=os.path.join(_here, 'results'))
    ap.add_argument('--no_per_shape', action='store_true')
    ap.add_argument('--paths', action='store_true',
                    help='Also draw the driven paths of every mission (27 figures).')
    ap.add_argument('--box_thresholds', type=str, default='',
                    help='Comma-separated swept-mass thresholds (e.g. 0.75,0.8,0.9,0.95,0.99): '
                         'one executions-to-threshold box plot each, in plots/executions_to_threshold/.')
    ap.add_argument('--box_only', action='store_true',
                    help='Only write the --box_thresholds figures, nothing else.')
    ap.add_argument('--svgd_rounds', type=str, default='0,5',
                    help='Planning rounds (0-based) for the SVGD-convergence figures.')
    args = ap.parse_args()

    root = os.path.join(args.out_root, args.out_tag)
    data, cfg = load_all(root)
    if not data:
        print("[plot_mission_eval] no rounds stored yet.")
        return
    n_max = int(max(len(a['n_exec']) for v in data.values() for a in v.values()))
    n_shapes = len({s for v in data.values() for s in v})
    plots = os.path.join(root, 'plots')
    os.makedirs(plots, exist_ok=True)
    cap = int(cfg.get('max_rounds', n_max))
    if args.box_thresholds:
        d = os.path.join(plots, 'executions_to_threshold')
        os.makedirs(d, exist_ok=True)
        for thr in [float(t) for t in args.box_thresholds.split(',') if t]:
            plot_box(data, os.path.join(d, f'executions_to_{thr * 100:g}_box.png'), cap, n_shapes,
                     threshold=thr)
        print(f"[plot_mission_eval] wrote threshold box plots to {d}")
        if args.box_only:
            return
    for metric in METRICS:
        plot_overview(data, metric, os.path.join(plots, f'overview_{metric}.png'), n_max, n_shapes)
    plot_cdf(data, os.path.join(plots, 'executions_to_99_cdf.png'), cap, n_shapes)
    plot_box(data, os.path.join(plots, 'executions_to_99_box.png'), cap, n_shapes)
    for rnd in [int(r) for r in args.svgd_rounds.split(',') if r != '']:
        plot_svgd_rounds(root, data, rnd, os.path.join(plots, f'svgd_convergence_round{rnd + 1}.png'))
    if not args.no_per_shape:
        d = os.path.join(plots, 'per_shape')
        os.makedirs(d, exist_ok=True)
        for c in COND_ORDER:
            for st in STRAT_ORDER:
                for metric in ('E_truth', 'swept_mass'):
                    plot_per_shape(data, metric, c, st, os.path.join(d, f'{c}_{st}_{metric}.png'))
    if args.paths:
        d = os.path.join(plots, 'paths')
        os.makedirs(d, exist_ok=True)
        from apply_cfm_belief import _white_inferno          # project WHITE_INFERNO colormap
        cmap = _white_inferno()
        for (c, st, m) in sorted(data):
            plot_paths(root, c, st, m, os.path.join(d, f'{c}_{st}_{m}.png'), cmap)
    write_tables(data, os.path.join(root, 'tables'), cap)
    print(f"[plot_mission_eval] wrote figures to {plots} and tables to {os.path.join(root, 'tables')} "
          f"({len(data)} sets, {n_shapes} shapes, up to {n_max} units)")


if __name__ == '__main__':
    main()
