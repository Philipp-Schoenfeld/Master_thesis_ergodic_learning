r"""
run_eval_matrix.py
===================
Orchestrator der vollen Auswertungsmatrix: fuer jede Holdout-Form, jede
Wissensstufe und jede Methoden-Variante wird eine Trajektorie erzeugt, gegen
die wahre Dichte bewertet und gespeichert (Bahn, Metriken, Einzel-Viz).
Danach: Tabellen und Sammel-Plots ueber alle Formen.

Beispiele
---------
    # Pilot: 3 Formen, schneller SVGD-Sweep, zur Mechanik-Pruefung
    python run_eval_matrix.py --pilot --svgd_iters 0,25 --out_tag pilot_smoke

    # Voller Lauf (NICHT ohne Ruecksprache starten, siehe CLAUDE.md)
    python run_eval_matrix.py --out_tag full_run
"""

import argparse
import csv
import json
import os
import sys
import time

# ── sys.path-Bootstrap, eigenstaendig (dieselbe Loesung wie __init__.py) ───
_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
          os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import apply_cfm_belief as acb                                   # noqa: E402
from common.data import load_truth                               # noqa: E402
from common.metrics import coverage_vs_truth, path_length as pl  # noqa: E402
from common.svgd_refine import SvgdRefiner                        # noqa: E402

import variant_runner as vr                                       # noqa: E402
from metrics_explore_exploit import (ExploreExploitErgodic, add_length_ratios,  # noqa: E402
                                     add_J, smoothness_energy, steps_to_full_coverage)
import viz                                                         # noqa: E402

DEFAULT_CKPT = os.path.join(_root, 'transfer', 'netz2d_startpunkt.pt')
PILOT_SHAPES = ['A', 'rand_gmm_20', 'organic_20']


def build_planner(ckpt, device):
    p = acb.CfmPlanner(ckpt=ckpt, device=device, pts=vr.PTS_RENDER,
                       steps=100, cfg_weight=2.0)
    if not p.start_cond:
        raise RuntimeError(
            f"{os.path.basename(ckpt)} ist nicht startpunkt-konditioniert; "
            "die Replanning-Varianten (one_replan/replan_1_6) setzen jede "
            "Runde am Endpunkt der vorigen an.")
    return p


def variant_id(method, sub, svgd_iters):
    if svgd_iters is None:
        return f"{method}_{sub}"
    return f"{method}_{sub}_svgd{svgd_iters}"


def compute_row(curve, truth, phi_k_truth, ee, shape_name, method, sub,
                svgd_iters, knowledge_condition):
    """All per-trajectory metrics as a plain dict -- no file I/O. Split out
    of `score_and_save` so a candidate curve can be scored without writing
    `trajectory.npy`/`viz.png` for it, e.g. to evaluate many candidates in
    `run_best_of_n_matrix.py` and only save the one that's kept."""
    metrics = ee.score(curve, phi_k_truth)
    steps_frac, steps_reached = steps_to_full_coverage(
        curve, truth, knowledge_condition=knowledge_condition)
    row = {
        'shape': shape_name, 'knowledge_condition': knowledge_condition,
        'method': method, 'subvariant': sub,
        'svgd_iters': -1 if svgd_iters is None else int(svgd_iters),
        'variant_id': variant_id(method, sub, svgd_iters),
        'coverage': float(coverage_vs_truth(curve, truth)),
        'path_len': pl(curve),
        'smoothness_energy': smoothness_energy(curve),
        'steps_to_full_coverage': steps_frac,
        'steps_to_full_coverage_reached': float(steps_reached),
        **metrics,
    }
    add_length_ratios(row)
    add_J(row)
    return row


def save_trajectory_files(curve, row, truth, raw_dir):
    """`trajectory.npy` + `metrics.json` + `viz.png` for one already-scored
    row. Split out of `score_and_save` so `run_best_of_n_matrix.py` can save
    exactly the one candidate it kept, after scoring all N -- without
    re-deriving the file path/title logic a second time."""
    d = os.path.join(raw_dir, row['knowledge_condition'], row['method'],
                     row['variant_id'], row['shape'])
    os.makedirs(d, exist_ok=True)
    curve_np = curve.detach().cpu().numpy()
    np.save(os.path.join(d, 'trajectory.npy'), curve_np)
    with open(os.path.join(d, 'metrics.json'), 'w') as f:
        json.dump(row, f, indent=2)
    viz.plot_single_trajectory(
        truth.detach().cpu().numpy(), curve_np,
        os.path.join(d, 'viz.png'),
        title=f"{row['shape']} | {row['method']}/{row['subvariant']} | "
              f"{row['knowledge_condition']} | svgd={row['svgd_iters']}")


def score_and_save(curve, truth, phi_k_truth, ee, shape_name, method, sub,
                   svgd_iters, knowledge_condition, raw_dir, save_files=True):
    row = compute_row(curve, truth, phi_k_truth, ee, shape_name, method, sub,
                      svgd_iters, knowledge_condition)
    if save_files:
        save_trajectory_files(curve, row, truth, raw_dir)
    return row


def write_csv(rows, path):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


NUMERIC_METRIC_KEYS = [
    'coverage', 'path_len', 'E_ergodic_total', 'E_ergodic_explore',
    'E_ergodic_exploit', 'explore_exploit_ratio',
    'E_ergodic_total_per_length', 'E_ergodic_explore_per_length',
    'E_ergodic_exploit_per_length', 'coverage_per_length', 'J',
    'smoothness_energy', 'steps_to_full_coverage',
    'steps_to_full_coverage_reached',
]


def _group(rows, keys):
    """rows -> {(werte der `keys`,): [passende Zeilen]}. Kein pandas im
    Projekt-venv installiert, daher von Hand statt ueber `groupby`."""
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


#: Explicit optimisation direction per metric -- used by `best_of_n_metrics`
#: (`run_best_of_n_matrix.py`) to decide which of N candidates counts as
#: "best" for a given metric panel. `None` marks a diagnostic quantity with
#: no principled better/worse direction (matches its `METRIC_INFO['direction']`
#: text below, e.g. "no blanket better/worse"); for those, best-of-N reports
#: the mean across candidates in place of a "best" pick, since picking one
#: would be arbitrary.
METRIC_DIRECTION = {
    'E_ergodic_total': 'min', 'E_ergodic_explore': 'min', 'E_ergodic_exploit': 'min',
    'E_ergodic_total_per_length': 'min', 'E_ergodic_explore_per_length': 'min',
    'E_ergodic_exploit_per_length': 'min',
    'explore_exploit_ratio': None,
    'coverage': 'min', 'coverage_per_length': 'min',
    'path_len': None,
    'J': 'min',
    'smoothness_energy': 'min',
    'steps_to_full_coverage': 'min',
    'steps_to_full_coverage_reached': 'max',
}


def best_of_n_metrics(candidate_rows, metric_keys=NUMERIC_METRIC_KEYS):
    """`candidate_rows`: per-candidate metric dicts (`compute_row`'s return
    shape) for N candidates of the *same* (shape, variant, knowledge
    condition) test case -- e.g. N different CFM samples or N random-walk
    seeds. Returns a dict of fields to merge into one representative row:

        <metric>       the best candidate's value, direction-aware
                       (`METRIC_DIRECTION`); for directionless metrics
                       (`path_len`, `explore_exploit_ratio`) the mean, since
                       there is no principled "best" to pick there.
        <metric>_mean  mean across all N candidates.
        <metric>_std   standard deviation across all N candidates.

    `_mean`/`_std` are always populated, for every metric -- the "plot the
    mean and standard deviation alongside it" Philipp asked for
    (2026-09-17), shown as an overlay by `plot_metric_bars(...,
    show_distribution=True)`.
    """
    out = {}
    for k in metric_keys:
        vals = [r[k] for r in candidate_rows if k in r and r[k] is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[f'{k}_mean'] = float(arr.mean())
        out[f'{k}_std'] = float(arr.std())
        direction = METRIC_DIRECTION.get(k)
        if direction == 'min':
            out[k] = float(arr.min())
        elif direction == 'max':
            out[k] = float(arr.max())
        else:
            out[k] = float(arr.mean())
    return out


def _agg_stats(rows, keys=NUMERIC_METRIC_KEYS):
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[f'{k}_mean'] = float(arr.mean())
        out[f'{k}_median'] = float(np.median(arr))
        out[f'{k}_max'] = float(arr.max())
    return out


def summarise(rows, tables_dir):
    write_csv(rows, os.path.join(tables_dir, 'all_runs.csv'))
    for keys, name in [
        (['method', 'subvariant', 'svgd_iters'], 'summary_by_method.csv'),
        (['knowledge_condition'], 'summary_by_knowledge_condition.csv'),
        (['shape'], 'summary_by_shape.csv'),
    ]:
        g = _group(rows, keys)
        out_rows = []
        for kv, grp in g.items():
            row = dict(zip(keys, kv))
            row['n'] = len(grp)
            row.update(_agg_stats(grp))
            out_rows.append(row)
        write_csv(out_rows, os.path.join(tables_dir, name))


#: Formula + direction ("smaller/bigger = better") per metric, shown under
#: the title in `plot_metric_bars`. Formulas taken from
#: `metrics_explore_exploit.py` (E_*, explore_exploit_ratio, J) resp.
#: `exploration/common/metrics.py::coverage_vs_truth` (coverage) and
#: `path_length` (path_len) -- see there for the derivation.
METRIC_INFO = {
    'E_ergodic_total': dict(
        formula=r'$E_{\mathrm{total}} = w \cdot \frac{1}{2}\sum_k \Lambda_k\,(c_k(\gamma) - \phi_k)^2$',
        direction='smaller = better'),
    'E_ergodic_explore': dict(
        formula=r'$E_{\mathrm{explore}} = w \cdot \frac{1}{2}\sum_{\|k\|\leq\theta} \Lambda_k\,(c_k(\gamma) - \phi_k)^2$',
        direction='smaller = better (exploration alone: low frequencies / global structure)'),
    'E_ergodic_exploit': dict(
        formula=r'$E_{\mathrm{exploit}} = w \cdot \frac{1}{2}\sum_{\|k\|>\theta} \Lambda_k\,(c_k(\gamma) - \phi_k)^2$',
        direction='smaller = better (exploitation: high frequencies / ergodic coverage of the ground truth)'),
    'explore_exploit_ratio': dict(
        formula=r'$E_{\mathrm{exploit}} \,/\, E_{\mathrm{explore}}$',
        direction='no blanket better/worse -- diagnostic for the weighting, not for quality'),
    'coverage': dict(
        formula=r'$\frac{\sum_i \phi(x_i)\cdot \min_t \|x_i - \gamma(t)\|}{\sum_i \phi(x_i)}$',
        direction='smaller = better'),
    'path_len': dict(
        formula=r'$L(\gamma) = \sum_t \|\gamma(t{+}1) - \gamma(t)\|$',
        direction='no blanket better/worse -- a budget/cost quantity, not a quality metric'),
    'J': dict(
        formula=r'$J = E_{\mathrm{total}} + \lambda_{\mathrm{len}} \cdot L(\gamma),\ \ \lambda_{\mathrm{len}}=0.02$',
        direction='smaller = better'),
    'smoothness_energy': dict(
        formula=r'$w \cdot \sum_t \|\gamma(t{+}1) - 2\gamma(t) + \gamma(t{-}1)\|^2,\ \ w=15$ (TSVEC solver term, path resampled to 128 pts)',
        direction='smaller = better (less steering effort/acceleration)'),
    'steps_to_full_coverage': dict(
        formula=r'$\min\{s/L(\gamma) : \mathrm{swept\_mass}(\gamma[0{:}s]) \geq 0.99\}$ (sensor radius 0.06; 0 for ground\_truth by construction)',
        direction='smaller = better (fewer steps to fully explore the unknown region); 1.0 with '
                  'steps_to_full_coverage_reached=0 means the threshold was never reached'),
    'steps_to_full_coverage_reached': dict(
        formula=r'fraction of trajectories that ever swept $\geq 99\%$ of the true target mass',
        direction='bigger = better (99% is a strict threshold -- most single-shot variants '
                  'never reach it, so the reach *rate* itself is informative, not just the step count)'),
}

DEFAULT_BAR_METRICS = ['E_ergodic_total', 'explore_exploit_ratio', 'coverage',
                       'path_len', 'E_ergodic_explore', 'E_ergodic_exploit', 'J',
                       'smoothness_energy', 'steps_to_full_coverage',
                       'steps_to_full_coverage_reached']

#: Bar/marker color for the Optuna-tuned ideal configuration (`eid_optuna_ideal_v2`,
#: see `variant_runner.STRATEGIES` -- the 950-trial TPE+Hyperband result run
#: locally). Highlighted in every plot so it stands out against the
#: hand-tuned/baseline variants. Same green as "generated trajectories" in
#: the project's plotting style guide.
#:
#: Deliberately does NOT match `mi_optuna_gross` (the `ideal_v2_gross_cluster`
#: study, run on the cluster -- see `variant_runner.py` comment there: of
#: 5037 trials, 98% crashed immediately, so it's not a fair 16-dim search and
#: must not be highlighted as "the" Optuna result).
OPTUNA_IDEAL_COLOR = '#00C853'
#: Label color for the MLP value-model policy's predicted configs
#: (`mlp_policy_*`, see `predict_mlp_policy_configs.py` /
#: `run_mlp_policy_eval.py`) -- a different color from `OPTUNA_IDEAL_COLOR`
#: so both can be highlighted at once without being confused for each other.
MLP_POLICY_COLOR = '#1565C0'
#: Neutral fill for every bar; kept grey (not blue) precisely so
#: `MLP_POLICY_COLOR` still stands out as a label color against it.
BAR_COLOR = '#90A4AE'
_OPTUNA_IDEAL_STRATEGY = 'eid_optuna_ideal_v2'
_MLP_POLICY_STRATEGY = 'mlp_policy'


def is_optuna_ideal(variant_id):
    return _OPTUNA_IDEAL_STRATEGY in variant_id


def is_mlp_policy(variant_id):
    return _MLP_POLICY_STRATEGY in variant_id


#: `(predicate, color, legend text)` -- checked in order, first match wins,
#: so a variant_id matching more than one predicate still gets one color.
LABEL_HIGHLIGHTS = [
    (is_optuna_ideal, OPTUNA_IDEAL_COLOR, 'Optuna ideal config (eid_optuna_ideal_v2)'),
    (is_mlp_policy, MLP_POLICY_COLOR, 'MLP value-model policy prediction (mlp_policy_*)'),
]


#: Marker color for the mean/std overlay in `plot_metric_bars(...,
#: show_distribution=True)` -- distinct from `BAR_COLOR` (the best-of-N bar
#: itself) and from both `LABEL_HIGHLIGHTS` colors, so all three stay
#: visually separable on the same panel.
DISTRIBUTION_MARKER_COLOR = '#E65100'


def plot_metric_bars(rows, plots_dir, metrics=DEFAULT_BAR_METRICS,
                     show_distribution=False):
    """`show_distribution=True`: for rows that carry `<metric>_mean`/
    `<metric>_std` (best-of-N rows, see `best_of_n_metrics` /
    `run_best_of_n_matrix.py`), overlay a diamond marker at the mean with an
    error bar at +-1 std, next to the best-of-N bar itself. Rows without
    those fields (deterministic variants: heuristic, linear, lawnmower) are
    left as plain bars, no overlay -- there is nothing to average."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conditions = sorted({r['knowledge_condition'] for r in rows})
    out_dir = os.path.join(plots_dir, 'metric_bars')
    os.makedirs(out_dir, exist_ok=True)

    for metric in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), facecolor='white',
                                 squeeze=False)
        for ax, cond in zip(axes.reshape(-1), conditions):
            sub = [r for r in rows if r['knowledge_condition'] == cond]
            g = _group(sub, ['variant_id'])
            pairs = sorted(((kv[0], np.mean([r[metric] for r in grp]))
                           for kv, grp in g.items()), key=lambda x: x[1])
            names = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
            ax.barh(names, vals, color=BAR_COLOR, alpha=0.85)
            if show_distribution:
                mean_key, std_key = f'{metric}_mean', f'{metric}_std'
                for yi, name in enumerate(names):
                    grp = g[(name,)]
                    means = [r[mean_key] for r in grp if r.get(mean_key) is not None]
                    stds = [r[std_key] for r in grp if r.get(std_key) is not None]
                    if not means:
                        continue
                    m, s = float(np.mean(means)), float(np.mean(stds))
                    ax.errorbar([m], [yi], xerr=[s], fmt='D',
                               color=DISTRIBUTION_MARKER_COLOR, ms=4,
                               capsize=3, elinewidth=1, zorder=5)
                    ax.text(m, yi + 0.28, f'μ={m:.3g}±{s:.2g}',
                           fontsize=5, ha='center', va='bottom',
                           color=DISTRIBUTION_MARKER_COLOR)
            ax.set_title(cond, fontsize=9, color='#1A1A2E')
            ax.tick_params(labelsize=6)
            for label, n in zip(ax.get_yticklabels(), names):
                for predicate, color, _legend in LABEL_HIGHLIGHTS:
                    if predicate(n):
                        label.set_color(color)
                        label.set_fontweight('bold')
                        break
            ax.set_facecolor('white')
            ax.grid(alpha=0.2, axis='x')
        for ax in axes.reshape(-1)[len(conditions):]:
            ax.axis('off')
        info = METRIC_INFO.get(metric)
        fig.suptitle(metric, color='#1A1A2E', fontsize=13, y=0.99)
        if info is not None:
            fig.text(0.5, 0.95, info['formula'], ha='center', va='top',
                     fontsize=11, color='#1A1A2E')
            fig.text(0.5, 0.87, info['direction'], ha='center', va='top',
                     fontsize=9, color='#555', style='italic')
            top = 0.81
        else:
            top = 0.96
        used = {c for predicate, c, _t in LABEL_HIGHLIGHTS
               if any(predicate(r['variant_id']) for r in rows)}
        legend_y = 0.02
        for predicate, color, text in LABEL_HIGHLIGHTS:
            if color not in used:
                continue
            fig.text(0.5, legend_y, f'bold label = {text}', ha='center',
                     va='bottom', fontsize=8, color=color, fontweight='bold')
            legend_y += 0.025
        n_legend_lines = len(used)
        if show_distribution:
            fig.text(0.5, legend_y, f'◆ = mean ± std over the best-of-N candidates '
                     '(only for variants that sample multiple candidates -- CFM, random walk)',
                     ha='center', va='bottom', fontsize=8,
                     color=DISTRIBUTION_MARKER_COLOR, fontweight='bold')
            n_legend_lines += 1
        fig.tight_layout(rect=[0, 0.03 + 0.025 * max(n_legend_lines - 1, 0), 1, top])
        fig.savefig(os.path.join(out_dir, f'{metric}_by_method.png'),
                   dpi=130, facecolor='white')
        plt.close(fig)


def plot_svgd_sweeps(rows, plots_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = os.path.join(plots_dir, 'svgd_sweeps')
    os.makedirs(out_dir, exist_ok=True)
    swept = [r for r in rows if r['svgd_iters'] >= 0]
    for (method, sub), grp in _group(swept, ['method', 'subvariant']).items():
        fig, ax = plt.subplots(figsize=(6, 4.5), facecolor='white')
        for (cond,), g2 in _group(grp, ['knowledge_condition']).items():
            by_iters = _group(g2, ['svgd_iters'])
            pts = sorted((kv[0], np.mean([r['E_ergodic_total'] for r in grp2]))
                        for kv, grp2 in by_iters.items())
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker='o', label=cond)
        ax.set_xlabel('SVGD iterations'); ax.set_ylabel('E_ergodic_total (mean)')
        ax.set_title(f'{method}/{sub}', color='#1A1A2E')
        ax.legend(fontsize=7); ax.grid(alpha=0.2); ax.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{method}_{sub}.png'),
                   dpi=130, facecolor='white')
        plt.close(fig)


def plot_tradeoff(rows, plots_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = os.path.join(plots_dir, 'explore_exploit_tradeoff')
    os.makedirs(out_dir, exist_ok=True)
    for (cond,), grp in _group(rows, ['knowledge_condition']).items():
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
        g = _group(grp, ['variant_id'])
        for (name,), grp2 in g.items():
            ex = np.mean([r['E_ergodic_explore'] for r in grp2])
            ei = np.mean([r['E_ergodic_exploit'] for r in grp2])
            highlight = next((c for predicate, c, _t in LABEL_HIGHLIGHTS
                             if predicate(name)), None)
            ax.scatter([ex], [ei], color=BAR_COLOR, alpha=0.85,
                      s=40, zorder=3 if highlight else 2)
            ax.annotate(name, (ex, ei), fontsize=6 if highlight else 5,
                       color=highlight or '#555',
                       fontweight='bold' if highlight else 'normal')
        ax.set_xlabel('E_ergodic_explore (low frequencies)')
        ax.set_ylabel('E_ergodic_exploit (high frequencies)')
        ax.set_title(cond, color='#1A1A2E')
        ax.grid(alpha=0.2); ax.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'tradeoff_{cond}.png'),
                   dpi=130, facecolor='white')
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true',
                    help='Nur PILOT_SHAPES statt aller 24 Holdout-Formen.')
    ap.add_argument('--shapes', type=str, default=None,
                    help='Komma-Liste von Formnamen, ueberschreibt --pilot.')
    ap.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--svgd_iters', type=str, default='0,25,500,1000')
    ap.add_argument('--n_peaks', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--truth_res', type=int, default=96)
    ap.add_argument('--no_viz', action='store_true',
                    help='Einzel-Trajektorien-PNGs ueberspringen (schneller Smoke-Test).')
    args = ap.parse_args()

    svgd_iters = [int(x) for x in args.svgd_iters.split(',') if x != '']
    if args.shapes:
        shapes = [s.strip() for s in args.shapes.split(',')]
    elif args.pilot:
        shapes = PILOT_SHAPES
    else:
        shapes = None   # alle 24 Validierungsformen

    device = args.device
    names, truths = load_truth(labels=shapes, n=999, split='val',
                               resolution=args.truth_res, device=device)
    print(f"[eval_matrix] {len(names)} Formen: {names}")

    planner = build_planner(args.ckpt, device)
    refiner = SvgdRefiner(seed=args.seed)
    ee = ExploreExploitErgodic(device=device)

    out_dir = os.path.join(_here, 'results', args.out_tag)
    raw_dir = os.path.join(out_dir, 'raw')
    tables_dir = os.path.join(out_dir, 'tables')
    plots_dir = os.path.join(out_dir, 'plots')
    for d in (raw_dir, tables_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        cfg = dict(vars(args))
        cfg.update(shapes_resolved=names,
                  knowledge_conditions=vr.KNOWLEDGE_CONDITIONS,
                  kappa=vr.KAPPA, length_unit=vr.LENGTH_UNIT)
        json.dump(cfg, f, indent=2)

    rows = []
    panels = {}   # (knowledge_condition, variant_id) -> [(name, truth_np, curve_np)]

    def _collect(cond, row, curve, truth_np, name):
        key = (cond, row['variant_id'])
        panels.setdefault(key, []).append((name, truth_np, curve.detach().cpu().numpy()))

    t0 = time.time()
    for i, name in enumerate(names):
        truth = truths[i]
        truth_np = truth.detach().cpu().numpy()
        phi_k_truth = ee.target_coeffs(truth)

        # -- Wissensstufen-unabhaengige Baselines: einmal je Form ----------
        fixed = {
            ('lawnmower', 'fixed'): vr.lawnmower_variant(),
            ('random_walk', 'fixed'): vr.random_walk_variant(
                seed=args.seed * 131 + i),
        }
        for (method, sub), curve in fixed.items():
            curve = curve.to(device)
            base_row = score_and_save(curve, truth, phi_k_truth, ee, name,
                                      method, sub, None, 'shared', raw_dir,
                                      save_files=not args.no_viz)
            for cond in vr.KNOWLEDGE_CONDITIONS:
                row = dict(base_row, knowledge_condition=cond)
                rows.append(row)
                _collect(cond, row, curve, truth_np, name)

        # -- Wissensstufen-abhaengige Varianten -----------------------------
        for cond in vr.KNOWLEDGE_CONDITIONS:
            belief = vr.build_belief(cond, truth, seed=args.seed, device=device)

            for scheme in ('no_replan', 'one_replan', 'replan_1_6'):
                for n_iters in svgd_iters:
                    b = belief.clone()
                    if scheme == 'no_replan':
                        curve = vr.cfm_no_replan(planner, b, n_iters, refiner)
                    elif scheme == 'one_replan':
                        curve = vr.cfm_one_replan(planner, b, truth, cond,
                                                  n_iters, refiner,
                                                  seed=args.seed)
                    else:
                        curve = vr.cfm_replan_1_6(planner, b, truth, cond,
                                                  n_iters, refiner)
                    row = score_and_save(curve, truth, phi_k_truth, ee, name,
                                         'cfm', scheme, n_iters, cond,
                                         raw_dir, save_files=not args.no_viz)
                    rows.append(row)
                    _collect(cond, row, curve, truth_np, name)

            for kind in ('diagonal', 'straight_top'):
                for n_iters in svgd_iters:
                    curve = vr.linear_variant(kind, belief, n_iters, refiner)
                    row = score_and_save(curve, truth, phi_k_truth, ee, name,
                                         'linear', kind, n_iters, cond,
                                         raw_dir, save_files=not args.no_viz)
                    rows.append(row)
                    _collect(cond, row, curve, truth_np, name)

            for n_iters in svgd_iters:
                curve = vr.heuristic_variant(belief, n_iters, refiner,
                                            n_peaks=args.n_peaks)
                row = score_and_save(curve, truth, phi_k_truth, ee, name,
                                     'heuristic', 'peaks', n_iters, cond,
                                     raw_dir, save_files=not args.no_viz)
                rows.append(row)
                _collect(cond, row, curve, truth_np, name)

        print(f"[eval_matrix] [{i + 1}/{len(names)}] {name} fertig, "
             f"{time.time() - t0:.1f}s seit Start, {len(rows)} Zeilen")

    summarise(rows, tables_dir)
    plot_metric_bars(rows, plots_dir)
    plot_svgd_sweeps(rows, plots_dir)
    plot_tradeoff(rows, plots_dir)

    panel_dir = os.path.join(plots_dir, 'holdout_panels')
    for (cond, vid), items in panels.items():
        d = os.path.join(panel_dir, cond)
        os.makedirs(d, exist_ok=True)
        viz.plot_holdout_panel(items, os.path.join(d, f'{vid}.png'),
                               title=f'{vid} | {cond}')

    print(f"[eval_matrix] fertig: {len(rows)} Zeilen, "
         f"{time.time() - t0:.1f}s gesamt -> {out_dir}")


if __name__ == '__main__':
    main()
