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
from metrics_explore_exploit import ExploreExploitErgodic, add_length_ratios  # noqa: E402
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


def score_and_save(curve, truth, phi_k_truth, ee, shape_name, method, sub,
                   svgd_iters, knowledge_condition, raw_dir, save_files=True):
    metrics = ee.score(curve, phi_k_truth)
    row = {
        'shape': shape_name, 'knowledge_condition': knowledge_condition,
        'method': method, 'subvariant': sub,
        'svgd_iters': -1 if svgd_iters is None else int(svgd_iters),
        'variant_id': variant_id(method, sub, svgd_iters),
        'coverage': float(coverage_vs_truth(curve, truth)),
        'path_len': pl(curve),
        **metrics,
    }
    add_length_ratios(row)

    if save_files:
        d = os.path.join(raw_dir, knowledge_condition, method,
                         row['variant_id'], shape_name)
        os.makedirs(d, exist_ok=True)
        curve_np = curve.detach().cpu().numpy()
        np.save(os.path.join(d, 'trajectory.npy'), curve_np)
        with open(os.path.join(d, 'metrics.json'), 'w') as f:
            json.dump(row, f, indent=2)
        viz.plot_single_trajectory(
            truth.detach().cpu().numpy(), curve_np,
            os.path.join(d, 'viz.png'),
            title=f"{shape_name} | {method}/{sub} | {knowledge_condition} | "
                  f"svgd={row['svgd_iters']}")
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
    'E_ergodic_exploit_per_length', 'coverage_per_length',
]


def _group(rows, keys):
    """rows -> {(werte der `keys`,): [passende Zeilen]}. Kein pandas im
    Projekt-venv installiert, daher von Hand statt ueber `groupby`."""
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


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


def plot_metric_bars(rows, plots_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = ['E_ergodic_total', 'explore_exploit_ratio', 'coverage', 'path_len']
    conditions = sorted({r['knowledge_condition'] for r in rows})
    out_dir = os.path.join(plots_dir, 'metric_bars')
    os.makedirs(out_dir, exist_ok=True)

    for metric in metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='white',
                                 squeeze=False)
        for ax, cond in zip(axes.reshape(-1), conditions):
            sub = [r for r in rows if r['knowledge_condition'] == cond]
            g = _group(sub, ['variant_id'])
            pairs = sorted(((kv[0], np.mean([r[metric] for r in grp]))
                           for kv, grp in g.items()), key=lambda x: x[1])
            names = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
            ax.barh(names, vals, color='#1565C0', alpha=0.85)
            ax.set_title(cond, fontsize=9, color='#1A1A2E')
            ax.tick_params(labelsize=6)
            ax.set_facecolor('white')
            ax.grid(alpha=0.2, axis='x')
        for ax in axes.reshape(-1)[len(conditions):]:
            ax.axis('off')
        fig.suptitle(metric, color='#1A1A2E')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
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
        ax.set_xlabel('SVGD-Iterationen'); ax.set_ylabel('E_ergodic_total (Mittel)')
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
            ax.scatter([ex], [ei], color='#00C853', alpha=0.85, s=40)
            ax.annotate(name, (ex, ei), fontsize=5, color='#555')
        ax.set_xlabel('E_ergodic_explore (niedrige Frequenzen)')
        ax.set_ylabel('E_ergodic_exploit (hohe Frequenzen)')
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
