#!/usr/bin/env python3
r"""
run_mlp_policy_eval.py
=======================
Wertet die vom MLP-Wertmodell (Policy A, Zufallsmaske) vorhergesagten
Konfigurationen durch die Standard-Evaluierungsmatrix-Pipeline aus.

Das MLP sagt je Wissensstufe eine feste (phi_model, param, svgd_iters)-
Konfiguration vorher (gespeichert in `mlp_policy_predicted_configs.json`,
erzeugt von `predict_mlp_policy_configs.py`).  Diese Konfiguration wird
genau wie eine Strategie aus `variant_runner.STRATEGIES` behandelt und
durch `cfm_ideal_no_replan` + `cfm_ideal_replan_1_6` gejagt -- dasselbe
Bewertungsprotokoll wie fuer `eid_optuna_ideal_v2`.

Besonderheit: Weil die Konfiguration pro Wissensstufe *unterschiedlich*
ist (das MLP passt seine Empfehlung an die Wissensstufe an), gibt es
hier keine einzelne, globale Strategie.  Stattdessen werden vier
Pseudo-Strategien `mlp_policy_<cond>` dynamisch registriert.

Ausgabe
-------
Ergebnisse werden in `results/<out_tag>/` abgelegt (eigenes Verzeichnis,
nicht in den bestehenden Lauf geschrieben).  Anschliessend kann
`merge_mlp_into_run.py` die Zeilen in die CSV eines bestehenden Laufs
einmischen und `regen_metric_bars.py` die Plots neu erzeugen.

Schnelltest (Sekunden)
-----------------------
    python run_mlp_policy_eval.py --pilot --out_tag smoke_mlp --no_viz

Voller Lauf (Minuten, NICHT ohne Ruecksprache starten)
-------------------------------------------------------
    python run_mlp_policy_eval.py --out_tag mlp_policy_eval_20260917
"""

import argparse
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch

import apply_cfm_belief as acb                                     # noqa: E402
from common.data import load_truth                                  # noqa: E402
from common.svgd_refine import SvgdRefiner                          # noqa: E402

import variant_runner as vr                                         # noqa: E402
from metrics_explore_exploit import ExploreExploitErgodic           # noqa: E402
from run_eval_matrix import (score_and_save, summarise,             # noqa: E402
                             plot_metric_bars, plot_tradeoff,
                             DEFAULT_CKPT)
import viz                                                          # noqa: E402

MLP_CONFIGS_JSON = os.path.join(_here, 'mlp_policy_predicted_configs.json')

PILOT_SHAPES = ['A', 'rand_gmm_20', 'organic_20']

# ── Standard-Fallback-Felder (wie bei den nicht-Optuna-Strategien) ──────────
_MLP_DEFAULTS = dict(
    debt_weight=0.6, visit_sat=1.0, visit_halflife=3.0,
    phi_mode='uniform', phi_quantile=0.5,
    gp_noise=0.05, gp_lengthscale=0.08,
    n_particles=vr.N_PARTICLES, cfg_weight=2.0,
)


def load_mlp_configs(path=MLP_CONFIGS_JSON):
    """Laedt `mlp_policy_predicted_configs.json` und gibt ein
    {condition: {phi_model, param, svgd_iters, ...}} dict zurueck."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    per_cond = {}
    for cond, info in data['per_condition'].items():
        entry = dict(_MLP_DEFAULTS)
        entry['phi_model'] = info['phi_model']
        entry['param'] = float(info['param'])
        entry['svgd_iters'] = int(info['svgd_iters'])
        per_cond[cond] = entry
    return per_cond, data


def _build_mlp_strategy(cond_entry, device):
    """Baut `(args, svgd_iters, cfg_weight)` aus einem MLP-Konfig-Dict,
    analog zu `variant_runner.build_strategy_args`."""
    from exploration_optimierung.mission import build_mission_args
    s = cond_entry
    args = build_mission_args(
        str(device), phi_model=s['phi_model'], param=s['param'],
        debt_weight=s['debt_weight'], visit_sat=s['visit_sat'],
        sensor_radius=0.06, gp_noise=s['gp_noise'],
        n_particles=s['n_particles'], meas_noise=0.05, max_obs=64,
        visit_halflife=s['visit_halflife'], phi_mode=s['phi_mode'])
    args.phi_quantile = s['phi_quantile']
    return args, s['svgd_iters'], s['cfg_weight']


def run_mlp_no_replan(planner, belief, cond_entry, refiner):
    """Wie `cfm_ideal_no_replan`, aber Strategie kommt aus dem MLP-Dict."""
    args, svgd_iters, cfg_weight = _build_mlp_strategy(cond_entry, belief.device)
    planner.cfg_weight = cfg_weight
    mu, sd = belief.posterior_grid()
    phi = acb.zieldichte(mu, sd, args.kappa, args)
    parts = acb.phi_particles(phi, args.n_particles, mode=args.phi_mode,
                              quantile=args.phi_quantile, device=belief.device)
    cps = planner.plan(parts, n_candidates=1)
    curve = planner.render(cps)[0]
    return vr.svgd(refiner, curve, phi, svgd_iters)


def run_mlp_replan_1_6(planner, belief, truth, condition, cond_entry, refiner):
    """Wie `cfm_ideal_replan_1_6`, aber Strategie kommt aus dem MLP-Dict."""
    from common.metrics import trim_to_length, path_length as _pl
    from exploration_optimierung.mission import resample_arclength, LENGTH_UNIT
    args, svgd_iters, cfg_weight = _build_mlp_strategy(cond_entry, belief.device)
    planner.cfg_weight = cfg_weight
    driven = None
    for _ in range(vr.N_REPLAN_ROUNDS):
        mu, sd = belief.posterior_grid()
        visit = vr._visit_field(driven, args, belief.device)
        phi, _v = acb.debt_density(mu, sd, visit, args.kappa, args)
        parts = acb.phi_particles(phi, args.n_particles, mode=args.phi_mode,
                                  quantile=args.phi_quantile, device=belief.device)
        start = None if driven is None else driven[-1]
        cps = planner.plan(parts, n_candidates=1, start=start)
        curve = planner.render(cps)[0]
        curve = vr.svgd(refiner, curve, phi, svgd_iters, start=start)

        seg = trim_to_length(curve, LENGTH_UNIT)
        n_pts = max(8, int(round(72 * _pl(seg) / LENGTH_UNIT)))
        seg = resample_arclength(seg, n_pts)

        vr._observe_segment(belief, seg, truth, condition,
                            noise=args.noise,
                            sensor_radius=args.sensor_radius,
                            max_obs=args.max_obs)
        driven = seg if driven is None else torch.cat([driven, seg], dim=0)
    return driven


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true',
                    help='Nur 3 Pilot-Formen statt aller 25 Holdout-Formen.')
    ap.add_argument('--shapes', type=str, default=None)
    ap.add_argument('--conditions', type=str,
                    default=','.join(vr.KNOWLEDGE_CONDITIONS))
    ap.add_argument('--replan_schemes', type=str, default='no_replan,replan_1_6')
    ap.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--mlp_configs', type=str, default=MLP_CONFIGS_JSON)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--truth_res', type=int, default=96)
    ap.add_argument('--no_viz', action='store_true')
    args = ap.parse_args()

    mlp_per_cond, mlp_meta = load_mlp_configs(args.mlp_configs)
    conditions = [c for c in args.conditions.split(',') if c]
    schemes = [s for s in args.replan_schemes.split(',') if s]

    if args.shapes:
        shapes = [s.strip() for s in args.shapes.split(',')]
    elif args.pilot:
        shapes = PILOT_SHAPES
    else:
        shapes = None

    device = args.device
    names, truths = load_truth(labels=shapes, n=999, split='val',
                               resolution=args.truth_res, device=device)
    print(f"[mlp_eval] {len(names)} Formen: {names}")
    print(f"[mlp_eval] Wissensstufen: {conditions}  Schemata: {schemes}")
    for cond in conditions:
        c = mlp_per_cond[cond]
        print(f"[mlp_eval]   {cond:>12}: model={c['phi_model']:<7} "
              f"param={c['param']:<10.4f} svgd_iters={c['svgd_iters']}")

    planner = acb.CfmPlanner(ckpt=args.ckpt, device=device, pts=vr.PTS_RENDER,
                              steps=100, cfg_weight=2.0)
    if not planner.start_cond:
        raise RuntimeError(
            f"{os.path.basename(args.ckpt)} ist nicht startpunkt-konditioniert; "
            "replan_1_6 braucht start=.")
    refiner = SvgdRefiner(seed=args.seed)
    ee = ExploreExploitErgodic(device=device)

    out_dir = os.path.join(_here, 'results', args.out_tag)
    raw_dir = os.path.join(out_dir, 'raw')
    tables_dir = os.path.join(out_dir, 'tables')
    plots_dir = os.path.join(out_dir, 'plots')
    for d in (raw_dir, tables_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(
            mlp_configs=args.mlp_configs,
            mlp_meta=mlp_meta,
            conditions=conditions, schemes=schemes,
            shapes_resolved=names,
            ckpt=args.ckpt, seed=args.seed, truth_res=args.truth_res,
        ), f, indent=2)

    rows = []
    panels = {}

    def _collect(cond, row, curve, truth_np, name):
        key = (cond, row['variant_id'])
        panels.setdefault(key, []).append((name, truth_np, curve.detach().cpu().numpy()))

    t0 = time.time()
    for i, name in enumerate(names):
        truth = truths[i]
        truth_np = truth.detach().cpu().numpy()
        phi_k_truth = ee.target_coeffs(truth)

        for cond in conditions:
            cond_entry = mlp_per_cond[cond]
            belief = vr.build_belief(
                cond, truth, seed=args.seed, device=device,
                gp_noise=cond_entry['gp_noise'],
                gp_lengthscale=cond_entry['gp_lengthscale'])

            for scheme in schemes:
                b = belief.clone()
                # variant_id-Schema: `mlp_policy__<scheme>` (Doppelunterstrich,
                # wie `eid_optuna_ideal_v2__no_replan`).
                # method='mlp_policy', sub='_<scheme>' -> variant_id(method,sub,None)
                # = 'mlp_policy__<scheme>'.  Erscheint so in JEDEM Subplot
                # (nicht nur in dem der Bedingung, der die Config gehoert),
                # weil der variant_id keine Bedingung eincodiert.
                strat_name = 'mlp_policy'
                sub = f'_{scheme}'   # fuehrendes _ gibt den __ im variant_id
                if scheme == 'no_replan':
                    curve = run_mlp_no_replan(planner, b, cond_entry, refiner)
                elif scheme == 'replan_1_6':
                    curve = run_mlp_replan_1_6(planner, b, truth, cond,
                                               cond_entry, refiner)
                else:
                    raise KeyError(f"unbekanntes Replan-Schema {scheme!r}")

                row = score_and_save(curve, truth, phi_k_truth, ee, name,
                                     'mlp_policy', sub, None, cond,
                                     raw_dir, save_files=not args.no_viz)
                row.update(representation='particles',
                           strategy=strat_name,
                           replan_scheme=scheme)
                rows.append(row)
                _collect(cond, row, curve, truth_np, name)

        print(f"[mlp_eval] [{i + 1}/{len(names)}] {name} fertig, "
              f"{time.time() - t0:.1f}s seit Start, {len(rows)} Zeilen")

    summarise(rows, tables_dir)
    plot_metric_bars(rows, plots_dir)
    plot_tradeoff(rows, plots_dir)

    panel_dir = os.path.join(plots_dir, 'holdout_panels')
    for (cond, vid), items in panels.items():
        d = os.path.join(panel_dir, cond)
        os.makedirs(d, exist_ok=True)
        viz.plot_holdout_panel(items, os.path.join(d, f'{vid}.png'),
                               title=f'{vid} | {cond}')

    print(f"[mlp_eval] fertig: {len(rows)} Zeilen, "
          f"{time.time() - t0:.1f}s gesamt -> {out_dir}")
    return out_dir


if __name__ == '__main__':
    main()
