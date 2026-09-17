#!/usr/bin/env python3
r"""
eval_optuna_best.py
====================
Wertet die von der Optuna-Studie (`optuna_search.py`, Raum `ideal`) gefundene
beste Einstellung mit derselben Infrastruktur aus wie die handgeschriebene
Suche in `optimize.py`/`plots.py`: Panel-Bild ueber alle 25 Holdout-Formen,
eine Metrik-Tabelle je Form (CSV) und die vollstaendige (n, q)-Kurve.

`plots.panel()` faehrt die Mission ueber `mission.build_mission_args`, die den
12-dimensionalen `ideal`-Suchraum nicht kennt (kein `phi_mode`, `phi_quantile`,
`cfg_weight`, `gp_lengthscale`) — ein direkter Aufruf wuerde also eine andere,
falsche Einstellung fahren als die, die Optuna tatsaechlich gefunden hat.
Dieses Skript baut die Missionsargumente stattdessen ueber
`optuna_search.build_args`, das genau diesen Raum abdeckt, und fuehrt danach
denselben Rollout + dieselben Abbildungen wie gewohnt aus (Panel, Streuung je
Form, Abdeckungskurve — Funktionen aus `plots.py` werden dafuer direkt
wiederverwendet, nicht neu geschrieben).

    python -m exploration_optimierung.eval_optuna_best
    python -m exploration_optimierung.eval_optuna_best --best results/optuna/best.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'exploration_optimierung'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402

from . import DEFAULT_CKPT, RESULTS_DIR                      # noqa: E402
from . import mission as M                                   # noqa: E402
from . import objective as OBJ                                # noqa: E402
from . import optuna_search as OS                             # noqa: E402
from . import plots as P                                      # noqa: E402

from visualize_checkpoint import WHITE_INFERNO                # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, 'optuna')

#: Welcher Optuna-Parametername der freie Regler eines Modells ist — dieselbe
#: Zuordnung wie in `optuna_search.suggest_config`, hier rueckwaerts, um aus
#: `best.json['params']` wieder ein `cfg`-dict wie zur Suchzeit zu bauen.
_PARAM_KEY = {'ucb': 'kappa', 'eid': 'kappa', 'stretch': 'kappa', 'mi': 'kappa',
             'mass': 'w', 'ei': 'xi', 'niveau': 'tau'}


def cfg_from_params(params, space='ideal'):
    """`best.json['params']` (oder ein beliebiger Optuna-Trial) -> `cfg`-dict,
    formgleich mit dem, was `optuna_search.suggest_config` waehrend der Suche
    gebaut hat. Fehlende Groessen (die in `basis`/`ideal` fest verdrahtet
    waren) kommen aus denselben Konstanten wie dort, nicht neu erfunden.
    """
    phi_model = params['phi_model']
    pkey = _PARAM_KEY[phi_model]
    cfg = dict(
        phi_model=phi_model,
        param=float(params[pkey]),
        svgd_iters=int(params['svgd_iters']),
        debt_weight=float(params['debt_weight']),
        visit_sat=float(params['visit_sat']),
        visit_halflife=float(params['visit_halflife']),
        phi_mode=params['phi_mode'],
        n_particles=int(params['n_particles']),
    )
    cfg['phi_quantile'] = (float(params['phi_quantile'])
                           if cfg['phi_mode'] == 'quantile' else 0.5)
    cfg.update(OS.DEFAULTS_GROSS, **OS.FIXED_BASIS)
    cfg['visit_bandwidth'] = OS.FIXED_ALWAYS['sensor_radius']
    if space in ('ideal', 'gross'):
        cfg.update(
            gp_lengthscale=float(params['gp_lengthscale']),
            gp_noise=float(params['gp_noise']),
            cfg_weight=float(params['cfg_weight']),
        )
    if space == 'gross':
        for k in ('gp_variance', 'gp_res', 'max_obs', 'visit_bandwidth', 'flow_steps'):
            if k in params:
                cfg[k] = params[k]
    cfg['phi_gamma'] = float(params.get('phi_gamma', 1.0))
    return cfg


def rollout(cfg, names, truths, planner, n_max, seed=0, pool=None):
    """Ein vollstaendiger Rollout ueber `n_max` Runden, dieselbe Schleife wie
    `optuna_search.run_trial`, nur ohne Pruning (hier wird ausgewertet, nicht
    gesucht).
    """
    args_ns = OS.build_args(cfg, truths.device)
    planner.cfg_weight = cfg['cfg_weight']
    planner.steps = int(cfg.get('flow_steps', 100))

    m = M.LaengenMission(planner, truths, names, args_ns,
                         svgd_iters=cfg['svgd_iters'], seed=seed, pool=pool,
                         gp_res=int(cfg.get('gp_res', 64)),
                         gp_lengthscale=cfg['gp_lengthscale'],
                         gp_variance=float(cfg.get('gp_variance', 1.0)))
    torch.manual_seed(seed)

    grenzen = [[] for _ in names]
    rows = []
    for r in range(n_max):
        rows += m.round(r, n_max=n_max)
        for row in rows:
            row['plan_s'] = m.plan_s / max(m.S, 1)
            row['svgd_s'] = m.svgd_s / max(m.S, 1)
        for i in range(len(names)):
            grenzen[i].append(m.driven[i].shape[0])
    for row in rows:
        row.update(phi_model=cfg['phi_model'], param=cfg['param'],
                   svgd_iters=int(cfg['svgd_iters']), seed=int(seed))
    return m, rows, grenzen


def draw_panel(m, names, truths, grenzen, n_exec, best, tag):
    """Ein Feld je Holdout-Form — dieselbe Zeichnung wie `plots.panel`, aber
    auf der bereits gefahrenen Mission `m` (die mit der korrekten
    `ideal`-Einstellung gebaut wurde), statt sie mit den falschen,
    unvollstaendigen Argumenten ein zweites Mal zu fahren.
    """
    S = len(names)
    cols = 5
    zeilen = int(np.ceil(S / cols))
    fig, axes = plt.subplots(zeilen, cols, figsize=(2.6 * cols, 2.75 * zeilen),
                             facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    for i, nm in enumerate(names):
        ax = axes[i]
        ax.imshow(truths[i].detach().cpu().numpy(), extent=[0, 1, 0, 1],
                  origin='lower', cmap=WHITE_INFERNO, vmin=0, vmax=1,
                  alpha=0.55, aspect='auto', zorder=0)
        # nur das erste Stueck bis zur gewaehlten Rundenzahl zeichnen.
        cut = grenzen[i][n_exec - 1]
        p = m.driven[i][:cut].detach().cpu().numpy()
        ax.plot(p[:, 0], p[:, 1], color=P.C_GEN, lw=1.8, alpha=0.95, zorder=3)
        gz = [g - 1 for g in grenzen[i][:n_exec - 1]]
        if gz:
            ax.scatter(p[gz, 0], p[gz, 1], s=14, color=P.C_MARK, zorder=4,
                       linewidths=0)
        ax.scatter([p[0, 0]], [p[0, 1]], s=26, marker='o', facecolor='white',
                   edgecolor=P.C_GT, linewidths=1.4, zorder=5)
        row = best['per_shape'].get(nm, {})
        P.style_axes(ax, f"{nm}\nq={row.get('cov_norm', float('nan')):.3f}")
    for ax in axes[S:]:
        ax.axis('off')

    q = float(np.mean([best['per_shape'][n]['cov_norm'] for n in names]))
    pname = _PARAM_KEY[best['phi_model']]
    fig.suptitle(
        f"Optuna-Bestwert ({tag}) — {best['phi_model']}, {pname}={best['param']:.3f}, "
        f"phi_mode={best['phi_mode']} q={best.get('phi_quantile', 0.5):.3f}, "
        f"{best['svgd_iters']} SVGD-Iterationen, {n_exec} Ausfuehrungen "
        f"({n_exec * M.LENGTH_UNIT:.2f} Laengeneinheiten Weg)\n"
        f"gruen: gefahrene Bahn   rot: Ende einer Ausfuehrung (Neuplanung)   "
        f"blau: Start   |   mittleres q = {q:.3f}",
        fontsize=10, color=P.C_DARK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return P.save(fig, f'panel_{tag}.png')


def write_alle_laeufe(rows, path):
    keys = ['phi_model', 'param', 'svgd_iters', 'seed', 'shape', 'n_exec',
           'cov', 'cov_norm', 'erg_truth', 'belief_rmse', 'info_gain',
           'path_len', 'n_obs', 'plan_s', 'svgd_s']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def write_kurven(table, cfg, path):
    keys = ['phi_model', 'param', 'svgd_iters', 'n_exec', 'q', 'J', 'cov_norm',
           'erg_truth', 'time_s']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in table:
            rec = {k: r.get(k) for k in keys}
            rec['phi_model'] = cfg['phi_model']
            rec['param'] = cfg['param']
            rec['svgd_iters'] = cfg['svgd_iters']
            w.writerow(rec)


def vergleichs_balken(vergleich, tag):
    """Gruppiertes Balkendiagramm J/q/erg_truth je Einstellung.

    `vergleich` ist das dict aus `vergleich_<tag>.json`: ein Eintrag
    `optuna_ideal` plus einer je Zeile aus `bestwerte_beide.json`. Die neue
    Einstellung wird in der Projektfarbe fuer generierte/neue Ergebnisse
    hervorgehoben (`plots.C_GEN`), die bisherigen in Grautoenen — so bleibt
    auf einen Blick sichtbar, welche Saeule die neu gefundene ist.
    """
    metriken = ['J', 'q', 'erg_truth']
    namen = list(vergleich)
    farben = [P.C_GEN if n == 'optuna_ideal' else '#9098a8' for n in namen]

    fig, axes = plt.subplots(1, len(metriken), figsize=(3.6 * len(metriken), 3.4),
                             facecolor='white')
    for ax, m in zip(axes, metriken):
        werte = [vergleich[n][m] for n in namen]
        bars = ax.bar(range(len(namen)), werte, color=farben, alpha=0.9)
        for b, v in zip(bars, werte):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                   ha='center', va='bottom', fontsize=8, color=P.C_DARK)
        ax.set_xticks(range(len(namen)))
        ax.set_xticklabels(namen, rotation=25, fontsize=8, ha='right')
        P.style_plot(ax, ylabel=m, title=m)
    fig.suptitle('Optuna-Bestwert (ideal-Raum) gegen die bisherige, von Hand '
                'gesuchte Einstellung — kleiner ist besser, alle bei n=6',
                fontsize=10, color=P.C_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return P.save(fig, f'vergleich_{tag}.png')


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--best', default=os.path.join(OUT_DIR, 'best.json'),
                   help="Bericht der Optuna-Studie (--report in optuna_search.py).")
    p.add_argument('--space', default='ideal', choices=['basis', 'ideal', 'gross'])
    p.add_argument('--tag', default='optuna_ideal_best')
    p.add_argument('--n_max', type=int, default=12,
                   help="Runden fuer die vollstaendige (n,q)-Kurve. Die "
                        "Panel-Abbildung zeigt zusaetzlich die vom Trial "
                        "berichtete Rundenzahl gesondert.")
    p.add_argument('--n_shapes', type=int, default=25)
    p.add_argument('--truth_res', type=int, default=96)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--baseline', default=os.path.join(RESULTS_DIR, 'bestwerte_beide.json'),
                   help="Bisherige, von Hand gesuchte Einstellung zum Vergleich.")
    cfg_cli = p.parse_args(argv)

    device = cfg_cli.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(cfg_cli.best, encoding='utf-8') as f:
        best_report = json.load(f)
    params = best_report['params']
    attrs = best_report.get('attrs', {})
    cfg = cfg_from_params(params, space=cfg_cli.space)

    print(f"\n{'=' * 70}")
    print(f"  Optuna-Bestwert #{best_report['number']}  J={best_report['J']:.4f}  "
         f"(Studie meldete n_exec={attrs.get('n_exec', '?')}, q={attrs.get('q', float('nan')):.4f})")
    for k, v in sorted(cfg.items()):
        print(f"    {k:16s} {v}")
    print(f"{'=' * 70}\n")

    names, truths = M.load_holdout(resolution=cfg_cli.truth_res, device=device,
                                   limit=cfg_cli.n_shapes)
    planner = M.build_planner(ckpt=cfg_cli.ckpt, device=device,
                              flow_steps=int(cfg.get('flow_steps', 100)))
    print(f"  {len(names)} Holdout-Formen geladen, Planer bereit.")

    pool = None
    if cfg['svgd_iters'] > 0:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
        if n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=n_workers,
                                       initializer=M._worker_init,
                                       initargs=(cfg_cli.seed,))
            print(f"  SVGD in {n_workers} Arbeitsprozessen")

    n_max = max(cfg_cli.n_max, int(round(attrs.get('n_exec', 6))) + 1)
    try:
        print(f"  Rollout ueber {n_max} Runden, seed={cfg_cli.seed} ...")
        m, rows, grenzen = rollout(cfg, names, truths, planner, n_max,
                                   seed=cfg_cli.seed, pool=pool)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    best, table = OBJ.score_trace(rows)
    if best is None:
        print("  Rollout lieferte keine auswertbare Spur.")
        return 1
    n_exec = int(best['n_exec'])
    best['phi_model'] = cfg['phi_model']
    best['param'] = cfg['param']
    best['phi_mode'] = cfg['phi_mode']
    best['phi_quantile'] = cfg['phi_quantile']
    best['svgd_iters'] = cfg['svgd_iters']
    best['per_shape'] = {r['shape']: r for r in rows if r['n_exec'] == n_exec}

    print(f"\n  Eigener Rollout (seed={cfg_cli.seed}): beste Rundenzahl n={n_exec}, "
         f"J={best['J']:.4f}, q={best['q']:.4f}, erg_truth={best['erg_truth']:.4f}")

    alle_laeufe_path = os.path.join(OUT_DIR, f'alle_laeufe_{cfg_cli.tag}.csv')
    kurven_path = os.path.join(OUT_DIR, f'kurven_{cfg_cli.tag}.csv')
    write_alle_laeufe(rows, alle_laeufe_path)
    write_kurven(table, cfg, kurven_path)
    print(f"  Geschrieben -> {alle_laeufe_path}")
    print(f"  Geschrieben -> {kurven_path}")

    draw_panel(m, names, truths, grenzen, n_exec, best, cfg_cli.tag)

    ranking_row = dict(best)
    ranking_row['param_name'] = _PARAM_KEY[cfg['phi_model']]
    P.formen_streuung(alle_laeufe_path, ranking_row, cfg_cli.tag)
    P.abdeckungs_kurven(kurven_path, [ranking_row], cfg_cli.tag)

    # ------------------------------------------------------------------
    # Vergleich mit der bisherigen, von Hand gesuchten Einstellung.
    # ------------------------------------------------------------------
    vergleich = {'optuna_ideal': {
        'J': best['J'], 'q': best['q'], 'cov_norm': best['cov_norm'],
        'erg_truth': best['erg_truth'], 'n_exec': n_exec,
        'phi_model': cfg['phi_model'], 'param': cfg['param'],
        'svgd_iters': cfg['svgd_iters'],
    }}
    if os.path.isfile(cfg_cli.baseline):
        with open(cfg_cli.baseline, encoding='utf-8') as f:
            bisherige = json.load(f)
        for rec in bisherige:
            vergleich[f"bisher_{rec['studie']}"] = {
                k: rec[k] for k in ('J', 'q', 'cov_norm', 'erg_truth', 'n_exec',
                                    'phi_model', 'param', 'svgd_iters')}

    vpath = os.path.join(OUT_DIR, f'vergleich_{cfg_cli.tag}.json')
    with open(vpath, 'w', encoding='utf-8') as f:
        json.dump(vergleich, f, indent=2)
    print(f"  Geschrieben -> {vpath}")
    vergleichs_balken(vergleich, cfg_cli.tag)

    print(f"{'Einstellung':<22}{'J':>8}{'q':>8}{'cov_norm':>10}{'erg_truth':>11}{'n':>4}")
    for name, rec in vergleich.items():
        print(f"{name:<22}{rec['J']:>8.4f}{rec['q']:>8.4f}{rec['cov_norm']:>10.4f}"
             f"{rec['erg_truth']:>11.4f}{rec['n_exec']:>4d}")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
