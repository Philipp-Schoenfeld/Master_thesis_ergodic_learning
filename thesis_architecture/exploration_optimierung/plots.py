#!/usr/bin/env python3
r"""
plots.py
========
Die Abbildungen zur Suche — Bilder *und* Zahlen ueber die ganze Holdout-Menge.

Drei Abbildungen, weil drei verschiedene Fragen offen sind:

1. **`panel_<tag>.png`** — die beste Einstellung, gefahren auf allen 25
   Holdout-Formen. Ein Feld je Form: die wahre Dichte blass im Hintergrund,
   darueber die gefahrene Bahn, die Rundengrenzen als Punkte. Das ist die
   Abbildung, die zeigt, ob eine gute Kennzahl auch eine gute Bahn ist — eine
   Mittelwertstabelle allein kann eine Einstellung gut aussehen lassen, die auf
   der Haelfte der Formen ins Leere faehrt.
2. **`parameter_<tag>.png`** — J ueber dem freien Parameter, eine Kurve je
   Zieldichte-Modell. Zeigt, ob das Optimum ein flaches Plateau ist (dann ist
   der genaue Wert gleichgueltig) oder eine schmale Spitze (dann ist er
   wichtig, und ein Raster mit sieben Punkten ist womoeglich zu grob).
3. **`abdeckung_<tag>.png`** — der bezogene Abdeckungsfehler q ueber der Zahl
   der Ausfuehrungen, je Modell die beste Einstellung. Das ist die eigentliche
   Antwort auf "wie viele Ausfuehrungen brauche ich": man liest die Stelle ab,
   an der die Kurve abknickt, unabhaengig davon, ob man `lambda_len` teilt.

    python -m exploration_optimierung.plots
    python -m exploration_optimierung.plots --tag ohne_svgd
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
import sys

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'exploration_optimierung'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402

from . import DEFAULT_CKPT, RESULTS_DIR          # noqa: E402
from . import mission as M                       # noqa: E402
from . import objective as O                     # noqa: E402

from visualize_checkpoint import WHITE_INFERNO   # noqa: E402

# Projektpalette, siehe CLAUDE.md
C_GEN = '#00C853'      # gefahrene Bahn
C_GT = '#1565C0'
C_MARK = '#D81B60'
C_DARK = '#1A1A2E'
C_GREY = '#555'
MODELL_FARBE = {'ucb': '#1565C0', 'eid': '#00C853',
                'mass': '#D81B60', 'niveau': '#F9A825'}


def style_axes(ax, title=None, fontsize=8):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=fontsize, color=C_DARK, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')


def style_plot(ax, xlabel='', ylabel='', title=''):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2, lw=0.5, color='gray')
    ax.set_xlabel(xlabel, fontsize=9, color=C_GREY)
    ax.set_ylabel(ylabel, fontsize=9, color=C_GREY)
    if title:
        ax.set_title(title, fontsize=10, color=C_DARK)
    ax.tick_params(labelsize=8, colors=C_GREY)
    for s in ax.spines.values():
        s.set_color('#ccc')


def save(fig, name, dpi=150):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Gespeichert -> {path}")
    return path


# ---------------------------------------------------------------------------

def panel(best, cfg, tag):
    """Die beste Einstellung auf allen Holdout-Formen, ein Feld je Form.

    Der Rollout wird dafuer noch einmal gefahren — die Suche legt nur die
    Kennzahlen ab, nicht die Bahnen. Das kostet eine einzige Auswertung
    (rund zwei Minuten ohne SVGD) und ist der Preis dafuer, dass der
    Zwischenspeicher schlank bleibt.
    """
    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    names, truths = M.load_holdout(resolution=cfg.truth_res, device=device,
                                   limit=cfg.n_shapes)
    planner = M.build_planner(ckpt=cfg.ckpt, device=device,
                              flow_steps=cfg.flow_steps)

    n_exec = int(best['n_exec'])
    print(f"  Fahre die beste Einstellung nach: {best['phi_model']} "
          f"{best['param_name']}={best['param']:.3f}, "
          f"{best['svgd_iters']} SVGD-Iterationen, {n_exec} Ausfuehrungen ...")
    args = M.build_mission_args(device, best['phi_model'], best['param'],
                                debt_weight=cfg.debt_weight,
                                visit_sat=cfg.visit_sat,
                                visit_halflife=cfg.visit_halflife,
                                sensor_radius=cfg.sensor_radius,
                                gp_noise=cfg.gp_noise,
                                n_particles=cfg.n_particles,
                                meas_noise=cfg.meas_noise, max_obs=cfg.max_obs)
    # Derselbe Prozesspool wie in der Suche. Ohne ihn liefe die Verfeinerung
    # hier seriell: bei 400 Iterationen ueber 25 Formen und acht Runden sind
    # das rund 30 Minuten statt drei — und zwar am Ende eines mehrstuendigen
    # Laufs, also genau dort, wo es am meisten stoert.
    pool = None
    if int(best['svgd_iters']) > 0:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
        if n_workers > 1:
            pool = ProcessPoolExecutor(max_workers=n_workers,
                                       initializer=M._worker_init,
                                       initargs=(0,))
            print(f"  SVGD in {n_workers} Arbeitsprozessen")
    try:
        m = M.LaengenMission(planner, truths, names, args,
                             svgd_iters=int(best['svgd_iters']), seed=0,
                             pool=pool)
        # Rundengrenzen mitschreiben, damit im Bild sichtbar wird, wo eine
        # Ausfuehrung endet und neu geplant wurde.
        grenzen = [[] for _ in names]
        rows = []
        for r in range(n_exec):
            rows += m.round(r)
            for i in range(len(names)):
                grenzen[i].append(m.driven[i].shape[0])
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    S = len(names)
    cols = 5
    zeilen = int(np.ceil(S / cols))
    fig, axes = plt.subplots(zeilen, cols, figsize=(2.6 * cols, 2.75 * zeilen),
                             facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    per_shape = {r['shape']: r for r in rows if r['n_exec'] == n_exec}
    for i, nm in enumerate(names):
        ax = axes[i]
        ax.imshow(truths[i].detach().cpu().numpy(), extent=[0, 1, 0, 1],
                  origin='lower', cmap=WHITE_INFERNO, vmin=0, vmax=1,
                  alpha=0.55, aspect='auto', zorder=0)
        p = m.driven[i].detach().cpu().numpy()
        ax.plot(p[:, 0], p[:, 1], color=C_GEN, lw=1.8, alpha=0.95, zorder=3)
        gz = [g - 1 for g in grenzen[i][:-1]]
        if gz:
            ax.scatter(p[gz, 0], p[gz, 1], s=14, color=C_MARK, zorder=4,
                       linewidths=0)
        ax.scatter([p[0, 0]], [p[0, 1]], s=26, marker='o', facecolor='white',
                   edgecolor=C_GT, linewidths=1.4, zorder=5)
        row = per_shape.get(nm, {})
        style_axes(ax, f"{nm}\nq={row.get('cov_norm', float('nan')):.3f}")
    for ax in axes[S:]:
        ax.axis('off')

    q = float(np.mean([per_shape[n]['cov_norm'] for n in names]))
    fig.suptitle(
        f"Laengeneinheit-Mission auf der Holdout-Menge — {best['phi_model']}, "
        f"{best['param_name']}={best['param']:.3f}, "
        f"{best['svgd_iters']} SVGD-Iterationen, {n_exec} Ausfuehrungen "
        f"({n_exec * M.LENGTH_UNIT:.2f} Laengeneinheiten Weg)\n"
        f"gruen: gefahrene Bahn   rot: Ende einer Ausfuehrung (Neuplanung)   "
        f"blau: Start   |   mittleres q = {q:.3f}",
        fontsize=10, color=C_DARK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return save(fig, f'panel_{tag}.png'), rows


def parameter_kurven(rows, tag):
    """J ueber dem freien Parameter, je Modell eine Kurve."""
    nach_modell = {}
    for r in rows:
        nach_modell.setdefault(r['phi_model'], []).append(r)
    if not nach_modell:
        return None

    fig, axes = plt.subplots(1, len(nach_modell),
                             figsize=(3.6 * len(nach_modell), 3.2),
                             facecolor='white', squeeze=False)
    for ax, (mod, rs) in zip(axes[0], sorted(nach_modell.items())):
        # je Parameterwert die beste Iterationszahl
        best = {}
        for r in rs:
            p = float(r['param'])
            if p not in best or float(r['J']) < float(best[p]['J']):
                best[p] = r
        xs = sorted(best)
        ys = [float(best[x]['J']) for x in xs]
        ns = [int(best[x]['n_exec']) for x in xs]
        farbe = MODELL_FARBE.get(mod, C_GEN)
        ax.plot(xs, ys, '-o', color=farbe, lw=2.0, ms=5, alpha=0.95)
        k = int(np.argmin(ys))
        ax.scatter([xs[k]], [ys[k]], s=110, facecolor='none', edgecolor=C_MARK,
                   linewidths=2.0, zorder=5)
        ax.annotate(f"{xs[k]:.3g}\nn={ns[k]}", (xs[k], ys[k]),
                    textcoords='offset points', xytext=(8, 8), fontsize=8,
                    color=C_MARK)
        pname = M.PHI_MODELS[mod][0]
        if pname == 'kappa':
            ax.set_xscale('log')
        style_plot(ax, xlabel=pname, ylabel='J (kleiner ist besser)',
                   title=mod)
    fig.suptitle('Zielfunktion ueber dem freien Parameter — Kreis: das gewaehlte Optimum',
                 fontsize=10, color=C_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save(fig, f'parameter_{tag}.png')


def abdeckungs_kurven(kurven_csv, ranked, tag):
    """q ueber der Zahl der Ausfuehrungen, je Modell die beste Einstellung."""
    if not os.path.isfile(kurven_csv):
        return None
    daten = {}
    with open(kurven_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            k = (row['phi_model'], row['param'], row['svgd_iters'])
            daten.setdefault(k, []).append(row)

    beste = {}
    for r in ranked:
        if r['phi_model'] not in beste:
            beste[r['phi_model']] = (r['phi_model'], str(r['param']),
                                     str(r['svgd_iters']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor='white')
    for mod, key in sorted(beste.items()):
        rs = sorted(daten.get(key, []), key=lambda r: int(r['n_exec']))
        if not rs:
            continue
        n = [int(r['n_exec']) for r in rs]
        q = [float(r['q']) for r in rs]
        J = [float(r['J']) for r in rs]
        farbe = MODELL_FARBE.get(mod, C_GEN)
        pname = M.PHI_MODELS[mod][0]
        lbl = f"{mod} ({pname}={float(key[1]):.3g}, svgd={key[2]})"
        ax1.plot(n, q, '-o', color=farbe, lw=2.0, ms=4, alpha=0.95, label=lbl)
        ax2.plot(n, J, '-o', color=farbe, lw=2.0, ms=4, alpha=0.95, label=lbl)
        k = int(np.argmin(J))
        ax2.scatter([n[k]], [J[k]], s=110, facecolor='none', edgecolor=C_MARK,
                    linewidths=2.0, zorder=5)

    style_plot(ax1, xlabel='Ausfuehrungen n (je eine Laengeneinheit)',
               ylabel='q — Restanteil des Abdeckungsfehlers',
               title='Abdeckung ueber der gefahrenen Strecke')
    style_plot(ax2, xlabel='Ausfuehrungen n',
               ylabel='J = q + λ·n (+ λ_t·t)',
               title='Zielfunktion — Kreis: gewaehlte Rundenzahl')
    ax1.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return save(fig, f'abdeckung_{tag}.png')


# ---------------------------------------------------------------------------

def waermekarte(rows, tag):
    r"""J ueber (Parameter x SVGD-Iterationen), eine Karte je Zieldichte-Modell.

    Das ist die Abbildung, fuer die es das volle Kreuzprodukt gibt. Sie
    beantwortet die Frage, die eine Parameterkurve allein offen laesst:
    **verschiebt sich das Optimum des Parameters, wenn mehr verfeinert wird?**
    Faellt der beste Wert bei 400 Iterationen auf ein anderes kappa als bei 0,
    dann ersetzt SVGD einen Teil dessen, was sonst die Zieldichte leisten muss —
    und die beiden Regler sind nicht unabhaengig einstellbar.

    Nach einem Lauf mit `--search abstieg` ist die Matrix luecken haft (der
    Abstieg misst die Iterationszahl nur bei einem Parameterwert). Die Luecken
    bleiben als graue Felder stehen, statt weggerechnet zu werden — sie sind
    eine ehrliche Auskunft darueber, was gemessen wurde und was nicht.
    """
    nach_modell = {}
    for r in rows:
        nach_modell.setdefault(r['phi_model'], []).append(r)
    if not nach_modell:
        return None

    fig, axes = plt.subplots(1, len(nach_modell),
                             figsize=(3.9 * len(nach_modell), 3.6),
                             facecolor='white', squeeze=False)
    for ax, (mod, rs) in zip(axes[0], sorted(nach_modell.items())):
        ps = sorted({float(r['param']) for r in rs})
        its = sorted({int(r['svgd_iters']) for r in rs})
        Z = np.full((len(its), len(ps)), np.nan)
        for r in rs:
            i, j = its.index(int(r['svgd_iters'])), ps.index(float(r['param']))
            v = float(r['J'])
            if np.isnan(Z[i, j]) or v < Z[i, j]:      # ueber Seeds das beste
                Z[i, j] = v

        cmap = plt.get_cmap('viridis_r').copy()
        cmap.set_bad('#e8e8e8')                        # nicht gemessen
        im = ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, aspect='auto',
                       origin='lower', interpolation='nearest')
        if np.isfinite(Z).any():
            i, j = np.unravel_index(np.nanargmin(Z), Z.shape)
            ax.scatter([j], [i], s=150, facecolor='none', edgecolor=C_MARK,
                       linewidths=2.2, zorder=5)
        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels([f"{p:.3g}" for p in ps], fontsize=7, rotation=60)
        ax.set_yticks(range(len(its)))
        ax.set_yticklabels([str(i) for i in its], fontsize=7)
        pname = M.PHI_MODELS[mod][0]
        style_plot(ax, xlabel=pname, ylabel='SVGD-Iterationen',
                   title=f"{mod} — J")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            labelsize=7, colors=C_GREY)

    fig.suptitle('Wirkung der beiden Regler aufeinander — J, klein ist besser '
                 '(Kreis: bestes Feld, grau: nicht gemessen)',
                 fontsize=10, color=C_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return save(fig, f'waermekarte_{tag}.png')


def formen_streuung(pfad, best, tag):
    r"""Der bezogene Abdeckungsfehler je einzelner Holdout-Form.

    Ein Mittelwert von q = 0,2 kann aus 25 gleich guten Bahnen entstehen oder
    aus 20 sehr guten und 5 gescheiterten. Fuer die Frage, ob eine Einstellung
    *taugt*, ist das der entscheidende Unterschied, und er ist in keiner der
    gemittelten Tabellen zu sehen. Deshalb hier je Form ein Balken, dazu der
    Verlauf ueber die Ausfuehrungen als duenne Linien.
    """
    if not os.path.isfile(pfad):
        return None
    with open(pfad, encoding='utf-8') as f:
        alle = list(csv.DictReader(f))
    if not alle:
        return None

    p, it = float(best['param']), int(best['svgd_iters'])
    mine = [r for r in alle
            if r['phi_model'] == best['phi_model']
            and abs(float(r['param']) - p) < 1e-6
            and int(r['svgd_iters']) == it]
    if not mine:
        return None
    n_best = int(best['n_exec'])

    je_form = {}
    for r in mine:
        je_form.setdefault(r['shape'], {})[int(r['n_exec'])] = float(r['cov_norm'])
    namen = sorted(je_form, key=lambda s: je_form[s].get(n_best, 9.9))
    werte = [je_form[s].get(n_best, float('nan')) for s in namen]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.2), facecolor='white')
    mittel = float(np.nanmean(werte))
    farben = [C_MARK if v > mittel else C_GEN for v in werte]
    ax1.bar(range(len(namen)), werte, color=farben, alpha=0.9)
    ax1.axhline(mittel, color=C_GT, lw=1.6, ls='--',
                label=f'Mittel q = {mittel:.3f}')
    ax1.set_xticks(range(len(namen)))
    ax1.set_xticklabels(namen, rotation=70, fontsize=7, ha='right')
    style_plot(ax1, ylabel='q je Form',
               title=f'Streuung ueber die Holdout-Menge bei n = {n_best}')
    ax1.legend(fontsize=8, framealpha=0.9)

    for s in namen:
        ns = sorted(je_form[s])
        ax2.plot(ns, [je_form[s][n] for n in ns], '-', color=C_GEN, lw=1.0,
                 alpha=0.35)
    ns = sorted({n for s in namen for n in je_form[s]})
    mit = [float(np.nanmean([je_form[s].get(n, np.nan) for s in namen]))
           for n in ns]
    ax2.plot(ns, mit, '-o', color=C_GT, lw=2.4, ms=5, label='Mittel')
    ax2.axvline(n_best, color=C_MARK, lw=1.4, ls='--',
                label=f'gewaehlt: n = {n_best}')
    style_plot(ax2, xlabel='Ausfuehrungen n', ylabel='q',
               title='Verlauf je Form (duenn) und im Mittel (dick)')
    ax2.legend(fontsize=8, framealpha=0.9)

    pname = M.PHI_MODELS[best['phi_model']][0]
    fig.suptitle(f"Beste Einstellung: {best['phi_model']}, {pname}={p:.3g}, "
                 f"{it} SVGD-Iterationen", fontsize=10, color=C_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return save(fig, f'formen_{tag}.png')


def lade_ranking(tag):
    path = os.path.join(RESULTS_DIR, f'suche_{tag}.csv')
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('param', 'J', 'q', 'cov', 'cov_norm', 'erg_truth',
                  'belief_rmse', 'info_gain', 'path_len', 'time_s'):
            if r.get(k) not in (None, ''):
                r[k] = float(r[k])
        for k in ('svgd_iters', 'n_exec', 'seed'):
            if r.get(k) not in (None, ''):
                r[k] = int(r[k])
        r['param_name'] = M.PHI_MODELS[r['phi_model']][0]
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tag', default=None,
                   help='Welche Suche gezeichnet wird. Ohne Angabe alles, '
                        'was in results/ liegt.')
    p.add_argument('--kein_panel', action='store_true',
                   help='Die Panel-Abbildung ueberspringen (sie faehrt die '
                        'beste Einstellung noch einmal).')
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--n_shapes', type=int, default=25)
    p.add_argument('--truth_res', type=int, default=96)
    p.add_argument('--flow_steps', type=int, default=100)
    p.add_argument('--debt_weight', type=float, default=0.6)
    p.add_argument('--visit_sat', type=float, default=1.0)
    p.add_argument('--visit_halflife', type=float, default=3.0)
    p.add_argument('--sensor_radius', type=float, default=0.06)
    p.add_argument('--gp_noise', type=float, default=0.05)
    p.add_argument('--meas_noise', type=float, default=0.02)
    p.add_argument('--n_particles', type=int, default=256)
    p.add_argument('--max_obs', type=int, default=64)
    cfg = p.parse_args(argv)

    tags = ([cfg.tag] if cfg.tag else
            [t for t in ('ohne_svgd', 'mit_svgd')
             if os.path.isfile(os.path.join(RESULTS_DIR, f'suche_{t}.csv'))])
    if not tags:
        print("  Keine Suchergebnisse in results/. Erst optimize.py laufen "
              "lassen.")
        return 1

    for tag in tags:
        ranked = lade_ranking(tag)
        if not ranked:
            print(f"  suche_{tag}.csv fehlt oder ist leer — uebersprungen.")
            continue
        print(f"\n=== Abbildungen fuer '{tag}' ({len(ranked)} Einstellungen) ===")
        parameter_kurven(ranked, tag)
        abdeckungs_kurven(os.path.join(RESULTS_DIR, f'kurven_{tag}.csv'),
                          ranked, tag)
        waermekarte(ranked, tag)
        formen_streuung(os.path.join(RESULTS_DIR, f'alle_laeufe_{tag}.csv'),
                        ranked[0], tag)
        if not cfg.kein_panel:
            panel(ranked[0], cfg, tag)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
