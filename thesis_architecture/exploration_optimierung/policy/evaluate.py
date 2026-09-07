r"""
evaluate.py
===========
Alle Regler auf der **vollen Holdout-Menge**, Bilder *und* Zahlen.

Gefahren werden bis zu vier Regler durch denselben Simulator, mit denselben
Formen, denselben Seeds und derselben Zielfunktion:

    fest      die Betriebseinstellung der Rastersuche (niveau, tau = 0,61,
              25 SVGD-Iterationen) — der Bezugswert, den es zu schlagen gilt.
    gelernt_a das ueberwachte Wertmodell aus `train.py` (Option A).
    rl_b      die PPO-Politik aus `ppo.py` (Option B).
    orakel    die gierige Ein-Runden-Suche (`--mit_orakel`) — teuer, nicht
              einsetzbar, aber die Obergrenze des kurzsichtigen Waehlens.

Warum vier Spalten und nicht zwei
---------------------------------
"Gelernt schlaegt fest" allein sagt wenig: der Abstand koennte fast das ganze
Erreichbare sein oder ein Zehntel davon. Erst mit der Orakel-Spalte steht die
Zahl in einem Rahmen — und erst mit A *und* B nebeneinander laesst sich die
eigentliche Frage beantworten, ob die Vorausschau von PPO ueber das
kurzsichtige Nachahmen hinaus etwas bringt.

Ausgabe
-------
    results/policy_metriken.csv       jede Form, jede Rundenzahl, jeder Regler
    results/policy_vergleich.json     Aggregat: J*, n*, q, cov, E_erg je Regler
    results/policy_panel.png          25 Formen, alle Regler uebereinander
    results/policy_kurven.png         q(n) und J(n) je Regler
    results/policy_aktionen.png       was die gelernten Regler tatsaechlich waehlen

    python -m exploration_optimierung.policy.evaluate
    python -m exploration_optimierung.policy.evaluate --mit_orakel --seeds 2
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402

from .. import DEFAULT_CKPT, RESULTS_DIR                     # noqa: E402
from .. import mission as M                                  # noqa: E402
from .. import objective as O                                # noqa: E402
from ..plots import save, style_axes, style_plot             # noqa: E402
from . import FESTE_POLICY                                   # noqa: E402
from .budget import Zeitbudget                               # noqa: E402
from .features import MODELL_ORDNUNG, aktionsraster, param_norm  # noqa: E402
from .model import FesteRichtlinie, OrakelRichtlinie, WertRichtlinie  # noqa: E402
from .train import MODELL_PT as MODELL_A                     # noqa: E402
from .ppo import MODELL_PT as MODELL_B, RLRichtlinie         # noqa: E402

from visualize_checkpoint import WHITE_INFERNO               # noqa: E402

def metriken_csv(tag='policy'):
    return os.path.join(RESULTS_DIR, f'{tag}_metriken.csv')


def vergleich_json(tag='policy'):
    return os.path.join(RESULTS_DIR, f'{tag}_vergleich.json')

FARBE = {'fest': '#1565C0', 'gelernt_a': '#00C853', 'rl_b': '#D81B60',
         'orakel': '#F9A825'}
TITEL = {'fest': 'fest (Studie)', 'gelernt_a': 'A · Wertmodell',
         'rl_b': 'B · PPO', 'orakel': 'Orakel (Obergrenze)'}

ZEILEN_COLS = ['regler', 'seed', 'shape', 'n_exec', 'cov', 'cov_norm',
               'erg_truth', 'belief_rmse', 'info_gain', 'path_len', 'n_obs',
               'plan_s', 'svgd_s', 'gewaehlt_modell', 'gewaehlt_param',
               'gewaehlt_svgd']


def fahre(planner, truths, names, args, policy, n_max, seed=0, pool=None,
          svgd_iters=0, on_round=None):
    """Eine Mission mit einem Regler. -> (zeilen, mission, rundengrenzen)."""
    m = M.LaengenMission(planner, truths, names, args, svgd_iters=svgd_iters,
                         seed=seed, pool=pool, policy=policy)
    if hasattr(policy, 'mission'):
        policy.mission = m          # die Orakel-Richtlinie braucht den Planer
    torch.manual_seed(seed)
    zeilen, grenzen = [], [[] for _ in names]
    for r in range(n_max):
        zeilen += m.round(r, n_max=n_max)
        for i in range(len(names)):
            grenzen[i].append(m.driven[i].shape[0])
        if on_round is not None:
            on_round(r, n_max)
    for z in zeilen:
        z['plan_s'] = m.plan_s / max(m.S, 1)
        z['svgd_s'] = m.svgd_s / max(m.S, 1)
    return zeilen, m, grenzen


# ---------------------------------------------------------------------------
# Abbildungen
# ---------------------------------------------------------------------------

def panel(names, truths, bahnen, grenzen, q_je_regler, n_exec, tag='policy'):
    """25 Formen, je Feld die Bahnen aller Regler uebereinander."""
    S = len(names)
    cols = 5
    zeilen_n = int(np.ceil(S / cols))
    fig, axes = plt.subplots(zeilen_n, cols,
                             figsize=(2.7 * cols, 2.95 * zeilen_n),
                             facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    for i, nm in enumerate(names):
        ax = axes[i]
        ax.imshow(truths[i].detach().cpu().numpy(), extent=[0, 1, 0, 1],
                  origin='lower', cmap=WHITE_INFERNO, vmin=0, vmax=1,
                  alpha=0.55, aspect='auto', zorder=0)
        teile = []
        for k, (regler, pfade) in enumerate(bahnen.items()):
            p = pfade[i].detach().cpu().numpy()
            # Der erste Regler kraeftig, die weiteren blasser — bei vier
            # Bahnen im selben Feld ist sonst nicht mehr zu sehen, welche
            # welche ist.
            ax.plot(p[:, 0], p[:, 1], color=FARBE[regler], lw=2.0 if k == 0 else 1.5,
                    alpha=0.95 if k == 0 else 0.75, zorder=3 + k)
            gz = [g - 1 for g in grenzen[regler][i][:-1]]
            if gz:
                ax.scatter(p[gz, 0], p[gz, 1], s=9, color=FARBE[regler],
                           zorder=6, linewidths=0)
            teile.append(f"{regler.split('_')[0]} {q_je_regler[regler][i]:.2f}")
        ax.scatter([0.5], [0.5], s=18, marker='+', color='#555', zorder=2)
        style_axes(ax, f"{nm}\nq: " + "  ".join(teile), fontsize=7)
    for ax in axes[S:]:
        ax.axis('off')

    beschriftung = "   ".join(
        f"{TITEL[r]} (q̄={np.mean(q_je_regler[r]):.3f})" for r in bahnen)
    fig.suptitle(
        f"Laengeneinheit-Mission auf der vollen Holdout-Menge, "
        f"{n_exec} Ausfuehrungen — Regler im Vergleich\n{beschriftung}",
        fontsize=10.5, color='#1A1A2E', y=0.996)
    for r in bahnen:
        fig.text(0.5, 0.0, '', color=FARBE[r])       # Farben in der Legende
    handles = [plt.Line2D([], [], color=FARBE[r], lw=2.2, label=TITEL[r])
               for r in bahnen]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=[0, 0.015, 1, 0.965])
    return save(fig, f'{tag}_panel.png')


def kurven(tabellen, tag='policy'):
    """q(n) und J(n) je Regler."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), facecolor='white')
    for regler, tab in tabellen.items():
        n = [t['n_exec'] for t in tab]
        ax1.plot(n, [t['q'] for t in tab], '-o', ms=4, lw=1.8,
                 color=FARBE[regler], label=TITEL[regler])
        ax2.plot(n, [t['J'] for t in tab], '-o', ms=4, lw=1.8,
                 color=FARBE[regler], label=TITEL[regler])
        best = min(tab, key=lambda t: t['J'])
        ax2.plot([best['n_exec']], [best['J']], 'o', ms=10, mfc='none',
                 mec=FARBE[regler], mew=1.8)
    style_plot(ax1, 'Ausfuehrungen n', 'q (bezogener Abdeckungsfehler)',
               'Guete ueber der Missionslaenge')
    style_plot(ax2, 'Ausfuehrungen n', 'J = q + 0,02·n + 0,004·t',
               'Zielfunktion (Ring = Optimum des Reglers)')
    ax1.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    return save(fig, f'{tag}_kurven.png')


def aktionen(gewaehlt, tag='policy'):
    """Was die Regler tatsaechlich waehlen — je Runde Modell und Parameter.

    Die Abbildung beantwortet die Frage, die eine J-Tabelle offen laesst:
    *worin* besteht die gelernte Regel? Eine Richtlinie, die immer dasselbe
    waehlt, ist trotz besserer Zahlen keine kontextabhaengige Richtlinie —
    das waere hier sofort zu sehen.
    """
    regler = [r for r in gewaehlt if gewaehlt[r]]
    if not regler:
        return None
    fig, axes = plt.subplots(1, len(regler),
                             figsize=(4.4 * len(regler), 3.6),
                             facecolor='white', squeeze=False)
    for ax, r in zip(axes[0], regler):
        for modell in MODELL_ORDNUNG:
            xs = [g['runde'] + np.random.uniform(-0.16, 0.16)
                  for g in gewaehlt[r] if g['modell'] == modell]
            ys = [param_norm(modell, g['param'])
                  for g in gewaehlt[r] if g['modell'] == modell]
            if xs:
                ax.scatter(xs, ys, s=14, alpha=0.45, linewidths=0,
                           label=f"{modell} ({len(xs)})")
        style_plot(ax, 'Runde', 'Parameter (normiert)', TITEL[r])
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.tight_layout()
    return save(fig, f'{tag}_aktionen.png')


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n_shapes', type=int, default=25)
    p.add_argument('--n_max', type=int, default=8)
    p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--split', default='val', choices=['val', 'train'])
    p.add_argument('--modell_a', default=MODELL_A)
    p.add_argument('--modell_b', default=MODELL_B)
    p.add_argument('--mit_orakel', action='store_true',
                   help="die teure Obergrenze mitfahren (K-mal so lange)")
    p.add_argument('--param_punkte', type=int, default=4)
    p.add_argument('--svgd_buckets', nargs='*', type=int, default=[0, 25, 100])
    p.add_argument('--flow_steps', type=int, default=100)
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--tag', default='policy')
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--max_minuten', type=float, default=None,
                   help="Zeitbudget. Reicht es nicht mehr fuer den naechsten "
                        "Regler, wird er ausgelassen und die Auswertung mit "
                        "den bereits gefahrenen geschrieben.")
    a = p.parse_args(argv)

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    planner = M.build_planner(ckpt=a.ckpt, device=device,
                              flow_steps=a.flow_steps)
    names, truths = M.load_holdout(resolution=96, device=device,
                                   limit=a.n_shapes, split=a.split)
    args = M.build_mission_args(device, phi_model=FESTE_POLICY['phi_model'],
                                param=FESTE_POLICY['param'])
    print(f"Auswertung auf {len(names)} Formen ({a.split}), "
          f"n_max = {a.n_max}, {a.seeds} Seed(s)  [{device}]")

    regler = {'fest': FesteRichtlinie(FESTE_POLICY['phi_model'],
                                      FESTE_POLICY['param'],
                                      FESTE_POLICY['svgd_iters'])}
    if os.path.exists(a.modell_a):
        regler['gelernt_a'] = WertRichtlinie.laden(a.modell_a, device=device)
        print(f"  Option A geladen: {a.modell_a}")
    else:
        print(f"  Option A uebersprungen (kein Modell unter {a.modell_a})")
    if os.path.exists(a.modell_b):
        regler['rl_b'] = RLRichtlinie.laden(a.modell_b, device=device)
        print(f"  Option B geladen: {a.modell_b}")
    else:
        print(f"  Option B uebersprungen (kein Modell unter {a.modell_b})")
    if a.mit_orakel:
        regler['orakel'] = OrakelRichtlinie(
            aktionsraster(MODELL_ORDNUNG, a.param_punkte, a.svgd_buckets))
        print(f"  Orakel mit {len(regler['orakel'].kandidaten)} Kandidaten "
              "je Entscheidung (teuer)")

    n_workers = a.workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
    pool = (ProcessPoolExecutor(max_workers=n_workers,
                                initializer=M._worker_init, initargs=(0,))
            if n_workers > 1 else None)

    alle_zeilen, bahnen, grenzen_je, tabellen, gewaehlt = [], {}, {}, {}, {}
    aggregat = {}
    budget = Zeitbudget(a.max_minuten, name='Auswertung')
    letzte_dauer = 0.0
    try:
        for name, pol in regler.items():
            if name != 'fest' and budget.abgelaufen(letzte_dauer):
                print(f"  {TITEL[name]} ausgelassen ({budget.grund()})")
                continue
            spuren = []
            t0 = time.perf_counter()
            for seed in range(a.seeds):
                zeilen, m, grenzen = fahre(planner, truths, names, args, pol,
                                           a.n_max, seed=seed, pool=pool)
                for z in zeilen:
                    z['regler'], z['seed'] = name, seed
                spuren += zeilen
                if seed == 0:                      # Bilder aus dem ersten Seed
                    bahnen[name] = [d for d in m.driven]
                    grenzen_je[name] = grenzen
            alle_zeilen += spuren
            best, tab = O.score_trace(spuren)
            tabellen[name] = tab
            gewaehlt[name] = [dict(runde=z['n_exec'] - 1,
                                   modell=z.get('gewaehlt_modell'),
                                   param=z.get('gewaehlt_param'),
                                   svgd=z.get('gewaehlt_svgd'))
                              for z in spuren if z.get('gewaehlt_modell')]
            aggregat[name] = dict(
                J=best['J'], n_exec=best['n_exec'], q=best['q'],
                cov=best['cov'], erg_truth=best['erg_truth'],
                belief_rmse=best['belief_rmse'], path_len=best['path_len'],
                time_s=best['time_s'], sekunden=time.perf_counter() - t0)
            letzte_dauer = time.perf_counter() - t0
            print(f"  {TITEL[name]:<22} J* = {best['J']:.4f} bei n = "
                  f"{best['n_exec']:2d}   q = {best['q']:.4f}   "
                  f"E_erg = {best['erg_truth']:.5f}   "
                  f"[{letzte_dauer / 60:.1f} min]", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    # -- paarweiser Vergleich je Form, bei gleicher Rundenzahl ---------------
    n_ref = aggregat['fest']['n_exec']
    def q_je_form(name):
        out = []
        for nm in names:
            werte = [z['cov_norm'] for z in alle_zeilen
                     if z['regler'] == name and z['shape'] == nm
                     and z['n_exec'] == n_ref]
            out.append(float(np.mean(werte)) if werte else float('nan'))
        return out

    q_form = {r: q_je_form(r) for r in aggregat}
    for name in aggregat:
        if name == 'fest':
            continue
        d = np.asarray(q_form['fest']) - np.asarray(q_form[name])
        aggregat[name]['gegen_fest'] = dict(
            n_exec=n_ref, mittel=float(np.mean(d)), median=float(np.median(d)),
            bester=float(np.max(d)), schlechtester=float(np.min(d)),
            formen_besser=int((d > 0).sum()), formen=len(names))
        # Bewusst ohne Sonderzeichen: die Windows-Konsole des Projekts laeuft
        # auf cp1252 und bricht bei griechischen Buchstaben ab.
        print(f"  {TITEL[name]:<22} gegen fest bei n = {n_ref}: "
              f"q besser um {np.mean(d):+.4f} im Mittel, auf "
              f"{(d > 0).sum()}/{len(names)} Formen")

    # -- die zweite Orakelzahl ----------------------------------------------
    # `orakel` oben waehlt nur die *Einstellung* und laesst die Mission danach
    # neu planen; weil die Flow-ODE stochastisch ist, faehrt sie nicht die
    # geprobte Bahn. Der eigenstaendige Rollout in `oracle.py` committet
    # dagegen genau die Bahn, die er ausprobiert hat — das ist die schaerfere
    # Obergrenze. Beide Zahlen gehoeren nebeneinander, sonst ist die eine
    # geschoent und die andere unlesbar.
    orakel_js = os.path.join(RESULTS_DIR, 'policy_orakel.json')
    if os.path.exists(orakel_js):
        try:
            with open(orakel_js, 'r', encoding='utf-8') as f:
                od = json.load(f)
            aggregat['orakel_rollout'] = dict(
                J=od.get('J'), n_exec=od.get('n_exec'), q=od.get('q'),
                quelle=os.path.basename(orakel_js),
                hinweis=('eigenstaendiger Rollout: committet die geprobte '
                         'Bahn, andere Formenzahl/Rundenzahl moeglich — '
                         'Obergrenze der Guete, kein einsetzbares Verfahren'))
            print(f"  Orakel (Rollout, aus {os.path.basename(orakel_js)}): "
                  f"J* = {od.get('J'):.4f} bei n = {od.get('n_exec')}")
        except (OSError, ValueError, TypeError) as exc:
            print(f"  {orakel_js} nicht lesbar ({exc}) — uebersprungen.")

    # -- Ausgaben ------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(metriken_csv(a.tag), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ZEILEN_COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(alle_zeilen)
    print(f"\n  Gespeichert -> {metriken_csv(a.tag)}  "
          f"({len(alle_zeilen)} Zeilen)")

    with open(vergleich_json(a.tag), 'w', encoding='utf-8') as f:
        json.dump(dict(konfiguration=vars(a), formen=names,
                       aggregat=aggregat, tabellen=tabellen,
                       q_je_form={r: dict(zip(names, v))
                                  for r, v in q_form.items()}), f, indent=2,
                  ensure_ascii=False, default=str)
    print(f"  Gespeichert -> {vergleich_json(a.tag)}")

    panel(names, truths, bahnen, grenzen_je, q_form, n_ref, tag=a.tag)
    kurven(tabellen, tag=a.tag)
    aktionen(gewaehlt, tag=a.tag)
    return aggregat


if __name__ == '__main__':
    main()
