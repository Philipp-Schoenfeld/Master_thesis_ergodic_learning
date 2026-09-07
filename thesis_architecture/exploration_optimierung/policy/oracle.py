r"""
oracle.py
=========
Das gierige Orakel: probiert jede Runde **jeden** Kandidaten wirklich aus und
faehrt den besten. Zwei Aufgaben in einem Lauf.

1. **Obergrenze.** Ein Regler, der die Einstellung je Runde waehlt, kann
   hoechstens so gut sein wie einer, der jede Wahl vorher ausprobiert. Ohne
   diese Zahl ist nicht zu beurteilen, ob eine gelernte Richtlinie viel oder
   wenig vom Erreichbaren holt — der Vergleich mit der festen Einstellung
   allein sagt nur, ob sie *besser als nichts* ist. (Die Grenze ist die des
   *kurzsichtigen* Waehlens: das Orakel sieht eine Runde weit, nicht bis zum
   Missionsende. Ein vorausschauender Regler koennte sie im Prinzip
   ueberbieten — das zu pruefen ist gerade die Aufgabe von Option B.)
2. **Trainingsdatensatz.** Jede Kandidatenbewertung ist ein Datenpunkt
   `(Zustand, Aktion) -> Guete`. Aus einem einzigen Rollout fallen damit
   `Formen x Runden x Kandidaten` Zeilen ab, nicht nur `Formen x Runden`
   Entscheidungen. Genau darauf trainiert Option A ihr Wertmodell (und Option
   B ihr Verhaltensklonen), ohne dass dafuer ein zweiter teurer Lauf noetig
   waere.

Auswahl nur gegen den Glauben
-----------------------------
Bewertet wird `coverage_vs_truth(bahn, mu_hat)` — die Abdeckung gegen den
**GP-Mittelwert**, nicht gegen die Wahrheit. Nach der Wahrheit auszuwaehlen
waere ein Blick auf das Ergebnis und wuerde sowohl die Obergrenze als auch den
Datensatz unbrauchbar machen: ein daraus gelernter Regler bekaeme in der
Anwendung eine Eingabe, die es dort nicht gibt. Dieselbe Trennung wie in
`greedy_per_round.run_greedy` und `apply_cfm_belief.best_candidate`. Die
Wahrheit kommt erst danach, rein zur Auswertung der gefahrenen Bahn.

Zur Zeitspalte
--------------
Das Orakel plant `K`-mal so viel, wie es faehrt. In `plan_s`/`svgd_s` steht
trotzdem nur der Aufwand des **committeten** Kandidaten (Gesamtzeit geteilt
durch K), weil die Spalte sonst den Zeitterm von J vollstaendig dominieren
wuerde und die Zahl als Obergrenze der *Guete* unlesbar waere. Das Orakel ist
kein einsetzbares Verfahren, sondern ein Massstab — seine echten Suchkosten
stehen separat als `wallclock_s` in der JSON-Zusammenfassung.

    python -m exploration_optimierung.policy.oracle --n_shapes 25 --seeds 2
    python -m exploration_optimierung.policy.oracle --n_shapes 3 --n_max 4 --schnell
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .. import DEFAULT_CKPT, RESULTS_DIR
from .. import mission as M
from .. import objective as O
from . import DATASET_CSV, FESTE_POLICY
from .budget import Zeitbudget
from .features import (MODELL_ORDNUNG, ZUSTANDS_MERKMALE, aktionsraster,
                       zustands_merkmale)

import apply_cfm_belief as acb                                   # noqa: E402
from common.metrics import coverage_vs_truth                     # noqa: E402
from common.observation import measure, thin                     # noqa: E402


DATASET_COLS = (['seed', 'shape', 'runde', 'n_max', 'kandidat',
                 'modell', 'param', 'svgd']
                + ZUSTANDS_MERKMALE
                + ['score_roh', 'score_norm', 'ist_bester'])


def _plane_gebatcht(planner, parts, starts, chunk):
    """Bahnen fuer viele Wolken auf einmal — in Bloecken von `chunk`.

    Das Orakel bewertet `Formen x Kandidaten` Wolken je Runde (bei 25 Formen
    und 48 Kandidaten also 1200). Alle auf einmal durch das Netz zu schicken
    spart die Aufrufe, sprengt aber auf einer 8-GB-Karte den Speicher; ein
    Aufruf je Form waere umgekehrt 25-mal langsamer als noetig. Bloecke sind
    der Mittelweg, `--plan_batch` stellt sie ein.
    """
    out = []
    for a in range(0, parts.shape[0], chunk):
        b = min(a + chunk, parts.shape[0])
        st = None if starts is None else starts[a:b]
        with torch.no_grad():
            cps = planner.plan(parts[a:b], n_candidates=b - a, start=st)
            out.append(planner.render(cps))
    return torch.cat(out, dim=0)


def bewerte_kandidaten(mission, zustaende, kandidaten, plan_batch=128):
    """Alle Kandidaten fuer alle Formen einer Runde durchspielen.

    -> (segmente, scores): `segmente[i][k]` ist das Stueck, das Form i unter
    Kandidat k fahren wuerde, `scores[i][k]` seine Abdeckung gegen den
    Glauben (kleiner ist besser).

    Die Funktion ist die gemeinsame Grundlage des Orakels und des
    `--wide`-Zweigs von `greedy_per_round.py`: dort wird sie zur Auswahl
    benutzt, hier zusaetzlich zum Mitschreiben aller Bewertungen.
    """
    S, K = len(zustaende), len(kandidaten)
    device = mission.device

    phis, parts = [], []
    for z in zustaende:
        for aktion in kandidaten:
            a = mission._args_fuer(aktion)
            phi, _ = acb.debt_density(z.mu, z.sd, z.visit, a.kappa, a)
            phis.append(phi)
            parts.append(acb.phi_particles(phi, a.n_particles,
                                           mode=a.phi_mode, device=str(device)))
    parts = torch.stack(parts)

    starts = None
    if any(z.driven is not None for z in zustaende):
        starts = torch.stack([z.position() for z in zustaende
                              for _ in range(K)])

    t0 = time.perf_counter()
    curves = _plane_gebatcht(mission.planner, parts, starts, plan_batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    mission.plan_s += (time.perf_counter() - t0) / max(K, 1)

    # SVGD nach Iterationszahl gruppiert — `refine_batch` kennt nur eine
    # gemeinsame Zahl, der Kandidatenraum aber mehrere.
    t0 = time.perf_counter()
    refined = [None] * (S * K)
    for n_iters in sorted(set(int(k[2]) for k in kandidaten)):
        idx = [i * K + k for i in range(S)
               for k in range(K) if int(kandidaten[k][2]) == n_iters]
        teil = M.refine_batch(curves[idx], [phis[j] for j in idx], n_iters,
                              nxi=mission.nxi_refine, pool=mission.pool)
        for j, pos in enumerate(idx):
            refined[pos] = teil[j]
    mission.svgd_s += (time.perf_counter() - t0) / max(K, 1)

    segmente, scores = [], []
    for i, z in enumerate(zustaende):
        mh = z.mu_hat()
        segs_i, sc_i = [], []
        for k in range(K):
            curve = torch.as_tensor(refined[i * K + k], device=device,
                                    dtype=torch.float32).clamp(0.0, 1.0)
            seg = mission._execute_one_unit(curve, i)
            voll = (seg if z.driven is None
                    else torch.cat([z.driven, seg], dim=0))
            segs_i.append(seg)
            sc_i.append(float(coverage_vs_truth(voll, mh)))
        segmente.append(segs_i)
        scores.append(sc_i)
    return segmente, scores


def orakel_rollout(planner, truths, names, args, kandidaten, n_max, seed=0,
                   pool=None, plan_batch=128, sammler=None, on_round=None,
                   budget=None):
    """Eine volle Mission, in der jede Runde der beste Kandidat committet wird.

    `sammler` bekommt je Entscheidung die Datensatzzeilen; ohne ihn laeuft nur
    der Rollout (fuer die reine Obergrenze). `budget` ist ein
    `budget.Zeitbudget`: laeuft es ab, endet die Mission nach der laufenden
    Runde statt mittendrin — die Spur ist dann kuerzer, aber vollstaendig
    auswertbar.
    -> (zeilen, gewaehlt, mission).
    """
    m = M.LaengenMission(planner, truths, names, args, svgd_iters=0, seed=seed,
                         pool=pool)
    torch.manual_seed(seed)

    gewaehlt = {n: [] for n in names}
    zeilen = []
    runden_sek = 0.0
    for r in range(n_max):
        if budget is not None and budget.abgelaufen(runden_sek):
            print(f"    Abbruch nach {r} von {n_max} Runden "
                  f"({budget.grund()}) — geschrieben wird, was vorliegt.",
                  flush=True)
            break
        t_runde = time.perf_counter()
        felder = [m._felder(i) for i in range(m.S)]
        zustaende = m.zustaende(r, n_max, felder=felder)
        segmente, scores = bewerte_kandidaten(m, zustaende, kandidaten,
                                              plan_batch=plan_batch)

        for i, z in enumerate(zustaende):
            sc = np.asarray(scores[i], dtype=np.float64)
            bester = int(sc.argmin())
            spanne = float(sc.max() - sc.min())
            if sammler is not None:
                merk = zustands_merkmale(z)
                for k, (modell, param, svgd) in enumerate(kandidaten):
                    zeile = dict(zip(ZUSTANDS_MERKMALE, merk.tolist()))
                    zeile.update(
                        seed=seed, shape=names[i], runde=r, n_max=n_max,
                        kandidat=k, modell=modell, param=float(param),
                        svgd=int(svgd), score_roh=float(sc[k]),
                        # Auf die Spanne der Entscheidung bezogen: 0 = bester
                        # Kandidat dieser Runde, 1 = schlechtester. Ohne den
                        # Bezug lernte das Modell vor allem, wie schwer die
                        # Form ist, statt welche Einstellung in einer Lage die
                        # richtige ist — die Niveaus verschieben sich zwischen
                        # Formen und Runden um ein Vielfaches der Unterschiede
                        # *innerhalb* einer Entscheidung.
                        score_norm=float((sc[k] - sc.min()) / max(spanne, 1e-12)),
                        ist_bester=int(k == bester))
                    sammler.append(zeile)

            aktion = kandidaten[bester]
            seg = segmente[i][bester]
            a = m._args_fuer(aktion)
            pts, vals = measure(seg, truths[i], noise_std=a.noise,
                                sensor_radius=a.sensor_radius)
            m.beliefs[i].observe(*thin(pts, vals, max_points=a.max_obs))
            m.driven[i] = (seg if m.driven[i] is None
                           else torch.cat([m.driven[i], seg], dim=0))
            gewaehlt[names[i]].append(aktion)
            zeilen.append(m._row(i, r, aktion))
        runden_sek = time.perf_counter() - t_runde
        if on_round is not None:
            on_round(r, n_max)

    for row in zeilen:
        row['plan_s'] = m.plan_s / max(m.S, 1)
        row['svgd_s'] = m.svgd_s / max(m.S, 1)
    return zeilen, gewaehlt, m


def schreibe_datensatz(zeilen, pfad=DATASET_CSV, anhaengen=False):
    os.makedirs(os.path.dirname(pfad), exist_ok=True)
    neu = anhaengen and os.path.exists(pfad)
    with open(pfad, 'a' if neu else 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=DATASET_COLS, extrasaction='ignore')
        if not neu:
            w.writeheader()
        w.writerows(zeilen)
    return pfad


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n_shapes', type=int, default=25)
    p.add_argument('--n_max', type=int, default=12)
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--models', nargs='*', default=MODELL_ORDNUNG,
                   help="Phi-Modelle im Kandidatenraum (A-breit: alle vier)")
    p.add_argument('--param_punkte', type=int, default=4,
                   help="Parameterwerte je Modell im Kandidatenraum")
    p.add_argument('--svgd_buckets', nargs='*', type=int, default=[0, 25, 100],
                   help="SVGD-Budgets im Kandidatenraum")
    p.add_argument('--plan_batch', type=int, default=128)
    p.add_argument('--split', default='val', choices=['val', 'train'],
                   help="'train': Datensatz auf den Trainingsformen des "
                        "Planernetzes erzeugen und die 25 Validierungsformen "
                        "ausschliesslich zum Testen behalten (siehe README, "
                        "Abschnitt zur Aufteilung der Formen)")
    p.add_argument('--flow_steps', type=int, default=100)
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--out', default=DATASET_CSV)
    p.add_argument('--workers', type=int, default=None,
                   help="Prozesse fuer SVGD; 0 = seriell")
    p.add_argument('--max_minuten', type=float, default=None,
                   help="Zeitbudget. Laeuft es ab (oder kommt SIGTERM), endet "
                        "der Lauf nach der laufenden Runde und schreibt, was "
                        "vorliegt — statt nach Stunden ohne Ergebnis "
                        "abgeschnitten zu werden.")
    p.add_argument('--schnell', action='store_true',
                   help="kleiner Kandidatenraum und weniger Flow-Schritte — "
                        "nur zum Durchtesten der Kette, nicht fuer Zahlen")
    a = p.parse_args(argv)

    if a.schnell:
        a.param_punkte = 2
        a.svgd_buckets = [0, 25]
        a.flow_steps = 40

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    kandidaten = aktionsraster(a.models, a.param_punkte, a.svgd_buckets)
    print(f"Device: {device}   Kandidaten/Runde: {len(kandidaten)} "
          f"({len(a.models)} Modelle x {a.param_punkte} Parameter x "
          f"{len(a.svgd_buckets)} SVGD-Budgets)")

    planner = M.build_planner(ckpt=a.ckpt, device=device,
                              flow_steps=a.flow_steps)
    names, truths = M.load_holdout(resolution=96, device=device,
                                   limit=a.n_shapes, split=a.split)
    print(f"Formen ({len(names)}, Split '{a.split}'): {', '.join(names)}")

    args = M.build_mission_args(device, phi_model=FESTE_POLICY['phi_model'],
                                param=FESTE_POLICY['param'])

    n_workers = a.workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
    pool = None
    if n_workers > 1 and any(s > 0 for s in a.svgd_buckets):
        pool = ProcessPoolExecutor(max_workers=n_workers,
                                   initializer=M._worker_init, initargs=(0,))
        print(f"SVGD in {n_workers} Arbeitsprozessen")

    budget = Zeitbudget(a.max_minuten, name='Orakel')
    alle_zeilen, spuren, zusammen = [], [], []
    t_start = time.perf_counter()
    try:
        for seed in range(a.seeds):
            if budget.abgelaufen() and seed > 0:
                print(f"  Seed {seed} nicht mehr begonnen ({budget.grund()}).")
                break
            t0 = time.perf_counter()
            vor_seed = len(alle_zeilen)

            def fortschritt(r, n, _seed=seed, _t0=t0):
                dt = time.perf_counter() - _t0
                print(f"  Seed {_seed}: Runde {r + 1}/{n}  "
                      f"({dt / (r + 1):.1f}s/Runde, noch "
                      f"{dt / (r + 1) * (n - r - 1) / 60:.1f} min)", flush=True)

            zeilen, gewaehlt, _m = orakel_rollout(
                planner, truths, names, args, kandidaten, a.n_max, seed=seed,
                pool=pool, plan_batch=a.plan_batch, sammler=alle_zeilen,
                on_round=fortschritt, budget=budget)
            if not zeilen:
                break
            for row in zeilen:
                row['seed'] = seed
            spuren += zeilen
            best, tabelle = O.score_trace(zeilen)
            zusammen.append(dict(seed=seed, J=best['J'], q=best['q'],
                                 n_exec=best['n_exec'],
                                 wallclock_s=time.perf_counter() - t0,
                                 gewaehlt={k: [list(x) for x in v]
                                           for k, v in gewaehlt.items()},
                                 tabelle=tabelle))
            print(f"  Seed {seed}: J* = {best['J']:.4f} bei n = "
                  f"{best['n_exec']}  (q = {best['q']:.4f}, "
                  f"{(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
            # Nach jedem Seed schreiben statt erst am Ende: ein Lauf ueber
            # mehrere Stunden soll nicht alles verlieren, wenn er in der
            # letzten Stunde abbricht (Zeitlimit, Absturz, Stromausfall).
            schreibe_datensatz(alle_zeilen[vor_seed:], a.out, anhaengen=seed > 0)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    pfad = a.out
    print(f"\nDatensatz: {pfad}  ({len(alle_zeilen)} Zeilen, "
          f"{len(kandidaten)} Kandidaten je Entscheidung)")

    best_alle, tabelle_alle = O.score_trace(spuren)
    js = os.path.join(RESULTS_DIR, 'policy_orakel.json')
    with open(js, 'w', encoding='utf-8') as f:
        json.dump(dict(
            kandidaten=[list(k) for k in kandidaten],
            n_shapes=len(names), n_max=a.n_max, seeds=a.seeds,
            shapes=names, flow_steps=a.flow_steps, split=a.split,
            J=best_alle['J'], q=best_alle['q'], n_exec=best_alle['n_exec'],
            tabelle=tabelle_alle, je_seed=zusammen,
            wallclock_s=time.perf_counter() - t_start), f, indent=2,
            ensure_ascii=False)
    print(f"Zusammenfassung: {js}")
    print(f"\nOrakel ueber alle Seeds: J* = {best_alle['J']:.4f} bei n = "
          f"{best_alle['n_exec']}  (q = {best_alle['q']:.4f})")
    return best_alle


if __name__ == '__main__':
    main()
