r"""
optimize.py
===========
Die Suche: welche Einstellung deckt die Zielverteilung mit den wenigsten
Ausfuehrungen ab?

Was gesucht wird
----------------
Je Zieldichte-Modell **ein** freier Parameter, und zwar der, den man auch von
Hand einstellen wuerde:

    ucb     kappa   Gewicht der Unsicherheit gegen das bereits Gefundene
    eid     kappa   Mischung aus aufgedeckter Dichte und Informationsdichte
    mass    w       Anteil der Zielmasse, der auf Erkundung entfaellt
    niveau  tau     Schwelle, ab der ein Ort zum Traeger der Form zaehlt

Dazu die **Zahl der Ausfuehrungen** n, und im zweiten Durchlauf die **Zahl der
SVGD-Iterationen**.

Warum n nichts kostet
---------------------
n wird nicht gesucht, sondern abgelesen. Ein Rollout ueber `--n_max` Runden
enthaelt alle kuerzeren Missionen als Praefix, also liefert er die Guete fuer
jedes n zwischen 1 und n_max auf einmal. Die Suche laeuft damit nur ueber den
Modellparameter (und ggf. die SVGD-Iterationen).

Wie gesucht wird
----------------
* **Ohne SVGD** (eindimensional): Grobraster aus `mission.PARAM_GRID`, danach
  eine lokale Verfeinerung um den besten Punkt (drei Auswertungen, halbierte
  Schrittweite). Ein Grobraster statt eines Abstiegs, weil ein einzelner
  Rollout ueber 25 Formen verrauscht ist und ein lokales Verfahren auf einem
  Rauschgipfel haengen bleibt; das Raster sieht die ganze Kurve.
* **Mit SVGD** (zweidimensional): Koordinatenabstieg in drei Durchgaengen —
  Parameter bei mittlerer Iterationszahl, dann Iterationszahl beim besten
  Parameter, dann Parameter-Verfeinerung. Das volle Kreuzprodukt waere das
  Fuenffache an Rechenzeit fuer eine Wechselwirkung, die es zwischen einer
  Zieldichte und einem nachgeschalteten Solver kaum geben kann: SVGD
  verfeinert *gegen dasselbe* Phi, das der Parameter erzeugt hat.

Benutzung
---------
    # Aufwandsschaetzung, rechnet nichts
    python -m exploration_optimierung.optimize --mode beide --estimate_only

    # der eigentliche Lauf (fortsetzbar: einfach erneut starten)
    python -m exploration_optimierung.optimize --mode beide

    # Rangfolge mit anderen Gewichten, ohne neu zu rechnen
    python -m exploration_optimierung.optimize --reweight --lambda_len 0.04

Fortsetzbarkeit: jede fertige Auswertung liegt als JSON unter
`results/cache/`. Ein Neustart ueberspringt, was schon da ist — ein
abgebrochener Lauf kostet hoechstens die angefangene Auswertung.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import torch

if __package__ in (None, ''):                       # direkter Aufruf
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = 'exploration_optimierung'

from . import DEFAULT_CKPT, RESULTS_DIR             # noqa: E402
from . import mission as M                          # noqa: E402
from . import objective as O                        # noqa: E402

CACHE_DIR = os.path.join(RESULTS_DIR, 'cache')

#: Raster der SVGD-Iterationen. 0 ist bewusst dabei: der zweite Durchlauf darf
#: zu dem Ergebnis kommen, dass sich die Verfeinerung nicht lohnt, und dafuer
#: muss "gar kein SVGD" im Suchraum liegen.
SVGD_GRID = [0, 25, 50, 100, 200, 400]
SVGD_MID = 100

# Konsole auf UTF-8: die Ausgaben tragen griechische Buchstaben und Pfeile,
# und cp1252 wirft darauf eine Ausnahme — ein Lauf soll nicht an einem print
# sterben. Dieselbe Vorsichtsmassnahme wie in `constraints/common.py`.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Zwischenspeicher
# ---------------------------------------------------------------------------

#: Die Felder der Konfiguration, die das *Ergebnis* einer Auswertung
#: veraendern. Sie stehen im Schluessel des Zwischenspeichers und entscheiden
#: zugleich, welche abgelegten Spuren zu einem Lauf gehoeren. Bewusst auch
#: `n_max`, `debt_weight`, `sensor_radius` und `flow_steps`: ein
#: zwischengespeichertes Ergebnis unter anderen Randbedingungen
#: wiederzuverwenden waere der stillste denkbare Fehler.
CONFIG_FIELDS = ('n_max', 'n_shapes', 'debt_weight', 'visit_sat',
                 'visit_halflife', 'sensor_radius', 'gp_noise', 'meas_noise',
                 'n_particles', 'flow_steps', 'n_prior', 'truth_res')


def config_payload(cfg):
    """Der konfigurationsabhaengige Teil des Schluessels — ohne die Einstellung."""
    d = {k: getattr(cfg, k) for k in CONFIG_FIELDS}
    d['ckpt'] = os.path.basename(cfg.ckpt)
    return d


def cache_key(cfg, phi_model, param, svgd_iters, seed):
    """Ein Schluessel, der *alles* enthaelt, was das Ergebnis veraendert."""
    payload = dict(phi_model=phi_model, param=round(float(param), 6),
                   svgd_iters=int(svgd_iters), seed=int(seed),
                   **config_payload(cfg))
    blob = json.dumps(payload, sort_keys=True)
    h = hashlib.sha1(blob.encode()).hexdigest()[:16]
    return f"{phi_model}_p{float(param):.4f}_s{int(svgd_iters)}_r{int(seed)}_{h}", payload


def im_speicher(cfg, phi_model, param, svgd_iters, seed):
    """Liegt diese Auswertung schon fertig auf der Platte?"""
    name, _ = cache_key(cfg, phi_model, param, svgd_iters, seed)
    return os.path.isfile(os.path.join(CACHE_DIR, name + '.json'))


def cache_load(name):
    path = os.path.join(CACHE_DIR, name + '.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)['rows']
    except (json.JSONDecodeError, KeyError, OSError):
        return None      # angefangene/kaputte Datei: einfach neu rechnen


def cache_store(name, payload, rows):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = os.path.join(CACHE_DIR, name + '.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'key': payload, 'rows': rows}, f)
    os.replace(tmp, os.path.join(CACHE_DIR, name + '.json'))


# ---------------------------------------------------------------------------
# Fortschritt und Restzeit
# ---------------------------------------------------------------------------

class Progress:
    """Restzeitschaetzung in Runden statt in Auswertungen.

    Eine Auswertung dauert je nach SVGD-Iterationen zwischen einer und zehn
    Minuten; ein Balken ueber Auswertungen wuerde also munter springen. Die
    Runde ist dagegen die Einheit, deren Aufwand sich sauber modellieren
    laesst:

        t_Runde(iters) = t_plan + iters * t_svgd_pro_iter

    Beide Konstanten werden zu Beginn kurz gemessen (`calibrate`) und danach
    laufend an den tatsaechlichen Zeiten nachgezogen. Das ist der Grund, warum
    die angezeigte Ankunftszeit nach der ersten Auswertung steht und nicht
    erst gegen Ende brauchbar wird.
    """

    def __init__(self, t_plan=8.0, t_svgd_iter=0.0057, quiet=False):
        self.t_plan = t_plan
        self.t_svgd_iter = t_svgd_iter
        self.quiet = quiet
        self.t0 = time.time()
        self.done_rounds = 0.0
        self.done_cost = 0.0
        self.total_cost = 0.0
        self.obs = {}                    # svgd_iters -> (n, Summe Sekunden)

    def round_cost(self, svgd_iters):
        n, s = self.obs.get(int(svgd_iters), (0, 0.0))
        if n >= 2:
            return s / n
        return self.t_plan + int(svgd_iters) * self.t_svgd_iter

    def plan_total(self, schedule, n_max):
        """schedule: Liste von (label, svgd_iters, n_evals)."""
        self.total_cost = sum(n_ev * n_max * self.round_cost(it)
                              for _, it, n_ev in schedule)
        return self.total_cost

    def observe(self, svgd_iters, n_rounds, seconds):
        n, s = self.obs.get(int(svgd_iters), (0, 0.0))
        self.obs[int(svgd_iters)] = (n + n_rounds, s + seconds)
        self.done_rounds += n_rounds
        self.done_cost += seconds

    def credit_cached(self, svgd_iters, n_rounds):
        """Aus dem Zwischenspeicher bediente Auswertungen kosten nichts —
        sie muessen aber aus dem Restbudget verschwinden, sonst behauptet die
        Anzeige nach einem Neustart eine Restzeit, die es nicht gibt."""
        self.total_cost -= n_rounds * self.round_cost(svgd_iters)

    def eta(self):
        rest = max(self.total_cost - self.done_cost, 0.0)
        # Die Modellkosten sind eine Schaetzung; sobald echte Zeiten vorliegen,
        # wird der Rest mit dem gemessenen Verhaeltnis skaliert.
        return rest

    def line(self, text):
        if self.quiet:
            return
        el = time.time() - self.t0
        rest = self.eta()
        end = datetime.now() + timedelta(seconds=rest)
        frac = (self.done_cost / self.total_cost) if self.total_cost > 0 else 0.0
        bar_n = int(round(24 * min(max(frac, 0.0), 1.0)))
        bar = '#' * bar_n + '.' * (24 - bar_n)
        print(f"  [{bar}] {100*frac:5.1f}%  "
              f"vergangen {fmt_dt(el)}  Rest {fmt_dt(rest)}  "
              f"fertig ~{end.strftime('%H:%M')}  | {text}", flush=True)


def fmt_dt(sec):
    sec = int(max(sec, 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


# ---------------------------------------------------------------------------
# Eine Auswertung
# ---------------------------------------------------------------------------

class Evaluator:
    """Rollout + Bewertung, mit Zwischenspeicher und Fortschrittsmeldung."""

    def __init__(self, cfg, planner, names, truths, pool, prog):
        self.cfg, self.planner = cfg, planner
        self.names, self.truths = names, truths
        self.pool, self.prog = pool, prog
        self.results = []                # alle Spuren dieses Laufs
        self.n_eval = 0

    def __call__(self, phi_model, param, svgd_iters, seed=0):
        cfg = self.cfg
        name, payload = cache_key(cfg, phi_model, param, svgd_iters, seed)
        pname = M.PHI_MODELS[phi_model][0]

        rows = cache_load(name)
        if rows is not None:
            self.prog.credit_cached(svgd_iters, cfg.n_max)
            self._record(phi_model, param, svgd_iters, seed, rows)
            self.prog.line(f"{phi_model:<6} {pname}={param:<6.3f} "
                           f"svgd={svgd_iters:<4d} (Zwischenspeicher)")
            return rows

        self.n_eval += 1
        t0 = time.time()
        rows, _ = M.run_config(
            self.planner, self.truths, self.names, phi_model, param,
            cfg.n_max, svgd_iters=svgd_iters, seed=seed, pool=self.pool,
            debt_weight=cfg.debt_weight, visit_sat=cfg.visit_sat,
            visit_halflife=cfg.visit_halflife,
            sensor_radius=cfg.sensor_radius, gp_noise=cfg.gp_noise,
            n_particles=cfg.n_particles, meas_noise=cfg.meas_noise,
            max_obs=cfg.max_obs)
        dt = time.time() - t0
        cache_store(name, payload, rows)
        self.prog.observe(svgd_iters, cfg.n_max, dt)
        rec = self._record(phi_model, param, svgd_iters, seed, rows)
        self.prog.line(f"{phi_model:<6} {pname}={param:<6.3f} "
                       f"svgd={svgd_iters:<4d} -> J={rec['J']:.4f} "
                       f"bei n={rec['n_exec']} (q={rec['q']:.3f}, {fmt_dt(dt)})")
        return rows

    def _record(self, phi_model, param, svgd_iters, seed, rows):
        key = {'phi_model': phi_model, 'param': float(param),
               'svgd_iters': int(svgd_iters), 'seed': int(seed)}
        self.results.append({'key': key, 'rows': rows})
        best, _ = O.score_trace(rows, lambda_len=self.cfg.lambda_len,
                                lambda_time=self.cfg.lambda_time,
                                quality=self.cfg.quality)
        return dict(key, **best)

    def score(self, phi_model, param, svgd_iters, seed=0):
        rows = self(phi_model, param, svgd_iters, seed=seed)
        best, _ = O.score_trace(rows, lambda_len=self.cfg.lambda_len,
                                lambda_time=self.cfg.lambda_time,
                                quality=self.cfg.quality)
        return best['J'], best


# ---------------------------------------------------------------------------
# Suchstrategien
# ---------------------------------------------------------------------------

def refine_points(grid, best, lo, hi, n=3):
    """Drei Punkte um den besten Rasterpunkt, mit halber Schrittweite.

    Genommen wird der halbe Abstand zu den *Nachbarn im Raster*, nicht ein
    fester Bruchteil: bei einem logarithmischen Raster (kappa) ist der Abstand
    nach oben ein Vielfaches des Abstands nach unten, und ein fester Bruchteil
    wuerde eine der beiden Seiten kaum absuchen.
    """
    i = grid.index(best)
    left = (grid[i] - grid[i - 1]) if i > 0 else (grid[1] - grid[0])
    right = (grid[i + 1] - grid[i]) if i + 1 < len(grid) else (grid[-1] - grid[-2])
    cand = [best - 0.5 * left, best - 0.25 * left + 0.25 * right,
            best + 0.5 * right][:n]
    out = []
    for c in cand:
        c = float(min(max(c, lo), hi))
        if all(abs(c - g) > 1e-4 for g in grid) and all(abs(c - o) > 1e-4 for o in out):
            out.append(c)
    return out


def arbeitsliste(cfg):
    r"""Alle Auswertungen der vollen Suche, in der Reihenfolge des Abarbeitens.

    Der Unterschied des Kreuzprodukts zum Koordinatenabstieg ist nicht die
    gefundene Einstellung — die ist meist dieselbe — sondern **was danach
    ausgewertet werden kann.** Der Abstieg misst die Iterationszahl nur beim
    *einen* besten Parameterwert und laesst damit genau die Frage offen, ob ein
    anderer Parameter unter mehr SVGD-Iterationen besser waere. Das
    Kreuzprodukt fuellt die ganze Matrix und macht die Wechselwirkung als
    Waermekarte sichtbar. In `mode='beide'` ist die Studie ohne SVGD dabei
    kostenlos enthalten: die 0 steht in `SVGD_GRID`, ihre Spalte *ist* der
    ungefuehrte Zweig, und die Ausgabe trennt beide anschliessend wieder.

    Die Reihenfolge ist die eigentliche Entscheidung hier. Sortiert wird
    **nach SVGD-Iterationen zuerst**, dann nach Modell, dann nach Parameter:

        Durchgang 1:  svgd =   0,  alle Modelle x alle Parameterwerte
        Durchgang 2:  svgd =  25,  alle Modelle x alle Parameterwerte
        ...

    Damit ist **jedes Praefix der Liste eine in sich abgeschlossene Studie**.
    Nach Durchgang 1 liegt die vollstaendige Untersuchung ohne SVGD vor — nicht
    ein angefangenes Stueck davon, sondern das fertige Ergebnis, mit allen
    Abbildungen. Jeder weitere Durchgang fuellt eine ganze Zeile der Waermekarte
    nach.

    Die naheliegende Alternative — Modell fuer Modell komplett durchrechnen —
    haette die unangenehme Eigenschaft, dass ein bei 60 % abgebrochener Lauf
    zwei Modelle vollstaendig und zwei gar nicht vermessen haette. Man koennte
    dann ueber die Modelle nicht vergleichen, und der Vergleich ist der Zweck
    der Studie.
    """
    iters = [0] if cfg.mode == 'ohne_svgd' else list(SVGD_GRID)
    jobs = []
    for seed in range(cfg.seeds):
        for it in iters:
            for m in cfg.models:
                pname = M.PHI_MODELS[m][0]
                for p in M.param_grid(pname, cfg.param_points):
                    jobs.append((m, float(p), int(it), int(seed)))
    return jobs


def status_bericht(cfg, jobs):
    """Was ist fertig, was ist offen — je Durchgang eine Zeile."""
    offen = [j for j in jobs if not im_speicher(cfg, *j)]
    fertig = len(jobs) - len(offen)
    print(f"\n  Arbeitsliste: {len(jobs)} Auswertungen, "
          f"{fertig} fertig, {len(offen)} offen")
    print(f"    {'Durchgang':<22}{'fertig':>10}{'offen':>8}")
    print("    " + "-" * 40)
    for seed in range(cfg.seeds):
        for it in ([0] if cfg.mode == 'ohne_svgd' else SVGD_GRID):
            grp = [j for j in jobs if j[2] == it and j[3] == seed]
            if not grp:
                continue
            n_f = sum(1 for j in grp if im_speicher(cfg, *j))
            marke = '  <-- laeuft' if 0 < n_f < len(grp) else ''
            label = f"svgd={it}" + (f", seed={seed}" if cfg.seeds > 1 else "")
            print(f"    {label:<22}{n_f:>10}{len(grp)-n_f:>8}{marke}")
    return offen


def search_ohne_svgd(ev, phi_model, seed=0):
    """Eindimensionale Suche ueber den Modellparameter, ohne Verfeinerung."""
    pname = M.PHI_MODELS[phi_model][0]
    grid = list(M.PARAM_GRID[pname])
    lo, hi = M.PARAM_RANGE[pname]

    scored = [(ev.score(phi_model, p, 0, seed=seed)[0], p) for p in grid]
    best_p = min(scored)[1]
    for p in refine_points(grid, best_p, lo, hi):
        scored.append((ev.score(phi_model, p, 0, seed=seed)[0], p))
    J, p = min(scored)
    return {'phi_model': phi_model, 'param_name': pname, 'param': p,
            'svgd_iters': 0, 'J': J}


def search_mit_svgd(ev, phi_model, seed=0):
    """Koordinatenabstieg ueber (Modellparameter, SVGD-Iterationen)."""
    pname = M.PHI_MODELS[phi_model][0]
    grid = list(M.PARAM_GRID[pname])
    lo, hi = M.PARAM_RANGE[pname]

    # 1) Parameter bei mittlerer Iterationszahl
    scored = [(ev.score(phi_model, p, SVGD_MID, seed=seed)[0], p) for p in grid]
    best_p = min(scored)[1]

    # 2) Iterationszahl beim besten Parameter
    it_scored = [(ev.score(phi_model, best_p, it, seed=seed)[0], it)
                 for it in SVGD_GRID if it != SVGD_MID]
    it_scored.append((min(scored)[0], SVGD_MID))
    best_it = min(it_scored)[1]

    # 3) Parameter nachziehen
    fine = [(ev.score(phi_model, p, best_it, seed=seed)[0], p)
            for p in refine_points(grid, best_p, lo, hi)]
    if best_it != SVGD_MID:
        fine.append((ev.score(phi_model, best_p, best_it, seed=seed)[0], best_p))
    else:
        fine.append((min(scored)[0], best_p))
    J, p = min(fine)
    return {'phi_model': phi_model, 'param_name': pname, 'param': p,
            'svgd_iters': best_it, 'J': J}


def schedule_for(mode, models, n_refine=3, search='abstieg', n_grid=7):
    """(label, svgd_iters, n_evals) je Stufe — fuer die Aufwandsschaetzung.

    Die *Werte* der Parameter stehen vorher nicht fest, ihre Anzahl und die
    jeweilige Iterationszahl schon. Genau das braucht die Schaetzung.
    """
    sched = []
    if search == 'voll':
        iters = [0] if mode == 'ohne_svgd' else list(SVGD_GRID)
        for m in models:
            for it in iters:
                sched.append((f'{m}: volles Raster, SVGD {it}', it, n_grid))
        return sched

    for m in models:
        if mode in ('ohne_svgd', 'beide'):
            sched.append((f'{m}: Raster ohne SVGD', 0, n_grid))
            sched.append((f'{m}: Verfeinerung ohne SVGD', 0, n_refine))
        if mode in ('mit_svgd', 'beide'):
            sched.append((f'{m}: Raster mit SVGD', SVGD_MID, n_grid))
            for it in SVGD_GRID:
                if it != SVGD_MID:
                    sched.append((f'{m}: SVGD-Iterationen {it}', it, 1))
            sched.append((f'{m}: Verfeinerung mit SVGD', SVGD_MID, n_refine))
    return sched


# ---------------------------------------------------------------------------
# Kalibrierung
# ---------------------------------------------------------------------------

def calibrate(cfg, planner, names, truths, pool, quiet=False):
    """Kosten einer Runde messen: Planung, und SVGD je Iteration.

    Zwei kurze Messungen statt einer Faustzahl, weil der Unterschied zwischen
    einer RTX 2070 und einer Cluster-GPU (und zwischen 4 und 12 Kernen fuer
    SVGD) genau die Groesse ist, die eine Aufwandsschaetzung brauchbar oder
    wertlos macht.
    """
    if not quiet:
        print("  Kalibriere Rundenkosten ...", flush=True)
    args = M.build_mission_args(str(truths.device), 'ucb', 3.0,
                                debt_weight=cfg.debt_weight,
                                visit_sat=cfg.visit_sat,
                                visit_halflife=cfg.visit_halflife,
                                sensor_radius=cfg.sensor_radius,
                                gp_noise=cfg.gp_noise,
                                n_particles=cfg.n_particles,
                                meas_noise=cfg.meas_noise, max_obs=cfg.max_obs)
    m = M.LaengenMission(planner, truths, names, args, svgd_iters=0,
                         seed=0, pool=pool)
    m.round(0)                                   # warmlaufen (CUDA-Kernel)
    t0 = time.time(); m.round(1); t_plan = time.time() - t0

    t_iter = 0.0
    if cfg.mode in ('mit_svgd', 'beide'):
        probe = 50
        m2 = M.LaengenMission(planner, truths, names, args,
                              svgd_iters=probe, seed=0, pool=pool)
        m2.round(0)
        t0 = time.time(); m2.round(1); t_svgd = time.time() - t0
        t_iter = max((t_svgd - t_plan) / probe, 1e-5)
    if not quiet:
        print(f"    Planung {t_plan:.2f} s/Runde ({len(names)} Formen), "
              f"SVGD {1000*t_iter:.2f} ms je Iteration und Runde", flush=True)
    return t_plan, t_iter


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------

#: Jede einzelne Messung, die die Mission erzeugt. Die Spalten sind bewusst
#: *nicht* aggregiert: eine Zeile ist genau ein (Einstellung, Form, Rundenzahl).
ROW_COLS = ['phi_model', 'param', 'svgd_iters', 'seed', 'shape', 'n_exec',
            'cov', 'cov_norm', 'erg_truth', 'belief_rmse', 'info_gain',
            'path_len', 'n_obs', 'plan_s', 'svgd_s']


def write_all_rows(results, tag, still=False):
    r"""Die Rohtabelle: jede Form, jede Rundenzahl, jede Einstellung.

    Die drei anderen CSV-Dateien sind Verdichtungen — `suche_` mittelt ueber
    die Formen *und* waehlt die beste Rundenzahl, `kurven_` mittelt nur ueber
    die Formen. Beide beantworten die Frage der Studie, aber keine von beiden
    laesst sich nachtraeglich anders auswerten: sobald ueber die Formen
    gemittelt ist, ist nicht mehr zu sehen, ob ein Mittelwert aus 25 gleich
    guten Bahnen entsteht oder aus 20 guten und 5 gescheiterten.

    Diese Datei ist deshalb die, aus der spaeter jedes Diagramm gebaut werden
    kann, das jetzt noch niemand vorhergesehen hat — Streuung ueber die Formen,
    einzelne schwierige Buchstaben, der Zusammenhang zwischen Informationsgewinn
    und Abdeckung, die Wirkung des Parameters getrennt nach Formtyp. Sie ist
    lang und schmal (bei der vollen Suche rund 50 000 Zeilen, etwa 5 MB) und
    laesst sich mit `pandas.read_csv` in einer Zeile aufmachen.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'alle_laeufe_{tag}.csv')
    n = 0
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLS, extrasaction='ignore')
        w.writeheader()
        for res in results:
            for r in res['rows']:
                w.writerow(r)
                n += 1
    if not still:
        print(f"  Gespeichert -> {path}  ({n} Messzeilen)")
    return path


def write_outputs(cfg, results, tag, still=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ranked = O.rank(results, lambda_len=cfg.lambda_len,
                    lambda_time=cfg.lambda_time, quality=cfg.quality,
                    n_min=cfg.n_min, n_max=cfg.n_max)

    cols = ['phi_model', 'param', 'svgd_iters', 'seed', 'n_exec', 'J', 'q',
            'cov', 'cov_norm', 'erg_truth', 'belief_rmse', 'info_gain',
            'path_len', 'time_s']
    path = os.path.join(RESULTS_DIR, f'suche_{tag}.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ranked:
            w.writerow({c: r.get(c) for c in cols})
    if not still:
        print(f"  Gespeichert -> {path}")

    # Vollstaendige (n, q)-Kurven: damit laesst sich die Gewichtsfrage
    # nachtraeglich anders beantworten, ohne irgendetwas neu zu rechnen.
    path2 = os.path.join(RESULTS_DIR, f'kurven_{tag}.csv')
    with open(path2, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['phi_model', 'param', 'svgd_iters', 'seed', 'n_exec', 'q',
                    'cov', 'cov_norm', 'erg_truth', 'belief_rmse',
                    'info_gain', 'path_len', 'time_s', 'J'])
        for r in ranked:
            for t in r['table']:
                w.writerow([r['phi_model'], r['param'], r['svgd_iters'],
                            r['seed'], t['n_exec'], t['q'], t['cov'],
                            t['cov_norm'], t['erg_truth'], t['belief_rmse'],
                            t['info_gain'], t['path_len'], t['time_s'],
                            t['J']])
    if not still:
        print(f"  Gespeichert -> {path2}")

    write_all_rows(results, tag, still=still)

    # Die vollstaendige Konfiguration daneben, damit eine Tabelle spaeter noch
    # ihre eigenen Randbedingungen kennt. Ohne das ist in einem halben Jahr
    # nicht mehr zu klaeren, ob eine Zahl bei debt_weight 0,6 oder 0,4 entstand.
    path3 = os.path.join(RESULTS_DIR, f'lauf_{tag}.json')
    with open(path3, 'w', newline='', encoding='utf-8') as f:
        json.dump({'zeitpunkt': datetime.now().isoformat(timespec='seconds'),
                   'konfiguration': {k: v for k, v in vars(cfg).items()},
                   'svgd_raster': SVGD_GRID,
                   'laengeneinheit': M.LENGTH_UNIT,
                   'n_spuren': len(results)}, f, indent=2, default=str)
    if not still:
        print(f"  Gespeichert -> {path3}")
    return ranked


def print_ranking(ranked, cfg, title):
    print(f"\n  {title}")
    print(f"    {'Modell':<8}{'Parameter':>12}{'SVGD':>7}{'n':>5}"
          f"{'J':>9}{'q':>8}{'cov':>9}{'E_erg':>10}{'rmse':>8}{'t/s':>8}")
    print("    " + "-" * 84)
    seen = set()
    for r in ranked:
        pname = M.PHI_MODELS[r['phi_model']][0]
        print(f"    {r['phi_model']:<8}{pname}={r['param']:<8.3f}"
              f"{r['svgd_iters']:>7d}{r['n_exec']:>5d}{r['J']:>9.4f}"
              f"{r['q']:>8.3f}{r['cov']:>9.4f}{r['erg_truth']:>10.5f}"
              f"{r['belief_rmse']:>8.4f}{r['time_s']:>8.1f}")
        seen.add(r['phi_model'])
    if ranked:
        b = ranked[0]
        pname = M.PHI_MODELS[b['phi_model']][0]
        print(f"\n    Bestes Ergebnis: {b['phi_model']} mit {pname}={b['param']:.3f}, "
              f"{b['svgd_iters']} SVGD-Iterationen, {b['n_exec']} Ausfuehrungen")
        print(f"    -> {b['n_exec']} x {M.LENGTH_UNIT:.3f} = "
              f"{b['n_exec']*M.LENGTH_UNIT:.2f} Laengeneinheiten Weg, "
              f"Restfehler q = {b['q']:.3f} des blinden Ausgangswerts")


def schreibe_studien(cfg, quelle, still=False):
    r"""Beide Studien aus einer Menge von Spuren schreiben. -> beste Einstellungen

    Wird von zwei Stellen benutzt: am Ende eines Laufs und von `--reweight`.
    Dass es *dieselbe* Funktion ist, ist der Punkt — der Zwischenstand eines
    noch laufenden Clusterjobs soll exakt die Dateien mit exakt den Namen
    erzeugen, die auch der fertige Lauf erzeugt, sonst passt `plots.py` nicht
    darauf und man baut sich fuer den Zwischenstand eine zweite Auswertung, die
    dann von der endgueltigen abweicht.

    Die Trennung laeuft ueber `svgd_iters`: die Spalte 0 ist die Studie ohne
    Verfeinerung, alles darueber die mit. Die Studie 'mit_svgd' bekommt beide
    Teile, weil 'gar kein SVGD' eine zulaessige Antwort auf die Frage nach der
    besten Iterationszahl ist und in der Rangfolge mitstehen muss.
    """
    ohne = [r for r in quelle if r['key']['svgd_iters'] == 0]
    mit = [r for r in quelle if r['key']['svgd_iters'] > 0]
    best = []
    if cfg.mode in ('ohne_svgd', 'beide') and ohne:
        cfg_o = argparse.Namespace(**vars(cfg))
        cfg_o.lambda_time = 0.0
        ranked = write_outputs(cfg_o, ohne, 'ohne_svgd', still=still)
        if not still:
            print_ranking(ranked[:12], cfg_o, 'Ohne SVGD — Rangfolge (beste 12)')
            print_best_per_model(ranked, cfg_o)
        if ranked:
            best.append(dict(ranked[0], studie='ohne_svgd',
                             param_name=M.PHI_MODELS[ranked[0]['phi_model']][0]))
    if cfg.mode in ('mit_svgd', 'beide') and mit:
        ranked = write_outputs(cfg, mit + ohne, 'mit_svgd', still=still)
        if not still:
            print_ranking(ranked[:12], cfg, 'Mit SVGD — Rangfolge (beste 12)')
            print_best_per_model(ranked, cfg)
        if ranked:
            best.append(dict(ranked[0], studie='mit_svgd',
                             param_name=M.PHI_MODELS[ranked[0]['phi_model']][0]))
    return best


def print_best_per_model(ranked, cfg):
    print("\n  Bestes je Zieldichte-Modell")
    print(f"    {'Modell':<8}{'Parameter':>12}{'SVGD':>7}{'n':>5}{'J':>9}{'q':>8}")
    print("    " + "-" * 49)
    best = {}
    for r in ranked:
        if r['phi_model'] not in best:
            best[r['phi_model']] = r
    for m in M.PHI_MODELS:
        r = best.get(m)
        if r is None:
            continue
        pname = M.PHI_MODELS[m][0]
        print(f"    {m:<8}{pname}={r['param']:<8.3f}{r['svgd_iters']:>7d}"
              f"{r['n_exec']:>5d}{r['J']:>9.4f}{r['q']:>8.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', default='beide',
                   choices=['ohne_svgd', 'mit_svgd', 'beide'],
                   help='Welche der beiden Studien gefahren wird.')
    p.add_argument('--models', default='ucb,eid,mass,niveau',
                   help='Zieldichte-Modelle, kommagetrennt.')
    p.add_argument('--n_max', type=int, default=12,
                   help='Rundenobergrenze. Jede kleinere Rundenzahl faellt '
                        'als Praefix mit ab und kostet nichts extra.')
    p.add_argument('--n_min', type=int, default=1,
                   help='Kleinste zugelassene Zahl von Ausfuehrungen.')
    p.add_argument('--n_shapes', type=int, default=25,
                   help='Zahl der Holdout-Formen (25 = die volle Menge).')
    p.add_argument('--search', default='abstieg', choices=['abstieg', 'voll'],
                   help="'abstieg' = Grobraster + lokale Verfeinerung, bei SVGD "
                        "Koordinatenabstieg (schnell, findet die Einstellung). "
                        "'voll' = vollstaendiges Kreuzprodukt Parameter x "
                        "SVGD-Iterationen (rund doppelt so teuer, liefert dafuer "
                        "die ganze Wirkungsmatrix fuer die Auswertung).")
    p.add_argument('--param_points', type=int, default=None,
                   help='Zahl der Rasterpunkte je Parameter. Ohne Angabe das '
                        'handgesetzte Raster mit 7 Punkten; 12 bis 16 fuellen '
                        f'den Bereich dichter (kappa {M.PARAM_RANGE["kappa"]}, '
                        f'w {M.PARAM_RANGE["w"]}, tau {M.PARAM_RANGE["tau"]}).')
    p.add_argument('--seeds', type=int, default=1,
                   help='Wiederholungen mit anderem Zufallsstartwert. >1 macht '
                        'die Streuung sichtbar und kostet linear mehr.')
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--workers', type=int, default=0,
                   help='Prozesse fuer die SVGD-Verfeinerung. 0 = automatisch '
                        '(CPU-Kerne minus zwei).')

    g = p.add_argument_group('Zielfunktion')
    g.add_argument('--lambda_len', type=float, default=O.DEFAULT_LAMBDA_LEN)
    g.add_argument('--lambda_time', type=float, default=None,
                   help='Voreinstellung: 0 ohne SVGD, sonst '
                        f'{O.DEFAULT_LAMBDA_TIME}.')
    g.add_argument('--quality', default='cov', choices=list(O.QUALITY_KEYS))

    g = p.add_argument_group('Mission')
    g.add_argument('--debt_weight', type=float, default=0.6,
                   help='Wie stark ein Besuch die Anziehung eines *gefundenen* '
                        'Gebiets senkt. 0 = gar nicht, 1 = auf null.')
    g.add_argument('--visit_sat', type=float, default=1.0,
                   help='Ab welchem Anteil des Besuchsmaximums ein Gebiet als '
                        'bedient gilt. 1.0 laesst die Altersgewichtung wirken, '
                        '0.25 (GUI-Voreinstellung) ueberdeckt sie.')
    g.add_argument('--visit_halflife', type=float, default=3.0,
                   help='Halbwertszeit der Besuchssperre in Laengeneinheiten. '
                        '0 schaltet die Alterung ab.')
    g.add_argument('--sensor_radius', type=float, default=0.06)
    g.add_argument('--gp_noise', type=float, default=0.05)
    g.add_argument('--meas_noise', type=float, default=0.02)
    g.add_argument('--n_particles', type=int, default=256)
    g.add_argument('--max_obs', type=int, default=64)
    g.add_argument('--n_prior', type=int, default=0,
                   help='Vorabmessungen je Form. 0 = ohne jedes Vorwissen '
                        'starten, wie in der Aufgabenstellung.')
    g.add_argument('--truth_res', type=int, default=96)
    g.add_argument('--flow_steps', type=int, default=100,
                   help='Schritte der Flow-ODE. 50 halbiert die Planungszeit '
                        'bei leicht groberen Bahnen.')

    g = p.add_argument_group('Ablauf')
    g.add_argument('--estimate_only', action='store_true',
                   help='Nur den Aufwand schaetzen und beenden.')
    g.add_argument('--reweight', action='store_true',
                   help='Rangfolge allein aus dem Zwischenspeicher neu bilden.')
    g.add_argument('--status', action='store_true',
                   help='Nur zeigen, was fertig und was offen ist, dann '
                        'beenden. Laedt weder Netz noch Formen.')
    g.add_argument('--max_evals', type=int, default=None,
                   help='Hoechstens so viele *neue* Auswertungen rechnen, dann '
                        'sauber beenden und alle Tabellen schreiben. Der '
                        'naechste Aufruf mit demselben Befehl macht weiter. '
                        'Nur bei --search voll.')
    g.add_argument('--zwischenstand', type=int, default=16,
                   help='Alle N Auswertungen die Tabellen neu schreiben, damit '
                        'ein noch laufender Job jederzeit einen abholbaren '
                        'Stand hat. 0 schaltet das ab.')
    g.add_argument('--max_stunden', type=float, default=None,
                   help='Zeitbudget in Stunden. Nach Ablauf wird die laufende '
                        'Auswertung noch fertig gerechnet, dann wird sauber '
                        'beendet. Nur bei --search voll.')
    g.add_argument('--tag', default=None)
    a = p.parse_args(argv)
    a.models = [m.strip() for m in a.models.split(',') if m.strip()]
    for m in a.models:
        if m not in M.PHI_MODELS:
            p.error(f"unbekanntes Modell {m!r}; bekannt: {sorted(M.PHI_MODELS)}")
    return a


def load_cached_results(cfg):
    """Alle Spuren aus dem Zwischenspeicher, die zu dieser Konfiguration passen.

    Verglichen wird gegen `config_payload`, also gegen *alle* Randbedingungen —
    nicht nur gegen `n_max` und `n_shapes`. Das ist der Unterschied zwischen
    "alles, was zufaellig dieselbe Formenzahl hatte" und "alles, was zu diesem
    Lauf gehoert", und bei fortgesetzten Laeufen faellt der Unterschied sonst
    niemandem auf: die Zahlen sehen plausibel aus, stammen aber aus zwei
    verschiedenen Einstellungen.
    """
    out = []
    if not os.path.isdir(CACHE_DIR):
        return out
    want = config_payload(cfg)
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith('.json'):
            continue
        try:
            with open(os.path.join(CACHE_DIR, fn), encoding='utf-8') as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        k = d['key']
        if any(k.get(f) != v for f, v in want.items()):
            continue
        if k['phi_model'] not in cfg.models:
            continue
        out.append({'key': {'phi_model': k['phi_model'], 'param': k['param'],
                            'svgd_iters': k['svgd_iters'], 'seed': k['seed']},
                    'rows': d['rows']})
    return out


def main(argv=None):
    cfg = parse_args(argv)
    if cfg.lambda_time is None:
        cfg.lambda_time = (0.0 if cfg.mode == 'ohne_svgd'
                           else O.DEFAULT_LAMBDA_TIME)
    tag = cfg.tag or cfg.mode

    print("\n=== Laengeneinheit-Mission: Suche nach der besten Einstellung ===")
    print(f"  Modelle       : {', '.join(cfg.models)}")
    print(f"  Studie        : {cfg.mode}")
    print(f"  Formen        : {cfg.n_shapes} Holdout   Runden bis {cfg.n_max}")
    print(f"  Suche         : {cfg.search}   "
          f"{len(M.param_grid('kappa', cfg.param_points))} Parameterpunkte   "
          f"{cfg.seeds} Startwert(e)")
    if cfg.search == 'voll':
        print(f"  SVGD-Raster   : {', '.join(str(i) for i in SVGD_GRID)} "
              f"(volles Kreuzprodukt mit dem Parameterraster)")
    print(f"  Laengeneinheit: {M.LENGTH_UNIT:.4f} (Diagonale von [0,1]^2)")
    print(f"  Zielfunktion  : J = q + {cfg.lambda_len}*n "
          f"+ {cfg.lambda_time}*t   (q ueber '{cfg.quality}')")

    if cfg.reweight:
        results = load_cached_results(cfg)
        if not results:
            print("\n  Kein passendes Ergebnis im Zwischenspeicher.")
            return 1
        print(f"\n  {len(results)} Spuren aus dem Zwischenspeicher ausgewertet.")
        if cfg.search == 'voll':
            jobs = arbeitsliste(cfg)
            status_bericht(cfg, jobs)
        schreibe_studien(cfg, results)
        print("\n  Abbildungen erzeugen:")
        print("    python -m exploration_optimierung.plots\n")
        return 0

    # Der Statusbericht braucht weder Netz noch Formen — er liest nur, welche
    # Schluessel auf der Platte liegen. Deshalb vor dem Laden.
    jobs = arbeitsliste(cfg) if cfg.search == 'voll' else None
    if cfg.status:
        if jobs is None:
            print("\n  --status gibt es nur fuer --search voll; der "
                  "Koordinatenabstieg entscheidet erst waehrend des Laufs, "
                  "welche Auswertungen er braucht.")
            return 1
        status_bericht(cfg, jobs)
        print()
        return 0

    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Geraet        : {device}")

    t0 = time.time()
    names, truths = M.load_holdout(resolution=cfg.truth_res, device=device,
                                   limit=cfg.n_shapes)
    print(f"  {len(names)} Formen geladen ({fmt_dt(time.time()-t0)}): "
          f"{', '.join(names[:6])}{' ...' if len(names) > 6 else ''}")
    planner = M.build_planner(ckpt=cfg.ckpt, device=device,
                              flow_steps=cfg.flow_steps)
    print(f"  Netz          : {os.path.basename(cfg.ckpt)} "
          f"(nxi={planner.nxi}, startpunkt-konditioniert)")

    n_workers = cfg.workers or max(1, (os.cpu_count() or 4) - 2)
    pool = None
    if cfg.mode in ('mit_svgd', 'beide') and n_workers > 1:
        pool = ProcessPoolExecutor(max_workers=n_workers,
                                   initializer=M._worker_init, initargs=(0,))
        print(f"  SVGD          : {n_workers} Arbeitsprozesse")

    try:
        t_plan, t_iter = calibrate(cfg, planner, names, truths, pool)
        prog = Progress(t_plan=t_plan, t_svgd_iter=t_iter)
        n_grid = len(M.param_grid('kappa', cfg.param_points))
        sched = schedule_for(cfg.mode, cfg.models, search=cfg.search,
                             n_grid=n_grid)
        total = prog.plan_total(sched, cfg.n_max) * cfg.seeds
        prog.total_cost = total
        n_evals = sum(n for _, _, n in sched) * cfg.seeds

        # Bei der vollen Suche steht die Arbeitsliste fest: was schon auf der
        # Platte liegt, wird nicht noch einmal gerechnet, und Fortschritt wie
        # Restzeit beziehen sich nur auf das, was dieser Aufruf tatsaechlich
        # vor sich hat.
        lauf = None
        if jobs is not None:
            offen = status_bericht(cfg, jobs)
            lauf = offen
            if cfg.max_evals is not None:
                lauf = lauf[:max(0, cfg.max_evals)]
            prog.total_cost = sum(cfg.n_max * prog.round_cost(it)
                                  for _, _, it, _ in lauf)
            n_evals = len(lauf)
            if not lauf:
                print("\n  Nichts offen — alles schon gerechnet. Es werden nur "
                      "die Tabellen neu geschrieben.")

        print(f"\n  Geplant: {n_evals} Auswertungen x {cfg.n_max} Runden "
              f"x {len(names)} Formen")
        total = prog.total_cost
        print(f"  Geschaetzter Aufwand: {fmt_dt(total)} "
              f"(fertig gegen {(datetime.now()+timedelta(seconds=total)).strftime('%d.%m. %H:%M')})")
        if lauf is None:
            for label, it, n_ev in sched:
                print(f"    {label:<38} {n_ev:>2} x {cfg.n_max} Runden a "
                      f"{prog.round_cost(it):5.1f} s = {fmt_dt(n_ev*cfg.n_max*prog.round_cost(it))}")
        else:
            for it in ([0] if cfg.mode == 'ohne_svgd' else SVGD_GRID):
                n_ev = sum(1 for j in lauf if j[2] == it)
                if n_ev:
                    print(f"    svgd={it:<34d} {n_ev:>2} x {cfg.n_max} Runden a "
                          f"{prog.round_cost(it):5.1f} s = "
                          f"{fmt_dt(n_ev*cfg.n_max*prog.round_cost(it))}")
        if cfg.estimate_only:
            print("\n  --estimate_only: es wurde nichts gerechnet.\n")
            return 0

        print()
        ev = Evaluator(cfg, planner, names, truths, pool, prog)
        best = []
        abbruch = None
        if lauf is not None:
            # Zeile fuer Zeile. Jede fertige Auswertung liegt danach auf der
            # Platte; ein Abbruch — durch Budget, Strg+C oder Stromausfall —
            # kostet hoechstens die gerade laufende.
            t_ende = (prog.t0 + cfg.max_stunden * 3600.0
                      if cfg.max_stunden else None)
            try:
                for i, (m, p, it, seed) in enumerate(lauf, 1):
                    if t_ende is not None and time.time() >= t_ende:
                        abbruch = (f"Zeitbudget von {cfg.max_stunden} h "
                                   f"erreicht")
                        break
                    print(f"  [{i}/{len(lauf)}]", end=' ')
                    ev(m, p, it, seed)
                    # Tabellen zwischendurch schreiben. Ein Clusterjob laeuft
                    # 23 Stunden; ohne das gaebe es waehrenddessen nichts
                    # abzuholen ausser dem Zwischenspeicher, und ein
                    # Zwischenstand nach der ersten Nacht waere nur ueber einen
                    # zweiten Auswertungslauf zu bekommen.
                    if (cfg.zwischenstand and i % cfg.zwischenstand == 0
                            and i < len(lauf)):
                        schreibe_studien(cfg, load_cached_results(cfg),
                                         still=True)
                        print(f"       Zwischenstand geschrieben "
                              f"({i}/{len(lauf)} dieses Aufrufs)", flush=True)
            except KeyboardInterrupt:
                abbruch = "durch Strg+C unterbrochen"
            if cfg.max_evals is not None and abbruch is None and lauf:
                abbruch = f"Budget von {cfg.max_evals} Auswertungen erreicht"
        else:
            for seed in range(cfg.seeds):
                for m in cfg.models:
                    if cfg.mode in ('ohne_svgd', 'beide'):
                        best.append(dict(search_ohne_svgd(ev, m, seed=seed),
                                         studie='ohne_svgd', seed=seed))
                    if cfg.mode in ('mit_svgd', 'beide'):
                        best.append(dict(search_mit_svgd(ev, m, seed=seed),
                                         studie='mit_svgd', seed=seed))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    print(f"\n  Fertig in {fmt_dt(time.time()-prog.t0)}, "
          f"{ev.n_eval} gerechnete Auswertungen.\n")

    # Bei der vollen Suche werden die Tabellen aus *allem* gebaut, was auf der
    # Platte liegt — nicht nur aus dem, was dieser Aufruf gerechnet hat. Nur so
    # enthaelt ein fortgesetzter Lauf auch die Ergebnisse der vorigen Sitzungen,
    # und nur so sind die Tabellen nach jedem Teilstueck vollstaendig aktuell.
    quelle = load_cached_results(cfg) if lauf is not None else ev.results
    aus_studien = schreibe_studien(cfg, quelle)
    if lauf is not None:
        best = aus_studien

    path = os.path.join(RESULTS_DIR, f'bestwerte_{tag}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([{k: v for k, v in b.items() if k != 'table'} for b in best],
                  f, indent=2)
    print(f"\n  Gespeichert -> {path}")

    if lauf is not None:
        rest = [j for j in jobs if not im_speicher(cfg, *j)]
        if rest:
            grund = abbruch or "Budget aufgebraucht"
            print(f"\n  {grund}. Noch offen: {len(rest)} von {len(jobs)} "
                  f"Auswertungen (~{fmt_dt(sum(cfg.n_max*prog.round_cost(j[2]) for j in rest))}).")
            print("  Fortsetzen mit demselben Befehl — was fertig ist, wird "
                  "uebersprungen:")
            print("    " + " ".join(sys.argv[:1] and
                                    ['python', '-m',
                                     'exploration_optimierung.optimize']
                                    + sys.argv[1:]))
            print("  Stand jederzeit ansehen:  ... --status")
        else:
            print(f"\n  Arbeitsliste vollstaendig abgearbeitet "
                  f"({len(jobs)} Auswertungen).")

    print("\n  Abbildungen erzeugen:")
    print("    python -m exploration_optimierung.plots\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
