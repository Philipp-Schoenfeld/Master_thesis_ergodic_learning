r"""
optuna_search.py
=================
Bayessche Suche (Optuna/TPE) ueber den vollen Einstellungsraum der
Laengeneinheit-Mission — als Ersatz fuer den handgeschriebenen Abstieg in
`optimize.py`, nicht als zweites Universum daneben.

Was unveraendert wiederverwendet wird
-------------------------------------
* **Die Zielfunktion.** `objective.score_trace` mit denselben Gewichten
  (`lambda_len=0.02`, `lambda_time=0.004`). J bleibt minimiert, und die beste
  Rundenzahl `n_exec` faellt weiterhin aus *einer* Spur ab, statt gesucht zu
  werden — eine Mission ueber `n_max` Runden enthaelt alle kuerzeren als
  Praefix (siehe `mission.py`-Kopf). `n_exec` ist deshalb bewusst **keine**
  Optuna-Dimension: es waere verschenkte Rechenzeit.
* **Der Rollout.** `LaengenMission` mit `policy=None`, damit die
  RNG-Reihenfolge der bereits veroeffentlichten Studien erhalten bleibt.
* **Der SVGD-Prozesspool** aus `mission._worker_init`.

Was `optimize.py` bisher *nicht* durchsucht hat und hier dazukommt
------------------------------------------------------------------
`debt_weight`, `visit_sat`, `visit_halflife` (die Abdeckungsschuld war
komplett ungetunt), sowie die **Repraesentation der Zieldichte**: `phi_mode`
(`uniform`/`density`/`quantile`), `phi_quantile` und `n_particles`. Das ist
die Achse "wie wird die Zielmenge dem Netz vorgelegt" — die
Architektur-Achse (Partikel vs. Spektral vs. SDF) ist derzeit nicht
durchsuchbar, weil nur das Partikelnetz einen brauchbaren Checkpoint hat.

Bewusst **fest**: `sensor_radius`, `gp_noise`, `meas_noise`, `max_obs`. Die
beschreiben Roboter und Messprozess, nicht das Verfahren — sie mitzudrehen
wuerde die Aufgabe leichter machen statt die Methode besser, und die Zahlen
waeren nicht mehr mit dem bestehenden Lauf vergleichbar.

Eigener Cache, mit Absicht
--------------------------
`optimize.py`s Cache-Key (`CONFIG_FIELDS`) enthaelt weder `max_obs` noch
`phi_mode`/`phi_quantile`. Wer die variiert und denselben Cache benutzt,
liest stillschweigend fremde Spuren. Diese Studie schreibt deshalb in
`results/optuna/cache/` mit einem Key ueber *alle* variierten Groessen.

Pruning
-------
Ein Versuch ist ein Rollout ueber alle Holdout-Formen und kostet Minuten.
Die Rundenschleife wird deshalb hier selbst gefahren (statt `run()`), damit
nach jeder Runde ein Zwischen-J an Optuna gemeldet werden kann und der
`MedianPruner` aussichtslose Einstellungen frueh abbricht. Das Zwischen-J
dient nur dem Vergleich *auf derselben Stufe*; gewertet wird am Ende immer
die vollstaendige Spur.

Beispiel
--------
    python -m exploration_optimierung.optuna_search --trials 200 --study tuning_v1
    python -m exploration_optimierung.optuna_search --study tuning_v1 --report
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from . import DEFAULT_CKPT, RESULTS_DIR                     # noqa: E402
from . import mission as M                                  # noqa: E402
from . import objective as OBJ                              # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, 'optuna')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')

#: Immer fest: Roboter- und Welteigenschaften, keine Stellschrauben des
#: Verfahrens. `sensor_radius` ist der Sensor, `meas_noise` das echte
#: Messrauschen der Welt. Wer die mitdreht, macht die Aufgabe leichter statt
#: die Methode besser.
FIXED_ALWAYS = dict(sensor_radius=0.06, meas_noise=0.02)

#: Im kleinen Raum zusaetzlich fest. `gp_noise` ist bewusst *nicht* dasselbe
#: wie `meas_noise`: es ist die vom Glauben *angenommene* Rauschstaerke, also
#: ein Modellparameter (siehe die Messreihe im Kopf von `belief.py`), und
#: wird im grossen Raum mitgesucht.
FIXED_BASIS = dict(gp_noise=0.05, max_obs=64)

#: Voreinstellungen der Groessen, die erst der grosse Raum oeffnet.
DEFAULTS_GROSS = dict(gp_lengthscale=0.08, gp_variance=1.0, gp_res=64,
                      visit_bandwidth=None, cfg_weight=2.0, flow_steps=100,
                      phi_gamma=1.0)

#: Modelle, die `acquisition.PHI_MODELLE` kann, `mission.PHI_MODELS` aber nie
#: freigeschaltet hat — im grossen Raum ueber ein direktes Setzen von
#: `args.phi_model` erreichbar (siehe `build_args`).
EXTRA_MODELS = ('stretch', 'ei', 'mi')


# ---------------------------------------------------------------------------
# Suchraum
# ---------------------------------------------------------------------------

def suggest_config(trial, space='basis'):
    """Ein Punkt im Einstellungsraum. `param` haengt am Modell — Optuna
    behandelt das als bedingten Raum, TPE kommt damit um.

    Drei Raeume:

    ``basis``  9 Dimensionen, die erste Studie.
    ``ideal``  12 Dimensionen. Gegenueber ``gross`` fehlen bewusst
               ``gp_variance`` (entartet mit kappa: sigma skaliert mit
               sqrt(variance), variance vervierfachen ist dasselbe wie kappa
               verdoppeln — zwei Regler fuer einen Effekt), ``flow_steps``
               (Rechen-, kein Guetereglern; die ODE saettigt, und der
               Zeitterm in J wuerde das Signal verrauschen) sowie
               ``gp_res``/``max_obs``/``visit_bandwidth`` (schwache Effekte
               zum Preis je einer vollen Dimension).
    ``gross``  16 Dimensionen, alles was erreichbar ist.
    """
    models = ['ucb', 'eid', 'mass', 'niveau']
    if space in ('ideal', 'gross'):
        models = models + list(EXTRA_MODELS)
    phi_model = trial.suggest_categorical('phi_model', models)

    # Bereiche aus `mission.PARAM_RANGE`, damit die Skala dieselbe bleibt wie
    # im handgeschriebenen Raster: kappa logarithmisch, w und tau linear.
    if phi_model in ('ucb', 'eid', 'stretch', 'mi'):
        param = trial.suggest_float('kappa', 0.15, 10.0, log=True)
    elif phi_model == 'mass':
        param = trial.suggest_float('w', 0.05, 0.95)
    elif phi_model == 'ei':
        param = trial.suggest_float('xi', 0.001, 0.5, log=True)
    else:
        param = trial.suggest_float('tau', 0.02, 0.90)

    cfg = dict(
        phi_model=phi_model,
        param=float(param),
        # Obergrenze 150 statt 400: `svgd_iters` ist der mit Abstand teuerste
        # Regler (Kostenmodell: Runde = 8s + iters*5.7ms), und die besten 20
        # aus Studie 1 lagen samt und sonders bei 25-50. Der Deckel kauft
        # Versuche, ohne eine Region zu verlieren, die je gewonnen haette.
        svgd_iters=trial.suggest_int('svgd_iters', 0, 150, step=25),
        debt_weight=trial.suggest_float('debt_weight', 0.0, 1.0),
        visit_sat=trial.suggest_float('visit_sat', 0.1, 1.0),
        visit_halflife=trial.suggest_float('visit_halflife', 0.5, 8.0),
        phi_mode=trial.suggest_categorical('phi_mode', ['uniform', 'density', 'quantile']),
        n_particles=trial.suggest_categorical('n_particles', [128, 256, 512]),
    )
    cfg['phi_quantile'] = (trial.suggest_float('phi_quantile', 0.1, 0.9)
                           if cfg['phi_mode'] == 'quantile' else 0.5)

    if space == 'basis':
        cfg.update(DEFAULTS_GROSS, **FIXED_BASIS)
        cfg['visit_bandwidth'] = FIXED_ALWAYS['sensor_radius']
        return cfg

    if space == 'ideal':
        # Alles auf Voreinstellung ausser den drei Groessen, die nach der
        # Sichtung wirklich etwas entscheiden koennen.
        cfg.update(DEFAULTS_GROSS, **FIXED_BASIS)
        cfg['visit_bandwidth'] = FIXED_ALWAYS['sensor_radius']
        cfg.update(
            # Korrelationslaenge des Glaubens — war seit jeher auf 0.08
            # verdrahtet und bestimmt, wie weit eine Messung traegt.
            gp_lengthscale=trial.suggest_float('gp_lengthscale', 0.02, 0.25, log=True),
            # Angenommenes Rauschen des GP (nicht das der Welt). `belief.py`
            # fuehrt dazu eine Messreihe, die 0.05 statt 0.01 nahelegt — also
            # eine Groesse mit belegtem Effekt.
            gp_noise=trial.suggest_float('gp_noise', 0.01, 0.2, log=True),
            # Fuehrungsstaerke der Erzeugung.
            cfg_weight=trial.suggest_float('cfg_weight', 1.0, 4.0),
        )
        cfg['phi_gamma'] = (trial.suggest_float('phi_gamma', 0.01, 4.0, log=True)
                            if phi_model == 'mi' else 1.0)
        return cfg

    # ── nur grosser Raum ────────────────────────────────────────────────
    cfg.update(
        # Glaube: die Korrelationslaenge war bis jetzt auf 0.08 verdrahtet.
        gp_lengthscale=trial.suggest_float('gp_lengthscale', 0.02, 0.25, log=True),
        gp_variance=trial.suggest_float('gp_variance', 0.25, 4.0, log=True),
        gp_noise=trial.suggest_float('gp_noise', 0.01, 0.2, log=True),
        gp_res=trial.suggest_categorical('gp_res', [32, 48, 64, 96]),
        max_obs=trial.suggest_int('max_obs', 16, 256, log=True),
        # Besuchskern, bisher stillschweigend an den Sensorradius gekoppelt.
        visit_bandwidth=trial.suggest_float('visit_bandwidth', 0.02, 0.15),
        # Erzeugung: Fuehrungsstaerke und ODE-Aufloesung des Netzes.
        cfg_weight=trial.suggest_float('cfg_weight', 1.0, 4.0),
        flow_steps=trial.suggest_int('flow_steps', 25, 200, step=25),
    )
    cfg['phi_gamma'] = (trial.suggest_float('phi_gamma', 0.01, 4.0, log=True)
                        if phi_model == 'mi' else 1.0)
    return cfg


def build_args(cfg, device):
    """Namensobjekt fuer `zieldichte`/`debt_density`.

    Die drei Modelle aus `EXTRA_MODELS` kennt `build_mission_args` nicht (sie
    stehen nicht in `mission.PHI_MODELS`). Statt dessen dortige Tabelle zu
    erweitern — was die Bedeutung des `param`-Feldes fuer bestehende Studien
    verschoebe — wird hier mit einem gueltigen Basismodell gebaut und danach
    der interne Modellname gesetzt. `zieldichte` liest genau diese Felder.
    """
    base = cfg['phi_model']
    known = base not in EXTRA_MODELS
    args = M.build_mission_args(
        str(device), phi_model=(base if known else 'ucb'),
        param=(cfg['param'] if known else 1.0),
        debt_weight=cfg['debt_weight'], visit_sat=cfg['visit_sat'],
        visit_halflife=cfg['visit_halflife'], n_particles=cfg['n_particles'],
        phi_mode=cfg['phi_mode'], gp_noise=cfg['gp_noise'],
        max_obs=cfg['max_obs'], **FIXED_ALWAYS)

    if not known:
        args.phi_model = base
        if base in ('stretch', 'mi'):
            args.kappa = cfg['param']
        elif base == 'ei':
            args.phi_xi = cfg['param']
    if base == 'mi':
        args.phi_gamma = cfg['phi_gamma']

    # `build_mission_args` verdrahtet beide fest; hier nachgezogen, weil sie
    # im grossen Raum eigene Dimensionen sind.
    args.phi_quantile = cfg['phi_quantile']
    args.visit_bandwidth = cfg['visit_bandwidth'] or FIXED_ALWAYS['sensor_radius']
    return args


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_key(cfg, ctx, seed):
    payload = {**cfg, **ctx, 'seed': seed}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:20], payload


def cache_load(key):
    p = os.path.join(CACHE_DIR, f'{key}.json')
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)['rows']
    except Exception:
        return None


def cache_store(key, payload, rows):
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, f'{key}.json')
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'key': payload, 'rows': rows}, f)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Ein Versuch = ein Rollout
# ---------------------------------------------------------------------------

def run_trial(cfg, planner, truths, names, n_max, seed, pool, trial=None,
              lambda_len=OBJ.DEFAULT_LAMBDA_LEN, lambda_time=OBJ.DEFAULT_LAMBDA_TIME,
              quality='cov', step_base=0, js_done=None):
    """Rollout + Bewertung. Gibt (J, best_record, rows) zurueck.

    Die Rundenschleife steht hier statt in `LaengenMission.run`, damit nach
    jeder Runde geprunt werden kann; `torch.manual_seed` wird dabei wie dort
    genau einmal vor der Schleife gesetzt, damit die Zufallsfolge dieselbe
    bleibt.
    """
    import optuna

    args = build_args(cfg, truths.device)

    # Die Erzeugungsparameter haengen am Planer, nicht an den Missionsargumenten.
    # Der Planer wird einmal gebaut und ueber alle Versuche geteilt (das Laden
    # des Checkpoints kostet mehr als ein ganzer Versuch), deshalb hier je
    # Versuch umgestellt statt neu gebaut.
    planner.cfg_weight = cfg['cfg_weight']
    planner.steps = int(cfg['flow_steps'])

    m = M.LaengenMission(planner, truths, names, args,
                         svgd_iters=cfg['svgd_iters'], seed=seed, pool=pool,
                         gp_res=int(cfg['gp_res']),
                         gp_lengthscale=cfg['gp_lengthscale'],
                         gp_variance=cfg['gp_variance'])
    torch.manual_seed(seed)

    rows, S = [], max(m.S, 1)
    for r in range(n_max):
        rows += m.round(r, n_max=n_max)
        for row in rows:
            row['plan_s'] = m.plan_s / S
            row['svgd_s'] = m.svgd_s / S
        if trial is not None:
            part, _ = OBJ.score_trace(rows, lambda_len=lambda_len,
                                      lambda_time=lambda_time, quality=quality)
            if part is not None:
                # Gemeldet wird immer die *laufende Schaetzung des Endwerts*:
                # Mittel ueber die fertigen Seeds plus den Teilstand des
                # aktuellen. So ist die gemeldete Groesse auf jeder
                # Ressourcenstufe dasselbe Mass und ueber Versuche hinweg
                # vergleichbar — die Voraussetzung dafuer, dass Hyperband
                # ueberhaupt richtig entscheidet.
                est = float(np.mean(list(js_done or []) + [part['J']]))
                trial.report(est, step_base + r + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    best, _table = OBJ.score_trace(rows, lambda_len=lambda_len,
                                   lambda_time=lambda_time, quality=quality)
    return (best['J'] if best else float('inf')), best, rows


def make_objective(planner, truths, names, args_ns, pool):
    """Zielfunktion ueber mehrere Seeds, mit Pruner-Gate nach dem ersten.

    Warum ueberhaupt mehrere Seeds: bei ein paar tausend Versuchen findet TPE
    mit hoher Wahrscheinlichkeit eine Einstellung, die einen *guenstigen
    Zufallszug* erwischt hat, statt wirklich gut zu sein — klassische
    Optimierer-Ueberanpassung. Ein Ein-Seed-Optimum ist damit nicht
    belastbar.

    Warum trotzdem bezahlbar: Seed 0 laeuft mit rundenweisem Pruning wie
    bisher. Nur wer *danach* noch aussichtsreich ist, bekommt die restlichen
    Seeds. Schlechte Einstellungen kosten weiterhin einen Seed, gute werden
    verifiziert. Gewertet wird das Mittel ueber alle gerechneten Seeds; weil
    nur vollstaendig durchgelaufene Versuche einen Wert bekommen, sind alle
    gewerteten Zahlen Mittel ueber dieselbe Seed-Zahl und damit vergleichbar.
    """
    import optuna

    seeds = list(range(args_ns.seeds))
    ctx = dict(n_max=args_ns.n_max, n_shapes=len(names),
               truth_res=args_ns.truth_res, ckpt=os.path.basename(args_ns.ckpt),
               quality=args_ns.quality, space=args_ns.space, **FIXED_ALWAYS)

    def _objective(trial):
        cfg = suggest_config(trial, space=args_ns.space)
        js, bests, n_cached = [], [], 0

        for i, seed in enumerate(seeds):
            key, payload = cache_key(cfg, ctx, seed)
            rows = cache_load(key)
            if rows is not None:
                best, _ = OBJ.score_trace(rows, quality=args_ns.quality)
                n_cached += 1
                if best is not None and i < len(seeds) - 1:
                    # Ein zwischengespeicherter Seed kostet nichts; gemeldet
                    # wird nur der Leitersprosse zuliebe, gepruned wird nicht.
                    trial.report(float(np.mean(js + [float(best['J'])])),
                                 (i + 1) * args_ns.n_max)
            else:
                _J, best, rows = run_trial(
                    cfg, planner, truths, names, args_ns.n_max, seed, pool,
                    trial=trial, quality=args_ns.quality,
                    # Durchlaufende Ressourcenachse: Seed i belegt die
                    # Sprossen i*n_max+1 .. (i+1)*n_max. Damit entspricht ein
                    # Sprossenabstand ueberall demselben Rechenaufwand — sonst
                    # verteilt Hyperband sein Budget nach einer Leiter, deren
                    # Stufen unterschiedlich teuer sind.
                    step_base=i * args_ns.n_max, js_done=js)
                cache_store(key, payload, rows)

            if best is None:
                return float('inf')
            js.append(float(best['J']))
            bests.append(best)

        trial.set_user_attr('cached_seeds', n_cached)
        trial.set_user_attr('J_seeds', [round(x, 5) for x in js])
        trial.set_user_attr('J_std', float(np.std(js)))
        for k in ('n_exec', 'q', 'cov', 'cov_norm', 'erg_truth',
                  'belief_rmse', 'info_gain', 'path_len', 'time_s'):
            trial.set_user_attr(k, float(np.mean([b[k] for b in bests])))
        return float(np.mean(js))

    return _objective


# ---------------------------------------------------------------------------
# Bericht
# ---------------------------------------------------------------------------

def write_report(study, out_dir):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for t in study.trials:
        rec = {'number': t.number, 'state': str(t.state), 'J': t.value}
        rec.update({f'p_{k}': v for k, v in t.params.items()})
        rec.update({k: v for k, v in t.user_attrs.items()})
        rows.append(rec)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(out_dir, 'trials.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    done = [t for t in study.trials if t.value is not None]
    if done:
        best = study.best_trial
        with open(os.path.join(out_dir, 'best.json'), 'w') as f:
            json.dump({'number': best.number, 'J': best.value,
                       'params': best.params, 'attrs': best.user_attrs,
                       'n_trials': len(study.trials),
                       'n_complete': len(done)}, f, indent=2)
        print(f"\n  Bester Versuch #{best.number}:  J={best.value:.4f}")
        for k, v in sorted(best.params.items()):
            print(f"    {k:16s} {v}")
        for k in ('n_exec', 'q', 'cov_norm', 'erg_truth', 'time_s'):
            if k in best.user_attrs:
                print(f"    {k:16s} {best.user_attrs[k]:.4f}")


#: Wird vom Signal-Handler gesetzt; der Optuna-Callback liest ihn nach jedem
#: Versuch. So endet ein Abbruch *zwischen* zwei Versuchen statt mitten in
#: einem — der laufende Versuch wird fertig gerechnet und abgelegt.
_PAUSE = {'flag': False}


def _install_pause_handler():
    import signal

    def _handler(signum, _frame):
        if _PAUSE['flag']:
            print("\n  [pause] zweites Signal — harter Abbruch, "
                 "der laufende Versuch geht verloren.")
            raise KeyboardInterrupt
        _PAUSE['flag'] = True
        print(f"\n  [pause] Signal {signum} empfangen. Der laufende Versuch "
             "wird noch fertig gerechnet, danach wird angehalten.\n"
             "         Fortsetzen: derselbe Aufruf mit demselben --study.\n"
             "         (nochmal Strg+C = sofort abbrechen)")

    for sig in (getattr(signal, 'SIGINT', None), getattr(signal, 'SIGTERM', None)):
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass        # z. B. in einem Nicht-Hauptthread


def _pause_callback(study, _trial):
    if _PAUSE['flag']:
        study.stop()


def _clear_stale(study):
    """Versuche, die ein harter Abbruch als RUNNING zurueckgelassen hat.

    Sie blockieren nichts, verfaelschen aber die Zaehlung und stehen fuer
    immer als 'laeuft' in der Datenbank. Beim Fortsetzen werden sie auf FAIL
    gesetzt, damit die Studie ehrlich bleibt.
    """
    from optuna.trial import TrialState
    stale = [t for t in study.get_trials(deepcopy=False)
             if t.state == TrialState.RUNNING]
    if not stale:
        return
    fixed = 0
    for t in stale:
        try:
            study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
            fixed += 1
        except Exception:
            pass
    print(f"  {len(stale)} verwaiste(r) Versuch(e) aus einem frueheren "
         f"Abbruch gefunden, {fixed} als FAIL markiert.")


def main():
    import optuna

    p = argparse.ArgumentParser()
    p.add_argument('--study', type=str, default='tuning_v1')
    p.add_argument('--space', type=str, default='ideal',
                   choices=['basis', 'ideal', 'gross'],
                   help="'basis' = 9 Dim. (erste Studie); 'ideal' = 12 Dim. "
                        "ohne die entarteten/schwachen Groessen; 'gross' = "
                        "16 Dim., alles Erreichbare.")
    p.add_argument('--seeds', type=int, default=3,
                   help='Rollout-Seeds je Versuch; gewertet wird das Mittel. '
                        '1 = schnell aber nicht seed-robust.')
    p.add_argument('--min_resource', type=int, default=4,
                   help='Unterste Hyperband-Sprosse in Runden. Niedriger = '
                        'schnelleres Sieben, aber mehr gute Einstellungen '
                        'wegen eines unguenstigen Seeds verworfen.')
    p.add_argument('--random_startup', type=int, default=50,
                   help='Erste N Versuche rein zufaellig. Dient zugleich als '
                        'Zufallssuche-Kontrolle gegen TPE.')
    p.add_argument('--trials', type=int, default=200)
    p.add_argument('--timeout_h', type=float, default=None,
                   help='Wanduhr-Budget in Stunden (zusaetzlich zu --trials).')
    p.add_argument('--n_max', type=int, default=12)
    p.add_argument('--n_shapes', type=int, default=25)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--truth_res', type=int, default=96)
    p.add_argument('--quality', type=str, default='cov', choices=list(OBJ.QUALITY_KEYS))
    p.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--workers', type=int, default=0,
                   help='SVGD-Prozesse; 0 = cpu_count-2.')
    p.add_argument('--report', action='store_true',
                   help='Nur den Bericht aus der bestehenden Studie schreiben.')
    a = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    storage = f"sqlite:///{os.path.join(OUT_DIR, 'study.db').replace(os.sep, '/')}"
    # Ressourcenachse = insgesamt gerechnete Runden ueber alle Seeds.
    max_resource = a.n_max * a.seeds
    study = optuna.create_study(
        study_name=a.study, storage=storage, load_if_exists=True,
        direction='minimize',
        # `n_startup_trials` ist hier doppelt gebucht und das mit Absicht:
        # die ersten `a.random_startup` Versuche zieht TPE rein zufaellig.
        # Das *ist* die Zufallssuche-Kontrolle — sie kostet keine
        # zusaetzliche Zeit, weil TPE diese Versuche anschliessend ohnehin
        # als Startmenge braucht. Der Vergleich "beste 50 Zufallsversuche
        # gegen den Rest" beantwortet damit gratis, ob TPE seinen Aufwand
        # verdient hat.
        sampler=optuna.samplers.TPESampler(
            seed=a.seed, multivariate=True, group=True,
            n_startup_trials=a.random_startup),
        # Hyperband statt Median: laut Benchmarks die beste Paarung mit TPE,
        # und der Median-Pruner hat in Studie 1 nur 31 % erwischt.
        # min_resource=4 statt 1 ist eine bewusste Bremse: bei sigma in der
        # Groessenordnung des Fortschritts wuerde eine Entscheidung nach
        # einer einzigen Runde gute Einstellungen wegen eines unguenstigen
        # Seeds verwerfen.
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=a.min_resource, max_resource=max_resource,
            reduction_factor=3))

    if a.report:
        write_report(study, OUT_DIR)
        return

    _clear_stale(study)
    _install_pause_handler()

    done = len([t for t in study.get_trials(deepcopy=False)
                if t.state.is_finished()])
    print(f"\n{'=' * 70}")
    print(f"  Optuna-Suche  study={a.study}  space={a.space}")
    print(f"  bereits in der Datenbank: {done} Versuche")
    print(f"  dieser Aufruf: +{a.trials} Versuche, Deckel {a.timeout_h}h")
    print(f"  n_max={a.n_max}  n_shapes={a.n_shapes}  quality={a.quality}")
    print(f"  Rollout-Seeds je Versuch: {a.seeds}   Sampler-Seed: {a.seed}")
    print(f"  Hyperband: Sprossen {a.min_resource} -> ... -> {max_resource} "
         f"Runden (Faktor 3)")
    print(f"  erste {a.random_startup} Versuche zufaellig (= Kontrollgruppe)")
    print(f"  immer fest: {FIXED_ALWAYS}")
    if a.space == 'basis':
        print(f"  zusaetzlich fest: {FIXED_BASIS}")
    print(f"  Strg+C haelt nach dem laufenden Versuch an (nichts geht verloren).")
    print(f"{'=' * 70}\n")

    names, truths = M.load_holdout(resolution=a.truth_res, device=a.device,
                                   limit=a.n_shapes)
    planner = M.build_planner(ckpt=a.ckpt, device=a.device)
    print(f"  {len(names)} Formen geladen, Planer bereit\n")

    pool = None
    workers = a.workers or max(1, (os.cpu_count() or 4) - 2)
    ctx_pool = None
    try:
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            ctx_pool = ProcessPoolExecutor(max_workers=workers,
                                           initializer=M._worker_init,
                                           initargs=(a.seed,))
            pool = ctx_pool
            print(f"  SVGD-Pool mit {workers} Prozessen\n")

        t0 = time.time()
        try:
            study.optimize(make_objective(planner, truths, names, a, pool),
                           n_trials=a.trials,
                           timeout=(a.timeout_h * 3600 if a.timeout_h else None),
                           gc_after_trial=True,
                           callbacks=[_pause_callback])
        except KeyboardInterrupt:
            print("\n  hart abgebrochen.")
        dt = time.time() - t0
        if _PAUSE['flag']:
            print(f"\n  angehalten nach {dt / 60:.1f} min. Fortsetzen mit:\n"
                 f"    python -m exploration_optimierung.optuna_search "
                 f"--study {a.study} --space {a.space} --trials N --timeout_h H")
        else:
            print(f"\n  fertig nach {dt:.0f}s")
    finally:
        if ctx_pool is not None:
            ctx_pool.shutdown(wait=True)
        write_report(study, OUT_DIR)


if __name__ == '__main__':
    main()
