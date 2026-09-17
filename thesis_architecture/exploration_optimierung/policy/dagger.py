r"""
dagger.py
=========
Closes the myopic-training / multi-round-deployment gap diagnosed for
Option A (see `README.md` and the training-vs-rollout numbers this script's
docstring below refers to).

`train.py`'s cross-validated regret already beats the fixed setting
per decision (0.364 vs. 0.426 on held-out shapes), yet the full-mission
rollout in `evaluate.py` is worse than the fixed setting (J=0.306 vs. 0.281).
The reason is not a lack of signal in the 17 state features -- it is that
`oracle.py`'s dataset only ever contains states visited by the ORACLE's own
greedy rollout. A policy with any per-round approximation error then drives
itself into states that rollout never covered, and that error compounds over
the mission's rounds. This is a covariate-shift problem (DAgger's classic
setting), not a representation-capacity problem, so a richer state encoder
would not fix it either.

This script closes that loop:

    1. Train an initial Option A model on the existing oracle dataset.
    2. Let that model drive its OWN missions (on training-split shapes).
    3. At every state it actually visits, ask the oracle for the true best
       candidate among the same grid `oracle.py` used -- dense supervision,
       exactly like `oracle.py` itself, just on a different state
       distribution (`oracle.orakel_rollout`'s new `driver=` argument makes
       this possible: the policy chooses what gets driven, the oracle still
       labels every candidate).
    4. Append those rows to the dataset and retrain.
    5. Repeat for a few rounds -- across them the training distribution
       converges toward the states the deployed policy actually encounters.

Keep `--split train` here, exactly like `oracle.py`/`ppo.py`: the 25
validation shapes stay reserved for `evaluate.py` and must never appear in
this loop, or the whole point of fixing the leak is undone again.

    python -m exploration_optimierung.policy.dagger --runden 4
"""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .. import DEFAULT_CKPT, RESULTS_DIR
from .. import mission as M
from . import DATASET_CSV, FESTE_POLICY, POLICY_DIR
from .budget import Zeitbudget
from .model import WertRichtlinie
from .oracle import orakel_rollout, schreibe_datensatz
from .train import (_vorhersage, bedauern, formen_falten, lade_datensatz,
                    trainiere, wahl_fest, wahl_nach_wert)

MODELL_PT = os.path.join(POLICY_DIR, 'policy_a_dagger.pt')
BERICHT_JSON = os.path.join(RESULTS_DIR, 'policy_a_dagger.json')


def _fahrer(richtlinie):
    """Wrap a `WertRichtlinie` as a per-state driver for `orakel_rollout`.

    Returns an index into `richtlinie.kandidaten` -- callers must pass that
    exact list as `orakel_rollout`'s `kandidaten` argument too, so the index
    means the same candidate on both sides.
    """
    def waehle(z):
        with torch.no_grad():
            return int(richtlinie.werte([z]).argmin(dim=1).item())
    return waehle


def _aggregiertes_modell(csv_pfad, device, folds, epochen, breite, lr):
    """Retrain Option A on the CSV as it currently stands.

    -> (WertRichtlinie trained on ALL rows, mean cross-validated regret
    dict). Mirrors `train.py --folds ... `'s two-stage structure (per-fold
    numbers for an honest estimate, then a final model on everything) so the
    reported regret stays comparable across DAgger rounds.
    """
    X, y, gruppen, entscheidung, kandidaten = lade_datensatz(csv_pfad)
    formen = sorted(set(gruppen.tolist()))
    falten = formen_falten(gruppen, min(folds, len(formen)))
    ergebnisse = []
    for maske_val in falten:
        idx_val = np.where(maske_val)[0]
        idx_train = np.where(~maske_val)[0]
        netz, (mittel, streuung), _ = trainiere(
            X, y, idx_train, idx_val, device, epochen=epochen, breite=breite,
            lr=lr, still=True)
        vor = _vorhersage(netz, X, mittel, streuung, device)
        w_modell = wahl_nach_wert(vor, entscheidung, idx_val)
        w_fest = wahl_fest(kandidaten, entscheidung, idx_val)
        ergebnisse.append(dict(
            bedauern_modell=bedauern(y, entscheidung, w_modell),
            bedauern_fest=(bedauern(y, entscheidung, w_fest)
                          if w_fest else None)))

    alle = np.arange(X.shape[0])
    netz, (mittel, streuung), _ = trainiere(
        X, y, alle, alle, device, epochen=epochen, breite=breite, lr=lr,
        still=True)
    raster = sorted(set(tuple(k) for k in kandidaten))
    richtlinie = WertRichtlinie(netz, mittel, streuung, raster, device=device)

    mit_fest = [e['bedauern_fest'] for e in ergebnisse
               if e['bedauern_fest'] is not None]
    bedauern_falten = dict(
        modell=float(np.mean([e['bedauern_modell'] for e in ergebnisse])),
        fest=float(np.mean(mit_fest)) if mit_fest else None,
        n_zeilen=int(X.shape[0]), n_formen=len(formen))
    return richtlinie, bedauern_falten


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datensatz', default=DATASET_CSV,
                   help="starting oracle dataset; must already exist "
                        "(run `policy.oracle` first, ideally with "
                        "--split train)")
    p.add_argument('--out_datensatz', default=os.path.join(
        os.path.dirname(DATASET_CSV), 'policy_datensatz_dagger.csv'),
                   help="aggregated dataset grown across DAgger rounds; "
                        "starts as a copy of --datensatz and is never "
                        "written back into it")
    p.add_argument('--runden', type=int, default=4)
    p.add_argument('--missionen_pro_runde', type=int, default=15,
                   help="fresh self-driven missions to collect states from "
                        "per round")
    p.add_argument('--n_max', type=int, default=10)
    p.add_argument('--split', default='train', choices=['val', 'train'],
                   help="rollout shapes for collecting new states -- keep "
                        "this at 'train', same reasoning as oracle.py/ppo.py: "
                        "'val' is reserved for evaluate.py")
    p.add_argument('--n_shapes_pool', type=int, default=None,
                   help="how many split shapes to draw the per-round "
                        "sample from (default: missionen_pro_runde * "
                        "runden, so rounds overlap less)")
    p.add_argument('--plan_batch', type=int, default=128)
    p.add_argument('--zufallsmaske', action='store_true',
                   help="each shape starts with a random, spatially-coherent "
                        "region already fully known and the rest fully "
                        "unknown, instead of the fully-blind default -- see "
                        "oracle.py --zufallsmaske. Off by default.")
    p.add_argument('--unbekannt_min', type=float, default=0.5)
    p.add_argument('--unbekannt_max', type=float, default=0.9)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochen', type=int, default=200)
    p.add_argument('--breite', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--flow_steps', type=int, default=100)
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=None,
                   help="processes for SVGD; 0 = serial")
    p.add_argument('--out', default=MODELL_PT)
    p.add_argument('--bericht', default=BERICHT_JSON)
    p.add_argument('--max_minuten', type=float, default=None)
    a = p.parse_args(argv)

    if not os.path.exists(a.datensatz):
        raise SystemExit(f"{a.datensatz} not found -- run `policy.oracle` "
                         "first.")
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(os.path.dirname(a.out_datensatz), exist_ok=True)
    if os.path.abspath(a.out_datensatz) != os.path.abspath(a.datensatz):
        shutil.copyfile(a.datensatz, a.out_datensatz)
    print(f"Aggregierter Datensatz: {a.out_datensatz} "
          f"(Start: Kopie von {a.datensatz})")

    planner = M.build_planner(ckpt=a.ckpt, device=device,
                              flow_steps=a.flow_steps)
    pool_groesse = a.n_shapes_pool or (a.missionen_pro_runde * a.runden)
    names_pool, truths_pool = M.load_holdout(
        resolution=96, device=device, limit=pool_groesse, split=a.split)
    print(f"Formen-Pool ({len(names_pool)}, Split '{a.split}'): "
          f"{a.missionen_pro_runde} je Runde ueber {a.runden} Runden.")

    args = M.build_mission_args(device, phi_model=FESTE_POLICY['phi_model'],
                                param=FESTE_POLICY['param'])
    unbekannt_bereich = ((a.unbekannt_min, a.unbekannt_max)
                        if a.zufallsmaske else None)
    n_workers = a.workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 2)
    pool = None
    if n_workers > 1:
        pool = ProcessPoolExecutor(max_workers=n_workers,
                                   initializer=M._worker_init, initargs=(0,))

    rng = np.random.default_rng(a.seed)
    budget = Zeitbudget(a.max_minuten, name='DAgger')
    verlauf = []
    bedauern_falten = None
    t0 = time.perf_counter()
    try:
        for runde in range(a.runden):
            if budget.abgelaufen():
                print(f"\nAbbruch nach {runde} von {a.runden} Runden "
                      f"({budget.grund()}) -- der Datensatz bis hierher "
                      "bleibt gueltig.")
                break
            print(f"\n=== DAgger-Runde {runde + 1}/{a.runden}  "
                  f"[{(time.perf_counter() - t0) / 60:.1f} min] ===")
            richtlinie, bedauern_falten = _aggregiertes_modell(
                a.out_datensatz, device, a.folds, a.epochen, a.breite, a.lr)
            fest_txt = (f"{bedauern_falten['fest']:.4f}"
                       if bedauern_falten['fest'] is not None else 'n/a')
            print(f"  Bedauern (Kreuzvalidierung, {bedauern_falten['n_zeilen']} "
                  f"Zeilen): Modell {bedauern_falten['modell']:.4f}  "
                  f"Fest {fest_txt}")

            k = min(a.missionen_pro_runde, len(names_pool))
            idx = rng.choice(len(names_pool), size=k, replace=False)
            names_r = [names_pool[i] for i in idx]
            truths_r = truths_pool[idx]

            neue_zeilen = []
            orakel_rollout(
                planner, truths_r, names_r, args, richtlinie.kandidaten,
                a.n_max, seed=a.seed + runde, pool=pool,
                plan_batch=a.plan_batch, sammler=neue_zeilen,
                driver=_fahrer(richtlinie), budget=budget,
                unbekannt_bereich=unbekannt_bereich)
            print(f"  {len(neue_zeilen)} neue Zeilen aus "
                  f"{len(names_r)} selbst gefahrenen Missionen: "
                  f"{', '.join(names_r)}")
            schreibe_datensatz(neue_zeilen, a.out_datensatz, anhaengen=True)

            verlauf.append(dict(runde=runde, bedauern_falten=bedauern_falten,
                                neue_zeilen=len(neue_zeilen), formen=names_r))
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    print("\nEndmodell auf dem vollstaendig aggregierten Datensatz ...")
    richtlinie, bedauern_falten = _aggregiertes_modell(
        a.out_datensatz, device, a.folds, a.epochen, a.breite, a.lr)
    richtlinie.speichern(a.out)
    print(f"Gespeichert -> {a.out}")

    os.makedirs(os.path.dirname(a.bericht), exist_ok=True)
    with open(a.bericht, 'w', encoding='utf-8') as f:
        json.dump(dict(konfiguration=vars(a), verlauf=verlauf,
                       bedauern_final=bedauern_falten,
                       sekunden=time.perf_counter() - t0), f, indent=2,
                  ensure_ascii=False, default=str)
    print(f"Bericht gespeichert -> {a.bericht}")


if __name__ == '__main__':
    main()
