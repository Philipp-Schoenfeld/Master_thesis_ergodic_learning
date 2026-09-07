r"""
train.py
========
Option A trainieren: das Wertmodell aus `model.py` auf den Orakel-Entscheidungen.

Was gemessen wird — und warum nicht der Trainingsfehler
-------------------------------------------------------
Der mittlere quadratische Fehler auf `score_norm` sagt wenig darueber, ob die
Richtlinie *gut waehlt*: eine Schaetzung darf beliebig danebenliegen, solange
sie den richtigen Kandidaten nach vorn sortiert. Berichtet wird deshalb vor
allem das **Bedauern** (Regret) auf der Validierungsmenge:

    Bedauern = score_norm des gewaehlten Kandidaten

also 0, wenn der beste Kandidat der Entscheidung getroffen wurde, und 1, wenn
der schlechteste getroffen wurde. Drei Vergleichswerte stehen daneben, sonst
ist die Zahl nicht einzuordnen:

    Orakel   0,0     (per Definition — es *ist* der beste Kandidat)
    Zufall   ~0,5    (Mittel ueber alle Kandidaten der Entscheidung)
    Fest     der Wert der Betriebseinstellung der Studie (niveau, tau = 0,61,
             25 SVGD-Iterationen), also des naechstliegenden Kandidaten im
             Raster. **Das ist die eigentliche Messlatte**: eine gelernte
             Richtlinie, die schlechter waehlt als die eine feste Einstellung,
             ist keine Verbesserung, egal wie klein ihr MSE ist.

Aufgeteilt wird **nach Formen**, nicht zufaellig
------------------------------------------------
Zeilen derselben Form sind untereinander stark abhaengig (dieselbe
Zielverteilung, aufeinanderfolgende Runden derselben Mission). Ein zufaelliger
Split haette in Training und Validierung dieselben Formen und wuerde
Auswendiglernen als Generalisierung ausweisen. Die Kreuzvalidierung laeuft
deshalb ueber Formgruppen: jede Falte haelt ganze Formen zurueck, und die
berichtete Zahl ist die auf **nie gesehenen Formen**.

Zur Einordnung fuer die Thesis: die 25 Formen sind zugleich die Holdout-Menge
des Flow-Matching-Netzes. Die hier berichteten Faltenzahlen sind damit
Generalisierung *ueber Formen* bei festem Planernetz — nicht Generalisierung
auf eine voellig unabhaengige Datenlage. Wer das strenger braucht, laesst
`oracle.py` auf den Trainingsformen des Netzes laufen und benutzt die 25
Validierungsformen ausschliesslich zum Testen (`--train_shapes`).

    python -m exploration_optimierung.policy.train
    python -m exploration_optimierung.policy.train --folds 5 --epochen 300
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .. import RESULTS_DIR
from . import DATASET_CSV, FESTE_POLICY, POLICY_DIR
from .features import (MERKMALE, ZUSTANDS_MERKMALE, aktions_merkmale,
                       param_norm, svgd_norm)
from .model import WertNetz, WertRichtlinie, normierung

MODELL_PT = os.path.join(POLICY_DIR, 'policy_a.pt')
BERICHT_JSON = os.path.join(RESULTS_DIR, 'policy_a_training.json')


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------

def lade_datensatz(pfad=DATASET_CSV):
    """CSV -> (X, y, gruppen, entscheidungen, kandidaten).

    `entscheidungen` ist die Zuordnung Zeile -> laufende Nummer der
    Entscheidung (seed, Form, Runde); alle Kandidatenzeilen einer Entscheidung
    teilen sie. Ohne diese Gruppierung liesse sich das Bedauern nicht rechnen,
    weil dafuer *alle* Kandidaten einer Lage nebeneinander liegen muessen.
    """
    with open(pfad, 'r', newline='', encoding='utf-8') as f:
        zeilen = list(csv.DictReader(f))
    if not zeilen:
        raise SystemExit(f"{pfad} ist leer — erst `policy.oracle` laufen lassen.")

    schluessel, entscheidung = {}, []
    X, y, gruppen, kand = [], [], [], []
    for z in zeilen:
        s = (z['seed'], z['shape'], z['runde'])
        if s not in schluessel:
            schluessel[s] = len(schluessel)
        entscheidung.append(schluessel[s])
        zvec = [float(z[m]) for m in ZUSTANDS_MERKMALE]
        aktion = (z['modell'], float(z['param']), int(z['svgd']))
        X.append(np.concatenate([np.asarray(zvec, dtype=np.float32),
                                 aktions_merkmale(aktion)]))
        y.append(float(z['score_norm']))
        gruppen.append(z['shape'])
        kand.append(aktion)
    return (np.stack(X).astype(np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(gruppen), np.asarray(entscheidung), kand)


def formen_falten(gruppen, k):
    """Formen auf k Falten verteilen -> Liste von Bool-Masken (Validierung)."""
    formen = sorted(set(gruppen.tolist()))
    falten = [formen[i::k] for i in range(k)]
    return [np.isin(gruppen, f) for f in falten if f]


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------

def bedauern(y, entscheidung, wahl_index):
    """Mittleres `score_norm` der gewaehlten Kandidaten.

    `wahl_index` ist ein dict Entscheidung -> Index der gewaehlten Zeile.
    """
    return float(np.mean([y[i] for i in wahl_index.values()]))


def wahl_nach_wert(vorhersage, entscheidung, indizes):
    """Je Entscheidung die Zeile mit dem kleinsten geschaetzten Wert."""
    best = {}
    for i in indizes:
        e = int(entscheidung[i])
        if e not in best or vorhersage[i] < vorhersage[best[e]]:
            best[e] = i
    return best


def wahl_fest(kandidaten, entscheidung, indizes):
    """Je Entscheidung die Zeile, die der festen Betriebseinstellung entspricht.

    Die feste Einstellung (tau = 0,6067) liegt nicht exakt auf dem groberen
    Kandidatenraster des Orakels; genommen wird der naechstliegende Kandidat
    in derselben normierten Skala, in der auch das Modell arbeitet.
    """
    ziel = np.asarray([param_norm(FESTE_POLICY['phi_model'],
                                  FESTE_POLICY['param']),
                       svgd_norm(FESTE_POLICY['svgd_iters'])])
    best = {}
    for i in indizes:
        m, p, s = kandidaten[i]
        if m != FESTE_POLICY['phi_model']:
            continue
        d = float(np.linalg.norm(np.asarray([param_norm(m, p), svgd_norm(s)])
                                 - ziel))
        e = int(entscheidung[i])
        if e not in best or d < best[e][1]:
            best[e] = (i, d)
    return {e: v[0] for e, v in best.items()}


def wichtigkeiten(netz, X, y, entscheidung, idx, mittel, streuung, device,
                  wiederholungen=3, rng=None):
    """Permutationswichtigkeit, gemessen am **Bedauern**, nicht am MSE.

    Ein Merkmal ist genau dann wichtig, wenn sein Zerstoeren die *Wahl*
    verschlechtert. Ein Merkmal, das die Schaetzung verschiebt, aber die
    Rangfolge der Kandidaten unberuehrt laesst, ist fuer diese Aufgabe
    gleichgueltig — das faengt nur eine Kennzahl ein, die auf der Wahl beruht.
    """
    rng = rng or np.random.default_rng(0)
    basis = bedauern(y, entscheidung,
                     wahl_nach_wert(_vorhersage(netz, X, mittel, streuung,
                                                device), entscheidung, idx))
    out = {}
    for j, name in enumerate(MERKMALE):
        werte = []
        for _ in range(wiederholungen):
            Xp = X.copy()
            Xp[idx, j] = Xp[rng.permutation(idx), j]
            v = _vorhersage(netz, Xp, mittel, streuung, device)
            werte.append(bedauern(y, entscheidung,
                                  wahl_nach_wert(v, entscheidung, idx)))
        out[name] = float(np.mean(werte) - basis)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _vorhersage(netz, X, mittel, streuung, device, batch=65536):
    netz.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    with torch.no_grad():
        for a in range(0, X.shape[0], batch):
            b = min(a + batch, X.shape[0])
            t = torch.as_tensor((X[a:b] - mittel) / streuung, device=device)
            out[a:b] = netz(t).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def trainiere(X, y, idx_train, idx_val, device, epochen=200, breite=64,
              lr=2e-3, batch=1024, gewichtsverfall=1e-4, still=False):
    """Ein Modell auf `idx_train`; gibt (Netz, Normierung, Verlauf) zurueck.

    Die Normierung wird **nur aus der Trainingsmenge** gebildet — sonst flosse
    ueber Mittelwert und Streuung Information der Validierungsformen ins
    Modell, und die Faltenzahlen waeren beschoenigt.
    """
    mittel, streuung = normierung(X[idx_train])
    Xt = torch.as_tensor((X[idx_train] - mittel) / streuung, device=device)
    yt = torch.as_tensor(y[idx_train], device=device)
    Xv = torch.as_tensor((X[idx_val] - mittel) / streuung, device=device)
    yv = torch.as_tensor(y[idx_val], device=device)

    netz = WertNetz(n_ein=X.shape[1], breite=breite).to(device)
    opt = torch.optim.AdamW(netz.parameters(), lr=lr,
                            weight_decay=gewichtsverfall)
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochen)
    verlust = nn.MSELoss()

    verlauf = []
    bestes, bester_val = None, float('inf')
    for ep in range(epochen):
        netz.train()
        perm = torch.randperm(Xt.shape[0], device=device)
        summe = 0.0
        for a in range(0, Xt.shape[0], batch):
            b = perm[a:a + batch]
            opt.zero_grad()
            l = verlust(netz(Xt[b]), yt[b])
            l.backward()
            opt.step()
            summe += float(l) * len(b)
        plan.step()
        netz.eval()
        with torch.no_grad():
            val = float(verlust(netz(Xv), yv)) if Xv.shape[0] else float('nan')
        verlauf.append(dict(epoche=ep, train=summe / max(Xt.shape[0], 1),
                            val=val))
        if val < bester_val:
            bester_val = val
            bestes = {k: v.detach().clone() for k, v in netz.state_dict().items()}
        if not still and (ep % 25 == 0 or ep == epochen - 1):
            print(f"    Epoche {ep:3d}  Training {verlauf[-1]['train']:.4f}  "
                  f"Validierung {val:.4f}")
    if bestes is not None:
        netz.load_state_dict(bestes)
    return netz, (mittel, streuung), verlauf


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datensatz', default=DATASET_CSV)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochen', type=int, default=200)
    p.add_argument('--breite', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--device', default=None)
    p.add_argument('--out', default=MODELL_PT)
    p.add_argument('--bericht', default=BERICHT_JSON)
    a = p.parse_args(argv)

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, gruppen, entscheidung, kandidaten = lade_datensatz(a.datensatz)
    formen = sorted(set(gruppen.tolist()))
    n_entsch = len(set(entscheidung.tolist()))
    print(f"Datensatz: {X.shape[0]} Zeilen, {n_entsch} Entscheidungen, "
          f"{len(formen)} Formen, {X.shape[1]} Merkmale  [{device}]")

    falten = formen_falten(gruppen, min(a.folds, len(formen)))
    ergebnisse = []
    for f, maske_val in enumerate(falten):
        idx_val = np.where(maske_val)[0]
        idx_train = np.where(~maske_val)[0]
        val_formen = sorted(set(gruppen[idx_val].tolist()))
        print(f"\nFalte {f + 1}/{len(falten)} — zurueckgehalten: "
              f"{', '.join(val_formen)}")
        netz, (mittel, streuung), verlauf = trainiere(
            X, y, idx_train, idx_val, device, epochen=a.epochen,
            breite=a.breite, lr=a.lr)

        vor = _vorhersage(netz, X, mittel, streuung, device)
        w_modell = wahl_nach_wert(vor, entscheidung, idx_val)
        w_fest = wahl_fest(kandidaten, entscheidung, idx_val)
        # Zufall: das Mittel ueber alle Kandidaten *ist* das erwartete
        # Bedauern einer zufaelligen Wahl, ohne dass dafuer gewuerfelt werden
        # muss.
        zufall = float(np.mean(y[idx_val]))
        treffer = float(np.mean([y[i] < 1e-9 for i in w_modell.values()]))

        rec = dict(
            falte=f, val_formen=val_formen,
            n_val=int(len(idx_val)), n_entscheidungen=len(w_modell),
            mse_val=float(np.mean((vor[idx_val] - y[idx_val]) ** 2)),
            bedauern_modell=bedauern(y, entscheidung, w_modell),
            bedauern_fest=bedauern(y, entscheidung, w_fest) if w_fest else None,
            bedauern_zufall=zufall,
            trefferquote=treffer,
            wichtigkeiten=wichtigkeiten(netz, X, y, entscheidung, idx_val,
                                        mittel, streuung, device),
            verlauf=verlauf[-1])
        ergebnisse.append(rec)
        print(f"  Bedauern  Modell {rec['bedauern_modell']:.4f}   "
              f"Fest {rec['bedauern_fest']:.4f}   "
              f"Zufall {rec['bedauern_zufall']:.4f}   "
              f"(Orakel 0,0)   Treffer {treffer:.1%}")

    mittelwerte = {
        k: float(np.mean([r[k] for r in ergebnisse]))
        for k in ('mse_val', 'bedauern_modell', 'bedauern_fest',
                  'bedauern_zufall', 'trefferquote')}
    print("\nMittel ueber die Falten:")
    print(f"  Bedauern   Modell {mittelwerte['bedauern_modell']:.4f}   "
          f"Fest {mittelwerte['bedauern_fest']:.4f}   "
          f"Zufall {mittelwerte['bedauern_zufall']:.4f}")
    besser = mittelwerte['bedauern_fest'] - mittelwerte['bedauern_modell']
    print(f"  Vorsprung vor der festen Einstellung: {besser:+.4f} "
          f"(positiv = gelernte Richtlinie waehlt besser)")

    # Endmodell auf allen Daten. Die Faltenzahlen oben gelten fuer *dieses*
    # Modell nur naeherungsweise — sie sind die Schaetzung dessen, was ein so
    # trainiertes Modell auf neuen Formen leistet, nicht eine Messung an ihm
    # selbst. Deshalb steht beides getrennt im Bericht.
    print("\nEndmodell auf allen Formen ...")
    alle = np.arange(X.shape[0])
    netz, (mittel, streuung), verlauf = trainiere(
        X, y, alle, alle, device, epochen=a.epochen, breite=a.breite, lr=a.lr,
        still=True)
    raster = sorted(set(tuple(k) for k in kandidaten))
    richtlinie = WertRichtlinie(
        netz, mittel, streuung, raster, device=device,
        meta=dict(datensatz=os.path.basename(a.datensatz),
                  n_zeilen=int(X.shape[0]), n_entscheidungen=n_entsch,
                  formen=formen, folds=len(falten),
                  bedauern_falten=mittelwerte['bedauern_modell'],
                  bedauern_fest=mittelwerte['bedauern_fest']))
    richtlinie.speichern(a.out)
    print(f"Modell gespeichert -> {a.out}")

    os.makedirs(os.path.dirname(a.bericht), exist_ok=True)
    with open(a.bericht, 'w', encoding='utf-8') as f:
        json.dump(dict(datensatz=a.datensatz, n_zeilen=int(X.shape[0]),
                       n_entscheidungen=n_entsch, formen=formen,
                       kandidaten=len(raster), falten=ergebnisse,
                       mittel=mittelwerte, endmodell_verlauf=verlauf[-1],
                       modell=a.out), f, indent=2, ensure_ascii=False)
    print(f"Bericht gespeichert -> {a.bericht}")
    return mittelwerte


if __name__ == '__main__':
    main()
