r"""
model.py
========
Option A: eine Richtlinie, die aus dem Glaubenszustand die Einstellung waehlt.

Warum ein **Wertmodell** und keine direkte Vorhersage der Aktion
----------------------------------------------------------------
Naheliegend waere, aus dem Zustand direkt die beste Einstellung vorherzusagen
(Klassifikation ueber das Modell, Regression auf den Parameter). Dagegen
sprechen zwei Dinge, die in den Orakeldaten sichtbar sind:

* Die Aktion ist **gemischt** (Modell diskret, Parameter stetig, SVGD-Budget
  diskret). Eine direkte Vorhersage braucht drei Koepfe und eine Annahme
  darueber, wie sie zusammenhaengen; ein Wertmodell braucht nur *einen*
  Ausgang und behandelt die Zusammenhaenge implizit.
* Der beste Kandidat ist oft **nicht eindeutig**. Die J-Landschaft der Studie
  ist innerhalb eines Modells flach, und dasselbe gilt fuer die
  Rundenentscheidungen: haeufig liegen mehrere Kandidaten dicht beieinander,
  und welcher davon gerade das Minimum haelt, ist zu einem guten Teil
  Rauschen. Eine Klassifikation auf den Argmin lernt genau dieses Rauschen
  mit. Ein Wertmodell lernt stattdessen die *ganze* Bewertungskurve ueber die
  Kandidaten — aus einer Entscheidung werden K Trainingszeilen statt einer,
  und flache Bereiche werden als flach gelernt statt als willkuerliche
  Praeferenz.

Das Modell schaetzt also `wert(zustand, aktion) -> Guete` (klein = gut, in der
auf die Entscheidung bezogenen Skala `score_norm` aus `oracle.py`). Gefahren
wird zur Laufzeit der Kandidat mit dem kleinsten geschaetzten Wert — dieselbe
Auswahlregel wie beim Orakel, nur mit einer Schaetzung statt einem echten
Probelauf, und damit K-mal billiger.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

from .features import (AKTIONS_MERKMALE, MERKMALE, ZUSTANDS_MERKMALE,
                       aktions_merkmale, aktionsraster, zustands_merkmale)


class WertNetz(nn.Module):
    """Kleines MLP `(Zustand, Aktion) -> geschaetzte Guete`.

    Zwei versteckte Schichten mit 64 Einheiten: rund 6 000 Parameter auf rund
    20 000 Trainingszeilen. Groesser waere bei dieser Datenmenge nicht
    sinnvoll — die k-fache Kreuzvalidierung in `train.py` zeigt die Luecke
    zwischen Trainings- und Validierungsfehler, und sie oeffnet sich schon bei
    dieser Groesse sichtbar.
    """

    def __init__(self, n_ein=len(MERKMALE), breite=64):
        super().__init__()
        self.netz = nn.Sequential(
            nn.Linear(n_ein, breite), nn.SiLU(),
            nn.Linear(breite, breite), nn.SiLU(),
            nn.Linear(breite, 1))

    def forward(self, x):
        return self.netz(x).squeeze(-1)


class WertRichtlinie:
    """Das trainierte Wertmodell als Regler fuer `mission.LaengenMission`.

    Aufrufbar als `policy(r, zustaende) -> [aktion, ...]`, also genau die
    Schnittstelle, die der Simulator erwartet. Damit laeuft die gelernte
    Richtlinie durch dieselbe Mission wie die feste Einstellung und das
    Orakel; unterschiedliche Ergebnisse sind Unterschiede der Entscheidung,
    nicht des Aufbaus.
    """

    def __init__(self, netz, mittel, streuung, kandidaten, device='cpu',
                 meta=None):
        self.netz = netz.to(device).eval()
        self.device = torch.device(device)
        self.mittel = torch.as_tensor(mittel, dtype=torch.float32,
                                      device=self.device)
        self.streuung = torch.as_tensor(streuung, dtype=torch.float32,
                                        device=self.device)
        self.kandidaten = [tuple(k) for k in kandidaten]
        self.meta = meta or {}
        # Aktionsmerkmale haengen nicht vom Zustand ab und werden einmal
        # gebaut statt in jeder Runde neu.
        self._akt = torch.as_tensor(
            np.stack([aktions_merkmale(k) for k in self.kandidaten]),
            dtype=torch.float32, device=self.device)
        self.letzte_wahl = []

    # -- Auswertung ---------------------------------------------------------
    def werte(self, zustaende):
        """-> (S, K) geschaetzte Guete je Form und Kandidat."""
        S, K = len(zustaende), len(self.kandidaten)
        z = torch.as_tensor(np.stack([zustands_merkmale(x) for x in zustaende]),
                            dtype=torch.float32, device=self.device)
        x = torch.cat([z.unsqueeze(1).expand(S, K, z.shape[-1]),
                       self._akt.unsqueeze(0).expand(S, K, self._akt.shape[-1])],
                      dim=-1)
        x = (x - self.mittel) / self.streuung
        with torch.no_grad():
            return self.netz(x.reshape(S * K, -1)).reshape(S, K)

    def __call__(self, r, zustaende):
        w = self.werte(zustaende)
        idx = w.argmin(dim=1).tolist()
        self.letzte_wahl = [self.kandidaten[i] for i in idx]
        return list(self.letzte_wahl)

    # -- Ablage -------------------------------------------------------------
    def speichern(self, pfad):
        os.makedirs(os.path.dirname(pfad), exist_ok=True)
        torch.save(dict(
            state_dict=self.netz.state_dict(),
            mittel=self.mittel.cpu().numpy(),
            streuung=self.streuung.cpu().numpy(),
            kandidaten=[list(k) for k in self.kandidaten],
            merkmale=MERKMALE,
            zustands_merkmale=ZUSTANDS_MERKMALE,
            aktions_merkmale=AKTIONS_MERKMALE,
            meta=self.meta), pfad)
        return pfad

    @classmethod
    def laden(cls, pfad, device='cpu'):
        ck = torch.load(pfad, map_location=device, weights_only=False)
        if list(ck.get('merkmale', MERKMALE)) != list(MERKMALE):
            raise ValueError(
                f"{os.path.basename(pfad)} wurde mit einer anderen "
                "Merkmalsliste trainiert — `features.MERKMALE` hat sich "
                "geaendert, das Modell muss neu trainiert werden.")
        netz = WertNetz(n_ein=len(ck['merkmale']))
        netz.load_state_dict(ck['state_dict'])
        return cls(netz, ck['mittel'], ck['streuung'], ck['kandidaten'],
                   device=device, meta=ck.get('meta', {}))


class FesteRichtlinie:
    """Die Betriebseinstellung der Studie als Regler — der Bezugswert.

    Technisch ueberfluessig (ohne `policy` faehrt die Mission dasselbe), aber
    die Auswertung vergleicht damit alle Regler ueber genau denselben Pfad
    durch den Simulator, statt einen davon als Sonderfall zu behandeln.
    """

    def __init__(self, modell, param, svgd_iters):
        self.aktion = (modell, float(param), int(svgd_iters))

    def __call__(self, r, zustaende):
        return [self.aktion] * len(zustaende)


class OrakelRichtlinie:
    """Das gierige Orakel als Regler — die Obergrenze im selben Rahmen.

    Probiert je Runde alle Kandidaten wirklich durch (`oracle.bewerte_-
    kandidaten`) und gibt den besten zurueck. Teuer und nicht einsetzbar; hier,
    damit die Auswertung alle vier Regler ueber dieselbe Schleife fahren kann.

    Achtung: die Mission plant den zurueckgegebenen Kandidaten anschliessend
    **erneut**, und die Flow-ODE ist stochastisch — die gefahrene Bahn ist
    also nicht Bit fuer Bit die geprobte. Das ist der ehrlichere Vergleich:
    bewertet wird die *Einstellung*, nicht ein einzelner gluecklicher Zug.
    Wer die geprobte Bahn selbst committen will, nimmt `oracle.orakel_rollout`.
    """

    def __init__(self, kandidaten=None, plan_batch=128):
        self.kandidaten = [tuple(k) for k in (kandidaten or aktionsraster())]
        self.plan_batch = plan_batch
        self.mission = None          # von `evaluate.py` gesetzt
        self.letzte_wahl = []

    def __call__(self, r, zustaende):
        from .oracle import bewerte_kandidaten
        if self.mission is None:
            raise RuntimeError("OrakelRichtlinie braucht `.mission`, weil sie "
                               "den Planer und die Missionsparameter benutzt.")
        _segs, scores = bewerte_kandidaten(self.mission, zustaende,
                                           self.kandidaten,
                                           plan_batch=self.plan_batch)
        self.letzte_wahl = [self.kandidaten[int(np.argmin(s))] for s in scores]
        return list(self.letzte_wahl)


def normierung(X):
    """Mittelwert und Streuung je Spalte; konstante Spalten bekommen 1.0.

    One-Hot-Spalten eines nur einmal vorkommenden Modells waeren sonst durch
    null geteilt.
    """
    mittel = X.mean(axis=0)
    streuung = X.std(axis=0)
    streuung[streuung < 1e-6] = 1.0
    return mittel.astype(np.float32), streuung.astype(np.float32)


def lade_bericht(pfad):
    with open(pfad, 'r', encoding='utf-8') as f:
        return json.load(f)
