r"""
exploration_optimierung.policy
==============================
Gelernte Regler fuer die Laengeneinheit-Mission — die beiden Optionen A und B
aus der Auswertung der Rastersuche.

Ausgangslage
------------
`optimize.py` hat *eine* feste Betriebseinstellung gesucht und gefunden
(niveau, tau = 0,61, 25 SVGD-Iterationen, n = 6; J = 0,258 ueber die volle
Holdout-Menge). Fest heisst: dieselbe Einstellung in jeder Runde und fuer jede
Form. Die Frage dieses Pakets ist, ob eine Einstellung, die sich **nach dem
Glaubenszustand richtet**, besser ist.

Drei Regler, ein Simulator
--------------------------
Alle drei fahren dieselbe Mission (`mission.LaengenMission`) und werden mit
derselben Zielfunktion (`objective.J`) gemessen; sie unterscheiden sich nur
darin, wer die Aktion `(phi_modell, param, svgd_iters)` je Runde liefert:

    fest      `interactive_sim.OPTIMAL_POLICY['mit_svgd']` — der Bezugswert.
    orakel    `oracle.py`: probiert jede Runde alle Kandidaten *wirklich* aus
              und committet den besten. Kein einsetzbarer Regler (K-mal so
              teuer), sondern die **Obergrenze** dessen, was ein rein
              reaktiver Regler erreichen kann — ohne sie ist nicht zu
              beurteilen, ob eine gelernte Richtlinie gut oder nur besser als
              nichts ist.
    gelernt   Option A (`model.WertRichtlinie`, ueberwacht auf den
              Orakel-Entscheidungen trainiert) und Option B
              (`ppo.HybridPolitik`, per PPO im Missionsloop trainiert, mit
              Verhaltensklonen aus denselben Orakeldaten als Startpunkt).

Reihenfolge der Skripte
-----------------------
    python -m exploration_optimierung.policy.oracle    --n_shapes 25 --seeds 2
    python -m exploration_optimierung.policy.train     # Option A
    python -m exploration_optimierung.policy.ppo       # Option B (BC + PPO)
    python -m exploration_optimierung.policy.evaluate  # alle Regler, volle Holdout-Menge

Was der Regler sehen darf
-------------------------
Die Merkmale in `features.py` sind **ohne die Wahrheit** berechenbar: GP-
Mittelwert, GP-Unsicherheit, Besuchsdichte, gefahrene Bahn, Rundennummer. Die
wahre Dichte geht nur in zwei Dinge ein: in die Auswertung (cov, J) und — bei
Option B — in die Belohnung waehrend des Trainings. Das ist der uebliche
asymmetrische Aufbau (privilegierte Belohnung, unprivilegierte Beobachtung):
der Simulator darf mehr wissen als der Agent, der Agent traegt aber nichts
davon in die Anwendung hinueber.
"""

import os

from .. import ARCH_DIR, RESULTS_DIR, ROOT_DIR, DEFAULT_CKPT  # noqa: F401

#: Ablage der gelernten Regler. Getrennt von `results/`, weil dort die
#: Messwerte der Studien liegen und hier Modellgewichte — zwei verschiedene
#: Dinge mit verschiedener Lebensdauer.
POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ablage')

#: Datensatz der Orakel-Entscheidungen (Option A und das Vortraining von B).
DATASET_CSV = os.path.join(RESULTS_DIR, 'policy_datensatz.csv')

#: Die feste Vergleichseinstellung, identisch mit
#: `exploration/interactive_sim.OPTIMAL_POLICY['mit_svgd']`.
FESTE_POLICY = dict(phi_model='niveau', param=0.6067, svgd_iters=25, n_exec=6)
