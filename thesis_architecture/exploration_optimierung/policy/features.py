r"""
features.py
===========
Was ein Regler sehen darf — und in welcher Zahlenform.

Der Zustand einer Runde besteht aus drei Feldern auf dem GP-Gitter (Mittelwert
`mu`, Unsicherheit `sd`, altersgewichtete Besuchsdichte `visit`), der bisher
gefahrenen Bahn und der Rundennummer. Ein Netz koennte darauf direkt arbeiten
(64x64x3 als Bild); hier stehen stattdessen **17 Kennzahlen**, aus drei
Gruenden:

1. **Datenmenge.** Der Orakel-Datensatz hat rund 20 000 Entscheidungen aus 25
   Formen. Ein Faltungsnetz auf 12 288 Eingangswerten wuerde daraus die Formen
   auswendig lernen, nicht die Regel.
2. **Interpretierbarkeit.** Mit benannten Merkmalen laesst sich hinterher
   sagen, *warum* die Richtlinie so entscheidet (Permutationswichtigkeit in
   `train.py`) — bei einem Bild-Encoder waere das eine eigene Untersuchung.
3. **Gleiche Eingabe fuer A und B.** Die ueberwachte Richtlinie und der
   RL-Agent bekommen denselben Vektor; ein Unterschied im Ergebnis ist damit
   ein Unterschied der Lernverfahren, nicht der Wahrnehmung.

Keines der Merkmale benutzt die wahre Dichte. Das ist keine Stilfrage: die
ganze Aufgabe besteht darin, ohne sie auszukommen. `glaube_cov` etwa ist die
Abdeckung der Bahn **gegen den Glauben** `mu_hat`, nicht gegen die Wahrheit —
dieselbe Trennung, die `greedy_per_round.run_greedy` bei der Auswahl macht.

Die Aktionsmerkmale beschreiben eine *Einstellung* (Modell, Regler,
SVGD-Budget) so, dass ein Modell ueber Einstellungen hinweg verallgemeinern
kann: das Modell als One-Hot, der freie Parameter auf [0,1] normiert **in der
Skala, in der er gemeint ist** (kappa logarithmisch, w und tau linear — genau
wie `mission.param_grid`), das SVGD-Budget logarithmisch, weil der Unterschied
zwischen 0 und 25 Iterationen groesser ist als der zwischen 200 und 400.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..mission import LENGTH_UNIT, PARAM_RANGE, PHI_MODELS

from common.metrics import coverage_vs_truth, path_length     # noqa: E402


#: Reihenfolge ist Teil der Schnittstelle: die gespeicherten Modelle in
#: `ablage/` haengen an ihr. Wer ein Merkmal einfuegt, muss die Modelle neu
#: trainieren — deshalb steht die Liste hier und nicht verstreut im Code.
ZUSTANDS_MERKMALE = [
    'runde',              # bisherige Ausfuehrungen, auf n_max bezogen
    'rest',               # verbleibendes Budget, auf n_max bezogen
    'mu_masse',           # mittlerer Glaubenswert (wie viel wurde gefunden)
    'mu_max',             # staerkster Glaubenswert
    'mu_konzentration',   # Partizipationsverhaeltnis: 1 = flach, ~0 = punktuell
    'mu_traeger',         # Anteil Zellen ueber einem Viertel des Maximums
    'sd_mittel',          # mittlere Restunsicherheit
    'sd_q10',             # 10-%-Quantil der Unsicherheit (schon Bekanntes)
    'sd_q90',             # 90-%-Quantil (noch voellig Unbekanntes)
    'besucht_anteil',     # Anteil Zellen mit mehr als halber Besuchssaettigung
    'besucht_mittel',     # mittlere gesaettigte Besuchsdichte
    'glaube_cov',         # Abdeckung der Bahn GEGEN DEN GLAUBEN, bezogen
    'pos_x', 'pos_y',     # aktuelle Position
    'pos_zu_schwerpunkt',  # Abstand zum Schwerpunkt des Glaubens
    'weg_laenge',         # gefahrene Strecke in Laengeneinheiten, auf n_max bezogen
    'n_obs',              # Zahl der Messungen, auf das Budget bezogen
]

#: Reihenfolge der One-Hot-Spalten der Modelle.
MODELL_ORDNUNG = ['ucb', 'eid', 'mass', 'niveau']

AKTIONS_MERKMALE = ([f'ist_{m}' for m in MODELL_ORDNUNG]
                    + ['param_norm', 'svgd_norm'])

MERKMALE = ZUSTANDS_MERKMALE + AKTIONS_MERKMALE

#: Bezugswert der SVGD-Normierung. 400 ist der groesste Wert des Rasters der
#: Studie (`optimize.SVGD_GRID`).
SVGD_MAX = 400.0


@dataclass
class Zustand:
    """Die Sicht des Reglers auf eine Form in einer Runde.

    Haelt die rohen Felder (fuer eine spaetere Bild-Variante und fuer die
    Kandidatenbewertung im Orakel) *und* liefert daraus den Merkmalsvektor.
    """
    mu: torch.Tensor            # (R, R) GP-Mittelwert
    sd: torch.Tensor            # (R, R) GP-Standardabweichung
    visit: torch.Tensor         # (R, R) altersgewichtete Besuchsdichte oder None
    driven: torch.Tensor        # (T, 2) bisher gefahrene Bahn oder None
    runde: int                  # 0-basiert: wie viele Ausfuehrungen schon liefen
    n_max: int                  # geplante Gesamtzahl der Ausfuehrungen
    n_obs: int = 0
    name: str = ''

    def mu_hat(self):
        """Der Glaube als normierte Dichte — die 'Zielverteilung nach heutigem
        Wissensstand', gegen die online gemessen werden darf."""
        m = self.mu.clamp(min=0.0)
        return m / m.sum().clamp(min=1e-12)

    def position(self):
        if self.driven is None:
            return torch.tensor([0.5, 0.5], device=self.mu.device)
        return self.driven[-1]

    def merkmale(self):
        return zustands_merkmale(self)


def _quantil(t, q):
    # torch.quantile bricht bei sehr grossen Tensoren ab; das Gitter ist mit
    # 64x64 = 4096 Werten klein genug, aber die Umwandlung kostet nichts.
    return float(torch.quantile(t.reshape(-1), q))


def zustands_merkmale(z):
    """`Zustand` -> np.ndarray (len(ZUSTANDS_MERKMALE),), float32.

    Alle Groessen sind entweder Anteile oder auf eine feste Skala bezogen,
    damit sie ueber Formen und Missionslaengen hinweg vergleichbar bleiben —
    ein Merkmal, das mit `n_max` waechst, waere sonst ein verstecktes Mass
    fuer die Missionslaenge statt fuer den Zustand.
    """
    mu = z.mu
    sd = z.sd
    n_max = max(int(z.n_max), 1)
    N = mu.numel()

    mu_pos = mu.clamp(min=0.0)
    summe = float(mu_pos.sum())
    mu_max = float(mu_pos.max())
    if summe > 1e-12:
        p = mu_pos / summe
        # Partizipationsverhaeltnis: 1/(N * sum p^2). 1,0 heisst gleichverteilt
        # ueber das ganze Gitter, kleine Werte heissen "die Masse sitzt auf
        # wenigen Zellen" — ein Mass dafuer, wie sehr sich die Form schon
        # herausgeschaelt hat.
        konz = float(1.0 / (N * float((p ** 2).sum()) + 1e-12))
    else:
        konz = 1.0
    traeger = (float((mu_pos > 0.25 * mu_max).float().mean())
               if mu_max > 1e-9 else 0.0)

    if z.visit is None:
        v_anteil, v_mittel = 0.0, 0.0
    else:
        v = (z.visit / z.visit.max().clamp(min=1e-12)).clamp(0.0, 1.0)
        v_anteil = float((v > 0.5).float().mean())
        v_mittel = float(v.mean())

    if z.driven is None:
        glaube_cov, weg = 1.0, 0.0
    else:
        mh = z.mu_hat()
        mitte = torch.tensor([[0.5, 0.5]], device=mh.device, dtype=torch.float32)
        blind = float(coverage_vs_truth(mitte, mh))
        glaube_cov = float(coverage_vs_truth(z.driven, mh)) / max(blind, 1e-12)
        weg = float(path_length(z.driven)) / LENGTH_UNIT / n_max

    pos = z.position()
    # Schwerpunkt des Glaubens auf [0,1]^2; ohne Glauben die Bildmitte.
    R = mu.shape[-1]
    achse = torch.linspace(0, 1, R, device=mu.device)
    if summe > 1e-12:
        p = mu_pos / summe
        cx = float((p.sum(dim=0) * achse).sum())
        cy = float((p.sum(dim=1) * achse).sum())
    else:
        cx, cy = 0.5, 0.5
    d_schwer = math.hypot(float(pos[0]) - cx, float(pos[1]) - cy)

    werte = [
        z.runde / n_max,
        (n_max - z.runde) / n_max,
        summe / N,
        mu_max,
        konz,
        traeger,
        float(sd.mean()),
        _quantil(sd, 0.1),
        _quantil(sd, 0.9),
        v_anteil,
        v_mittel,
        glaube_cov,
        float(pos[0]), float(pos[1]),
        d_schwer,
        weg,
        z.n_obs / (64.0 * n_max),
    ]
    return np.asarray(werte, dtype=np.float32)


def param_norm(modell, param):
    """Freier Parameter -> [0,1], in der Skala, in der er gemeint ist.

    kappa logarithmisch (der Schritt 0,3 -> 0,6 ist derselbe Eingriff wie
    3 -> 6), w und tau linear. Dieselbe Begruendung wie bei
    `mission.param_grid`, und dieselbe Umkehrung steht in `param_von_norm`.
    """
    pname, _ = PHI_MODELS[modell]
    lo, hi = PARAM_RANGE[pname]
    x = float(min(max(float(param), lo), hi))
    if pname == 'kappa':
        return (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (x - lo) / (hi - lo)


def param_von_norm(modell, u):
    """Umkehrung von `param_norm`; `u` wird auf [0,1] beschnitten."""
    pname, _ = PHI_MODELS[modell]
    lo, hi = PARAM_RANGE[pname]
    u = float(min(max(float(u), 0.0), 1.0))
    if pname == 'kappa':
        return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
    return float(lo + u * (hi - lo))


def svgd_norm(iters):
    """SVGD-Budget -> [0,1], logarithmisch.

    log1p, damit 0 (keine Verfeinerung) ein echter Punkt der Skala ist und
    nicht ins Unendliche faellt; der Abstand 0 -> 25 ist damit rund so gross
    wie 25 -> 400, was der gemessenen Wirkung entspricht (Lehre 2 der Studie:
    die ersten 25 Iterationen bringen fast alles).
    """
    return math.log1p(max(0.0, float(iters))) / math.log1p(SVGD_MAX)


def aktions_merkmale(aktion):
    """`(modell, param, svgd_iters)` -> np.ndarray (len(AKTIONS_MERKMALE),)."""
    modell, param, svgd = aktion
    hot = [1.0 if modell == m else 0.0 for m in MODELL_ORDNUNG]
    return np.asarray(hot + [param_norm(modell, param), svgd_norm(svgd)],
                      dtype=np.float32)


def paar_merkmale(zustand_vec, aktion):
    """Zustands- und Aktionsmerkmale zu einer Zeile zusammensetzen."""
    return np.concatenate([np.asarray(zustand_vec, dtype=np.float32),
                           aktions_merkmale(aktion)])


def aktionsraster(modelle=None, param_punkte=4, svgd_buckets=(0, 25, 100)):
    """Der diskrete Kandidatenraum: Kreuzprodukt Modell x Parameter x SVGD.

    Dasselbe Raster benutzen das Orakel (es probiert alle Kandidaten aus), die
    ueberwachte Richtlinie (sie bewertet alle und nimmt den besten) und die
    Auswertung. Ein gemeinsames Raster ist Bedingung dafuer, dass die drei
    Zahlen ueberhaupt vergleichbar sind.
    """
    from ..mission import param_grid
    modelle = list(modelle or MODELL_ORDNUNG)
    raster = []
    for m in modelle:
        pname, _ = PHI_MODELS[m]
        for p in param_grid(pname, param_punkte):
            for s in svgd_buckets:
                raster.append((m, float(p), int(s)))
    return raster
