r"""
mission.py
==========
Die Laengeneinheit-Mission: planen, *eine* Laengeneinheit fahren, neu planen.

Der Ablauf einer Runde
----------------------
1. Aus dem GP-Glauben wird die Zieldichte Phi gebildet — mit Abdeckungsschuld,
   also `apply_cfm_belief.debt_density`. Sie leistet genau das, worum es hier
   geht:

   * **Erkundet und nichts gefunden -> faellt auf null.** Dort ist mu = 0 und
     sigma nach dem Besuch fast null; `sd_eff = sd * (1 - v)` loescht den Rest.
     Der Ort zieht nicht mehr.
   * **Erkundet und etwas gefunden -> faellt nur anteilig ab.**
     `mu_eff = mu * (1 - w * v * (1 - sigma))` senkt die Anziehung um den
     Anteil `w` (`--debt_weight`), nicht auf null. Was bleibt, ist weiterhin
     *proportional zu mu* — ein starkes Gebiet bleibt also attraktiver als ein
     schwaches und wird oefter wieder angefahren, genau in dem Verhaeltnis, in
     dem seine Dichte steht.
   * **Die Absenkung erholt sich mit der Zeit.** Die Aufenthaltsdichte, die
     `debt_density` bekommt, ist hier nach dem *Alter* eines Besuchs gewichtet
     (`visitation_recent`, Halbwertszeit `--visit_halflife` in
     Laengeneinheiten). Ein eben befahrenes Gebiet ist gesperrt, ein vor
     sechs Ausfuehrungen befahrenes fast wieder frei. Ohne diese Gewichtung
     gibt es keine Rueckkehr: `debt_density` saettigt bei einem Viertel des
     Besuchsmaximums, und ein einmal befahrenes Gebiet bleibt bis zum Ende der
     Mission ueber dieser Schwelle. Der Selbsttest haelt genau das fest.

2. Aus Phi wird eine Partikelwolke gezogen, das trainierte, *startpunkt-*
   konditionierte Netz plant daraus eine vollstaendige Bahn ab der aktuellen
   Position.
3. Optional verfeinert SVGD diese Bahn gegen dieselbe Runden-Zieldichte.
4. Von dieser Bahn wird **genau eine Laengeneinheit** abgefahren. Eine
   Laengeneinheit ist die Diagonale der Zieldomaene, also sqrt(2) auf [0,1]^2.
5. Entlang des gefahrenen Stuecks wird gemessen, der Glaube fortgeschrieben,
   und es geht zurueck zu 1.

Warum alle Formen gleichzeitig laufen
-------------------------------------
Der teuerste Schritt ist die Flow-ODE des Netzes, und die kostet fuer S Formen
in *einem* Aufruf spuerbar weniger als S Aufrufe (auf der RTX 2070 SUPER
gemessen: 25 Formen in 7,5 s statt 25 x 0,72 s = 18 s). `generate_particle_-
trajectories` nimmt eine vorgebatchte Wolke `(S, N, 3)` und `S` Startpunkte
entgegen, Zeile i gehoert zu Wolke i. Deshalb laufen hier alle Holdout-Formen
in einer Runde im Gleichschritt statt nacheinander.

Warum eine Rolloutspur und keine feste Rundenzahl
-------------------------------------------------
Eine Mission ueber `n` Runden enthaelt die Missionen ueber 1..n-1 Runden als
Praefixe. Die Zahl der Ausfuehrungen muss deshalb **nicht** gesucht werden: ein
einziger Rollout bis `n_max` liefert die Guete fuer *jede* Rundenzahl. Die Suche
in `optimize.py` bezahlt damit nur fuer die uebrigen Parameter, und `n` faellt
als Nebenprodukt ab. Das ist der Grund, warum die ganze Studie auf einem
Arbeitsplatzrechner in Stunden statt in Tagen laeuft.
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from . import DEFAULT_CKPT  # noqa: F401  (Pfad-Bootstrap)

import apply_cfm_belief as acb                                   # noqa: E402
from common.belief import GPBelief                               # noqa: E402
from common.data import load_truth                               # noqa: E402
from common.metrics import (coverage_vs_truth, belief_rmse,      # noqa: E402
                            path_length, trim_to_length)
from common.observation import measure, thin                     # noqa: E402


#: Eine Laengeneinheit = die Diagonale der Zieldomaene [0,1]^2.
LENGTH_UNIT = math.sqrt(2.0)

#: Stuetzpunkte je Laengeneinheit auf dem abgefahrenen Stueck. Die geplanten
#: Bahnen sind unterschiedlich lang, ein Zuschnitt auf eine feste Weglaenge
#: liefert deshalb je nach Runde 50 bis 110 der urspruenglichen Punkte. Ohne
#: Neuabtastung haenge damit die Messdichte — und ueber sie der Glaube — an
#: einer Groesse, die mit der Sache nichts zu tun hat.
PTS_PER_UNIT = 72

#: Die vier Zieldichte-Modelle der Studie und der jeweils *eine* freie
#: Parameter, den sie haben. Die Namen sind die der GUI (`interactive_sim.PHI_UI`),
#: die Werte die internen aus `common/acquisition.py`.
PHI_MODELS = {
    'ucb':    ('kappa', 'ucb'),    # Phi = mu + kappa*sigma
    'eid':    ('kappa', 'eid'),    # Phi = mu_hat + kappa*EID_hat
    'mass':   ('w',     'mass'),   # Phi = (1-w)*mu_hat + w*sigma_hat
    'niveau': ('tau',   'lse'),    # Phi = P(f > tau)
}

#: Suchbereiche der freien Parameter. Bewusst grosszuegig — die Verfeinerung
#: in `optimize.py` arbeitet innerhalb dieser Grenzen weiter.
PARAM_RANGE = {
    'kappa': (0.15, 10.0),
    'w':     (0.05, 0.95),
    'tau':   (0.02, 0.90),
}

#: Grobraster je Parameter. `kappa` logarithmisch, weil der Unterschied
#: zwischen 0,3 und 0,6 dieselbe Bedeutung hat wie der zwischen 3 und 6;
#: `w` und `tau` linear, weil beide echte Anteile bzw. Schwellen sind.
PARAM_GRID = {
    'kappa': [0.25, 0.5, 1.0, 2.0, 3.5, 6.0, 9.0],
    'w':     [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95],
    'tau':   [0.05, 0.12, 0.20, 0.30, 0.45, 0.60, 0.80],
}


def param_grid(pname, n_points=None):
    r"""Das Raster eines Parameters — von Hand gesetzt oder gleichmaessig erzeugt.

    Ohne `n_points` kommt das handgesetzte Raster aus `PARAM_GRID` zurueck: sieben
    Punkte, die dort dichter stehen, wo die Erfahrung aus den Probelaeufen das
    Optimum vermuten laesst.

    Mit `n_points` wird ueber `PARAM_RANGE` gleichmaessig erzeugt, und zwar in der
    Skala, in der der Parameter *gemeint* ist:

    * `kappa` **logarithmisch**. Der Regler gewichtet Unsicherheit gegen bereits
      Gefundenes; der Unterschied zwischen 0,3 und 0,6 ist derselbe Eingriff wie
      der zwischen 3 und 6. Ein lineares Raster ueber [0,15, 10] verbraeuchte
      zwei Drittel seiner Punkte oberhalb von 3, wo die Zieldichte laengst reine
      Unsicherheit ist und sich nichts mehr aendert.
    * `w` und `tau` **linear**. Beide sind echte Anteile bzw. Schwellen auf einer
      beschraenkten Skala, auf der jeder Abschnitt gleich viel bedeutet.

    Die handgesetzten Raster sind Teilmengen der erzeugten nur zufaellig — wer
    `--param_points` setzt, rechnet die sieben Punkte also neu. Der
    Zwischenspeicher greift dann nicht, was er auch nicht soll: es sind andere
    Parameterwerte.
    """
    if n_points is None:
        return list(PARAM_GRID[pname])
    lo, hi = PARAM_RANGE[pname]
    n = max(2, int(n_points))
    if pname == 'kappa':
        vals = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    else:
        vals = np.linspace(lo, hi, n)
    return [round(float(v), 4) for v in vals]


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

def build_mission_args(device, phi_model='ucb', param=3.0, debt_weight=0.6,
                       visit_sat=1.0, sensor_radius=0.06, gp_noise=0.05,
                       n_particles=256, meas_noise=0.02, max_obs=64,
                       visit_halflife=3.0, phi_mode='uniform'):
    r"""Das Namensobjekt, das `apply_cfm_belief.zieldichte`/`debt_density` liest.

    Formgleich mit `interactive_sim.build_args`, damit GUI und Batch-Suche
    denselben Zieldichte-Code fahren. Der einzige Unterschied ist die
    Uebersetzung des *freien Parameters* in das, was `zieldichte` erwartet:

        ucb, eid   `param` ist kappa und geht direkt durch.
        mass       `zieldichte` rechnet intern w = kappa/(1+kappa). Damit der
                   eingestellte Anteil `w` exakt ankommt und nicht ueber die
                   verzerrte kappa-Skala, wird hier umgekehrt gerechnet:
                   kappa = w/(1-w). Dieselbe Umrechnung wie im 'mass'-Zweig
                   der GUI.
        niveau     `zieldichte` benutzt fuer 'lse' allein `phi_tau`; kappa
                   wirkt dort nicht und bleibt auf einem beliebigen Wert.

    Ergebnis ist also: *ein* Regler je Modell, und zwar der, den man auch
    einstellen wuerde.
    """
    if phi_model not in PHI_MODELS:
        raise KeyError(f"unbekanntes Phi-Modell {phi_model!r}; "
                       f"bekannt: {sorted(PHI_MODELS)}")
    pname, internal = PHI_MODELS[phi_model]

    a = argparse.Namespace()
    a.phi_model = internal
    a.phi_tau = 0.25
    a.phi_xi = 0.01
    a.phi_gamma = 1.0
    a.gp_noise = gp_noise
    # `visit_sat` entscheidet mit darueber, ob die Altersgewichtung ueberhaupt
    # bei der Zieldichte ankommt. `debt_density` rechnet
    # v = visit / (visit_sat * max(visit)) und schneidet bei 1 ab; bei 0,25
    # (der Voreinstellung der GUI) gilt damit alles, was mehr als ein Viertel
    # des Maximums erreicht, als vollstaendig bedient — und das ist ein frueh
    # befahrenes Gebiet auch nach acht Runden noch. Die Alterung waere
    # sichtbar in `visitation_recent` und unsichtbar in Phi. Deshalb hier 1,0:
    # gesaettigt ist nur, was gerade das Maximum haelt, also das zuletzt
    # Befahrene. Der Selbsttest haelt beide Faelle fest.
    a.visit_sat = visit_sat
    a.debt_weight = debt_weight
    a.device = device
    a.phi_mode = phi_mode
    a.phi_quantile = 0.5
    a.n_particles = n_particles
    a.noise = meas_noise
    a.sensor_radius = sensor_radius
    a.visit_bandwidth = sensor_radius
    a.visit_halflife = visit_halflife
    a.max_obs = max_obs
    a.obstacle = None

    if pname == 'kappa':
        a.kappa = float(param)
    elif pname == 'w':
        w = min(max(float(param), 0.0), 0.98)
        a.kappa = w / max(1e-6, 1.0 - w)
    else:                                   # tau
        a.phi_tau = float(param)
        a.kappa = 1.0
    a.phi_ui = phi_model
    a.param_name = pname
    a.param = float(param)
    return a


# ---------------------------------------------------------------------------
# Hilfsgroessen
# ---------------------------------------------------------------------------

def resample_arclength(curve, n_pts):
    """Bahn (T,2) gleichmaessig nach Bogenlaenge auf `n_pts` Punkte abtasten.

    Ohne das haengt die Zahl der Messungen je Laengeneinheit daran, wie lang
    die *geplante* Bahn war, aus der das Stueck geschnitten wurde — eine
    Groesse, die mit der Guete der Einstellung nichts zu tun hat und den
    Vergleich zweier Parameterwerte verfaelschen wuerde.
    """
    if curve.shape[0] < 2:
        return curve
    seg = (curve[1:] - curve[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1, device=curve.device, dtype=curve.dtype),
                     seg.cumsum(0)])
    total = float(cum[-1])
    if total <= 1e-9:
        return curve[:1]
    want = torch.linspace(0.0, total, n_pts, device=curve.device,
                          dtype=curve.dtype)
    idx = torch.searchsorted(cum, want).clamp(1, curve.shape[0] - 1)
    lo, hi = cum[idx - 1], cum[idx]
    t = ((want - lo) / (hi - lo).clamp(min=1e-12)).unsqueeze(-1)
    return curve[idx - 1] * (1 - t) + curve[idx] * t


def visitation_recent(path, res, bandwidth, device, half_life=3.0):
    r"""Aufenthaltsdichte, in der ein *frischer* Besuch schwerer wiegt als ein alter.

    `apply_cfm_belief.visitation_field` gewichtet jeden gefahrenen Punkt gleich.
    Das genuegt fuer die Variante D dort, hier aber nicht: die Aufgabe verlangt
    ausdruecklich, dass ein eben erkundetes Gebiet nur *nicht sofort wieder*
    angefahren wird — spaeter aber schon, und zwar umso oefter, je hoeher seine
    Intensitaet ist.

    Mit gleichgewichteten Besuchen passiert das nicht. Der Grund ist die
    Saettigung: `debt_density` rechnet mit `v = visit / (visit_sat * max(visit))`
    und schneidet bei 1 ab. Bei `visit_sat = 0.25` sitzt ein einmal befahrenes
    Gebiet damit beim Vierfachen der Saettigungsschwelle und bleibt dort — es
    muesste anderswo *viermal* so lange verweilt werden, damit sein relativer
    Wert ueberhaupt zu sinken beginnt. Ein Gebiet, das in Runde 1 befahren
    wurde, ist in Runde 10 also genauso gesperrt wie unmittelbar danach.

    Hier bekommt deshalb jeder Bahnpunkt ein Gewicht

        gamma(a) = 2^(-a / half_life)

    mit `a` seinem Alter, gemessen als Bogenlaenge bis zum Bahnende, und
    `half_life` in Laengeneinheiten. Bei der Voreinstellung 3 ist die Sperre
    eines Gebiets nach drei weiteren Ausfuehrungen halbiert und nach sechs auf
    ein Viertel gefallen. Die Rueckkehr ist damit erlaubt, aber verzoegert —
    und weil `debt_density` den verbliebenen Anteil mit `mu` multipliziert,
    faellt sie fuer starke Gebiete frueher an als fuer schwache. Genau das ist
    die geforderte Besuchshaeufigkeit proportional zur Intensitaet.

    `half_life <= 0` schaltet die Gewichtung ab und ergibt exakt
    `acb.visitation_field` — die Einstellung, unter der die GUI und die
    Missionen in `apply_cfm_belief.py` gefahren sind.
    """
    if half_life is None or half_life <= 0:
        return acb.visitation_field(path, res, bandwidth, device)

    path = path.to(device)
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, res, device=device),
        torch.linspace(0, 1, res, device=device), indexing='ij')
    cells = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    seg = (path[1:] - path[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1, device=device, dtype=seg.dtype),
                     seg.cumsum(0)])
    age = cum[-1] - cum                                   # (T,)
    w = torch.pow(torch.tensor(0.5, device=device), age / half_life)

    d2 = torch.cdist(cells, path) ** 2                    # (R*R, T)
    k = (torch.exp(-d2 / (2.0 * bandwidth ** 2)) * w.unsqueeze(0)).sum(dim=1)
    return k.view(res, res)


def blind_coverage(truth):
    """Abdeckungsfehler ohne jede Bahn — der Bezugswert einer Form.

    `coverage_vs_truth` ist ein dichtegewichteter mittlerer Abstand und
    deshalb formabhaengig: eine weit gestreute Verteilung hat schon bei
    perfekter Bahn einen groesseren Wert als eine kompakte. Ein Mittel ueber
    25 Formen waere ohne Bezug also von den grossflaechigen Formen dominiert.

    Bezogen wird deshalb auf den Wert, den ein Roboter *ohne* Bahn erreicht
    (ein Punkt in der Mitte). `cov_norm = cov / cov_blind` liest sich damit als
    "welcher Anteil des anfaenglichen Abdeckungsfehlers ist noch uebrig" — eine
    Groesse zwischen 0 und ~1, die ueber Formen hinweg vergleichbar ist.
    """
    mid = torch.tensor([[0.5, 0.5]], device=truth.device, dtype=torch.float32)
    return float(coverage_vs_truth(mid, truth))


# ---------------------------------------------------------------------------
# SVGD in Arbeitsprozessen
# ---------------------------------------------------------------------------
# Die Verfeinerung laeuft in NumPy auf der CPU und ist je Form unabhaengig —
# die einzige Stelle der Schleife, an der sich mehrere Kerne lohnen. Der
# Prozesspool wird in `optimize.py` aufgebaut und hier nur benutzt; das
# Arbeitsmodul haelt bewusst nur `common.svgd_refine` (kein Matplotlib, kein
# MuJoCo), weil Windows die Prozesse per `spawn` startet und jeder Import
# vollstaendig wiederholt wird.

_WORKER_REFINER = None


def _worker_init(seed):
    global _WORKER_REFINER
    from common.svgd_refine import SvgdRefiner
    _WORKER_REFINER = SvgdRefiner(seed=seed)


def _worker_refine(job):
    curve_np, phi_np, n_iters, nxi = job
    return _WORKER_REFINER.refine(curve_np, phi_np, n_iters, nxi=nxi)


def refine_batch(curves, phis, n_iters, nxi=25, pool=None, refiner=None):
    """SVGD auf alle Formen einer Runde. Gibt eine Liste von (T,2)-Arrays.

    `pool` ist ein `ProcessPoolExecutor` mit `_worker_init` als Initialisierer;
    ohne ihn wird seriell mit `refiner` gerechnet (Smoke-Test, ein einzelner
    Kern, oder `n_iters == 0`).
    """
    jobs = [(c.detach().cpu().numpy().astype(np.float64),
             p.detach().cpu().numpy().astype(np.float64), int(n_iters), nxi)
            for c, p in zip(curves, phis)]
    if n_iters <= 0:
        return [j[0] for j in jobs]
    if pool is None:
        if refiner is None:
            from common.svgd_refine import SvgdRefiner
            refiner = SvgdRefiner(seed=0)
        return [refiner.refine(*j[:3], nxi=j[3]) for j in jobs]
    return list(pool.map(_worker_refine, jobs))


# ---------------------------------------------------------------------------
# Die Mission
# ---------------------------------------------------------------------------

class LaengenMission:
    r"""Eine Laengeneinheit-Mission, im Gleichschritt ueber alle Holdout-Formen.

    Args:
        planner:   `apply_cfm_belief.CfmPlanner` (startpunkt-konditioniert).
        truths:    (S, R_t, R_t) wahre Dichten, max-normiert.
        names:     Formnamen, nur fuer die Ausgabe.
        args:      aus `build_mission_args`.
        svgd_iters: 0 = keine Verfeinerung.
        n_prior:   Vorabmessungen je Form. Voreinstellung 0 — "wir starten ohne
                   Wissen ueber die Zielverteilung". In Runde 0 ist Phi damit
                   gleichverteilt (mu = 0, sigma ueberall gleich) und die erste
                   Bahn ist eine beliebige raumfuellende; ab Runde 1 traegt der
                   Glaube Struktur. Das ist der in `exploration/README.md`
                   beschriebene entartete Anfangszustand — hier kein Fehler,
                   sondern die Aufgabenstellung.
    """

    def __init__(self, planner, truths, names, args, svgd_iters=0,
                 gp_res=64, n_prior=0, seed=0, nxi_refine=25, pool=None):
        self.planner = planner
        self.truths = truths
        self.names = names
        self.args = args
        self.svgd_iters = int(svgd_iters)
        self.gp_res = gp_res
        self.seed = seed
        self.nxi_refine = nxi_refine
        self.pool = pool
        self.device = truths.device
        self.S = truths.shape[0]

        self.ergodic = acb.ErgodicScore(K=8, device=str(self.device))
        self.cov_blind = [blind_coverage(truths[i]) for i in range(self.S)]

        self.beliefs, self.beliefs0 = [], []
        for i in range(self.S):
            b = GPBelief(grid_res=gp_res, lengthscale=0.08,
                         noise=args.gp_noise, device=str(self.device))
            if n_prior > 0:
                g = torch.Generator().manual_seed(seed * 977 + i)
                pts = acb.prior_points('zufall', n_prior, generator=g,
                                       device=str(self.device))
                _, vals = measure(pts, truths[i], noise_std=args.noise)
                b.observe(pts, vals)
            self.beliefs.append(b)
            self.beliefs0.append(b.clone())

        self.driven = [None] * self.S
        self.plan_s = 0.0
        self.svgd_s = 0.0

    # -- eine Runde ---------------------------------------------------------
    def _phi_and_particles(self):
        phis, parts = [], []
        for i in range(self.S):
            mu, sd = self.beliefs[i].posterior_grid()
            visit = (visitation_recent(self.driven[i], self.gp_res,
                                       self.args.visit_bandwidth,
                                       str(self.device),
                                       half_life=self.args.visit_halflife
                                       * LENGTH_UNIT)
                     if self.driven[i] is not None else None)
            phi, _ = acb.debt_density(mu, sd, visit, self.args.kappa, self.args)
            phis.append(phi)
            parts.append(acb.phi_particles(phi, self.args.n_particles,
                                           mode=self.args.phi_mode,
                                           device=str(self.device)))
        return phis, torch.stack(parts)

    def _execute_one_unit(self, curve, i):
        """Von `curve` genau eine Laengeneinheit ab der aktuellen Position.

        Nach einer SVGD-Verfeinerung liegt der erste Kurvenpunkt nicht mehr
        exakt auf der aktuellen Position — SVGD bewegt alle Kontrollpunkte
        frei, auch den ersten, den das Netz zuvor hart gesetzt hatte. Der
        entstehende Versatz wird als kurzes gerades Anschlussstueck gefahren
        und **zaehlt zur Weglaenge**, statt als Sprung unter den Tisch zu
        fallen; sonst waere die gefahrene Strecke systematisch zu kurz
        gebucht, und zwar genau in dem Zweig, der mit SVGD verglichen wird.
        """
        if self.driven[i] is not None:
            here = self.driven[i][-1]
            gap = float((curve[0] - here).norm())
            if gap > 1e-6:
                al = torch.linspace(0, 1, 8, device=curve.device,
                                    dtype=curve.dtype).unsqueeze(-1)
                link = here.unsqueeze(0) * (1 - al) + curve[0].unsqueeze(0) * al
                curve = torch.cat([link, curve], dim=0)
        seg = trim_to_length(curve, LENGTH_UNIT)
        n_pts = max(8, int(round(PTS_PER_UNIT * path_length(seg) / LENGTH_UNIT)))
        return resample_arclength(seg, n_pts)

    def _row(self, i, r):
        drv = self.driven[i]
        cov = float(coverage_vs_truth(drv, self.truths[i]))
        return {
            'shape': self.names[i],
            'n_exec': r + 1,
            'cov': cov,
            'cov_norm': cov / max(self.cov_blind[i], 1e-12),
            'erg_truth': self.ergodic(drv, self.truths[i]),
            'belief_rmse': belief_rmse(self.beliefs[i], self.truths[i]),
            'info_gain': float(self.beliefs0[i].total_uncertainty()
                               - self.beliefs[i].total_uncertainty()),
            'path_len': path_length(drv),
            'n_obs': int(self.beliefs[i].n_obs),
        }

    def round(self, r):
        """Eine Planung + eine gefahrene Laengeneinheit fuer alle Formen."""
        phis, parts = self._phi_and_particles()

        starts = None
        if any(d is not None for d in self.driven):
            starts = torch.stack([
                (self.driven[i][-1] if self.driven[i] is not None
                 else torch.tensor([0.5, 0.5], device=self.device))
                for i in range(self.S)])

        t0 = time.perf_counter()
        with torch.no_grad():
            cps = self.planner.plan(parts, n_candidates=self.S, start=starts)
            curves = self.planner.render(cps)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.plan_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        refined = refine_batch(curves, phis, self.svgd_iters,
                               nxi=self.nxi_refine, pool=self.pool)
        self.svgd_s += time.perf_counter() - t0

        rows = []
        for i in range(self.S):
            curve = torch.as_tensor(refined[i], device=self.device,
                                    dtype=torch.float32).clamp(0.0, 1.0)
            seg = self._execute_one_unit(curve, i)
            pts, vals = measure(seg, self.truths[i], noise_std=self.args.noise,
                                sensor_radius=self.args.sensor_radius)
            self.beliefs[i].observe(*thin(pts, vals,
                                          max_points=self.args.max_obs))
            self.driven[i] = (seg if self.driven[i] is None
                              else torch.cat([self.driven[i], seg], dim=0))
            rows.append(self._row(i, r))
        return rows

    def run(self, n_max, on_round=None):
        """`n_max` Runden. -> Liste von Zeilen, eine je (Form, Rundenzahl).

        Weil jede Runde eine vollstaendige Zeile schreibt, enthaelt die
        zurueckgegebene Spur die Ergebnisse fuer *jede* Zahl von Ausfuehrungen
        von 1 bis `n_max`. Genau daraus liest `objective.py` die beste
        Rundenzahl ab, ohne dafuer je erneut zu rechnen.
        """
        torch.manual_seed(self.seed)
        rows = []
        for r in range(n_max):
            rows += self.round(r)
            if on_round is not None:
                on_round(r, n_max)
        for row in rows:
            row['plan_s'] = self.plan_s / max(self.S, 1)
            row['svgd_s'] = self.svgd_s / max(self.S, 1)
        return rows


# ---------------------------------------------------------------------------
# Aufbau
# ---------------------------------------------------------------------------

def load_holdout(resolution=96, device='cuda', shapes=None, limit=None):
    """Die Holdout-Formen als wahre Dichten.

    `split='val'` der Trainingsdatenbank ist deckungsgleich mit
    `shape_library.VALIDATION_SHAPES` (25 Formen) — geprueft, nicht
    angenommen. Es wird ueber die Datenbank geladen, weil `load_truth` die
    Formen dort schon max-normiert und in der Reihenfolge liefert, die auch
    die uebrigen Auswertungen des Projekts benutzen.
    """
    names, truths = load_truth(labels=shapes, n=(limit or 999), split='val',
                               resolution=resolution, device=device)
    if limit is not None:
        names, truths = names[:limit], truths[:limit]
    return names, truths


def build_planner(ckpt=DEFAULT_CKPT, device='cuda', pts=256, flow_steps=100,
                  cfg_weight=2.0):
    p = acb.CfmPlanner(ckpt=ckpt, device=device, pts=pts, steps=flow_steps,
                       cfg_weight=cfg_weight)
    if not p.start_cond:
        raise RuntimeError(
            f"{os.path.basename(ckpt)} ist nicht startpunkt-konditioniert. "
            "Die Laengeneinheit-Mission setzt jede Runde am Endpunkt der "
            "vorigen an; ohne `start_cond` springt die Bahn dort hin, wo das "
            "Netz sie gerade beginnen laesst.")
    return p


def run_config(planner, truths, names, phi_model, param, n_max,
               svgd_iters=0, seed=0, pool=None, on_round=None, **kw):
    """Ein vollstaendiger Rollout fuer eine Einstellung. -> Zeilen."""
    args = build_mission_args(str(truths.device), phi_model=phi_model,
                              param=param, **kw)
    m = LaengenMission(planner, truths, names, args, svgd_iters=svgd_iters,
                       seed=seed, pool=pool)
    rows = m.run(n_max, on_round=on_round)
    for row in rows:
        row.update(phi_model=phi_model, param=float(param),
                   svgd_iters=int(svgd_iters), seed=int(seed))
    return rows, m
