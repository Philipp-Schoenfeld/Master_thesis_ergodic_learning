r"""
apply_cfm_belief.py
===================
Das **bereits trainierte** CFM+ErgLoss-Netz auf glaubenskonditionierte
Zieldichten anwenden — ohne es neu zu trainieren.

Die Idee
--------
Bisher bekommt das Netz zur Auswertung die Zieldichte der Holdout-Form direkt:
die volle Wahrheit, aus `pdf_on_grid(density_params)`. Das setzt voraus, dass
das Feld vor der Fahrt bekannt ist — genau die Annahme, die bei echter
Erkundung fehlt.

Hier wird stattdessen ein Glaube ueber dem Feld gefuehrt und daraus die
Zieldichte gebaut:

    Phi(x) = mu(x) + kappa * sigma(x)

Das Netz selbst bleibt unangetastet. Es konditioniert ohnehin auf eine
gewichtete Partikelwolke und kann nicht sehen, woraus die gezogen wurde — das
ist der Grund, warum diese Frage ohne eine einzige Architekturaenderung
angehbar ist.

Warum der Glaube die fehlende Historie ersetzt
----------------------------------------------
Das CFM-Netz hat keinen Historien-Eingang: es weiss beim Nachplanen nicht, wo
es schon war. In `exploration/` war genau das der Grund, warum Variante C beim
zweiten und dritten Segment fast dieselbe Schleife noch einmal fuhr.

Hier tritt das Problem nicht auf, und zwar aus einem strukturellen Grund: nach
dem Abfahren einer Bahn ist sigma entlang dieser Bahn klein geworden, also ist
Phi dort klein. **Die Historie steckt im Glauben.** Das Netz braucht sie nicht
als Eingang, weil sie schon in seiner Konditionierung steht. Der Nachteil des
supervised Netzes faellt in dieser Konstruktion also weg.

Die zwei Skalen-Fallen
----------------------
1. **Gewichtskanal.** Die Trainingsdichten des Hauptrunners sind auf
   `max = 1` normiert (`d_map /= d_map.max()` in `_load_shapes`), und der
   dritte Partikelkanal traegt genau diesen Wert. `ucb_density` normiert per
   Default auf `sum = 1` — auf einem 64x64-Gitter ist das ein Faktor ~4000
   daneben, und das Netz saehe einen Gewichtskanal, der numerisch Null ist.
   Deshalb `norm='max'`.

2. **Traeger.** Mit `sample_mode='uniform'` liegen die Trainingspartikel
   gleichverteilt ueber `{d > 1e-5}` und tragen den Dichtewert als Gewicht.
   Bei Phi ist sigma ueberall positiv, der Traeger also das ganze Quadrat.
   `--phi_mode` macht diese Wahl explizit und pruefbar statt stumm.

Missionen
---------
    orakel     Partikel aus der *wahren* Dichte, eine Planung. Obergrenze.
    glaube-1   Phi aus dem Vorab-Glauben, eine Planung. Kein Nachplanen.
    glaube-R   Phi-Schleife ueber `--rounds` Runden mit kappa-Zeitplan.
    grad-R     Dieselbe Schleife, aber mit direkter Gradientenoptimierung
               statt Netz — zeigt, ob das Netz gegenueber dem Verfahren
               verliert oder nur gegenueber der Zieldichte.
    maeher     Bahnplaner fester Geometrie auf die Laenge von glaube-R
               gebracht.

Verglichen wird bei **gleicher Weglaenge** ueber Anytime-Kurven, nicht am
Endwert: glaube-R faehrt R-mal so weit wie glaube-1, und bei freier Weglaenge
gewinnt fast immer, wer weiter faehrt.

Beispiel
--------
    python -u apply_cfm_belief.py \
        --ckpt checkpoints/<...ERGLOSS-w300...>.pt \
        --shapes 12 --rounds 3 --kappa0 3.0 --kappa1 0.3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, _arch, os.path.join(_arch, 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.acquisition import (ucb_density, is_degenerate,
                                particles_from_density, kappa_schedule,
                                phi_from_belief, PHI_MODELLE)
from common.baselines import lawnmower_path
from common.belief import GPBelief, MaskiertesWissen, muster_maske
from common.data import load_truth
from common.metrics import (anytime_curve, at_budget, belief_rmse,
                            coverage_vs_truth, information_gain, path_length)
from common.observation import measure, thin
from common.planner import GradientPlanner


# ===========================================================================
# Zieldichte -> Partikel, in der Konvention des Trainings
# ===========================================================================

def phi_particles(phi, n_particles, mode='uniform', quantile=0.5,
                  device=None, generator=None):
    """Partikelwolke (N,3) aus einer max-normierten Zieldichte.

    `mode` steuert, wie der Traeger gebildet wird — die zweite Skalen-Falle
    aus dem Modulkopf:

      'uniform'  Orte gleichverteilt ueber `{phi > 1e-5}`, Gewicht = phi.
                 Formal identisch zu `sample_particles(mode='uniform')` im
                 Hauptrunner. Bei einer UCB-Dichte ist der Traeger allerdings
                 das ganze Quadrat, waehrend er im Training die Umgebung eines
                 Buchstabens war.
      'density'  Orte proportional zu phi. Die Wolke konzentriert sich dort,
                 wo Phi gross ist — geometrisch naeher am Trainingsbild, aber
                 mit einer anderen Gewichtssemantik.
      'quantile' Orte gleichverteilt ueber `{phi > Quantil(phi, q)}`, Gewicht
                 = phi. Der Kompromiss: ein echter, begrenzter Traeger wie im
                 Training, ohne die Gewichtssemantik zu wechseln.
    """
    device = device or phi.device
    if mode == 'quantile':
        thr = torch.quantile(phi.reshape(-1), quantile)
        masked = torch.where(phi > thr, phi, torch.zeros_like(phi))
        return particles_from_density(masked, n_particles, device=device,
                                      mode='uniform', generator=generator)
    return particles_from_density(phi, n_particles, device=device, mode=mode,
                                  generator=generator)


# ===========================================================================
# Ergodische Metrik gegen die Wahrheit (nur Auswertung, kein Training)
# ===========================================================================

class ErgodicScore:
    """E = sum_k Lambda_k (c_k - phi_k)^2 gegen ein festes Wahrheitsgitter.

    Dieselbe Konvention wie `ergodic_metric.py`, damit die Zahlen hier an die
    Reihe aus den ErgLoss-Ablationen anschliessen. Das Wahrheitsgitter wird
    dabei als vollstaendige gewichtete Punktmenge behandelt — jede Zelle ein
    Partikel —, also ohne den Stichprobenfehler einer gezogenen Wolke.
    """

    def __init__(self, K=8, device='cpu'):
        from ergodic_metric import make_k_grid
        k_idx, Lam = make_k_grid(K)
        self.k_idx = torch.tensor(k_idx, device=device)
        self.Lam = torch.tensor(Lam, dtype=torch.float32, device=device)
        self.device = device
        self._cache = {}

    def target(self, truth):
        key = id(truth)
        if key not in self._cache:
            from ergodic_metric import target_coeffs_from_particles
            R = truth.shape[-1]
            ys, xs = torch.meshgrid(
                torch.linspace(0, 1, R, device=self.device),
                torch.linspace(0, 1, R, device=self.device), indexing='ij')
            cloud = torch.stack([xs.reshape(-1), ys.reshape(-1),
                                 truth.to(self.device).reshape(-1)], dim=-1)
            self._cache[key] = target_coeffs_from_particles(
                cloud.unsqueeze(0), self.k_idx, weighted=True)
        return self._cache[key]

    def __call__(self, curve, truth):
        from ergodic_metric import trajectory_coeffs
        c = trajectory_coeffs(curve.unsqueeze(0).to(self.device), self.k_idx)
        d = c - self.target(truth)
        return float((self.Lam * d ** 2).sum())


# ===========================================================================
# Planer
# ===========================================================================

class CfmPlanner:
    """Das trainierte CFM-Netz als Planer.

    Bewusst nicht `common.planner.ModelPlanner` wiederverwendet: hier wird
    zusaetzlich `--dry_run` gebraucht (zufaellig initialisiertes Netz, um die
    Pipeline ohne Checkpoint durchzutesten) und die Kandidatenauswahl laeuft
    gegen Phi statt gegen die Wahrheit.
    """

    def __init__(self, ckpt=None, nxi=25, pts=128, deg=5, steps=100,
                 cfg_weight=2.0, D=384, n_particles=256, device='cpu'):
        from obstacles import bspline_basis_matrix

        self.device = torch.device(device)
        self.steps, self.cfg_weight = steps, cfg_weight

        ck = None
        if ckpt is not None:
            ck = torch.load(ckpt, map_location=device, weights_only=False)
            nxi = ck.get('nxi', nxi)
            D = ck.get('D', D)
            n_particles = ck.get('n_particles', n_particles)
        # Startpunkt-konditionierte Checkpoints tragen `start_cond` und
        # brauchen die andere Architektur — sie haben einen zusaetzlichen
        # `start_emb`-Block, den das alte Netz nicht laden kann.
        self.start_cond = bool(ck.get('start_cond', False)) if ck else False
        if self.start_cond:
            from flow_matching_cond_particles_start import (
                ParticleCrossAttnFlowNetwork, generate_particle_trajectories)
        else:
            from flow_matching_cond_particles_crossattn import (
                ParticleCrossAttnFlowNetwork, generate_particle_trajectories)
        self._gen = generate_particle_trajectories
        self.nxi, self.n_particles = nxi, n_particles

        self.B = torch.from_numpy(
            bspline_basis_matrix(nxi, pts, deg)).float().to(self.device)
        # `n_particles` ist ein reiner Datenparameter — der Tokenizer arbeitet
        # ueber die Sequenzachse und kennt keine feste Wolkengroesse.
        self.model = ParticleCrossAttnFlowNetwork(nxi=nxi, D=D).to(self.device)
        if ck is not None:
            self.model.load_state_dict(ck['model_state_dict'])
        self.model.eval()
        self.last_wallclock = 0.0

    def render(self, cps):
        return torch.einsum('pi,kid->kpd', self.B, cps.float().to(self.device))

    def plan(self, particles, n_candidates=1, start=None):
        """`start` wirkt nur bei einem startpunkt-konditionierten Checkpoint.

        Dort geht der Punkt als FiLM-Konditionierung ins Netz, und der erste
        Kontrollpunkt wird anschliessend hart darauf gesetzt. Bei einem alten
        Checkpoint wird er ignoriert.
        """
        t0 = time.perf_counter()
        kw = {}
        if start is not None and self.start_cond:
            kw['start'] = start.to(self.device).reshape(-1)
        out = self._gen(self.model, particles.to(self.device),
                        num_samples=n_candidates, nxi=self.nxi,
                        steps=self.steps, device=str(self.device),
                        cfg_weight=self.cfg_weight, **kw)
        cps = out[0] if isinstance(out, tuple) else out
        self.last_wallclock = time.perf_counter() - t0
        return cps.detach()


def best_candidate(curves, phi):
    """Kandidat mit der besten Abdeckung *bezueglich Phi* — nicht bezueglich
    der Wahrheit. Nach der Wahrheit auszuwaehlen waere ein Blick auf das
    Ergebnis und wuerde die Mission unbrauchbar machen."""
    if curves.shape[0] == 1:
        return curves[0]
    scores = [float(coverage_vs_truth(c, phi)) for c in curves]
    return curves[int(np.argmin(scores))]


# ===========================================================================
# Missionen
# ===========================================================================

def run_mission(planner, truth, belief0, args, mode, rounds, ergodic,
                use_truth_density=False, generator=None):
    """Eine Mission auf einer Form. -> dict mit Bahn, Runden und Metriken."""
    belief = belief0.clone()
    seg_curves, rounds_log = [], []
    wallclock = 0.0

    for r in range(rounds):
        # ---- Zieldichte fuer diese Runde ---------------------------------
        if use_truth_density:
            phi = truth.to(args.device)
            phi = phi / phi.max().clamp(min=1e-12)
            kap = float('nan')
        else:
            mu, sd = belief.posterior_grid()
            kap = kappa_schedule(r, rounds, args.kappa0, args.kappa1)
            # norm='max': Trainingskonvention des Gewichtskanals, siehe Modulkopf
            phi = zieldichte(mu, sd, kap, args)
            if r == 0 and is_degenerate(phi):
                print("      [warn] Phi ist praktisch gleichverteilt — "
                      "ohne Vormessungen ist Erkundung nicht definiert.")

        parts = phi_particles(phi, planner.n_particles if hasattr(
            planner, 'n_particles') else args.n_particles,
            mode=args.phi_mode, quantile=args.phi_quantile,
            device=args.device, generator=generator)

        # ---- Planen -------------------------------------------------------
        # Beide Planer bekommen dieselbe Wolke (N,3) und *keine* Historie:
        # getestet wird ja gerade, ob Phi die Historie ersetzt.
        cps = planner.plan(parts, n_candidates=args.n_candidates)
        wallclock += getattr(planner, 'last_wallclock', 0.0)

        curves = planner.render(cps)
        curve = best_candidate(curves, phi)

        # ---- Ausfuehren (ggf. nur ein Praefix) -----------------------------
        # Mit `--execute_frac < 1` ist das genau Variante D: lang planen, kurz
        # fahren, neu planen. Der Glaube traegt dabei die Historie, das Netz
        # braucht keinen Gedaechtnis-Eingang.
        if args.execute_frac < 1.0:
            k = max(2, int(args.execute_frac * curve.shape[0]))
            curve = curve[:k]

        # Die naechste Runde plant von vorn und beginnt nicht dort, wo diese
        # aufgehoert hat — das Netz hat keinen Start-Eingang. Die Anfahrt wird
        # eingefuegt und mitgezaehlt, sonst bekommt das Verfahren eine Strecke
        # geschenkt, die ein Roboter fahren muesste.
        if seg_curves:
            prev_end = seg_curves[-1][-1].to(curve.device)
            al = torch.linspace(0, 1, args.transit_pts,
                                device=curve.device).unsqueeze(-1)
            link = prev_end.unsqueeze(0) * (1 - al) + curve[0].unsqueeze(0) * al
            curve = torch.cat([link, curve], dim=0)
        seg_curves.append(curve.detach())

        # ---- Messen und Glauben fortschreiben ------------------------------
        before = belief.clone()
        pts, vals = measure(curve.detach(), truth,
                            noise_std=args.noise,
                            sensor_radius=args.sensor_radius)
        belief.observe(*thin(pts, vals, max_points=args.max_obs))

        rounds_log.append({
            'round': r, 'kappa': kap,
            'phi': phi.detach().cpu().numpy(),
            'curve': curve.detach().cpu().numpy(),
            'info_gain': information_gain(before, belief),
            'belief_rmse': belief_rmse(belief, truth),
        })

    path = torch.cat(seg_curves, dim=0)
    return {
        'path': path,
        'rounds': rounds_log,
        'belief': belief,
        'wallclock': wallclock,
        'coverage': float(coverage_vs_truth(path, truth)),
        'ergodic': ergodic(path, truth),
        'info_gain': information_gain(belief0, belief),
        'belief_rmse': belief_rmse(belief, truth),
        'path_len': path_length(path),
    }


def _probe_points_fn():
    """`_probe_points` aus Variante B laden statt nachzubauen.

    Der Helfer muss denselben Messprozess abbilden wie `observation.measure`,
    Sensorring eingeschlossen — ohne den war die Vorausschau in Variante B um
    rund 40 % zu pessimistisch. Zwei Kopien dieser Logik waeren genau die Art
    von Duplikat, das spaeter auseinanderlaeuft.
    """
    import importlib.util
    path = os.path.join(_here, 'variant_b_diffsim', 'run_b.py')
    spec = importlib.util.spec_from_file_location('_run_b', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._probe_points


def refine(cps0, belief, parts, grad, args, probe_fn, steps=None):
    """Differenzierbare Verfeinerung von Kontrollpunkten (Variante B).

    Das ist der Punkt, an dem sich ein trainiertes Netz und Variante B nicht
    ausschliessen, sondern ergaenzen. B *ist* kein Konditionierungsschema, das
    man dem Netz geben koennte — B ist ein Optimierer, der Gradienten durch die
    Kontrollpunkte braucht, und ein Vorwaertspass hat nichts zu optimieren.

    Umgekehrt ist B aber genau der nicht-konvexe Optimierer, fuer den die
    Arbeit einen gelernten Warmstart verspricht: statt aus Rauschen zu starten,
    startet die Optimierung bei dem, was das Netz in einem Durchgang erzeugt
    hat. Gemessen wird deshalb beides — dieselbe Verfeinerung aus dem Netz
    heraus und aus Rauschen heraus, mit gleich vielen Schritten.

        min_cps   lambda_unc * U(cps)  +  lambda_cov * E_erg(cps, Phi)  +  Reg

    U ist die erwartete Restunsicherheit nach dem Abfahren. Sie ist exakt
    berechenbar, weil die Posterior-Varianz eines GP nur von den Messorten
    abhaengt, nicht von den Messwerten.
    """
    steps = steps or args.refine_steps
    cps = cps0.clone().detach().to(args.device).requires_grad_(True)
    pb = parts.unsqueeze(0).expand(cps.shape[0], -1, -1).to(args.device)
    opt = torch.optim.Adam([cps], lr=args.refine_lr)
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        curve = torch.einsum('pi,kid->kpd', grad.B.float(), cps)
        unc = sum(belief.uncertainty_after(
            probe_fn(curve[k], args.n_probe, args.sensor_radius))
            for k in range(curve.shape[0]))
        cov = grad.erg.coverage_error(cps, pb).sum()
        reg = grad._regularisers(cps).sum()
        (args.lambda_unc * unc + args.lambda_cov * cov + reg).backward()
        opt.step()
    return cps.detach().clamp(0.0, 1.0), time.perf_counter() - t0


def run_variant_b(planner, grad, truth, belief0, args, ergodic, probe_fn,
                  warm=True, generator=None):
    """Variante B, einmal mit gelerntem Warmstart und einmal aus Rauschen."""
    belief = belief0.clone()
    mu, sd = belief.posterior_grid()
    phi = zieldichte(mu, sd, args.kappa0, args)
    parts = phi_particles(phi, args.n_particles, mode=args.phi_mode,
                          quantile=args.phi_quantile, device=args.device,
                          generator=generator)

    net_s = 0.0
    if warm:
        cps0 = planner.plan(parts, n_candidates=1)
        net_s = planner.last_wallclock
    else:
        g = torch.Generator(device='cpu').manual_seed(args.seed)
        cps0 = (0.3 * torch.randn(1, planner.nxi, 2, generator=g) + 0.5)

    before = float(coverage_vs_truth(
        torch.einsum('pi,kid->kpd', grad.B.float(),
                     cps0.float().to(args.device))[0].detach(), truth))
    cps, ref_s = refine(cps0, belief, parts, grad, args, probe_fn)
    curve = torch.einsum('pi,kid->kpd', grad.B.float(), cps)[0].detach()

    b_before = belief.clone()
    pts, vals = measure(curve, truth, noise_std=args.noise,
                        sensor_radius=args.sensor_radius)
    belief.observe(*thin(pts, vals, max_points=args.max_obs))

    return {
        'path': curve, 'rounds': [], 'belief': belief,
        'wallclock': net_s + ref_s,
        'coverage': float(coverage_vs_truth(curve, truth)),
        'ergodic': ergodic(curve, truth),
        'info_gain': information_gain(b_before, belief),
        'belief_rmse': belief_rmse(belief, truth),
        'path_len': path_length(curve),
        'cov_start': before, 'net_s': net_s, 'refine_s': ref_s,
    }


def run_selection(planner, truth, belief0, args, ergodic, probe_fn,
                  generator=None):
    """Das Netz zieht K Kandidaten, die Vorausschau waehlt aus.

    Der billigste Weg, die Idee von Variante B mit einem trainierten Netz zu
    verbinden: `uncertainty_after` wird hier nur *ausgewertet*, nicht
    differenziert. Kein Optimierer, kein Gradient — ein Vorwaertspass und K
    Bewertungen. Wenn das schon nahe an die volle Verfeinerung herankommt, ist
    die Optimierung den Aufwand nicht wert.
    """
    belief = belief0.clone()
    mu, sd = belief.posterior_grid()
    phi = zieldichte(mu, sd, args.kappa0, args)
    parts = phi_particles(phi, args.n_particles, mode=args.phi_mode,
                          quantile=args.phi_quantile, device=args.device,
                          generator=generator)

    cps = planner.plan(parts, n_candidates=args.select_k)
    wall = planner.last_wallclock
    curves = planner.render(cps)

    t0 = time.perf_counter()
    with torch.no_grad():
        scores = [float(belief.uncertainty_after(
            probe_fn(c, args.n_probe, args.sensor_radius))) for c in curves]
    wall += time.perf_counter() - t0
    curve = curves[int(np.argmin(scores))].detach()

    b_before = belief.clone()
    pts, vals = measure(curve, truth, noise_std=args.noise,
                        sensor_radius=args.sensor_radius)
    belief.observe(*thin(pts, vals, max_points=args.max_obs))
    return {
        'path': curve, 'rounds': [], 'belief': belief, 'wallclock': wall,
        'coverage': float(coverage_vs_truth(curve, truth)),
        'ergodic': ergodic(curve, truth),
        'info_gain': information_gain(b_before, belief),
        'belief_rmse': belief_rmse(belief, truth),
        'path_len': path_length(curve),
        'spread': float(np.std(scores)),
    }


def zieldichte(mu, sd, kappa, args):
    r"""Die Zieldichte nach dem gewaehlten Modell — immer auf Maximum 1.

    Der kappa-Zeitplan bleibt die eine Stellschraube, die ueber alle Modelle
    laeuft, auch wenn kappa nicht in jedem davon woertlich vorkommt. Die
    Uebersetzung:

        ucb, stretch, mi   kappa direkt
        eid                kappa direkt — Mischungsverhaeltnis zwischen der
                           aufgedeckten Dichte und der Informationsdichte.
        mass               w = kappa/(1+kappa) — der Anteil der Zielmasse,
                           der auf Erkundung entfaellt. kappa = 3 wird zu
                           w = 0,75, kappa = 0,3 zu w = 0,23. Der Zeitplan
                           "erst erkunden, dann abdecken" bleibt damit erhalten.
        lse                die Schwelle bleibt fest; kappa wirkt hier nicht.
                           Das Modell fragt nach dem Traeger, und der haengt
                           nicht davon ab, wie erkundungsfreudig man ist.
        ei                 wie lse: kein kappa.

    Bei `--phi_model ucb` (Voreinstellung) ist das Ergebnis bitgleich mit dem
    bisherigen `ucb_density(..., norm='max')`.
    """
    m = args.phi_model
    if m == 'ucb':
        return ucb_density(mu, sd, kappa=kappa, norm='max')
    if m in ('stretch', 'mi'):
        kw = dict(kappa=kappa)
        if m == 'mi':
            kw['gamma'] = args.phi_gamma
        return phi_from_belief(mu, sd, modell=m, **kw)
    if m == 'eid':
        return phi_from_belief(mu, sd, modell='eid', kappa=kappa,
                               noise=args.gp_noise)
    if m == 'mass':
        return phi_from_belief(mu, sd, modell='mass',
                               w=float(kappa) / (1.0 + float(kappa)))
    if m == 'lse':
        return phi_from_belief(mu, sd, modell='lse', tau=args.phi_tau)
    if m == 'ei':
        return phi_from_belief(mu, sd, modell='ei', xi=args.phi_xi)
    raise KeyError(m)


def prior_points(pattern, n, generator=None, device='cpu', rand=0.0):
    r"""Messorte nach einem *Muster* statt zufaellig ueber die Flaeche.

    Zufaellig gestreute Vormessungen erzeugen einen Glauben, der ueberall
    gleich lueckenhaft ist — jede Messung senkt sigma ein wenig, nirgends
    entsteht ein wirklich bekannter Bereich. Damit misst der Vergleich der
    Zieldichten nur, wie sie mit gleichmaessigem Halbwissen umgehen.

    Diese Muster erzeugen stattdessen eine **Kante**: hier ist das Feld
    bekannt, dort gar nicht. Genau daran laesst sich ablesen, ob eine
    Zieldichte den unbekannten Teil auch wirklich ansteuert.

        zufall       gleichverteilt ueber das ganze Quadrat (bisheriges
                     Verhalten, bleibt die Voreinstellung)
        haelfte      nur die linke Haelfte ist bekannt
        quadranten   zwei diagonal gegenueberliegende Viertel sind bekannt.
                     Der bekannte Teil ist damit *unzusammenhaengend* — eine
                     Bahn muss zweimal ueber unbekanntes Gebiet.
        loch         alles ausser einer Scheibe in der Mitte ist bekannt. Das
                     Unbekannte ist hier von Wissen *umschlossen*; es gibt
                     keine Kante zum Rand hin, an der man sich entlanghangeln
                     koennte.

    `rand` streut zusaetzlich einen Anteil der Punkte ueber die ganze Flaeche.
    Bei 0 ist die Kante hart.
    """
    def drin(p):
        x, y = p[:, 0], p[:, 1]
        if pattern == 'zufall':
            return torch.ones_like(x, dtype=torch.bool)
        if pattern == 'haelfte':
            return x < 0.5
        if pattern == 'quadranten':
            return ((x < 0.5) & (y < 0.5)) | ((x >= 0.5) & (y >= 0.5))
        if pattern == 'loch':
            return ((x - 0.5) ** 2 + (y - 0.5) ** 2) > 0.28 ** 2
        raise KeyError(pattern)

    n_struk = int(round(n * (1.0 - rand)))
    gesammelt = []
    # Verwerfungsverfahren: gleichverteilt ziehen, was ausserhalb liegt
    # wegwerfen. Bei einem Gebiet von einem Viertel der Flaeche sind das im
    # Mittel vier Versuche je Treffer — bei diesen Zahlen belanglos.
    while sum(len(g) for g in gesammelt) < n_struk:
        kand = torch.rand(max(64, 4 * n_struk), 2, generator=generator)
        gesammelt.append(kand[drin(kand)])
    pts = torch.cat(gesammelt, dim=0)[:n_struk]
    if n - n_struk > 0:
        pts = torch.cat([pts, torch.rand(n - n_struk, 2, generator=generator)],
                        dim=0)
    return pts.to(device)


def visitation_field(path, res, bandwidth, device):
    r"""Aufenthaltsdichte der bereits gefahrenen Bahn auf dem GP-Gitter.

    Ein Gauss-Kern der Breite `bandwidth` um jeden gefahrenen Punkt, aufsummiert.
    Die Breite ist per Voreinstellung der Sensorradius: was der Sensor beim
    Vorbeifahren erfasst hat, gilt als besucht — nicht nur der Punkt unter dem
    Roboter.

    Der Wert waechst mit der Zahl der Punkte, die in der Naehe lagen, ist also
    ein Mass fuer die *dort verbrachte Zeit* und nicht nur fuer "war schon mal
    da". Genau das braucht die Abdeckungsschuld.
    """
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, res, device=device),
        torch.linspace(0, 1, res, device=device), indexing='ij')
    cells = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)   # (R*R, 2)
    d2 = torch.cdist(cells, path.to(device)) ** 2                   # (R*R, T)
    k = torch.exp(-d2 / (2.0 * bandwidth ** 2)).sum(dim=1)
    return k.view(res, res)


def debt_density(mu, sd, visit, kappa, args, floor=1e-6):
    r"""Zieldichte mit Abdeckungsschuld — die Zieldichte der Variante D.

    Gegenueber `ucb_density` aendern sich beide Summanden, und zwar getrennt:

    **Die Unsicherheit wird geloescht, wo schon gefahren wurde.**
    Der Gauss-Prozess senkt sigma dort zwar von selbst, aber nur ueber seine
    Korrelationslaenge und nie ganz auf null. Fuer eine Schleife, die alle zehn
    Prozent neu plant, ist das zu weich: das Restrauschen an einer eben
    abgefahrenen Stelle reicht, damit das Netz noch einmal hinfaehrt.

        sigma_eff = sigma * (1 - v)

    **Die Anziehung faellt proportional zu Wissen und verbrachter Zeit.**
    Ein Gebiet, das oft besucht *und* dabei gut vermessen wurde, hat seine
    Nachfrage bereits bedient — es soll nicht weiter ziehen. Ein Gebiet, das
    zwar besucht, aber weiterhin unsicher ist, behaelt seine Anziehung.

        mu_eff = mu * (1 - w * v * (1 - sigma))
                            ^   ^        ^
                            |   |        Wissen dort
                            |   verbrachte Zeit
                            debt_weight

    `v` ist die auf ihr Maximum bezogene und bei `--visit_sat` gesaettigte
    Aufenthaltsdichte: ab einem Viertel des Maximums gilt ein Gebiet als
    vollstaendig bedient. Ohne diese Saettigung wuerde eine einzige lange
    Verweildauer alle anderen Besuche kleinrechnen.
    """
    if visit is None:
        v = torch.zeros_like(mu)
    else:
        vmax = visit.max().clamp(min=1e-12)
        v = (visit / (args.visit_sat * vmax)).clamp(0.0, 1.0)

    sd_eff = sd * (1.0 - v)
    know = (1.0 - sd.clamp(0.0, 1.0))
    mu_eff = mu.clamp(min=0.0) * (1.0 - (args.debt_weight * v * know).clamp(0.0, 1.0))

    phi = zieldichte(mu_eff, sd_eff.clamp(min=1e-6), kappa, args)
    return phi, v


def run_variant_d(planner, truth, belief0, args, ergodic, generator=None):
    r"""Variante D mit dem trainierten Netz: lang planen, kurz fahren, umbuchen.

    Je Runde wird eine vollstaendige Bahn erzeugt, aber nur `--d_execute_frac`
    davon abgefahren. Danach wird der Glaube fortgeschrieben, die
    Abdeckungsschuld nachgezogen und neu geplant.

    **Der Anschluss.** Das Netz hat keinen Start-Eingang, die neu geplante Bahn
    beginnt also irgendwo. Bei zwanzig Runden waeren zwanzig Anfahrten fuer sich
    genommen schon eine erhebliche Strecke. Deshalb wird per Voreinstellung
    nicht am Anfang der neuen Bahn eingestiegen, sondern an ihrem der aktuellen
    Position naechstgelegenen Punkt (`--d_join nearest`) — und von dort ein
    Stueck weitergefahren. Die verbleibende Anfahrt wird trotzdem gefahren,
    gemessen und mitgezaehlt.
    """
    belief = belief0.clone()
    driven, log = [], []
    wallclock, transit_len = 0.0, 0.0
    R = args.d_rounds

    for r in range(R):
        mu, sd = belief.posterior_grid()
        kap = kappa_schedule(r, R, args.kappa0, args.kappa1)
        here = torch.cat(driven, dim=0) if driven else None
        visit = (visitation_field(here, belief.res, args.visit_bandwidth,
                                  args.device) if here is not None else None)
        phi, v = debt_density(mu, sd, visit, kap, args)

        parts = phi_particles(phi, args.n_particles, mode=args.phi_mode,
                              quantile=args.phi_quantile, device=args.device,
                              generator=generator)
        # Startpunkt-Konditionierung: die neue Bahn beginnt dort, wo der
        # Roboter gerade steht. Damit entfaellt sowohl der `nearest`-Einstieg
        # als auch die Anfahrt — beides waren Behelfe fuer ein Netz ohne
        # Start-Eingang, und bei zwanzig Runden summierten sich die Anfahrten
        # zu einer erheblichen Strecke.
        pos = here[-1] if here is not None else None
        kann_start = getattr(planner, 'start_cond', False) and pos is not None
        cps = (planner.plan(parts, n_candidates=args.n_candidates, start=pos)
               if kann_start else
               planner.plan(parts, n_candidates=args.n_candidates))
        wallclock += getattr(planner, 'last_wallclock', 0.0)
        curve = best_candidate(planner.render(cps), phi)

        T = curve.shape[0]
        k = max(2, int(round(args.d_execute_frac * T)))
        if kann_start and args.d_join == 'netz':
            seg = curve[:k]
        elif here is not None and args.d_join == 'nearest':
            i0 = int(torch.cdist(here[-1:].to(curve.device), curve).argmin())
            idx = (torch.arange(i0, i0 + k, device=curve.device) % T)
            seg = curve[idx]
        else:
            seg = curve[:k]

        if here is not None and not (kann_start and args.d_join == 'netz'):
            al = torch.linspace(0, 1, args.transit_pts,
                                device=curve.device).unsqueeze(-1)
            link = here[-1].to(curve.device).unsqueeze(0) * (1 - al) \
                   + seg[0].unsqueeze(0) * al
            transit_len += path_length(link)
            seg = torch.cat([link, seg], dim=0)
        elif here is not None:
            # Restsprung messen, falls das Netz den Startpunkt nicht exakt
            # trifft — er wird gefahren und gezaehlt wie jede andere Strecke.
            transit_len += float(torch.linalg.norm(
                seg[0] - here[-1].to(seg.device)))

        driven.append(seg.detach())
        before = belief.clone()
        pts, vals = measure(seg.detach(), truth, noise_std=args.noise,
                            sensor_radius=args.sensor_radius)
        belief.observe(*thin(pts, vals, max_points=args.max_obs))

        if r % max(1, R // 3) == 0 or r == R - 1:
            log.append({'round': r, 'kappa': kap,
                        'phi': phi.detach().cpu().numpy(),
                        'curve': seg.detach().cpu().numpy(),
                        'visit': (v.detach().cpu().numpy() if visit is not None
                                  else np.zeros_like(phi.cpu().numpy())),
                        'info_gain': information_gain(before, belief),
                        'belief_rmse': belief_rmse(belief, truth)})

    path = torch.cat(driven, dim=0)
    return {
        'path': path, 'rounds': log, 'belief': belief, 'wallclock': wallclock,
        'coverage': float(coverage_vs_truth(path, truth)),
        'ergodic': ergodic(path, truth),
        'info_gain': information_gain(belief0, belief),
        'belief_rmse': belief_rmse(belief, truth),
        'path_len': path_length(path),
        'transit_len': transit_len,
    }


def run_two_stage(planner, truth, belief0, args, ergodic, generator=None):
    """Variante E mit dem trainierten Netz: erst sigma, dann mu.

    Die zwei Phasen sind zwei getrennte Konditionierungen desselben,
    unveraenderten Netzes:

      Phase 1   Phi = sigma            reine Unsicherheit. mu wird auf Null
                                       gesetzt, nicht bloss klein gewichtet —
                                       die Phase soll die Aufgabe wirklich
                                       nicht kennen.
      Phase 2   Phi = mu               die inzwischen *aufgedeckte* Dichte.
                                       sigma faellt weg, es wird nur noch
                                       abgedeckt, was Phase 1 gefunden hat.

    Zwei Verluste nacheinander sind wohlgestellt, zwei gleichzeitig nicht —
    das ist der Grund, warum diese Zerlegung ueberhaupt als Baseline taugt.
    Der Preis ist das doppelte Budget, und der wird hier auch wirklich bezahlt:

    **Die Verbindungsfahrt wird mitgezaehlt.** Das CFM-Netz hat keinen
    Start-Eingang — anders als `GradientPlanner`, der `start=` kennt. Phase 2
    beginnt also irgendwo, nicht dort, wo Phase 1 endete. Diese Luecke
    stillschweigend zu ueberspringen wuerde E eine Strecke schenken, die ein
    Roboter fahren muesste. Sie wird deshalb als gerades Stueck eingefuegt,
    abgemessen *und* gemessen: der Sensor laeuft waehrend der Verbindungsfahrt
    ja weiter.
    """
    belief = belief0.clone()
    wallclock = 0.0
    log = []

    def _phase(phi, tag):
        nonlocal wallclock
        parts = phi_particles(phi, args.n_particles, mode=args.phi_mode,
                              quantile=args.phi_quantile, device=args.device,
                              generator=generator)
        cps = planner.plan(parts, n_candidates=args.n_candidates)
        wallclock += getattr(planner, 'last_wallclock', 0.0)
        curve = best_candidate(planner.render(cps), phi)
        log.append({'round': len(log), 'kappa': float('nan'), 'tag': tag,
                    'phi': phi.detach().cpu().numpy(),
                    'curve': curve.detach().cpu().numpy()})
        return curve

    # ── Phase 1: reine Erkundung ───────────────────────────────────────────
    mu, sd = belief.posterior_grid()
    phi_explore = zieldichte(torch.zeros_like(mu), sd, 1.0, args)
    curve1 = _phase(phi_explore, 'sigma')

    before = belief.clone()
    pts, vals = measure(curve1.detach(), truth,
                        noise_std=args.noise, sensor_radius=args.sensor_radius)
    belief.observe(*thin(pts, vals, max_points=args.max_obs))
    log[-1]['info_gain'] = information_gain(before, belief)
    log[-1]['belief_rmse'] = belief_rmse(belief, truth)

    # ── Phase 2: reine Abdeckung des Aufgedeckten ──────────────────────────
    mu2, sd2 = belief.posterior_grid()
    phi_cover = zieldichte(mu2, torch.zeros_like(sd2) + 1e-6, 0.0, args)
    if is_degenerate(phi_cover):
        print("      [warn] mu ist nach Phase 1 praktisch flach — Phase 2 hat "
              "keine aufgedeckte Dichte, die sie abdecken koennte.")
    curve2 = _phase(phi_cover, 'mu')

    # ── Verbindungsfahrt, ehrlich abgerechnet ──────────────────────────────
    a_ = torch.linspace(0, 1, args.transit_pts,
                        device=curve1.device).unsqueeze(-1)
    transit = curve1[-1].unsqueeze(0) * (1 - a_) + curve2[0].unsqueeze(0) * a_

    before2 = belief.clone()
    seg = torch.cat([transit, curve2], dim=0)
    pts, vals = measure(seg.detach(), truth,
                        noise_std=args.noise, sensor_radius=args.sensor_radius)
    belief.observe(*thin(pts, vals, max_points=args.max_obs))
    log[-1]['info_gain'] = information_gain(before2, belief)
    log[-1]['belief_rmse'] = belief_rmse(belief, truth)

    path = torch.cat([curve1, transit, curve2], dim=0).detach()
    return {
        'path': path,
        'rounds': log,
        'belief': belief,
        'wallclock': wallclock,
        'coverage': float(coverage_vs_truth(path, truth)),
        'ergodic': ergodic(path, truth),
        'info_gain': information_gain(belief0, belief),
        'belief_rmse': belief_rmse(belief, truth),
        'path_len': path_length(path),
        'transit_len': path_length(transit),
        'cov_phase2': float(coverage_vs_truth(curve2.detach(), truth)),
    }


# ===========================================================================
# Visualisierung (Projektstil)
# ===========================================================================

def _white_inferno():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mcolors.LinearSegmentedColormap.from_list('white_inferno', inf)


def _style(ax):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')


def plot_shapes(records, out_path, rounds, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cmap = _white_inferno()
    n = len(records)
    ncols = 2 + rounds
    fig, axes = plt.subplots(n, ncols, figsize=(2.6 * ncols, 2.6 * n),
                             facecolor='white', squeeze=False)

    for i, rec in enumerate(records):
        truth = rec['truth']
        # Spalte 0: Wahrheit + Orakelbahn
        ax = axes[i][0]; _style(ax)
        ax.imshow(truth, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                  alpha=0.55, vmin=0.0)
        obs = rec['prior_pts']
        if obs is not None and len(obs):
            ax.scatter(obs[:, 0], obs[:, 1], s=6, alpha=0.3, color='#444444')
        mk = rec.get('prior_maske')
        if mk is not None:
            # Unbekanntes Gebiet grau ueberlegen, damit die Kante sichtbar ist
            import numpy as _np
            ueber = _np.zeros(mk.shape + (4,), dtype=float)
            ueber[..., :3] = 0.35
            ueber[..., 3] = (~mk.astype(bool)) * 0.30
            ax.imshow(ueber, origin='lower', extent=[0, 1, 0, 1], zorder=2)
        if 'orakel' in rec:
            o = rec['orakel']['path'].cpu().numpy()
            ax.plot(o[:, 0], o[:, 1], color='#1565C0', lw=2.5, alpha=0.9)
        ax.set_ylabel(rec['name'], color='#1A1A2E', fontsize=11)
        if i == 0:
            ax.set_title('Wahrheit + Orakel', color='#1A1A2E', fontsize=10)

        # Spalten 1..R: Phi pro Runde + die dort geplante Bahn
        for r in range(rounds):
            ax = axes[i][1 + r]; _style(ax)
            if 'glaube' not in rec or r >= len(rec['glaube']['rounds']):
                ax.axis('off'); continue
            rd = rec['glaube']['rounds'][r]
            ax.imshow(rd['phi'], origin='lower', extent=[0, 1, 0, 1],
                      cmap=cmap, alpha=0.55, vmin=0.0, vmax=1.0)
            c = rd['curve']
            ax.plot(c[:, 0], c[:, 1], color='#00C853', lw=2.2, alpha=0.95)
            if i == 0:
                ax.set_title(f"$\\Phi$ Runde {r + 1}  ($\\kappa$={rd['kappa']:.1f})",
                             color='#1A1A2E', fontsize=10)

        # letzte Spalte: Wahrheit + gesamte gefahrene Bahn
        ax = axes[i][-1]; _style(ax)
        ax.imshow(truth, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                  alpha=0.55, vmin=0.0)
        if 'glaube' in rec:
            g = rec['glaube']['path'].cpu().numpy()
            ax.plot(g[:, 0], g[:, 1], color='#00C853', lw=2.2, alpha=0.95)
        if i == 0:
            ax.set_title('Wahrheit + Gesamtbahn', color='#1A1A2E', fontsize=10)

    fig.suptitle(title, color='#1A1A2E', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130, facecolor='white')
    plt.close(fig)
    print(f"  [viz] {out_path}")


def plot_two_stage(records, out_path):
    """Variante E: was jede Phase sieht und was sie daraus macht.

    Vier Spalten, weil genau vier Bilder die Zerlegung tragen: die Dichte, auf
    die Phase 1 konditioniert (reines sigma), das Ergebnis dieser Fahrt, die
    Dichte, die daraus aufgedeckt wurde (reines mu), und die Gesamtbahn
    inklusive der Verbindungsfahrt. Die Verbindungsfahrt ist gestrichelt —
    sie ist die Strecke, die das Verfahren kostet, ohne dass sie geplant wurde.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cmap = _white_inferno()
    recs = [r for r in records if 'zweistufig' in r]
    if not recs:
        return
    fig, axes = plt.subplots(len(recs), 4, figsize=(10.6, 2.7 * len(recs)),
                             facecolor='white', squeeze=False)
    titles = ['$\\Phi=\\sigma$ (Phase 1)', 'Wahrheit + Phase 1',
              '$\\Phi=\\mu$ (Phase 2, aufgedeckt)', 'Wahrheit + Gesamtbahn']

    for i, rec in enumerate(recs):
        E = rec['zweistufig']
        truth, log = rec['truth'], E['rounds']
        c1 = log[0]['curve']
        c2 = log[1]['curve']

        for j, (fieldname, field) in enumerate(
                [('phi1', log[0]['phi']), ('truth', truth),
                 ('phi2', log[1]['phi']), ('truth', truth)]):
            ax = axes[i][j]; _style(ax)
            vmax = 1.0 if fieldname.startswith('phi') else None
            ax.imshow(field, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                      alpha=0.55, vmin=0.0, vmax=vmax)
            if i == 0:
                ax.set_title(titles[j], color='#1A1A2E', fontsize=10)

        axes[i][0].plot(c1[:, 0], c1[:, 1], color='#00C853', lw=2.2, alpha=0.95)
        axes[i][1].plot(c1[:, 0], c1[:, 1], color='#00C853', lw=2.2, alpha=0.95)
        axes[i][2].plot(c2[:, 0], c2[:, 1], color='#D81B60', lw=2.2, alpha=0.95)

        ax = axes[i][3]
        ax.plot(c1[:, 0], c1[:, 1], color='#00C853', lw=2.2, alpha=0.6)
        ax.plot([c1[-1, 0], c2[0, 0]], [c1[-1, 1], c2[0, 1]],
                color='#444444', lw=1.4, ls='--', alpha=0.9)
        ax.plot(c2[:, 0], c2[:, 1], color='#D81B60', lw=2.2, alpha=0.95)
        ax.set_xlabel(f"Verbindung {E['transit_len']:.2f} von "
                      f"{E['path_len']:.2f}", color='#555', fontsize=9)
        axes[i][0].set_ylabel(rec['name'], color='#1A1A2E', fontsize=11)

    fig.suptitle('Variante E mit dem trainierten CFM+ErgLoss-Netz — '
                 'erst $\\sigma$, dann $\\mu$', color='#1A1A2E', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130, facecolor='white')
    plt.close(fig)
    print(f"  [viz] {out_path}")


def plot_anytime(curves_by_mission, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keys = [('coverage', 'Abdeckungsabstand (kleiner besser)'),
            ('info_gain', 'Informationsgewinn (groesser besser)'),
            ('belief_rmse', 'RMSE des Glaubens (kleiner besser)')]
    colors = {'orakel': '#1565C0', 'glaube-1': '#7E57C2', 'glaube-R': '#00C853',
              'zweistufig': '#D81B60', 'glaube-D': '#00838F',
              'grad-R': '#EF6C00', 'maeher': '#607D8B',
              'B-warm': '#00838F', 'B-kalt': '#9E9E9E', 'B-auswahl': '#5E35B1'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor='white')
    for ax, (key, lab) in zip(axes, keys):
        for name, runs in curves_by_mission.items():
            if not runs:
                continue
            L = min(len(r) for r in runs)
            xs = np.mean([[c['path_len'] for c in r[:L]] for r in runs], axis=0)
            ys = np.mean([[c[key] for c in r[:L]] for r in runs], axis=0)
            ax.plot(xs, ys, label=name, lw=2.2,
                    color=colors.get(name, '#444444'))
        ax.set_xlabel('gefahrene Weglaenge', color='#555')
        ax.set_title(lab, color='#1A1A2E', fontsize=11)
        ax.grid(alpha=0.2)
        for s in ax.spines.values():
            s.set_color('#ccc')
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, color='#1A1A2E', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130, facecolor='white')
    plt.close(fig)
    print(f"  [viz] {out_path}")


# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default=None,
                   help='Checkpoint des trainierten CFM+ErgLoss-Netzes. Ohne '
                        'Angabe wird der neueste Treffer von --ckpt_glob '
                        'genommen.')
    p.add_argument('--ckpt_glob',
                   default=os.path.join(_arch, 'checkpoints', '*ERGLOSS*.pt'),
                   help='Suchmuster, wenn --ckpt fehlt.')
    p.add_argument('--dry_run', action='store_true',
                   help='Zufaellig initialisiertes Netz — testet nur, ob die '
                        'Pipeline durchlaeuft. Die Zahlen sind bedeutungslos.')
    p.add_argument('--shapes', type=int, default=12)
    p.add_argument('--truth_res', type=int, default=128,
                   help='Aufloesung der Wahrheit; 128 wie im Training.')
    p.add_argument('--gp_res', type=int, default=64)
    p.add_argument('--lengthscale', type=float, default=0.08)
    p.add_argument('--gp_noise', type=float, default=0.05,
                   help='Rauschterm des GP. Bewusst groesser als das echte '
                        'Messrauschen (--noise): Messpunkte auf einer Bahn '
                        'liegen dicht beieinander und sind keine unabhaengigen '
                        'Stichproben. Mit 1e-2 interpoliert der GP das Rauschen '
                        'und der Posterior-Mittelwert schiesst nach [-2.5, 3.9] '
                        'ueber, obwohl die Wahrheit in [0,1] liegt.')
    p.add_argument('--prior_pattern', default='zufall',
                   choices=['zufall', 'haelfte', 'quadranten', 'loch'],
                   help='Wie die Vormessungen verteilt werden. `zufall` ist '
                        'das bisherige Verhalten und bleibt Voreinstellung.')
    p.add_argument('--prior_rand', type=float, default=0.0,
                   help='Anteil der Vormessungen, der trotz Muster ueber die '
                        'ganze Flaeche gestreut wird.')
    p.add_argument('--n_prior', type=int, default=12,
                   help='Vormessungen. Bei 0 ist Phi fuer *jedes* kappa '
                        'gleichverteilt und der Vergleich bedeutungslos.')
    p.add_argument('--n_particles', type=int, default=256)
    p.add_argument('--kappa0', type=float, default=3.0)
    p.add_argument('--kappa1', type=float, default=0.3)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--execute_frac', type=float, default=1.0)
    p.add_argument('--phi_model', default='ucb', choices=sorted(PHI_MODELLE),
                   help='Wie aus dem Glauben eine Zieldichte wird. `ucb` ist '
                        'die bisherige Modellierung und bleibt die '
                        'Voreinstellung.')
    p.add_argument('--phi_tau', type=float, default=0.25,
                   help='Schwelle des Niveaumengen-Modells.')
    p.add_argument('--phi_xi', type=float, default=0.01,
                   help='Mindestzugewinn beim Modell `ei`.')
    p.add_argument('--phi_gamma', type=float, default=1.0,
                   help='Bereits eingeholte Information bei `mi`.')
    p.add_argument('--prior_mode', default='messungen',
                   choices=['messungen', 'wahrheit'],
                   help="Wie das Vorwissen entsteht. 'messungen' zieht "
                        "--n_prior Punktmessungen im bekannten Gebiet und "
                        "laesst den GP interpolieren (bisheriges Verhalten). "
                        "'wahrheit' stellt dort die Grundwahrheit exakt zur "
                        "Verfuegung (sigma = 0) und ausserhalb gar kein Wissen.")
    p.add_argument('--sigma_bekannt', type=float, default=0.0,
                   help='Rest-Unsicherheit im bekannten Gebiet bei '
                        '--prior_mode wahrheit.')
    p.add_argument('--phi_mode', default='uniform',
                   choices=['uniform', 'density', 'quantile'])
    p.add_argument('--phi_quantile', type=float, default=0.5)
    p.add_argument('--n_candidates', type=int, default=1)
    p.add_argument('--no_two_stage', action='store_true',
                   help='Variante E (erst sigma, dann mu) weglassen.')
    p.add_argument('--missions', nargs='+', default=None,
                   choices=['orakel', 'glaube-1', 'glaube-R', 'zweistufig',
                            'glaube-D', 'B-warm', 'B-kalt', 'B-auswahl',
                            'grad-R', 'maeher'],
                   help='Nur diese Missionen fahren. Ohne Angabe bleibt das '
                        'bisherige Verhalten unveraendert: alle ausser den '
                        'B-Missionen, die weiterhin an --variant_b haengen.')
    p.add_argument('--variant_b', action='store_true',
                   help='Variante B mitmessen: gelernter Warmstart der '
                        'differenzierbaren Verfeinerung, dieselbe Verfeinerung '
                        'aus Rauschen als Kontrolle, und Kandidatenauswahl '
                        'per Vorausschau ohne jede Optimierung.')
    p.add_argument('--refine_steps', type=int, default=100)
    p.add_argument('--refine_lr', type=float, default=0.03)
    p.add_argument('--n_probe', type=int, default=40)
    p.add_argument('--lambda_unc', type=float, default=1.0)
    p.add_argument('--lambda_cov', type=float, default=20000.0,
                   help='Der Unsicherheitsterm liegt bei mehreren hundert, der '
                        'ergodische Fehler bei 1e-2 — ohne 1e4 ist die '
                        'Abdeckung im Gesamtziel praktisch nicht vertreten.')
    p.add_argument('--select_k', type=int, default=8)
    p.add_argument('--d_rounds', type=int, default=20,
                   help='Planungsschritte der Variante D.')
    p.add_argument('--d_execute_frac', type=float, default=0.10,
                   help='Anteil der geplanten Bahn, der je Runde gefahren wird.')
    p.add_argument('--d_join', default='nearest', choices=['nearest', 'start', 'netz'],
                   help='Wo in die neu geplante Bahn eingestiegen wird. '
                        '`nearest` haelt die Anfahrten kurz, `start` faehrt '
                        'stur ab Bahnanfang und macht die Kosten sichtbar.')
    p.add_argument('--visit_bandwidth', type=float, default=None,
                   help='Kernbreite der Aufenthaltsdichte. Ohne Angabe der '
                        'Sensorradius.')
    p.add_argument('--visit_sat', type=float, default=0.25,
                   help='Ab diesem Anteil des Maximums gilt ein Gebiet als '
                        'vollstaendig bedient.')
    p.add_argument('--debt_weight', type=float, default=1.0,
                   help='Wie stark besuchte Gebiete ihre Anziehung verlieren. '
                        '0 schaltet die Abdeckungsschuld ab und laesst nur das '
                        'Loeschen der Unsicherheit uebrig.')
    p.add_argument('--transit_pts', type=int, default=16,
                   help='Stuetzpunkte der Verbindungsfahrt zwischen den beiden '
                        'Phasen von Variante E. Das Netz hat keinen '
                        'Start-Eingang, also beginnt Phase 2 nicht dort, wo '
                        'Phase 1 endete — die Strecke wird mitgezaehlt.')
    p.add_argument('--cfg_weight', type=float, default=2.0)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--nxi', type=int, default=25)
    p.add_argument('--pts', type=int, default=128)
    p.add_argument('--noise', type=float, default=0.02)
    p.add_argument('--sensor_radius', type=float, default=0.06)
    p.add_argument('--max_obs', type=int, default=96)
    p.add_argument('--grad_steps', type=int, default=200)
    p.add_argument('--no_grad_baseline', action='store_true')
    p.add_argument('--grad_target_length', type=float, default=None,
                   help='Laengenvorgabe je Runde fuer den Gradientenvergleich. '
                        'Ohne Angabe: der Median der vom Netz erzeugten Laenge, '
                        'damit beide Verfahren dieselbe Strecke fahren.')
    p.add_argument('--anytime_points', type=int, default=12)
    p.add_argument('--budgets', type=float, nargs='+', default=[0.33, 0.66, 1.0],
                   help='Vergleichsbudgets als Vielfache der glaube-R-Laenge.')
    p.add_argument('--viz_shapes', type=int, default=4)
    p.add_argument('--save_paths', action='store_true',
                   help='Alle gefahrenen Bahnen als bahnen.json ablegen — je '
                        'Form und Mission. Damit lassen sich saemtliche '
                        'Missionen nachtraeglich zeichnen, ohne den Lauf zu '
                        'wiederholen.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'cfm_belief'))
    p.add_argument('--device', default=None)
    a = p.parse_args()

    a.device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if a.visit_bandwidth is None:
        a.visit_bandwidth = max(a.sensor_radius, 1e-3)
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    if a.ckpt is None and not a.dry_run:
        import glob
        hits = sorted(glob.glob(a.ckpt_glob), key=os.path.getmtime)
        if not hits:
            p.error(f'Kein Checkpoint unter {a.ckpt_glob} gefunden — '
                    '--ckpt explizit angeben (oder --dry_run).')
        a.ckpt = hits[-1]
        print(f"Checkpoint automatisch gewaehlt ({len(hits)} Treffer):")
        for h in hits[-5:]:
            print(f"    {'->' if h == a.ckpt else '  '} {os.path.basename(h)}")

    print(f"Geraet: {a.device}   Checkpoint: {a.ckpt or '(dry run)'}")
    # Die Wahrheit wird gleich auf dem Rechengeraet geladen. Damit gibt es im
    # ganzen Lauf nur *ein* Geraet, und die Klasse von Fehlern, an der Job
    # 133748 gescheitert ist ("cuda:0 and cpu"), kann strukturell nicht mehr
    # auftreten. Auf die CPU zurueck geht es nur an der Zeichenkante, wo
    # Matplotlib NumPy-Arrays braucht.
    names, truths = load_truth(n=a.shapes, split='val',
                               resolution=a.truth_res, device=a.device)
    print(f"Holdout-Formen: {', '.join(names)}")

    planner = CfmPlanner(ckpt=a.ckpt, nxi=a.nxi, pts=a.pts, steps=a.steps,
                         cfg_weight=a.cfg_weight, n_particles=a.n_particles,
                         device=a.device)
    # Der Gradientenplaner wird auch dann gebraucht, wenn `grad-R` gar nicht
    # gefahren wird: Variante B benutzt seine B-Spline-Basis, seinen
    # ergodischen Verlust und seine Regularisierer fuer die Verfeinerung.
    grad = None
    need_grad = (not a.no_grad_baseline) and (
        a.missions is None or 'grad-R' in a.missions
        or any(m.startswith('B-') for m in a.missions))
    if need_grad:
        grad = GradientPlanner(nxi=a.nxi, pts=a.pts, K=8, steps=a.grad_steps,
                               device=a.device)
    ergodic = ErgodicScore(K=8, device=a.device)

    records, rows = [], []
    if a.missions:
        # Reihenfolge der Ausgabe bleibt die kanonische, nicht die der Eingabe.
        keys = [k for k in ('orakel','glaube-1','glaube-R','zweistufig','glaube-D','B-warm','B-kalt','B-auswahl','grad-R','maeher') if k in a.missions]
        a.variant_b = any(k.startswith('B-') for k in keys)
    else:
        keys = ['orakel', 'glaube-1', 'glaube-R', 'zweistufig']
        if a.variant_b:
            keys += ['B-warm', 'B-kalt', 'B-auswahl']
        keys += ['grad-R', 'maeher']
    want = set(keys)
    print(f"Missionen: {', '.join(keys)}")
    if a.prior_mode == 'wahrheit':
        print(f"Vorwissen: Muster '{a.prior_pattern}', Grundwahrheit im "
              f"bekannten Gebiet (sigma={a.sigma_bekannt}), ausserhalb kein Wissen")
    elif a.prior_pattern != 'zufall':
        print(f"Vorwissen: Muster '{a.prior_pattern}', {a.n_prior} Messungen"
              + (f", davon {a.prior_rand:.0%} gestreut" if a.prior_rand else ""))
    anytime = {k: [] for k in keys}
    probe_fn = _probe_points_fn() if a.variant_b else None

    # ---- Durchgang 1: alles, was das Netz plant ---------------------------
    # Der Gradienten-Vergleich laeuft bewusst erst danach. Ohne Laengenvorgabe
    # faehrt er von sich aus rund ein Zehntel der Strecke des Netzes, und ein
    # Vergleich bei gleicher Weglaenge waere dann fuer jedes sinnvolle Budget
    # leer. Er bekommt deshalb die Laenge, die das Netz tatsaechlich erzeugt
    # hat — bekannt erst, nachdem das Netz gelaufen ist.
    for i, name in enumerate(names):
        truth = truths[i]
        print(f"\n[{i + 1}/{len(names)}] {name}")
        gen = torch.Generator(device=a.device).manual_seed(a.seed * 1000 + i)

        if a.prior_mode == 'wahrheit':
            # Im bekannten Gebiet liegt die Grundwahrheit vor, ausserhalb gar
            # kein Wissen. Das ist etwas anderes als `messungen`, wo dort nur
            # n Punktmessungen gezogen werden und der GP dazwischen raet.
            maske = muster_maske(a.prior_pattern, a.gp_res, device=a.device)
            wahr = torch.nn.functional.interpolate(
                truth[None, None].to(a.device), size=(a.gp_res, a.gp_res),
                mode='bilinear', align_corners=True)[0, 0]
            b0 = MaskiertesWissen(maske, wahr, sigma_bekannt=a.sigma_bekannt,
                                  grid_res=a.gp_res, lengthscale=a.lengthscale,
                                  noise=a.gp_noise, device=a.device)
            prior_pts = None
            prior_maske = maske.cpu().numpy()
        else:
            b0 = GPBelief(grid_res=a.gp_res, lengthscale=a.lengthscale,
                          noise=a.gp_noise, device=a.device)
            prior_maske = None
            if a.n_prior > 0:
                g = torch.Generator().manual_seed(a.seed * 977 + i)
                pp = prior_points(a.prior_pattern, a.n_prior, generator=g,
                                  device=a.device, rand=a.prior_rand)
                _, vv = measure(pp, truth, noise_std=a.noise)
                b0.observe(pp, vv)
                prior_pts = pp.cpu().numpy()
            else:
                prior_pts = None

        rec = {'name': name, 'truth': truth.cpu().numpy(), 'b0': b0, 'gen': gen,
               'prior_pts': prior_pts, 'prior_maske': prior_maske}
        if 'orakel' in want:
            rec['orakel'] = run_mission(planner, truth, b0, a, 'cfm', 1,
                                        ergodic, use_truth_density=True,
                                        generator=gen)
        if 'glaube-1' in want:
            rec['glaube-1'] = run_mission(planner, truth, b0, a, 'cfm', 1,
                                          ergodic, generator=gen)
        if 'glaube-R' in want:
            rec['glaube'] = run_mission(planner, truth, b0, a, 'cfm', a.rounds,
                                        ergodic, generator=gen)
        if 'glaube-D' in want:
            rec['glaube-D'] = run_variant_d(planner, truth, b0, a, ergodic,
                                            generator=gen)
        if 'zweistufig' in want and not a.no_two_stage:
            rec['zweistufig'] = run_two_stage(planner, truth, b0, a, ergodic,
                                              generator=gen)
        if a.variant_b:
            if grad is None:
                p.error('--variant_b braucht den Gradientenpfad; '
                        '--no_grad_baseline schliesst ihn aus.')
            if 'B-warm' in want:
                rec['B-warm'] = run_variant_b(planner, grad, truth, b0, a,
                                              ergodic, probe_fn, warm=True,
                                              generator=gen)
            if 'B-kalt' in want:
                rec['B-kalt'] = run_variant_b(planner, grad, truth, b0, a,
                                              ergodic, probe_fn, warm=False,
                                              generator=gen)
            if 'B-auswahl' in want:
                rec['B-auswahl'] = run_selection(planner, truth, b0, a,
                                                 ergodic, probe_fn,
                                                 generator=gen)

        if 'maeher' not in want:
            records.append(rec)
            continue
        # Der Maeher bekommt die Laenge der laengsten tatsaechlich gefahrenen
        # Mission, damit er ein echter Vergleich bleibt und nicht an einer
        # Mission haengt, die vielleicht gar nicht gefahren wurde.
        ref_len = max(m['path_len'] for m in rec.values()
                      if isinstance(m, dict) and 'path_len' in m)
        lawn = lawnmower_path(n_points=a.pts * a.rounds,
                              target_length=ref_len).float().to(a.device)
        bl = b0.clone()
        lp, lv = measure(lawn, truth, noise_std=a.noise,
                         sensor_radius=a.sensor_radius)
        bl.observe(*thin(lp, lv, max_points=a.max_obs))
        rec['maeher'] = {
            'path': lawn, 'rounds': [], 'belief': bl, 'wallclock': 0.0,
            'coverage': float(coverage_vs_truth(lawn, truth)),
            'ergodic': ergodic(lawn, truth),
            'info_gain': information_gain(b0, bl),
            'belief_rmse': belief_rmse(bl, truth),
            'path_len': path_length(lawn),
        }
        for key in ('orakel', 'glaube-1', 'glaube', 'zweistufig', 'glaube-D',
                    'B-warm', 'B-kalt', 'B-auswahl', 'maeher'):
            # `rec` enthaelt ohnehin nur die gefahrenen Missionen
            if key not in rec:
                continue
            m = rec[key]
            print(f"    {key:9s} cov={m['coverage']:.4f} erg={m['ergodic']:.5f} "
                  f"ig={m['info_gain']:8.2f} rmse={m['belief_rmse']:.4f} "
                  f"len={m['path_len']:.2f} t={m['wallclock'] * 1e3:.0f}ms")
        records.append(rec)

    # ---- Durchgang 2: Gradientenvergleich auf die Netzlaenge gebracht -----
    if grad is not None and 'grad-R' in want:
        seg = float(np.median([r['glaube']['path_len'] for r in records
                               if 'glaube' in r] or [4.0])) / a.rounds
        grad.target_length = a.grad_target_length or seg
        print(f"\nGradientenvergleich mit Laengenvorgabe "
              f"{grad.target_length:.2f} je Runde ({a.rounds} Runden).")
        for rec in records:
            truth = truths[names.index(rec['name'])]
            rec['grad'] = run_mission(grad, truth, rec['b0'], a, 'grad',
                                      a.rounds, ergodic, generator=rec['gen'])
            m = rec['grad']
            print(f"  {rec['name']:16s} cov={m['coverage']:.4f} "
                  f"erg={m['ergodic']:.5f} ig={m['info_gain']:8.2f} "
                  f"rmse={m['belief_rmse']:.4f} len={m['path_len']:.2f}")

    # ---- Anytime-Kurven und Zeilen ----------------------------------------
    for rec in records:
        truth = truths[names.index(rec['name'])]
        for key, mission in (('orakel', 'orakel'), ('glaube-1', 'glaube-1'),
                             ('glaube-R', 'glaube'), ('zweistufig', 'zweistufig'),
                             ('glaube-D', 'glaube-D'),
                             ('B-warm', 'B-warm'), ('B-kalt', 'B-kalt'),
                             ('B-auswahl', 'B-auswahl'),
                             ('grad-R', 'grad'), ('maeher', 'maeher')):
            if mission not in rec:
                continue
            m = rec[mission]
            anytime[key].append(anytime_curve(
                m['path'], truth, rec['b0'], n_points=a.anytime_points,
                noise=a.noise, sensor_radius=a.sensor_radius,
                max_obs=a.max_obs))
            rows.append({
                'shape': rec['name'], 'mission': key,
                'coverage': m['coverage'], 'ergodic': m['ergodic'],
                'info_gain': m['info_gain'], 'belief_rmse': m['belief_rmse'],
                'path_len': m['path_len'], 'wallclock': m['wallclock'],
            })

    # ---- Zusammenfassung --------------------------------------------------
    import csv
    csv_path = os.path.join(a.out_dir, 'metriken.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  [csv] {csv_path}")

    print("\n" + "=" * 78)
    print(f"{'Mission':12s} {'Abdeckung':>11s} {'Ergodisch':>11s} "
          f"{'InfoGewinn':>11s} {'RMSE':>9s} {'Laenge':>8s} {'ms':>7s}")
    print("-" * 78)
    for key in anytime.keys():
        sel = [r for r in rows if r['mission'] == key]
        if not sel:
            continue
        m = {k: float(np.mean([r[k] for r in sel]))
             for k in ('coverage', 'ergodic', 'info_gain', 'belief_rmse',
                       'path_len', 'wallclock')}
        print(f"{key:12s} {m['coverage']:11.4f} {m['ergodic']:11.5f} "
              f"{m['info_gain']:11.2f} {m['belief_rmse']:9.4f} "
              f"{m['path_len']:8.2f} {m['wallclock'] * 1e3:7.0f}")
    print("=" * 78)

    # ---- Vergleich bei gleicher Weglaenge ---------------------------------
    # Nicht ein einziges Minimum ueber alle Missionen: die kuerzeste Bahn
    # bestimmt sonst allein den Vergleichspunkt, und alle anderen werden bei
    # einem Budget beurteilt, das sie nach wenigen Prozent ihrer Fahrt
    # erreichen. Stattdessen mehrere Budgets, abgeleitet von der Bahnlaenge
    # der eigentlich untersuchten Mission (glaube-R), und ein '—' fuer jede
    # Mission, die ein Budget gar nicht ausschoepft — das ist eine Aussage
    # ueber die Mission, kein fehlender Wert.
    # Bezugslaenge: glaube-R, wenn gefahren, sonst die laengste vorhandene.
    _refkey = 'glaube-R' if anytime.get('glaube-R') else max(
        (k for k in anytime if anytime[k]),
        key=lambda k: np.median([c[-1]['path_len'] for c in anytime[k]]))
    ref = float(np.median([c[-1]['path_len'] for c in anytime[_refkey]]))
    budgets = [f * ref for f in a.budgets]
    print(f"\nAbdeckung bei fester Weglaenge (Referenz {_refkey} = {ref:.2f}):")
    head = ' '.join(f"{b:>9.2f}" for b in budgets)
    print(f"{'Mission':12s} {head}")
    print('-' * (13 + 10 * len(budgets)))
    for key, runs in anytime.items():
        if not runs:
            continue
        cells = []
        for bud in budgets:
            vals = [at_budget(c, bud, 'coverage') for c in runs]
            vals = [v for v in vals if v is not None]
            # Nur werten, wenn die Mehrheit der Formen das Budget erreicht.
            cells.append(f"{np.mean(vals):9.4f}" if len(vals) > len(runs) // 2
                         else f"{'—':>9s}")
        print(f"{key:12s} " + ' '.join(cells))

    with open(os.path.join(a.out_dir, 'anytime.json'), 'w') as f:
        json.dump(anytime, f)

    if a.save_paths:
        dump = {'meta': {k: v for k, v in vars(a).items()
                         if isinstance(v, (int, float, str, bool, type(None)))},
                'formen': []}
        for rec in records:
            e = {'name': rec['name'],
                 'truth': rec['truth'].tolist(),
                 'prior': (rec['prior_pts'].tolist()
                           if rec['prior_pts'] is not None else []),
                 'maske': (rec['prior_maske'].astype('uint8').tolist()
                           if rec.get('prior_maske') is not None else None),
                 'missionen': {}}
            for key, mission in (('orakel', 'orakel'), ('glaube-1', 'glaube-1'),
                                 ('glaube-R', 'glaube'),
                                 ('zweistufig', 'zweistufig'),
                                 ('glaube-D', 'glaube-D'),
                                 ('B-warm', 'B-warm'), ('B-kalt', 'B-kalt'),
                                 ('B-auswahl', 'B-auswahl'), ('grad-R', 'grad'),
                                 ('maeher', 'maeher')):
                if mission not in rec:
                    continue
                m = rec[mission]
                e['missionen'][key] = {
                    'bahn': m['path'].cpu().numpy().round(4).tolist(),
                    'coverage': m['coverage'], 'ergodic': m['ergodic'],
                    'belief_rmse': m['belief_rmse'], 'path_len': m['path_len'],
                    'phi': [r['phi'].round(4).tolist() for r in m['rounds']],
                    'visit': [r['visit'].round(4).tolist()
                              for r in m['rounds'] if 'visit' in r],
                    'kappa': [r['kappa'] for r in m['rounds']],
                    'runde': [r['round'] for r in m['rounds']],
                }
            dump['formen'].append(e)
        pth = os.path.join(a.out_dir, 'bahnen.json')
        with open(pth, 'w') as f:
            json.dump(dump, f)
        print(f"  [json] {pth}")

    plot_shapes(records[:a.viz_shapes],
                os.path.join(a.out_dir, 'formen.png'), a.rounds,
                f'Trainiertes CFM+ErgLoss-Netz, $\\Phi$ = {a.phi_model}, '
                f'phi_mode={a.phi_mode}, Vorwissen: {a.prior_pattern} '
                f'({a.prior_mode})')
    plot_anytime(anytime, os.path.join(a.out_dir, 'anytime.png'),
                 'Guete gegen gefahrene Weglaenge')
    plot_two_stage(records[:a.viz_shapes],
                   os.path.join(a.out_dir, 'zweistufig.png'))

    if a.variant_b and all('B-warm' in r and 'B-kalt' in r and 'B-auswahl' in r
                           for r in records):
        w = [r['B-warm'] for r in records]
        c = [r['B-kalt'] for r in records]
        sel = [r['B-auswahl'] for r in records]
        print("\nVariante B mit dem Netz:")
        print(f"  Warmstart   Abdeckung {np.mean([x['coverage'] for x in w]):.4f}"
              f"   davon Netz {np.mean([x['net_s'] for x in w]) * 1e3:.0f} ms, "
              f"Verfeinerung {np.mean([x['refine_s'] for x in w]) * 1e3:.0f} ms")
        print(f"  Kaltstart   Abdeckung {np.mean([x['coverage'] for x in c]):.4f}"
              f"   (gleiche Schrittzahl, Start aus Rauschen)")
        print(f"  Auswahl     Abdeckung {np.mean([x['coverage'] for x in sel]):.4f}"
              f"   {a.select_k} Kandidaten, keine Optimierung, "
              f"{np.mean([x['wallclock'] for x in sel]) * 1e3:.0f} ms")
        print(f"  Der Warmstart startet bei Abdeckung "
              f"{np.mean([x['cov_start'] for x in w]):.4f} und endet bei "
              f"{np.mean([x['coverage'] for x in w]):.4f}.")
    elif any('B-warm' in r for r in records):
        w = [r['B-warm'] for r in records if 'B-warm' in r]
        print(f"\nB-warm: Abdeckung {np.mean([x['coverage'] for x in w]):.4f}, "
              f"Start bei {np.mean([x['cov_start'] for x in w]):.4f}  "
              f"(Netz {np.mean([x['net_s'] for x in w])*1e3:.0f} ms, "
              f"Verfeinerung {np.mean([x['refine_s'] for x in w])*1e3:.0f} ms)")

    ds = [r['glaube-D'] for r in records if 'glaube-D' in r]
    if ds:
        sh = float(np.mean([d['transit_len'] / max(d['path_len'], 1e-9)
                            for d in ds]))
        print(f"\nVariante D: {a.d_rounds} Runden a {a.d_execute_frac:.0%}, "
              f"Anfahrten {sh * 100:.1f} % der Strecke "
              f"(Einstieg: {a.d_join})")

    ts = [r['zweistufig'] for r in records if 'zweistufig' in r]
    if ts:
        share = float(np.mean([t['transit_len'] / max(t['path_len'], 1e-9)
                               for t in ts]))
        print(f"\nVariante E: die ungeplante Verbindungsfahrt ist im Mittel "
              f"{share * 100:.1f} % der Gesamtstrecke")
        print(f"           (Abdeckung nur der zweiten Phase: "
              f"{float(np.mean([t['cov_phase2'] for t in ts])):.4f})")


if __name__ == '__main__':
    main()
