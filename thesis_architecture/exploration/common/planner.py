r"""
planner.py
==========
Der Teil, der aus einer Zielverteilung eine Trajektorie macht — austauschbar.

Zwei Implementierungen hinter derselben Schnittstelle, und die Austauschbarkeit
ist der Punkt:

* `GradientPlanner` optimiert B-Spline-Kontrollpunkte direkt gegen den
  ergodischen Verlust. Braucht keinen Checkpoint, laeuft ueberall, und steht
  stellvertretend fuer "Solver von vorn" — also fuer das, was der gelernte
  Warm-Start ersetzen soll.
* `ModelPlanner` laedt ein trainiertes Netz und generiert amortisiert.

Damit sind alle fuenf Varianten sofort lauffaehig und spaeter ohne Aenderung auf
das Netz umstellbar. Vor allem aber wird der Vergleich, um den es in der Arbeit
geht, zu einem Austausch einer Zeile: dieselbe Mission, derselbe Glaube,
derselbe Messprozess — nur der Planer wechselt. Genau dort zahlt sich
Amortisierung aus, weil im Receding Horizon dutzendfach pro Mission geplant wird.
"""

import os
import sys
import time

import torch

_arch = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', '..'))
if _arch not in sys.path:
    sys.path.insert(0, _arch)


class BasePlanner:
    """Schnittstelle: Partikelwolke rein, Kontrollpunkte raus."""

    def plan(self, particles, n_candidates=1, init=None, history=None,
             start=None):
        raise NotImplementedError

    def render(self, cps):
        """Kontrollpunkte -> Kurve. (K,nxi,2) -> (K,T,2)"""
        return torch.einsum('pi,kid->kpd', self.B, cps.float())


class GradientPlanner(BasePlanner):
    """Direkte Optimierung der Kontrollpunkte gegen den ergodischen Verlust.

    Kein gelerntes Modell, keine Datenbank — ein Gradientenabstieg von einer
    zufaelligen oder uebergebenen Initialisierung. Dass er startwertabhaengig
    ist, ist kein Mangel der Implementierung, sondern genau die Eigenschaft, die
    der Warm-Start adressiert; `init` existiert, damit sich das messen laesst.

    Args:
        metric: 'fourier' oder 'sinkhorn' — dieselbe Wahl wie im Training.
        steps:  Optimierungsschritte pro Planung. Die Wanduhrzeit hiervon ist
                die Groesse, gegen die ein amortisierter Generator antritt.
    """

    def __init__(self, nxi=25, pts=128, deg=5, K=8, metric='fourier',
                 steps=150, lr=0.05, device='cpu', seed=0,
                 boundary_weight=10.0, smooth_weight=0.5,
                 target_length=None, length_weight=20.0):
        from obstacles import bspline_basis_matrix
        from ergodic_metric import ErgodicLoss

        self.nxi, self.device = nxi, torch.device(device)
        self.steps, self.lr, self.seed = steps, lr, seed
        self.boundary_weight = boundary_weight
        self.smooth_weight = smooth_weight
        self.target_length = target_length
        self.length_weight = length_weight
        self.B = torch.from_numpy(bspline_basis_matrix(nxi, pts, deg)).to(device)
        self.erg = ErgodicLoss(nxi=nxi, K=K, pts=pts, deg=deg, weight=1.0,
                               t_power=0.0, metric=metric).to(device)
        self.last_wallclock = 0.0

    def _regularisers(self, cps):
        """Rand-, Beschleunigungs- und optionale Laengenstrafe.

        Die ersten beiden entsprechen den Termen der Solver-Energie: ohne sie
        laeuft die Optimierung aus dem Einheitsquadrat heraus oder erzeugt
        Zickzack, das die Metrik senkt, aber nicht fahrbar ist.

        Die dritte, `target_length`, existiert allein fuer den Vergleich. Ohne
        sie produziert jede Variante die Laenge, die ihr Optimierer zufaellig
        ergibt — im ersten Vergleichslauf zwischen 1.5 und 7. Ein Vergleich bei
        festem Wegbudget ist dann nur in dem schmalen Bereich moeglich, den alle
        erreichen, und "weiter fahren" gewinnt fast immer. Mit einer Zielaenge
        fahren alle Varianten gleich weit, und die Tabelle vergleicht Verfahren
        statt Budgets.
        """
        out = (cps - cps.clamp(0.0, 1.0)).pow(2).sum(dim=(1, 2))
        acc = cps[:, 2:] - 2 * cps[:, 1:-1] + cps[:, :-2]
        reg = (self.boundary_weight * out
               + self.smooth_weight * acc.pow(2).sum(dim=(1, 2)))

        if self.target_length is not None:
            curve = torch.einsum('pi,kid->kpd', self.B, cps)
            L = (curve[:, 1:] - curve[:, :-1]).norm(dim=-1).sum(dim=1)
            reg = reg + self.length_weight * (L - self.target_length).pow(2)
        return reg

    def _coverage(self, cps, parts, hist):
        """Ergodischer Fehler auf Historie + neuem Abschnitt."""
        if hist is None:
            return self.erg.coverage_error(cps, parts)
        curve = torch.einsum('pi,kid->kpd', self.erg.B, cps.float())
        full = torch.cat([hist, curve], dim=1)
        return self._erg_on_curve(full, parts)

    def _erg_on_curve(self, curve, parts):
        """Ergodischer Fehler direkt auf einer Kurve statt auf Kontrollpunkten."""
        from ergodic_metric import trajectory_coeffs, target_coeffs_from_particles
        e = self.erg
        if e.metric == 'sinkhorn':
            from ergodic_metric import sinkhorn_error
            return sinkhorn_error(curve, parts.float(), e._sinkhorn,
                                  e.weighted_target)
        c = trajectory_coeffs(curve, e.k_idx)
        phi = target_coeffs_from_particles(parts.float(), e.k_idx,
                                           e.weighted_target)
        return (e.Lambda * (c - phi) ** 2).sum(dim=-1)

    def plan(self, particles, n_candidates=1, init=None, history=None,
             start=None, length=None, length_cfg_weight=0.0):
        """`history` ist die bereits abgefahrene Bahn.

        `length`/`length_cfg_weight` gehoeren zur laengenkonditionierten
        `CfmPlanner`-Schnittstelle und werden hier ignoriert — der
        Gradientenplaner hat sein eigenes `target_length` (siehe
        `_regularisers`), nicht abhaengig vom Aufrufer.

        Sie wird der Abdeckung vorangestellt, statt ignoriert zu werden. Das ist
        die Buchfuehrung, an der Receding-Horizon-Verfahren scheitern, wenn man
        sie weglaesst: plant man jeden Abschnitt frisch gegen die volle
        Zielverteilung, wird das Verfahren wieder gierig und pendelt zwischen
        gleich attraktiven Gebieten, statt die Abdeckungsschuld abzutragen.

        `start` erzwingt den ersten Kontrollpunkt, damit ein Abschnitt dort
        beginnt, wo der vorige endete.
        """
        t0 = time.perf_counter()
        g = torch.Generator(device='cpu').manual_seed(self.seed)
        if init is None:
            cps = (0.3 * torch.randn(n_candidates, self.nxi, 2, generator=g) + 0.5)
            cps = cps.to(self.device)
        else:
            cps = init.clone().to(self.device)
            if cps.dim() == 2:
                cps = cps.unsqueeze(0)
        cps = cps.clamp(0.02, 0.98).requires_grad_(True)

        K = cps.shape[0]
        parts = particles.unsqueeze(0).expand(K, -1, -1).to(self.device)
        hist = None
        if history is not None and history.numel():
            hist = history.to(self.device)
            if hist.dim() == 2:
                hist = hist.unsqueeze(0).expand(K, -1, -1)
        s0 = None if start is None else start.to(self.device).view(1, 1, 2)

        opt = torch.optim.Adam([cps], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad(set_to_none=True)
            eff = cps if s0 is None else torch.cat([s0.expand(K, 1, 2),
                                                    cps[:, 1:]], dim=1)
            err = self._coverage(eff, parts, hist)
            loss = (err + self._regularisers(eff)).sum()
            loss.backward()
            opt.step()
        if s0 is not None:
            with torch.no_grad():
                cps[:, 0] = s0.view(1, 2)

        self.last_wallclock = time.perf_counter() - t0
        return cps.detach().clamp(0.0, 1.0)


class ModelPlanner(BasePlanner):
    """Amortisierte Planung mit einem trainierten Flow-Matching-Netz.

    Der Eingang des Netzes bleibt unveraendert: es konditioniert ohnehin auf
    Partikelwolken, und ob die aus einer bekannten Dichte oder aus einem
    UCB-Glauben gezogen wurden, sieht es nicht. Das ist der Grund, warum diese
    ganze Fragestellung ohne Architekturaenderung angehbar ist.
    """

    def __init__(self, checkpoint, nxi=25, pts=128, deg=5, steps=100,
                 cfg_weight=2.0, device='cpu'):
        from obstacles import bspline_basis_matrix
        self.device = torch.device(device)
        self.nxi, self.steps, self.cfg_weight = nxi, steps, cfg_weight
        self.B = torch.from_numpy(bspline_basis_matrix(nxi, pts, deg)).to(device)
        self.last_wallclock = 0.0

        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        # Startpunkt-konditionierte Checkpoints tragen `start_cond` und brauchen
        # die andere Architektur. Ohne den Schluessel bleibt alles wie bisher.
        self.start_cond = bool(ck.get('start_cond', False))
        if self.start_cond:
            from flow_matching_cond_particles_start import (
                ParticleCrossAttnFlowNetwork, generate_particle_trajectories)
        else:
            from flow_matching_cond_particles_crossattn import (
                ParticleCrossAttnFlowNetwork, generate_particle_trajectories)
        self._gen = generate_particle_trajectories
        # `n_particles` gehoert zum Datenlader, nicht zum Netz — der
        # Partikel-Tokenizer arbeitet ueber die Sequenzachse.
        self.model = ParticleCrossAttnFlowNetwork(
            nxi=ck.get('nxi', nxi), D=ck.get('D', 384)).to(device)
        self.model.load_state_dict(ck['model_state_dict'])
        self.model.eval()

    def plan(self, particles, n_candidates=1, init=None, start=None,
             length=None, length_cfg_weight=0.0):
        """`start` wirkt nur bei einem startpunkt-konditionierten Checkpoint.

        Dort geht der Punkt als FiLM-Konditionierung ins Netz und der erste
        Kontrollpunkt wird anschliessend hart darauf gesetzt. Bei einem alten
        Checkpoint wird er stillschweigend ignoriert — der Aufrufer muss
        `start_cond` pruefen, wenn er sich darauf verlassen will.

        `length`/`length_cfg_weight` sind hier ebenso stillschweigend
        ignoriert: diese Klasse laedt immer die Basis- oder
        Startpunkt-Architektur, nie die laengenkonditionierte — dafuer ist
        `CfmPlanner` in `apply_cfm_belief.py` zustaendig.
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
