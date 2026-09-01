r"""
interactive_sim.py
===================
Interaktive Play/Pause-Simulation: Exploration/Exploitation live ansehen.

Zuerst sieht man, wie das Vorwissen ueber die Zielverteilung verteilt ist (ein
paar zufaellige Vormessungen, der Rest unbekannt). Mit Play faehrt ein Agent
(Kreis, Durchmesser = Sensorradius, also genau der Bereich, den eine Messung
abdeckt) die vom gewaehlten Solver geplante Bahn ab. Im linken Panel deckt das
Abfahren die tatsaechliche Zieldichte in den besuchten Gebieten auf
(Fog-of-War); im rechten Panel steht fortlaufend die aktuelle Zieldichte
Phi = f(GP-Glaube), aus der der Solver seine naechste Bahn plant.

Nutzt ausschliesslich das trainierte, startpunkt-konditionierte Netz
(`netz2d_startpunkt.pt`) ueber `apply_cfm_belief.CfmPlanner` -- dieselbe
Klasse und denselben Zieldichte-/Schulden-Code wie die Batch-Missionen
'glaube-1' (Solver A), 'glaube-R' (Solver C) und 'glaube-D' (Solver D) in
`apply_cfm_belief.py`. Was hier neu ist, ist nur die Aufteilung dieser
Missionen in einzelne Runden, die sich anhalten und animiert abfahren lassen.

    python interactive_sim.py
    python interactive_sim.py --ckpt ../../transfer/netz2d_startpunkt.pt
"""

import argparse
import contextlib
import os
import sys
import threading
import time
import traceback

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault('MPLBACKEND', 'TkAgg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.colors as mcolors      # noqa: E402
import matplotlib.patches as mpatches    # noqa: E402
from matplotlib.widgets import RadioButtons, Slider, Button  # noqa: E402

from mujoco_sim.board import MAX_TIP_SPEED, max_ui_speed  # noqa: E402

from common.data import load_truth               # noqa: E402
from common.belief import GPBelief                # noqa: E402
from common.observation import measure, thin       # noqa: E402
from common.acquisition import kappa_schedule       # noqa: E402
from common.metrics import (coverage_vs_truth, information_gain,  # noqa: E402
                            path_length)

import apply_cfm_belief as acb                      # noqa: E402
import ergodic_core as ergo                         # noqa: E402
import svgd_engine as svgde                         # noqa: E402
from obstacles import CircleObstacle, CompositeObstacle, polish_out_of_obstacle, bspline_basis_matrix # noqa: E402


DEFAULT_CKPT = os.path.join(_root, 'transfer', 'netz2d_startpunkt.pt')

# Wieviele Runden Solver C/D in dieser interaktiven Ansicht fahren. Die
# Batch-Skripte (`apply_cfm_belief.py`) nutzen 3 bzw. 20 -- hier kleiner
# gehalten, damit jede Replanung nur eine kurze, sichtbare Pause verursacht.
C_ROUNDS = 4
D_ROUNDS = 8
D_EXECUTE_FRAC = 0.15

# ucb/mass/eid direkt aus common/acquisition.py; 'niveau' ist die
# Niveaumengen-Schaetzung, in acquisition.py als `phi_lse` gefuehrt.
PHI_UI = ['ucb', 'mass', 'eid', 'niveau']
PHI_INTERNAL = {'ucb': 'ucb', 'mass': 'mass', 'eid': 'eid', 'niveau': 'lse'}

SOLVERS = ['A (Open Loop)', 'C (Receding Horizon)', 'D (Ergodic Debt)']
SOLVER_DESC = {
    'A': 'A — plan once, execute once',
    'C': 'C — receding horizon (belief grows)',
    'D': 'D — plan long, execute short, replan',
}

INIT_MODES = ['CFM', 'Linear', 'Heuristic', 'Manual']
INIT_DESC = {
    'CFM': 'Trained CFM-Net (amortized)',
    'Linear': 'Diagonal + Gradient Optimization',
    'Heuristic': 'TSP+Serpentine + Gradient Opt.',
    'Manual': 'Draw manually + Gradient Opt.',
}

# Erklaerungstexte fuer die Info-Knoepfe neben den Live-Metriken.
INFO_TEXT = {
    'length': (
        "Driven Length\n\n"
        "Arc length of the path actually driven so far, in domain units "
        "[0,1]x[0,1] (common/metrics.path_length)."),
    'erg_phi': (
        "Ergodicity vs. Phi\n\n"
        "Fourier ergodicity error (K=8, same metric as in training) "
        "between the spatial distribution of the path driven so far and "
        "the CURRENT target density Phi that the solver is aiming for. 0 "
        "means: the path perfectly matches Phi. Changes with every "
        "replanning because Phi itself changes."),
    'explore': (
        "Exploration (Resolved Uncertainty)\n\n"
        "Sum of GP standard deviation over the grid at mission start "
        "minus now (common/metrics.information_gain) — how much of the "
        "unknown space has already been clarified by measurements. There is "
        "no established 'ergodicity of exploration'; this is the "
        "proxy used in this project. Higher is better."),
    'erg_truth': (
        "Ergodicity vs. Truth\n\n"
        "Same Fourier ergodicity error as on the left, but computed against the "
        "actual ground truth target density, which the solver never gets to see. "
        "The actual goal of the mission — Phi is merely a means "
        "to this end."),
    'coverage': (
        "Total Coverage Error\n\n"
        "Density-weighted mean distance of each grid cell of the true "
        "target density to the nearest path point "
        "(common/metrics.coverage_vs_truth). Lower is better; unlike "
        "the ergodicity metrics, this ignores the order/dwell time "
        "and only asks whether every location was visited eventually."),
    'solvers': (
        "Solvers\n\n"
        "A (Open Loop): Plans one long trajectory based on initial knowledge and executes it fully without replanning.\n\n"
        "C (Receding Horizon): Plans a path, executes a small segment, updates the GP belief with new measurements, and replans.\n\n"
        "D (Ergodic Debt): Like C, but plans against a 'debt' density that subtracts areas already visited, forcing the agent to explore new regions."
    ),
}


# ===========================================================================
# Stil (Projektkonvention, siehe CLAUDE.md)
# ===========================================================================

def white_inferno():
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mcolors.LinearSegmentedColormap.from_list('white_inferno', inf)


def style(ax):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')


# ===========================================================================
# Missions-Logik: dieselben Bausteine wie apply_cfm_belief.py, aber als
# unterbrechbare Runden-Generatoren statt einer einzigen Endlosschleife.
# ===========================================================================

class Mission:
    """Eine Solver-Instanz (A/C/D) auf einer Form, Runde fuer Runde.

    Jede Runde plant `self.planner` (das trainierte Netz) eine Bahn gegen die
    aktuelle Zieldichte Phi, die aus `self.belief` (GP-Posterior) abgeleitet
    wird -- identisch zu `zieldichte`/`debt_density` in apply_cfm_belief.py.
    Zwischen den Runden wird `self.belief` mit den Messungen entlang der
    gefahrenen Strecke fortgeschrieben. Die Trennung in Runden-Generatoren
    (statt einer Schleife bis zum Ende) ist der einzige Unterschied zu den
    Batch-Missionen dort -- so kann die aufrufende Animation nach jeder Runde
    anhalten und das Abfahren selbst animieren.
    """

    def __init__(self, planner, truth, belief, args, variant, svgd, nxi_ui=25):
        self.planner = planner
        self.truth = truth
        self.belief = belief
        self.args = args
        self.variant = variant
        self.svgd = svgd
        self.nxi_ui = nxi_ui
        self.driven = []
        self.current_step = 'inference'
        
    def _project_bspline(self, curve):
        if self.nxi_ui == 25:
            return curve
        from obstacles import bspline_basis_matrix
        T = curve.shape[0]
        B = torch.from_numpy(bspline_basis_matrix(self.nxi_ui, T, 5)).float().to(curve.device)
        result = torch.linalg.lstsq(B, curve)
        P = result.solution
        curve_proj = B @ P
        return curve_proj

    def path_so_far(self):
        return torch.cat(self.driven, dim=0) if self.driven else None

    def _observe(self, seg):
        pts, vals = measure(seg.detach(), self.truth, noise_std=self.args.noise,
                            sensor_radius=self.args.sensor_radius)
        self.belief.observe(*thin(pts, vals, max_points=self.args.max_obs))

    def _refine(self, curve, phi):
        """SVGD-Nachverfeinerung des Netz-Vorschlags gegen dieselbe Runden-
        Zieldichte Phi -- ein reiner Durchreicher, solange `svgd_iters<=0`."""
        self.current_step = 'svgd'
        if self.args.svgd_iters <= 0:
            return self._project_bspline(curve)
        device, dtype = curve.device, curve.dtype
        refined = self.svgd.refine(curve.detach().cpu().numpy(),
                                   phi.detach().cpu().numpy(),
                                   self.args.svgd_iters,
                                   nxi=self.nxi_ui,
                                   obstacle=self.args.obstacle,
                                   obstacle_weight=50000.0)
        return torch.from_numpy(refined).to(device=device, dtype=dtype)

    def rounds(self):
        if self.variant == 'A':
            return self._rounds_A()
        if self.variant == 'C':
            return self._rounds_C()
        return self._rounds_D()

    def _get_init(self, phi, pos=None):
        """Initialisierung fuer den GradientPlanner, falls vorhanden.

        - CfmPlanner hat kein `_default_init` -> None -> kein init-Argument.
        - Linear-Modus: `_default_init` ist gesetzt -> Diagonale vom aktuellen Agenten.
        - Heuristik-Modus: `_default_init is None` -> aus Phi generieren, TSP
          startend vom aktuellen Agenten."""
        if not hasattr(self.planner, '_default_init'):
            return {}  # CfmPlanner — kein init noetig
        
        sp_np = np.array([0.5, 0.5])
        if pos is not None:
            sp_np = pos.detach().cpu().numpy()

        if self.planner._default_init is not None:
            if self.planner._default_init == 'Manual':
                drawn = getattr(self, 'app', None).manual_init_drawn
                if getattr(self, 'app', None):
                    self.app.manual_init_drawn = None
                return {'init': drawn}
            else:
                # Linear-Modus: baue eine frische Diagonale vom Startpunkt aus
                t = np.linspace(0, 1, 25)[:, None]
                # Ziele auf das gegenueberliegende Quadrat-Zentrum (0.9, 0.9) oder (0.1, 0.1)
                target = np.array([0.9, 0.9]) if sp_np.sum() < 1.0 else np.array([0.1, 0.1])
                cps = sp_np + t * (target - sp_np)
                cps = np.clip(cps, 0.02, 0.98)
                init_t = torch.from_numpy(cps).float().unsqueeze(0).to(self.planner.device)
                return {'init': init_t}
        
        # Heuristik: aus Phi-Gitter eine Serpentinen-Bahn bauen
        return {'init': self._heuristic_init_from_phi(phi, sp_np)}

    def _heuristic_init_from_phi(self, phi, start_pos_np):
        """Baut aus dem Phi-Gitter eine TSP+Serpentinen-Initialisierung.

        1. Finde die k staerksten Peaks (lokale Maxima oder Top-Gewichtszellen)
        2. Ordne sie per Greedy-TSP (naechster Nachbar)
        3. Verbinde sie mit Serpentinen-Kurven entlang der Hauptachse
        4. Interpoliere auf nxi=25 Kontrollpunkte
        """
        phi_np = phi.detach().cpu().numpy().copy()
        R = phi_np.shape[-1]
        
        if self.args.obstacle is not None:
            gx = np.linspace(0, 1, R)
            gy = np.linspace(0, 1, R)
            XX, YY = np.meshgrid(gx, gy)
            obs_mask = self.args.obstacle.mask(XX, YY, inflated=True)
            phi_np[obs_mask] = 0.0

        # Top-k Zellen als Cluster-Zentren
        flat = phi_np.ravel()
        n_peaks = min(8, max(3, int((flat > 0.3 * flat.max()).sum() / (R * 0.5))))
        top_idx = np.argsort(flat)[-n_peaks * R:]  # ueberdimensioniert, wird geclustered
        top_y, top_x = np.unravel_index(top_idx, phi_np.shape)
        top_pts = np.stack([top_x / (R - 1), top_y / (R - 1)], axis=-1)
        top_w = flat[top_idx]

        # K-Means-artig: gewichtetes Clustering auf wenige Zentren
        from scipy.cluster.vq import kmeans2
        n_centers = min(n_peaks, len(top_pts))
        if n_centers < 2:
            # Zu wenig Struktur -> einfache Diagonale
            t = np.linspace(0, 1, 25)[:, None]
            cps = 0.05 + 0.9 * np.tile(t, (1, 2))
            return torch.from_numpy(cps).float().unsqueeze(0).to(
                self.planner.device)
        centers, _ = kmeans2(top_pts, n_centers, minit='points')

        # Greedy-TSP: vom Agenten aus startend
        curr = start_pos_np
        remaining = list(range(len(centers)))
        order = []
        while remaining:
            dists = [np.linalg.norm(curr - centers[i]) for i in remaining]
            best = remaining.pop(int(np.argmin(dists)))
            order.append(best)
            curr = centers[best]
        ordered = centers[order]

        # Serpentinen zwischen den Zentren
        all_pts = [np.array([start_pos_np])]
        for i, c in enumerate(ordered):
            prev = all_pts[-1][-1]
            # Anfahrt
            n_transit = max(3, int(np.linalg.norm(c - prev) * 40))
            transit = np.linspace(prev, c, n_transit)
            # Lokale Serpentine um das Zentrum
            spread = 0.08
            n_swing = 12
            tau = np.linspace(-1, 1, n_swing)
            dx = np.array([spread, 0])
            dy = np.array([0, spread * 0.4])
            serpentine = c[None, :] + np.outer(tau, dx) + \
                np.outer(np.sin(2 * np.pi * tau), dy)
            serpentine = np.clip(serpentine, 0.02, 0.98)
            all_pts.extend([transit, serpentine])

        combined = np.vstack(all_pts)
        # Auf 25 Kontrollpunkte interpolieren
        idx = np.linspace(0, len(combined) - 1, 25).astype(int)
        cps = combined[idx]
        cps = np.clip(cps, 0.02, 0.98)
        
        cps_t = torch.from_numpy(cps).float().unsqueeze(0).to(self.planner.device)
        
        if self.args.obstacle is not None:
            B_eval = torch.from_numpy(bspline_basis_matrix(25, 256, 5)).float().to(self.planner.device)
            cps_t = polish_out_of_obstacle(cps_t, self.args.obstacle, B_eval)
            
        return cps_t

    def _rounds_A(self):
        with torch.no_grad():
            mu, sd = self.belief.posterior_grid()
            if getattr(self.args, 'allknowing_mode', False):
                phi = self.truth
            else:
                phi = acb.zieldichte(mu, sd, self.args.kappa0, self.args)
            parts = acb.phi_particles(phi, self.args.n_particles,
                                      mode=self.args.phi_mode,
                                      device=self.args.device)
            here = self.path_so_far()
            pos = here[-1] if here is not None else None
            
            init_kw = self._get_init(phi, pos=pos)
            if pos is not None:
                init_kw['start'] = pos
                
            if self.planner.__class__.__name__ == 'CfmPlanner':
                init_kw['obstacle'] = self.args.obstacle
                init_kw['obstacle_weight'] = 5000.0
                
            cps = self.planner.plan(parts, n_candidates=self.args.n_candidates, **init_kw)
            curve = acb.best_candidate(self.planner.render(cps), phi)
            curve = self._refine(curve, phi)
        yield dict(round=0, n_rounds=1, phi=phi, mu=mu, sd=sd, seg=curve,
                   kappa=self.args.kappa0)
        with torch.no_grad():
            self._observe(curve)
            self.driven.append(curve.detach())

    def _rounds_C(self, n_rounds=C_ROUNDS):
        for r in range(n_rounds):
            with torch.no_grad():
                kap = kappa_schedule(r, n_rounds, self.args.kappa0, self.args.kappa1)
                mu, sd = self.belief.posterior_grid()
                if getattr(self.args, 'allknowing_mode', False):
                    phi = self.truth
                else:
                    phi = acb.zieldichte(mu, sd, kap, self.args)
                parts = acb.phi_particles(phi, self.args.n_particles,
                                          mode=self.args.phi_mode,
                                          device=self.args.device)
                
                here = self.path_so_far()
                pos = here[-1] if here is not None else None
                
                init_kw = self._get_init(phi, pos=pos)
                if pos is not None:
                    init_kw['start'] = pos
                    
                if self.planner.__class__.__name__ == 'CfmPlanner':
                    init_kw['obstacle'] = self.args.obstacle
                    init_kw['obstacle_weight'] = 5000.0
                    
                cps = self.planner.plan(parts, n_candidates=self.args.n_candidates, **init_kw)
                curve = acb.best_candidate(self.planner.render(cps), phi)
                curve = self._refine(curve, phi)
                if self.driven:
                    prev_end = self.driven[-1][-1].to(curve.device)
                    al = torch.linspace(0, 1, self.args.transit_pts,
                                        device=curve.device).unsqueeze(-1)
                    link = prev_end.unsqueeze(0) * (1 - al) + curve[0].unsqueeze(0) * al
                    curve = torch.cat([link, curve], dim=0)
            yield dict(round=r, n_rounds=n_rounds, phi=phi, mu=mu, sd=sd,
                      seg=curve, kappa=kap)
            with torch.no_grad():
                self._observe(curve)
                self.driven.append(curve.detach())

    def _rounds_D(self, n_rounds=D_ROUNDS):
        for r in range(n_rounds):
            with torch.no_grad():
                kap = kappa_schedule(r, n_rounds, self.args.kappa0, self.args.kappa1)
                here = self.path_so_far()
                visit = (acb.visitation_field(here, self.belief.res,
                                              self.args.visit_bandwidth,
                                              self.args.device)
                         if here is not None else None)
                mu, sd = self.belief.posterior_grid()
                if getattr(self.args, 'allknowing_mode', False):
                    phi = self.truth
                    v = visit if visit is not None else torch.zeros_like(phi)
                else:
                    phi, v = acb.debt_density(mu, sd, visit, kap, self.args)
                parts = acb.phi_particles(phi, self.args.n_particles,
                                          mode=self.args.phi_mode,
                                          device=self.args.device)
                pos = here[-1] if here is not None else None
                init_kw = self._get_init(phi, pos=pos)
                if pos is not None:
                    init_kw['start'] = pos
                    
                if self.planner.__class__.__name__ == 'CfmPlanner':
                    init_kw['obstacle'] = self.args.obstacle
                    init_kw['obstacle_weight'] = 5000.0
                    
                cps = self.planner.plan(parts, n_candidates=self.args.n_candidates, **init_kw)
                curve = acb.best_candidate(self.planner.render(cps), phi)
                curve = self._refine(curve, phi)
                T = curve.shape[0]
                k = max(2, int(round(self.args.d_execute_frac * T)))
                seg = curve[:k]
            yield dict(round=r, n_rounds=n_rounds, phi=phi, mu=mu, sd=sd,
                      seg=seg, kappa=kap, visit=v)
            with torch.no_grad():
                self._observe(seg)
                self.driven.append(seg.detach())


def build_args(device, phi_model='ucb', kappa0=3.0, svgd_iters=0, phi_tau=0.25, sensor_radius=0.06):
    """Ein leichtgewichtiges Namensobjekt mit denselben Feldern wie das
    argparse-Namespace aus apply_cfm_belief.py -- `zieldichte`/`debt_density`
    lesen nur diese Attribute, kein CLI-Parsing noetig."""
    a = argparse.Namespace()
    a.svgd_iters = svgd_iters
    a.phi_model = phi_model
    a.phi_tau = phi_tau
    a.phi_xi = 0.01
    a.phi_gamma = 1.0
    a.gp_noise = 0.05
    a.visit_sat = 0.25
    a.debt_weight = 1.0
    a.device = device
    a.phi_mode = 'uniform'
    a.phi_quantile = 0.5
    a.target_length = None
    a.target_length_cfg = 0.0
    a.n_candidates = 1
    a.n_particles = 256
    a.noise = 0.02
    a.sensor_radius = sensor_radius
    a.max_obs = 96
    a.transit_pts = 16
    a.d_execute_frac = D_EXECUTE_FRAC
    a.visit_bandwidth = a.sensor_radius
    a.kappa0 = kappa0
    a.kappa1 = max(0.05, kappa0 * 0.15)
    a.obstacle = None
    return a


class SvgdRefiner:
    """Verfeinert eine geplante Bahn mit dem bestehenden SVGD-Solver
    (SE3_SVGD/svgd_engine.py + ergodic_core.py) — Ziel ist die aktuelle
    Zieldichte Phi der jeweiligen Runde, nicht ein fest verdrahtetes Ziel wie
    in `SE3_SVGD/tsvec_2d.py`.

    Ablauf pro Aufruf: `N_PARTICLES` Kopien der vom Netz gelieferten Bahn,
    leicht verrauscht, als Partikelschwarm; SVGD zieht sie ueber `n_iters`
    Adam-Schritte auf die Fourier-Koeffizienten von Phi; zurueckgegeben wird
    der Partikel mit der niedrigsten Endenergie. `n_iters=0` ist ein reiner
    Durchreicher (kein SVGD) -- das ist die 0-Stellung des Reglers in der GUI.

    Energie- und Adam-Gewichte identisch zu `SE3_SVGD/tsvec_2d.py`
    (W_ERGODIC=600, W_SMOOTH=15, W_BOUNDARY=30); K=8 statt dessen K=10, damit
    dieselbe Fourier-Basis wie die "Ergodizitaet vs. Phi"-Live-Metrik benutzt
    wird und deren Anzeige die SVGD-Verfeinerung direkt sichtbar macht.
    """

    K = 8
    DIM = 2
    N_PARTICLES = 8
    JITTER = 0.01
    W_ERGODIC, W_SMOOTH, W_BOUNDARY = 600.0, 15.0, 30.0

    def __init__(self, seed=0):
        self.k_idx = ergo.build_fourier_indices(self.K, self.DIM)
        self.Lambda_k = ergo.compute_lambda_k(self.k_idx)
        self.rng = np.random.default_rng(seed)

    def _phi_k(self, phi_grid):
        R = phi_grid.shape[-1]
        xs = np.linspace(0, 1, R)
        Xg, Yg = np.meshgrid(xs, xs)
        grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
        grid_w = np.clip(phi_grid.ravel(), 0.0, None)
        return ergo.compute_target_fourier_coeffs(grid_pts, grid_w, self.k_idx)

    def _energy_and_grad(self, phi_k, obstacle=None, obstacle_weight=20.0):
        k_idx, Lambda_k = self.k_idx, self.Lambda_k
        W_E, W_S, W_B = self.W_ERGODIC, self.W_SMOOTH, self.W_BOUNDARY

        def fn(X_flat, T):
            X = X_flat.reshape(T, 2)
            F = ergo.fourier_basis_nd(X, k_idx)
            c_k = F.mean(axis=0)
            dF = ergo.fourier_basis_grad_nd(X, k_idx)
            diff = c_k - phi_k
            erg_cost = 0.5 * np.sum(Lambda_k * diff ** 2)
            erg_grad = np.einsum('m,tmd->td', Lambda_k * diff, dF) / T

            sm_cost = svgde.compute_smoothness_cost_numpy(X)
            sm_grad = svgde.compute_smoothness_grad_numpy(X)
            bd_cost = svgde.compute_boundary_cost_numpy(X)
            bd_grad = svgde.compute_boundary_grad_numpy(X)

            energy = W_E * erg_cost + W_S * sm_cost + W_B * bd_cost
            grad = W_E * erg_grad + W_S * sm_grad + W_B * bd_grad
            
            if obstacle is not None:
                energy += obstacle_weight * obstacle.penalty(X)
                grad += obstacle_weight * obstacle.grad_penalty(X)

            return energy, grad.ravel()
        return fn

    def refine(self, curve_np, phi_np, n_iters, nxi=None, obstacle=None, obstacle_weight=20.0):
        """curve_np: (T,2) Startbahn (vom Netz). phi_np: (R,R) Zieldichte,
        Werte in [0,1]. -> (T,2) verfeinerte Bahn."""
        if n_iters <= 0:
            return curve_np
        T = curve_np.shape[0]
        phi_k = self._phi_k(phi_np)
        energy_fn_orig = self._energy_and_grad(phi_k, obstacle=obstacle, obstacle_weight=obstacle_weight)

        if nxi is not None and nxi != T:
            from obstacles import bspline_basis_matrix
            B = bspline_basis_matrix(nxi, T, 5)
            P, _, _, _ = np.linalg.lstsq(B, curve_np, rcond=None)
            
            def energy_fn(P_flat, N_pts):
                P_mat = P_flat.reshape(nxi, 2)
                X = B @ P_mat
                energy, grad_X = energy_fn_orig(X.ravel(), T)
                grad_X_mat = grad_X.reshape(T, 2)
                grad_P_mat = B.T @ grad_X_mat
                return energy, grad_P_mat.ravel()

            jitter = self.rng.normal(scale=self.JITTER, size=(self.N_PARTICLES, nxi, 2))
            init = P[None] + jitter
            particles = init.reshape(self.N_PARTICLES, nxi * 2)
            
            with open(os.devnull, 'w') as devnull, \
                    contextlib.redirect_stderr(devnull):
                particles, _ = svgde.run_svgd_numpy(
                    particles, nxi, int(n_iters), energy_fn, dim=self.DIM,
                    label='SVGD-Verfeinerung')
                    
            particles = particles.reshape(self.N_PARTICLES, nxi, 2)
            scores = [energy_fn(p.ravel(), nxi)[0] for p in particles]
            best_P = particles[int(np.argmin(scores))]
            return B @ best_P
        else:
            jitter = self.rng.normal(scale=self.JITTER, size=(self.N_PARTICLES, T, 2))
            init = np.clip(curve_np[None] + jitter, 0.02, 0.98)
            particles = init.reshape(self.N_PARTICLES, T * 2)

            with open(os.devnull, 'w') as devnull, \
                    contextlib.redirect_stderr(devnull):
                particles, _ = svgde.run_svgd_numpy(
                    particles, T, int(n_iters), energy_fn_orig, dim=self.DIM,
                    label='SVGD-Verfeinerung')

            particles = particles.reshape(self.N_PARTICLES, T, 2)
            scores = [energy_fn_orig(p.ravel(), T)[0] for p in particles]
            return particles[int(np.argmin(scores))]


# ===========================================================================
# Interaktive Ansicht
# ===========================================================================

class App:
    N_PRIOR = 12
    GP_RES = 64
    TRUTH_RES = 96
    FRAME_MS = 55

    #: Agentengeschwindigkeit beim Start, in cm/s auf dem Brett. Bei 20 cm/s
    #: folgt der MuJoCo-Arm mit ~1.5 mm mittlerem Fehler; das Slider-Maximum
    #: MAX_TIP_SPEED ist die Grenze, ab der er sichtbar zurueckfaellt.
    DEFAULT_SPEED_CM_S = 20
    PLAN_TIME_EST_INIT = 1.5  # Sekunden; wird nach der ersten Runde kalibriert

    def __init__(self, ckpt, shapes, device, seed, shared_truth=None, agent_info_array=None, shared_obstacles=None):
        self.shared_truth = shared_truth
        self.agent_info_array = agent_info_array
        self.shared_obstacles = shared_obstacles
        self.eraser_mode = False
        self.allknowing_mode = False
        self.device = device
        self.seed = seed
        print(f"Loading checkpoint: {ckpt}")
        self.planner = acb.CfmPlanner(ckpt=ckpt, device=device)
        self.ergodic = acb.ErgodicScore(K=8, device=device)
        self.names, self.truths = load_truth(n=shapes, resolution=self.TRUTH_RES,
                                             device=device)
        print(f"Shapes: {', '.join(self.names)}")

        self.shape_i = 0
        self.solver = 'A'
        self.phi_ui = 'ucb'
        self.phi_view = 'Φ'   # Toggle: 'Φ', 'μ' oder 'σ'
        self.init_mode = 'CFM'  # 'CFM', 'Linear', 'Heuristic', 'Manual'
        self.kappa = 3.0     # fuer ucb/eid/stretch/mi
        
        # State for manual drawing initialization
        self.manual_init_drawn = None
        self.waiting_for_manual_init = False
        self.is_drawing_manual = False
        self.manual_path = []
        self.manual_line = None
        self.mass_w = 0.5    # fuer mass -- direkter Anteil, nicht ueber kappa
        self.niveau_tau = 0.25 # fuer niveau -- Schwellwert tau
        self.svgd_iters = 0  # 0 = kein SVGD, Regler in der GUI geht bis 1000
        self.svgd = SvgdRefiner(seed=seed)

        self.gx = np.linspace(0, 1, self.TRUTH_RES)
        self.gy = np.linspace(0, 1, self.TRUTH_RES)
        self.XX, self.YY = np.meshgrid(self.gx, self.gy)

        # Planung (Netz-Vorwaertspass) laeuft in einem Hintergrund-Thread,
        # damit die GUI waehrend dessen nicht einfriert; der Fortschrittsbalken
        # zeigt die verstrichene Zeit gegen eine laufend kalibrierte Schaetzung.
        self.busy = False
        self.est_plan_time = self.PLAN_TIME_EST_INIT
        self.est_svgd_time = 0.5
        self._plan_thread = None
        self._plan_result = None
        self._busy_t0 = 0.0
        self.belief0 = None    # Glaube bei Missionsbeginn, fuer information_gain
        self._last_phi = None  # letztes Phi, solange waehrend Planung kein neues da ist
        self._last_mu = None   # letzter GP-Mittelwert
        self._last_sd = None   # letzte GP-Unsicherheit
        
        self.custom_truth_np = None
        self.custom_prior_mask = None
        
        self.nxi_ui = 25       # Default number of B-Spline control points
        self.sensor_radius_ui = 0.06  # Default agent radius
        self.target_length_ui = 0.0   # Default Target Length
        
        self.obstacles = []
        self.obs_patches = []
        self._dragging_obs = False

        if self.shared_truth is not None:
            with self.shared_truth.get_lock():
                np_truth = np.frombuffer(self.shared_truth.get_obj(), dtype=np.float64).reshape((self.TRUTH_RES, self.TRUTH_RES))
                np_truth[:] = self.truths[self.shape_i].detach().cpu().numpy()[:]

        self._build_figure()
        self.reset(replan=True)

        self.timer = self.fig.canvas.new_timer(interval=self.FRAME_MS)
        self.timer.add_callback(self._tick)
        self.timer.start()

        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)

        plt.show()

    # ── Aufbau ───────────────────────────────────────────────────────────
    def _build_figure(self):
        self._info_open = None
        self.fig = plt.figure(figsize=(17.0, 8.2), facecolor='white')

        # ── Obere Zeile: zwei Panels + Colorbar + Metriken ────────────────
        # Panels etwas tiefer ansetzen, damit Status/Round-Text oben Platz hat
        panel_bottom = 0.34
        panel_h = 0.54
        panel_w = 0.33

        self.ax_world = self.fig.add_axes([0.07, panel_bottom, panel_w, panel_h])
        self.ax_phi = self.fig.add_axes([0.42, panel_bottom, panel_w, panel_h])
        style(self.ax_world); style(self.ax_phi)
        self.cmap = white_inferno()

        self.ax_world.set_title('World — Revealed Target Density', fontsize=11,
                                color='#1A1A2E')
        self.ax_phi.set_title('', fontsize=11, color='#1A1A2E')

        # ── Toggle Φ / μ / σ — oben rechts ueber dem rechten Panel ────────
        ax_phi_view = self.fig.add_axes([0.42 + panel_w - 0.10, panel_bottom + panel_h,
                                         0.10, 0.06])
        ax_phi_view.set_facecolor('white')
        for sp in ax_phi_view.spines.values():
            sp.set_visible(False)
        self.r_phi_view = RadioButtons(ax_phi_view, ['Φ', 'μ', 'σ'],
                                       active=0)
        self.r_phi_view.on_clicked(self._on_phi_view)
        for lbl in self.r_phi_view.labels:
            lbl.set_fontsize(9)

        im0 = np.zeros((self.TRUTH_RES, self.TRUTH_RES, 4))
        self.img_world = self.ax_world.imshow(im0, origin='lower',
                                              extent=[0, 1, 0, 1], zorder=1)
        self.line_driven, = self.ax_world.plot([], [], color='#00C853', lw=2.2,
                                               alpha=0.95, zorder=3)
        self.line_preview, = self.ax_world.plot([], [], color='#00C853', lw=1.6,
                                                alpha=0.3, ls='--', zorder=3)
        self.scat_prior = self.ax_world.scatter([], [], s=6, alpha=0.3,
                                                color='#444444', zorder=2)
        self.agent_disk = mpatches.Circle((0.5, 0.5), self.sensor_radius_ui, facecolor='#1565C0',
                                          edgecolor='#1565C0', alpha=0.22, lw=1.2,
                                          zorder=4)
        self.ax_world.add_patch(self.agent_disk)
        self.agent_dot, = self.ax_world.plot([0.5], [0.5], 'o', color='#1565C0',
                                             ms=4.5, zorder=5)

        # Drag-and-Drop Obstacle Template
        self.ax_obs_template = self.fig.add_axes([0.015, 0.45, 0.04, 0.1])
        self.ax_obs_template.set_axis_off()
        self.ax_obs_template.set_xlim(0, 1)
        self.ax_obs_template.set_ylim(0, 1)
        self.ax_obs_template.set_aspect('equal')
        
        template_radius = 0.4
        self.ax_obs_template.add_patch(mpatches.Circle((0.5, 0.5), template_radius,
                                facecolor='#9E9E9E', alpha=0.75,
                                edgecolor='#424242', lw=1.5, zorder=6))
        self.ax_obs_template.text(0.5, -0.15, 'Drag Obstacle', ha='center', 
                                  va='top', fontsize=9, color='#424242', transform=self.ax_obs_template.transAxes)

        # Ghost patches for dragging (initially hidden)
        default_obs = CircleObstacle(margin=0.0) # just to get default radius/margin
        self.obs_drag_patch = mpatches.Circle((0.5, 0.5), default_obs.radius,
                                facecolor='#9E9E9E', alpha=0.5,
                                edgecolor='#424242', lw=1.5, zorder=7, visible=False)
        self.obs_drag_margin = mpatches.Circle((0.5, 0.5), default_obs.effective_radius,
                                    facecolor='none', edgecolor='#424242',
                                    lw=0.8, ls=':', alpha=0.4, zorder=7, visible=False)
        self.ax_world.add_patch(self.obs_drag_patch)
        self.ax_world.add_patch(self.obs_drag_margin)

        # Kleines Fenster: die volle Grundwahrheit, nie verdeckt — zum
        # Abgleich, wie viel das Fog-of-War-Panel links schon aufgedeckt hat.
        # Bewusst eine eigenstaendige Achse in Figur-Koordinaten statt eines
        # `ax_world.inset_axes(...)`-Kindes: zwei verschachtelte
        # aspect='equal'-Achsen, staendig per Timer neu gezeichnet und dabei
        # der Fenstergroesse des Nutzers ausgesetzt, sind ein bekannter Ausloeser
        # fuer RecursionError in matplotlibs Aspect-Layout — eine unabhaengige
        # Achse hat keine Eltern-Kind-Positionskopplung, die das ausloesen kann.
        truth_x = 0.07 + panel_w - 0.115 - 0.005
        truth_y = panel_bottom + panel_h - 0.17 - 0.005
        self.ax_truth = self.fig.add_axes([truth_x, truth_y, 0.105, 0.17])
        style(self.ax_truth)
        self.ax_truth.grid(False)
        for s in self.ax_truth.spines.values():
            s.set_color('#1A1A2E'); s.set_linewidth(1.0)
        self.ax_truth.set_title('Ground Truth', fontsize=7.5, color='#1A1A2E', pad=2)
        self.img_truth = self.ax_truth.imshow(
            np.zeros((self.TRUTH_RES, self.TRUTH_RES)), origin='lower',
            extent=[0, 1, 0, 1], cmap=self.cmap, vmin=0, vmax=1, alpha=0.85,
            zorder=10)

        self.img_phi = self.ax_phi.imshow(np.zeros((self.GP_RES, self.GP_RES)),
                                          origin='lower', extent=[0, 1, 0, 1],
                                          cmap=self.cmap, vmin=0, vmax=1, alpha=0.72,
                                          zorder=1)
        self.line_plan, = self.ax_phi.plot([], [], color='#00838F', lw=1.8,
                                           alpha=0.9, zorder=2)

        # Farbskala fuer Φ
        cbar_x = 0.39 + panel_w + 0.01
        ax_cbar = self.fig.add_axes([cbar_x, panel_bottom, 0.012, panel_h])
        self.cbar = self.fig.colorbar(self.img_phi, cax=ax_cbar)
        self.cbar.set_label('Φ (normalized)', fontsize=8.5, color='#555555')
        self.cbar.ax.tick_params(labelsize=7.5, color='#999999',
                                 labelcolor='#555555')
        self.cbar.outline.set_edgecolor('#ccc')

        # ── Live-Metriken (rechte Spalte) ────────────────────────────────
        met_x = cbar_x + 0.04
        self.fig.text(met_x, 0.92, 'Live Metrics', fontsize=11,
                      fontweight='bold', color='#1A1A2E')
        self.metric_txt = {}
        self._info_buttons = []
        metric_rows = [
            ('length', 'Driven Length'),
            ('erg_phi', 'Ergodicity vs. Φ'),
            ('explore', 'Exploration (Info Gain)'),
            ('erg_truth', 'Ergodicity vs. Truth'),
            ('coverage', 'Coverage Error'),
        ]
        row_h, gap, top0 = 0.092, 0.014, 0.89
        for i, (key, label) in enumerate(metric_rows):
            top = top0 - i * (row_h + gap)
            self.fig.text(met_x, top, label, fontsize=8, color='#555555')
            self.metric_txt[key] = self.fig.text(
                met_x, top - 0.035, '–', fontsize=14.5, fontweight='bold',
                color='#1A1A2E')
            self._info_buttons.append(
                self._make_info_button(0.965, top - 0.008, key))

        # Gemeinsame Erklaerungs-Box: zeigt den Text des zuletzt geklickten
        # Info-Knopfs, ueberlagert dafuer temporaer die Panels darunter.
        self.info_box = self.fig.text(
            0.20, 0.86, '', fontsize=8.8, color='#1A1A2E', ha='left', va='top',
            wrap=True, visible=False, zorder=50,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffefa',
                      edgecolor='#1565C0', linewidth=1.2))

        # ── Status-Texte (zentriert ueber den beiden Panels) ─────────────
        center_x = 0.39 / 2 + 0.03 / 2 + panel_w / 2    # Mitte zwischen den Panels
        self.status_txt = self.fig.text(center_x, 0.96, '', ha='center',
                                        fontsize=11.5, color='#1A1A2E')
        self.round_txt = self.fig.text(center_x, 0.935, '', ha='center',
                                       fontsize=9, color='#555555')

        # ── Fortschrittsbalken fuer Planung/Berechnung ──────────────────────
        prog_y = panel_bottom - 0.065
        
        self.ax_prog_inf = self.fig.add_axes([0.07, prog_y + 0.012, 0.68, 0.01])
        self.ax_prog_inf.set_xlim(0, 1); self.ax_prog_inf.set_ylim(0, 1)
        self.ax_prog_inf.axis('off')
        self.prog_track_inf = mpatches.Rectangle((0, 0), 1, 1, facecolor='#eaecf2', edgecolor='#ccc', lw=0.8)
        self.prog_fill_inf = mpatches.Rectangle((0, 0), 0.0, 1, facecolor='#00838F', edgecolor='none')
        self.ax_prog_inf.add_patch(self.prog_track_inf)
        self.ax_prog_inf.add_patch(self.prog_fill_inf)

        self.ax_prog_svgd = self.fig.add_axes([0.07, prog_y, 0.68, 0.01])
        self.ax_prog_svgd.set_xlim(0, 1); self.ax_prog_svgd.set_ylim(0, 1)
        self.ax_prog_svgd.axis('off')
        self.prog_track_svgd = mpatches.Rectangle((0, 0), 1, 1, facecolor='#eaecf2', edgecolor='#ccc', lw=0.8)
        self.prog_fill_svgd = mpatches.Rectangle((0, 0), 0.0, 1, facecolor='#8E24AA', edgecolor='none')
        self.ax_prog_svgd.add_patch(self.prog_track_svgd)
        self.ax_prog_svgd.add_patch(self.prog_fill_svgd)
        
        self.prog_txt = self.fig.text(0.07 + 0.345, prog_y + 0.025, '',
                                      ha='center', fontsize=8.5, color='#555555')

        # ── Widgets (unteres Drittel) ─────────────────────────────────────
        ctrl_bottom = 0.03
        ctrl_h = 0.20

        # Solver-Auswahl
        ax_solver = self.fig.add_axes([0.07, ctrl_bottom, 0.09, ctrl_h])
        ax_solver.set_title('Solver', fontsize=9.5)
        self.r_solver = RadioButtons(ax_solver, SOLVERS, active=0)
        self.r_solver.on_clicked(self._on_solver)
        self._info_buttons.append(self._make_info_button(0.07 + 0.09 - 0.02, ctrl_bottom + ctrl_h + 0.01, 'solvers'))

        # Zielfunktion-Auswahl
        ax_phi_model = self.fig.add_axes([0.17, ctrl_bottom, 0.10, ctrl_h])
        ax_phi_model.set_title('Target Function Φ', fontsize=9.5)
        self.r_phi = RadioButtons(ax_phi_model, PHI_UI, active=0)
        self.r_phi.on_clicked(self._on_phi)

        # Planer-Auswahl (Netz / Linear / Heuristik)
        ax_init = self.fig.add_axes([0.28, ctrl_bottom, 0.11, ctrl_h])
        ax_init.set_title('Planner', fontsize=9.5)
        self.r_init = RadioButtons(ax_init, INIT_MODES, active=0)
        self.r_init.on_clicked(self._on_init_mode)

        btn_x = 0.41
        btn_w = 0.12
        btn_half = 0.055
        btn_gap = 0.01

        # Form-Auswahl
        ax_shape = self.fig.add_axes([btn_x, ctrl_bottom, btn_w, 0.05])
        ax_shape.axis('off')
        self.shape_txt = ax_shape.text(0.5, 0.95, '', ha='center', fontsize=10.5,
                                       color='#1A1A2E', transform=ax_shape.transAxes)
        ax_prev = self.fig.add_axes([btn_x, ctrl_bottom, btn_half, 0.035])
        self.b_prev = Button(ax_prev, '◀ Shape')
        self.b_prev.on_clicked(lambda e: self._change_shape(-1))
        ax_next = self.fig.add_axes([btn_x + btn_half + btn_gap, ctrl_bottom, btn_half, 0.035])
        self.b_next = Button(ax_next, 'Shape ▶')
        self.b_next.on_clicked(lambda e: self._change_shape(+1))

        # Clear Obstacles
        ax_obs = self.fig.add_axes([btn_x, 0.09, btn_w, 0.026])
        self.b_obs = Button(ax_obs, 'Clear Obstacles')
        self.b_obs.on_clicked(self._on_clear_obstacles)

        # Define Target
        ax_def_tgt = self.fig.add_axes([btn_x, 0.12, btn_w, 0.026])
        self.b_def_tgt = Button(ax_def_tgt, 'Define Target')
        self.b_def_tgt.on_clicked(self._open_drawing_window)

        # Eraser Mode
        ax_eraser = self.fig.add_axes([btn_x, 0.15, btn_w, 0.026])
        self.b_eraser = Button(ax_eraser, 'Eraser Mode: OFF')
        def toggle_eraser(event):
            self.eraser_mode = not self.eraser_mode
            self.b_eraser.label.set_text(f'Eraser Mode: {"ON" if self.eraser_mode else "OFF"}')
            self.fig.canvas.draw_idle()
        self.b_eraser.on_clicked(toggle_eraser)

        # Allknowing Mode
        ax_allknowing = self.fig.add_axes([btn_x, 0.18, btn_w, 0.026])
        self.b_allknowing = Button(ax_allknowing, 'Mode: Explore')
        def toggle_allknowing(event):
            self.allknowing_mode = not self.allknowing_mode
            self.b_allknowing.label.set_text(f'Mode: {"Allknowing" if self.allknowing_mode else "Explore"}')
            self.reset(replan=True)
            self.fig.canvas.draw_idle()
        self.b_allknowing.on_clicked(toggle_allknowing)

        # Play / Reset
        ax_play = self.fig.add_axes([btn_x, 0.21, btn_half, 0.038])
        self.b_play = Button(ax_play, '▶ Play')
        self.b_play.on_clicked(self._on_play)
        ax_reset = self.fig.add_axes([btn_x + btn_half + btn_gap, 0.21, btn_half, 0.038])
        self.b_reset = Button(ax_reset, 'Reset')
        self.b_reset.on_clicked(lambda e: self.reset(replan=True))

        # Open Writeboard
        ax_open_wb = self.fig.add_axes([btn_x, 0.26, btn_w, 0.026])
        self.b_open_wb = Button(ax_open_wb, 'Open Writeboard')
        self.b_open_wb.on_clicked(self._on_open_writeboard)

        # Open MuJoCo
        ax_open_mj = self.fig.add_axes([btn_x, 0.29, btn_w, 0.026])
        self.b_open_mj = Button(ax_open_mj, 'Open MuJoCo')
        self.b_open_mj.on_clicked(self._on_open_mujoco)

        slider_x = 0.63
        slider_w = 0.14
        
        # Tuning-Regler (κ oder w)
        self._tuning_rect = (slider_x, 0.14, slider_w, 0.028)
        self._build_tuning_slider()

        # Tempo-Slider: Agentengeschwindigkeit in cm/s auf dem Brett, damit sie
        # direkt mit der des Roboterarms vergleichbar ist. Das Maximum ist das,
        # was der Arm noch sauber abfaehrt (siehe mujoco_sim/board.py).
        ax_speed = self.fig.add_axes([slider_x, 0.04, slider_w, 0.028])
        self.s_speed = Slider(ax_speed, 'Speed cm/s', 2, round(MAX_TIP_SPEED * 100),
                              valinit=self.DEFAULT_SPEED_CM_S, valstep=1,
                              color='#555555')

        # SVGD-Slider
        ax_svgd = self.fig.add_axes([slider_x, 0.09, slider_w, 0.028])
        self.s_svgd = Slider(ax_svgd, 'SVGD Iters', 0, 1000, valinit=self.svgd_iters, valstep=25, color='#00838F')
        self.s_svgd.on_changed(self._on_svgd)

        # B-Spline Pts-Slider
        ax_bspline = self.fig.add_axes([slider_x, 0.19, slider_w, 0.028])
        self.s_bspline = Slider(ax_bspline, 'B-Spline Pts', 5, 50, valinit=self.nxi_ui, valstep=1, color='#8E24AA')
        self.s_bspline.on_changed(self._on_nxi)
        self.s_bspline.ax.axvline(25, color='black', linestyle='--', linewidth=1)
        
        # Agent Radius-Slider
        ax_radius = self.fig.add_axes([slider_x, 0.24, slider_w, 0.028])
        self.s_radius = Slider(ax_radius, 'Agent Radius', 0.01, 0.15, valinit=self.sensor_radius_ui, valstep=0.01, color='#FF8F00')
        self.s_radius.on_changed(self._on_sensor_radius)

        # Target Length-Slider
        ax_length = self.fig.add_axes([slider_x, 0.29, slider_w, 0.028])
        self.s_length = Slider(ax_length, 'Target Length', 0.0, 10.0, valinit=self.target_length_ui, valstep=0.1, color='#FF5722')
        self.s_length.on_changed(self._on_target_length)

        # ── Live-Ansichten für μ und σ unten rechts ──────────────────────────
        self.ax_mu_live = self.fig.add_axes([0.79, 0.03, 0.09, 0.20])
        self.ax_sigma_live = self.fig.add_axes([0.90, 0.03, 0.09, 0.20])
        
        for ax, title in [(self.ax_mu_live, 'μ (Mean)'), (self.ax_sigma_live, 'σ (Uncertainty)')]:
            style(ax)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_color('#1A1A2E')
                s.set_linewidth(1.0)
            ax.set_title(title, fontsize=8, color='#1A1A2E', pad=3)
            
        self.img_mu_live = self.ax_mu_live.imshow(
            np.zeros((self.GP_RES, self.GP_RES)), origin='lower',
            extent=[0, 1, 0, 1], cmap=self.cmap, vmin=0, vmax=1, alpha=0.9)
            
        self.img_sigma_live = self.ax_sigma_live.imshow(
            np.zeros((self.GP_RES, self.GP_RES)), origin='lower',
            extent=[0, 1, 0, 1], cmap=self.cmap, vmin=0, vmax=1, alpha=0.9)

    # ── Zustand ──────────────────────────────────────────────────────────
    def reset(self, replan):
        if self.custom_truth_np is not None:
            truth = self.custom_truth_np
        else:
            truth = self.truths[self.shape_i]
            
        belief = GPBelief(grid_res=self.GP_RES, lengthscale=0.08, noise=0.05,
                          device=self.device)
                          
        if self.custom_prior_mask is not None:
            y_idx, x_idx = np.where(self.custom_prior_mask)
            pts = np.stack([x_idx / (self.TRUTH_RES - 1), y_idx / (self.TRUTH_RES - 1)], axis=-1)
            if len(pts) > 500:
                idx = np.random.choice(len(pts), 500, replace=False)
                pts = pts[idx]
            prior_pts = torch.from_numpy(pts).float().to(self.device)
        else:
            g = torch.Generator().manual_seed(self.seed * 977 + self.shape_i)
            prior_pts = acb.prior_points('zufall', self.N_PRIOR, generator=g,
                                         device=self.device)
        _, prior_vals = measure(prior_pts, truth, noise_std=0.02)
        belief.observe(prior_pts, prior_vals)
        self.belief0 = belief.clone()  # Referenzpunkt fuer information_gain
        self._last_phi = None
        self._last_mu = None
        self._last_sd = None

        if self.phi_ui == 'mass':
            # `apply_cfm_belief.zieldichte` rechnet fuer 'mass' intern
            # w = kappa/(1+kappa) -- hier umgekehrt, damit der direkt
            # eingestellte Anteil w exakt (nicht nur ueber die verzerrte
            # Kappa-Skala) ankommt.
            w = min(max(self.mass_w, 0.0), 0.98)
            kappa0 = w / max(1e-6, 1.0 - w)
        else:
            kappa0 = self.kappa
        args = build_args(self.device, PHI_INTERNAL[self.phi_ui], kappa0,
                          svgd_iters=self.svgd_iters, phi_tau=self.niveau_tau,
                          sensor_radius=self.sensor_radius_ui)
        args.obstacle = CompositeObstacle(self.obstacles) if len(self.obstacles) > 0 else None
        args.allknowing_mode = self.allknowing_mode
        if self.target_length_ui > 0:
            args.target_length = self.target_length_ui
            args.target_length_cfg = 2.0
        else:
            args.target_length = None
            args.target_length_cfg = 0.0

        # ── Planer-Auswahl ────────────────────────────────────────────────
        if self.init_mode == 'CFM':
            planner = self.planner
        else:
            from common.planner import GradientPlanner
            # steps=0 macht den Planer zu einem reinen Initialisierungs-
            # Provider, analog zum trainierten Netz. Er gibt einfach die
            # Startbahn aus, die dann via SVGD nachbearbeitet oder
            # direkt abgefahren wird.
            planner = GradientPlanner(
                nxi=25, pts=128, deg=5, K=8, metric='fourier',
                steps=0, lr=0.05, device=self.device, seed=self.seed)
            # Flag fuer `_get_init`:
            if self.init_mode == 'Linear':
                planner._default_init = "Linear"
            elif self.init_mode == 'Manual':
                planner._default_init = "Manual"
            else:  # 'Heuristic'
                planner._default_init = None  # wird dynamisch pro Runde aus Phi gebaut

        self.mission = Mission(planner, truth, belief, args, self.solver,
                               svgd=self.svgd, nxi_ui=self.nxi_ui)
        self.mission.app = self
        self.gen = self.mission.rounds()
        self.truth_np = truth.detach().cpu().numpy()
        self.prior_pts_np = prior_pts.detach().cpu().numpy()
        self.visible = np.zeros((self.TRUTH_RES, self.TRUTH_RES), dtype=bool)
        self._reveal(self.prior_pts_np, radius=args.sensor_radius)
        self.img_truth.set_data(self.truth_np)

        self.driven_np = np.zeros((0, 2))
        self.cur = None
        self.play_idx = 0
        self.play_arc = 0.0
        self.playing = False
        self.finished = False
        self.b_play.label.set_text('▶ Play')
        self.prog_fill_inf.set_width(0.0)
        self.prog_fill_svgd.set_width(0.0)
        self.prog_txt.set_text('')
        for txt in self.metric_txt.values():
            txt.set_text('–')

        if self.custom_truth_np is None:
            self.shape_txt.set_text(self.names[self.shape_i])
        else:
            self.shape_txt.set_text('Custom')
        self.status_txt.set_text('Prior Knowledge — Play plans Round 1')
        self.round_txt.set_text('')
        self._update_phi_title()
        self._advance_round()  # Runde 1 im Hintergrund vorplanen, noch nicht fahren
        self._redraw()

    def _reveal(self, pts, radius):
        if pts is None or len(pts) == 0:
            return
        pts = np.atleast_2d(pts)
        d2 = ((self.XX[None] - pts[:, 0, None, None]) ** 2
              + (self.YY[None] - pts[:, 1, None, None]) ** 2)
        self.visible |= (d2 <= radius ** 2).any(axis=0)

    def _advance_round(self):
        """Naechste Runde vom Missions-Generator holen, im Hintergrund-Thread.

        Der Netz-Vorwaertspass blockiert 1-2 Sekunden; ohne Thread wuerde die
        Tk-Eventschleife (und damit Play/Pause/Reset) fuer diese Zeit
        einfrieren. `gen`/`result` werden als lokale Variablen gebunden statt
        ueber `self` gelesen: startet der Nutzer waehrenddessen per Reset eine
        neue Mission, liest der noch laufende alte Thread trotzdem weiter aus
        seinem eigenen (jetzt verwaisten) Generator und kann `self.mission`
        der neuen Mission nicht mehr verfaelschen.
        """
        if self.init_mode == 'Manual' and self.manual_init_drawn is None:
            self.waiting_for_manual_init = True
            self.prog_txt.set_text('Wait for initialization input (Draw in World)')
            self.fig.canvas.draw_idle()
            return
            
        if hasattr(self, 'mission'):
            self.mission.current_step = 'inference'
        self.busy = True
        self._busy_t0 = time.perf_counter()
        gen = self.gen
        result = {}
        self._plan_result = result

        def worker():
            try:
                result['val'] = next(gen)
            except StopIteration:
                result['stop'] = True
            except Exception as exc:  # noqa: BLE001 — der GUI-Thread soll das sehen
                result['err'] = exc

        self._plan_thread = threading.Thread(target=worker, daemon=True)
        self._plan_thread.start()

    def _poll_plan(self):
        """Von _tick waehrend `self.busy` aufgerufen: Balken fuellen, Ergebnis
        uebernehmen, sobald der Hintergrund-Thread fertig ist."""
        elapsed = time.perf_counter() - self._busy_t0
        
        step = 'inference'
        if hasattr(self, 'mission'):
            step = getattr(self.mission, 'current_step', 'inference')
            
        if step == 'inference':
            frac = min(0.96, elapsed / max(self.est_plan_time, 0.05))
            self.prog_fill_inf.set_width(frac)
            self.prog_fill_svgd.set_width(0.0)
            self.prog_txt.set_text(f'Inference ... {elapsed:.1f}s')
        else:
            self.prog_fill_inf.set_width(1.0)
            if not hasattr(self, '_svgd_t0'):
                self._svgd_t0 = time.perf_counter()
            svgd_elapsed = time.perf_counter() - self._svgd_t0
            frac = min(0.96, svgd_elapsed / max(self.est_svgd_time, 0.05))
            self.prog_fill_svgd.set_width(frac)
            self.prog_txt.set_text(f'SVGD ... {svgd_elapsed:.1f}s')

        if self._plan_thread.is_alive():
            return

        if step == 'inference':
            self.est_plan_time = 0.6 * self.est_plan_time + 0.4 * elapsed
        else:
            svgd_elapsed = time.perf_counter() - getattr(self, '_svgd_t0', time.perf_counter())
            self.est_svgd_time = 0.6 * self.est_svgd_time + 0.4 * svgd_elapsed
            inf_elapsed = getattr(self, '_svgd_t0', time.perf_counter()) - self._busy_t0
            self.est_plan_time = 0.6 * self.est_plan_time + 0.4 * inf_elapsed
            
        if hasattr(self, '_svgd_t0'):
            del self._svgd_t0

        self.busy = False
        self.prog_fill_inf.set_width(0.0)
        self.prog_fill_svgd.set_width(0.0)
        self.prog_txt.set_text('')

        res = self._plan_result
        if 'err' in res:
            self.finished = True
            self.playing = False
            self.b_play.label.set_text('▶ Play')
            self.status_txt.set_text(f'Planning error: {res["err"]}')
            return
        if res.get('stop'):
            self.cur = None
            self.finished = True
            self.playing = False
            self.b_play.label.set_text('▶ Play')
            self.status_txt.set_text('Mission finished — Reset for new run')
            return

        self.cur = res['val']
        self.play_idx = 0
        self.play_arc = 0.0
        r, n = self.cur['round'], self.cur['n_rounds']
        self.status_txt.set_text(f"Solver {self.solver} — Round {r + 1}/{n}")

    # ── Der Regler unter Phi: κ fuer die meisten Modelle, w fuer 'mass' ──
    def _build_tuning_slider(self):
        """`mass` (common/acquisition.phi_mass) nimmt direkt einen Anteil
        w in [0,1) -- 'wie viel der Zielmasse auf Erkundung entfaellt' --
        nicht kappa. `apply_cfm_belief.zieldichte` rechnet fuer 'mass' intern
        ohnehin kappa -> w = kappa/(1+kappa) um; ohne diesen Regler waere w
        nur ueber diese versteckte, nichtlineare Umrechnung einstellbar, und
        das Label 'κ' wuerde beschreiben, was gar nicht eingestellt wird.
        Deshalb wird der Regler beim Wechsel zu/von 'mass' neu aufgebaut,
        statt denselben Regler mit doppelter Bedeutung zu ueberladen."""
        x, y, w, h = self._tuning_rect
        ax = self.fig.add_axes([x, y, w, h])
        if self.phi_ui == 'mass':
            s = Slider(ax, 'w (Exploration Ratio)', 0.0, 0.95,
                      valinit=self.mass_w, valstep=0.01, color='#1565C0')
        elif self.phi_ui == 'niveau':
            s = Slider(ax, 'τ (Level Set Threshold)', 0.01, 0.99,
                      valinit=self.niveau_tau, valstep=0.01, color='#1565C0')
        else:
            s = Slider(ax, 'κ (explore ↔ exploit)', 0.0, 6.0,
                      valinit=self.kappa, valstep=0.1, color='#1565C0')
        s.on_changed(self._on_tuning)
        self.s_kappa = s

    def _rebuild_tuning_slider(self):
        self.s_kappa.ax.remove()
        self._build_tuning_slider()
        self.fig.canvas.draw_idle()

    # ── Widget-Callbacks ─────────────────────────────────────────────────
    def _on_solver(self, label):
        self.solver = label[0]
        self.reset(replan=True)

    def _on_phi(self, label):
        self.phi_ui = label
        self._rebuild_tuning_slider()
        self._update_phi_title()
        self.reset(replan=True)

    def _on_tuning(self, val):
        if self.phi_ui == 'mass':
            self.mass_w = float(val)
        elif self.phi_ui == 'niveau':
            self.niveau_tau = float(val)
        else:
            self.kappa = float(val)
        self.reset(replan=True)

    def _on_init_mode(self, label):
        self.init_mode = label
        self.reset(replan=True)

    def _on_svgd(self, val):
        self.svgd_iters = int(val)
        self.reset(replan=True)
        
    def _on_nxi(self, val):
        self.nxi_ui = int(val)
        self.reset(replan=True)

    def _on_sensor_radius(self, val):
        self.sensor_radius_ui = float(val)
        self.agent_disk.set_radius(self.sensor_radius_ui)
        self.reset(replan=True)

    def _on_target_length(self, val):
        self.target_length_ui = float(val)
        self.reset(replan=True)

    def _on_open_writeboard(self, event):
        import multiprocessing as mp
        if not hasattr(self, 'p_wb') or not self.p_wb.is_alive():
            from writeboard import run_writeboard
            self.p_wb = mp.Process(target=run_writeboard, args=(self.shared_truth, self.agent_info_array, self.shared_obstacles))
            self.p_wb.start()

    def _on_open_mujoco(self, event):
        import multiprocessing as mp
        if not hasattr(self, 'p_mj') or not self.p_mj.is_alive():
            from mujoco_sim.run_mujoco import run_mujoco_sim
            self.p_mj = mp.Process(target=run_mujoco_sim, args=(self.agent_info_array, self.shared_truth))
            self.p_mj.start()

    def _change_shape(self, d):
        self.custom_truth_np = None
        self.custom_prior_mask = None
        self.shape_i = (self.shape_i + d) % len(self.names)
        if self.shared_truth is not None:
            with self.shared_truth.get_lock():
                np_truth = np.frombuffer(self.shared_truth.get_obj(), dtype=np.float64).reshape((self.TRUTH_RES, self.TRUTH_RES))
                np_truth[:] = self.truths[self.shape_i].detach().cpu().numpy()[:]
        self.reset(replan=True)

    def _on_clear_obstacles(self, event):
        if self.shared_obstacles is not None:
            with self.shared_obstacles.get_lock():
                arr = np.frombuffer(self.shared_obstacles.get_obj(), dtype=np.float64)
                arr[:] = 0.0
        else:
            self.obstacles.clear()
            for p in self.obs_patches:
                p.remove()
            self.obs_patches.clear()
        self.reset(replan=True)

    def _on_press(self, event):
        if getattr(self, 'waiting_for_manual_init', False) and event.inaxes == self.ax_world:
            self.is_drawing_manual = True
            self.manual_path = [(event.xdata, event.ydata)]
            if getattr(self, 'manual_line', None) is not None:
                self.manual_line.remove()
                self.manual_line = None
            self.manual_line, = self.ax_world.plot([event.xdata], [event.ydata], color='#D81B60', lw=2.5, ls='--', zorder=20)
            self.fig.canvas.draw_idle()
            return

        if event.inaxes == self.ax_obs_template:
            if len(self.obstacles) >= 10:
                print("Max 10 obstacles allowed.")
                return
            self._dragging_obs = True
            self.obs_drag_patch.set_visible(False)
            self.obs_drag_margin.set_visible(False)

    def _on_motion(self, event):
        if getattr(self, 'is_drawing_manual', False) and event.inaxes == self.ax_world:
            self.manual_path.append((event.xdata, event.ydata))
            xdata, ydata = zip(*self.manual_path)
            self.manual_line.set_data(xdata, ydata)
            self.fig.canvas.draw_idle()
            return
            
        if getattr(self, '_dragging_obs', False):
            if event.inaxes == self.ax_world:
                self.obs_drag_patch.center = (event.xdata, event.ydata)
                self.obs_drag_margin.center = (event.xdata, event.ydata)
                self.obs_drag_patch.set_visible(True)
                self.obs_drag_margin.set_visible(True)
            else:
                self.obs_drag_patch.set_visible(False)
                self.obs_drag_margin.set_visible(False)
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if getattr(self, 'is_drawing_manual', False):
            self.is_drawing_manual = False
            if len(self.manual_path) > 1:
                arr = np.array(self.manual_path)
                dists = np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=1))
                cum_dist = np.insert(np.cumsum(dists), 0, 0)
                if cum_dist[-1] > 0.01:
                    eval_dists = np.linspace(0, cum_dist[-1], 25)
                    interp_x = np.interp(eval_dists, cum_dist, arr[:, 0])
                    interp_y = np.interp(eval_dists, cum_dist, arr[:, 1])
                    cps = np.stack([interp_x, interp_y], axis=-1)
                    cps = np.clip(cps, 0.02, 0.98)
                    self.manual_init_drawn = torch.from_numpy(cps).float().unsqueeze(0).to(self.device)
                    self.waiting_for_manual_init = False
                    if getattr(self, 'manual_line', None) is not None:
                        self.manual_line.remove()
                        self.manual_line = None
                    self._advance_round()
                else:
                    if getattr(self, 'manual_line', None) is not None:
                        self.manual_line.remove()
                        self.manual_line = None
            else:
                if getattr(self, 'manual_line', None) is not None:
                    self.manual_line.remove()
                    self.manual_line = None
            self.fig.canvas.draw_idle()
            return

        if getattr(self, '_dragging_obs', False):
            self._dragging_obs = False
            self.obs_drag_patch.set_visible(False)
            self.obs_drag_margin.set_visible(False)
            if event.inaxes == self.ax_world:
                new_obs = CircleObstacle(center=(event.xdata, event.ydata), margin=0.0)
                if self.shared_obstacles is not None:
                    with self.shared_obstacles.get_lock():
                        arr = np.frombuffer(self.shared_obstacles.get_obj(), dtype=np.float64)
                        for i in range(10):
                            if arr[i*3+2] == 0.0:  # empty slot
                                arr[i*3] = event.xdata
                                arr[i*3+1] = event.ydata
                                arr[i*3+2] = new_obs.radius
                                break
                else:
                    self.obstacles.append(new_obs)
                    patch = mpatches.Circle(new_obs.center, new_obs.radius, facecolor='#9E9E9E', alpha=0.75, edgecolor='#424242', lw=1.5, zorder=6)
                    margin = mpatches.Circle(new_obs.center, new_obs.effective_radius, facecolor='none', edgecolor='#424242', lw=0.8, ls=':', alpha=0.6, zorder=6)
                    self.ax_world.add_patch(patch)
                    self.ax_world.add_patch(margin)
                    self.obs_patches.extend([patch, margin])
                self.reset(replan=True)
            self.fig.canvas.draw_idle()

    def _on_play(self, event):
        if self.finished:
            self.reset(replan=True)
            self.playing = True
        else:
            self.playing = not self.playing
        self.b_play.label.set_text('❚❚ Pause' if self.playing else '▶ Play')

    def _update_phi_title(self):
        """Titel des rechten Panels: zeigt die mathematische Definition
        des gewaehlten Phi-Modells, oder den Komponentennamen bei mu/sigma."""
        if self.phi_view == 'μ':
            t = 'μ — Posterior Mean (known)'
        elif self.phi_view == 'σ':
            t = 'σ — Posterior Uncertainty (unknown)'
        else:
            phi_defs = {
                'ucb':    'Φ = μ + κ·σ',
                'mass':   'Φ = (1−w)·μ̂ + w·σ̂',
                'eid':    'Φ = μ̂ + κ·EID   (EID ∝ ||∇μ||² + ||∇σ||²)',
                'niveau': 'Φ = P(f(x) > τ)   (f: true density, P: GP-prob)',
            }
            defn = phi_defs.get(self.phi_ui, 'Φ')
            t = f'{defn}   —   Target Density from GP Belief'
        self.ax_phi.set_title(t, fontsize=11, color='#1A1A2E', loc='left')
        self.fig.canvas.draw_idle()

    def _on_phi_view(self, label):
        """Toggle zwischen Φ (kombiniert), μ (Mittelwert), σ (Unsicherheit)
        im rechten Anzeige-Panel. Kein Reset noetig — nur die Darstellung
        wechselt, die Planung laeuft unveraendert weiter."""
        self.phi_view = label
        self._update_phi_title()

    def _make_info_button(self, x, y, key):
        # Plain 'i' statt eines Kreis-Symbols: DejaVu Sans (matplotlibs
        # Standardfont) hat kein Glyph fuer 'ⓘ'/'ℹ' und wuerde sonst ein
        # Tofu-Kaestchen zeichnen.
        ax = self.fig.add_axes([x, y, 0.026, 0.03])
        btn = Button(ax, 'i', color='#eaecf2', hovercolor='#d7dae3')
        btn.label.set_fontsize(9)
        btn.label.set_fontstyle('italic')
        btn.label.set_fontweight('bold')
        btn.on_clicked(lambda evt, k=key: self._toggle_info(k))
        return btn

    def _toggle_info(self, key):
        """Klick auf denselben Knopf schliesst die Box wieder, ein anderer
        Knopf tauscht nur den Text aus — nie mehr als eine Erklaerung offen."""
        if self._info_open == key:
            self._info_open = None
            self.info_box.set_visible(False)
        else:
            self._info_open = key
            self.info_box.set_text(INFO_TEXT[key])
            self.info_box.set_visible(True)
        self.fig.canvas.draw_idle()

    # ── Drawing Window ───────────────────────────────────────────────────
    def _open_drawing_window(self, event):
        self.fig_draw = plt.figure(figsize=(12, 6), facecolor='white')
        
        self.ax_draw_target = self.fig_draw.add_axes([0.05, 0.2, 0.4, 0.7])
        self.ax_draw_known = self.fig_draw.add_axes([0.55, 0.2, 0.4, 0.7])
        
        self.ax_draw_target.set_title('Draw Target Density\n(Hold mouse to increase density)', fontsize=11)
        self.ax_draw_known.set_title('Draw Known Area\n(Click/drag to mark as known)', fontsize=11)
        
        style(self.ax_draw_target)
        style(self.ax_draw_known)
        
        self.draw_target_grid = np.zeros((self.TRUTH_RES, self.TRUTH_RES))
        self.draw_known_grid = np.zeros((self.TRUTH_RES, self.TRUTH_RES), dtype=bool)
        
        self.img_draw_target = self.ax_draw_target.imshow(
            self.draw_target_grid, origin='lower', extent=[0, 1, 0, 1], cmap=self.cmap, vmin=0, vmax=1)
        self.img_draw_known = self.ax_draw_known.imshow(
            self.draw_known_grid.astype(float), origin='lower', extent=[0, 1, 0, 1], cmap='gray', vmin=0, vmax=1)
            
        ax_save = self.fig_draw.add_axes([0.4, 0.05, 0.2, 0.08])
        self.b_save = Button(ax_save, 'Save and Close')
        self.b_save.on_clicked(self._save_drawing)
        
        self.is_drawing = False
        self.drawing_axes = None
        self.last_mouse_event = None
        
        self.fig_draw.canvas.mpl_connect('button_press_event', self._on_draw_press)
        self.fig_draw.canvas.mpl_connect('button_release_event', self._on_draw_release)
        self.fig_draw.canvas.mpl_connect('motion_notify_event', self._on_draw_motion)
        
        self.draw_timer = self.fig_draw.canvas.new_timer(interval=20)
        self.draw_timer.add_callback(self._on_draw_timer_tick)
        
        plt.show(block=False)
        
    def _on_draw_press(self, event):
        if event.inaxes in (self.ax_draw_target, self.ax_draw_known):
            self.is_drawing = True
            self.drawing_axes = event.inaxes
            self.last_mouse_event = event
            self._apply_brush(event)
            self.draw_timer.start()

    def _on_draw_release(self, event):
        self.is_drawing = False
        self.drawing_axes = None
        self.last_mouse_event = None
        self.draw_timer.stop()
        
    def _on_draw_motion(self, event):
        if self.is_drawing and event.inaxes == self.drawing_axes:
            self.last_mouse_event = event
            self._apply_brush(event)
            
    def _on_draw_timer_tick(self):
        if self.is_drawing and self.last_mouse_event is not None:
            self._apply_brush(self.last_mouse_event)
            
    def _apply_brush(self, event):
        if event.xdata is None or event.ydata is None:
            return
            
        res = self.TRUTH_RES
        x_idx = int(np.clip(event.xdata * res, 0, res - 1))
        y_idx = int(np.clip(event.ydata * res, 0, res - 1))
        
        if self.drawing_axes == self.ax_draw_target:
            sigma = 2.0
            y, x = np.ogrid[-y_idx:res-y_idx, -x_idx:res-x_idx]
            blob = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            self.draw_target_grid += blob * 0.25
            self.draw_target_grid = np.clip(self.draw_target_grid, 0, 1)
            self.img_draw_target.set_data(self.draw_target_grid)
            self.fig_draw.canvas.draw_idle()
            
        elif self.drawing_axes == self.ax_draw_known:
            radius = 3
            y, x = np.ogrid[-y_idx:res-y_idx, -x_idx:res-x_idx]
            mask = x**2 + y**2 <= radius**2
            self.draw_known_grid[mask] = True
            self.img_draw_known.set_data(self.draw_known_grid.astype(float))
            self.fig_draw.canvas.draw_idle()
            
    def _save_drawing(self, event):
        self.draw_timer.stop()
        
        if self.draw_target_grid.max() > 0:
            norm_grid = self.draw_target_grid / self.draw_target_grid.max()
            self.custom_truth_np = torch.from_numpy(norm_grid).float().to(self.device)
            if self.shared_truth is not None:
                with self.shared_truth.get_lock():
                    np_truth = np.frombuffer(self.shared_truth.get_obj(), dtype=np.float64).reshape((self.TRUTH_RES, self.TRUTH_RES))
                    np_truth[:] = norm_grid[:]
        else:
            self.custom_truth_np = None
            
        if self.draw_known_grid.any():
            self.custom_prior_mask = self.draw_known_grid.copy()
        else:
            self.custom_prior_mask = None
            
        plt.close(self.fig_draw)
        
        if self.custom_truth_np is not None:
            self.shape_txt.set_text('Custom')
            
        self.reset(replan=True)

    # ── Animation ────────────────────────────────────────────────────────
    def _tick(self):
        """Timer-Callback. Tkinter faengt Ausnahmen aus `after()`-Callbacks
        ab, druckt sie auf stderr und laeuft dann einfach weiter — bei einem
        Fehler, der bei jedem Tick erneut auftritt (z.B. ein Layout-Bug, der
        bei jedem Redraw wiederkehrt), wirkt die GUI dadurch eingefroren,
        obwohl der Prozess weiterlaeuft. Deshalb hier explizit auffangen: den
        Timer anhalten und den Fehler sichtbar machen, statt still haengen zu
        bleiben."""
        try:
            self._tick_inner()
        except Exception:
            traceback.print_exc()
            self.timer.stop()
            self.playing = False
            try:
                self.status_txt.set_text('Error — Simulation stopped (see console)')
                self.round_txt.set_text('Try reset or close window')
                self.fig.canvas.draw_idle()
            except Exception:
                pass

    def _tick_inner(self):
        if self.shared_obstacles is not None:
            with self.shared_obstacles.get_lock():
                arr = np.frombuffer(self.shared_obstacles.get_obj(), dtype=np.float64).copy()
            new_obstacles = []
            for i in range(10):
                x, y, r = arr[i*3], arr[i*3+1], arr[i*3+2]
                if r > 0:
                    new_obstacles.append(CircleObstacle(center=(x, y), margin=0.0))
            current_obs_data = [(o.center[0], o.center[1], o.radius) for o in self.obstacles]
            new_obs_data = [(o.center[0], o.center[1], o.radius) for o in new_obstacles]
            if current_obs_data != new_obs_data:
                self.obstacles = new_obstacles
                for p in self.obs_patches:
                    p.remove()
                self.obs_patches.clear()
                for obs in self.obstacles:
                    patch = mpatches.Circle(obs.center, obs.radius, facecolor='#9E9E9E', alpha=0.75, edgecolor='#424242', lw=1.5, zorder=6)
                    margin = mpatches.Circle(obs.center, obs.effective_radius, facecolor='none', edgecolor='#424242', lw=0.8, ls=':', alpha=0.6, zorder=6)
                    self.ax_world.add_patch(patch)
                    self.ax_world.add_patch(margin)
                    self.obs_patches.extend([patch, margin])
                if not self.busy:
                    self.reset(replan=True)
                    return

        if self.shared_truth is not None:
            with self.shared_truth.get_lock():
                np_truth = np.frombuffer(self.shared_truth.get_obj(), dtype=np.float64).reshape((self.TRUTH_RES, self.TRUTH_RES))
                self.truth_np = np_truth.copy()
            self.custom_truth_np = torch.from_numpy(self.truth_np).float().to(self.device)
            self.img_truth.set_data(self.truth_np)
            if hasattr(self, 'mission'):
                self.mission.truth = self.custom_truth_np
                
        if self.agent_info_array is not None:
            pos = (0.5, 0.5)
            if self.cur is not None:
                seg = self.cur['seg'].detach().cpu().numpy()
                pos = seg[max(self.play_idx - 1, 0)] if len(seg) else (0.5, 0.5)
            elif len(self.driven_np) > 0:
                pos = self.driven_np[-1]
            with self.agent_info_array.get_lock():
                self.agent_info_array[:] = [pos[0], pos[1], self.sensor_radius_ui, 1.0 if self.eraser_mode else 0.0]

        if self.busy:
            self._poll_plan()
            self._redraw()
            return
        if self.playing and not self.finished and self.cur is not None:
            seg = self.cur['seg'].detach().cpu().numpy()
            new_idx = self._advance_index(seg, self.play_idx)
            pts = seg[self.play_idx:new_idx]
            self._reveal(pts, radius=self.mission.args.sensor_radius)
            
            if self.eraser_mode and len(pts) > 0 and self.shared_truth is not None:
                with self.shared_truth.get_lock():
                    np_truth = np.frombuffer(self.shared_truth.get_obj(), dtype=np.float64).reshape((self.TRUTH_RES, self.TRUTH_RES))
                    pts_2d = np.atleast_2d(pts)
                    d2 = ((self.XX[None] - pts_2d[:, 0, None, None]) ** 2
                          + (self.YY[None] - pts_2d[:, 1, None, None]) ** 2)
                    mask = (d2 <= self.mission.args.sensor_radius ** 2).any(axis=0)
                    np_truth[mask] = 0.0
                    self.truth_np = np_truth.copy()
                self.custom_truth_np = torch.from_numpy(self.truth_np).float().to(self.device)
                self.mission.truth = self.custom_truth_np
            
            self.play_idx = new_idx
            if self.play_idx >= len(seg):
                self.driven_np = np.concatenate([self.driven_np, seg], axis=0)
                self.cur = None  # Runde fertig gefahren — naechste wird geplant
                self._advance_round()
        self._redraw()

    def _advance_index(self, seg, idx):
        """
        Naechster Playback-Index, getaktet nach **Bogenlaenge** statt nach
        Index.

        Vorher wurde die Bahn in einer festen Zahl von Ticks abgefahren
        (`len(seg) / (14 * speed)` Punkte pro Tick), d. h. das Tempo hing an der
        Laenge der geplanten Bahn: eine lange Runde wurde genauso schnell
        "durchgespult" wie eine kurze, oft weit schneller als der Roboterarm
        folgen kann. Jetzt legt der Agent pro Sekunde eine feste Strecke auf dem
        Brett zurueck -- dieselbe physikalische Geschwindigkeit, mit der die
        MuJoCo-Seite den Arm faehrt.
        """
        if len(seg) < 2:
            return len(seg)
        arc = np.concatenate([[0.0],
                              np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
        if idx <= 0:                      # neue Bahn -> Wegzaehler zuruecksetzen
            self.play_arc = 0.0
        speed_ui = max_ui_speed(self.s_speed.val / 100.0)   # UI-Einheiten pro Sekunde
        # Der Weg wird als Fliesskomma aufsummiert. Wuerde stattdessen pro Tick
        # mindestens ein Stuetzpunkt uebersprungen, liefe eine grob abgetastete
        # Bahn schneller als eingestellt -- bei 120 Punkten waeren aus 20 cm/s
        # rund 32 cm/s geworden.
        self.play_arc += speed_ui * (self.FRAME_MS / 1000.0)
        new_idx = int(np.searchsorted(arc, self.play_arc, side='left'))
        return int(min(len(seg), max(idx, new_idx)))

    # ── Zeichnen ─────────────────────────────────────────────────────────
    def _redraw(self):
        rgba = self.cmap(np.clip(self.truth_np, 0, 1))
        if getattr(self, 'allknowing_mode', False):
            rgba[..., 3] = 0.55
        else:
            rgba[..., 3] = np.where(self.visible, 0.55, 0.0)
        self.img_world.set_data(rgba)
        self.scat_prior.set_offsets(self.prior_pts_np)

        if self.cur is not None:
            seg = self.cur['seg'].detach().cpu().numpy()
            done = seg[:self.play_idx]
            rest = seg[self.play_idx:]
            full_driven = np.concatenate([self.driven_np, done], axis=0)
            self.line_driven.set_data(full_driven[:, 0], full_driven[:, 1])
            self.line_preview.set_data(rest[:, 0], rest[:, 1])
            pos = seg[max(self.play_idx - 1, 0)] if len(seg) else (0.5, 0.5)
            self.agent_disk.center = tuple(pos)
            self.agent_disk.set_radius(self.mission.args.sensor_radius)
            self.agent_dot.set_data([pos[0]], [pos[1]])

            phi = self.cur['phi'].detach().cpu().numpy()
            mu_grid = self.cur['mu'].detach().cpu().numpy()
            sd_grid = self.cur['sd'].detach().cpu().numpy()
            self._last_mu = mu_grid
            self._last_sd = sd_grid
            
            # Live-Panels unten rechts
            self.img_mu_live.set_data(mu_grid.clip(0) / max(mu_grid.max(), 1e-12))
            self.img_sigma_live.set_data(sd_grid / max(sd_grid.max(), 1e-12))

            # Je nach Toggle: Φ, μ oder σ anzeigen
            if self.phi_view == 'μ':
                show = mu_grid.clip(0) / max(mu_grid.max(), 1e-12)
            elif self.phi_view == 'σ':
                show = sd_grid / max(sd_grid.max(), 1e-12)
            else:
                show = phi
            self.img_phi.set_data(show)
            self.line_plan.set_data(seg[:, 0], seg[:, 1])
            if self.phi_ui == 'niveau':
                tag = 'τ=0.25 (fest)'
            elif self.phi_ui == 'mass':
                kr = self.cur['kappa']
                tag = 'w=%.2f' % (kr / (1.0 + kr))
            else:
                tag = 'κ=%.2f' % self.cur['kappa']
            svgd_tag = (f'  ·  +SVGD×{self.svgd_iters}' if self.svgd_iters > 0
                       else '')
            self.round_txt.set_text(
                f"{SOLVER_DESC[self.solver]}  ·  Φ={self.phi_ui}  ·  {tag}"
                f"{svgd_tag}")
        else:
            self.line_driven.set_data(self.driven_np[:, 0], self.driven_np[:, 1])
            self.line_preview.set_data([], [])

        self._update_metrics()
        self.fig.canvas.draw_idle()

    def _update_metrics(self):
        """Live-Metriken neu berechnen. Absichtlich uebersprungen, solange
        `self.busy`: der Hintergrund-Thread mutiert `self.mission.belief`
        gerade (Cholesky-Cache invalidieren, neue Punkte anhaengen), und ein
        gleichzeitiger Lesezugriff vom Hauptthread waere eine echte Race
        Condition (z.B. X/y kurzzeitig unterschiedlich lang) statt nur eines
        harmlosen veralteten Werts. Die Zahlen frieren also waehrend der
        kurzen Planungspause ein und springen danach auf den aktuellen Stand."""
        if self.busy:
            return

        if self.cur is not None:
            seg = self.cur['seg'].detach().cpu().numpy()
            full = np.concatenate([self.driven_np, seg[:self.play_idx]], axis=0)
            self._last_phi = self.cur['phi']
        else:
            full = self.driven_np

        if full.shape[0] < 2:
            for txt in self.metric_txt.values():
                txt.set_text('–')
            return

        path_t = torch.from_numpy(full).float().to(self.device)
        self.metric_txt['length'].set_text(f"{float(path_length(path_t)):.2f}")

        if self._last_phi is not None:
            erg_phi = self.ergodic(path_t, self._last_phi.to(self.device))
            self.metric_txt['erg_phi'].set_text(f"{erg_phi:.4f}")

        erg_truth = self.ergodic(path_t, self.mission.truth)
        self.metric_txt['erg_truth'].set_text(f"{erg_truth:.4f}")

        cov = float(coverage_vs_truth(path_t, self.mission.truth))
        self.metric_txt['coverage'].set_text(f"{cov:.4f}")

        ig = information_gain(self.belief0, self.mission.belief)
        self.metric_txt['explore'].set_text(f"{ig:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--shapes', type=int, default=12,
                   help='Wie viele Holdout-Formen zum Durchklicken geladen werden.')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    if not os.path.isfile(a.ckpt):
        p.error(f'Checkpoint nicht gefunden: {a.ckpt}')

    App(a.ckpt, a.shapes, device, a.seed)


if __name__ == '__main__':
    main()
