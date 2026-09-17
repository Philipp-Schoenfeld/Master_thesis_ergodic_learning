r"""
variant_runner.py
==================
Erzeugt fuer eine Form und eine Wissensstufe alle Trajektorien-Varianten der
Auswertungsmatrix. Reine Erzeugung — Bewertung und Speichern passiert in
`run_eval_matrix.py`.

Baut auf dem bestehenden `exploration/`-Unterbau auf:

    CfmPlanner        (apply_cfm_belief.py)      gelerntes Netz, startpunkt-
                                                  konditioniert
    SvgdRefiner        (common/svgd_refine.py)    SVGD-Verfeinerung, n_iters
                                                  als freier Parameter
    GPBelief/          (common/belief.py)         Wissensstufen
    MaskiertesWissen
    ucb_density,       (common/acquisition.py)    Phi = mu + kappa*sigma
    particles_from_density
    measure, thin      (common/observation.py)    Messprozess
    lawnmower_path,    (common/baselines.py)      Referenzbahnen
    random_path
    resample_arclength,(exploration_optimierung/  Bogenlaengen-Neuabtastung,
    trim_to_length,    mission.py,                Kuerzung, Laengeneinheit
    LENGTH_UNIT        common/metrics.py)
"""

import math

import numpy as np
import torch

import apply_cfm_belief as acb
from common.acquisition import ucb_density, particles_from_density
from common.baselines import lawnmower_path, lanes_for_length
from common.belief import GPBelief, MaskiertesWissen, muster_maske
from common.metrics import trim_to_length, path_length as _pl
from common.observation import measure, thin
from common.svgd_refine import SvgdRefiner
from exploration_optimierung.mission import resample_arclength, LENGTH_UNIT

from init_baselines import (diagonal_path, straight_top_path,
                            random_walk_path, heuristic_peak_path)

KNOWLEDGE_CONDITIONS = ['ground_truth', 'none_known', 'half_known', 'ten_samples']
SVGD_ITERS = (0, 25, 500, 1000)
N_REPLAN_ROUNDS = 6                     # "replanning every 1/6 (6 in total)"
KAPPA = 2.0                             # Phi = mu + KAPPA*sigma, fester Regler
N_PARTICLES = 256
GP_RES = 64
NXI = 25
PTS_RENDER = 128
TARGET_LENGTH = N_REPLAN_ROUNDS * LENGTH_UNIT   # Bezugslaenge fuer Maeander


# ── Wissensstufen ───────────────────────────────────────────────────────────

def build_belief(condition, truth, seed=0, device='cpu', gp_noise=0.05,
                 gp_lengthscale=0.08):
    """GPBelief/MaskiertesWissen fuer eine der vier Wissensstufen.

    `noise=0.05`, nicht der `GPBelief`-Default 0.01 — siehe die Messung im
    Docstring von `belief.py`: 0.01 laesst den Posterior auf kurzen Bahnen
    weit ueber [0,1] hinausschiessen.

    `gp_lengthscale=0.08` ist derselbe Default wie `GPBelief`/`mission.py`
    selbst ("altes Verhalten") — bestehende Aufrufer ohne dieses Argument
    bleiben also unveraendert. Strategien mit einer eigenen, getunten
    Korrelationslaenge (siehe `STRATEGIES`) geben sie explizit mit.
    """
    if condition == 'ground_truth':
        maske = muster_maske('alles', GP_RES, device=device)
        return MaskiertesWissen(maske, truth, sigma_bekannt=0.0,
                                grid_res=GP_RES, noise=gp_noise,
                                lengthscale=gp_lengthscale, device=device)
    if condition == 'none_known':
        return GPBelief(grid_res=GP_RES, noise=gp_noise,
                        lengthscale=gp_lengthscale, device=device)
    if condition == 'half_known':
        maske = muster_maske('haelfte', GP_RES, device=device)
        return MaskiertesWissen(maske, truth, sigma_bekannt=0.0,
                                grid_res=GP_RES, noise=gp_noise,
                                lengthscale=gp_lengthscale, device=device)
    if condition == 'ten_samples':
        b = GPBelief(grid_res=GP_RES, noise=gp_noise,
                     lengthscale=gp_lengthscale, device=device)
        g = torch.Generator(device='cpu').manual_seed(seed * 1013 + 7)
        pts = torch.rand(10, 2, generator=g).to(device)
        p, v = measure(pts, truth, noise_std=gp_noise)
        b.observe(p, v)
        return b
    raise KeyError(condition)


def needs_belief_update(condition):
    """Ground-Truth-Bedingung: Glaube ist schon exakt, Updates aendern nichts."""
    return condition != 'ground_truth'


def phi_from_belief(belief, norm='max'):
    mu, sd = belief.posterior_grid()
    return ucb_density(mu, sd, kappa=KAPPA, norm=norm)


# ── SVGD-Verfeinerung, einheitlich ──────────────────────────────────────────

def svgd(refiner, curve, phi, n_iters, start=None, nxi=None):
    curve_np = curve.detach().cpu().numpy().astype(np.float64)
    phi_np = phi.detach().cpu().numpy().astype(np.float64)
    start_np = None if start is None else start.detach().cpu().numpy()
    out = refiner.refine(curve_np, phi_np, n_iters, nxi=nxi or NXI, start=start_np)
    return torch.as_tensor(out, dtype=torch.float32).clamp(0.0, 1.0)


def _observe_segment(belief, seg, truth, condition, noise=0.05,
                     sensor_radius=0.06, max_obs=64):
    if not needs_belief_update(condition):
        return
    pts, vals = measure(seg, truth, noise_std=noise, sensor_radius=sensor_radius)
    belief.observe(*thin(pts, vals, max_points=max_obs))


# ── CFM: drei Replanning-Schemata ───────────────────────────────────────────

def cfm_no_replan(planner, belief, svgd_iters, refiner):
    """Variante A-artig: einmal planen, keine Nachfuehrung."""
    phi = phi_from_belief(belief, norm='max')
    parts = acb.phi_particles(phi, N_PARTICLES, device=belief.device)
    cps = planner.plan(parts, n_candidates=1)
    curve = planner.render(cps)[0]
    return svgd(refiner, curve, phi_from_belief(belief, norm='sum'), svgd_iters)


def cfm_one_replan(planner, belief, truth, condition, svgd_iters, refiner, seed=0):
    """Genau ein Nachplanungsschritt: erste Haelfte fahren, dann neu planen."""
    torch.manual_seed(seed)
    phi1 = phi_from_belief(belief, norm='max')
    parts1 = acb.phi_particles(phi1, N_PARTICLES, device=belief.device)
    cps1 = planner.plan(parts1, n_candidates=1)
    curve1 = planner.render(cps1)[0]
    curve1 = svgd(refiner, curve1, phi_from_belief(belief, norm='sum'), svgd_iters)

    half = trim_to_length(curve1, max(_pl(curve1) / 2.0, 1e-6))
    _observe_segment(belief, half, truth, condition)

    phi2 = phi_from_belief(belief, norm='max')
    parts2 = acb.phi_particles(phi2, N_PARTICLES, device=belief.device)
    start = half[-1]
    cps2 = planner.plan(parts2, n_candidates=1, start=start)
    curve2 = planner.render(cps2)[0]
    curve2 = svgd(refiner, curve2, phi_from_belief(belief, norm='sum'),
                 svgd_iters, start=start)
    return torch.cat([half, curve2], dim=0)


def cfm_replan_1_6(planner, belief, truth, condition, svgd_iters, refiner):
    """Variante D-artig: genau eine Laengeneinheit fahren, neu planen, 6x."""
    driven = None
    for _ in range(N_REPLAN_ROUNDS):
        phi_max = phi_from_belief(belief, norm='max')
        parts = acb.phi_particles(phi_max, N_PARTICLES, device=belief.device)
        start = None if driven is None else driven[-1]
        cps = planner.plan(parts, n_candidates=1, start=start)
        curve = planner.render(cps)[0]
        curve = svgd(refiner, curve, phi_from_belief(belief, norm='sum'),
                    svgd_iters, start=start)

        seg = trim_to_length(curve, LENGTH_UNIT)
        n_pts = max(8, int(round(72 * _pl(seg) / LENGTH_UNIT)))
        seg = resample_arclength(seg, n_pts)

        _observe_segment(belief, seg, truth, condition)
        driven = seg if driven is None else torch.cat([driven, seg], dim=0)
    return driven


# ── Linear / heuristisch: einmal initialisieren, SVGD-Sweep ────────────────

def linear_variant(kind, belief, svgd_iters, refiner):
    raw = diagonal_path(PTS_RENDER) if kind == 'diagonal' else straight_top_path(PTS_RENDER)
    phi = phi_from_belief(belief, norm='sum')
    return svgd(refiner, raw, phi, svgd_iters)


def heuristic_variant(belief, svgd_iters, refiner, n_peaks=12):
    phi = phi_from_belief(belief, norm='sum')
    raw = heuristic_peak_path(phi, n_points=PTS_RENDER, n_peaks=n_peaks)
    return svgd(refiner, raw, phi, svgd_iters)


# ── Dichte-unabhaengige Baselines: einmal je Form ───────────────────────────

def lawnmower_variant():
    return lawnmower_path(n_points=PTS_RENDER, target_length=TARGET_LENGTH)


def random_walk_variant(seed=0):
    return random_walk_path(n_points=PTS_RENDER, seed=seed)


# ── Registrierung: was existiert, was braucht die Wissensstufe ─────────────
#: (method, subvariant) -> ob eine Trajektorie pro Wissensstufe neu erzeugt
#: werden muss (True) oder einmal je Form reicht (False, Maeander/Irrfahrt).
KNOWLEDGE_DEPENDENT = {
    ('cfm', 'no_replan'): True, ('cfm', 'one_replan'): True,
    ('cfm', 'replan_1_6'): True,
    ('linear', 'diagonal'): True, ('linear', 'straight_top'): True,
    ('heuristic', 'peaks'): True,
    ('lawnmower', 'fixed'): False, ('random_walk', 'fixed'): False,
}


# =============================================================================
# Follow-up (2026-09-12): the project's own TUNED acquisition settings,
# instead of the hand-picked `ucb`/kappa=2.0 above, plus the coverage-debt
# mechanism the first run's replanning loop was missing.
#
# Values below are the project's own published/tuned points, not new picks:
#   - niveau (lse) tau=0.4307/svgd=0 and tau=0.6067/svgd=25:
#     `exploration_optimierung/results/bestwerte_beide.json`, also hardcoded
#     as `OPTIMAL_POLICY` in `exploration/interactive_sim.py:96-101`.
#   - ucb kappa=0.4597, mass w=0.53, eid kappa=2.4662: per-model tuned values
#     from `exploration_optimierung/greedy_per_round.py`'s `FIXED_PARAM`
#     (line 56), which also names `niveau` "der veroeffentlichte Gewinner".
# `svgd_iters` for ucb/mass/eid isn't separately published per model, so each
# is tested at both of niveau's two published operating points (0 and 25) —
# consistent treatment rather than an invented number.
# =============================================================================

STRATEGIES = {
    'niveau_svgd0':      dict(phi_model='niveau', param=0.4307, svgd_iters=0),
    'niveau_svgd25':     dict(phi_model='niveau', param=0.6067, svgd_iters=25),
    'ucb_tuned_svgd0':   dict(phi_model='ucb',    param=0.4597, svgd_iters=0),
    'ucb_tuned_svgd25':  dict(phi_model='ucb',    param=0.4597, svgd_iters=25),
    'mass_tuned_svgd0':  dict(phi_model='mass',   param=0.53,   svgd_iters=0),
    'mass_tuned_svgd25': dict(phi_model='mass',   param=0.53,   svgd_iters=25),
    'eid_tuned_svgd0':   dict(phi_model='eid',    param=2.4662, svgd_iters=0),
    'eid_tuned_svgd25':  dict(phi_model='eid',    param=2.4662, svgd_iters=25),

    # ==========================================================================
    # Follow-up (2026-09-16): Optuna-Studie `ideal_v2` (`exploration_optimierung/
    # optuna_search.py --space ideal`, 950 Versuche, TPE+Hyperband). Bester
    # Versuch #883, J=0.2513 (vs. J=0.2581 fuer den obigen `eid_tuned`-Punkt,
    # siehe `exploration_optimierung/results/optuna/best.json`). Anders als die
    # Punkte oben deckt dieser Fund neun statt drei Groessen ab — die uebrigen
    # sechs waren in `build_strategy_args`/`build_belief` bislang projektweit
    # fest verdrahtet (debt_weight=0.6, visit_sat=1.0, visit_halflife=3.0,
    # phi_mode='uniform', gp_noise=0.05, gp_lengthscale=0.08, n_particles=256,
    # cfg_weight=2.0) und werden fuer diesen Eintrag durch die getunten Werte
    # ersetzt; die acht Eintraege oben lesen dieselben Felder weiterhin ueber
    # `.get(key, alter_wert)` und bleiben dadurch unveraendert.
    # ==========================================================================
    'eid_optuna_ideal_v2': dict(
        phi_model='eid', param=0.6080837837539514, svgd_iters=0,
        debt_weight=0.9002740182042122, visit_sat=0.4696706948278474,
        visit_halflife=2.2153603772076282, phi_mode='quantile',
        phi_quantile=0.14978348922195622, gp_noise=0.1249491528099695,
        gp_lengthscale=0.12192310997439061, n_particles=128,
        cfg_weight=1.5987308781507361),
}


def build_strategy_args(strategy_name, device):
    """`mission.build_mission_args`, keyed by the project's own tuned point.

    Reused verbatim rather than hand-built: it already translates each
    model's public parameter (kappa/w/tau) into what `zieldichte`/
    `debt_density` expect. Every field beyond `phi_model`/`param`/`svgd_iters`
    is read from the strategy dict with `.get(key, alter_wert)`, where
    `alter_wert` is the value every strategy used before per-strategy tuning
    existed (`debt_weight=0.6, visit_sat=1.0, visit_halflife=3.0,
    phi_mode='uniform', gp_noise=0.05, n_particles=N_PARTICLES, cfg_weight=2.0`)
    — strategies that don't set these keys behave exactly as before.

    Returns `(args, svgd_iters, cfg_weight)`; `cfg_weight` is the planner's
    guidance strength and isn't part of `build_mission_args` (that's a
    `LaengenMission`/`CfmPlanner` setting), so it's threaded back separately
    for the caller to assign to `planner.cfg_weight`.
    """
    from exploration_optimierung.mission import build_mission_args
    s = STRATEGIES[strategy_name]
    args = build_mission_args(
        str(device), phi_model=s['phi_model'], param=s['param'],
        debt_weight=s.get('debt_weight', 0.6), visit_sat=s.get('visit_sat', 1.0),
        sensor_radius=0.06, gp_noise=s.get('gp_noise', 0.05),
        n_particles=s.get('n_particles', N_PARTICLES), meas_noise=0.05, max_obs=64,
        visit_halflife=s.get('visit_halflife', 3.0),
        phi_mode=s.get('phi_mode', 'uniform'))
    args.phi_quantile = s.get('phi_quantile', 0.5)
    return args, s['svgd_iters'], s.get('cfg_weight', 2.0)


def _visit_field(driven, args, device):
    if driven is None:
        return None
    from exploration_optimierung.mission import visitation_recent, LENGTH_UNIT as _LU
    return visitation_recent(driven, GP_RES, args.visit_bandwidth, str(device),
                             half_life=args.visit_halflife * _LU)


# ── CFM unter der abgestimmten Strategie ────────────────────────────────────

def cfm_ideal_no_replan(planner, belief, strategy_name, refiner):
    """Wie `cfm_no_replan`, aber Phi kommt aus der abgestimmten Strategie
    statt aus festem `ucb`/kappa=2.0. Kein Debt-Term noetig — eine einzelne
    Planung hat nichts zu vergessen."""
    args, svgd_iters, cfg_weight = build_strategy_args(strategy_name, belief.device)
    planner.cfg_weight = cfg_weight
    mu, sd = belief.posterior_grid()
    phi = acb.zieldichte(mu, sd, args.kappa, args)
    parts = acb.phi_particles(phi, args.n_particles, mode=args.phi_mode,
                              quantile=args.phi_quantile, device=belief.device)
    cps = planner.plan(parts, n_candidates=1)
    curve = planner.render(cps)[0]
    return svgd(refiner, curve, phi, svgd_iters)


def cfm_ideal_replan_1_6(planner, belief, truth, condition, strategy_name, refiner):
    """Wie `cfm_replan_1_6`, aber mit `debt_density` statt blossem `ucb_density`
    — Fix fuer Ursache #2 der Lawnmower-Auswertung: die Zieldichte vergisst
    jetzt, was gerade abgefahren wurde, statt jede Runde bei Null anzufangen."""
    args, svgd_iters, cfg_weight = build_strategy_args(strategy_name, belief.device)
    planner.cfg_weight = cfg_weight
    driven = None
    for _ in range(N_REPLAN_ROUNDS):
        mu, sd = belief.posterior_grid()
        visit = _visit_field(driven, args, belief.device)
        phi, _v = acb.debt_density(mu, sd, visit, args.kappa, args)
        parts = acb.phi_particles(phi, args.n_particles, mode=args.phi_mode,
                                  quantile=args.phi_quantile, device=belief.device)
        start = None if driven is None else driven[-1]
        cps = planner.plan(parts, n_candidates=1, start=start)
        curve = planner.render(cps)[0]
        curve = svgd(refiner, curve, phi, svgd_iters, start=start)

        seg = trim_to_length(curve, LENGTH_UNIT)
        n_pts = max(8, int(round(72 * _pl(seg) / LENGTH_UNIT)))
        seg = resample_arclength(seg, n_pts)

        _observe_segment(belief, seg, truth, condition, noise=args.noise,
                         sensor_radius=args.sensor_radius, max_obs=args.max_obs)
        driven = seg if driven is None else torch.cat([driven, seg], dim=0)
    return driven


# ── Spektral-Repraesentation (neuer Wrapper, siehe spectral_planner.py) ────
#: Achtung: `checkpoints/cond_spectral_crossattn_ep900.pt` wurde auf einem
#: KLEINEREN, andersartigen Formen-Satz trainiert (`sigma, rand_poly_7, G,
#: spiral_2cw, star_5, W, lissajous_1_3, heart, 5, phi`), nicht auf
#: `shape_library.VALIDATION_SHAPES`. Ergebnisse hier pruefen Generalisierung
#: ueber den Datensatz hinweg, nicht nur den Repraesentationsvergleich an
#: sich — bei der Auswertung offen mitfuehren, nicht verschweigen.

def spectral_ideal_no_replan(planner, belief, strategy_name, refiner):
    args, svgd_iters, cfg_weight = build_strategy_args(strategy_name, belief.device)
    planner.cfg_weight = cfg_weight
    mu, sd = belief.posterior_grid()
    phi = acb.zieldichte(mu, sd, args.kappa, args)
    cps = planner.plan(phi, n_candidates=1)
    curve = planner.render(cps)[0]
    return svgd(refiner, curve, phi, svgd_iters, nxi=planner.nxi)


def spectral_ideal_replan_1_6(planner, belief, truth, condition, strategy_name, refiner):
    args, svgd_iters, cfg_weight = build_strategy_args(strategy_name, belief.device)
    planner.cfg_weight = cfg_weight
    driven = None
    for _ in range(N_REPLAN_ROUNDS):
        mu, sd = belief.posterior_grid()
        visit = _visit_field(driven, args, belief.device)
        phi, _v = acb.debt_density(mu, sd, visit, args.kappa, args)
        start = None if driven is None else driven[-1]
        cps = planner.plan(phi, n_candidates=1, start=start)
        curve = planner.render(cps)[0]
        curve = svgd(refiner, curve, phi, svgd_iters, start=start, nxi=planner.nxi)

        seg = trim_to_length(curve, LENGTH_UNIT)
        n_pts = max(8, int(round(72 * _pl(seg) / LENGTH_UNIT)))
        seg = resample_arclength(seg, n_pts)

        _observe_segment(belief, seg, truth, condition, noise=args.noise,
                         sensor_radius=args.sensor_radius, max_obs=args.max_obs)
        driven = seg if driven is None else torch.cat([driven, seg], dim=0)
    return driven


#: representation -> (module, attribute) to build its planner, and its
#: (no_replan, replan_1_6) generator functions. `run_ideal_matrix.py` reads
#: this instead of hardcoding the two representations inline.
REPRESENTATIONS = {
    'particles': dict(no_replan=cfm_ideal_no_replan, replan_1_6=cfm_ideal_replan_1_6),
    'spectral': dict(no_replan=spectral_ideal_no_replan, replan_1_6=spectral_ideal_replan_1_6),
}
