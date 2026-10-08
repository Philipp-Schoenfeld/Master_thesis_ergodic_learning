r"""
rq_experiments/common.py
=========================
Geteilte Bausteine fuer RQ1 (Laufzeit) und RQ2 (Robustheit bei festem
Budget): Sun-Trials (Suns eigenes Randomisierungsprotokoll aus Benchmark Q1),
ein zeitgestoppelter Sun-Solver (identischer Rechenkern wie
`ergodic_solver.py`, nur mit Wall-Clock-Messung je Checkpoint) und die
CFM-Varianten (roh, als Warm-Start fuer Sun, mit `SvgdRefiner`).

Nichts hier aendert `ergodic_solver.py` oder `svgd_refine.py` -- es importiert
nur deren Bausteine, damit Messung und Datengenerierung denselben Kern
benutzen (siehe `sun_refine.py`s Docstring fuer dieselbe Begruendung).
"""

import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
_arch = os.path.dirname(_mat)
_root = os.path.dirname(_arch)
for _p in (_mat, os.path.join(_arch, 'exploration'), _arch,
          os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                    # noqa: E402
import jax.numpy as jnp                       # noqa: E402

import ergodic_solver as es                   # noqa: E402
from common.sun_refine import run_batch as sun_refine_batch  # noqa: E402
from sinkhorn_jax import sinkhorn_divergence   # noqa: E402

# ── Suns Randomisierungsprotokoll (Benchmark Q1: "trimodal GMM + Startpunkt
#    zufaellig in jedem Trial") ───────────────────────────────────────────────

#: Kovarianz-"Archetypen" aus `stein_flow_coverage.ipynb` (schmal-hoch,
#: diagonal, schmal-hoch) -- Trials variieren nur Lage/Gewicht/Rotation davon,
#: damit die Komponenten weiterhin wie bei Sun aussehen (keine entarteten
#: Gauss-Keulen).
_BASE_COVS = [
    np.array([[0.002, 0.0], [0.0, 0.04]]),
    np.array([[0.02, -0.018], [-0.018, 0.02]]),
    np.array([[0.002, 0.0], [0.0, 0.04]]),
]


def _rotate(cov, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return R @ cov @ R.T


def random_trimodal_gmm(rng):
    """Ein zufaelliges trimodales GMM nach Suns Protokoll -> shape_def dict
    (kompatibel mit `shape_library.make_pdf_and_score` und
    `ergodic_solver._generate_initial_trajectory`'s GMM-Zweig)."""
    means = rng.uniform(0.2, 0.8, size=(3, 2))
    covs = [np.clip(_rotate(c, rng.uniform(0, np.pi)), -0.05, 0.05)
           for c in _BASE_COVS]
    # Diagonale numerisch positiv-definit halten (Rotation einer validen Kovarianz
    # bleibt es analytisch; der Clip oben ist nur eine Sicherheitsgrenze gegen
    # extreme Drehwinkel-Rundungsfehler).
    w = rng.uniform(0.2, 1.0, size=3)
    w /= w.sum()
    return {'means': means.tolist(), 'covs': [c.tolist() for c in covs],
           'weights': w.tolist()}


def random_x0(rng, margin=0.1):
    return tuple(rng.uniform(margin, 1.0 - margin, size=2))


def make_trial(seed):
    """Ein Trial wie bei Sun: (shape_def, x0, pdf_fn, score_fn, phi_grid)."""
    from shape_library import make_pdf_and_score
    rng = np.random.default_rng(seed)
    shape_def = random_trimodal_gmm(rng)
    x0 = random_x0(rng)
    pdf_fn, score_fn = make_pdf_and_score(shape_def)
    return dict(seed=seed, shape_def=shape_def, x0=x0, pdf_fn=pdf_fn, score_fn=score_fn)


def density_grid(pdf_fn, res=96):
    """phi-Gitter (res,res) in [0,1]^2, Werte in [0, max]. Zeile = y, wie in
    `SvgdRefiner._phi_k` / `sun_refine._bilinear`."""
    xs = np.linspace(0, 1, res)
    X, Y = np.meshgrid(xs, xs)
    pts = jnp.stack([jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())], axis=-1)
    phi = np.asarray(jax.vmap(lambda p: pdf_fn(jnp.array([p[0], p[1]])))(pts)).reshape(res, res)
    return phi


# ── Ergodik-Metrik: EINE feste Referenz fuer alle Sun-Vergleiche (siehe
#    Thesis-Doc, Abschnitt "Forschungsfragen", Fussnote zu Metrik-Mismatch).
#    K=8 2D-Fourier-Modi, Dichte normiert auf Summe 1 ueber dem Gitter --
#    identisch zur Eval-Matrix (`SvgdRefiner.K`), damit RQ1/2 und die
#    bestehende Eval-Matrix dieselbe Zahl berichten. ──────────────────────────

K_FREQ = 8
_k1, _k2 = np.meshgrid(np.arange(K_FREQ), np.arange(K_FREQ), indexing='ij')
K_IDX = np.stack([_k1.ravel(), _k2.ravel()], axis=-1).astype(np.float64)   # (64, 2)
LAMBDA_K = (1.0 + (K_IDX ** 2).sum(axis=1)) ** (-1.5)


def target_coeffs(phi_grid):
    """phi (R,R), Zeile = y -> (64,) Fourier-Zielkoeffizienten (Summe 1)."""
    R = phi_grid.shape[0]
    xs = np.linspace(0, 1, R)
    X, Y = np.meshgrid(xs, xs)
    w = np.clip(phi_grid, 0.0, None).ravel()
    w = w / max(w.sum(), 1e-12)
    F = np.cos(np.pi * K_IDX[None, :, 0] * X.ravel()[:, None]) \
        * np.cos(np.pi * K_IDX[None, :, 1] * Y.ravel()[:, None])
    return (F * w[:, None]).sum(axis=0)


def ergodic_error(traj_xy, phi_k_target):
    """(T,2)-Bahn vs. Zielkoeffizienten -> skalarer ergodischer Fehler
    0.5 * sum Lambda_k (c_k - phi_k)^2 (dieselbe Formel wie
    `ergodic_core.py` / `ergodic_energy_torch.py`)."""
    tr = np.asarray(traj_xy)
    F = (np.cos(np.pi * K_IDX[None, :, 0] * tr[:, 0:1])
        * np.cos(np.pi * K_IDX[None, :, 1] * tr[:, 1:2]))
    c_k = F.mean(axis=0)
    return float(0.5 * np.sum(LAMBDA_K * (c_k - phi_k_target) ** 2))


# ── Zeitgestoppelter Sun-Solver: derselbe Kern wie ergodic_solver.py,
#    Wall-Clock-Messung an gegebenen Checkpoints ──────────────────────────────

def _sync(x):
    jax.block_until_ready(x)
    return x


def timed_solve(pm, linearize_dyn, solve_lqr, score_fn, x0j, u_traj0, checkpoints,
                step_size=0.01, h=0.01, score_scale=1.0, warmup=True):
    """Dynamik-agnostische Fassung von Suns FM-Stein-Schleife: `pm` ist eine
    beliebige `lqrax.LQR`-Unterklasse mit `.positions(x_traj) -> (T,2)`
    (siehe `dynamics_zoo.py`), `linearize_dyn`/`solve_lqr` deren
    JIT-kompilierte `linearize_dyn`/`solve`-Methoden (CPU, wie bei Sun). Der
    Stein-Kernel selbst (`es._make_stein_grad`) braucht nur die ersten zwei
    Zustandskomponenten (x, y) und ist damit unveraendert fuer jede Dynamik
    nutzbar.

    Stoppt die Wall-Clock-Zeit an jedem Eintrag in `checkpoints` (aufsteigend
    sortierte Iterationszahlen, `0` erlaubt = Startzustand ohne Iteration).
    `warmup=True`: ein Dummy-Durchlauf VOR der Zeitmessung loest die
    JIT-Kompilierung aus (nicht Teil der gemessenen Zeit, wie ueberall sonst
    in diesem Projekt und bei Sun selbst).

    -> Liste von dicts {iters, time_s, traj_xy (T+1,2)}, eine je Checkpoint.
    """
    stein_grad_jit = es._make_stein_grad(score_fn, score_scale)
    return _timed_loop(pm, linearize_dyn, solve_lqr, lambda x_traj: stein_grad_jit(x_traj, h=h),
                       x0j, u_traj0, checkpoints, step_size=step_size, warmup=warmup)


def _timed_loop(pm, linearize_dyn, solve_lqr, ref_flow_fn, x0j, u_traj0, checkpoints,
                step_size=0.01, warmup=True):
    """The flow-matching loop body shared by `timed_solve` (Stein reference
    flow) and `timed_sinkhorn_solve` (Sinkhorn reference flow, see below):
    only `ref_flow_fn(x_traj) -> h_tilde` differs between the two reference
    flows Sun et al. define; the LQ flow-matching step itself
    (`u += step_size * v`, `v` from the Riccati solve) does not. Factored out
    2026-10 when RQ4 added the Sinkhorn flow, so the two reference flows
    cannot drift apart in how they drive the shared LQR step. New code
    (English, per the project's code-language convention), unlike the two
    functions that call it."""
    z0 = jnp.zeros(pm.x_dim)
    targets = sorted(set(int(c) for c in checkpoints))

    def step(u):
        x_traj, A, B = linearize_dyn(x0j, u)
        dx = ref_flow_fn(x_traj)
        v, _ = solve_lqr(z0, A, B, dx)
        return u + step_size * v

    if warmup and max(targets) > 0:
        _sync(step(u_traj0))

    out = []
    u = u_traj0
    done_idx = 0
    if targets[0] == 0:
        out.append(dict(iters=0, time_s=0.0, traj_xy=pm.positions(pm.traj_sim(x0j, u))))
        done_idx = 1
    if done_idx >= len(targets):
        return out

    t0 = time.perf_counter()
    for it in range(1, targets[-1] + 1):
        u = step(u)
        if it == targets[done_idx]:
            _sync(u)
            elapsed = time.perf_counter() - t0
            out.append(dict(iters=it, time_s=elapsed, traj_xy=pm.positions(pm.traj_sim(x0j, u))))
            done_idx += 1
            if done_idx >= len(targets):
                break
    return out


def timed_sun_solve(score_fn, x0j, u_traj0, checkpoints, dt=0.05, step_size=0.01,
                    h=0.01, score_scale=1.0, warmup=True):
    """`timed_solve` mit Suns eigener Point-Mass-2.-Ordnung-Dynamik
    (`ergodic_solver.PointMassLQR` ueber `es._build_lqr`) -- die Variante, die
    RQ1/RQ2 benutzen."""
    pm, linearize_dyn, solve_lqr = es._build_lqr(dt)
    pm.positions = lambda x_traj: np.array(x_traj)[:, :2]
    return timed_solve(pm, linearize_dyn, solve_lqr, score_fn, x0j, u_traj0, checkpoints,
                       step_size=step_size, h=h, score_scale=score_scale, warmup=warmup)


def sun_cold_x0_u0(x0, dt, tsteps):
    """Suns eigene Initialisierung: u=0, Anfangsgeschwindigkeit Richtung
    Zentrum (exakt wie `stein_flow_coverage.ipynb` / der Default-Zweig von
    `run_ergodic_coverage`)."""
    T = dt * tsteps
    x0j = jnp.array([x0[0], x0[1], 2.0 * (0.5 - x0[0]) / T, 2.0 * (0.5 - x0[1]) / T])
    return x0j, jnp.zeros((tsteps, 2))


def heuristic_x0_u0(x0, shape_def, dt, tsteps):
    """Unsere Heuristik-Initialisierung (TSP + Lissajous + PID-Tracking),
    identisch zum Generator-Pfad fuer GMM-`shape_def`s."""
    p_traj, u_traj_np, v0 = es._generate_initial_trajectory(x0, shape_def, tsteps, dt)
    x0j = jnp.array([x0[0], x0[1], v0[0], v0[1]])
    return x0j, jnp.array(u_traj_np)


def cfm_x0_u0(curve_xy, start, dt, tsteps):
    """CFM-Netzbahn (beliebig viele Punkte) -> (x0, u0) ueber dasselbe
    PID-Tracking wie `custom_p_traj` im Generator, damit "CFM als Warm-Start"
    exakt denselben Pfad nimmt wie bei der Datengenerierung."""
    curve = np.asarray(curve_xy, dtype=np.float64)
    idx_eval = np.linspace(0, len(curve) - 1, tsteps + 1)
    idx = np.arange(len(curve))
    p_traj = np.column_stack([np.interp(idx_eval, idx, curve[:, 0]),
                             np.interp(idx_eval, idx, curve[:, 1])])
    p_traj[0] = np.asarray(start)
    u_traj_np, v0 = es._pid_track(p_traj, dt, tsteps)
    x0j = jnp.array([start[0], start[1], v0[0], v0[1]])
    return x0j, jnp.array(u_traj_np)


# ── CFM: roh und als SvgdRefiner-Eingabe ──────────────────────────────────────

def _cfm_plan(planner, parts, start, device):
    """Shared tail of `cfm_raw_curve`/`cfm_raw_curve_from_particles`: one
    `plan`+`render` call, timed. `parts`: (N,3) torch tensor (x, y, weight),
    already on `device`."""
    import torch
    t0 = time.perf_counter()
    kw = {}
    if start is not None and getattr(planner, 'start_cond', False):
        kw['start'] = torch.as_tensor(start, dtype=torch.float32, device=device)
    cps = planner.plan(parts, n_candidates=1, **kw)
    curve = planner.render(cps)[0]
    if device != 'cpu':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return curve.detach().cpu().numpy().astype(np.float64), elapsed


def cfm_raw_curve(planner, phi_grid, start=None, device='cpu'):
    """Ein CFM-Vorwaertsdurchlauf -> ((T,2) Bahn in numpy, Sekunden).
    `phi_grid`: (R,R)-Dichte (dient als Partikelquelle, siehe
    `apply_cfm_belief.phi_particles`)."""
    import torch
    import apply_cfm_belief as acb
    phi_t = torch.as_tensor(np.asarray(phi_grid), dtype=torch.float32, device=device)
    parts = acb.phi_particles(phi_t, getattr(planner, 'n_particles', 256), device=device)
    return _cfm_plan(planner, parts, start, device)


def cfm_raw_curve_from_particles(planner, particles_xy, start=None, device='cpu', weight=1.0):
    """RQ4 (added 2026-10): condition CFM directly on a SAMPLE set -- no
    density grid at all, unlike `cfm_raw_curve`. This is the whole point of
    comparing against it: Sun et al.'s own stated limitation of the Stein
    flow is that it needs a closed-form score, unavailable when the target
    is only samples (an icon's point cloud); CFM's particle conditioning
    (`(x, y, weight)` triples, see `common.acquisition.particles_from_density`)
    never needed a density in the first place.

    particles_xy: (N, 2) numpy array in [0,1]^2 (e.g. `load_icon`'s output).
    `weight`: uniform weight per particle (matches an unweighted point
    cloud; `particles_from_density(mode='uniform')` does the same for a
    density's support). -> ((T,2) curve, seconds), same as `cfm_raw_curve`."""
    import torch
    n = len(particles_xy)
    w = np.full((n, 1), weight, dtype=np.float32)
    parts = torch.as_tensor(np.concatenate([np.asarray(particles_xy, dtype=np.float32), w], axis=1),
                            dtype=torch.float32, device=device)
    return _cfm_plan(planner, parts, start, device)


def cfm_svgd_timed(curve_xy, phi_grid, start, checkpoints, nxi=25):
    """CFM-Bahn mit `SvgdRefiner`s Standardverfahren verfeinert -- seit
    2026-10-06 Suns FM-Stein-Loeser auf dem Dichtegitter (`backend='sun'`),
    hier direkt ueber `sun_refine.run_batch` aufgerufen. Wall-Clock an jedem
    Checkpoint; jeder Checkpoint ist ein EIGENER Durchlauf von 0 (derselben
    Startbahn) bis `k` (kein Fortsetzen eines vorherigen Laufs), misst also
    "wie lange braucht ein frischer Aufruf mit Budget k" -- das ist die
    Groesse, die ein Nutzer bzw. Planer tatsaechlich aufwendet. `run_batch`
    mit `record=False` nutzt den undynamischen `fori_loop`-Zweig
    (`sun_refine._run_batch_plain`), kompiliert also nur einmal unabhaengig
    von `n_iters`. -> Liste wie `timed_sun_solve`."""
    curve = np.asarray(curve_xy, dtype=np.float64)[None]
    phi = np.asarray(phi_grid, dtype=np.float64)
    st = None if start is None else np.asarray(start, dtype=np.float64)[None]
    targets = sorted(set(int(c) for c in checkpoints))
    if not targets:
        return []
    # Warmup (JIT) auf dem kleinsten Budget, nicht mitgezaehlt.
    sun_refine_batch(curve, phi, st, targets[0], nxi, record=False)
    out = []
    t0 = time.perf_counter()
    for k in targets:
        res = sun_refine_batch(curve, phi, st, k, nxi, record=False)
        elapsed = time.perf_counter() - t0
        out.append(dict(iters=k, time_s=elapsed, traj_xy=res['final_pos'][0]))
    return out


# =============================================================================
# RQ4 -- FM-Sinkhorn (non-smooth, sample-represented targets). Added 2026-10;
# everything above this point predates RQ4 and is unchanged (see the module
# docstring and `evaluation_full_matrix/rq_experiments/README.md`). New code
# from here on follows the project's English-only convention for anything
# newly written, even though the rest of this file is German.
# =============================================================================

def timed_sinkhorn_solve(target_samples, x0j, u_traj0, checkpoints, dt=0.05,
                         step_size=0.005, epsilon=0.01, n_sinkhorn_iters=50,
                         cost_scale=1e3, warmup=True):
    """Sun's flow-matching loop (`timed_solve`'s dynamics-agnostic form,
    specialised here to `ergodic_solver.PointMassLQR`) with the reference
    flow h_tilde replaced by the Sinkhorn-divergence gradient
    (`sinkhorn_jax.sinkhorn_gradient`) instead of the Stein-variational
    gradient -- Sun et al.'s second reference flow, matching
    `sinkhorn_flow_coverage.ipynb`: same LQ flow-matching machinery
    (`es._build_lqr`), only the reference flow computation differs. Needed
    because the target here is a SAMPLE set (an icon's point cloud, or this
    project's own particle-conditioned shapes), not a density with a
    closed-form score -- Sun's own stated limitation of the Stein flow (no
    closed-form score from samples; score matching "cannot be conducted in
    real time" and is "inaccurate in regions with low sample density").

    `step_size=0.005`, `cost_scale=1e3`: Sun's own values
    (`sinkhorn_flow_coverage.ipynb`: `step_size = 0.005`,
    `return sinkhorn_cost * 1e3` inside `sinkhorn_div`). The scale matters a
    lot here, unlike for the Stein flow: measured on this project's own GMM
    trials, the raw Sinkhorn-divergence gradient peaks around 0.005 against
    the Stein score's ~14 for the same trajectory, so without `cost_scale`
    the LQ flow-matching step barely moves the trajectory at Sun's
    `step_size`, however many iterations are spent (verified: 300 iterations
    at `cost_scale=1` moved the coverage error by under 10%; restoring
    `cost_scale=1e3` fixes this).

    target_samples: (M, 2) target point cloud, already in [0,1]^2.
    -> same checkpoint-list contract as `timed_solve`/`timed_sun_solve`."""
    pm, linearize_dyn, solve_lqr = es._build_lqr(dt)
    pm.positions = lambda x_traj: np.array(x_traj)[:, :2]
    target_j = jnp.asarray(target_samples, dtype=jnp.float32)

    # x_traj is the full (T, 4) point-mass state; the Sinkhorn cost only ever
    # looks at the (T, 2) position slice (matching the Stein kernel's own
    # `x[:2]` convention, see `ergodic_solver._kernel`), so grad w.r.t. the
    # full state naturally comes out zero on the velocity columns rather than
    # raising a shape mismatch against the (M, 2) target samples. JIT-compiled
    # (`es._make_stein_grad` does the same for the Stein flow): without it,
    # each of the up to `max(checkpoints)` outer iterations would retrace and
    # re-execute the unrolled Sinkhorn loop in eager mode, which is the
    # difference between single-digit seconds and several minutes here.
    ref_flow = jax.jit(lambda x_traj: -jax.grad(
        lambda xt: cost_scale * sinkhorn_divergence(
            xt[:, :2], target_j, epsilon, n_sinkhorn_iters))(x_traj))
    return _timed_loop(pm, linearize_dyn, solve_lqr, ref_flow, x0j, u_traj0, checkpoints,
                       step_size=step_size, warmup=warmup)


def load_icon(name, icons_dir=None, n_particles=256, seed=0):
    """One of Sun's Q3.A test icons (`tutorials/test_objects/2d/<name>.txt`
    in https://github.com/MurpheyLab/lqr-flow-matching, fetched once and
    checked into `rq_experiments/sun_icons/` -- the project website's own
    notebooks load these the same way, over the network each run).

    Subsampled to `n_particles` (deterministically, via a fixed seed): the
    raw files hold 900-3600 points, far more than this project's own
    particle-cloud convention (`N_PARTICLES=256` in `variant_runner.py`,
    also what `phi_particles` draws). Besides matching that convention, it
    keeps the Sinkhorn cost matrix small (`n_particles` x trajectory length)
    and JIT-compiles once regardless of which icon is loaded; pass
    `n_particles=None` for the full, unsampled point cloud.
    -> (n_particles, 2) numpy array in [0,1]^2 (or (N,2) if unsampled)."""
    icons_dir = icons_dir or os.path.join(_here, 'sun_icons')
    path = os.path.join(icons_dir, f'{name}.txt')
    pts = np.loadtxt(path)[:, :2]
    if n_particles is not None and len(pts) > n_particles:
        idx = np.random.default_rng(seed).choice(len(pts), size=n_particles, replace=False)
        pts = pts[idx]
    return pts


ICON_NAMES = ['star', 'heart', 'sword', 'eifeltower', 'trophy', 'fire',
             'scissors', 'lock', 'thunder', 'airplane']


def icon_density_grid(name, res=96, sigma=0.015, icons_dir=None):
    """An icon's point cloud rendered to a density grid (Gaussian-splatted
    points, normalised to max 1) -- for methods that need a GRID rather than
    samples (the Fourier coverage metric, CFM's particle-cloud conditioning
    via `phi_particles`, which itself resamples FROM a grid)."""
    pts = load_icon(name, icons_dir)
    xs = np.linspace(0, 1, res)
    X, Y = np.meshgrid(xs, xs)
    grid = np.zeros((res, res))
    for p in pts:
        grid += np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2) / (2 * sigma ** 2))
    return grid / max(grid.max(), 1e-12)


def coverage_error(traj_xy, target_samples_or_phik, k_idx=None, lambda_k=None):
    """Sun's Q3 "coverage error" IS the Fourier ergodic metric of Mathew and
    Mezic, "Metrics for ergodicity and design of ergodic dynamics for
    multi-agent systems", Physica D 240(4-5), 2011 -- Sun et al.'s own Q3.A/
    Q3.B description cites "the trajectory uniformity metric in Mathew and
    Mezic [31] (equation 4)", which is exactly the weighted sum-of-squared-
    Fourier-coefficient-difference metric already implemented as
    `ergodic_error`/`target_coeffs` above (K=8, confirmed against Mathew and
    Mezic's original formulation via a web search, 2026-10 -- see the
    thesis-doc conversation this module was built from). No separate "RQ4
    metric" is needed: this is a thin alias so RQ4's own scripts can name the
    metric they are actually using without re-deriving it.

    `target_samples_or_phik`: either an (M,2) point cloud (coefficients are
    computed from it directly, unweighted) or an already-computed (64,)
    Fourier coefficient vector (as `target_coeffs` returns for a density
    grid) -- accepts both so the same call works whether the target started
    as samples (icons) or a grid (this project's own shapes)."""
    arr = np.asarray(target_samples_or_phik)
    if arr.ndim == 2 and arr.shape[1] == 2:
        F = (np.cos(np.pi * K_IDX[None, :, 0] * arr[:, 0:1])
            * np.cos(np.pi * K_IDX[None, :, 1] * arr[:, 1:2]))
        phik = F.mean(axis=0)
    else:
        phik = arr
    return ergodic_error(traj_xy, phik)


# =============================================================================
# Cluster job support (added for the combined all-RQ job, 2026-10): a single
# `--time_budget_h` convention across all six `rqN_*.py` scripts, matching
# the pattern already used elsewhere in this project
# (`run_mission_eval.py`/`run_svgd_convergence.py`'s own `--time_budget_h`).
# Each script checks this before starting a new unit of work (a trial, a
# round) and stops submitting new ones once exceeded -- the already-written
# CSV rows stay valid and resumable, so a follow-up `sbatch` job (or the next
# step of a combined job script) picks up exactly where this one left off.
# =============================================================================

def time_budget_exceeded(t_start, budget_h):
    """`budget_h=None` -> never exceeded (no limit)."""
    return budget_h is not None and (time.time() - t_start) / 3600.0 > budget_h
