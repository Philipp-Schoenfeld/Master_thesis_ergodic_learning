r"""
sun_refine.py
=============
Nachverfeinerung einer geplanten Bahn mit Suns FM-Stein-Loeser (Sun, Pinosky,
Murphey, RSS 2025: Stein-Variational-Gradient-Fluss + LQ-Flow-Matching) --
DERSELBE Kern wie die Datengenerierung in
`ergodic_dataset_generator/ergodic_solver.py`: Kernel (`_kernel`), Dynamik
(`PointMassLQR`, gleiche Q/R), PID-Tracking der Startbahn (`_pid_track`),
dt = 0.05, 200 Schritte, Schrittweite 0.01, Bandbreite h = 0.01 und die
Update-Regel u <- u + eta * v. Diese Bausteine werden von dort importiert,
nicht kopiert, damit Datengenerierung und Verfeinerung nicht auseinanderlaufen.

Was gegenueber der Datengenerierung dazukommt, ist nur die Schnittstelle:

* Ziel ist ein Dichtegitter `phi` (R x R ueber [0,1]^2, Zeile = y, wie in
  `SvgdRefiner._phi_k`), keine analytische Dichte. Der Score ist der Gradient
  von log(phi_eff), bilinear interpoliert, mit
      phi_eff = phi + FLOOR_REL * max(phi) * blur(phi) / max(blur(phi)) + 1e-12 * max(phi),
  blur = Gauss mit BLUR_SIGMA (Anteil der Kantenlaenge). Suns GMM-Ziele haben
  ueberall Auslaeufer; ein Gitter mit Nullbereichen (Buchstaben, Wahrheit)
  haette dort keinen Gradienten, und eine Bahn ausserhalb des Traegers wuerde
  nicht angezogen. Der Gradient des Logarithmus haengt nicht von FLOOR_REL ab,
  der geglaettete Anteil zeigt also auch weit weg vom Traeger zur Masse,
  waehrend er auf dem Traeger gegen phi verschwindet.
* Ein weicher Rand (wie `W_BOUNDARY` im alten Refiner) und optional Hindernisse
  werden als Strafterme vom Log-Score abgezogen -- dieselbe Konstruktion wie im
  Hindernis-Experiment in `src/methods/Stein_Flow_matching/`. Ohne sie haette
  die Kernel-Abstossung in Regionen mit phi = 0 keinen Gegenspieler.
* Die Eingabebahn (beliebig viele Punkte) wird auf 201 Punkte gebracht und per
  PID-Tracking in eine Steuerfolge u0 uebersetzt (`custom_p_traj`-Weg des
  Generators); der Startpunkt ist der Anfangszustand und damit exakt erfuellt.
* Ausgabe wie beim alten Refiner: (T,2)-Bahn; mit `nxi` als B-Spline-Fit auf
  `nxi` Kontrollpunkte (dieselbe Basis wie `SvgdRefiner`), sonst linear auf T
  Punkte gebracht.

`n_iters` zaehlt FM-Stein-Iterationen (bei der Datengenerierung 600).
Einzel- und Stapelaufruf laufen durch dieselbe vmap-Funktion, ein Kandidat ist
also im Batch exakt so wie allein.
"""

import os
import sys
from functools import partial

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_expl = os.path.dirname(_here)
_arch = os.path.dirname(_expl)
for _p in (_arch, os.path.join(_arch, 'ergodic_dataset_generator')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                   # noqa: E402
import jax.numpy as jnp                      # noqa: E402

import ergodic_solver as es                  # noqa: E402


#: Loeser-Konstanten, identisch zu den Defaults von `generate_dataset.py`.
DT = 0.05
TSTEPS = 200
STEP_SIZE = 0.01
H = 0.01
#: Gewicht des geglaetteten Auslaeufers relativ zum Maximum von phi und
#: dessen Breite (Standardabweichung als Anteil der Kantenlaenge).
FLOOR_REL = 1e-3
BLUR_SIGMA = 0.15
#: Weicher Rand (Abstand zu den Kanten von [0,1]^2, wie BOUNDARY_MARGIN im
#: alten Refiner) und sein Gewicht im Log-Score.
BOUNDARY_MARGIN = 0.03
W_BOUNDARY = 1000.0
#: Hindernisgewicht im Log-Score fuer `obstacle_weight = 20` (der Default des
#: alten Refiners); andere Werte skalieren linear.
W_OBSTACLE = 2000.0
#: B-Spline-Basis wie im alten Refiner (`bspline_basis_matrix(nxi, T, 5)`).
DEGREE = 5

_PM = es.PointMassLQR(dt=DT)


# ── Ziel: Log-Dichte aus dem Gitter + Strafterme ──────────────────────────────

def _bilinear(phi, p):
    """phi: (R, R) ueber [0,1]^2, Zeile = y; p: (2,) -> Wert (Rand geklemmt)."""
    R = phi.shape[0]
    g = jnp.clip(p, 0.0, 1.0) * (R - 1)
    j0 = jnp.clip(jnp.floor(g).astype(jnp.int32), 0, R - 2)
    fx, fy = g[0] - j0[0], g[1] - j0[1]
    c, r = j0[0], j0[1]
    return ((1 - fx) * (1 - fy) * phi[r, c] + fx * (1 - fy) * phi[r, c + 1]
            + (1 - fx) * fy * phi[r + 1, c] + fx * fy * phi[r + 1, c + 1])


def _log_target(x, phi, floor, obs_c, obs_r, w_obs):
    p = x[:2]
    val = jnp.log(_bilinear(phi, p) + floor)
    m = BOUNDARY_MARGIN
    val -= W_BOUNDARY * 0.5 * jnp.sum(jnp.maximum(m - p, 0.0) ** 2
                                      + jnp.maximum(p - (1.0 - m), 0.0) ** 2)
    dist = jnp.sqrt(jnp.sum((p[None] - obs_c) ** 2, axis=-1) + 1e-12)
    val -= w_obs * 0.5 * jnp.sum(jnp.maximum(obs_r - dist, 0.0) ** 2)
    return val


_score = jax.grad(_log_target, argnums=0)


def _stein_grad(traj, phi, floor, obs_c, obs_r, w_obs, h):
    """Wie `ergodic_solver._make_stein_grad`, nur mit dem Gitter-Score."""
    def unit(x1, x2):
        return (es._kernel(x2, x1, h) * _score(x2, phi, floor, obs_c, obs_r, w_obs)
                + es._d_kernel(x2, x1, h))

    def state(x):
        return jnp.mean(jax.vmap(unit, in_axes=(None, 0))(x, traj), axis=0)

    return jax.vmap(state)(traj)


def _iteration(u, x0, phi, floor, obs_c, obs_r, w_obs, h, step):
    """Eine Iteration von Algorithmus 1 bei Sun (= Schleifenkoerper von
    `run_ergodic_coverage`)."""
    x_traj, A, B = _PM.linearize_dyn(x0, u)
    dx = _stein_grad(x_traj, phi, floor, obs_c, obs_r, w_obs, h)
    v, _ = _PM.solve(jnp.zeros(4), A, B, dx)
    return u + step * v


def _positions(u, x0):
    """(TSTEPS+1, 2): Startpunkt + simulierte Positionen."""
    return jnp.concatenate([x0[None, :2], _PM.traj_sim(x0, u)[:, :2]], axis=0)


_it_batch = jax.vmap(_iteration, in_axes=(0, 0, 0, 0, None, None, None, None, None))


@partial(jax.jit, static_argnums=(8,))
def _run_batch_logged(u0, x0, phi, floor, obs_c, obs_r, w_obs, fit, n_iters,
                      h=H, step=STEP_SIZE):
    """u0: (C, TSTEPS, 2), x0: (C, 4), phi: (C, R, R), floor: (C,),
    fit: (nxi, TSTEPS+1) lineare Fit-Abbildung Positionen -> Kontrollpunkte
    (`_fit_matrix`).
    -> (u_final (C, TSTEPS, 2), log (n_iters, C, nxi, 2)). Die Laenge des Logs
    ist Teil der Form, `n_iters` ist hier deshalb statisch."""
    pos = jax.vmap(_positions)

    def body(u, _):
        u = _it_batch(u, x0, phi, floor, obs_c, obs_r, w_obs, h, step)
        return u, jnp.einsum('ij,cjd->cid', fit, pos(u, x0))

    return jax.lax.scan(body, u0, None, length=n_iters)


@jax.jit
def _run_batch_plain(u0, x0, phi, floor, obs_c, obs_r, w_obs, n_iters,
                     h=H, step=STEP_SIZE):
    """Wie `_run_batch_logged` ohne Log; `n_iters` ist dynamisch, ein Wechsel
    der Iterationszahl (GUI-Regler, SVGD-Sweeps) kompiliert also nicht neu."""
    return jax.lax.fori_loop(
        0, n_iters, lambda _, u: _it_batch(u, x0, phi, floor, obs_c, obs_r, w_obs, h, step), u0)


# ── Hilfen: Eingabe vorbereiten, Ausgabe zurueckgeben ─────────────────────────

def _resample(curve, n):
    """(T, 2) -> (n, 2), linear ueber den Punktindex (wie `_downsample_or_pad`,
    ohne Glaettung: die Netz-Bahn ist bereits glatt)."""
    curve = np.asarray(curve, dtype=np.float64)
    s = np.linspace(0, len(curve) - 1, n)
    idx = np.arange(len(curve))
    return np.column_stack([np.interp(s, idx, curve[:, 0]), np.interp(s, idx, curve[:, 1])])


def _obstacle_arrays(obstacle, obstacle_weight):
    """CircleObstacle / CompositeObstacle -> (centers (M,2), radii (M,), w)."""
    if obstacle is None:
        return np.zeros((1, 2)), np.zeros(1), 0.0
    parts = getattr(obstacle, 'obstacles', None) or [obstacle]
    c = np.array([o.center for o in parts], dtype=np.float64)
    r = np.array([o.effective_radius for o in parts], dtype=np.float64)
    return c, r, W_OBSTACLE * float(obstacle_weight) / 20.0


def _basis(nxi, pts):
    from obstacles import bspline_basis_matrix
    return np.asarray(bspline_basis_matrix(nxi, pts, DEGREE), dtype=np.float64)


def make_grid_score(phi_grid):
    """Standalone JAX score function nabla log p(x) from ANY density grid
    (R, R) over [0,1]^2, row = y. Reuses the exact bilinear interpolation and
    floor/blur construction `run_batch` already applies internally
    (`effective_targets`, `_bilinear`), but exposes it for callers outside
    the refiner loop -- e.g. RQ3's GP-belief posterior (`GPBelief.posterior_grid`
    -> `apply_cfm_belief.ucb_density`/`zieldichte`), which is not one of the
    GMM-based `ergodic_solver` targets `run_ergodic_coverage` was built for.

    Added 2026-10 for the Sun-comparison research questions beyond RQ1/RQ2/RQ5
    (see `evaluation_full_matrix/rq_experiments/README.md`); every other
    function in this module predates that work and is unchanged."""
    phi_eff, floor = effective_targets(np.asarray(phi_grid, dtype=np.float64)[None])
    phi_eff_j = jnp.asarray(phi_eff[0], dtype=jnp.float32)
    floor_j = jnp.float32(floor[0])

    def log_p(x):
        return jnp.log(_bilinear(phi_eff_j, x[:2]) + floor_j)

    return jax.grad(log_p)


def prepare(curves, starts):
    """Startbahnen -> (u0 (C,TSTEPS,2), x0 (C,4)) ueber das PID-Tracking des
    Generators. `starts`: (C,2) oder None (dann der erste Bahnpunkt)."""
    us, xs = [], []
    for c, curve in enumerate(curves):
        p = _resample(curve, TSTEPS + 1)
        if starts is not None:
            p[0] = np.asarray(starts[c], dtype=np.float64)
        u, v0 = es._pid_track(p, DT, TSTEPS)
        us.append(u)
        xs.append([p[0, 0], p[0, 1], v0[0], v0[1]])
    return np.stack(us), np.asarray(xs, dtype=np.float64)


def _fit_matrix(nxi):
    """(nxi, TSTEPS+1): Kontrollpunkte = F @ Positionen. Kleinste Quadrate mit
    festgehaltenem erstem Kontrollpunkt = erster Position: bei der geklemmten
    B-Spline ist das genau der Kurvenanfang, der Startpunkt (Anfangszustand der
    Dynamik) bleibt also auch im Fit exakt."""
    B = _basis(nxi, TSTEPS + 1)
    M = np.eye(TSTEPS + 1)
    M[:, 0] -= B[:, 0]
    F = np.zeros((nxi, TSTEPS + 1))
    F[0, 0] = 1.0
    F[1:] = np.linalg.pinv(B[:, 1:]) @ M
    return F


def effective_targets(phis):
    """(C, R, R) -> (phi_eff (C, R, R), floor (C,)), siehe Modul-Docstring."""
    from scipy.ndimage import gaussian_filter
    out, fl = [], []
    for p in phis:
        m = max(float(np.max(p)), 1e-12)
        b = gaussian_filter(p, sigma=BLUR_SIGMA * (p.shape[0] - 1), mode='nearest')
        out.append(p + FLOOR_REL * m * b / max(float(np.max(b)), 1e-30))
        fl.append(1e-12 * m)
    return np.stack(out), np.array(fl)


def run_batch(curves, phis, starts, n_iters, nxi, record=False, obstacle=None,
              obstacle_weight=20.0):
    """Stapel-Verfeinerung. curves: (C, T, 2); phis: (C, R, R) oder (R, R);
    starts: (C, 2), (2,) oder None.
    -> dict(final_cps (C,nxi,2), final_pos (C,TSTEPS+1,2), init_cps (C,nxi,2),
            log (C, n_iters, nxi, 2) oder None) -- alles numpy. Gerechnet wird in
            float32 wie in der Datengenerierung."""
    curves = np.asarray(curves, dtype=np.float64)
    C, T = curves.shape[:2]
    phis = np.asarray(phis, dtype=np.float64)
    if phis.ndim == 2:
        phis = np.broadcast_to(phis, (C,) + phis.shape)
    phis = np.clip(phis, 0.0, None)
    if starts is not None:
        starts = np.asarray(starts, dtype=np.float64)
        if starts.ndim == 1:
            starts = np.broadcast_to(starts, (C, 2))
    u0, x0 = prepare(curves, starts)
    B_T = _basis(nxi, T)
    init_cps = np.linalg.lstsq(B_T, curves.transpose(1, 0, 2).reshape(T, C * 2),
                               rcond=None)[0].reshape(nxi, C, 2).transpose(1, 0, 2)
    fit = _fit_matrix(nxi)
    obs_c, obs_r, w_obs = _obstacle_arrays(obstacle, obstacle_weight)
    # float32 wie in der Datengenerierung (JAX-Standard, kein x64).
    f32 = lambda a: jnp.asarray(np.asarray(a, dtype=np.float32))
    phi_eff, floor = effective_targets(phis)
    args = (f32(u0), f32(x0), f32(phi_eff), f32(floor), f32(obs_c), f32(obs_r),
            jnp.float32(w_obs))
    if record:
        u, log = _run_batch_logged(*args, f32(fit), int(n_iters))
    else:
        u, log = _run_batch_plain(*args, jnp.int32(n_iters)), None
    pos = np.asarray(jax.vmap(_positions)(u, f32(x0)), dtype=np.float64)
    final_cps = np.einsum('ij,cjd->cid', fit, pos)
    if log is not None:
        log = np.asarray(log).transpose(1, 0, 2, 3)
    return dict(final_cps=final_cps, final_pos=pos, init_cps=init_cps, log=log)


class SunSteinRefiner:
    """Gleiche Schnittstelle wie `SvgdRefiner.refine`, Kern von Sun et al.
    (`ergodic_solver.py`). Deterministisch: `seed` wird nur fuer die
    Schnittstelle angenommen."""

    def __init__(self, seed=0):
        self.seed = seed

    def refine(self, curve_np, phi_np, n_iters, nxi=None, obstacle=None,
               obstacle_weight=20.0, start=None, trajectory_log=None):
        """curve_np: (T,2) Startbahn; phi_np: (R,R) Zieldichte; `start`: (2,)
        Startpunkt (= Anfangszustand). `trajectory_log` (nur mit nxi != T):
        Eintrag 0 = B-Spline-Fit der Startbahn, Eintrag i = Kontrollpunkte nach
        Iteration i -- wie beim alten Refiner. -> (T,2) verfeinerte Bahn."""
        if trajectory_log is not None and (nxi is None or nxi == curve_np.shape[0]):
            raise ValueError("trajectory_log braucht den B-Spline-Zweig (nxi != T)")
        if n_iters <= 0:
            if start is not None:
                curve_np = curve_np.copy()
                curve_np[0] = np.asarray(start, dtype=curve_np.dtype)
            return curve_np
        T = curve_np.shape[0]
        k = nxi if (nxi is not None and nxi != T) else 25
        out = run_batch(curve_np[None], phi_np, None if start is None else np.asarray(start)[None],
                        n_iters, k, record=trajectory_log is not None, obstacle=obstacle,
                        obstacle_weight=obstacle_weight)
        if trajectory_log is not None:
            trajectory_log.append(out['init_cps'][0].copy())
            trajectory_log.extend(c.copy() for c in out['log'][0])
        if nxi is not None and nxi != T:
            curve_out = _basis(nxi, T) @ out['final_cps'][0]
        else:
            curve_out = _resample(out['final_pos'][0], T)
        curve_out = curve_out.astype(curve_np.dtype, copy=False)
        if start is not None:
            curve_out = curve_out.copy()
            curve_out[0] = np.asarray(start, dtype=curve_out.dtype)
        return curve_out
