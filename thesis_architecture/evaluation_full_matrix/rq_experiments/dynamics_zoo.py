r"""
dynamics_zoo.py
================
RQ5 (Dynamiken, spiegelt Sun et al. Fig. 11): die sechs Dynamiken aus Fig. 11
-- Point-Mass und Differential-Drive je 1./2. Ordnung, plus Dubins Car je
1./2. Ordnung -- als `LQR`-Unterklassen im Stil von Suns eigenen
`PointMassLQR`-Klassen (`stein_flow_coverage.ipynb`,
`stein_flow_coverage_diffdrive.ipynb`). Zwei davon (Point-Mass 2. Ordnung,
Diff-Drive 1. Ordnung) sind direkt aus Suns Notebooks uebernommen, die
anderen vier sind naheliegende Erweiterungen -- das Paper nennt dazu keine
Gleichungen oder Gewichte (siehe Thesis-Doc, Abschnitt "Fig. 11
durchgespielt"), Q/R hier sind also eigene, plausible Wahlen, keine
Sun-Werte.

Jede Klasse bringt zusaetzlich `flat_map(curve_xy, dt, tsteps) -> (x0, u0)`:
die Umrechnung einer (beliebig langen) xy-Kurve -- der Netzausgabe -- in eine
Anfangssteuerfolge fuer GENAU DIESE Dynamik, ueber die differentielle
Flachheit in (x, y) (siehe Thesis-Doc-Tabelle). Point-Mass und Diff-Drive
sind exakt flach; Dubins zusaetzlich nur bis auf die Kruemmungsschranke
kappa_max -- `flat_map` clippt dort hart (siehe `_dubins_omega`).
"""

import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
_arch = os.path.dirname(_mat)
for _p in (_here, _mat, os.path.join(_arch, 'ergodic_dataset_generator')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                     # noqa: E402
import jax.numpy as jnp                        # noqa: E402
from lqrax import LQR                          # noqa: E402

import ergodic_solver as es                    # noqa: E402


def build(dynamics_cls, dt=0.05, **kw):
    """`dynamics_cls(dt, **kw)` -> (pm, linearize_dyn_jit, solve_jit), bereit
    fuer `exp_common.timed_solve` -- LQR auf der CPU wie bei Sun ("lqr solving
    on CPU is faster")."""
    cpu = jax.devices('cpu')[0]
    pm = dynamics_cls(dt=dt, **kw)
    return pm, jax.jit(pm.linearize_dyn, device=cpu), jax.jit(pm.solve, device=cpu)


def estimate_speed(curve_xy, dt, tsteps):
    """Bogenlaenge der (geglaetteten) Kurve geteilt durch die Trajektoriendauer
    dt*tsteps -- die Geschwindigkeit, mit der eine FESTE Geschwindigkeit sie in
    genau dieser Zeit abfahren wuerde. Fuer Dubins zwingend noetig: Autos mit
    konstanter Geschwindigkeit koennen nicht langsamer werden, `DUBINS_V0` als
    globale Konstante waere nur fuer Kurven zufaellig richtig, deren
    Bogenlaenge gerade `DUBINS_V0*dt*tsteps` ist -- sonst schiesst die
    simulierte Bahn (bei zu hoher Geschwindigkeit) weit ueber das
    Einheitsquadrat hinaus oder bleibt (bei zu niedriger) weit vor dem Ziel
    stehen, in beiden Faellen schon bei 0 FM-Stein-Iterationen."""
    p = _resample_xy(curve_xy, tsteps + 1, smooth_sigma=_DERIV_SMOOTH_SIGMA)
    L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return L / (dt * tsteps)


def build_for_curve(dynamics_cls, curve_xy, dt, tsteps, **kw):
    """Wie `build`, bestimmt aber fuer Dynamiken mit konstanter Geschwindigkeit
    (`v0`-Attribut, siehe `estimate_speed`) `v0` zuerst aus der gegebenen
    Kurve, BEVOR die Instanz (und damit `dyn()`s JIT-Spur) entsteht -- `v0`
    danach zu aendern wirkt nicht mehr zuverlaessig auf bereits kompilierte
    `linearize_dyn`/`solve`. Dynamiken ohne `v0` (alle ausser Dubins) sind
    unveraendert."""
    if 'v0' in dynamics_cls.__init__.__code__.co_varnames and 'v0' not in kw:
        kw['v0'] = estimate_speed(curve_xy, dt, tsteps)
    return build(dynamics_cls, dt=dt, **kw)


def _resample_xy(curve_xy, n, smooth_sigma=0.0):
    curve = np.asarray(curve_xy, dtype=np.float64)
    s = np.linspace(0, len(curve) - 1, n)
    idx = np.arange(len(curve))
    p = np.column_stack([np.interp(s, idx, curve[:, 0]), np.interp(s, idx, curve[:, 1])])
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        p0 = p[0].copy()
        p = gaussian_filter1d(p, sigma=smooth_sigma, axis=0, mode='nearest')
        p[0] = p0     # Startpunkt bleibt exakt (wie `_finish_gmm_trajectory` im Generator)
    return p


#: Vor jeder Ableitung geglaettet (dieselbe Begruendung und derselbe
#: `gaussian_filter1d`-Griff wie in `ergodic_solver._finish_gmm_trajectory`:
#: "Smooth the path to avoid infinite acceleration at corners/transits") --
#: ein per Finite-Differenzen aus einer rohen Punktfolge geschaetztes
#: Geschwindigkeits-/Beschleunigungsprofil hat an den Raendern und an
#: scharfen Ecken sonst Spitzen, die die Dynamik (vor allem 2. Ordnung) aus
#: dem Einheitsquadrat tragen koennen.
_DERIV_SMOOTH_SIGMA = 2.0


def _derivatives(p, dt):
    """(T,2) Positionen, gleichmaessig in dt -> (v (T,2), a (T,2)) per
    zentraler Differenz (Raender einseitig), wie der `MPDLayer`-Gedanke in
    CLAUDE.md: Geschwindigkeit/Beschleunigung implizit aus der Kurve."""
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    return v, a


# ── 1) Point-Mass, 1. Ordnung: Zustand (x,y), Steuerung = Geschwindigkeit ───

class PointMass1LQR(LQR):
    """xdot = u. `flat_map`: u = Bahngeschwindigkeit direkt."""

    def __init__(self, dt, Q=None, R=None):
        Q = Q if Q is not None else jnp.diag(jnp.array([1.0, 1.0]))
        R = R if R is not None else jnp.diag(jnp.array([0.05, 0.05]))
        super().__init__(dt, x_dim=2, u_dim=2, Q=Q, R=R)

    def dyn(self, xt, ut):
        return ut

    def flat_map(self, curve_xy, dt, tsteps):
        p = _resample_xy(curve_xy, tsteps + 1, smooth_sigma=_DERIV_SMOOTH_SIGMA)
        v, _ = _derivatives(p, dt)
        x0 = jnp.array(p[0])
        u0 = jnp.array(v[:-1])
        return x0, u0

    def positions(self, x_traj):
        return np.asarray(x_traj)[:, :2]


# ── 2) Point-Mass, 2. Ordnung: Suns eigene Klasse (ergodic_solver.py) ───────

class PointMass2LQR(es.PointMassLQR):
    """Identisch zu Suns `PointMassLQR` (`ergodic_solver.py`); hier nur um
    `flat_map`/`positions` ergaenzt, damit alle sechs Dynamiken dieselbe
    Schnittstelle haben."""

    def flat_map(self, curve_xy, dt, tsteps):
        """NICHT `_derivatives` (zweifache Finite-Differenz von p): auf
        beliebigen (auch unseren eigenen synthetischen oder B-Spline-)Kurven
        verstaerkt das Rauschen in der zweiten Ableitung so stark, dass die
        Replay-Simulation (u0 allein, 0 Stein-Iterationen) schon weit vom
        Einheitsquadrat wegdriftet -- gemessen: Startgeschwindigkeit bis
        |v0|~2 und Beschleunigungsspitzen bis ~9.5 bei einer CFM-Testkurve,
        die selbst brav in [0,1]^2 liegt. Dieselbe PD-Tracking-Loesung wie
        `exp_common.cfm_x0_u0` (= `ergodic_solver._pid_track`, Kp=150, Kd=20,
        exakte diskrete Doppelintegrator-Schrittweise) statt Finite-Differenzen
        -- numerisch robust, weil sie die Dynamik beim Erzeugen von u0 schon
        kennt statt sie nachtraeglich zu erraten."""
        p = _resample_xy(curve_xy, tsteps + 1, smooth_sigma=_DERIV_SMOOTH_SIGMA)
        u0_np, v0 = es._pid_track(p, dt, tsteps)
        x0 = jnp.array([p[0, 0], p[0, 1], v0[0], v0[1]])
        return x0, jnp.array(u0_np)

    def positions(self, x_traj):
        return np.asarray(x_traj)[:, :2]


# ── 3) Differential-Drive, 1. Ordnung: Suns eigene Klasse
#    (stein_flow_coverage_diffdrive.ipynb) ──────────────────────────────────

class DiffDrive1LQR(LQR):
    """Zustand (x,y,theta), Steuerung (v,omega). xdot = [v cos th, v sin th, w]."""

    def __init__(self, dt, Q=None, R=None):
        Q = Q if Q is not None else jnp.diag(jnp.array([1.0, 1.0, 1e-5]))
        R = R if R is not None else jnp.diag(jnp.array([0.5, 0.001]))
        super().__init__(dt, x_dim=3, u_dim=2, Q=Q, R=R)

    def dyn(self, xt, ut):
        return jnp.array([ut[0] * jnp.cos(xt[2]), ut[0] * jnp.sin(xt[2]), ut[1]])

    def flat_map(self, curve_xy, dt, tsteps):
        p = _resample_xy(curve_xy, tsteps + 1, smooth_sigma=_DERIV_SMOOTH_SIGMA)
        v, _ = _derivatives(p, dt)
        speed = np.linalg.norm(v, axis=1)
        theta = np.arctan2(v[:, 1], v[:, 0])
        theta = np.unwrap(theta)
        omega = np.gradient(theta, dt)
        x0 = jnp.array([p[0, 0], p[0, 1], theta[0]])
        u0 = jnp.array(np.stack([speed[:-1], omega[:-1]], axis=1))
        return x0, u0

    def positions(self, x_traj):
        return np.asarray(x_traj)[:, :2]


# ── 4) Differential-Drive, 2. Ordnung: (x,y,theta,v,omega), Steuerung
#    (Beschleunigung, Winkelbeschleunigung) ──────────────────────────────────

class DiffDrive2LQR(LQR):
    def __init__(self, dt, Q=None, R=None):
        Q = Q if Q is not None else jnp.diag(jnp.array([1.0, 1.0, 1e-5, 1e-3, 1e-3]))
        R = R if R is not None else jnp.diag(jnp.array([0.2, 0.05]))
        super().__init__(dt, x_dim=5, u_dim=2, Q=Q, R=R)

    def dyn(self, xt, ut):
        v, w = xt[3], xt[4]
        return jnp.array([v * jnp.cos(xt[2]), v * jnp.sin(xt[2]), w, ut[0], ut[1]])

    def flat_map(self, curve_xy, dt, tsteps):
        p = _resample_xy(curve_xy, tsteps + 1, smooth_sigma=_DERIV_SMOOTH_SIGMA)
        v, _ = _derivatives(p, dt)
        speed = np.linalg.norm(v, axis=1)
        theta = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        omega = np.gradient(theta, dt)
        accel = np.gradient(speed, dt)
        alpha = np.gradient(omega, dt)
        x0 = jnp.array([p[0, 0], p[0, 1], theta[0], speed[0], omega[0]])
        u0 = jnp.array(np.stack([accel[:-1], alpha[:-1]], axis=1))
        return x0, u0

    def positions(self, x_traj):
        return np.asarray(x_traj)[:, :2]


# ── 5) Dubins Car, 1. Ordnung: feste Geschwindigkeit v0, Steuerung omega ────

#: v0 an der Groessenordnung typischer Bahnlaengen in diesem Projekt
#: ausgerichtet (CFM/Heuristik-Bahnen ~4.5-7 Laengeneinheiten ueber die volle
#: Trajektorie, siehe `evaluation_full_matrix/results/.../SUMMARY.md`
#: Abschnitt 11) -- bei dt*tsteps=10s Trajektoriendauer entspricht das einer
#: mittleren Geschwindigkeit von ~0.5-0.7. Point-Mass/Diff-Drive koennen
#: selbst langsamer werden, Dubins mit fester Geschwindigkeit nicht; v0 MUSS
#: also von vornherein in dieser Groessenordnung liegen, sonst verlaesst die
#: Bahn das Einheitsquadrat (mit v0=1.0 waeren es 10 Laengeneinheiten).
DUBINS_V0 = 0.5
DUBINS_KAPPA_MAX = 8.0   # 1/Laengeneinheit -> minimaler Kurvenradius 1/(8*0.5)=0.25


def _dubins_omega(theta, dt, kappa_max):
    omega = np.gradient(np.unwrap(theta), dt)
    return np.clip(omega, -kappa_max * DUBINS_V0, kappa_max * DUBINS_V0)


class Dubins1LQR(LQR):
    """Zustand (x,y,theta), Steuerung omega, feste Geschwindigkeit `DUBINS_V0`.
    `flat_map` clippt omega hart auf `DUBINS_KAPPA_MAX` -- die eine Stelle, an
    der die Flachheit nur bis auf die Kruemmungsschranke gilt (siehe
    Thesis-Doc)."""

    def __init__(self, dt, v0=DUBINS_V0, Q=None, R=None):
        self.v0 = v0
        Q = Q if Q is not None else jnp.diag(jnp.array([1.0, 1.0, 1e-5]))
        R = R if R is not None else jnp.diag(jnp.array([0.02]))
        super().__init__(dt, x_dim=3, u_dim=1, Q=Q, R=R)

    def dyn(self, xt, ut):
        return jnp.array([self.v0 * jnp.cos(xt[2]), self.v0 * jnp.sin(xt[2]), ut[0]])

    def flat_map(self, curve_xy, dt, tsteps):
        p = _resample_xy(curve_xy, tsteps + 1, smooth_sigma=_DERIV_SMOOTH_SIGMA)
        v, _ = _derivatives(p, dt)
        theta = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        omega = _dubins_omega(theta, dt, DUBINS_KAPPA_MAX)
        x0 = jnp.array([p[0, 0], p[0, 1], theta[0]])
        u0 = jnp.array(omega[:-1])[:, None]
        return x0, u0

    def positions(self, x_traj):
        return np.asarray(x_traj)[:, :2]


# ── 6) Dubins Car, 2. Ordnung: Zustand (x,y,theta,omega), Steuerung omega_dot

class Dubins2LQR(LQR):
    def __init__(self, dt, v0=DUBINS_V0, Q=None, R=None):
        self.v0 = v0
        Q = Q if Q is not None else jnp.diag(jnp.array([1.0, 1.0, 1e-5, 1e-3]))
        R = R if R is not None else jnp.diag(jnp.array([0.01]))
        super().__init__(dt, x_dim=4, u_dim=1, Q=Q, R=R)

    def dyn(self, xt, ut):
        return jnp.array([self.v0 * jnp.cos(xt[2]), self.v0 * jnp.sin(xt[2]), xt[3], ut[0]])

    def flat_map(self, curve_xy, dt, tsteps):
        p = _resample_xy(curve_xy, tsteps + 1, smooth_sigma=_DERIV_SMOOTH_SIGMA)
        v, _ = _derivatives(p, dt)
        theta = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
        omega = _dubins_omega(theta, dt, DUBINS_KAPPA_MAX)
        alpha = np.gradient(omega, dt)
        x0 = jnp.array([p[0, 0], p[0, 1], theta[0], omega[0]])
        u0 = jnp.array(alpha[:-1])[:, None]
        return x0, u0

    def positions(self, x_traj):
        return np.asarray(x_traj)[:, :2]


def cold_x0_u0(pm, start, dt, tsteps):
    """"Sun kalt" fuer eine beliebige Dynamik: u0 = 0 ueberall (Suns eigene
    Formel ist buchstaeblich "u=0"); fuer `PointMass2LQR` zusaetzlich Suns
    Anfangsgeschwindigkeit Richtung Zentrum (`sun_cold_x0_u0`), weil dort der
    Zustand ohne sie entartet (reiner Stillstand, siehe Docstring oben). Alle
    anderen Zustandskomponenten jenseits (x,y) (Blickrichtung, Geschwindigkeit,
    Drehrate) starten bei 0 bzw. zeigen auf die Mitte des Suchraums -- derselbe
    Geist wie Suns Formel, nur je Dynamik ausbuchstabiert."""
    if isinstance(pm, PointMass2LQR):
        T = dt * tsteps
        x0 = jnp.array([start[0], start[1], 2.0 * (0.5 - start[0]) / T,
                       2.0 * (0.5 - start[1]) / T])
    else:
        theta0 = float(np.arctan2(0.5 - start[1], 0.5 - start[0]))
        extra = {PointMass1LQR: [], DiffDrive1LQR: [theta0],
                DiffDrive2LQR: [theta0, 0.0, 0.0], Dubins1LQR: [theta0],
                Dubins2LQR: [theta0, 0.0]}[type(pm)]
        x0 = jnp.array([start[0], start[1]] + extra)
    return x0, jnp.zeros((tsteps, pm.u_dim))


#: Reihenfolge wie in Sun et al. Fig. 11.
DYNAMICS = {
    'Point-Mass 1st': PointMass1LQR,
    'Point-Mass 2nd': PointMass2LQR,
    'Diff-Drive 1st': DiffDrive1LQR,
    'Diff-Drive 2nd': DiffDrive2LQR,
    'Dubins 1st':     Dubins1LQR,
    'Dubins 2nd':     Dubins2LQR,
}
#: Welche davon bereits in Suns Repo vorhanden sind (siehe Anhang B der
#: Thesis-Doc) -- fuer die Legenden/Annotation der RQ5-Plots.
FROM_SUN_REPO = {'Point-Mass 2nd', 'Diff-Drive 1st'}
