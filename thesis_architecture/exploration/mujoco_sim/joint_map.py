"""
joint_map.py
============
Smooth, single-branch joint-space map of *orthogonal* tool poses over the board.

Online differential IK from the previous command ("continuation") can drift
into a joint-limit pocket or onto a wrong branch and then holds the tool tilted
for a long time.  Here the board is instead sampled once on a grid: starting
from a pose at the board centre, every grid cell is solved by continuation
from its already-solved neighbour (tool axis = hard constraint), so the whole
map lives on one IK branch and is validated at build time.  Live, the arm only
interpolates this map and polishes the result with a few IK steps.

The centre pose decides whether one branch can cover the whole board, so
several centre solutions are screened on a coarse grid and the one that covers
the board exactly, with the most joint-limit margin and a smooth map, wins.
"""

import hashlib
import os

import numpy as np

from . import board as board_mod
from .board import TOOL_AXIS

#: A map cell counts as exact below these errors.
EXACT_POS = 5e-4        # m
EXACT_AXIS = 1e-3       # rad
SCREEN_N = 9
SCREEN_KEEP = 6
SMOOTH_STEP = np.radians(30.0)   # max joint change between screening cells


class JointMap:
    """``q[iv, iu]`` on a regular (n x n) grid over board coordinates [0,1]^2."""

    def __init__(self, q, report):
        self.q = q
        self.n = q.shape[0]
        self.report = report

    def query(self, uv):
        """Bilinear interpolation of the joint configuration at ``uv`` (2,)."""
        u, v = np.clip(np.asarray(uv, dtype=np.float64), 0.0, 1.0) * (self.n - 1)
        j0 = min(int(u), self.n - 2)
        i0 = min(int(v), self.n - 2)
        b, a = u - j0, v - i0
        q = self.q
        return ((1 - a) * (1 - b) * q[i0, j0] + (1 - a) * b * q[i0, j0 + 1]
                + a * (1 - b) * q[i0 + 1, j0] + a * b * q[i0 + 1, j0 + 1])


def _world_fn(n, standoff):
    def world(iv, iu):
        return board_mod.uv_to_world(np.array([iu / (n - 1.0), iv / (n - 1.0)]), standoff)
    return world


def _fill_grid(ik, q_c, n, standoff, iterations):
    """Continuation over an (n x n) grid starting from the centre pose ``q_c``."""
    world = _world_fn(n, standoff)
    ic = n // 2
    ik.posture = q_c.copy()
    q = np.full((n, n, ik.n), np.nan)
    pos_err = np.zeros((n, n))
    axis_err = np.zeros((n, n))
    q[ic, ic] = q_c

    def fill(iv, iu, seed):
        qq, pe, ae = ik.solve(seed, world(iv, iu), TOOL_AXIS, iterations=iterations,
                              priority='axis')
        q[iv, iu], pos_err[iv, iu], axis_err[iv, iu] = qq, pe, ae

    for iu in range(ic + 1, n):                       # centre row, to the right
        fill(ic, iu, q[ic, iu - 1])
    for iu in range(ic - 1, -1, -1):                  # centre row, to the left
        fill(ic, iu, q[ic, iu + 1])
    for iu in range(n):                               # every column, up and down
        for iv in range(ic + 1, n):
            fill(iv, iu, q[iv - 1, iu])
        for iv in range(ic - 1, -1, -1):
            fill(iv, iu, q[iv + 1, iu])
    return q, pos_err, axis_err


def _stats(ik, q, pos_err, axis_err):
    margin = (np.minimum(q - ik.lo, ik.hi - q) / ik.span).min()
    # Joint 7 only spins the cylindrical eraser about its own axis; ignore it
    # for smoothness (it has no effect on the tip or on the tool axis).
    step = max(np.abs(np.diff(q[..., :6], axis=0)).max(),
               np.abs(np.diff(q[..., :6], axis=1)).max())
    return {
        'max_pos_err': float(pos_err.max()),
        'max_axis_err': float(axis_err.max()),
        'min_margin': float(margin),          # fraction of the joint range
        'max_cell_step': float(step),
    }


def _centre_candidates(ik, target, rng):
    """Distinct valid centre solutions, best joint-limit margin first."""
    seeds = [rng.uniform(ik.lo, ik.hi) for _ in range(24)]
    seeds += [np.array([0.0, rng.uniform(-1, 1.2), 0.0, rng.uniform(-3.0, -0.6), 0.0,
                        rng.uniform(0.0, 3.7), rng.uniform(-1, 1)]) for _ in range(24)]
    ik.posture = np.array([0.0, 0.3, 0.0, -1.8, 0.0, 1.2, 0.0])
    cands = []
    for s in seeds:
        q, pe, ae = ik.solve(s, target, TOOL_AXIS, iterations=300, priority='axis')
        if pe < 1e-4 and ae < 1e-3 and not any(np.abs(q - c).max() < 0.05 for c in cands):
            cands.append(q)
    cands.sort(key=lambda q: -(np.minimum(q - ik.lo, ik.hi - q) / ik.span).min())
    return cands


def build_joint_map(ik, standoff, n=25, iterations=80, verbose=True):
    """Build the map; ``n`` must be odd so that the centre is a grid point."""
    if n % 2 == 0:
        raise ValueError("n must be odd")
    rng = np.random.default_rng(0)
    centre = board_mod.uv_to_world(np.array([0.5, 0.5]), standoff)
    cands = _centre_candidates(ik, centre, rng)
    if not cands:
        raise RuntimeError("cannot reach the board centre with the tool axis "
                           "orthogonal to the board")

    best = None
    for q_c in cands[:SCREEN_KEEP]:
        q, pe, ae = _fill_grid(ik, q_c, SCREEN_N, standoff, iterations)
        s = _stats(ik, q, pe, ae)
        exact = s['max_pos_err'] < EXACT_POS and s['max_axis_err'] < EXACT_AXIS
        smooth = s['max_cell_step'] <= SMOOTH_STEP
        key = (not exact, not smooth, -s['min_margin'])
        if best is None or key < best[0]:
            best = (key, q_c)
    q_c = best[1]
    if best[0][0] and verbose:
        print("  warning: no centre solution covers the board orthogonally")

    q, pos_err, axis_err = _fill_grid(ik, q_c, n, standoff, iterations)
    report = _stats(ik, q, pos_err, axis_err)
    if verbose:
        print("  joint map %dx%d: max tip error %.3f mm | max tilt %.4f deg | "
              "min joint-limit margin %.1f%% of range | max step between cells %.1f deg"
              % (n, n, report['max_pos_err'] * 1e3, np.degrees(report['max_axis_err']),
                 100 * report['min_margin'], np.degrees(report['max_cell_step'])))
    return JointMap(q, report), q_c


_REPORT_KEYS = ('max_pos_err', 'max_axis_err', 'min_margin', 'max_cell_step')
#: Bump when the build algorithm changes so that stale cached maps are rebuilt.
MAP_VERSION = 2


def get_joint_map(ik, standoff, cache_dir, key_parts, n=25, verbose=True):
    """Load the map from the on-disk cache or build (and cache) it."""
    key = hashlib.sha1("|".join(str(x) for x in
                                key_parts + (round(standoff, 6), n, MAP_VERSION))
                       .encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, "jointmap_%s.npz" % key)
    if os.path.exists(path):
        try:
            blob = np.load(path)
            report = {k: float(blob[k]) for k in _REPORT_KEYS}
            ik.posture = blob['posture'].copy()
            return JointMap(blob['q'], report)
        except Exception as exc:                       # corrupt/stale cache
            print("  ignoring joint-map cache (%s)" % exc)
    if verbose:
        print("  building the orthogonal joint map (once, then cached) ...")
    jmap, q_c = build_joint_map(ik, standoff, n=n, verbose=verbose)
    ik.posture = q_c.copy()
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, q=jmap.q, posture=q_c, **jmap.report)
    return jmap
