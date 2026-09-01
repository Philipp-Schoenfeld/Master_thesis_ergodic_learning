"""
board.py
========
Single source of truth for the writeboard geometry in the MuJoCo scene, plus
loaders for the ergodic trajectories / target densities stored in the SQLite
datasets.

Board layout
------------
The board is a flat square panel standing upright in front of the robot.  Its
painted face lies in the world Y-Z plane at ``BOARD_FACE_X``; the robot
approaches it from -X, so the tool has to point along **+X**.

Board coordinates ``(u, v)`` are the same normalised [0,1]^2 coordinates that
the 2D GUI (``interactive_sim.py`` / ``writeboard.py``) uses:

    u = 0 -> +Y edge (left  of the board, seen from the robot)
    u = 1 -> -Y edge (right of the board)
    v = 0 -> bottom edge
    v = 1 -> top edge

Keeping this mapping in one place means the board can be moved or resized
without regenerating ``ergodic_dataset_robot.db``: trajectories are converted
back to (u, v) on load and then re-projected through the current constants.
"""

import json
import os
import sqlite3

import numpy as np

# -- Board geometry ----------------------------------------------------------
BOARD_ORIGIN = np.array([0.500, 0.0, 0.45])   # centre of the board body
BOARD_SIZE = 0.6                              # edge length of the square face
SITE_OFFSET_X = 0.003                         # pixel sites sit in front of the body
SITE_HALF_THICK = 0.001                       # half thickness of a pixel site
BOARD_FACE_X = float(BOARD_ORIGIN[0] + SITE_OFFSET_X + SITE_HALF_THICK)

#: Direction the tool has to point in to face the board.
TOOL_AXIS = np.array([1.0, 0.0, 0.0])

#: Resolution of the coloured pixel-site grid baked into the MuJoCo model.
SITE_RES = 64
#: Resolution of the shared target-density array used by the 2D GUI.
TRUTH_RES = 96

#: How far the eraser tip stays off the painted surface (m).  The position
#: servo overshoots by up to ~6 mm during the fastest direction changes, so a
#: few millimetres of clearance keep the pad from sinking into the pixel grid.
DEFAULT_STANDOFF = 0.006

#: Fastest tip speed the arm still tracks cleanly along the board (m/s).
#: Measured by sweeping a circular target across the board in the live mode:
#: mean tracking error is 0.6 mm at 0.10 m/s, 1.5 mm at 0.20, 4.2 mm at 0.30
#: and 6.9 mm at 0.40, and above ~0.3 m/s the tip starts pushing through the
#: board surface.  The 2D control centre paces its agent against this so both
#: views show the same motion.
MAX_TIP_SPEED = 0.30

#: Panda joint velocity limits (rad/s, datasheet).  The MuJoCo model does not
#: carry them, but the live IK has to respect them: when the continuation
#: switches to another IK branch it would otherwise command a ~46 rad/s
#: reconfiguration that the physical arm cannot follow.
JOINT_VMAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])


def max_ui_speed(tip_speed=MAX_TIP_SPEED):
    """``tip_speed`` in m/s expressed in board/GUI units per second."""
    return tip_speed / BOARD_SIZE

# -- Mapping used by ``3D_ergodic_learning/project_robot_board.py`` -----------
# Kept explicit so trajectories written by that script can be un-projected back
# to board coordinates regardless of where the board sits today.
LEGACY_Y0, LEGACY_Y_SPAN = 0.3, -0.6
LEGACY_Z0, LEGACY_Z_SPAN = 0.15, 0.6

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPLORATION = os.path.dirname(_HERE)
_ARCH = os.path.dirname(_EXPLORATION)
_ROOT = os.path.dirname(_ARCH)

ROBOT_DB = os.path.join(_ROOT, '3D_ergodic_learning', 'ergodic_dataset_robot.db')
SOURCE_DB = os.path.join(_ARCH, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')


# -- Coordinate transforms ---------------------------------------------------
def uv_to_world(uv, standoff=0.0):
    """Board coordinates (..., 2) in [0,1]^2 -> world positions (..., 3)."""
    uv = np.asarray(uv, dtype=np.float64)
    out = np.empty(uv.shape[:-1] + (3,), dtype=np.float64)
    out[..., 0] = BOARD_FACE_X - standoff
    out[..., 1] = BOARD_ORIGIN[1] + BOARD_SIZE * (0.5 - uv[..., 0])
    out[..., 2] = BOARD_ORIGIN[2] + BOARD_SIZE * (uv[..., 1] - 0.5)
    return out


def world_to_uv(pos):
    """World positions (..., 3) -> board coordinates (..., 2)."""
    pos = np.asarray(pos, dtype=np.float64)
    u = 0.5 - (pos[..., 1] - BOARD_ORIGIN[1]) / BOARD_SIZE
    v = 0.5 + (pos[..., 2] - BOARD_ORIGIN[2]) / BOARD_SIZE
    return np.stack([u, v], axis=-1)


def _legacy_pos_to_uv(pos):
    """Un-project the hard-coded board mapping of ``project_robot_board.py``."""
    u = (pos[..., 1] - LEGACY_Y0) / LEGACY_Y_SPAN
    v = (pos[..., 2] - LEGACY_Z0) / LEGACY_Z_SPAN
    return np.stack([u, v], axis=-1)


# -- Trajectory loading ------------------------------------------------------
def load_trajectory_uv(shape_name):
    """
    Board-coordinate ergodic trajectory for ``shape_name``.

    Prefers the projected robot dataset; falls back to the raw 2D dataset so
    the simulation still runs if ``ergodic_dataset_robot.db`` was never built.

    Returns (uv, dt) with uv of shape (N, 2) and dt the solver timestep [s].
    """
    if os.path.exists(ROBOT_DB):
        con = sqlite3.connect(ROBOT_DB)
        try:
            row = con.execute(
                "SELECT traj_pos, n_points FROM ergodic_pairs_robot WHERE shape_name=?",
                (shape_name,),
            ).fetchone()
        finally:
            con.close()
        if row is not None:
            pos = np.frombuffer(row[0], dtype=np.float32).reshape(row[1], 3).astype(np.float64)
            return _legacy_pos_to_uv(pos), _lookup_dt(shape_name)

    uv, dt = _load_trajectory_from_source(shape_name)
    if uv is None:
        raise KeyError(
            "No ergodic trajectory for shape %r.\n  looked in %s\n        and %s"
            % (shape_name, ROBOT_DB, SOURCE_DB)
        )
    return uv, dt


def _load_trajectory_from_source(shape_name):
    if not os.path.exists(SOURCE_DB):
        return None, 0.05
    con = sqlite3.connect(SOURCE_DB)
    try:
        row = con.execute(
            "SELECT trajectory, tsteps, dt FROM ergodic_pairs WHERE shape_name=?",
            (shape_name,),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None, 0.05
    uv = np.frombuffer(row[0], dtype=np.float32).reshape(row[1], 2).astype(np.float64)
    return uv, float(row[2])


def _lookup_dt(shape_name, default=0.05):
    if not os.path.exists(SOURCE_DB):
        return default
    con = sqlite3.connect(SOURCE_DB)
    try:
        row = con.execute(
            "SELECT dt FROM ergodic_pairs WHERE shape_name=?", (shape_name,)
        ).fetchone()
    finally:
        con.close()
    return float(row[0]) if row else default


def list_shapes():
    """Shape names available in the robot dataset (sorted)."""
    db = ROBOT_DB if os.path.exists(ROBOT_DB) else SOURCE_DB
    table = 'ergodic_pairs_robot' if db == ROBOT_DB else 'ergodic_pairs'
    con = sqlite3.connect(db)
    try:
        return sorted(r[0] for r in con.execute("SELECT DISTINCT shape_name FROM " + table))
    finally:
        con.close()


# -- Target density ----------------------------------------------------------
def load_density(shape_name, resolution=TRUTH_RES):
    """
    Rasterise the target density of ``shape_name`` onto a (res, res) grid,
    normalised to [0, 1] and indexed as ``grid[v_index, u_index]`` -- the same
    convention as ``shape_library.pdf_on_grid`` and the 2D GUI.

    Re-implemented in plain NumPy so the simulation does not need JAX.
    Returns ``None`` when the shape has no stored density parameters.
    """
    if not os.path.exists(SOURCE_DB):
        return None
    con = sqlite3.connect(SOURCE_DB)
    try:
        row = con.execute(
            "SELECT density_params FROM ergodic_pairs WHERE shape_name=?", (shape_name,)
        ).fetchone()
    finally:
        con.close()
    if row is None:
        return None

    spec = json.loads(row[0])
    axis = np.linspace(0.0, 1.0, resolution)
    gx, gy = np.meshgrid(axis, axis)            # gx varies along u, gy along v
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    if spec.get('type') == 'analytical':
        vals = _analytical_density(pts, spec)
    else:
        vals = _gmm_density(pts, spec)

    grid = vals.reshape(resolution, resolution)
    peak = float(grid.max())
    return grid / peak if peak > 0 else grid


def _gmm_density(pts, spec):
    means = np.asarray(spec['means'], dtype=np.float64)      # (K, 2)
    covs = np.asarray(spec['covs'], dtype=np.float64)        # (K, 2, 2)
    weights = np.asarray(spec['weights'], dtype=np.float64)  # (K,)
    weights = weights / weights.sum()

    dets = np.linalg.det(covs)
    inv = np.linalg.inv(covs)
    diff = pts[:, None, :] - means[None, :, :]               # (P, K, 2)
    maha = np.einsum('pki,kij,pkj->pk', diff, inv, diff)
    norm = weights / (2.0 * np.pi * np.sqrt(np.maximum(dets, 1e-30)))
    return (norm[None, :] * np.exp(-0.5 * maha)).sum(axis=1)


def _analytical_density(pts, spec):
    segments = np.asarray(spec['segments'], dtype=np.float64)   # (S, 2, 2)
    sigma = float(spec.get('sigma', 0.025))
    a, b = segments[:, 0, :], segments[:, 1, :]
    ab = b - a                                                  # (S, 2)
    denom = np.maximum((ab * ab).sum(axis=1), 1e-8)
    ap = pts[:, None, :] - a[None, :, :]                        # (P, S, 2)
    t = np.clip((ap * ab[None]).sum(axis=2) / denom[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    d2 = ((pts[:, None, :] - proj) ** 2).sum(axis=2)
    # ``max`` (not sum) keeps the density uniform where strokes cross.
    return np.exp(-d2 / (2.0 * sigma ** 2)).max(axis=1)
