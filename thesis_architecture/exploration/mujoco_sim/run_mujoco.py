"""
run_mujoco.py
=============
Real-time MuJoCo simulation of a Franka Panda holding a board eraser in front
of the writeboard.

Two modes:

``run_shape_playback(shape)``
    Standalone: replay the stored ergodic trajectory of one shape (e.g. ``A``)
    from ``3D_ergodic_learning/ergodic_dataset_robot.db``.  The whole joint
    trajectory is planned up front, so playback is exact and deterministic.

        python -m mujoco_sim.run_mujoco --shape A

``run_mujoco_sim(agent_info_array, shared_truth)``
    Driven live by the 2D control centre (``main.py`` / ``interactive_sim.py``)
    through shared memory.

The target density is drawn on the board with a grid of coloured mini-sites
(``wb_pix_<v>_<u>``) using the project's WHITE_INFERNO colormap.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:                       # allow `python run_mujoco.py`
    sys.path.insert(0, _PARENT)

from mujoco_sim import board as board_mod          # noqa: E402
from mujoco_sim.ik import TaskPriorityIK, plan_joint_path  # noqa: E402
from mujoco_sim.board import (                     # noqa: E402
    BOARD_FACE_X, BOARD_ORIGIN, BOARD_SIZE, SITE_RES, TOOL_AXIS, TRUTH_RES,
    DEFAULT_STANDOFF, JOINT_VMAX, MAX_TIP_SPEED,
)

# -- Eraser tool --------------------------------------------------------------
ERASER_RADIUS = 0.018        # m
ERASER_HALF_LEN = 0.035      # m
ERASER_CENTRE = 0.095        # distance from the hand frame origin along the mount axis
ERASER_TIP = ERASER_CENTRE + ERASER_HALF_LEN
#: Gripper command that closes the fingers onto the eraser (ctrl 0..255 ~ 0..0.04 m).
GRIPPER_CTRL = ERASER_RADIUS / 0.04 * 255.0

N_ARM_JOINTS = 7
RETRACT = 0.12               # how far the tool backs off the board between passes

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, 'cache')


def _build_white_inferno_lut(n=256):
    """WHITE_INFERNO colormap as a uint8 LUT (white fading into inferno)."""
    try:
        import matplotlib
        inf = matplotlib.colormaps['inferno'](np.linspace(0, 1, n))[:, :3]
        blend = 60
        ramp = np.linspace(0, 1, blend)[:, None]
        inf[:blend] = (1 - ramp) * 1.0 + ramp * inf[:blend]
        return (inf * 255).astype(np.uint8)
    except ImportError:
        g = np.linspace(255, 0, n).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)


CMAP_LUT = _build_white_inferno_lut()

#: Mounting positions for the eraser, as offsets along a hand-frame axis.
_AXES = {
    'X': np.array([1.0, 0.0, 0.0]), '-X': np.array([-1.0, 0.0, 0.0]),
    'Y': np.array([0.0, 1.0, 0.0]), '-Y': np.array([0.0, -1.0, 0.0]),
    'Z': np.array([0.0, 0.0, 1.0]), '-Z': np.array([0.0, 0.0, -1.0]),
}


def _quat_z_to(direction):
    """Quaternion whose local +Z maps onto ``direction`` (unit vector)."""
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, np.asarray(direction, dtype=np.float64))
    return quat


# -- Scene --------------------------------------------------------------------
def _build_scene(model_dir, orthogonal_axis="Z", site_res=SITE_RES):
    """
    Build the Panda scene plus writeboard pixel grid and the eraser tool.

    ``orthogonal_axis`` selects which hand-frame axis the eraser sticks out
    along.  ``"Z"`` is the natural one -- that is where the gripper fingers are,
    so the eraser looks like it is actually being held.  Whatever is chosen, the
    ``eraser_tip`` site is oriented so that **its local +Z points along the
    eraser**, which is the axis the IK aligns with the board normal.
    """
    panda_scene = os.path.join(model_dir, "menagerie", "franka_emika_panda", "scene.xml")
    spec = mujoco.MjSpec.from_file(panda_scene)

    # -- Writeboard: a grid of mini boxes acting as pixels --------------------
    wb = spec.worldbody.add_body()
    wb.name = "writeboard_body"
    wb.pos = BOARD_ORIGIN.tolist()

    half = BOARD_SIZE / site_res / 2.0
    for iv in range(site_res):
        for iu in range(site_res):
            u = iu / (site_res - 1.0)
            v = iv / (site_res - 1.0)
            s = wb.add_site()
            s.name = "wb_pix_%d_%d" % (iv, iu)
            s.type = mujoco.mjtGeom.mjGEOM_BOX
            s.size = [board_mod.SITE_HALF_THICK, half, half]
            s.pos = [board_mod.SITE_OFFSET_X,
                     BOARD_SIZE * (0.5 - u),
                     BOARD_SIZE * (v - 0.5)]
            s.rgba = [1.0, 1.0, 1.0, 1.0]

    # -- Eraser held in the gripper ------------------------------------------
    direction = _AXES[orthogonal_axis]
    quat = _quat_z_to(direction)

    hand = next(b for b in spec.bodies if b.name == "hand")
    eraser = hand.add_geom()
    eraser.name = "eraser_geom"
    eraser.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    eraser.size = [ERASER_RADIUS, ERASER_HALF_LEN, 0]
    eraser.pos = (ERASER_CENTRE * direction).tolist()
    eraser.quat = quat.tolist()
    eraser.rgba = [0.93, 0.36, 0.51, 1.0]
    eraser.contype = 0
    eraser.conaffinity = 0

    felt = hand.add_geom()                       # darker "felt" pad on the front
    felt.name = "eraser_pad"
    felt.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    felt.size = [ERASER_RADIUS * 1.02, 0.004, 0]
    felt.pos = ((ERASER_TIP - 0.004) * direction).tolist()
    felt.quat = quat.tolist()
    felt.rgba = [0.18, 0.18, 0.22, 1.0]
    felt.contype = 0
    felt.conaffinity = 0

    tip = hand.add_site()
    tip.name = "eraser_tip"
    tip.type = mujoco.mjtGeom.mjGEOM_SPHERE
    tip.size = [0.004, 0.004, 0.004]
    tip.pos = (ERASER_TIP * direction).tolist()
    tip.quat = quat.tolist()                     # local +Z == eraser axis
    tip.rgba = [0.0, 0.78, 0.33, 1.0]

    return spec.compile()


class BoardPainter:
    """Maps a normalised density grid onto the board's pixel sites."""

    def __init__(self, model, site_res=SITE_RES):
        self.model = model
        self.res = site_res
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "wb_pix_%d_%d" % (iv, iu))
               for iv in range(site_res) for iu in range(site_res)]
        if min(ids) < 0:
            raise RuntimeError("writeboard pixel sites missing from the model")
        self.site_ids = np.array(ids, dtype=np.int32)
        self._rgba = np.ones((site_res * site_res, 4), dtype=np.float64)

    def paint(self, density):
        """``density`` is (res, res) in [0, 1], indexed [v, u]."""
        grid = np.asarray(density, dtype=np.float64)
        if grid.shape != (self.res, self.res):
            vi = np.linspace(0, grid.shape[0] - 1, self.res).astype(np.int32)
            ui = np.linspace(0, grid.shape[1] - 1, self.res).astype(np.int32)
            grid = grid[np.ix_(vi, ui)]
        idx = np.clip(grid * 255.0, 0, 255).astype(np.intp)
        self._rgba[:, :3] = CMAP_LUT[idx].reshape(-1, 3) / 255.0
        self.model.site_rgba[self.site_ids] = self._rgba


def erase_disc(density, uv, radius_uv):
    """Zero the density inside a disc centred at board coordinate ``uv``."""
    res = density.shape[0]
    axis = np.linspace(0.0, 1.0, res)
    du = axis[None, :] - uv[0]
    dv = axis[:, None] - uv[1]
    density[(du * du + dv * dv) <= radius_uv * radius_uv] = 0.0


# -- Motion helpers -----------------------------------------------------------
def _smoothstep(x):
    return x * x * (3.0 - 2.0 * x)


class Segment:
    """A timed joint-space segment of the playback schedule."""

    def __init__(self, q_path, duration, label, ease=False, drawing=False):
        self.q = np.atleast_2d(np.asarray(q_path, dtype=np.float64))
        self.duration = float(duration)
        self.label = label
        self.ease = ease
        self.drawing = drawing

    def at(self, t):
        """Joint command and path fraction at local time ``t``."""
        s = 0.0 if self.duration <= 0 else np.clip(t / self.duration, 0.0, 1.0)
        if self.ease:
            s = _smoothstep(s)
        if len(self.q) == 1:
            return self.q[0], s
        pos = s * (len(self.q) - 1)
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, len(self.q) - 1)
        a = pos - i0
        return (1.0 - a) * self.q[i0] + a * self.q[i1], s


class Schedule:
    """Cyclic sequence of segments; the first ``n_intro`` run only once."""

    def __init__(self, segments, n_intro):
        self.segments = segments
        self.n_intro = n_intro
        self.intro_time = sum(s.duration for s in segments[:n_intro])
        self.cycle_time = sum(s.duration for s in segments[n_intro:])

    def at(self, t):
        if t < self.intro_time:
            for seg in self.segments[:self.n_intro]:
                if t < seg.duration:
                    return seg, seg.at(t)
                t -= seg.duration
            seg = self.segments[self.n_intro - 1]
            return seg, seg.at(seg.duration)

        t = (t - self.intro_time) % max(self.cycle_time, 1e-9)
        for seg in self.segments[self.n_intro:]:
            if t < seg.duration:
                return seg, seg.at(t)
            t -= seg.duration
        seg = self.segments[-1]
        return seg, seg.at(seg.duration)

    def single_pass_time(self):
        """End of the first pass: intro, one drawing run, and the retract."""
        total = self.intro_time
        for seg in self.segments[self.n_intro:]:
            total += seg.duration
            if seg.label == 'retract':
                break
        return total


def _joint_lerp(q_a, q_b, n=2):
    return np.linspace(q_a, q_b, n)


def servo_lead_time(model, n_joints=N_ARM_JOINTS):
    """
    Look-ahead that cancels the position servos' velocity lag.

    The Panda actuators are PD servos (``gainprm=kp``, ``biasprm=[0,-kp,-kd]``),
    so a joint moving at ``qd`` settles ``kd/kp * qd`` behind its command.
    Commanding ``q_des(t + kd/kp)`` cancels that to first order -- which is free
    here because the whole joint path is planned ahead of time.  Measured on the
    'A' trajectory this cuts the tip lag from 28 mm to 6 mm.
    """
    kp = np.asarray(model.actuator_gainprm[:n_joints, 0], dtype=np.float64)
    kd = -np.asarray(model.actuator_biasprm[:n_joints, 2], dtype=np.float64)
    valid = kp > 0
    if not valid.any():
        return 0.0
    return float(np.median(kd[valid] / kp[valid]))


def _cartesian_segment(ik, q_seed, p_from, p_to, n=40, tool_axis=TOOL_AXIS):
    """IK a straight Cartesian line, continuing from ``q_seed``."""
    qs = np.empty((n, ik.n))
    q = np.asarray(q_seed, dtype=np.float64).copy()
    for i, a in enumerate(np.linspace(0.0, 1.0, n)):
        target = (1.0 - a) * p_from + a * p_to
        q, _, _ = ik.solve(q, target, tool_axis, iterations=30)
        qs[i] = q
    return qs


# -- Trajectory planning (with on-disk cache) ---------------------------------
def _cache_key(shape_name, standoff, axis, densify, n_pts):
    payload = "|".join(str(x) for x in (
        shape_name, round(standoff, 6), axis, densify, n_pts,
        BOARD_FACE_X, BOARD_SIZE, tuple(BOARD_ORIGIN), ERASER_TIP, SITE_RES))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def plan_playback(model, shape_name, standoff=DEFAULT_STANDOFF, axis='Z',
                  densify=4, restarts=8, use_cache=True):
    """
    Plan everything needed to replay ``shape_name``: the drawing joint path and
    the approach/retract segments.  Returns ``(schedule, path_pos, report)``.
    """
    uv, dt = board_mod.load_trajectory_uv(shape_name)
    n_raw = len(uv)
    idx = np.linspace(0, n_raw - 1, max(densify * n_raw, n_raw))
    uv_dense = np.stack([np.interp(idx, np.arange(n_raw), uv[:, k]) for k in range(2)], axis=1)
    path_pos = board_mod.uv_to_world(uv_dense, standoff)

    ik = TaskPriorityIK(model, tool_axis=TOOL_AXIS)

    key = _cache_key(shape_name, standoff, axis, densify, n_raw)
    cache_file = os.path.join(CACHE_DIR, "qpath_%s_%s.npz" % (shape_name, key))
    q_draw = None
    if use_cache and os.path.exists(cache_file):
        try:
            blob = np.load(cache_file)
            q_draw = blob['q_draw']
            report = {k: float(blob[k]) for k in
                      ('max_pos_err', 'mean_pos_err', 'max_axis_err',
                       'mean_axis_err', 'max_joint_step')}
            report['n_waypoints'] = len(q_draw)
            report['cached'] = True
            print("  loaded cached joint path (%s)" % os.path.basename(cache_file))
        except Exception as exc:                      # corrupt/stale cache
            print("  ignoring cache (%s)" % exc)
            q_draw = None

    if q_draw is None:
        print("  planning joint trajectory over %d waypoints ..." % len(path_pos))
        t0 = time.perf_counter()
        q_draw, report = plan_joint_path(ik, path_pos, TOOL_AXIS, restarts=restarts)
        report['cached'] = False
        print("  planned in %.1f s" % (time.perf_counter() - t0))
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez_compressed(cache_file, q_draw=q_draw,
                                **{k: v for k, v in report.items()
                                   if k not in ('n_waypoints', 'cached')})

    # Home pose from the model keyframe.
    scratch = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, scratch, key_id)
    q_home = scratch.qpos[:N_ARM_JOINTS].copy()

    back = np.array([RETRACT, 0.0, 0.0])
    p_first, p_last = path_pos[0], path_pos[-1]
    q_pre, _, _ = ik.solve(q_draw[0], p_first - back, TOOL_AXIS, iterations=300)
    q_post, _, _ = ik.solve(q_draw[-1], p_last - back, TOOL_AXIS, iterations=300)

    draw_time = n_raw * dt
    segments = [
        Segment(_joint_lerp(q_home, q_pre), 2.5, 'approach', ease=True),
        Segment(_cartesian_segment(ik, q_pre, p_first - back, p_first), 1.0, 'engage'),
        Segment(q_draw, draw_time, 'draw', drawing=True),
        Segment(_cartesian_segment(ik, q_draw[-1], p_last, p_last - back), 0.8, 'retract'),
        Segment(_joint_lerp(q_post, q_pre), 1.5, 'return', ease=True),
        Segment(_cartesian_segment(ik, q_pre, p_first - back, p_first), 1.0, 'engage'),
    ]
    # The intro (approach + first engage) runs once; the rest loops.
    return Schedule(segments, n_intro=2), path_pos, report


# -- Standalone playback ------------------------------------------------------
def run_shape_playback(shape_name='A', speed=1.0, erase=False, loop=True,
                       standoff=DEFAULT_STANDOFF, axis='Z', show_path=True,
                       use_cache=True, restarts=8, fps=60.0):
    print("Building MuJoCo scene ...")
    model = _build_scene(_HERE, orthogonal_axis=axis)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    print("Planning '%s' ..." % shape_name)
    schedule, path_pos, report = plan_playback(
        model, shape_name, standoff=standoff, axis=axis,
        use_cache=use_cache, restarts=restarts)
    print("  planned tip accuracy: max %.1f um (mean %.1f um) | "
          "tool tilt max %.2f deg (mean %.2f deg)"
          % (report['max_pos_err'] * 1e6, report['mean_pos_err'] * 1e6,
             np.degrees(report['max_axis_err']), np.degrees(report['mean_axis_err'])))

    density = board_mod.load_density(shape_name, SITE_RES)
    if density is None:
        print("  no density parameters stored for %r -- board stays blank" % shape_name)
        density = np.zeros((SITE_RES, SITE_RES))
    painter = BoardPainter(model)
    painter.paint(density)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eraser_tip")
    dt = model.opt.timestep
    n_sub = max(1, int(round((1.0 / fps) / dt)))
    erase_radius_uv = ERASER_RADIUS / BOARD_SIZE
    lead = servo_lead_time(model)

    # Start already settled in the home pose.
    seg, (q_cmd, _) = schedule.at(0.0)
    data.ctrl[:N_ARM_JOINTS] = q_cmd
    data.ctrl[N_ARM_JOINTS] = GRIPPER_CTRL
    mujoco.mj_forward(model, data)

    sim_t = 0.0
    stop_at = None if loop else schedule.single_pass_time()
    print("Launching viewer -- %s: %.1f s intro + %.1f s per pass (speed x%.2f, "
          "servo lead %.0f ms)%s"
          % (shape_name, schedule.intro_time / speed, schedule.cycle_time / speed,
             speed, lead * 1e3, ", erasing ON" if erase else ""))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _draw_reference_path(viewer, path_pos if show_path else None)
        last_label = None
        finished = False
        while viewer.is_running():
            frame_start = time.perf_counter()

            for _ in range(n_sub):
                seg, _ = schedule.at(sim_t * speed)
                # Command the pose the arm should be in `lead` seconds from now.
                _, (q_cmd, _frac) = schedule.at((sim_t + lead) * speed)
                data.ctrl[:N_ARM_JOINTS] = q_cmd
                data.ctrl[N_ARM_JOINTS] = GRIPPER_CTRL
                data.qfrc_applied[:] = data.qfrc_bias      # gravity compensation
                mujoco.mj_step(model, data)
                sim_t += dt
                if stop_at is not None and sim_t * speed >= stop_at:
                    finished = True
                    break

            tip = data.site_xpos[tip_id].copy()
            if erase and seg.drawing:
                erase_disc(density, board_mod.world_to_uv(tip), erase_radius_uv)
                painter.paint(density)

            if seg.label != last_label:
                print("  [%6.2f s] %s" % (sim_t, seg.label))
                last_label = seg.label

            if show_path:
                _update_tip_marker(viewer, tip)
            viewer.sync()

            if finished:
                print("Done -- close the viewer window to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(1.0 / fps)
                break

            sleep = 1.0 / fps - (time.perf_counter() - frame_start)
            if sleep > 0:
                time.sleep(sleep)


def _draw_reference_path(viewer, path_pos):
    """Show the reference trajectory as green dots plus a live tip marker."""
    scn = getattr(viewer, 'user_scn', None)
    if scn is None:
        return
    scn.ngeom = 0
    if path_pos is not None:
        stride = max(1, len(path_pos) // 400)
        for pt in path_pos[::stride]:
            if scn.ngeom >= scn.maxgeom - 1:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=[0.0025, 0, 0], pos=pt, mat=np.eye(3).flatten(),
                                rgba=np.array([0.0, 0.78, 0.33, 0.85]))
            scn.ngeom += 1
    # Reserve the last slot for the moving marker.
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.008, 0, 0], pos=np.zeros(3), mat=np.eye(3).flatten(),
                        rgba=np.array([0.08, 0.40, 0.75, 1.0]))
    scn.ngeom += 1


def _update_tip_marker(viewer, tip):
    scn = getattr(viewer, 'user_scn', None)
    if scn is not None and scn.ngeom > 0:
        scn.geoms[scn.ngeom - 1].pos[:] = tip


# -- Live mode driven by the 2D control centre --------------------------------
def run_mujoco_sim(agent_info_array, shared_truth, truth_res=TRUTH_RES,
                   axis='Z', standoff=DEFAULT_STANDOFF, fps=60.0,
                   max_tip_speed=MAX_TIP_SPEED):
    """
    Mirror the 2D agent from ``interactive_sim.py`` with the Panda.

    The 2D agent teleports between planning rounds and can sweep far faster
    than the arm.  Its position is therefore treated as a *goal* that the tool
    chases at no more than ``max_tip_speed`` m/s: the arm trails behind on fast
    moves instead of being handed a command it cannot reach, which is what used
    to throw the tip tens of millimetres off the board plane.

    The joint command is rate-limited to ``JOINT_VMAX`` on top of that.  Online
    IK continuation occasionally has to hop to a different branch -- most
    reliably in the low centre of the board -- and without the limit that hop
    is issued as a single ~46 rad/s step, which shows up as the tool briefly
    leaving the board.  Ramping it out cuts the worst excursion from 95 mm to
    26 mm and keeps the tip behind the board surface throughout.
    """
    model = _build_scene(_HERE, orthogonal_axis=axis)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    painter = BoardPainter(model)
    ik = TaskPriorityIK(model, tool_axis=TOOL_AXIS)

    # Settle on the board centre before following the agent.
    centre = board_mod.uv_to_world(np.array([0.5, 0.5]), standoff)
    q_cmd, pos_err, _ = ik.solve_from_restarts(centre, restarts=16)
    if pos_err > 1e-3:
        print("[mujoco] warning: could not reach the board centre (%.1f mm)" % (pos_err * 1e3))
    data.qpos[:N_ARM_JOINTS] = q_cmd
    data.ctrl[:N_ARM_JOINTS] = q_cmd
    data.ctrl[N_ARM_JOINTS] = GRIPPER_CTRL
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    n_sub = max(1, int(round((1.0 / fps) / dt)))
    frame_dt = n_sub * dt
    max_step_uv = max_tip_speed * frame_dt / BOARD_SIZE
    lead = servo_lead_time(model)
    cursor = np.array([0.5, 0.5])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()

            with agent_info_array.get_lock():
                goal = np.array([float(agent_info_array[0]),
                                 float(agent_info_array[1])])
            step = np.clip(goal, 0.0, 1.0) - cursor
            dist = np.linalg.norm(step)
            if dist > max_step_uv:
                step *= max_step_uv / dist
            cursor += step

            # Aim where the cursor will be one servo time-constant from now, so
            # the arm sits on the agent instead of trailing it (see
            # ``servo_lead_time``).  With a stationary cursor this is a no-op.
            aim = np.clip(cursor + (lead / frame_dt) * step, 0.0, 1.0)
            target = board_mod.uv_to_world(aim, standoff)
            q_next, _, _ = ik.solve(q_cmd, target, TOOL_AXIS, iterations=10)

            # Never ask for more joint speed than the real arm has.
            delta = q_next - q_cmd
            excess = np.abs(delta) / (JOINT_VMAX * frame_dt)
            if excess.max() > 1.0:
                delta /= excess.max()
            q_cmd = q_cmd + delta

            for _ in range(n_sub):
                data.ctrl[:N_ARM_JOINTS] = q_cmd
                data.ctrl[N_ARM_JOINTS] = GRIPPER_CTRL
                data.qfrc_applied[:] = data.qfrc_bias
                mujoco.mj_step(model, data)

            if shared_truth is not None:
                with shared_truth.get_lock():
                    grid = np.frombuffer(shared_truth.get_obj(), dtype=np.float64)
                    grid = grid.reshape((truth_res, truth_res)).copy()
                peak = grid.max()
                if peak > 0:
                    grid /= peak
                painter.paint(np.clip(grid, 0.0, 1.0))

            viewer.sync()
            sleep = 1.0 / fps - (time.perf_counter() - frame_start)
            if sleep > 0:
                time.sleep(sleep)


# -- CLI ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Replay a stored ergodic trajectory with the Panda eraser.")
    p.add_argument('--shape', default='A',
                   help="shape name in ergodic_dataset_robot.db (default: A)")
    p.add_argument('--speed', type=float, default=1.0,
                   help="playback speed multiplier (default: 1.0)")
    p.add_argument('--erase', action='store_true',
                   help="wipe the density away where the eraser passes")
    p.add_argument('--once', action='store_true', help="play a single pass, do not loop")
    p.add_argument('--axis', default='Z', choices=sorted(_AXES),
                   help="hand-frame axis the eraser is mounted on (default: Z)")
    p.add_argument('--standoff', type=float, default=DEFAULT_STANDOFF,
                   help="gap between eraser tip and board surface in m")
    p.add_argument('--no-path', action='store_true', help="hide the reference trajectory")
    p.add_argument('--no-cache', action='store_true', help="always re-plan the joint path")
    p.add_argument('--restarts', type=int, default=8, help="IK seed restarts (default: 8)")
    p.add_argument('--list-shapes', action='store_true', help="print available shapes and exit")
    args = p.parse_args(argv)

    if args.list_shapes:
        names = board_mod.list_shapes()
        print("%d shapes:" % len(names))
        for i in range(0, len(names), 8):
            print("  " + "  ".join("%-22s" % n for n in names[i:i + 8]))
        return 0

    run_shape_playback(shape_name=args.shape, speed=args.speed, erase=args.erase,
                       loop=not args.once, standoff=args.standoff, axis=args.axis,
                       show_path=not args.no_path, use_cache=not args.no_cache,
                       restarts=args.restarts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
