"""
run_mujoco_3d.py
================
Live MuJoCo view of the 3D active-sensing simulation.

The control centre (``interactive_sim_3d.py``) writes into shared memory:

``fig_sel``  [version, split_code, entry_index]   which held-out figure to show
``agent``    [x, y, z, ax, ay, az, radius]        tool tip / tool axis, figure frame
``path``     see ``PATH_*`` below                 trajectory in the figure frame

This process builds the scene for the selected figure (mesh + density, placed
so that the lit side faces the robot), lets the Panda follow the agent with the
tool pointing along the inward surface normal, and draws the trajectory on the
figure: earlier rounds green, the current round red.  When the figure changes
the viewer is relaunched.
"""

import contextlib
import json
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPL3 = os.path.dirname(_HERE)
if _EXPL3 not in sys.path:
    sys.path.insert(0, _EXPL3)

from mujoco_3d import scene3d as S                         # noqa: E402
from mujoco_3d.ik import TaskPriorityIK                    # noqa: E402

# -- shared path layout (figure frame, 3 floats per point) -----------------------
PATH_HEADER = 4
PATH_MAX_PAST = 700
PATH_MAX_CURRENT = 300
PATH_MAX_PREVIEW = 200
PATH_ARRAY_SIZE = PATH_HEADER + 3 * (PATH_MAX_PAST + PATH_MAX_CURRENT + PATH_MAX_PREVIEW)
FIG_SEL_SIZE = 4
AGENT_SIZE = 8
SPLITS = ('val_form', 'val_beides', 'val_flaeche', 'volume')


class LiveArm3D:
    """Follows a goal (tip position + tool axis, world frame) with the Panda.

    The goal is chased at no more than ``max_tip_speed`` (m/s) and the tool
    axis turns at no more than ``MAX_AXIS_RATE`` (rad/s); the joint command is
    limited to the Panda's joint velocities.  IK runs from the previous command
    (continuation), tool axis first, then position."""

    def __init__(self, model, jmap, start_pos, start_axis, fps=60.0, ik_iters=6):
        self.model = model
        self.jmap = jmap
        self.data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        mujoco.mj_resetDataKeyframe(model, self.data, key)
        self.ik = jmap.ik
        self.q_home = S.home_pose(model)
        self.ik.posture = self.q_home.copy()
        self.tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'eraser_tip')
        self.ik_iters = ik_iters
        dt = model.opt.timestep
        self.n_sub = max(1, int(round((1.0 / fps) / dt)))
        self.frame_dt = self.n_sub * dt
        self.lead = S.servo_lead_time(model)
        self.q_prev = None

        # Start already on the figure: the home pose is far from the surface and
        # the continuation IK can lock onto a bad branch on the way there.
        axis = np.asarray(start_axis, float) / np.linalg.norm(start_axis)
        target = np.asarray(start_pos, float) - axis * S.STANDOFF
        q, pe, ae = self.ik.solve(jmap.query(start_pos), target, axis, iterations=300, max_step=0.3)
        if pe > 1e-3 or ae > np.radians(2.0):
            q, pe, ae = self.ik.solve_from_restarts(target, axis, restarts=24, iterations=300,
                                                    max_step=0.3)
        self.start_error = (pe, ae)
        self.q_cmd = q.copy()
        self.data.qpos[:S.N_ARM_JOINTS] = q
        self.data.ctrl[:S.N_ARM_JOINTS] = q
        self.data.ctrl[S.N_ARM_JOINTS] = S.GRIPPER_CTRL
        mujoco.mj_forward(model, self.data)
        self.cursor = np.asarray(start_pos, float).copy()   # surface point being followed
        self.axis = axis.copy()                              # tool axis being followed

    def step(self, goal_pos, goal_axis, max_tip_speed=S.MAX_TIP_SPEED):
        step = np.asarray(goal_pos, float) - self.cursor
        dist = np.linalg.norm(step)
        lim = max_tip_speed * self.frame_dt
        if dist > lim:
            step *= lim / dist
        self.cursor = self.cursor + step

        # turn the tool axis towards the goal axis at a limited rate
        ga = np.asarray(goal_axis, float)
        ga = ga / max(np.linalg.norm(ga), 1e-9)
        ang = float(np.arccos(np.clip(self.axis @ ga, -1.0, 1.0)))
        lim_a = S.MAX_AXIS_RATE * self.frame_dt
        if ang > lim_a:
            R = S.rotation_between(self.axis, ga)
            # rotation by the fraction lim_a/ang about the same axis
            w = np.cross(self.axis, ga)
            s = np.linalg.norm(w)
            if s > 1e-9:
                k = w / s
                th = lim_a
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            self.axis = R @ self.axis
        else:
            self.axis = ga
        self.axis /= np.linalg.norm(self.axis)

        target = self.cursor - self.axis * S.STANDOFF
        # Continue from the previous command (smooth); only if that gets stuck (joint
        # limit, wrong branch) restart from the validated joint map and take the better.
        kw = dict(iterations=self.ik_iters, max_step=0.3, priority='axis', tol_pos=3e-4, tol_axis=0.005)
        q_next, pe, ae = self.ik.solve(self.q_cmd, target, self.axis, **kw)
        if pe > 3e-3 or ae > np.radians(2.0):
            q_map, pe2, ae2 = self.ik.solve(self.jmap.query(self.cursor), target, self.axis, **kw)
            if pe2 + 0.02 * ae2 < pe + 0.02 * ae:
                q_next = q_map
        delta = q_next - self.q_cmd
        excess = np.abs(delta) / (S.JOINT_VMAX * self.frame_dt)
        if excess.max() > 1.0:
            delta /= excess.max()
        q_old = self.q_cmd
        self.q_cmd = self.q_cmd + delta
        # command where the joints should be one servo time-constant from now
        q_ctrl = self.q_cmd + (self.lead / self.frame_dt) * (self.q_cmd - q_old)

        for _ in range(self.n_sub):
            self.data.ctrl[:S.N_ARM_JOINTS] = q_ctrl
            self.data.ctrl[S.N_ARM_JOINTS] = S.GRIPPER_CTRL
            self.data.qfrc_applied[:] = self.data.qfrc_bias       # gravity compensation
            mujoco.mj_step(self.model, self.data)

    def tool_state(self):
        pos = self.data.site_xpos[self.tip_id].copy()
        z = self.data.site_xmat[self.tip_id].reshape(3, 3)[:, 2]
        return pos, z


class PathOverlay3D:
    """Draws the trajectory on the figure: green past, red current round (solid
    for the part driven so far, faint for the rest of the plan)."""

    PAST_RGBA = (0.0, 0.78, 0.33, 0.95)
    CURRENT_RGBA = (1.00, 0.09, 0.27, 1.00)
    PREVIEW_RGBA = (1.00, 0.09, 0.27, 0.45)
    RADIUS = 0.0022
    LIFT = 0.004                    # above the surface (m)
    MAX_JUMP = 0.12                 # do not join points further apart (m)
    MIN_INTERVAL = 0.05

    def __init__(self, shared_path, placement, lit):
        self.shared, self.pl, self.lit = shared_path, placement, lit
        self._last, self._t = None, 0.0

    def _read(self):
        with self.shared.get_lock():
            raw = np.frombuffer(self.shared.get_obj(), dtype=np.float64).copy()
        n = [int(np.clip(raw[0], 0, PATH_MAX_PAST)), int(np.clip(raw[1], 0, PATH_MAX_CURRENT)),
             int(np.clip(raw[2], 0, PATH_MAX_PREVIEW))]
        off = [PATH_HEADER, PATH_HEADER + 3 * PATH_MAX_PAST,
               PATH_HEADER + 3 * (PATH_MAX_PAST + PATH_MAX_CURRENT)]
        return [raw[o:o + 3 * k].reshape(-1, 3) for o, k in zip(off, n)]

    def _world(self, pts):
        """Figure-frame points -> world, lifted off the lit surface along its normal."""
        if len(pts) == 0:
            return pts
        p, nrm = self.lit.project(pts)
        return self.pl.point(p) + self.pl.direction(nrm) * self.LIFT

    def draw(self, viewer):
        scn = getattr(viewer, 'user_scn', None)
        now = time.perf_counter()
        if scn is None or now - self._t < self.MIN_INTERVAL:
            return
        past, cur, pre = self._read()
        sig = (past.tobytes(), cur.tobytes(), pre.tobytes())
        if sig == self._last:
            return
        self._last, self._t = sig, now
        if len(past) and len(cur):
            cur = np.vstack([past[-1:], cur])
        tail = cur if len(cur) else past
        if len(tail) and len(pre):
            pre = np.vstack([tail[-1:], pre])
        lock = getattr(viewer, 'lock', None)
        with (lock() if lock is not None else contextlib.nullcontext()):
            scn.ngeom = 0
            for pts, rgba in ((past, self.PAST_RGBA), (cur, self.CURRENT_RGBA),
                              (pre, self.PREVIEW_RGBA)):
                self._lines(scn, self._world(pts), rgba)

    def _lines(self, scn, pts, rgba):
        for k in range(len(pts) - 1):
            if scn.ngeom >= scn.maxgeom:
                return
            if np.linalg.norm(pts[k + 1] - pts[k]) > self.MAX_JUMP:
                continue
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, self.RADIUS,
                                 pts[k], pts[k + 1])
            scn.ngeom += 1


# -- placement cache ----------------------------------------------------------------
def _placement_for(entry, figure, lit, base_model):
    os.makedirs(S.CACHE_DIR, exist_ok=True)
    key = entry.get('placement_key') or '%s__%s__%s' % (entry['shape'], entry['surface'],
                                                       entry.get('view_id', 0))
    key = '%s__c%d_jm3' % (key, int(S.MAX_TILT_DEG))
    if getattr(figure, 'shrink', 1.0) != 1.0:
        key += '_s%.2f' % figure.shrink
    path = os.path.join(S.CACHE_DIR, 'placement_%s.json' % key)
    centre = 0.5 * (figure.pts.min(axis=0) + figure.pts.max(axis=0))
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            return S.Placement(S.figure_rotation(figure.view), d['scale'], centre, d['shift']), d['reach']
        except Exception:
            pass
    pl, frac = S.choose_placement(base_model, figure, lit)
    json.dump({'scale': pl.scale, 'shift': pl.shift.tolist(), 'reach': frac}, open(path, 'w'))
    return pl, frac


def _read_sel(fig_sel):
    with fig_sel.get_lock():
        return int(fig_sel[0]), int(fig_sel[1]), int(fig_sel[2])


def run_mujoco_3d(fig_sel, agent, shared_path, fps=60.0):
    import paths3d
    paths3d.setup()
    import figures3d as F
    from surface3d import make_lit

    base_model = S.build_spec().compile()
    entry_cache = {}
    work_dir = os.path.join(S.CACHE_DIR, 'scenes')
    version = None
    while True:
        version, split_code, idx = _read_sel(fig_sel)
        split = SPLITS[split_code]
        if split not in entry_cache:
            entry_cache[split] = F.list_entries(split)
        entry = entry_cache[split][idx]
        print('[mujoco3d] building %s on %s ...' % (entry['shape'], entry['surface']), flush=True)
        fig = F.build_figure(entry)
        lit = make_lit(fig)
        placement, reach = _placement_for(entry, fig, lit, base_model)
        print('[mujoco3d] scale %.2f at %s, %.0f%% of the lit surface reachable'
              % (placement.scale, np.round(placement.shift, 2), 100 * reach), flush=True)
        model = S.build_model(fig, placement, work_dir)
        with agent.get_lock():
            a0 = np.frombuffer(agent.get_obj(), dtype=np.float64).copy()
        start_axis = S.limit_axes(placement.direction(a0[3:6]), -S.FACE_DIR)
        if np.linalg.norm(start_axis) < 1e-6:
            start_axis = -S.FACE_DIR
        from mujoco_3d.joint_map3d import JointMap3D
        t0 = time.perf_counter()
        ik = TaskPriorityIK(model, tool_axis=(0, 0, 1))
        ik.posture = S.home_pose(model).copy()
        pts_w, axes_w = S.surface_samples(lit, placement, 1500)
        jmap = JointMap3D(ik, pts_w, axes_w, S.home_pose(model), n_restarts=3)
        print('[mujoco3d] joint map: %.0f%% of %d surface nodes valid (%.1fs)'
              % (100 * jmap.valid_fraction, len(pts_w), time.perf_counter() - t0), flush=True)
        arm = LiveArm3D(model, jmap, placement.point(a0[:3]), start_axis, fps=fps)
        print('[mujoco3d] arm placed at the start: tip %.1f mm, tilt %.1f deg off'
              % (arm.start_error[0] * 1e3, np.degrees(arm.start_error[1])), flush=True)
        overlay = PathOverlay3D(shared_path, placement, lit)
        changed = False
        with mujoco.viewer.launch_passive(model, arm.data) as viewer:
            viewer.cam.lookat[:] = placement.shift
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, 120.0, -15.0
            while viewer.is_running():
                t0 = time.perf_counter()
                v, _, _ = _read_sel(fig_sel)
                if v != version:
                    changed = True
                    break
                with agent.get_lock():
                    a = np.frombuffer(agent.get_obj(), dtype=np.float64).copy()
                goal = placement.point(a[:3])
                axis = placement.direction(a[3:6])
                if np.linalg.norm(axis) < 1e-6:
                    axis = placement.direction(-fig.view)
                axis = S.limit_axes(axis, -S.FACE_DIR)
                arm.step(goal, axis)
                tip, _ = arm.tool_state()
                lag = float(np.linalg.norm(tip - (goal - axis / np.linalg.norm(axis) * S.STANDOFF)))
                with agent.get_lock():
                    agent[7] = lag                  # the control centre paces the agent by this
                overlay.draw(viewer)
                viewer.sync()
                sleep = 1.0 / fps - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        if not changed:
            return                                 # the user closed the window
