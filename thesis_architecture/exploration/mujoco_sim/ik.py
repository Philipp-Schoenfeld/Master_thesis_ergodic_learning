"""
ik.py
=====
Task-priority differential inverse kinematics for the eraser tip of the Panda.

Why not a plain 6D pose IK
--------------------------
The eraser is a cylinder, so its rotation about its own axis is meaningless.
Constraining the full SE(3) pose therefore wastes a joint and drives the wrist
straight into ``joint7``/``joint6`` limits -- which is exactly what used to make
the arm freeze roughly 180 deg away from the target and need a random-jitter
escape hack.

This solver instead uses a strict priority stack on the 7 arm joints:

    1. tip **position**            (3 DOF, hard)
    2. tool **axis** vs board normal (2 effective DOF, in the null space of 1)
    3. stay away from joint limits   (remainder)

Two details matter for it to actually converge:

* ``mju_subQuat`` / axis errors have to be expressed in the **world** frame,
  because the rotational rows of ``mj_jacSite`` are world-frame.  Mixing local
  error with a world Jacobian is what produced the stuck-at-180-deg behaviour.
* Joints that saturate must be **locked out and the step re-solved**.  Simply
  clipping ``q + dq`` corrupts the step direction and stalls the solve several
  millimetres away from a perfectly reachable target.
"""

import numpy as np
import mujoco

#: A joint closer than this to a limit counts as saturated.
_LIMIT_EPS = 1e-4


def axis_error(z_cur, z_target):
    """World-frame rotation vector that rotates ``z_cur`` onto ``z_target``."""
    cross = np.cross(z_cur, z_target)
    sin = np.linalg.norm(cross)
    cos = float(z_cur @ z_target)
    if sin < 1e-12:
        # Aligned, or exactly anti-parallel: pick an arbitrary perpendicular.
        return np.zeros(3) if cos > 0.0 else np.array([0.0, 0.0, np.pi])
    return (np.arctan2(sin, cos) / sin) * cross


class TaskPriorityIK:
    """
    Differential IK on a MuJoCo site.

    The solver keeps its own :class:`mujoco.MjData` so it can evaluate forward
    kinematics at candidate configurations without touching the simulated
    state.  ``solve`` is cheap enough (~0.1 ms per iteration) to run online.
    """

    def __init__(self, model, site_name='eraser_tip', n_joints=7,
                 tool_axis=(1.0, 0.0, 0.0)):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError("site %r not found in model" % site_name)
        self.n = n_joints
        self.tool_axis = np.asarray(tool_axis, dtype=np.float64)

        self.lo = model.jnt_range[:n_joints, 0].copy()
        self.hi = model.jnt_range[:n_joints, 1].copy()
        self.mid = 0.5 * (self.lo + self.hi)
        self.span = self.hi - self.lo

        self._jac = np.zeros((6, model.nv))
        self._eye = np.eye(n_joints)

    # -- forward kinematics -------------------------------------------------
    def _fk(self, q):
        self.data.qpos[:self.n] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def tip_pose(self, q):
        """(position, rotation matrix) of the tip site at configuration ``q``."""
        self._fk(q)
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def errors(self, q, target_pos, tool_axis=None):
        """(position error [m], tool-axis error [rad]) at ``q``."""
        pos, rot = self.tip_pose(q)
        axis = self.tool_axis if tool_axis is None else tool_axis
        return (float(np.linalg.norm(target_pos - pos)),
                float(np.linalg.norm(axis_error(rot[:, 2], axis))))

    # -- one damped step ----------------------------------------------------
    def _step(self, q, e_pos, e_axis, jac_p, jac_r, free, k_limit, rcond, damp_r):
        """Priority-stacked joint step restricted to the ``free`` joints."""
        mask = np.diag(free.astype(np.float64))
        jp, jr = jac_p @ mask, jac_r @ mask

        jp_pinv = np.linalg.pinv(jp, rcond=rcond)        # truncated -> exact projector
        dq = jp_pinv @ e_pos
        null_1 = self._eye - jp_pinv @ jp

        jr_n = jr @ null_1
        jr_n_pinv = jr_n.T @ np.linalg.inv(jr_n @ jr_n.T + damp_r * np.eye(3))
        dq = dq + null_1 @ (jr_n_pinv @ (e_axis - jr @ dq))

        null_2 = null_1 @ (self._eye - jr_n_pinv @ jr_n)
        dq = dq + null_2 @ (-k_limit * (q - self.mid) / self.span ** 2)
        return mask @ dq

    def solve(self, q_init, target_pos, tool_axis=None, iterations=25,
              max_step=0.15, k_limit=0.5, rcond=1e-3, damp_r=1e-2,
              max_pos_err=0.05, max_axis_err=0.1):
        """
        Iterate towards ``target_pos`` with the tool pointing along ``tool_axis``.

        Returns ``(q, pos_err, axis_err)``.  ``q_init`` is used as the seed, so
        calling this along a dense path performs a continuation and keeps the
        arm on one smooth IK branch.
        """
        axis = self.tool_axis if tool_axis is None else np.asarray(tool_axis, float)
        q = np.clip(np.asarray(q_init, dtype=np.float64)[:self.n], self.lo, self.hi)

        for _ in range(iterations):
            self._fk(q)
            rot = self.data.site_xmat[self.site_id].reshape(3, 3)
            e_pos = np.clip(target_pos - self.data.site_xpos[self.site_id],
                            -max_pos_err, max_pos_err)
            e_axis = np.clip(axis_error(rot[:, 2], axis), -max_axis_err, max_axis_err)

            mujoco.mj_jacSite(self.model, self.data, self._jac[:3], self._jac[3:],
                              self.site_id)
            jac_p = self._jac[:3, :self.n].copy()
            jac_r = self._jac[3:, :self.n].copy()

            # Clamping loop: lock joints the step would push past their limit
            # and re-solve, so the remaining joints take over the task.
            free = np.ones(self.n, dtype=bool)
            dq = np.zeros(self.n)
            for _ in range(self.n):
                dq = self._step(q, e_pos, e_axis, jac_p, jac_r, free,
                                k_limit, rcond, damp_r)
                blocked = (((q <= self.lo + _LIMIT_EPS) & (dq < 0.0))
                           | ((q >= self.hi - _LIMIT_EPS) & (dq > 0.0))) & free
                if not blocked.any():
                    break
                free &= ~blocked
                if not free.any():
                    dq = np.zeros(self.n)
                    break

            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm
            q = np.clip(q + dq, self.lo, self.hi)

        pos_err, axis_err = self.errors(q, target_pos, axis)
        return q, pos_err, axis_err

    # -- global seeding -----------------------------------------------------
    def solve_from_restarts(self, target_pos, tool_axis=None, restarts=12,
                            iterations=400, rng=None, **kw):
        """Best-of-N random-restart solve, for when no good seed is available."""
        rng = np.random.default_rng(0) if rng is None else rng
        best = None
        for _ in range(restarts):
            seed = rng.uniform(self.lo, self.hi)
            q, pe, ae = self.solve(seed, target_pos, tool_axis,
                                   iterations=iterations, **kw)
            score = pe + 0.02 * ae
            if best is None or score < best[0]:
                best = (score, q, pe, ae)
        return best[1], best[2], best[3]


#: A branch tracking better than this counts as exact; below it the residual is
#: solver noise and should not decide which branch wins.
_EXACT_POS = 5e-4


def _branch_score(pos_max, axis_max, jump):
    """
    Rank IK branches: reach the path first, then keep the tool flat on the
    board, then move smoothly.  Position error is only a final tie-break --
    every usable branch tracks to well under a millimetre, so ranking on it
    directly would let numerical noise pick a badly tilted wrist.
    """
    return (pos_max > _EXACT_POS, round(axis_max, 3), round(jump, 3), pos_max)


def _track(ik, q0, path_pos, tool_axis, iterations):
    """Continuation along ``path_pos`` starting from ``q0``."""
    q = np.asarray(q0, dtype=np.float64).copy()
    qs = np.empty((len(path_pos), ik.n))
    pos_err = np.empty(len(path_pos))
    axis_err = np.empty(len(path_pos))
    for i, target in enumerate(path_pos):
        q, pe, ae = ik.solve(q, target, tool_axis, iterations=iterations)
        qs[i], pos_err[i], axis_err[i] = q, pe, ae
    jump = float(np.abs(np.diff(qs, axis=0)).max()) if len(qs) > 1 else 0.0
    score = _branch_score(float(pos_err.max()), float(axis_err.max()), jump)
    return score, qs, pos_err, axis_err


def plan_joint_path(ik, path_pos, tool_axis=None, restarts=8, seed=0,
                    iterations=25, finalists=3, screen_stride=8, verbose=True):
    """
    Plan one continuous joint trajectory through ``path_pos`` (N, 3).

    Random seeds land on very different IK branches, and only some of them can
    carry the tool across the whole board without stalling on a joint limit.
    So: screen every seed on a strided subsample of the path (cheap), then
    re-track the most promising ones at full resolution and keep the branch
    with the smallest worst-case position error (ties broken by tool tilt, then
    by joint smoothness).

    Returns ``(q_path, report)`` where ``q_path`` is (N, n_joints).
    """
    rng = np.random.default_rng(seed)
    coarse = path_pos[::max(1, screen_stride)]
    screened = []

    for attempt in range(restarts):
        q0, pe0, _ = ik.solve(rng.uniform(ik.lo, ik.hi), path_pos[0], tool_axis,
                              iterations=500)
        if pe0 > 5e-4:
            if verbose:
                print("    seed %d: cannot reach the first waypoint (%.2f mm)"
                      % (attempt, pe0 * 1e3))
            continue
        score, _, _, err = _track(ik, q0, coarse, tool_axis, iterations)
        if verbose:
            print("    seed %d screened: %s | max tilt %5.2f deg | max dq %.3f rad"
                  % (attempt, "reaches" if not score[0] else "STALLS ",
                     np.degrees(score[1]), score[2]))
        screened.append((score, q0))

    if not screened:
        raise RuntimeError(
            "IK could not reach the first trajectory waypoint from any of the "
            "%d random seeds -- is the board inside the robot's workspace?" % restarts
        )

    screened.sort(key=lambda item: item[0])
    best = None
    for rank, (_, q0) in enumerate(screened[:max(1, finalists)]):
        result = _track(ik, q0, path_pos, tool_axis, iterations)
        if verbose:
            print("    finalist %d: max pos %.4f mm | max tilt %5.2f deg | max dq %.3f rad"
                  % (rank, result[0][3] * 1e3, np.degrees(result[0][1]), result[0][2]))
        if best is None or result[0] < best[0]:
            best = result

    (_stalled, max_axis, jump, max_pos), qs, pos_err, axis_err = best
    report = {
        'max_pos_err': max_pos,
        'mean_pos_err': float(pos_err.mean()),
        'max_axis_err': max_axis,
        'mean_axis_err': float(axis_err.mean()),
        'max_joint_step': jump,
        'n_waypoints': len(qs),
    }
    return qs, report
