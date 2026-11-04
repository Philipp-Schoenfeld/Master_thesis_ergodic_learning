"""
joint_map3d.py
==============
A joint-configuration map over the lit surface of the figure.

Solving the IK online by continuation lets the arm drift onto a bad branch (it
ends up at a joint limit and loses the trajectory).  Instead, the map solves the
IK once for a set of nodes spread over the surface, spreading breadth-first over
a neighbour graph so that neighbouring nodes share one smooth branch.  At run
time the map provides the seed configuration for the per-frame IK, so the arm
stays on the validated branch (same idea as the 2D ``joint_map``).
"""

import numpy as np
from scipy.spatial import cKDTree

from mujoco_3d import scene3d as S

POS_OK = 2e-3
AXIS_OK = np.radians(3.0)


class JointMap3D:
    def __init__(self, ik, pts, axes, q_home, k=8, n_restarts=6):
        """``pts`` (N, 3) world surface points, ``axes`` (N, 3) tool axes there."""
        self.ik = ik
        self.pts = np.asarray(pts, float)
        self.axes = np.asarray(axes, float)
        n = len(self.pts)
        self.q = np.zeros((n, S.N_ARM_JOINTS))
        self.valid = np.zeros(n, dtype=bool)
        self.pos_err = np.full(n, np.inf)
        tree = cKDTree(self.pts)
        _, nbr = tree.query(self.pts, k=min(k + 1, n))

        def solve(i, seed, iters=25):
            q, pe, ae = ik.solve(seed, self.pts[i] - self.axes[i] * S.STANDOFF, self.axes[i],
                                 iterations=iters, max_step=0.3, tol_pos=5e-4, tol_axis=0.01)
            self.pos_err[i] = pe
            return q, (pe < POS_OK and ae < AXIS_OK)

        order = np.argsort(np.linalg.norm(self.pts - self.pts.mean(axis=0), axis=1))
        best = None
        for r in range(n_restarts):
            result = self._spread(order[r % min(len(order), 10)], solve, nbr, q_home, restart=r)
            score = result[1].sum()
            if best is None or score > best[0]:
                best = (score, *result)
            if score >= 0.97 * n:
                break
        _, self.q, self.valid = best
        self._vtree = cKDTree(self.pts[self.valid]) if self.valid.any() else None
        self._vq = self.q[self.valid]
        self.valid_fraction = float(self.valid.mean())

    def _spread(self, root, solve, nbr, q_home, restart):
        n = len(self.pts)
        q = np.zeros((n, S.N_ARM_JOINTS))
        valid = np.zeros(n, dtype=bool)
        seed_of = np.zeros((n, S.N_ARM_JOINTS))
        rng = np.random.default_rng(restart)
        seed = q_home if restart == 0 else rng.uniform(self.ik.lo, self.ik.hi)
        q[root], ok = solve(root, seed, iters=400)
        valid[root] = ok
        seed_of[root] = q[root] if ok else q_home
        seen = np.zeros(n, dtype=bool)
        seen[root] = True
        queue = [root]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in nbr[u][1:]:
                if seen[v]:
                    continue
                seen[v] = True
                qv, ok = solve(v, seed_of[u])
                q[v], valid[v] = qv, ok
                seed_of[v] = qv if ok else seed_of[u]
                queue.append(v)
        for v in np.flatnonzero(~seen):                       # disconnected pieces
            qv, ok = solve(v, q_home, iters=200)
            q[v], valid[v] = qv, ok
        return q, valid

    def query(self, p, k=4):
        """Seed configuration for world position ``p`` (inverse-distance blend of the
        nearest valid nodes)."""
        if self._vtree is None:
            return self.ik.posture.copy()
        d, i = self._vtree.query(np.asarray(p, float), k=min(k, len(self._vq)))
        d, i = np.atleast_1d(d), np.atleast_1d(i)
        w = 1.0 / (d + 1e-3)
        return (w[:, None] * self._vq[i]).sum(axis=0) / w.sum()
