r"""
waypoints.py
============
Constraint 2: **waypoint / anchor pinning** -- force the curve through fixed
poses at chosen phase positions (start, end, via-points).

    E(C) = 0.5 * sum_k || C(phi_k) - p_k ||^2

Unlike the obstacle and keep-in penalties this one never vanishes: it is a
two-sided quadratic, so it keeps pulling until the pin is met exactly (the
same shape of penalty the 3D surface attraction uses).

The repo already ships a *trained* answer to this question for the start point
(`flow_matching_cond_particles_start.py`, whose `start_cond` checkpoints take
the start pose as a FiLM input). This module is the inference-only complement:
it needs no retraining and works with any checkpoint, at the cost of the
network not knowing about the pin while it draws the rest of the path. That
trade-off is exactly the one noted in the start-conditioned runner -- hard-
setting a start point the network never saw produces a jump, whereas pulling
it there over the ODE ramp lets the surrounding curve follow along.

Only the *phase* is pinned, not the arc length: `phi` indexes the B-spline's
parameter domain, so `phi=0.5` is the midpoint of the parameterisation.
"""

import numpy as np
import torch


class WaypointPins:
    """Pin `C(phi_k)` to `p_k` for a list of (phase, point) pairs."""

    def __init__(self, pins):
        """pins: iterable of (phase in [0, 1], point of length nd)."""
        self.phases = [float(f) for f, _ in pins]
        self.points = [tuple(float(v) for v in p) for _, p in pins]

    def __repr__(self):
        return f"WaypointPins({list(zip(self.phases, self.points))})"

    def _indices(self, pts):
        return [min(pts - 1, max(0, int(round(f * (pts - 1))))) for f in self.phases]

    def _targets(self, curve):
        return torch.as_tensor(self.points, dtype=curve.dtype, device=curve.device)

    def energy(self, curve):
        """Scalar penalty for `common.curve_energy_grad`."""
        idx = self._indices(curve.shape[1])
        got = curve[:, idx, :]                       # (B, K, nd)
        return 0.5 * ((got - self._targets(curve)[None]) ** 2).sum()

    def errors(self, curve):
        """(B, K) distance from each pin -- what the experiment reports."""
        idx = self._indices(curve.shape[1])
        return (curve[:, idx, :] - self._targets(curve)[None]).norm(dim=-1)

    def max_error(self, curve):
        return self.errors(curve).max().item()

    def draw(self, ax, zorder=4):
        pts = np.array(self.points)
        ax.scatter(pts[:, 0], pts[:, 1], s=90, marker='X', color='#D81B60',
                   edgecolors='white', linewidths=1.2, zorder=zorder,
                   label='Waypoints')
        for (x, y), f in zip(self.points, self.phases):
            ax.annotate(f"φ={f:g}", (x, y), textcoords='offset points',
                        xytext=(7, 6), fontsize=7, color='#D81B60')
