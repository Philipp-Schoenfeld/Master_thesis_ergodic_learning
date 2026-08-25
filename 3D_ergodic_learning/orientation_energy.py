r"""
orientation_energy.py
=====================
Stufe 2: the orientation-dependent terms of the objective, and the SE(3) energy
that composes them with the positional one.

`ergodic_energy_torch.ErgodicEnergy` is left untouched — it is the faithful port
of the solver and stays the reference. `SE3Energy` wraps it, so a run without
orientation is byte-identical to before and a run with orientation is an
explicit opt-in.

Three terms, all in the same hinge-squared / inner-product style as the existing
ones so their magnitudes stay comparable:

**Pointing.** ``sum(1 - <R e_z, d(x)>)`` where d is the direction to the target
surface. Zero when the sensor looks straight at the surface, 2 when it looks
directly away. Per point it lives in [0, 2], so over T = 100 points the raw term
is O(10) for a mediocre solution — hence a default weight around 0.1 to sit
alongside an ergodic term of O(1..5).

**Standoff.** A hinge band, not a target value: any distance inside
[d0 - band, d0 + band] costs nothing, outside it costs quadratically. A hard
target would fight the ergodic term for no benefit — what matters is being
within the sensor's working range, not at one exact height.

**Angular smoothness.** The geodesic second difference on SO(3), i.e. the
rotational analogue of the existing ``W_SMOOTH`` acceleration term. Written with
the relative-rotation angle rather than a chart, so it is valid for large
rotations.

A note on why these weights are not "verbatim from the solver" like the others:
they are not in the 2D solver at all. `W_ERGODIC`, `W_SMOOTH` and `W_BOUNDARY`
are copied unchanged precisely so results stay comparable; these three are new
and their defaults are reasoned from magnitude, not inherited. They will need
tuning, and the runner logs each term separately so that tuning is possible.
"""

import torch
import torch.nn as nn

from ergodic_energy_torch import (ErgodicEnergy, ergodic_term, smoothness_term,
                                  boundary_term, obstacle_term, K_DEFAULT)
from orientation import geodesic_angle, sensor_axis, rot6d_to_matrix

# New weights — reasoned from term magnitude, NOT inherited from the solver.
W_POINT = 0.10          # raw term O(10) over 100 points -> contribution O(1)
W_STANDOFF = 300.0      # hinge on a length in domain widths; (0.05)^2 * 300 ~ 0.75
W_ANGSMOOTH = 2.0       # raw term is squared radians, typically O(0.1)

STANDOFF_TARGET = 0.12  # desired distance from the surface, in domain widths
STANDOFF_BAND = 0.03    # free band around it


def pointing_term(R, target_dir, w=W_POINT, axis=2):
    """w * sum(1 - <R e_axis, d>). R: (B, T, 3, 3), target_dir: (B, T, 3) -> (B,)"""
    a = sensor_axis(R, axis)
    align = (a * target_dir).sum(-1).clamp(-1.0, 1.0)
    return w * (1.0 - align).sum(dim=1)


def standoff_term(dist, target=STANDOFF_TARGET, band=STANDOFF_BAND, w=W_STANDOFF):
    """w * 0.5 * sum(excess^2) outside the band. dist: (B, T) -> (B,)"""
    excess = (dist - target).abs() - band
    return w * 0.5 * excess.clamp(min=0.0).pow(2).sum(dim=1)


def angular_smoothness_term(R, w=W_ANGSMOOTH):
    """w * sum(|theta_{i+1} - theta_i|^2), the geodesic second difference.

    R: (B, T, 3, 3) -> (B,). Rotational counterpart of the positional
    acceleration penalty; no 0.5 factor, matching `smoothness_term`.
    """
    if R.shape[1] < 3:
        return R.new_zeros(R.shape[0])
    step = geodesic_angle(R[:, :-1], R[:, 1:])          # (B, T-1)
    accel = step[:, 1:] - step[:, :-1]                  # (B, T-2)
    return w * accel.pow(2).sum(dim=1)


class SE3Energy(nn.Module):
    """Positional energy plus the orientation terms, over one interface.

    Args:
        base: an `ErgodicEnergy`; built here if not supplied.
        ergodic_on: 'position' evaluates coverage where the robot is,
            'footprint' evaluates it where the sensor beam lands. The second is
            the meaningful one for standoff inspection — with a standoff the
            robot is deliberately *not* on the surface, so scoring its own
            position rewards the wrong thing. It also makes coverage depend on
            orientation, which is what turns orientation from decoration into
            part of the task.
    """

    def __init__(self, base=None, K=K_DEFAULT, basis=None,
                 w_point=W_POINT, w_standoff=W_STANDOFF, w_angsmooth=W_ANGSMOOTH,
                 standoff_target=STANDOFF_TARGET, standoff_band=STANDOFF_BAND,
                 ergodic_on='position', sensor_axis_index=2, **base_kwargs):
        super().__init__()
        self.base = base if base is not None else ErgodicEnergy(
            K=K, basis=basis, **base_kwargs)
        self.w_point = w_point
        self.w_standoff = w_standoff
        self.w_angsmooth = w_angsmooth
        self.standoff_target = standoff_target
        self.standoff_band = standoff_band
        self.ergodic_on = ergodic_on
        self.axis = sensor_axis_index

    def extra_repr(self):
        return (f"ergodic_on={self.ergodic_on}, w_point={self.w_point}, "
                f"w_standoff={self.w_standoff}, w_angsmooth={self.w_angsmooth}, "
                f"standoff={self.standoff_target}+-{self.standoff_band}")

    def render(self, X):
        return self.base.render(X)

    def render_rot(self, rot6d):
        """(B, nxi, 6) control values -> (B, T, 3, 3) along the curve.

        The B-spline basis is applied in the 6D ambient space and the projection
        to SO(3) happens per curve point. See the note in `orientation.py` for
        why that is both valid and a deliberate simplification.
        """
        if self.base.basis is None:
            return rot6d_to_matrix(rot6d)
        r = torch.einsum('ti,bic->btc', self.base.basis, rot6d)
        return rot6d_to_matrix(r)

    def forward(self, X, phi_k, rot6d=None, field=None, return_terms=False):
        """
        X:     (B, nxi, 3) control points
        phi_k: (M,) or (B, M)
        rot6d: (B, nxi, 6) orientation control values, or None for position-only
        field: a `SurfaceField` for this target; required once rot6d is given
        """
        P = self.render(X)                                    # (B, T, 3)

        terms = {
            'smooth':   smoothness_term(P, self.base.w_smooth),
            'boundary': boundary_term(P, self.base.boundary_margin,
                                      self.base.w_boundary),
        }
        if self.base.use_obstacle:
            terms['obstacle'] = obstacle_term(
                P, self.base.obstacle_center, self.base.obstacle_radius,
                self.base.w_obstacle)

        if rot6d is None:
            coverage_pts = P
        else:
            if field is None:
                raise ValueError("orientation terms need a SurfaceField")
            R = self.render_rot(rot6d)                        # (B, T, 3, 3)
            a = sensor_axis(R, self.axis)
            d = field.direction(P)
            dist = field.distance(P)

            terms['point'] = pointing_term(R, d, self.w_point, self.axis)
            terms['standoff'] = standoff_term(dist, self.standoff_target,
                                              self.standoff_band, self.w_standoff)
            terms['angsmooth'] = angular_smoothness_term(R, self.w_angsmooth)

            coverage_pts = (field.footprint(P, a)
                            if self.ergodic_on == 'footprint' else P)

        terms['ergodic'] = ergodic_term(coverage_pts, self.base.k_idx,
                                        self.base.Lambda, phi_k,
                                        self.base.w_ergodic)

        total = sum(terms.values())
        return (total, terms) if return_terms else total


# ── Evaluation-only diagnostics ──────────────────────────────────────────────
def pointing_error_deg(R, target_dir, axis=2):
    """Mean angle between the sensor axis and the ideal direction, in degrees."""
    a = sensor_axis(R, axis)
    cos = (a * target_dir).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos)).mean(dim=1)


def incidence_ok_fraction(R, target_dir, max_deg=30.0, axis=2):
    """Fraction of curve points whose pointing error is within `max_deg`.

    Closer to what an inspection task actually cares about than the mean error:
    a path that is perfect for 80 % of its length and useless for the rest is
    not the same as one that is mediocre throughout, but the mean cannot tell
    them apart.
    """
    a = sensor_axis(R, axis)
    cos = (a * target_dir).sum(-1).clamp(-1.0, 1.0)
    err = torch.rad2deg(torch.acos(cos))
    return (err <= max_deg).float().mean(dim=1)


def standoff_error(dist, target=STANDOFF_TARGET):
    """Mean |distance - target|, in domain widths. dist: (B, T) -> (B,)."""
    return (dist - target).abs().mean(dim=1)


# ===========================================================================
# Stufe 3: orientation as a training objective inside the CFM loss
# ===========================================================================

class ParticleSurface:
    """Surface distance and direction taken from the conditioning particles.

    A drop-in stand-in for the query half of `SurfaceField`, and the *only* one
    of the two that may be used inside the training loop. `SurfaceField` is
    built from the density volume, which is never augmented; the training batch
    is rotated, scaled and translated, and the particles are augmented in
    lockstep with the trajectory. Querying the grid field from inside training
    would therefore compare an augmented curve against an unaugmented surface —
    silently, and worst exactly where the augmentation is strongest. This is the
    same reason `ergodic_metric` estimates phi_k from particles rather than from
    the grid.

    Occupancy is a threshold on mu relative to the per-sample maximum. With
    `sample_mode='uniform'` the particles cover the support and mu carries the
    density, so an unthresholded nearest neighbour would treat empty space as
    surface. mu enters only through this mask, never through the gradient, so
    the distance stays exactly differentiable in the query point.
    """

    def __init__(self, particles, mu_thresh=0.5, eps=1e-9):
        self.q = particles[..., :3]                              # (B, N, 3)
        mu = particles[..., 3]                                   # (B, N)
        peak = mu.amax(dim=1, keepdim=True).clamp(min=eps)
        self.occ = (mu >= mu_thresh * peak)                      # (B, N) bool
        self.eps = eps

    def _nearest(self, pts):
        """-> (delta to nearest occupied particle, its distance). pts: (B,T,3)"""
        d = torch.cdist(pts, self.q)                             # (B, T, N)
        d = d.masked_fill(~self.occ.unsqueeze(1), float('inf'))
        idx = d.argmin(dim=-1)                                   # (B, T)
        near = torch.gather(self.q, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        delta = near - pts                                       # (B, T, 3)
        return delta, delta.norm(dim=-1)

    def direction(self, pts):
        """Unit vector from each point towards the nearest occupied particle."""
        delta, dist = self._nearest(pts)
        return delta / dist.clamp(min=self.eps).unsqueeze(-1)

    def distance(self, pts):
        return self._nearest(pts)[1]

    def footprint(self, pts, axis):
        """Where the beam lands: pts + distance * axis. See SurfaceField."""
        return pts + self.distance(pts).unsqueeze(-1) * axis


class OrientationLoss(nn.Module):
    """Pointing, standoff and angular smoothness as a CFM training term.

    The counterpart of `ErgodicLoss` for the rotation block, and the reason it
    exists is the same: an imitation target caps the network at whatever
    produced it. The supervised branch trains the rotation against Stufe-0
    look-at frames computed from the stored curve — frames that are not ground
    truth (the database holds no orientation at all) but a geometric
    construction, so a network fitting them perfectly has merely reproduced a
    formula it could have evaluated directly. Scoring the orientation against
    the objective instead removes that ceiling, exactly as the ergodic term did
    for position.

    Applies to the one-step endpoint estimate `x1_hat`, with the same `t^power`
    ramp as `ErgodicLoss` and for the same reason: the estimate is meaningless
    at small t.

    Args:
        nxi, pts, deg: B-spline geometry, as in ErgodicLoss
        weight:  lambda multiplying this term in the total loss
        t_power: exponent of the t-ramp; 0 disables it
        w_point / w_standoff / w_angsmooth: relative weights of the three terms,
                 defaulting to the values reasoned about at the top of this file
        mu_thresh: occupancy threshold for `ParticleSurface`
    """

    def __init__(self, nxi=25, pts=128, deg=5, weight=1.0, t_power=2.0,
                 w_point=W_POINT, w_standoff=W_STANDOFF,
                 w_angsmooth=W_ANGSMOOTH, standoff_target=STANDOFF_TARGET,
                 standoff_band=STANDOFF_BAND, mu_thresh=0.5, axis=2):
        super().__init__()
        from obstacles import bspline_basis_matrix

        self.weight, self.t_power = weight, t_power
        self.w_point, self.w_standoff = w_point, w_standoff
        self.w_angsmooth = w_angsmooth
        self.standoff_target, self.standoff_band = standoff_target, standoff_band
        self.mu_thresh, self.axis = mu_thresh, axis
        self.register_buffer('B', torch.from_numpy(
            bspline_basis_matrix(nxi, pts, deg)))

    def extra_repr(self):
        return (f"weight={self.weight}, t_power={self.t_power}, "
                f"w_point={self.w_point}, w_standoff={self.w_standoff}, "
                f"w_angsmooth={self.w_angsmooth}, "
                f"standoff={self.standoff_target}+-{self.standoff_band}")

    def render(self, x1_hat):
        """(B, nxi, 9) -> curve (B, T, 3) and frames (B, T, 3, 3), float32."""
        x = x1_hat.float()
        P = torch.einsum('pi,bid->bpd', self.B, x[..., :3])
        # The 6D rotation is interpolated along the same basis and orthonormal-
        # ised afterwards; interpolating the matrices themselves would leave
        # SO(3). This mirrors what the runners do for visualisation.
        r6 = torch.einsum('pi,bic->bpc', self.B, x[..., 3:9])
        return P, rot6d_to_matrix(r6)

    def terms(self, x1_hat, particles, surface=None):
        """Per-sample unweighted terms. -> dict of (B,)"""
        P, R = self.render(x1_hat)
        surf = surface or ParticleSurface(particles.float(), self.mu_thresh)
        return {
            'point': pointing_term(R, surf.direction(P), self.w_point, self.axis),
            'standoff': standoff_term(surf.distance(P), self.standoff_target,
                                      self.standoff_band, self.w_standoff),
            'angsmooth': angular_smoothness_term(R, self.w_angsmooth),
        }

    def forward(self, x1_hat, particles, t, surface=None, return_parts=False):
        parts = self.terms(x1_hat, particles, surface)
        err = sum(parts.values())
        if self.t_power > 0:
            err = err * t.float() ** self.t_power
        if return_parts:
            return err.mean(), {k: v.mean().detach() for k, v in parts.items()}
        return err.mean()
