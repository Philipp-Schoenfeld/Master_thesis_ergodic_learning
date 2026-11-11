"""
mission3d.py
============
Active sensing on a 3D figure: the same round-based logic as the 2D control
centre, with the belief, the particle cloud and the trajectories living on the
surface of a mesh.

One round = belief -> target density Phi -> particle cloud on the surface ->
network plan (start-conditioned) -> snap onto the lit surface -> drive ->
measure -> update the belief.

Solvers
    A  plan once, execute once
    B  plan, drive one length unit, replan (with a coverage debt that fades)
    C  receding horizon: execute every full plan, replan
    D  plan long, execute a short part, replan (with coverage debt)
    E  C with five rounds and a manually chosen kappa per round
"""

import math

import numpy as np
import torch
from scipy.spatial import cKDTree

import paths3d

paths3d.setup()
import surfaces                                           # noqa: E402
from ergodic_metric import (make_k_grid, trajectory_coeffs,        # noqa: E402
                            target_coeffs_from_particles)
from surface3d import make_lit, snap_plan               # noqa: E402
from belief3d import make_belief, Sensor                   # noqa: E402
from planner3d import OBSTACLE_WEIGHT                      # noqa: E402

C_ROUNDS = 4
D_ROUNDS = 8
B_ROUNDS = 12
E_ROUNDS = 5
D_EXECUTE_FRAC = 0.15
LENGTH_UNIT = 0.8          # path length B drives per round (plans are ~2-3.5 long)
#: Clearance added to every obstacle's radius (figure units).  The eraser is 18 mm in radius and
#: its pad sits within ~7 cm of the surface, so with the figure scaled by up to 0.64 in the MuJoCo
#: scene the tool needs about 0.04 beyond the sphere to pass without clipping it.
OBSTACLE_MARGIN = 0.04
E_DEFAULT_KAPPAS =[round(float(k), 4) for k in np.geomspace(1.0, 0.005, E_ROUNDS)]


# -- target density models (elementwise; mirrors common/acquisition.py) ---------
def _norm_max(phi, floor=1e-6):
    phi = phi.clamp(min=0.0) + floor
    return phi / phi.max().clamp(min=1e-12)


def _neighbour_gradient_sq(f, nbr_idx, nbr_dist):
    """|grad f|^2 on an unstructured point set from k nearest neighbours."""
    d = (f[:, None] - f[nbr_idx]) / nbr_dist.clamp(min=1e-6)
    return (d ** 2).mean(dim=1)


def phi_ucb(mu, sd, kappa, **kw):
    return _norm_max(mu + kappa * sd)


def phi_mass(mu, sd, kappa, **kw):
    w = float(kappa) / (1.0 + float(kappa))
    m = mu.clamp(min=0.0); m = m / m.sum().clamp(min=1e-12)
    s = sd.clamp(min=0.0); s = s / s.sum().clamp(min=1e-12)
    return _norm_max((1.0 - w) * m + w * s)


def phi_lse(mu, sd, tau=0.25, **kw):
    z = (mu - tau) / sd.clamp(min=1e-9)
    return _norm_max(0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))))


def phi_eid(mu, sd, kappa, noise=0.05, nbr=None, **kw):
    """Expected information density; gradients from the k-NN graph of the surface."""
    eid = (_neighbour_gradient_sq(mu, *nbr) + _neighbour_gradient_sq(sd, *nbr)) / max(noise ** 2, 1e-12)
    m = mu.clamp(min=0.0); m = m / m.sum().clamp(min=1e-12)
    e = eid.clamp(min=0.0); e = e / e.sum().clamp(min=1e-12)
    return _norm_max(m + float(kappa) * e)


PHI_MODELS = {'ucb': phi_ucb, 'mass': phi_mass, 'eid': phi_eid, 'lse': phi_lse}


class Args:
    """Mission settings (names follow the 2D ``build_args``)."""

    def __init__(self, **kw):
        self.phi_model = 'ucb'
        self.kappa0 = 0.5
        self.kappa1 = 0.3
        self.phi_tau = 0.25
        self.gp_noise = 0.05
        self.sensor_radius = 0.05
        self.noise = 0.02
        self.max_obs = 96
        self.n_particles = 512
        self.phi_cutoff = 0.0
        self.belief_kernel = 'rbf'             # 'rbf' | 'ak' (belief3d.KERNELS)
        self.snap = True                       # project the plan onto the lit surface
        self.net_orientation = False           # tool axis from the network instead of the surface normal
        self.truth_known = False               # plan against the true density instead of the GP belief
        self.obstacle = None                   # SphereObstacle / CompositeObstacle: repels the plan
        self.obstacle_weight = OBSTACLE_WEIGHT # strength of the repulsion during inference
        self.debt_weight = 1.0
        self.debt_weight_lse = 0.15
        self.visit_sat = 0.25
        self.visit_halflife = 3.0
        self.visit_bandwidth = None            # default: sensor radius
        self.d_execute_frac = D_EXECUTE_FRAC
        self.kappas = list(E_DEFAULT_KAPPAS)   # solver E
        self.__dict__.update(kw)


def trim_to_length(curve, axes, max_len):
    """Cut a (T, 3) path (and its axes) to at most ``max_len`` arc length."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= max_len:
        return curve, axes
    k = int(np.searchsorted(cum, max_len))
    k = max(2, min(k, len(curve)))
    return curve[:k], axes[:k]


class Mission3D:
    def __init__(self, planner, figure, args, variant='A', seed=0, device=None):
        self.planner, self.fig, self.args, self.variant = planner, figure, args, variant
        self.device = torch.device(device or planner.device)
        self.lit = make_lit(figure)
        self.sensor = Sensor(figure, noise=args.noise, seed=seed)
        self.belief = make_belief(args.belief_kernel, figure.pts, figure.nrm, seed=seed,
                                  noise=args.gp_noise, device=self.device)
        idx, vals = self.sensor.prior(12)
        self.prior_idx = idx
        self.belief.observe_indices(idx, vals)
        self.belief0 = self.belief.clone()
        self.driven = []                       # list of (T,3) numpy segments
        self.driven_axes = []
        self.current_step = 'inference'
        self.gen = None
        self.pending = None
        self.rng = np.random.default_rng(seed + 1)
        tree = cKDTree(figure.pts)
        dist, idx = tree.query(figure.pts, k=9)
        self.nbr = (torch.from_numpy(idx[:, 1:]).long().to(self.device),
                    torch.from_numpy(dist[:, 1:]).float().to(self.device))
        self._pts_t = torch.from_numpy(figure.pts).to(self.device)
        self.lit_mask = torch.from_numpy(figure.lit.astype(np.float32)).to(self.device)
        self._truth_t = torch.from_numpy(np.asarray(figure.w, dtype=np.float32)).to(self.device)

    # -- helpers -------------------------------------------------------------
    def end_point(self):
        return self.driven[-1][-1] if self.driven else None

    def path_so_far(self):
        return np.concatenate(self.driven, axis=0) if self.driven else None

    def planning_belief(self, mu, sd):
        """What the planner sees as (mean, sigma): the GP posterior or, with
        ``args.truth_known``, the true density with zero uncertainty (the oracle case of the 2D
        side: the target density is the truth, no acquisition trade-off)."""
        if self.args.truth_known:
            return self._truth_t, torch.zeros_like(self._truth_t)
        return mu, sd

    def phi_plain(self, mu, sd, kappa, from_belief=True):
        """Target density of a round.  ``from_belief``: (mu, sd) are the raw GP posterior and
        are replaced by the truth when it is known; the coverage-debt solvers pass an already
        depleted (mu, sd) and set it to False."""
        a = self.args
        if from_belief:
            mu, sd = self.planning_belief(mu, sd)
        mu = mu.clamp(0.0, 1.0)                # the target density is normalised to max 1
        if a.truth_known:
            phi = _norm_max(mu)                # known density: no uncertainty term to trade off
        else:
            phi = PHI_MODELS[a.phi_model](mu, sd, kappa=kappa, tau=a.phi_tau, noise=a.gp_noise,
                                          nbr=self.nbr)
        return phi * self.lit_mask             # the robot only works on the lit side

    def visit_field(self, path, half_life):
        """Gaussian visitation density on the belief points; with ``half_life > 0``
        a recent visit weighs more than an old one (age = arc length to the end)."""
        a = self.args
        bw = a.visit_bandwidth or a.sensor_radius
        p = torch.from_numpy(path).float().to(self.device)
        d2 = torch.cdist(self._pts_t, p) ** 2
        k = torch.exp(-d2 / (2.0 * bw ** 2))
        if half_life and half_life > 0:
            seg = torch.norm(p[1:] - p[:-1], dim=-1)
            cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
            age = cum[-1] - cum
            k = k * (2.0 ** (-age / (half_life * LENGTH_UNIT)))[None, :]
        return k.sum(dim=1)

    def debt_phi(self, mu, sd, visit, kappa):
        a = self.args
        mu, sd = self.planning_belief(mu, sd)
        v = (torch.zeros_like(mu) if visit is None
             else (visit / (a.visit_sat * visit.max().clamp(min=1e-12))).clamp(0.0, 1.0))
        sd_eff = sd * (1.0 - v)
        know = 1.0 - sd.clamp(0.0, 1.0)
        w_debt = a.debt_weight_lse if a.phi_model == 'lse' else a.debt_weight
        mu_eff = mu.clamp(min=0.0) * (1.0 - (w_debt * v * know).clamp(0.0, 1.0))
        return self.phi_plain(mu_eff, sd_eff.clamp(min=1e-6), kappa, from_belief=False), v

    def cloud(self, phi):
        """Particle cloud (N, 4) = (x, y, z, weight): positions drawn uniformly over
        the support {Phi >= cutoff * max Phi} of the surface, weight = Phi."""
        a = self.args
        w = phi.detach().cpu().numpy().astype(np.float64)
        w = w / max(float(w.max()), 1e-12)
        if a.phi_cutoff > 0:
            w = np.where(w >= a.phi_cutoff, w, 0.0)
        return surfaces.particles_from_projection(self.fig.pts, w, a.n_particles,
                                                  seed=int(self.rng.integers(0, 2 ** 31)))

    def plan_round(self, phi):
        """Network plan from the end of the driven path.

        ``args.snap``: project the positions onto the lit surface (else the raw network
        positions are used).  ``args.net_orientation``: tool axes from the network's
        own orientation head (else the inward surface normal).

        ``args.obstacle``: the plan is repelled from it during the flow integration
        (inference) and again in every snapping iteration (refinement)."""
        a = self.args
        obs = a.obstacle
        self.lit.set_obstacle(obs)              # also keeps the tool-axis lookups obstacle-free
        parts = self.cloud(phi)
        start = self.end_point()
        if start is None:
            start = self.fig.start
        self.current_step = 'inference'
        cps, rot6 = self.planner.plan(parts, start=start, obstacle=obs,
                                      obstacle_weight=a.obstacle_weight)
        raw_curve, net_axes = self.planner.render(cps, rot6)
        if a.snap:
            self.current_step = 'snap'
            _, curve, normal_axes = snap_plan(cps, self.planner.B.cpu(), self.lit, start=start,
                                              obstacle=obs)
        else:
            curve = raw_curve.numpy().astype(np.float32)
            _, nrm = self.lit.project(curve)
            normal_axes = -nrm
            normal_axes = normal_axes / np.maximum(np.linalg.norm(normal_axes, axis=1, keepdims=True), 1e-9)
        if a.net_orientation and net_axes is not None:
            axes = net_axes.numpy()
        else:
            axes = normal_axes
        return curve.astype(np.float32), np.asarray(axes, dtype=np.float32), parts

    def replan(self, info):
        """Plan the round described by ``info`` again (same belief state and target
        density, current settings); returns the new info."""
        phi = torch.from_numpy(info['phi']).to(self.device)
        curve, axes, parts = self.plan_round(phi)
        kind, val = info.get('trim') or (None, None)
        if kind == 'frac':
            k = max(2, int(round(val * len(curve))))
            seg, ax = curve[:k], axes[:k]
        elif kind == 'len':
            seg, ax = trim_to_length(curve, axes, val)
        else:
            seg, ax = curve, axes
        new = dict(info, seg=seg, axes=ax, parts=parts)
        self.pending = (seg, ax)
        return new

    def observe(self, seg):
        idx, vals = self.sensor.measure(seg, self.args.sensor_radius, self.args.max_obs)
        if len(idx):
            self.belief.observe_indices(idx, vals)

    def commit(self, seg, axes):
        self.driven.append(np.asarray(seg, dtype=np.float32))
        self.driven_axes.append(np.asarray(axes, dtype=np.float32))
        self.observe(seg)

    # -- round generators ----------------------------------------------------
    def rounds(self):
        return {'A': self._rounds_A, 'B': self._rounds_B, 'C': self._rounds_C,
                'D': self._rounds_D, 'E': self._rounds_E}[self.variant]()

    def _info(self, r, n, phi, mu, sd, seg, axes, kappa, parts, visit=None, trim=None):
        self.pending = (seg, axes)             # what gets committed once the round is driven
        return dict(trim=trim, round=r, n_rounds=n, phi=phi.detach().cpu().numpy(), mu=mu.cpu().numpy(),
                    sd=sd.cpu().numpy(), seg=seg, axes=axes, kappa=kappa, parts=parts,
                    visit=None if visit is None else visit.cpu().numpy())

    def _rounds_A(self):
        mu, sd = self.belief.posterior()
        phi = self.phi_plain(mu, sd, self.args.kappa0)
        seg, axes, parts = self.plan_round(phi)
        yield self._info(0, 1, phi, mu, sd, seg, axes, self.args.kappa0, parts)
        self.commit(*self.pending)

    def _rounds_C(self, n_rounds=C_ROUNDS, kappas=None):
        a = self.args
        for r in range(n_rounds):
            kap = float(kappas[r]) if kappas is not None else \
                a.kappa0 + (a.kappa1 - a.kappa0) * (r / max(n_rounds - 1, 1))
            mu, sd = self.belief.posterior()
            phi = self.phi_plain(mu, sd, kap)
            seg, axes, parts = self.plan_round(phi)
            yield self._info(r, n_rounds, phi, mu, sd, seg, axes, kap, parts)
            self.commit(*self.pending)

    def _rounds_E(self):
        return self._rounds_C(n_rounds=len(self.args.kappas), kappas=self.args.kappas)

    def _rounds_D(self, n_rounds=D_ROUNDS):
        a = self.args
        for r in range(n_rounds):
            kap = a.kappa0 + (a.kappa1 - a.kappa0) * (r / max(n_rounds - 1, 1))
            mu, sd = self.belief.posterior()
            here = self.path_so_far()
            visit = self.visit_field(here, 0.0) if here is not None else None
            phi, v = self.debt_phi(mu, sd, visit, kap)
            curve, axes, parts = self.plan_round(phi)
            k = max(2, int(round(a.d_execute_frac * len(curve))))
            seg, ax = curve[:k], axes[:k]
            yield self._info(r, n_rounds, phi, mu, sd, seg, ax, kap, parts, visit=v,
                             trim=('frac', a.d_execute_frac))
            self.commit(*self.pending)

    def _rounds_B(self, n_rounds=B_ROUNDS):
        a = self.args
        saved = (a.debt_weight, a.visit_sat)
        a.debt_weight, a.visit_sat = 0.6, 1.0          # values of the length-unit study
        try:
            for r in range(n_rounds):
                mu, sd = self.belief.posterior()
                here = self.path_so_far()
                visit = self.visit_field(here, a.visit_halflife) if here is not None else None
                phi, v = self.debt_phi(mu, sd, visit, a.kappa0)
                curve, axes, parts = self.plan_round(phi)
                seg, ax = trim_to_length(curve, axes, LENGTH_UNIT)
                yield self._info(r, n_rounds, phi, mu, sd, seg, ax, a.kappa0, parts, visit=v,
                                 trim=('len', LENGTH_UNIT))
                self.commit(*self.pending)
        finally:
            a.debt_weight, a.visit_sat = saved


# -- metrics against the ground truth (only the evaluator knows it) ---------------
class Metrics3D:
    def __init__(self, figure, device='cpu', K=6):
        k_idx, self.Lam = make_k_grid(K)
        self.k_idx = torch.tensor(k_idx, device=device)
        self.Lam = torch.tensor(self.Lam, dtype=torch.float32, device=device)
        self.device = device
        self.pts = figure.pts
        self.w = figure.w
        truth = torch.from_numpy(np.concatenate([figure.pts, figure.w[:, None]], axis=1)).float().to(device)
        self.phi_truth = target_coeffs_from_particles(truth.unsqueeze(0), self.k_idx, True)

    def ergodic(self, path, particles=None):
        c = trajectory_coeffs(torch.from_numpy(np.asarray(path)).float().to(self.device).unsqueeze(0), self.k_idx)
        phi = self.phi_truth if particles is None else target_coeffs_from_particles(
            torch.from_numpy(np.asarray(particles)).float().to(self.device).unsqueeze(0), self.k_idx, True)
        return float((self.Lam * (c - phi) ** 2).sum())

    def coverage(self, path):
        p = np.asarray(path, dtype=np.float32)[::2]
        d = np.linalg.norm(self.pts[:, None, :] - p[None], axis=2).min(axis=1)
        return float((d * self.w).sum() / max(float(self.w.sum()), 1e-9))
