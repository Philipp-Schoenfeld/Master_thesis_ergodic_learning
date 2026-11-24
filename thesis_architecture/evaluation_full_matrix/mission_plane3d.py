r"""
mission_plane3d.py
==================
The five-round annealed replanning mission of `sample_annealed_mission.run_mission` in 3D on a
planar patch (`plane3d.Plane`): belief, sensor, target density, network inference, baselines and
metrics all work on 3D world coordinates. Used by `run_exploration_comparison_3d.py`.

Import rule: the 3D project (`3D_ergodic_learning/`) has modules whose names collide with the 2D
ones (`flow_matching_cond_particles_crossattn`, `obstacles`, `ergodic_energy_torch`, ...), so this
module puts the 3D folders first on `sys.path` (`exploration_3d/paths3d.setup`) and imports NOTHING
from the 2D pipeline that would pull in a 2D network module. In particular `common.data` is not
used (it inserts `thesis_architecture/` at the front of `sys.path`); `load_truth` below is a copy.

Everything that is a port of a 2D component keeps its 2D settings (`ucb_tuned_svgd0` strategy of
`variant_runner`: debt density with debt_weight 0.6, visit_sat 1.0, half life 3 length units, GP
noise 0.05, lengthscale 0.08, measurement noise 0.05, 4-point sensor ring of radius 0.06, at most
64 measurements per round), with lengths in plane units (`plane3d` module docstring).

Path sources (`path_fn` baselines: no network, no SVGD) are generated in the plane's (u, v) chart and
embedded isometrically (up to the scale s); they are 3D curves on the patch, nothing is projected.
"""
import json
import math
import os
import sqlite3
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_exp3d = os.path.join(_arch, 'exploration_3d')
if _exp3d not in sys.path:
    sys.path.insert(0, _exp3d)
import paths3d                                                          # noqa: E402

paths3d.setup()
import net_zoo3d                                                        # noqa: E402
from belief3d import SurfaceBelief, AttentiveSurfaceBelief              # noqa: E402
from flow_matching_cond_particles_crossattn import (                    # noqa: E402
    ParticleCrossAttnFlowNetwork, generate_particle_trajectories)
from flow_matching_cond_particles_dit import ParticleDiTFlowNetwork, dit_kwargs   # noqa: E402
from obstacles import bspline_basis_matrix                              # noqa: E402  (the 3D one)

import flow_matching_cond_particles_crossattn as _fm3d                  # noqa: E402
assert os.path.dirname(os.path.abspath(_fm3d.__file__)) == paths3d.D3, \
    f'wrong flow-matching module imported: {_fm3d.__file__} (must be the 3D one)'

_gen = os.path.join(_arch, 'ergodic_dataset_generator')
if _gen not in sys.path:
    sys.path.append(_gen)              # appended: shape_library has no 3D namesake

GP_RES = 64             # variant_runner.GP_RES
TRUTH_RES = 96          # run_svgd_convergence.TRUTH_RES
NXI, N_POINTS, DEGREE = 25, 128, 5
LENGTH_UNIT = math.sqrt(2.0)
PTS_PER_UNIT = 72
TRUTH_DB = os.path.join(_gen, 'ergodic_dataset_775.db')

#: the 3D networks of the comparison: key -> (file-name pattern, label). The pair unet630 / dit630
#: is the 3D counterpart of the 2D pair (LM-teacher DB, Sinkhorn loss lambda = 630, start held by
#: inpainting); see `exploration_3d/net_zoo3d.py` for the descriptions.
NETS3D = {
    'unet630': (r'lm630unet.*ep0750', 'U-Net D512 λ=630 (3D)'),
    'dit630': (r'lm630dit.*ep0750', 'DiT-B λ=630 (3D)'),
    'dits_old': (r'dbcmpold.*_ep0150', 'DiT-S λ=630, old LM DB (3D)'),
    'nooffset200': (r'^nooffset200_flow3d.*_ep0300\.pt$', 'U-Net D384 Fourier loss (3D)'),
    'selfsup': (r'^selfsup_3d_selfsup3d_date_08_18_15h57min.*_final\.pt$', 'Self-supervised 3D + TSVEC'),
}


# ===========================================================================
# Settings (2D `ucb_tuned_svgd0` values, lengths in plane units)
# ===========================================================================

def default_settings(**kw):
    import types
    a = types.SimpleNamespace(
        sensor_radius=0.06, noise=0.05, max_obs=64, n_ring=4, gp_noise=0.05, gp_lengthscale=0.08,
        debt_weight=0.6, visit_sat=1.0, visit_halflife=3.0, visit_bandwidth=0.06, ak_fit_iters=200,
        cfg_weight=2.0, flow_steps=100, w_surface=50.0, w_align=1.0, force_weight=20.0,
        force_max=0.5)
    a.__dict__.update(kw)
    return a


def load_truth(labels=None, n=10_000, split='val', resolution=TRUTH_RES, db_path=TRUTH_DB):
    """Copy of `exploration/common/data.load_truth` (see the import rule in the module docstring):
    the heldout densities (n, R, R) on [0, 1]^2, indexed [y, x], normalised to max 1."""
    from shape_library import pdf_on_grid
    conn = sqlite3.connect(db_path)
    rows = conn.execute('SELECT shape_name, density_params FROM ergodic_pairs WHERE split=? '
                        'ORDER BY id ASC', (split,)).fetchall()
    conn.close()
    picked, seen = [], set()
    for name, params in rows:
        if (labels is not None and name not in labels) or name in seen:
            continue
        seen.add(name)
        picked.append((name, params))
        if labels is None and len(picked) >= n:
            break
    names, grids = [], []
    for name, params in picked:
        d, _, _ = pdf_on_grid(json.loads(params), resolution=resolution)
        d = np.asarray(d, dtype=np.float32)
        grids.append(d / max(d.max(), 1e-12))
        names.append(name)
    return names, torch.tensor(np.stack(grids))


# ===========================================================================
# Curve helpers (any dimension)
# ===========================================================================

def path_length(curve):
    return float((curve[1:] - curve[:-1]).norm(dim=-1).sum()) if curve.shape[0] > 1 else 0.0


def resample_arclength(curve, n_pts):
    if curve.shape[0] < 2:
        return curve
    seg = (curve[1:] - curve[:-1]).norm(dim=-1)
    cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
    total = float(cum[-1])
    if total <= 1e-9:
        return curve[:1]
    want = torch.linspace(0.0, total, n_pts, device=curve.device, dtype=curve.dtype)
    idx = torch.searchsorted(cum, want).clamp(1, curve.shape[0] - 1)
    lo, hi = cum[idx - 1], cum[idx]
    t = ((want - lo) / (hi - lo).clamp(min=1e-12)).unsqueeze(-1)
    return curve[idx - 1] + t * (curve[idx] - curve[idx - 1])


def trim_to_length(curve, max_len):
    """Cut a polyline to at most `max_len` arc length (last point interpolated exactly)."""
    if max_len <= 0:
        return curve[:1]
    seg = (curve[1:] - curve[:-1]).norm(dim=-1)
    cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
    if float(cum[-1]) <= max_len:
        return curve
    k = int(torch.searchsorted(cum, torch.tensor(max_len, dtype=cum.dtype)))
    t = (max_len - float(cum[k - 1])) / max(float(seg[k - 1]), 1e-12)
    end = curve[k - 1] + t * (curve[k] - curve[k - 1])
    return torch.cat([curve[:k], end[None]], dim=0)


def bilinear(field, uv):
    """(H, W) field indexed [y, x], uv (N, 2) in [0, 1]^2 -> (N,)"""
    H, W = field.shape[-2:]
    g = (uv.float() * 2.0 - 1.0).view(1, -1, 1, 2)
    return F.grid_sample(field.float().to(uv.device).view(1, 1, H, W), g, mode='bilinear',
                         align_corners=True).view(-1)


# ===========================================================================
# Belief, sensor, target density
# ===========================================================================

class PlaneBelief:
    """GP belief (RBF or Attentive Kernel) of `exploration_3d/belief3d.py` on the 3D grid points of
    the patch (GP_RES x GP_RES); measurements are taken at arbitrary 3D points. The kernel acts on
    3D positions (and the surface normal, which is constant on a plane, so that factor is 1)."""

    def __init__(self, kernel, plane, seed, a, device):
        self.plane, self.G = plane, GP_RES
        pts = plane.grid_world(GP_RES, dtype=torch.float64).numpy()
        nrm = np.tile(plane.n, (pts.shape[0], 1))
        kw = dict(lengthscale=a.gp_lengthscale * plane.s, noise=a.gp_noise, device=device,
                  dtype=torch.float64)
        if kernel == 'rbf':
            self.b = SurfaceBelief(pts, nrm, **kw)
        elif kernel == 'ak':
            self.b = AttentiveSurfaceBelief(pts, nrm, fit_iters=a.ak_fit_iters, seed=seed, **kw)
        else:
            raise KeyError(f'belief kernel {kernel!r} has no 3D version (rbf, ak)')
        self._nfeat = torch.as_tensor(plane.n / self.b.nl, dtype=torch.float64, device=self.b.device)

    @property
    def n_obs(self):
        return self.b.n_obs

    def observe(self, xyz, vals):
        b = self.b
        feat = torch.cat([xyz.to(b.device, b.dtype) / b.ls,
                          self._nfeat.expand(xyz.shape[0], 3)], dim=1)
        with b._lock:
            b.X = torch.cat([b.X, feat], dim=0)
            b.y = torch.cat([b.y, vals.to(b.device, b.dtype).reshape(-1)], dim=0)
            b._cache = None
            if isinstance(b, AttentiveSurfaceBelief):
                b._fit()

    def posterior_grid(self):
        mu, sd = self.b.posterior()
        return mu.view(self.G, self.G), sd.view(self.G, self.G)


def measure(seg, truth, plane, a):
    """Measurements along a driven segment (T, 3): every path point plus a ring of `n_ring` points
    at the sensor radius IN the plane (the 2D `common.observation.measure`), values = ground truth
    at those points + Gaussian noise (global torch RNG, seeded per round like 2D)."""
    pts = seg
    if a.sensor_radius > 0 and a.n_ring > 0:
        ang = torch.arange(a.n_ring, device=seg.device, dtype=seg.dtype) * (2 * math.pi / a.n_ring)
        E = torch.as_tensor(plane.E, dtype=seg.dtype, device=seg.device)
        off = (torch.cos(ang)[:, None] * E[0] + torch.sin(ang)[:, None] * E[1]) * (a.sensor_radius * plane.s)
        pts = torch.cat([seg, (seg.unsqueeze(1) + off.unsqueeze(0)).reshape(-1, 3)], dim=0)
    uv = plane.to_uv(pts).clamp(0.0, 1.0)
    pts = plane.to_world(uv)
    vals = bilinear(truth, uv)
    if a.noise > 0:
        vals = vals + a.noise * torch.randn(vals.shape, device=vals.device, dtype=vals.dtype)
    return pts, vals


def thin(pts, vals, max_points):
    n = pts.shape[0]
    if n <= max_points:
        return pts, vals
    idx = torch.linspace(0, n - 1, max_points, device=pts.device).long()
    return pts[idx], vals[idx]


def visitation(driven, cells, plane, a):
    """`exploration_optimierung.mission.visitation_recent` on the 3D grid points: Gaussian
    visitation, a visit's weight halves every `visit_halflife` length units of later path."""
    s = plane.s
    seg = (driven[1:] - driven[:-1]).norm(dim=-1) / s
    cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
    w = torch.pow(torch.tensor(0.5, device=driven.device), (cum[-1] - cum) / (a.visit_halflife * LENGTH_UNIT))
    d2 = (torch.cdist(cells, driven) / s) ** 2
    k = (torch.exp(-d2 / (2.0 * a.visit_bandwidth ** 2)) * w.unsqueeze(0)).sum(dim=1)
    return k.view(GP_RES, GP_RES)


def debt_density(mu, sd, visit, kappa, a, floor=1e-6):
    """`apply_cfm_belief.debt_density` with the 'ucb' target (`common.acquisition.ucb_density`,
    norm='max'), elementwise -- identical formula to 2D."""
    if visit is None:
        v = torch.zeros_like(mu)
    else:
        v = (visit / (a.visit_sat * visit.max().clamp(min=1e-12))).clamp(0.0, 1.0)
    sd_eff = (sd * (1.0 - v)).clamp(min=1e-6)
    know = 1.0 - sd.clamp(0.0, 1.0)
    mu_eff = mu.clamp(min=0.0) * (1.0 - (a.debt_weight * v * know).clamp(0.0, 1.0))
    phi = (mu_eff + kappa * sd_eff).clamp(min=0.0) + floor
    return phi / phi.max().clamp(min=1e-12), v


def particle_cloud(phi, plane, n, cutoff, device):
    """(n, 4) = (x, y, z, weight) on the patch: positions uniform over the support of the (cut)
    target, jittered within a cell, weight = target value -- `particles_from_density(mode='uniform')`
    of 2D, in 3D world coordinates (the convention the 3D networks were trained on)."""
    if cutoff > 0.0:
        phi = torch.where(phi >= cutoff * phi.max(), phi, torch.zeros_like(phi))
    R = phi.shape[-1]
    flat = phi.reshape(-1).to(device)
    w = (flat > 1e-5).float()
    if float(w.sum()) <= 0:
        w = torch.ones_like(flat)
    idx = torch.multinomial(w, n, replacement=True)
    sy, sx = idx // R, idx % R
    jx = (torch.rand(n, device=device) - 0.5) / (R - 1)
    jy = (torch.rand(n, device=device) - 0.5) / (R - 1)
    uv = torch.stack([(sx.float() / (R - 1) + jx).clamp(0, 1), (sy.float() / (R - 1) + jy).clamp(0, 1)], -1)
    return torch.cat([plane.to_world(uv), flat[idx, None]], dim=-1)


# ===========================================================================
# Planners (3D networks)
# ===========================================================================

def find_checkpoint3d(method, override=None):
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(override)
        return override
    import re
    pat = re.compile(NETS3D[method][0])
    hits = [f for f in net_zoo3d._candidate_files() if pat.search(os.path.basename(f))]
    if not hits:
        raise FileNotFoundError(f"no 3D checkpoint for '{method}' (pattern {NETS3D[method][0]!r}) under "
                                f"{os.path.join(paths3d.ROOT, 'transfer')} or {os.path.join(paths3d.D3, 'checkpoints')}")
    return max(hits, key=os.path.getmtime)


def plane_force_fn(plane, B, a):
    """`force(cps) -> dE/dcps` for the flow integration: Constraint 5 of the 3D pipeline
    (`mesh_alignment.SurfaceTangentAlignment`, w_surface 50 / w_align 1) for the plane, i.e.
    E = w_s/2 mean_t SDF^2 + w_a/2 mean_t (tangent . n)^2 per sample, gradient normalised by the
    basis column sums like `mesh_alignment.curve_energy_grad`."""
    n_t = torch.as_tensor(plane.n, dtype=torch.float32)
    c_t = torch.as_tensor(plane.c, dtype=torch.float32)

    def force(cps):
        with torch.enable_grad():
            c = cps.detach().float().requires_grad_(True)
            curve = torch.einsum('pi,bid->bpd', B.to(c.device), c)
            n, cc = n_t.to(c.device), c_t.to(c.device)
            sdf = (curve - cc) @ n
            t = curve[:, 1:] - curve[:, :-1]
            t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            e = (a.w_surface * 0.5 * (sdf ** 2).mean(-1) + a.w_align * 0.5 * ((t @ n) ** 2).mean(-1)).sum()
            (g,) = torch.autograd.grad(e, c)
        return g / B.to(c.device).sum(0).clamp(min=1e-8)[None, :, None]
    return force


class NetPlanner3D:
    """A 3D SE(3) flow-matching network (start-conditioned, 25 x 3 control points + 6D orientation)
    as the candidate source; the plane constraint enters as an inference-time force (same recipe
    as `exploration_3d/mission3d.plan_round`). Only the positions are used."""

    def __init__(self, ckpt, plane, device, a):
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        kw = dict(nxi=int(ck.get('nxi', NXI)), nd=3, D=int(ck.get('D', 384)),
                  predict_orientation=bool(ck.get('orientation', True)),
                  start_cond=bool(ck.get('start_cond', False)),
                  obstacle_cond=bool(ck.get('obstacle_cond', False)))
        net = (ParticleDiTFlowNetwork(**kw, **dit_kwargs(ck)) if ck.get('backbone', 'unet') == 'dit'
               else ParticleCrossAttnFlowNetwork(**kw))
        net.load_state_dict(ck['model_state_dict'])
        self.model = net.to(device).eval()
        self.nxi, self.n_particles = kw['nxi'], int(ck.get('n_particles', 512))
        self.start_cond = kw['start_cond']
        self.info = dict(ckpt=os.path.basename(ckpt), epoch=ck.get('epoch'), backbone=ck.get('backbone', 'unet'),
                         start_cond=self.start_cond, n_particles=self.n_particles)
        del ck
        self.device, self.a = device, a
        assert self.nxi == NXI, f'nxi={self.nxi}, expected {NXI}'
        self.B = torch.from_numpy(bspline_basis_matrix(NXI, N_POINTS, DEGREE)).float().to(device)
        self.force = plane_force_fn(plane, self.B, a)

    @torch.no_grad()
    def plan(self, parts, n, start):
        cps, _ = generate_particle_trajectories(
            self.model, parts.to(self.device).float(), num_samples=n, nxi=self.nxi, nd=3,
            steps=self.a.flow_steps, device=str(self.device), cfg_weight=self.a.cfg_weight,
            start=start.reshape(1, 3).float() if self.start_cond else None,
            force=self.force, force_weight=self.a.force_weight, force_max=self.a.force_max)
        cps = cps.float()
        cps[:, 0] = start.to(cps.device).float()       # (inpainting nets already hold it)
        return cps


class SelfsupPlanner3D:
    """The 3D self-supervised single-pass generator (`3D_ergodic_learning/
    flow_matching_particles_selfsupervised.py`). It has no start input: the first control point is
    set to the agent position, the TSVEC refinement (start pinned, plane attachment) does the rest."""

    def __init__(self, ckpt, plane, device, a):
        from flow_matching_particles_selfsupervised import SelfSupervisedParticleGenerator
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        net = SelfSupervisedParticleGenerator(nxi=int(ck.get('nxi', NXI)), nd=3, D=int(ck.get('D', 128)),
                                              predict_orientation=bool(ck.get('orientation', False)))
        net.load_state_dict(ck['model_state_dict'])
        self.model = net.to(device).eval()
        self.nxi, self.n_particles, self.start_cond = net.nxi, int(ck.get('n_particles', 128)), False
        self.info = dict(ckpt=os.path.basename(ckpt), selfsupervised=True, n_particles=self.n_particles)
        del ck
        self.device = device

    @torch.no_grad()
    def plan(self, parts, n, start):
        cps, _ = self.model.generate(parts.to(self.device).float(), num_samples=n, device=str(self.device))
        cps = cps.float().clone()
        cps[:, 0] = start.to(cps.device).float()
        return cps


def build_planner3d(method, plane, device, a, override=None):
    if method not in NETS3D:
        return None
    ckpt = find_checkpoint3d(method, override)
    cls = SelfsupPlanner3D if method == 'selfsup' else NetPlanner3D
    return cls(ckpt, plane, device, a)


# ===========================================================================
# Model-free path sources (all budget matched: `walk_len` plane units per round)
# ===========================================================================

def _to_world_curve(plane, uv, device='cpu'):
    return plane.to_world(torch.as_tensor(uv, dtype=torch.float32)).to(device)


def random_walk_fn(plane, walk_len, seed, step_std=0.035, n_points=2048):
    """`init_baselines.random_walk_path`: Gaussian steps reflected at the patch edges, resampled by
    arc length, cut to the budget."""
    def fn(r, start, phi):
        rng = torch.Generator().manual_seed(int(seed) * 1000 + int(r))
        steps = (torch.randn(max(n_points * 4, 512), 2, generator=rng) * step_std).numpy()
        pos = plane.to_uv(start.detach().cpu().double()).numpy().clip(0, 1)
        pts = [pos.copy()]
        for st in steps:
            nxt = pos + st
            nxt = np.where(nxt < 0, -nxt, np.where(nxt > 1, 2 - nxt, nxt)).clip(0, 1)
            pts.append(nxt)
            pos = nxt
        walk = resample_arclength(_to_world_curve(plane, np.stack(pts)), n_points)
        return trim_to_length(walk, walk_len * plane.s)
    return fn


SWEEP_OFFSETS = (0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875)


def lawnmower_fn(plane, sweep_len, n_rounds, margin=0.06):
    """`sample_annealed_mission_overview.lawnmower_round_fn` on the patch: one coarse sweep per
    round, alternating direction, lane grid shifted to fill the gaps of earlier sweeps."""
    w = 1.0 - 2 * margin
    n_lanes = max(2, int(round((sweep_len - w) / max(w, 1e-9))))
    spacing = w / n_lanes
    sweeps, x_end = [], 0
    for k in range(n_rounds):
        ys = [margin + (i + SWEEP_OFFSETS[k % len(SWEEP_OFFSETS)]) * spacing for i in range(n_lanes)]
        if k % 2:
            ys = ys[::-1]
        pts, side = [], x_end
        for y in ys:
            x0, x1 = (margin, 1.0 - margin) if side == 0 else (1.0 - margin, margin)
            pts += [(x0, y), (x1, y)]
            side = 1 - side
        x_end = 0 if pts[-1][0] < 0.5 else 1
        sweeps.append(_to_world_curve(plane, np.array(pts)))

    def fn(r, start, phi):
        return sweeps[r] if r == 0 else torch.cat([start.detach().cpu().float()[None], sweeps[r]])
    return fn


def linear_ray_uv(start_uv, angle_deg, length, margin=0.04):
    """`init_baselines.linear_ray_path` in the chart: billiard ray in [margin, 1-margin]^2."""
    lo, hi = margin, 1.0 - margin
    start = np.asarray(start_uv, dtype=np.float64).reshape(2)
    p = np.clip(start, lo, hi)
    th = np.deg2rad(angle_deg)
    d = np.array([np.cos(th), np.sin(th)])
    pts = [start.copy()]
    if np.abs(p - start).max() > 1e-12:
        pts.append(p.copy())
    remaining = float(length)
    for _ in range(1000):
        if remaining <= 1e-9:
            break
        for k in range(2):
            if (p[k] <= lo + 1e-9 and d[k] < 0) or (p[k] >= hi - 1e-9 and d[k] > 0):
                d[k] = -d[k]
        t_wall = [np.inf, np.inf]
        for k in range(2):
            if d[k] > 1e-12:
                t_wall[k] = (hi - p[k]) / d[k]
            elif d[k] < -1e-12:
                t_wall[k] = (lo - p[k]) / d[k]
        t = min(t_wall[0], t_wall[1], remaining)
        p = p + t * d
        pts.append(p.copy())
        remaining -= t
        for k in range(2):
            if t_wall[k] <= t + 1e-12:
                d[k] = -d[k]
    pts = np.stack(pts)
    return pts if len(pts) > 1 else np.concatenate([pts, pts])


def ray_curves(plane, start, n_init, walk_len, n_points=N_POINTS):
    """(n_init, n_points, 3): straight rays in the plane from the agent, headings 360 i / n_init."""
    s_uv = plane.to_uv(start.detach().cpu().double()).numpy()
    return torch.stack([resample_arclength(_to_world_curve(plane, linear_ray_uv(s_uv, 360.0 * i / n_init, walk_len)),
                                           n_points) for i in range(n_init)])


def ray_init_fn(plane, n_init, walk_len):
    return lambda r, start, phi: ray_curves(plane, start, n_init, walk_len)


def best_ray_fn(plane, ee, n_init, walk_len):
    """`mission_baselines.best_ray_path_fn`: the ray with the lowest 3D ergodic error against the
    round's target, driven as it is."""
    def fn(r, start, phi):
        rays = ray_curves(plane, start, n_init, walk_len).to(ee.device)
        E = ee.energy(rays, ee.target_coeffs(phi.to(ee.device), plane))
        return rays[int(torch.argmin(E))].cpu()
    return fn


def gp_goal_fn(plane, walk_len, seed, source='mean', n_points=512, min_goal_dist=0.02, max_goals=500):
    """`mission_baselines.gp_goal_path_fn`: straight 3D lines to goals drawn from the GP posterior
    mean (or the round's target, `source='phi'`) until the round's budget is used up."""
    def fn(r, start, phi, mu=None, sd=None):
        gen = torch.Generator().manual_seed(int(seed) * 1000 + int(r))
        dens = (mu if source == 'mean' and mu is not None else phi).detach().float().cpu().clamp(min=0.0)
        res, flat = dens.shape[-1], dens.reshape(-1)
        if float(flat.max()) < 1e-3:
            flat = torch.ones_like(flat)
        cur = start.detach().cpu().double().numpy()
        pts, remaining = [cur.copy()], float(walk_len) * plane.s
        for _ in range(max_goals):
            if remaining <= 1e-9:
                break
            row, col = divmod(int(torch.multinomial(flat / flat.sum(), 1, generator=gen)), res)
            goal = plane.to_world(np.array([col / (res - 1), row / (res - 1)]))
            vec = goal - cur
            dist = float(np.linalg.norm(vec))
            if dist < min_goal_dist * plane.s:
                continue
            take = min(dist, remaining)
            cur = cur + vec / dist * take
            pts.append(cur.copy())
            remaining -= take
        if len(pts) < 2:
            pts.append(pts[0] + 1e-6)
        return resample_arclength(torch.as_tensor(np.stack(pts), dtype=torch.float32), n_points)
    fn.wants_belief = True
    return fn


def top_points(field, plane, n, min_dist, seed):
    """The `n` highest cells of `field` (GP_RES x GP_RES) with non-maximum suppression (a disc of
    `min_dist` plane units around every pick is excluded, as `init_baselines._find_peaks`), as 3D
    world points. A flat field (round 0: zero-mean GP without measurements) gets a seeded random
    tie-break, i.e. a random spread of points."""
    f = field.detach().double().cpu().reshape(-1).clone()
    if float(f.max() - f.min()) < 1e-3:
        f = f + 1e-6 * torch.rand(f.shape, generator=torch.Generator().manual_seed(int(seed)), dtype=f.dtype)
    uv = plane.grid_uv(field.shape[-1], dtype=torch.float64)
    picks = []
    for _ in range(n):
        i = int(torch.argmax(f))
        if not torch.isfinite(f[i]):
            break
        picks.append(uv[i])
        f[((uv - uv[i]) ** 2).sum(-1) < min_dist ** 2] = -float('inf')
    return plane.to_world(torch.stack(picks)).numpy()


def tsp_open_path(start, pts, exact_max=12):
    """Shortest open path start -> all `pts` (any order), 3D Euclidean distances. Exact Held-Karp
    dynamic programme for up to `exact_max` points, else nearest neighbour + 2-opt."""
    m = len(pts)
    if m <= 1:
        return list(range(m))
    D = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d0 = np.linalg.norm(pts - start[None], axis=-1)
    if m <= exact_max:
        full = 1 << m
        dp = np.full((full, m), np.inf)
        parent = np.full((full, m), -1, dtype=np.int64)
        for j in range(m):
            dp[1 << j, j] = d0[j]
        for mask in range(1, full):
            row = dp[mask]
            if not np.isfinite(row).any():
                continue
            ends = [j for j in range(m) if mask >> j & 1 and np.isfinite(row[j])]
            for k in range(m):
                if mask >> k & 1:
                    continue
                cand = row[ends] + D[ends, k]
                b = int(np.argmin(cand))
                nm = mask | (1 << k)
                if cand[b] < dp[nm, k]:
                    dp[nm, k], parent[nm, k] = cand[b], ends[b]
        j, mask = int(np.argmin(dp[full - 1])), full - 1
        order = []
        while j >= 0:
            order.append(j)
            pj = parent[mask, j]
            mask ^= 1 << j
            j = int(pj)
        return order[::-1]
    # nearest neighbour from the start, then 2-opt on the open path (start fixed at position 0)
    order, left = [], set(range(m))
    cur = None
    while left:
        cur = min(left, key=lambda k: d0[k] if cur is None else D[cur, k])
        order.append(cur)
        left.discard(cur)
    # full distance matrix with the start as node m
    Df = np.zeros((m + 1, m + 1))
    Df[:m, :m], Df[m, :m], Df[:m, m] = D, d0, d0
    seq = [m] + order
    improved = True
    while improved:
        improved = False
        for i in range(1, m):                    # reverse seq[i..k]; seq[0] = start stays
            a = seq[i - 1]
            for k in range(i + 1, m + 1):
                b, c = seq[i], seq[k]
                nxt = seq[k + 1] if k + 1 <= m else None
                delta = Df[a, c] - Df[a, b] + ((Df[b, nxt] - Df[c, nxt]) if nxt is not None else 0.0)
                if delta < -1e-12:
                    seq[i:k + 1] = seq[i:k + 1][::-1]
                    improved = True
                    a = seq[i - 1]
    return seq[1:]


def _tour_length(start, pts, order):
    p = np.concatenate([start[None], pts[order]])
    return float(np.linalg.norm(p[1:] - p[:-1], axis=-1).sum())


#: most belief maxima `tsp_fill` may use per round (the suppression disc limits it anyway: ~115
#: points fit into the patch at min_dist 0.1)
TSP_FILL_MAX = 200


def tsp_fn(plane, walk_len, seed, n_points_tsp=12, min_dist=0.10, source='mean', n_points=512,
           fill=False):
    """NEW baseline 'tsp': the `n_points_tsp` highest points of the current GP belief (posterior
    mean; `source='phi'`: the round's target density) with non-maximum suppression, connected by
    the shortest open travelling-salesman path from the agent (exact for <= 12 points), straight
    3D segments, cut to the round's budget. No network, no SVGD.

    `fill=True` (method 'tsp_fill'): if that tour is shorter than the budget, the next-highest
    maxima are added (in order of height) until the shortest tour through them reaches the budget
    (bisection over the number of points; nearest neighbour + 2-opt beyond 12 points), then cut
    to the budget -- so the method drives as far as the others. If every maximum the suppression
    leaves is used and the tour is still shorter, it stays shorter."""
    budget = walk_len * plane.s

    def fn(r, start, phi, mu=None, sd=None):
        field = mu if source == 'mean' and mu is not None else phi
        s = start.detach().cpu().double().numpy()
        if not fill:
            pts = top_points(field, plane, n_points_tsp, min_dist, seed * 1000 + r)
            order = tsp_open_path(s, pts)
        else:
            cand = top_points(field, plane, TSP_FILL_MAX, min_dist, seed * 1000 + r)
            lo = min(n_points_tsp, len(cand))
            order = tsp_open_path(s, cand[:lo])
            pts = cand[:lo]
            if _tour_length(s, pts, order) < budget and len(cand) > lo:
                best = None
                hi = len(cand)
                o_hi = tsp_open_path(s, cand)
                if _tour_length(s, cand, o_hi) < budget:
                    pts, order = cand, o_hi              # all maxima are not enough
                else:
                    best = (cand, o_hi)
                    while hi - lo > 1:                   # smallest m with tour >= budget
                        mid = (lo + hi) // 2
                        o = tsp_open_path(s, cand[:mid])
                        if _tour_length(s, cand[:mid], o) >= budget:
                            hi, best = mid, (cand[:mid], o)
                        else:
                            lo = mid
                    pts, order = best
        poly = torch.as_tensor(np.concatenate([s[None], pts[order]]), dtype=torch.float32)
        return resample_arclength(trim_to_length(poly, walk_len * plane.s), n_points)
    fn.wants_belief = True
    return fn


# ===========================================================================
# One mission
# ===========================================================================

def _now(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def run_mission(name, truth, plane, kernel, planner, bs, ee, B32, a, device, n_rounds, n_init,
                n_iters, kappas, start_uv, seed, cutoff=0.0, on_round=None, path_fn=None,
                init_fn=None):
    """3D port of `sample_annealed_mission.run_mission` (full execution, zero prior knowledge).
    Sources as there: `path_fn` (baseline path, no refinement), `init_fn` (network-free candidates
    + refinement) or `planner` (network candidates + refinement). Candidate selection: lowest 3D
    ergodic error against the round's own target. The driven curve is the winner with its start
    pinned, snapped onto the patch (`plane.snap`: removes the residual of the attachment term and
    clamps (u, v) to [0, 1] like the 2D clamp); `diag['plane_dev_max']` is the largest distance
    from the plane (plane units) BEFORE that snap."""
    belief = PlaneBelief(kernel, plane, seed, a, device)
    cells = plane.grid_world(GP_RES, device=device)
    start = plane.to_world(torch.as_tensor(start_uv, dtype=torch.float32)).to(device)
    s = plane.s
    segments, phis = [], []
    for r in range(n_rounds):
        t_0 = _now(device)
        torch.manual_seed(seed * 1000 + r)
        mu, sd = belief.posterior_grid()
        mu, sd = mu.float(), sd.float()
        visit = visitation(torch.cat(segments).to(device), cells, plane, a) if segments else None
        kap = kappas[r]
        phi, _ = debt_density(mu, sd, visit, kap, a)
        phi = phi.float()
        if float(phi.max()) < 1e-6:
            phi = torch.ones_like(phi)
        t_1 = _now(device)
        if path_fn is not None:
            raw = (path_fn(r, start, phi, mu=mu, sd=sd) if getattr(path_fn, 'wants_belief', False)
                   else path_fn(r, start, phi))
            winner = raw.to(device).float().clone()
            winner[0] = start
            E_init = E = torch.full((1,), float('nan'))
            t_2 = t_3 = _now(device)
        else:
            if init_fn is not None:
                curves = init_fn(r, start, phi).to(device).float()
            else:
                parts = particle_cloud(phi, plane, planner.n_particles, cutoff, device)
                cps = planner.plan(parts, n_init, start)
                curves = torch.einsum('pi,cid->cpd', B32, cps.to(device))
            phi_k = ee.target_coeffs(phi, plane)
            E_init = ee.energy(curves, phi_k)
            t_2 = _now(device)
            if n_iters <= 0:
                rhs = curves.permute(1, 0, 2).reshape(curves.shape[1], -1)
                P0 = torch.linalg.lstsq(B32, rhs).solution.reshape(B32.shape[1], n_init, 3).permute(1, 0, 2)
                P0 = P0.contiguous()
                P0[:, 0] = start
                final_cps = P0
            else:
                out = bs.run(curves.cpu().numpy(), phi_k.cpu().numpy(), start.cpu().numpy(),
                             [seed * 100000 + r * 1000 + i for i in range(n_init)], n_iters)
                final_cps = torch.as_tensor(out['final_cps'], device=device, dtype=torch.float32)
            final_curves = torch.einsum('pi,cid->cpd', B32, final_cps)
            E = ee.energy(final_curves, phi_k)
            sel = int(torch.argmin(E))
            winner = final_curves[sel].clone()
            winner[0] = start
            t_3 = _now(device)
        dev = float(plane.sdf(winner).abs().max()) / s
        winner = plane.snap(winner)
        n_pts = max(8, int(round(PTS_PER_UNIT * path_length(winner) / s / LENGTH_UNIT)))
        seg = resample_arclength(winner, n_pts)
        pts, vals = measure(seg, truth.to(device), plane, a)
        pts, vals = thin(pts, vals, a.max_obs)
        belief.observe(pts, vals)
        t_4 = _now(device)
        segments.append(seg.detach().cpu())
        phis.append(phi.detach().cpu())
        if on_round is not None:
            on_round(r, belief, segments, dict(
                E_init_best=float(E_init.min()), E_init_med=float(E_init.median()),
                E_ref_best=float(E.min()), E_ref_med=float(E.median()), plane_dev_max=dev,
                t_target=t_1 - t_0, t_init=t_2 - t_1, t_refine=t_3 - t_2, t_belief=t_4 - t_3,
                t_round=t_4 - t_0))
        start = seg[-1].detach()
        print(f'    [{name}] round {r}: kappa={kap:.3f}  path_len={path_length(seg) / s:.2f}  '
              f'n_obs={belief.n_obs}  plane_dev={dev:.1e}', flush=True)
    return segments, phis


# ===========================================================================
# Metrics against the ground truth
# ===========================================================================

GT_SUPPORT = 0.05
HI_LEVELS = (0.5, 0.75)
REVISIT_RADIUS = 0.03
W_SMOOTH_METRIC = 15.0


def _truth_like(truth, mu):
    t = truth.to(device=mu.device, dtype=mu.dtype)
    if t.shape != mu.shape:
        t = F.interpolate(t[None, None], size=mu.shape, mode='bilinear', align_corners=True)[0, 0]
    return t


def belief_quality(mu, sd, truth):
    """`run_mission_eval.belief_quality`."""
    mu, sd = mu.double(), sd.double().clamp(min=1e-3)
    t = _truth_like(truth, mu)
    err = mu - t
    a, b = mu.flatten() - mu.mean(), t.flatten() - t.mean()
    corr = float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-12))
    nll = float((torch.log(sd) + 0.5 * (err / sd) ** 2).mean() + 0.5 * np.log(2 * np.pi))
    return dict(rmse=float(err.pow(2).mean().sqrt()), corr=corr, nll=nll,
                cov95=float((err.abs() <= 1.96 * sd).double().mean()), sd_mean=float(sd.mean()))


class Evaluator3D:
    """Per-shape metrics (the 2D `run_exploration_comparison.Evaluator`), distances in 3D world
    coordinates converted to plane units; ergodic errors with the 3D cube metric."""

    def __init__(self, truth, plane, ee, device):
        self.truth, self.plane, self.ee, self.s = truth.float().cpu(), plane, ee, plane.s
        self.R = truth.shape[-1]
        self.cells = plane.grid_world(self.R)                          # (R*R, 3), CPU
        self.w = self.truth.reshape(-1).clamp(min=0.0)
        self.w_sum = float(self.w.sum().clamp(min=1e-12))
        self.hi = {lv: self.truth.reshape(-1) >= lv for lv in HI_LEVELS}
        self.phi_k_truth = ee.target_coeffs(self.truth.to(ee.device), plane)

    def truth_at(self, pts):
        uv = self.plane.to_uv(pts.cpu().float()).clamp(0, 1)
        idx = (uv * (self.R - 1)).round().long()
        return self.truth[idx[:, 1], idx[:, 0]]

    def path_stats(self, pts, prefix=''):
        t = self.truth_at(pts)
        return {f'{prefix}out_gt_frac': float((t < GT_SUPPORT).float().mean()),
                f'{prefix}in_hi50_frac': float((t >= 0.5).float().mean()),
                f'{prefix}in_hi75_frac': float((t >= 0.75).float().mean())}

    def dist_to(self, path):
        return torch.cdist(self.cells, path.cpu().float()).min(dim=1).values / self.s

    def coverage(self, driven):
        d = self.dist_to(driven)
        out = {'cov': float((d * self.w).sum() / self.w_sum)}
        for key, rad in (('swept03', 0.03), ('swept06', 0.06), ('swept10', 0.10)):
            out[key] = float(((d <= rad).float() * self.w).sum() / self.w_sum)
        for lv in HI_LEVELS:
            m = self.hi[lv]
            out[f'swept_hi{int(lv * 100)}'] = float((d[m] <= 0.06).float().mean()) if m.any() else float('nan')
        return out

    def belief(self, belief):
        mu, sd = belief.posterior_grid()
        mu, sd = mu.float().cpu(), sd.float().cpu()
        bq = belief_quality(mu, sd, self.truth)
        mu_n = mu.clamp(min=0.0)
        mu_n = mu_n / mu_n.max().clamp(min=1e-6)
        t = _truth_like(self.truth, mu_n)
        pa, pb = mu_n / mu_n.sum().clamp(min=1e-12), t / t.sum().clamp(min=1e-12)
        ma, mb = mu_n >= HI_LEVELS[0], t >= HI_LEVELS[0]
        iou = float((ma & mb).sum() / (ma | mb).sum().clamp(min=1))
        return mu, sd, mu_n, dict(belief_rmse=bq['rmse'], belief_corr=bq['corr'], belief_nll=bq['nll'],
                                  belief_cov95=bq['cov95'], belief_sd_mean=bq['sd_mean'],
                                  belief_tv=float(0.5 * (pa - pb).abs().sum()), belief_iou_hi=iou)

    def smoothness(self, seg):
        """`metrics_explore_exploit.smoothness_energy` in plane units (128-point resampling)."""
        c = resample_arclength(seg.float() / self.s, 128)
        acc = c[2:] - 2 * c[1:-1] + c[:-2]
        return float(W_SMOOTH_METRIC * acc.pow(2).sum())

    def turning_per_length(self, seg):
        v = seg[1:] - seg[:-1]
        v = v[v.norm(dim=-1) > 1e-9]
        if v.shape[0] < 2:
            return 0.0
        v = v / v.norm(dim=-1, keepdim=True)
        ang = torch.acos((v[1:] * v[:-1]).sum(-1).clamp(-1.0, 1.0))
        return float(ang.sum() / max(path_length(seg) / self.s, 1e-9))

    def round_metrics(self, driven, seg, earlier, belief, diag):
        ee = self.ee
        sc = ee.score(driven.to(ee.device), self.phi_k_truth)
        mu, sd, mu_n, bm = self.belief(belief)
        row = dict(E_truth=sc['E_ergodic_total'], E_truth_explore=sc['E_ergodic_explore'],
                   E_truth_exploit=sc['E_ergodic_exploit'],
                   E_belief=ee.score(driven.to(ee.device), ee.target_coeffs(mu_n.to(ee.device), self.plane))['E_ergodic_total'],
                   path_len=path_length(driven) / self.s, seg_len=path_length(seg) / self.s,
                   n_obs=int(belief.n_obs), **bm, **self.coverage(driven),
                   **self.path_stats(driven), **self.path_stats(seg, 'seg_'),
                   seg_smooth=self.smoothness(seg), seg_curv=self.turning_per_length(seg),
                   revisit_frac=(float((torch.cdist(seg, earlier).min(dim=1).values / self.s
                                        <= REVISIT_RADIUS).float().mean()) if earlier is not None else 0.0))
        row.update(diag)
        return row, mu, sd

    def steps_to_coverage(self, curve, threshold, sensor_radius=0.06):
        """`metrics_explore_exploit.steps_to_full_coverage` -> (fraction of the path, reached)."""
        if curve.shape[0] < 2:
            return 1.0, False
        d_run = torch.cummin(torch.cdist(self.cells, curve.float()) / self.s, dim=1).values
        frac = ((d_run <= sensor_radius).float() * self.w.unsqueeze(1)).sum(0) / self.w_sum
        seg = (curve[1:] - curve[:-1]).norm(dim=-1)
        cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
        hit = (frac >= threshold).nonzero(as_tuple=True)[0]
        if len(hit) == 0:
            return 1.0, False
        return float(cum[int(hit[0])] / cum[-1].clamp(min=1e-12)), True
