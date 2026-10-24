r"""
pipeline.py -- one cell of the teacher-trajectory study
========================================================
Goal (Philipp, 2026-10-24): find out how to generate the best teacher trajectories for the
training DB. Idea under test: optimise RAW WAYPOINTS (ergodic + very smooth), then PROJECT them
onto B-splines, which adds a further smoothing effect; optionally polish in control-point space.

One cell = (shape, config):

    init (C chords or teacher warm starts, start point x0 pinned)
      -> solver (tsvec | adam | lm | sun_sinkhorn | sun_stein) on waypoints or control points
      -> stage 'wp'            : the optimised waypoints (T = 128, time-uniform)
      -> stage 'bs<nxi>[d<g>]' : least-squares B-spline projection (clamped basis, CP 0 = x0)
      -> stage 'bs<nxi>_pol<k>': k Adam steps in control-point space after the projection

Every stage is scored with the SAME metrics (`stage_metrics`): the independent ergodicity
measures of `ergodic_validation.ergodicity_metrics` (Hellinger, tv_c, low_excess, ...), the
project's smoothness convention (`smoothness_and_pathlen`: W_SMOOTH * sum accel^2 on the
arclength-resampled path) and time-domain velocity / acceleration / jerk RMS and maxima, so the
effect of the projection can be read off as the difference between two stages.

Reuses (does not copy): `exploration/common/{tsvec_svgd,adam_svgd,sun_refine,sun_sinkhorn}`,
`ergodic_energy_torch`, `ergodic_validation/ergodicity_metrics`, `evaluation_full_matrix/
run_smoothness_comparison.smoothness_and_pathlen`.
"""
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_arch, os.path.join(_arch, 'exploration'), os.path.join(_arch, 'evaluation_full_matrix'),
           os.path.join(_arch, 'ergodic_dataset_generator'), os.path.join(_root, 'SE3_SVGD'),
           os.path.join(_root, 'src'), os.path.join(_root, 'bsplinax-main')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

N_POINTS = 128          # waypoints / render resolution (same as the smoothness comparison)
DEGREE = 5              # B-spline degree of the dataset pipeline
NXI = 25                # control points of the dataset pipeline
TRUTH_RES = 96
DB_PATH = os.path.join(_arch, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')
BETA = 100.0            # TSVEC / Adam temperature (see run_smoothness_comparison_v2)
ADAM_LR = 2e-3
POLISH_LR = 5e-4        # Adam step of the control-point polish after the projection


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Cfg:
    """One solver configuration.

    solver        'tsvec' | 'adam' | 'lm' | 'gui_adam' | 'sun_sinkhorn' | 'sun_stein'
                  ('gui_adam' = the GUI refiner: `BatchedSvgdTorch`, 8 particles per candidate, K=8)
    nxi           None = optimise raw waypoints; int = optimise that many B-spline control points
                  (degree DEGREE) directly (no projection stage needed, 'wp' is the rendered curve)
    w_smooth      weight of the smoothness force (0 = off); tsvec/adam/lm: on the energy, sun: on the
                  Stein / Sinkhorn gradient (`smoothness_weight`)
    smooth_order  1 velocity, 2 acceleration, 3 jerk (tsvec/adam/lm/sun_sinkhorn; sun_stein: 2)
    dynamics      sun only: 'pointmass' | 'jerk' (JerkPenalizedLQR)
    damping       tsvec only: Levenberg-Marquardt damping override (None = module default)
    init          'chord' (ring of chords from x0) | 'teacher' (current DB trajectory + smooth noise)
    n_iters       solver iterations
    """
    name: str
    solver: str
    nxi: Optional[int] = None
    w_smooth: float = 15.0
    smooth_order: int = 2
    dynamics: str = 'pointmass'
    damping: Optional[float] = None
    init: str = 'chord'
    n_iters: int = 1000
    K: int = 10                                  # Fourier modes per axis of the ergodic energy
    lr: Optional[float] = None                   # gui_adam only: Adam step size (None = 2e-3)
    schedule: Tuple[float, ...] = ()             # lm only: smoothness-weight continuation, n_iters split evenly
    # post-processing stages evaluated for every cell (cheap, all share the solver run)
    projections: Tuple[Tuple[int, int], ...] = ((25, 5), (20, 5), (35, 5), (25, 3))
    polish_iters: Tuple[int, ...] = (100, 300)
    polish_w: Optional[float] = None       # None -> w_smooth of the cell

    def to_dict(self):
        d = asdict(self)
        d['projections'] = [list(p) for p in self.projections]
        d['polish_iters'] = list(self.polish_iters)
        return d


# ---------------------------------------------------------------------------------------------
# data: shapes, start points, teacher trajectories
# ---------------------------------------------------------------------------------------------

@dataclass
class ShapeData:
    name: str
    density: np.ndarray          # (R, R) float32, max-normalised, [iy, ix]
    x0: np.ndarray               # (2,) start point
    teacher: np.ndarray          # (T_teacher, 2) DB trajectory incl. x0 as first point


def load_shapes(labels=None, split='val', db_path=DB_PATH, resolution=TRUTH_RES):
    """Holdout shapes with density, start point x0 and the teacher trajectory stored in the DB."""
    from shape_library import pdf_on_grid
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT shape_name, density_params, trajectory, x0 FROM ergodic_pairs "
                        "WHERE split=? ORDER BY id ASC", (split,)).fetchall()
    conn.close()
    out, seen = [], set()
    for name, params, blob, x0 in rows:
        if name in seen or (labels is not None and name not in labels):
            continue
        seen.add(name)
        d, _, _ = pdf_on_grid(json.loads(params), resolution=resolution)
        d = np.asarray(d, dtype=np.float32)
        x0 = np.asarray(json.loads(x0), dtype=np.float64)[:2]
        traj = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2).astype(np.float64)
        out.append(ShapeData(name, d / max(float(d.max()), 1e-12), x0, np.vstack([x0[None], traj])))
    if labels is not None:
        order = {n: i for i, n in enumerate(labels)}
        out.sort(key=lambda s: order[s.name])
    return out


def list_shape_names(split='val', db_path=DB_PATH):
    """Distinct shape names of a DB split in id order."""
    conn = sqlite3.connect(db_path)
    names = [r[0] for r in conn.execute("SELECT shape_name FROM ergodic_pairs WHERE split=? ORDER BY id ASC", (split,))]
    conn.close()
    return list(dict.fromkeys(names))


def resample_time(curve, n):
    """Linear time-uniform resampling (m, 2) -> (n, 2); batched over leading axes."""
    curve = np.asarray(curve, dtype=np.float64)
    m = curve.shape[-2]
    if m == n:
        return curve.copy()
    src = np.linspace(0.0, 1.0, m)
    dst = np.linspace(0.0, 1.0, n)
    flat = curve.reshape(-1, m, 2)
    out = np.empty((flat.shape[0], n, 2))
    for i, c in enumerate(flat):
        out[i, :, 0] = np.interp(dst, src, c[:, 0])
        out[i, :, 1] = np.interp(dst, src, c[:, 1])
    return out.reshape(curve.shape[:-2] + (n, 2))


# ---------------------------------------------------------------------------------------------
# B-spline projection
# ---------------------------------------------------------------------------------------------

_BASIS_CACHE = {}


def basis(nxi=NXI, n_points=N_POINTS, degree=DEGREE):
    """Clamped B-spline basis (n_points, nxi) float64 (curve = B @ control_points)."""
    key = (nxi, n_points, degree)
    if key not in _BASIS_CACHE:
        from obstacles import bspline_basis_matrix
        _BASIS_CACHE[key] = bspline_basis_matrix(nxi, n_points, degree).astype(np.float64)
    return _BASIS_CACHE[key]


def project_bspline(curves, nxi=NXI, degree=DEGREE, x0=None):
    """Least-squares control points of dense curves (C, T, 2) -> (C, nxi, 2).

    The fit is in the TIME parametrisation (index = time), because the ergodic measure is a time
    average: re-parametrising by arclength would change the visitation density. With the clamped
    basis B[0] = e_0, so control point 0 is the first curve point; if `x0` is given it is pinned
    exactly and the remaining points are fitted to the residual."""
    B = basis(nxi, curves.shape[1], degree)
    C, T, _ = curves.shape
    rhs = curves.transpose(1, 0, 2).reshape(T, C * 2)
    if x0 is None:
        P = np.linalg.lstsq(B, rhs, rcond=None)[0]
    else:
        x0 = np.asarray(x0, dtype=np.float64)
        rhs = rhs - np.outer(B[:, 0], np.tile(x0, C))
        P = np.empty((nxi, C * 2))
        P[0] = np.tile(x0, C)
        P[1:] = np.linalg.lstsq(B[:, 1:], rhs, rcond=None)[0]
    return P.reshape(nxi, C, 2).transpose(1, 0, 2)


def render(P, n_points=N_POINTS, degree=DEGREE):
    B = basis(P.shape[1], n_points, degree)
    return np.einsum('ti,cid->ctd', B, np.asarray(P, dtype=np.float64))


# ---------------------------------------------------------------------------------------------
# initial curves
# ---------------------------------------------------------------------------------------------

def chord_inits(x0, n, n_points=N_POINTS, radius=0.35, centre=(0.5, 0.5)):
    """n chords from the start point x0 to points on a ring around the workspace centre."""
    ang = 2.0 * np.pi * np.arange(n) / n
    ends = np.clip(np.asarray(centre) + radius * np.stack([np.cos(ang), np.sin(ang)], 1), 0.05, 0.95)
    s = np.linspace(0.0, 1.0, n_points)[None, :, None]
    return (1.0 - s) * np.asarray(x0)[None, None, :] + s * ends[:, None, :]


def teacher_inits(sd: ShapeData, n, seed, noise=0.02, n_points=N_POINTS):
    """The DB teacher (resampled) as candidate 0 plus n-1 smooth perturbations: Gaussian noise on
    the control points of its B-spline projection (start pinned)."""
    base = resample_time(sd.teacher, n_points)[None]
    P = project_bspline(base, NXI, DEGREE, x0=sd.x0)[0]
    rng = np.random.default_rng(seed)
    Ps = np.repeat(P[None], n, axis=0)
    Ps[1:, 1:] += noise * rng.standard_normal((n - 1, NXI - 1, 2))
    curves = render(np.clip(Ps, 0.03, 0.97))
    curves[0] = base[0]
    curves[:, 0] = sd.x0
    return curves


def make_inits(cfg: Cfg, sd: ShapeData, n, seed):
    if cfg.init == 'chord':
        return chord_inits(sd.x0, n)
    if cfg.init == 'teacher':
        return teacher_inits(sd, n, seed)
    raise ValueError(f'unknown init {cfg.init!r}')


# ---------------------------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------------------------

def _phi_k(density, k_idx, device):
    from ergodic_energy_torch import target_coeffs_from_grid
    return target_coeffs_from_grid(torch.as_tensor(density, dtype=torch.float32, device=device), k_idx)


def _param_init(cfg, curves):
    """-> (B or None, P0 (C, D_in, 2) float32): raw waypoints, or LS control points."""
    if cfg.nxi is None:
        return None, curves.astype(np.float32)
    B = basis(cfg.nxi, N_POINTS, DEGREE).astype(np.float32)
    return B, project_bspline(curves, cfg.nxi, DEGREE).astype(np.float32)


def _levenberg_marquardt(energy, P0, phi_k, n_iters, start=None):
    """Batched, per-candidate Levenberg-Marquardt on the TSVEC energy 0.5 |r|^2 (ergodic +
    smoothness + boundary residuals of `TsvecSvgd._residual_and_dX`). No swarm interaction:
    a deterministic point-estimate optimiser used as a reference for the SVGD solvers.
    Diversity comes from the initial curves only."""
    from common.tsvec_svgd import PARAM_CLIP
    dev, dt = energy.device, energy.dtype
    P = torch.as_tensor(P0, dtype=dt, device=dev).clone()
    if start is not None:
        start = torch.as_tensor(start, dtype=dt, device=dev)
        P[:, 0] = start
    N, D_in = P.shape[0], P.shape[1]
    D = D_in * 2
    BK = energy._dX_dP(N_POINTS, D_in)
    eye = torch.eye(D, device=dev, dtype=dt)

    def eval_point(Q):
        X = energy.render(Q)
        r, dr = energy._residual_and_dX(X, phi_k)
        J = torch.matmul(dr, BK.unsqueeze(0))
        return 0.5 * (r ** 2).sum(-1), J, r

    E, J, r = eval_point(P)
    lam = torch.full((N,), 1e-2, device=dev, dtype=dt)
    for _ in range(n_iters):
        H = torch.bmm(J.transpose(1, 2), J)
        g = torch.bmm(J.transpose(1, 2), r.unsqueeze(-1))
        diag = H.diagonal(dim1=1, dim2=2).mean(-1).view(N, 1, 1)
        lhs = H + lam.view(N, 1, 1) * diag * eye + 1e-9 * eye
        step = -torch.linalg.solve(lhs, g).squeeze(-1)
        Q = (P.reshape(N, D) + step).clamp(*PARAM_CLIP).reshape(P.shape)
        if start is not None:
            Q[:, 0] = start
        E_t, J_t, r_t = eval_point(Q)
        ok = E_t < E
        P = torch.where(ok.view(N, 1, 1), Q, P)
        E = torch.where(ok, E_t, E)
        J = torch.where(ok.view(N, 1, 1), J_t, J)
        r = torch.where(ok.view(N, 1), r_t, r)
        lam = torch.where(ok, (lam / 3.0).clamp(min=1e-6), (lam * 4.0).clamp(max=1e6))
    return P


def run_solver(cfg: Cfg, sd: ShapeData, curves, device):
    """-> dense final curves (C, T_out, 2) float64 (T_out = 128 or 201 for the Sun solvers)."""
    if cfg.solver in ('tsvec', 'adam', 'lm'):
        from common.adam_svgd import AdamSvgd
        from common.tsvec_svgd import TsvecSvgd
        B, P0 = _param_init(cfg, curves)
        if cfg.solver == 'tsvec':
            kw = {} if cfg.damping is None else dict(damping=cfg.damping)
            solver = TsvecSvgd(B=B, K=cfg.K, device=device, w_smooth=cfg.w_smooth, beta=BETA,
                               smooth_order=cfg.smooth_order, **kw)
            phi_k = _phi_k(sd.density, solver.k_idx, device)
            P = solver.run(P0, phi_k, n_iters=cfg.n_iters, record=False, start=sd.x0)['final']
            X = solver.render(P)
        elif cfg.solver == 'adam':
            solver = AdamSvgd(B=B, K=cfg.K, device=device, w_smooth=cfg.w_smooth, beta=BETA, lr=ADAM_LR,
                              smooth_order=cfg.smooth_order)
            phi_k = _phi_k(sd.density, solver.k_idx, device)
            P = solver.run(P0, phi_k, n_iters=cfg.n_iters, record=False, start=sd.x0)['final']
            X = solver.render(P)
        else:
            phases = cfg.schedule or (cfg.w_smooth,)
            P, per = P0, max(1, cfg.n_iters // len(phases))
            for w in phases:
                energy = TsvecSvgd(B=B, K=cfg.K, device=device, w_smooth=w, beta=1.0,
                                   smooth_order=cfg.smooth_order)
                phi_k = _phi_k(sd.density, energy.k_idx, device)
                P = _levenberg_marquardt(energy, P, phi_k, per, start=sd.x0)
            X = energy.render(P)
        return X.detach().cpu().numpy().astype(np.float64)

    if cfg.solver == 'gui_adam':
        # Philipp's reference setup (2026-10-24): Adam-SVGD of the GUI refiner (`SvgdRefiner`,
        # backend 'tsvec') on raw waypoints: every candidate its own swarm of 8 jittered copies, energy
        # 600 * ergodic (K=8) + 15 * accel^2 + boundary + start pin (W_START), best particle by energy,
        # then the hard snap of point 0 to the start.
        from svgd_batched import BatchedSvgdTorch
        from common.svgd_refine import SvgdRefiner
        if cfg.nxi is not None or cfg.smooth_order != 2:
            raise ValueError('gui_adam: raw waypoints and the acceleration force only')
        bs = BatchedSvgdTorch(np.eye(N_POINTS, dtype=np.float32), device, torch.float64,
                              compute_dtype=torch.float32 if str(device).startswith('cuda') else torch.float64,
                              smoothness_weight=cfg.w_smooth, lr=cfg.lr)
        phi_k = SvgdRefiner(0, backend='tsvec')._phi_k(sd.density)
        out = bs.run(np.asarray(curves, dtype=np.float64), phi_k, sd.x0, list(range(len(curves))),
                     cfg.n_iters, record=False)
        X = np.asarray(out['final_cps'], dtype=np.float64)
        X[:, 0] = sd.x0
        return X

    if cfg.nxi is not None:
        raise ValueError('sun solvers optimise a simulated trajectory; nxi must be None')
    init = np.asarray(curves, dtype=np.float64).copy()
    init[:, 0] = sd.x0
    if cfg.solver == 'sun_sinkhorn':
        from common import sun_sinkhorn
        out = sun_sinkhorn.run_batch(init, sd.density, cfg.n_iters, N_POINTS, record=False, log_space='raw',
                                     dynamics=cfg.dynamics, smoothness_weight=cfg.w_smooth,
                                     smooth_order=cfg.smooth_order)
        return np.asarray(out['final_pos'], dtype=np.float64)
    if cfg.solver == 'sun_stein':
        from common import sun_refine
        if cfg.smooth_order != 2:
            raise ValueError('sun_stein only supports the acceleration force (smooth_order=2)')
        out = sun_refine.run_batch(init, sd.density, sd.x0, cfg.n_iters, N_POINTS, record=False,
                                   smoothness_weight=cfg.w_smooth, log_space='raw', dynamics=cfg.dynamics)
        return np.asarray(out['final_pos'], dtype=np.float64)
    raise ValueError(f'unknown solver {cfg.solver!r}')


def polish(P0, sd: ShapeData, w_smooth, order, n_iters, device):
    """Independent Adam descent (no swarm interaction) on the TSVEC energy in control-point space,
    start pinned. P0: (C, nxi, 2) -> (C, nxi, 2)."""
    from common.adam_svgd import AdamSvgd
    B = basis(P0.shape[1], N_POINTS, DEGREE).astype(np.float32)
    solver = AdamSvgd(B=B, K=10, device=device, w_smooth=w_smooth, beta=BETA, lr=POLISH_LR,
                      use_kernel=False, smooth_order=order)
    phi_k = _phi_k(sd.density, solver.k_idx, device)
    out = solver.run(P0.astype(np.float32), phi_k, n_iters=n_iters, record=False, start=sd.x0)
    return out['final'].detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------

def _arclength_resample(X, n):
    seg = np.linalg.norm(np.diff(X, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-12:
        return np.repeat(X[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, X[:, 0]), np.interp(u, s, X[:, 1])], axis=1)


def derivative_stats(X):
    """Time-domain derivative statistics of time-uniform curves (C, T, 2), total time = 1, so
    v = dX * (T-1), a = d2X * (T-1)^2, j = d3X * (T-1)^3; plus the arclength curvature."""
    C, T, _ = X.shape
    h = float(T - 1)
    v = np.diff(X, axis=1) * h
    a = np.diff(X, n=2, axis=1) * h ** 2
    j = np.diff(X, n=3, axis=1) * h ** 3
    rms = lambda z: np.sqrt((z ** 2).sum(-1).mean(-1))
    mx = lambda z: np.linalg.norm(z, axis=-1).max(-1)
    kmax, k95 = np.empty(C), np.empty(C)
    for c in range(C):
        Y = _arclength_resample(X[c], T)
        d1 = np.gradient(Y, axis=0)
        d2 = np.gradient(d1, axis=0)
        sp = np.linalg.norm(d1, axis=1)
        kappa = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(sp, 1e-9) ** 3
        kappa = kappa[2:-2]                 # drop the one-sided boundary differences
        kmax[c], k95[c] = kappa.max(), np.percentile(kappa, 95)
    return dict(vel_rms=rms(v), acc_rms=rms(a), jerk_rms=rms(j), acc_max=mx(a), jerk_max=mx(j),
                curv_max=kmax, curv_p95=k95)


def stage_metrics(X, density, x0):
    """X: (C, T, 2) time-uniform curves -> dict of (C,) arrays (all keys numeric); `start_err` =
    distance of the first point to the required start point x0."""
    from ergodic_validation import ergodicity_metrics as em
    from run_smoothness_comparison import smoothness_and_pathlen
    m = em.evaluate(X, density)
    smooth, plen = smoothness_and_pathlen(X.astype(np.float32))
    m['smooth'] = smooth.astype(np.float64)
    m['path_len_arc'] = plen.astype(np.float64)
    m.update(derivative_stats(X))
    m['outside'] = ((X < 0.03) | (X > 0.97)).any(-1).mean(-1)
    m['start_err'] = np.linalg.norm(X[:, 0] - np.asarray(x0)[None], axis=-1)
    return {k: np.asarray(v, dtype=np.float64) for k, v in m.items()}


# ---------------------------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------------------------

def stage_name_projection(nxi, degree):
    return f'bs{nxi}' if degree == DEGREE else f'bs{nxi}d{degree}'


def run_cell(cfg: Cfg, sd: ShapeData, n_init, device, seed=0):
    """-> (stages, finals): stages = {stage_name: metrics dict of (C,) arrays}; finals =
    {stage_name: array} (waypoints (C, 128, 2) for 'wp', control points (C, nxi, 2) for 'bs*',
    rendered curve is B @ P)."""
    curves = make_inits(cfg, sd, n_init, seed)
    X = run_solver(cfg, sd, curves, device)
    X = resample_time(X, N_POINTS)
    # a diverged candidate (NaN / inf, e.g. an over-strong force in the Sun solvers) is replaced by
    # its init so the metrics stay computable, and flagged: analyses must drop `diverged == 1` rows
    bad = ~np.isfinite(X).all(axis=(1, 2))
    if bad.any():
        X[bad] = curves[bad]
    X = np.clip(X, 0.0, 1.0)
    X[:, 0] = sd.x0
    stages, finals = {}, {}
    stages['wp'] = stage_metrics(X, sd.density, sd.x0)
    finals['wp'] = X.astype(np.float32)

    projections = cfg.projections
    if cfg.nxi is not None:                    # already a B-spline curve: the projection is trivial
        projections = ()
    cps_by_stage = {}
    for nxi, deg in projections:
        P = project_bspline(X, nxi, deg, x0=sd.x0)
        name = stage_name_projection(nxi, deg)
        stages[name] = stage_metrics(render(P, N_POINTS, deg), sd.density, sd.x0)
        finals[name] = P.astype(np.float32)
        cps_by_stage[name] = P
    base = cps_by_stage.get(stage_name_projection(NXI, DEGREE))
    if base is not None and cfg.polish_iters:
        w = cfg.w_smooth if cfg.polish_w is None else cfg.polish_w
        for k in cfg.polish_iters:
            P = polish(base, sd, w, cfg.smooth_order, k, device)
            name = f'bs{NXI}_pol{k}'
            stages[name] = stage_metrics(render(P, N_POINTS, DEGREE), sd.density, sd.x0)
            finals[name] = P.astype(np.float32)
    for m in stages.values():
        m['diverged'] = bad.astype(np.float64)
    return stages, finals


def teacher_reference(sd: ShapeData):
    """Baseline row: the trajectory stored in the DB, resampled to 128 points ('wp') and projected
    to the dataset's B-spline ('bs25')."""
    X = resample_time(sd.teacher, N_POINTS)[None]
    stages = {'wp': stage_metrics(X, sd.density, sd.x0)}
    P = project_bspline(X, NXI, DEGREE, x0=sd.x0)
    stages[stage_name_projection(NXI, DEGREE)] = stage_metrics(render(P), sd.density, sd.x0)
    for m in stages.values():
        m['diverged'] = np.zeros(1)
    return stages, {'wp': X.astype(np.float32), stage_name_projection(NXI, DEGREE): P.astype(np.float32)}
