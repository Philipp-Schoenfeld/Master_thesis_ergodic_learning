r"""
workspace_obstacles.py
======================
Circular obstacles in the 2D workspace [0, 1]^2 for the active-sensing benchmark with obstacles
(`run_exploration_comparison.py --obstacles dice6`), and the obstacle-aware versions of the
model-free path sources. Everything here is opt-in: the benchmark without `--obstacles` never
imports anything from this module's path functions.

Layout 'dice6': six equal circles arranged like the six pips of a die (two columns x three rows),
centres x in {0.27, 0.73}, y in {0.2, 0.5, 0.8}, radius 0.08 plus a 0.01 clearance margin (the
same `obstacles.CircleObstacle` convention as the CFM inference guidance and the other baselines:
guidance and solvers keep out of `radius + margin`, the reported violation is measured against the
physical `radius`). The default start (0.5, 0.5) and the lawnmower corner (0.06, 0.06) are free.

Obstacle avoidance of the path sources (each one keeps its own character):
  straight rays (linray, and the initial rays of lintsvec / linadam)
                specular (billiard) reflection at the walls AND at the obstacle circles
  random walk   a Gaussian step whose straight segment would touch an obstacle is redrawn
  lawnmower     lanes (and the transfer to the first lane) that cross an obstacle go around it
                along the circle (the shorter arc)
  GP-goal line  goals are drawn from the belief restricted to the free space, the straight line to
                the goal goes around obstacles along the circle (the shorter arc)
"""
import math

import numpy as np
import torch

from exploration_optimierung.mission import resample_arclength
from obstacles import CircleObstacle, CompositeObstacle

DICE6_CENTERS = ((0.27, 0.2), (0.27, 0.5), (0.27, 0.8), (0.73, 0.2), (0.73, 0.5), (0.73, 0.8))
DEFAULT_RADIUS = 0.08
DEFAULT_MARGIN = 0.01
LAYOUTS = {'dice6': DICE6_CENTERS}
#: arcs are laid this far outside the inflated radius, so rounding never puts a point inside
ARC_EPS = 1e-3
#: spacing of the points on a detour arc
ARC_STEP = 0.005


def make_obstacles(layout='dice6', radius=DEFAULT_RADIUS, margin=DEFAULT_MARGIN):
    """-> `obstacles.CompositeObstacle` of the named layout (None for layout None / 'none')."""
    if layout in (None, '', 'none'):
        return None
    if layout not in LAYOUTS:
        raise ValueError(f'unknown obstacle layout {layout!r}; known: {sorted(LAYOUTS)}')
    return CompositeObstacle([CircleObstacle(c, radius, margin) for c in LAYOUTS[layout]])


def tag(layout, radius):
    """Name suffix of the output folder / DB, e.g. 'obstacles_dice6_r0.08'."""
    return f'obstacles_{layout}_r{radius:g}'


def arrays(obst):
    """-> centres (M, 2), physical radii (M,), inflated radii (M,) as float64 numpy arrays."""
    parts = obst.obstacles
    return (np.array([o.center for o in parts], dtype=np.float64),
            np.array([o.radius for o in parts], dtype=np.float64),
            np.array([o.effective_radius for o in parts], dtype=np.float64))


# ===========================================================================
# Free-space masks and violation statistics
# ===========================================================================

def free_mask(obst, res, inflated=True, device='cpu'):
    """(res, res) bool tensor, indexed [y, x] on linspace(0, 1, res) like every density grid of the
    pipeline: True where the cell centre lies outside every obstacle."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, res), torch.linspace(0, 1, res), indexing='ij')
    m = obst.mask(xs.numpy(), ys.numpy(), inflated=inflated)
    return torch.as_tensor(~m, device=device)


def mask_density(density, obst, inflated=True):
    """Density grid with the obstacle cells set to 0 (same dtype / device)."""
    m = free_mask(obst, density.shape[-1], inflated=inflated, device=density.device)
    return density * m.to(density.dtype)


def penetration(pts, obst, inflated=False):
    """(n, 2) points -> (n,) depth inside the deepest obstacle (0 outside); physical radius unless
    `inflated`."""
    p = pts.detach().cpu().double().numpy() if torch.is_tensor(pts) else np.asarray(pts, np.float64)
    c, r, re = arrays(obst)
    d = np.linalg.norm(p[:, None, :] - c[None], axis=-1)
    return np.clip((re if inflated else r)[None] - d, 0.0, None).max(axis=1)


def path_stats(pts, obst):
    """Violation statistics of a driven path against the physical obstacles: deepest penetration,
    share of points inside, and the share of the path LENGTH inside (dense check on the segments)."""
    p = pts.detach().cpu().double().numpy() if torch.is_tensor(pts) else np.asarray(pts, np.float64)
    if len(p) < 2:
        pen = penetration(p, obst)
        return dict(pen_max=float(pen.max()), in_frac=float((pen > 0).mean()), len_in_frac=0.0)
    dense = densify(p, 0.002)
    pen = penetration(dense, obst)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=-1)
    inside = (pen[1:] > 0) | (pen[:-1] > 0)
    return dict(pen_max=float(pen.max()), in_frac=float((penetration(p, obst) > 0).mean()),
                len_in_frac=float(seg[inside].sum() / max(seg.sum(), 1e-12)))


def densify(p, step):
    """Polyline (n, 2) -> the same polyline with extra points so no segment is longer than `step`."""
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        t = np.linspace(0.0, 1.0, k + 1)[1:, None]
        out.append(a + t * (b - a))
    return np.concatenate(out, axis=0)


def segment_hits(a, b, c, r):
    """-> True if the closed segment a-b comes closer than r to the centre c."""
    ab = b - a
    L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-18 else float(np.clip((c - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - c)) < r


def segment_hits_any(a, b, c, r):
    """Vectorised `segment_hits` over all obstacles: c (M, 2), r (M,) -> bool."""
    ab = b - a
    L2 = float(ab @ ab)
    t = np.zeros(len(c)) if L2 < 1e-18 else np.clip((c - a) @ ab / L2, 0.0, 1.0)
    d = a[None] + t[:, None] * ab[None] - c
    return bool(((d * d).sum(-1) < r * r).any())


def push_out(p, obst):
    """Point (2,) radially out of any inflated obstacle it lies in (onto radius + ARC_EPS)."""
    c, _, re = arrays(obst)
    p = np.asarray(p, dtype=np.float64).copy()
    for cm, rm in zip(c, re):
        d = p - cm
        n = float(np.linalg.norm(d))
        if n < rm:
            d = d / n if n > 1e-12 else np.array([1.0, 0.0])
            p = cm + (rm + ARC_EPS) * d
    return p


# ===========================================================================
# Detours: straight segments around the obstacles along the circle
# ===========================================================================

def _arc(c, r, a, b, side_hint):
    """Points on the circle (c, r) from a to b (both on the circle), along the shorter arc; on a
    tie (segment through the centre) the side `side_hint` (+1 counter-clockwise, -1 clockwise)."""
    th0 = math.atan2(a[1] - c[1], a[0] - c[0])
    th1 = math.atan2(b[1] - c[1], b[0] - c[0])
    ccw = (th1 - th0) % (2 * math.pi)                 # counter-clockwise sweep in [0, 2 pi)
    cw = ccw - 2 * math.pi                            # clockwise sweep in (-2 pi, 0]
    if abs(abs(ccw) - abs(cw)) < 1e-9:
        sweep = ccw if side_hint >= 0 else cw
    else:
        sweep = ccw if abs(ccw) < abs(cw) else cw
    n = max(2, int(math.ceil(abs(sweep) * r / ARC_STEP)) + 1)
    th = th0 + np.linspace(0.0, sweep, n)
    return np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)], axis=-1)


def detour_segment(a, b, obst):
    """Straight segment a -> b as a polyline (k, 2) that goes around every inflated obstacle it
    crosses, along the circle. Endpoints inside an obstacle are pushed out first."""
    a, b = push_out(a, obst), push_out(b, obst)
    c, _, re = arrays(obst)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-12:
        return np.stack([a, b])
    u = d / L
    hits = []
    for cm, rm in zip(c, re):
        R = rm + ARC_EPS
        f = a - cm
        bq = float(f @ u)
        disc = bq * bq - (float(f @ f) - R * R)
        if disc <= 0.0:
            continue
        s = math.sqrt(disc)
        t0, t1 = -bq - s, -bq + s
        if t1 <= 0.0 or t0 >= L:
            continue
        hits.append((max(t0, 0.0), min(t1, L), cm, R))
    if not hits:
        return np.stack([a, b])
    hits.sort(key=lambda h: h[0])
    pts = [a]
    for t0, t1, cm, R in hits:
        # shorter arc; a segment exactly through the centre goes counter-clockwise
        pts.extend(_arc(cm, R, a + t0 * u, a + t1 * u, 1.0))
    pts.append(b)
    return np.asarray(pts)


def detour_polyline(vertices, obst):
    """Polyline (n, 2) -> polyline with every segment routed around the obstacles."""
    v = np.asarray(vertices, dtype=np.float64)
    out = [detour_segment(v[0], v[1], obst)]
    for a, b in zip(v[1:-1], v[2:]):
        out.append(detour_segment(a, b, obst)[1:])
    return np.concatenate(out, axis=0)


def polyline_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum())


def truncate_polyline(p, max_len):
    """First `max_len` of path length of the polyline (n, 2), last point interpolated."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= max_len:
        return p
    k = int(np.searchsorted(cum, max_len))
    t = (max_len - cum[k - 1]) / max(seg[k - 1], 1e-12)
    return np.concatenate([p[:k], (p[k - 1] + t * (p[k] - p[k - 1]))[None]], axis=0)


# ===========================================================================
# Obstacle-aware path sources
# ===========================================================================

def reflect_ray_path(start, angle_deg, length, obst, n_points=128, margin=0.04):
    """`init_baselines.linear_ray_path` with obstacles: the straight ray leaves `start` at
    `angle_deg` and is reflected specularly at the walls of [margin, 1-margin]^2 AND at the inflated
    obstacle circles (billiard path), `length` long in total."""
    lo, hi = margin, 1.0 - margin
    c, _, re = arrays(obst)
    start = np.asarray(start, dtype=np.float64).reshape(2)
    p = push_out(np.clip(start, lo, hi), obst)
    th = np.deg2rad(angle_deg)
    d = np.array([np.cos(th), np.sin(th)])
    pts = [start.copy()]
    if np.abs(p - start).max() > 1e-12:
        pts.append(p.copy())
    remaining = float(length)
    for _ in range(4000):
        if remaining <= 1e-9:
            break
        for k in range(2):                              # turn away from a wall we sit on
            if (p[k] <= lo + 1e-9 and d[k] < 0) or (p[k] >= hi - 1e-9 and d[k] > 0):
                d[k] = -d[k]
        t_wall = [np.inf, np.inf]
        for k in range(2):
            if d[k] > 1e-12:
                t_wall[k] = (hi - p[k]) / d[k]
            elif d[k] < -1e-12:
                t_wall[k] = (lo - p[k]) / d[k]
        t_obs, m_hit = np.inf, -1
        for m, (cm, rm) in enumerate(zip(c, re)):
            f = p - cm
            if float(f @ d) >= 0.0:                     # moving away from (or tangent to) this circle
                continue
            bq = float(f @ d)
            disc = bq * bq - (float(f @ f) - rm * rm)
            if disc <= 0.0:
                continue
            t0 = -bq - math.sqrt(disc)
            if 1e-10 < t0 < t_obs:
                t_obs, m_hit = t0, m
        t = min(t_wall[0], t_wall[1], t_obs, remaining)
        p = p + t * d
        pts.append(p.copy())
        remaining -= t
        if m_hit >= 0 and t_obs <= t + 1e-12:
            n = (p - c[m_hit]) / max(np.linalg.norm(p - c[m_hit]), 1e-12)
            d = d - 2.0 * float(d @ n) * n
            p = c[m_hit] + (re[m_hit] + 1e-9) * n       # sit exactly on the rim, outside
        else:
            for k in range(2):
                if t_wall[k] <= t + 1e-12:
                    d[k] = -d[k]
    curve = torch.tensor(np.stack(pts), dtype=torch.float32)
    if curve.shape[0] < 2:
        curve = torch.cat([curve, curve], dim=0)
    return resample_arclength(curve, n_points)


def random_walk_path(obst, n_points=128, seed=0, step_std=0.035, start=None, max_redraw=50):
    """`init_baselines.random_walk_path` with obstacles: Gaussian steps reflected at the walls; a
    step whose straight segment would come within the inflated radius of an obstacle is redrawn
    (up to `max_redraw` times, then the walker stays put for that step). After the arc-length
    resampling to `n_points` (as without obstacles) the chords are routed around the obstacles,
    so the result has slightly more than `n_points` points."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    c, _, re = arrays(obst)
    raw_n = max(n_points * 4, 512)
    pos = (start.clone().double().numpy() if start is not None else np.array([0.5, 0.5]))
    pos = push_out(pos, obst)
    pts = [pos.copy()]
    for _ in range(raw_n):
        for _try in range(max_redraw):
            s = (torch.randn(2, generator=g) * step_std).double().numpy()
            nxt = pos + s
            for k in range(2):                          # reflect at [0, 1]^2
                if nxt[k] < 0:
                    nxt[k] = -nxt[k]
                elif nxt[k] > 1:
                    nxt[k] = 2 - nxt[k]
            nxt = np.clip(nxt, 0.0, 1.0)
            if not segment_hits_any(pos, nxt, c, re):
                break
        else:
            nxt = pos
        pts.append(nxt.copy())
        pos = nxt
    curve = resample_arclength(torch.tensor(np.stack(pts), dtype=torch.float32), n_points)
    # the arc-length resampling (as in the obstacle-free walk) turns the many short steps into
    # long chords (~0.17 for 2048 points) that can cut an obstacle although every step is clear;
    # those chords go around the obstacle along the circle instead
    return torch.as_tensor(detour_polyline(curve.double().numpy(), obst), dtype=torch.float32)


def lawnmower_round_fn(sweep_len, n_rounds, obst, margin=0.06):
    """`sample_annealed_mission_overview.lawnmower_round_fn` with obstacles: the same lane plan,
    every lane (and the transfer from the agent to the first lane) routed around the obstacles."""
    from sample_annealed_mission_overview import lawnmower_round_fn as plain
    base = plain(sweep_len, n_rounds, margin)

    def fn(r, start, phi):
        verts = base(r, start, phi).double().numpy()
        return torch.as_tensor(detour_polyline(verts, obst), dtype=torch.float32)
    return fn


def sample_free_goal(density, obst, generator, floor=1e-3, max_tries=100):
    """`mission_baselines.sample_goal` from the density restricted to the free space."""
    from mission_baselines import sample_goal
    d = mask_density(density.detach().float().cpu().clamp(min=0.0), obst)
    for _ in range(max_tries):
        goal = sample_goal(d, generator, floor=floor)
        if penetration(goal[None], obst, inflated=True)[0] <= 0.0:
            return goal
    return push_out(goal, obst)


def gp_goal_path_fn(walk_len, seed, obst, source='mean', n_points=512, min_goal_dist=0.02,
                    max_goals=500):
    """`mission_baselines.gp_goal_path_fn` with obstacles: goals from the free part of the belief,
    each leg routed around the obstacles, the polyline cut to `walk_len`."""
    if source not in ('mean', 'phi'):
        raise ValueError(f"source must be 'mean' or 'phi', got {source!r}")

    def fn(r, start, phi, mu=None, sd=None):
        gen = torch.Generator().manual_seed(int(seed) * 1000 + int(r))
        density = mu if (source == 'mean' and mu is not None) else phi
        cur = push_out(start.detach().cpu().double().numpy().reshape(2), obst)
        pts, remaining = [cur.copy()], float(walk_len)
        for _ in range(max_goals):
            if remaining <= 1e-9:
                break
            goal = sample_free_goal(density, obst, gen)
            if float(np.linalg.norm(goal - cur)) < min_goal_dist:
                continue
            leg = detour_segment(cur, goal, obst)
            L = polyline_length(leg)
            if L <= remaining:
                pts.extend(leg[1:])
                remaining -= L
                cur = leg[-1]
            else:
                pts.extend(truncate_polyline(leg, remaining)[1:])
                remaining = 0.0
        if len(pts) < 2:
            pts.append(pts[0] + 1e-6)
        return resample_arclength(torch.as_tensor(np.stack(pts), dtype=torch.float32), n_points)

    fn.wants_belief = True
    return fn


def best_ray_path_fn(ee, n_init, walk_len, n_points, obst):
    """`mission_baselines.best_ray_path_fn` with obstacle-reflected rays."""
    from ergodic_energy_torch import coeffs_from_points

    def fn(r, start, phi):
        s = start.detach().cpu().numpy()
        rays = torch.stack([reflect_ray_path(s, 360.0 * i / n_init, walk_len, obst, n_points)
                            for i in range(n_init)]).to(ee.device).float()
        phi_k = ee.target_coeffs(phi.to(ee.device))
        c = coeffs_from_points(rays, ee.k_idx)
        E = (ee.w * 0.5 * ee.Lambda * (c - phi_k.unsqueeze(0)) ** 2).sum(-1)
        return rays[int(torch.argmin(E))].cpu()
    return fn


def ray_init_fn(n_init, walk_len, n_points, obst):
    """init_fn of lintsvec / linadam with obstacle-reflected rays."""
    def fn(r, start, phi):
        s = start.detach().cpu().numpy()
        return torch.stack([reflect_ray_path(s, 360.0 * i / n_init, walk_len, obst, n_points)
                            for i in range(n_init)])
    return fn


# ===========================================================================
# Candidate selection and the final clearance step of the SVGD-refined methods
# ===========================================================================

def curve_violation(curves, obst):
    """(C, T, 2) dense curves -> (C,) deepest penetration into the INFLATED obstacles (the zone the
    planners are asked to keep out of)."""
    c, _, re = arrays(obst)
    ct = torch.as_tensor(c, dtype=curves.dtype, device=curves.device)
    rt = torch.as_tensor(re, dtype=curves.dtype, device=curves.device)
    d = (curves.unsqueeze(2) - ct.view(1, 1, -1, 2)).norm(dim=-1)
    return torch.clamp(rt.view(1, 1, -1) - d, min=0.0).amax(dim=(1, 2))


def select_feasible(E, viol, tol=1e-4):
    """Index of the candidate with the lowest ergodic error among those whose violation is <= tol;
    if none is feasible, the least violating one."""
    ok = viol <= tol
    if bool(ok.any()):
        Em = torch.where(ok, E, torch.full_like(E, float('inf')))
        return int(torch.argmin(Em))
    return int(torch.argmin(viol))


#: weight of the points inside an obstacle in the least-squares step of `clear_obstacles`
WEIGHT_INSIDE = 50.0
#: points of the dense rendering used for the clearance check and for driving the winner: with the
#: 128-point rendering a fast stretch of a curve has chords long enough to cut an obstacle edge
DENSE_PTS = 4096


def dense_basis(nxi, device, dtype=torch.float32, deg=5):
    from obstacles import basis_torch
    return basis_torch(nxi, DENSE_PTS, deg, device=device, dtype=dtype)


def _project_out_points(curve, obst, push=2e-3):
    """(C, T, 2) -> target points: every point inside an inflated obstacle moved radially onto
    radius + `push`, all other points unchanged."""
    c, _, re = arrays(obst)
    tgt = curve.clone()
    for cm, rm in zip(c, re):
        ct = torch.as_tensor(cm, dtype=curve.dtype, device=curve.device)
        d = tgt - ct
        n = d.norm(dim=-1, keepdim=True)
        inside = (n < rm).squeeze(-1)
        if bool(inside.any()):
            u = d / n.clamp(min=1e-12)
            tgt = torch.where(inside.unsqueeze(-1), ct + (rm + push) * u, tgt)
    return tgt


def clear_obstacles(cps, obst, B=None, start=None, max_iters=100, tol=1e-4, damping=1e-4,
                    w_inside=WEIGHT_INSIDE, nudge_depth=0.01):
    """Final clearance step of a B-spline candidate, control point 0 held at `start` so the round
    still begins where the agent is. cps: (C, nxi, 2); checked on the dense `DENSE_PTS` rendering
    (`B` is only used for its device). `tol` is the remaining depth inside the INFLATED radius, so
    the physical obstacle stays at least `margin - tol` away.

    Gauss-Newton on the hinge penalty: every rendered point inside an obstacle gets the target
    "radially onto the rim (+2e-3)", every other point the target "stay"; the free control points
    move by the damped, weighted least-squares fit of that displacement field (inside points
    weighted `w_inside`, outside points 1, so the many "stay" targets cannot hold the inside points
    back) -- a few iterations instead of
    the slow linear convergence of plain descent, which left violations behind on the cluster
    test). A curve that does not improve (runs exactly through a centre, where the radial target
    lies along the curve) is nudged sideways as in `obstacles._break_symmetry`."""
    from obstacles import _break_symmetry
    cps = torch.as_tensor(cps)
    dev = B.device if B is not None else cps.device
    Bd = dense_basis(cps.shape[1], dev, dtype=torch.float64)
    x = cps.to(dtype=torch.float64, device=dev).clone()
    if start is not None:
        st = torch.as_tensor(start, dtype=x.dtype, device=dev).reshape(1, 2)
        x[:, 0] = st
    free = slice(1, None) if start is not None else slice(0, None)
    Bf = Bd[:, free]                                                    # (T, nf)
    eye = torch.eye(Bf.shape[1], dtype=Bf.dtype, device=dev)
    last = None
    for _ in range(max_iters):
        curve = torch.einsum('pi,bid->bpd', Bd, x)
        v = curve_violation(curve[:, 1:] if start is not None else curve, obst)
        if float(v.max()) <= tol:
            break
        resid = _project_out_points(curve, obst) - curve                # (C, T, 2)
        w = 1.0 + (w_inside - 1.0) * (resid.abs().sum(-1) > 0).to(Bf.dtype)   # (C, T)
        for b in range(x.shape[0]):
            if float(v[b]) <= tol:
                continue
            A = Bf.T @ (w[b, :, None] * Bf)
            A = A + damping * torch.trace(A) / A.shape[0] * eye
            rhs = Bf.T @ (w[b, :, None] * resid[b])                     # (nf, 2)
            x[b, free] = x[b, free] + torch.linalg.solve(A, rhs)
        if last is not None:
            stuck = (v > nudge_depth) & (v > 0.95 * last)
            if bool(stuck.any()):
                deep = max(obst.obstacles, key=lambda o: float(torch.clamp(
                    o.effective_radius - (curve - torch.as_tensor(o.center, dtype=curve.dtype,
                                                                  device=dev)).norm(dim=-1),
                    min=0.0).max()))
                x = _break_symmetry(x, deep, Bd, active=stuck)
                if start is not None:
                    x[:, 0] = st
        last = v
    return x.to(dtype=torch.as_tensor(cps).dtype if torch.is_floating_point(torch.as_tensor(cps))
                else torch.float32)


def guarantee_clear(curve, obst, tol=1e-4):
    """Last line of defence for a driven dense curve (T, 2) (point 0 = the agent, outside): if
    any later point is still inside an inflated obstacle, the points inside are moved radially onto
    the rim and every chord that still crosses an obstacle is replaced by the circle arc
    (`detour_polyline`), so the result is collision-free by construction.
    -> (curve, 1.0 if the detour was needed else 0.0)."""
    p = curve.detach().cpu().double().numpy()
    # checked on the chords as well (densified), not only on the points: the curve is driven as a
    # polyline, and a fast stretch has chords long enough to cut the clearance margin
    if penetration(densify(p, 0.002)[1:], obst, inflated=True).max() <= tol:
        return curve, 0.0
    poly = detour_polyline(p, obst)
    return torch.as_tensor(poly, dtype=curve.dtype, device=curve.device), 1.0


def draw(ax, obst, zorder=2.5):
    """Obstacles in the project style (grey disc, dotted clearance margin)."""
    if obst is not None:
        obst.draw(ax, zorder=zorder)
