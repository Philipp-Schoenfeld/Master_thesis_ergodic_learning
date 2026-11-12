r"""
mission_baselines.py
=====================
Model-free path sources for the exploration comparison (`run_exploration_comparison.py`), in
the `path_fn(r, start, phi) -> (T, 2)` contract of `sample_annealed_mission.run_mission`
(random walk and lawnmower live in `init_baselines.py` / `sample_annealed_mission_overview.py`):

  gp_goal_path_fn   "straight line to a point sampled from the current GP estimate". Every round
                    draws a goal from the belief (probability proportional to the clipped
                    posterior mean, or to the round's target density with ``source='phi'``) and
                    drives a straight line to it. If the goal is reached before the round's path
                    budget is used up, the next goal is drawn from the same (not yet updated)
                    belief and the walk continues, so every round has the same budget as the other
                    methods. No network, no SVGD, the planner ignores the ergodic metric.
  best_ray_path_fn  "straight line initialisation": the `n_init` straight rays (billiard
                    reflection at the workspace walls) the ergodic searches start from, but
                    WITHOUT any search -- the ray with the lowest ergodic error against the
                    round's target density is driven as it is. Isolates what the SVGD refinement
                    adds on top of its own initialisation.

Both are budget matched: `walk_len` path length per round (default 6.1 requested = the networks'
mean path length per round in the exploration comparison).
"""
import numpy as np
import torch

from ergodic_energy_torch import coeffs_from_points
from exploration_optimierung.mission import resample_arclength
from init_baselines import linear_ray_path


def sample_goal(density, generator, floor=1e-3):
    """Draw one point (x, y) in [0, 1]^2 from a (R, R) density grid, indexed [y, x]. A density
    that is (almost) zero everywhere -- e.g. the zero-mean GP before the first measurement --
    falls back to the uniform distribution."""
    d = density.detach().float().cpu().clamp(min=0.0)
    res = d.shape[-1]
    flat = d.reshape(-1)
    if float(flat.max()) < floor:
        flat = torch.ones_like(flat)
    idx = int(torch.multinomial(flat / flat.sum(), 1, generator=generator))
    row, col = divmod(idx, res)
    return np.array([col / (res - 1), row / (res - 1)], dtype=np.float64)


def gp_goal_polyline(start, density, walk_len, generator, min_goal_dist=0.02, max_goals=500):
    """Vertices of the polyline start -> goal_1 -> goal_2 -> ... cut to `walk_len` in total length."""
    cur = np.asarray(start, dtype=np.float64).reshape(2)
    pts, remaining = [cur.copy()], float(walk_len)
    for _ in range(max_goals):
        if remaining <= 1e-9:
            break
        goal = sample_goal(density, generator)
        vec = goal - cur
        dist = float(np.linalg.norm(vec))
        if dist < min_goal_dist:                      # goal on top of the agent: draw again
            continue
        take = min(dist, remaining)
        cur = cur + vec / dist * take
        pts.append(cur.copy())
        remaining -= take
    if len(pts) < 2:
        pts.append(pts[0] + 1e-6)
    return np.stack(pts)


def gp_goal_path_fn(walk_len, seed, source='mean', n_points=512):
    """path_fn of the 'line to a GP-sampled point' baseline; `source` 'mean' = the GP posterior
    mean (the current GP estimate, default), 'phi' = the round's target density (UCB mixture of
    mean and uncertainty). It asks `run_mission` for the belief fields (`wants_belief`)."""
    if source not in ('mean', 'phi'):
        raise ValueError(f"source must be 'mean' or 'phi', got {source!r}")

    def fn(r, start, phi, mu=None, sd=None):
        gen = torch.Generator().manual_seed(int(seed) * 1000 + int(r))
        density = mu if (source == 'mean' and mu is not None) else phi
        poly = gp_goal_polyline(start.detach().cpu().numpy(), density, walk_len, gen)
        return resample_arclength(torch.as_tensor(poly, dtype=torch.float32), n_points)

    fn.wants_belief = True
    return fn


def best_ray_path_fn(ee, n_init, walk_len, n_points):
    """path_fn of the 'straight line initialisation' baseline (no refinement, see module doc)."""
    def fn(r, start, phi):
        s = start.detach().cpu().numpy()
        rays = torch.stack([linear_ray_path(s, 360.0 * i / n_init, walk_len, n_points)
                            for i in range(n_init)]).to(ee.device).float()
        phi_k = ee.target_coeffs(phi.to(ee.device))
        c = coeffs_from_points(rays, ee.k_idx)
        E = (ee.w * 0.5 * ee.Lambda * (c - phi_k.unsqueeze(0)) ** 2).sum(-1)
        return rays[int(torch.argmin(E))].cpu()
    return fn
