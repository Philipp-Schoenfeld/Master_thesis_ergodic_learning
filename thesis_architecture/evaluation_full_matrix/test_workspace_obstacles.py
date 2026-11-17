r"""
test_workspace_obstacles.py
===========================
Checks of the obstacle benchmark building blocks (`workspace_obstacles.py` and the obstacle term of
`common.tsvec_svgd.TsvecSvgd` / `common.adam_svgd.AdamSvgd`). CPU only, about a minute.

    python test_workspace_obstacles.py          # or: python -m pytest -q test_workspace_obstacles.py
"""
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sample_annealed_mission  # noqa: E402,F401  (sets up the remaining import paths)
import workspace_obstacles as wo  # noqa: E402
from common.adam_svgd import AdamSvgd  # noqa: E402
from common.tsvec_svgd import TsvecSvgd  # noqa: E402
from obstacles import bspline_basis_matrix  # noqa: E402

OBST = wo.make_obstacles('dice6', 0.08, 0.01)
RNG = np.random.default_rng(0)


def free_start():
    while True:
        p = RNG.uniform(0.05, 0.95, size=2)
        if wo.penetration(p[None], OBST, inflated=True)[0] == 0.0:
            return p


def assert_clear(path, what, inflated_tol=1e-6):
    st = wo.path_stats(path, OBST)
    assert st['pen_max'] == 0.0, f'{what}: physical penetration {st["pen_max"]:.4g}'
    p = path.detach().cpu().numpy() if torch.is_tensor(path) else path
    inf = wo.penetration(wo.densify(np.asarray(p, np.float64), 0.002), OBST, inflated=True).max()
    assert inf <= inflated_tol, f'{what}: inside the clearance margin by {inf:.4g}'


def test_layout():
    c, r, re = wo.arrays(OBST)
    assert c.shape == (6, 2) and np.allclose(r, 0.08) and np.allclose(re, 0.09)
    for s in ((0.5, 0.5), (0.06, 0.06)):            # default start / lawnmower corner are free
        assert wo.penetration(np.array([s]), OBST, inflated=True)[0] == 0.0
    m = wo.free_mask(OBST, 96, inflated=False)
    blocked = 1.0 - float(m.float().mean())
    assert 0.09 < blocked < 0.15, blocked


def test_detour_through_centre_and_tangent():
    for a, b in (((0.05, 0.5), (0.95, 0.5)), ((0.27, 0.02), (0.27, 0.98)), ((0.1, 0.1), (0.9, 0.9))):
        poly = wo.detour_segment(np.array(a), np.array(b), OBST)
        assert np.allclose(poly[0], a) and np.allclose(poly[-1], b)
        assert_clear(poly, f'detour {a}->{b}')


def test_reflect_rays():
    for _ in range(40):
        s = free_start()
        for ang in np.linspace(0, 360, 8, endpoint=False):
            ray = wo.reflect_ray_path(s, ang, 6.1, OBST, n_points=128)
            L = float((ray[1:] - ray[:-1]).norm(dim=-1).sum())
            # the 128-point resampling cuts the reflection corners (more bounces than without
            # obstacles: mean 5.86 vs. 5.98 measured), the polyline itself is exactly 6.1 long
            assert 5.4 < L <= 6.1 + 1e-4, L
            assert wo.path_stats(ray, OBST)['pen_max'] == 0.0
            assert torch.allclose(ray[0], torch.as_tensor(s, dtype=torch.float32))
            dense = wo.reflect_ray_path(s, ang, 6.1, OBST, n_points=4096)
            assert_clear(dense, f'ray from {s} at {ang}')


def test_random_walk():
    from common.metrics import trim_to_length
    for k in range(60):
        s = torch.as_tensor(free_start(), dtype=torch.float32)
        walk = trim_to_length(wo.random_walk_path(OBST, n_points=2048, seed=k, start=s), 6.1)
        assert_clear(walk, f'random walk {k}')
        # the mission drives the walk resampled to ~72 points per sqrt(2) of length
        from exploration_optimierung.mission import resample_arclength
        assert_clear(resample_arclength(walk, 311), f'random walk {k} as driven', inflated_tol=2e-3)


def test_lawnmower():
    fn = wo.lawnmower_round_fn(6.1, 5, OBST)
    start = torch.tensor([0.06, 0.06])
    for r in range(5):
        path = fn(r, start, None)
        assert_clear(path, f'lawnmower round {r}')
        start = path[-1]


def test_gp_goal():
    for k in range(10):
        mu = torch.rand(48, 48) ** 4
        fn = wo.gp_goal_path_fn(6.1, k, OBST)
        s = torch.as_tensor(free_start(), dtype=torch.float32)
        path = fn(0, s, mu, mu=mu, sd=None)
        L = float((path[1:] - path[:-1]).norm(dim=-1).sum())
        assert abs(L - 6.1) < 0.1, L                     # 512-point resampling cuts the arcs a bit
        assert_clear(path, f'gp goal {k}', inflated_tol=2e-4)


def test_clear_obstacles_keeps_start():
    B = torch.as_tensor(bspline_basis_matrix(25, 128, 5))
    start = torch.tensor([0.5, 0.5])
    x = torch.linspace(0.05, 0.95, 25)
    cps = torch.stack([torch.stack([x, torch.full_like(x, 0.2)], -1),       # straight through 2 centres
                       torch.stack([torch.full_like(x, 0.27), x], -1)])     # straight through 3 centres
    cps[:, 0] = start
    out = wo.clear_obstacles(cps, OBST, B, start)
    curves = torch.einsum('pi,bid->bpd', wo.dense_basis(25, 'cpu'), out)   # the rendering that is driven
    assert torch.equal(out[:, 0], start.expand(2, 2))
    assert float(wo.curve_violation(curves[:, 1:], OBST).max()) <= 1e-4
    for c in curves:
        assert wo.path_stats(c, OBST)['pen_max'] == 0.0


def _random_candidates(mode, n, seed):
    """'smooth': control points as a random walk from the start (like refined candidates);
    'harsh': uniform control points with 6 of them dropped into obstacle centres (pathological)."""
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    centres = wo.arrays(OBST)[0]
    for _ in range(n):
        start = torch.as_tensor(wo.push_out(g.random(2) * 0.9 + 0.05, OBST), dtype=torch.float32)
        if mode == 'smooth':
            cps = (start + (torch.randn(1, 25, 2) * 0.08).cumsum(1)).clamp(0.03, 0.97)
        else:
            cps = torch.rand(1, 25, 2) * 0.9 + 0.05
            k = g.integers(1, 25, size=6)
            cps[0, k] = (torch.as_tensor(centres[g.integers(0, 6, 6)], dtype=torch.float32)
                         + 0.03 * torch.randn(6, 2))
        cps[:, 0] = start
        yield start, cps


def test_clearance_smooth_candidates():
    Bd = wo.dense_basis(25, 'cpu')
    for start, cps in _random_candidates('smooth', 100, 1):
        out = wo.clear_obstacles(cps, OBST, None, start)
        cur = torch.einsum('pi,id->pd', Bd, out[0])
        assert torch.allclose(out[0, 0], start)
        assert float(wo.curve_violation(cur[None, 1:].double(), OBST)[0]) <= 1e-4
        assert_clear(cur, 'cleared smooth candidate', inflated_tol=2e-4)


def test_guarantee_clear_pathological():
    Bd = wo.dense_basis(25, 'cpu')
    for start, cps in _random_candidates('harsh', 100, 2):
        out = wo.clear_obstacles(cps, OBST, None, start)
        cur, _ = wo.guarantee_clear(torch.einsum('pi,id->pd', Bd, out[0]), OBST)
        assert torch.allclose(cur[0], start)
        assert wo.path_stats(cur, OBST)['pen_max'] == 0.0


def test_svgd_obstacle_term():
    B = bspline_basis_matrix(25, 128, 5)
    torch.manual_seed(0)
    P = 0.2 + 0.6 * torch.rand(4, 25, 2, dtype=torch.float64)
    phi = torch.rand(100, dtype=torch.float64) * 0.1
    s0 = TsvecSvgd(B, K=10, dtype=torch.float64)
    s1 = TsvecSvgd(B, K=10, dtype=torch.float64, obstacles=OBST, w_obstacle=500.0)
    sz = TsvecSvgd(B, K=10, dtype=torch.float64, obstacles=OBST, w_obstacle=0.0)
    X = s1.render(P)
    r0, J0 = s0._residual_and_dX(X, phi)
    rz, Jz = sz._residual_and_dX(X, phi)
    assert torch.equal(r0, rz) and torch.equal(J0, Jz)           # weight 0 = previous behaviour
    r1, J1 = s1._residual_and_dX(X, phi)
    assert torch.equal(r1[:, :r0.shape[1]], r0)
    BK = s1._dX_dP(128, 25)
    f = s1._residual_fn(phi)
    for n in range(P.shape[0]):
        Ja = torch.autograd.functional.jacobian(f, P[n].reshape(-1))
        assert float((J1[n] @ BK - Ja).abs().max()) < 1e-10


def _through_obstacles_init():
    B = bspline_basis_matrix(25, 128, 5)
    x = np.linspace(0.05, 0.95, 25)
    # slightly off-centre: a line exactly through a centre gets a purely tangential push
    rows = [np.stack([x, np.full_like(x, y + 0.015)], -1) for y in (0.2, 0.5, 0.8)]
    rows += [np.stack([np.full_like(x, xx - 0.015), x], -1) for xx in (0.27, 0.73)]
    return B, np.stack(rows).astype(np.float32)


def test_tsvec_run_reduces_violation():
    B, P0 = _through_obstacles_init()
    phi = np.zeros(100)
    phi[0] = 1.0
    Bt = torch.as_tensor(B)
    for w in (0.0, 1000.0):
        out = TsvecSvgd(B, K=10, obstacles=OBST, w_obstacle=w).run(P0, phi, n_iters=300, record=False)
        v = wo.curve_violation(torch.einsum('pi,nid->npd', Bt, out['final']), OBST)
        if w == 0.0:
            v_off = float(v.mean())
        else:
            # measured: 0.070 -> 0.013 (w=1000, 300 it.); the SVGD keeps a thermal residual, the
            # final clearance step of the mission removes it
            assert float(v.mean()) < 0.4 * v_off, (float(v.mean()), v_off)   # uniform target: 0.074 -> 0.026


def test_adam_run_reduces_violation():
    B, P0 = _through_obstacles_init()
    phi = np.zeros(100)
    phi[0] = 1.0
    Bt = torch.as_tensor(B)
    res = {}
    for w in (0.0, 1000.0):
        out = AdamSvgd(B, K=10, beta=100.0, obstacles=OBST, w_obstacle=w).run(P0, phi, n_iters=300,
                                                                            record=False)
        res[w] = float(wo.curve_violation(torch.einsum('pi,nid->npd', Bt, out['final']), OBST).mean())
    assert res[1000.0] < 0.25 * res[0.0], res          # Adam moves <= lr per step: slower


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'  ok  {t.__name__}')
    print(f'{len(tests)} tests passed')
