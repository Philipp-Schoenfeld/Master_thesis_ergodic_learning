"""Tests of the 3D exploration comparison building blocks (plane3d.py, mission_plane3d.py).

    python -m pytest test_plane3d.py -q
"""
import itertools

import numpy as np
import torch

import mission_plane3d as mp   # noqa: F401  (3D import path first)
import plane3d as p3


def test_plane_chart_roundtrip_and_sdf():
    for key in p3.PLANE_NORMALS:
        pl = p3.Plane(key)
        uv = torch.rand(20, 2, dtype=torch.float64)
        x = pl.to_world(uv)
        assert torch.allclose(pl.to_uv(x), uv, atol=1e-12)
        assert float(pl.sdf(x).abs().max()) < 1e-12
        assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0       # the patch lies in the unit cube
        # side length of the patch is the plane scale
        assert abs(float((pl.to_world(torch.tensor([[1.0, 0.0]], dtype=torch.float64))
                          - pl.to_world(torch.tensor([[0.0, 0.0]], dtype=torch.float64))).norm()) - pl.s) < 1e-12


def test_flat_plane_is_the_training_plane():
    pl = p3.Plane('flat')
    np.testing.assert_allclose(pl.n, [0, 0, 1])
    np.testing.assert_allclose(pl.to_world(np.array([0.5, 0.5])), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(pl.to_world(np.array([0.0, 0.0])), [p3.MARGIN, p3.MARGIN, 0.5])


def test_tsvec3d_jacobian_matches_finite_differences():
    torch.manual_seed(0)
    for key in ('flat', 'diagonal'):
        pl = p3.Plane(key)
        T, nxi = 30, 7
        B = np.random.default_rng(0).random((T, nxi)).astype(np.float32)
        B /= B.sum(1, keepdims=True)
        s = p3.TsvecSvgd3D(B, pl, K=3, dtype=torch.float64)
        P = pl.to_world(torch.rand(2, nxi, 2, dtype=torch.float64) * 1.2 - 0.1) \
            + 0.02 * torch.randn(2, nxi, 1, dtype=torch.float64) * torch.as_tensor(pl.n)
        phi = torch.rand(27, dtype=torch.float64) * 0.1
        _, J, _ = s.residual_and_J(P, phi)
        Pf, eps = P.reshape(2, -1), 1e-6
        for d in range(Pf.shape[1]):
            Pp, Pm = Pf.clone(), Pf.clone()
            Pp[:, d] += eps
            Pm[:, d] -= eps
            rp = s.residual_and_J(Pp.reshape(P.shape), phi)[0]
            rm = s.residual_and_J(Pm.reshape(P.shape), phi)[0]
            assert torch.allclose(J[:, :, d], (rp - rm) / (2 * eps), atol=1e-6)


def test_ergodic3d_coeffs_match_direct_basis():
    ee = p3.Ergodic3D(K=4)
    pts = torch.rand(50, 3)
    direct = torch.cos(np.pi * pts[:, None, :] * ee.k_idx[None]).prod(-1).mean(0)
    assert torch.allclose(ee.coeffs(pts), direct, atol=1e-5)


def test_tsp_exact_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(5):
        pts, start = rng.random((7, 3)), rng.random(3)
        cost = lambda o: (np.linalg.norm(pts[o[0]] - start)                      # noqa: E731
                          + sum(np.linalg.norm(pts[o[i]] - pts[o[i + 1]]) for i in range(len(o) - 1)))
        best = min(cost(list(o)) for o in itertools.permutations(range(7)))
        order = mp.tsp_open_path(start, pts)
        assert sorted(order) == list(range(7))
        assert abs(cost(order) - best) < 1e-9


def test_tsp_path_stays_on_plane_and_in_budget():
    pl = p3.Plane('tilted')
    mu = torch.zeros(mp.GP_RES, mp.GP_RES)
    mu[10, 50] = mu[40, 20] = mu[55, 55] = 1.0
    fn = mp.tsp_fn(pl, walk_len=1.0, seed=0, n_points_tsp=3)
    start = pl.to_world(torch.tensor([0.5, 0.5]))
    path = fn(0, start, mu, mu=mu, sd=None)
    assert float(pl.sdf(path).abs().max()) < 1e-5
    assert mp.path_length(path) <= 1.0 * pl.s + 1e-4


def test_tsp_two_opt_beats_or_matches_nearest_neighbour():
    rng = np.random.default_rng(3)
    pts, start = rng.random((40, 3)), rng.random(3)
    order = mp.tsp_open_path(start, pts)
    assert sorted(order) == list(range(40))
    nn, left, cur = [], set(range(40)), None
    while left:
        cur = min(left, key=lambda k: np.linalg.norm(pts[k] - (start if cur is None else pts[cur])))
        nn.append(cur)
        left.discard(cur)
    assert mp._tour_length(start, pts, order) <= mp._tour_length(start, pts, nn) + 1e-9


def test_tsp_fill_uses_the_budget():
    pl = p3.Plane('flat')
    torch.manual_seed(0)
    mu = torch.rand(mp.GP_RES, mp.GP_RES)
    start = pl.to_world(torch.tensor([0.5, 0.5]))
    plain = mp.tsp_fn(pl, walk_len=6.1, seed=0, n_points_tsp=12)(0, start, mu, mu=mu, sd=None)
    filled = mp.tsp_fn(pl, walk_len=6.1, seed=0, n_points_tsp=12, fill=True)(0, start, mu, mu=mu, sd=None)
    assert mp.path_length(plain) < 0.8 * 6.1 * pl.s          # 12 maxima alone do not fill it
    # the arc-length resampling cuts the polyline's corners a little (as for gpgoal / rw)
    assert 0.95 * 6.1 * pl.s < mp.path_length(filled) <= 6.1 * pl.s + 1e-4
    assert float(pl.sdf(filled).abs().max()) < 1e-5
