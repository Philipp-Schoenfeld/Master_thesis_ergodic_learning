r"""
test_teacher_study.py -- self-test of the teacher-study machinery (CPU, no checkpoint, ~1-2 min)
=================================================================================================
Run:  python test_teacher_study.py     (exit code 0 = all passed)

 1. `tsvec_svgd._diff_matrix`: order 2 == `_accel_matrix`; orders 1 / 3 == np.diff x (T-1)^(order-2).
 2. `TsvecSvgd(smooth_order=...)`: the analytic residual of `_residual_and_dX` equals the forward
    residual `_residual_fn`, and its Jacobian matches finite differences, for orders 1, 2 and 3.
 3. `AdamSvgd.run(start=...)` pins the first point exactly; start=None is unchanged.
 4. `sun_sinkhorn.run_batch(smoothness_weight=0)` is bit-identical to the old call; w > 0 changes the path.
 5. `project_bspline`: exact for a curve that is a B-spline already, pins x0, keeps B[0] = x0.
 6. Levenberg-Marquardt lowers the energy of every candidate and respects the start pin.
 7. `derivative_stats` on a circle: RMS acceleration and curvature in closed form.
 8. A diverged (NaN) candidate is flagged and does not crash the metrics.
 9. `analyze_teacher_study.select` picks the smoothest candidate within the Hellinger tolerance.
10. End to end: `run_teacher_study.py --grid smoke` (all solvers, forces and stages once) and
    `analyze_teacher_study.py` on its output both exit 0 and write the expected files.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, _arch, os.path.join(_arch, 'exploration')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline as pl                                                      # noqa: E402


def _toy_density(R=48):
    xs = np.linspace(0, 1, R)
    X, Y = np.meshgrid(xs, xs)
    d = np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / 0.02) + 0.5 * np.exp(-((X - 0.75) ** 2 + (Y - 0.3) ** 2) / 0.01)
    return (d / d.max()).astype(np.float32)


def _toy_shape():
    return pl.ShapeData('toy', _toy_density(), np.array([0.2, 0.2]),
                        np.vstack([[0.2, 0.2], pl.chord_inits([0.2, 0.2], 1)[0, 1:]]))


def test_diff_matrix():
    from common.tsvec_svgd import _accel_matrix, _diff_matrix
    T = 40
    np.testing.assert_array_equal(_diff_matrix(T, 2).numpy(), _accel_matrix(T).numpy())
    x = torch.randn(T, 2, dtype=torch.float64)
    for order in (1, 3):
        D = _diff_matrix(T, order, dtype=torch.float64)
        ref = np.diff(x.numpy(), n=order, axis=0) * float(T - 1) ** (order - 2)
        np.testing.assert_allclose((D @ x).numpy(), ref, atol=1e-10)
    print('[test] _diff_matrix orders 1/2/3: OK')


def test_tsvec_orders_jacobian():
    from common.tsvec_svgd import TsvecSvgd
    rng = np.random.default_rng(0)
    T = 32
    X = torch.as_tensor(rng.uniform(0.2, 0.8, size=(1, T, 2)), dtype=torch.float64)
    phi = torch.as_tensor(rng.normal(size=(36,)) * 0.1, dtype=torch.float64)
    for order in (1, 2, 3):
        s = TsvecSvgd(B=None, K=6, dtype=torch.float64, w_smooth=7.0, smooth_order=order)
        r, dr = s._residual_and_dX(X, phi)
        f = s._residual_fn(phi)
        np.testing.assert_allclose(r[0].numpy(), f(X[0].reshape(-1)).numpy(), atol=1e-10)
        eps = 1e-6
        J_fd = np.zeros((r.shape[1], T * 2))
        base = X[0].reshape(-1).clone()
        for i in range(T * 2):
            d = base.clone()
            d[i] += eps
            J_fd[:, i] = (f(d).numpy() - f(base).numpy()) / eps
        err = np.abs(dr[0].numpy() - J_fd).max() / max(np.abs(J_fd).max(), 1e-12)
        assert err < 1e-4, f'order {order}: relative Jacobian error {err}'
    print('[test] TsvecSvgd(smooth_order=1/2/3): residual == forward, Jacobian == finite differences, OK')


def test_adam_start_pin():
    from common.adam_svgd import AdamSvgd
    rng = np.random.default_rng(1)
    P0 = rng.uniform(0.3, 0.7, size=(3, 20, 2)).astype(np.float32)
    phi = torch.as_tensor(rng.normal(size=(36,)) * 0.1, dtype=torch.float32)
    s = AdamSvgd(B=None, K=6, w_smooth=15.0, beta=10.0, use_kernel=False)
    start = np.array([0.11, 0.22])
    out = s.run(P0, phi, 20, record=True, start=start)
    np.testing.assert_allclose(out['log'][:, :, 0].numpy(), np.broadcast_to(start, (21, 3, 2)), atol=1e-7)
    a = s.run(P0, phi, 5, record=False)['final']
    b = s.run(P0, phi, 5, record=False, start=None)['final']
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    print('[test] AdamSvgd start pin exact, start=None unchanged: OK')


def test_sinkhorn_smoothness_default():
    from common import sun_sinkhorn
    sd = _toy_shape()
    curves = pl.chord_inits(sd.x0, 2)
    kw = dict(record=False, log_space='raw')
    a = sun_sinkhorn.run_batch(curves, sd.density, 3, 128, **kw)['final_pos']
    b = sun_sinkhorn.run_batch(curves, sd.density, 3, 128, smoothness_weight=0.0, **kw)['final_pos']
    c = sun_sinkhorn.run_batch(curves, sd.density, 3, 128, smoothness_weight=3000.0, **kw)['final_pos']
    np.testing.assert_array_equal(a, b)
    assert np.abs(a - c).max() > 1e-7, 'smoothness force had no effect'
    print('[test] sun_sinkhorn smoothness_weight=0 bit-identical, w>0 acts: OK')


def test_project_bspline():
    rng = np.random.default_rng(2)
    P = rng.uniform(0.2, 0.8, size=(3, pl.NXI, 2))
    curves = pl.render(P)
    x0 = curves[0, 0]
    Q = pl.project_bspline(curves, pl.NXI, pl.DEGREE)
    np.testing.assert_allclose(Q, P, atol=1e-6)
    pinned = pl.project_bspline(curves, pl.NXI, pl.DEGREE, x0=np.array([0.5, 0.5]))
    np.testing.assert_allclose(pinned[:, 0], 0.5, atol=1e-12)
    np.testing.assert_allclose(pl.render(pinned)[:, 0], 0.5, atol=1e-9)
    assert np.abs(pl.basis()[0, 0] - 1.0) < 1e-9, 'basis is not clamped'
    print('[test] project_bspline: exact on B-splines, start pinned, clamped basis: OK')


def test_lm_decreases_energy():
    from common.tsvec_svgd import TsvecSvgd
    sd = _toy_shape()
    curves = pl.chord_inits(sd.x0, 4)
    e = TsvecSvgd(B=None, K=10, dtype=torch.float32, w_smooth=15.0, beta=1.0)
    phi = pl._phi_k(sd.density, e.k_idx, 'cpu')

    def energy(P):
        r, _ = e._residual_and_dX(e.render(torch.as_tensor(P, dtype=torch.float32)), phi)
        return 0.5 * (r ** 2).sum(-1).numpy()

    P0 = curves.astype(np.float32)
    P0[:, 0] = sd.x0
    P = pl._levenberg_marquardt(e, P0, phi, 40, start=sd.x0).numpy()
    assert (energy(P) < energy(P0)).all(), (energy(P0), energy(P))
    np.testing.assert_allclose(P[:, 0], np.broadcast_to(sd.x0, (4, 2)), atol=1e-6)
    print(f'[test] LM: energy {energy(P0).mean():.2f} -> {energy(P).mean():.2f}, start pinned: OK')


def test_derivative_stats_circle():
    T, r = 400, 0.3
    t = np.linspace(0, 1, T)
    X = np.stack([0.5 + r * np.cos(2 * np.pi * t), 0.5 + r * np.sin(2 * np.pi * t)], axis=1)[None]
    s = pl.derivative_stats(X)
    acc = r * (2 * np.pi) ** 2
    assert abs(s['acc_rms'][0] - acc) / acc < 0.01, (s['acc_rms'], acc)
    assert abs(s['curv_p95'][0] - 1 / r) / (1 / r) < 0.02, (s['curv_p95'], 1 / r)
    print('[test] derivative_stats on a circle (acceleration, curvature): OK')


def test_diverged_flag():
    sd = _toy_shape()
    cfg = pl.Cfg('t', 'lm', None, 15.0, 2, n_iters=2, projections=((25, 5),), polish_iters=())
    orig = pl.run_solver

    def fake(cfg, sd, curves, device):
        X = curves.copy()
        X[1] = np.nan
        return X
    pl.run_solver = fake
    try:
        stages, _ = pl.run_cell(cfg, sd, 3, 'cpu')
    finally:
        pl.run_solver = orig
    for m in stages.values():
        assert m['diverged'].tolist() == [0.0, 1.0, 0.0], m['diverged']
        assert all(np.isfinite(v).all() for v in m.values())
    print('[test] diverged candidate flagged, metrics stay finite: OK')


def test_selection_rule():
    import analyze_teacher_study as an
    d = dict(shape=np.array(['s'] * 5), config=np.array(['a', 'b', 'c', 'd', 'teacher_db']),
             stage=np.array(['bs25'] * 5), cand=np.zeros(5, int),
             hellinger=np.array([0.10, 0.14, 0.20, 0.12, 0.05]), smooth=np.array([5.0, 1.0, 0.1, 2.0, 0.01]),
             diverged=np.zeros(5))
    sel = an.select(d, ['bs25'], 0.05, 'smooth')
    j, hb = sel['s']
    assert d['config'][j] == 'b' and abs(hb - 0.10) < 1e-12, (d['config'][j], hb)    # c too far, teacher excluded
    d['diverged'][1] = 1.0
    assert d['config'][an.select(d, ['bs25'], 0.05, 'smooth')['s'][0]] == 'd'
    print('[test] selection rule: smoothest within tolerance, diverged and DB rows excluded: OK')


def test_end_to_end():
    with tempfile.TemporaryDirectory():
        tag = f'_selftest_{int(time.time())}'
        t = time.time()
        r = subprocess.run([sys.executable, os.path.join(_here, 'run_teacher_study.py'), '--grid', 'smoke',
                            '--shapes', 'A', '--n_init', '3', '--device', 'cpu', '--out_tag', tag],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        print(f'[test] run_teacher_study.py --grid smoke: {time.time() - t:.0f}s')
        run_dir = os.path.join(_here, 'results', tag)
        assert os.path.exists(os.path.join(run_dir, 'metrics.csv'))
        n_done = len(open(os.path.join(run_dir, 'done.txt')).read().split())
        assert n_done == 11, n_done                      # 10 smoke configs + the DB reference
        # resume: a second call must skip everything
        r2 = subprocess.run([sys.executable, os.path.join(_here, 'run_teacher_study.py'), '--grid', 'smoke',
                             '--shapes', 'A', '--n_init', '3', '--device', 'cpu', '--out_tag', tag],
                            capture_output=True, text=True)
        assert r2.returncode == 0 and 'rows' not in r2.stdout, r2.stdout[-2000:]
        r3 = subprocess.run([sys.executable, os.path.join(_here, 'analyze_teacher_study.py'), '--out_tag', tag],
                            capture_output=True, text=True)
        assert r3.returncode == 0, r3.stdout[-3000:] + r3.stderr[-3000:]
        for f in ('summary_config_stage.csv', 'selection.csv', 'report.md', 'plots/pareto.png',
                  'plots/force_sweeps.png', 'plots/projection_effect.png', 'plots/selection_gallery.png'):
            assert os.path.exists(os.path.join(run_dir, f)), f
        import shutil
        shutil.rmtree(run_dir)
    print('[test] end to end (smoke grid, resume, analysis, figures): OK')


if __name__ == '__main__':
    test_diff_matrix()
    test_tsvec_orders_jacobian()
    test_adam_start_pin()
    test_sinkhorn_smoothness_default()
    test_project_bspline()
    test_lm_decreases_energy()
    test_derivative_stats_circle()
    test_diverged_flag()
    test_selection_rule()
    test_end_to_end()
    print('ALL TESTS PASSED')
