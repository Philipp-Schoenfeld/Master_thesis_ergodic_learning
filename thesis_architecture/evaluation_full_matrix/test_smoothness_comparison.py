r"""
test_smoothness_comparison.py -- self-test of the smoothness-comparison pipeline
==================================================================================
CPU-only, no checkpoint needed for most of it (part 4 needs the GPU + the real
CFM checkpoint). ~1-3 minutes.   Run:  python test_smoothness_comparison.py
(exit code 0 = all passed)

 1. `_linear_resample_matrix` == `_resample`'s np.interp, exactly (both are
    the raw-waypoint logging's resampling step -- a matrix vs. the function it
    must reproduce bit-for-bit).
 2. `sun_refine.run_batch`: `log_space='raw'` runs and returns the right
    shapes; `smoothness_weight=0.0` (default) changes NOTHING about the
    'cps' path (same call with/without the new kwargs, bit-identical);
    `smoothness_weight>0` visibly reduces the raw-waypoint path's own
    acceleration cost relative to `smoothness_weight=0` on a deliberately
    jagged initial curve (sanity, not an exact number).
 3. `state_codec` round trip at n_states=1 (the `cfm_only` variant's case) and
    at nxi=128 (the raw-waypoint variants' point count, not just the usual 25).
 4. `smoothness_db.py` round trip (save_runs / iter_runs, with and without
    decoded states).
 5. end-to-end `run_smoothness_comparison.py --dry_run` (tiny: 1 shape, 2
    trajectories, a few iterations) + `plot_smoothness_comparison.py` on its
    output -- both must exit 0 and produce the expected files.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import subprocess
import sys
import tempfile
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

RUNNER = os.path.join(_here, 'run_smoothness_comparison.py')
PLOTTER = os.path.join(_here, 'plot_smoothness_comparison.py')


def run_cli(script, *args):
    t = time.time()
    r = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise AssertionError(f"{os.path.basename(script)} failed with exit code {r.returncode}")
    return r.stdout, time.time() - t


def test_resample_matrix():
    from common.sun_refine import _linear_resample_matrix, _resample
    rng = np.random.default_rng(0)
    curve = np.cumsum(rng.normal(size=(201, 2)), axis=0)
    for n_out in (25, 128, 201, 7):
        M = _linear_resample_matrix(n_out, curve.shape[0])
        via_matrix = M @ curve
        via_interp = _resample(curve, n_out)
        err = np.abs(via_matrix - via_interp).max()
        assert err < 1e-9, f"n_out={n_out}: max |diff|={err}"
    print("[test] _linear_resample_matrix == _resample: OK")


def _toy_phi(R=48):
    xs = np.linspace(0, 1, R)
    X, Y = np.meshgrid(xs, xs)
    phi = np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / 0.02) + 0.5 * np.exp(
        -((X - 0.75) ** 2 + (Y - 0.3) ** 2) / 0.01)
    return (phi / phi.max()).astype(np.float64)


def test_run_batch_smoothness_and_log_space():
    from common.sun_refine import run_batch
    phi = _toy_phi()
    T, C, n_iters = 64, 3, 5
    rng = np.random.default_rng(1)
    curves = np.stack([np.linspace([0.1, 0.1], [0.9, 0.9], T) + 0.05 * rng.normal(size=(T, 2))
                       for _ in range(C)])

    out_cps = run_batch(curves, phi, None, n_iters, 25, record=True)
    out_cps_explicit = run_batch(curves, phi, None, n_iters, 25, record=True,
                                 smoothness_weight=0.0, log_space='cps')
    assert out_cps['log'].shape == (C, n_iters, 25, 2)
    np.testing.assert_array_equal(out_cps['log'], out_cps_explicit['log'])
    np.testing.assert_array_equal(out_cps['final_cps'], out_cps_explicit['final_cps'])
    print("[test] smoothness_weight=0.0 / log_space='cps' defaults: bit-identical, OK")

    out_raw = run_batch(curves, phi, None, n_iters, T, record=True, log_space='raw')
    assert out_raw['log'].shape == (C, n_iters, T, 2)
    assert out_raw['init_cps'].shape == (C, T, 2)
    assert np.all(np.isfinite(out_raw['log']))
    print("[test] log_space='raw': correct shapes, finite values, OK")

    # jaggedness sanity: a much larger smoothness_weight should reduce the
    # raw path's own acceleration cost relative to weight 0, on average.
    def accel_cost(pos):
        a = pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]
        return float((a ** 2).sum(axis=(1, 2)).mean())

    out_plain = run_batch(curves, phi, None, 40, T, record=False, log_space='raw',
                          smoothness_weight=0.0)
    out_smooth = run_batch(curves, phi, None, 40, T, record=False, log_space='raw',
                           smoothness_weight=2000.0)
    c_plain, c_smooth = accel_cost(out_plain['final_pos']), accel_cost(out_smooth['final_pos'])
    assert c_smooth < c_plain, f"expected smoother final path, got {c_smooth} >= {c_plain}"
    print(f"[test] smoothness_weight=2000 reduces raw-path acceleration cost: "
         f"{c_plain:.4g} -> {c_smooth:.4g}, OK")


def test_state_codec_edge_cases():
    from state_codec import pack_states, unpack_states
    rng = np.random.default_rng(2)
    for n_states, nxi in ((1, 128), (601, 25), (2, 128)):
        a = rng.uniform(-0.2, 1.2, size=(n_states, nxi, 2)).astype(np.float32)
        b = pack_states(a)
        a2 = unpack_states(b, n_states, nxi)
        tol = (1.5 - (-0.5)) / 65535.0  # see state_codec.py
        assert np.abs(a - a2).max() < tol + 1e-5, f"n_states={n_states}, nxi={nxi}"
    print("[test] state_codec round trip (n_states=1, nxi=128): OK")


def test_smoothness_db_roundtrip():
    import smoothness_db as sdb
    from state_codec import pack_states
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        conn = sdb.open_db(os.path.join(d, 'x.db'))
        rows = []
        for c in range(4):
            n_states = 1 if c == 0 else 11
            states = rng.uniform(0, 1, size=(n_states, 128, 2)).astype(np.float32)
            rows.append(dict(
                cand_idx=c, init_param=None, n_iters=n_states - 1, nxi=128, n_points=128,
                log_space='raw', smoothness_weight=0.0 if c % 2 == 0 else 5.0,
                n_states=n_states, init_curve=states[0], states=pack_states(states),
                smooth_series=np.linspace(1, 0, n_states).astype(np.float32),
                path_len_series=np.linspace(2, 1, n_states).astype(np.float32)))
        sdb.save_runs(conn, 'toy_shape', 'linear_svgd_raw', rows)
        got = list(sdb.iter_runs(conn, shape='toy_shape', variant='linear_svgd_raw', with_states=True))
        assert len(got) == 4
        for r, g in zip(rows, got):
            assert g['cps'].shape == (r['n_states'], 128, 2)
            assert g['n_states'] == r['n_states']
        have = sdb.existing_cands(conn, 'toy_shape', 'linear_svgd_raw')
        assert have == {0, 1, 2, 3}
    print("[test] smoothness_db round trip: OK")


def test_end_to_end_dry_run():
    with tempfile.TemporaryDirectory():
        out_tag = f'_selftest_{int(time.time())}'
        out, dt = run_cli(RUNNER, '--dry_run', '--shapes', 'A', '--n_init', '2',
                          '--n_iters', '4', '--variants',
                          'cfm_only,cfm_svgd,linear_svgd_bspline,linear_svgd_raw,'
                          'linear_svgd_raw_smooth', '--out_tag', out_tag)
        print(f"[test] run_smoothness_comparison.py --dry_run: {dt:.1f}s")
        import smoothness_db as sdb
        db_path = os.path.join(_here, 'results', out_tag, 'smoothness_comparison.db')
        assert os.path.exists(db_path), out
        conn = sdb.open_db(db_path)
        n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert n == 5 * 2, f"expected 10 rows (5 variants x 2 cand), got {n}"
        out, dt = run_cli(PLOTTER, '--out_tag', out_tag)
        print(f"[test] plot_smoothness_comparison.py: {dt:.1f}s")
        plot_dir = os.path.join(_here, 'results', out_tag, 'plots')
        assert os.path.exists(os.path.join(plot_dir, 'aggregate.png'))
        assert os.path.exists(os.path.join(plot_dir, 'per_shape', 'A.png'))
        assert os.path.exists(os.path.join(_here, 'results', out_tag, 'summary.csv'))
        print("[test] end-to-end dry run + plots: OK")


if __name__ == '__main__':
    test_resample_matrix()
    test_run_batch_smoothness_and_log_space()
    test_state_codec_edge_cases()
    test_smoothness_db_roundtrip()
    test_end_to_end_dry_run()
    print("ALL TESTS PASSED")
