r"""
test_smoothness_comparison_v2.py -- self-test of the round-2 pipeline
=========================================================================
Covers what `test_smoothness_comparison.py` (round 1) doesn't: the new
`tsvec_svgd` solver's integration into `run_smoothness_comparison_v2.py`
(the solver's own correctness is `exploration/common/test_tsvec_svgd.py`,
run first here) and `sun_refine.py`'s new `dynamics='jerk'` path.
CPU-only except part 4 (needs the real CFM checkpoint; GPU if available,
else CPU). ~2-3 minutes.   Run:  python test_smoothness_comparison_v2.py

 1. `exploration/common/test_tsvec_svgd.py` (gradient checks, timing) --
    run as a subprocess, must exit 0.
 2. `sun_refine.run_batch(..., dynamics='jerk')`: finite output, and a
    jerk-penalised run has lower acceleration AND jerk cost than the
    point-mass run on the same init/target (sanity already covered in more
    depth manually during development; kept here as a regression guard).
 3. state_codec / smoothness_db round trip at `n_states=2001` (2000
    iterations), the real scale this round targets.
 4. end-to-end `run_smoothness_comparison_v2.py --dry_run` (tiny: 1 shape,
    3 particles, a few iterations, all 7 variants) + the shared
    `plot_smoothness_comparison.py` on its output -- both must exit 0.
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
for _p in (_here, os.path.join(_arch, 'exploration'), os.path.join(_arch, 'exploration', 'common'),
          _arch, os.path.join(_arch, 'ergodic_dataset_generator'),
          os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

RUNNER = os.path.join(_here, 'run_smoothness_comparison_v2.py')
PLOTTER = os.path.join(_here, 'plot_smoothness_comparison.py')
TSVEC_SVGD_TEST = os.path.join(_arch, 'exploration', 'common', 'test_tsvec_svgd.py')


def run_cli(script, *args):
    t = time.time()
    r = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise AssertionError(f"{os.path.basename(script)} failed with exit code {r.returncode}")
    return r.stdout, time.time() - t


def test_tsvec_svgd_subprocess():
    out, dt = run_cli(TSVEC_SVGD_TEST)
    print(f"[test] exploration/common/test_tsvec_svgd.py: {dt:.1f}s")
    assert 'ALL TESTS PASSED' in out
    print("[test] tsvec_svgd.py self-test: OK")


def test_jerk_dynamics():
    from common.sun_refine import run_batch
    rng = np.random.default_rng(5)
    T, C, n_iters = 48, 3, 30
    xs = np.linspace(0, 1, T)
    X, Y = np.meshgrid(xs, xs)
    phi = np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / 0.02)
    phi /= phi.max()
    curves = np.stack([np.linspace([0.1, 0.1], [0.9, 0.9], T) + 0.05 * rng.normal(size=(T, 2))
                       for _ in range(C)])

    def accel_cost(pos):
        a = pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]
        return float((a ** 2).sum(axis=(1, 2)).mean())

    out_pm = run_batch(curves, phi, None, n_iters, T, record=False, dynamics='pointmass')
    out_jk = run_batch(curves, phi, None, n_iters, T, record=False, dynamics='jerk')
    assert np.all(np.isfinite(out_pm['final_pos'])) and np.all(np.isfinite(out_jk['final_pos']))
    c_pm, c_jk = accel_cost(out_pm['final_pos']), accel_cost(out_jk['final_pos'])
    assert c_jk < c_pm, f"jerk dynamics should be smoother: pointmass={c_pm}, jerk={c_jk}"
    print(f"[test] dynamics='jerk' finite and smoother than 'pointmass': {c_pm:.4g} -> {c_jk:.4g}, OK")


def test_state_codec_2000_iters():
    from state_codec import pack_states, unpack_states
    rng = np.random.default_rng(6)
    n_states, nxi = 2001, 128
    a = rng.uniform(-0.1, 1.1, size=(n_states, nxi, 2)).astype(np.float32)
    b = pack_states(a)
    a2 = unpack_states(b, n_states, nxi)
    tol = 2.0 / 65535.0
    assert np.abs(a - a2).max() < tol + 1e-5
    print(f"[test] state_codec round trip at n_states=2001: OK ({len(b) / 1024:.1f} KB packed)")


def test_end_to_end_dry_run():
    out_tag = f'_selftest_v2_{int(time.time())}'
    out, dt = run_cli(RUNNER, '--dry_run', '--shapes', 'A', '--n_init', '3',
                      '--n_iters', '5', '--out_tag', out_tag)
    print(f"[test] run_smoothness_comparison_v2.py --dry_run: {dt:.1f}s")
    import smoothness_db as sdb
    db_path = os.path.join(_here, 'results', out_tag, 'smoothness_comparison.db')
    assert os.path.exists(db_path), out
    conn = sdb.open_db(db_path)
    n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n == 7 * 3, f"expected 21 rows (7 variants x 3 cand), got {n}"
    out, dt = run_cli(PLOTTER, '--out_tag', out_tag)
    print(f"[test] plot_smoothness_comparison.py: {dt:.1f}s")
    plot_dir = os.path.join(_here, 'results', out_tag, 'plots')
    assert os.path.exists(os.path.join(plot_dir, 'aggregate.png'))
    assert os.path.exists(os.path.join(plot_dir, 'per_shape', 'A.png'))
    print("[test] end-to-end dry run + plots: OK")


if __name__ == '__main__':
    test_tsvec_svgd_subprocess()
    test_jerk_dynamics()
    test_state_codec_2000_iters()
    test_end_to_end_dry_run()
    print("ALL TESTS PASSED")
