r"""
test_ergodicity_metrics.py -- the metrics themselves must be right before they judge any solver
===============================================================================================
  1. exact cases: all time in the left half of a left/right-symmetric target -> tv_2 = 0.5;
     visitation of a uniform grid of points vs. the uniform target -> tv ~ 0, hellinger small
  2. i.i.d. samples from the target (the best a T-sample time average can do) score far better
     than straight chords through the centre, for every metric and every target
  3. the spectral metric of i.i.d. samples falls with the number of samples (consistency)
  4. the time fraction of i.i.d. samples in the 75 % blob is ~0.75
Prints the finite-sample floor of every metric (used to set the solver pass thresholds).
Run: python test_ergodicity_metrics.py   (exit 0 = passed, CPU, < 1 min)
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, _arch, os.path.join(_arch, 'exploration'), os.path.join(_arch, 'evaluation_full_matrix')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import ergodicity_metrics as em
import targets as tg


def chords(n, T):
    ang = np.linspace(0, np.pi, n, endpoint=False)
    t = np.linspace(-0.45, 0.45, T)
    return np.stack([np.stack([0.5 + t * np.cos(a), 0.5 + t * np.sin(a)], -1) for a in ang])


def test_exact_cases():
    R = 96
    sym = tg.blobs_75_25(R)[:, ::-1] + tg.blobs_75_25(R)       # left/right symmetric
    X = np.stack([np.full(400, 0.25), np.linspace(0.1, 0.9, 400)], -1)
    m = em.evaluate(X, sym)
    assert abs(m['tv_2'][0] - 0.5) < 0.02, m['tv_2']
    g = (np.arange(64) + 0.5) / 64
    gx, gy = np.meshgrid(g, g)
    U = np.stack([gx.ravel(), gy.ravel()], -1)
    mu = em.evaluate(U, tg.uniform(R))
    assert mu['tv_8'][0] < 1e-9 and mu['hellinger'][0] < 0.05, (mu['tv_8'], mu['hellinger'])
    print(f"[metrics] exact cases: half-domain tv_2={m['tv_2'][0]:.3f} (0.5), "
          f"uniform grid tv_8={mu['tv_8'][0]:.1e}, hellinger={mu['hellinger'][0]:.3f}, OK")


def test_oracle_beats_chords():
    floors = {}
    for name, d in tg.load(list(tg.SYNTHETIC)):
        o = em.evaluate(em.oracle_samples(d, 30, 128, seed=1), d)
        c = em.evaluate(chords(30, 128), d)
        for k in ('hellinger', 'tv_4', 'tv_8', 'spec_rel_20', 'ergodic_k10'):
            assert np.median(o[k]) < np.median(c[k]), (name, k, np.median(o[k]), np.median(c[k]))
        assert np.median(o['coverage']) > np.median(c['coverage']) or name == 'uniform', name
        floors[name] = {k: float(np.median(o[k])) for k in o}
        print(f"[metrics] {name:12s} oracle (iid, T=128) vs chords: "
              + ', '.join(f"{k} {np.median(o[k]):.3f}/{np.median(c[k]):.3f}"
                          for k in ('hellinger', 'tv_4', 'low_excess', 'coverage', 'spec_rel_20', 'ergodic_k10')))
    return floors


def test_spectral_consistency():
    d = tg.sun_gmm3()
    vals = [np.median(em.spectral_rel(em.oracle_samples(d, 20, T, seed=2), d)) for T in (64, 256, 2048)]
    assert vals[0] > vals[1] > vals[2], vals
    print(f"[metrics] spectral error of iid samples falls with T (64/256/2048): "
          + ' > '.join(f'{v:.4f}' for v in vals) + ', OK')


def test_blob_fraction():
    d = tg.blobs_75_25()
    X = em.oracle_samples(d, 30, 512, seed=3)
    frac = (X[..., 0] < 0.51).mean()
    expected = tg.region_mass(d, lambda x, y: x < 0.51)
    assert abs(frac - expected) < 0.03 and abs(expected - 0.75) < 0.02, (frac, expected)
    print(f"[metrics] blob time fraction of iid samples {frac:.3f} vs target mass {expected:.3f}, OK")


if __name__ == '__main__':
    test_exact_cases()
    test_oracle_beats_chords()
    test_spectral_consistency()
    test_blob_fraction()
    print('ALL METRIC TESTS PASSED')
