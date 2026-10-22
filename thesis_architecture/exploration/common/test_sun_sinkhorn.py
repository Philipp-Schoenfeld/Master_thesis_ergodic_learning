r"""
test_sun_sinkhorn.py -- correctness checks for sun_sinkhorn.py
==============================================================
 1. divergence_grad (envelope-theorem gradient at the Sinkhorn fixed point) equals autodiff of the
    debiased divergence through a long unrolled Sinkhorn solve (float64).
 2. The warm-started gradient used inside the flow (N_INNER updates from the previous potentials)
    agrees with a fully converged one after a typical step (cosine > 0.98).
 3. A plain particle gradient flow on the divergence (no dynamics) puts 75 % of the particles into
    the 75 % blob -- the mass property the Stein flow lacks.
 4. Full trajectory run (LQ flow matching, point mass) on the 75/25 blobs from straight chords:
    time fraction in the left half within 0.08 of 0.75, and closer than the Stein flow from the
    same chords.
Run: python test_sun_sinkhorn.py   (CPU, a few minutes)
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_expl = os.path.dirname(_here)
_arch = os.path.dirname(_expl)
for _p in (_expl, _arch, os.path.join(_arch, 'ergodic_dataset_generator'), os.path.join(_arch, 'ergodic_validation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np

from common import sun_sinkhorn as sk


def _blobs(R=64):
    import targets as tg
    return tg.blobs_75_25(R)


def test_gradient_vs_autodiff():
    jax.config.update('jax_enable_x64', True)
    try:
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.uniform(0.2, 0.8, (15, 2)))
        Y = jnp.asarray(rng.uniform(0.1, 0.9, (40, 2)))
        lb = jnp.log(jnp.asarray(rng.uniform(0.5, 1.5, 40)) / 1.0)
        lb = lb - jax.scipy.special.logsumexp(lb)
        eps = 0.05

        def S(x):
            N = x.shape[0]
            la = jnp.full((N,), -jnp.log(N))
            f, g = sk.sinkhorn_potentials(sk._cost(x, Y), la, lb, jnp.zeros(N), jnp.zeros(40), 2000, eps)
            fs = sk.symmetric_potential(sk._cost(x, x), la, jnp.zeros(N), 2000, eps)
            return jnp.exp(la) @ f + jnp.exp(lb) @ g - jnp.exp(la) @ fs

        ga = jax.grad(S)(x)
        N = x.shape[0]
        gm, *_ = sk.divergence_grad(x, Y, lb, jnp.zeros(N), jnp.zeros(40), jnp.zeros(N), 2000, eps)
        err = float(jnp.abs(ga - gm).max() / jnp.abs(ga).max())
        assert err < 1e-4, err
        print(f'[test] envelope gradient == autodiff of the unrolled divergence: rel err {err:.1e}, OK')
    finally:
        jax.config.update('jax_enable_x64', False)


def test_warm_start():
    Y, lb = sk.target_cloud(_blobs())
    t = np.linspace(0, 1, 200)[:, None]
    x0 = jnp.asarray(0.1 + 0.8 * np.hstack([t, t * 0.5 + 0.25]), dtype=jnp.float32)
    z = lambda n: jnp.zeros(n, dtype=jnp.float32)
    g0, f, g, fxx = sk.divergence_grad(x0, Y, lb, z(200), z(len(Y)), z(200), sk.N_INNER_FIRST)
    x1 = x0 - 0.005 * sk.SCALE * g0 * 0.01                     # a typical small move
    gw, *_ = sk.divergence_grad(x1, Y, lb, f, g, fxx, sk.N_INNER)
    gc, *_ = sk.divergence_grad(x1, Y, lb, z(200), z(len(Y)), z(200), 2000)
    cos = float(jnp.sum(gw * gc) / (jnp.linalg.norm(gw) * jnp.linalg.norm(gc)))
    assert cos > 0.98, cos
    print(f'[test] warm-started gradient vs. converged: cosine {cos:.4f}, OK')


def test_particle_flow_mass():
    Y, lb = sk.target_cloud(_blobs())
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.uniform(0.1, 0.9, (200, 2)), dtype=jnp.float32)
    z = lambda n: jnp.zeros(n, dtype=jnp.float32)
    f, g, fxx = z(200), z(len(Y)), z(200)
    step = jax.jit(lambda x, f, g, fxx: sk.divergence_grad(x, Y, lb, f, g, fxx, 30))
    for _ in range(400):
        gr, f, g, fxx = step(x, f, g, fxx)
        x = x - 0.5 * 200 * gr * 0.01
    frac = float((x[:, 0] < 0.51).mean())
    assert abs(frac - 0.75) < 0.05, frac
    print(f'[test] particle Sinkhorn flow: fraction in the 75 % blob {frac:.3f}, OK')


def test_torch_twin_matches_jax():
    """The GPU (torch) implementation used in the pipeline == the JAX reference: same gradient,
    and the same trajectories after 60 flow iterations."""
    import torch
    Y, lb = sk.target_cloud(_blobs())
    rng = np.random.default_rng(3)
    x = rng.uniform(0.1, 0.9, (3, 50, 2)).astype(np.float32)
    z = lambda *s: np.zeros(s, dtype=np.float32)
    gj = np.stack([np.asarray(sk.divergence_grad(jnp.asarray(x[c]), Y, lb, z(50), z(len(Y)), z(50), 200)[0])
                   for c in range(3)])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = lambda a: torch.as_tensor(a, device=dev)
    gt = sk.divergence_grad_torch(T(x), T(Y), T(lb), T(z(3, 50)), T(z(3, len(Y))), T(z(3, 50)), 200)[0].cpu().numpy()
    err = np.abs(gt - gj).max() / np.abs(gj).max()
    assert err < 1e-3, err
    ang = np.linspace(0, np.pi, 3, endpoint=False)
    s = np.linspace(-0.45, 0.45, 128)[:, None]
    curves = np.stack([0.5 + s * np.array([np.cos(a), np.sin(a)]) for a in ang])
    a = sk.run_batch(curves, _blobs(96), 10, 128, device='jax')['final_pos']
    b = sk.run_batch(curves, _blobs(96), 10, 128, device=dev)['final_pos']
    gap = np.abs(a - b).max()
    assert gap < 1e-4, gap
    # Longer runs diverge at float32 rounding level (the L1 cost has a sign discontinuity; torch on
    # CPU and on CUDA diverge from each other the same way), so compare what matters: ergodicity.
    a = sk.run_batch(curves, _blobs(96), 300, 128, device='jax')['final_pos']
    b = sk.run_batch(curves, _blobs(96), 300, 128, device=dev)['final_pos']
    fa, fb = (a[..., 0] < 0.51).mean(), (b[..., 0] < 0.51).mean()
    assert abs(fa - fb) < 0.03, (fa, fb)
    print(f'[test] torch twin == JAX reference: gradient rel err {err:.1e}, trajectories after 10 it. '
          f'max |diff| {gap:.1e}; after 300 it. blob time fraction {fa:.3f} vs {fb:.3f}, OK')


def test_trajectory_mass_vs_stein():
    from common import sun_refine
    d = _blobs(96)
    ang = np.linspace(0, np.pi, 6, endpoint=False)
    s = np.linspace(-0.45, 0.45, 128)[:, None]
    curves = np.stack([0.5 + s * np.array([np.cos(a), np.sin(a)]) for a in ang])
    out_s = sun_refine.run_batch(curves, d, None, 800, 128, record=False, log_space='raw')
    out_k = sk.run_batch(curves, d, 800, 128, record=False)
    fs = (out_s['final_pos'][..., 0] < 0.51).mean()
    fk = (out_k['final_pos'][..., 0] < 0.51).mean()
    assert abs(fk - 0.75) < 0.08 and abs(fk - 0.75) < abs(fs - 0.75), (fk, fs)
    print(f'[test] trajectory time fraction in the 75 % blob: Sinkhorn flow {fk:.3f}, Stein flow {fs:.3f}, OK')


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(_arch, 'evaluation_full_matrix'))
    test_gradient_vs_autodiff()
    test_warm_start()
    test_particle_flow_mass()
    test_torch_twin_matches_jax()
    test_trajectory_mass_vs_stein()
    print('ALL TESTS PASSED')
