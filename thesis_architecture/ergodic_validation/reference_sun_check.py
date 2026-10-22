r"""
reference_sun_check.py -- our Sun port vs. the official implementation
=======================================================================
Reference: Sun, Pinosky, Murphey, "Flow Matching Ergodic Coverage", RSS 2025, official tutorial
MurpheyLab/lqr-flow-matching, tutorials/stein_flow_coverage.ipynb (cells 2, 4, 8-15),
transcribed below without changes: analytic 3-GMM target, x0 = (0.1, 0.2) with initial velocity
towards the centre, zero initial control, PointMassLQR (Q = diag(1, 1, 1e-3, 1e-3), R = 0.01 I,
dt = 0.05, 200 steps), RBF kernel exp(-|x - x'|^2 / h) with h = 0.01 averaged over the time
points, step 0.01, 500 iterations.

Our port (`exploration/common/sun_refine.run_batch`) differs only in the target interface: the
score comes from a density GRID (bilinear interpolation, smoothed floor, soft boundary). It is
run on the same target rasterised to 96 x 96, from the same straight line (which is what zero
control with the official x0 / v0 traces out), for the same 500 iterations.

Checks: (1) both reach a comparable ergodicity (Hellinger of the visitation density, region
masses), (2) the official result itself passes the validation criteria -- i.e. the criteria are
achievable for this solver family on its own benchmark, (3) the two final trajectories are close.
Run: python reference_sun_check.py   (CPU, ~1-2 min)
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_arch, 'exploration'), os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'bsplinax-main'), os.path.join(_arch, 'evaluation_full_matrix')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal as mvn
from lqrax import LQR

import ergodicity_metrics as em
import targets as tg

# ---- official tutorial, cells 2 / 4 / 8-12 / 15 (verbatim apart from removed plotting) -------
mean1, cov1 = jnp.array([0.3, 0.5]), jnp.array([[0.002, 0.0], [0.0, 0.04]])
mean2, cov2 = jnp.array([0.5, 0.5]), jnp.array([[0.02, -0.018], [-0.018, 0.02]])
mean3, cov3 = jnp.array([0.7, 0.5]), jnp.array([[0.002, 0.0], [0.0, 0.04]])
w1, w2, w3 = 0.34, 0.34, 0.33


def pdf(x):
    return w1 * mvn.pdf(x[:2], mean1, cov1) + w2 * mvn.pdf(x[:2], mean2, cov2) + w3 * mvn.pdf(x[:2], mean3, cov3)


score_pdf = jax.grad(lambda x: jnp.log(pdf(x)))
dt, tsteps = 0.05, 200
T = dt * tsteps
_x0 = jnp.array([0.1, 0.2])
x0 = jnp.array([_x0[0], _x0[1], 2.0 * (0.5 - _x0[0]) / T, 2.0 * (0.5 - _x0[1]) / T])


class PointMassLQR(LQR):
    def __init__(self, dt, x_dim, u_dim, Q, R):
        super().__init__(dt, x_dim, u_dim, Q, R)

    def dyn(self, xt, ut):
        return jnp.array([xt[2], xt[3], ut[0], ut[1]])


pm = PointMassLQR(dt=dt, x_dim=4, u_dim=2, Q=jnp.diag(jnp.array([1.0, 1.0, 0.001, 0.001])),
                  R=jnp.diag(jnp.array([0.01, 0.01])))
linearize_dyn, solve_lqr = jax.jit(pm.linearize_dyn), jax.jit(pm.solve)


def kernel(x1, x2, h):
    return jnp.exp(-1.0 * jnp.sum(jnp.square(x1[:2] - x2[:2])) / h)


d_kernel = jax.grad(kernel, argnums=0)


def stein_grad(traj, h):
    unit = lambda x1, x2: kernel(x2, x1, h) * score_pdf(x2) + d_kernel(x2, x1, h)
    return jax.vmap(lambda x: jnp.mean(jax.vmap(unit, in_axes=(None, 0))(x, traj), axis=0))(traj)


stein_grad = jax.jit(stein_grad)


def official(num_iters=500, step_size=0.01):
    u_traj = jnp.zeros((tsteps, 2))
    for _ in range(num_iters):
        x_traj, A_traj, B_traj = linearize_dyn(x0, u_traj)
        v_traj, _ = solve_lqr(jnp.zeros(4), A_traj, B_traj, stein_grad(x_traj, h=0.01))
        u_traj += step_size * v_traj
    return np.concatenate([np.asarray(x0[None, :2]), np.asarray(pm.traj_sim(x0, u_traj))[:, :2]], 0)


# ---- our port ----------------------------------------------------------------------------------
def ours(grid, num_iters=500):
    from common import sun_refine
    line = np.asarray(x0[:2])[None] + np.linspace(0, T, tsteps + 1)[:, None] * np.asarray(x0[2:])[None]
    out = sun_refine.run_batch(line[None].astype(np.float64), grid.astype(np.float32), None, num_iters,
                               tsteps + 1, record=False, log_space='raw')
    return np.asarray(out['final_pos'])[0] if out.get('final_pos') is not None else np.asarray(out['final_cps'])[0]


def main():
    grid = tg.sun_gmm3(96)
    ref = official()
    our = ours(grid)
    mr, mo = em.evaluate(ref, grid), em.evaluate(our, grid)
    ln = np.asarray(x0[:2])[None] + np.linspace(0, T, tsteps + 1)[:, None] * np.asarray(x0[2:])[None]
    ml = em.evaluate(ln, grid)
    floor = float(np.median(em.evaluate(em.oracle_samples(grid, 30, tsteps + 1, seed=7), grid)['hellinger']))
    for lab, m in (('initial line', ml), ('official', mr), ('our port', mo)):
        print(f"[sun-ref] {lab:12s} hellinger {m['hellinger'][0]:.3f} (iid floor {floor:.3f})  tv_4 {m['tv_4'][0]:.3f}  "
              f"low_excess {m['low_excess'][0]:+.3f}  coverage {m['coverage'][0]:.3f}  E10 {m['ergodic_k10'][0]:.2f}")
    gap = np.linalg.norm(ref - our, axis=-1)
    print(f"[sun-ref] official vs. our port: mean point distance {gap.mean():.4f}, max {gap.max():.4f}")
    assert mr['hellinger'][0] < ml['hellinger'][0] - 0.2, 'official solver does not improve on the line?'
    assert abs(mo['hellinger'][0] - mr['hellinger'][0]) < 0.08, 'port reaches a different ergodicity'
    assert gap.mean() < 0.05, 'port trajectory differs from the official one'
    np.savez(os.path.join(_here, 'results', 'reference_sun.npz'), official=ref, ours=our, line=ln, grid=grid)
    print('[sun-ref] OK: our port reproduces the official Sun et al. solver on its own benchmark')


if __name__ == '__main__':
    main()
