"""
alm_solver.py
==============
RQ6's missing building block: an Augmented Lagrangian solver over B-spline
control points with EXPLICIT Lagrange multipliers, which nothing in this
repository has (see the thesis-doc conversation this module was built from
for the full gap analysis). Concretely, this is the extension TSVEC's own
paper (Li et al.) names as future work ("the current framework can be
readily extended to an Augmented Lagrangian constrained optimization
approach; however, for simplicity, we leave this for future work") and the
kind of solver Flow-Opt (Idoko et al.) predicts a warm start lambda_0 FOR --
neither paper's own repository/released code contains one, so this is new,
not a port.

Problem (single robot, control points xi in R^(nxi x 2), B = B-spline basis
so the curve is B @ xi):

    minimize_xi   f(xi)                         ergodic coverage + smoothness
    subject to    xi[0] = start                  equality   (start pin)
                  0 <= xi <= 1                    affine inequality (workspace)
                  ||xi_i - c|| >= r  for all i    quadratic inequality (obstacle)

This mirrors Flow-Opt's own constraint shape (Eq. 11a-11d in that paper: an
affine equality `A xi = b`, an affine inequality `G xi <= h`, and a quadratic
inequality `g(xi) <= 0`) applied to a single-robot ergodic-coverage
trajectory instead of Flow-Opt's multi-robot setting, and is deliberately
NOT a copy of `obstacles.py`'s existing obstacle handling: that module
enforces the same geometric fact (stay outside a circle) as a smooth PENALTY
added to the cost, with no multiplier and no exact satisfaction guarantee.
Here it is a genuine inequality constraint with its own dual variable mu,
updated by the standard Augmented Lagrangian rule -- the actual mechanism
TSVEC defers and Flow-Opt's lambda_0 initialises.

What this module deliberately does NOT do: train a network to predict
(xi_0, lambda_0, mu_0) the way Flow-Opt does (a second transformer,
self-supervised on the fixed-point residual of ITS OWN solver, Eq. 12/14 in
that paper). That is a separate, substantially larger piece of work -- a new
network architecture, a new self-supervised training loop, and a GPU training
budget -- and is out of scope here. `rq6_alm_lambda0.py` instead tests the
narrower, still-meaningful question this solver makes answerable for the
first time in this codebase: does warm-starting (lambda, mu) at all help
convergence, using the cheapest available informed source (the previous
replanning round's converged multipliers) as a stand-in for a trained
predictor. See that script's docstring for why this is an honest scoping of
the question, not an attempt to claim a trained lambda_0 predictor was built.
"""

import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
_arch = os.path.dirname(_mat)
for _p in (_here, _mat, os.path.join(_arch, 'ergodic_dataset_generator')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import jax                                      # noqa: E402
import jax.numpy as jnp                         # noqa: E402

NXI = 25
N_EVAL = 128          # points the curve is evaluated at for cost/constraints
#: Same weights as `SE3_SVGD/tsvec_2d.py` (W_ERGODIC/W_SMOOTH), so the
#: unconstrained part of the objective is not a new, untested formulation --
#: only the constraint HANDLING (penalty vs. ALM) differs from that solver.
W_ERGODIC = 600.0
W_SMOOTH = 15.0
K_FREQ = 8


def _fourier_coeffs(curve, k_idx):
    """curve: (T, 2) -> (K*K,) time-averaged 2D Fourier coefficients
    (cos(pi k1 x) cos(pi k2 y)), same convention as `exp_common.target_coeffs`
    / `ergodic_core.py`."""
    F = (jnp.cos(jnp.pi * k_idx[None, :, 0] * curve[:, 0:1])
        * jnp.cos(jnp.pi * k_idx[None, :, 1] * curve[:, 1:2]))
    return F.mean(axis=0)


def _smoothness(curve):
    """Sum of squared second differences (discrete acceleration), the same
    quantity `SE3_SVGD/svgd_engine.py::compute_smoothness_cost_numpy` scores."""
    accel = curve[2:] - 2 * curve[1:-1] + curve[:-2]
    return jnp.sum(accel ** 2)


class AlmBSplineSolver:
    """Augmented Lagrangian solver for one ergodic-coverage trajectory.

    `basis`: (N_EVAL, nxi) clamped B-spline basis (see `obstacles.bspline_basis_matrix`).
    `phi_k_target`: (K*K,) target Fourier coefficients (`exp_common.target_coeffs`).
    `obstacle_center`/`obstacle_radius`: the quadratic inequality constraint;
    `obstacle_radius=0` disables it (only the equality + box inequality remain).
    """

    def __init__(self, basis, phi_k_target, k_idx, start, obstacle_center=(0.5, 0.5),
                obstacle_radius=0.08, rho0=10.0, rho_growth=1.3, rho_max=1e5,
                inner_lr=2e-3, inner_steps=50):
        self.B = jnp.asarray(basis, dtype=jnp.float32)
        self.nxi = self.B.shape[1]
        self.phi_k_target = jnp.asarray(phi_k_target, dtype=jnp.float32)
        self.k_idx = jnp.asarray(k_idx, dtype=jnp.float32)
        self.start = jnp.asarray(start, dtype=jnp.float32)
        self.c = jnp.asarray(obstacle_center, dtype=jnp.float32)
        self.r = float(obstacle_radius)
        self.rho0, self.rho_growth, self.rho_max = rho0, rho_growth, rho_max
        self.inner_lr, self.inner_steps = inner_lr, inner_steps
        self._build()

    # -- the three constraint blocks -----------------------------------------
    def _h_eq(self, xi):
        """Equality residual (nxi,2) -> (2,): only the first control point is
        constrained (A xi = b with A = selection of row 0, b = start)."""
        return xi[0] - self.start

    def _g_box(self, xi):
        """Affine inequality g<=0 for the workspace box 0<=xi<=1, flattened
        to (nxi*2*2,): [xi - 1 (upper), -xi (lower)]."""
        return jnp.concatenate([(xi - 1.0).reshape(-1), (-xi).reshape(-1)])

    def _g_obstacle(self, xi):
        """Quadratic inequality g<=0 per control point: r^2 - ||xi_i-c||^2 <= 0
        (i.e. stay at distance >= r from the obstacle centre). Returns
        shape (nxi,); an all-zero array (no multiplier mass) if disabled."""
        if self.r <= 0:
            return jnp.zeros((self.nxi,))
        d2 = jnp.sum((xi - self.c[None, :]) ** 2, axis=-1)
        return self.r ** 2 - d2

    def _objective(self, xi):
        curve = self.B @ xi
        c_k = _fourier_coeffs(curve, self.k_idx)
        erg = 0.5 * jnp.sum((c_k - self.phi_k_target) ** 2)
        return W_ERGODIC * erg + W_SMOOTH * _smoothness(curve)

    def _build(self):
        def lagrangian(xi, lam, mu, rho):
            f = self._objective(xi)
            h = self._h_eq(xi)
            g = jnp.concatenate([self._g_box(xi), self._g_obstacle(xi)])
            eq_term = jnp.dot(lam, h) + 0.5 * rho * jnp.sum(h ** 2)
            ineq_active = jnp.maximum(0.0, mu + rho * g)
            ineq_term = (jnp.sum(ineq_active ** 2) - jnp.sum(mu ** 2)) / (2 * rho)
            return f + eq_term + ineq_term

        self._lagrangian = lagrangian
        self._grad_xi = jax.jit(jax.grad(lagrangian, argnums=0))

        # Adam, not plain gradient descent: the ergodic term alone is
        # W_ERGODIC=600 times a Fourier-coefficient mismatch, the same scale
        # `svgd_batched.py`/`sun_refine.py` optimise with Adam at `ADAM_LR=2e-3`
        # (not a fixed-step GD, which diverges to NaN within a handful of
        # outer iterations at this objective scale -- verified while building
        # this solver). Same Adam hyperparameters, so the ALM's inner solve
        # is not a new, untuned optimiser choice.
        b1, b2, adam_eps = 0.9, 0.999, 1e-8

        def inner_minimise(xi, lam, mu, rho):
            def body(carry, t):
                xi_, m, v = carry
                g = self._grad_xi(xi_, lam, mu, rho)
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                m_hat = m / (1 - b1 ** (t + 1))
                v_hat = v / (1 - b2 ** (t + 1))
                xi_new = xi_ - self.inner_lr * m_hat / (jnp.sqrt(v_hat) + adam_eps)
                return (xi_new, m, v), None
            (xi_out, _, _), _ = jax.lax.scan(
                body, (xi, jnp.zeros_like(xi), jnp.zeros_like(xi)),
                jnp.arange(self.inner_steps))
            return xi_out

        self._inner_minimise = jax.jit(inner_minimise)

        def outer_step(xi, lam, mu, rho):
            xi = self._inner_minimise(xi, lam, mu, rho)
            h = self._h_eq(xi)
            g = jnp.concatenate([self._g_box(xi), self._g_obstacle(xi)])
            lam = lam + rho * h
            mu = jnp.maximum(0.0, mu + rho * g)
            rho = jnp.minimum(rho * self.rho_growth, self.rho_max)
            return xi, lam, mu, rho

        self._outer_step = jax.jit(outer_step)

    # -- public API -----------------------------------------------------------
    def n_constraints(self):
        # box: nxi control points x 2 coords x {upper, lower} = 4*nxi; obstacle: nxi.
        return 4 * self.nxi + self.nxi

    def init_multipliers(self):
        return jnp.zeros(2), jnp.zeros(self.n_constraints())

    def violation(self, xi):
        """Max constraint violation (0 = fully feasible): max(|h|, relu(g))."""
        h = self._h_eq(xi)
        g = jnp.concatenate([self._g_box(xi), self._g_obstacle(xi)])
        return float(jnp.maximum(jnp.abs(h).max(), jnp.maximum(g, 0.0).max()))

    def objective(self, xi):
        return float(self._objective(xi))

    def solve(self, xi0, lam0=None, mu0=None, n_outer=30, tol=1e-3, record=False):
        """Run up to `n_outer` ALM outer iterations from `xi0` (and, if given,
        `lam0`/`mu0` -- the "warm-started multipliers" RQ6 is about), stopping
        early once `violation(xi) <= tol`.
        -> dict(xi, lam, mu, outer_iters (iterations actually run), feasible
        (bool), log (list of (outer_iter, objective, violation)) if `record`)."""
        xi = jnp.asarray(xi0, dtype=jnp.float32)
        lam, mu = self.init_multipliers() if lam0 is None else (
            jnp.asarray(lam0, dtype=jnp.float32), jnp.asarray(mu0, dtype=jnp.float32))
        rho = self.rho0
        log = []
        for it in range(n_outer):
            xi, lam, mu, rho = self._outer_step(xi, lam, mu, rho)
            v = self.violation(xi)
            if record:
                log.append((it + 1, self.objective(xi), v))
            if v <= tol:
                return dict(xi=np.asarray(xi), lam=np.asarray(lam), mu=np.asarray(mu),
                           outer_iters=it + 1, feasible=True, log=log)
        return dict(xi=np.asarray(xi), lam=np.asarray(lam), mu=np.asarray(mu),
                   outer_iters=n_outer, feasible=False, log=log)
