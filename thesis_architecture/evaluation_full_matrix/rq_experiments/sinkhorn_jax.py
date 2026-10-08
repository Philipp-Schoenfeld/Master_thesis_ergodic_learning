"""
sinkhorn_jax.py
================
A self-contained, log-domain-stabilized Sinkhorn divergence in plain JAX --
written specifically because `ott-jax` (the package Sun et al.'s own
`sinkhorn_flow_coverage.ipynb` depends on) does not import in this project's
environment:

    >>> import ott
    TypeError: register_dataclass() missing 2 required positional arguments
    ('data_fields', 'meta_fields')

`ott-jax==0.6.0` (the latest release on PyPI at the time of writing) calls
`jax.tree_util.register_dataclass` with the pre-0.4.28 two-positional-argument
signature; the installed `jax==0.4.35` requires the newer keyword-argument
form. No newer `ott-jax` release exists to fix this, and downgrading `jax`
project-wide would risk breaking `lqrax`/`ergodic_solver.py`, which the rest
of the Sun comparison (RQ1/RQ2/RQ5, the data generator itself) depends on.
Implementing the ~40 lines of entropic optimal transport directly avoids the
dependency entirely.

Matches Sun et al.'s own choices (`sinkhorn_flow_coverage.ipynb`): an L1
("cityblock") ground cost, uniform weights on both point sets, and the
Sinkhorn DIVERGENCE (removes the entropic bias of plain entropic OT):

    D_eps(p, q) = OT_eps(p, q) - 0.5 * OT_eps(p, p) - 0.5 * OT_eps(q, q)

`sinkhorn_gradient` differentiates the WHOLE unrolled iteration with
`jax.grad` (as the Sun notebook does through `ott`'s own autodiff path),
rather than an implicit-function/envelope-theorem shortcut -- simpler to get
right, and the problem sizes here (a few hundred trajectory points against a
few hundred target samples) make the extra autodiff cost negligible.
"""

import jax
import jax.numpy as jnp


def _pairwise_l1(x, y):
    """x: (N, d), y: (M, d) -> (N, M) L1 distances."""
    return jnp.sum(jnp.abs(x[:, None, :] - y[None, :, :]), axis=-1)


def sinkhorn_ot(x, y, epsilon, n_iters=50, w_x=None, w_y=None):
    """Entropic OT cost between point clouds `x` (N, d) and `y` (M, d),
    log-domain Sinkhorn iteration (numerically stable for small `epsilon`).
    `w_x`/`w_y`: optional (N,)/(M,) weights, default uniform. -> scalar."""
    n, m = x.shape[0], y.shape[0]
    log_a = jnp.full((n,), -jnp.log(n)) if w_x is None else jnp.log(w_x)
    log_b = jnp.full((m,), -jnp.log(m)) if w_y is None else jnp.log(w_y)
    C = _pairwise_l1(x, y)

    def body(carry, _):
        f, g = carry
        f = -epsilon * jax.scipy.special.logsumexp(
            log_b[None, :] + (g[None, :] - C) / epsilon, axis=1)
        g = -epsilon * jax.scipy.special.logsumexp(
            log_a[:, None] + (f[:, None] - C) / epsilon, axis=0)
        return (f, g), None

    (f, g), _ = jax.lax.scan(body, (jnp.zeros(n), jnp.zeros(m)), None, length=n_iters)
    return jnp.sum(jnp.exp(log_a) * f) + jnp.sum(jnp.exp(log_b) * g)


def sinkhorn_divergence(x, y, epsilon, n_iters=50):
    """D_eps(x, y) = OT_eps(x, y) - 0.5 OT_eps(x, x) - 0.5 OT_eps(y, y).
    x: (N, d), y: (M, d) -> scalar, >= 0, 0 iff the two empirical
    distributions coincide (unlike plain entropic OT, which has a strictly
    positive bias at x=y for epsilon>0)."""
    xy = sinkhorn_ot(x, y, epsilon, n_iters)
    xx = sinkhorn_ot(x, x, epsilon, n_iters)
    yy = sinkhorn_ot(y, y, epsilon, n_iters)
    return xy - 0.5 * xx - 0.5 * yy


def sinkhorn_gradient(x, y, epsilon, n_iters=50):
    """-d/dx of the Sinkhorn divergence, matching the convention of Sun's
    `sinkhorn_grad = jit(lambda _xs: -1.0 * grad(sinkhorn_div)(_xs))`: the
    DESCENT direction for the divergence, used as `h_tilde` in flow matching.
    x: (N, d) -> (N, d)."""
    return -jax.grad(lambda x_: sinkhorn_divergence(x_, y, epsilon, n_iters))(x)
