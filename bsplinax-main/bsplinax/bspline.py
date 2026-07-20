import abc
from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "high")


@jax.jit
def _safe_div(num: jnp.ndarray, den: jnp.ndarray):
    den_safe = jnp.where(den == 0.0, 1.0, den)
    return (num / den_safe) * (den != 0.0)


@partial(jax.jit, static_argnames=("degree"))
def bspline_basis_at_s(s: float, knots: jnp.ndarray, degree: int):
    """Compute the B-spline basis function values at a given phase s.

    Args:
        s (float): The phase parameter.
        knots (jnp.ndarray): The knot vector.
        degree (int): The degree of the B-spline.
        is_clamped (bool): Whether the B-spline is clamped at the boundaries.

    Returns:
        jnp.ndarray: The B-spline basis function values at s.

    """
    num_knots = knots.shape[0]
    num_control_points = num_knots - degree - 1
    last_row_len = num_knots - 1

    B = jnp.zeros((degree + 1, last_row_len), dtype=knots.dtype)

    # degree-0
    u_left, u_right = knots[:-1], knots[1:]
    B0 = ((s >= u_left) & (s < u_right)).astype(knots.dtype)
    B = B.at[0].set(B0)

    idx = jnp.arange(last_row_len)

    def step_k(k, B_acc):
        L = last_row_len - k
        # safe gathers (no OOB even though we later mask tail)
        u_i = jnp.take(knots, idx, mode="clip")
        u_i_k = jnp.take(knots, idx + k, mode="clip")
        u_i1 = jnp.take(knots, idx + 1, mode="clip")
        u_i_k1 = jnp.take(knots, idx + k + 1, mode="clip")

        Bprev_i = jnp.take(B_acc[k - 1], idx, mode="clip")
        Bprev_ip = jnp.take(B_acc[k - 1], idx + 1, mode="clip")

        a = _safe_div((s - u_i) * Bprev_i, (u_i_k - u_i))
        b = _safe_div((u_i_k1 - s) * Bprev_ip, (u_i_k1 - u_i1))

        Bk_full = a + b
        Bk = jnp.where(idx < L, Bk_full, 0.0)  # mask invalid tail
        return B_acc.at[k].set(Bk)

    B = jax.lax.fori_loop(1, degree + 1, step_k, B)
    B_s = B[degree, :num_control_points]

    return B_s


class BsplineBasis(abc.ABC):
    r"""Abstract base class for B-spline basis function matrices (position, velocity, acceleration, jerk).
    We use an uniformly spaced knot vector u \in [0, 1], adapted if a clamped B-spline is used.
    Given a vector of control points w \in R^{n_b}, the B-spline basis matrices allow for computing paths
    and derivatives for a point in phase space s \in [0, 1].
        p(s) = \sum_{i=0}^{n_b-1} B_i(s) w_i
        dp/ds(s) = \sum_{i=0}^{n_b-1} dB_ds_i(s) w_i
        ...
    Note that in this form dB_ds is not a ``true" B-spline basis, because it does not satisfy the
    partition of unity property, since \sum_{i=0}^{n_b-1} dB_ds_i(s) != 0.
    """

    def __init__(
        self,
        degree: int = 5,
        num_control_points: int = 16,
        is_clamped: bool = False,
        num_phase_points: int = 1024,
        compute_derivatives: bool = True,
        **kwargs,
    ):
        self.degree = degree
        self.num_control_points = num_control_points
        self.is_clamped = is_clamped
        self.type_bspline = "clamped" if is_clamped else "standard"
        # u = [u_0, ..., u_m], num_knots_intervals := m
        self.num_knots_intervals = self.num_control_points + self.degree
        self.num_knots = self.num_knots_intervals + 1
        self.num_phase_points = num_phase_points

        # Build knots vector
        if is_clamped:
            # For setting boundary conditions, the first and last knot points should be repeated degree+1 times.
            self.knots = jnp.pad(
                jnp.linspace(0.0, 1.0, self.num_control_points - self.degree + 1),
                (self.degree, self.degree),
                mode="edge",
            )
        else:
            self.knots = jnp.linspace(0.0, 1.0, self.num_knots)

        # Evaluate B-spline bases and derivatives at all phase points
        self.ss = jnp.linspace(0.0, 1.0, self.num_phase_points)

        # Store basis and derivatives matrices evaluated at all phase points
        def f_aux(s: jnp.ndarray) -> jnp.ndarray:
            return bspline_basis_at_s(s, self.knots, self.degree)

        self.B = jax.vmap(f_aux)(self.ss)
        if compute_derivatives:
            self.dB_ds = jax.vmap(jax.jacrev(f_aux))(self.ss)
            self.d2B_ds2 = jax.vmap(jax.jacrev(jax.jacrev(f_aux)))(self.ss)
            self.d3B_ds3 = jax.vmap(jax.jacrev(jax.jacrev(jax.jacrev(f_aux))))(self.ss)
        else:
            self.dB_ds = None
            self.d2B_ds2 = None
            self.d3B_ds3 = None

        # There is a discontinuity at the last knot point (s=1), so derivatives from
        # automatic differentiation are not well defined. We set them manually.
        # https://github.com/pkicki/cnp-b/blob/dd96d2c2e036613f52d170e6a876e9771b593bff/utils/bspline.py
        # TODO - review this part of the code
        if is_clamped:
            M = self.num_knots_intervals - 2 * self.degree

            # B (last row)
            self.B = self.B.at[-1, -1].set(1.0)

            # dB_ds (last 2 cols): [ -d, +d ] * M
            if self.degree >= 1 and compute_derivatives:
                # TODO - check coefficients...
                coeff1 = M * self.degree
                self.dB_ds = self.dB_ds.at[-1, -2].set(-coeff1)
                self.dB_ds = self.dB_ds.at[-1, -1].set(+coeff1)

            # d2B_ds2 (last 3 cols): [ +1, -2, +1 ] * d(d-1) * M^2
            if self.degree >= 2 and compute_derivatives:
                # TODO - check coefficients...
                coeff2 = (M**2) * self.degree * (self.degree - 1)
                self.d2B_ds2 = self.d2B_ds2.at[-1, -3].set(1 / 2 * coeff2)
                self.d2B_ds2 = self.d2B_ds2.at[-1, -2].set(-3 / 2 * coeff2)
                self.d2B_ds2 = self.d2B_ds2.at[-1, -1].set(+coeff2)

            # d3B_ds3 (last 4 cols): [ -1, +3, -3, +1 ] * d(d-1)(d-2) * M^3
            if self.degree >= 3 and compute_derivatives:
                # TODO - is missing (self.degree - 1)? Check coefficients...
                coeff3 = (M**3) * self.degree * (self.degree - 2)
                self.d3B_ds3 = self.d3B_ds3.at[-1, -1].set(6 * coeff3)
                self.d3B_ds3 = self.d3B_ds3.at[-1, -2].set(-10.5 * coeff3)
                self.d3B_ds3 = self.d3B_ds3.at[-1, -3].set(5.5 * coeff3)
                self.d3B_ds3 = self.d3B_ds3.at[-1, -4].set(-1.0 * coeff3)


class BsplineBasisClamped(BsplineBasis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is_clamped=True)


if __name__ == "__main__":
    bspline = BsplineBasis(degree=5, num_control_points=12, num_phase_points=1024)
    bspline = BsplineBasisClamped(degree=3, num_control_points=8, num_phase_points=248)
