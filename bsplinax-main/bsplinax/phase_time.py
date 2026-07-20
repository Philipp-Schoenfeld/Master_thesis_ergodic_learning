import abc

import jax
import jax.numpy as jnp


@jax.jit
def convert_r_to_t(r: jnp.ndarray) -> jnp.ndarray:
    """Trapezoidal integration of dt(s) = 1/r(s) * ds, with r(s)=ds/dt(s) and ds = 1/N.
    Produces t[0]=0 and cumulative times at each sample.
    """
    n = r.shape[0]
    if n == 1:
        return jnp.zeros_like(r)

    dt = 1.0 / r / n  # dt_k at each sample
    # Increments between consecutive samples using central (trapezoidal) rule
    increments = 0.5 * (dt[:-1] + dt[1:])  # length n-1
    t = jnp.concatenate([jnp.zeros_like(dt[:1]), jnp.cumsum(increments)], axis=-1)  # length n
    return t


class PhaseTime(abc.ABC):
    def __init__(self, total_duration: float = 5.0, num_time_points: int = 248, **kwargs):  # seconds
        self.total_duration = total_duration
        self.num_time_points = num_time_points

        # phase variable
        self.ss = jnp.linspace(0.0, 1.0, num_time_points)
        self.ds = 1.0 / num_time_points

        self.ratio = 1.0
        # Given r(s)=ds/dt(s), we want t_1 = T (trajectory duration)
        # But t_1 = \int_{0}^{1} dt/ds(s) ds = \int_{0}^{1} 1/r(s) ds = T_tmp
        # Adjust r(s) such that T_tmp = T
        rs_tmp = self.r_fn(self.ss)
        T_tmp = convert_r_to_t(rs_tmp)[-1]
        self.ratio = T_tmp / self.total_duration

        # r(s)
        self.r = self._r_fn(self.ss)
        # r(s)^-1 = dt/ds(s)
        self.r_inv = 1.0 / self.r

        # derivatives wrt s
        self.dr_ds = jax.vmap(jax.jacrev(self._r_fn))(self.ss)
        self.d2r_ds2 = jax.vmap(jax.jacrev(jax.jacrev(self._r_fn)))(self.ss)

        # time variable
        self.tt = convert_r_to_t(self.r)

    def _r_fn(self, s: jnp.ndarray) -> jnp.ndarray:
        """Implements r(s) = ds/dt as a function of s, and corrects it to match the desired duration."""
        return self.ratio * self.r_fn(s)

    @abc.abstractmethod
    def r_fn(self, s: jnp.ndarray) -> jnp.ndarray:
        """Implements r(s) = ds/dt as a function of s."""
        raise NotImplementedError

    def phi_s(self, s: jnp.ndarray) -> jnp.ndarray:
        r""" t = \phi(s)"""
        raise NotImplementedError

    def phi_inv_t(self, t: jnp.ndarray) -> jnp.ndarray:
        r"""S = \phi^{-1}(t)"""
        raise NotImplementedError


class PhaseTimeLinear(PhaseTime):
    r"""Linear phase-time mapping.
    Phase and time evolving linearly with t = phi(s) = s * T, with T being the total duration.
    """

    def r_fn(self, s: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(s) / self.total_duration

    def phi_s(self, s: jnp.ndarray) -> jnp.ndarray:
        return s * self.total_duration

    def phi_inv_t(self, t: jnp.ndarray) -> jnp.ndarray:
        return t / self.total_duration


class PhaseTimeParabola(PhaseTime):
    r"""Parabola phase-time mapping.
    Prevents the velocity and acceleration changing abruptly at the start and end of the trajectory.
    """

    def __init__(self, c1: float = -5.0, c2: float = 1.0, **kwargs):
        assert c1 < 0, "c1 must be less than 0"
        assert 0 < c2 <= 1.0, "c2 must be between 0 and 1"
        self.c1 = c1
        self.c2 = c2
        super().__init__(**kwargs)

    def r_fn(self, s: jnp.ndarray) -> jnp.ndarray:
        return self.c1 * s * (s - 1) + 1 / (self.c2 * self.total_duration)


if __name__ == "__main__":
    rs = jnp.ones(248) / 5.0
    t = convert_r_to_t(rs)
    phase_time_linear = PhaseTimeLinear(total_duration=5.0, num_time_points=1024)
    phase_time_parabola = PhaseTimeParabola(trajectory_duration=5.0, num_time_points=1024)
