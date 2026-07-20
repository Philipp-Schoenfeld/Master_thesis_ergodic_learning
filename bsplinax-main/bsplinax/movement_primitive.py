import jax
import jax.numpy as jnp
from flax import struct

from bsplinax.bspline import BsplineBasisClamped
from bsplinax.phase_time import PhaseTime, PhaseTimeLinear


@struct.dataclass
class Trajectory:
    q: jnp.ndarray | None = None
    dq: jnp.ndarray | None = None
    ddq: jnp.ndarray | None = None
    dddq: jnp.ndarray | None = None

    def __getitem__(self, idx) -> "Trajectory":
        def maybe_index(x):
            return None if x is None else x[idx]

        return Trajectory(
            q=maybe_index(self.q),
            dq=maybe_index(self.dq),
            ddq=maybe_index(self.ddq),
            dddq=maybe_index(self.dddq),
        )


class BsplineMP:
    """B-spline movement primitive."""

    def __init__(
        self,
        control_points: jnp.ndarray = None,
        phase_time: PhaseTime = None,
        phase_time_cls: type[PhaseTime] = PhaseTimeLinear,
        phase_time_kwargs: dict = {},
        *args,
        **kwargs,
    ):
        self.bspline_basis = BsplineBasisClamped(*args, **kwargs)
        self.control_points = control_points
        if phase_time is not None:
            self.phase_time = phase_time
        else:
            self.phase_time = phase_time_cls(
                *args, **kwargs, **phase_time_kwargs, num_time_points=self.bspline_basis.num_phase_points
            )

    @property
    def degree(self) -> int:
        return self.bspline_basis.degree

    @property
    def num_control_points(self) -> int:
        return self.bspline_basis.num_control_points

    @property
    def num_dof(self) -> int:
        assert self.control_points is not None
        return self.control_points.shape[-1]

    def set_control_points(self, control_points: jnp.ndarray):
        self.control_points = control_points

    def set_boundary_derivatives_to_zero(self):
        # Repeat control points to set the initial and final velocities and accelerations to zero (default)
        assert self.control_points is not None
        # Set velocities to zero
        self.control_points = self.control_points.at[1].set(self.control_points[0])
        self.control_points = self.control_points.at[-2].set(self.control_points[-1])
        # Set accelerations to zero
        self.control_points = self.control_points.at[2].set(self.control_points[0])
        self.control_points = self.control_points.at[-3].set(self.control_points[-1])

    def set_boundary_conditions(
        self,
        q_initial: jnp.ndarray = None,
        q_final: jnp.ndarray = None,
        dq_dt_initial: jnp.ndarray = None,
        dq_dt_final: jnp.ndarray = None,
        d2q_dt2_initial: jnp.ndarray = None,
        d2q_dt2_final: jnp.ndarray = None,
    ):
        """Sets boundary conditions for the B-spline movement primitive.
        Defaults derivatives to zero if not specified.
        """
        assert self.control_points is not None

        # Set initial and final positions
        if q_initial is not None:
            self.control_points = self.control_points.at[0].set(q_initial)
        if q_final is not None:
            self.control_points = self.control_points.at[-1].set(q_final)

        # Set initial and final velocities
        if dq_dt_initial is None:
            dq_dt_initial = jnp.zeros(self.num_dof)
        dp_ds_initial = dq_dt_initial / self.phase_time.r[0]
        self.control_points = self.control_points.at[1].set(
            self.control_points[0] - dp_ds_initial / self.bspline_basis.dB_ds[0, 0]
        )
        if dq_dt_final is None:
            dq_dt_final = jnp.zeros(self.num_dof)
        dp_ds_final = dq_dt_final / self.phase_time.r[-1]
        self.control_points = self.control_points.at[-2].set(
            self.control_points[-1] - dp_ds_final / self.bspline_basis.dB_ds[-1, -1]
        )

        # Set initial and final accelerations
        if d2q_dt2_initial is None:
            d2q_dt2_initial = jnp.zeros(self.num_dof)
        d2p_ds2_initial = (d2q_dt2_initial - dp_ds_initial * self.phase_time.dr_ds[0] * self.phase_time.r[0]) / (
            self.phase_time.r[0] ** 2
        )
        self.control_points = self.control_points.at[2].set(
            (
                d2p_ds2_initial
                - self.bspline_basis.d2B_ds2[0, 0] * self.control_points[0]
                - self.bspline_basis.d2B_ds2[0, 1] * self.control_points[1]
            )
            / self.bspline_basis.d2B_ds2[0, 2]
        )
        if d2q_dt2_final is None:
            d2q_dt2_final = jnp.zeros(self.num_dof)
        d2p_ds2_final = (d2q_dt2_final - dp_ds_final * self.phase_time.dr_ds[-1] * self.phase_time.r[-1]) / (
            self.phase_time.r[-1] ** 2
        )
        self.control_points = self.control_points.at[-3].set(
            (
                d2p_ds2_final
                - self.bspline_basis.d2B_ds2[-1, -1] * self.control_points[-1]
                - self.bspline_basis.d2B_ds2[-1, -2] * self.control_points[-2]
            )
            / self.bspline_basis.d2B_ds2[-1, -3]
        )

    def set_constraint_at_time(
        self,
        t: float,
        q: jnp.ndarray,
        dq_dt: jnp.ndarray = None,
        d2q_dt2: jnp.ndarray = None,
        control_points: jnp.ndarray = None
    ) -> jnp.ndarray:
        """Find the closest (in L2-norm) control points such that the B-spline
        trajectory satisfies position/velocity/acceleration constraints at time t.
        
        min_w ||w - w0||^2
        s.t. A w = c
        w0: current control points
        A: basis functions evaluated at phase s(t)
        c: constraint values (position/velocity/acceleration)
        """
        if control_points is None:
            control_points = self.control_points
        assert control_points is not None

        # Index of closest time
        assert 0.0 <= t <= self.phase_time.total_duration, f"t={t} outside [0, {self.phase_time.total_duration}]"
        idx = jnp.argmin(jnp.abs(self.phase_time.tt - t))

        rows = []
        rhs = []
        # Position constraint
        if q is not None:
            rows.append(self.bspline_basis.B[idx])
            rhs.append(q)

        # Velocity constraint
        if dq_dt is not None:
            rows.append(self.bspline_basis.dB_ds[idx])
            dq_ds = dq_dt / self.phase_time.r[idx]
            rhs.append(dq_ds)

        # Acceleration constraint
        if d2q_dt2 is not None:
            rows.append(self.bspline_basis.d2B_ds2[idx])
            dq_ds = dq_dt / self.phase_time.r[idx]
            d2q_ds2 = (d2q_dt2 - dq_ds * self.phase_time.dr_ds[idx] * self.phase_time.r[idx]) / (self.phase_time.r[idx] ** 2)
            rhs.append(d2q_ds2)

        # Basis matrix A and constraints vector c
        A = jnp.stack(rows, axis=0)
        c = jnp.stack(rhs, axis=0)

        # Compute correction
        Aw0 = A @ control_points
        delta = c - Aw0
        # Minimal-norm solution of A * correction = delta
        correction = jnp.linalg.lstsq(A, delta, rcond=None)[0]
        
        # Update control points
        control_points_new = control_points + correction
        self.control_points = self.control_points.at[:].set(control_points_new)

    def get_trajectory(self, control_points: jnp.ndarray = None, in_phase: bool = False) -> Trajectory:
        if control_points is None:
            control_points = self.control_points
        p = self.bspline_basis.B @ control_points
        dp_ds = self.bspline_basis.dB_ds @ control_points
        d2p_ds2 = self.bspline_basis.d2B_ds2 @ control_points
        d3p_ds3 = self.bspline_basis.d3B_ds3 @ control_points
        if in_phase:
            # Trajectory in phase
            return Trajectory(q=p, dq=dp_ds, ddq=d2p_ds2, dddq=d3p_ds3)
        else:
            # Trajectory in time
            r_s = self.phase_time.r
            dr_ds = self.phase_time.dr_ds
            d2r_ds2 = self.phase_time.d2r_ds2
            return Trajectory(
                q=p,
                dq=dp_ds * r_s[..., None],
                ddq=d2p_ds2 * r_s[..., None] ** 2 + dp_ds * dr_ds[..., None] * r_s[..., None],
                dddq=d3p_ds3 * r_s[..., None] ** 3
                + d2p_ds2 * 2 * dr_ds[..., None] * r_s[..., None] ** 2
                + d2p_ds2 * dr_ds[..., None] * r_s[..., None] ** 2
                + dp_ds * d2r_ds2[..., None] * r_s[..., None] ** 2
                + dp_ds * dr_ds[..., None] ** 2 * r_s[..., None],
            )


if __name__ == "__main__":
    bmp = BsplineMP(degree=4, num_control_points=12, num_phase_points=512, total_duration=7.0)
    control_points = jnp.repeat(jnp.linspace(0, 1, bmp.num_control_points)[:, None], 2, axis=-1)
    key = jax.random.PRNGKey(0)
    # add some noise
    control_points = control_points.at[1:-1].add(0.025 * jax.random.normal(key, shape=control_points[1:-1].shape))
    bmp.set_control_points(control_points)
    bmp.set_boundary_derivatives_to_zero()
    bmp.set_boundary_conditions(q_initial=control_points[0], q_final=control_points[-1])
    trajectory = bmp.get_trajectory()
