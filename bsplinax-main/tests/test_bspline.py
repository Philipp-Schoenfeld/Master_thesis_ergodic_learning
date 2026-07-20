import jax
import jax.numpy as jnp
import pytest

from bsplinax.bspline import bspline_basis_at_s
from bsplinax.movement_primitive import BsplineMP
from bsplinax.phase_time import PhaseTime, PhaseTimeLinear, PhaseTimeParabola


@pytest.mark.parametrize("is_clamped", [False, True])
def test_bspline_partition_of_unity_param(is_clamped: bool):
    degree = 3
    num_control_points = 6
    num_knots = num_control_points + degree + 1

    if not is_clamped:
        knots = jnp.linspace(0.0, 1.0, num_knots)
    else:
        # pad endpoints with repeated knots for clamped B-spline
        knots = jnp.pad(jnp.linspace(0.0, 1.0, num_control_points - degree + 1), (degree, degree), mode="edge")

    if not is_clamped:
        # evaluate B-spline basis functions inside valid domain (avoid endpoints)
        ss = jnp.linspace(knots[degree], knots[-degree - 1], 17)
    else:
        ss = jnp.linspace(0.0, knots[-1], 17, endpoint=False)

    for s in ss:
        B_s = bspline_basis_at_s(s, knots, degree)
        partition_sum = jnp.sum(B_s)
        # Check partition of unity
        assert jnp.isclose(partition_sum, 1.0, atol=1e-6), f"is_clamped={is_clamped}, s={s}, sum={partition_sum}"


@pytest.mark.parametrize("phase_time_cls", [PhaseTimeLinear, PhaseTimeParabola])
def test_phase_time(phase_time_cls: type[PhaseTime]):
    phase_time = phase_time_cls(num_time_points=512, total_duration=7.0)
    assert jnp.isclose(phase_time.tt[0], 0.0)
    assert jnp.isclose(phase_time.tt[-1], 7.0)


def test_bspline_movement_primitive():
    bmp = BsplineMP(degree=4, num_control_points=12, num_phase_points=512, total_duration=7.0)
    control_points = jnp.repeat(jnp.linspace(0, 1, bmp.num_control_points)[:, None], 2, axis=-1)
    key = jax.random.PRNGKey(0)
    # add some noise
    control_points = control_points.at[1:-1].add(0.025 * jax.random.normal(key, shape=control_points[1:-1].shape))
    bmp.set_control_points(control_points)
    bmp.set_boundary_derivatives_to_zero()
    bmp.set_boundary_conditions(q_initial=control_points[0], q_final=control_points[-1])
    trajectory = bmp.get_trajectory()
    assert jnp.allclose(trajectory.q[0], control_points[0])
    assert jnp.allclose(trajectory.q[-1], control_points[-1])
    assert jnp.allclose(trajectory.dq[0], jnp.zeros_like(control_points[0]))
    assert jnp.allclose(trajectory.dq[-1], jnp.zeros_like(control_points[-1]))
    assert jnp.allclose(trajectory.ddq[0], jnp.zeros_like(control_points[0]))
    assert jnp.allclose(trajectory.ddq[-1], jnp.zeros_like(control_points[-1]))

    # set some arbitrary velocity and acceleration boundary conditions
    dq_dt_initial = jnp.array([-0.1, 0.7])
    dq_dt_final = jnp.array([0.2, 0.3])
    d2q_dt2_initial = jnp.array([0.1, 0.2])
    d2q_dt2_final = jnp.array([-0.3, 0.6])
    bmp.set_boundary_conditions(
        dq_dt_initial=dq_dt_initial,
        dq_dt_final=dq_dt_final,
        d2q_dt2_initial=d2q_dt2_initial,
        d2q_dt2_final=d2q_dt2_final,
    )
    trajectory = bmp.get_trajectory()
    assert jnp.allclose(trajectory.dq[0], dq_dt_initial)
    assert jnp.allclose(trajectory.dq[-1], dq_dt_final)
    assert jnp.allclose(trajectory.ddq[0], d2q_dt2_initial)
    assert jnp.allclose(trajectory.ddq[-1], d2q_dt2_final)


if __name__ == "__main__":
    pytest.main([__file__])
