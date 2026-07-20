import jax
import jax.numpy as jnp
import scipy
from matplotlib import pyplot as plt

from bsplinax.movement_primitive import BsplineMP, Trajectory

if __name__ == "__main__":
    """
    Fit a B-spline to a synthetic trajectory of positions and velocities.
    """
    # Generate synthetic trajectory data
    total_duration = 3.0  # seconds
    freq_sampling = 10  # Hz
    num_sampled_points = int(total_duration * freq_sampling)
    tt_sampling = jnp.linspace(0, total_duration, num_sampled_points)
    q_t_data_raw = jnp.repeat(jnp.linspace(0, 1, num_sampled_points)[:, None], 2, axis=-1)
    q_t_data_raw = q_t_data_raw.at[:, 1].set(jnp.sin(q_t_data_raw[:, 0] * 2 * jnp.pi) * 0.5 + 0.5)
    # add some noise to dof
    q_t_data_raw = q_t_data_raw.at[:, 0].set(
        q_t_data_raw[:, 0] + 0.01 * jax.random.normal(jax.random.PRNGKey(0), q_t_data_raw[:, 0].shape)
    )
    q_t_data_raw = q_t_data_raw.at[:, 1].set(
        q_t_data_raw[:, 1] + 0.05 * jax.random.normal(jax.random.PRNGKey(0), q_t_data_raw[:, 1].shape)
    )
    # Smooth with a savgol filter
    q_t_data_raw = scipy.signal.savgol_filter(q_t_data_raw, window_length=5, polyorder=2, axis=0)
    # Get finite difference velocities
    dq_dt_data_raw = jnp.gradient(q_t_data_raw, tt_sampling, axis=0)

    # Interpolate to desired number of points
    num_phase_points = 512
    ss_raw = jnp.linspace(0.0, 1.0, q_t_data_raw.shape[0])
    ss_interp = jnp.linspace(0.0, 1.0, num_phase_points)
    vmap_jnp_interp = jax.vmap(lambda col: jnp.interp(ss_interp, ss_raw, col))
    q_t_data = vmap_jnp_interp(q_t_data_raw.T).T
    dq_dt_data = vmap_jnp_interp(dq_dt_data_raw.T).T
    traj_data = Trajectory(q=q_t_data, dq=dq_dt_data)

    # Fit control points to data (position and velocity) in phase space
    bmp = BsplineMP(degree=3, num_control_points=16, num_phase_points=num_phase_points, total_duration=total_duration)
    B = jnp.concatenate([bmp.bspline_basis.B, bmp.bspline_basis.dB_ds], axis=0)
    p_dp = jnp.concatenate([traj_data.q, traj_data.dq / bmp.phase_time.r[:, None]], axis=0)
    control_points = jnp.linalg.lstsq(B, p_dp, rcond=None)[0]
    bmp.set_control_points(control_points)
    q_initial = traj_data.q[0]
    q_final = traj_data.q[-1]
    dq_dt_initial = traj_data.dq[0]
    dq_dt_final = traj_data.dq[-1]
    bmp.set_boundary_conditions(
        q_initial=q_initial, q_final=q_final, dq_dt_initial=dq_dt_initial, dq_dt_final=dq_dt_final
    )
    traj_pred = bmp.get_trajectory()

    # Plot data and fitted B-spline
    # In 2D space
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(traj_data.q[:, 0], traj_data.q[:, 1], label="Trajectory", s=1)
    ax.scatter(bmp.control_points[:, 0], bmp.control_points[:, 1], color="green", label="Control Points")
    ax.plot(traj_pred.q[:, 0], traj_pred.q[:, 1], color="orange", label="Fitted B-spline")
    ax.quiver(
        q_initial[0],
        q_initial[1],
        dq_dt_initial[0],
        dq_dt_initial[1],
        color="red",
        label="dq_dt initial desired",
        angles="xy",
        scale_units="xy",
        scale=5,
    )
    ax.quiver(
        traj_pred.q[0, 0],
        traj_pred.q[0, 1],
        traj_pred.dq[0, 0],
        traj_pred.dq[0, 1],
        color="blue",
        label="dq_dt initial",
        angles="xy",
        scale_units="xy",
        scale=5,
    )
    ax.quiver(
        q_final[0],
        q_final[1],
        dq_dt_final[0],
        dq_dt_final[1],
        color="red",
        label="dq_dt final desired",
        angles="xy",
        scale_units="xy",
        scale=5,
    )
    ax.quiver(
        traj_pred.q[-1, 0],
        traj_pred.q[-1, 1],
        traj_pred.dq[-1, 0],
        traj_pred.dq[-1, 1],
        color="blue",
        label="dq_dt final",
        angles="xy",
        scale_units="xy",
        scale=5,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Trajectory")
    ax.legend()

    # In time
    fig, axs = plt.subplots(figsize=(8, 6), nrows=traj_pred.q.shape[-1], ncols=2)
    tt = bmp.phase_time.tt
    for i, ax in enumerate(axs):
        ax[0].scatter(tt[0], q_initial[i], color="red", s=10, label="initial")
        ax[0].scatter(tt[-1], q_final[i], color="red", s=10, label="final")
        ax[0].plot(tt, traj_data.q[:, i], label="data")
        ax[0].plot(tt, traj_pred.q[:, i], label="prediction")
        ax[1].scatter(tt[0], dq_dt_initial[i], color="red", s=10, label="initial")
        ax[1].scatter(tt[-1], dq_dt_final[i], color="red", s=10, label="final")
        ax[1].plot(tt, traj_data.dq[:, i], label="data")
        ax[1].plot(tt, traj_pred.dq[:, i], label="prediction")

        ax[0].set_ylabel(f"d_{i}")

    axs[0, 0].set_title("Position")
    axs[0, 1].set_title("Velocity")
    axs[-1, 0].set_xlabel("t")
    axs[-1, 1].set_xlabel("t")

    plt.show()
