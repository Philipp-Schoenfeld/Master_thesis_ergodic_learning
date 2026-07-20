import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from bsplinax.movement_primitive import BsplineMP
from bsplinax.phase_time import PhaseTimeLinear

if __name__ == "__main__":
    bmp = BsplineMP(
        degree=4, num_control_points=12, num_phase_points=512, total_duration=3.0, phase_time_cls=PhaseTimeLinear
    )
    control_points = jnp.repeat(jnp.linspace(0, 1, bmp.num_control_points)[:, None], 2, axis=-1)
    # Add some noise to the inner control points
    key = jax.random.PRNGKey(0)
    control_points = control_points.at[1:-1].add(0.025 * jax.random.normal(key, shape=control_points[1:-1].shape))
    bmp.set_control_points(control_points)
    bmp.set_boundary_derivatives_to_zero()

    # Set boundary conditions
    q_initial = control_points[0]
    q_final = control_points[-1]
    dq_dt_initial = jnp.array([0.1, 0.2])
    dq_dt_final = jnp.array([0.1, -0.1])
    d2q_dt2_initial = jnp.array([0.0, 0.0])
    d2q_dt2_final = jnp.array([0.0, 0.0])
    bmp.set_boundary_conditions(
        q_initial=q_initial,
        q_final=q_final,
        dq_dt_initial=dq_dt_initial,
        dq_dt_final=dq_dt_final,
        d2q_dt2_initial=d2q_dt2_initial,
        d2q_dt2_final=d2q_dt2_final,
    )
    
    # Set a position constraint at time t=total_duration/2
    t_des = bmp.phase_time.total_duration / 2
    q_des = jnp.array([0.7, 0.6])
    dq_dt_des = jnp.array([0.1, 0.4])
    d2q_dt2_des = jnp.array([-1.0, -0.5])
    bmp.set_constraint_at_time(t_des, q=q_des, dq_dt=dq_dt_des, d2q_dt2=d2q_dt2_des)

    # Compute trajectories
    traj_t = bmp.get_trajectory()
    traj_s = bmp.get_trajectory(in_phase=True)

    # Plot trajectories in 2D
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(traj_t.q[:, 0], traj_t.q[:, 1], label="B-spline path")
    ax.plot(bmp.control_points[:, 0], bmp.control_points[:, 1], "x", color="red", label="Control points")
    for i in range(bmp.control_points.shape[0]):
        ax.text(bmp.control_points[i, 0], bmp.control_points[i, 1], str(i), fontsize=12, ha="right")
    ax.quiver(
        q_initial[0],
        q_initial[1],
        dq_dt_initial[0],
        dq_dt_initial[1],
        color="red",
        label="dq_dt initial desired",
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax.quiver(
        bmp.control_points[0, 0],
        bmp.control_points[0, 1],
        traj_t.dq[0, 0],
        traj_t.dq[0, 1],
        color="blue",
        label="dq_dt initial",
        angles="xy",
        scale_units="xy",
        scale=1,
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
        scale=1,
    )
    ax.quiver(
        bmp.control_points[-1, 0],
        bmp.control_points[-1, 1],
        traj_t.dq[-1, 0],
        traj_t.dq[-1, 1],
        color="blue",
        label="dq_dt final",
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax.legend()
    ax.set_xlabel("d_0")
    ax.set_ylabel("d_1")
    fig.tight_layout()

    # Plot trajectories in phase space
    fig, axs = plt.subplots(bmp.control_points.shape[-1], 4, figsize=(8, 4))
    ss = bmp.bspline_basis.ss
    for i in range(traj_s.q.shape[-1]):
        axs[i, 0].plot(ss, traj_s.q[:, i], label=f"p_{i}(s)")
        axs[i, 1].plot(ss, traj_s.dq[:, i], label=f"dp_{i}/ds(s)")
        axs[i, 2].plot(ss, traj_s.ddq[:, i], label=f"d2p_{i}/ds2(s)")
        axs[i, 3].plot(ss, traj_s.dddq[:, i], label=f"d3p_{i}/ds3(s)")
    for i in range(axs.shape[-1]):
        axs[0, i].set_title(f"d^{i}p/ds^{i}(s)")
        axs[-1, i].set_xlabel("s")
    for i in range(traj_s.q.shape[-1]):
        axs[i, 0].set_ylabel(f"d_{i}")
    fig.tight_layout()

    # Plot trajectories in time
    fig, axs = plt.subplots(bmp.control_points.shape[-1], 4, figsize=(8, 4))
    tt = bmp.phase_time.tt
    for i in range(bmp.num_dof):
        axs[i, 0].plot(tt, traj_t.q[:, i], label=f"q_{i}(t)")
        axs[i, 1].plot(tt, traj_t.dq[:, i], label=f"dq_{i}/dt(t)")
        axs[i, 2].plot(tt, traj_t.ddq[:, i], label=f"d2q_{i}/dt2(t)")
        axs[i, 3].plot(tt, traj_t.dddq[:, i], label=f"d3q_{i}/dt3(t)")
    # Plot desired boundaries
    for i in range(bmp.num_dof):
        axs[i, 0].scatter([tt[0], tt[-1]], [q_initial[i], q_final[i]], color="red", marker="o")
        axs[i, 1].scatter([tt[0], tt[-1]], [dq_dt_initial[i], dq_dt_final[i]], color="red", marker="o")
        axs[i, 2].scatter([tt[0], tt[-1]], [d2q_dt2_initial[i], d2q_dt2_final[i]], color="red", marker="o")
    for i in range(axs.shape[-1]):
        axs[0, i].set_title(f"d^{i}q/dt^{i}(t)")
        axs[-1, i].set_xlabel("t")
    for i in range(bmp.num_dof):
        axs[i, 0].set_ylabel(f"d_{i}")
    for i in range(traj_s.q.shape[-1]):
        axs[i, 0].scatter(t_des, q_des[i], color="red", marker="x")
        axs[i, 1].scatter(t_des, dq_dt_des[i], color="red", marker="x")
        axs[i, 2].scatter(t_des, d2q_dt2_des[i], color="red", marker="x")
    fig.tight_layout()

    plt.show()
