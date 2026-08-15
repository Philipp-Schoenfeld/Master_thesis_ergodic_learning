import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('/home/philipp/Documents/Uni/Master_thesis/thesis_architecture/ergodic_dataset_generator')
from ergodic_solver import _build_lqr

def inverse_euler(p_traj, dt):
    # p_traj has length T+1. We want u to have length T.
    T = len(p_traj) - 1
    v = np.zeros((T+1, 2))
    u = np.zeros((T, 2))
    for t in range(T):
        v[t] = (p_traj[t+1] - p_traj[t]) / dt
    v[T] = v[T-1]
    for t in range(T):
        u[t] = (v[t+1] - v[t]) / dt
    return u, v[0]

dt = 0.05
T = 200
t_eval = np.linspace(0, 2*np.pi, T+1)
p_traj = np.column_stack([np.sin(t_eval), np.cos(t_eval)])

u, v0 = inverse_euler(p_traj, dt)

pm, _, _ = _build_lqr(dt)
x0j = jnp.array([p_traj[0,0], p_traj[0,1], v0[0], v0[1]])
init_sim_traj = pm.traj_sim(x0j, jnp.array(u))
p_sim = np.array(init_sim_traj)[:, :2]

print("Max diff:", np.max(np.abs(p_sim - p_traj[:-1])))
