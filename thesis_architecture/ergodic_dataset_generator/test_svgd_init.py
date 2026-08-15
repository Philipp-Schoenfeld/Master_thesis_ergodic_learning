import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/philipp/Documents/Uni/Master_thesis/thesis_architecture/ergodic_dataset_generator')
from ergodic_solver import run_ergodic_coverage, _generate_initial_trajectory, _build_lqr
from shape_library import get_shape, make_pdf_and_score

shape_def = get_shape('test_gmm_1')
_, score_fn = make_pdf_and_score(shape_def)

dt = 0.05
tsteps = 200
x0 = (0.1, 0.2)
target_means = shape_def['means']

p_traj, u_traj_np, v0 = _generate_initial_trajectory(x0, target_means, tsteps, dt)
x0j = jnp.array([x0[0], x0[1], v0[0], v0[1]])
u_traj = jnp.array(u_traj_np)

pm, _, _ = _build_lqr(dt)
init_sim_traj = pm.traj_sim(x0j, u_traj)
init_sim_traj = np.array(init_sim_traj)[:, :2]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("TSP Path vs Simulated Init Traj")
plt.plot(p_traj[:, 0], p_traj[:, 1], 'b-', label='TSP Path (p_traj)')
plt.plot(init_sim_traj[:, 0], init_sim_traj[:, 1], 'r--', label='Simulated (u_traj)')
plt.scatter([m[0] for m in target_means], [m[1] for m in target_means], c='k', marker='x', label='Means')
plt.legend()

plt.subplot(1, 2, 2)
plt.title("u_traj magnitude")
plt.plot(np.linalg.norm(u_traj_np, axis=1))

plt.savefig('debug_init.png')
print("Max diff:", np.max(np.abs(p_traj - init_sim_traj)))
