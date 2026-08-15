import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('/home/philipp/Documents/Uni/Master_thesis/thesis_architecture/ergodic_dataset_generator')
from ergodic_solver import _build_lqr, _make_stein_grad
from shape_library import get_shape, make_pdf_and_score
import matplotlib.pyplot as plt

def pid_track(p_traj, dt):
    T = len(p_traj) - 1
    u = np.zeros((T, 2))
    p = p_traj[0].copy()
    v = (p_traj[1] - p_traj[0]) / dt
    v0 = v.copy()
    Kp = 150.0
    Kd = 20.0
    for t in range(T):
        p_target = p_traj[t+1]
        v_target = (p_traj[min(t+2, T)] - p_traj[t]) / (2*dt) if t < T-1 else (p_traj[T] - p_traj[T-1])/dt
        u[t] = Kp * (p_target - p) + Kd * (v_target - v)
        p = p + v * dt + 0.5 * u[t] * dt**2
        v = v + u[t] * dt
    return u, v0

shape_def = get_shape('test_gmm_2')
_, score_fn = make_pdf_and_score(shape_def)
means = shape_def['means']
x0 = (0.1, 0.2)
tsteps = 200
dt = 0.05

# TSP Path
unvisited = list(means)
path = [x0]
curr = x0
while unvisited:
    dists = [np.linalg.norm(np.array(curr) - np.array(m)) for m in unvisited]
    idx = np.argmin(dists)
    curr = unvisited.pop(idx)
    path.append(curr)
path = np.array(path)

# Interpolate
diffs = np.diff(path, axis=0)
dists = np.linalg.norm(diffs, axis=1)
cum_dists = np.concatenate(([0], np.cumsum(dists)))
total_dist = cum_dists[-1]
t_eval = np.linspace(0, total_dist, tsteps+1)
x = np.interp(t_eval, cum_dists, path[:, 0])
y = np.interp(t_eval, cum_dists, path[:, 1])
p_traj = np.column_stack([x, y])

from scipy.ndimage import gaussian_filter1d
p_traj = gaussian_filter1d(p_traj, sigma=3, axis=0, mode='nearest')
p_traj[0] = x0

u, v0 = pid_track(p_traj, dt)

pm, linearize_dyn, solve_lqr = _build_lqr(dt)
x0j = jnp.array([p_traj[0,0], p_traj[0,1], v0[0], v0[1]])
init_sim_traj = pm.traj_sim(x0j, jnp.array(u))
p_sim = np.array(init_sim_traj)[:, :2]

print("Max tracking error:", np.max(np.abs(p_sim - p_traj[:-1])))

# Now run SVGD for 50 iters to see if it explodes
u_traj = jnp.array(u)
stein_grad_jit = _make_stein_grad(score_fn)
z0 = jnp.zeros(4)

for i in range(50):
    x_traj, A_traj, B_traj = linearize_dyn(x0j, u_traj)
    stein_dx               = stein_grad_jit(x_traj, h=0.01)
    v_traj, _              = solve_lqr(z0, A_traj, B_traj, stein_dx)
    u_traj                += 0.01 * v_traj
    
    if i % 10 == 0:
        max_dx = np.max(np.abs(stein_dx))
        print(f"Iter {i}: max stein_dx = {max_dx:.2f}")

final_traj = pm.traj_sim(x0j, u_traj)
final_traj = np.array(final_traj)[:, :2]

plt.plot(p_traj[:, 0], p_traj[:, 1], 'w--', label='TSP Init')
plt.plot(final_traj[:, 0], final_traj[:, 1], 'c-', label='Final Traj')
plt.scatter([m[0] for m in means], [m[1] for m in means], c='r')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('test_svgd.png')
