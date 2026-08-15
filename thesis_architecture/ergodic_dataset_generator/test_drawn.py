import numpy as np
from shape_library import get_shape, make_pdf_and_score
from ergodic_solver import run_ergodic_coverage, _pid_track

# simulate drawn path from (0.2, 0.2) to (0.8, 0.8)
path = np.linspace([0.2, 0.2], [0.8, 0.8], 50)
diffs = np.diff(path, axis=0)
dists = np.linalg.norm(diffs, axis=1)
cum_dists = np.concatenate(([0], np.cumsum(dists)))
total_dist = cum_dists[-1]

tsteps = 200
idx_eval = np.linspace(0, total_dist, tsteps + 1)
p_traj = np.column_stack([
    np.interp(idx_eval, cum_dists, path[:, 0]),
    np.interp(idx_eval, cum_dists, path[:, 1])
])

print("p_traj max diff:", np.max(np.abs(np.diff(p_traj, axis=0))))

u_traj, v0 = _pid_track(p_traj, dt=0.05, tsteps=tsteps)
print("u_traj max:", np.max(np.abs(u_traj)))
print("u_traj has nan?", np.isnan(u_traj).any())

shape_def = get_shape('test_gmm_1')
_, score_fn = make_pdf_and_score(shape_def)

traj_xy, _ = run_ergodic_coverage(
    score_fn, x0=p_traj[0], shape_def=shape_def, 
    custom_p_traj=p_traj, dt=0.05, tsteps=tsteps, num_iters=200, verbose=True
)

print("traj_xy has nan?", np.isnan(traj_xy).any())
print("traj_xy head:\n", traj_xy[:5])

