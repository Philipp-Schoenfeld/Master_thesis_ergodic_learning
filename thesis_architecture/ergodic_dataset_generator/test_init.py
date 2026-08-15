import numpy as np

def _generate_initial_trajectory(x0, means, tsteps, dt):
    from scipy.ndimage import gaussian_filter1d
    
    # 1. Greedy TSP
    unvisited = list(means)
    path = [x0]
    curr = x0
    while unvisited:
        dists = [np.linalg.norm(np.array(curr) - np.array(m)) for m in unvisited]
        idx = np.argmin(dists)
        curr = unvisited.pop(idx)
        path.append(curr)
    path = np.array(path)
    
    # 2. Linear interpolation by distance
    diffs = np.diff(path, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dists[-1]
    
    if total_dist < 1e-5:
        p_traj = np.tile(path[0], (tsteps, 1))
    else:
        t_eval = np.linspace(0, total_dist, tsteps)
        x = np.interp(t_eval, cum_dists, path[:, 0])
        y = np.interp(t_eval, cum_dists, path[:, 1])
        p_traj = np.column_stack([x, y])
    
    # 3. Smooth the path so second derivatives are not delta functions
    p_traj = gaussian_filter1d(p_traj, sigma=tsteps/20.0, axis=0, mode='nearest')
    
    # Force start exactly at x0
    p_traj[0] = x0
    
    # 4. Compute controls (second derivative)
    u_traj = np.zeros_like(p_traj)
    for t in range(1, tsteps - 1):
        u_traj[t] = (p_traj[t+1] - 2*p_traj[t] + p_traj[t-1]) / (dt**2)
    u_traj[0] = u_traj[1]
    u_traj[-1] = u_traj[-2]
    
    # Initial velocity
    v0 = (p_traj[1] - p_traj[0]) / dt
    
    return p_traj, u_traj, v0

x0 = (0.1, 0.2)
means = [[0.5, 0.5], [0.8, 0.2], [0.2, 0.8]]
tsteps = 200
dt = 0.05
p, u, v0 = _generate_initial_trajectory(x0, means, tsteps, dt)
print("p_traj shape:", p.shape)
print("u_traj shape:", u.shape)
print("max u:", np.max(np.abs(u)))
print("v0:", v0)
