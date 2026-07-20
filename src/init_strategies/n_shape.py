import numpy as np

def init_particles(N, T, noise_std=0.02):
    pts = np.array([
        [0.25, 0.15],
        [0.25, 0.85],
        [0.75, 0.15],
        [0.75, 0.85]
    ])
    
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    cum_dists /= cum_dists[-1]
    
    t = np.linspace(0, 1, T)
    base_traj = np.stack([
        np.interp(t, cum_dists, pts[:, 0]),
        np.interp(t, cum_dists, pts[:, 1])
    ], axis=-1)
    
    particles = []
    for _ in range(N):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(T, 2))
        traj = base_traj + noise
        traj = np.clip(traj, 0.02, 0.98)
        particles.append(traj.ravel())
        
    return np.array(particles), base_traj
