import numpy as np

def init_particles(N, T, degree=5, noise_std=0.02):
    """
    Polynomial approximation of the N-shape trajectory.
    We first generate a high-resolution ideal N-shape, then fit a polynomial
    of the specified degree to it (parametric in t), and sample from that.
    """
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
    
    t_fine = np.linspace(0, 1, 100)
    x_fine = np.interp(t_fine, cum_dists, pts[:, 0])
    y_fine = np.interp(t_fine, cum_dists, pts[:, 1])
    
    # Fit parametric polynomial
    px = np.polyfit(t_fine, x_fine, degree)
    py = np.polyfit(t_fine, y_fine, degree)
    
    t = np.linspace(0, 1, T)
    x_poly = np.polyval(px, t)
    y_poly = np.polyval(py, t)
    
    base_traj = np.stack([x_poly, y_poly], axis=-1)
    
    particles = []
    for _ in range(N):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(T, 2))
        traj = base_traj + noise
        traj = np.clip(traj, 0.02, 0.98)
        particles.append(traj.ravel())
        
    return np.array(particles), base_traj
