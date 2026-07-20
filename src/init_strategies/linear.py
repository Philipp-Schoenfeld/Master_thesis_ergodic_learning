import numpy as np

def init_particles(N, T, noise_std=0.02):
    t = np.linspace(0, 1, T)[:, None]
    start = np.array([0.2, 0.15])
    end = np.array([0.8, 0.85])
    base_traj = (1 - t) * start + t * end
    
    particles = []
    for _ in range(N):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(T, 2))
        traj = base_traj + noise
        traj = np.clip(traj, 0.02, 0.98)
        particles.append(traj.ravel())
        
    return np.array(particles), base_traj
