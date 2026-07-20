import numpy as np

def init_particles(N, T, **kwargs):
    particles = []
    for _ in range(N):
        traj = np.random.uniform(0.05, 0.95, size=(T, 2))
        particles.append(traj.ravel())
        
    return np.array(particles), None
