import numpy as np


def init_particles(N, T, dim=2, **kwargs):
    """
    Uniform random initialization in [0.05, 0.95]^dim.

    Args:
        N: number of particles
        T: number of time steps
        dim: spatial dimension (2 or 3)

    Returns:
        particles: (N, T*dim) flattened trajectories
        base_traj: None
    """
    particles = []
    for _ in range(N):
        traj = np.random.uniform(0.05, 0.95, size=(T, dim))
        particles.append(traj.ravel())

    return np.array(particles), None
