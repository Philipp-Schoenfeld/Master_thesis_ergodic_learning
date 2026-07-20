import numpy as np


# Default endpoints per dimension
_ENDPOINTS = {
    2: {'start': [0.2, 0.15], 'end': [0.8, 0.85]},
    3: {'start': [0.2, 0.15, 0.15], 'end': [0.8, 0.85, 0.85]},
}


def init_particles(N, T, dim=2, noise_std=0.02):
    """
    Linear interpolation from start to end with Gaussian noise.

    Args:
        N: number of particles
        T: number of time steps
        dim: spatial dimension (2 or 3)
        noise_std: standard deviation of Gaussian noise

    Returns:
        particles: (N, T*dim) flattened trajectories
        base_traj: (T, dim) noiseless base trajectory
    """
    endpoints = _ENDPOINTS.get(dim, {
        'start': [0.2] * dim,
        'end': [0.8] * dim,
    })
    start = np.array(endpoints['start'])
    end = np.array(endpoints['end'])

    t = np.linspace(0, 1, T)[:, None]  # (T, 1)
    base_traj = (1 - t) * start + t * end  # (T, dim)

    particles = []
    for _ in range(N):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(T, dim))
        traj = base_traj + noise
        traj = np.clip(traj, 0.02, 0.98)
        particles.append(traj.ravel())

    return np.array(particles), base_traj
