import numpy as np


# N-shape waypoints per dimension
_WAYPOINTS = {
    2: [
        [0.25, 0.15],
        [0.25, 0.85],
        [0.75, 0.15],
        [0.75, 0.85],
    ],
    3: [
        [0.25, 0.15, 0.2],
        [0.25, 0.85, 0.5],
        [0.75, 0.15, 0.8],
        [0.75, 0.85, 0.5],
    ],
}


def init_particles(N, T, dim=2, noise_std=0.02):
    """
    N-shape waypoint interpolation with Gaussian noise.

    Args:
        N: number of particles
        T: number of time steps
        dim: spatial dimension (2 or 3)
        noise_std: standard deviation of Gaussian noise

    Returns:
        particles: (N, T*dim) flattened trajectories
        base_traj: (T, dim) noiseless base trajectory
    """
    pts = np.array(_WAYPOINTS.get(dim, _WAYPOINTS[2]))

    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    cum_dists /= cum_dists[-1]

    t = np.linspace(0, 1, T)
    base_traj = np.stack([
        np.interp(t, cum_dists, pts[:, d]) for d in range(dim)
    ], axis=-1)  # (T, dim)

    particles = []
    for _ in range(N):
        noise = np.random.normal(loc=0.0, scale=noise_std, size=(T, dim))
        traj = base_traj + noise
        traj = np.clip(traj, 0.02, 0.98)
        particles.append(traj.ravel())

    return np.array(particles), base_traj
