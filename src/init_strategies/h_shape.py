import numpy as np


def init_particles(N, T, noise_std=0.02, **kwargs):
    """
    Piecewise-linear H-shape trajectory (straight-line connections).

    The H is traced as a single connected stroke:
        bottom-left → top-left  [left leg, going up]
        mid-left    → mid-right [crossbar]
        top-right   → bottom-right [right leg, going down]

    Points are placed with equal spacing along arc-length, then T samples
    are drawn with np.interp — no polynomial smoothing, pure straight lines.

    Parameters
    ----------
    N         : int   — number of particles (batch size)
    T         : int   — number of control points per trajectory
    noise_std : float — Gaussian noise std added to each particle

    Returns
    -------
    particles : np.ndarray (N, T*2)  — ravelled noisy trajectories
    base_traj : np.ndarray (T, 2)   — noiseless reference trajectory
    """
    # Key waypoints for the single-stroke H
    pts = np.array([
        [0.25, 0.15],   # bottom of left leg
        [0.25, 0.85],   # top of left leg
        [0.25, 0.50],   # back to crossbar height (left end)
        [0.75, 0.50],   # crossbar (right end)
        [0.75, 0.85],   # top of right leg
        [0.75, 0.15],   # bottom of right leg
    ])

    # Arc-length parameterisation → uniform t in [0,1]
    diffs     = np.diff(pts, axis=0)
    dists     = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    cum_dists = cum_dists / cum_dists[-1]

    # Sample T points with straight-line (piecewise-linear) interpolation
    t = np.linspace(0, 1, T)
    x = np.interp(t, cum_dists, pts[:, 0])
    y = np.interp(t, cum_dists, pts[:, 1])

    base_traj = np.stack([x, y], axis=-1)   # (T, 2)

    particles = []
    for _ in range(N):
        noise = np.random.normal(0.0, noise_std, (T, 2))
        traj  = np.clip(base_traj + noise, 0.02, 0.98)
        particles.append(traj.ravel())

    return np.array(particles), base_traj
