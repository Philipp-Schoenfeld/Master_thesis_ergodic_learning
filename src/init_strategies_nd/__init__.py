"""
Dimension-parametric initialization strategies for SVGD trajectory particles.
Supports 2D and 3D (and in principle any dimension).
"""

from .linear import init_particles as linear_init
from .n_shape import init_particles as n_shape_init
from .random_init import init_particles as random_init


def get_initialization(strategy_name, N, T, dim=2, **kwargs):
    """
    Get initial particle trajectories for a given strategy.

    Args:
        strategy_name: one of 'linear', 'n_shape', 'random'
        N: number of particles
        T: number of time steps
        dim: spatial dimension (2 or 3)
        **kwargs: passed to the strategy (e.g. noise_std)

    Returns:
        particles: (N, T*dim) flattened trajectories
        base_traj: (T, dim) base trajectory or None
    """
    strategies = {
        'linear': linear_init,
        'n_shape': n_shape_init,
        'random': random_init,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")

    return strategies[strategy_name](N, T, dim=dim, **kwargs)
