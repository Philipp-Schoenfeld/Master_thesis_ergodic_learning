from .linear import init_particles as linear_init
from .n_shape import init_particles as n_shape_init
from .polynomial import init_particles as polynomial_init
from .random_init import init_particles as random_init
from .rrt_init import init_particles as rrt_init

def get_initialization(strategy_name, N, T, **kwargs):
    strategies = {
        'linear': linear_init,
        'n_shape': n_shape_init,
        'polynomial': polynomial_init,
        'random': random_init,
        'rrt': rrt_init
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}")
        
    return strategies[strategy_name](N, T, **kwargs)
