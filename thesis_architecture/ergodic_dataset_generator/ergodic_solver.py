"""
ergodic_solver.py
=================
Functional wrapper around the Stein variational flow ergodic coverage solver.

Core logic is identical to stein_flow_coverage.py — only refactored into a
callable function that accepts an arbitrary JAX score function so it can be
driven by any GMM-based target density.

Usage:
    from ergodic_solver import run_ergodic_coverage
    from shape_library import SHAPES, make_pdf_and_score

    pdf_fn, score_fn = make_pdf_and_score(SHAPES['N_shape'])
    traj_xy = run_ergodic_coverage(score_fn, x0=[0.1, 0.2])
    # traj_xy : (tsteps+1, 2) numpy array — 2D positions only
"""

import numpy as np
import jax
import jax.numpy as jnp

# ── LQR import ────────────────────────────────────────────────────────────────

try:
    from lqrax import LQR
except ImportError as e:
    raise ImportError(
        "lqrax is required: pip install lqrax\n"
        "GitHub: https://github.com/MaxMSun/lqrax"
    ) from e

# ── Point-mass dynamics ───────────────────────────────────────────────────────

class PointMassLQR(LQR):
    """Second-order point-mass dynamics: state = [px, py, vx, vy]."""
    def __init__(self, dt, x_dim=4, u_dim=2, Q=None, R=None):
        if Q is None:
            Q = jnp.diag(jnp.array([1.0, 1.0, 0.001, 0.001]))
        if R is None:
            R = jnp.diag(jnp.array([0.01, 0.01]))
        super().__init__(dt, x_dim, u_dim, Q, R)

    def dyn(self, xt, ut):
        return jnp.array([xt[2], xt[3], ut[0], ut[1]])


def _build_lqr(dt=0.05):
    """Instantiate and JIT-compile the LQR solver (CPU is faster for this)."""
    cpu = jax.devices('cpu')[0]
    pm = PointMassLQR(dt=dt)
    linearize_dyn = jax.jit(pm.linearize_dyn, device=cpu)
    solve_lqr     = jax.jit(pm.solve,         device=cpu)
    return pm, linearize_dyn, solve_lqr


# ── Stein kernel ──────────────────────────────────────────────────────────────

def _kernel(x1, x2, h):
    """RBF kernel on position (first 2 dims of state)."""
    return jnp.exp(-jnp.sum(jnp.square(x1[:2] - x2[:2])) / h)

_d_kernel = jax.grad(_kernel, argnums=0)


def _make_stein_grad(score_fn, score_scale=1.0):
    """Build a JIT-compiled Stein gradient function for the given score."""
    def stein_grad_unit(x1, x2, h):
        return _kernel(x2, x1, h) * (score_scale * score_fn(x2)) + _d_kernel(x2, x1, h)

    def stein_grad_state(x, x_traj, h):
        vals = jax.vmap(stein_grad_unit, in_axes=(None, 0, None))(x, x_traj, h)
        return jnp.mean(vals, axis=0)

    def stein_grad(traj, h):
        return jax.vmap(stein_grad_state, in_axes=(0, None, None))(traj, traj, h)

    # GPU is faster for the Stein gradient; fall back to CPU if not available
    try:
        target = jax.devices('cuda')[0]
    except Exception:
        target = jax.devices('cpu')[0]

    return jax.jit(stein_grad, device=target)


# ── Public API ────────────────────────────────────────────────────────────────

def run_ergodic_coverage(
    score_fn,
    x0        = (0.1, 0.2),
    shape_def = None,
    custom_p_traj = None,
    dt        = 0.05,
    tsteps    = 200,
    num_iters = 200,
    step_size = 0.01,
    h         = 0.01,
    Kp=10.0,
    Kd=5.0,
    score_scale=1.0,
    verbose   = False,
):
    """
    Run Stein variational flow matching ergodic coverage optimisation.

    Parameters
    ----------
    score_fn  : callable (4,) → (4,)
        Score function ∇ log p(x) from shape_library.make_pdf_and_score().
    x0        : tuple (px, py)
        2D start position in [0,1]^2.
    dt        : float
        Integration timestep.
    tsteps    : int
        Number of trajectory timesteps.
    num_iters : int
        Number of Stein flow optimisation iterations.
    step_size : float
        Gradient ascent step size.
    h         : float
        RBF kernel bandwidth.
    verbose   : bool
        Print iteration count.

    Returns
    -------
    traj_xy : np.ndarray, shape (tsteps+1, 2)
        Final optimised trajectory — 2D positions only.
    """
    T   = dt * tsteps
    if custom_p_traj is not None:
        p_traj = np.array(custom_p_traj)
        u_traj_np, v0 = _pid_track(p_traj, dt, tsteps)
        x0j = jnp.array([x0[0], x0[1], v0[0], v0[1]])
        u_traj = jnp.array(u_traj_np)
    elif shape_def is not None and ('means' in shape_def or shape_def.get('type') == 'analytical'):
        p_traj, u_traj_np, v0 = _generate_initial_trajectory(x0, shape_def, tsteps, dt)
        x0j = jnp.array([x0[0], x0[1], v0[0], v0[1]])
        u_traj = jnp.array(u_traj_np)
    else:
        x0j = jnp.array([
            x0[0], x0[1],
            2.0 * (0.5 - x0[0]) / T,
            2.0 * (0.5 - x0[1]) / T,
        ])
        u_traj = jnp.zeros((tsteps, 2))

    pm, linearize_dyn, solve_lqr = _build_lqr(dt)
    stein_grad_jit = _make_stein_grad(score_fn, score_scale)
    z0 = jnp.zeros(4)

    init_traj = np.array(pm.traj_sim(x0j, u_traj))[:, :2]

    if verbose:
        from tqdm import tqdm
        itr = tqdm(range(num_iters), desc='  Ergodic opt', unit='it', position=1, leave=False)
    else:
        itr = range(num_iters)

    for _ in itr:
        x_traj, A_traj, B_traj = linearize_dyn(x0j, u_traj)
        stein_dx               = stein_grad_jit(x_traj, h=h)
        v_traj, _              = solve_lqr(z0, A_traj, B_traj, stein_dx)
        u_traj                += step_size * v_traj

    final_traj = pm.traj_sim(x0j, u_traj)
    return np.array(final_traj)[:, :2], init_traj


def _downsample_or_pad(points, tsteps):
    from scipy.ndimage import gaussian_filter1d
    idx_eval = np.linspace(0, len(points)-1, tsteps + 1)
    p_traj = np.column_stack([
        np.interp(idx_eval, np.arange(len(points)), points[:, 0]),
        np.interp(idx_eval, np.arange(len(points)), points[:, 1])
    ])
    p_traj = gaussian_filter1d(p_traj, sigma=2, axis=0, mode='nearest')
    return p_traj, *_pid_track(p_traj, 0.05, tsteps)


def _generate_initial_trajectory(x0, shape_def, tsteps, dt):
    """
    Generate an initial path covering the target shape.
    Supports standard GMMs or 'analytical' segment-based shapes.
    """
    if shape_def.get('type') == 'analytical':
        segments = shape_def['segments']
        unvisited = list(segments)
        curr_pt = np.array(x0)
        all_points = [np.array([x0])]
        
        while unvisited:
            best_dist = float('inf')
            best_idx = -1
            best_reverse = False
            
            for i, (p1, p2) in enumerate(unvisited):
                d1 = np.linalg.norm(curr_pt - np.array(p1))
                d2 = np.linalg.norm(curr_pt - np.array(p2))
                if d1 < best_dist:
                    best_dist = d1
                    best_idx = i
                    best_reverse = False
                if d2 < best_dist:
                    best_dist = d2
                    best_idx = i
                    best_reverse = True
                    
            seg = unvisited.pop(best_idx)
            p_start = np.array(seg[1] if best_reverse else seg[0])
            p_end = np.array(seg[0] if best_reverse else seg[1])
            
            dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
            L = np.hypot(dx, dy)
            if L < 1e-4:
                continue
                
            mu = (p_start + p_end) / 2
            E1 = np.array([dx, dy]) / L * (L/2)
            E2 = np.array([-dy, dx]) / L * 0.025
            
            num_swings = float(max(1, int(np.round(1.0 + 2.0 * (L / 0.7)))))
            n_pts = max(10, int(L * 200))
            tau = np.linspace(-1, 1, n_pts)
            
            curve = mu[None, :] + np.outer(tau, E1) + 0.3 * np.outer(np.sin(num_swings * np.pi * (tau + 1)), E2)
            
            dist_to_start = np.linalg.norm(p_start - curr_pt)
            if dist_to_start > 1e-3:
                transit_pts = max(5, int(dist_to_start * 100))
                transit = np.linspace(curr_pt, p_start, transit_pts)
                all_points.append(transit)
                
            all_points.append(curve)
            curr_pt = curve[-1]
            
        combined = np.vstack(all_points)
        return _downsample_or_pad(combined, tsteps)

    else:
        means = np.array(shape_def['means'])
        covs = np.array(shape_def['covs'])
        weights = np.array(shape_def['weights'])
    
    from scipy.ndimage import gaussian_filter1d
    weights = weights / np.sum(weights)
    max_w = np.max(weights) if len(weights) > 0 else 1.0
    
    # 1. Greedy TSP to order the means
    unvisited = list(range(len(means)))
    path_indices = []
    
    # Find the closest mean to x0
    dists_to_x0 = [np.linalg.norm(np.array(x0) - means[i]) for i in unvisited]
    curr_idx = unvisited.pop(np.argmin(dists_to_x0))
    path_indices.append(curr_idx)
    
    while unvisited:
        curr_mean = means[curr_idx]
        dists = [np.linalg.norm(curr_mean - means[idx]) for idx in unvisited]
        best = np.argmin(dists)
        curr_idx = unvisited.pop(best)
        path_indices.append(curr_idx)
    
    # 2. Build continuous path mapping Lissajous curves for each component
    points = []
    
    # Start with a transit from x0 to the first curve
    points.append(np.array([x0]))
    
    for idx in path_indices:
        mu = means[idx]
        cov = covs[idx]
        w = weights[idx]
        rel_w = w / max_w
        num_swings = 1.0 + 1.0 * rel_w # Maps to [1.0, 2.0] gentle swings based on relative density
        
        # eigen decomposition to find principal axes
        vals, vecs = np.linalg.eigh(cov)
        E1 = vecs[:, 1] * np.sqrt(vals[1])
        E2 = vecs[:, 0] * np.sqrt(vals[0])
        
        # Number of points proportional to weight (min 10)
        n_pts = max(10, int(w * 500))
        tau = np.linspace(-1, 1, n_pts)
        
        # Dynamic Serpentine: progress along Major Axis (E1), oscillate gently along Minor Axis (E2)
        curve = mu[None, :] + 1.0 * np.outer(tau, E1) + 0.3 * np.outer(np.sin(num_swings * np.pi * (tau + 1)), E2)
        
        # Anfahrt vom letzten Punkt. Die Zahl der Stuetzpunkte haengt an der
        # *Laenge* der Anfahrt, nicht an der Zahl der Komponenten. Mit einem
        # Startpunkt irgendwo auf der Flaeche kann die erste Anfahrt fast die
        # ganze Diagonale lang sein; mit einer festen, kleinen Punktzahl wuerde
        # sie nach der Interpolation auf `tsteps` zu wenige Abtastpunkte
        # bekommen und der PID-Regler muesste sie mit unrealistischer
        # Geschwindigkeit abfahren.
        last_pt = points[-1][-1]
        dist = float(np.linalg.norm(np.asarray(curve[0]) - np.asarray(last_pt)))
        n_transit = max(5, int(round(dist * 120)), int(20 / len(means)))
        transit = np.linspace(last_pt, curve[0], n_transit)[1:-1]
        points.append(transit)
        points.append(curve)

    points = np.vstack(points)
    
    # 3. Interpolate to exact tsteps + 1
    idx_eval = np.linspace(0, len(points)-1, tsteps + 1)
    p_traj = np.column_stack([
        np.interp(idx_eval, np.arange(len(points)), points[:, 0]),
        np.interp(idx_eval, np.arange(len(points)), points[:, 1])
    ])
    
    # 4. Smooth the path to avoid infinite acceleration at corners/transits
    p_traj = gaussian_filter1d(p_traj, sigma=3, axis=0, mode='nearest')
    p_traj[0] = x0
    
    # 5. PID Tracking
    u_traj, v0 = _pid_track(p_traj, dt, tsteps)
        
    return p_traj, u_traj, v0


def _pid_track(p_traj, dt, tsteps):
    """PID Tracking of a smooth path using exact discrete double integrator."""
    import numpy as np
    T = tsteps
    u_traj = np.zeros((T, 2))
    p = p_traj[0].copy()
    v = (p_traj[1] - p_traj[0]) / dt
    v0 = v.copy()
    
    Kp = 150.0
    Kd = 20.0
    
    for t in range(T):
        p_target = p_traj[t+1]
        v_target = (p_traj[min(t+2, T)] - p_traj[t]) / (2*dt) if t < T-1 else (p_traj[T] - p_traj[T-1]) / dt
        
        # PD control law
        u_traj[t] = Kp * (p_target - p) + Kd * (v_target - v)
        
        # Update state using exact integration matching LQR (PointMassLQR)
        p = p + v * dt + 0.5 * u_traj[t] * (dt**2)
        v = v + u_traj[t] * dt
        
    return u_traj, v0
