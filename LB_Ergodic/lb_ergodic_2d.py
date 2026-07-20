#!/usr/bin/env python3
"""
2D Laplace-Beltrami Ergodic Coverage — Letter "N" target

Implements "Ergodic Exploration over Meshable Surfaces" (Dong et al.,
ICRA 2025) on a 2D Delaunay mesh of the unit square.

Instead of analytical Fourier basis functions, the ergodic metric is
built from the first K eigenvectors of the discrete Laplace-Beltrami
operator on the mesh.  Trajectory controls are optimised via L-BFGS-B.

Uses the same five initialisation strategies and comparison-grid
visualisation as the SE3_SVGD, Stein_Flow_matching, and HEDAC testbenches.
"""

import time
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.sparse import csc_matrix, diags, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize

sys.path.append("/home/philipp/Documents/Uni/Master_thesis")
from init_strategies import get_initialization

np.random.seed(42)

# ============================================================================
# 1. Target Distribution
# ============================================================================

TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')

if TARGET_SHAPE == 'N':
    N_SEGMENTS = [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.25, 0.85], [0.75, 0.15]),
        ([0.75, 0.15], [0.75, 0.85]),
    ]
elif TARGET_SHAPE == 'H':
    N_SEGMENTS = [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
        ([0.25, 0.50], [0.75, 0.50]),
    ]
elif TARGET_SHAPE == 'II':
    N_SEGMENTS = [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.75, 0.15], [0.75, 0.85]),
    ]
else:
    raise ValueError(f"Unknown target shape: {TARGET_SHAPE}")
STROKE_WIDTH = 0.045


def _dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (len_sq + 1e-12), 0, 1)
    return np.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)


def target_distribution(x, y):
    d_min = np.full_like(x, 1e10)
    for (ax, ay), (bx, by) in N_SEGMENTS:
        d_min = np.minimum(d_min, _dist_to_segment(x, y, ax, ay, bx, by))
    return np.exp(-d_min ** 2 / (2 * STROKE_WIDTH ** 2))


# High-res visualisation grid
grid_res = 200
_xs_vis = np.linspace(0, 1, grid_res)
_ys_vis = np.linspace(0, 1, grid_res)
Xg, Yg = np.meshgrid(_xs_vis, _ys_vis)
Zg = target_distribution(Xg, Yg)

# ============================================================================
# 2. Fourier Ergodic Metric (for cross-method comparison)
# ============================================================================

K_FOURIER = 10
k_indices = np.array([[k1, k2] for k1 in range(K_FOURIER)
                       for k2 in range(K_FOURIER)])
Lambda_k_fourier = (1.0 + np.sum(k_indices ** 2, axis=1)) ** (-1.5)


def fourier_basis(pts):
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)


_grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
_grid_w = Zg.ravel()
_grid_w = _grid_w / _grid_w.sum()
phi_k_fourier = np.sum(_grid_w[:, None] * fourier_basis(_grid_pts), axis=0)

W_ERGODIC = 600.0
W_SMOOTH = 15.0
W_BOUNDARY = 30.0

USE_OBSTACLE = False
OBSTACLE_CENTER = [0.5, 0.5]
OBSTACLE_RADIUS = 0.12
W_OBSTACLE = 50000.0

def compute_fourier_energy(traj_flat, T):
    """Same Fourier energy metric used by tsvec_2d / flow_matching_2d."""
    X = traj_flat.reshape(T, 2)
    energy = 0.0
    accel = X[2:] - 2 * X[1:-1] + X[:-2]
    energy += W_SMOOTH * np.sum(accel ** 2)
    Fk = fourier_basis(X)
    c_k = np.mean(Fk, axis=0)
    diff_k = c_k - phi_k_fourier
    energy += W_ERGODIC * 0.5 * np.sum(Lambda_k_fourier * diff_k ** 2)
    margin = 0.03
    lo = np.minimum(X - margin, 0.0)
    hi = np.maximum(X - (1.0 - margin), 0.0)
    energy += W_BOUNDARY * 0.5 * (np.sum(lo ** 2) + np.sum(hi ** 2))
    
    if USE_OBSTACLE:
        dx = X[:, 0] - OBSTACLE_CENTER[0]
        dy = X[:, 1] - OBSTACLE_CENTER[1]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        violation = np.maximum(OBSTACLE_RADIUS - dist, 0.0)
        energy += W_OBSTACLE * 0.5 * np.sum(violation**2)
        
    return energy

# ============================================================================
# 3. Hyperparameters
# ============================================================================

N_PARTICLES = 10
T = 100               # trajectory time-steps
dt = 0.01             # integration step
K_LB = 20             # number of LB eigenvectors
MESH_RES = 35         # points per side for the mesh
SIGMA_SENSOR = 0.05   # Gaussian sensor std
W_CONTROL = 0.0005    # control effort regularisation
MAX_LBFGS_ITER = 120  # L-BFGS-B iterations per trajectory
INIT_NOISE_STD = 0.02

# ============================================================================
# 4. Mesh Generation & Laplace-Beltrami Operator
# ============================================================================


def generate_mesh(res):
    """Create a Delaunay mesh of [0,1]² with slight interior jitter."""
    xs = np.linspace(0, 1, res)
    ys = np.linspace(0, 1, res)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Small jitter on interior points for mesh quality
    interior = ((pts[:, 0] > 0) & (pts[:, 0] < 1) &
                (pts[:, 1] > 0) & (pts[:, 1] < 1))
    pts[interior] += np.random.uniform(-0.3 / res, 0.3 / res,
                                        size=(interior.sum(), 2))
    pts = np.clip(pts, 0, 1)
    tri = Delaunay(pts)
    return pts, tri.simplices


def build_laplacian_and_mass(verts, tris):
    """Build the cotangent stiffness matrix L (positive semi-definite)
    and lumped mass matrix M for the 2D triangle mesh."""
    m = len(verts)
    rows, cols, vals = [], [], []
    mass = np.zeros(m)

    for tri in tris:
        v = verts[tri]          # (3, 2)
        for loc in range(3):
            i = tri[loc]
            j = tri[(loc + 1) % 3]
            k = tri[(loc + 2) % 3]
            # Angle at vertex i → opposite to edge (j, k)
            eij = v[(loc + 1) % 3] - v[loc]
            eik = v[(loc + 2) % 3] - v[loc]
            cos_a = np.dot(eij, eik)
            sin_a = abs(np.cross(eij, eik)) + 1e-12
            cot_a = cos_a / sin_a
            w = 0.5 * cot_a
            # Off-diagonal (negative) and diagonal (positive)
            rows.extend([j, k, j, k])
            cols.extend([k, j, j, k])
            vals.extend([-w, -w, w, w])
        # Lumped mass: area / 3 per vertex
        area = 0.5 * abs(np.cross(v[1] - v[0], v[2] - v[0]))
        for vi in tri:
            mass[vi] += area / 3.0

    L = csc_matrix(coo_matrix((vals, (rows, cols)), shape=(m, m)))
    M = diags(mass)
    return L, M, mass


print("Building mesh and Laplace-Beltrami operator …")
mesh_verts, mesh_tris = generate_mesh(MESH_RES)
L_mat, M_mat, mass_vec = build_laplacian_and_mass(mesh_verts, mesh_tris)
n_verts = len(mesh_verts)

# ============================================================================
# 5. Eigen-decomposition
# ============================================================================

print(f"  Mesh: {n_verts} vertices, {len(mesh_tris)} triangles")
print(f"  Solving for {K_LB} LB eigenpairs …")
eigenvalues, eigenvectors = eigsh(L_mat, k=K_LB, M=M_mat, sigma=0,
                                   which='LM')
# Sort by eigenvalue (ascending)
order = np.argsort(eigenvalues)
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Discount factor  Λ_k = exp(-0.1 √λ_k)
Lambda_k_lb = np.exp(-0.1 * np.sqrt(np.abs(eigenvalues)))

# ============================================================================
# 6. Project Information Map onto LB Basis
# ============================================================================

phi_mesh = target_distribution(mesh_verts[:, 0], mesh_verts[:, 1])
# Normalise so ∫φ dA ≈ φ^T M 1 = 1
phi_mesh /= (mass_vec * phi_mesh).sum() + 1e-12

# φ_k = f_k^T M φ
phi_k_lb = eigenvectors.T @ (M_mat @ phi_mesh)

print(f"  Eigenvalues: {eigenvalues[:5].round(3)} …")
print(f"  φ_k (first 5): {phi_k_lb[:5].round(5)}")

# ============================================================================
# 7. Ergodic Objective (LB metric) with Analytical Gradient
# ============================================================================


def ergodic_cost_and_grad(u_flat, x0):
    """Compute the LB ergodic metric and its gradient w.r.t. controls.

    Decision variable:  u_flat ∈ ℝ^{2T}  (flattened T×2 controls)
    Dynamics:           x[t+1] = x[t] + dt·u[t]  (single integrator)
    Sensor:             s_t(w) = exp(−‖w−x[t]‖²/(2σ²))
    Statistics:         μ(w) = (1/(T+1)) Σ_t s_t(w),  normalised
    Metric:             E = Σ_k Λ_k (μ_k − φ_k)²
    """
    u = u_flat.reshape(T, 2)

    # ----- Forward simulate trajectory -----
    x = np.zeros((T + 1, 2))
    x[0] = x0
    for t in range(T):
        x[t + 1] = x[t] + dt * u[t]

    # ----- Sensor readings at all mesh vertices -----
    # diff[t, j, :] = mesh_verts[j] - x[t]   shape (T+1, m, 2)
    diff = mesh_verts[None, :, :] - x[:, None, :]        # (T+1, m, 2)
    dist_sq = np.sum(diff ** 2, axis=-1)                  # (T+1, m)
    S = np.exp(-dist_sq / (2.0 * SIGMA_SENSOR ** 2))      # (T+1, m)

    # ----- Time-averaged statistics -----
    mu_raw = np.mean(S, axis=0)                            # (m,)
    mu_int = (mass_vec * mu_raw).sum()
    mu_norm = mu_raw / (mu_int + 1e-12)                    # normalised

    # ----- Project onto LB basis -----
    mu_k = eigenvectors.T @ (M_mat @ mu_norm)              # (K,)

    # ----- Ergodic cost -----
    diff_k = mu_k - phi_k_lb
    cost = np.sum(Lambda_k_lb * diff_k ** 2)

    # ----- Control regularisation -----
    cost += W_CONTROL * np.sum(u ** 2)

    # ----- Boundary penalty -----
    margin = 0.02
    lo = np.minimum(x - margin, 0.0)
    hi = np.maximum(x - (1.0 - margin), 0.0)
    cost += 50.0 * (np.sum(lo ** 2) + np.sum(hi ** 2))

    # ======================= GRADIENT =======================
    # ∂E/∂μ_k = 2 Λ_k (μ_k − φ_k)
    dE_dmu_k = 2.0 * Lambda_k_lb * diff_k                 # (K,)

    # ∂E/∂μ_norm[j] = Σ_k dE_dmu_k[k] · M[j,j] · f_k[j]
    dE_dmu_norm = mass_vec * (eigenvectors @ dE_dmu_k)     # (m,)

    # μ_norm = μ_raw / mu_int  →  ∂μ_norm/∂μ_raw
    # Simplified (ignoring normalisation-through-grad for speed):
    dE_dmu_raw = dE_dmu_norm / (mu_int + 1e-12)            # (m,)

    # ∂μ_raw/∂x[t] = (1/(T+1)) · S[t,j]/σ² · diff[t,j]
    # g[t] = Σ_j dE_dmu_raw[j] · (1/(T+1)) · S[t,j]/σ² · diff[t,j]
    weighted_S = (dE_dmu_raw[None, :] * S /
                  ((T + 1) * SIGMA_SENSOR ** 2))            # (T+1, m)
    g = np.einsum('tj,tjd->td', weighted_S, diff)          # (T+1, 2)

    # Boundary gradient
    g += 100.0 * (lo + hi)                                  # (T+1, 2)

    # Obstacle gradient
    if USE_OBSTACLE:
        dx = x[:, 0] - OBSTACLE_CENTER[0]
        dy = x[:, 1] - OBSTACLE_CENTER[1]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        violation = np.maximum(OBSTACLE_RADIUS - dist, 0.0)
        
        cost += W_OBSTACLE * 0.5 * np.sum(violation**2)
        g[:, 0] += W_OBSTACLE * violation * (-dx / dist)
        g[:, 1] += W_OBSTACLE * violation * (-dy / dist)

    # ∂x[t]/∂u[s] = dt for s < t  →  grad_u = dt · reverse_cumsum(g)[1:]
    g_rev_cumsum = np.cumsum(g[::-1], axis=0)[::-1]        # (T+1, 2)
    grad_u = dt * g_rev_cumsum[1:]                          # (T, 2)

    # Control regularisation gradient
    grad_u += 2.0 * W_CONTROL * u

    return cost, grad_u.ravel()

# ============================================================================
# 8. Trajectory Optimiser
# ============================================================================


def optimise_trajectory(x0, u_init):
    """Run L-BFGS-B to minimise the LB ergodic metric."""
    res = minimize(
        ergodic_cost_and_grad, u_init.ravel(), args=(x0,),
        method='L-BFGS-B', jac=True,
        options={'maxiter': MAX_LBFGS_ITER, 'ftol': 1e-9, 'gtol': 1e-6}
    )
    u_opt = res.x.reshape(T, 2)
    # Simulate final trajectory
    traj = np.zeros((T + 1, 2))
    traj[0] = x0
    for t in range(T):
        traj[t + 1] = traj[t] + dt * u_opt[t]
    traj = np.clip(traj, 0, 1)
    # Sub-sample to T points (drop x0, keep T waypoints)
    return traj[1:]

# ============================================================================
# 9. Run Over Initialisation Strategies
# ============================================================================

def run_benchmark(out_dir: str, save_npy: bool = False, use_obstacle: bool = False):
    global USE_OBSTACLE
    USE_OBSTACLE = use_obstacle
    
    os.makedirs(out_dir, exist_ok=True)
    strategies = ["linear", "n_shape", "polynomial", "rrt"]
    results = {}
    benchmark_data = {}

    print(f"\n2D LB-Ergodic  |  {N_PARTICLES} particles, T={T}, K_LB={K_LB}, "
          f"L-BFGS iters={MAX_LBFGS_ITER}")
    print(f"Mesh: {n_verts} verts, sensor σ={SIGMA_SENSOR}, "
          f"w_ctrl={W_CONTROL}")
    print("-" * 65)

    for strat in strategies:
        print(f"Running strategy: {strat}")
        t_start = time.time()

        init_p, base_t = get_initialization(strat, N_PARTICLES, T,
                                            noise_std=INIT_NOISE_STD)
        pos_trajs = init_p.reshape(N_PARTICLES, T, 2)

        final_trajs = np.zeros_like(pos_trajs)
        for i in range(N_PARTICLES):
            x0 = pos_trajs[i, 0]
            u_init = np.diff(pos_trajs[i], axis=0) / dt
            u_init = np.vstack([u_init, u_init[-1:]])
            u_init = np.clip(u_init, -5, 5)

            opt_traj = optimise_trajectory(x0, u_init)
            final_trajs[i] = opt_traj

        final_flat = final_trajs.reshape(N_PARTICLES, -1)

        energies = np.array([compute_fourier_energy(final_flat[j], T)
                             for j in range(N_PARTICLES)])
        best = int(np.argmin(energies))
        elapsed = time.time() - t_start
        print(f"  -> Best energy: {energies[best]:.3f} (Time: {elapsed:.2f}s)\n")

        results[strat] = {
            'initial': init_p,
            'base_traj': base_t,
            'final': final_flat,
            'best_idx': best,
        }
        
        benchmark_data[strat] = {
            'mean_cost': float(np.mean(energies)),
            'best_cost': float(energies[best]),
            'time_s': float(elapsed)
        }
        
        if save_npy:
            np.save(os.path.join(out_dir, f"{strat}_trajs.npy"), final_flat)

    fig, axes = plt.subplots(len(strategies), 2, figsize=(12, 5 * len(strategies)))
    cmap = 'YlOrRd'
    colors = plt.cm.rainbow(np.linspace(0, 1, N_PARTICLES))

    def plot_particles(ax, parts, title,
                       highlight_best=False, best_idx=-1, base_traj=None):
        ax.contourf(Xg, Yg, Zg, levels=30, cmap=cmap, alpha=0.6)
        ax.contour(Xg, Yg, Zg, levels=6, colors='k', linewidths=0.3, alpha=0.3)

        if USE_OBSTACLE:
            circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='gray', alpha=0.8, zorder=5)
            ax.add_patch(circle)

        if base_traj is not None:
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--',
                    color='white', lw=3.0, zorder=8)
            ax.plot(base_traj[:, 0], base_traj[:, 1], '--',
                    color='black', lw=1.5, zorder=9, label='Base Trajectory')

        for i in range(N_PARTICLES):
            tr = parts[i].reshape(T, 2)
            lw = 2.5 if (highlight_best and i == best_idx) else 0.8
            al = 1.0 if (highlight_best and i == best_idx) else 0.4
            ax.plot(tr[:, 0], tr[:, 1], '-', color=colors[i], lw=lw, alpha=al)
            ax.plot(tr[0, 0], tr[0, 1], 'o', color=colors[i], ms=4)

        if highlight_best and best_idx >= 0:
            best_tr = parts[best_idx].reshape(T, 2)
            ax.plot(best_tr[:, 0], best_tr[:, 1], '-', color=colors[best_idx],
                    lw=2.5, label=f'Best (#{best_idx})', zorder=10)
            ax.legend(loc='lower right', fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    for row, strat in enumerate(strategies):
        res = results[strat]
        ax = axes[row, 0]
        plot_particles(ax, res['initial'], f'[{strat}] Initial',
                       base_traj=res['base_traj'])
        if res['base_traj'] is not None:
            ax.legend(loc='lower right', fontsize=9)

        ax = axes[row, 1]
        plot_particles(ax, res['final'], f'[{strat}] LB-Ergodic Result',
                       highlight_best=True, best_idx=res['best_idx'])

    plt.tight_layout()
    out = os.path.join(out_dir, f'lb_ergodic_2d_{os.environ.get("TARGET_SHAPE", "N")}_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump({
            'T': T,
            'dt': dt,
            'MAX_LBFGS_ITER': MAX_LBFGS_ITER,
            'N_PARTICLES': N_PARTICLES
        }, f, indent=4)
        
    return benchmark_data

if __name__ == "__main__":
    TARGET_SHAPE = os.environ.get('TARGET_SHAPE', 'N')
    run_benchmark(out_dir=f'/home/philipp/Documents/Uni/Master_thesis/results/LB_Ergodic_{TARGET_SHAPE}')
