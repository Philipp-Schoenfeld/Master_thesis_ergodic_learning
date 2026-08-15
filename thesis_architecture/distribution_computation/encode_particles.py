import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def encode(density_map, n_particles=500):
    """Encode a density map into a discrete set of particles (Point Cloud)."""
    flat_density = density_map.flatten()
    if flat_density.sum() == 0:
        flat_density += 1e-8
    flat_density /= flat_density.sum()
    
    H, W = density_map.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    # Map back to [0, 1]
    x_flat = x_coords.flatten() / (W - 1)
    y_flat = y_coords.flatten() / (H - 1)
    
    indices = np.random.choice(len(flat_density), size=n_particles, p=flat_density)
    
    # Add a tiny bit of uniform noise to de-grid the sampled positions
    noise_x = np.random.uniform(-0.5/(W-1), 0.5/(W-1), size=n_particles)
    noise_y = np.random.uniform(-0.5/(H-1), 0.5/(H-1), size=n_particles)
    
    particles = np.vstack((x_flat[indices] + noise_x, y_flat[indices] + noise_y)).T
    # Clip to bounds
    particles = np.clip(particles, 0, 1)
    
    return particles

def viz_encoded(ax, particles):
    """Visualize the point cloud as a scatter plot."""
    ax.scatter(particles[:, 0], particles[:, 1], s=4, c='crimson', alpha=0.6, edgecolors='none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Point Cloud (N={len(particles)})", fontsize=10)

def revert(particles, resolution=256):
    """Revert particles back to a continuous density map using KDE."""
    # Fit Kernel Density Estimator
    kde = gaussian_kde(particles.T, bw_method='scott')
    
    # Evaluate on a grid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    reverted_map = np.reshape(kde(positions).T, X.shape)
    
    # Normalize for visualization
    if reverted_map.max() > 0:
        reverted_map /= reverted_map.max()
        
    return reverted_map
