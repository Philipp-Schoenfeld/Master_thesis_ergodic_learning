import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def encode(density_map, n_components=15, n_samples=5000):
    """Encode a continuous density map into a GMM by sampling."""
    flat_density = density_map.flatten()
    flat_density /= flat_density.sum()
    
    H, W = density_map.shape
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    # Map back to [0, 1]
    x_flat = x_coords.flatten() / (W - 1)
    y_flat = y_coords.flatten() / (H - 1)
    
    indices = np.random.choice(len(flat_density), size=n_samples, p=flat_density)
    samples = np.vstack((x_flat[indices], y_flat[indices])).T
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(samples)
    return gmm

def viz_encoded(ax, gmm):
    """Visualize the GMM components as ellipses."""
    # Plot a faint background grid
    ax.set_facecolor('#f0f0f0')
    max_w = np.max(gmm.weights_)
    for mean, covar, weight in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0]) * 180.0 / np.pi
        
        # Scale alpha relative to the highest weight
        alpha = np.clip((weight / max_w) * 0.8 + 0.2, 0.1, 0.9)
        ell = Ellipse(mean, v[0]*2, v[1]*2, angle=180.0+angle, color='crimson', alpha=alpha)
        ax.add_patch(ell)
        ax.scatter(mean[0], mean[1], c='black', s=5, zorder=5)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("GMM (15 Components)", fontsize=10)

def revert(gmm, resolution=256):
    """Revert GMM back to a continuous density map."""
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    # score_samples returns log prob
    log_prob = gmm.score_samples(pos.reshape(-1, 2))
    reverted_map = np.exp(log_prob).reshape(resolution, resolution)
    # Normalize for visualization
    reverted_map /= reverted_map.max()
    return reverted_map
