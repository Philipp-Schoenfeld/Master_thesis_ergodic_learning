import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def encode(density_map, target_res=16):
    """Encode a high-res density map into a low-res discrete grid."""
    H, W = density_map.shape
    zoom_factor = target_res / H
    # Downsample using local averaging (order=1 is bilinear)
    # To avoid aliasing, we can use order=3 (bicubic), but downsampling is best done 
    # via block mean or just spline interpolation.
    grid = zoom(density_map, zoom_factor, order=1)
    grid = np.clip(grid, 0, None)
    return grid

def viz_encoded(ax, grid):
    """Visualize the discrete grid as a pixelated heatmap."""
    ax.imshow(grid, origin='lower', extent=[0, 1, 0, 1], cmap='Blues', interpolation='nearest')
    
    # Draw faint grid lines to emphasize the discrete nature
    res = grid.shape[0]
    for i in range(res + 1):
        ax.axhline(i / res, color='black', lw=0.5, alpha=0.2)
        ax.axvline(i / res, color='black', lw=0.5, alpha=0.2)
        
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Discrete Grid ({res}x{res})", fontsize=10)

def revert(grid, resolution=256):
    """Revert the low-res grid back to high-res via bicubic interpolation."""
    current_res = grid.shape[0]
    zoom_factor = resolution / current_res
    
    # order=3 means bicubic interpolation, which gives smooth continuous results
    reverted_map = zoom(grid, zoom_factor, order=3)
    reverted_map = np.clip(reverted_map, 0, None)
    
    if reverted_map.max() > 0:
        reverted_map /= reverted_map.max()
        
    return reverted_map
