import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

def encode(density_map, k_max=10):
    """Encode a density map into Spectral Cosine Coefficients."""
    # Compute 2D Discrete Cosine Transform
    # norm='ortho' ensures proper scaling for inverse transform
    c_all = dctn(density_map, norm='ortho')
    
    # Truncate to the first k_max x k_max frequencies (low-pass filter)
    c_trunc = c_all[:k_max, :k_max]
    return c_trunc

def viz_encoded(ax, c_trunc):
    """Visualize the spectral coefficients as a heatmap."""
    # Plot the 2D array of coefficients
    # We use a diverging colormap because coefficients can be negative
    c_max = np.max(np.abs(c_trunc))
    im = ax.imshow(c_trunc, cmap='coolwarm', vmin=-c_max, vmax=c_max, origin='upper')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Spectral Coeffs ({c_trunc.shape[0]}x{c_trunc.shape[1]})", fontsize=10)

def revert(c_trunc, resolution=256):
    """Revert spectral coefficients back to a spatial density map."""
    # Pad the truncated coefficients back to the full resolution with zeros
    c_padded = np.zeros((resolution, resolution))
    k_y, k_x = c_trunc.shape
    c_padded[:k_y, :k_x] = c_trunc
    
    # Inverse 2D DCT
    reverted_map = idctn(c_padded, norm='ortho')
    
    # The inverse transform might have tiny negative ringing artifacts, clip them
    reverted_map = np.clip(reverted_map, 0, None)
    
    # Normalize for visualization
    if reverted_map.max() > 0:
        reverted_map /= reverted_map.max()
        
    return reverted_map
