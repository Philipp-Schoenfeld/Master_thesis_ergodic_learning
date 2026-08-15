import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import encode_spectral
import encode_grid

def encode(density_map, k_max=8, target_res=16):
    """Encode density into Hybrid representation (Low-freq Spectral + Low-res Grid)."""
    c_trunc = encode_spectral.encode(density_map, k_max=k_max)
    grid = encode_grid.encode(density_map, target_res=target_res)
    return (c_trunc, grid)

def viz_encoded(ax, encoded_tuple):
    """Visualize Hybrid (Spatial grid as main, Spectral as inset)."""
    c_trunc, grid = encoded_tuple
    
    # Plot grid as main
    ax.imshow(grid, origin='lower', extent=[0, 1, 0, 1], cmap='Blues', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Hybrid (Grid + Spec)", fontsize=10)
    
    # Create inset for spectral
    axins = inset_axes(ax, width="40%", height="40%", loc='upper right')
    c_max = np.max(np.abs(c_trunc))
    axins.imshow(c_trunc, cmap='coolwarm', vmin=-c_max, vmax=c_max, origin='upper')
    axins.set_xticks([])
    axins.set_yticks([])
    # White border around inset to make it stand out
    for spine in axins.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

def revert(encoded_tuple, resolution=256):
    """Revert hybrid representation by averaging the spatial and spectral reconstructions."""
    c_trunc, grid = encoded_tuple
    
    reverted_spectral = encode_spectral.revert(c_trunc, resolution=resolution)
    reverted_grid = encode_grid.revert(grid, resolution=resolution)
    
    # Average the two reconstructions
    reverted_map = (reverted_spectral + reverted_grid) / 2.0
    
    if reverted_map.max() > 0:
        reverted_map /= reverted_map.max()
        
    return reverted_map
