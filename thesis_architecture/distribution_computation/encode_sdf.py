import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

def encode(density_map, level_ratio=0.2):
    """Encode a density map into Analytical SDF segments (Polygons)."""
    # Find contours at a specific level
    level = np.max(density_map) * level_ratio
    contours = find_contours(density_map, level)
    
    H, W = density_map.shape
    segments = []
    
    for contour in contours:
        # contour is (N, 2) array of (y, x) indices
        # Convert to (x, y) in [0, 1]
        contour_x = contour[:, 1] / (W - 1)
        contour_y = contour[:, 0] / (H - 1)
        pts = np.column_stack((contour_x, contour_y))
        
        # Create line segments between consecutive points
        for i in range(len(pts) - 1):
            segments.append((pts[i], pts[i+1]))
            
    return np.array(segments) # shape (N_segments, 2, 2)

def viz_encoded(ax, segments):
    """Visualize the extracted analytical polygons."""
    for seg in segments:
        ax.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]], color='crimson', lw=1.5)
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Analytical Segments ({len(segments)})", fontsize=10)

def _dist_to_segment_sq(p, a, b):
    # p is (N, 2), a is (2,), b is (2,)
    ab = b - a
    l2 = np.sum(ab**2)
    if l2 == 0:
        return np.sum((p - a)**2, axis=1)
    
    # Dot product of (p - a) and ab
    t = np.sum((p - a) * ab, axis=1) / l2
    t = np.clip(t, 0.0, 1.0)
    
    # Projection
    proj = a + t[:, np.newaxis] * ab
    
    return np.sum((p - proj)**2, axis=1)

def revert(segments, resolution=256, sigma=0.025):
    """Revert segments back to a continuous density via exponential SDF."""
    if len(segments) == 0:
        return np.zeros((resolution, resolution))
        
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack((X.ravel(), Y.ravel()))
    
    # Calculate min distance to any segment for all points
    min_dist_sq = np.full(len(pts), np.inf)
    
    for seg in segments:
        d2 = _dist_to_segment_sq(pts, seg[0], seg[1])
        min_dist_sq = np.minimum(min_dist_sq, d2)
        
    # Convert distance to density
    reverted_map = np.exp(-min_dist_sq / (2 * sigma**2)).reshape(resolution, resolution)
    
    if reverted_map.max() > 0:
        reverted_map /= reverted_map.max()
        
    return reverted_map
