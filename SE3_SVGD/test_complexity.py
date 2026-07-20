import numpy as np

def _dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (len_sq + 1e-12), 0, 1)
    return np.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)

STROKE_WIDTH = 0.045
grid_res = 200
_xs = np.linspace(0, 1, grid_res)
_ys = np.linspace(0, 1, grid_res)
Xg, Yg = np.meshgrid(_xs, _ys)
_grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)

K_FOURIER = 10
k_indices = np.array([[k1, k2] for k1 in range(K_FOURIER) for k2 in range(K_FOURIER)])

def fourier_basis(pts):
    args = np.pi * pts[:, None, :] * k_indices[None, :, :]
    return np.prod(np.cos(args), axis=-1)

shapes = {
    'N': [([0.25, 0.15], [0.25, 0.85]), ([0.25, 0.85], [0.75, 0.15]), ([0.75, 0.15], [0.75, 0.85])],
    'H': [([0.25, 0.15], [0.25, 0.85]), ([0.75, 0.15], [0.75, 0.85]), ([0.25, 0.50], [0.75, 0.50])],
    'II': [([0.25, 0.15], [0.25, 0.85]), ([0.75, 0.15], [0.75, 0.85])]
}

for name, segments in shapes.items():
    d_min = np.full_like(Xg, 1e10)
    for (ax, ay), (bx, by) in segments:
        d_min = np.minimum(d_min, _dist_to_segment(Xg, Yg, ax, ay, bx, by))
    Zg = np.exp(-d_min ** 2 / (2 * STROKE_WIDTH ** 2))
    
    _grid_w = Zg.ravel()
    _grid_w = _grid_w / _grid_w.sum()
    phi_k = np.sum(_grid_w[:, None] * fourier_basis(_grid_pts), axis=0)
    
    freq_mags = np.linalg.norm(k_indices, axis=1)
    mask = freq_mags > 0
    
    # Metric 1: spectral complexity
    spectral_complexity = np.sum(freq_mags[mask] * np.abs(phi_k[mask])) / np.sum(np.abs(phi_k[mask]))
    
    # Metric 2: total arc length
    arc_length = sum(np.linalg.norm(np.array(b) - np.array(a)) for a, b in segments)
    
    # Metric 3: spatial entropy
    entropy = -np.sum(_grid_w[_grid_w > 1e-5] * np.log(_grid_w[_grid_w > 1e-5]))
    
    print(f"Shape: {name}")
    print(f"  Spectral Complexity: {spectral_complexity:.3f}")
    print(f"  Arc Length: {arc_length:.3f}")
    print(f"  Spatial Entropy: {entropy:.3f}")
