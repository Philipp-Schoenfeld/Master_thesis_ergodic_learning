import numpy as np
import matplotlib.pyplot as plt
import os

STROKE_WIDTH = 0.025

# Helpers for curves
def make_circle_segments(cx, cy, r, n=16):
    t = np.linspace(0, 2*np.pi, n+1)
    pts = np.column_stack([cx + r*np.cos(t), cy + r*np.sin(t)])
    return [(pts[i].tolist(), pts[i+1].tolist()) for i in range(n)]

SHAPES = {
    'A': [
        ([0.5, 0.85], [0.25, 0.15]),
        ([0.5, 0.85], [0.75, 0.15]),
        ([0.35, 0.45], [0.65, 0.45]),
    ],
    'G': [
        ([0.7, 0.75], [0.5, 0.85]),
        ([0.5, 0.85], [0.25, 0.7]),
        ([0.25, 0.7], [0.25, 0.3]),
        ([0.25, 0.3], [0.5, 0.15]),
        ([0.5, 0.15], [0.75, 0.3]),
        ([0.75, 0.3], [0.75, 0.5]),
        ([0.55, 0.5], [0.75, 0.5]),
    ],
    'W': [
        ([0.15, 0.85], [0.35, 0.15]),
        ([0.35, 0.15], [0.50, 0.50]),
        ([0.50, 0.50], [0.65, 0.15]),
        ([0.65, 0.15], [0.85, 0.85]),
    ],
    'phi': [
        ([0.5, 0.85], [0.5, 0.15])
    ] + make_circle_segments(0.5, 0.5, 0.2, 16),
    'sigma': make_circle_segments(0.5, 0.4, 0.25, 16) + [
        ([0.5, 0.65], [0.8, 0.65])
    ],
    'Ö': make_circle_segments(0.5, 0.4, 0.25, 16) + [
        ([0.35, 0.8], [0.35, 0.82]),
        ([0.65, 0.8], [0.65, 0.82])
    ],
    'B': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.25, 0.85], [0.65, 0.85]),
        ([0.65, 0.85], [0.75, 0.7]),
        ([0.75, 0.7], [0.65, 0.5]),
        ([0.65, 0.5], [0.25, 0.5]),
        ([0.65, 0.5], [0.75, 0.32]),
        ([0.75, 0.32], [0.65, 0.15]),
        ([0.65, 0.15], [0.25, 0.15]),
    ],
    'R': [
        ([0.25, 0.15], [0.25, 0.85]),
        ([0.25, 0.85], [0.65, 0.85]),
        ([0.65, 0.85], [0.75, 0.7]),
        ([0.75, 0.7], [0.65, 0.5]),
        ([0.65, 0.5], [0.25, 0.5]),
        ([0.55, 0.5], [0.75, 0.15]),
    ],
    'psi': [
        ([0.5, 0.15], [0.5, 0.85]),
        ([0.25, 0.75], [0.25, 0.5]),
        ([0.25, 0.5], [0.5, 0.35]),
        ([0.75, 0.75], [0.75, 0.5]),
        ([0.75, 0.5], [0.5, 0.35]),
    ],
    '8': [
        ([0.5, 0.85], [0.7, 0.675]),
        ([0.7, 0.675], [0.5, 0.5]),
        ([0.5, 0.5], [0.3, 0.675]),
        ([0.3, 0.675], [0.5, 0.85]),
        ([0.5, 0.5], [0.75, 0.325]),
        ([0.75, 0.325], [0.5, 0.15]),
        ([0.5, 0.15], [0.25, 0.325]),
        ([0.25, 0.325], [0.5, 0.5]),
    ],
}

def _dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return np.sqrt((px - ax)**2 + (py - ay)**2)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / len_sq, 0, 1)
    return np.sqrt((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2)

def target_distribution(x, y, segments):
    d_min = np.full_like(x, 1e10)
    for (ax, ay), (bx, by) in segments:
        d_min = np.minimum(d_min, _dist_to_segment(x, y, ax, ay, bx, by))
    return np.exp(-d_min ** 2 / (2 * STROKE_WIDTH ** 2))

def generate_initialization(segments):
    all_points = []
    
    # We apply the Serpentine initialization directly to the explicit line segments
    for (ax, ay), (bx, by) in segments:
        mu = np.array([(ax+bx)/2, (ay+by)/2])
        dx, dy = bx - ax, by - ay
        L = np.hypot(dx, dy)
        if L < 1e-4:
            continue
            
        # Principal axes (E1 along line, E2 perpendicular)
        E1 = np.array([dx, dy]) / L * (L/2)
        E2 = np.array([-dy, dx]) / L * STROKE_WIDTH
        
        # 1-3 gentle swings based on length
        num_swings = 1.0 + 2.0 * (L / 0.7)
        
        n_pts = max(10, int(L * 200))
        tau = np.linspace(-1, 1, n_pts)
        
        # Dynamic Serpentine
        curve = mu[None, :] + np.outer(tau, E1) + 0.3 * np.outer(np.sin(num_swings * np.pi * (tau + 1)), E2)
        all_points.append(curve)
        
    return all_points

def main():
    os.makedirs('visualizations', exist_ok=True)
    
    # Evaluate on dense grid
    grid_res = 150
    _xs = np.linspace(0, 1, grid_res)
    _ys = np.linspace(0, 1, grid_res)
    Xg, Yg = np.meshgrid(_xs, _ys)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()

    for idx, (name, segments) in enumerate(SHAPES.items()):
        ax = axes[idx]
        
        # 1. Density Map (Analytical Distance Field)
        Zg = target_distribution(Xg, Yg, segments)
        ax.imshow(Zg, origin='lower', extent=[0, 1, 0, 1], cmap='Reds', alpha=0.9)
        
        # 2. Initialization Path (Serpentine)
        init_paths = generate_initialization(segments)
        for p in init_paths:
            ax.plot(p[:, 0], p[:, 1], color='white', linestyle='--', linewidth=1.5, alpha=0.8)
            
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('visualizations/segments_grid.png', dpi=150)
    print('Saved visualizations/segments_grid.png')

if __name__ == '__main__':
    main()
