import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sqlite3

# =============================================================================
# 1. EXISTING CHARACTER WAYPOINTS (60 shapes)
# =============================================================================

alphabet = {
    'A': [(0,0), (0.5,1), (1,0), (0.75,0.5), (0.25,0.5)],
    'B': [(0,0), (0,1), (0.5,1), (0.8,0.8), (0.5,0.5), (0,0.5), (0.5,0.5), (0.8,0.2), (0.5,0), (0,0)],
    'C': [(1,0.2), (0.8,0), (0.2,0), (0,0.2), (0,0.8), (0.2,1), (0.8,1), (1,0.8)],
    'D': [(0,0), (0,1), (0.5,1), (1,0.8), (1,0.2), (0.5,0), (0,0)],
    'E': [(1,1), (0,1), (0,0.5), (0.8,0.5), (0,0.5), (0,0), (1,0)],
    'F': [(1,1), (0,1), (0,0.5), (0.8,0.5), (0,0.5), (0,0)],
    'G': [(1,0.8), (0.8,1), (0.2,1), (0,0.8), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,0.5), (0.5,0.5)],
    'H': [(0,1), (0,0), (0,0.5), (1,0.5), (1,1), (1,0)],
    'I': [(0.2,1), (0.8,1), (0.5,1), (0.5,0), (0.2,0), (0.8,0)],
    'J': [(0,0.5), (0.2,0), (0.8,0), (1,0.2), (1,1)],
    'K': [(0,1), (0,0), (0,0.5), (1,1), (0,0.5), (1,0)],
    'L': [(0,1), (0,0), (1,0)],
    'M': [(0,0), (0,1), (0.5,0.5), (1,1), (1,0)],
    'N': [(0,0), (0,1), (1,0), (1,1)],
    'O': [(0.5,1), (0.2,1), (0,0.8), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,0.8), (0.8,1), (0.5,1)],
    'P': [(0,0), (0,1), (0.8,1), (1,0.8), (1,0.6), (0.8,0.5), (0,0.5)],
    'Q': [(0.5,1), (0.2,1), (0,0.8), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,0.8), (0.8,1), (0.5,1), (0.5,0.5), (0.8,0.2), (1,0)],
    'R': [(0,0), (0,1), (0.8,1), (1,0.8), (1,0.6), (0.8,0.5), (0,0.5), (1,0)],
    'S': [(1,0.8), (0.8,1), (0.2,1), (0,0.8), (0,0.6), (0.2,0.5), (0.8,0.5), (1,0.4), (1,0.2), (0.8,0), (0.2,0), (0,0.2)],
    'T': [(0,1), (1,1), (0.5,1), (0.5,0)],
    'U': [(0,1), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,1)],
    'V': [(0,1), (0.5,0), (1,1)],
    'W': [(0,1), (0.25,0), (0.5,0.5), (0.75,0), (1,1)],
    'X': [(0,1), (1,0), (0.5,0.5), (0,0), (1,1)],
    'Y': [(0,1), (0.5,0.5), (1,1), (0.5,0.5), (0.5,0)],
    'Z': [(0,1), (1,1), (0,0), (1,0)]
}

numbers = {
    '0': [(0.5,1), (0.2,1), (0,0.8), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,0.8), (0.8,1), (0.5,1)],
    '1': [(0.2,0.8), (0.5,1), (0.5,0), (0.2,0), (0.8,0)],
    '2': [(0,0.8), (0.2,1), (0.8,1), (1,0.8), (1,0.6), (0,0), (1,0)],
    '3': [(0,0.8), (0.2,1), (0.8,1), (1,0.8), (0.5,0.5), (1,0.2), (0.8,0), (0.2,0), (0,0.2)],
    '4': [(0.8,0), (0.8,1), (0,0.4), (1,0.4)],
    '5': [(1,1), (0,1), (0,0.6), (0.8,0.6), (1,0.4), (1,0.2), (0.8,0), (0.2,0), (0,0.2)],
    '6': [(1,0.8), (0.8,1), (0.2,1), (0,0.8), (0,0), (0.8,0), (1,0.2), (1,0.4), (0.8,0.5), (0,0.5)],
    '7': [(0,1), (1,1), (0.4,0)],
    '8': [(0.5,0.5), (0.2,0.6), (0,0.8), (0.2,1), (0.8,1), (1,0.8), (0.8,0.6), (0.5,0.5), (0.2,0.4), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (0.8,0.4), (0.5,0.5)],
    '9': [(0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,1), (0.2,1), (0,0.8), (0,0.6), (0.2,0.5), (1,0.5)]
}

greek = {
    'alpha': [(0.8, 0.8), (0.2, 0.8), (0, 0.5), (0.2, 0), (0.8, 0), (0.2, 1), (1, 0)],
    'beta': [(0,-0.2), (0,1), (0.5,1), (0.8,0.8), (0.5,0.5), (0,0.5), (0.5,0.5), (0.8,0.2), (0.5,0), (0,0)],
    'gamma': [(0,1), (0.4,0), (0.3,-0.2), (0.4,-0.4), (0.6,-0.4), (0.7,-0.2), (0.5,0.2), (1,1)],
    'delta': [(0.8, 1), (0.4, 0.8), (0.2, 0.5), (0.2, 0.2), (0.4, 0), (0.6, 0), (0.8, 0.2), (0.8, 0.5), (0.4, 0.5)],
    'epsilon': [(0.8,0.8), (0.2,0.8), (0,0.5), (0.5,0.5), (0,0.5), (0.2,0), (0.8,0)],
    'zeta': [(0,1), (1,1), (0,0), (0.5,-0.2), (0.5,-0.4), (0,-0.4)],
    'eta': [(0,1), (0,0), (0,0.8), (0.8,0.8), (0.8,0), (0.8,-0.4)],
    'theta': [(0.5,1), (0.2,1), (0,0.8), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,0.8), (0.8,1), (0.5,1), (0.5,0.5), (0,0.5), (1,0.5)],
    'iota': [(0.5,1), (0.5,0), (0.8,0)],
    'kappa': [(0,1), (0,0), (0,0.5), (1,1), (0.2,0.5), (1,0)],
    'lamda': [(0.5,1), (0,0), (0.5,1), (1,0)],
    'mu': [(0,0.8), (0,-0.4), (0,0), (0.5,0), (0.8,0.8), (0.8,0)],
    'nu': [(0,0.8), (0.5,0), (1,0.8)],
    'xi': [(0,1), (1,1), (0.5,0.5), (0,0.5), (1,0.5), (0.5,0), (0,0), (0.5,-0.2), (0.5,-0.4), (0,-0.4)],
    'omicron': [(0.5,0.8), (0.2,0.8), (0,0.6), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,0.6), (0.8,0.8), (0.5,0.8)],
    'pi': [(0,1), (1,1), (0.8,1), (0.8,0), (0.8,1), (0.2,1), (0.2,0)],
    'rho': [(0,-0.4), (0,0.8), (0.8,0.8), (1,0.6), (1,0.4), (0.8,0.2), (0,0.2)],
    'sigma': [(0.8,0.8), (0.2,0.8), (0,0.6), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,0.6), (0.8,0.8), (1,0.8)],
    'tau': [(0,0.8), (1,0.8), (0.5,0.8), (0.5,0), (0.8,0)],
    'upsilon': [(0,0.8), (0.2,0), (0.8,0), (1,0.8)],
    'phi': [(0.5,1), (0.5,-0.4), (0.5,0.5), (0.8,0.8), (0.2,0.8), (0,0.5), (0.2,0.2), (0.8,0.2), (1,0.5), (0.8,0.8)],
    'chi': [(0,0.8), (1,-0.2), (0.5,0.3), (0,-0.2), (1,0.8)],
    'psi': [(0,0.8), (0,0.2), (0.2,0), (0.8,0), (1,0.2), (1,0.8), (1,0.2), (0.5,0), (0.5,-0.4), (0.5,1)],
    'omega': [(0,0.8), (0.2,0), (0.5,0.4), (0.8,0), (1,0.8)]
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def normalize_to_unit(pts, margin=0.05):
    """Normalize points to fit in [margin, 1-margin]^2, preserving aspect ratio."""
    pts = np.array(pts, dtype=np.float64)
    mn, mx = pts.min(0), pts.max(0)
    span = max((mx - mn).max(), 1e-8)
    return (pts - (mn + mx) / 2.0) / span * (1 - 2 * margin) + 0.5

def sample_path(waypoints, num_points=200):
    """Evenly resample waypoints to num_points via linear interpolation."""
    waypoints = np.array(waypoints, dtype=np.float64)
    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total = seg_lengths.sum()
    if total == 0:
        return np.repeat(waypoints[[0]], num_points, axis=0)
    cum = np.insert(np.cumsum(seg_lengths), 0, 0)
    t = np.linspace(0, total, num_points)
    out = np.zeros((num_points, 2))
    out[:, 0] = np.interp(t, cum, waypoints[:, 0])
    out[:, 1] = np.interp(t, cum, waypoints[:, 1])
    return out

def bezier_curve(control_points, n_pts=200):
    """Evaluate a Bézier curve using De Casteljau's algorithm."""
    cp = np.array(control_points, dtype=np.float64)
    t_vals = np.linspace(0, 1, n_pts)
    result = np.zeros((n_pts, 2))
    for i, t in enumerate(t_vals):
        p = cp.copy()
        for k in range(len(p) - 1, 0, -1):
            p[:k] = (1 - t) * p[:k] + t * p[1:k + 1]
        result[i] = p[0]
    return result

# =============================================================================
# 3. PROCEDURAL SHAPE GENERATORS
# =============================================================================

def make_regular_polygon(n, num_points=200):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    pts = np.vstack([pts, pts[:1]])  # close
    return normalize_to_unit(sample_path(pts, num_points))

def make_star(n, inner_ratio=0.4, num_points=200):
    angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    radii = np.array([1.0 if i % 2 == 0 else inner_ratio for i in range(2 * n)])
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    pts = np.vstack([pts, pts[:1]])
    return normalize_to_unit(sample_path(pts, num_points))

def make_spiral(turns, direction=1, num_points=200):
    theta = np.linspace(0, turns * 2 * np.pi, num_points)
    r = theta / (turns * 2 * np.pi)
    pts = np.column_stack([r * np.cos(direction * theta),
                           r * np.sin(direction * theta)])
    return normalize_to_unit(pts)

def make_log_spiral(turns=2, num_points=200):
    theta = np.linspace(0.1, turns * 2 * np.pi, num_points)
    r = 0.1 * np.exp(0.12 * theta)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return normalize_to_unit(pts)

def make_lissajous(a, b, delta=np.pi / 2, num_points=200):
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    pts = np.column_stack([np.sin(a * t + delta), np.sin(b * t)])
    return normalize_to_unit(pts)

def make_sine_wave(periods, num_points=200):
    x = np.linspace(0, 1, num_points)
    y = 0.5 + 0.4 * np.sin(2 * np.pi * periods * x)
    return np.column_stack([x, y])

def make_zigzag(n_zigs, num_points=200):
    wps = [(0.0, 0.5)]
    for i in range(n_zigs):
        x = (i + 0.5) / n_zigs
        y = 0.9 if i % 2 == 0 else 0.1
        wps.append((x, y))
    wps.append((1.0, 0.5))
    return sample_path(wps, num_points)

def make_staircase(n_steps=4, num_points=200):
    wps = []
    for i in range(n_steps):
        x0 = i / n_steps
        y = i / (n_steps - 1) if n_steps > 1 else 0.5
        wps.append((x0, y))
        wps.append(((i + 1) / n_steps, y))
    return normalize_to_unit(sample_path(wps, num_points))

def make_heart(num_points=200):
    t = np.linspace(0, 2 * np.pi, num_points)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    return normalize_to_unit(np.column_stack([x, y]))

def make_infinity(num_points=200):
    t = np.linspace(0, 2 * np.pi, num_points)
    denom = 1 + np.sin(t) ** 2
    x = np.cos(t) / denom
    y = np.sin(t) * np.cos(t) / denom
    return normalize_to_unit(np.column_stack([x, y]))

# Symbol waypoints (normalized later via sample_path)
symbol_waypoints = {
    'cross':       [(0.5,0), (0.5,1), (0.5,0.5), (0,0.5), (1,0.5)],
    'diamond':     [(0.5,0), (1,0.5), (0.5,1), (0,0.5), (0.5,0)],
    'checkmark':   [(0,0.5), (0.35,0), (1,1)],
    'lightning':   [(0.3,1), (0.55,0.6), (0.35,0.55), (0.7,0)],
    'arrow_right': [(0,0.5), (0.8,0.5), (0.6,0.8), (0.8,0.5), (0.6,0.2)],
    'arrow_up':    [(0.5,0), (0.5,0.8), (0.2,0.6), (0.5,0.8), (0.8,0.6)],
}

def make_random_convex_polygon(seed, n_verts, num_points=200):
    rng = np.random.RandomState(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_verts))
    radii = rng.uniform(0.3, 1.0, n_verts)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    pts = np.vstack([pts, pts[:1]])  # close
    return normalize_to_unit(sample_path(pts, num_points))

def make_random_bezier(seed, n_control, num_points=200):
    rng = np.random.RandomState(seed)
    cp = rng.uniform(-1, 1, (n_control, 2))
    pts = bezier_curve(cp, num_points)
    return normalize_to_unit(pts)

def make_random_lines(seed, n_waypoints, num_points=200):
    rng = np.random.RandomState(seed)
    wps = rng.uniform(0.05, 0.95, (n_waypoints, 2))
    return sample_path(wps, num_points)

# =============================================================================
# 4. GENERATE ALL 150 SHAPES
# =============================================================================

def generate_all_shapes(num_points=200):
    """
    Returns an OrderedDict: label → (num_points, 2) numpy array.
    Total: 60 characters + 90 procedural = 150 shapes.
    """
    from collections import OrderedDict
    shapes = OrderedDict()

    # ── Characters (60) ──
    all_chars = {**alphabet, **numbers, **greek}
    for name, wps in all_chars.items():
        shapes[name] = sample_path(wps, num_points)

    # ── Regular polygons (8): triangle → decagon ──
    for n in range(3, 11):
        shapes[f'poly_{n}'] = make_regular_polygon(n, num_points)

    # ── Stars (6): 3- through 8-pointed ──
    for n in range(3, 9):
        shapes[f'star_{n}'] = make_star(n, inner_ratio=0.4, num_points=num_points)

    # ── Spirals (6) ──
    for turns in [1, 2, 3]:
        shapes[f'spiral_{turns}cw']  = make_spiral(turns,  1, num_points)
    for turns in [1, 2]:
        shapes[f'spiral_{turns}ccw'] = make_spiral(turns, -1, num_points)
    shapes['log_spiral'] = make_log_spiral(2, num_points)

    # ── Lissajous curves (8) ──
    for (a, b) in [(1,2), (1,3), (2,3), (3,2), (3,4), (2,5), (3,5), (4,5)]:
        shapes[f'lissajous_{a}_{b}'] = make_lissajous(a, b, num_points=num_points)

    # ── Waves (6) ──
    for p in [1, 2, 3]:
        shapes[f'sine_{p}'] = make_sine_wave(p, num_points)
    for z in [3, 5]:
        shapes[f'zigzag_{z}'] = make_zigzag(z, num_points)
    shapes['staircase'] = make_staircase(5, num_points)

    # ── Symbols (8) ──
    shapes['heart']    = make_heart(num_points)
    shapes['infinity'] = make_infinity(num_points)
    for name, wps in symbol_waypoints.items():
        shapes[name] = sample_path(wps, num_points)

    # ── Random convex polygons (60) ──
    for i in range(60):
        n_verts = 4 + (i % 5)  # 4,5,6,7,8 vertices cycling
        shapes[f'rand_poly_{i}'] = make_random_convex_polygon(
            seed=100 + i, n_verts=n_verts, num_points=num_points)

    # ── Random Bézier curves (80) ──
    for i in range(80):
        n_cp = 3 + (i % 4)  # 3,4,5,6 control points cycling
        shapes[f'rand_bezier_{i}'] = make_random_bezier(
            seed=200 + i, n_control=n_cp, num_points=num_points)

    # ── Random connected line segments (58) ──
    for i in range(58):
        n_wp = 4 + (i % 4)  # 4,5,6,7 waypoints cycling
        shapes[f'rand_lines_{i}'] = make_random_lines(
            seed=300 + i, n_waypoints=n_wp, num_points=num_points)

    assert len(shapes) == 300, f"Expected 300 shapes, got {len(shapes)}"
    return shapes

# =============================================================================
# 5. MAIN — GENERATE, VISUALIZE, SAVE
# =============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    num_points = 200

    shapes = generate_all_shapes(num_points)
    print(f"Generated {len(shapes)} shapes")

    # ── Visualization (20 × 15 grid) ──
    n_rows, n_cols = 20, 15
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 40))
    axes = axes.flatten()

    for idx, (name, pts) in enumerate(shapes.items()):
        ax = axes[idx]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=np.arange(len(pts)), cmap='viridis', s=6, zorder=2)
        ax.plot(pts[:, 0], pts[:, 1], 'k-', lw=0.8, alpha=0.3, zorder=1)
        ax.set_title(name, fontsize=7)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.55, 1.15)
        ax.set_aspect('equal')
        ax.axis('off')

    for idx in range(len(shapes), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'all_shapes_visualization.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {plot_path}")

    # ── Save to SQLite DB ──
    db_path = os.path.join(script_dir, 'character_trajectories.db')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Recreate table with label column
    cur.execute('DROP TABLE IF EXISTS runs')
    cur.execute('''
        CREATE TABLE runs (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            x0_x  REAL,
            x0_y  REAL,
            trajectory BLOB,
            shape TEXT,
            label TEXT
        )
    ''')

    for name, pts in shapes.items():
        traj = pts.astype(np.float32)
        shape_str = f"{traj.shape[0]},{traj.shape[1]}"
        cur.execute(
            'INSERT INTO runs (x0_x, x0_y, trajectory, shape, label) '
            'VALUES (?, ?, ?, ?, ?)',
            (float(traj[0, 0]), float(traj[0, 1]), traj.tobytes(), shape_str, name))

    conn.commit()
    conn.close()
    print(f"Database saved to {db_path}  ({len(shapes)} shapes)")

    # ── Save to JSON (sampled only) ──
    json_path = os.path.join(script_dir, 'character_trajectories_sampled.json')
    with open(json_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in shapes.items()}, f, indent=2)
    print(f"JSON saved to {json_path}")


if __name__ == '__main__':
    main()
