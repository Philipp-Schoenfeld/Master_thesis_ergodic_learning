#!/usr/bin/env python3
"""
visualize_obstacle_holdout.py
=============================
Preview the planned inference-time obstacle on all holdout targets.

Draws the 25 val-split shapes (density heatmap + ground-truth trajectory) with
the obstacle overlaid, annotated with how much of the target density and how
many ground-truth trajectory points fall inside it. No checkpoint needed.

Usage:
  python -u visualize_obstacle_holdout.py
  python -u visualize_obstacle_holdout.py --radius 0.15 --center 0.5 0.5
"""

import argparse, os, sys, json, sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from visualize_checkpoint import load_holdout_shapes, draw_panel, _DB_PATH
from obstacles import CircleObstacle, OBSTACLE_CENTER, OBSTACLE_RADIUS


def load_full_trajectories():
    """Full (un-subsampled) val trajectories, for the coverage statistic."""
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT shape_name, trajectory FROM ergodic_pairs "
                "WHERE split='val' ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    return {name: np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
            for name, blob in rows}


def main():
    p = argparse.ArgumentParser(description='Preview the obstacle on holdout targets.')
    p.add_argument('--center', type=float, nargs=2, default=list(OBSTACLE_CENTER))
    p.add_argument('--radius', type=float, default=OBSTACLE_RADIUS)
    p.add_argument('--margin', type=float, default=0.0,
                   help='Guidance cushion to draw as a dotted ring (0 = hide).')
    p.add_argument('--grid_res', type=int, default=100)
    p.add_argument('--nxi', type=int, default=25)
    p.add_argument('--max_cols', type=int, default=5)
    p.add_argument('--out', default=os.path.join(
        _here, 'Trajectory_data_generator', 'obstacle_holdout_preview.png'))
    args = p.parse_args()

    obstacle = CircleObstacle(center=tuple(args.center), radius=args.radius,
                              margin=args.margin)
    print(f"  Obstacle: {obstacle}  (area {np.pi * args.radius ** 2:.3f} "
          f"of the unit square)")

    print("  Loading holdout shapes from DB...")
    shapes, densities = load_holdout_shapes(args.nxi, grid_res=args.grid_res)
    full_trajs = load_full_trajectories()
    labels = list(shapes.keys())
    print(f"  Found {len(labels)} holdout shapes.")

    xs = np.linspace(0, 1, args.grid_res)
    X, Y = np.meshgrid(xs, xs)
    in_mask = obstacle.mask(X, Y)

    panels, stats = [], []
    for lbl in labels:
        d_map = densities[lbl]
        dens_in = d_map[in_mask].sum() / max(d_map.sum(), 1e-12)
        traj = full_trajs[lbl]
        # Measured against the raw radius (not the guidance margin), so the
        # number stays comparable to the density fraction above.
        traj_in = float(obstacle.mask(traj[:, 0], traj[:, 1]).mean())
        stats.append((lbl, dens_in, traj_in))
        panels.append(dict(
            label=f"'{lbl}'\ndens_in={dens_in:.3f}  traj_in={traj_in:.3f}",
            base_cp=shapes[lbl], gen_cps=[], density_grid=d_map,
            particles=None, obstacle=obstacle,
        ))

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n  {'shape':<28}{'dens_in':>9}{'traj_in':>9}")
    for lbl, d, t in stats:
        print(f"  {lbl:<28}{d:>9.3f}{t:>9.3f}")
    mean_d = np.mean([s[1] for s in stats])
    mean_t = np.mean([s[2] for s in stats])
    print(f"  {'MEAN':<28}{mean_d:>9.3f}{mean_t:>9.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n = len(panels)
    n_cols = min(n, args.max_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5 * n_rows),
                             facecolor='white', squeeze=False)
    fig.suptitle(
        f"Planned obstacle on holdout targets — circle at "
        f"({obstacle.center[0]:.2f}, {obstacle.center[1]:.2f}), r={obstacle.radius:.2f}"
        f"   |   mean density inside {mean_d:.1%}, mean GT points inside {mean_t:.1%}",
        fontsize=14, fontweight='bold', color='#1A1A2E', y=1.01)

    for i, panel in enumerate(panels):
        ax = axes[i // n_cols][i % n_cols]
        draw_panel(ax, panel['base_cp'], panel['gen_cps'], panel['density_grid'],
                   panel['particles'], panel['label'], obstacle=panel['obstacle'])
        if i == 0:
            ax.legend(frameon=True, fontsize=7, loc='upper right',
                      facecolor='white', edgecolor='#ddd', framealpha=0.9)

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis('off')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved → {args.out}")


if __name__ == '__main__':
    main()
