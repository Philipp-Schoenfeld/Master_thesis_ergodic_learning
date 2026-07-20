import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, '/home/philipp/Documents/Uni/Master_thesis/Unified_Pipeline')
from log_surrogate_mmd import FourierErgodicMetric

for shape in ['N', 'H', 'II']:
    dir_path = f'/home/philipp/Documents/Uni/Master_thesis/results/Unified_Pipeline_{shape}'
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} not found.")
        continue
    
    final_trajs = np.load(os.path.join(dir_path, 'final_trajs.npy')) # (N, T*2)
    control_points = np.load(os.path.join(dir_path, 'control_points.npy')) # (N, n_ctrl, 2)
    
    with open(os.path.join(dir_path, 'settings.json'), 'r') as f:
        settings = json.load(f)
        
    T = settings['T']
    n_particles = settings['N_PARTICLES']
    
    metric = FourierErgodicMetric(target_shape=shape, K=10)
    Xg, Yg, Zg = metric.Xg, metric.Yg, metric.Zg
    
    fig, ax = plt.subplots(figsize=(8, 8))
    # Background target density
    ax.contourf(Xg, Yg, Zg, levels=30, cmap='YlOrRd', alpha=0.4)
    ax.contour(Xg, Yg, Zg, levels=6, colors='k', linewidths=0.3, alpha=0.2)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))
    
    for i in range(n_particles):
        tr = final_trajs[i].reshape(T, 2)
        cp = control_points[i]
        
        # Plot dense trajectory
        ax.plot(tr[:, 0], tr[:, 1], '-', color=colors[i], lw=2.0, alpha=0.5)
        
        # Plot control polygon (the B-spline points)
        ax.plot(cp[:, 0], cp[:, 1], '--o', color='black', markerfacecolor=colors[i], 
                markeredgecolor='black', lw=1.0, markersize=5, alpha=0.7, zorder=5)
        
        # Mark start of the control polygon
        ax.plot(cp[0, 0], cp[0, 1], 's', color=colors[i], markersize=7, markeredgecolor='black', zorder=6)
        
    ax.set_title(f"Phase 3 Final B-Splines — Target '{shape}'\n"
                 f"Dense Trajectories (colored) & Control Polygons (black dashed)", 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Save image
    out_file = os.path.join(dir_path, 'spline_visualization.png')
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_file}")
