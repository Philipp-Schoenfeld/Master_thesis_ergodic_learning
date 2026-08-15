import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import os
import sys
from tqdm import tqdm

from shape_library import pdf_on_grid

_here = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(_here, 'ergodic_dataset.db')
if not os.path.exists(db_path):
    print(f"Error: {db_path} not found.")
    sys.exit(1)

conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT shape_name, density_params, trajectory FROM ergodic_pairs ORDER BY split, id ASC")
rows = cur.fetchall()

n = len(rows)
print(f"Total trajectories generated: {n}")

batch_size = 100
num_batches = math.ceil(n / batch_size)

for b_idx in range(num_batches):
    batch = rows[b_idx * batch_size : (b_idx + 1) * batch_size]
    cols = 10
    rows_count = math.ceil(len(batch) / cols)
    
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols*2, rows_count*2), facecolor='#0f0f1a')
    axes = axes.flatten()
    
    for i, (name, params_json, traj_blob) in enumerate(tqdm(batch, desc=f"Rendering batch {b_idx+1}/{num_batches}", leave=False)):
        ax = axes[i]
        ax.set_facecolor('#0f0f1a')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        shape_def = json.loads(params_json)
        if shape_def.get('type') != 'analytical':
            shape_def['type'] = 'gmm'
            shape_def['means'] = np.array(shape_def['means'])
            shape_def['covs'] = np.array(shape_def['covs'])
            shape_def['weights'] = np.array(shape_def['weights'])
            
        pdf_grid, _, _ = pdf_on_grid(shape_def, resolution=50)
        ax.imshow(pdf_grid, origin='lower', extent=[0, 1, 0, 1], cmap='inferno', aspect='equal')
        
        traj_xy = np.frombuffer(traj_blob, dtype=np.float32).reshape(-1, 2)
        ax.plot(traj_xy[:, 0], traj_xy[:, 1], color='#FF00FF', lw=1.0, alpha=0.7)
        ax.scatter(traj_xy[0, 0], traj_xy[0, 1], s=20, c='white', zorder=5)
        ax.scatter(traj_xy[-1, 0], traj_xy[-1, 1], s=20, c='#FFD700', marker='*', zorder=5)
        
        ax.set_title(name, color='white', fontsize=8, pad=2)

    for i in range(len(batch), len(axes)):
        axes[i].axis('off')
        axes[i].set_facecolor('#0f0f1a')

    fig.suptitle(f'Ergodic Trajectories Dataset - Part {b_idx+1}/{num_batches}', color='white', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(_here, f'dataset_grid_part_{b_idx+1}.png')
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

print("All visualisations rendered successfully!")
