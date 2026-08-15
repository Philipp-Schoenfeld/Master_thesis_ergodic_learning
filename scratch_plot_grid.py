import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from math import ceil
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_architecture', 'ergodic_dataset_generator'))
from shape_library import pdf_on_grid

def plot_final_grids():
    db_path = 'thesis_architecture/ergodic_dataset_generator/ergodic_dataset_775.db'
    out_dir = 'thesis_architecture/ergodic_dataset_generator/visualizations/final_trajectories_overview'
    os.makedirs(out_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT shape_name, density_params, trajectory FROM ergodic_pairs ORDER BY id")
    rows = c.fetchall()
    
    batch_size = 100
    n_batches = ceil(len(rows) / batch_size)
    
    for b in tqdm(range(n_batches), desc="Generating Overview Grids"):
        batch = rows[b*batch_size : (b+1)*batch_size]
        cols = 10
        rows_plot = ceil(len(batch) / cols)
        
        fig, axes = plt.subplots(rows_plot, cols, figsize=(cols*2, rows_plot*2))
        axes = axes.flatten()
        
        for i, (name, dens_json, traj_blob) in enumerate(batch):
            ax = axes[i]
            
            shape_def = json.loads(dens_json)
            if shape_def.get('type') != 'analytical':
                shape_def['type'] = 'gmm'
                shape_def['means'] = np.array(shape_def['means'])
                shape_def['covs'] = np.array(shape_def['covs'])
                shape_def['weights'] = np.array(shape_def['weights'])
                
            pdf_grid, _, _ = pdf_on_grid(shape_def, resolution=50)
            ax.imshow(pdf_grid, origin='lower', extent=[0, 1, 0, 1], cmap='inferno')
            
            # Use float32 !
            traj = np.frombuffer(traj_blob, dtype=np.float32).reshape(-1, 2)
                
            ax.plot(traj[:, 0], traj[:, 1], color='#4FC3F7', lw=1.5, alpha=0.8)
            ax.scatter(traj[:, 0], traj[:, 1], color='#4FC3F7', s=1.0, alpha=0.9)
            
            ax.set_title(name, fontsize=8, color='white')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
        for i in range(len(batch), len(axes)):
            axes[i].axis('off')
            
        fig.patch.set_facecolor('#0f0f1a')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'final_trajectories_grid_{b+1}.png'), facecolor='#0f0f1a', dpi=150)
        plt.close()

if __name__ == '__main__':
    plot_final_grids()
