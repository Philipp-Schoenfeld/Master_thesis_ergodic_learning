import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from shape_library import VALIDATION_SHAPES, pdf_on_grid

db_path = 'ergodic_dataset.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, shape_name in enumerate(VALIDATION_SHAPES):
    cur.execute("SELECT trajectory, density_params FROM ergodic_pairs WHERE shape_name=?", (shape_name,))
    row = cur.fetchone()
    if not row:
        print(f"Shape {shape_name} not found in DB.")
        continue
    
    # trajectory is stored as raw numpy bytes (BLOB)
    traj = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 2)
    density_params = json.loads(row[1])
    
    # Calculate density background
    density_map, gx, gy = pdf_on_grid(density_params, resolution=100)
    
    ax = axes[i]
    # Plot density
    ax.imshow(density_map, origin='lower', extent=[0, 1, 0, 1], cmap='Blues', alpha=0.6)
    
    # Plot trajectory
    if traj.shape[0] >= 2:
        ax.plot(traj[:, 0], traj[:, 1], color='crimson', linewidth=1.5, alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], color='black', s=20, marker='x')
    
    ax.set_title(shape_name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('holdout_shapes_grid.png', dpi=150)
print("Saved holdout_shapes_grid.png")
conn.close()
