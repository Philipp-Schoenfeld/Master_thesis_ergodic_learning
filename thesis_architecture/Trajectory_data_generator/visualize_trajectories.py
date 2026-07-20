import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn
from jax import vmap
import os

def pdf(x):
    mean_simple = jnp.array([0.5, 0.5])
    cov_simple = jnp.array([
        [0.02, 0.0],
        [0.0, 0.02]
    ])
    w1 = 0.34
    return w1 * mvn.pdf(x[:2], mean_simple, cov_simple)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "stein_coverage_results.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT x0_x, x0_y, trajectory, shape FROM runs LIMIT 3")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No data found in the database.")
        return

    # Prepare background grid
    grids_x, grids_y = jnp.meshgrid(
        jnp.linspace(0.0, 1.0, 100),
        jnp.linspace(0.0, 1.0, 100)
    )
    grids = jnp.array([grids_x.ravel(), grids_y.ravel()]).T
    pdf_grids = vmap(pdf)(grids).reshape(grids_x.shape)
    pdf_grids = np.array(pdf_grids)
    clevels = np.linspace(pdf_grids.min(), pdf_grids.max(), 11)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    ax.set_title("First 3 Ergodic Trajectories")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.contourf(grids_x, grids_y, pdf_grids, levels=clevels[1:], cmap='Purples')

    colors = ['C1', 'C2', 'C3']
    
    for i, row in enumerate(rows):
        x0_x, x0_y, traj_blob, shape_str = row
        
        # Parse shape
        shape = tuple(map(int, shape_str.split(',')))
        
        # Load trajectory
        traj = np.frombuffer(traj_blob, dtype=np.float32).reshape(shape)
        
        color = colors[i % len(colors)]
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], linestyle='-', linewidth=2,
                marker='o', markersize=3, color=color, alpha=0.7, label=f"Traj {i+1}")
        
        # Plot start point
        ax.plot(x0_x, x0_y, linestyle='', marker='o', markersize=10, color='k')
    
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(script_dir, "first_3_trajectories.png")
    plt.show()
    #plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
