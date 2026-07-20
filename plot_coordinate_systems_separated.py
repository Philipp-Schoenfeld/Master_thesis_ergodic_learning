import os
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories_separate(dataset_path, output_path):
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    N_PARTICLES, T_times_2 = data.shape
    T = T_times_2 // 2
    
    # Randomly select 5 trajectories
    np.random.seed(42)
    sample_indices = np.random.choice(N_PARTICLES, size=min(5, N_PARTICLES), replace=False)
    
    samples = []
    for i in sample_indices:
        tr = data[i].reshape(T, 2)
        samples.append(tr)

    # Setup the plot: 5 rows (one per trajectory) x 4 columns (different coordinate systems)
    fig = plt.figure(figsize=(20, 20))
    
    R, r_torus = 2, 1
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    X_surf = (R + r_torus * np.cos(V)) * np.cos(U)
    Y_surf = (R + r_torus * np.cos(V)) * np.sin(U)
    Z_surf = r_torus * np.sin(V)
    
    time_steps = np.linspace(0, 1, T)

    for i, tr in enumerate(samples):
        color = f'C{i}' # Use a unique color for each trajectory
        traj_name = f"Traj {sample_indices[i]}"
        
        # 1. Cartesian Coordinates
        ax1 = fig.add_subplot(5, 4, i*4 + 1)
        if i == 0: ax1.set_title("1. Cartesian (x, y)")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.plot(tr[:, 0], tr[:, 1], '-', color=color, label=traj_name)
        ax1.plot(tr[0, 0], tr[0, 1], 'ko', ms=3)
        ax1.legend(loc="upper right")

        # 2. Polar Coordinates
        ax2 = fig.add_subplot(5, 4, i*4 + 2, projection='polar')
        if i == 0: ax2.set_title("2. Polar (origin at 0.5,0.5)")
        x_c = tr[:, 0] - 0.5
        y_c = tr[:, 1] - 0.5
        r = np.sqrt(x_c**2 + y_c**2)
        theta = np.arctan2(y_c, x_c)
        ax2.plot(theta, r, '-', color=color)
        ax2.plot(theta[0], r[0], 'ko', ms=3)
        
        # 3. 3D Time-Series
        ax3 = fig.add_subplot(5, 4, i*4 + 3, projection='3d')
        if i == 0: ax3.set_title("3. 3D Time-Series (x, y, time)")
        ax3.plot(tr[:, 0], tr[:, 1], time_steps, '-', color=color)
        ax3.scatter(tr[0, 0], tr[0, 1], time_steps[0], color='k', s=10)
        
        # 4. Torus Mapping
        ax4 = fig.add_subplot(5, 4, i*4 + 4, projection='3d')
        if i == 0: ax4.set_title("4. Torus Mapping (x->θ, y->φ)")
        ax4.plot_wireframe(X_surf, Y_surf, Z_surf, color='gray', alpha=0.1)
        theta_torus = 2 * np.pi * tr[:, 0]
        phi_torus = 2 * np.pi * tr[:, 1]
        x_torus = (R + r_torus * np.cos(phi_torus)) * np.cos(theta_torus)
        y_torus = (R + r_torus * np.cos(phi_torus)) * np.sin(theta_torus)
        z_torus = r_torus * np.sin(phi_torus)
        ax4.plot(x_torus, y_torus, z_torus, '-', color=color, linewidth=2)
        ax4.scatter(x_torus[0], y_torus[0], z_torus[0], color='k', s=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved successfully to {output_path}")

if __name__ == "__main__":
    dataset_path = "results/Gauss_Dataset/gauss_bspline_dataset_100.npy"
    output_path = "results/Gauss_Dataset/coordinate_systems_separated_viz.png"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}.")
    else:
        plot_trajectories_separate(dataset_path, output_path)
