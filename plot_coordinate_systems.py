import os
import numpy as np
import matplotlib.pyplot as plt

def plot_different_coordinate_systems(dataset_path, output_path):
    print(f"Loading dataset from {dataset_path}...")
    data = np.load(dataset_path)
    N_PARTICLES, T_times_2 = data.shape
    T = T_times_2 // 2
    
    # Randomly select 5 trajectories
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(N_PARTICLES, size=min(5, N_PARTICLES), replace=False)
    
    # Reshape and extract 5 trajectories
    samples = []
    for i in sample_indices:
        tr = data[i].reshape(T, 2)
        samples.append(tr)

    # Setup the plot
    fig = plt.figure(figsize=(20, 5))
    
    # 1. Cartesian Coordinates (2D)
    ax1 = fig.add_subplot(141)
    ax1.set_title("1. Cartesian Coordinates (x, y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    for idx, tr in enumerate(samples):
        ax1.plot(tr[:, 0], tr[:, 1], '-', label=f"Traj {sample_indices[idx]}")
        ax1.plot(tr[0, 0], tr[0, 1], 'ko', ms=3)
    ax1.legend()

    # 2. Polar Coordinates (r, theta) centered at (0.5, 0.5)
    ax2 = fig.add_subplot(142, projection='polar')
    ax2.set_title("2. Polar Coordinates (origin at 0.5,0.5)")
    for tr in samples:
        x_c = tr[:, 0] - 0.5
        y_c = tr[:, 1] - 0.5
        r = np.sqrt(x_c**2 + y_c**2)
        theta = np.arctan2(y_c, x_c)
        ax2.plot(theta, r, '-')
        ax2.plot(theta[0], r[0], 'ko', ms=3)
    
    # 3. 3D Time-Series (x, y, t)
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.set_title("3. 3D Time-Series (x, y, time)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("time t")
    time_steps = np.linspace(0, 1, T)
    for tr in samples:
        ax3.plot(tr[:, 0], tr[:, 1], time_steps, '-')
        ax3.scatter(tr[0, 0], tr[0, 1], time_steps[0], color='k', s=10)
    
    # 4. Torus Mapping (3D)
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.set_title("4. Torus Mapping (x->θ, y->φ)")
    R, r_torus = 2, 1
    # Plot faint torus surface
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)
    X = (R + r_torus * np.cos(V)) * np.cos(U)
    Y = (R + r_torus * np.cos(V)) * np.sin(U)
    Z = r_torus * np.sin(V)
    ax4.plot_wireframe(X, Y, Z, color='gray', alpha=0.1)
    
    for tr in samples:
        theta = 2 * np.pi * tr[:, 0]
        phi = 2 * np.pi * tr[:, 1]
        x_torus = (R + r_torus * np.cos(phi)) * np.cos(theta)
        y_torus = (R + r_torus * np.cos(phi)) * np.sin(theta)
        z_torus = r_torus * np.sin(phi)
        ax4.plot(x_torus, y_torus, z_torus, '-', linewidth=2)
        ax4.scatter(x_torus[0], y_torus[0], z_torus[0], color='k', s=10)
    
    # Finalize and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved successfully to {output_path}")

if __name__ == "__main__":
    dataset_path = "results/Gauss_Dataset/gauss_bspline_dataset_100.npy"
    output_path = "results/Gauss_Dataset/coordinate_systems_viz.png"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}.")
    else:
        plot_different_coordinate_systems(dataset_path, output_path)
