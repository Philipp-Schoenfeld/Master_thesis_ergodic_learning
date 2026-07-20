import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Visualize 3D trajectories interactively.")
    parser.add_argument('--file', type=str, default="results/SE3_SVGD_BSpline_3D_N/n_shape_trajs.npy", help="Path to the .npy file containing trajectories.")
    parser.add_argument('--shape', type=str, default="N", choices=["N", "H", "II"], help="Target shape to project.")
    args = parser.parse_args()

    file_path = args.file
    target_shape = args.shape

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    data = np.load(file_path)
    if len(data.shape) == 2:
        N_PARTICLES, length = data.shape
        DIM = 3
        T = length // DIM
        data = data.reshape(N_PARTICLES, T, DIM)
    else:
        N_PARTICLES, T, DIM = data.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.rainbow(np.linspace(0, 1, N_PARTICLES))

    for i in range(N_PARTICLES):
        ax.plot(data[i, :, 0], data[i, :, 1], data[i, :, 2], '-', color=colors[i], lw=1.5, alpha=0.7)
        ax.scatter(data[i, 0, 0], data[i, 0, 1], data[i, 0, 2], color=colors[i], s=15)

    # Target shape: projected letter on a diagonal plane (z = 0.5*x + 0.5*y)
    SHAPE_SEGMENTS_2D = {
        'N': [
            ([0.25, 0.15], [0.25, 0.85]),
            ([0.25, 0.85], [0.75, 0.15]),
            ([0.75, 0.15], [0.75, 0.85]),
        ],
        'H': [
            ([0.25, 0.15], [0.25, 0.85]),
            ([0.75, 0.15], [0.75, 0.85]),
            ([0.25, 0.50], [0.75, 0.50]),
        ],
        'II': [
            ([0.25, 0.15], [0.25, 0.85]),
            ([0.75, 0.15], [0.75, 0.85]),
        ],
    }

    if target_shape in SHAPE_SEGMENTS_2D:
        for (a, b) in SHAPE_SEGMENTS_2D[target_shape]:
            a3 = [a[0], a[1], 0.5 * a[0] + 0.5 * a[1]]
            b3 = [b[0], b[1], 0.5 * b[0] + 0.5 * b[1]]
            ax.plot([a3[0], b3[0]], [a3[1], b3[1]], [a3[2], b3[2]], 'k--', lw=2.0, alpha=0.8, zorder=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f"Interactive 3D Trajectory Visualization ({target_shape})")

    output_file = "interactive_3d_viz.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Image saved to {output_file}")

    plt.show()

if __name__ == "__main__":
    main()
