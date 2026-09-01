import os, sqlite3, sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')

DB_IN = os.path.join(_root, 'thesis_architecture',
                     'ergodic_dataset_generator', 'ergodic_dataset_775.db')
DB_OUT = os.path.join(_here, 'ergodic_dataset_robot.db')

def frames_from_normals(pos, nrm, eps=1e-8):
    z = -nrm / np.linalg.norm(nrm, axis=-1, keepdims=True).clip(eps)
    
    # Use a fixed UP vector to prevent the wrist from twisting along the trajectory
    up = np.zeros_like(pos)
    up[:, 2] = 1.0
    
    x = up - (up * z).sum(-1, keepdims=True) * z
    x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(eps)
    
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)

def matrix_to_rot6(R):
    return np.concatenate([R[..., 0], R[..., 1]], axis=-1)

def main():
    print(f"Reading from {DB_IN}")
    print(f"Writing to {DB_OUT}")

    if os.path.exists(DB_OUT):
        os.remove(DB_OUT)
        
    con_in = sqlite3.connect(DB_IN)
    con_out = sqlite3.connect(DB_OUT)
    
    con_out.execute('''
        CREATE TABLE ergodic_pairs_robot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shape_name TEXT,
            traj_pos BLOB,
            traj_rot6 BLOB,
            n_points INTEGER
        )
    ''')

    rows = con_in.execute("SELECT shape_name, trajectory, tsteps FROM ergodic_pairs").fetchall()
    
    for row in rows:
        shape_name, traj_blob, tsteps = row
        xy = np.frombuffer(traj_blob, dtype=np.float32).reshape((tsteps, 2))
        
        # Calculate exactly the 3D position on the robot board
        pos = np.zeros((tsteps, 3), dtype=np.float64)
        pos[:, 0] = 0.49
        pos[:, 1] = 0.3 - 0.6 * xy[:, 0]
        pos[:, 2] = 0.15 + 0.6 * xy[:, 1]
        
        # Calculate normals (pointing to -X, so that z=-nrm points to +X, into the board)
        nrm = np.zeros((tsteps, 3), dtype=np.float64)
        nrm[:, 0] = -1.0
        
        R = frames_from_normals(pos, nrm)
        rot6 = matrix_to_rot6(R)
        
        con_out.execute(
            "INSERT INTO ergodic_pairs_robot (shape_name, traj_pos, traj_rot6, n_points) VALUES (?, ?, ?, ?)",
            (shape_name, pos.astype(np.float32).tobytes(), rot6.astype(np.float32).tobytes(), tsteps)
        )
        print(f"Projected shape {shape_name}")

    con_out.commit()
    con_out.close()
    con_in.close()
    print("Done!")

if __name__ == '__main__':
    main()
