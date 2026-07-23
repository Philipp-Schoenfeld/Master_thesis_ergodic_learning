import os
import sys
import torch
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flow_matching_cond_mpd_unet import CondMpdUNetFlowNetwork, generate_cond_trajectories
from bsplinax.bspline import BsplineBasisClamped

DB_PATH = os.path.join(_here, 'Trajectory_data_generator', 'stein_coverage_results.db')

def cp_to_bspline(cps, pts=512, deg=5):
    nxi = cps.shape[0]
    B = np.array(BsplineBasisClamped(degree=deg, num_control_points=nxi,
                                     num_phase_points=pts,
                                     compute_derivatives=False).B)
    return B @ cps

def get_random_trajectories(n=5, nxi=20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT trajectory, shape FROM runs ORDER BY RANDOM() LIMIT ?", (n,))
    rows = cur.fetchall()
    conn.close()
    
    trajs = []
    for blob, sh in rows:
        shape = tuple(map(int, sh.split(',')))
        xy = np.frombuffer(blob, dtype=np.float32).reshape(shape)[:, :2]
        idx = np.linspace(0, len(xy)-1, nxi).astype(int)
        trajs.append(xy[idx])
    return np.array(trajs)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    nxi = 20
    nd = 2
    D = 384
    
    # 1. Load the model
    model = CondMpdUNetFlowNetwork(nxi=nxi, nd=nd, D=D).to(device)
    ckpt_path = os.path.join(_here, 'checkpoints', 'cond_mpd_startpoint.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # 2. Get 5 random trajectories from DB
    real_trajs = get_random_trajectories(n=5, nxi=nxi)
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for i in range(5):
        real_traj = real_trajs[i]
        start_pos = real_traj[0] # shape (2,)
        
        # Format the start position as conditioning
        # Most likely trained by repeating the start position
        ref_cps = np.tile(start_pos, (nxi, 1)) # shape (nxi, 2)
        ref_tensor = torch.tensor(ref_cps, dtype=torch.float32).to(device)
        
        # 3. Generate trajectory
        gen_t = generate_cond_trajectories(model, ref_tensor, num_samples=1, nxi=nxi, nd=nd, steps=100, device=str(device))
        gen_cps = gen_t[0].cpu().numpy()
        
        # 4. Plot
        ax = axes[i]
        
        # Plot real trajectory
        real_bspline = cp_to_bspline(real_traj)
        ax.plot(real_bspline[:, 0], real_bspline[:, 1], 'b-', lw=2, label='Real (DB)', alpha=0.6)
        ax.scatter(real_traj[:, 0], real_traj[:, 1], color='blue', s=10, alpha=0.3)
        
        # Plot generated trajectory
        gen_bspline = cp_to_bspline(gen_cps)
        ax.plot(gen_bspline[:, 0], gen_bspline[:, 1], 'r-', lw=2, label='Generated', alpha=0.8)
        ax.scatter(gen_cps[:, 0], gen_cps[:, 1], color='red', s=15, alpha=0.6)
        
        # Mark start point
        ax.scatter([start_pos[0]], [start_pos[1]], color='green', marker='*', s=150, label='Start Point', zorder=5)
        
        ax.set_title(f"Sample {i+1}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig('visualize_startpoint_results.png', dpi=150)
    print("Saved visualize_startpoint_results.png")

if __name__ == '__main__':
    main()
