import os
import sys
import multiprocessing as mp
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_arch, 'ergodic_dataset_generator'), os.path.join(_root, 'SE3_SVGD')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interactive_sim import App, DEFAULT_CKPT
from writeboard import run_writeboard

def run_interactive_sim(ckpt, shapes, device, seed, shared_truth, agent_info_array, shared_obstacles, shared_path):
    App(ckpt, shapes, device, seed, shared_truth=shared_truth, agent_info_array=agent_info_array,
        shared_obstacles=shared_obstacles, shared_path=shared_path)

def run_writeboard(truth_array, agent_info_array, shared_obstacles):
    from writeboard import run_writeboard
    run_writeboard(truth_array, agent_info_array, shared_obstacles)

def run_mujoco(agent_info_array, shared_truth, shared_path=None):
    from mujoco_sim.run_mujoco import run_mujoco_sim
    run_mujoco_sim(agent_info_array, shared_truth, shared_path=shared_path)

if __name__ == '__main__':
    # Using 'spawn' to ensure PyTorch and Matplotlib play nicely in multiprocessing
    mp.set_start_method('spawn')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Default: transfer/netz2d_startpunkt_lm_inpaint.pt; optionally pass another
    # checkpoint as the first argument:  python main.py <path/to/model.pt>
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    shapes = 12
    seed = 0
    
    # 96x96 array for target distribution (shared memory)
    truth_array = mp.Array('d', 96 * 96)
    
    # [pos_x, pos_y, radius, eraser_mode, surface_mode (0 = wall, 1 = table)]
    agent_info_array = mp.Array('d', 5)
    agent_info_array[:] = [0.5, 0.5, 0.06, 0.0, 0.0]

    # [x, y, radius] * 10
    shared_obstacles = mp.Array('d', 30)

    # Trajectory projected onto the drawing surface (layout: mujoco_sim/board.py)
    from mujoco_sim.board import PATH_ARRAY_SIZE
    shared_path = mp.Array('d', PATH_ARRAY_SIZE)

    print("Starting Control Center (Schaltzentrale)...")
    p1 = mp.Process(target=run_interactive_sim, args=(ckpt, shapes, device, seed, truth_array, agent_info_array, shared_obstacles, shared_path))
    
    p1.start()
    p1.join()
