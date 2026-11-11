"""Start the 3D active-sensing control centre:  python main.py [checkpoint.pt]"""

import multiprocessing as mp
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import paths3d                                           # noqa: E402

paths3d.setup()


def run_control_centre(ckpt, device, seed, fig_sel, agent, shared_path, shared_obs):
    from interactive_sim_3d import run_interactive_sim_3d
    run_interactive_sim_3d(ckpt, device, seed, fig_sel, agent, shared_path, shared_obs)


if __name__ == '__main__':
    mp.set_start_method('spawn')       # PyTorch and Matplotlib behave better with 'spawn'
    import torch
    from mujoco_3d.run_mujoco_3d import PATH_ARRAY_SIZE, FIG_SEL_SIZE, AGENT_SIZE, OBS_ARRAY_SIZE

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = sys.argv[1] if len(sys.argv) > 1 else paths3d.DEFAULT_CKPT

    fig_sel = mp.Array('d', FIG_SEL_SIZE)      # [version, split, entry index]
    agent = mp.Array('d', AGENT_SIZE)          # [x, y, z, axis x, y, z, radius, -]
    shared_path = mp.Array('d', PATH_ARRAY_SIZE)
    shared_obs = mp.Array('d', OBS_ARRAY_SIZE)  # [n, (x, y, z, radius, margin) * OBS_MAX], figure frame

    print('Starting the 3D control centre (checkpoint: %s) ...' % ckpt)
    p = mp.Process(target=run_control_centre, args=(ckpt, device, 0, fig_sel, agent, shared_path, shared_obs))
    p.start()
    p.join()
