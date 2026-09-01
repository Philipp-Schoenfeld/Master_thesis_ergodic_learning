"""
test_mujoco_tex_isolated.py
===========================
Superseded -- kept as a thin shim so the old command line still works.

The scene building, board painting and trajectory following that used to live
here now sit in ``mujoco_sim/``:

    mujoco_sim/board.py       board geometry + trajectory/density loading
    mujoco_sim/ik.py          task-priority IK for the eraser tip
    mujoco_sim/run_mujoco.py  scene, playback and the live GUI-driven mode

Run the playback directly with:

    python -m mujoco_sim.run_mujoco --shape A

Note on ``--roll``: the old version tried to pin the full SE(3) pose, including
the rotation about the eraser's own axis.  The eraser is a cylinder, so that
rotation is meaningless and pinning it only used up a joint and drove the wrist
into its limits.  The new solver leaves it free, so ``--roll`` no longer exists.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mujoco_sim.run_mujoco import run_shape_playback


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--shape', default='A')
    parser.add_argument('--axis', default='Z',
                        choices=['X', '-X', 'Y', '-Y', 'Z', '-Z'],
                        help="hand-frame axis the eraser is mounted on")
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--erase', action='store_true')
    args = parser.parse_args()

    print(__doc__)
    run_shape_playback(shape_name=args.shape, axis=args.axis,
                       speed=args.speed, erase=args.erase)


if __name__ == '__main__':
    main()
