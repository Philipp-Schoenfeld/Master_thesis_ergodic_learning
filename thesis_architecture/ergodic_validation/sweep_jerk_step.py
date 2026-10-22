r"""
sweep_jerk_step.py -- flow step size for the jerk-penalised dynamics with the Sinkhorn flow
===========================================================================================
With the official Sinkhorn step (0.005) the jerk variant overshot on ring / uniform (the weakly
controllable end of the trajectory drifted into a corner, Hellinger 0.57 -> 0.71 after iteration
100). This sweep runs smaller steps on the targets where it failed plus a control target.
Usage: python sweep_jerk_step.py
"""
import json
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'evaluation_full_matrix'), os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'), os.path.join(_root, 'bsplinax-main')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import ergodicity_metrics as em
import targets as tg

STEPS = (0.0025, 0.00125)
TARGETS = ('ring', 'uniform', 'korean_5')


def main():
    import run_smoothness_comparison as rsc1
    from common import sun_refine as sr
    from common import sun_sinkhorn as sk
    lin, _ = rsc1.linear_curves(30)
    out_path = os.path.join(_here, 'results', 'sweep_jerk_step.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name, grid in tg.load(list(TARGETS)):
        for step in STEPS:
            key = f'{name}|{step}'
            if key in res:
                continue
            t0 = time.time()
            out = sk.run_batch(lin, grid, 2000, 128, record=True, dynamics='jerk', step=step)
            trace = []
            for it in (100, 300, 1000, 2000):
                X = out['log'][:, it - 1]
                trace.append(float(np.median(em.evaluate(X, grid)['hellinger'])))
            X = sr._resample_all(out['final_pos'], None, 128)
            m = em.evaluate(X, grid)
            res[key] = {k: float(np.median(v)) for k, v in m.items()}
            res[key]['trace'] = trace
            r = res[key]
            print(f"{name:9s} step {step:.5f}  hell {r['hellinger']:.3f} (trace 100/300/1000/2000: "
                  f"{', '.join(f'{h:.2f}' for h in trace)})  low {r['low_excess']:+.3f}  accel {r['accel']:.1e}  "
                  f"len {r['path_len']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
            json.dump(res, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
