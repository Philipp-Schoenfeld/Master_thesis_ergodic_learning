r"""
sweep_jerk.py -- weights of the jerk-penalised dynamics (our own extension of Sun et al.)
==========================================================================================
`sun_refine.JerkPenalizedLQR` (state [p, v, a], control = jerk) was introduced 2026-10-09 with
provisional, unswept weights Q = diag(1, 1, 1e-3, 1e-3, 1e-2, 1e-2), R = 0.05 I. The first bench
run showed it too sluggish (korean_5 / uniform not ergodic, short paths). This sweep varies the jerk
penalty R and the acceleration-state weight (Sinkhorn flow, linear init, 30 candidates, 2000
iterations) and reports ergodicity (Hellinger, time outside the support) next to smoothness
(acceleration cost of the time-uniform samples) -- the variant must stay smoother than the point
mass while becoming ergodic.
Result (2026-10-12, korean_5): with the Sinkhorn flow the CURRENT weights are already ergodic
(Hellinger 0.33 vs. 0.57 with the Stein flow, path length 4.1); R = 0.01 makes the LQ flow-matching
step unstable at the official step size 0.005 (Hellinger 0.66, path length 108). Weights kept, the
sweep was stopped there.
Usage: python sweep_jerk.py [--targets korean_5,uniform,A]
"""
import argparse
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

import jax.numpy as jnp
import numpy as np

import ergodicity_metrics as em
import targets as tg

CONFIGS = {
    'jerk R=0.05 qa=1e-2 (current)': (0.05, 1e-2),
    'jerk R=0.01 qa=1e-2': (0.01, 1e-2),
    'jerk R=0.002 qa=1e-2': (0.002, 1e-2),
    'jerk R=0.01 qa=1e-3': (0.01, 1e-3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='korean_5,uniform,A')
    ap.add_argument('--n_iters', type=int, default=2000)
    ap.add_argument('--out', default=os.path.join(_here, 'results', 'sweep_jerk.json'))
    a = ap.parse_args()
    import run_smoothness_comparison as rsc1
    from common import sun_refine as sr
    from common import sun_sinkhorn as sk
    lin, _ = rsc1.linear_curves(30)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name, grid in tg.load(a.targets.split(',')):
        for label, cfg in CONFIGS.items():
            key = f'{name}|{label}'
            if key in res:
                continue
            t0 = time.time()
            if cfg is None:
                out = sk.run_batch(lin, grid, a.n_iters, 128, dynamics='pointmass')
            else:
                R, qa = cfg
                pm = sr.JerkPenalizedLQR(dt=sr.DT, Q=jnp.diag(jnp.array([1.0, 1.0, 1e-3, 1e-3, qa, qa])),
                                         R=jnp.diag(jnp.array([R, R])))
                out = sk.run_batch(lin, grid, a.n_iters, 128, dynamics='jerk', lqr=pm)
            X = sr._resample_all(out['final_pos'], None, 128)
            m = em.evaluate(X, grid)
            res[key] = {k: float(np.median(v)) for k, v in m.items()}
            res[key]['seconds'] = time.time() - t0
            r = res[key]
            print(f"{name:9s} {label:32s} hell {r['hellinger']:.3f}  low {r['low_excess']:+.3f}  "
                  f"accel {r['accel']:.2e}  len {r['path_len']:.2f}  ({r['seconds']:.0f}s)", flush=True)
            json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
