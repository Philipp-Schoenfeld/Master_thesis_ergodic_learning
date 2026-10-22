r"""
sweep_modes.py -- how many Fourier modes do the spectral solvers need to be ergodic on sharp shapes?
=====================================================================================================
The TSVEC and Adam solvers optimise the truncated spectral metric with K x K modes. A letter-shaped
target is only resolved by high modes, so with K = 10 the optimum of the truncated metric can put a
lot of time outside the letter. This sweep runs both solvers (B-spline and waypoints + smoothness,
linear init, 2000 iterations, 30 candidates) for several K and judges them with the independent
metrics of ergodicity_metrics.py. Smooth GMM targets are included as a control (higher K must not
make them worse).
Usage: python sweep_modes.py --targets A,korean_5,sun_gmm3 --K 10,16,24 --out results/sweep_modes.json
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

import numpy as np
import torch

import ergodicity_metrics as em
import targets as tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='A,korean_5,sun_gmm3')
    ap.add_argument('--K', default='10,16,24')
    ap.add_argument('--solvers', default='tsvec_bspline,tsvec_waypoint_smooth,adam_bspline,adam_waypoint_smooth')
    ap.add_argument('--n_iters', type=int, default=2000)
    ap.add_argument('--out', default=os.path.join(_here, 'results', 'sweep_modes.json'))
    a = ap.parse_args()
    import run_smoothness_comparison as rsc1
    from common.adam_svgd import AdamSvgd
    from common.tsvec_svgd import TsvecSvgd
    from ergodic_energy_torch import W_SMOOTH, target_coeffs_from_grid
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    B32 = rsc1.basis_matrix()
    lin, _ = rsc1.linear_curves(30)
    P_bs = np.linalg.lstsq(B32, lin.transpose(1, 0, 2).reshape(128, 60), rcond=None)[0] \
        .reshape(25, 30, 2).transpose(1, 0, 2).astype(np.float32)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name, grid in tg.load(a.targets.split(',')):
        for K in [int(k) for k in a.K.split(',')]:
            for sol in a.solvers.split(','):
                key = f'{name}|{sol}|K{K}'
                if key in res:
                    continue
                bs = 'bspline' in sol
                B, P0 = (B32, P_bs) if bs else (None, lin.astype(np.float32))
                if sol.startswith('tsvec'):
                    s = TsvecSvgd(B=B, K=K, device=dev, w_smooth=W_SMOOTH, beta=100.0)
                else:
                    s = AdamSvgd(B=B, K=K, device=dev, w_smooth=W_SMOOTH, beta=100.0)
                phi = target_coeffs_from_grid(torch.as_tensor(grid, dtype=torch.float32, device=dev), s.k_idx)
                t0 = time.time()
                P = s.run(P0, phi, n_iters=a.n_iters, record=False)['final']
                X = s.render(P).detach().cpu().numpy()
                m = em.evaluate(X, grid)
                res[key] = {k: float(np.median(v)) for k, v in m.items()}
                res[key]['seconds'] = time.time() - t0
                r = res[key]
                print(f"{name:10s} {sol:22s} K={K:2d}  hell {r['hellinger']:.3f}  tv4 {r['tv_4']:.3f}  "
                      f"low {r['low_excess']:+.3f}  cov {r['coverage']:.3f}  len {r['path_len']:.1f}  "
                      f"({r['seconds']:.0f}s)", flush=True)
                json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
