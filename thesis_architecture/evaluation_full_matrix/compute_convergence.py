"""Convergence statistics of the smoothness comparison, from the stored iteration states.

For every (shape, variant) the ergodic error (project metric, K = 10) of the stored states is
evaluated at ~60 log-spaced iterations and averaged over the 30 candidates. From that curve:

  n90 = first evaluated iteration at which 90 % of the total improvement (initial error minus
        final error) is reached, i.e. error <= final + 0.1 * (initial - final).

`cfm_only` has no iterations (NaN). Writes `results/<out_tag>/convergence.npz`
(keys: its, shapes, variants, curves[shape, variant, iteration], n90[shape, variant]) AND, per
(shape, variant), the same checkpoints/curve/n90 into the DB's `convergence` table
(`smoothness_db.save_convergence`) -- the DB stays the complete, durable record of a run.

Usage:  python compute_convergence.py --out_tag smoothness_comparison_v3_20261011
"""
import argparse
import os

import numpy as np
import torch

import smoothness_db as sdb
from plot_smoothness_comparison import ErgodicSeriesCache

_here = os.path.dirname(os.path.abspath(__file__))
N_CHECKPOINTS = 60


def checkpoints(n_iters):
    its = np.unique(np.round(np.geomspace(1, n_iters, N_CHECKPOINTS - 1)).astype(int))
    return np.concatenate([[0], its])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    args = ap.parse_args()
    conn = sdb.open_db(os.path.join(_here, 'results', args.out_tag, 'smoothness_comparison.db'))
    cfg = sdb.get_meta(conn, 'config')
    shapes, variants, n_iters = cfg['shapes'], cfg['variants'], cfg['n_iters']
    its = checkpoints(n_iters)
    ec = ErgodicSeriesCache(conn)
    curves = np.full((len(shapes), len(variants), len(its)), np.nan, dtype=np.float32)
    n90 = np.full((len(shapes), len(variants)), np.nan, dtype=np.float32)
    for si, shape in enumerate(shapes):
        phi_k = ec.phi_k(shape)
        for vi, variant in enumerate(variants):
            rows = list(sdb.iter_runs(conn, shape=shape, variant=variant, with_states=True))
            if not rows:
                continue
            per_cand = []
            for r in rows:
                states = r['cps']                                   # (n_states, nxi, 2)
                if states.shape[0] == 1:                            # no iterations (cfm_only)
                    idx = np.array([0])
                else:
                    idx = its
                sel = states[idx]
                dense = np.einsum('pi,sid->spd', ec.basis(r['nxi']), sel) if r['log_space'] == 'cps' else sel
                from ergodic_energy_torch import ergodic_term
                e = ergodic_term(torch.as_tensor(dense, dtype=torch.float32), ec.k_idx, ec.Lambda, phi_k).numpy()
                full = np.full(len(its), np.nan, dtype=np.float32)
                full[:len(e)] = e
                per_cand.append(full)
            curve = np.nanmean(np.stack(per_cand), axis=0)
            curves[si, vi] = curve
            if not np.isnan(curve[1:]).all():
                e0, ef = curve[0], curve[-1]
                hit = np.nonzero(curve <= ef + 0.1 * (e0 - ef))[0]
                n90[si, vi] = its[hit[0]] if len(hit) else np.nan
            sdb.save_convergence(conn, shape, variant, its, curve, n90[si, vi])
        print(f"[convergence] {shape} done ({si + 1}/{len(shapes)})", flush=True)
    out = os.path.join(_here, 'results', args.out_tag, 'convergence.npz')
    np.savez(out, its=its, shapes=np.array(shapes), variants=np.array(variants), curves=curves, n90=n90)
    for vi, v in enumerate(variants):
        print(f"[convergence] {v:30s} median n90 = {np.nanmedian(n90[:, vi]) if not np.isnan(n90[:, vi]).all() else float('nan'):7.1f}")
    print(out)


if __name__ == '__main__':
    main()
