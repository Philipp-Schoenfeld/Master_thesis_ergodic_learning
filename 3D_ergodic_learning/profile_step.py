#!/usr/bin/env python3
r"""
profile_step.py
===============
Attributes the cost of one 3D training step to its parts.

Written because the first orientation run came in at 878 s/epoch, which makes a
500-epoch run a 122-hour proposition, and the log cannot say why. The obvious
suspect — particle sampling from 64^3 volumes on the CPU — was measured at
~330 ms/batch, i.e. 3.4 % of the budget, so it is not the answer.

Reports both absolute times and the share of the step, so the result transfers
even when run at a smaller batch than the real job.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, '..', 'bsplinax-main'),
           os.path.join(_here, '..', 'thesis_architecture',
                        'ergodic_dataset_generator')):
    _p = os.path.normpath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def timeit(fn, n, sync, warmup=2):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=int, default=384)
    ap.add_argument('--nxi', type=int, default=25)
    ap.add_argument('--n_particles', type=int, default=512)
    ap.add_argument('--mini_batch', type=int, default=128)
    ap.add_argument('--erg_K', type=int, default=6)
    ap.add_argument('--erg_pts', type=int, default=128)
    ap.add_argument('--grid_res', type=int, default=64)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--device', type=str, default=None)
    ap.add_argument('--no_autocast', action='store_true')
    a = ap.parse_args()

    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_cuda = dev.type == 'cuda'
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"device={dev}, batch={a.mini_batch}, D={a.D}, "
          f"N={a.n_particles}, erg_pts={a.erg_pts}, autocast="
          f"{not a.no_autocast}\n")

    from flow_matching_cond_particles_crossattn import (
        ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss)
    from ergodic_metric import ErgodicLoss
    from orientation_energy import OrientationLoss, ParticleSurface
    from data_3d import sample_particles, augment_batch

    B, nxi, nd = a.mini_batch, a.nxi, 3
    model = ParticleCrossAttnFlowNetwork(nxi=nxi, nd=nd, D=a.D,
                                         predict_orientation=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}\n")

    x1 = torch.rand(B, nxi, nd + 6, device=dev)
    parts = torch.cat([torch.rand(B, a.n_particles, 3, device=dev),
                       torch.rand(B, a.n_particles, 1, device=dev)], dim=-1)
    t = torch.rand(B, device=dev)

    erg_pos = ErgodicLoss(nxi=nxi, K=a.erg_K, pts=a.erg_pts, nd=nd,
                          weight=100.0, ergodic_on='position').to(dev)
    erg_fp = ErgodicLoss(nxi=nxi, K=a.erg_K, pts=a.erg_pts, nd=nd,
                         weight=100.0, ergodic_on='footprint').to(dev)
    ori = OrientationLoss(nxi=nxi, pts=a.erg_pts, weight=0.012).to(dev)

    ctx = (torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
           if not a.no_autocast else torch.enable_grad())

    def full_step(**kw):
        def run():
            opt.zero_grad(set_to_none=True)
            with ctx:
                loss, _ = compute_particle_cfm_loss(model, x1, parts, **kw)
            loss.backward()
            opt.step()
        return run

    vols_cpu = torch.rand(200, a.grid_res, a.grid_res, a.grid_res)
    idx_cpu = torch.randint(0, 200, (B,))

    rows = []
    rows.append(("CFM allein (fwd+bwd+step)", timeit(full_step(), a.repeats, sync)))
    rows.append(("  + ErgLoss on=position",
                 timeit(full_step(ergodic=erg_pos), a.repeats, sync)))
    rows.append(("  + ErgLoss on=footprint",
                 timeit(full_step(ergodic=erg_fp), a.repeats, sync)))
    rows.append(("  + ErgLoss footprint + OriLoss",
                 timeit(full_step(ergodic=erg_fp, orientation=ori, w_cfm_rot=0.5),
                        a.repeats, sync)))
    rows.append(("sample_particles (CPU->GPU)",
                 timeit(lambda: sample_particles(vols_cpu, idx_cpu,
                                                 a.n_particles, dev,
                                                 mode='uniform'),
                        a.repeats, sync)))
    rows.append(("augment_batch",
                 timeit(lambda: augment_batch(x1, parts, p_flip=0.0),
                        a.repeats, sync)))

    base = rows[0][1]
    print(f"{'Komponente':34s} {'ms/Batch':>10} {'vs CFM':>9}")
    print("-" * 56)
    for name, dt in rows:
        rel = f"{dt / base:6.2f}x" if base > 0 else "—"
        print(f"{name:34s} {dt * 1000:10.1f} {rel:>9}")

    step = rows[3][1] + rows[4][1] + rows[5][1]
    print("-" * 56)
    print(f"{'Schritt gesamt (geschaetzt)':34s} {step * 1000:10.1f}")
    print(f"\n  Aufschluesselung des vollen Schritts:")
    print(f"    CFM-Netz          {rows[0][1] / step * 100:5.1f} %")
    print(f"    ErgLoss+Footprint {(rows[2][1] - rows[0][1]) / step * 100:5.1f} %")
    print(f"    OriLoss           {(rows[3][1] - rows[2][1]) / step * 100:5.1f} %")
    print(f"    sample_particles  {rows[4][1] / step * 100:5.1f} %")
    print(f"    augment_batch     {rows[5][1] / step * 100:5.1f} %")

    n_batches = int(np.ceil(775 * 15 / B))
    print(f"\n  Hochrechnung: {n_batches} Batches/Epoche"
          f" -> {step * n_batches / 60:.1f} min/Epoche")
    print(f"  500 Epochen -> {step * n_batches * 500 / 3600:.1f} h")


if __name__ == '__main__':
    main()
