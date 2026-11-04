#!/usr/bin/env python3
"""Step time and peak GPU memory of candidate 3D backbones on real batches.

Uses the exact training step of `flow_matching_runner_particles.train` (same
augmentation, bf16 autocast, CFM + Sinkhorn footprint ErgLoss + orientation
loss, grad clipping, AdamW) on entries of the 3D DB, so the numbers can be
turned directly into an epoch budget. Prints one line per config and the raw
loss parts at init.

    python bench_backbones_3d.py --db3d ergodic_dataset_3d_lm.db --mini_batch 256
"""
import argparse
import time

import numpy as np
import torch

from data_3d import augment_batch
from data_surfaces import load_surface_db
from ergodic_metric import ErgodicLoss
from flow_matching_cond_particles_crossattn import (
    ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss)
from flow_matching_cond_particles_dit import ParticleDiTFlowNetwork
from orientation_energy import OrientationLoss

CONFIGS = {
    'unet-D384':     dict(backbone='unet', D=384),
    'unet-D512':     dict(backbone='unet', D=512),
    'dit-D384-d12':  dict(backbone='dit', D=384, depth=12, heads=6),
    'dit-D512-d12':  dict(backbone='dit', D=512, depth=12, heads=8),
    'dit-D512-d16':  dict(backbone='dit', D=512, depth=16, heads=8),
    'dit-D768-d12':  dict(backbone='dit', D=768, depth=12, heads=12),
}


def build(cfg, device):
    if cfg['backbone'] == 'dit':
        return ParticleDiTFlowNetwork(nxi=25, D=cfg['D'], depth=cfg['depth'],
                                      n_heads=cfg['heads'], predict_orientation=True,
                                      start_cond=True, start_inpaint=True).to(device)
    return ParticleCrossAttnFlowNetwork(nxi=25, D=cfg['D'], predict_orientation=True,
                                        start_cond=True, start_inpaint=True).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db3d', default='ergodic_dataset_3d_lm.db')
    p.add_argument('--mini_batch', type=int, default=256)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--n_entries', type=int, default=4000)
    p.add_argument('--configs', nargs='+', default=list(CONFIGS))
    a = p.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = dev.type == 'cuda'
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    torch.backends.cudnn.benchmark = True
    print(f"device: {torch.cuda.get_device_name(0) if cuda else 'cpu'}")
    entries = load_surface_db(a.db3d, splits=('train',), target_cps=True)[:a.n_entries]
    x1 = torch.from_numpy(np.stack([e['x1'] for e in entries])).to(dev)
    parts = torch.from_numpy(np.stack([e['parts'] for e in entries])).to(dev)
    print(f"{len(entries)} entries, batch {a.mini_batch}")

    erg = ErgodicLoss(nxi=25, pts=128, deg=5, weight=630.0, t_power=2.0, nd=3,
                      ergodic_on='footprint', metric='sinkhorn',
                      sinkhorn_blur=0.05).to(dev)
    ori = OrientationLoss(nxi=25, pts=128, deg=5, weight=0.012, t_power=2.0,
                          w_point=0.1, w_standoff=300.0, w_angsmooth=2.0,
                          standoff_target=0.12, standoff_band=0.03).to(dev)

    for name in a.configs:
        cfg = CONFIGS[name]
        torch.manual_seed(0)
        model = build(cfg, dev)
        n_par = sum(q.numel() for q in model.parameters()) / 1e6
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        times, last = [], {}
        try:
            for s in range(a.warmup + a.steps):
                idx = torch.randint(0, len(entries), (a.mini_batch,), device=dev)
                b, pa = augment_batch(x1[idx], parts[idx], p_flip=0.0, rot_range=20.0,
                                      scale_range=(0.75, 1.25), trans_range=0.08,
                                      noise_std=0.015, rot_full=True)
                sync()
                t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    loss, pr = compute_particle_cfm_loss(model, b, pa, p_drop=0.1,
                                                         ergodic=erg, orientation=ori,
                                                         w_cfm_rot=0.5)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sync()
                if s >= a.warmup:
                    times.append(time.perf_counter() - t0)
                last = {k: v.item() for k, v in pr.items()}
        except torch.cuda.OutOfMemoryError:
            # One config too large for this card must not end the whole benchmark.
            print(f"{name:<14} {n_par:6.1f}M  OUT OF MEMORY at batch {a.mini_batch}",
                  flush=True)
            del model, opt
            torch.cuda.empty_cache()
            continue
        t = sorted(times)[len(times) // 2]
        mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
        print(f"{name:<14} {n_par:6.1f}M  {t:.3f} s/step (median of {len(times)})  "
              f"peak {mem:5.1f} GiB  | cfm {last['cfm']:.3f} cfm_rot {last['cfm_rot']:.3f} "
              f"erg {last['erg']:.2e} ori {last['ori']:.2f}", flush=True)
        del model, opt
        if cuda:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
