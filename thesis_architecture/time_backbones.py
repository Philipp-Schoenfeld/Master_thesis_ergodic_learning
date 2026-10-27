"""Seconds per training step of the U-Net and the DiT backbone, with and without the Sinkhorn
ErgLoss, in the training configuration of run_job_start_lm*.bash (batch 256, 256 particles,
nxi 25, D384, bf16 autocast, AdamW, grad clipping). Random data: only the time matters.

    python time_backbones.py --steps 30
"""
import argparse
import os
import sys
import time

import torch

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
for _p in (os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ergodic_metric import ErgodicLoss
from flow_matching_cond_particles_dit import ParticleDiTFlowNetwork
from flow_matching_cond_particles_start import ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss

p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=30)
p.add_argument('--warmup', type=int, default=5)
p.add_argument('--batch', type=int, default=256)
p.add_argument('--n_particles', type=int, default=256)
p.add_argument('--D', type=int, default=384)
p.add_argument('--heads', type=int, default=6, help='DiT heads (DiT-S: 6).')
p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = p.parse_args()
dev = torch.device(a.device)
if dev.type == 'cuda':
    torch.backends.cudnn.benchmark = True

erg = ErgodicLoss(nxi=25, K=8, pts=128, deg=5, weight=630.0, t_power=2.0, weighted_target=True,
                  metric='sinkhorn', sinkhorn_blur=0.05, sinkhorn_scaling=0.5,
                  sinkhorn_ratio=1.0).to(dev)
x1 = torch.rand(a.batch, 25, 2, device=dev)
P = torch.rand(a.batch, a.n_particles, 3, device=dev)


def sync():
    if dev.type == 'cuda':
        torch.cuda.synchronize()


models = {'unet': lambda: ParticleCrossAttnFlowNetwork(nxi=25, D=a.D, start_inpaint=True),
          'dit': lambda: ParticleDiTFlowNetwork(nxi=25, D=a.D, n_heads=a.heads, start_inpaint=True)}
for name, build in models.items():
    model = build().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    n_par = sum(q.numel() for q in model.parameters()) / 1e6
    for ergodic in (None, erg):
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for k in range(a.warmup + a.steps):
            if k == a.warmup:
                sync()
                t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                loss, _ = compute_particle_cfm_loss(model, x1, P, p_drop=0.1, ergodic=ergodic)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sync()
        s = (time.perf_counter() - t0) / a.steps
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else float('nan')
        print(f"{name:5s} {n_par:5.1f}M  ergloss={'on ' if ergodic else 'off'}  "
              f"{s:.3f} s/step  -> {666 * s:.0f} s/epoch (666 steps)  peak {mem:.1f} GiB", flush=True)
    del model, opt
