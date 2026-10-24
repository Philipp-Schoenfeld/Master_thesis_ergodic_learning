"""Full-epoch robustness check after the lambda0 tanh-bound fix (see
alm_solver_torch.py's _LAMBDA0_BOUND): runs all ~900 real batches with the
same skip-on-non-finite-gradient safety net flow_matching_runner_ergodic.py's
train() now has, and reports how often (if ever) a step still had to be
skipped -- the goal is zero skips, but even if not, the model must survive
the full epoch without permanent corruption (finite loss throughout)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from obstacles import basis_torch
from flow_matching_cond_spectral_crossattn import (
    SpectralCrossAttnFlowNetwork, compute_spectral_cfm_loss_dfl,
)
import alm_solver_torch as alm_torch
from flow_matching_runner_ergodic import (
    _load_shapes, _build_dataset, _make_ergodic_k_grid,
    trajectory_to_spectral_batch_torch, augment_batch_torch,
)

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

NXI, D, S = 25, 384, 256
MINI_BATCH = 64
COPIES = int(os.environ.get('COPIES_PER_CHAR', '75'))
N_BATCHES = int(os.environ.get('N_BATCHES', '900'))

train_shapes, holdout_shapes, _, _ = _load_shapes(NXI)
x1, _ = _build_dataset(train_shapes, holdout_shapes, copies_per_char=COPIES, S=S, device=device)
print("x1:", x1.shape, " N_BATCHES:", N_BATCHES)

n_lambda = alm_torch.n_lambda_for_alm(NXI)
model = SpectralCrossAttnFlowNetwork(
    nxi=NXI, nd=2, D=D, S=S, n_lambda=n_lambda, predict_lambda=True, n_heads=8,
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

K = int(np.ceil(np.sqrt(S)))
k_idx_full_np, Lambda_full_np = _make_ergodic_k_grid(K)
k_idx_full = torch.tensor(k_idx_full_np, dtype=torch.float32, device=device)
Lambda_full = torch.tensor(Lambda_full_np, dtype=torch.float32, device=device)
k_idx_cond = torch.tensor(k_idx_full_np[:S], dtype=torch.long, device=device)
k_idx_cond_batch = k_idx_cond.unsqueeze(0).expand(MINI_BATCH, -1, -1).contiguous()

dfl_basis = basis_torch(NXI, pts=128, deg=5, device=device)

perm = torch.randperm(x1.shape[0], device=device)
use_cuda = device.type == 'cuda'

n_total = min(N_BATCHES * MINI_BATCH, (x1.shape[0] // MINI_BATCH) * MINI_BATCH)
n_skipped = 0
n_done = 0
for i in range(0, n_total, MINI_BATCH):
    idx = perm[i:i + MINI_BATCH]
    batch_clean = x1[idx]
    batch_aug = augment_batch_torch(batch_clean)
    spec_cond = trajectory_to_spectral_batch_torch(batch_aug, k_idx_full, Lambda_full, S=S)

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda' if use_cuda else 'cpu', dtype=torch.bfloat16):
        loss = compute_spectral_cfm_loss_dfl(
            model, batch_aug, spec_cond, k_idx_cond_batch[:batch_aug.shape[0]],
            p_drop=0.1, dfl_weight=0.1, dfl_basis=dfl_basis, dfl_nxi=NXI,
            dfl_n_outer=5, dfl_inner_steps=8,
        )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    batch_idx = i // MINI_BATCH
    if torch.isfinite(grad_norm):
        opt.step()
        n_done += 1
    else:
        n_skipped += 1
        print(f"  [batch {batch_idx}] skipped (grad_norm={grad_norm.item()})")

    if batch_idx % 50 == 0:
        m = compute_spectral_cfm_loss_dfl.last_metrics
        print(f"batch {batch_idx}: loss={loss.item():.6g}  grad_norm={grad_norm.item():.6g}  "
              + "  ".join(f"{k}={v.item():.6g}" for k, v in m.items()))

    any_param_bad = any(not torch.isfinite(p).all() for p in model.parameters())
    if any_param_bad:
        print(f"  !!! model parameters corrupted at batch {batch_idx} despite skip-guard !!!")
        break

print(f"\ndone. n_done={n_done}  n_skipped={n_skipped}  "
      f"final params finite: {all(torch.isfinite(p).all().item() for p in model.parameters())}")
