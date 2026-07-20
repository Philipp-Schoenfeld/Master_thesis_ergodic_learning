"""
flow_matching_cond_patch_unet.py
================================
Conditional PatchUNet flow matching network.

Conditioning signal: reference control points (the target shape),
encoded by a ShapeEncoder into a D-dim context vector and added to
the sinusoidal time embedding before injecting into every token.

    x (B, nxi, nd)       — noisy control points at flow time t
    t (B,)               — ODE integration time in [0, 1]
    ref_cps (B, nxi, nd) — reference (target) control points

    → vθ (B, nxi, nd)    — predicted velocity field
"""

import os
import sys
import torch
import torch.nn as nn

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from flow_matching_patch_unet import (
    PatchifyLayer, SinusoidalTimeEmbedding, UNetBackbone, OutputMLPHead,
)


# ---------------------------------------------------------------------------
# Shape Encoder
# ---------------------------------------------------------------------------

class ShapeEncoder(nn.Module):
    """
    Encode reference control points → D-dim shape context vector.

    Pipeline:
        (B, nxi, nd)  →  PatchifyLayer  →  (B, D, nxi)
                      →  global avg-pool →  (B, D)
                      →  MLP            →  (B, D)
    """

    def __init__(self, nd: int, D: int):
        super().__init__()
        self.patch = PatchifyLayer(nd=nd, D=D)
        self.proj  = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.SiLU(),
            nn.Linear(D * 2, D),
        )

    def forward(self, ref_cps: torch.Tensor) -> torch.Tensor:
        """ref_cps: (B, nxi, nd)  →  (B, D)"""
        tokens = self.patch(ref_cps)     # (B, D, nxi)
        pooled = tokens.mean(dim=-1)     # (B, D)
        return self.proj(pooled)         # (B, D)


# ---------------------------------------------------------------------------
# Full Conditional Model
# ---------------------------------------------------------------------------

class CondPatchUNetFlowNetwork(nn.Module):
    """
    Conditional PatchUNet for shape-guided trajectory generation.

    The reference control points encode *which shape* to generate.
    Combined conditioning:
        cond = SinusoidalTimeEmbedding(t) + ShapeEncoder(ref_cps)
        tokens += cond[:, :, None]   — broadcast over nxi positions
    """

    def __init__(self, nxi: int = 20, nd: int = 2,
                 D: int = 256, kernel_size: int = 3):
        super().__init__()
        self.nxi = nxi
        self.nd  = nd
        self.D   = D

        self.patchify  = PatchifyLayer(nd=nd, D=D)
        self.time_emb  = SinusoidalTimeEmbedding(D=D)
        self.shape_enc = ShapeEncoder(nd=nd, D=D)
        self.backbone  = UNetBackbone(D=D, kernel_size=kernel_size)
        self.head      = OutputMLPHead(D=D, nd=nd)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                ref_cps: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, nxi, nd)
        t       : (B,)
        ref_cps : (B, nxi, nd) — target shape reference
        Returns vθ (B, nxi, nd)
        """
        tokens = self.patchify(x)                        # (B, D, nxi)
        cond   = self.time_emb(t) + self.shape_enc(ref_cps)  # (B, D)
        tokens = tokens + cond[:, :, None]               # (B, D, nxi)
        tokens = self.backbone(tokens)                   # (B, D, nxi)
        return  self.head(tokens)                        # (B, nxi, nd)


# ---------------------------------------------------------------------------
# Training utility
# ---------------------------------------------------------------------------

def compute_cond_cfm_loss(model: nn.Module,
                          x1_batch: torch.Tensor,
                          ref_cps_batch: torch.Tensor) -> torch.Tensor:
    """
    Conditional CFM loss (straight OT path).

    x1_batch      : (B, nxi, nd) — clean target trajectories
    ref_cps_batch : (B, nxi, nd) — per-sample reference CPs (condition)
    """
    B, nxi, nd = x1_batch.shape
    device = x1_batch.device

    x0    = torch.randn_like(x1_batch)
    t     = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)
    xt    = (1 - t_exp) * x0 + t_exp * x1_batch
    ut    = x1_batch - x0

    vt = model(xt, t, ref_cps_batch)
    return torch.mean((vt - ut) ** 2)


# ---------------------------------------------------------------------------
# Generation utility
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_cond_trajectories(
    model: nn.Module,
    ref_cps: torch.Tensor,     # (nxi, nd)  or  (num_samples, nxi, nd)
    num_samples: int = 1,
    nxi: int = 20,
    nd:  int = 2,
    steps: int = 100,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Euler integration conditioned on ref_cps.

    Returns (num_samples, nxi, nd) generated control points.
    """
    model.eval()
    if ref_cps.ndim == 2:                              # (nxi, nd) → broadcast
        ref_cps = ref_cps.unsqueeze(0).expand(num_samples, -1, -1)
    ref_cps = ref_cps.to(device)

    x  = torch.randn(num_samples, nxi, nd, device=device)
    dt = 1.0 / steps

    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=device)
        x = x + model(x, t, ref_cps) * dt

    return x   # (num_samples, nxi, nd)
