r"""
flow_matching_cond_mpd_unet.py
================================
Conditional MPD (Motion Planning Diffusion) Flow Matching Network.

Data Representation & Architectural Breakdown:
----------------------------------------------
1. State Representation (Dimensions):
   At any single point in time, the state of the robot is represented as a single vector
   concatenating position and velocity:
       s = [q^T, \dot{q}^T]^T \in \mathbb{R}^d
   where d is the total state dimension (e.g., d = 14 for a 7-DOF arm: 7 positions + 7 velocities).

2. Trajectory Matrix (Horizon H / nxi):
   A trajectory is a discrete-time sequence of these states over a fixed horizon:
       \tau = (s_0, ..., s_{H-1}) \in \mathbb{R}^{H \times d}
   where H (nxi) is the horizon length / number of waypoints.

3. Deep Learning Tensor Shape (Batching B):
   Raw input tensor provided by dataset has shape:
       x \in (B, H, d)  /  (B, nxi, nd)

4. U-Net Integration (Spatial-Temporal Permutation & 1D Temporal Conv):
   Standard Conv1d expects (Batch, Channels, Sequence_Length).
   - Permute: (B, H, d) → (B, d, H) where state dimension d acts as input channels,
     and horizon H acts as the temporal/sequence dimension.
   - Temporal Convolution: A 1D Convolution with kernel_size=3 (stride=1, padding=1)
     slides across the temporal axis H, extracting local kinematic features between
     neighboring waypoints (t-1, t, t+1) and projecting d state channels to embedding D.
   - Output to Backbone: Tensor of shape (B, D, H) is fed into the U-Net backbone.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Sinusoidal Time Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """
    Maps scalar t ∈ [0, 1]  →  R^D via sinusoidal positional encoding.
    """
    def __init__(self, D: int):
        super().__init__()
        assert D % 2 == 0, "D must be even for sinusoidal embedding."
        half = D // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * -(torch.log(torch.tensor(10_000.0)) / (half - 1))
        )
        self.register_buffer("freqs", freqs)   # (half,)
        self.proj = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.SiLU(),
            nn.Linear(D * 2, D),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)                                      # (B,)
        args = t[:, None] * self.freqs[None, :]             # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, D)
        return self.proj(emb)                               # (B, D)


# ---------------------------------------------------------------------------
# 2. 1D U-Net building blocks
# ---------------------------------------------------------------------------

class ConvResBlock(nn.Module):
    """
    Residual 1D-CNN block: two Conv1d + GroupNorm + SiLU layers with a
    residual skip.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act   = nn.SiLU()

        self.residual = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.residual(x))


class UNetBackbone(nn.Module):
    """
    Pure 1D U-Net backbone operating on token sequences (B, D, nξ).
    """
    def __init__(self, D: int, kernel_size: int = 3):
        super().__init__()
        self.enc1 = ConvResBlock(D,   D,   kernel_size, stride=1)
        self.enc2 = ConvResBlock(D,   D*2, kernel_size, stride=2)
        self.enc3 = ConvResBlock(D*2, D*4, kernel_size, stride=2)

        self.bottleneck = nn.Sequential(
            ConvResBlock(D*4, D*4, kernel_size, stride=1),
            ConvResBlock(D*4, D*4, kernel_size, stride=1),
        )

        self.dec1 = ConvResBlock(D*4 + D*2, D*2, kernel_size, stride=1)
        self.dec2 = ConvResBlock(D*2 + D,   D,   kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)   # (B, D,   nξ)
        e2 = self.enc2(e1)  # (B, 2D, nξ/2)
        e3 = self.enc3(e2)  # (B, 4D, nξ/4)

        b = self.bottleneck(e3)  # (B, 4D, nξ/4)

        b_up = F.interpolate(b,  size=e2.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([b_up, e2], dim=1))          # (B, 2D, nξ/2)

        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d1_up, e1], dim=1))         # (B, D,  nξ)

        return d2


# ---------------------------------------------------------------------------
# 3. Output MLP Head
# ---------------------------------------------------------------------------

class OutputMLPHead(nn.Module):
    """
    Projects D-dimensional tokens back to nd-dimensional velocity vectors.
    """
    def __init__(self, D: int, nd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.SiLU(),
            nn.Linear(D, nd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1))   # (B, nξ, nd)


# ---------------------------------------------------------------------------
# 4. MPD Tokenization Layer (1D Temporal Convolution)
# ---------------------------------------------------------------------------

class MPDLayer(nn.Module):
    """
    MPD Tokenisation via 1D Temporal CNN.
    """
    def __init__(self, nd: int, D: int, kernel_size: int = 3):
        super().__init__()
        self.nd = nd
        self.D  = D
        self.conv = nn.Conv1d(
            in_channels=nd,
            out_channels=D,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 1)                           # (B, d, H)
        h = self.conv(h)                                 # (B, D, H)
        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)  # (B, D, H)
        return h


# ---------------------------------------------------------------------------
# 5. Shape Encoder (MPD Version)
# ---------------------------------------------------------------------------

class ShapeEncoderMPD(nn.Module):
    """
    Encode reference control points / target trajectory (B, H, d) → D-dim shape context vector
    using the MPD 1D temporal convolution layer.
    """
    def __init__(self, nd: int, D: int):
        super().__init__()
        self.mpd_layer = MPDLayer(nd=nd, D=D)
        self.proj  = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.SiLU(),
            nn.Linear(D * 2, D),
        )

    def forward(self, ref_cps: torch.Tensor) -> torch.Tensor:
        tokens = self.mpd_layer(ref_cps)     # (B, D, H)
        pooled = tokens.mean(dim=-1)         # Global temporal pooling over H → (B, D)
        return self.proj(pooled)             # Context vector (B, D)


# ---------------------------------------------------------------------------
# 6. Full Conditional Model
# ---------------------------------------------------------------------------

class CondMpdUNetFlowNetwork(nn.Module):
    """
    Conditional MPD UNet for shape-guided / motion-planning trajectory generation.
    """
    def __init__(self, nxi: int = 20, nd: int = 2,
                 D: int = 256, kernel_size: int = 3):
        super().__init__()
        self.nxi = nxi  # Horizon H
        self.nd  = nd   # State dimension d
        self.D   = D    # Embedding dimension D

        self.mpd_layer = MPDLayer(nd=nd, D=D, kernel_size=kernel_size)
        self.time_emb  = SinusoidalTimeEmbedding(D=D)
        self.shape_enc = ShapeEncoderMPD(nd=nd, D=D)
        self.backbone  = UNetBackbone(D=D, kernel_size=kernel_size)
        self.head      = OutputMLPHead(D=D, nd=nd)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                ref_cps: torch.Tensor) -> torch.Tensor:
        # 1. MPD Tokenization: (B, H, d) permute → (B, d, H) → Conv1d → (B, D, H)
        tokens = self.mpd_layer(x)                           # (B, D, H)

        # 2. Conditioning: Time embedding + Shape context vector
        cond   = self.time_emb(t) + self.shape_enc(ref_cps)  # (B, D)

        # 3. Inject conditioning into tokens along temporal axis H
        tokens = tokens + cond[:, :, None]                   # (B, D, H)

        # 4. U-Net Backbone processing
        tokens = self.backbone(tokens)                       # (B, D, H)

        # 5. Output Head projection back to (B, H, d)
        return self.head(tokens)                            # (B, H, d)


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
