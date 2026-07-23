r"""
flow_matching_cond_mpd_film_unet.py
================================
Conditional MPD (Motion Planning Diffusion) Flow Matching Network with FiLM.

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

5. FiLM (Feature-wise Linear Modulation):
   - Rather than globally adding the context vector to the tokens once, the context
     cond = time_emb(t) + shape_enc(ref_cps) is passed down into the U-Net.
   - Inside every FiLMConvResBlock, a small MLP projector maps the context to
     gamma (scale) and beta (shift).
   - Features are modulated strictly after the first GroupNorm:
     F_neu = (1 + gamma) * F + beta
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from flow_matching_patch_unet import (
    SinusoidalTimeEmbedding, OutputMLPHead,
)
from flow_matching_cond_mpd_unet import (
    MPDLayer, ShapeEncoderMPD
)


# ---------------------------------------------------------------------------
# 1. FiLM Residual Block
# ---------------------------------------------------------------------------

class FiLMConvResBlock(nn.Module):
    """
    Residual 1D-CNN block with FiLM (Feature-wise Linear Modulation).
    
    The condition vector (B, cond_dim) is projected to gamma and beta for each
    feature channel. Modulation is applied after the first convolution's GroupNorm.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        
        # 1st Conv + Norm
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        
        # 2nd Conv + Norm
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        
        self.act   = nn.SiLU()

        # 1x1 projection for the residual path when channels or stride differ
        self.residual = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        
        # FiLM Projector: context -> gamma & beta (2 * out_ch)
        self.film_proj = nn.Linear(cond_dim, out_ch * 2)
        
        # Initialize gamma and beta to 0 (identity modulation at start of training)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_ch, L)
        cond: (B, cond_dim)
        """
        # 1. Standard Conv + Normalization
        h = self.norm1(self.conv1(x))
        
        # 2. FiLM Modulation
        # Project cond -> (B, 2*out_ch) and split into gamma, beta
        film_params = self.film_proj(cond)                  # (B, 2*out_ch)
        gamma, beta = film_params.chunk(2, dim=-1)          # (B, out_ch), (B, out_ch)
        
        # Expand dims for broadcasting over temporal axis: (B, out_ch, 1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        
        h = (1 + gamma) * h + beta
        
        # 3. Activation & Second Conv
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        
        # 4. Residual Skip Connection
        return self.act(h + self.residual(x))


# ---------------------------------------------------------------------------
# 2. 1D U-Net Backbone with FiLM
# ---------------------------------------------------------------------------

class FiLMUNetBackbone(nn.Module):
    """
    1D U-Net backbone where every residual block is conditioned via FiLM.
    
    Encoder:
        Enc1: D   → D   [stride=1]
        Enc2: D   → 2D  [stride=2]
        Enc3: 2D  → 4D  [stride=2]
    Bottleneck:
        4D → 4D (two residual blocks)
    Decoder:
        Dec1: upsample + cat(skip enc2)  6D → 2D
        Dec2: upsample + cat(skip enc1)  3D → D
    """

    def __init__(self, D: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        # --- Encoder ---
        self.enc1 = FiLMConvResBlock(D,   D,   cond_dim, kernel_size, stride=1)
        self.enc2 = FiLMConvResBlock(D,   D*2, cond_dim, kernel_size, stride=2)
        self.enc3 = FiLMConvResBlock(D*2, D*4, cond_dim, kernel_size, stride=2)

        # --- Bottleneck ---
        # Using a regular ModuleList so we can pass 'cond' in manually
        self.bottleneck = nn.ModuleList([
            FiLMConvResBlock(D*4, D*4, cond_dim, kernel_size, stride=1),
            FiLMConvResBlock(D*4, D*4, cond_dim, kernel_size, stride=1),
        ])

        # --- Decoder ---
        # After upsample + concat: (D*4 + D*2) = D*6  →  D*2
        self.dec1 = FiLMConvResBlock(D*4 + D*2, D*2, cond_dim, kernel_size, stride=1)
        # After upsample + concat: (D*2 + D)  = D*3  →  D
        self.dec2 = FiLMConvResBlock(D*2 + D,   D,   cond_dim, kernel_size, stride=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, nxi)  — temporal tokens
            cond: (B, cond_dim) — conditioning context
        Returns:
            out: (B, D, nxi)
        """
        # Encoder
        e1 = self.enc1(x, cond)   # (B, D,   nxi)
        e2 = self.enc2(e1, cond)  # (B, 2D, nxi/2)
        e3 = self.enc3(e2, cond)  # (B, 4D, nxi/4)

        # Bottleneck
        b = self.bottleneck[0](e3, cond)
        b = self.bottleneck[1](b, cond)  # (B, 4D, nxi/4)

        # Decoder — upsample to match skip connection length, then concat
        b_up = F.interpolate(b,  size=e2.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([b_up, e2], dim=1), cond)          # (B, 2D, nxi/2)

        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d1_up, e1], dim=1), cond)         # (B, D,  nxi)

        return d2


# ---------------------------------------------------------------------------
# 3. Full Conditional Model
# ---------------------------------------------------------------------------

class CondMpdFiLMUNetFlowNetwork(nn.Module):
    """
    Conditional MPD UNet with FiLM modulation for shape-guided trajectory generation.

    Integrates:
    - MPD Tokenization (Spatial-Temporal Permutation + 1D Temporal Conv, kernel_size=3)
    - Sinusoidal Time Embedding + Shape Context Conditioning
    - FiLM-modulated 1D UNet Backbone
    - Output Head back to (B, H, d)
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
        
        # Backbone now expects cond_dim (D, since time_emb and shape_enc both output D)
        self.backbone  = FiLMUNetBackbone(D=D, cond_dim=D, kernel_size=kernel_size)
        self.head      = OutputMLPHead(D=D, nd=nd)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                ref_cps: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, H, d) — noisy trajectory / state sequence at flow time t
        t       : (B,)      — flow matching ODE integration time in [0, 1]
        ref_cps : (B, H, d) — target reference trajectory / control points

        Returns predicted velocity field v_theta of shape (B, H, d)
        """
        # 1. MPD Tokenization: (B, H, d) permute → (B, d, H) → Conv1d → (B, D, H)
        tokens = self.mpd_layer(x)                           # (B, D, H)

        # 2. Compute Context Vector: Time embedding + Shape context vector
        # (Both output D-dimensional vectors, which are summed)
        cond = self.time_emb(t) + self.shape_enc(ref_cps)    # (B, D)

        # 3. U-Net Backbone processing with FiLM modulation
        tokens = self.backbone(tokens, cond)                 # (B, D, H)

        # 4. Output Head projection back to (B, H, d)
        return self.head(tokens)                             # (B, H, d)
