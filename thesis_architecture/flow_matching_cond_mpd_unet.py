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

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from flow_matching_patch_unet import (
    SinusoidalTimeEmbedding, UNetBackbone, OutputMLPHead,
)


# ---------------------------------------------------------------------------
# MPD Tokenization Layer (1D Temporal Convolution)
# ---------------------------------------------------------------------------

class MPDLayer(nn.Module):
    """
    MPD Tokenisation via 1D Temporal CNN.

    - Input:  x of shape (B, H, d)  [Batch, Horizon nxi, State Dimension nd]
    - Permute: (B, d, H) — treating state dimension d as input channels and H as sequence length.
    - Conv1d: kernel_size=3 extracts local kinematic features between neighboring
              waypoints (t-1, t, t+1) and projects d channels → D embedding.
    - Output: tokens of shape (B, D, H) ready for U-Net backbone.
    """
    def __init__(self, nd: int, D: int, kernel_size: int = 3):
        super().__init__()
        self.nd = nd
        self.D  = D
        # kernel_size=3, padding=1 keeps sequence length H (nxi) identical
        self.conv = nn.Conv1d(
            in_channels=nd,
            out_channels=D,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, d)  — Raw trajectory tensor (Batch, Horizon nxi, State Dimension nd)
        Returns:
            tokens: (B, D, H)  — D-dim token embedding per waypoint
        """
        # Step 1 & 2: Permute (B, H, d) → (B, d, H) for Conv1d temporal convolution
        h = x.permute(0, 2, 1)                           # (B, d, H)

        # Step 3: 1D Temporal Convolution (d → D, window over H)
        h = self.conv(h)                                 # (B, D, H)

        # Step 4: LayerNorm over embedding dimension D
        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)  # (B, D, H)
        return h


# ---------------------------------------------------------------------------
# Shape Encoder (MPD Version)
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
        """ref_cps: (B, H, d)  →  (B, D)"""
        tokens = self.mpd_layer(ref_cps)     # (B, D, H)
        pooled = tokens.mean(dim=-1)         # Global temporal pooling over H → (B, D)
        return self.proj(pooled)             # Context vector (B, D)


# ---------------------------------------------------------------------------
# Full Conditional Model
# ---------------------------------------------------------------------------

class CondMpdUNetFlowNetwork(nn.Module):
    """
    Conditional MPD UNet for shape-guided / motion-planning trajectory generation.

    Integrates:
    - MPD Tokenization (Spatial-Temporal Permutation + 1D Temporal Conv, kernel_size=3)
    - Sinusoidal Time Embedding
    - Shape Context Conditioning via ShapeEncoderMPD
    - 1D UNet Backbone
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
        self.backbone  = UNetBackbone(D=D, kernel_size=kernel_size)
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

        # 2. Conditioning: Time embedding + Shape context vector
        cond   = self.time_emb(t) + self.shape_enc(ref_cps)  # (B, D)

        # 3. Inject conditioning into tokens along temporal axis H
        tokens = tokens + cond[:, :, None]                   # (B, D, H)

        # 4. U-Net Backbone processing
        tokens = self.backbone(tokens)                       # (B, D, H)

        # 5. Output Head projection back to (B, H, d)
        return self.head(tokens)                            # (B, H, d)

