r"""
flow_matching_tsvec_unet.py
================================
TSVEC (Time, Space, Velocity, Ergodic Context) Flow Matching Network.
Phase 1: Unpooled MPD Shape Encoding + Self & Cross Attention in Bottleneck.

This architecture builds on the FiLM-UNet but introduces Transformer-like capabilities:
1. Self-Attention: Allows spatially distant B-Spline tokens in the bottleneck to 
   communicate globally, perceiving macro-topology (e.g. loops).
2. Cross-Attention: The U-Net tokens (Queries) actively read features from an 
   uncompressed, sequence-based environmental context map (Keys/Values).
   In Phase 1, this context map is the (B, H, D) representation of the target shape.

Key stability mechanisms (DiT-style):
- Sinusoidal Positional Encodings on Q/K/V to preserve B-Spline temporal ordering.
- Zero-initialized out_proj so Attention starts as identity (AdaLN-Zero principle).
"""

import math
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
from flow_matching_cond_mpd_unet import MPDLayer
from flow_matching_cond_mpd_film_unet import FiLMConvResBlock


# ---------------------------------------------------------------------------
# 0. Positional Encoding (sinusoidal, for sequence tokens)
# ---------------------------------------------------------------------------

class SequencePositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding for 1D sequences.
    Creates a (1, max_len, embed_dim) buffer that is added to token sequences
    so that Attention preserves the temporal ordering of B-Spline control points.
    """
    def __init__(self, embed_dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter, but moves with .to(device))
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)  →  (B, L, C) with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


# ---------------------------------------------------------------------------
# 1. Unpooled Shape Encoder (Phase 1 TargetDensityEncoder)
# ---------------------------------------------------------------------------

class ShapeEncoderMPD_Unpooled(nn.Module):
    """
    Encodes reference control points into an uncompressed sequence of features.
    Unlike the original ShapeEncoderMPD, this does NOT use Global Average Pooling.
    Outputs: (B, H, D) to be used as Keys and Values in Cross-Attention.
    """
    def __init__(self, nd: int, D: int):
        super().__init__()
        self.mpd_layer = MPDLayer(nd=nd, D=D)
        
        # Pointwise projection applied to each token in the sequence
        self.proj = nn.Sequential(
            nn.Linear(D, D * 2),
            nn.SiLU(),
            nn.Linear(D * 2, D),
        )

    def forward(self, ref_cps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_cps: (B, H, nd)
        Returns:
            context_sequence: (B, H, D)
        """
        tokens = self.mpd_layer(ref_cps)         # (B, D, H)
        tokens_flat = tokens.permute(0, 2, 1)    # (B, H, D)
        return self.proj(tokens_flat)            # (B, H, D)


# ---------------------------------------------------------------------------
# 2. Attention Blocks (with PE + Zero-Init)
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """
    Self-Attention over the spatial sequence dimension L.
    
    Stability features:
    - Sinusoidal Positional Encoding added to Q/K/V before attention,
      so tokens retain their temporal B-Spline ordering in the "Conference Room".
    - out_proj initialized to zero (DiT AdaLN-Zero principle), so the block
      outputs exactly 0.0 at epoch 0, preserving the CNN's kinematic inductive bias.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.pe = SequencePositionalEncoding(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Zero-init: Attention starts as identity → no initial shock to residual stream
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        """
        x_flat = x.permute(0, 2, 1)              # (B, L, C)
        x_norm = self.norm(x_flat)
        
        # Add positional encoding so tokens know their B-Spline position
        x_pos = self.pe(x_norm)
        
        # Self-attention: Q = K = V = position-aware tokens
        attn_out, _ = self.mha(query=x_pos, key=x_pos, value=x_pos)
        
        # Residual connection and permute back to (B, C, L)
        return x + attn_out.permute(0, 2, 1)


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention where U-Net tokens (Queries) read from environmental context (Keys/Values).
    
    Stability features:
    - Separate Positional Encodings for Queries and Keys/Values.
    - out_proj zero-initialized for stable gradient flow at init.
    """
    def __init__(self, embed_dim: int, kv_dim: int, num_heads: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        
        # Positional encodings: separate for query sequence and KV sequence
        self.pe_q = SequencePositionalEncoding(embed_dim)
        self.pe_kv = SequencePositionalEncoding(kv_dim)
        
        # PyTorch MHA supports different dimension for K/V via kdim and vdim
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, kdim=kv_dim, vdim=kv_dim, batch_first=True
        )
        
        # Zero-init: Cross-Attention starts as identity
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        x  : (B, C, L)      - Queries from the U-Net backbone
        kv : (B, K_len, kv_dim) - Context features (Keys/Values)
        """
        x_flat = x.permute(0, 2, 1)              # (B, L, C)
        q = self.pe_q(self.norm_q(x_flat))        # Norm → PE → Query
        kv_norm = self.pe_kv(self.norm_kv(kv))    # Norm → PE → Key/Value
        
        attn_out, _ = self.mha(query=q, key=kv_norm, value=kv_norm)
        
        return x + attn_out.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# 3. TSVEC Backbone (FiLM + Attention Bottleneck)
# ---------------------------------------------------------------------------

class TSVECUNetBackbone(nn.Module):
    """
    1D U-Net backbone using FiLM for time conditioning, and Self/Cross attention 
    in the bottleneck for spatial/ergodic context.
    """
    def __init__(self, D: int, cond_dim: int, kv_dim: int, kernel_size: int = 3):
        super().__init__()
        # --- Encoder ---
        self.enc1 = FiLMConvResBlock(D,   D,   cond_dim, kernel_size, stride=1)
        self.enc2 = FiLMConvResBlock(D,   D*2, cond_dim, kernel_size, stride=2)
        self.enc3 = FiLMConvResBlock(D*2, D*4, cond_dim, kernel_size, stride=2)

        # --- Bottleneck (The Conference Room) ---
        self.bottleneck_conv1 = FiLMConvResBlock(D*4, D*4, cond_dim, kernel_size, stride=1)
        
        # Self-Attention: Tokens communicate with each other globally
        self.bottleneck_self_attn = SelfAttentionBlock(embed_dim=D*4, num_heads=4)
        
        # Cross-Attention: Tokens read from the uncompressed environment map
        self.bottleneck_cross_attn = CrossAttentionBlock(embed_dim=D*4, kv_dim=kv_dim, num_heads=4)
        
        self.bottleneck_conv2 = FiLMConvResBlock(D*4, D*4, cond_dim, kernel_size, stride=1)

        # --- Decoder ---
        self.dec1 = FiLMConvResBlock(D*4 + D*2, D*2, cond_dim, kernel_size, stride=1)
        self.dec2 = FiLMConvResBlock(D*2 + D,   D,   cond_dim, kernel_size, stride=1)

    def forward(self, x: torch.Tensor, cond_film: torch.Tensor, cond_kv: torch.Tensor) -> torch.Tensor:
        """
        x         : (B, D, nxi)
        cond_film : (B, cond_dim) -> For FiLM blocks (time)
        cond_kv   : (B, H, kv_dim) -> For Cross-Attention (shape/ergodic map)
        """
        # Encoder
        e1 = self.enc1(x, cond_film)   # (B, D,   nxi)
        e2 = self.enc2(e1, cond_film)  # (B, 2D, nxi/2)
        e3 = self.enc3(e2, cond_film)  # (B, 4D, nxi/4)

        # Bottleneck (The Conference Room)
        b = self.bottleneck_conv1(e3, cond_film)      # 1D Kinematic extraction
        b = self.bottleneck_self_attn(b)              # Global Topology (Self-Attn + PE)
        b = self.bottleneck_cross_attn(b, cond_kv)    # Environment Reading (Cross-Attn + PE)
        b = self.bottleneck_conv2(b, cond_film)       # 1D Kinematic extraction

        # Decoder
        b_up = F.interpolate(b, size=e2.shape[-1], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([b_up, e2], dim=1), cond_film)
        
        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([d1_up, e1], dim=1), cond_film)

        return d2


# ---------------------------------------------------------------------------
# 4. Full TSVEC Model Wrapper
# ---------------------------------------------------------------------------

class TSVECFlowNetwork(nn.Module):
    """
    The hybrid generator architecture:
    - FiLM (Time-Conditioning)
    - Cross-Attention (Spatial-Conditioning) with Positional Encoding
    - Self-Attention (Global topology) with Positional Encoding
    - 1D Convolutions (Kinematic Ck-smoothness)
    - Zero-Initialized Attention out_proj (DiT AdaLN-Zero stability)
    """
    def __init__(self, nxi: int = 20, nd: int = 2, D: int = 256, kernel_size: int = 3):
        super().__init__()
        self.nxi = nxi
        self.nd  = nd
        self.D   = D

        self.mpd_layer = MPDLayer(nd=nd, D=D, kernel_size=kernel_size)
        
        # Condition 1: Time for FiLM
        self.time_emb  = SinusoidalTimeEmbedding(D=D)
        
        # Condition 2: Target shape/density for Cross-Attention (Phase 1 unpooled seq)
        self.shape_enc_unpooled = ShapeEncoderMPD_Unpooled(nd=nd, D=D)
        
        # The Hybrid Backbone
        self.backbone = TSVECUNetBackbone(D=D, cond_dim=D, kv_dim=D, kernel_size=kernel_size)
        self.head     = OutputMLPHead(D=D, nd=nd)

    def forward(self, x: torch.Tensor, t: torch.Tensor, ref_cps: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, H, d) — noisy trajectory
        t       : (B,)      — flow matching integration time
        ref_cps : (B, H, d) — reference trajectory target
        """
        # 1. MPD Tokenization
        tokens = self.mpd_layer(x)                             # (B, D, H)

        # 2. Extract Conditions
        cond_film = self.time_emb(t)                           # (B, D)
        cond_kv   = self.shape_enc_unpooled(ref_cps)           # (B, H, D)

        # 3. Backbone (FiLM + Attention)
        tokens = self.backbone(tokens, cond_film, cond_kv)     # (B, D, H)

        # 4. Output Head projection
        return self.head(tokens)                               # (B, H, d)
