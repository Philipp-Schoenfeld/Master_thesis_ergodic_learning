r"""
flow_matching_cond_mpd_unet_char_film_cfg.py
====================================
Conditional MPD (Motion Planning Diffusion) Flow Matching Network.

- FiLM architecture: Condition scales/shifts U-Net blocks.
- Classifier-Free Guidance (CFG): Supports condition dropout for extrapolation.
- Positional Encodings: Waypoints are position-aware for better sequence logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Sinusoidal Time Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar t ∈ [0, 1]  →  R^D via sinusoidal positional encoding."""

    def __init__(self, D: int):
        super().__init__()
        assert D % 2 == 0, "D must be even for sinusoidal embedding."
        half = D // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * -(torch.log(torch.tensor(10_000.0)) / (half - 1))
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(D, D * 2), nn.SiLU(), nn.Linear(D * 2, D),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        args = t[:, None] * self.freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# 2. 1D U-Net building blocks (with FiLM)
# ---------------------------------------------------------------------------

class ConvResBlock(nn.Module):
    """Residual 1D-CNN block: Conv1d + GroupNorm + FiLM + SiLU × 2, with skip."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        
        # FiLM projection: outputs Scale (gamma) and Shift (beta)
        self.film_proj = nn.Linear(cond_dim, out_ch * 2)
        # Init to zero so it acts as identity at the start of training (gamma=0, beta=0)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act   = nn.SiLU()
        self.residual = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
            if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        
        # Apply FiLM after norm, before activation (standard adaGN pattern from DiT)
        if cond is not None:
            film_params = self.film_proj(cond).unsqueeze(-1)  # (B, 2 * out_ch, 1)
            gamma, beta = film_params.chunk(2, dim=1)         # (B, out_ch, 1)
            h = h * (1.0 + gamma) + beta
        
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.residual(x))


class UNetBackbone(nn.Module):
    """Pure 1D U-Net backbone operating on token sequences (B, D, nξ)."""

    def __init__(self, D: int, kernel_size: int = 3):
        super().__init__()
        self.enc1 = ConvResBlock(D,   D,   cond_dim=D, kernel_size=kernel_size, stride=1)
        self.enc2 = ConvResBlock(D,   D*2, cond_dim=D, kernel_size=kernel_size, stride=2)
        self.enc3 = ConvResBlock(D*2, D*4, cond_dim=D, kernel_size=kernel_size, stride=2)
        
        # Bottleneck unrolled to pass condition to both
        self.bot1 = ConvResBlock(D*4, D*4, cond_dim=D, kernel_size=kernel_size, stride=1)
        self.bot2 = ConvResBlock(D*4, D*4, cond_dim=D, kernel_size=kernel_size, stride=1)
        
        self.dec1 = ConvResBlock(D*4 + D*2, D*2, cond_dim=D, kernel_size=kernel_size, stride=1)
        self.dec2 = ConvResBlock(D*2 + D,   D,   cond_dim=D, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        e1 = self.enc1(x, cond)
        e2 = self.enc2(e1, cond)
        e3 = self.enc3(e2, cond)
        
        b = self.bot1(e3, cond)
        b = self.bot2(b, cond)
        
        b_up  = F.interpolate(b,  size=e2.shape[-1], mode='linear', align_corners=False)
        d1    = self.dec1(torch.cat([b_up, e2], dim=1), cond)
        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d2    = self.dec2(torch.cat([d1_up, e1], dim=1), cond)
        return d2


# ---------------------------------------------------------------------------
# 3. Output MLP Head
# ---------------------------------------------------------------------------

class OutputMLPHead(nn.Module):
    """Projects D-dimensional tokens back to nd-dimensional velocity vectors."""

    def __init__(self, D: int, nd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D), nn.LayerNorm(D), nn.SiLU(), nn.Linear(D, nd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1))


# ---------------------------------------------------------------------------
# 4. MPD Tokenization Layer (1D Temporal Convolution)
# ---------------------------------------------------------------------------

class MPDLayer(nn.Module):
    """MPD Tokenisation via 1D Temporal CNN."""

    def __init__(self, nd: int, D: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(nd, D, kernel_size, stride=1,
                              padding=kernel_size // 2)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.permute(0, 2, 1))                       # (B, D, H)
        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)      # (B, D, H)
        return h


# ---------------------------------------------------------------------------
# 5. Shape Encoder (Trajectory-based — generalises to unseen shapes)
# ---------------------------------------------------------------------------

class ShapeEncoderMPD(nn.Module):
    """
    Encode a reference trajectory (B, H, d) → D-dim shape context vector
    via 1D temporal convolution + global mean pooling + MLP projection.
    Now includes Positional Encoding.
    """

    def __init__(self, nd: int, D: int, nxi: int = 20):
        super().__init__()
        self.mpd_layer = MPDLayer(nd=nd, D=D)
        self.pos_emb = nn.Parameter(torch.randn(1, D, nxi) * 0.02)
        
        self.proj = nn.Sequential(
            nn.Linear(D, D * 2), nn.SiLU(), nn.Linear(D * 2, D),
        )

    def forward(self, ref_cps: torch.Tensor) -> torch.Tensor:
        tokens = self.mpd_layer(ref_cps)      # (B, D, H)
        tokens = tokens + self.pos_emb        # Inject sequence order knowledge
        pooled = tokens.mean(dim=-1)          # (B, D)
        return self.proj(pooled)              # (B, D)


# ---------------------------------------------------------------------------
# 6. Full Conditional Model (FiLM + CFG + PosEnc)
# ---------------------------------------------------------------------------

class CondMpdUNetFlowNetwork(nn.Module):
    """
    Conditional MPD UNet for trajectory generation.
    Supports Classifier-Free Guidance (CFG).
    """

    def __init__(self, nxi: int = 20, nd: int = 2,
                 D: int = 128, kernel_size: int = 3):
        super().__init__()
        self.nxi = nxi
        self.nd  = nd
        self.D   = D

        self.mpd_layer = MPDLayer(nd=nd, D=D, kernel_size=kernel_size)
        self.pos_emb   = nn.Parameter(torch.randn(1, D, nxi) * 0.02)
        
        self.time_emb  = SinusoidalTimeEmbedding(D=D)
        self.shape_enc = ShapeEncoderMPD(nd=nd, D=D, nxi=nxi)
        
        # Null shape embedding for CFG dropout
        self.null_shape_emb = nn.Parameter(torch.zeros(D))
        
        self.backbone  = UNetBackbone(D=D, kernel_size=kernel_size)
        self.head      = OutputMLPHead(D=D, nd=nd)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                ref_cps: torch.Tensor, cond_drop_mask: torch.Tensor = None) -> torch.Tensor:
        tokens = self.mpd_layer(x)                               # (B, D, H)
        tokens = tokens + self.pos_emb                           # inject positional info
        
        shape_cond = self.shape_enc(ref_cps)                     # (B, D)
        
        # Apply CFG Dropout
        if cond_drop_mask is not None:
            mask = cond_drop_mask.view(-1, 1).to(shape_cond.dtype)
            shape_cond = shape_cond * (1.0 - mask) + self.null_shape_emb * mask
            
        cond = self.time_emb(t) + shape_cond                     # (B, D)
        tokens = self.backbone(tokens, cond)                     # (B, D, H)  <- FiLM injection
        return self.head(tokens)                                 # (B, H, d)


# ---------------------------------------------------------------------------
# Training utility (with CFG Dropout)
# ---------------------------------------------------------------------------

def compute_cond_cfm_loss(model: nn.Module,
                          x1_batch: torch.Tensor,
                          ref_cps_batch: torch.Tensor,
                          p_drop: float = 0.1) -> torch.Tensor:
    B = x1_batch.shape[0]
    device = x1_batch.device
    x0    = torch.randn_like(x1_batch)
    t     = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)
    xt    = (1 - t_exp) * x0 + t_exp * x1_batch
    ut    = x1_batch - x0
    
    # 10% chance to drop the condition completely
    cond_drop_mask = (torch.rand(B, device=device) < p_drop)
    
    vt = model(xt, t, ref_cps_batch, cond_drop_mask=cond_drop_mask)
    return torch.mean((vt - ut) ** 2)


# ---------------------------------------------------------------------------
# Generation utility (with CFG Extrapolation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_cond_trajectories(
    model: nn.Module,
    ref_cps: torch.Tensor,          # (nxi, nd) or (num_samples, nxi, nd)
    num_samples: int = 1,
    nxi: int = 20,
    nd:  int = 2,
    steps: int = 100,
    device: str = 'cpu',
    cfg_weight: float = 2.0,        # CFG Extrapolation strength
) -> torch.Tensor:
    model.eval()
    if ref_cps.ndim == 2:
        ref_cps = ref_cps.unsqueeze(0).expand(num_samples, -1, -1).contiguous()
    ref_cps = ref_cps.to(device)
    x  = torch.randn(num_samples, nxi, nd, device=device)
    dt = 1.0 / steps
    
    # Batch mask: top half = Conditioned (False), bottom half = Null (True)
    mask_batch = torch.cat([
        torch.zeros(num_samples, dtype=torch.bool, device=device),
        torch.ones(num_samples,  dtype=torch.bool, device=device)
    ], dim=0)
    ref_batch = torch.cat([ref_cps, ref_cps], dim=0)
    
    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=device)
        t_batch = torch.cat([t, t], dim=0)
        x_batch = torch.cat([x, x], dim=0)
        
        # Run backbone once with doubled batch size for maximum efficiency
        v_batch = model(x_batch, t_batch, ref_batch, cond_drop_mask=mask_batch)
        v_cond, v_null = v_batch.chunk(2, dim=0)
        
        # Classifier-Free Guidance Extrapolation
        v_cfg = v_null + cfg_weight * (v_cond - v_null)
        
        x = x + v_cfg * dt
        
    return x
