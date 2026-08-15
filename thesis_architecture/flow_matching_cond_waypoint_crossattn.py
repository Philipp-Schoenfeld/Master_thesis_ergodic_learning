r"""
flow_matching_cond_waypoint_crossattn.py
========================================
Conditional Flow Matching Network with:
- Waypoint Conditioning via WaypointTokenizer (replaces SpectralTokenizer)
- Cross-Attention in the U-Net bottleneck
- FiLM for time conditioning
- CFG support (condition dropout on waypoint tokens)
- Dual output head: Flow velocity + optional Lagrange multipliers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Sinusoidal Time Embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar t in [0, 1] -> R^D via sinusoidal positional encoding."""

    def __init__(self, D: int):
        super().__init__()
        assert D % 2 == 0, "D must be even for sinusoidal embedding."
        half = D // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * -(math.log(10_000.0) / (half - 1))
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(D, D * 2), nn.SiLU(), nn.Linear(D * 2, D),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1)
        args = t[:, None] * self.freqs[None, :]                 # (B, D//2)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)        # (B, D)
        return self.proj(emb)                                    # (B, D)


# ---------------------------------------------------------------------------
# 2. WaypointTokenizer
# ---------------------------------------------------------------------------

class WaypointTokenizer(nn.Module):
    """
    Translates raw trajectory waypoints into tokens for cross-attention.
    Uses a 1D Convolution to capture local geometric context (curvature, direction),
    followed by an MLP to refine the features.
    """

    def __init__(self, D: int, nd: int = 2, kernel_size: int = 5):
        super().__init__()
        self.D = D
        # 1D Convolution to look at neighboring waypoints
        self.conv = nn.Conv1d(nd, D, kernel_size, padding=kernel_size // 2)
        self.conv_norm = nn.LayerNorm(D)
        
        # Project the local features deeper
        self.mlp = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D),
        )
        self.out_norm = nn.LayerNorm(D)

    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        """
        waypoints: (B, nxi, 2) — coordinate waypoints
        Returns:   (B, nxi, D) waypoint tokens
        """
        # Conv expects (B, Channels, Length)
        h = waypoints.permute(0, 2, 1)                               # (B, nd, nxi)
        h = self.conv(h)                                             # (B, D, nxi)
        h = h.permute(0, 2, 1)                                       # (B, nxi, D)
        h = F.silu(self.conv_norm(h))
        
        tokens = self.mlp(h)                                         # (B, nxi, D)
        tokens = self.out_norm(tokens)                               # (B, nxi, D)
        return tokens


# ---------------------------------------------------------------------------
# 3. 1D U-Net building blocks (with FiLM for time conditioning)
# ---------------------------------------------------------------------------

class ConvResBlock(nn.Module):
    """Residual 1D-CNN block: Conv1d + GroupNorm + FiLM + SiLU x 2, with skip."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

        # FiLM projection for TIME conditioning (Scale + Shift)
        self.film_proj = nn.Linear(cond_dim, out_ch * 2)
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
        h = self.norm1(self.conv1(x))                            # (B, out_ch, L)

        if cond is not None:
            film_params = self.film_proj(cond).unsqueeze(-1)     # (B, 2*out_ch, 1)
            gamma, beta = film_params.chunk(2, dim=1)            # (B, out_ch, 1) each
            h = h * (1.0 + gamma) + beta

        h = self.act(h)
        h = self.norm2(self.conv2(h))                            # (B, out_ch, L)
        return self.act(h + self.residual(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, n_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=q_dim, num_heads=n_heads, batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(q_dim)

        self.kv_proj = nn.Sequential(
            nn.Linear(kv_dim, q_dim), nn.SiLU(), nn.Linear(q_dim, q_dim),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        x: (B, q_dim, L)
        kv: (B, nxi, kv_dim)
        Returns: (B, q_dim, L)
        """
        if kv is None:
            return x
        
        B, C, L = x.shape
        q = x.permute(0, 2, 1)                      # (B, L, C)
        
        kv_proj = self.kv_proj(kv)                  # (B, nxi, C)
        
        q_norm = self.cross_attn_norm(q)
        ca_out, _ = self.cross_attn(q_norm, kv_proj, kv_proj)
        q = q + ca_out                              # Clean residual skip connection
        
        return q.permute(0, 2, 1)                   # (B, C, L)


# ---------------------------------------------------------------------------
# 4. U-Net Backbone with Cross-Attention Bottleneck and Decoders
# ---------------------------------------------------------------------------

class UNetBackboneWaypoint(nn.Module):
    def __init__(self, D: int, n_heads: int = 8, kernel_size: int = 3):
        super().__init__()
        D4 = D * 4  

        self.enc1 = ConvResBlock(D,   D,   cond_dim=D, kernel_size=kernel_size, stride=1)
        self.enc2 = ConvResBlock(D,   D*2, cond_dim=D, kernel_size=kernel_size, stride=2)
        self.enc3 = ConvResBlock(D*2, D4,  cond_dim=D, kernel_size=kernel_size, stride=2)

        self.bot1 = ConvResBlock(D4, D4, cond_dim=D, kernel_size=kernel_size, stride=1)
        self.bot2 = ConvResBlock(D4, D4, cond_dim=D, kernel_size=kernel_size, stride=1)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=D4, num_heads=n_heads, batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(D4)

        self.bot_cross_attn = CrossAttentionBlock(q_dim=D4, kv_dim=D, n_heads=n_heads)
        self.dec1_cross_attn = CrossAttentionBlock(q_dim=D*2, kv_dim=D, n_heads=n_heads)
        self.dec2_cross_attn = CrossAttentionBlock(q_dim=D, kv_dim=D, n_heads=n_heads)

        self.dec1 = ConvResBlock(D4 + D*2, D*2, cond_dim=D, kernel_size=kernel_size, stride=1)
        self.dec2 = ConvResBlock(D*2 + D,  D,   cond_dim=D, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor,
                waypoint_tokens: torch.Tensor = None) -> torch.Tensor:
        e1 = self.enc1(x, time_cond)                             
        e2 = self.enc2(e1, time_cond)                            
        e3 = self.enc3(e2, time_cond)                            

        b = self.bot1(e3, time_cond)                             

        B_size, C, L = b.shape
        b_seq = b.permute(0, 2, 1)                              
        b_seq_norm = self.self_attn_norm(b_seq)
        sa_out, _ = self.self_attn(b_seq_norm, b_seq_norm, b_seq_norm)  
        b_seq = b_seq + sa_out                                  
        b = b_seq.permute(0, 2, 1)                              

        b = self.bot_cross_attn(b, waypoint_tokens)              
        b = self.bot2(b, time_cond)                              

        b_up  = F.interpolate(b, size=e2.shape[-1], mode='linear', align_corners=False)
        d1    = self.dec1(torch.cat([b_up, e2], dim=1), time_cond)  
        d1 = self.dec1_cross_attn(d1, waypoint_tokens)

        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d2    = self.dec2(torch.cat([d1_up, e1], dim=1), time_cond) 
        d2 = self.dec2_cross_attn(d2, waypoint_tokens)
        return d2


# ---------------------------------------------------------------------------
# 5. Output Heads
# ---------------------------------------------------------------------------

class FlowHead(nn.Module):
    def __init__(self, D: int, nd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D), nn.LayerNorm(D), nn.SiLU(), nn.Linear(D, nd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1))


class LambdaHead(nn.Module):
    def __init__(self, D: int, n_lambda: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(),
            nn.Linear(D, D // 2), nn.SiLU(),
            nn.Linear(D // 2, n_lambda),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=-1)                                  
        return self.net(pooled)                                  


# ---------------------------------------------------------------------------
# 6. MPD Tokenization Layer (1D Temporal Convolution)
# ---------------------------------------------------------------------------

class MPDLayer(nn.Module):
    def __init__(self, nd: int, D: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(nd, D, kernel_size, stride=1,
                              padding=kernel_size // 2)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.permute(0, 2, 1))                       
        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)      
        return h


# ---------------------------------------------------------------------------
# 7. Full Model: Waypoint Cross-Attention Flow Network
# ---------------------------------------------------------------------------

class WaypointCrossAttnFlowNetwork(nn.Module):
    def __init__(self, nxi: int = 20, nd: int = 2, D: int = 128,
                 n_lambda: int = 6, predict_lambda: bool = True,
                 n_heads: int = 8, kernel_size: int = 3):
        super().__init__()
        self.nxi = nxi
        self.nd  = nd
        self.D   = D
        self.predict_lambda = predict_lambda

        self.mpd_layer = MPDLayer(nd=nd, D=D, kernel_size=kernel_size)
        self.pos_emb   = nn.Parameter(torch.randn(1, D, nxi) * 0.02)

        self.time_emb = SinusoidalTimeEmbedding(D=D)
        
        self.waypoint_tokenizer = WaypointTokenizer(D=D)
        self.null_waypoint_tokens = nn.Parameter(torch.zeros(1, nxi, D))

        self.backbone = UNetBackboneWaypoint(
            D=D, n_heads=n_heads, kernel_size=kernel_size,
        )

        self.flow_head = FlowHead(D=D, nd=nd)
        if predict_lambda:
            self.lambda_head = LambdaHead(D=D, n_lambda=n_lambda)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                waypoints: torch.Tensor,
                cond_drop_mask: torch.Tensor = None):
        tokens = self.mpd_layer(x)                               
        tokens = tokens + self.pos_emb                           

        time_cond = self.time_emb(t)                             

        waypoint_tokens = self.waypoint_tokenizer(waypoints)   

        if cond_drop_mask is not None:
            mask = cond_drop_mask.view(-1, 1, 1).to(waypoint_tokens.dtype)  
            waypoint_tokens = waypoint_tokens * (1.0 - mask) + self.null_waypoint_tokens * mask

        out = self.backbone(tokens, time_cond, waypoint_tokens)      

        v_t = self.flow_head(out)                                
        lambda0 = None
        if self.predict_lambda:
            lambda0 = self.lambda_head(out)                      

        return v_t, lambda0


# ---------------------------------------------------------------------------
# 8. Training utility (with CFG Dropout)
# ---------------------------------------------------------------------------

def compute_waypoint_cfm_loss(
    model: nn.Module,
    x1_batch: torch.Tensor,
    waypoint_batch: torch.Tensor,
    p_drop: float = 0.0,
) -> torch.Tensor:
    B = x1_batch.shape[0]
    device = x1_batch.device
    x0    = torch.randn_like(x1_batch)
    t     = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)
    xt    = (1 - t_exp) * x0 + t_exp * x1_batch
    ut    = x1_batch - x0

    cond_drop_mask = (torch.rand(B, device=device) < p_drop)

    v_t, lambda0 = model(xt, t, waypoint_batch, cond_drop_mask=cond_drop_mask)

    loss = torch.mean((v_t - ut) ** 2)
    return loss


# ---------------------------------------------------------------------------
# 9. Generation utility (with CFG Extrapolation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_waypoint_trajectories(
    model: nn.Module,
    waypoints: torch.Tensor,
    num_samples: int = 1,
    nxi: int = 20,
    nd:  int = 2,
    steps: int = 100,
    device: str = 'cpu',
    cfg_weight: float = 2.0,
) -> tuple:
    model.eval()
    if waypoints.ndim == 2:  
        waypoints = waypoints.unsqueeze(0).expand(num_samples, -1, -1).contiguous()
    waypoints = waypoints.to(device)

    x  = torch.randn(num_samples, nxi, nd, device=device)
    dt = 1.0 / steps

    mask_batch = torch.cat([
        torch.zeros(num_samples, dtype=torch.bool, device=device),
        torch.ones(num_samples,  dtype=torch.bool, device=device),
    ], dim=0)
    waypoint_batch = torch.cat([waypoints, waypoints], dim=0)

    lambda0_accum = None

    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=device)
        t_batch = torch.cat([t, t], dim=0)
        x_batch = torch.cat([x, x], dim=0)

        v_batch, lam_batch = model(
            x_batch, t_batch, waypoint_batch, cond_drop_mask=mask_batch,
        )
        v_cond, v_null = v_batch.chunk(2, dim=0)

        v_cfg = v_null + cfg_weight * (v_cond - v_null)
        x = x + v_cfg * dt

        if lam_batch is not None:
            lam_cond, _ = lam_batch.chunk(2, dim=0)
            lambda0_accum = lam_cond

    return x, lambda0_accum
