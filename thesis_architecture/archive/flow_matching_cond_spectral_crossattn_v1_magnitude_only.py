r"""
flow_matching_cond_spectral_crossattn.py
========================================
Conditional Flow Matching Network with:
- Spectral Conditioning via SpectralTokenizer (replaces ShapeEncoderMPD)
- Cross-Attention in the U-Net bottleneck (replaces Global Average Pooling)
- FiLM for time conditioning (retained from previous architecture)
- CFG support (condition dropout on spectral tokens)
- Dual output head: Flow velocity + optional Lagrange multipliers

Architecture lineage:
  _char.py → _char_film.py → _char_film_cfg.py → THIS FILE (spectral_crossattn)
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
# 2. Sinusoidal Frequency Positional Encoding
# ---------------------------------------------------------------------------

class FrequencyPositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal positional encoding for spectral frequency bands.
    Encodes the 2D indices (k1, k2) by adding two 1D positional encodings.
    """

    def __init__(self, D: int, max_len: int = 256):
        super().__init__()
        assert D % 2 == 0, "D must be even for sinusoidal encoding."
        pe_1d = torch.zeros(max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, D, 2, dtype=torch.float32)
            * -(math.log(10_000.0) / D)
        )
        pe_1d[:, 0::2] = torch.sin(position * div_term)
        pe_1d[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_1d", pe_1d)              # (max_len, D)

    def forward(self, x: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D) - token embeddings
        k_indices: (B, S, 2) - integer indices for k1 and k2
        Returns: (B, S, D) with positional encoding added.
        """
        k1 = k_indices[:, :, 0]  # (B, S)
        k2 = k_indices[:, :, 1]  # (B, S)
        
        pe_k1 = self.pe_1d[k1]  # (B, S, D)
        pe_k2 = self.pe_1d[k2]  # (B, S, D)
        
        return x + pe_k1 + pe_k2


# ---------------------------------------------------------------------------
# 3. SpectralTokenizer (replaces ShapeEncoderMPD)
# ---------------------------------------------------------------------------

class SpectralTokenizer(nn.Module):
    """
    Translates spectral coefficients into frequency tokens for cross-attention.

    Input:  (B, S, 2) spectral coefficients [real, imag channels], (B, S, 2) k_indices
    Output: (B, S, D) frequency tokens (un-pooled, for use as K/V in cross-attention)

    Preserves full phase information: real and imaginary parts are fed as
    separate input channels instead of collapsing to magnitude-only.
    """

    def __init__(self, S: int, D: int):
        super().__init__()
        self.S = S
        self.D = D
        # Per-frequency MLP: project each [real, imag] pair to D dimensions.
        # Input dim is 2 (not 1) to preserve phase; previously np.abs() was used
        # which made the encoding invariant to time-shift / traversal direction.
        self.freq_mlp = nn.Sequential(
            nn.Linear(2, D), nn.SiLU(), nn.Linear(D, D),
        )
        # 2D Frequency positional encoding (max_len=256 to support arbitrary index ranges)
        self.freq_pos_enc = FrequencyPositionalEncoding2D(D=D, max_len=256)
        # LayerNorm for stable outputs
        self.norm = nn.LayerNorm(D)

    def forward(self, spec_ri: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        """
        spec_ri:   (B, S, 2) — [real, imag] of each spectral coefficient
        k_indices: (B, S, 2) — 2D frequency indices
        Returns:   (B, S, D) frequency tokens
        """
        tokens = self.freq_mlp(spec_ri)                              # (B, S, D)
        tokens = self.freq_pos_enc(tokens, k_indices)                # (B, S, D) + pos
        tokens = self.norm(tokens)                                   # (B, S, D)
        return tokens


# ---------------------------------------------------------------------------
# 4. 1D U-Net building blocks (with FiLM for time conditioning)
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

        # FiLM after norm, before activation (adaGN pattern from DiT)
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
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

        self.kv_proj = nn.Sequential(
            nn.Linear(kv_dim, q_dim), nn.SiLU(), nn.Linear(q_dim, q_dim),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        x: (B, q_dim, L)
        kv: (B, S, kv_dim)
        Returns: (B, q_dim, L)
        """
        if kv is None:
            return x
        
        B, C, L = x.shape
        q = x.permute(0, 2, 1)                      # (B, L, C)
        
        kv_proj = self.kv_proj(kv)                  # (B, S, C)
        ca_out, _ = self.cross_attn(q, kv_proj, kv_proj)
        q = self.cross_attn_norm(q + ca_out)        # (B, L, C)
        
        return q.permute(0, 2, 1)                   # (B, C, L)


# ---------------------------------------------------------------------------
# 5. U-Net Backbone with Cross-Attention Bottleneck and Decoders
# ---------------------------------------------------------------------------

class UNetBackboneSpectral(nn.Module):
    """
    1D U-Net backbone with:
    - FiLM conditioning (time) in every ConvResBlock
    - Self-Attention + Cross-Attention in the bottleneck

    The spectral condition enters ONLY via cross-attention (not FiLM).
    FiLM handles temporal awareness (where in the flow am I?).
    Cross-Attention handles spatial awareness (which frequencies to cover?).
    """

    def __init__(self, D: int, S: int, n_heads: int = 8, kernel_size: int = 3):
        super().__init__()
        D4 = D * 4  # bottleneck channel dim

        # ── Encoder ──
        self.enc1 = ConvResBlock(D,   D,   cond_dim=D, kernel_size=kernel_size, stride=1)
        self.enc2 = ConvResBlock(D,   D*2, cond_dim=D, kernel_size=kernel_size, stride=2)
        self.enc3 = ConvResBlock(D*2, D4,  cond_dim=D, kernel_size=kernel_size, stride=2)

        # ── Bottleneck Conv Blocks ──
        self.bot1 = ConvResBlock(D4, D4, cond_dim=D, kernel_size=kernel_size, stride=1)
        self.bot2 = ConvResBlock(D4, D4, cond_dim=D, kernel_size=kernel_size, stride=1)

        # ── Self-Attention on B-spline tokens ──
        self.self_attn = nn.MultiheadAttention(
            embed_dim=D4, num_heads=n_heads, batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(D4)
        # Zero-init output projection for stable residual
        nn.init.zeros_(self.self_attn.out_proj.weight)
        nn.init.zeros_(self.self_attn.out_proj.bias)

        # ── Cross-Attention Blocks ──
        self.bot_cross_attn = CrossAttentionBlock(q_dim=D4, kv_dim=D, n_heads=n_heads)
        self.dec1_cross_attn = CrossAttentionBlock(q_dim=D*2, kv_dim=D, n_heads=n_heads)
        self.dec2_cross_attn = CrossAttentionBlock(q_dim=D, kv_dim=D, n_heads=n_heads)

        # ── Decoder ──
        self.dec1 = ConvResBlock(D4 + D*2, D*2, cond_dim=D, kernel_size=kernel_size, stride=1)
        self.dec2 = ConvResBlock(D*2 + D,  D,   cond_dim=D, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor,
                spectral_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        x:               (B, D, nxi) — tokenized noisy trajectory
        time_cond:        (B, D) — time embedding (used for FiLM)
        spectral_tokens:  (B, S, D) — frequency tokens from SpectralTokenizer
        Returns:          (B, D, nxi)
        """
        # ── Encoder ──
        e1 = self.enc1(x, time_cond)                             # (B, D, 20)
        e2 = self.enc2(e1, time_cond)                            # (B, D*2, 10)
        e3 = self.enc3(e2, time_cond)                            # (B, D*4, 5)

        # ── Bottleneck block 1 ──
        b = self.bot1(e3, time_cond)                             # (B, D*4, 5)

        # ── Self-Attention on B-spline macro-tokens ──
        # Reshape to (B, L, D*4) for attention
        B_size, C, L = b.shape
        b_seq = b.permute(0, 2, 1)                              # (B, L=5, D*4)
        sa_out, _ = self.self_attn(b_seq, b_seq, b_seq)          # (B, L, D*4)
        b_seq = self.self_attn_norm(b_seq + sa_out)              # residual + norm
        b = b_seq.permute(0, 2, 1)                              # (B, D*4, 5)

        # ── Cross-Attention: Q=B-spline, KV=Spectral ──
        b = self.bot_cross_attn(b, spectral_tokens)              # (B, D*4, 5)

        # ── Bottleneck block 2 ──
        b = self.bot2(b, time_cond)                              # (B, D*4, 5)

        # ── Decoder ──
        b_up  = F.interpolate(b, size=e2.shape[-1], mode='linear', align_corners=False)
        d1    = self.dec1(torch.cat([b_up, e2], dim=1), time_cond)  # (B, D*2, 10)
        d1 = self.dec1_cross_attn(d1, spectral_tokens)

        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d2    = self.dec2(torch.cat([d1_up, e1], dim=1), time_cond) # (B, D, 20)
        d2 = self.dec2_cross_attn(d2, spectral_tokens)
        return d2


# ---------------------------------------------------------------------------
# 6. Output Heads
# ---------------------------------------------------------------------------

class FlowHead(nn.Module):
    """Projects D-dimensional tokens back to nd-dimensional velocity vectors."""

    def __init__(self, D: int, nd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D), nn.LayerNorm(D), nn.SiLU(), nn.Linear(D, nd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, nxi) -> (B, nxi, nd)"""
        return self.net(x.permute(0, 2, 1))


class LambdaHead(nn.Module):
    """
    Predicts initial Lagrange multipliers for SE(3) constraints.
    Uses global mean pooling over bottleneck features + MLP.
    """

    def __init__(self, D: int, n_lambda: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(),
            nn.Linear(D, D // 2), nn.SiLU(),
            nn.Linear(D // 2, n_lambda),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, nxi) -> (B, n_lambda)"""
        pooled = x.mean(dim=-1)                                  # (B, D)
        return self.net(pooled)                                  # (B, n_lambda)


# ---------------------------------------------------------------------------
# 7. MPD Tokenization Layer (1D Temporal Convolution)
# ---------------------------------------------------------------------------

class MPDLayer(nn.Module):
    """MPD Tokenisation via 1D Temporal CNN."""

    def __init__(self, nd: int, D: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(nd, D, kernel_size, stride=1,
                              padding=kernel_size // 2)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, nxi, nd) -> (B, D, nxi)"""
        h = self.conv(x.permute(0, 2, 1))                       # (B, D, nxi)
        h = self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)      # (B, D, nxi)
        return h


# ---------------------------------------------------------------------------
# 8. Full Model: Spectral Cross-Attention Flow Network
# ---------------------------------------------------------------------------

class SpectralCrossAttnFlowNetwork(nn.Module):
    """
    Conditional Flow Matching network for trajectory generation.

    Conditioning: Spectral coefficients -> SpectralTokenizer -> Cross-Attention
    Time:         Sinusoidal embedding -> FiLM in every ConvResBlock
    CFG:          Condition dropout on spectral tokens
    Output:       Flow velocity v_t + optional Lagrange multipliers lambda_0
    """

    def __init__(self, nxi: int = 20, nd: int = 2, D: int = 128,
                 S: int = 50, n_lambda: int = 6, predict_lambda: bool = True,
                 n_heads: int = 8, kernel_size: int = 3):
        super().__init__()
        self.nxi = nxi
        self.nd  = nd
        self.D   = D
        self.S   = S
        self.predict_lambda = predict_lambda

        # Trajectory tokenizer
        self.mpd_layer = MPDLayer(nd=nd, D=D, kernel_size=kernel_size)
        self.pos_emb   = nn.Parameter(torch.randn(1, D, nxi) * 0.02)

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(D=D)

        # Spectral condition encoder (replaces ShapeEncoderMPD)
        self.spectral_tokenizer = SpectralTokenizer(S=S, D=D)

        # Null spectral tokens for CFG dropout (B, S, D)
        self.null_spectral_tokens = nn.Parameter(torch.zeros(1, S, D))

        # U-Net backbone with cross-attention
        self.backbone = UNetBackboneSpectral(
            D=D, S=S, n_heads=n_heads, kernel_size=kernel_size,
        )

        # Dual output heads
        self.flow_head = FlowHead(D=D, nd=nd)
        if predict_lambda:
            self.lambda_head = LambdaHead(D=D, n_lambda=n_lambda)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                spectral_coeffs: torch.Tensor,
                k_indices: torch.Tensor,
                cond_drop_mask: torch.Tensor = None):
        """
        x:               (B, nxi, nd) — noisy trajectory
        t:               (B,) — diffusion time
        spectral_coeffs: (B, S, 2) — spectral condition [real, imag per coefficient]
        k_indices:       (B, S, 2) — spectral frequencies 2D (placeholder grid)
        cond_drop_mask:  (B,) bool — True = drop condition (CFG)

        Returns:
            v_t:     (B, nxi, nd) — predicted flow velocity
            lambda0: (B, n_lambda) or None — predicted Lagrange multipliers
        """
        # ── Tokenize trajectory ──
        tokens = self.mpd_layer(x)                               # (B, D, nxi)
        tokens = tokens + self.pos_emb                           # + positional encoding

        # ── Time embedding ──
        time_cond = self.time_emb(t)                             # (B, D)

        # ── Spectral condition ──
        spec_tokens = self.spectral_tokenizer(spectral_coeffs, k_indices)   # (B, S, D)

        # ── CFG Dropout: replace with null tokens ──
        if cond_drop_mask is not None:
            mask = cond_drop_mask.view(-1, 1, 1).to(spec_tokens.dtype)  # (B, 1, 1)
            spec_tokens = spec_tokens * (1.0 - mask) + self.null_spectral_tokens * mask

        # ── U-Net with cross-attention ──
        out = self.backbone(tokens, time_cond, spec_tokens)      # (B, D, nxi)

        # ── Dual output ──
        v_t = self.flow_head(out)                                # (B, nxi, nd)
        lambda0 = None
        if self.predict_lambda:
            lambda0 = self.lambda_head(out)                      # (B, n_lambda)

        return v_t, lambda0


# ---------------------------------------------------------------------------
# 9. Training utility (with CFG Dropout)
# ---------------------------------------------------------------------------

def compute_spectral_cfm_loss(
    model: nn.Module,
    x1_batch: torch.Tensor,
    spectral_batch: torch.Tensor,
    k_idx_batch: torch.Tensor,
    p_drop: float = 0.0,
) -> torch.Tensor:
    """
    Conditional CFM loss with spectral conditioning and CFG dropout.

    x1_batch:       (B, nxi, nd) — target trajectories
    spectral_batch: (B, S, 2) — spectral coefficients [real, imag] (condition)
    k_idx_batch:    (B, S, 2) — frequencies
    p_drop:         probability of dropping condition (CFG)
    """
    B = x1_batch.shape[0]
    device = x1_batch.device
    x0    = torch.randn_like(x1_batch)
    t     = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)
    xt    = (1 - t_exp) * x0 + t_exp * x1_batch
    ut    = x1_batch - x0

    # CFG condition dropout
    cond_drop_mask = (torch.rand(B, device=device) < p_drop)

    v_t, lambda0 = model(xt, t, spectral_batch, k_idx_batch, cond_drop_mask=cond_drop_mask)

    # Flow matching loss (MSE on velocity field)
    loss = torch.mean((v_t - ut) ** 2)

    # Lambda loss is NOT computed here — it requires ground-truth lambdas
    # which are only available when paired with a TSVEC solver.
    return loss


# ---------------------------------------------------------------------------
# 10. Generation utility (with CFG Extrapolation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_spectral_trajectories(
    model: nn.Module,
    spectral_coeffs: torch.Tensor,   # (S, 2) or (num_samples, S, 2)
    k_indices: torch.Tensor,         # (S, 2) or (num_samples, S, 2)
    num_samples: int = 1,
    nxi: int = 20,
    nd:  int = 2,
    steps: int = 100,
    device: str = 'cpu',
    cfg_weight: float = 2.0,
) -> tuple:
    """
    Generate trajectories conditioned on spectral coefficients.

    Returns:
        x:       (num_samples, nxi, nd) — generated trajectories
        lambda0: (num_samples, n_lambda) or None — predicted Lagrange multipliers
    """
    model.eval()
    if spectral_coeffs.ndim == 2:  # (S, 2) — single shape
        spectral_coeffs = spectral_coeffs.unsqueeze(0).expand(num_samples, -1, -1).contiguous()
    spectral_coeffs = spectral_coeffs.to(device)

    if k_indices.ndim == 2:
        k_indices = k_indices.unsqueeze(0).expand(num_samples, -1, -1).contiguous()
    k_indices = k_indices.to(device)

    x  = torch.randn(num_samples, nxi, nd, device=device)
    dt = 1.0 / steps

    # CFG batching: [conditioned ; unconditioned]
    mask_batch = torch.cat([
        torch.zeros(num_samples, dtype=torch.bool, device=device),
        torch.ones(num_samples,  dtype=torch.bool, device=device),
    ], dim=0)
    spec_batch = torch.cat([spectral_coeffs, spectral_coeffs], dim=0)
    k_idx_batch = torch.cat([k_indices, k_indices], dim=0)

    lambda0_accum = None

    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=device)
        t_batch = torch.cat([t, t], dim=0)
        x_batch = torch.cat([x, x], dim=0)

        v_batch, lam_batch = model(
            x_batch, t_batch, spec_batch, k_idx_batch, cond_drop_mask=mask_batch,
        )
        v_cond, v_null = v_batch.chunk(2, dim=0)

        # CFG extrapolation
        v_cfg = v_null + cfg_weight * (v_cond - v_null)
        x = x + v_cfg * dt

        # Keep lambda from the last step (conditioned half)
        if lam_batch is not None:
            lam_cond, _ = lam_batch.chunk(2, dim=0)
            lambda0_accum = lam_cond

    return x, lambda0_accum
