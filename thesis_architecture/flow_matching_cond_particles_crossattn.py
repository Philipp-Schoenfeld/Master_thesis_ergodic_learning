r"""
flow_matching_cond_particles_crossattn.py
=========================================
Conditional Flow Matching Network with:
- Particle Conditioning via ParticleTokenizer
- Cross-Attention in the U-Net bottleneck
- FiLM for time conditioning
- CFG support (condition dropout on particle tokens)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
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
        args = t[:, None] * self.freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)

class GaussianFourierProjection2D(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(2, embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x: torch.Tensor):
        x_proj = x @ self.W * 2.0 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ParticleTokenizer(nn.Module):
    """
    Translates raw point cloud particles [x, y, mu] into tokens for cross-attention.
    Applies Gaussian Fourier Features to (x,y) to avoid Spectral Bias,
    then concatenates mu and processes via MLP.
    """
    def __init__(self, D: int):
        super().__init__()
        self.D = D
        self.pos_enc = GaussianFourierProjection2D(embed_dim=D//2, scale=1.0)
        self.mlp = nn.Sequential(
            nn.Linear(D//2 + 1, D), nn.SiLU(), nn.Linear(D, D),
        )
        self.out_norm = nn.LayerNorm(D)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        particles: (B, N, 3) — coordinate + density
        Returns:   (B, N, D) particle tokens
        """
        xy = particles[:, :, :2]
        mu = particles[:, :, 2:3]
        pos_feat = self.pos_enc(xy)
        features = torch.cat([pos_feat, mu], dim=-1)
        tokens = self.mlp(features)
        tokens = self.out_norm(tokens)
        return tokens

class ConvResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 128,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
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
        h = self.norm1(self.conv1(x))
        if cond is not None:
            film_params = self.film_proj(cond).unsqueeze(-1)
            gamma, beta = film_params.chunk(2, dim=1)
            h = h * (1.0 + gamma) + beta
        h = self.act(h)
        h = self.norm2(self.conv2(h))
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
            nn.LayerNorm(q_dim),
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        if kv is None:
            return x
        B, C, L = x.shape
        q = x.permute(0, 2, 1)
        kv_proj = self.kv_proj(kv)
        q_norm = self.cross_attn_norm(q)
        ca_out, _ = self.cross_attn(q_norm, kv_proj, kv_proj)
        q = q + ca_out
        return q.permute(0, 2, 1)

class UNetBackboneParticles(nn.Module):
    def __init__(self, D: int, n_heads: int = 8, kernel_size: int = 3):
        super().__init__()
        D4 = D * 4  
        self.enc1 = ConvResBlock(D,   D,   cond_dim=D, kernel_size=kernel_size, stride=1)
        self.enc2 = ConvResBlock(D,   D*2, cond_dim=D, kernel_size=kernel_size, stride=2)
        self.enc3 = ConvResBlock(D*2, D4,  cond_dim=D, kernel_size=kernel_size, stride=2)

        self.bot1 = ConvResBlock(D4, D4, cond_dim=D, kernel_size=kernel_size, stride=1)
        self.bot2 = ConvResBlock(D4, D4, cond_dim=D, kernel_size=kernel_size, stride=1)

        self.self_attn = nn.MultiheadAttention(embed_dim=D4, num_heads=n_heads, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(D4)

        self.bot_cross_attn = CrossAttentionBlock(q_dim=D4, kv_dim=D, n_heads=n_heads)
        self.dec1_cross_attn = CrossAttentionBlock(q_dim=D*2, kv_dim=D, n_heads=n_heads)
        self.dec2_cross_attn = CrossAttentionBlock(q_dim=D, kv_dim=D, n_heads=n_heads)

        self.dec1 = ConvResBlock(D4 + D*2, D*2, cond_dim=D, kernel_size=kernel_size, stride=1)
        self.dec2 = ConvResBlock(D*2 + D,  D,   cond_dim=D, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor, particle_tokens: torch.Tensor = None) -> torch.Tensor:
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

        b = self.bot_cross_attn(b, particle_tokens)              
        b = self.bot2(b, time_cond)                              

        b_up  = F.interpolate(b, size=e2.shape[-1], mode='linear', align_corners=False)
        d1    = self.dec1(torch.cat([b_up, e2], dim=1), time_cond)  
        d1 = self.dec1_cross_attn(d1, particle_tokens)

        d1_up = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d2    = self.dec2(torch.cat([d1_up, e1], dim=1), time_cond) 
        d2 = self.dec2_cross_attn(d2, particle_tokens)
        return d2

class FlowHead(nn.Module):
    def __init__(self, D: int, nd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D), nn.LayerNorm(D), nn.SiLU(), nn.Linear(D, nd),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1))

class MPDLayer(nn.Module):
    def __init__(self, nd: int, D: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(nd, D, kernel_size, stride=1, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(D)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.permute(0, 2, 1))                       
        return self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)

class ParticleCrossAttnFlowNetwork(nn.Module):
    def __init__(self, nxi: int = 20, nd: int = 2, D: int = 128,
                 n_heads: int = 8, kernel_size: int = 3):
        super().__init__()
        self.nxi = nxi
        self.nd  = nd
        self.D   = D

        self.mpd_layer = MPDLayer(nd=nd, D=D, kernel_size=kernel_size)
        self.pos_emb   = nn.Parameter(torch.randn(1, D, nxi) * 0.02)
        self.time_emb = SinusoidalTimeEmbedding(D=D)
        
        self.particle_tokenizer = ParticleTokenizer(D=D)
        
        self.null_particle_token = nn.Parameter(torch.zeros(1, 1, D))

        self.backbone = UNetBackboneParticles(D=D, n_heads=n_heads, kernel_size=kernel_size)
        self.flow_head = FlowHead(D=D, nd=nd)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                particles: torch.Tensor,
                cond_drop_mask: torch.Tensor = None):
        tokens = self.mpd_layer(x)                               
        tokens = tokens + self.pos_emb                           
        time_cond = self.time_emb(t)                             

        particle_tokens = self.particle_tokenizer(particles)   

        if cond_drop_mask is not None:
            mask = cond_drop_mask.view(-1, 1, 1).to(particle_tokens.dtype)  
            # Broadcasting self.null_particle_token along sequence length
            particle_tokens = particle_tokens * (1.0 - mask) + self.null_particle_token * mask

        out = self.backbone(tokens, time_cond, particle_tokens)      

        v_t = self.flow_head(out)                                

        return v_t, None

def compute_particle_cfm_loss(
    model: nn.Module,
    x1_batch: torch.Tensor,
    particle_batch: torch.Tensor,
    p_drop: float = 0.0,
    ergodic=None,
) -> tuple:
    """Conditional flow-matching loss, optionally with an ergodic coverage term.

    Returns (total_loss, components) where components maps a name to a detached
    scalar tensor, so the runner can log the two terms separately.
    """
    B = x1_batch.shape[0]
    device = x1_batch.device
    x0    = torch.randn_like(x1_batch)
    t     = torch.rand(B, device=device)
    t_exp = t.view(B, 1, 1)
    xt    = (1 - t_exp) * x0 + t_exp * x1_batch
    ut    = x1_batch - x0

    cond_drop_mask = (torch.rand(B, device=device) < p_drop)
    v_t, lambda0 = model(xt, t, particle_batch, cond_drop_mask=cond_drop_mask)

    loss_cfm = torch.mean((v_t - ut) ** 2)
    total = loss_cfm
    parts = {'cfm': loss_cfm.detach()}

    if ergodic is not None and ergodic.weight > 0.0:
        # Flow matching predicts a velocity, so the endpoint has to be estimated
        # before a trajectory-level metric can be applied to it.
        x1_hat = xt + (1.0 - t_exp) * v_t
        loss_erg = ergodic(x1_hat, particle_batch, t)
        total = total + ergodic.weight * loss_erg
        parts['erg'] = loss_erg.detach()

    return total, parts

@torch.no_grad()
def generate_particle_trajectories(
    model: nn.Module,
    particles: torch.Tensor,
    num_samples: int = 1,
    nxi: int = 20,
    nd:  int = 2,
    steps: int = 100,
    device: str = 'cpu',
    cfg_weight: float = 2.0,
    obstacle=None,
    obstacle_weight: float = 20.0,
    obstacle_t_start: float = 0.3,
    bspline_pts: int = 256,
    bspline_deg: int = 5,
    polish_steps: int = 250,
    generator: torch.Generator = None,
) -> tuple:
    """Integrate the flow ODE, optionally repelling the curve from an obstacle.

    With `obstacle=None` the behaviour is unchanged. Otherwise a repulsion term
    is added to the velocity at inference time only — the model itself never
    sees the obstacle and is conditioned on the unmodified target density.
    """
    model.eval()
    if particles.ndim == 2:
        particles = particles.unsqueeze(0)
    if particles.shape[0] == 1 and num_samples > 1:
        particles = particles.expand(num_samples, -1, -1).contiguous()
    particles = particles.to(device)

    x  = torch.randn(num_samples, nxi, nd, device=device, generator=generator)
    dt = 1.0 / steps

    B_basis = None
    if obstacle is not None:
        from obstacles import basis_torch, curve_repulsion_grad
        B_basis = basis_torch(nxi, bspline_pts, bspline_deg, device=device)

    mask_batch = torch.cat([
        torch.zeros(num_samples, dtype=torch.bool, device=device),
        torch.ones(num_samples,  dtype=torch.bool, device=device),
    ], dim=0)
    particle_batch = torch.cat([particles, particles], dim=0)

    lambda0_accum = None

    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=device)
        t_batch = torch.cat([t, t], dim=0)
        x_batch = torch.cat([x, x], dim=0)

        v_batch, lam_batch = model(
            x_batch, t_batch, particle_batch, cond_drop_mask=mask_batch,
        )
        v_cond, v_null = v_batch.chunk(2, dim=0)
        v = v_null + cfg_weight * (v_cond - v_null)

        if obstacle is not None:
            # Ramp: at small t the state is still essentially Gaussian noise, so
            # repelling it is meaningless and only distorts the flow. Start at
            # t_start and grow quadratically to full strength at t = 1.
            t_now = step * dt
            if t_now >= obstacle_t_start:
                s = (t_now - obstacle_t_start) / max(1.0 - obstacle_t_start, 1e-8)
                v = v - (obstacle_weight * s ** 2) * curve_repulsion_grad(
                    x, obstacle, B_basis)

        x = x + v * dt

        if lam_batch is not None:
            lam_cond, _ = lam_batch.chunk(2, dim=0)
            lambda0_accum = lam_cond

    # Polish: the ramp alone does not guarantee hard clearance. A few pure
    # descent steps on the penalty drive the remaining violation to zero while
    # leaving the shape produced by the flow essentially untouched.
    if obstacle is not None and polish_steps > 0:
        from obstacles import polish_out_of_obstacle
        x = polish_out_of_obstacle(x, obstacle, B_basis, max_iters=polish_steps)

    return x, lambda0_accum
