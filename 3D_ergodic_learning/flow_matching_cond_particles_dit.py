r"""
flow_matching_cond_particles_dit.py  —  3D port
===============================================
Diffusion-Transformer (DiT) backbone with adaLN-Zero for the 3D SE(3) particle
CFM -- an opt-in alternative to the 1D U-Net of
`flow_matching_cond_particles_crossattn.py` (runner: `--backbone dit`).

The blocks are those of the 2D DiT, `thesis_architecture/
flow_matching_cond_particles_dit.py` (see there for the references: DiT,
Peebles & Xie 2023; PixArt-alpha cross-attention; RDT-1B; Dasari et al. 2024),
which beat the 87.5 M U-Net in the 2D comparison `cfm_models_20261028`
(Hellinger 0.298 vs 0.335 at lambda 630) at half the parameters and a quarter
of the step time. They are copied rather than imported: both directories carry
modules of the same name (`flow_matching_cond_particles_crossattn`), and an
import across them would bind whichever comes first on sys.path.

Same interface as the 3D U-Net, so `compute_particle_cfm_loss`,
`generate_particle_trajectories`, the ErgLoss, the orientation objective and
the runner work unchanged:

    forward(x, t, particles, cond_drop_mask, start, start_drop_mask) -> (v_pos, v_rot)
    attributes nxi, nd, D, in_dim, predict_orientation, start_cond, start_inpaint

What changes against the 2D DiT, and only that:

* the state is (position 3 + rot6 6) = 9 channels: the Conv1d token embedding
  takes `in_dim` channels and the final layer returns `in_dim`, split into the
  position and rotation velocity,
* the particle tokenizer and the start embedding are the 3D ones
  (x, y, z | mu; start in R^3),
* the start point can be dropped per sample (`start_drop_mask`) or omitted
  (null start token), exactly like the 3D U-Net.

    x (B, nxi, 9) -> Conv1d k=3 embed + fixed 1D sin-cos pos. emb.
    c = TimestepEmbedder(1000 t) + StartEmbedding(p0)
    particles (B, N, 4) -> ParticleTokenizer -> (B, N, D)
    depth x [adaLN-Zero self-attn -> cross-attn to particles -> adaLN-Zero MLP]
    FinalLayer -> (B, nxi, 9) = (v_pos | v_rot)

All newly added output projections start at zero (adaLN modulation, cross-
attention projection, final linear), so the net outputs v = 0 at step 0.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching_cond_particles_crossattn import (  # noqa: F401  (re-exported)
    ND, ParticleTokenizer, StartEmbedding, _adopt_inpaint_flag,
    compute_particle_cfm_loss, generate_particle_trajectories,
)
from obstacle_cond import ObstacleEmbedding, ObstacleCondMixin


def modulate(x, shift, scale):
    """DiT: per-sample affine modulation, broadcast over the sequence axis."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed(embed_dim, length):
    """Fixed 1D sin-cos table (L, D), the 1D part of DiT's/MAE's 2D table."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum('m,d->md', np.arange(length, dtype=np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


class TimestepEmbedder(nn.Module):
    """DiT: sinusoidal frequency embedding of the (scaled) time, then an MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class Attention(nn.Module):
    """Multi-head self-attention (timm `Attention` as used by DiT), via SDPA."""

    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0, "hidden size must be divisible by the number of heads"
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class CrossAttention(nn.Module):
    """Queries from the control-point tokens, keys/values from the particle tokens
    (PixArt-alpha `MultiHeadCrossAttention`)."""

    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0, "hidden size must be divisible by the number of heads"
        self.n_heads = n_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cond):
        B, L, D = x.shape
        h = self.n_heads
        q = self.q(x).view(B, L, h, D // h).transpose(1, 2)
        k, v = self.kv(cond).view(B, cond.shape[1], 2, h, D // h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class DiTCrossBlock(nn.Module):
    """DiT block with adaLN-Zero plus a PixArt-alpha cross-attention to the particles."""

    def __init__(self, hidden_size, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, n_heads)
        self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, n_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.GELU(approximate='tanh'),
            nn.Linear(mlp_hidden, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x, c, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(self.norm_cross(x), cond)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """DiT: adaLN-modulated LayerNorm, then a linear map to the velocity per control point."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class ParticleDiTFlowNetwork(ObstacleCondMixin, nn.Module):
    """Drop-in replacement for the 3D `ParticleCrossAttnFlowNetwork` (see module docstring)."""

    def __init__(self, nxi: int = 25, nd: int = ND, D: int = 384, depth: int = 12,
                 n_heads: int = 6, mlp_ratio: float = 4.0, kernel_size: int = 3,
                 predict_orientation: bool = False, start_cond: bool = False,
                 start_inpaint: bool = False, obstacle_cond: bool = False):
        super().__init__()
        self.nxi, self.nd, self.D = nxi, nd, D
        self.obstacle_cond = obstacle_cond
        if obstacle_cond:
            self.obstacle_emb = ObstacleEmbedding(D=D)
            self.null_obstacle_token = nn.Parameter(torch.zeros(1, D))
        self.depth, self.n_heads, self.mlp_ratio = depth, n_heads, mlp_ratio
        self.predict_orientation = predict_orientation
        self.start_cond = start_cond
        self.in_dim = nd + (6 if predict_orientation else 0)
        if start_inpaint:
            self.register_buffer('start_inpaint_flag', torch.ones(()))
        self._register_load_state_dict_pre_hook(_adopt_inpaint_flag, with_module=True)

        # Conv1d k=3 token embedding: the MPD layer of the U-Net, here in the
        # role of DiT's patch embedding (finite differences of neighbouring cps).
        self.x_embedder = nn.Conv1d(self.in_dim, D, kernel_size, padding=kernel_size // 2)
        self.register_buffer(
            'pos_embed', torch.from_numpy(get_1d_sincos_pos_embed(D, nxi)).unsqueeze(0),
            persistent=False)
        self.t_embedder = TimestepEmbedder(D)
        if start_cond:
            self.start_emb = StartEmbedding(D=D, nd=nd)
            self.null_start_token = nn.Parameter(torch.zeros(1, D))
        self.particle_tokenizer = ParticleTokenizer(D=D, nd=nd)
        self.null_particle_token = nn.Parameter(torch.zeros(1, 1, D))

        self.blocks = nn.ModuleList(
            [DiTCrossBlock(D, n_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(D, self.in_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """DiT `initialize_weights` + PixArt-alpha's zero cross-attention projection."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.obstacle_cond:
            # `_basic_init` above re-initialised every Linear, including the last
            # layer of the obstacle embedding that must start at zero.
            nn.init.zeros_(self.obstacle_emb.net[-1].weight)
            nn.init.zeros_(self.obstacle_emb.net[-1].bias)

    @property
    def start_inpaint(self) -> bool:
        return 'start_inpaint_flag' in self._buffers

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                particles: torch.Tensor,
                cond_drop_mask: torch.Tensor = None,
                start: torch.Tensor = None,
                start_drop_mask: torch.Tensor = None,
                obstacle: torch.Tensor = None,
                obstacle_present: torch.Tensor = None):
        """x: (B, nxi, in_dim). Returns (v_pos, v_rot) with v_rot None if off."""
        h = self.x_embedder(x.permute(0, 2, 1)).permute(0, 2, 1) + self.pos_embed
        c = self.t_embedder(1000.0 * t.view(-1))
        if self.start_cond:
            if start is not None:
                start_e = self.start_emb(start)
                if start_drop_mask is not None:
                    m = start_drop_mask.view(-1, 1).to(start_e.dtype)
                    start_e = start_e * (1.0 - m) + self.null_start_token * m
            else:
                start_e = self.null_start_token.expand(c.shape[0], -1)
            c = c + start_e
        if self.obstacle_cond:
            c = c + self._obstacle_vector(c, obstacle, obstacle_present)

        particle_tokens = self.particle_tokenizer(particles)
        if cond_drop_mask is not None:
            mask = cond_drop_mask.view(-1, 1, 1).to(particle_tokens.dtype)
            particle_tokens = particle_tokens * (1.0 - mask) + self.null_particle_token * mask

        for block in self.blocks:
            h = block(h, c, particle_tokens)
        v = self.final_layer(h, c)
        if self.predict_orientation:
            return v[..., :self.nd], v[..., self.nd:]
        return v, None


def dit_kwargs(ckpt: dict) -> dict:
    """Constructor arguments of a `backbone == 'dit'` checkpoint (for the loaders)."""
    return dict(depth=ckpt.get('dit_depth', 12), n_heads=ckpt.get('dit_heads', 6),
                mlp_ratio=ckpt.get('dit_mlp_ratio', 4.0))
