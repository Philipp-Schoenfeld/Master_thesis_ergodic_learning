r"""
flow_matching_cond_particles_dit.py
===================================
Diffusion-Transformer (DiT) backbone with adaLN-Zero for the start-conditioned
particle CFM -- an opt-in alternative to the 1D U-Net of
`flow_matching_cond_particles_start.py` (runner: `--backbone dit`).

Same interface as `ParticleCrossAttnFlowNetwork` there: forward(x, t, particles,
cond_drop_mask, start) -> (v, None), the same particle tokenizer, start embedding,
CFG null token and `start_inpaint` flag. Loss and sampler are model-agnostic and
are re-exported from that module unchanged, so CFM loss, ErgLoss, inpainting
and CFG behave identically and only the backbone differs.

Architecture and the implementations it follows
-----------------------------------------------
    x (B, nxi, 2) -> Conv1d k=3 embed + fixed 1D sin-cos pos. emb.   [DiT PatchEmbed; MPD k=3]
    c = TimestepEmbedder(1000 t) + StartEmbedding(p0)               [DiT: c = t_emb + y_emb]
    particles (B, N, 3) -> ParticleTokenizer -> (B, N, D)           (shared with the U-Net)
    depth x block:
        x = x + gate_msa * SelfAttn(modulate(LN(x), shift_msa, scale_msa))   [DiT adaLN-Zero]
        x = x + CrossAttn(LN(x), particle tokens)                           [PixArt-alpha]
        x = x + gate_mlp * MLP(modulate(LN(x), shift_mlp, scale_mlp))       [DiT adaLN-Zero]
    FinalLayer: modulate(LN(x), shift, scale) -> Linear(D, nd)              [DiT FinalLayer]

* Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023,
  github.com/facebookresearch/DiT (models.py): `modulate`, `TimestepEmbedder`,
  `DiTBlock`, `FinalLayer`, `initialize_weights` (xavier-uniform linears,
  N(0, 0.02) timestep MLP, zero adaLN modulation and zero final linear, so every
  block starts as the identity and the net outputs v = 0). DiT-S sizes
  (hidden 384, 12 blocks, 6 heads) are the defaults here.
* Chen et al., "PixArt-alpha", ICLR 2024, github.com/PixArt-alpha/PixArt-alpha
  (diffusion/model/nets/PixArt.py, `PixArtBlock`): a DiT block with an extra
  cross-attention to a token sequence between self-attention and MLP, its output
  projection zero-initialised. That is how a *set* condition enters here:
  adaLN only takes a global vector, and pooling the 256 particle tokens into one
  would be exactly the global average pooling this project avoids.
* Liu et al., "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation",
  ICLR 2025, github.com/thu-ml/RoboticsDiffusionTransformer (models/rdt/blocks.py):
  the same self-attn -> cross-attn -> FFN block for an action/trajectory sequence,
  with a pre-norm in front of the cross-attention (used here as well).
* Dasari et al., "The Ingredients for Robotic Diffusion Transformers", 2024,
  arXiv:2410.10088: adaLN-Zero for the global conditioning is what made their
  DiT policy train stably; plain cross-attention / in-context conditioning alone
  was "far less stable" and removing the zero init cost performance.
* Chi et al., "Diffusion Policy", RSS 2023 (transformer_for_diffusion.py) is the
  earlier trajectory transformer (time as a token, pre-LN decoder with
  cross-attention to observations); kept only as the baseline those works improve on.

Time scaling: CFM uses t in [0, 1]; DiT's sinusoidal embedding (max period 1e4)
expects a step index in [0, 1000], so t is multiplied by 1000 as in the
flow-matching DiTs of Stable Diffusion 3 (Esser et al., ICML 2024).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching_cond_particles_start import (  # noqa: F401  (re-exported for the loaders)
    ParticleTokenizer, StartEmbedding,
    compute_particle_cfm_loss, generate_particle_trajectories,
)


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


class ParticleDiTFlowNetwork(nn.Module):
    """Drop-in replacement for `flow_matching_cond_particles_start.ParticleCrossAttnFlowNetwork`.

    `start_inpaint` works as there (buffer `start_inpaint_flag`, adopted by a
    `load_state_dict` pre-hook), so the shared loss and sampler pin control
    point 0 for inpainting checkpoints.
    """

    def __init__(self, nxi: int = 25, nd: int = 2, D: int = 384, depth: int = 12,
                 n_heads: int = 6, mlp_ratio: float = 4.0, kernel_size: int = 3,
                 start_inpaint: bool = False):
        super().__init__()
        self.nxi, self.nd, self.D = nxi, nd, D
        self.depth, self.n_heads, self.mlp_ratio = depth, n_heads, mlp_ratio
        if start_inpaint:
            self.register_buffer('start_inpaint_flag', torch.ones(()))
        self._register_load_state_dict_pre_hook(self._adopt_inpaint_flag)

        # Conv1d k=3 token embedding: the MPD layer of the U-Net, here in the
        # role of DiT's patch embedding (finite differences of neighbouring cps).
        self.x_embedder = nn.Conv1d(nd, D, kernel_size, padding=kernel_size // 2)
        self.register_buffer(
            'pos_embed', torch.from_numpy(get_1d_sincos_pos_embed(D, nxi)).unsqueeze(0),
            persistent=False)
        self.t_embedder = TimestepEmbedder(D)
        self.start_emb = StartEmbedding(D=D, nd=nd)
        self.particle_tokenizer = ParticleTokenizer(D=D)
        self.null_particle_token = nn.Parameter(torch.zeros(1, 1, D))

        self.blocks = nn.ModuleList(
            [DiTCrossBlock(D, n_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(D, nd)
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

    @property
    def start_inpaint(self) -> bool:
        return 'start_inpaint_flag' in self._buffers

    def _adopt_inpaint_flag(self, state_dict, prefix, *args):
        if prefix + 'start_inpaint_flag' in state_dict and not self.start_inpaint:
            self.register_buffer('start_inpaint_flag',
                                 torch.ones((), device=self.pos_embed.device))

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                particles: torch.Tensor,
                cond_drop_mask: torch.Tensor = None,
                start: torch.Tensor = None):
        h = self.x_embedder(x.permute(0, 2, 1)).permute(0, 2, 1) + self.pos_embed
        c = self.t_embedder(1000.0 * t.view(-1))
        if start is not None:
            c = c + self.start_emb(start)

        particle_tokens = self.particle_tokenizer(particles)
        if cond_drop_mask is not None:
            mask = cond_drop_mask.view(-1, 1, 1).to(particle_tokens.dtype)
            particle_tokens = particle_tokens * (1.0 - mask) + self.null_particle_token * mask

        for block in self.blocks:
            h = block(h, c, particle_tokens)
        return self.final_layer(h, c), None


def dit_kwargs(ckpt: dict) -> dict:
    """Constructor arguments of a `backbone == 'dit'` checkpoint (for the loaders)."""
    return dict(depth=ckpt.get('dit_depth', 12), n_heads=ckpt.get('dit_heads', 6),
                mlp_ratio=ckpt.get('dit_mlp_ratio', 4.0))
