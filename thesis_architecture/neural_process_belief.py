"""
neural_process_belief.py
========================
Density belief from a few point measurements with an attentive neural process
(deterministic path of Kim et al., "Attentive Neural Processes", ICLR 2019).

Context tokens are the measurements (x, y, value); they pass through
self-attention, and every query location reads them via cross-attention. The
head predicts a Gaussian over the density value at the query. The network is
meta-trained once on many shapes (train_belief_np.py) and then used frozen:
a new belief costs one forward pass instead of a GP fit.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from attentive_kernel_belief import normalize_belief


class FourierFeatures2D(nn.Module):
    def __init__(self, n_features, scale):
        super().__init__()
        self.register_buffer('W', torch.randn(2, n_features // 2) * scale)

    def forward(self, xy):
        proj = 2.0 * math.pi * xy @ self.W
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class CrossAttentionBlock(nn.Module):
    """Queries attend to the context only; queries never attend to each other."""

    def __init__(self, D, heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(D)
        self.norm_k = nn.LayerNorm(D)
        self.norm_v = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(self, q, keys, values):
        q = q + self.attn(self.norm_q(q), self.norm_k(keys), self.norm_v(values),
                          need_weights=False)[0]
        return q + self.ff(self.norm_ff(q))


class AttentiveNeuralProcess(nn.Module):
    def __init__(self, D=128, heads=4, n_self=2, n_cross=2, fourier_scale=4.0):
        super().__init__()
        self.config = dict(D=D, heads=heads, n_self=n_self, n_cross=n_cross,
                           fourier_scale=fourier_scale)
        self.pos = FourierFeatures2D(D, fourier_scale)
        self.ctx_in = nn.Sequential(nn.Linear(D + 1, D), nn.GELU(), nn.Linear(D, D))
        self.qry_in = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(D, heads, 2 * D, dropout=0.0, activation='gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(n_self)])
        self.cross_attn = nn.ModuleList([CrossAttentionBlock(D, heads) for _ in range(n_cross)])
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2))

    def encode(self, context):
        """context: (B, K, 3) with (x, y, value) -> keys, values, each (B, K, D).

        As in the ANP, keys embed only the context locations, with the same
        encoder as the queries, so attending to nearby measurements is easy to
        learn; the measured values enter through the values only.
        """
        pos = self.pos(context[..., :2])
        h = self.ctx_in(torch.cat([pos, context[..., 2:3]], dim=-1))
        for layer in self.self_attn:
            h = layer(h)
        return self.qry_in(pos), h

    def decode(self, ctx, xq):
        """ctx: (keys, values) from encode, xq: (B, Q, 2) -> mean, std, each (B, Q)."""
        keys, values = ctx
        q = self.qry_in(self.pos(xq))
        for block in self.cross_attn:
            q = block(q, keys, values)
        mean, raw_std = self.head(q).unbind(-1)
        return mean, 0.01 + F.softplus(raw_std)

    def forward(self, context, xq):
        return self.decode(self.encode(context), xq)

    @torch.no_grad()
    def predict_grid(self, context, grid_res=64, q_chunk=1024):
        """Belief grids (B, G, G) in the same format as fit_belief_grids."""
        B = context.shape[0]
        lin = torch.linspace(0.0, 1.0, grid_res, device=context.device)
        gy, gx = torch.meshgrid(lin, lin, indexing='ij')
        xq = torch.stack([gx, gy], dim=-1).reshape(1, -1, 2).expand(B, -1, -1)
        ctx = self.encode(context)
        mean = torch.cat([self.decode(ctx, xq[:, i:i + q_chunk])[0]
                          for i in range(0, xq.shape[1], q_chunk)], dim=1)
        return normalize_belief(mean.reshape(B, grid_res, grid_res))


def gaussian_nll(mean, std, target):
    return (torch.log(std) + 0.5 * ((target - mean) / std).pow(2)).mean()


def load_np(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = AttentiveNeuralProcess(**ckpt['config']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt
