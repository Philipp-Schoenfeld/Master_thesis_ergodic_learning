r"""
flow_matching_cond_particles_crossattn.py  —  3D port
=====================================================
Conditional Flow Matching Network with:
- Particle Conditioning via ParticleTokenizer  (now [x, y, z, mu])
- Cross-Attention in the U-Net bottleneck
- FiLM for time conditioning
- CFG support (condition dropout on particle tokens)

What actually changed from the 2D version
-----------------------------------------
Very little, and that is the point. The U-Net operates along the *trajectory*
axis, not along space, so its convolutions, attention and skip connections are
untouched by the coordinate dimension. Only the two places that look at raw
coordinates had to move:

* `GaussianFourierProjection3D` draws its fixed random matrix as (3, D/2)
  instead of (2, D/2).
* `ParticleTokenizer` splits its input as (x, y, z | mu) instead of (x, y | mu).

`MPDLayer` and `FlowHead` already took `nd` as a constructor argument, so
passing 3 is enough. Everything else is byte-identical in behaviour.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ND = 3


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


class GaussianFourierProjection3D(nn.Module):
    """Random Fourier features on 3D coordinates.

    Fixed (not learned) random projection, then sin/cos. This is the standard
    remedy for the spectral bias of MLPs on raw low-dimensional coordinates:
    without it the tokenizer resolves only smooth, low-frequency structure and
    blurs the sharp boundary between 'inside the shape' and 'outside'.
    """

    def __init__(self, embed_dim: int, scale: float = 1.0, nd: int = ND):
        super().__init__()
        self.W = nn.Parameter(torch.randn(nd, embed_dim // 2) * scale,
                              requires_grad=False)

    def forward(self, x: torch.Tensor):
        x_proj = x @ self.W * 2.0 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# Kept under the old name too, so 2D-era imports fail loudly rather than
# silently binding to something with different behaviour.
GaussianFourierProjection2D = None


class ParticleTokenizer(nn.Module):
    """
    Translates raw point cloud particles [x, y, z, mu] into tokens for
    cross-attention. Applies Gaussian Fourier Features to (x,y,z) to avoid
    spectral bias, then concatenates mu and processes via MLP.
    """
    def __init__(self, D: int, nd: int = ND):
        super().__init__()
        self.D = D
        self.nd = nd
        self.pos_enc = GaussianFourierProjection3D(embed_dim=D // 2, scale=1.0, nd=nd)
        self.mlp = nn.Sequential(
            nn.Linear(D // 2 + 1, D), nn.SiLU(), nn.Linear(D, D),
        )
        self.out_norm = nn.LayerNorm(D)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        particles: (B, N, nd + 1) — coordinates + density
        Returns:   (B, N, D) particle tokens
        """
        xyz = particles[:, :, :self.nd]
        mu = particles[:, :, self.nd:self.nd + 1]
        pos_feat = self.pos_enc(xyz)
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
    """1D U-Net along the trajectory axis. Unchanged by the move to 3D."""

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

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor,
                particle_tokens: torch.Tensor = None) -> torch.Tensor:
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


class OrientationHead(nn.Module):
    """Predicts a 6D rotation representation per control point (Stufe 1).

    Structurally the same MLP as `FlowHead`, but the final layer is
    zero-initialised and a constant identity encoding is added, so the head
    starts out predicting exactly the identity rotation everywhere. Same
    reasoning as the zero-initialised FiLM projections: a newly added branch
    should contribute nothing at step zero rather than inject noise into a
    network that is otherwise ready to train.

    Output is 6D and *not* projected here — the projection to SO(3) happens
    after the B-spline basis has been applied, so the interpolation stays in the
    vector space where it is valid.
    """

    def __init__(self, D: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, D), nn.LayerNorm(D), nn.SiLU(), nn.Linear(D, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # 6D encoding of the identity: first two columns of I.
        self.register_buffer('identity',
                             torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, nxi) -> (B, nxi, 6)"""
        return self.net(x.permute(0, 2, 1)) + self.identity


class MPDLayer(nn.Module):
    """Kinematic tokenisation: kernel_size=3 over the *time* axis.

    Dimension-agnostic — nd enters only as the input channel count. The kernel
    still spans (t-1, t, t+1), so the implicit finite differences that give
    velocity and acceleration work exactly as in 2D.
    """
    def __init__(self, nd: int, D: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(nd, D, kernel_size, stride=1, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(D)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.permute(0, 2, 1))
        return self.norm(h.permute(0, 2, 1)).permute(0, 2, 1)


class ParticleCrossAttnFlowNetwork(nn.Module):
    """Flow-matching network over positions, optionally over orientation too.

    With `predict_orientation=False` (the default) this is byte-identical to the
    position-only network: no extra module is constructed and the second return
    value stays None, so existing checkpoints and callers are unaffected.

    With it enabled the state the flow transports becomes
    (position, 6D rotation) in R^(3+6). That the 6D block is a plain vector
    space is what keeps the linear CFM interpolation and the MSE objective
    valid — see `orientation.py`.
    """

    def __init__(self, nxi: int = 25, nd: int = ND, D: int = 128,
                 n_heads: int = 8, kernel_size: int = 3,
                 predict_orientation: bool = False):
        super().__init__()
        self.nxi = nxi
        self.nd  = nd
        self.D   = D
        self.predict_orientation = predict_orientation
        # The sequence input carries position and, if enabled, the 6D rotation.
        self.in_dim = nd + (6 if predict_orientation else 0)

        self.mpd_layer = MPDLayer(nd=self.in_dim, D=D, kernel_size=kernel_size)
        self.pos_emb   = nn.Parameter(torch.randn(1, D, nxi) * 0.02)
        self.time_emb = SinusoidalTimeEmbedding(D=D)

        self.particle_tokenizer = ParticleTokenizer(D=D, nd=nd)

        self.null_particle_token = nn.Parameter(torch.zeros(1, 1, D))

        self.backbone = UNetBackboneParticles(D=D, n_heads=n_heads, kernel_size=kernel_size)
        self.flow_head = FlowHead(D=D, nd=nd)
        if predict_orientation:
            self.rot_head = FlowHead(D=D, nd=6)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                particles: torch.Tensor,
                cond_drop_mask: torch.Tensor = None):
        """x: (B, nxi, in_dim). Returns (v_pos, v_rot) with v_rot None if off."""
        tokens = self.mpd_layer(x)
        tokens = tokens + self.pos_emb
        time_cond = self.time_emb(t)

        particle_tokens = self.particle_tokenizer(particles)

        if cond_drop_mask is not None:
            mask = cond_drop_mask.view(-1, 1, 1).to(particle_tokens.dtype)
            particle_tokens = particle_tokens * (1.0 - mask) + self.null_particle_token * mask

        out = self.backbone(tokens, time_cond, particle_tokens)
        v_t = self.flow_head(out)
        v_rot = self.rot_head(out) if self.predict_orientation else None
        return v_t, v_rot


def compute_particle_cfm_loss(
    model: nn.Module,
    x1_batch: torch.Tensor,
    particle_batch: torch.Tensor,
    p_drop: float = 0.0,
    ergodic=None,
    orientation=None,
    w_cfm_rot: float = 1.0,
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
    v_pos, v_rot = model(xt, t, particle_batch, cond_drop_mask=cond_drop_mask)
    v_t = v_pos if v_rot is None else torch.cat([v_pos, v_rot], dim=-1)

    # Position and orientation are reported separately: they live on different
    # scales, and a single number cannot show which of the two is still moving.
    nd = model.nd
    loss_pos = torch.mean((v_t[..., :nd] - ut[..., :nd]) ** 2)
    total = loss_pos
    parts = {'cfm': loss_pos.detach()}
    if v_rot is not None:
        # `w_cfm_rot` exists so the imitation of the Stufe-0 frames can be turned
        # down or off independently of the objective terms below. The frames are
        # a geometric construction, not measured data, so three regimes are worth
        # telling apart: pure imitation (w_cfm_rot=1, no orientation loss), pure
        # objective (w_cfm_rot=0), and imitation as a warm start for it.
        loss_rot = torch.mean((v_t[..., nd:] - ut[..., nd:]) ** 2)
        total = total + w_cfm_rot * loss_rot
        parts['cfm_rot'] = loss_rot.detach()

    want_erg = ergodic is not None and ergodic.weight > 0.0
    want_ori = orientation is not None and orientation.weight > 0.0

    if want_erg or want_ori:
        # Flow matching predicts a velocity, so the endpoint has to be estimated
        # before a trajectory-level objective can be applied to it.
        x1_hat = xt + (1.0 - t_exp) * v_t
        rot_hat = x1_hat[..., nd:] if v_rot is not None else None

        # One surface for both terms: building it is a cdist over the particle
        # cloud, and the two objectives query the same geometry.
        surf = None
        if want_ori or (want_erg and getattr(ergodic, 'ergodic_on', 'position')
                        == 'footprint'):
            from orientation_energy import ParticleSurface
            thresh = getattr(orientation, 'mu_thresh', None)
            if thresh is None:
                thresh = getattr(ergodic, 'mu_thresh', 0.5)
            surf = ParticleSurface(particle_batch.float(), thresh)

        if want_erg:
            loss_erg = ergodic(x1_hat[..., :nd], particle_batch, t,
                               rot6d=rot_hat, surface=surf)
            total = total + ergodic.weight * loss_erg
            parts['erg'] = loss_erg.detach()

        if want_ori:
            if rot_hat is None:
                raise ValueError("orientation loss needs a model with a "
                                 "rotation head (--orientation)")
            loss_ori, ori_parts = orientation(x1_hat, particle_batch, t,
                                              surface=surf, return_parts=True)
            total = total + orientation.weight * loss_ori
            parts['ori'] = loss_ori.detach()
            parts.update({f'ori_{k}': v for k, v in ori_parts.items()})

    return total, parts


@torch.no_grad()
def generate_particle_trajectories(
    model: nn.Module,
    particles: torch.Tensor,
    num_samples: int = 1,
    nxi: int = 25,
    nd:  int = ND,
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

    state_dim = getattr(model, 'in_dim', nd)
    x  = torch.randn(num_samples, nxi, state_dim, device=device, generator=generator)
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

    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=device)
        t_batch = torch.cat([t, t], dim=0)
        x_batch = torch.cat([x, x], dim=0)

        v_pos, v_rot = model(x_batch, t_batch, particle_batch,
                             cond_drop_mask=mask_batch)
        v_batch = v_pos if v_rot is None else torch.cat([v_pos, v_rot], dim=-1)
        v_cond, v_null = v_batch.chunk(2, dim=0)
        v = v_null + cfg_weight * (v_cond - v_null)

        if obstacle is not None:
            # Ramp: at small t the state is still essentially Gaussian noise, so
            # repelling it is meaningless and only distorts the flow. Start at
            # t_start and grow quadratically to full strength at t = 1.
            # Only the positional block is repelled — the obstacle says nothing
            # about which way the sensor should face.
            t_now = step * dt
            if t_now >= obstacle_t_start:
                s = (t_now - obstacle_t_start) / max(1.0 - obstacle_t_start, 1e-8)
                g = curve_repulsion_grad(x[..., :nd], obstacle, B_basis)
                v = v.clone()
                v[..., :nd] = v[..., :nd] - (obstacle_weight * s ** 2) * g

        x = x + v * dt

    # Polish: the ramp alone does not guarantee hard clearance. A few pure
    # descent steps on the penalty drive the remaining violation to zero while
    # leaving the shape produced by the flow essentially untouched.
    if obstacle is not None and polish_steps > 0:
        from obstacles import polish_out_of_obstacle
        pos = polish_out_of_obstacle(x[..., :nd].contiguous(), obstacle,
                                     B_basis, max_iters=polish_steps)
        x = torch.cat([pos, x[..., nd:]], dim=-1) if x.shape[-1] > nd else pos

    # (positions, 6D rotations) — the second is None when orientation is off, so
    # the return signature stays compatible with position-only callers.
    if x.shape[-1] > nd:
        return x[..., :nd].contiguous(), x[..., nd:].contiguous()
    return x, None
