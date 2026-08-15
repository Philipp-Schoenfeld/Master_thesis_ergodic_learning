r"""
flow_matching_particles_selfsupervised.py
=========================================
Single-pass trajectory generator for self-supervised training against the
solver energy (`ergodic_energy_torch.py`).

Unlike the flow-matching networks, this one is not a velocity field: there is no
flow time t, no Euler loop and no CFG. One forward pass maps
(noise, conditioning) directly to a trajectory, so it amortises the solver's 600
iterations into a single evaluation.

Reused unchanged from `flow_matching_cond_particles_crossattn.py`:
    MPDLayer, ParticleTokenizer, UNetBackboneParticles, FlowHead
Dropped: SinusoidalTimeEmbedding (no flow time), null_particle_token and the
conditioning-dropout mask (CFG only makes sense with a CFM objective).

The comparison baseline (`flow_matching_cond_particles_crossattn.py`,
`flow_matching_runner_particles.py`) is imported from, never modified.
"""

import torch
import torch.nn as nn

from flow_matching_cond_particles_crossattn import (
    MPDLayer, ParticleTokenizer, UNetBackboneParticles, FlowHead,
)


class SelfSupervisedParticleGenerator(nn.Module):
    """(noise, particles) -> trajectory control points, in one pass.

    Args:
        nxi: number of B-spline control points emitted.
        nd:  coordinate dimension (2).
        D:   model width.
    """

    def __init__(self, nxi: int = 25, nd: int = 2, D: int = 384,
                 n_heads: int = 8, kernel_size: int = 3):
        super().__init__()
        self.nxi, self.nd, self.D = nxi, nd, D

        self.mpd_layer = MPDLayer(nd=nd, D=D, kernel_size=kernel_size)
        self.pos_emb = nn.Parameter(torch.randn(1, D, nxi) * 0.02)
        self.particle_tokenizer = ParticleTokenizer(D=D)
        self.backbone = UNetBackboneParticles(D=D, n_heads=n_heads,
                                              kernel_size=kernel_size)
        self.head = FlowHead(D=D, nd=nd)

    def forward(self, z: torch.Tensor, particles: torch.Tensor) -> torch.Tensor:
        """z: (B, nxi, nd) noise, particles: (B, N, 3) -> (B, nxi, nd).

        The noise enters as the *sequence input*, exactly where x0 entered the
        flow-matching net. A global z routed through the FiLM path instead would
        be inert at initialisation, because `ConvResBlock.film_proj` is
        zero-initialised — there would be no gradient separating the K
        candidates, which is precisely where the diversity term has to bite.

        `time_cond=None` is safe: ConvResBlock guards with `if cond is not None`,
        so no dead FiLM path is left behind.
        """
        tokens = self.mpd_layer(z) + self.pos_emb
        particle_tokens = self.particle_tokenizer(particles)
        out = self.backbone(tokens, None, particle_tokens)
        # Constant offset, not a sigmoid: squashing would flatten the gradient
        # exactly at the domain edge, which is where the boundary term works.
        return 0.5 + self.head(out)

    @torch.no_grad()
    def generate(self, particles: torch.Tensor, num_samples: int = 1,
                 device: str = 'cpu', generator: torch.Generator = None):
        """Convenience sampler for visualisation. particles: (N, 3) or (B, N, 3)."""
        self.eval()
        if particles.dim() == 2:
            particles = particles.unsqueeze(0)
        if particles.shape[0] == 1 and num_samples > 1:
            particles = particles.expand(num_samples, -1, -1).contiguous()
        particles = particles.to(device)
        z = torch.randn(particles.shape[0], self.nxi, self.nd,
                        device=device, generator=generator)
        return self(z, particles)


def compute_selfsupervised_loss(model, particles, phi_k, energy, n_candidates=1,
                                diversity_weight=0.0, generator=None):
    """Energy of K candidates per target, minus their diversity.

    Args:
        particles: (B, N, 3) conditioning for B targets.
        phi_k:     (B, M) target coefficients for the same B targets.
        energy:    an `ErgodicEnergy` module.
        n_candidates: K trajectories generated per target from different noise.

    Returns:
        (loss, parts) where parts holds the unweighted pieces for logging.
    """
    from ergodic_energy_torch import diversity_reward_batched

    B, K = particles.shape[0], n_candidates
    nxi, nd = model.nxi, model.nd

    z = torch.randn(B * K, nxi, nd, device=particles.device, generator=generator)
    cond = particles.repeat_interleave(K, dim=0) if K > 1 else particles
    xi = model(z, cond)                                      # (B*K, nxi, nd)

    # float32 on purpose: under autocast(bfloat16) the ergodic term is a
    # difference of coefficients of order 0.1-1 whose gap can be ~1e-3, and
    # bfloat16's ~3 decimal digits would erase it before it is ever squared.
    with torch.autocast(device_type=particles.device.type, enabled=False):
        phi = phi_k.repeat_interleave(K, dim=0) if K > 1 else phi_k
        E = energy(xi.float(), phi.float())                  # (B*K,)
        loss = E.mean()
        parts = {'energy': E.mean().detach()}

        if K > 1 and diversity_weight > 0.0:
            div = diversity_reward_batched(xi.float().view(B, K, nxi, nd))
            loss = loss - diversity_weight * div
            parts['diversity'] = div.detach()

    return loss, parts
