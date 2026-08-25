r"""
flow_matching_particles_selfsupervised.py  —  3D port
=====================================================
Single-pass trajectory generator for self-supervised training against the
solver energy (`ergodic_energy_torch.py`).

Unlike the flow-matching networks, this one is not a velocity field: there is no
flow time t, no Euler loop and no CFG. One forward pass maps
(noise, conditioning) directly to a trajectory, so it amortises the solver's
iterations into a single evaluation.

Reused unchanged from `flow_matching_cond_particles_crossattn.py` (the 3D one in
this folder):
    MPDLayer, ParticleTokenizer, UNetBackboneParticles, FlowHead
Dropped: SinusoidalTimeEmbedding (no flow time), null_particle_token and the
conditioning-dropout mask (CFG only makes sense with a CFM objective).

3D changes: only `nd` and the offset. The output offset that used to place the
initial trajectory at the centre of the unit square now places it at the centre
of the unit cube — same constant 0.5, but it is worth being explicit that it
applies to all three coordinates.
"""

import torch
import torch.nn as nn

from flow_matching_cond_particles_crossattn import (
    MPDLayer, ParticleTokenizer, UNetBackboneParticles, FlowHead,
    OrientationHead, ND,
)


class SelfSupervisedParticleGenerator(nn.Module):
    """(noise, particles) -> trajectory control points, in one pass.

    Args:
        nxi: number of B-spline control points emitted.
        nd:  coordinate dimension (3).
        D:   model width.
        predict_orientation: also emit a 6D rotation per control point (Stufe 1).
            Default False, in which case the module is byte-identical to the
            position-only version.

    This is the branch where orientation belongs first: there is no orientation
    ground truth anywhere in the dataset, but the pointing, standoff and angular
    smoothness terms are defined analytically, so a self-supervised generator can
    learn orientation from the objective alone.
    """

    def __init__(self, nxi: int = 25, nd: int = ND, D: int = 384,
                 n_heads: int = 8, kernel_size: int = 3,
                 predict_orientation: bool = False):
        super().__init__()
        self.nxi, self.nd, self.D = nxi, nd, D
        self.predict_orientation = predict_orientation

        self.mpd_layer = MPDLayer(nd=nd, D=D, kernel_size=kernel_size)
        self.pos_emb = nn.Parameter(torch.randn(1, D, nxi) * 0.02)
        self.particle_tokenizer = ParticleTokenizer(D=D, nd=nd)
        self.backbone = UNetBackboneParticles(D=D, n_heads=n_heads,
                                              kernel_size=kernel_size)
        self.head = FlowHead(D=D, nd=nd)
        if predict_orientation:
            self.rot_head = OrientationHead(D=D)

    def forward(self, z: torch.Tensor, particles: torch.Tensor):
        """z: (B, nxi, nd) noise, particles: (B, N, nd+1).

        Returns (cps, rot6d) with rot6d None when orientation is off.

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
        # Constant offset to the centre of the unit cube, not a sigmoid:
        # squashing would flatten the gradient exactly at the domain edge, which
        # is where the boundary term works.
        cps = 0.5 + self.head(out)
        rot6d = self.rot_head(out) if self.predict_orientation else None
        return cps, rot6d

    @torch.no_grad()
    def generate(self, particles: torch.Tensor, num_samples: int = 1,
                 device: str = 'cpu', generator: torch.Generator = None):
        """Convenience sampler. particles: (N, nd+1) or (B, N, nd+1).

        Returns (cps, rot6d); rot6d is None unless orientation is enabled.
        """
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
                                diversity_weight=0.0, generator=None,
                                fields=None):
    """Energy of K candidates per target, minus their diversity.

    Args:
        particles: (B, N, nd+1) conditioning for B targets.
        phi_k:     (B, M) target coefficients for the same B targets.
        energy:    an `ErgodicEnergy` or an `SE3Energy`.
        n_candidates: K trajectories generated per target from different noise.
        fields:    list of B `SurfaceField`s, required once the model predicts
                   orientation — the pointing and standoff terms need something
                   to point at.

    Returns:
        (loss, parts) where parts holds the unweighted pieces for logging.
    """
    from ergodic_energy_torch import diversity_reward_batched

    B, K = particles.shape[0], n_candidates
    nxi, nd = model.nxi, model.nd

    z = torch.randn(B * K, nxi, nd, device=particles.device, generator=generator)
    cond = particles.repeat_interleave(K, dim=0) if K > 1 else particles
    xi, rot6d = model(z, cond)                               # (B*K, nxi, ·)

    # float32 on purpose: under autocast(bfloat16) the ergodic term is a
    # difference of coefficients of order 0.1-1 whose gap can be ~1e-3, and
    # bfloat16's ~3 decimal digits would erase it before it is ever squared.
    with torch.autocast(device_type=particles.device.type, enabled=False):
        phi = phi_k.repeat_interleave(K, dim=0) if K > 1 else phi_k

        if rot6d is None:
            E, terms = energy(xi.float(), phi.float(), return_terms=True)
        else:
            if fields is None:
                raise ValueError("orientation is enabled but no SurfaceFields "
                                 "were passed to the loss")
            # Each target has its own field, so the K candidates of one target
            # are scored per target and then concatenated.
            Es, term_list = [], []
            for b in range(B):
                sl = slice(b * K, (b + 1) * K)
                e_b, t_b = energy(xi[sl].float(), phi[sl].float(),
                                  rot6d=rot6d[sl].float(), field=fields[b],
                                  return_terms=True)
                Es.append(e_b)
                term_list.append(t_b)
            E = torch.cat(Es)
            terms = {k: torch.cat([t[k] for t in term_list])
                     for k in term_list[0]}

        loss = E.mean()
        parts = {'energy': E.mean().detach()}
        for k, v in terms.items():
            parts[k] = v.mean().detach()

        if K > 1 and diversity_weight > 0.0:
            div = diversity_reward_batched(xi.float().view(B, K, nxi, nd))
            loss = loss - diversity_weight * div
            parts['diversity'] = div.detach()

    return loss, parts
