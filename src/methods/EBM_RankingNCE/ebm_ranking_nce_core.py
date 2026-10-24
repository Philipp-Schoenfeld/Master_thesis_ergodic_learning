r"""
ebm_ranking_nce_core.py
========================
Energy-based policy for ergodic-coverage trajectory generation, trained with
the debiased Ranking-NCE objective rather than the classic Implicit Behavior
Cloning (IBC) loss.

Why Ranking-NCE instead of plain InfoNCE/IBC
----------------------------------------------
Original IBC (Florence et al.) trains an energy E(context, action) with a
softmax cross-entropy over a batch of counter-example actions, implicitly
assuming the negative-sampling proposal has *constant* density across
candidates. Once negatives are drawn from a non-uniform proposal (e.g. a
derivative-free optimizer that concentrates samples near the current energy
minimum, or Gaussian noise around the ground truth), that assumption breaks
and the objective is biased at the population level (see arXiv:2309.05803).

The fix applied here is the standard multi-class noise-contrastive estimation
correction: every candidate's logit is `-E(candidate) - log q(candidate)`,
i.e. the energy is corrected by the *known* log-density of the proposal that
generated it. The softmax normalizer Z(c) of the true unnormalized
distribution cancels in the softmax regardless, so this is exact (not an
approximation) as long as q is known in closed form -- which it is here,
since negatives are Gaussian perturbations of the ground-truth action at a
sampled noise scale sigma (a "noise-to-data" family of EBMs, several sigma
scales mixed across a batch rather than a single fixed one).

Action / context tensor conventions (matching the flow-matching baselines in
thesis_architecture/):
  - trajectory / action:     (B, nxi, nd)   -- B-spline control points
  - particle conditioning:   (B, N, 3)      -- [x, y, density weight]
  - spectral conditioning:   (B, S, 2)      -- [real, imag] per coefficient,
                                                paired with k_indices (S, 2)

This module is self-contained (no dependency on thesis_architecture) so it
can be unit-tested in isolation; wiring it into the shared dataset loaders is
a separate step once the method is adopted into the comparison matrix.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Context encoders
# ----------------------------------------------------------------------------

class GaussianFourierProjection2D(nn.Module):
    """Fixed random Fourier features for 2D coordinates (avoids spectral bias)."""

    def __init__(self, embed_dim: int, scale: float = 1.0):
        super().__init__()
        assert embed_dim % 2 == 0
        self.W = nn.Parameter(torch.randn(2, embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x @ self.W * 2.0 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ParticleSetEncoder(nn.Module):
    """
    Permutation-invariant encoder for point-cloud conditioning.

    Note on pooling: this mean-pools over the *conditioning* particle set to
    obtain a single context vector for the scalar energy head. That is a
    standard permutation-invariant set encoding and is unrelated to the
    project-wide ban on global-average-pooling the *trajectory*/target-density
    tokens inside the main VITA cross-attention architecture -- there is no
    cross-attention here to collapse, only a context summary feeding an MLP.
    """

    def __init__(self, D: int):
        super().__init__()
        self.pos_enc = GaussianFourierProjection2D(embed_dim=D // 2, scale=1.0)
        self.mlp = nn.Sequential(
            nn.Linear(D // 2 + 1, D), nn.SiLU(), nn.Linear(D, D),
        )
        self.out_norm = nn.LayerNorm(D)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """particles: (B, N, 3) -> (B, D) pooled context embedding."""
        xy = particles[:, :, :2]
        mu = particles[:, :, 2:3]
        feat = torch.cat([self.pos_enc(xy), mu], dim=-1)
        tokens = self.out_norm(self.mlp(feat))         # (B, N, D)
        return tokens.mean(dim=1)                       # (B, D)


class SpectralSetEncoder(nn.Module):
    """Encodes leading spectral coefficients (real/imag + frequency) into a single context vector."""

    def __init__(self, S: int, D: int):
        super().__init__()
        self.S = S
        self.freq_proj = nn.Linear(2, D // 2)
        self.mlp = nn.Sequential(
            nn.Linear(D // 2 + 2, D), nn.SiLU(), nn.Linear(D, D),
        )
        self.out_norm = nn.LayerNorm(D)

    def forward(self, spec_ri: torch.Tensor, k_indices: torch.Tensor) -> torch.Tensor:
        """
        spec_ri:   (B, S, 2) real/imag coefficients
        k_indices: (S, 2) integer frequency indices, shared across the batch
        Returns:   (B, D) pooled context embedding
        """
        B = spec_ri.shape[0]
        freq_feat = self.freq_proj(k_indices.float())            # (S, D//2)
        freq_feat = freq_feat.unsqueeze(0).expand(B, -1, -1)      # (B, S, D//2)
        feat = torch.cat([freq_feat, spec_ri], dim=-1)            # (B, S, D//2+2)
        tokens = self.out_norm(self.mlp(feat))                    # (B, S, D)
        return tokens.mean(dim=1)                                  # (B, D)


# ----------------------------------------------------------------------------
# Trajectory (action) encoder
# ----------------------------------------------------------------------------

class TrajectoryEncoder(nn.Module):
    """
    1D-CNN encoder over B-spline control points, kernel_size=3, following the
    project convention that a kernel_size=3 temporal convolution implicitly
    computes finite differences (velocity/acceleration) without needing an
    explicit derivative channel in the state vector.
    """

    def __init__(self, nd: int, D: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(nd, D, kernel_size, padding=pad), nn.SiLU(),
            nn.Conv1d(D, D, kernel_size, padding=pad), nn.SiLU(),
            nn.Conv1d(D, D, kernel_size, padding=pad),
        )
        self.out_norm = nn.LayerNorm(D)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, nxi, nd) -> (B, D)"""
        x = action.permute(0, 2, 1)       # (B, nd, nxi)
        x = self.net(x)                    # (B, D, nxi)
        x = x.mean(dim=-1)                 # (B, D)  (pool over control points, not over context)
        return self.out_norm(x)


class SigmaEmbedding(nn.Module):
    """Sinusoidal embedding of the noise scale sigma, for the noise-conditional energy head."""

    def __init__(self, D: int):
        super().__init__()
        assert D % 2 == 0
        half = D // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10_000.0) / max(half - 1, 1)))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))

    def forward(self, log_sigma: torch.Tensor) -> torch.Tensor:
        """log_sigma: (B,) -> (B, D)"""
        args = log_sigma[:, None] * self.freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class StartEmbedding(nn.Module):
    """
    Embeds a global robot start point p0 into the energy head, mirroring the
    FiLM start-point design in flow_matching_cond_particles_start.py: the
    start point is a single value for the whole trajectory (not a per-token
    feature), so it is concatenated once into the pooled head features rather
    than added as another particle/trajectory token.

    The final linear layer is zero-initialized so the start point has no
    effect at the beginning of training (matches the project's AdaLN-Zero /
    FiLM-zero convention) -- the energy starts out identical to the
    start-point-agnostic model and only grows a dependency on p0 as training
    proceeds, rather than destabilizing the loss from step 0.
    """

    def __init__(self, D: int):
        super().__init__()
        self.pos_enc = GaussianFourierProjection2D(embed_dim=D, scale=1.0)
        self.proj = nn.Linear(D, D)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        """p0: (B, 2) -> (B, D)"""
        return self.proj(self.pos_enc(p0))


# ----------------------------------------------------------------------------
# Energy network
# ----------------------------------------------------------------------------

class EnergyNet(nn.Module):
    """
    Scalar energy E(context, action[, sigma]) for a B-spline control-point
    trajectory under a given target-density context.

    `context_mode` selects which conditioning the context encoder expects:
      - 'particles': context tensor is (B, N, 3)
      - 'spectral':  context tensor is (B, S, 2), plus a shared k_indices (S,2)
    """

    def __init__(self, context_mode: str, nxi: int, nd: int, D: int = 128,
                 S: Optional[int] = None, use_sigma_cond: bool = True,
                 use_start_cond: bool = False):
        super().__init__()
        assert context_mode in ("particles", "spectral")
        self.context_mode = context_mode
        self.nxi, self.nd = nxi, nd
        self.use_sigma_cond = use_sigma_cond
        self.use_start_cond = use_start_cond

        if context_mode == "particles":
            self.context_encoder = ParticleSetEncoder(D)
        else:
            assert S is not None, "S (number of spectral coefficients) is required for context_mode='spectral'"
            self.context_encoder = SpectralSetEncoder(S, D)

        self.traj_encoder = TrajectoryEncoder(nd, D)
        self.sigma_embed = SigmaEmbedding(D) if use_sigma_cond else None
        self.start_embed = StartEmbedding(D) if use_start_cond else None

        head_in = D * (2 + int(use_sigma_cond) + int(use_start_cond))
        self.head = nn.Sequential(
            nn.Linear(head_in, D), nn.SiLU(),
            nn.Linear(D, D), nn.SiLU(),
            nn.Linear(D, 1),
        )

    def encode_context(self, context: torch.Tensor, k_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.context_mode == "spectral":
            assert k_indices is not None, "k_indices is required for context_mode='spectral'"
            return self.context_encoder(context, k_indices)
        return self.context_encoder(context)

    def forward(self, context_emb: torch.Tensor, action: torch.Tensor,
                log_sigma: Optional[torch.Tensor] = None,
                start: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        context_emb: (B, D) -- already pooled via `encode_context`
        action:      (B, nxi, nd)
        log_sigma:   (B,) or None
        start:       (B, 2) robot start point, or None -- required if
                     `use_start_cond=True` (see StartEmbedding)
        Returns:     (B,) scalar energy
        """
        traj_emb = self.traj_encoder(action)
        feats = [context_emb, traj_emb]
        if self.use_sigma_cond:
            if log_sigma is None:
                log_sigma = torch.zeros(action.shape[0], device=action.device)
            feats.append(self.sigma_embed(log_sigma))
        if self.use_start_cond:
            assert start is not None, "start (B, 2) is required when use_start_cond=True"
            feats.append(self.start_embed(start))
        x = torch.cat(feats, dim=-1)
        return self.head(x).squeeze(-1)


# ----------------------------------------------------------------------------
# Ranking-NCE loss
# ----------------------------------------------------------------------------

def sample_log_uniform_sigma(batch_size: int, sigma_min: float, sigma_max: float,
                              device: torch.device) -> torch.Tensor:
    """Draws (B,) noise scales log-uniformly in [sigma_min, sigma_max]."""
    u = torch.rand(batch_size, device=device)
    log_sigma = math.log(sigma_min) + u * (math.log(sigma_max) - math.log(sigma_min))
    return log_sigma.exp()


def ranking_nce_loss(energy_net: EnergyNet, context: torch.Tensor, action_pos: torch.Tensor,
                      k_indices: Optional[torch.Tensor] = None, n_negatives: int = 8,
                      sigma_min: float = 0.01, sigma_max: float = 1.0,
                      energy_reg: float = 1e-3,
                      start: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
    """
    Multi-class Ranking-NCE loss with a noise-to-data Gaussian proposal family.

    For each (context_i, action_pos_i), draws `n_negatives` Gaussian negatives
    at a per-sample noise scale sigma_i ~ LogUniform(sigma_min, sigma_max), and
    trains the energy so that the softmax over
        logit_j = -E(context_i, candidate_j) - log q(candidate_j | sigma_i)
    places candidate 0 (the ground-truth action) first. The -log q correction
    makes this consistent (unbiased in the population limit) regardless of how
    skewed the Gaussian proposal density is -- unlike dropping that term, which
    is exactly the bias in the original IBC formulation. `sigma_max` should
    span a sizeable fraction of the actual action-space scale (not just a
    small local wiggle) so the noise family also shapes the energy away from
    the immediate neighborhood of each ground truth, not only right at it.

    `energy_reg` adds a small `energy_reg * mean(E^2)` penalty over all
    candidates. Without it the softmax is invariant to adding any constant
    offset per logit, so nothing stops the unconstrained output head from
    drifting to an arbitrarily large magnitude; this is the standard
    stabilization used when training EBMs (e.g. IGEBM-style energy
    regularization) and keeps `langevin_refine`/`cem_optimize` well-behaved.

    Returns (loss, metrics) where metrics contains 'acc' (top-1 ranking
    accuracy of the ground-truth action among the candidates, a useful
    training-health signal) and 'sigma_mean'.
    """
    B, nxi, nd = action_pos.shape
    D_act = nxi * nd
    device = action_pos.device

    sigma = sample_log_uniform_sigma(B, sigma_min, sigma_max, device)   # (B,)
    eps = torch.randn(B, n_negatives, nxi, nd, device=device)
    negatives = action_pos.unsqueeze(1) + sigma.view(B, 1, 1, 1) * eps   # (B, K, nxi, nd)
    candidates = torch.cat([action_pos.unsqueeze(1), negatives], dim=1)  # (B, K+1, nxi, nd)
    n_cand = candidates.shape[1]

    # log q(candidate | sigma): index 0 has zero offset, index >=1 has offset sigma*eps
    log_norm_const = -0.5 * D_act * torch.log(2.0 * math.pi * sigma ** 2)          # (B,)
    sq_eps = eps.pow(2).sum(dim=(-2, -1))                                           # (B, K)
    log_q_pos = log_norm_const                                                      # (B,)
    log_q_neg = log_norm_const.unsqueeze(1) - 0.5 * sq_eps                          # (B, K)
    log_q = torch.cat([log_q_pos.unsqueeze(1), log_q_neg], dim=1)                   # (B, K+1)

    context_emb = energy_net.encode_context(context, k_indices)                     # (B, D)
    context_emb_rep = context_emb.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)
    candidates_flat = candidates.reshape(B * n_cand, nxi, nd)
    sigma_rep = sigma.unsqueeze(1).expand(-1, n_cand).reshape(B * n_cand)
    log_sigma_rep = sigma_rep.log()
    start_rep = (start.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)
                 if start is not None else None)

    energies = energy_net(context_emb_rep, candidates_flat, log_sigma_rep, start_rep).view(B, n_cand)  # (B, K+1)
    logits = -energies - log_q

    target = torch.zeros(B, dtype=torch.long, device=device)   # ground truth is always index 0
    loss = F.cross_entropy(logits, target) + energy_reg * energies.pow(2).mean()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == 0).float().mean()

    return loss, {"acc": acc.item(), "sigma_mean": sigma.mean().item()}


# ----------------------------------------------------------------------------
# Inference: ranking, derivative-free (CEM) and gradient-based refinement
# ----------------------------------------------------------------------------

@torch.no_grad()
def rank_candidates(energy_net: EnergyNet, context: torch.Tensor, candidates: torch.Tensor,
                     k_indices: Optional[torch.Tensor] = None,
                     start: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scores and ranks several candidate trajectories per context (e.g. several
    CFM samples), lowest energy first.

    context:    per the `context_mode` convention, batch size B
    candidates: (B, M, nxi, nd)
    start:      (B, 2) or None, required if the net uses start conditioning
    Returns:    (energies (B, M), sort_idx (B, M) ascending by energy)
    """
    B, M, nxi, nd = candidates.shape
    context_emb = energy_net.encode_context(context, k_indices)                      # (B, D)
    context_emb_rep = context_emb.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
    cand_flat = candidates.reshape(B * M, nxi, nd)
    start_rep = start.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1) if start is not None else None
    energies = energy_net(context_emb_rep, cand_flat, start=start_rep).view(B, M)
    sort_idx = energies.argsort(dim=1)
    return energies, sort_idx


@torch.no_grad()
def cem_optimize(energy_net: EnergyNet, context: torch.Tensor, nxi: int, nd: int,
                  k_indices: Optional[torch.Tensor] = None, n_iters: int = 10,
                  n_samples: int = 256, elite_frac: float = 0.1,
                  init_mean: Optional[torch.Tensor] = None, init_std: float = 1.0,
                  start: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Derivative-free Cross-Entropy Method optimization of the action that
    minimizes the energy, batched over the context's leading dimension.

    Intended use is *local* search around a given warm start (e.g. a CFM
    sample), not global search from scratch: a Ranking-NCE energy trained
    with the noise-to-data family above only has well-shaped gradients/levels
    near the data manifold it was contrasted against (analogous to why
    denoising/score-matching models need an annealed sampler rather than a
    naive optimizer run from an arbitrary point). Pass `init_mean` close to a
    plausible trajectory (not zeros/large `init_std`) for this to reliably
    descend towards the true action rather than an unconstrained far-away
    region the energy was never trained to penalize.

    Returns (best_action (B, nxi, nd), best_energy (B,)).
    """
    B = context.shape[0]
    device = context.device
    D_act = nxi * nd
    n_elite = max(1, int(round(n_samples * elite_frac)))

    context_emb = energy_net.encode_context(context, k_indices)   # (B, D)

    mean = (init_mean.reshape(B, D_act).clone() if init_mean is not None
            else torch.zeros(B, D_act, device=device))
    std = torch.full((B, D_act), init_std, device=device)

    for _ in range(n_iters):
        samples = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(B, n_samples, D_act, device=device)
        actions = samples.view(B, n_samples, nxi, nd)

        context_emb_rep = context_emb.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)
        actions_flat = actions.reshape(B * n_samples, nxi, nd)
        start_rep = (start.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)
                     if start is not None else None)
        energies = energy_net(context_emb_rep, actions_flat, start=start_rep).view(B, n_samples)

        elite_idx = energies.argsort(dim=1)[:, :n_elite]                               # (B, n_elite)
        elite = torch.gather(samples, 1, elite_idx.unsqueeze(-1).expand(-1, -1, D_act))  # (B, n_elite, D_act)
        mean = elite.mean(dim=1)
        std = elite.std(dim=1) + 1e-6

    best_action = mean.view(B, nxi, nd)
    best_energy = energy_net(context_emb, best_action, start=start)
    return best_action, best_energy


def langevin_refine(energy_net: EnergyNet, context: torch.Tensor, action_init: torch.Tensor,
                     k_indices: Optional[torch.Tensor] = None, n_steps: int = 20,
                     step_size: float = 1e-2, noise_scale: float = 0.0,
                     start: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient-based refinement of a given warm-start action by descending the
    energy landscape (plain gradient descent for noise_scale=0, Langevin
    dynamics otherwise). Intended for "recovering" an out-of-distribution CFM
    sample without needing a full re-solve.

    Returns (refined_action (B, nxi, nd), energy_history (n_steps+1, B)).
    """
    context_emb = energy_net.encode_context(context, k_indices).detach()
    action = action_init.clone().detach().requires_grad_(True)
    history = []

    with torch.no_grad():
        history.append(energy_net(context_emb, action, start=start).detach().clone())

    for _ in range(n_steps):
        energy = energy_net(context_emb, action, start=start)
        grad, = torch.autograd.grad(energy.sum(), action)
        with torch.no_grad():
            noise = (noise_scale * math.sqrt(2.0 * step_size) * torch.randn_like(action)
                     if noise_scale > 0 else 0.0)
            action = (action - step_size * grad + noise).detach().requires_grad_(True)
            history.append(energy_net(context_emb, action, start=start).detach().clone())

    return action.detach(), torch.stack(history, dim=0)
