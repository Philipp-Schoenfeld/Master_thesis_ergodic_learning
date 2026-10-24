#!/usr/bin/env python3
"""
test_ebm_ranking_nce.py
========================
Sanity checks for the Ranking-NCE energy-based policy (`ebm_ranking_nce_core.py`).

Not a correctness-against-reference test (there is no existing reference
implementation in this repo) -- instead verifies the properties that matter
for this to be a usable second generative head:
  1. shapes / no-NaN forward passes for both conditioning modes
  2. the Ranking-NCE loss is actually learnable (ranking accuracy rises,
     ground-truth action ends up lower-energy than noise negatives)
  3. `rank_candidates` prefers a near-ground-truth candidate over a far one
  4. `cem_optimize` (derivative-free) refines a warm start into a
     lower-energy, closer-to-truth action
  5. `langevin_refine` (gradient-based) reduces energy of a perturbed
     warm-start -- the "recover an out-of-distribution CFM sample" use case
  6. the log-uniform noise schedule respects its bounds

Usage:
  python -u test_ebm_ranking_nce.py
"""

import os
import sys
import math

import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from ebm_ranking_nce_core import (
    EnergyNet,
    ranking_nce_loss,
    rank_candidates,
    cem_optimize,
    langevin_refine,
    sample_log_uniform_sigma,
)

torch.manual_seed(0)

FAILED = []


def check(name, ok, detail=''):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def make_synthetic_spectral_problem(n_contexts, S, nxi, nd, device):
    """Fixed random contexts, each paired with one fixed ground-truth action."""
    contexts = torch.randn(n_contexts, S, 2, device=device)
    actions = torch.randn(n_contexts, nxi, nd, device=device)
    k_indices = torch.stack([
        torch.arange(S, device=device) % 4,
        torch.arange(S, device=device) // 4,
    ], dim=-1)
    return contexts, actions, k_indices


def main():
    device = torch.device('cpu')
    S, nxi, nd, D = 6, 8, 2, 64
    n_contexts = 16

    contexts, true_actions, k_indices = make_synthetic_spectral_problem(n_contexts, S, nxi, nd, device)

    # ── 1. shapes / forward sanity (spectral mode) ──────────────────────────
    net = EnergyNet(context_mode='spectral', nxi=nxi, nd=nd, D=D, S=S).to(device)
    ctx_emb = net.encode_context(contexts, k_indices)
    check('spectral context embedding has shape (B, D)', tuple(ctx_emb.shape) == (n_contexts, D))

    energies = net(ctx_emb, true_actions)
    check('energy forward returns shape (B,)', tuple(energies.shape) == (n_contexts,))
    check('initial energies are finite', torch.isfinite(energies).all().item())

    # ── 2. Ranking-NCE loss is learnable ─────────────────────────────────────
    # sigma spans from small local wiggles to a sizeable fraction of the actual
    # action scale (true_actions ~ N(0,1) per dim in nxi*nd=16 dims), so the
    # noise-to-data family also shapes the energy beyond the immediate
    # neighborhood of each ground truth -- see `cem_optimize` docstring.
    loss_kwargs = dict(n_negatives=8, sigma_min=0.05, sigma_max=3.0)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    loss0, _ = ranking_nce_loss(net, contexts, true_actions, k_indices, **loss_kwargs)
    for _ in range(1500):
        opt.zero_grad()
        loss, metrics = ranking_nce_loss(net, contexts, true_actions, k_indices, **loss_kwargs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()

    # average over several fresh noise draws -- a single training step's
    # minibatch accuracy is noisy, since it resamples both sigma and eps
    with torch.no_grad():
        eval_accs = [ranking_nce_loss(net, contexts, true_actions, k_indices, **loss_kwargs)[1]['acc']
                     for _ in range(20)]
    mean_acc = sum(eval_accs) / len(eval_accs)

    check('loss decreases with training', loss.item() < loss0.item(),
          f'{loss0.item():.3f} -> {loss.item():.3f}')
    check('ranking accuracy improves to a high value', mean_acc > 0.7,
          f'mean_acc={mean_acc:.3f}')

    with torch.no_grad():
        ctx_emb = net.encode_context(contexts, k_indices)
        e_true = net(ctx_emb, true_actions)
        random_negatives = torch.randn_like(true_actions)
        e_random = net(ctx_emb, random_negatives)
    check('ground-truth actions end up lower-energy than random actions',
          (e_true < e_random).float().mean().item() > 0.85,
          f'mean(E_true)={e_true.mean():.2f}, mean(E_random)={e_random.mean():.2f}')

    # ── 3. rank_candidates prefers near-truth over far candidates ───────────
    near = true_actions + 0.05 * torch.randn_like(true_actions)
    far = true_actions + 3.0 * torch.randn_like(true_actions)
    candidates = torch.stack([near, far], dim=1)   # (B, 2, nxi, nd)
    cand_energies, sort_idx = rank_candidates(net, contexts, candidates, k_indices)
    check('rank_candidates ranks the near-truth candidate first',
          (sort_idx[:, 0] == 0).float().mean().item() > 0.85,
          f'frac_near_first={(sort_idx[:, 0] == 0).float().mean().item():.3f}')

    # ── 4. CEM (derivative-free) local search around a warm start ───────────
    # As documented on `cem_optimize`: this noise-to-data energy only has a
    # well-shaped landscape near each ground truth, so the realistic use case
    # (and the one tested here) is local refinement around a decent warm
    # start -- e.g. a CFM sample -- not a global search from zero/pure noise.
    # Uses the full 16-context batch and a majority/mean criterion rather than
    # requiring every single item to improve, since both training (only 1200
    # steps on 16 fixed contexts) and the warm-start noise are stochastic.
    warm_start = true_actions + 0.6 * torch.randn_like(true_actions)   # stand-in for an imperfect CFM sample
    init_mean = warm_start.view(n_contexts, nxi * nd)
    best_action, best_energy = cem_optimize(net, contexts, nxi, nd, k_indices,
                                             n_iters=10, n_samples=256, elite_frac=0.1,
                                             init_mean=init_mean, init_std=0.4)
    with torch.no_grad():
        ctx_emb_all = net.encode_context(contexts, k_indices)
        init_energy = net(ctx_emb_all, warm_start)
    check('CEM reduces mean energy relative to the warm start',
          best_energy.mean().item() < init_energy.mean().item(),
          f'init={init_energy.mean():.2f}, cem={best_energy.mean():.2f}')

    dist_warm = (warm_start - true_actions).pow(2).sum(dim=(-2, -1)).sqrt()
    dist_cem = (best_action - true_actions).pow(2).sum(dim=(-2, -1)).sqrt()
    frac_improved = (dist_cem < dist_warm).float().mean().item()
    check('CEM moves most contexts closer to the true action than the warm start',
          frac_improved > 0.7 and dist_cem.mean().item() < dist_warm.mean().item(),
          f'frac_improved={frac_improved:.2f}, mean dist_warm={dist_warm.mean():.3f}, mean dist_cem={dist_cem.mean():.3f}')

    # ── 5. Langevin / gradient refinement recovers a perturbed warm-start ───
    perturbed = true_actions + 1.0 * torch.randn_like(true_actions)
    refined, history = langevin_refine(net, contexts, perturbed, k_indices,
                                        n_steps=30, step_size=5e-3, noise_scale=0.0)
    frac_energy_improved = (history[-1] < history[0]).float().mean().item()
    check('gradient refinement decreases energy for most contexts',
          frac_energy_improved > 0.7 and history[-1].mean().item() < history[0].mean().item(),
          f'frac_improved={frac_energy_improved:.2f}, before={history[0].mean():.2f}, after={history[-1].mean():.2f}')
    dist_perturbed = (perturbed - true_actions).pow(2).sum(dim=(-2, -1)).sqrt()
    dist_refined = (refined - true_actions).pow(2).sum(dim=(-2, -1)).sqrt()
    check('gradient refinement moves closer to the true action on average',
          dist_refined.mean().item() < dist_perturbed.mean().item(),
          f'before={dist_perturbed.mean():.3f}, after={dist_refined.mean():.3f}')
    check('refined action has no NaNs/Infs', torch.isfinite(refined).all().item())

    # ── 6. particle-conditioning mode: shape / no-NaN sanity only ────────────
    net_particles = EnergyNet(context_mode='particles', nxi=nxi, nd=nd, D=D)
    particles = torch.rand(5, 30, 3)
    actions_p = torch.randn(5, nxi, nd)
    loss_p, metrics_p = ranking_nce_loss(net_particles, particles, actions_p,
                                          k_indices=None, n_negatives=4)
    loss_p.backward()
    grad_norm = sum(p.grad.norm().item() for p in net_particles.parameters() if p.grad is not None)
    check('particle-conditioning loss is finite', torch.isfinite(loss_p).item())
    check('particle-conditioning backward produces nonzero gradients', grad_norm > 0.0)

    # ── 7b. start-point conditioning: zero-init inertness + gradient flow ──
    net_start = EnergyNet(context_mode='particles', nxi=nxi, nd=nd, D=D, use_start_cond=True)
    particles_s = torch.rand(5, 30, 3)
    actions_s = torch.randn(5, nxi, nd)
    start_a = torch.rand(5, 2)
    start_b = torch.rand(5, 2)
    with torch.no_grad():
        ctx_emb_s = net_start.encode_context(particles_s)
        e_a = net_start(ctx_emb_s, actions_s, start=start_a)
        e_b = net_start(ctx_emb_s, actions_s, start=start_b)
    check('start conditioning is inert at zero-init (start_embed zero-initialized)',
          torch.allclose(e_a, e_b, atol=1e-6),
          f'max diff={torch.abs(e_a - e_b).max().item():.2e}')

    loss_s, _ = ranking_nce_loss(net_start, particles_s, actions_s, k_indices=None,
                                  n_negatives=4, start=start_a)
    loss_s.backward()
    start_grad_norm = net_start.start_embed.proj.weight.grad.norm().item()
    check('start-embedding receives gradient through the Ranking-NCE loss', start_grad_norm > 0.0,
          f'grad_norm={start_grad_norm:.2e}')

    # ── 7. noise schedule respects its bounds ────────────────────────────────
    sigmas = sample_log_uniform_sigma(10_000, sigma_min=0.05, sigma_max=0.5, device=device)
    check('log-uniform noise schedule stays within [sigma_min, sigma_max]',
          (sigmas >= 0.05).all().item() and (sigmas <= 0.5).all().item())
    expected_log_mean = 0.5 * (math.log(0.05) + math.log(0.5))
    log_mean = sigmas.log().mean().item()
    check('log-uniform noise schedule is centered in log-space',
          abs(log_mean - expected_log_mean) < 0.05,
          f'log_mean={log_mean:.3f}, expected~{expected_log_mean:.3f}')

    print()
    if FAILED:
        print(f"  {len(FAILED)} check(s) FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("  All checks passed -- Ranking-NCE energy policy is learnable and usable for ranking/refinement.")


if __name__ == '__main__':
    main()
