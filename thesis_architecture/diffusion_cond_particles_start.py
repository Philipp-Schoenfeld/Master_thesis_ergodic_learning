r"""
diffusion_cond_particles_start.py
=================================
Diffusion training objective and sampler for the start-conditioned particle
U-Net of `flow_matching_cond_particles_start.py` -- an opt-in alternative to
conditional flow matching (runner: `--objective diffusion`).

The network is untouched: it still maps (x_t, t, particles, start) to a tensor
of shape (B, nxi, nd). Only what that output means (loss) and how it is
integrated (sampler) differ, so U-Net, particle cross-attention, FiLM start
conditioning, CFG dropout, inpainting and ErgLoss are shared with the CFM runs
and a comparison isolates the training objective.

Forward process
---------------
Variance preserving, cosine schedule, continuous time (Nichol & Dhariwal 2021,
"Improved DDPM"; Kingma et al. 2021, "Variational Diffusion Models"). The flow
time convention of this repo is kept -- t = 0 is noise, t = 1 is data -- so the
time embedding, the ErgLoss t-ramp and the inpainting see the same t as in CFM:

    x_t = alpha(t) x1 + sigma(t) eps,    alpha = sin(pi/2 t),  sigma = cos(pi/2 t)

In DDPM notation the noise level is s = 1 - t and alpha_bar(s) = cos^2(pi/2 s),
i.e. the cosine schedule without its offset (not needed in continuous time).
The difference to CFM is the path (variance preserving instead of the straight
line (1 - t) x0 + t x1) and the sampler (stochastic DDPM / DDIM steps instead
of the Euler ODE).

Prediction target (`--diff_pred`, stored in the model as a buffer)
------------------------------------------------------------------
    v   : v = alpha eps - sigma x1   (Salimans & Ho 2022)   -- default
    eps : eps                          (Ho et al. 2020; MPD, Carvalho et al. 2024)
    x0  : x1 itself
The loss is the plain MSE on that target ("simple" loss of Ho et al.). The
endpoint estimate the ErgLoss needs is exact algebra for v and x0; for eps it
divides by alpha, which vanishes at t -> 0 -- alpha is clamped there, and the
t^2 ramp of the ErgLoss gives those times almost no weight anyway. With ErgLoss
v (or x0) is the well-conditioned choice.

Sampler
-------
DDIM update (Song et al. 2021) on a uniform t grid with `eta`: eta = 1 is the
DDPM ancestral step (posterior variance), eta = 0 deterministic DDIM. CFG is
applied to the raw network output, which for every parametrisation is
equivalent to CFG on eps (eps_hat is linear in the output at fixed x_t).
"""

import math
import torch
import torch.nn as nn

PRED_CODES = {'v': 1, 'eps': 2, 'x0': 3}
PRED_NAMES = {c: n for n, c in PRED_CODES.items()}

# Lower bound for alpha when an eps prediction is turned into an endpoint.
_ALPHA_MIN = 1e-2


def alpha_sigma(t: torch.Tensor):
    """Cosine VP schedule in flow time (t = 0 noise, t = 1 data)."""
    a = 0.5 * math.pi * t
    return torch.sin(a), torch.cos(a)


def _bcast(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return x.view(-1, *([1] * (like.ndim - 1)))


def target_from(pred: str, x1, eps, alpha, sigma):
    if pred == 'v':
        return alpha * eps - sigma * x1
    if pred == 'eps':
        return eps
    return x1


def x1_eps_from_output(pred: str, out, xt, alpha, sigma):
    """(x1_hat, eps_hat) from the network output; alpha, sigma broadcast to xt."""
    if pred == 'v':
        return alpha * xt - sigma * out, sigma * xt + alpha * out
    if pred == 'eps':
        return (xt - sigma * out) / alpha.clamp(min=_ALPHA_MIN), out
    return out, (xt - alpha * out) / sigma.clamp(min=_ALPHA_MIN)


def compute_particle_diffusion_loss(
    model: nn.Module,
    x1_batch: torch.Tensor,
    particle_batch: torch.Tensor,
    p_drop: float = 0.0,
    ergodic=None,
) -> tuple:
    """Diffusion counterpart of `compute_particle_cfm_loss` (same signature, same
    start conditioning, CFG dropout, inpainting and ErgLoss handling).

    Returns (total_loss, components); the denoising term is logged as 'diff'.
    """
    pred = model.diffusion_pred
    assert pred is not None, "model was not built with diffusion_pred"
    B = x1_batch.shape[0]
    device = x1_batch.device
    eps   = torch.randn_like(x1_batch)
    t     = torch.rand(B, device=device)
    alpha, sigma = (_bcast(s, x1_batch) for s in alpha_sigma(t))
    xt    = alpha * x1_batch + sigma * eps
    tgt   = target_from(pred, x1_batch, eps, alpha, sigma)

    cond_drop_mask = (torch.rand(B, device=device) < p_drop)
    # Start point = first point of the *augmented* target, as in the CFM loss.
    start = x1_batch[:, 0, :].detach()
    inpaint = getattr(model, 'start_inpaint', False)
    if inpaint:
        xt = xt.clone()
        xt[:, 0, :] = start
    out, _ = model(xt, t, particle_batch, cond_drop_mask=cond_drop_mask, start=start)

    # With inpainting control point 0 is given, not generated: no target for it.
    loss_diff = (torch.mean((out[:, 1:] - tgt[:, 1:]) ** 2) if inpaint
                 else torch.mean((out - tgt) ** 2))
    total = loss_diff
    parts = {'diff': loss_diff.detach()}

    if ergodic is not None and ergodic.weight > 0.0:
        x1_hat, _ = x1_eps_from_output(pred, out, xt, alpha, sigma)
        if inpaint:
            x1_hat = torch.cat([start[:, None, :].to(x1_hat.dtype), x1_hat[:, 1:]], dim=1)
        loss_erg = ergodic(x1_hat, particle_batch, t)
        total = total + ergodic.weight * loss_erg
        parts['erg'] = loss_erg.detach()

    return total, parts


@torch.no_grad()
def generate_particle_trajectories_diffusion(
    model: nn.Module,
    particles: torch.Tensor,
    num_samples: int = 1,
    nxi: int = 20,
    nd: int = 2,
    steps: int = 100,
    device: str = 'cpu',
    cfg_weight: float = 2.0,
    generator: torch.Generator = None,
    start: torch.Tensor = None,
    eta: float = 1.0,
    obstacle=None,
    guide=None,
    guide_weight: float = 0.0,
    **_ignored,
) -> tuple:
    """DDPM (eta = 1) / DDIM (eta = 0) sampling with CFG; same inputs and outputs as
    `generate_particle_trajectories`, which dispatches here for diffusion models.

    Obstacle repulsion and belief guidance are written for the CFM velocity and
    are not ported; asking for them raises instead of silently ignoring them.
    """
    if obstacle is not None or (guide is not None and guide_weight > 0.0):
        raise NotImplementedError(
            "obstacle / guide sampling is only implemented for CFM checkpoints")
    pred = model.diffusion_pred
    model.eval()
    if particles.ndim == 2:
        particles = particles.unsqueeze(0)
    if particles.shape[0] == 1 and num_samples > 1:
        particles = particles.expand(num_samples, -1, -1).contiguous()
    particles = particles.to(device)

    x = torch.randn(num_samples, nxi, nd, device=device, generator=generator)

    start_b = None
    if start is not None:
        start_b = start.to(device).reshape(-1, nd)
        if start_b.shape[0] == 1 and num_samples > 1:
            start_b = start_b.expand(num_samples, -1).contiguous()
    inpaint = start_b is not None and getattr(model, 'start_inpaint', False)

    mask_batch = torch.cat([
        torch.zeros(num_samples, dtype=torch.bool, device=device),
        torch.ones(num_samples,  dtype=torch.bool, device=device),
    ], dim=0)
    particle_batch = torch.cat([particles, particles], dim=0)
    start_batch = None if start_b is None else torch.cat([start_b, start_b], dim=0)

    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    for i in range(steps):
        if inpaint:
            x[:, 0, :] = start_b.to(x.dtype)
        t_now, t_next = ts[i], ts[i + 1]
        t_batch = t_now.expand(2 * num_samples)
        out_batch, _ = model(torch.cat([x, x], dim=0), t_batch, particle_batch,
                             cond_drop_mask=mask_batch, start=start_batch)
        out_cond, out_null = out_batch.float().chunk(2, dim=0)
        out = out_null + cfg_weight * (out_cond - out_null)

        a, s = alpha_sigma(t_now)
        a_n, s_n = alpha_sigma(t_next)
        x1_hat, eps_hat = x1_eps_from_output(pred, out, x, a, s)
        # DDIM with eta: sigma_eta^2 = eta^2 (s_n / s)^2 (1 - a^2 / a_n^2); zero at t_next = 1.
        var = (eta ** 2) * (s_n / s.clamp(min=1e-12)) ** 2 * (1.0 - (a / a_n) ** 2)
        var = var.clamp(min=0.0)
        dir_coef = (s_n ** 2 - var).clamp(min=0.0).sqrt()
        x = a_n * x1_hat + dir_coef * eps_hat
        if eta > 0.0 and i < steps - 1:
            noise = torch.randn(x.shape, device=x.device, generator=generator)
            x = x + var.sqrt() * noise

    if start_b is not None:
        # Same as the CFM sampler: the first control point *is* the start point.
        x = x.clone()
        x[:, 0, :] = start_b

    return x, None
