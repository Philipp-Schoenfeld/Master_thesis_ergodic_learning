r"""
alm_solver_torch.py
====================
Differentiable PyTorch port of `evaluation_full_matrix/rq_experiments/alm_solver.py`
(`AlmBSplineSolver`, JAX), used to train the Lagrange-multiplier head
(`LambdaHead` in `flow_matching_cond_spectral_crossattn.py`) with a
Decision-Focused Learning (DFL) loss instead of plain imitation: the loss
measures how much a *predicted* lambda0/mu0 warm-start actually reduces the
constraint violation / objective of a few unrolled Augmented-Lagrangian (ALM)
outer iterations, rather than how close it is to some ground-truth multiplier
(no such ground truth exists anywhere in this codebase for this head -- see
`alm_solver.py`'s own docstring).

Problem mirrored exactly from `alm_solver.py` (same objective weights,
constraint blocks and Adam inner-solve hyperparameters, so the unconstrained
part of the objective is not a new, untested formulation):

    minimize_xi   f(xi)                         ergodic coverage + smoothness
    subject to    xi[0] = start                  equality   (start pin)
                  0 <= xi <= 1                    affine inequality (workspace)
                  ||xi_i - c|| >= r  for all i    quadratic inequality (obstacle)

Batched over the training minibatch (B) and fully unrolled in plain PyTorch
ops (no `.detach()` in the hot path), so gradients flow from the final
violation/objective all the way back through every inner Adam step and every
outer dual update to the predicted (lambda0, mu0) -- the standard
"differentiable unrolled optimisation" trick (e.g. TOAST's merit-function
training, Flow-Opt's fixed-point residual).

Obstacle constraint is OFF by default (`obstacle_radius=0.0`): the ergodic
training data in this repo (`ergodic_pairs.db` / character outlines) carries
no obstacle annotation, so only the start-pin equality and the workspace box
inequality are active. The obstacle multiplier slots are still allocated in
`LambdaHead`'s output (dead weight, zero gradient) so the head's output
dimension does not have to change again if obstacle-aware training is added
later -- see `n_lambda_for_alm`.
"""

import torch
import torch.nn.functional as F

W_ERGODIC = 600.0
W_SMOOTH = 15.0


# ---------------------------------------------------------------------------
# Head <-> solver dimension bookkeeping
# ---------------------------------------------------------------------------

def n_lambda_for_alm(nxi: int, obstacle_enabled: bool = True) -> int:
    """Total LambdaHead output dimension matching the ALM constraint set:
    2 (equality start-pin) + 4*nxi (box inequality) + nxi (obstacle, always
    allocated even if disabled -- see module docstring)."""
    n_box = 4 * nxi
    n_obstacle = nxi if obstacle_enabled else 0
    return 2 + n_box + n_obstacle


#: Soft bound on the raw LambdaHead output before it ever reaches the ALM
#: solver. Found empirically: an early-training LambdaHead (random/barely
#: trained weights, shared backbone with flow_head) occasionally emits a
#: raw lambda0 large enough that backpropagating the DFL loss through the
#: unrolled solver AND back through the shared backbone produces an
#: inf/nan gradient that poisons every model parameter, not just
#: lambda_head's own (see git history / diagnostics around this file).
#: tanh saturates smoothly with a bounded (<=1) gradient, so it both caps
#: the magnitude the solver ever sees and attenuates (rather than passes
#: through) any pathologically large upstream gradient on the way back --
#: unlike a hard clamp, which has zero gradient outside the bound and would
#: simply stop training the head whenever it saturates.
_LAMBDA0_BOUND = 10.0


def split_lambda0(lambda0: torch.Tensor, nxi: int, obstacle_enabled: bool = True):
    """lambda0: (B, n_lambda) raw network output ->
    lam: (B, 2) equality dual (unconstrained sign)
    mu:  (B, 4*nxi [+ nxi]) inequality dual, softplus'd to stay >= 0."""
    expected = n_lambda_for_alm(nxi, obstacle_enabled)
    assert lambda0.shape[-1] == expected, (
        f"lambda0 has dim {lambda0.shape[-1]}, expected {expected} for "
        f"nxi={nxi} (obstacle_enabled={obstacle_enabled}). Set "
        f"--n_lambda {expected} (or use n_lambda_for_alm(nxi)) when "
        f"--dfl_weight > 0."
    )
    lambda0 = _LAMBDA0_BOUND * torch.tanh(lambda0 / _LAMBDA0_BOUND)
    lam = lambda0[:, :2]
    mu = F.softplus(lambda0[:, 2:])
    return lam, mu


# ---------------------------------------------------------------------------
# Constraint blocks (batched)
# ---------------------------------------------------------------------------

def _h_eq(xi: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """xi: (B, nxi, 2), start: (B, 2) -> (B, 2)."""
    return xi[:, 0, :] - start


def _g_box(xi: torch.Tensor) -> torch.Tensor:
    """xi: (B, nxi, 2) -> (B, 4*nxi): [xi - 1 (upper), -xi (lower)]."""
    B = xi.shape[0]
    return torch.cat([(xi - 1.0).reshape(B, -1), (-xi).reshape(B, -1)], dim=-1)


def _g_obstacle(xi: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """xi: (B, nxi, 2) -> (B, nxi). All-zero (no multiplier mass) if disabled."""
    if radius <= 0:
        return torch.zeros(xi.shape[0], xi.shape[1], device=xi.device, dtype=xi.dtype)
    d2 = torch.sum((xi - center.view(1, 1, 2)) ** 2, dim=-1)
    return radius ** 2 - d2


def _fourier_coeffs(curve: torch.Tensor, k_idx: torch.Tensor) -> torch.Tensor:
    """curve: (B, T, 2), k_idx: (B, S, 2) -> (B, S). Same convention as
    `exp_common.target_coeffs` / the runners' `_ergodic_fourier_basis`:
    cos(pi k1 x) cos(pi k2 y), time-averaged."""
    kx = k_idx[:, None, :, 0]           # (B, 1, S)
    ky = k_idx[:, None, :, 1]           # (B, 1, S)
    cx = curve[:, :, 0:1]               # (B, T, 1)
    cy = curve[:, :, 1:2]               # (B, T, 1)
    Fk = torch.cos(torch.pi * kx * cx) * torch.cos(torch.pi * ky * cy)   # (B, T, S)
    return Fk.mean(dim=1)                                                # (B, S)


def _smoothness(curve: torch.Tensor) -> torch.Tensor:
    """curve: (B, T, 2) -> (B,): sum of squared second differences."""
    accel = curve[:, 2:, :] - 2 * curve[:, 1:-1, :] + curve[:, :-2, :]
    return torch.sum(accel ** 2, dim=(1, 2))


def objective(xi: torch.Tensor, basis: torch.Tensor, phi_k_target: torch.Tensor,
              k_idx: torch.Tensor) -> torch.Tensor:
    """xi: (B, nxi, 2), basis: (T, nxi) -> (B,) per-sample scalar objective."""
    curve = torch.einsum('tc,bcd->btd', basis, xi)   # (B, T, 2)
    c_k = _fourier_coeffs(curve, k_idx)               # (B, S)
    erg = 0.5 * torch.sum((c_k - phi_k_target) ** 2, dim=-1)
    return W_ERGODIC * erg + W_SMOOTH * _smoothness(curve)


def violation(xi: torch.Tensor, start: torch.Tensor, center: torch.Tensor,
              radius: float) -> torch.Tensor:
    """Max constraint violation per sample (0 = fully feasible): (B,)."""
    h = _h_eq(xi, start)
    g = torch.cat([_g_box(xi), _g_obstacle(xi, center, radius)], dim=-1)
    return torch.maximum(h.abs().amax(dim=-1), g.clamp(min=0).amax(dim=-1))


def _lagrangian(xi, lam, mu, rho, basis, phi_k_target, k_idx, start, center, radius):
    f = objective(xi, basis, phi_k_target, k_idx)             # (B,)
    h = _h_eq(xi, start)                                       # (B, 2)
    g = torch.cat([_g_box(xi), _g_obstacle(xi, center, radius)], dim=-1)  # (B, C)
    eq_term = torch.sum(lam * h, dim=-1) + 0.5 * rho * torch.sum(h ** 2, dim=-1)
    ineq_active = torch.clamp(mu + rho * g, min=0.0)
    ineq_term = (torch.sum(ineq_active ** 2, dim=-1) - torch.sum(mu ** 2, dim=-1)) / (2 * rho)
    return f + eq_term + ineq_term                             # (B,)


# ---------------------------------------------------------------------------
# Unrolled differentiable inner (Adam) / outer (dual update) ALM steps
# ---------------------------------------------------------------------------

#: Adam's bias-corrected denominator sqrt(v_hat) + adam_eps is differentiated
#: a SECOND time when the outer decision_focused_loss().backward() call
#: backpropagates through this whole unrolled loop (create_graph=True). That
#: second derivative has a 1/v_hat**1.5 term, which explodes whenever any
#: single control point happens to have a near-zero gradient at some inner
#: step -- found empirically: real training (flow_matching_runner_ergodic.py,
#: S=256, nxi=25) ran clean for ~180 batches, then one batch's second-order
#: backward produced an inf gradient into lambda0, which clip_grad_norm_
#: turned into nan, permanently corrupting every model parameter on the next
#: optimiser step (see git history around alm_solver_torch.py for the
#: diagnostic). 1e-8 (plain Adam's usual default) is far too small once a
#: second derivative of 1/sqrt(.) is involved; 1e-3 costs a little inner-loop
#: precision (this is a cheap warm-start evaluation, not a precise solve) for
#: a much flatter, better-conditioned second derivative.
_INNER_ADAM_EPS = 1e-3
#: Per-sample raw-gradient norm clip before it ever reaches the Adam moment
#: estimates, independent of the eps fix above: bounds the magnitude the
#: second-order backward pass has to differentiate through in the first
#: place, so a single pathological batch can no longer produce an inf/nan
#: gradient regardless of how small v_hat gets.
_INNER_GRAD_CLIP = 50.0


def _inner_minimise(xi, lam, mu, rho, basis, phi_k_target, k_idx, start, center,
                    radius, inner_lr, inner_steps, create_graph=True):
    """`inner_steps` unrolled Adam steps minimising the Lagrangian w.r.t. xi,
    with lam/mu/rho fixed. Same Adam hyperparameters as `alm_solver.py`
    (plain gradient descent diverges at this objective scale -- see that
    file's comment), except `adam_eps` and the gradient clip below, which
    `alm_solver.py` does not need since it is never differentiated a second
    time (see `_INNER_ADAM_EPS` comment).

    `create_graph=True` (default, matches all existing callers/training)
    keeps every inner step's graph alive so `decision_focused_loss().backward()`
    can differentiate through the whole unrolled solve a second time. A
    forward-only evaluation (e.g. measuring outer_iters-to-convergence with no
    backward pass at all) has no use for that and pays for it in GPU memory
    proportional to inner_steps -- at inner_steps=50 (alm_solver.py's own
    default, vs. 8 during DFL training) this OOM'd evaluating 1500 candidates
    at once. Pass `create_graph=False` there: `torch.autograd.grad` then frees
    each step's graph immediately after use."""
    b1, b2 = 0.9, 0.999
    x = xi if xi.requires_grad else xi.clone().requires_grad_(True)
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    for t in range(inner_steps):
        L = _lagrangian(x, lam, mu, rho, basis, phi_k_target, k_idx, start, center, radius).sum()
        g, = torch.autograd.grad(L, x, create_graph=create_graph)
        g_norm = g.flatten(1).norm(dim=-1).clamp(min=1e-12).view(-1, 1, 1)
        g = g * (_INNER_GRAD_CLIP / g_norm).clamp(max=1.0)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** (t + 1))
        v_hat = v / (1 - b2 ** (t + 1))
        x = x - inner_lr * m_hat / (torch.sqrt(v_hat) + _INNER_ADAM_EPS)
    return x


def _outer_step(xi, lam, mu, rho, basis, phi_k_target, k_idx, start, center, radius,
                inner_lr, inner_steps, rho_growth, rho_max, create_graph=True):
    xi = _inner_minimise(xi, lam, mu, rho, basis, phi_k_target, k_idx, start, center,
                         radius, inner_lr, inner_steps, create_graph=create_graph)
    h = _h_eq(xi, start)
    g = torch.cat([_g_box(xi), _g_obstacle(xi, center, radius)], dim=-1)
    lam = lam + rho * h
    mu = torch.clamp(mu + rho * g, min=0.0)
    rho = min(rho * rho_growth, rho_max)
    return xi, lam, mu, rho


def make_cold_start(start: torch.Tensor, nxi: int) -> torch.Tensor:
    """Degenerate 'all control points sitting at the start' cold start --
    the generic, label-independent initial guess the DFL loss warm-starts
    from. start: (B, 2) -> (B, nxi, 2)."""
    return start.unsqueeze(1).expand(-1, nxi, -1).clone()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def decision_focused_loss(
    lambda0: torch.Tensor,
    xi0: torch.Tensor,
    start: torch.Tensor,
    phi_k_target: torch.Tensor,
    k_idx: torch.Tensor,
    basis: torch.Tensor,
    nxi: int,
    obstacle_center=(0.5, 0.5),
    obstacle_radius: float = 0.0,
    n_outer: int = 5,
    inner_steps: int = 8,
    inner_lr: float = 2e-3,
    rho0: float = 10.0,
    rho_growth: float = 1.3,
    rho_max: float = 1e5,
    w_objective: float = 1e-3,
):
    """Decision-focused loss for the LambdaHead: run `n_outer` unrolled ALM
    outer iterations from the cold-started `xi0`, warm-starting the dual
    variables at the network-predicted (lam, mu) = split_lambda0(lambda0)
    instead of the zero-initialisation `alm_solver.py.init_multipliers()`
    uses. The loss is the mean constraint violation across those outer
    iterations (a differentiable proxy for "how many outer iterations the
    real solver would need", the quantity `rq6_alm_lambda0.py` measures as
    `outer_iters`) plus a small weight on the final objective.

    All inputs must already be float32 and on the same device; the caller
    is responsible for stepping outside any bf16 autocast context (the
    inner Adam recursion is numerically sensitive at this objective's scale,
    same caveat as `alm_solver.py`).

    Returns: (loss: 0-d tensor, info: dict of detached scalars for logging).
    """
    device = lambda0.device
    # obstacle_enabled=True unconditionally: the head's output dimension is
    # fixed at construction time (see n_lambda_for_alm) and always allocates
    # the obstacle mu slots, independently of whether `obstacle_radius` (a
    # runtime training hyperparameter) makes that constraint block active.
    lam, mu = split_lambda0(lambda0, nxi, obstacle_enabled=True)
    center = torch.as_tensor(obstacle_center, dtype=xi0.dtype, device=device)

    xi = xi0
    rho = rho0
    violations = []
    for _ in range(n_outer):
        xi, lam, mu, rho = _outer_step(
            xi, lam, mu, rho, basis, phi_k_target, k_idx, start, center,
            obstacle_radius, inner_lr, inner_steps, rho_growth, rho_max,
        )
        violations.append(violation(xi, start, center, obstacle_radius))

    viol_stack = torch.stack(violations, dim=0)   # (n_outer, B)
    mean_violation = viol_stack.mean()
    final_violation = viol_stack[-1].mean()
    final_objective = objective(xi, basis, phi_k_target, k_idx).mean()

    loss = mean_violation + w_objective * final_objective

    info = dict(
        dfl_loss=loss.detach(),
        dfl_mean_violation=mean_violation.detach(),
        dfl_final_violation=final_violation.detach(),
        dfl_final_objective=final_objective.detach(),
    )
    return loss, info
