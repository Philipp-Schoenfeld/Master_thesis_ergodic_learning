r"""
common.py
=========
Shared plumbing for the inference-time constraint experiments in this folder.

Every constraint here follows the same recipe as the obstacle repulsion in
`obstacles.py`: the trained flow-matching model is never retrained and never
sees the constraint. A constraint contributes a gradient dE/d(control points)
which is subtracted from the ODE velocity with a quadratic ramp, and a final
pure-descent polish drives the remaining violation to (near) zero.

Two gradient routes exist, and picking the right one matters:

* **Pointwise** penalties (obstacle, keep-in region, surface attraction)
  decompose over individual curve points, so the chain rule through the linear
  basis map collapses to ``B.T @ grad`` -- that is exactly what
  `obstacles.curve_repulsion_grad` computes, and it needs no autograd.
* **Coupled** penalties (curvature, arc length, tangent alignment) involve
  neighbouring curve points, so that shortcut is invalid. `curve_energy_grad`
  below takes the gradient with autograd through the same basis map instead.

Both routes return shape ``(B, nxi, nd)`` and are interchangeable as the
``force`` argument of `guided_generate`.

Evaluation is deliberately not per-shape-anecdotal: `run_over_shapes` drives
any constraint across the full holdout split, reports the constraint's own
violation *and* the solver's ergodic metrics before/after, and writes both a
panel grid and a CSV. A constraint that satisfies itself by wrecking coverage
is a failed constraint, and only the paired numbers show that.
"""

import csv
import os
import sys

import numpy as np
import torch

# Titles and metric labels carry Greek letters and arrows; the Windows console
# defaults to cp1252 and raises on them, which would kill a run purely for a
# print. Figures are unaffected -- this only widens stdout.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

# ── Path bootstrap (this file is the single place that knows the layout) ──────
_here = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.dirname(_here)                    # thesis_architecture/
ROOT_DIR = os.path.dirname(ARCH_DIR)                 # repo root
for _p in (os.path.join(ROOT_DIR, 'bsplinax-main'), os.path.join(ROOT_DIR, 'src'),
           ARCH_DIR, os.path.join(ARCH_DIR, 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shape_library import get_shape, pdf_on_grid, VALIDATION_SHAPES   # noqa: E402
from model_zoo import load_model                                      # noqa: E402
from flow_matching_runner_particles import sample_particles           # noqa: E402
from obstacles import (basis_torch, bspline_basis_matrix,             # noqa: E402,F401
                       curve_repulsion_grad)
from visualize_checkpoint import WHITE_INFERNO                        # noqa: E402
from ergodic_energy_torch import (ErgodicEnergy, make_k_grid,         # noqa: E402
                                  target_coeffs_from_grid,
                                  coverage_distance, K_DEFAULT)
from bsplinax.bspline import BsplineBasisClamped                      # noqa: E402

DEFAULT_CKPT = os.path.join(
    ARCH_DIR, 'exploration', 'modelle_und_Datenbank',
    'cond_particles_crossattn_flow_matching_particle_ergodic_date_08_28_09h13min_'
    'nxi64_D384_N256_C2_flip0.0_START_FLAT7540_LEN-pd0.1_LINEARFREQ_LR1E4_'
    'ERGLOSS-SINKHORN-w1300-blur0.05-tp2_final.pt')

HOLDOUT_SHAPES = list(VALIDATION_SHAPES)

# Project palette (see the visualization guidelines in CLAUDE.md).
C_GEN = '#00C853'      # generated / constrained trajectory
C_FREE = '#00C853'     # unguided reference, drawn pale + dashed
C_GT = '#1565C0'
C_MARK = '#D81B60'
C_DARK = '#1A1A2E'
C_GREY = '#555'


# ── Model + conditioning ─────────────────────────────────────────────────────
def pick_device(name=None):
    return torch.device(name if name else ('cuda' if torch.cuda.is_available() else 'cpu'))


def load_generator(device, ckpt=DEFAULT_CKPT, verbose=True):
    """Load the trained particle-conditioned flow model. Returns (model, meta)."""
    model, kind, meta = load_model(ckpt, device)
    assert kind == 'flow', f"expected a flow checkpoint, got kind={kind!r}"
    if verbose:
        print(f"Checkpoint: nxi={meta['nxi']} D={meta['D']} "
              f"N={meta['n_particles']} epoch={meta['epoch']}")
    return model, meta


def density_and_particles(shape_name, meta, device, grid_res=64, seed=0):
    """Target density grid (normalised) plus the particle conditioning drawn
    from it -- the exact conditioning path the training runner uses."""
    d_map, _, _ = pdf_on_grid(get_shape(shape_name), resolution=grid_res)
    if d_map.max() > 0:
        d_map = d_map / d_map.max()
    dens_t = torch.tensor(d_map, dtype=torch.float32, device=device).unsqueeze(0)
    idx_t = torch.tensor([0], dtype=torch.long, device=device)
    torch.manual_seed(seed)
    particles = sample_particles(dens_t, idx_t, meta['n_particles'], device, mode='uniform')[0]
    return d_map, particles


# ── Constraint gradients ─────────────────────────────────────────────────────
def curve_energy_grad(cps, energy_fn, B, normalize=True):
    """dE/d(control points) for ANY scalar functional of the dense curve.

    `curve_repulsion_grad` assumes the penalty decomposes point by point.
    Curvature, arc length and tangent alignment all couple neighbouring curve
    points, so the gradient is taken with autograd through the (linear) basis
    map ``curve = B @ cps`` instead.

    `normalize` divides by each control point's total basis weight, the same
    convention `curve_repulsion_grad` uses, so guidance weights stay roughly
    comparable across constraints instead of scaling with `pts`.
    """
    with torch.enable_grad():
        c = cps.detach().requires_grad_(True)
        curve = torch.einsum('pi,bid->bpd', B, c)
        e = energy_fn(curve)
        (g,) = torch.autograd.grad(e, c)
    if normalize:
        g = g / B.sum(0).clamp(min=1e-8)[None, :, None]
    return g


def pointwise_grad(shape_like, B, normalize=True):
    """Build a `force` callable for a penalty that decomposes point by point
    (anything exposing `grad_penalty`, i.e. the obstacles.py convention)."""
    def force(cps):
        return curve_repulsion_grad(cps, shape_like, B, normalize=normalize)
    return force


def energy_force(energy_fn, B, normalize=True):
    """Build a `force` callable from a scalar functional of the dense curve."""
    def force(cps):
        return curve_energy_grad(cps, energy_fn, B, normalize=normalize)
    return force


def _clip(g, max_abs):
    """Element-wise cap on the guidance gradient.

    Not cosmetic: penalties whose gradient has a heavy tail (curvature blows up
    like 1/|c'|^3 near a cusp) hand explicit Euler a step far larger than the
    control points themselves, and the ODE diverges instead of being steered.
    Measured on the letter 'A', the raw curvature gradient peaks around 3e2
    while the control points live at scale ~1.
    """
    return g if max_abs is None else g.clamp(min=-max_abs, max=max_abs)


# ── Inference-time guided generation ─────────────────────────────────────────
@torch.no_grad()
def guided_generate(model, meta, particles, force=None, num_samples=1, steps=100,
                    device='cpu', seed=0, force_weight=20.0, force_t_start=0.3,
                    polish_steps=250, polish_lr=1.0, polish_tol=1e-7,
                    max_force=None):
    """Integrate the flow ODE with an arbitrary inference-time constraint force.

    Mirrors `generate_particle_trajectories`'s obstacle branch exactly, only
    with the constraint abstracted behind `force(cps) -> dE/dcps`:

    * **Ramp** -- at small t the state is still essentially Gaussian noise, so
      constraining it is meaningless and only distorts the flow. The force
      starts at `force_t_start` and grows quadratically to full strength at
      t = 1.
    * **Polish** -- the ramp alone does not guarantee the constraint holds.
      Pure descent steps afterwards drive the residual violation down while
      leaving the shape produced by the flow essentially untouched.

    With `force=None` this is plain unguided generation, which is what the
    experiments use as their reference curve.
    """
    model.eval()
    nxi, nd = meta['nxi'], meta['nd']
    dev = torch.device(device) if isinstance(device, str) else device
    g = torch.Generator(device=dev).manual_seed(seed)

    if particles.ndim == 2:
        particles = particles.unsqueeze(0)
    if particles.shape[0] == 1 and num_samples > 1:
        particles = particles.expand(num_samples, -1, -1).contiguous()
    particles = particles.to(dev)

    x = torch.randn(num_samples, nxi, nd, device=dev, generator=g)
    dt = 1.0 / steps

    mask_batch = torch.cat([
        torch.zeros(num_samples, dtype=torch.bool, device=dev),
        torch.ones(num_samples, dtype=torch.bool, device=dev),
    ], dim=0)
    particle_batch = torch.cat([particles, particles], dim=0)

    for step in range(steps):
        t = torch.full((num_samples,), step * dt, device=dev)
        v_batch, _ = model(torch.cat([x, x], dim=0), torch.cat([t, t], dim=0),
                           particle_batch, cond_drop_mask=mask_batch)
        v_cond, v_null = v_batch.chunk(2, dim=0)
        v = v_null + meta['cfg_weight'] * (v_cond - v_null)

        if force is not None:
            t_now = step * dt
            if t_now >= force_t_start:
                s = (t_now - force_t_start) / max(1.0 - force_t_start, 1e-8)
                v = v - (force_weight * s ** 2) * _clip(force(x), max_force)

        x = x + v * dt

    if force is not None and polish_steps > 0:
        x = polish(x, force, iters=polish_steps, lr=polish_lr, tol=polish_tol,
                   max_force=max_force)
    return x


def polish(cps, force, iters=250, lr=1.0, tol=1e-7, max_force=None):
    """Pure descent on the constraint penalty, the generic counterpart of
    `obstacles.polish_out_of_obstacle`."""
    cps = cps.clone()
    for _ in range(iters):
        g = _clip(force(cps), max_force)
        if g.abs().max() < tol:
            break
        cps = cps - lr * g
    return cps


# ── Curve helpers ────────────────────────────────────────────────────────────
def curve_of(cps, B):
    """(B, nxi, nd) control points -> (B, pts, nd) dense curve, torch."""
    return torch.einsum('pi,bid->bpd', B, cps)


def cp_to_curve_np(cps, pts=512, deg=5):
    """(nxi, nd) numpy control points -> (pts, nd) numpy dense curve."""
    return bspline_basis_matrix(cps.shape[0], pts, deg) @ cps


def tangents(curve, h=None):
    """Central-difference tangents of a dense curve, (B, pts, nd).

    Same finite-difference philosophy as the `MPDLayer`'s kernel_size=3
    convolutions: derivatives come from neighbouring samples rather than from
    an explicit velocity channel in the state.
    """
    pts = curve.shape[1]
    h = h if h is not None else 1.0 / (pts - 1)
    d = torch.zeros_like(curve)
    d[:, 1:-1] = (curve[:, 2:] - curve[:, :-2]) / (2 * h)
    d[:, 0] = (curve[:, 1] - curve[:, 0]) / h
    d[:, -1] = (curve[:, -1] - curve[:, -2]) / h
    return d


def arc_length(curve):
    """(B, pts, nd) -> (B,) polyline length of the dense curve."""
    return (curve[:, 1:] - curve[:, :-1]).norm(dim=-1).sum(dim=-1)


def curvature(curve, h=None, eps=1e-8):
    """Discrete curvature kappa = |c' x c''| / |c'|^3, shape (B, pts-2).

    Padded to 3D internally so one expression covers planar and lifted curves;
    for a planar curve the cross product reduces to its z-component.
    """
    pts = curve.shape[1]
    h = h if h is not None else 1.0 / (pts - 1)
    d1 = (curve[:, 2:] - curve[:, :-2]) / (2 * h)
    d2 = (curve[:, 2:] - 2 * curve[:, 1:-1] + curve[:, :-2]) / (h ** 2)
    if curve.shape[-1] == 2:
        pad = torch.zeros_like(d1[..., :1])
        d1 = torch.cat([d1, pad], dim=-1)
        d2 = torch.cat([d2, pad], dim=-1)
    num = torch.cross(d1, d2, dim=-1).norm(dim=-1)
    return num / d1.norm(dim=-1).clamp(min=eps) ** 3


# ── Solver-side quality metrics ──────────────────────────────────────────────
class ErgodicMetrics:
    """The solver's own objective, used to price what a constraint costs.

    Mirrors `visualize_checkpoint._score_candidates`: the same `ErgodicEnergy`
    with the same K and the same solver-side B-spline basis, so the numbers are
    comparable with the rest of the thesis's evaluations.
    """

    def __init__(self, nxi, device, K=K_DEFAULT, solver_T=100, deg=5):
        self.device = device
        basis_np = np.array(BsplineBasisClamped(
            degree=deg, num_control_points=nxi, num_phase_points=solver_T,
            compute_derivatives=False).B)
        self.basis = torch.tensor(basis_np, dtype=torch.float32, device=device)
        self.energy = ErgodicEnergy(K=K, basis=self.basis).to(device)
        self.k_idx = torch.tensor(make_k_grid(K)[0], dtype=torch.float64)

    def phi_for(self, d_map):
        phi = target_coeffs_from_grid(torch.tensor(d_map, dtype=torch.float64), self.k_idx)
        return torch.tensor(phi.numpy(), dtype=torch.float32,
                            device=self.device).unsqueeze(0)

    def score(self, cps, phi, density_t):
        """(E_ergodic, coverage, path_length) for a single-sample cps."""
        _, terms = self.energy(cps, phi.expand(cps.shape[0], -1), return_terms=True)
        curves = torch.einsum('ti,bid->btd', self.basis, cps)
        return (terms['ergodic'][0].item(),
                coverage_distance(curves, density_t)[0].item(),
                arc_length(curves)[0].item())


# ── Plot helpers (project style: white, pale density, neon-green paths) ──────
def style_axes(ax, title=None, fontsize=9):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=fontsize, color=C_DARK, pad=4)
    ax.tick_params(labelsize=6, colors=C_GREY)
    for spine in ax.spines.values():
        spine.set_color('#ccc')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')


def draw_density(ax, d_map):
    ax.imshow(d_map, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO,
              vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)


def draw_free(ax, curve_np, label='ohne Constraint'):
    ax.plot(curve_np[:, 0], curve_np[:, 1], color=C_FREE, lw=1.6, alpha=0.35,
            ls='--', label=label, zorder=2.5)


def draw_guided(ax, curve_np, label='mit Constraint', color=C_GEN, alpha=0.95):
    ax.plot(curve_np[:, 0], curve_np[:, 1], color=color, lw=2.2, alpha=alpha,
            label=label, zorder=3)


def save(fig, out_dir, name, dpi=140):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved -> {path}")
    return path


def results_dir(script_file):
    return os.path.join(os.path.dirname(os.path.abspath(script_file)), 'results')


# ── Metrics reporting ────────────────────────────────────────────────────────
def write_metrics(rows, out_dir, name):
    """CSV of the per-shape rows. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    if not rows:
        return path
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved -> {path}")
    return path


def summarise(rows, keys=None, label='Mittel ueber Holdout'):
    """Print mean/median for every numeric column and return the means."""
    if not rows:
        return {}
    keys = keys or [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    print(f"\n  {label}  (n={len(rows)})")
    print(f"    {'Metrik':<28} {'Mittel':>12} {'Median':>12} {'max':>12}")
    means = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        means[k] = float(vals.mean())
        print(f"    {k:<28} {vals.mean():>12.4f} {np.median(vals):>12.4f} {vals.max():>12.4f}")
    return means


# ── The driver: one constraint, every holdout shape ──────────────────────────
def run_over_shapes(title, build, metrics_fn, decorate=None, shapes=None, seed=0,
                    steps=100, device=None, ckpt=DEFAULT_CKPT, out_dir='.',
                    tag='run', grid_res=64, max_cols=5, panel_title=None,
                    model_meta=None):
    """Evaluate one constraint across the holdout split.

    Args:
        build: ``build(ctx) -> dict(constraint=..., force=..., gen={...})``
            where ctx carries the per-shape context (free curve included, so a
            constraint may calibrate itself against the unguided result).
        metrics_fn: ``(constraint, free_curve, guided_curve) -> dict``, the
            constraint's own before/after numbers.
        decorate: optional ``(ax, constraint)`` to draw the constraint itself.

    Writes a panel grid and a CSV, prints a summary table, returns the rows.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = device or pick_device()
    shapes = shapes if shapes is not None else HOLDOUT_SHAPES
    model, meta = model_meta if model_meta else load_generator(device, ckpt)
    B = basis_torch(meta['nxi'], 256, 5, device=device)
    erg = ErgodicMetrics(meta['nxi'], device)

    n = len(shapes)
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.9 * n_rows),
                             facecolor='white', squeeze=False)

    rows = []
    print(f"\n{title}  --  {n} Holdout-Formen")
    for i, shape_name in enumerate(shapes):
        d_map, particles = density_and_particles(shape_name, meta, device,
                                                 grid_res=grid_res, seed=seed)
        dens_t = torch.tensor(d_map, dtype=torch.float32, device=device)
        phi = erg.phi_for(d_map)

        free = guided_generate(model, meta, particles, force=None, steps=steps,
                               device=device, seed=seed)
        free_curve = curve_of(free, B)

        ctx = dict(shape=shape_name, free_cps=free, free_curve=free_curve, B=B,
                   meta=meta, device=device, d_map=d_map)
        spec = build(ctx)
        constraint, force = spec['constraint'], spec['force']

        guided = guided_generate(model, meta, particles, force=force, steps=steps,
                                 device=device, seed=seed, **spec.get('gen', {}))
        guided_curve = curve_of(guided, B)

        e_free = erg.score(free, phi, dens_t)
        e_con = erg.score(guided, phi, dens_t)
        row = {'shape': shape_name}
        row.update(metrics_fn(constraint, free_curve, guided_curve))
        row.update({
            'E_erg_frei': e_free[0], 'E_erg_constr': e_con[0],
            'coverage_frei': e_free[1], 'coverage_constr': e_con[1],
            'laenge_frei': e_free[2], 'laenge_constr': e_con[2],
        })
        rows.append(row)

        ax = axes[i // n_cols][i % n_cols]
        draw_density(ax, d_map)
        if decorate:
            decorate(ax, constraint)
        draw_free(ax, cp_to_curve_np(free[0].cpu().numpy()))
        draw_guided(ax, cp_to_curve_np(guided[0].cpu().numpy()))
        head = panel_title(row) if panel_title else f"'{shape_name}'"
        style_axes(ax, head, fontsize=8)
        if i == 0:
            ax.legend(frameon=True, fontsize=6, loc='upper right',
                      facecolor='white', edgecolor='#ddd', framealpha=0.9)
        print(f"  [{i + 1:2d}/{n}] {shape_name:<24} " +
              '  '.join(f"{k}={v:.3f}" for k, v in row.items()
                        if isinstance(v, float) and not k.startswith(('E_erg', 'coverage', 'laenge'))))

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis('off')

    means = summarise(rows)
    fig.suptitle(f"{title}\nHoldout n={n}   "
                 f"E_erg {means.get('E_erg_frei', 0):.2f} → {means.get('E_erg_constr', 0):.2f}   "
                 f"coverage {means.get('coverage_frei', 0):.4f} → {means.get('coverage_constr', 0):.4f}",
                 fontsize=13, color=C_DARK, y=1.005)
    fig.tight_layout()
    save(fig, out_dir, f'{tag}_holdout.png')
    plt.close(fig)
    write_metrics(rows, out_dir, f'{tag}_metrics.csv')
    return rows


__all__ = [
    'DEFAULT_CKPT', 'HOLDOUT_SHAPES', 'WHITE_INFERNO',
    'C_GEN', 'C_FREE', 'C_GT', 'C_MARK', 'C_DARK', 'C_GREY',
    'pick_device', 'load_generator', 'density_and_particles',
    'curve_energy_grad', 'pointwise_grad', 'energy_force',
    'guided_generate', 'polish', 'basis_torch', 'curve_repulsion_grad',
    'curve_of', 'cp_to_curve_np', 'tangents', 'arc_length', 'curvature',
    'ErgodicMetrics', 'style_axes', 'draw_density', 'draw_free', 'draw_guided',
    'save', 'results_dir', 'write_metrics', 'summarise', 'run_over_shapes',
    'np', 'torch',
]
