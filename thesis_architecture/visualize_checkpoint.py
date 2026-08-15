#!/usr/bin/env python3
"""
visualize_checkpoint.py
=======================
Standalone script: load a saved checkpoint (.pt) and the 775-shape database,
then regenerate holdout visualizations using the current clean white-style.

Usage examples:
  # Particle model (auto-detected):
  python visualize_checkpoint.py --checkpoint checkpoints/cond_particles_crossattn_..._ep0020.pt

  # Ergodic/spectral model:
  python visualize_checkpoint.py --checkpoint checkpoints/flow_matching_ergodic_..._ep0250.pt

  # Override output directory:
  python visualize_checkpoint.py --checkpoint ... --out_dir my_vizzes/

  # Use all val shapes (default: 25):
  python visualize_checkpoint.py --checkpoint ... --n_gen 3 --steps 50
"""

import argparse, os, sys, json, sqlite3
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
from datetime import datetime

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.append(os.path.join(_here, 'ergodic_dataset_generator'))
from shape_library import pdf_on_grid, VALIDATION_SHAPES
from bsplinax.bspline import BsplineBasisClamped
from ergodic_energy_torch import (
    ErgodicEnergy, make_k_grid, target_coeffs_from_grid,
    coverage_distance, path_length, K_DEFAULT,
)
from model_zoo import load_model, generate, describe
from obstacles import CircleObstacle
from flow_matching_runner_particles import sample_particles

_DB_PATH = os.path.join(_here, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')

# ── White-Inferno Colormap ────────────────────────────────────────────────────
_inferno_colors = plt.colormaps['inferno'](np.linspace(0.0, 1.0, 256))
_n_white = 40
for _i in range(_n_white):
    _t = _i / _n_white
    _inferno_colors[_i] = (1 - _t) * np.array([1, 1, 1, 1]) + _t * _inferno_colors[_n_white]
WHITE_INFERNO = _mcolors.LinearSegmentedColormap.from_list('white_inferno', _inferno_colors)


# ── B-Spline helper ───────────────────────────────────────────────────────────
def cp_to_bspline(cps, pts=512, deg=5):
    nxi = cps.shape[0]
    B = np.array(BsplineBasisClamped(
        degree=deg, num_control_points=nxi,
        num_phase_points=pts, compute_derivatives=False).B)
    return B @ cps


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_panel(ax, base_cp, gen_cps, density_grid, particles, title,
               bspline_pts=512, bspline_deg=5, obstacle=None, free_cps=None):
    ax.set_facecolor('white')

    if density_grid is not None:
        d = density_grid.copy()
        if d.max() > 0:
            d /= d.max()
        ax.imshow(d, extent=[0, 1, 0, 1], origin='lower',
                  cmap=WHITE_INFERNO, vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)

    if particles is not None:
        ax.scatter(particles[:, 0], particles[:, 1],
                   c='#444444', s=6, alpha=0.3, zorder=1, edgecolors='none')

    # Above density/particles but below the trajectories, so a path that cuts
    # through the obstacle stays visible instead of being covered up.
    if obstacle is not None:
        obstacle.draw(ax, zorder=1.5)

    # Unguided reference run (same seed), drawn pale underneath the guided one.
    for i, cp in enumerate(free_cps if free_cps is not None else []):
        if len(cp) >= 6:
            ax.plot(*cp_to_bspline(cp, bspline_pts, bspline_deg).T,
                    color='#00C853', lw=1.6, alpha=0.35, ls='--',
                    label='Generated (no guidance)' if i == 0 else '', zorder=2.5)

    if base_cp is not None and len(base_cp) >= 6:
        ax.plot(*cp_to_bspline(base_cp, bspline_pts, bspline_deg).T,
                color='#1565C0', lw=2.5, label='Ground Truth', zorder=2)
        ax.scatter(base_cp[:, 0], base_cp[:, 1],
                   color='#1565C0', s=12, alpha=0.5, zorder=2)

    for i, cp in enumerate(gen_cps):
        alpha = 0.95 if i == 0 else 0.3
        if len(cp) >= 6:
            ax.plot(*cp_to_bspline(cp, bspline_pts, bspline_deg).T,
                    color='#00C853', lw=2.2, alpha=alpha,
                    label='Generated' if i == 0 else '', zorder=3)
        ax.scatter(cp[:, 0], cp[:, 1], color='#00C853',
                   s=8, alpha=max(0.1, alpha * 0.65), zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9, color='#1A1A2E', pad=4)
    ax.set_xlabel('x', fontsize=7, color='#555')
    ax.set_ylabel('y', fontsize=7, color='#555')
    ax.tick_params(labelsize=6, colors='#555')
    for spine in ax.spines.values():
        spine.set_color('#ccc')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')


def save_grid(panels, title, save_path, max_cols=5):
    """panels: list of dicts with keys: base_cp, gen_cps, density_grid, particles, label"""
    n = len(panels)
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 5 * n_rows),
                             facecolor='white', squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight='bold', color='#1A1A2E', y=1.01)

    for i, panel in enumerate(panels):
        ax = axes[i // n_cols][i % n_cols]
        draw_panel(ax, panel['base_cp'], panel['gen_cps'],
                   panel['density_grid'], panel.get('particles'), panel['label'],
                   obstacle=panel.get('obstacle'), free_cps=panel.get('free_cps'))
        if i == 0:
            ax.legend(frameon=True, fontsize=7, loc='upper right',
                      facecolor='white', edgecolor='#ddd', framealpha=0.9)

    for i in range(n, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved → {save_path}")


# ── Load holdout shapes from DB ───────────────────────────────────────────────
def load_holdout_shapes(nxi, grid_res=64):
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT shape_name, density_params, trajectory FROM ergodic_pairs "
        "WHERE split='val' ORDER BY id ASC"
    )
    rows = cur.fetchall()
    conn.close()

    shapes, densities = {}, {}
    for name, dens_json, traj_blob in rows:
        xy = np.frombuffer(traj_blob, dtype=np.float32).reshape(-1, 2)
        idx = np.linspace(0, len(xy) - 1, nxi).astype(int)
        params = json.loads(dens_json)
        d_map, _, _ = pdf_on_grid(params, resolution=grid_res)
        if d_map.max() > 0:
            d_map /= d_map.max()
        shapes[name] = xy[idx]
        densities[name] = d_map

    return shapes, densities


# ── Spectral helper (for ergodic runner checkpoints) ─────────────────────────
def compute_spectral_features(d_map, S, device='cpu'):
    from flow_matching_runner_ergodic import _make_ergodic_k_grid
    K = int(np.ceil(np.sqrt(S)))
    k_idx_np, Lambda_np = _make_ergodic_k_grid(K)
    H, W = d_map.shape
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    XX, YY = np.meshgrid(xs, ys)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    ck = []
    for (kx, ky) in k_idx_np[:S]:
        fk = np.cos(np.pi * kx * XX) * np.cos(np.pi * ky * YY)
        norm = (fk ** 2).sum() * dx * dy
        ck.append((d_map * fk).sum() * dx * dy / max(norm, 1e-9))
    spec = np.array(ck, dtype=np.float32)
    k_idx = k_idx_np[:S].astype(np.int32)
    return spec, k_idx


# ── Scoring and best-of-n selection ──────────────────────────────────────────
def _score_candidates(cps, energy, phi, density_t, basis):
    """Per-candidate metrics for one shape. cps: (n, nxi, 2)"""
    _, terms = energy(cps, phi.expand(cps.shape[0], -1), return_terms=True)
    curves = torch.einsum('ti,bid->btd', basis, cps)
    return dict(
        E_ergodic=terms['ergodic'], E_total=sum(terms.values()),
        coverage=coverage_distance(curves, density_t),
        path_len=path_length(curves),
    )


def render_checkpoint(ckpt_path, out_dir=None, n_gen=5, steps=100,
                      select_best=True, select_by='total', obstacle_mode='both',
                      device=None, grid_res=64, erg_K=K_DEFAULT, solver_T=100,
                      bspline_deg=5, seed=0, max_cols=5, n_particles=None,
                      obstacle_weight=20.0, obstacle_t_start=0.3, quiet=False):
    """Holdout figures for one checkpoint. Returns {variant: path}.

    With `select_best`, the n generated candidates are scored with the solver's
    own objective and only the winner is drawn — the grid then shows what the
    model can actually deliver rather than an average of its spread.

    `obstacle_mode='both'` writes two separate figures, unobstructed and guided,
    so the obstacle's effect is a comparison between images instead of clutter
    inside one.
    """
    device = torch.device(device if device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, kind, meta = load_model(ckpt_path, device)
    if n_particles:
        meta['n_particles'] = n_particles
    nxi = meta['nxi']

    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = out_dir or os.path.join(_here, 'Trajectory_data_generator', 'viz_rerun')
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%Hh%Mmin')

    shapes, densities = load_holdout_shapes(nxi, grid_res=grid_res)
    names = list(shapes.keys())

    basis = torch.tensor(np.array(BsplineBasisClamped(
        degree=bspline_deg, num_control_points=nxi,
        num_phase_points=solver_T, compute_derivatives=False).B),
        dtype=torch.float32, device=device)
    energy = ErgodicEnergy(K=erg_K, basis=basis).to(device)
    k_idx = torch.tensor(make_k_grid(erg_K)[0], dtype=torch.float64)

    variants = {'off': None, 'on': CircleObstacle()}
    if obstacle_mode == 'off':
        variants.pop('on')
    elif obstacle_mode == 'on':
        variants.pop('off')

    written = {}
    for vname, obstacle in variants.items():
        panels, stats = [], {k: [] for k in ('E_ergodic', 'E_total', 'coverage', 'path_len')}
        if not quiet:
            print(f"  [{vname:3s}] generating {len(names)} shapes "
                  f"({n_gen} candidates each)...")

        for i, name in enumerate(names):
            d_map = densities[name]
            dens_t = torch.tensor(d_map, dtype=torch.float32, device=device)
            phi = torch.tensor(
                target_coeffs_from_grid(torch.tensor(d_map, dtype=torch.float64), k_idx)
                .numpy(), dtype=torch.float32, device=device).unsqueeze(0)

            idx_t = torch.tensor([0], dtype=torch.long, device=device)
            parts = sample_particles(dens_t.unsqueeze(0), idx_t,
                                     meta['n_particles'], device, mode='uniform')[0]
            cps, _ = generate(model, kind, parts, n_gen, meta, steps, device,
                              seed + i, obstacle=obstacle,
                              obstacle_weight=obstacle_weight,
                              obstacle_t_start=obstacle_t_start)

            s = _score_candidates(cps, energy, phi, dens_t, basis)
            if select_best:
                key = 'E_total' if select_by == 'total' else 'E_ergodic'
                b = int(torch.argmin(s[key]).item())
                shown = cps[b:b + 1]
                m = {k: v[b].item() for k, v in s.items()}
            else:
                shown = cps
                m = {k: v.mean().item() for k, v in s.items()}
            for k, v in m.items():
                stats[k].append(v)

            panels.append(dict(
                label=f"'{name}'\nE_erg={m['E_ergodic']:.2f}  cov={m['coverage']:.3f}",
                base_cp=shapes[name], gen_cps=shown.cpu().numpy(),
                density_grid=d_map, particles=parts.cpu().numpy(),
                obstacle=obstacle))

        mean = {k: float(np.mean(v)) for k, v in stats.items()}
        sel = ('beste von %d' % n_gen) if select_best else ('%d Stichproben' % n_gen)
        head = f"Holdout — {describe(kind, meta)} — {sel}"
        head += "  |  mit Hindernis" if obstacle else "  |  ohne Hindernis"
        head += (f"\nMittel:  E_ergodic {mean['E_ergodic']:.3f}   "
                 f"E_total {mean['E_total']:.3f}   "
                 f"coverage {mean['coverage']:.4f}   "
                 f"Pfadlänge {mean['path_len']:.2f}")

        path = os.path.join(out_dir, f"viz_{stem}_{vname}obstacle_{stamp}.png")
        save_grid(panels, head, path, max_cols=max_cols)
        written[vname] = path
        if not quiet:
            print(f"         E_ergodic={mean['E_ergodic']:.3f}  "
                  f"E_total={mean['E_total']:.3f}  coverage={mean['coverage']:.4f}")

    return written


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='Holdout visualisation from a checkpoint, best-of-n, '
                    'with and without obstacle.')
    p.add_argument('--checkpoint', required=True, nargs='+',
                   help='One or more .pt checkpoints.')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--n_gen', type=int, default=5,
                   help='Candidates generated per shape before picking the best.')
    p.add_argument('--steps', type=int, default=100, help='ODE steps (flow models).')
    p.add_argument('--no_select_best', action='store_true',
                   help='Draw all candidates instead of only the best one.')
    p.add_argument('--select_by', choices=['total', 'ergodic'], default='total',
                   help="Objective used to pick the winner.")
    p.add_argument('--obstacle_mode', choices=['both', 'off', 'on'], default='both',
                   help='Which figures to write. Default writes both, separately.')
    p.add_argument('--obstacle_weight', type=float, default=20.0)
    p.add_argument('--obstacle_t_start', type=float, default=0.3)
    p.add_argument('--n_particles', type=int, default=None)
    p.add_argument('--grid_res', type=int, default=64)
    p.add_argument('--erg_K', type=int, default=K_DEFAULT)
    p.add_argument('--solver_T', type=int, default=100)
    p.add_argument('--bspline_deg', type=int, default=5)
    p.add_argument('--max_cols', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    for ck in args.checkpoint:
        if not os.path.isfile(ck):
            print(f"[SKIP] not found: {ck}")
            continue
        print(f"\n  {os.path.basename(ck)}")
        render_checkpoint(
            ck, out_dir=args.out_dir, n_gen=args.n_gen, steps=args.steps,
            select_best=not args.no_select_best, select_by=args.select_by,
            obstacle_mode=args.obstacle_mode, device=args.device,
            grid_res=args.grid_res, erg_K=args.erg_K, solver_T=args.solver_T,
            bspline_deg=args.bspline_deg, seed=args.seed, max_cols=args.max_cols,
            n_particles=args.n_particles, obstacle_weight=args.obstacle_weight,
            obstacle_t_start=args.obstacle_t_start)
    print()


if __name__ == '__main__':
    main()
