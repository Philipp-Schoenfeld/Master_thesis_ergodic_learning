#!/usr/bin/env python3
"""
test_ias_logo_pipeline.py
==========================
Testet den SVGD-Solver aus der Ergodic-Dataset-Generierung auf dem IAS-Logo
(TU Darmstadt Fachgruppe) als Zieldichte, mit drei verschiedenen
Initialisierungen:

  1. linear    — gerade Linie unten links -> oben links
  2. heuristic — dieselbe Greedy-TSP + Lissajous-Serpentinen-Heuristik, die
                 ergodic_dataset_generator/ergodic_solver.py auch beim Bau von
                 ergodic_dataset_775.db / ergodic_dataset_start.db verwendet
  3. model     — Warmstart aus dem trainierten CFM+ErgLoss-Checkpoint
                 (Startpunkt-konditioniertes Partikel-Netz)

Jede Initialisierung wird mit 500 und 1000 SVGD-Iterationen optimiert (6 Läufe
insgesamt), alle mit denselben sonstigen Solver-Parametern wie in
generate_dataset.py (dt=0.05, tsteps=200, step_size=0.01, h=0.01,
score_scale=1.0).

Zieldichte — Flächen und Linien mit Gauß-Abfall zum Rand
---------------------------------------------------------
Die erste Version dieses Skripts hat das Logo als Summe vieler kleiner
Punkt-Gaußglocken dargestellt (`shape_rasterizer.points_to_gmm`) — dieselbe
Konvention wie die Buchstaben-Formen der Trainingsdatenbank. Für ein Logo mit
großen zusammenhängenden Flächen (der Balken des „I", der Roboterkörper, der
Korpus des „S") ist das nicht die richtige Beschreibung: das Innere der Fläche
ist dabei nie wirklich flach, sondern eine Überlagerung einzelner Buckel, die
im Rechendichte-Bild sichtbar spricht (siehe `results_ias_logo_test/` aus der
ersten Version).

Diese Version baut die Dichte stattdessen direkt aus der Randform: eine
Euklidische Distanztransformation der binären Logo-Maske liefert für jeden
Bildpunkt den Abstand zum nächsten Maskenpixel. Daraus wird eine Dichte

    p(x) = 1                                    innerhalb der Maske
    p(x) = exp(-0.5 * (d(x) / sigma_edge)^2)     außerhalb, d = Abstand zum Rand

gebaut — exakt eine „dichte Fläche mit Gauß-Abfall zum Rand", flach im Inneren,
ohne die Sprünge einer harten Indikatorfunktion. Dieselbe Formel beschreibt
ebenso dünne Linien (die Kernbreite ist dann nur die Strichbreite der Maske,
keine gesonderte Fallunterscheidung nötig). Die Distanztransformation selbst
ist nicht in JAX differenzierbar; das Skript wertet sie einmalig auf dem
Pixelgitter aus und liest sie danach über eine bilineare, in x und y
differenzierbare Interpolation aus — genug für `jax.grad`, denn der Score
braucht nur die erste Ableitung.

Ergodische Metriken
--------------------
Jede der sechs optimierten Bahnen (und zur Einordnung auch die drei rohen
Initialisierungen) wird durch `ergodic_energy_torch.ErgodicEnergy` gemessen —
demselben Modul, mit dem `evaluate_models.py` die trainierten Netze gegen den
SVGD-Löser vergleicht (E_ergodic, E_smooth, E_boundary, E_total, sowie die
basisfreie `coverage_distance`). Da der Löser bereits dichte T-Punkt-Bahnen
liefert (keine B-Spline-Kontrollpunkte), wird `ErgodicEnergy(basis=None)`
direkt auf den Bahnpunkten ausgewertet — laut Modul-Docstring der
löser-äquivalente Modus.

Reine CPU/lokale Rechenlast: ein voller Durchlauf (Logo-Rasterung,
Distanztransformation, ein Modell-Forward-Pass, sechs SVGD-Läufe, alle
Metriken) dauert auf einem Laptop ohne GPU typischerweise 1-2 Minuten — kein
Cluster-Job nötig.

Aufruf (siehe --help für alle Optionen):
    python3 test_ias_logo_pipeline.py
    python3 test_ias_logo_pipeline.py --logo /pfad/zu/anderem_logo.png \\
        --checkpoint checkpoints/anderer_lauf_final.pt --iters 500 1000 2000
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
if 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from scipy.ndimage import distance_transform_edt

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_gen_dir = os.path.join(_here, 'ergodic_dataset_generator')
if _gen_dir not in sys.path:
    sys.path.insert(0, _gen_dir)

from shape_rasterizer import points_to_gmm          # noqa: E402
from ergodic_solver import run_ergodic_coverage       # noqa: E402

import torch                                           # noqa: E402
from ergodic_energy_torch import (                     # noqa: E402
    ErgodicEnergy, make_k_grid, K_DEFAULT,
    coverage_distance, path_length,
)

_DEFAULT_CHECKPOINT = os.path.join(
    _here, 'checkpoints',
    'cond_particles_crossattn_flow_matching_particle_ergodic_date_08_26_04h30min_'
    'nxi25_D384_N256_C75_flip0.0_START_FLAT400_L500_ERGLOSS-SINKHORN-w1300-blur0.05-tp2_ep0500.pt'
)

# ── Shared white-inferno colormap + trajectory palette (siehe CLAUDE.md) ──────
_inferno_colors = plt.colormaps['inferno'](np.linspace(0.0, 1.0, 256))
_n_white = 40
for _i in range(_n_white):
    _t = _i / _n_white
    _inferno_colors[_i] = (1 - _t) * np.array([1, 1, 1, 1]) + _t * _inferno_colors[_n_white]
WHITE_INFERNO = mcolors.LinearSegmentedColormap.from_list('white_inferno', _inferno_colors)

INIT_COLOR = '#a500ff'    # gleiche Farbe wie in generate_dataset.py fuer Initialisierungen
GT_COLOR = '#1565C0'
GEN_COLOR = '#00C853'
PARTICLE_COLOR = '#444444'

INIT_ORDER = ['linear', 'heuristic', 'model']
INIT_LABELS = {
    'linear': 'Linear (unten links -> oben links)',
    'heuristic': 'Heuristik (Greedy-TSP + Lissajous)',
    'model': 'CFM+ErgLoss Warmstart (gelernt)',
}


# ===========================================================================
# 1. IAS-Logo -> Zieldichte (Fläche + Gauß-Abfall zum Rand)
# ===========================================================================

def _bilinear_sample(grid, gx, gy):
    """Differenzierbare bilineare Interpolation von `grid` an (gx, gy)
    (Pixelkoordinaten, gebroadcastet). Liefert eine in gx/gy differenzierbare
    Ausgabe, genug fuer `jax.grad` auf dem Score."""
    Hh, Ww = grid.shape
    gx = jnp.clip(gx, 0.0, Ww - 1.0 - 1e-6)
    gy = jnp.clip(gy, 0.0, Hh - 1.0 - 1e-6)
    x0 = jnp.floor(gx).astype(jnp.int32)
    y0 = jnp.floor(gy).astype(jnp.int32)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = gx - x0, gy - y0
    v00, v01 = grid[y0, x0], grid[y0, x1]
    v10, v11 = grid[y1, x0], grid[y1, x1]
    return (v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy)
            + v10 * (1 - wx) * wy + v11 * wx * wy)


class IASDensityField:
    """Zieldichte aus einer Logo-Rastergrafik: flach (=1) innerhalb der
    Maske, Gauß-Abfall exp(-0.5*(d/sigma_edge)^2) im Abstand d zum Rand
    außerhalb — dieselbe Formel für dicke Flächen wie für dünne Linien, es
    gibt nur die eine Kernbreite `sigma_edge`.

    `pdf_fn`/`score_fn` sind JAX-Funktionen auf dem üblichen 4er-Zustand
    (x, y, vx, vy) für den SVGD-Löser; `grid(resolution)` liefert dieselbe
    Dichte als NumPy-Array für Visualisierung, Partikel-Sampling und die
    ergodische Zielkoeffizienten-Berechnung.
    """

    def __init__(self, logo_path, edge_sigma=0.025, fill_frac=0.8):
        img = Image.open(logo_path).convert('L')
        arr = np.array(img)
        mask = arr < 128
        if mask.mean() > 0.5:
            mask = ~mask   # helles Logo auf dunklem Grund -> Maske invertieren
        ys, xs = np.where(mask)
        if len(xs) < 10:
            raise ValueError(f"Zu wenige dunkle Pixel in {logo_path} gefunden.")

        x0b, x1b = xs.min(), xs.max()
        y0b, y1b = ys.min(), ys.max()
        bw, bh = max(x1b - x0b, 1), max(y1b - y0b, 1)
        self.scale = fill_frac / max(bw, bh)
        self.cx, self.cy = (x0b + x1b) / 2.0, (y0b + y1b) / 2.0
        self.edge_sigma = edge_sigma
        self.mask = mask

        # Unsignierter Pixelabstand von außerhalb der Maske zum naechsten
        # Maskenpixel; 0 ueberall innerhalb -> dort ist die Dichte exakt 1.
        dist_out_px = distance_transform_edt(~mask).astype(np.float32)
        self._dist_np = dist_out_px
        self._dist_j = jnp.array(dist_out_px)

        self.pdf_fn = jax.jit(self._pdf)
        self.score_fn = jax.jit(jax.grad(lambda x: jnp.log(self._pdf(x))))

    def _dist_norm_j(self, x, y):
        col = (x - 0.5) / self.scale + self.cx
        row = -(y - 0.5) / self.scale + self.cy
        return _bilinear_sample(self._dist_j, col, row) * self.scale

    def _pdf(self, x4):
        d = self._dist_norm_j(x4[0], x4[1])
        return jnp.exp(-0.5 * (d / self.edge_sigma) ** 2) + 1e-10

    def grid(self, resolution=128):
        """Dieselbe Dichte als (resolution, resolution) NumPy-Array auf
        [0,1]^2, ausgewertet ueber `scipy.ndimage.map_coordinates` (auch
        bilinear, aber schneller als der JAX-Pfad fuer ein ganzes Gitter)."""
        from scipy.ndimage import map_coordinates
        xs_ = np.linspace(0.0, 1.0, resolution)
        ys_ = np.linspace(0.0, 1.0, resolution)
        gx, gy = np.meshgrid(xs_, ys_)
        col = (gx - 0.5) / self.scale + self.cx
        row = -(gy - 0.5) / self.scale + self.cy
        d_px = map_coordinates(self._dist_np, [row.ravel(), col.ravel()],
                               order=1, mode='nearest')
        d_norm = d_px.reshape(resolution, resolution) * self.scale
        return np.exp(-0.5 * (d_norm / self.edge_sigma) ** 2)

    def sample_route_points(self, n_points=800, sigma=0.02, seed=0):
        """Punktwolke + GMM-`shape_def`, ausschließlich für die Greedy-TSP +
        Lissajous-Heuristik (`ergodic_solver._generate_initial_trajectory`)
        gebraucht, die eine Liste von Komponenten zum Abfahren erwartet —
        NICHT für die eigentliche Solver-Dichte, die kommt aus `pdf_fn`."""
        ys, xs = np.where(self.mask)
        px_all = (xs - self.cx) * self.scale + 0.5
        py_all = -(ys - self.cy) * self.scale + 0.5
        rng = np.random.default_rng(seed)
        n = min(n_points, len(px_all))
        idx = rng.choice(len(px_all), size=n, replace=False)
        pts = np.column_stack([px_all[idx], py_all[idx]])
        pts += rng.normal(0, 0.002, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        return points_to_gmm(pts, sigma=sigma), pts


# ===========================================================================
# 2. Initialisierungen
# ===========================================================================

def build_linear_init(x0, x1, tsteps):
    """Gerade Linie von x0 nach x1, tsteps+1 Punkte."""
    return np.linspace(np.array(x0), np.array(x1), tsteps + 1)


def build_model_init(checkpoint_path, d_map, x0, tsteps, device='cpu', seed=0):
    """Warmstart-Trajektorie aus dem trainierten Startpunkt-konditionierten
    CFM+ErgLoss-Checkpoint (flow_matching_cond_particles_start.py)."""
    import torch
    from flow_matching_cond_particles_start import (
        ParticleCrossAttnFlowNetwork, generate_particle_trajectories,
    )
    from flow_matching_runner_start import sample_particles, cp_to_bspline

    dev = torch.device(device)
    # Ohne expliziten Seed *vor* dem ersten Zufallsaufruf haengt PyTorchs
    # globaler RNG-Zustand vom OS-Entropie-Seed beim Prozessstart ab, nicht von
    # `seed` — Partikelstichprobe und Warmstart waeren dann zwischen zwei
    # Laeufen mit demselben `--seed` nicht reproduzierbar.
    torch.manual_seed(seed)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model = ParticleCrossAttnFlowNetwork(
        nxi=ckpt['nxi'], nd=ckpt['nd'], D=ckpt['D']).to(dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    grid_t = torch.tensor(d_map, dtype=torch.float32, device=dev).unsqueeze(0)
    idx_t = torch.tensor([0], dtype=torch.long, device=dev)
    particles = sample_particles(grid_t, idx_t, ckpt['n_particles'], device=dev,
                                 mode=ckpt.get('sample_mode', 'uniform'))[0]

    start_t = torch.tensor([list(x0)], dtype=torch.float32, device=dev)
    cps, _ = generate_particle_trajectories(
        model, particles, num_samples=1, nxi=ckpt['nxi'], nd=ckpt['nd'],
        steps=100, device=str(dev), cfg_weight=ckpt.get('cfg_weight', 2.0),
        start=start_t,
    )
    cps_np = cps[0].cpu().numpy()
    curve = cp_to_bspline(cps_np, pts=tsteps + 1, deg=5)
    return curve, cps_np, particles.cpu().numpy()


# ===========================================================================
# 3. Ergodische Metriken
# ===========================================================================

def compute_metrics(traj_np, density_map_np, k_idx, Lambda, phi_k, energy):
    """E_ergodic/E_smooth/E_boundary/E_total/coverage/path_len für eine
    einzelne dichte (T, 2)-Bahn, mit demselben Modul wie evaluate_models.py.
    `energy` ist ein `ErgodicEnergy(basis=None)` — die Bahn wird also direkt
    als Punktfolge gewertet, kein Umweg über B-Spline-Kontrollpunkte."""
    X = torch.tensor(traj_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, terms = energy(X, phi_k, return_terms=True)
        dens_t = torch.tensor(density_map_np, dtype=torch.float32)
        cov = coverage_distance(X, dens_t)
        plen = path_length(X)
    return dict(
        E_ergodic=terms['ergodic'].item(), E_smooth=terms['smooth'].item(),
        E_boundary=terms['boundary'].item(),
        E_total=sum(v.item() for v in terms.values()),
        coverage=cov.item(), path_len=plen.item(),
    )


# ===========================================================================
# 4. Solver-Läufe
# ===========================================================================

def run_all(density_field, x0, x1_linear, checkpoint_path, iters_list,
           dt=0.05, tsteps=200, step_size=0.01, h=0.01, score_scale=1.0,
           grid_res=128, erg_K=K_DEFAULT, device='cpu', seed=0, verbose=True):
    shape_def, route_pts = density_field.sample_route_points(seed=seed)
    d_map = density_field.grid(resolution=grid_res)

    k_idx_np, Lambda_np = make_k_grid(erg_K)
    k_idx = torch.from_numpy(k_idx_np).float()
    Lambda = torch.from_numpy(Lambda_np).float()
    from ergodic_energy_torch import target_coeffs_from_grid
    phi_k = target_coeffs_from_grid(torch.tensor(d_map, dtype=torch.float32), k_idx)
    energy = ErgodicEnergy(K=erg_K, basis=None)

    custom_trajs = {}
    extra = {}
    custom_trajs['linear'] = build_linear_init(x0, x1_linear, tsteps)
    custom_trajs['heuristic'] = None  # nutzt die interne Heuristik ueber shape_def
    print("  [model] generiere Warmstart aus Checkpoint ...")
    model_curve, model_cps, model_particles = build_model_init(
        checkpoint_path, d_map, x0, tsteps, device=device, seed=seed)
    custom_trajs['model'] = model_curve
    extra['model_particles'] = model_particles

    results = {}
    for name in INIT_ORDER:
        for n_iter in iters_list:
            label = f"{name}_iters{n_iter}"
            print(f"  [{label}] SVGD laeuft ({n_iter} Iterationen) ...")
            t0 = time.time()
            traj, init_traj = run_ergodic_coverage(
                density_field.score_fn, x0=x0, shape_def=shape_def,
                custom_p_traj=custom_trajs[name],
                dt=dt, tsteps=tsteps, num_iters=n_iter,
                step_size=step_size, h=h, score_scale=score_scale,
                verbose=verbose,
            )
            dt_wall = time.time() - t0
            m_final = compute_metrics(traj, d_map, k_idx, Lambda, phi_k, energy)
            m_init = compute_metrics(init_traj, d_map, k_idx, Lambda, phi_k, energy)
            print(f"  [{label}] fertig in {dt_wall:.1f}s  "
                 f"E_ergodic={m_final['E_ergodic']:.3f} (init: {m_init['E_ergodic']:.3f})  "
                 f"path_len={m_final['path_len']:.3f}")
            results[label] = dict(
                init_name=name, n_iter=n_iter, traj=traj, init_traj=init_traj,
                wall_time=dt_wall, metrics=m_final, init_metrics=m_init,
                path_len=m_final['path_len'],
            )

    return results, d_map, extra


# ===========================================================================
# 4. Visualisierung
# ===========================================================================

def _draw_panel(ax, d_map, init_traj, final_traj, x0, title, particles=None):
    ax.set_facecolor('white')
    ax.imshow(d_map, extent=[0, 1, 0, 1], origin='lower',
              cmap=WHITE_INFERNO, vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)
    if particles is not None:
        ax.scatter(particles[:, 0], particles[:, 1], c=PARTICLE_COLOR, s=6,
                  alpha=0.3, zorder=1, edgecolors='none')
    ax.plot(init_traj[:, 0], init_traj[:, 1], color=INIT_COLOR, lw=1.6,
           alpha=0.6, linestyle='--', label='Initialisierung', zorder=2)
    ax.plot(final_traj[:, 0], final_traj[:, 1], color=GEN_COLOR, lw=2.2,
           alpha=0.95, label='SVGD-Ergebnis', zorder=3)
    ax.scatter(*x0, color=GT_COLOR, s=30, marker='*', zorder=4, label='Start x0')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9, color='#1A1A2E', pad=4)
    ax.tick_params(labelsize=6, colors='#555')
    for spine in ax.spines.values():
        spine.set_color('#ccc')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')


def save_individual_figures(results, d_map, x0, out_dir):
    for label, r in results.items():
        fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor='white')
        title = (f"{INIT_LABELS[r['init_name']]}\n"
                f"{r['n_iter']} it, {r['wall_time']:.1f}s, L={r['path_len']:.2f}, "
                f"E_erg={r['metrics']['E_ergodic']:.3f} (init {r['init_metrics']['E_ergodic']:.3f})")
        _draw_panel(ax, d_map, r['init_traj'], r['traj'], x0, title)
        ax.legend(frameon=True, fontsize=7, loc='upper right',
                 facecolor='white', edgecolor='#ddd', framealpha=0.9)
        path = os.path.join(out_dir, f"result_{label}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Gespeichert -> {path}")


def save_comparison_grid(results, d_map, x0, iters_list, out_dir):
    n_rows, n_cols = len(INIT_ORDER), len(iters_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.2 * n_rows),
                             facecolor='white', squeeze=False)
    fig.suptitle('SVGD-Solver auf dem IAS-Logo — Initialisierungsvergleich',
                fontsize=14, fontweight='bold', color='#1A1A2E', y=1.01)
    for i, name in enumerate(INIT_ORDER):
        for j, n_iter in enumerate(iters_list):
            label = f"{name}_iters{n_iter}"
            r = results[label]
            ax = axes[i][j]
            title = (f"{INIT_LABELS[name]}\n{n_iter} it, {r['wall_time']:.1f}s, "
                    f"L={r['path_len']:.2f}, E_erg={r['metrics']['E_ergodic']:.3f}")
            _draw_panel(ax, d_map, r['init_traj'], r['traj'], x0, title)
            if i == 0 and j == 0:
                ax.legend(frameon=True, fontsize=6, loc='upper right',
                         facecolor='white', edgecolor='#ddd', framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Gespeichert -> {path}")


def save_init_overview(results, d_map, x0, iters_list, out_dir):
    """Reine Initialisierungen (vor der Optimierung) nebeneinander."""
    n_iter0 = iters_list[0]
    fig, axes = plt.subplots(1, len(INIT_ORDER), figsize=(5.2 * len(INIT_ORDER), 5.2),
                             facecolor='white', squeeze=False)
    for i, name in enumerate(INIT_ORDER):
        r = results[f"{name}_iters{n_iter0}"]
        ax = axes[0][i]
        ax.set_facecolor('white')
        ax.imshow(d_map, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO,
                 vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)
        ax.plot(r['init_traj'][:, 0], r['init_traj'][:, 1], color=INIT_COLOR,
               lw=2.0, alpha=0.9, zorder=2)
        ax.scatter(*x0, color=GT_COLOR, s=30, marker='*', zorder=3)
        ax.set_title(INIT_LABELS[name], fontsize=9, color='#1A1A2E')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.tick_params(labelsize=6, colors='#555')
        for spine in ax.spines.values():
            spine.set_color('#ccc')
        ax.grid(True, alpha=0.2, lw=0.4, color='gray')
    fig.suptitle('Rohe Initialisierungen vor der SVGD-Optimierung',
                fontsize=13, fontweight='bold', color='#1A1A2E', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'init_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Gespeichert -> {path}")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--logo', type=str,
                   default='/home/philipp/Documents/Uni/Master_thesis/iasLogo.jpg',
                   help='Pfad zum IAS-Logo-Bild.')
    p.add_argument('--checkpoint', type=str, default=_DEFAULT_CHECKPOINT,
                   help='Trainierter CFM+ErgLoss-Checkpoint fuer den Modell-Warmstart.')
    p.add_argument('--out_dir', type=str, default=os.path.join(_here, 'results_ias_logo_test'))
    p.add_argument('--iters', type=int, nargs='+', default=[500, 1000],
                   help='SVGD-Iterationszahlen, je Initialisierung einmal.')
    p.add_argument('--edge_sigma', type=float, default=0.025,
                   help='Gauss-Abfallbreite am Rand der Logo-Maske (0.02-0.03).')
    p.add_argument('--fill_frac', type=float, default=0.8,
                   help='Anteil des Einheitsquadrats, den die Logo-Bounding-Box fuellt.')
    p.add_argument('--erg_K', type=int, default=K_DEFAULT,
                   help='Fourier-Frequenzgitter K x K fuer die ergodische Metrik (Loeser-Konvention: 10).')
    p.add_argument('--x0', type=float, nargs=2, default=[0.12, 0.12],
                   help='Startpunkt unten links.')
    p.add_argument('--x1_linear', type=float, nargs=2, default=[0.12, 0.88],
                   help='Zielpunkt der linearen Initialisierung, oben links.')
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--tsteps', type=int, default=200)
    p.add_argument('--step_size', type=float, default=0.01)
    p.add_argument('--h', type=float, default=0.01)
    p.add_argument('--score_scale', type=float, default=1.0)
    p.add_argument('--grid_res', type=int, default=128)
    p.add_argument('--device', type=str, default='cpu',
                   help="'cpu' oder 'cuda' fuer den Modell-Warmstart (Solver laeuft ueber JAX separat).")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quiet', action='store_true', help='Keine tqdm-Fortschrittsbalken im Solver.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isfile(args.logo):
        print(f"[FEHLER] Logo-Datei nicht gefunden: {args.logo}")
        return 1
    if not os.path.isfile(args.checkpoint):
        print(f"[FEHLER] Checkpoint nicht gefunden: {args.checkpoint}")
        return 1

    print(f"\n{'=' * 78}\n  IAS-Logo SVGD-Initialisierungstest\n"
          f"  Logo: {args.logo}\n  Checkpoint: {os.path.basename(args.checkpoint)}\n"
          f"  Iterationen: {args.iters}\n{'=' * 78}\n")

    print("[1/4] Baue Zieldichte: Flaeche mit Gauss-Abfall zum Rand ...")
    density_field = IASDensityField(
        args.logo, edge_sigma=args.edge_sigma, fill_frac=args.fill_frac)
    print(f"  Distanztransformation auf {density_field.mask.shape} Pixeln, "
         f"edge_sigma={args.edge_sigma}")

    print("[2/4] Baue Initialisierungen und fuehre SVGD-Laeufe aus ...")
    results, d_map, extra = run_all(
        density_field, tuple(args.x0), tuple(args.x1_linear), args.checkpoint,
        args.iters, dt=args.dt, tsteps=args.tsteps, step_size=args.step_size,
        h=args.h, score_scale=args.score_scale, grid_res=args.grid_res,
        erg_K=args.erg_K, device=args.device, seed=args.seed, verbose=not args.quiet,
    )

    print("[3/4] Visualisiere Ergebnisse ...")
    save_init_overview(results, d_map, tuple(args.x0), args.iters, args.out_dir)
    save_individual_figures(results, d_map, tuple(args.x0), args.out_dir)
    save_comparison_grid(results, d_map, tuple(args.x0), args.iters, args.out_dir)

    print("[4/4] Schreibe summary.json und metrics.csv ...")
    summary = {
        label: dict(init_name=r['init_name'], n_iter=r['n_iter'],
                   wall_time_s=r['wall_time'], metrics=r['metrics'],
                   init_metrics=r['init_metrics'])
        for label, r in results.items()
    }
    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(dict(config=vars(args), results=summary), f, indent=2)
    print(f"  Gespeichert -> {summary_path}")

    metric_keys = ['E_ergodic', 'E_smooth', 'E_boundary', 'E_total', 'coverage', 'path_len']
    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['init', 'n_iter', 'wall_time_s'] + metric_keys
                  + [f'init_{k}' for k in metric_keys])
        for name in INIT_ORDER:
            for n_iter in args.iters:
                r = results[f"{name}_iters{n_iter}"]
                w.writerow([name, n_iter, f"{r['wall_time']:.3f}"]
                          + [f"{r['metrics'][k]:.6f}" for k in metric_keys]
                          + [f"{r['init_metrics'][k]:.6f}" for k in metric_keys])
    print(f"  Gespeichert -> {csv_path}")

    print(f"\n{'=' * 100}\n  Ergodische Metriken (E_ergodic, niedriger ist besser)\n{'=' * 100}")
    print(f"  {'Initialisierung':<20}{'it':>6}{'E_ergodic':>12}{'(vorher)':>12}"
         f"{'coverage':>11}{'path_len':>11}{'Zeit [s]':>10}")
    for name in INIT_ORDER:
        for n_iter in args.iters:
            r = results[f"{name}_iters{n_iter}"]
            m, mi = r['metrics'], r['init_metrics']
            print(f"  {name:<20}{n_iter:>6}{m['E_ergodic']:>12.4f}{mi['E_ergodic']:>12.4f}"
                 f"{m['coverage']:>11.4f}{m['path_len']:>11.3f}{r['wall_time']:>10.1f}")

    print(f"\nFertig. Alle Ergebnisse in: {args.out_dir}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
