"""Qualitative grid: final 30 trajectories per (shape, variant) over the density."""
import os
import sys

_eval = '/media/philipp/storage/Dokumente/Uni/Master_thesis/thesis_architecture/evaluation_full_matrix'
_arch = os.path.dirname(_eval)
for _p in (_eval, os.path.join(_arch, 'exploration'), _arch):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import smoothness_db as sdb
import viz
from plot_smoothness_comparison import VARIANT_LABELS
from ergodic_energy_torch import ergodic_term, make_k_grid, target_coeffs_from_grid

out_tag = sys.argv[1]
step_fracs = [float(s) for s in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
db_dir = os.path.join(_eval, 'results', out_tag)
conn = sdb.open_db(os.path.join(db_dir, 'smoothness_comparison.db'))
cfg = sdb.get_meta(conn, 'config', {})
shapes, variants = cfg['shapes'], cfg['variants']
k_idx, Lambda = (torch.as_tensor(a, dtype=torch.float32) for a in make_k_grid(10))
basis = {}


def dense_states(row):
    cps = row['cps']
    if row['log_space'] != 'cps':
        return cps
    nxi = row['nxi']
    if nxi not in basis:
        blob, n_pts = conn.execute("SELECT matrix, n_points FROM basis WHERE nxi=?", (nxi,)).fetchone()
        basis[nxi] = np.frombuffer(blob, dtype=np.float32).reshape(n_pts, nxi)
    return np.einsum('pi,sid->spd', basis[nxi], cps)


def panel(ax, truth, curves, title):
    viz.style_axes(ax)
    viz.draw_density(ax, truth)
    for c in range(1, len(curves)):
        ax.plot(curves[c, :, 0], curves[c, :, 1], color='#00C853', lw=1.0, alpha=0.3)
    ax.plot(curves[0, :, 0], curves[0, :, 1], color='#00C853', lw=2.2, alpha=0.95)
    ax.set_title(title, color='#1A1A2E', fontsize=8)


if step_fracs is None:
    fig, axes = plt.subplots(len(shapes), len(variants), figsize=(2.8 * len(variants), 2.9 * len(shapes)),
                             facecolor='white', squeeze=False)
    for r, shape in enumerate(shapes):
        truth = sdb.load_truth(conn, shape)
        phi_k = target_coeffs_from_grid(torch.as_tensor(truth.copy()), k_idx)
        for c, variant in enumerate(variants):
            rows = list(sdb.iter_runs(conn, shape=shape, variant=variant, with_states=True))
            final = np.stack([dense_states(row)[-1] for row in rows])
            e = ergodic_term(torch.as_tensor(final), k_idx, Lambda, phi_k).mean().item()
            panel(axes[r, c], truth, final,
                  f"{VARIANT_LABELS[variant]}\n{shape}: ergodic err {e:.2f}")
    fig.suptitle(f'Final 30 trajectories after {cfg["n_iters"]} iterations (fixed TSVEC-SVGD)',
                 color='#1A1A2E', fontsize=11)
    out = os.path.join(db_dir, 'qualitative_final.png')
else:
    shape, variant = sys.argv[3], sys.argv[4]
    truth = sdb.load_truth(conn, shape)
    phi_k = target_coeffs_from_grid(torch.as_tensor(truth.copy()), k_idx)
    rows = list(sdb.iter_runs(conn, shape=shape, variant=variant, with_states=True))
    states = np.stack([dense_states(row) for row in rows])          # (C, S, T, 2)
    n = states.shape[1] - 1
    steps = sorted({int(round(f * n)) for f in step_fracs})
    fig, axes = plt.subplots(1, len(steps), figsize=(2.9 * len(steps), 3.2), facecolor='white', squeeze=False)
    for ax, s in zip(axes[0], steps):
        e = ergodic_term(torch.as_tensor(states[:, s]), k_idx, Lambda, phi_k).mean().item()
        panel(ax, truth, states[:, s], f'iteration {s}: ergodic err {e:.2f}')
    fig.suptitle(f'{VARIANT_LABELS[variant]} -- {shape}', color='#1A1A2E', fontsize=10)
    out = os.path.join(db_dir, f'qualitative_steps_{shape}_{variant}.png')
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=120, facecolor='white')
print(out)
