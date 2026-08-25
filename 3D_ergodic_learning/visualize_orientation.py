#!/usr/bin/env python3
r"""
visualize_orientation.py
========================
Orientierung gross und lesbar, statt als Beiwerk im 25er-Raster.

Im Standard-Holdout-Bild sind die Sensorachsen ein paar Pixel gross — man sieht,
*dass* sie da sind, aber nicht, was sie tun. Dieses Skript stellt sie in den
Mittelpunkt und zeigt vor allem eine Groesse, die bisher in keinem Bild
vorkommt: den **Fussabdruck**, also den Ort, an dem der Sensorstrahl auf die
Flaeche trifft.

Das ist keine Kosmetik. Der Lauf optimiert Abdeckung `--erg_on footprint`, also
genau dort — nicht an der Roboterposition. Ein Bild, das nur die Bahn zeigt,
zeigt damit nicht die Groesse, die minimiert wurde, und laesst die zentrale
Frage offen: liegt der Strahl auf der Zielverteilung, auch wenn die Bahn mit
Standoff darueber schwebt?

Drei Ansichten je Form:

* **Perspektive** — Bahn, Sensorachsen, Fussabdruck raeumlich
* **Seitenriss** — die Ansicht, in der Standoff und Anstellwinkel wirklich
  ablesbar sind; von schraeg oben taeuscht die Projektion
* **Aufsicht** — Fussabdruck gegen die Zieldichte, also die eigentliche
  Abdeckungsfrage, ohne Hoeheninformation als Stoerung

    python visualize_orientation.py --checkpoint checkpoints/<datei>.pt --shapes 4
"""

import argparse
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.normpath(os.path.join(_here, '..', 'bsplinax-main')),
           os.path.normpath(os.path.join(_here, '..', 'thesis_architecture',
                                         'ergodic_dataset_generator'))):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import viz_3d                                                     # noqa: E402
from data_3d import density_volume, load_pairs, sample_particles  # noqa: E402
from model_zoo import generate, load_model                        # noqa: E402
from obstacles import basis_torch                                 # noqa: E402
from orientation import (SurfaceField, rot6d_to_matrix,           # noqa: E402
                         sensor_axis)
from orientation_energy import pointing_term, standoff_term       # noqa: E402

SENSOR_ORANGE = viz_3d.SENSOR_ORANGE
GEN_GREEN = viz_3d.GEN_GREEN
GT_BLUE = viz_3d.GT_BLUE
FOOTPRINT = '#C2185B'          # Fussabdruck: eigene Farbe, sonst nicht von der
                               # Sensorachse zu trennen


def panel_perspective(ax, curve, R, foot, volume, title):
    viz_3d.style_axes3d(ax, title)
    ax.view_init(elev=24, azim=-58)
    viz_3d.draw_density(ax, volume, max_points=2500, floor_shadow=True)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=GEN_GREEN, lw=2.2,
            alpha=0.95, zorder=5)
    viz_3d.draw_frames(ax, curve, R, n_arrows=18, length=0.11)
    ax.plot(foot[:, 0], foot[:, 1], foot[:, 2], color=FOOTPRINT, lw=1.4,
            alpha=0.85, zorder=6)


def panel_side(ax, curve, R, foot, volume, title, n_arrows=22):
    """Seitenriss: x gegen z, y wird weggelassen.

    Hier ist ablesbar, was die Perspektive verschluckt — wie hoch die Bahn
    ueber der Flaeche liegt und ob die Achsen wirklich nach unten zeigen.
    """
    ax.set_facecolor('white')
    ax.set_title(title, color='#1A1A2E', fontsize=10)
    R_ = volume.shape[-1]
    zs = np.linspace(0, 1, R_)
    prof = volume.sum(axis=(1, 2)) if volume.ndim == 3 else None
    if prof is not None and prof.max() > 0:
        ax.fill_betweenx(zs, 0, prof / prof.max() * 0.12, color='#EF6C00',
                         alpha=0.18, lw=0)

    ax.plot(curve[:, 0], curve[:, 2], color=GEN_GREEN, lw=2.2, alpha=0.95)
    ax.plot(foot[:, 0], foot[:, 2], color=FOOTPRINT, lw=1.3, alpha=0.8)
    idx = np.linspace(0, len(curve) - 1, n_arrows).astype(int)
    a = R_ and sensor_axis(torch.from_numpy(R[idx]).unsqueeze(0)).numpy()[0]
    ax.quiver(curve[idx, 0], curve[idx, 2], a[:, 0], a[:, 2],
              color=SENSOR_ORANGE, alpha=0.9, width=0.005,
              scale=14, zorder=6)
    ax.set_xlabel('x', color='#555'); ax.set_ylabel('z', color='#555')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    for s in ax.spines.values():
        s.set_color('#ccc')


def panel_top(ax, curve, foot, volume, title):
    """Aufsicht: Fussabdruck gegen die Zieldichte.

    Die eigentliche Abdeckungsfrage. Gruen ist, wo der Roboter war, rot, wo er
    hingesehen hat — und nur das zweite zaehlt fuer die Metrik.
    """
    ax.set_facecolor('white')
    ax.set_title(title, color='#1A1A2E', fontsize=10)
    proj = volume.sum(axis=0) if volume.ndim == 3 else volume
    ax.imshow(proj / max(proj.max(), 1e-12), origin='lower',
              extent=[0, 1, 0, 1], cmap=viz_3d.WHITE_INFERNO, alpha=0.55)
    ax.plot(curve[:, 0], curve[:, 1], color=GEN_GREEN, lw=1.6, alpha=0.55,
            label='Bahn')
    ax.plot(foot[:, 0], foot[:, 1], color=FOOTPRINT, lw=2.0, alpha=0.95,
            label='Fussabdruck')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax.set_xlabel('x', color='#555'); ax.set_ylabel('y', color='#555')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    for s in ax.spines.values():
        s.set_color('#ccc')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--shapes', type=int, default=4)
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--grid_res', type=int, default=64)
    p.add_argument('--bspline_pts', type=int, default=256)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--out', default='viz/orientation_detail.png')
    p.add_argument('--device', default=None)
    a = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dev = torch.device(a.device or
                       ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, kind, meta = load_model(a.checkpoint, dev)
    print(f"  Checkpoint: {os.path.basename(a.checkpoint)}")
    print(f"  Epoche {meta.get('epoch')}, Orientierung={meta.get('orientation')}, "
          f"frame_mode={meta.get('frame_mode')}")
    if not meta.get('orientation'):
        raise SystemExit("Dieser Checkpoint hat keinen Orientierungskopf — "
                         "es gibt nichts zu zeichnen.")

    traj, shape_defs, splits = load_pairs(meta['nxi'])
    val = [l for l in traj if splits[l] == 'val']
    labels = a.labels or val[:a.shapes]
    print(f"  Formen: {', '.join(labels)}")

    vols = np.stack([density_volume(shape_defs[l], resolution=a.grid_res)
                     for l in labels])
    parts = sample_particles(torch.tensor(vols), torch.arange(len(labels)),
                             meta['n_particles'], dev)
    basis = basis_torch(meta['nxi'], a.bspline_pts, 5, dev)

    fig, axes = plt.subplots(len(labels), 3, figsize=(15, 4.6 * len(labels)),
                             facecolor='white', squeeze=False)
    stats = []

    for i, lbl in enumerate(labels):
        cps, rot6d, _ = generate(model, kind, parts[i:i + 1], 1, meta,
                                 a.steps, dev, seed=0)
        if rot6d is None:
            raise SystemExit("Das Modell liefert keine Rotationen.")

        curve_t = torch.einsum('ti,ic->tc', basis, cps[0].to(dev))
        r_t = torch.einsum('ti,ic->tc', basis, rot6d[0].to(dev))
        R_t = rot6d_to_matrix(r_t)
        axis_t = sensor_axis(R_t.unsqueeze(0))[0]

        field = SurfaceField(vols[i], device=str(dev))
        foot_t = field.footprint(curve_t.unsqueeze(0), axis_t.unsqueeze(0))[0]
        d_t = field.direction(curve_t.unsqueeze(0))[0]
        dist_t = field.distance(curve_t.unsqueeze(0))[0]

        ang = torch.rad2deg(torch.arccos(
            (axis_t * d_t).sum(-1).clamp(-1, 1)))
        stats.append({
            'shape': lbl,
            'pointing_deg': float(ang.mean()),
            'within30': float((ang < 30).float().mean()),
            'standoff': float(dist_t.mean()),
            'standoff_sd': float(dist_t.std()),
        })

        curve = curve_t.detach().cpu().numpy()
        Rm = R_t.detach().cpu().numpy()
        foot = foot_t.detach().cpu().numpy()

        ax0 = fig.add_subplot(len(labels), 3, i * 3 + 1, projection='3d')
        axes[i][0].remove()
        panel_perspective(ax0, curve, Rm, foot, vols[i], f"'{lbl}' — Perspektive")
        panel_side(axes[i][1], curve, Rm, foot, vols[i], f"'{lbl}' — Seitenriss")
        panel_top(axes[i][2], curve, foot, vols[i], f"'{lbl}' — Aufsicht")

    fig.suptitle(f"Orientierung im Detail — {os.path.basename(a.checkpoint)[:60]}",
                 color='#1A1A2E', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor='white')
    plt.close(fig)

    print(f"\n  {'Form':<20}{'Zeigefehler':>13}{'<30 Grad':>11}"
          f"{'Standoff':>11}{'Streuung':>11}")
    print("  " + "-" * 64)
    for s in stats:
        print(f"  {s['shape'][:20]:<20}{s['pointing_deg']:11.1f} Grad"
              f"{s['within30']:10.0%}{s['standoff']:11.3f}{s['standoff_sd']:11.3f}")
    m = lambda k: sum(s[k] for s in stats) / len(stats)
    print("  " + "-" * 64)
    print(f"  {'Mittel':<20}{m('pointing_deg'):11.1f} Grad"
          f"{m('within30'):10.0%}{m('standoff'):11.3f}{m('standoff_sd'):11.3f}")
    print(f"\n  Standoff-Ziel war 0.12 +- 0.03")
    print(f"  Bild -> {a.out}\n")


if __name__ == '__main__':
    main()
