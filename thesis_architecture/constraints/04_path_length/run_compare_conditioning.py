#!/usr/bin/env python3
r"""
run_compare_conditioning.py
===========================
The third arm that the holdout table was missing: does the model's *learned*
length conditioning control path length, and does it pay a different price for
it than the inference-time force?

Three arms per shape, all from the same checkpoint, same seed, same particles,
same CFG weight:

    frei    null length token, no force        (what the other runners call free)
    kraft   null length token + TargetLength   (constraint 4)
    kond    length passed to the model         (its trained FiLM channel)

The ladder is deliberately **absolute**, not a multiple of each shape's own
free length. The embedding normalises as

    u = (log1p(L) - log1p(log_ref)) / log_scale

with log_ref = 11.05 for this checkpoint, so u = 0 sits at L ≈ 11 while the
unconditioned generations come out around L ≈ 6 (u ≈ -1.3). A relative ladder
would hide that offset; an absolute one shows where in its trained range the
conditioning actually has authority.

    python run_compare_conditioning.py [--shapes A digit_5] [--length_cfg 0.0]
"""
import argparse
import importlib
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (pick_device, load_generator, density_and_particles, basis_torch,
                    guided_generate, energy_force, curve_of, cp_to_curve_np,
                    arc_length, ErgodicMetrics, HOLDOUT_SHAPES, save, write_metrics,
                    summarise, results_dir, draw_density, draw_guided, style_axes,
                    C_DARK, C_GREY, C_GEN, C_MARK)
from length import TargetLength

LADDER = [4.0, 6.0, 8.0, 11.0, 14.0, 18.0]
C_KOND = '#1565C0'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', nargs='*', default=None)
    p.add_argument('--ladder', nargs='*', type=float, default=LADDER)
    p.add_argument('--length_cfg', type=float, default=0.0,
                   help="CFG weight on the length channel (its own guidance scale).")
    p.add_argument('--weight', type=float, default=30.0)
    p.add_argument('--max_force', type=float, default=0.5)
    p.add_argument('--polish_steps', type=int, default=400)
    p.add_argument('--polish_lr', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--tag', default='conditioning')
    args = p.parse_args()

    shapes = args.shapes if args.shapes else HOLDOUT_SHAPES
    device = pick_device()
    model, meta = load_generator(device)
    B = basis_torch(meta['nxi'], 256, 5, device=device)
    erg = ErgodicMetrics(meta['nxi'], device)
    out = results_dir(__file__)

    # The conditioning arm calls the trained module's own sampler rather than a
    # re-implementation, so the length channel is driven exactly as in training.
    gen_mod = importlib.import_module(meta['_module'])
    gen_cond = gen_mod.generate_particle_trajectories

    u_of = lambda L: (math.log1p(L) - math.log1p(meta['log_ref'])) / meta['log_scale']
    print(f"log_ref={meta['log_ref']:.2f}  log_scale={meta['log_scale']:.3f}  "
          f"length_cfg={args.length_cfg}")
    print("Ladder (Ziel -> u): " +
          "  ".join(f"{L:.0f}->{u_of(L):+.2f}" for L in args.ladder))

    rows = []
    for i, name in enumerate(shapes):
        d_map, particles = density_and_particles(name, meta, device, seed=args.seed)
        dens_t = torch.tensor(d_map, dtype=torch.float32, device=device)
        phi = erg.phi_for(d_map)

        free = guided_generate(model, meta, particles, force=None, steps=args.steps,
                               device=device, seed=args.seed)
        L_free = float(arc_length(curve_of(free, B))[0])
        e_free = erg.score(free, phi, dens_t)

        for L_t in args.ladder:
            con = TargetLength(target=L_t, mode='exact')
            kraft = guided_generate(model, meta, particles,
                                    force=energy_force(con.energy, B), steps=args.steps,
                                    device=device, seed=args.seed,
                                    force_weight=args.weight, force_t_start=0.3,
                                    max_force=args.max_force,
                                    polish_steps=args.polish_steps,
                                    polish_lr=args.polish_lr)
            g = torch.Generator(device=device).manual_seed(args.seed)
            kond, _ = gen_cond(model, particles, num_samples=1, nxi=meta['nxi'],
                               nd=meta['nd'], steps=args.steps, device=str(device),
                               cfg_weight=meta['cfg_weight'], generator=g,
                               length=L_t, length_cfg_weight=args.length_cfg)

            Lk = float(arc_length(curve_of(kraft, B))[0])
            Lc = float(arc_length(curve_of(kond, B))[0])
            ek = erg.score(kraft, phi, dens_t)
            ec = erg.score(kond, phi, dens_t)
            rows.append(dict(
                shape=name, ziel=L_t, u=round(u_of(L_t), 3),
                laenge_frei=round(L_free, 3),
                laenge_kraft=round(Lk, 3), laenge_kond=round(Lc, 3),
                relf_kraft=round(Lk / L_t - 1, 4), relf_kond=round(Lc / L_t - 1, 4),
                E_erg_frei=round(e_free[0], 4),
                E_erg_kraft=round(ek[0], 4), E_erg_kond=round(ec[0], 4),
                cov_frei=round(e_free[1], 5),
                cov_kraft=round(ek[1], 5), cov_kond=round(ec[1], 5)))
        print(f"  [{i + 1:2d}/{len(shapes)}] {name:<24} L_frei={L_free:.2f}", flush=True)

    write_metrics(rows, out, f'{args.tag}_metrics.csv')
    summarise(rows, keys=['relf_kraft', 'relf_kond'],
              label='Relativer Laengenfehler ueber alle Ziele')

    print(f"\n  {'Ziel':>6} {'u':>6} | {'L Kraft':>9} {'L Kond':>9} | "
          f"{'relF Kraft':>11} {'relF Kond':>10} | {'cov Kraft':>10} {'cov Kond':>9}")
    ladder_stats = []
    for L_t in args.ladder:
        sub = [r for r in rows if r['ziel'] == L_t]
        st = dict(ziel=L_t, u=sub[0]['u'],
                  Lk=np.mean([r['laenge_kraft'] for r in sub]),
                  Lc=np.mean([r['laenge_kond'] for r in sub]),
                  rk=np.mean([abs(r['relf_kraft']) for r in sub]),
                  rc=np.mean([abs(r['relf_kond']) for r in sub]),
                  ck=np.mean([r['cov_kraft'] for r in sub]),
                  cc=np.mean([r['cov_kond'] for r in sub]),
                  ek=np.mean([r['E_erg_kraft'] for r in sub]),
                  ec=np.mean([r['E_erg_kond'] for r in sub]))
        ladder_stats.append(st)
        print(f"  {st['ziel']:>6.1f} {st['u']:>+6.2f} | {st['Lk']:>9.2f} {st['Lc']:>9.2f} | "
              f"{st['rk']:>11.3f} {st['rc']:>10.3f} | {st['ck']:>10.4f} {st['cc']:>9.4f}")

    L_free_mean = np.mean([r['laenge_frei'] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), facecolor='white')

    ax = axes[0]
    lo, hi = min(args.ladder) - 1, max(args.ladder) + 1
    ax.plot([lo, hi], [lo, hi], color='#B0B4B0', lw=1.2, ls='--', label='perfekte Kontrolle')
    ax.axhline(L_free_mean, color=C_GREY, lw=1, ls=':',
               label=f'ungeführt (L≈{L_free_mean:.1f})')
    for r in rows:
        ax.plot([r['ziel']], [r['laenge_kond']], 'o', color=C_KOND, ms=2.5, alpha=.25)
        ax.plot([r['ziel']], [r['laenge_kraft']], 'o', color=C_GEN, ms=2.5, alpha=.25)
    ax.plot([s['ziel'] for s in ladder_stats], [s['Lk'] for s in ladder_stats],
            '-o', color=C_GEN, lw=2, ms=6, label='Kraft')
    ax.plot([s['ziel'] for s in ladder_stats], [s['Lc'] for s in ladder_stats],
            '-o', color=C_KOND, lw=2, ms=6, label='Konditionierung')
    ax.set_xlabel('angeforderte Länge', fontsize=9, color=C_GREY)
    ax.set_ylabel('erreichte Länge', fontsize=9, color=C_GREY)
    ax.set_title('Längenkontrolle: angefordert gegen erreicht', fontsize=10, color=C_DARK)

    ax2 = axes[1]
    ax2.plot([s['ziel'] for s in ladder_stats], [s['ck'] for s in ladder_stats],
             '-o', color=C_GEN, lw=2, ms=6, label='Kraft')
    ax2.plot([s['ziel'] for s in ladder_stats], [s['cc'] for s in ladder_stats],
             '-o', color=C_KOND, lw=2, ms=6, label='Konditionierung')
    ax2.axhline(np.mean([r['cov_frei'] for r in rows]), color=C_GREY, lw=1, ls=':',
                label='ungeführt')
    ax2.set_xlabel('angeforderte Länge', fontsize=9, color=C_GREY)
    ax2.set_ylabel('coverage (kleiner = besser)', fontsize=9, color=C_GREY)
    ax2.set_title('Was die Länge für die Abdeckung kostet', fontsize=10, color=C_DARK)

    for a in axes:
        a.set_facecolor('white')
        a.tick_params(labelsize=7, colors=C_GREY)
        for sp in a.spines.values():
            sp.set_color('#ccc')
        a.grid(True, alpha=0.2, lw=0.4, color='gray')
        a.legend(frameon=True, fontsize=8, facecolor='white', edgecolor='#ddd')

    fig.suptitle(f"Constraint 4 — Kraft gegen gelernte Konditionierung   "
                 f"(Holdout n={len(shapes)}, length_cfg={args.length_cfg})",
                 fontsize=12, color=C_DARK)
    fig.tight_layout()
    save(fig, out, f'{args.tag}_response.png')


if __name__ == '__main__':
    main()
