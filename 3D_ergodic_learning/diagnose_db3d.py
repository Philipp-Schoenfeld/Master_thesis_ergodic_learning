#!/usr/bin/env python3
"""Phase 0 of the DB improvement plan: where does the teacher's Sinkhorn error come from?

In the first lambda-630 comparison the projected DB teacher scored a Sinkhorn
footprint error of 4.0e-3 against 1.6e-3 for the networks trained on it. This
script splits that number up on a subset of entries (all holdout splits plus
`--n_train` train entries):

    floor        128 points drawn i.i.d. from the target itself (prob. ~ mu),
                 against the target: the level of a random sample. Not a lower
                 bound -- an optimised point set beats i.i.d. sampling
                 (quantisation vs Monte Carlo), and refined paths do go below it
    cps_pos      teacher LSQ control points, scored at the robot position
    cps_foot     same, scored at the beam footprint (the training metric)
    dense_pos    the full stored teacher path (128 points by index), at the
                 position -> loss due to the B-spline fit
    off_foot     teacher moved 0.12 along the surface normal, footprint
    refined      teacher refined on the training objective (refine_db3d.py):
                 an estimate of what is reachable per entry

plus standoff / pointing of the teacher against a local PCA plane
(surface_metrics.py) -- the teacher is built on the surface looking along the
normal, so these should come out ~0 and ~0 deg; a large value would mean the
DB rotations are broken rather than the metric.

The group `ebene` contains the planar surfaces (projection close to the
identity), so comparing it to the curved groups separates "the 2D teacher is
weak" from "the projection breaks it".

    python diagnose_db3d.py --db3d ergodic_dataset_3d_lm.db --out_dir results/db_diagnose
"""
import argparse
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_surfaces import load_surface_db
from ergodic_metric import sinkhorn_error
from orientation import rot6d_to_matrix, sensor_axis
from refine_db3d import add_refine_args, refiner_from_args
from surface_metrics import standoff_pointing

COLS = ['floor', 'cps_pos', 'cps_foot', 'dense_pos', 'off_foot', 'refined',
        'so_teacher', 'pt_teacher', 'so_refined', 'pt_refined']


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db3d', default='ergodic_dataset_3d_lm.db')
    p.add_argument('--n_train', type=int, default=2000)
    p.add_argument('--per_holdout', type=int, default=400, help='entries per holdout split')
    p.add_argument('--floor_draws', type=int, default=4)
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'db_diagnose'))
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    add_refine_args(p)
    a = p.parse_args()
    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    ent = load_surface_db(a.db3d, target_cps=True, keep_path=True)
    sel = []
    for sp in sorted({e['split'] for e in ent}):
        pool = [e for e in ent if e['split'] == sp]
        n = a.n_train if sp == 'train' else a.per_holdout
        sel += [pool[i] for i in sorted(rng.permutation(len(pool))[:n])]
    del ent
    print(f"{len(sel)} entries: "
          + ", ".join(f"{s}={sum(e['split'] == s for e in sel)}" for s in sorted({e['split'] for e in sel})))

    ref = refiner_from_args(a, dev)
    sk = ref.erg._sinkhorn
    pos_loss = lambda cps, parts: sinkhorn_error(torch.einsum('pi,bid->bpd', ref.B, cps.float()),
                                                 parts, sk)
    res = {c: [] for c in COLS}
    g = torch.Generator(device=dev).manual_seed(a.seed)
    for b0 in range(0, len(sel), a.batch):
        batch = sel[b0:b0 + a.batch]
        x1 = torch.from_numpy(np.stack([e['x1'] for e in batch])).to(dev)
        parts = torch.from_numpy(np.stack([e['parts'] for e in batch])).to(dev)
        B, T = len(batch), ref.B.shape[0]
        with torch.no_grad():
            w = parts[..., 3].clamp(min=0) + 1e-12
            fl = 0
            for _ in range(a.floor_draws):
                idx = torch.multinomial(w, T, replacement=True, generator=g)
                pts = torch.gather(parts[..., :3], 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                fl = fl + sinkhorn_error(pts, parts, sk) / a.floor_draws
            res['floor'].append(fl)
            res['cps_pos'].append(pos_loss(x1[..., :3], parts))
            res['cps_foot'].append(ref.erg.coverage_error(x1[..., :3], parts, rot6d=x1[..., 3:]))
            dense = torch.stack([torch.from_numpy(
                e['path_pos'][np.linspace(0, len(e['path_pos']) - 1, T).round().astype(int)])
                for e in batch]).to(dev)
            res['dense_pos'].append(sinkhorn_error(dense.float(), parts, sk))
            x_off = ref.offset(x1)
            res['off_foot'].append(ref.erg.coverage_error(x_off[..., :3], parts, rot6d=x_off[..., 3:]))
            curve = torch.einsum('pi,bid->bpd', ref.B, x1[..., :3])
            axis = sensor_axis(rot6d_to_matrix(torch.einsum('pi,bic->bpc', ref.B, x1[..., 3:])), 2)
            so, pt = standoff_pointing(curve, axis, parts)
            res['so_teacher'].append(so.mean(-1)); res['pt_teacher'].append(pt.mean(-1))
        x_ref, _ = ref.run(x_off, parts, a.iters, a.lr)
        m = ref.metrics(x_ref, parts)
        res['refined'].append(m['erg']); res['so_refined'].append(m['so']); res['pt_refined'].append(m['pt'])
        print(f"  {b0 + B}/{len(sel)}", flush=True)
    R = {c: torch.cat(v).cpu().numpy() for c, v in res.items()}

    split = np.array([e['split'] for e in sel]); grp = np.array([e['gruppe'] for e in sel])
    import csv
    with open(os.path.join(a.out_dir, 'diagnose_per_entry.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['name', 'split', 'surface', 'gruppe'] + COLS)
        for i, e in enumerate(sel):
            wr.writerow([e['name'], e['split'], e['surface'], e['gruppe']]
                        + [f"{R[c][i]:.6g}" for c in COLS])

    subsets = [('all', np.ones(len(sel), bool))]
    subsets += [(s, split == s) for s in sorted(set(split))]
    subsets += [(f"group {g_}", grp == g_) for g_ in sorted(set(grp))]
    md = ["# DB diagnosis (phase 0)", "",
          "Medians of the Sinkhorn error (columns floor..refined) and of the PCA-plane "
          "standoff / pointing. See the docstring of diagnose_db3d.py for the columns.", "",
          "| subset | n | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for name, m in subsets:
        md.append(f"| {name} | {m.sum()} | " + " | ".join(
            (f"{np.median(R[c][m]):.3e}" if not c.startswith(('so', 'pt'))
             else f"{np.median(R[c][m]):.3f}") for c in COLS) + " |")
    open(os.path.join(a.out_dir, 'diagnose_summary.md'), 'w').write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    erg_cols = ['floor', 'cps_pos', 'dense_pos', 'cps_foot', 'off_foot', 'refined']
    colors = ['#9E9E9E', '#90CAF9', '#42A5F5', '#1565C0', '#E69F00', '#00C853']
    names = [n for n, _ in subsets]
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(names)), 4.6), facecolor='white')
    ax.set_facecolor('white'); ax.grid(alpha=0.2)
    for s in ax.spines.values():
        s.set_color('#cccccc')
    wdt = 0.8 / len(erg_cols)
    for j, (c, col) in enumerate(zip(erg_cols, colors)):
        ax.bar(np.arange(len(names)) + j * wdt, [np.median(R[c][m]) for _, m in subsets],
               wdt, color=col, label=c)
    ax.set_xticks(np.arange(len(names)) + 0.4 - wdt / 2)
    ax.set_xticklabels(names, rotation=25, fontsize=8, color='#555555')
    ax.set_yscale('log')
    ax.set_ylabel('median Sinkhorn error', color='#1A1A2E')
    ax.set_title('Where the teacher loses: floor -> fit -> footprint -> standoff -> refined',
                 color='#1A1A2E', fontweight='bold')
    ax.legend(fontsize=7, ncol=3, frameon=True, facecolor='white', edgecolor='#ddd')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'diagnose_breakdown.png'), dpi=150, facecolor='white')
    plt.close(fig)
    print(f"-> {a.out_dir}")


if __name__ == '__main__':
    main()
