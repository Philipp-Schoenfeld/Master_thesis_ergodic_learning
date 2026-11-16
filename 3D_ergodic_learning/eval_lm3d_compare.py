#!/usr/bin/env python3
"""Compare 3D CFM checkpoints on the holdout splits of the projected LM DB.

Every model gets the same holdout entries (particle cloud + start point), samples
`--n_gen` trajectories with the same noise seed and is scored with the same
metrics, so the numbers are directly comparable even though the models were
trained with different losses / backbones:

    sinkhorn   debiased Sinkhorn (blur 0.05) between the beam footprint and the
               weighted target particles  (the lambda-630 training objective)
    fourier    spectral ergodic metric (K=6) of the same footprint (the
               objective of the older lambda-100 runs)
    standoff   mean |distance to the nearest occupied particle - 0.12|
    pointing   angle between sensor axis and direction to that particle
               (both biased near the surface, see surface_metrics.py)
    standoff_pca / pointing_pca  the same against a local PCA plane of the
               particles (unbiased; the on-surface teacher gives ~0.12 / ~0 deg)
    cp_rmse    RMSE of the generated control points to the DB teacher
    start_err  distance of control point 0 to the requested start

The DB teacher (LSQ control points of the stored path) is scored as a reference.

    python eval_lm3d_compare.py \
        --models "DiT-B (lambda 630)=checkpoints/...dit...pt" "U-Net D512 (lambda 630)=..." \
        --db3d ergodic_dataset_3d_lm.db --per_split 100 --out_dir results/lm3d_compare
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import viz_3d
from data_surfaces import load_surface_db, volume_from_particles
from ergodic_metric import ErgodicLoss
from flow_matching_cond_particles_crossattn import (
    ParticleCrossAttnFlowNetwork, generate_particle_trajectories)
from flow_matching_cond_particles_dit import ParticleDiTFlowNetwork, dit_kwargs
from obstacles import bspline_basis_matrix
from orientation import rot6d_to_matrix, sensor_axis
from orientation_energy import ParticleSurface
from surface_metrics import standoff_pointing

REF = 'DB teacher'
# Colour per model (distinct, colour-blind safe); the teacher is the deep blue of the
# trajectory style guide.
PALETTE = ['#00A86B', '#E69F00', '#7B5EA7', '#D55E00']
METRICS = [('sinkhorn', 'Sinkhorn footprint error (lower is better)'),
           ('fourier', 'Fourier ergodic error K=6 (lower is better)'),
           ('standoff', 'Standoff dev., nearest particle (lower is better)'),
           ('pointing', 'Pointing [deg], nearest particle (lower is better)'),
           ('standoff_pca', 'Standoff dev. |d - 0.12|, PCA plane (lower is better)'),
           ('pointing_pca', 'Pointing [deg] to normal, PCA plane (lower is better)'),
           ('cp_rmse', 'Control-point RMSE to teacher (lower is better)')]


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    kw = dict(nxi=ck.get('nxi', 25), nd=3, D=ck.get('D', 384),
              predict_orientation=bool(ck.get('orientation', True)),
              start_cond=bool(ck.get('start_cond', False)),
              obstacle_cond=bool(ck.get('obstacle_cond', False)))
    if ck.get('backbone', 'unet') == 'dit':
        model = ParticleDiTFlowNetwork(**kw, **dit_kwargs(ck))
    else:
        model = ParticleCrossAttnFlowNetwork(**kw)
    model.load_state_dict(ck['model_state_dict'])
    info = dict(epoch=ck.get('epoch', -1) + 1, loss=ck.get('loss', float('nan')),
                params=sum(p.numel() for p in model.parameters()) / 1e6,
                backbone=ck.get('backbone', 'unet'), lambda_erg=ck.get('lambda_erg'),
                inpaint=bool(model.start_inpaint))
    del ck
    return model.to(device).eval(), info


def pick_entries(entries, per_split, seed):
    rng = np.random.default_rng(seed)
    out = []
    for sp in sorted({e['split'] for e in entries}):
        pool = [e for e in entries if e['split'] == sp]
        idx = rng.permutation(len(pool))[:per_split]
        out += [pool[i] for i in sorted(idx)]
    return out


@torch.no_grad()
def score(cps, rot6, parts, losses, basis, standoff_target, mu_thresh=0.5):
    """cps (S,25,3), rot6 (S,25,6), parts (N,4) -> dict of (S,) arrays."""
    S = cps.shape[0]
    P = parts.unsqueeze(0).expand(S, -1, -1).contiguous()
    res = {}
    for name, loss in losses.items():
        res[name] = loss.coverage_error(cps, P, rot6d=rot6).cpu().numpy()
    curve = torch.einsum('pi,kid->kpd', basis, cps)
    ps = ParticleSurface(P, mu_thresh)
    res['standoff'] = (ps.distance(curve) - standoff_target).abs().mean(-1).cpu().numpy()
    R = rot6d_to_matrix(torch.einsum('pi,kic->kpc', basis, rot6).reshape(-1, 6)
                        ).reshape(S, -1, 3, 3)
    ax = sensor_axis(R, axis=2)
    ax = ax / ax.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    cosang = (ax * ps.direction(curve)).sum(-1).clamp(-1, 1)
    res['pointing'] = torch.rad2deg(torch.acos(cosang)).mean(-1).cpu().numpy()
    dist, ang = standoff_pointing(curve, ax, P)
    res['standoff_pca'] = (dist - standoff_target).abs().mean(-1).cpu().numpy()
    res['pointing_pca'] = ang.mean(-1).cpu().numpy()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', required=True, help='"label=checkpoint.pt" ...')
    p.add_argument('--db3d', default='ergodic_dataset_3d_lm.db')
    p.add_argument('--splits', nargs='+', default=['val_form', 'val_flaeche', 'val_beides'])
    p.add_argument('--per_split', type=int, default=100)
    p.add_argument('--n_gen', type=int, default=4)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cfg_weight', type=float, default=2.0)
    p.add_argument('--standoff_target', type=float, default=0.12)
    p.add_argument('--n_examples', type=int, default=3, help='examples per split')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dump_npz', action='store_true',
                   help='also save generated trajectories, particles and DB targets to trajectories.npz')
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'lm3d_compare'))
    p.add_argument('--device', default=None)
    a = p.parse_args()

    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(a.out_dir, exist_ok=True)
    models = []
    for m in a.models:
        label, path = m.split('=', 1)
        models.append((label, path))

    entries = pick_entries(load_surface_db(a.db3d, splits=tuple(a.splits), target_cps=True),
                           a.per_split, a.seed)
    print(f"{len(entries)} holdout entries: "
          + ", ".join(f"{s}={sum(e['split'] == s for e in entries)}" for s in a.splits))

    basis = torch.from_numpy(bspline_basis_matrix(25, 128, 5)).float().to(dev)
    losses = {'sinkhorn': ErgodicLoss(nxi=25, nd=3, pts=128, ergodic_on='footprint',
                                      metric='sinkhorn', sinkhorn_blur=0.05).to(dev),
              'fourier': ErgodicLoss(nxi=25, K=6, nd=3, pts=128,
                                     ergodic_on='footprint').to(dev)}

    def with_extras(sc, cps, gt_cps, start):
        sc['cp_rmse'] = ((cps - gt_cps).pow(2).sum(-1).mean(-1).sqrt()).cpu().numpy()
        sc['start_err'] = (cps[:, 0] - start).norm(dim=-1).cpu().numpy()
        sc['path_len'] = (cps[:, 1:] - cps[:, :-1]).norm(dim=-1).sum(-1).cpu().numpy()
        return sc

    # rows[label][i] = dict of (n_gen,) arrays ; gens[label][i] = (cps, rot6)
    rows, gens, infos = {REF: []}, {}, {}
    for e in entries:
        x1 = torch.from_numpy(e['x1']).to(dev)
        parts = torch.from_numpy(e['parts']).to(dev)
        st = torch.from_numpy(e['start']).to(dev)
        sc = score(x1[None, :, :3], x1[None, :, 3:], parts, losses, basis, a.standoff_target)
        rows[REF].append(with_extras(sc, x1[None, :, :3], x1[None, :, :3], st))

    for label, path in models:
        print(f"\n=== {label}: {os.path.basename(path)}")
        model, infos[label] = load_model(path, dev)
        print(f"  {infos[label]}")
        rows[label], gens[label] = [], []
        for i, e in enumerate(entries):
            x1 = torch.from_numpy(e['x1']).to(dev)
            parts = torch.from_numpy(e['parts']).to(dev)
            st = torch.from_numpy(e['start']).to(dev)
            g = torch.Generator(device=dev).manual_seed(a.seed * 100003 + i)
            cps, rot6 = generate_particle_trajectories(
                model, parts, num_samples=a.n_gen, steps=a.steps, device=str(dev),
                cfg_weight=a.cfg_weight, start=st, generator=g)
            sc = score(cps, rot6, parts, losses, basis, a.standoff_target)
            rows[label].append(with_extras(sc, cps, x1[None, :, :3], st))
            gens[label].append((cps.cpu().numpy(), rot6.cpu().numpy()))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(entries)}  sinkhorn so far "
                      f"{np.mean([r['sinkhorn'].mean() for r in rows[label]]):.3e}", flush=True)
        del model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    labels = [REF] + [l for l, _ in models]
    meta = [dict(name=e['name'], split=e['split'], surface=e['surface'], gruppe=e['gruppe'])
            for e in entries]
    if a.dump_npz:
        dump = dict(names=np.array([m['name'] for m in meta]), splits=np.array([m['split'] for m in meta]),
                    groups=np.array([m['gruppe'] for m in meta]), surfaces=np.array([m['surface'] for m in meta]),
                    parts=np.stack([e['parts'] for e in entries]).astype(np.float32),
                    db_x1=np.stack([e['x1'] for e in entries]).astype(np.float32),
                    labels=np.array(labels[1:]))
        for j, (l, _) in enumerate(models):
            dump[f'cps_{j}'] = np.stack([g[0][:3] for g in gens[l]]).astype(np.float32)
            dump[f'rot6_{j}'] = np.stack([g[1][:3] for g in gens[l]]).astype(np.float32)
        np.savez_compressed(os.path.join(a.out_dir, 'trajectories.npz'), **dump)
    # per entry means (mean over the n_gen samples)
    M = {l: {k: np.array([r[k].mean() for r in rows[l]]) for k in rows[l][0]} for l in labels}

    with open(os.path.join(a.out_dir, 'per_entry.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        keys = list(M[REF])
        w.writerow(['name', 'split', 'surface', 'gruppe', 'model'] + keys)
        for l in labels:
            for i, m in enumerate(meta):
                w.writerow([m['name'], m['split'], m['surface'], m['gruppe'], l]
                           + [f"{M[l][k][i]:.6g}" for k in keys])

    # ---- summary table ----
    def agg(l, k, mask):
        v = M[l][k][mask]
        return float(np.mean(v)), float(np.median(v)), float(np.std(v) / np.sqrt(len(v)))

    splits = a.splits
    masks = {'all': np.ones(len(entries), bool)}
    masks.update({s: np.array([m['split'] == s for m in meta]) for s in splits})
    summary = {s: {l: {k: dict(zip(('mean', 'median', 'sem'), agg(l, k, mk)))
                       for k in M[l]} for l in labels} for s, mk in masks.items()}
    json.dump(dict(models=infos, n_entries=len(entries), args=vars(a), summary=summary),
              open(os.path.join(a.out_dir, 'summary.json'), 'w'), indent=1)
    md = ["| split | model | " + " | ".join(k for k, _ in METRICS) + " | start_err | path_len |",
          "|---|---|" + "---|" * (len(METRICS) + 2)]
    for s in masks:
        for l in labels:
            r = summary[s][l]
            md.append(f"| {s} | {l} | " + " | ".join(f"{r[k]['mean']:.4g}" for k, _ in METRICS)
                      + f" | {r['start_err']['mean']:.2g} | {r['path_len']['mean']:.3g} |")
    open(os.path.join(a.out_dir, 'summary.md'), 'w').write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))

    plot_all(a, labels, models, meta, M, masks, summary, entries, gens, rows, dev)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
INK, GRID = '#1A1A2E', '#cccccc'


def _style(ax):
    ax.set_facecolor('white')
    ax.grid(alpha=0.2)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors='#555555', labelsize=8)


def plot_all(a, labels, models, meta, M, masks, summary, entries, gens, rows, dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    col = {REF: '#1565C0'}
    col.update({l: PALETTE[i % len(PALETTE)] for i, (l, _) in enumerate(models)})
    gen_labels = [l for l, _ in models]
    keys = [k for k, _ in METRICS]
    names = list(masks)

    # 1) metric bars per split
    ncol = 4
    fig, axes = plt.subplots(2, ncol, figsize=(4.4 * ncol, 8.6), facecolor='white')
    axes = axes.ravel()
    for ax in axes[len(keys):]:
        ax.axis('off')
    w = 0.8 / len(labels)
    for ax, (k, title) in zip(axes, METRICS):
        _style(ax)
        for j, l in enumerate(labels):
            v = [summary[s][l][k]['mean'] for s in names]
            e = [summary[s][l][k]['sem'] for s in names]
            ax.bar(np.arange(len(names)) + j * w, v, w, yerr=e, color=col[l], label=l,
                   alpha=0.9, error_kw=dict(ecolor='#555555', lw=1, capsize=2))
        ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2)
        ax.set_xticklabels(names, rotation=20, fontsize=8)
        ax.set_title(title, fontsize=9, color=INK)
        if k in ('sinkhorn', 'fourier'):
            ax.set_yscale('log')
    axes[0].legend(fontsize=7, frameon=True, facecolor='white', edgecolor='#ddd')
    fig.suptitle(f"Holdout metrics per split (mean +- s.e.m. over {len(meta)} entries, "
                 f"{a.n_gen} samples each)", color=INK, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'metrics_by_split.png'), dpi=150, facecolor='white')
    plt.close(fig)

    # 2) Sinkhorn error per surface group
    groups = sorted({m['gruppe'] for m in meta})
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 3, 4.4), facecolor='white')
    _style(ax)
    for j, l in enumerate(labels):
        v = [np.mean(M[l]['sinkhorn'][[m['gruppe'] == g for m in meta]]) for g in groups]
        ax.bar(np.arange(len(groups)) + j * w, v, w, color=col[l], label=l, alpha=0.9)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - w / 2)
    ax.set_xticklabels(groups, rotation=25, fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('Sinkhorn footprint error', color=INK)
    ax.set_title('Sinkhorn error per surface group', color=INK, fontweight='bold')
    ax.legend(fontsize=7, frameon=True, facecolor='white', edgecolor='#ddd')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'sinkhorn_by_group.png'), dpi=150, facecolor='white')
    plt.close(fig)

    # 3) paired scatter + error distribution of the first two generative models
    if len(gen_labels) >= 2:
        l1, l2 = gen_labels[0], gen_labels[1]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor='white')
        _style(axes[0])
        x, y = M[l2]['sinkhorn'], M[l1]['sinkhorn']
        sp_col = {s: c for s, c in zip(a.splits, ['#1565C0', '#E69F00', '#D55E00'])}
        for s in a.splits:
            mk = masks[s]
            axes[0].scatter(x[mk], y[mk], s=14, alpha=0.7, color=sp_col[s], label=s)
        lim = [min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.2]
        axes[0].plot(lim, lim, color='#555555', lw=1)
        axes[0].set_xscale('log'); axes[0].set_yscale('log')
        axes[0].set_xlabel(f'{l2}  (Sinkhorn error)', color=INK)
        axes[0].set_ylabel(f'{l1}  (Sinkhorn error)', color=INK)
        axes[0].set_title(f'Per entry: {l1} wins below the diagonal ({np.mean(y < x):.0%} of entries)',
                          fontsize=9, color=INK)
        axes[0].legend(fontsize=7, frameon=True, facecolor='white', edgecolor='#ddd')
        _style(axes[1])
        for l in labels:
            v = np.sort(M[l]['sinkhorn'])
            axes[1].plot(v, np.arange(1, len(v) + 1) / len(v), color=col[l], lw=2, label=l)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Sinkhorn footprint error', color=INK)
        axes[1].set_ylabel('Cumulative fraction of entries', color=INK)
        axes[1].set_title('Error distribution (CDF)', fontsize=9, color=INK)
        axes[1].legend(fontsize=7, frameon=True, facecolor='white', edgecolor='#ddd')
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, 'paired_sinkhorn.png'), dpi=150, facecolor='white')
        plt.close(fig)

    # 4) example grids: rows = entries, columns = models
    B = bspline_basis_matrix(25, 256, 5)

    def example_fig(idxs, fname, title):
        panels = []
        for i in idxs:
            e = entries[i]
            vol = volume_from_particles(e['parts'])
            for l in gen_labels:
                cps, rot6 = gens[l][i]
                R = rot6d_to_matrix(torch.from_numpy(
                    np.einsum('ti,ic->tc', B, rot6[0]))).numpy()
                panels.append(dict(
                    base=e['x1'][:, :3], gen_cps=cps, particles=e['parts'], volume=vol,
                    title=f"{l}\n{e['name']} / {e['surface']}  Sinkhorn "
                          f"{rows[l][i]['sinkhorn'].mean():.1e}",
                    bspline_pts=256, bspline_deg=5, gen_R=R, start=e['start']))
        viz_3d.save_grid(panels, os.path.join(a.out_dir, fname), title,
                         max_cols=len(gen_labels))

    rng = np.random.default_rng(a.seed + 1)
    for s in a.splits:
        pool = np.flatnonzero(masks[s])
        if len(pool):
            idx = sorted(rng.choice(pool, min(a.n_examples, len(pool)), replace=False))
            example_fig(idx, f'examples_{s}.png',
                        f'Holdout examples, split {s} (blue: DB teacher, green: generated)')
    if len(gen_labels) >= 2:
        d = M[gen_labels[0]]['sinkhorn'] - M[gen_labels[1]]['sinkhorn']
        order = np.argsort(d)
        example_fig(list(order[:2]), 'examples_best_for_first.png',
                    f'Largest advantage of {gen_labels[0]} over {gen_labels[1]}')
        example_fig(list(order[-2:]), 'examples_worst_for_first.png',
                    f'Largest advantage of {gen_labels[1]} over {gen_labels[0]}')
    print(f"\nplots + tables in {a.out_dir}")


if __name__ == '__main__':
    main()
