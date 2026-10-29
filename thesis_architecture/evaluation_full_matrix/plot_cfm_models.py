r"""
plot_cfm_models.py -- figures for the incremental model comparison (eval_start_cfm_models.py)
=============================================================================================
Reads every model stored under results/<out_tag>/models/ (plus the baselines) and writes
results/<out_tag>/plots/*.png. Models added later are picked up automatically; a model without
a finished part (e.g. no warm-start B.npz yet) is simply left out of that figure.

  tradeoff.png          Hellinger distance vs smoothness / boundary violations (mean, 95% CI over densities)
  distribution.png      per-case Hellinger distance as box plots, base densities vs large-area (#unk0) variants
  warm_start.png        solver quality vs Levenberg-Marquardt iterations, by initialisation
  samples.png           diversity of the 30 samples per case, best-of-N vs mean Hellinger, time per call
  gallery.png           same densities, ground truth next to every model's samples

Colour encoding (validated with the dataviz palette check): ground truth blue, U-Net green (light -> dark
with log lambda_erg), DiT purple, chords neutral grey dashed; marker shape (circle U-Net 180 epochs,
diamond U-Net long run, square DiT) and direct labels carry the identity as well.

    python plot_cfm_models.py --out_tag cfm_models_20261028
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from eval_start_cfm_models import RESULTS, _load_models, _boot_ci, _load_npz
from eval_start_cfm_model import C_GT, C_CHORD, _style, load_cases

INK, MUTED = '#1A1A2E', '#555555'
UNET_LIGHT, UNET_DARK = (0x7F, 0xDF, 0xA3), (0x00, 0x69, 0x2B)     # green ramp, anchored at the project green
DIT_LIGHT, DIT_DARK = (0xCE, 0x93, 0xD8), (0x6A, 0x1B, 0x9A)       # purple ramp around #8E24AA
DIFF_LIGHT, DIFF_DARK = (0xF0, 0x80, 0xA8), (0xC2, 0x18, 0x5B)      # pink ramp ending in #C2185B (CVD-validated against green and purple)
GALLERY_DENSITIES = ['greek_upper_0', 'digit_5', 'rand_gmm_20', 'organic_0', 'rand_gmm_complex_0']


def _lerp(a, b, t):
    return '#%02x%02x%02x' % tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def model_styles(models):
    """label -> dict(color, marker, name). Colour = backbone family, shade = log lambda_erg."""
    st = {}
    for fam, (lo, hi), marker in (('unet', (UNET_LIGHT, UNET_DARK), 'o'), ('dit', (DIT_LIGHT, DIT_DARK), 's'),
                                 ('unet-diff', (DIFF_LIGHT, DIFF_DARK), '^')):
        labs = [l for l, m in models.items() if m['meta']['backbone'] == fam]
        lams = np.log([max(models[l]['meta']['lambda_erg'], 1.0) for l in labs]) if labs else []
        for l, ll in zip(labs, lams):
            t = 0.5 if len(labs) == 1 or np.ptp(lams) == 0 else (ll - lams.min()) / np.ptp(lams)
            m = models[l]['meta']
            n_ep = (m['epoch'] or 0) + 1
            long_run = n_ep > 300
            base = {'unet': 'U-Net', 'dit': 'DiT', 'unet-diff': 'Diffusion U-Net'}[fam]
            name = (f"{base} λ={m['lambda_erg']:g}" if m['lambda_erg'] > 0 else f'{base}, no ErgLoss') + (f' ({n_ep} ep.)' if long_run else '')
            st[l] = dict(color=_lerp(lo, hi, t), marker='D' if long_run else marker, name=name)
    return st


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[plot] {name}', flush=True)


def _ax(ax, xlabel=None, ylabel=None, title=None):
    _style(ax)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, loc='left')


def _legend(ax, handles, **kw):
    ax.legend(handles=handles, fontsize=8, frameon=True, facecolor='white', edgecolor='#ddd', labelcolor=INK, **kw)


def _boot_rows(M, n_boot=2000, seed=0):
    """M (n_densities, n_x): mean and 95% bootstrap CI over densities (rows) -> (3, n_x)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(M), (n_boot, len(M)))
    means = M[idx].mean(1)
    return np.stack([M.mean(0), *np.percentile(means, [2.5, 97.5], axis=0)])


# ---------------------------------------------------------------------------------------------

# Label offsets (points) that keep the cluster around lambda=630 readable; unknown labels fall back to a rank pattern.
LABEL_OFFSETS = {
    'unet-l0': (-12, 12, 'right'), 'unet-l210': (-14, 14, 'right'), 'unet-l630': (-18, -26, 'right'),
    'diff-l630': (16, 28, 'left'), 'unet-l1300-600ep': (16, -28, 'left'), 'unet-l1890': (12, 16, 'left'),
    'dit-l630': (12, -14, 'left'),
}


def plot_tradeoff(models, st, base, dens, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor='white')
    gt = {k: float(base['A_ground_truth'][f'm__{k}'].mean()) for k in ('hellinger', 'smooth', 'outside')}
    for ax, (key, xlabel) in zip(axes, [('smooth', 'Smoothness energy (lower = smoother)'),
                                         ('outside', 'Fraction of points outside [0.03, 0.97]')]):
        _ax(ax, xlabel, 'Hellinger distance to target (lower = better)')
        ax.scatter([gt[key]], [gt['hellinger']], marker='*', s=190, color=C_GT, zorder=5, edgecolors='white', linewidths=0.8)
        ax.annotate('Ground truth', (gt[key], gt['hellinger']), xytext=(8, -3), textcoords='offset points',
                    fontsize=8.5, color=INK, va='center')
        xs = {l: m['A'][f'm__{key}'].mean() for l, m in models.items() if 'A' in m}
        order = {l: i for i, l in enumerate(sorted(xs, key=xs.get))}
        for l, m in models.items():
            if 'A' not in m:
                continue
            rank = order[l]
            h = m['A']['m__hellinger'].mean(1)
            x = m['A'][f'm__{key}'].mean(1)
            (hm, hlo, hhi), (xm, xlo, xhi) = _boot_ci(h, dens), _boot_ci(x, dens)
            s = st[l]
            ax.errorbar([xm], [hm], xerr=[[xm - xlo], [xhi - xm]], yerr=[[hm - hlo], [hhi - hm]], fmt='none',
                        ecolor=s['color'], elinewidth=1.4, capsize=3, alpha=0.8, zorder=3)
            ax.scatter([xm], [hm], marker=s['marker'], s=70, color=s['color'], edgecolors='white', linewidths=1.2, zorder=4)
            dx, dy, ha = LABEL_OFFSETS.get(l, [(8, 8, 'left'), (-8, 10, 'right'), (8, -18, 'left'), (8, 8, 'left')][rank % 4])
            ax.annotate(s['name'], (xm, hm), xytext=(dx, dy), textcoords='offset points', fontsize=8.5, color=INK, ha=ha,
                        arrowprops=dict(arrowstyle='-', color='#999999', lw=0.7, shrinkA=0, shrinkB=4))
        ax.margins(x=0.18, y=0.2)
    fig.suptitle('Quality–smoothness trade-off of the generated trajectories (mean over 144 holdout cases, 95% CI over densities)',
                 fontsize=11, color=INK, y=1.02)
    plt.tight_layout()
    _save(fig, out, 'tradeoff.png')


def plot_distribution(models, st, base, cases, out):
    kinds = [('base', 'Base densities (72 cases)'), ('unk0', 'Large-area variants "#unk0" (72 cases)')]
    labs = [l for l, m in models.items() if 'A' in m]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor='white', sharey=True)
    for ax, (kind, title) in zip(axes, kinds):
        _ax(ax, None, 'Hellinger distance to target' if kind == 'base' else None, title)
        mask = np.array([c['kind'] == kind for c in cases])
        data = [base['A_ground_truth']['m__hellinger'].mean(1)[mask]] + \
               [models[l]['A']['m__hellinger'].mean(1)[mask] for l in labs]
        cols = [C_GT] + [st[l]['color'] for l in labs]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showmeans=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color='#999'), capprops=dict(color='#999'),
                        meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor=INK, markersize=5))
        for patch, c in zip(bp['boxes'], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor('#666')
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(['Ground truth'] + [st[l]['name'] for l in labs], fontsize=8.5, color=INK, rotation=30, ha='right',
                           rotation_mode='anchor')
    _legend(axes[0], [Line2D([], [], color=INK, lw=1.6, label='median'),
                      Line2D([], [], marker='D', ls='', markerfacecolor='white', markeredgecolor=INK, label='mean')],
            loc='upper left')
    fig.suptitle('Per-case Hellinger distance (mean over 30 samples; box = quartiles, whiskers = 1.5 IQR)',
                 fontsize=11, color=INK, y=1.02)
    plt.tight_layout()
    _save(fig, out, 'distribution.png')


def plot_warm_start(models, st, base, proto, out):
    its = proto['lm_iters']
    B = {l: m['B'] for l, m in models.items() if 'B' in m}
    if not B or 'B_chord' not in base:
        return
    series = {l: (st[l]['color'], st[l]['marker'], st[l]['name'], '-', b) for l, b in B.items()}
    series['chord'] = (C_CHORD, 'x', 'Chords (no network)', '--', base['B_chord'])
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.4), facecolor='white')
    x = np.arange(len(its))
    for ax, (key, title, log, zoom) in zip(axes, [('hellinger', 'Hellinger distance to target', False, False),
                                                   ('hellinger', 'Hellinger distance (zoom, from 5 iterations)', False, True),
                                                   ('ergodic_k10', 'Fourier error (K=10)', True, False),
                                                   ('smooth', 'Smoothness energy', False, False)]):
        _ax(ax, 'Levenberg–Marquardt iterations', None, title)
        for l, (c, mk, name, ls, b) in series.items():
            M = b[f'm__{key}'].mean(2)                      # (n_densities, n_iters): mean over the initial curves
            mean, lo, hi = _boot_rows(M)
            ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
            ax.plot(x, mean, color=c, ls=ls, lw=2, marker=mk, ms=5.5, mec='white', mew=0.8, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in its], fontsize=8)
        if log:
            ax.set_yscale('log')
        if zoom:
            ax.set_xlim(0.7, len(its) - 0.7)
            lo_y = min(float(b['m__hellinger'].mean(2)[:, 1:].mean(0).min()) for *_, b in series.values())
            hi_y = max(float(b['m__hellinger'].mean(2)[:, 1:].mean(0).max()) for *_, b in series.values())
            ax.set_ylim(lo_y - 0.01, min(hi_y + 0.01, lo_y + 0.16))
    _legend(axes[0], [Line2D([], [], color=c, ls=ls, marker=mk, lw=2, ms=5.5, label=name)
                      for (c, mk, name, ls, _) in series.values()], loc='upper right')
    n = len(next(iter(series.values()))[4]['m__hellinger'])
    fig.suptitle(f'Warm start: solver quality after n iterations, by initialisation '
                 f'(mean over {n} densities × {B and next(iter(B.values()))["m__hellinger"].shape[2]} initial curves, 95% CI over densities)',
                 fontsize=11, color=INK, y=1.02)
    plt.tight_layout()
    _save(fig, out, 'warm_start.png')


def plot_samples(models, st, dens, out):
    labs = [l for l, m in models.items() if 'A' in m]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor='white')
    x = np.arange(len(labs))
    names = [st[l]['name'].replace(' (', '\n(') for l in labs]
    # (a) diversity: mean pairwise distance between the samples of one case
    _ax(axes[0], None, 'mean distance between samples', 'Sample diversity per case')
    for i, l in enumerate(labs):
        X = models[l]['A']['curves'][:, :10].astype(np.float64)               # (cases, 10, T, 2)
        d = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1).mean(-1)      # (cases, 10, 10)
        iu = np.triu_indices(10, 1)
        per_case = d[:, iu[0], iu[1]].mean(1)
        m, lo, hi = _boot_ci(per_case, dens)
        axes[0].bar(i, m, width=0.55, color=st[l]['color'])
        axes[0].errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=3, lw=1.2)
    # (b) best-of-N vs mean
    _ax(axes[1], None, 'Hellinger distance to target', 'Best of N samples vs average sample')
    n = models[labs[0]]['A']['m__hellinger'].shape[1]
    for i, l in enumerate(labs):
        H = models[l]['A']['m__hellinger']
        axes[1].bar(i - 0.2, H.mean(), width=0.36, color=st[l]['color'], alpha=0.45)
        axes[1].bar(i + 0.2, H.min(1).mean(), width=0.36, color=st[l]['color'])
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.22)
    _legend(axes[1], [Patch(facecolor='#999', alpha=0.45, label='average sample'),
                      Patch(facecolor='#999', label=f'best of {n}')], loc='upper right')
    # (c) generation time per sample
    gpus = {l: models[l]['meta']['gpu'].replace('NVIDIA ', '').replace('GeForce ', '') for l in labs}
    mixed = len(set(gpus.values())) > 1
    _ax(axes[2], None, 'milliseconds per sample',
        'Generation time (hatched: RTX 3090, solid: RTX A5000)' if mixed else f'Generation time ({next(iter(gpus.values()))})')
    for i, l in enumerate(labs):
        t = float(np.median(models[l]['A']['t_sample'])) / n * 1000
        axes[2].bar(i, t, width=0.55, color=st[l]['color'], hatch='//' if 'A5000' not in gpus[l] and mixed else None,
                    edgecolor='white' if 'A5000' not in gpus[l] and mixed else None, linewidth=0)
        axes[2].annotate(f'{t:.0f}', (i, t), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8.5, color=INK)
    flat = [st[l]['name'] for l in labs]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(flat, fontsize=8, color=INK, rotation=30, ha='right', rotation_mode='anchor')
    fig.suptitle('Diversity, best-of-N and cost of the generated samples (100 Euler steps)', fontsize=11, color=INK, y=1.02)
    plt.tight_layout()
    _save(fig, out, 'samples.png')


def plot_gallery(models, st, base, proto, db, out):
    from analyze_teacher_study import WHITE_INFERNO
    import pipeline as pl
    labs = [l for l, m in models.items() if 'A' in m]
    names = [c['name'] for c in proto['cases']]
    cases = {c['name']: c for c in load_cases(db, None, proto['seed'])}
    rows = []
    for d in GALLERY_DENSITIES:
        hit = [c for c in proto['cases'] if c['density'] == d and c['start'] == 'stored']
        if hit:
            rows.append(hit[0]['name'])
    if not rows:
        return
    ncol = 1 + len(labs)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.9 * ncol, 2.9 * len(rows)), facecolor='white', squeeze=False)
    for i, name in enumerate(rows):
        c = cases[name]
        ci = names.index(name)
        for j in range(ncol):
            ax = axes[i][j]
            ax.imshow(c['sd'].density, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO, vmin=0, vmax=1,
                      alpha=0.55, aspect='auto')
            if j == 0:
                g = base['A_ground_truth']['curves'][ci, 0]
                ax.plot(g[:, 0], g[:, 1], color=C_GT, lw=2.5, alpha=0.9)
                title = 'Ground truth'
            else:
                X = models[labs[j - 1]]['A']['curves'][ci]
                for k in range(5):
                    ax.plot(X[k, :, 0], X[k, :, 1], color='#00C853', lw=2.2 if k == 0 else 1.2,
                            alpha=0.95 if k == 0 else 0.3)
                h = models[labs[j - 1]]['A']['m__hellinger'][ci].mean()
                title = f'{st[labs[j - 1]]["name"]}  (H = {h:.2f})'
            ax.scatter([c['sd'].x0[0]], [c['sd'].x0[1]], color=C_GT, marker='*', s=110, edgecolors='white',
                       linewidths=0.6, zorder=4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('#ccc')
            ax.set_title(title if i == 0 else (f'H = {h:.2f}' if j else ''), fontsize=8.5, color=INK)
            if j == 0:
                ax.set_ylabel(c['density'], fontsize=9, color=INK)
    fig.suptitle('Generated trajectories on holdout densities (stored start ★; first sample bold, 4 more faint; H = mean Hellinger distance)',
                 fontsize=10.5, color=INK, y=1.0)
    plt.tight_layout()
    _save(fig, out, 'gallery.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--db', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                'ergodic_dataset_generator', 'ergodic_dataset_lm_nxi25.db'))
    ap.add_argument('--no_gallery', action='store_true', help='skip the gallery (needs the DB for the density grids)')
    a = ap.parse_args()
    out = os.path.join(RESULTS, a.out_tag)
    proto = json.load(open(os.path.join(out, 'protocol.json')))
    models = _load_models(out)
    base = {k: _load_npz(os.path.join(out, 'baselines', f'{k}.npz')) for k in ('A_ground_truth', 'A_chord', 'B_chord')
            if os.path.exists(os.path.join(out, 'baselines', f'{k}.npz'))}
    st = model_styles(models)
    dens = [c['density'] for c in proto['cases']]
    plots = os.path.join(out, 'plots')
    print(f'[plot] {len(models)} models: {list(models)}', flush=True)
    plot_tradeoff(models, st, base, dens, plots)
    plot_distribution(models, st, base, proto['cases'], plots)
    plot_warm_start(models, st, base, proto, plots)
    plot_samples(models, st, dens, plots)
    if not a.no_gallery:
        plot_gallery(models, st, base, proto, a.db, plots)


if __name__ == '__main__':
    main()
