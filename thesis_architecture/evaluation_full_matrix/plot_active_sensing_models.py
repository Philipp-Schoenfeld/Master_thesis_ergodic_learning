"""
Combined comparison of the active-sensing runs of `run_job_active_sensing_models.bash`.

Reads results/<tag>_<label>_tsvec_svgd/tables_active_sensing/rounds.csv for every label and writes
results/<tag>/{report.md, summary.json, plots/*.png}.

All planners run the same missions (25 holdout shapes, UCB target, belief ak, 100 TSVEC-SVGD
iterations per round), so shapes are paired: differences are bootstrapped over shapes.
Missions stop once 99 % of the mass is swept, so later rounds have fewer shapes; the comparison
across models therefore uses (a) the energy after the fixed rounds 0..2, which all shapes reach,
and (b) the final energy of the mission.

    python plot_active_sensing_models.py --tag as_models_20261028
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
INK, MUTED = '#1A1A2E', '#555555'

# label -> (display name, colour, marker); greens = U-Net family, purple = DiT, grey = no network
MODELS = {
    'dit630': ('DiT λ=630', '#8E24AA', 's'),
    'unet630': ('U-Net λ=630', '#00C853', 'o'),
    'lm1300': ('U-Net λ=1300 (600 ep.)', '#00692B', 'D'),
    'unet1890': ('U-Net λ=1890', '#2E7D32', 'v'),
    'unet0': ('U-Net, no ErgLoss', '#9CCC65', 'P'),
    'diff630': ('Diffusion U-Net λ=630', '#C2185B', '^'),
    'old500': ('old start model (λ=500)', '#7FDFA3', 'o'),
    'linear': ('linear rays (no network)', '#888888', 'x'),
}
CONDS = [('none_known', 'no prior samples'), ('ten_samples', 'ten known samples')]
FIXED_ROUNDS = [0, 1, 2]
PATH_GRID = [9, 12, 15]       # cumulative path lengths for the equal-length comparison


def load(tag):
    """-> {label: rounds dataframe}; 'linear' is taken from the run that computed it."""
    frames = {}
    for lab in MODELS:
        if lab == 'linear':
            continue
        p = os.path.join(RESULTS, f'{tag}_{lab}_tsvec_svgd', 'tables_active_sensing', 'rounds.csv')
        if not os.path.exists(p):
            print(f'[as] {lab}: {p} missing -- skipped')
            continue
        df = pd.read_csv(p)
        frames[lab] = df[df.method == 'cfm']
        if 'linear' in set(df.method):
            frames['linear'] = df[df.method == 'linear']
    return frames


def final_rows(df):
    """Last round of every (cond, shape)."""
    d = df[df['round'] >= 0].sort_values('round')
    return d.groupby(['cond', 'shape'], as_index=False).last()


def interp_at(df, cond, length):
    """Per shape: E at cumulative path length `length` (NaN where the mission path does not cover it)."""
    d = df[(df.cond == cond) & (df['round'].between(0, 2))]
    res = {}
    for shape, g in d.groupby('shape'):
        g = g.sort_values('round')
        x, y = g.path_len.values, g.E_truth.values
        if len(x) >= 2 and x[0] <= length <= x[-1]:
            res[shape] = float(np.interp(length, x, y))
    return pd.Series(res, dtype=float)


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def style(ax):
    ax.set_facecolor('white')
    ax.grid(alpha=0.2)
    for s in ax.spines.values():
        s.set_color('#cccccc')
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='as_models_20261028')
    ap.add_argument('--ref', default='linear', help='reference model for the paired differences')
    a = ap.parse_args()
    out = os.path.join(RESULTS, a.tag)
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    F = load(a.tag)
    labs = [l for l in MODELS if l in F]
    ref = a.ref if a.ref in F else labs[0]
    summ = dict(models=labs, ref=ref, conds={})

    # ------------------------------------------------------------------ tables
    L = [f'# Active sensing with different start-conditioned planners ({a.tag})', '',
         'UCB target, belief ak, 100 TSVEC-SVGD iterations per round, <= 5 rounds, 25 shapes per cell. '
         'E = ergodic energy against the true density (lower is better); mean ± SEM over shapes; '
         f'Δ = paired difference to **{MODELS[ref][0]}** (95 % bootstrap CI over shapes).', '']
    for cond, cname in CONDS:
        L += [f'## {cname}', '', '| model | E round 0 | E round 1 | E round 2 | E final | Δ final vs ref | '
              'swept mass | path length | reached 99 % | rounds to 99 % |', '|' + '---|' * 10]
        cs = summ['conds'][cond] = {}
        refF = final_rows(F[ref])
        refF = refF[refF.cond == cond].set_index('shape')
        for lab in labs:
            d = F[lab][F[lab].cond == cond]
            fr = final_rows(F[lab])
            fr = fr[fr.cond == cond].set_index('shape')
            per_r = []
            for r in FIXED_ROUNDS:
                v = d[d['round'] == r].set_index('shape').E_truth
                per_r.append((v.mean(), v.std(ddof=1) / np.sqrt(len(v))))
            diff = (fr.E_truth - refF.E_truth.reindex(fr.index)).dropna()
            dm, dlo, dhi = boot(diff) if lab != ref else (0.0, 0.0, 0.0)
            e_final = fr.E_truth
            reached = d.groupby('shape').reached99.max()
            r99 = d[d.reached99 == 1].groupby('shape')['round'].min() + 1
            cs[lab] = dict(E_round=[x[0] for x in per_r], E_round_sem=[x[1] for x in per_r],
                           E_final=float(e_final.mean()), E_final_sem=float(e_final.std(ddof=1) / np.sqrt(len(e_final))),
                           diff_final=[dm, dlo, dhi], swept=float(fr.swept_mass.mean()),
                           path_len=float(fr.path_len.mean()), frac_99=float(reached.mean()),
                           rounds_99=float(r99.mean()) if len(r99) else float('nan'), n=int(len(fr)))
            c = cs[lab]
            L.append(f"| {MODELS[lab][0]} | " + ' | '.join(f'{m:.2f} ± {s:.2f}' for m, s in per_r) +
                     f" | {c['E_final']:.2f} ± {c['E_final_sem']:.2f} | " +
                     ('–' if lab == ref else f'{dm:+.2f} [{dlo:+.2f}, {dhi:+.2f}]') +
                     f" | {c['swept']:.3f} | {c['path_len']:.1f} | {c['frac_99']:.0%} | {c['rounds_99']:.2f} |")
        L.append('')
    # E at equal cumulative path length (linear interpolation over rounds 0..2 per shape; only shapes whose
    # executed path covers the length). Rays travel ~30 % further per round than the networks, so the
    # per-round comparison favours them.
    L += ['## Energy at equal cumulative path length', '',
          'Per shape linear interpolation of E over the cumulative path length (rounds 0-2). n = shapes whose '
          f'path covers the length in both models. Δ = paired difference to **{MODELS[ref][0]}**.', '',
          '| path length | model | E | n | Δ vs ref (95 % CI) |', '|---|---|---|---|---|']
    summ['equal_path'] = {}
    for cond, cname in CONDS:
        for L_ in PATH_GRID:
            curves = {lab: interp_at(F[lab], cond, L_) for lab in labs}
            for lab in labs:
                v = curves[lab].dropna()
                both = pd.concat([curves[lab], curves[ref]], axis=1).dropna()
                dd = both.iloc[:, 0] - both.iloc[:, 1]
                dm, dlo, dhi = boot(dd) if lab != ref and len(dd) > 2 else (0.0, 0.0, 0.0)
                summ['equal_path'].setdefault(cond, {}).setdefault(str(L_), {})[lab] = dict(
                    E=float(v.mean()), n=int(len(v)), diff=[dm, dlo, dhi], n_pair=int(len(dd)))
                L.append(f"| {cname}, L={L_} | {MODELS[lab][0]} | {v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f} | {len(dd)} | " +
                         ('–' if lab == ref else f'{dm:+.2f} [{dlo:+.2f}, {dhi:+.2f}]') + ' |')
    L.append('')
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L))
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print('\n'.join(L))

    # ------------------------------------------------------------------ plot 1: energy vs round
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='white', sharey=True)
    for ax, (cond, cname) in zip(axes, CONDS):
        style(ax)
        for lab in labs:
            name, col, mk = MODELS[lab]
            c = summ['conds'][cond][lab]
            dd = F[lab][(F[lab].cond == cond) & F[lab]['round'].isin(FIXED_ROUNDS)]
            x = dd.groupby('round').path_len.mean().values
            ax.errorbar(x, c['E_round'], yerr=c['E_round_sem'], color=col, marker=mk, ms=6, lw=2 if lab != 'linear' else 1.5,
                        ls='--' if lab == 'linear' else '-', capsize=3, label=name, mec='#444444' if lab == 'old500' else col)
        ax.set_xlabel('mean cumulative path length (markers: end of rounds 1-3, all 25 shapes)', fontsize=9, color=MUTED)
        ax.set_title(cname, fontsize=10.5, color=INK, loc='left')
    axes[0].set_ylabel('ergodic energy vs. true density (mean ± SEM)', fontsize=9, color=MUTED)
    axes[0].legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#ddd', labelcolor=INK)
    fig.suptitle('Energy against travelled path length (100 SVGD iterations per round)', fontsize=11, color=INK, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'plots', 'energy_by_round.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ------------------------------------------------------------------ plot 2: final energy + paired difference
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='white')
    ax = axes[0]
    style(ax)
    w = 0.38
    for k, (cond, cname) in enumerate(CONDS):
        for i, lab in enumerate(labs):
            c = summ['conds'][cond][lab]
            ax.barh(i + (k - 0.5) * w, c['E_final'], w * 0.92, xerr=c['E_final_sem'], capsize=2,
                    color=MODELS[lab][1], alpha=1.0 if k == 0 else 0.55, edgecolor='white', error_kw=dict(ecolor='#444444', lw=1))
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels([MODELS[l][0] for l in labs], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel('final ergodic energy (mean ± SEM; missions end at different path lengths)', fontsize=8.5, color=MUTED)
    ax.set_title('Final energy (solid: no prior samples, pale: ten samples)', fontsize=10, color=INK, loc='left')
    ax = axes[1]
    style(ax)
    others = [l for l in labs if l != ref]
    for k, (cond, cname) in enumerate(CONDS):
        for i, lab in enumerate(others):
            m, lo, hi = summ['equal_path'][cond][str(PATH_GRID[-1])][lab]['diff']
            y = i + (k - 0.5) * 0.3
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], color=MODELS[lab][1], marker=MODELS[lab][2], ms=7, capsize=3, lw=2,
                        mfc=MODELS[lab][1] if k == 0 else 'white')
    ax.axvline(0, color='#888888', lw=1)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels([MODELS[l][0] for l in others], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(f'Δ energy vs. {MODELS[ref][0]} (negative = better)\nfilled: no samples, open: ten samples', fontsize=8.5, color=MUTED)
    ax.set_title(f'Paired difference at equal path length ({PATH_GRID[-1]}), 95 % CI over shapes', fontsize=10, color=INK, loc='left')
    plt.tight_layout()
    fig.savefig(os.path.join(out, 'plots', 'final_energy.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[plot] done ->', os.path.join(out, 'plots'))


if __name__ == '__main__':
    main()
