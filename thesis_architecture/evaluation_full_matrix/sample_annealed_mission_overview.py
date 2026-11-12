#!/usr/bin/env python3
r"""
sample_annealed_mission_overview.py
====================================
Overview of the five-round annealed replanning mini-pipeline of
`sample_annealed_mission.py` (CFM init + TSVEC-SVGD refine, kappa 1 -> 0.05, full
execution, zero prior knowledge) for a fixed set of setups, each drawn as one row
of round panels PLUS a metrics table.

Three groups (the same shape and seed everywhere, so rows differ only by the setup):

  kernels   U-Net lambda=210 with the Attentive-Kernel belief vs. the RBF belief
            (cloud cutoff = mid value for both)
  cutoffs   RBF belief, U-Net lambda=210, particle-cloud cutoff 0.1 / 0.3 / 0.75
  cutoffs_ak  the same sweep with the Attentive-Kernel belief
  final_rbf / final_ak
            the closing comparison, once per belief kernel: U-Net lambda=630, DiT lambda=630,
            a random walk (no network, no SVGD) and a lawnmower. Random walk and lawnmower get
            the same path budget per round as the SVGD-refined networks (--walk_len, default
            6.1 requested, ~5.9 after resampling = the networks' mean) and the same belief
            update / measurements, but ignore the belief.
            The lawnmower does one coarse sweep of the whole workspace per round (about every
            second lane of a dense lawnmower, lane count from common.baselines.lanes_for_length);
            the next round sweeps back in the opposite direction with the lane grid shifted into
            the gaps (see lawnmower_round_fn). It starts in the corner (0.06, 0.06).
  inits     AK belief, mid cutoff, different initialisers: U-Net without ErgLoss,
            U-Net lambda=1890, DiT lambda=630, diffusion U-Net lambda=630
            (+ the U-Net lambda=210 run of the kernels group as the reference row)

Cloud cutoff = the GUI's "Cloud Cutoff" slider (0 ... 0.5): the network's particle cloud
only holds particles where Phi >= cutoff * max(Phi); the refiner's target stays the full
Phi. Mid = 0.20 (kernels and inits groups); the cutoffs group sweeps 0.1 / 0.3 / 0.75.

Metrics per round (of the whole path driven so far, against the GROUND TRUTH the
mission never sees, except the two E_target rows):
  E_truth        ergodic error vs. the true density (K=10, solver weighting) -- lower is better
  swept mass     share of the true mass within the coverage radius 0.06 of the path
  coverage dist  density-weighted mean distance to the path (lower is better)
  path length    of the driven path
  belief RMSE / corr   belief mean vs. truth after the round's measurements
  E_target raw   best raw network candidate vs. the round's own target (init quality)
  E_target SVGD  best candidate after the TSVEC-SVGD refinement

Results are cached per setup (cache/*.pt in the output folder), so re-plotting or
adding a setup never recomputes finished ones.

    python sample_annealed_mission_overview.py                  # everything
    python sample_annealed_mission_overview.py --only ak_unet210_mid,rbf_unet210_mid
    python sample_annealed_mission_overview.py --smoke          # 2 rounds, tiny budget
    python sample_annealed_mission_overview.py --replot         # cached setups only
"""

import argparse
import copy
import csv
import glob
import os
import re
import sys
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

import sample_annealed_mission as sam                                   # noqa: E402
from common import net_zoo                                             # noqa: E402
from common.acquisition import kappa_schedule                          # noqa: E402
from common.baselines import lanes_for_length                          # noqa: E402
from common.data import load_truth                                     # noqa: E402
from common.metrics import coverage_vs_truth, path_length, trim_to_length   # noqa: E402
from metrics_explore_exploit import ExploreExploitErgodic, swept_mass_fraction   # noqa: E402
from run_mission_eval import belief_quality                            # noqa: E402
from run_ideal_matrix import PLANNER_BUILDERS                          # noqa: E402
from svgd_batched import BatchedTsvecSvgdTorch                         # noqa: E402
import run_svgd_convergence as rsc                                     # noqa: E402
import variant_runner as vr                                            # noqa: E402
from init_baselines import random_walk_path                            # noqa: E402
from exploration_optimierung.mission import resample_arclength         # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
TEXT, MUTED, SPINE = '#1A1A2E', '#555555', '#CCCCCC'
COVERAGE_RADIUS = 0.06

CUT_LOW, CUT_MID, CUT_HIGH = 0.05, 0.20, 0.40
# cutoffs group: own values (the *_mid setups of the other groups stay at CUT_MID)
SWEEP_CUTS = (0.1, 0.3, 0.75)

# net key -> (label, regex on the checkpoint file name; searched in transfer/ and its dated sub-folders)
NETS = {
    'unet210': ('U-Net λ=210', r'^(?!.*DIT)(?!.*DIFF).*LAMERG25'),
    'unet0': ('U-Net, no ErgLoss', r'LAMERG0_final'),
    'unet630': ('U-Net λ=630', r'^(?!.*DIT)(?!.*DIFF).*LAMERG50'),
    'unet1890': ('U-Net λ=1890', r'LAMERG75'),
    'dit630': ('DiT λ=630', r'DIT-d12-h6'),
    'diff630': ('Diffusion U-Net λ=630', r'DIFF-v'),
    # model-free baselines: no checkpoint, no refinement
    'rw': ('Random walk', None),
    'lawn': ('Lawnmower', None),
    # network-free initialiser, refined by the Adam-SVGD backend (only run by
    # run_exploration_comparison.py, not by this script's main())
    'linadam': ('Linear init + Adam-SVGD', None),
    # added 2026-11-11 for the extended exploration comparison (run_exploration_comparison.py)
    'old500': ('U-Net λ=500 (older start model)', r'^netz2d_startpunkt\.pt$'),
    'lm1300': ('U-Net λ=1300 (600 ep.)', r'LMTEACHER|netz2d_startpunkt_lm_inpaint'),
    'selfsup': ('Self-supervised generator', None),     # own checkpoint, see selfsup_planner.py
    'lintsvec': ('Linear init + TSVEC-SVGD', None),     # ergodic search WITHOUT a learned prior
    'gpgoal': ('Line to GP-sampled point', None),       # model-free, no SVGD
    'linray': ('Best straight ray', None),              # model-free, no SVGD
}
BASELINES = ('rw', 'lawn', 'gpgoal', 'linray')
#: straight-ray initialised, refined by an SVGD backend (no network)
RAY_REFINED = ('linadam', 'lintsvec')
#: nets without a start-conditioned checkpoint in `transfer/` (own loader)
OWN_LOADER = ('selfsup',)
#: setups only run by run_exploration_comparison.py (this script's default set excludes them)
COMPARISON_ONLY = RAY_REFINED + ('gpgoal', 'linray', 'selfsup', 'old500', 'lm1300')


@dataclass(frozen=True)
class Setup:
    key: str
    kernel: str       # 'ak' or 'rbf'
    net: str          # key of NETS
    cutoff: float
    kappa: float = None   # constant kappa for every round; None = the annealed schedule

    @property
    def is_baseline(self):
        return self.net in BASELINES

    @property
    def label(self):
        kern = {'ak': 'Attentive-Kernel belief', 'rbf': 'RBF belief'}[self.kernel]
        if self.is_baseline:
            return f'{NETS[self.net][0]}\n{kern}\nno SVGD'
        if self.net == 'linadam':
            return f'{NETS[self.net][0]}\n{kern}\nstraight rays, 8 headings'
        cut = f'cloud cutoff {self.cutoff:g}' if self.cutoff > 0 else 'no cloud cutoff'
        kap = '' if self.kappa is None else f'\nconstant κ = {self.kappa:g}'
        return f'{NETS[self.net][0]}\n{kern}\n{cut}{kap}'


SETUPS = {s.key: s for s in (
    Setup('ak_unet210_mid', 'ak', 'unet210', CUT_MID),
    Setup('rbf_unet210_mid', 'rbf', 'unet210', CUT_MID),
    Setup('rbf_unet210_low', 'rbf', 'unet210', CUT_LOW),
    Setup('rbf_unet210_high', 'rbf', 'unet210', CUT_HIGH),
    *(Setup(f'rbf_unet210_cut{c:g}', 'rbf', 'unet210', c) for c in SWEEP_CUTS),
    *(Setup(f'ak_unet210_cut{c:g}', 'ak', 'unet210', c) for c in SWEEP_CUTS),
    Setup('ak_unet0_mid', 'ak', 'unet0', CUT_MID),
    Setup('ak_unet1890_mid', 'ak', 'unet1890', CUT_MID),
    Setup('ak_dit630_mid', 'ak', 'dit630', CUT_MID),
    Setup('ak_diff630_mid', 'ak', 'diff630', CUT_MID),
    *(Setup(f'{k}_{n}', k, n, CUT_MID)
      for k in ('rbf', 'ak') for n in ('rw', 'lawn', 'linadam', 'lintsvec', 'gpgoal', 'linray')),
    *(Setup(f'{k}_{n}_mid', k, n, CUT_MID)
      for k in ('rbf', 'ak') for n in ('selfsup', 'old500', 'lm1300', 'unet0', 'unet1890')),
    *(Setup(f'{k}_unet630_mid', k, 'unet630', CUT_MID) for k in ('rbf', 'ak')),
    Setup('rbf_dit630_mid', 'rbf', 'dit630', CUT_MID),
    Setup('ak_unet210_nocut', 'ak', 'unet210', 0.0),
    Setup('rbf_unet210_nocut', 'rbf', 'unet210', 0.0),
    Setup('ak_unet210_nocut_k0.5', 'ak', 'unet210', 0.0, kappa=0.5),
    Setup('rbf_unet210_nocut_k0.5', 'rbf', 'unet210', 0.0, kappa=0.5),
)}

GROUPS = {
    'kernels': ('Belief kernel: Attentive Kernel vs. RBF (U-Net λ=210, cloud cutoff %g)' % CUT_MID,
                ['ak_unet210_mid', 'rbf_unet210_mid']),
    'kernels_nocutoff': ('Belief kernel: Attentive Kernel vs. RBF (U-Net λ=210, no cloud cutoff)',
                         ['ak_unet210_nocut', 'rbf_unet210_nocut']),
    'kernels_nocutoff_kappa0.5': ('Belief kernel: Attentive Kernel vs. RBF (U-Net λ=210, no cloud '
                                  'cutoff, constant κ = 0.5)',
                                  ['ak_unet210_nocut_k0.5', 'rbf_unet210_nocut_k0.5']),
    'cutoffs': ('Particle-cloud width: RBF belief, U-Net λ=210, cloud cutoff %g / %g / %g'
                % SWEEP_CUTS,
                [f'rbf_unet210_cut{c:g}' for c in SWEEP_CUTS]),
    'cutoffs_ak': ('Particle-cloud width: Attentive-Kernel belief, U-Net λ=210, cloud cutoff %g / %g / %g'
                   % SWEEP_CUTS,
                   [f'ak_unet210_cut{c:g}' for c in SWEEP_CUTS]),
    'final_rbf': ('Final comparison: RBF belief (cloud cutoff %g)' % CUT_MID,
                  ['rbf_unet630_mid', 'rbf_dit630_mid', 'rbf_rw', 'rbf_lawn']),
    'final_ak': ('Final comparison: Attentive-Kernel belief (cloud cutoff %g)' % CUT_MID,
                 ['ak_unet630_mid', 'ak_dit630_mid', 'ak_rw', 'ak_lawn']),
    'inits': ('Initialiser: Attentive-Kernel belief, cloud cutoff %g' % CUT_MID,
              ['ak_unet0_mid', 'ak_unet210_mid', 'ak_unet1890_mid', 'ak_dit630_mid',
               'ak_diff630_mid']),
}

# table rows: (result key, label, format)
TABLE_ROWS = (
    ('E_truth', 'E_truth ↓', '{:.2f}'),
    ('E_belief', 'E_belief ↓', '{:.2f}'),
    ('swept', 'swept mass ↑', '{:.0%}'),
    ('cov', 'coverage dist ↓', '{:.3f}'),
    ('path_len', 'path length', '{:.1f}'),
    ('out_gt_frac', 'path outside GT ↓', '{:.0%}'),
    ('in_hi50_frac', 'path in dense GT ↑', '{:.0%}'),
    ('belief_rmse', 'belief RMSE ↓', '{:.3f}'),
    ('belief_corr', 'belief corr ↑', '{:.2f}'),
    ('E_init_best', 'E_target raw net ↓', '{:.2f}'),
    ('E_ref_best', 'E_target after SVGD ↓', '{:.2f}'),
)


def find_checkpoint(net):
    pat = re.compile(NETS[net][1])
    files = glob.glob(os.path.join(net_zoo.TRANSFER, '*.pt')) + \
        glob.glob(os.path.join(net_zoo.TRANSFER, '*', '*.pt'))
    # the 3D surface models (`*flow3d*`, `lm3d_*`, `nooffset*`) sit in transfer/ too and match the
    # same architecture patterns (e.g. 'DIT-d12-h6'); they are not 2D planners
    hits = [f for f in files if pat.search(os.path.basename(f))
            and 'flow3d' not in os.path.basename(f)
            and os.path.basename(os.path.dirname(f)) not in ('old_models', 'gross')]
    if not hits:
        raise FileNotFoundError(f"no checkpoint for '{net}' (pattern {NETS[net][1]!r}) "
                                f"under {net_zoo.TRANSFER}")
    return max(hits, key=os.path.getmtime)


# ===========================================================================
# One setup = one mission with per-round metrics
# ===========================================================================

def baseline_path_fn(setup, a, seed):
    """Per-round path source of a model-free baseline, `walk_len` long per round."""
    if setup.net == 'rw':
        def fn(r, start, phi):
            walk = random_walk_path(n_points=2048, seed=seed * 1000 + r, start=start.detach().cpu())
            return trim_to_length(walk, a.walk_len)
        return fn
    return lawnmower_round_fn(a.walk_len, a.n_rounds)


#: lane-grid offsets (in lane spacings) of consecutive sweeps: 0, 1/2, 1/4, 3/4, 1/8, ... so
#: every sweep falls into the largest gap left by the earlier ones
SWEEP_OFFSETS = (0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875)


def lawnmower_round_fn(sweep_len, n_rounds, margin=0.06):
    """One round = one sweep of the whole workspace at coarse lane spacing (~`sweep_len` long),
    i.e. the sweep covers only about every second lane of a dense lawnmower. The next round
    sweeps back in the opposite direction (bottom->top, top->bottom, ...), its lane grid shifted
    by SWEEP_OFFSETS so it fills the gaps of the previous sweeps. Each sweep begins on the side
    where the previous one ended."""
    n_lanes = lanes_for_length(sweep_len, margin)
    spacing = (1.0 - 2 * margin) / n_lanes
    sweeps, x_end = [], 0                 # x_end: 0 = left, 1 = right side where the last sweep ended
    for k in range(n_rounds):
        ys = [margin + (i + SWEEP_OFFSETS[k % len(SWEEP_OFFSETS)]) * spacing for i in range(n_lanes)]
        if k % 2:
            ys = ys[::-1]                 # odd sweeps run top -> bottom
        pts, side = [], x_end
        for y in ys:
            x0, x1 = (margin, 1.0 - margin) if side == 0 else (1.0 - margin, margin)
            pts += [(x0, y), (x1, y)]
            side = 1 - side
        x_end = 0 if pts[-1][0] < 0.5 else 1
        sweeps.append(torch.tensor(pts, dtype=torch.float32))

    def fn(r, start, phi):
        # later sweeps first move from where the last one ended to their first lane, instead of
        # having their first lane point snapped onto the agent (which draws a slanted line)
        return sweeps[r] if r == 0 else torch.cat([start.detach().cpu().float()[None], sweeps[r]])
    return fn


def run_setup(setup, planner, truth, ee, bs, B32, base_args, a, kappas, device, seed):
    args = copy.copy(base_args)
    args.belief_kernel, args.ak_fit_iters, args.np_ckpt = setup.kernel, a.ak_fit_iters, None
    phi_k_truth = ee.target_coeffs(truth)
    rows = []

    def on_round(r, belief, segments, diag):
        driven = torch.cat(segments, dim=0)
        mu, sd = belief.posterior_grid()
        belief._cache = None
        bq = belief_quality(mu.float(), sd.float(), truth)
        rows.append(dict(
            round=r, kappa=kappas[r],
            E_truth=ee.score(driven, phi_k_truth)['E_ergodic_total'],
            swept=swept_mass_fraction(driven, truth, sensor_radius=COVERAGE_RADIUS),
            cov=float(coverage_vs_truth(driven, truth)), path_len=path_length(driven),
            belief_rmse=bq['rmse'], belief_corr=bq['corr'], **diag))

    t0 = time.time()
    path_fn = baseline_path_fn(setup, a, seed) if setup.is_baseline else None
    start_pos = (0.06, 0.06) if setup.net == 'lawn' else (0.5, 0.5)
    segments, phis, infos = sam.run_mission(
        setup.key, truth, planner, bs, ee, B32, args, device, a.n_rounds, a.n_init,
        a.n_iters, kappas, start_pos, seed, cutoff=setup.cutoff, on_round=on_round,
        path_fn=path_fn)
    return dict(setup=setup.key, segments=segments, phis=phis, rows=rows,
                kappas=kappas, seconds=time.time() - t0)


# ===========================================================================
# Plot
# ===========================================================================

def metrics_table(ax, res):
    ax.axis('off')
    n = len(res['rows'])
    cells = [['–' if row.get(key, float('nan')) != row.get(key, float('nan'))
              else fmt.format(row[key]) for row in res['rows']]
             for key, _, fmt in TABLE_ROWS]
    tab = ax.table(cellText=cells, rowLabels=[lab for _, lab, _ in TABLE_ROWS],
                   colLabels=[f'R{r}' for r in range(n)], cellLoc='center', loc='center',
                   bbox=[0.30, 0.08, 0.70, 0.80])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.6)
        cell.get_text().set_color(TEXT)
        if i == 0:
            cell.set_facecolor('#F3F3F6')
        if j == n - 1 and i > 0:                  # final value of the mission
            cell.get_text().set_fontweight('bold')
        if j == -1:
            cell._loc = 'right'
            cell.set_linewidth(0)
    ax.set_title('Metrics per round (last = final)', fontsize=10, color=TEXT)


def plot_rows(results, title, out, shape):
    n_rows, n_rounds = len(results), len(results[0]['segments'])
    cmap = sam.white_inferno()
    colors = sam.round_colors(n_rounds)
    fig, axes = plt.subplots(n_rows, n_rounds + 2, squeeze=False, facecolor='white',
                             figsize=(3.4 * (n_rounds + 1) + 5.6, 3.7 * n_rows),
                             gridspec_kw=dict(width_ratios=[1] * (n_rounds + 1) + [1.65]))
    truth = results[0]['truth']
    for i, res in enumerate(results):
        s = SETUPS[res['setup']]
        for r in range(n_rounds):
            sam.plot_round(axes[i, r], res['phis'][r], res['segments'][r], res['kappas'][r],
                           colors[r], row_label=s.label if r == 0 else None,
                           col_title=f'Round {r}' if i == 0 else None, cmap=cmap)
        sam.plot_summary(axes[i, n_rounds], truth, res['segments'], colors, res['kappas'],
                         col_title='All rounds (ground truth)' if i == 0 else None, cmap=cmap)
        metrics_table(axes[i, n_rounds + 1], res)
    kap = results[0]['kappas']
    kap_txt = (f'constant κ = {kap[0]:g}' if len(set(kap)) == 1 else
               'κ annealed ' + ' → '.join(f'{k:g}' for k in kap))
    nets = {SETUPS[r['setup']].net for r in results}
    parts = []
    if nets - set(BASELINES) - set(RAY_REFINED):
        parts.append('network init + TSVEC-SVGD refine')
    if 'linadam' in nets:
        parts.append('straight-ray init + Adam-SVGD refine')
    if 'lintsvec' in nets:
        parts.append('straight-ray init + TSVEC-SVGD refine')
    if nets & set(BASELINES):
        parts.append('random walk / lawnmower / GP-goal line / best ray: no network, no SVGD')
    method = '; '.join(parts)
    fig.suptitle(f'{title}\nshape {shape}; {method}, {kap_txt}, '
                 'full execution per round, zero prior knowledge', fontsize=12.5, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.95 if n_rows > 1 else 0.90])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ===========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shape_names', type=str, default='A',
                   help='Heldout (val-split) shape of the mission; same as sample_annealed_mission.py.')
    p.add_argument('--n_rounds', type=int, default=5)
    p.add_argument('--n_init', type=int, default=8, help='Candidates per round.')
    p.add_argument('--n_iters', type=int, default=500, help='TSVEC-SVGD iterations per round.')
    p.add_argument('--kappa0', type=float, default=1.0)
    p.add_argument('--kappa1', type=float, default=0.05)
    p.add_argument('--walk_len', type=float, default=6.1,
                   help='Path length per round of the random-walk / lawnmower baselines '
                        '(default: mean per-round length of the SVGD-refined networks).')
    p.add_argument('--ak_fit_iters', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--only', type=str, default=None,
                   help='Comma-separated setup keys to run (default: all). Keys: ' + ', '.join(SETUPS))
    p.add_argument('--replot', action='store_true', help='Plot from the cache only, compute nothing.')
    p.add_argument('--smoke', action='store_true',
                   help='Tiny budget (2 rounds, 2 candidates, 20 iterations), separate folder.')
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.smoke:
        a.n_rounds, a.n_init, a.n_iters = 2, 2, 20
    out_dir = a.out_dir or os.path.join(_here, 'plots', 'overview_annealed_mission' +
                                        ('_smoke' if a.smoke else ''))
    cache_dir = os.path.join(out_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    device = a.device
    keys = a.only.split(',') if a.only else [k for k in SETUPS if SETUPS[k].net not in COMPARISON_ONLY]
    unknown = [k for k in keys if k not in SETUPS]
    if unknown:
        raise SystemExit(f'unknown setup keys {unknown}; known: {list(SETUPS)}')

    names, truths = load_truth(labels=[a.shape_names], n=1, split='val',
                               resolution=rsc.TRUTH_RES, device=device)
    shape, truth = names[0], truths[0].detach().cpu()
    kappas = [kappa_schedule(r, a.n_rounds, a.kappa0, a.kappa1) for r in range(a.n_rounds)]
    tag = f'{shape}_s{a.seed}_r{a.n_rounds}_i{a.n_iters}_n{a.n_init}'
    cache_path = lambda k: os.path.join(cache_dir, f'{k}__{tag}.pt')   # noqa: E731

    # ---- compute (grouped by network so each checkpoint is loaded once) ----
    todo = [k for k in keys if not os.path.exists(cache_path(k))]
    if todo and not a.replot:
        missing = []
        for net in {SETUPS[k].net for k in todo} - set(BASELINES):
            try:
                find_checkpoint(net)
            except FileNotFoundError as e:
                missing.append(str(e))
        if missing:
            raise SystemExit('\n'.join(missing))
        base_args, _, _ = vr.build_strategy_args(sam.STRATEGY_NAME, device)
        ee = ExploreExploitErgodic(device=device)
        B_np = rsc.basis_matrix(rsc.NXI, rsc.N_POINTS, rsc.DEGREE)
        bs = BatchedTsvecSvgdTorch(B_np, device)
        B32 = torch.as_tensor(B_np, dtype=torch.float32, device=device)
        print(f'\n=== overview: shape {shape}, {len(todo)} setups to run, device={device}, '
              f'n_init={a.n_init}, n_iters={a.n_iters}, rounds={a.n_rounds} ===')
        for k in [k for k in todo if SETUPS[k].is_baseline]:
            print(f'  --- setup {k} (baseline, walk_len={a.walk_len:g}/round) ---')
            k_sched = kappas if SETUPS[k].kappa is None else [SETUPS[k].kappa] * a.n_rounds
            res = run_setup(SETUPS[k], None, truth, ee, None, None, base_args, a, k_sched,
                            device, a.seed)
            torch.save(res, cache_path(k))
            print(f'  {k}: {res["seconds"] / 60:.1f} min, final E_truth='
                  f'{res["rows"][-1]["E_truth"]:.2f}, path length='
                  f'{res["rows"][-1]["path_len"]:.1f}')
        for net in sorted({SETUPS[k].net for k in todo} - set(BASELINES)):
            ckpt = find_checkpoint(net)
            print(f'\n[net {net}] {os.path.basename(ckpt)}')
            planner = PLANNER_BUILDERS['particles'](ckpt, device)
            for k in [k for k in todo if SETUPS[k].net == net]:
                print(f'  --- setup {k} ---')
                k_sched = kappas if SETUPS[k].kappa is None else [SETUPS[k].kappa] * a.n_rounds
                res = run_setup(SETUPS[k], planner, truth, ee, bs, B32, base_args, a, k_sched,
                                device, a.seed)
                torch.save(res, cache_path(k))
                print(f'  {k}: {res["seconds"] / 60:.1f} min, final E_truth='
                      f'{res["rows"][-1]["E_truth"]:.2f}')
            del planner
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

    # ---- plots + metrics csv ----
    loaded = {k: torch.load(cache_path(k), weights_only=False)
              for k in SETUPS if os.path.exists(cache_path(k))}
    for res in loaded.values():
        res['truth'] = truth
    print('\n=== plots ===')
    for g, (title, g_keys) in GROUPS.items():
        have = [loaded[k] for k in g_keys if k in loaded]
        if have:
            plot_rows(have, title, os.path.join(out_dir, f'overview_{g}.png'), shape)
    for k, res in loaded.items():
        plot_rows([res], SETUPS[k].label.replace('\n', ', '),
                  os.path.join(out_dir, 'setups', f'{k}.png'), shape)
    csv_path = os.path.join(out_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        cols = ['round', 'kappa'] + [k for k, _, _ in TABLE_ROWS]
        w.writerow(['setup', 'kernel', 'net', 'cutoff', 'shape'] + cols)
        for k, res in loaded.items():
            s = SETUPS[k]
            for row in res['rows']:
                w.writerow([k, s.kernel, s.net, s.cutoff, shape] + [row.get(c, float('nan')) for c in cols])
    print(f'  Saved: {csv_path}')


if __name__ == '__main__':
    main()
