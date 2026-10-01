r"""
analyze.py -- intensive analysis of the partial paired-svgd run
==================================================================
Run with the local environment (pandas/matplotlib/numpy only, no torch/jax
needed -- this reads the already-computed `all_runs.csv`, it doesn't
generate any new trajectories).

Scope caveat (read before trusting any number here): this is
`best_of_30_paired_20260918` as it stood after both chained 24h jobs timed
out -- 8 of 25 holdout shapes fully done (A, a_lc, digit_5, greek_upper_0,
korean_5, rand_ana_poly_10, rand_gmm_10, rand_gmm_20), a 9th
(rand_ana_poly_20) partially done and EXCLUDED here for consistency. No
further jobs have run since 2026-09-20. Every table/plot below is scoped to
these 8 shapes and should be read as "early signal", not a final result.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'best_of_30_paired_20260918', 'tables', 'all_runs.csv')
TABLES = os.path.join(HERE, 'tables')
PLOTS = os.path.join(HERE, 'plots')
os.makedirs(TABLES, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

# ---- dataviz reference palette (light mode) --------------------------------
SURFACE = '#fcfcfb'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
CAT = {
    'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100',
    'magenta': '#e87ba4', 'green': '#008300', 'violet': '#4a3aa7', 'red': '#e34948',
}
STATUS_GOOD = '#0ca30c'
STATUS_CRITICAL = '#d03b3b'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK_SECONDARY,
    'text.color': INK_PRIMARY, 'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED,
    'grid.color': GRID, 'font.size': 11, 'axes.titlesize': 13, 'axes.titlecolor': INK_PRIMARY,
    'axes.titleweight': 'bold', 'savefig.facecolor': SURFACE, 'savefig.dpi': 150,
})

ONLY_FULLY_DONE_SHAPES = ['A', 'a_lc', 'digit_5', 'greek_upper_0', 'korean_5',
                          'rand_ana_poly_10', 'rand_gmm_10', 'rand_gmm_20']

MIN_BETTER = {'E_ergodic_total', 'E_ergodic_explore', 'E_ergodic_exploit',
             'E_ergodic_total_per_length', 'coverage', 'coverage_per_length',
             'J', 'smoothness_energy', 'steps_to_full_coverage'}
MAX_BETTER = {'steps_to_full_coverage_reached'}

FAMILY_COLOR = {
    'niveau': CAT['blue'], 'lse': CAT['blue'],
    'ucb': CAT['orange'],
    'mass': CAT['aqua'],
    'eid_tuned': CAT['magenta'], 'eid': CAT['magenta'],
    'eid_optuna_ideal_v2': CAT['violet'],
}
METHOD_COLOR = {
    'CFM (particles)': CAT['blue'], 'CFM (spectral)': CAT['violet'],
    'heuristic_tuned': CAT['orange'], 'linear_waypoints_tuned': CAT['yellow'],
    'random_walk': CAT['red'], 'lawnmower': CAT['aqua'],
}


def style_ax(ax, grid_axis='y'):
    ax.grid(axis=grid_axis, alpha=1.0, linewidth=0.8, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(length=0)


def bar_labels(ax, bars, fmt='{:.2f}'):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                   ha='center', va='bottom' if h >= 0 else 'top',
                   fontsize=9, color=INK_SECONDARY,
                   xytext=(0, 3 if h >= 0 else -3), textcoords='offset points')


print("Loading all_runs.csv ...")
df = pd.read_csv(SRC)
n_before = len(df)
df = df[df['shape'].isin(ONLY_FULLY_DONE_SHAPES)].copy()
print(f"{n_before} rows total -> {len(df)} rows after restricting to the "
     f"{len(ONLY_FULLY_DONE_SHAPES)} fully-completed shapes")

# Family/svgd_iters parsing for CFM rows (particles/spectral): svgd_iters is
# recorded as -1 for these in the raw CSV (the project's existing convention
# -- sub-variant name carries it, not the svgd_iters column), so parse it
# from `strategy` instead.
FAMILY_RE = {
    'niveau_svgd0': ('niveau', 0), 'niveau_svgd25': ('niveau', 25),
    'ucb_tuned_svgd0': ('ucb', 0), 'ucb_tuned_svgd25': ('ucb', 25),
    'mass_tuned_svgd0': ('mass', 0), 'mass_tuned_svgd25': ('mass', 25),
    'eid_tuned_svgd0': ('eid_tuned', 0), 'eid_tuned_svgd25': ('eid_tuned', 25),
    'eid_optuna_ideal_v2': ('eid_optuna_ideal_v2', 0),
}
cfm = df[df['method'].isin(['particles', 'spectral'])].copy()
cfm['cfm_family'] = cfm['strategy'].map(lambda s: FAMILY_RE.get(s, (None, None))[0])
cfm['cfm_svgd_iters'] = cfm['strategy'].map(lambda s: FAMILY_RE.get(s, (None, None))[1])

summary_lines = []


def log(line=''):
    print(line)
    summary_lines.append(line)


log("# Intensive-Auswertung: best_of_30_paired_20260918 (PARTIAL)")
log()
log(f"Datenbasis: {len(ONLY_FULLY_DONE_SHAPES)} von 25 Formen vollstaendig "
   f"({', '.join(ONLY_FULLY_DONE_SHAPES)}), {len(df)} Zeilen.")
log("Kein weiterer Job seit 2026-09-20 -- dies ist ein Zwischenstand, kein "
   "Endergebnis.")
log()

# =============================================================================
# 1. Paired-SVGD-Effekt: svgd0 vs. svgd25, SELBER Basis-Kandidat je Familie
# =============================================================================
log("## 1. Paired-SVGD-Effekt (svgd0 -> svgd25, no_replan, derselbe Kandidat)")
rows = []
pair_cfm = cfm[(cfm['replan_scheme'] == 'no_replan') & cfm['cfm_family'].notna()
              & (cfm['cfm_family'] != 'eid_optuna_ideal_v2')]
for rep in ['particles', 'spectral']:
    for fam in ['niveau', 'ucb', 'mass', 'eid_tuned']:
        sub = pair_cfm[(pair_cfm['representation'] == rep) & (pair_cfm['cfm_family'] == fam)]
        s0 = sub[sub['cfm_svgd_iters'] == 0]['E_ergodic_total']
        s25 = sub[sub['cfm_svgd_iters'] == 25]['E_ergodic_total']
        if len(s0) == 0 or len(s25) == 0:
            continue
        rows.append(dict(representation=rep, family=fam,
                         E_ergodic_total_svgd0=s0.mean(), E_ergodic_total_svgd25=s25.mean(),
                         delta=s25.mean() - s0.mean(),
                         pct_improvement=100 * (s0.mean() - s25.mean()) / s0.mean(),
                         n=len(s0)))
paired_df = pd.DataFrame(rows)
paired_df.to_csv(os.path.join(TABLES, '1_paired_svgd_effect.csv'), index=False)
log(paired_df.to_string(index=False))
log()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, rep in zip(axes, ['particles', 'spectral']):
    sub = paired_df[paired_df['representation'] == rep]
    x = np.arange(len(sub))
    w = 0.35
    b1 = ax.bar(x - w / 2, sub['E_ergodic_total_svgd0'], w, label='svgd0 (raw)',
               color=CAT['red'], zorder=2)
    b2 = ax.bar(x + w / 2, sub['E_ergodic_total_svgd25'], w, label='svgd25 (refined)',
               color=STATUS_GOOD, zorder=2)
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(sub['family'])
    ax.set_title(f'{rep}')
    style_ax(ax)
    if ax is axes[0]:
        ax.set_ylabel('E_ergodic_total (lower = better)')
    ax.margins(y=0.15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.suptitle('Paired-SVGD: same base candidate, raw vs. 25 SVGD iterations',
            fontsize=13, fontweight='bold', color=INK_PRIMARY, y=1.12)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '1_paired_svgd_effect.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 2. Random Walk: svgd0 vs. svgd500
# =============================================================================
log("## 2. Random Walk: svgd0 vs. svgd500 (derselbe Best-of-30-Kandidat)")
log("WICHTIGE EINORDNUNG: svgd500 verfeinert direkt gegen die EXAKTE "
   "Ground-Truth-Dichte (random_walk hat kein Wissensstufen-Konzept, nutzt "
   "immer die volle Wahrheit), mit 20x mehr Iterationen als jede andere "
   "Strategie hier verwendet. Der SVGD-Optimierer minimiert direkt dieselbe "
   "Energie, die E_ergodic_total misst -- bei genug Iterationen gegen die "
   "exakte Wahrheit konvergiert JEDE Startbahn nahe ans Optimum, unabhaengig "
   "von ihrer Qualitaet. Das testet \"kann der Solver selbst, mit genug Zeit "
   "und vollem Wissen, nahe loesen\", NICHT \"ist eine Zufallsbahn als Prior "
   "so gut wie ein trainiertes Netz\". Nicht fair vergleichbar mit den "
   "wissensstufen-beschraenkten CFM-/Heuristik-Zahlen unten.")
rw = df[df['method'] == 'random_walk']
rw_summary = rw.groupby('subvariant')[['E_ergodic_total', 'coverage', 'path_len',
                                      'smoothness_energy']].mean()
rw_summary.to_csv(os.path.join(TABLES, '2_random_walk_svgd_effect.csv'))
log(rw_summary.to_string())
log()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
vals = [rw[rw['subvariant'] == 'svgd0']['E_ergodic_total'].mean(),
       rw[rw['subvariant'] == 'svgd500']['E_ergodic_total'].mean()]
bars = ax.bar(['svgd0 (raw)', 'svgd500'], vals, color=[CAT['red'], STATUS_GOOD], zorder=2)
bar_labels(ax, bars)
ax.set_ylabel('E_ergodic_total (lower = better)')
ax.set_title('Random walk: deep SVGD refinement (500 iters)')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '2_random_walk_svgd_effect.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 3. Heuristik/Linear: SVGD-Sweep 0/500/1000 Iterationen
# =============================================================================
log("## 3. Heuristik/Linear-Waypoints: SVGD-Sweep (0/500/1000 Iterationen)")
hl = df[df['method'].isin(['heuristic_tuned', 'linear_waypoints_tuned'])]
hl_summary = hl.groupby(['method', 'family', 'svgd_iters'])['E_ergodic_total'].mean().reset_index()
hl_summary.to_csv(os.path.join(TABLES, '3_heuristic_linear_svgd_sweep.csv'), index=False)
log(hl_summary.to_string(index=False))
log()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, method in zip(axes, ['heuristic_tuned', 'linear_waypoints_tuned']):
    sub = hl_summary[hl_summary['method'] == method]
    for fam in sorted(sub['family'].unique()):
        s = sub[sub['family'] == fam].sort_values('svgd_iters')
        ax.plot(s['svgd_iters'].astype(str), s['E_ergodic_total'], marker='o',
               color=FAMILY_COLOR.get(fam, INK_MUTED), label=fam, linewidth=2, zorder=3)
    ax.set_title(method)
    ax.set_xlabel('SVGD iterations')
    style_ax(ax)
    if ax is axes[0]:
        ax.set_ylabel('E_ergodic_total (lower = better)')
axes[1].legend(frameon=False, loc='upper right', fontsize=9)
fig.suptitle('Heuristic/linear baselines: effect of growing SVGD budget',
            fontsize=13, fontweight='bold', color=INK_PRIMARY)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '3_heuristic_linear_svgd_sweep.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 4. Repraesentationsvergleich: particles vs. spectral
# =============================================================================
log("## 4. Repraesentationsvergleich: particles vs. spectral (alle CFM-Strategien)")
rep_cmp = cfm.groupby('representation')[['E_ergodic_total', 'coverage', 'smoothness_energy',
                                        'path_len']].mean()
rep_cmp.to_csv(os.path.join(TABLES, '4_representation_comparison.csv'))
log(rep_cmp.to_string())
log()
log("ACHTUNG: `spectral` lief hier (wie im ganzen Projekt) auf einem kleineren, "
   "andersartigen Trainings-Formensatz als `particles` -- dieser Vergleich testet "
   "auch Out-of-Distribution-Generalisierung, nicht nur Repraesentation an sich "
   "(siehe spectral_planner.py Modul-Docstring).")
log()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
metrics_show = ['E_ergodic_total', 'smoothness_energy']
for ax, m in zip(axes, metrics_show):
    vals = [rep_cmp.loc['particles', m], rep_cmp.loc['spectral', m]]
    bars = ax.bar(['particles', 'spectral'], vals, color=[CAT['blue'], CAT['violet']], zorder=2)
    bar_labels(ax, bars)
    ax.set_title(m)
    style_ax(ax)
fig.suptitle('particles vs. spectral (lower = better, both metrics)',
            fontsize=13, fontweight='bold', color=INK_PRIMARY)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '4_representation_comparison.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 5. Replan-Schema: no_replan vs. replan_1_6
# =============================================================================
log("## 5. Replan-Schema: no_replan vs. replan_1_6")
scheme_cmp = cfm.groupby('replan_scheme')[['E_ergodic_total', 'coverage', 'path_len']].mean()
scheme_cmp.to_csv(os.path.join(TABLES, '5_replan_scheme_comparison.csv'))
log(scheme_cmp.to_string())
log()

fig, ax = plt.subplots(figsize=(5, 4.5))
bars = ax.bar(scheme_cmp.index, scheme_cmp['E_ergodic_total'],
              color=[CAT['blue'], CAT['orange']], zorder=2)
bar_labels(ax, bars)
ax.set_ylabel('E_ergodic_total (lower = better)')
ax.set_title('Replan scheme')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '5_replan_scheme_comparison.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 6. Wissensstufen-Degradation
# =============================================================================
log("## 6. Wissensstufen: ground_truth -> half_known -> ten_samples -> none_known")
cond_order = ['ground_truth', 'half_known', 'ten_samples', 'none_known']
cond_cmp = cfm.groupby('knowledge_condition')[['E_ergodic_total', 'coverage']].mean().reindex(cond_order)
cond_cmp.to_csv(os.path.join(TABLES, '6_knowledge_condition_comparison.csv'))
log(cond_cmp.to_string())
log()

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(cond_order, cond_cmp['E_ergodic_total'], color=CAT['blue'], zorder=2)
bar_labels(ax, bars)
ax.set_ylabel('E_ergodic_total (lower = better)')
ax.set_title('Degradation with decreasing prior knowledge (CFM, all strategies)')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '6_knowledge_condition_comparison.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 7. Methodenvergleich (Kernergebnis): CFM vs. Heuristik vs. Linear vs. RW vs. Lawnmower
# =============================================================================
log("## 7. Methodenvergleich (Kernergebnis)")
log("Fairness-Hinweis: heuristic_tuned/linear_waypoints_tuned sind "
   "deterministisch/Einmal-Planung (kein Replan-Konzept) -- CFM wird hier "
   "deshalb auf `no_replan` beschraenkt, fuer einen Einmal-Planung-gegen-"
   "Einmal-Planung-Vergleich. CFMs replan_1_6-Option ist separat in Abschnitt "
   "5 zu sehen (deutlich schlechter: 23.5 vs. 15.0) -- in dieser Tabelle NICHT "
   "mitgemittelt, sonst wuerde CFM unfair schlechter aussehen.")
cfm_nr = cfm[cfm['replan_scheme'] == 'no_replan']
method_rows = []
method_subsets = {}  # label -> the exact filtered sub-dataframe behind it,
                     # reused by charts 11/12 below so "best per method" is
                     # defined identically everywhere on this page.
best_cfm_particles = cfm_nr[(cfm_nr['representation'] == 'particles')].groupby('strategy')['E_ergodic_total'].mean().idxmin()
best_cfm_spectral = cfm_nr[(cfm_nr['representation'] == 'spectral')].groupby('strategy')['E_ergodic_total'].mean().idxmin()
_lbl = 'CFM (particles, no_replan, best: %s)' % best_cfm_particles
_sub = cfm_nr[(cfm_nr['representation'] == 'particles') & (cfm_nr['strategy'] == best_cfm_particles)]
method_rows.append((_lbl, _sub['E_ergodic_total'].mean())); method_subsets[_lbl] = _sub
_lbl = 'CFM (spectral, no_replan, best: %s)' % best_cfm_spectral
_sub = cfm_nr[(cfm_nr['representation'] == 'spectral') & (cfm_nr['strategy'] == best_cfm_spectral)]
method_rows.append((_lbl, _sub['E_ergodic_total'].mean())); method_subsets[_lbl] = _sub
for method in ['heuristic_tuned', 'linear_waypoints_tuned']:
    m = df[df['method'] == method]
    # Best SINGLE (family, svgd_iters) combo -- averaging across svgd_iters
    # per family would mix in the un-refined svgd_iters=0 point (E~50-60,
    # see table 3) and badly understate how good this baseline actually is.
    grp = m.groupby(['family', 'svgd_iters'])['E_ergodic_total'].mean()
    best_fam, best_iters = grp.idxmin()
    _lbl = '%s (best: %s@%d)' % (method, best_fam, best_iters)
    _sub = m[(m['family'] == best_fam) & (m['svgd_iters'] == best_iters)]
    method_rows.append((_lbl, grp.loc[(best_fam, best_iters)])); method_subsets[_lbl] = _sub
# random_walk (svgd500) deliberately left out of this core comparison -- per
# the caveat in section 2, it refines against the exact ground truth with
# 20x the SVGD budget of every other entry here, so it isn't a fair point of
# comparison against the knowledge-limited CFM/heuristic numbers. It still
# gets its own dedicated analysis in section 2.
_lbl = 'lawnmower'
_sub = df[df['method'] == 'lawnmower']
method_rows.append((_lbl, _sub['E_ergodic_total'].mean())); method_subsets[_lbl] = _sub
method_df = pd.DataFrame(method_rows, columns=['method', 'E_ergodic_total']).sort_values('E_ergodic_total')
method_df.to_csv(os.path.join(TABLES, '7_method_level_comparison.csv'), index=False)
log(method_df.to_string(index=False))
log()

def _method_color(label):
    if 'random_walk' in label:
        return CAT['red']
    if 'linear_waypoints' in label:
        return CAT['yellow']
    if 'CFM (particles' in label:
        return CAT['blue']
    if 'CFM (spectral' in label:
        return CAT['violet']
    if 'heuristic_tuned' in label:
        return CAT['orange']
    if 'lawnmower' in label:
        return CAT['aqua']
    return INK_MUTED


def _short_method_label(label):
    """Short, UNIQUE per-method label for axis ticks -- must never collapse
    CFM (particles ...) and CFM (spectral ...) to the same string, or
    matplotlib silently merges them into one categorical bar position."""
    if label.startswith('CFM (particles'):
        return 'CFM particles'
    if label.startswith('CFM (spectral'):
        return 'CFM spectral'
    return label.split(' (')[0].replace('*see caveat above*', '').strip()


fig, ax = plt.subplots(figsize=(10, 5.5))
bar_colors = [_method_color(m) for m in method_df['method']]
short_labels7 = [_short_method_label(m) for m in method_df['method']]
bars = ax.barh(short_labels7, method_df['E_ergodic_total'], color=bar_colors, zorder=2)
for b in bars:
    w_ = b.get_width()
    ax.annotate(f'{w_:.2f}', (w_, b.get_y() + b.get_height() / 2), ha='left', va='center',
               fontsize=9, color=INK_SECONDARY, xytext=(4, 0), textcoords='offset points')
ax.invert_yaxis()
ax.set_xlabel('E_ergodic_total (lower = better)')
ax.set_title('Core comparison: best configuration point per method')
style_ax(ax, grid_axis='x')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '7_method_level_comparison.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 8. Pro-Form-Aufschluesselung (bester CFM-Kandidat)
# =============================================================================
log("## 8. Pro-Form-Aufschluesselung (CFM particles, bestes Strategie)")
per_shape = cfm_nr[(cfm_nr['representation'] == 'particles') & (cfm_nr['strategy'] == best_cfm_particles)]
per_shape_summary = per_shape.groupby('shape')['E_ergodic_total'].mean().reindex(ONLY_FULLY_DONE_SHAPES)
per_shape_summary.to_csv(os.path.join(TABLES, '8_per_shape_breakdown.csv'))
log(per_shape_summary.to_string())
log()

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(per_shape_summary.index, per_shape_summary.values, color=CAT['blue'], zorder=2)
bar_labels(ax, bars)
ax.set_ylabel('E_ergodic_total')
ax.set_title(f'Per shape: CFM particles / {best_cfm_particles}')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '8_per_shape_breakdown.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 9. steps_to_full_coverage_reached -- Erreichungsrate je Methode
# =============================================================================
log("## 9. 99%-Coverage-Erreichungsrate je Methode")
reach_rows = []
for method, sub in df.groupby('method'):
    reach_rows.append((method, sub['steps_to_full_coverage_reached'].mean()))
reach_df = pd.DataFrame(reach_rows, columns=['method', 'reach_rate']).sort_values('reach_rate', ascending=False)
reach_df.to_csv(os.path.join(TABLES, '9_coverage_reach_rate.csv'), index=False)
log(reach_df.to_string(index=False))
log("Hinweis: bei diesem strikten 99%-Schwellwert erreichen die meisten Varianten "
   "ihn fast nie (siehe caveat in CLUSTER_CONTEXT_best_of_n.md) -- die Rate selbst "
   "ist aussagekraeftiger als die Schrittzahl.")
log()

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(reach_df['method'], reach_df['reach_rate'], color=CAT['aqua'], zorder=2)
bar_labels(ax, bars, fmt='{:.0%}')
ax.set_ylabel('Share of test cases reaching 99% coverage')
ax.set_title('99% coverage reach rate per method')
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '9_coverage_reach_rate.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 10. Candidate-pool quality: coverage mean +/- std across the 30 raw
#     candidates considered at each decision point (Philipp's request,
#     2026-09-18), broken down PER KNOWLEDGE CONDITION (follow-up request --
#     the first version of this chart silently averaged over all 4
#     conditions together, which was the wrong call: ground_truth's pool and
#     none_known's pool are fundamentally different regimes and shouldn't be
#     blurred into one number). Only `coverage` carries this statistic for
#     free in the pre_svgd data -- E_ergodic_total/J are only ever computed
#     for the single selected/refined winner, not for all 30 raw candidates,
#     so this can't be extended to those metrics without regenerating
#     candidates (see the note below the chart).
# =============================================================================
log("## 10. Candidate-pool coverage: mean +/- std across the 30 raw candidates, per knowledge condition")
pool_rows = []
for cond in cond_order:
    for rep in ['particles', 'spectral']:
        for fam, strat0 in [('niveau', 'niveau_svgd0'), ('ucb', 'ucb_tuned_svgd0'),
                            ('mass', 'mass_tuned_svgd0'), ('eid_tuned', 'eid_tuned_svgd0')]:
            sub = cfm[(cfm['representation'] == rep) & (cfm['strategy'] == strat0)
                     & (cfm['replan_scheme'] == 'no_replan') & (cfm['knowledge_condition'] == cond)]
            if len(sub) == 0:
                continue
            pool_rows.append(dict(knowledge_condition=cond, group=f'{rep}/{fam}',
                                  coverage_mean=sub['coverage_mean'].mean(),
                                  coverage_std=sub['coverage_std'].mean(), n=len(sub)))
    rw_pool = df[(df['method'] == 'random_walk') & (df['subvariant'] == 'svgd0')
                & (df['knowledge_condition'] == cond)]
    if len(rw_pool) > 0:
        pool_rows.append(dict(knowledge_condition=cond, group='random_walk',
                              coverage_mean=rw_pool['coverage_mean'].mean(),
                              coverage_std=rw_pool['coverage_std'].mean(), n=len(rw_pool)))
pool_df = pd.DataFrame(pool_rows)
pool_df.to_csv(os.path.join(TABLES, '10_candidate_pool_coverage.csv'), index=False)
log(pool_df.to_string(index=False))
log("coverage_mean/_std are themselves averages over the 8 shapes within each "
   "(knowledge_condition, group) cell -- n=8 per row, not the spread BETWEEN "
   "shapes. random_walk's value is IDENTICAL across all 4 panels (0.097): each "
   "shape's random_walk row is computed once (method='shared', no belief "
   "involved at all) and the exact same row is replicated with a different "
   "knowledge_condition label for all 4 conditions -- not a real "
   "condition-dependence, just the label reused four times.")
log()

fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
for ax, cond in zip(axes.flat, cond_order):
    sub = pool_df[pool_df['knowledge_condition'] == cond]
    colors = [FAMILY_COLOR.get(g.split('/')[-1], CAT['red']) if '/' in g else CAT['red']
             for g in sub['group']]
    bars = ax.bar(sub['group'], sub['coverage_mean'], yerr=sub['coverage_std'],
                  color=colors, zorder=2, capsize=4,
                  error_kw=dict(ecolor=INK_SECONDARY, linewidth=1.3))
    bar_labels(ax, bars, fmt='{:.3f}')
    ax.set_title(cond)
    plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    style_ax(ax)
fig.supylabel('coverage (lower = better), mean +/- std over 30 candidates', fontsize=13)
fig.suptitle('Candidate-pool quality before selection/refinement, per knowledge condition',
            fontsize=15, fontweight='bold', color=INK_PRIMARY)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '10_candidate_pool_coverage.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 11. Length & smoothness/energy per method (Philipp's note: "wie lang, wie
#     smooth, wie viel energy") -- same best-per-method selection as
#     section 7, via method_subsets.
# =============================================================================
log("## 11. Path length and smoothness/energy per method")
len_energy_rows = []
for label, sub in method_subsets.items():
    len_energy_rows.append(dict(method=label, path_len=sub['path_len'].mean(),
                                smoothness_energy=sub['smoothness_energy'].mean()))
len_energy_df = pd.DataFrame(len_energy_rows).set_index('method').loc[method_df['method']]
len_energy_df.to_csv(os.path.join(TABLES, '11_length_energy_per_method.csv'))
log(len_energy_df.to_string())
log("smoothness_energy = w * sum(acceleration^2), 3-point finite difference, "
   "resampled to 128 points -- this IS the robotics energy/control-effort "
   "proxy, not a separate quantity from smoothness.")
log()

short_labels = [_short_method_label(m) for m in len_energy_df.index]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, col in zip(axes, ['path_len', 'smoothness_energy']):
    colors = [_method_color(m) for m in len_energy_df.index]
    bars = ax.barh(short_labels, len_energy_df[col], color=colors, zorder=2)
    for b in bars:
        w_ = b.get_width()
        ax.annotate(f'{w_:.2f}', (w_, b.get_y() + b.get_height() / 2), ha='left',
                   va='center', fontsize=9, color=INK_SECONDARY,
                   xytext=(4, 0), textcoords='offset points')
    ax.invert_yaxis()
    ax.set_title(col)
    style_ax(ax, grid_axis='x')
fig.suptitle('Length and smoothness/energy, same best-per-method point as section 7',
            fontsize=13, fontweight='bold', color=INK_PRIMARY)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '11_length_energy_per_method.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 12. Steps to recover ground truth (Philipp's note: "in wie viel steps habe
#     ich die ground truth recovered") -- IMPORTANT: checking this across all
#     4 conditions (not just ground_truth) reveals the metric is essentially
#     BINARY at the 99% threshold, not graded: 0.0 only because ground_truth
#     is forced to it by definition (metrics_explore_exploit.py:190 --
#     "no exploration needed by construction, not because of anything the
#     trajectory does"), and ~1.0 (never reached) for all three other
#     conditions in nearly every case. Showing ground_truth alone (the
#     original draft of this chart) would have hidden that and made the
#     metric look more informative than it is -- shown across all 4
#     conditions instead, which makes the degenerate pattern visible.
# =============================================================================
log("## 12. Steps (normalized arclength) to first reach 99% coverage, all 4 conditions")
cond_order_12 = ['ground_truth', 'half_known', 'ten_samples', 'none_known']
steps_methods = ['CFM (particles, no_replan, best: %s)' % best_cfm_particles,
                 'heuristic_tuned (best: mass@1000)', 'lawnmower']
steps_rows = []
for label in steps_methods:
    sub = method_subsets[label]
    for cond in cond_order_12:
        c = sub[sub['knowledge_condition'] == cond]
        if len(c) == 0:
            continue
        steps_rows.append(dict(method=label, knowledge_condition=cond,
                               steps_to_full_coverage=c['steps_to_full_coverage'].mean(),
                               reached=c['steps_to_full_coverage_reached'].mean(), n=len(c)))
steps_df = pd.DataFrame(steps_rows)
steps_df.to_csv(os.path.join(TABLES, '12_steps_to_recover_ground_truth.csv'), index=False)
log(steps_df.to_string(index=False))
log("This metric is essentially BINARY at the 99% threshold in this data: "
   "~0 only for ground_truth (forced by definition, not earned), ~1 (never "
   "reached) for every other condition, with one marginal exception "
   "(particles/none_known: 0.69% of cases reached it). A graded answer to "
   "\"how many steps to recover\" would need a looser threshold (e.g. 80-90%) "
   "than the project's current 99% default.")
log()

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(cond_order_12))
w = 0.2
for i, label in enumerate(steps_methods):
    sub = steps_df[steps_df['method'] == label].set_index('knowledge_condition').reindex(cond_order_12)
    ax.bar(x + (i - 1.5) * w, sub['steps_to_full_coverage'], w,
          label=_short_method_label(label), color=_method_color(label), zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(cond_order_12)
ax.set_ylabel('Normalized arclength to first reach 99% coverage')
ax.set_title('Steps to recover the ground truth, across all 4 knowledge conditions')
ax.legend(frameon=False, fontsize=9, loc='center right')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '12_steps_to_recover_ground_truth.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 13. Candidate-pool distribution comparison, Gaussian-approximated from
#     mean+-std (Philipp's note: "vergleiche die Verteilungen" / "sample von
#     den Verteilungen"). Explicitly labelled as an approximation -- the
#     true empirical shape (skew, multimodality) isn't recoverable from only
#     mean+std, see the caveat already in section 10/the checklist below.
# =============================================================================
log("## 13. Candidate-pool coverage distributions (Gaussian approximation from mean+-std)")
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.linspace(0, 0.22, 400)
palette_cycle = [CAT['blue'], CAT['orange'], CAT['aqua'], CAT['magenta'],
                 CAT['blue'], CAT['orange'], CAT['aqua'], CAT['magenta'], CAT['red']]
for (_, row), color in zip(pool_df.iterrows(), palette_cycle):
    mu, sigma = row['coverage_mean'], row['coverage_std']
    y = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ls = '--' if row['group'] == 'random_walk' else ('-' if 'particles' in row['group'] else ':')
    ax.plot(x, y, color=color, linewidth=2, label=row['group'], linestyle=ls, zorder=3)
ax.set_xlabel('coverage (lower = better)')
ax.set_ylabel('approximate density')
ax.set_title('Candidate-pool coverage distributions (Gaussian approx. from mean+-std)')
ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='upper right')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '13_candidate_pool_distributions.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 14. Explore vs. exploit balance per method -- stacked bar, E_total =
#     E_explore + E_exploit (additive by construction, see the page's
#     "Exploration vs. Exploitation" section).
# =============================================================================
log("## 14. Explore vs. exploit balance per method")
ee_rows = []
for label, sub in method_subsets.items():
    ee_rows.append(dict(method=label, E_ergodic_explore=sub['E_ergodic_explore'].mean(),
                        E_ergodic_exploit=sub['E_ergodic_exploit'].mean()))
ee_df = pd.DataFrame(ee_rows).set_index('method').loc[method_df['method']]
ee_df.to_csv(os.path.join(TABLES, '14_explore_exploit_balance.csv'))
log(ee_df.to_string())
log()

fig, ax = plt.subplots(figsize=(10, 5.5))
short_labels3 = [_short_method_label(m) for m in ee_df.index]
b1 = ax.barh(short_labels3, ee_df['E_ergodic_explore'], color=CAT['blue'],
            label='E_explore (global structure)', zorder=2)
b2 = ax.barh(short_labels3, ee_df['E_ergodic_exploit'], left=ee_df['E_ergodic_explore'],
            color=CAT['orange'], label='E_exploit (local hotspots)', zorder=2)
ax.invert_yaxis()
ax.set_xlabel('E_ergodic_total = E_explore + E_exploit (lower = better)')
ax.set_title('Explore vs. exploit balance per method')
ax.legend(frameon=False, loc='lower right')
style_ax(ax, grid_axis='x')
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '14_explore_exploit_balance.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 15. Knowledge-condition degradation PER METHOD (section 6 only showed the
#     aggregate across all CFM strategies combined -- this breaks it out by
#     method, revealing whether some methods degrade faster than others).
# =============================================================================
log("## 15. Knowledge-condition degradation per method")
deg_rows = []
for label, sub in method_subsets.items():
    for cond in cond_order:
        c = sub[sub['knowledge_condition'] == cond]
        if len(c) == 0:
            continue
        deg_rows.append(dict(method=label, knowledge_condition=cond,
                             E_ergodic_total=c['E_ergodic_total'].mean()))
deg_df = pd.DataFrame(deg_rows)
deg_df.to_csv(os.path.join(TABLES, '15_degradation_per_method.csv'), index=False)
log(deg_df.to_string(index=False))
log()

fig, ax = plt.subplots(figsize=(9, 5.5))
for label in method_df['method']:
    sub = deg_df[deg_df['method'] == label].set_index('knowledge_condition').reindex(cond_order)
    ax.plot(cond_order, sub['E_ergodic_total'], marker='o', linewidth=2,
           color=_method_color(label), label=_short_method_label(label), zorder=3)
ax.set_ylabel('E_ergodic_total (lower = better)')
ax.set_title('Knowledge-condition degradation, per method')
ax.legend(frameon=False, fontsize=9, loc='upper left')
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '15_degradation_per_method.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 16. Representation x knowledge-condition heatmap (interaction effect not
#     visible in the separate bar charts of sections 5/6).
# =============================================================================
log("## 16. Representation x knowledge-condition heatmap (E_ergodic_total)")
heat = cfm.pivot_table(index='representation', columns='knowledge_condition',
                       values='E_ergodic_total', aggfunc='mean').reindex(columns=cond_order)
heat.to_csv(os.path.join(TABLES, '16_representation_condition_heatmap.csv'))
log(heat.to_string())
log()

fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.imshow(heat.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(cond_order))); ax.set_xticklabels(cond_order)
ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        txt_color = 'white' if v > heat.values.mean() else INK_PRIMARY
        ax.text(j, i, f'{v:.1f}', ha='center', va='center', color=txt_color, fontsize=11)
ax.set_title('E_ergodic_total: representation x knowledge condition')
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '16_representation_condition_heatmap.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 17. Per-shape breakdown, ALL top methods (section 8 only showed CFM
#     particles) -- grouped bars per shape, one group of bars per method.
#     random_walk (svgd500) excluded here: per section 2's caveat it's
#     refined directly against the exact ground truth and trivially
#     collapses toward 0 on every shape -- including it would be misleading
#     without that caveat directly attached, and its bars are invisible
#     anyway next to the legitimately comparable methods.
# =============================================================================
log("## 17. Per-shape breakdown, all top methods (random_walk excluded, see caveat)")
shape_methods = [m for m in method_df['method'] if not m.startswith('random_walk')]
shape_rows = []
for label in shape_methods:
    sub = method_subsets[label]
    for shape in ONLY_FULLY_DONE_SHAPES:
        s = sub[sub['shape'] == shape]
        if len(s) == 0:
            continue
        shape_rows.append(dict(method=label, shape=shape, E_ergodic_total=s['E_ergodic_total'].mean()))
shape_df = pd.DataFrame(shape_rows)
shape_df.to_csv(os.path.join(TABLES, '17_per_shape_all_methods.csv'), index=False)
log(shape_df.to_string(index=False))
log()

fig, ax = plt.subplots(figsize=(13, 6))
n_methods = len(shape_methods)
w = 0.8 / n_methods
x = np.arange(len(ONLY_FULLY_DONE_SHAPES))
for i, label in enumerate(shape_methods):
    sub = shape_df[shape_df['method'] == label].set_index('shape').reindex(ONLY_FULLY_DONE_SHAPES)
    ax.bar(x + (i - n_methods / 2 + 0.5) * w, sub['E_ergodic_total'], w,
          color=_method_color(label), label=_short_method_label(label), zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(ONLY_FULLY_DONE_SHAPES, rotation=30, ha='right')
ax.set_ylabel('E_ergodic_total (lower = better)')
ax.set_title('Per-shape breakdown, all top methods')
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.18))
style_ax(ax)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, '17_per_shape_all_methods.png'), bbox_inches='tight')
plt.close(fig)

# =============================================================================
# 18. steps_to_full_coverage at looser thresholds (80%/90%, vs. the project
#     default 99% from section 12) -- Philipp's follow-up request after
#     section 12 showed the metric is near-binary at 99%. Needs the raw
#     trajectories re-evaluated at a different threshold (recompute_thresholds.py,
#     run separately -- no new generation, just re-reads the already-saved
#     winning curves). Only the 3 non-ground_truth conditions are averaged
#     here (ground_truth is forced to 0 regardless of threshold, see
#     section 12 -- including it would just dilute the real signal).
# =============================================================================
log("## 18. Coverage-reach rate at looser thresholds (80% / 90% / 99%)")
th_path = os.path.join(TABLES, '18_steps_thresholds_raw.csv')
if os.path.isfile(th_path):
    th_raw = pd.read_csv(th_path)
    th_raw = th_raw[th_raw['knowledge_condition'] != 'ground_truth']
    th_agg = th_raw.groupby('method')[['reached_80', 'reached_90', 'reached_99']].mean()
    th_agg = th_agg.reindex([m for m in ['CFM particles', 'CFM spectral', 'heuristic_tuned',
                                         'linear_waypoints_tuned', 'lawnmower']
                             if m in th_agg.index])
    th_agg.to_csv(os.path.join(TABLES, '18_coverage_reach_by_threshold.csv'))
    log(th_agg.to_string())
    log("Averaged over half_known/ten_samples/none_known only (ground_truth "
       "excluded -- forced to reached=1 regardless of threshold, would dilute "
       "the signal). lawnmower stands out: 100% reach rate at 80%, vs. "
       "near-0% for the ergodic-optimizing methods -- systematic sweeping "
       "guarantees eventual area coverage, something none of the "
       "density-matching methods are explicitly optimizing for.")
    log()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(th_agg))
    w = 0.25
    for i, (th, color) in enumerate(zip([80, 90, 99], [CAT['aqua'], CAT['orange'], CAT['red']])):
        bars = ax.bar(x + (i - 1) * w, th_agg[f'reached_{th}'], w, label=f'{th}% threshold',
                      color=color, zorder=2)
        bar_labels(ax, bars, fmt='{:.0%}')
    ax.set_xticks(x)
    ax.set_xticklabels(th_agg.index, rotation=15, ha='right')
    ax.set_ylabel('Share of (shape x condition) cases reaching the threshold')
    ax.set_title('Coverage-reach rate at looser thresholds (ground_truth excluded)')
    ax.legend(frameon=False)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, '18_coverage_reach_by_threshold.png'), bbox_inches='tight')
    plt.close(fig)
else:
    log("(recompute_thresholds.py has not been run yet -- skipping section 18)")
log()

# =============================================================================
# 19. svgd0 vs. svgd25 vs. svgd500 (Philipp's follow-up: extend the paired
#     families with a 3rd, much deeper SVGD budget, reusing the already-
#     cached raw candidate -- see extend_svgd500.py, run separately, no new
#     network inference). ground_truth excluded from the main comparison:
#     phi there IS the exact target, so 500 iterations trivially collapse
#     toward 0 regardless of the starting candidate (same artifact as
#     random_walk's svgd500 in section 2) -- shown as a separate footnote
#     instead of silently averaged in.
# =============================================================================
ext_path = os.path.join(HERE, '..', 'svgd500_extension.csv')
if os.path.isfile(ext_path):
    log("## 19. svgd0 vs. svgd25 vs. svgd500 (deeper SVGD budget)")
    ext = pd.read_csv(ext_path)
    EXT_FAMILY = {'niveau_svgd500': 'niveau', 'ucb_tuned_svgd500': 'ucb',
                 'mass_tuned_svgd500': 'mass', 'eid_tuned_svgd500': 'eid_tuned'}
    ext['cfm_family'] = ext['strategy'].map(EXT_FAMILY)
    ext_nogt = ext[ext['knowledge_condition'] != 'ground_truth']
    rows19 = []
    for rep in ['particles', 'spectral']:
        for fam in ['niveau', 'ucb', 'mass', 'eid_tuned']:
            s0 = pair_cfm[(pair_cfm['representation'] == rep) & (pair_cfm['cfm_family'] == fam)
                         & (pair_cfm['cfm_svgd_iters'] == 0)
                         & (pair_cfm['knowledge_condition'] != 'ground_truth')]['E_ergodic_total']
            s25 = pair_cfm[(pair_cfm['representation'] == rep) & (pair_cfm['cfm_family'] == fam)
                          & (pair_cfm['cfm_svgd_iters'] == 25)
                          & (pair_cfm['knowledge_condition'] != 'ground_truth')]['E_ergodic_total']
            s500 = ext_nogt[(ext_nogt['representation'] == rep) & (ext_nogt['cfm_family'] == fam)]['E_ergodic_total']
            rows19.append(dict(representation=rep, family=fam, svgd0=s0.mean(),
                              svgd25=s25.mean(), svgd500=s500.mean()))
    df19 = pd.DataFrame(rows19)
    df19.to_csv(os.path.join(TABLES, '19_svgd0_25_500.csv'), index=False)
    log(df19.to_string(index=False))
    gt500 = ext[ext['knowledge_condition'] == 'ground_truth']['E_ergodic_total'].mean()
    log(f"ground_truth excluded above (collapses toward 0 like random_walk's svgd500, "
       f"mean E_ergodic_total={gt500:.3f}) -- shown separately, not averaged in.")
    log()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, rep in zip(axes, ['particles', 'spectral']):
        sub = df19[df19['representation'] == rep]
        x = np.arange(len(sub))
        w = 0.25
        for i, (col, color) in enumerate(zip(['svgd0', 'svgd25', 'svgd500'],
                                             [CAT['red'], CAT['orange'], STATUS_GOOD])):
            bars = ax.bar(x + (i - 1) * w, sub[col], w, label=col, color=color, zorder=2)
            bar_labels(ax, bars, fmt='{:.1f}')
        ax.set_xticks(x)
        ax.set_xticklabels(sub['family'])
        ax.set_title(rep)
        style_ax(ax)
        if ax is axes[0]:
            ax.set_ylabel('E_ergodic_total (lower = better)')
        ax.margins(y=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.suptitle('SVGD budget: 0 vs. 25 vs. 500 iterations (ground_truth excluded)',
                fontsize=13, fontweight='bold', color=INK_PRIMARY, y=1.14)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, '19_svgd0_25_500.png'), bbox_inches='tight')
    plt.close(fig)
else:
    log("(svgd500_extension.csv not found yet -- skipping section 19)")
log()

with open(os.path.join(HERE, 'SUMMARY.md'), 'w') as f:
    f.write('\n'.join(summary_lines))

print()
print(f"Fertig. Tabellen -> {TABLES}")
print(f"Plots -> {PLOTS}")
print(f"Textzusammenfassung -> {os.path.join(HERE, 'SUMMARY.md')}")
