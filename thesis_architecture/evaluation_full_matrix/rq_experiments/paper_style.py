r"""
paper_style.py
==============
Diagramm-Bausteine fuer die RQ1-RQ6-Auswertungen, im Look von Sun et al.
(Fig. 4/5/6: Violinplots mit weissem Median-Strich; Fig. 9/10: Boxplots;
Fig. 11: qualitative Trajektorien-Gitter; Fig. 13: Zeit-ueber-Groesse-Kurven)
und gleichzeitig nach der verbindlichen Projekt-Optik aus `CLAUDE.md` (weisser
Hintergrund, `WHITE_INFERNO`, Ground-Truth-Blau `#1565C0`,
Generiert-Neongruen `#00C853`). `viz.white_inferno`/`viz.style_axes` werden
wiederverwendet statt zweimal gepflegt.

Diagrammtext ist durchgaengig Englisch (Titel, Achsen, Legenden) -- feste
Projektregel, unabhaengig von der Sprache des restlichen Codes/der Kommentare.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_mat = os.path.dirname(_here)
if _mat not in sys.path:
    sys.path.insert(0, _mat)

from viz import white_inferno, style_axes                          # noqa: E402

CMAP = white_inferno()

GT_BLUE = '#1565C0'
GEN_GREEN = '#00C853'
PARTICLE_GREY = '#444444'
#: Eine Farbe je Methode, stabil ueber alle RQ-Plots (Sun-Methoden in
#: Grautoenen wie in Suns eigenen Figures, unsere Methoden in Blau/Gruen).
METHOD_COLORS = {
    'sun':           '#757575',
    'sun_cold':      '#9E9E9E',
    'sun_heuristic': '#616161',
    'fm_sinkhorn':   '#8E24AA',
    'cfm_raw':       '#90CAF9',
    'cfm_fmstein':   GT_BLUE,
    'cfm_svgd':      GEN_GREEN,
    'cfm_warm':      GT_BLUE,
    'cold':          '#9E9E9E',
    'warm':          GT_BLUE,
}
METHOD_LABELS = {
    'sun':           'Sun (FM-Stein)',
    'sun_cold':      'Sun (cold)',
    'sun_heuristic': 'Sun + heuristic init',
    'fm_sinkhorn':   'Sun (FM-Sinkhorn)',
    'cfm_raw':       'CFM (raw)',
    'cfm_fmstein':   'CFM + FM-Stein',
    'cfm_svgd':      'CFM + SvgdRefiner',
    'cfm_warm':      'CFM (warm start)',
}


def _finish(fig, out_path, dpi=150, tight=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor='white')
    plt.close(fig)


# ── Violinplot im Sun-Stil (Fig. 4/5/6: weisser Median-Strich, log-Skala
#    optional, eine Spalte je Methode) ───────────────────────────────────────

def violin(data_by_method, title, ylabel, out_path, methods=None, log_y=False,
          figsize=(7.5, 5.2), censored_note=None):
    """data_by_method: {method: array-artige Werte}. `methods`: Reihenfolge
    (Default: Einfuegereihenfolge). NaNs werden als zensiert markiert und
    oben rechts gezaehlt (z. B. "nicht erreicht" bei Zeit-bis-Epsilon)."""
    methods = methods or list(data_by_method)
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.grid(alpha=0.2, color='#ccc', axis='y')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555')

    positions, censored = [], []
    for i, m in enumerate(methods):
        vals = np.asarray(data_by_method[m], dtype=float)
        ok = vals[np.isfinite(vals)]
        censored.append(int((~np.isfinite(vals)).sum()))
        if len(ok) == 0:
            continue
        vp = ax.violinplot([ok if not log_y else np.log10(np.clip(ok, 1e-9, None))],
                           positions=[i], widths=0.75, showmedians=False,
                           showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(METHOD_COLORS.get(m, '#90A4AE'))
            body.set_alpha(0.75)
            body.set_edgecolor('none')
        med = np.median(ok if not log_y else np.log10(np.clip(ok, 1e-9, None)))
        ax.hlines(med, i - 0.2, i + 0.2, color='white', linewidth=2.2, zorder=5)
        jitter = (np.random.default_rng(i).uniform(-0.08, 0.08, size=len(ok)))
        yv = ok if not log_y else np.log10(np.clip(ok, 1e-9, None))
        ax.scatter(np.full(len(ok), i) + jitter, yv, s=10, color='#1A1A2E',
                  alpha=0.35, zorder=4, linewidths=0)
        positions.append(i)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       rotation=15, ha='right', color='#1A1A2E')
    if log_y:
        ticks = ax.get_yticks()
        ax.set_yticklabels([f"{10 ** t:.3g}" for t in ticks])
    ax.set_ylabel(ylabel, color='#1A1A2E')
    import textwrap
    ax.set_title('\n'.join(textwrap.wrap(title, width=48)), color='#1A1A2E', fontsize=12, fontweight='bold')
    if any(censored):
        note = censored_note or 'not reached: ' + ', '.join(
            f"{METHOD_LABELS.get(m, m)} {c}/{len(np.asarray(data_by_method[m]))}"
            for m, c in zip(methods, censored) if c)
        import textwrap
        wrapped = '\n'.join(textwrap.wrap(note, width=70))
        fig.text(0.5, 0.01, wrapped, ha='center', va='bottom', fontsize=8, color='#555')
        fig.subplots_adjust(bottom=0.34 + 0.03 * wrapped.count('\n'))
    _finish(fig, out_path, tight=not any(censored))
    return out_path


# ── Anytime-Kurve im Sun-Fig.13-Stil (Median +/- IQR ueber der Zeit) ─────────

def anytime_curve(df, title, out_path, methods=None, x='time_s', y='E',
                  group='trial', figsize=(6.0, 4.5), log_x=False, log_y=True,
                  xlabel=None, ylabel=None):
    """df: DataFrame mit Spalten [method, trial, x, y]. Zeichnet je Methode
    die Median-Kurve von y ueber x, gebinnt auf ein gemeinsames x-Raster
    (Interpolation je Trial), mit schattiertem Interquartilsbereich."""
    import pandas as pd
    methods = methods or list(df['method'].unique())
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.grid(alpha=0.2, color='#ccc')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555')

    xmax = df[x].quantile(0.95) if len(df) else 1.0
    xs = (np.geomspace(max(df[x][df[x] > 0].min(), 1e-3), xmax, 40) if log_x
         else np.linspace(0, xmax, 40))
    for m in methods:
        sub = df[df['method'] == m]
        if sub.empty:
            continue
        curves = []
        for _, g in sub.groupby(group):
            g = g.sort_values(x)
            curves.append(np.interp(xs, g[x], g[y], left=g[y].iloc[0], right=g[y].iloc[-1]))
        curves = np.stack(curves)
        med = np.median(curves, axis=0)
        q1, q3 = np.percentile(curves, [25, 75], axis=0)
        c = METHOD_COLORS.get(m, '#90A4AE')
        ax.plot(xs, med, color=c, lw=2.0, label=METHOD_LABELS.get(m, m))
        ax.fill_between(xs, q1, q3, color=c, alpha=0.18, linewidth=0)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel or 'Wall-clock time (s)', color='#1A1A2E')
    ax.set_ylabel(ylabel or 'Ergodic error E (lower is better)', color='#1A1A2E')
    import textwrap
    ax.set_title('\n'.join(textwrap.wrap(title, width=55)), color='#1A1A2E', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=9, labelcolor='#1A1A2E')
    _finish(fig, out_path)
    return out_path


# ── Boxplot im Sun-Fig.9/10-Stil ──────────────────────────────────────────────

def box(data_by_method, title, ylabel, out_path, methods=None, log_y=False,
       figsize=(6.5, 4.8)):
    methods = methods or list(data_by_method)
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.grid(alpha=0.2, color='#ccc', axis='y')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555')
    vals = [np.asarray(data_by_method[m], dtype=float) for m in methods]
    vals = [v[np.isfinite(v)] for v in vals]
    bp = ax.boxplot(vals, patch_artist=True, medianprops=dict(color='white', linewidth=2),
                    whiskerprops=dict(color='#555'), capprops=dict(color='#555'),
                    flierprops=dict(markeredgecolor='#90A4AE', markersize=3))
    for patch, m in zip(bp['boxes'], methods):
        patch.set_facecolor(METHOD_COLORS.get(m, '#90A4AE'))
        patch.set_alpha(0.8)
        patch.set_edgecolor('none')
    if log_y:
        ax.set_yscale('log')
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       rotation=15, ha='right', color='#1A1A2E')
    ax.set_ylabel(ylabel, color='#1A1A2E')
    import textwrap
    ax.set_title('\n'.join(textwrap.wrap(title, width=48)), color='#1A1A2E', fontsize=12, fontweight='bold')
    _finish(fig, out_path)
    return out_path


# ── Qualitatives Trajektorien-Panel (Fig. 7/11-Stil): Dichte + eine oder
#    mehrere Bahnen pro Zelle, Raster aus Zellen ────────────────────────────

def trajectory_panel(cells, out_path, ncols=None, figsize_per_cell=2.6,
                     col_titles=None, row_titles=None, suptitle=None,
                     particles=None):
    """cells: 2D-Liste von dicts {'phi': (R,R) oder None, 'trajs': [(T,2),...],
    'start': (2,) oder None, 'title': str oder None}. Erste Bahn in
    `trajs` kraeftig (Projekt-Konvention), weitere Bahnen blasser."""
    nrows = len(cells)
    ncols = ncols or max(len(r) for r in cells)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_cell * ncols,
                                                    figsize_per_cell * nrows),
                             facecolor='white', squeeze=False)
    for i, row in enumerate(cells):
        for j in range(ncols):
            ax = axes[i][j]
            if j >= len(row) or row[j] is None:
                ax.axis('off')
                continue
            cell = row[j]
            style_axes(ax)
            phi = cell.get('phi')
            if phi is not None:
                ax.imshow(phi, origin='lower', extent=(0, 1, 0, 1), cmap=CMAP,
                         alpha=0.55, vmin=0)
            for k, tr in enumerate(cell.get('trajs', [])):
                tr = np.asarray(tr)
                if k == 0:
                    ax.plot(tr[:, 0], tr[:, 1], color=GEN_GREEN, lw=2.2, alpha=0.95,
                           zorder=3)
                else:
                    ax.plot(tr[:, 0], tr[:, 1], color=GEN_GREEN, lw=1.4, alpha=0.3,
                           zorder=2)
            if cell.get('start') is not None:
                sx, sy = cell['start']
                ax.plot(sx, sy, 'o', color='#1A1A2E', markersize=6, zorder=4)
            pts = cell.get('particles') if cell.get('particles') is not None else particles
            if pts is not None:
                pts = np.asarray(pts)
                ax.scatter(pts[:, 0], pts[:, 1], s=6, color=PARTICLE_GREY, alpha=0.3,
                          zorder=1)
            t = cell.get('title')
            if t:
                ax.set_title(t, fontsize=9, color='#1A1A2E')
        if row_titles and i < len(row_titles) and row_titles[i]:
            axes[i][0].set_ylabel(row_titles[i], fontsize=10, color='#1A1A2E')
    if col_titles:
        for j, t in enumerate(col_titles):
            if t:
                axes[0][j].set_title(t, fontsize=10, color='#1A1A2E', fontweight='bold')
    if suptitle:
        # A fixed *fraction* of the figure height (matplotlib's default)
        # puts the suptitle closer and closer to row 0's column titles as
        # more rows are added (the figure grows taller, each row looks
        # relatively smaller) -- with 10 rows this collided outright
        # (observed on the RQ4 qualitative grid, 2026-10). Pin it instead to
        # a near-constant on-page gap: 0.3 inch above the axes block,
        # converted to a figure-relative y.
        fig_h = figsize_per_cell * nrows
        fig.suptitle(suptitle, color='#1A1A2E', fontsize=13, fontweight='bold',
                    y=1 - 0.3 / (fig_h + 0.6))
        fig.subplots_adjust(top=1 - 0.65 / (fig_h + 0.6))
        _finish(fig, out_path, tight=False)
        return out_path
    _finish(fig, out_path)
    return out_path


# ── Balken (iterations/time-to-eps je Dynamik, Fig. 11-Ergaenzung;
#    Boxplot-Alternative fuer wenige Punkte je Zelle) ─────────────────────────

def grouped_bars(data, title, ylabel, out_path, group_labels, series_labels,
                 series_colors=None, figsize=(7.5, 4.6), log_y=False):
    """data: (n_groups, n_series) Werte (z. B. Dynamiken x Methoden). NaN =
    nicht erreicht: als schraffierter Balken bis zur Oberkante gezeichnet
    (NICHT als Balken der Hoehe 0 -- das saehe wie ein perfektes Ergebnis
    aus, waere also irrefuehrend)."""
    data = np.asarray(data, dtype=float)
    n_groups, n_series = data.shape
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    ax.grid(alpha=0.2, color='#ccc', axis='y')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555')
    width = 0.8 / n_series
    x = np.arange(n_groups)
    colors = series_colors or [METHOD_COLORS.get(s, None) for s in series_labels]
    ymax = np.nanmax(data) if np.isfinite(data).any() else 1.0
    nan_h = ymax * 1.08
    for j in range(n_series):
        xj = x + (j - (n_series - 1) / 2) * width
        ok = ~np.isnan(data[:, j])
        ax.bar(xj[ok], data[ok, j], width=width,
              label=series_labels[j] if ok.any() else None,
              color=colors[j] if colors[j] else None)
        nan_mask = ~ok
        if nan_mask.any():
            ax.bar(xj[nan_mask], nan_h, width=width, color='none', edgecolor='#9E9E9E',
                  hatch='////', linewidth=1.0,
                  label='not reached' if j == n_series - 1 or not ok.any() else None)
    if log_y:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, nan_h * 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha='right', color='#1A1A2E')
    ax.set_ylabel(ylabel, color='#1A1A2E')
    import textwrap
    ax.set_title('\n'.join(textwrap.wrap(title, width=55)), color='#1A1A2E',
                fontsize=12, fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=9, labelcolor='#1A1A2E')
    _finish(fig, out_path)
    return out_path
