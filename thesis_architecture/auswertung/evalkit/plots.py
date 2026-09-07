r"""
plots.py
========
Abbildungen im Projektstil — weiss, blasse Dichte, neongruene Bahnen.

Der Stil selbst kommt aus `constraints/common.py` (`style_axes`,
`draw_density`, `draw_guided`, `draw_free`, `save`, `WHITE_INFERNO`) und wird
hier nicht neu definiert, sondern importiert. Neu sind nur die Diagrammtypen,
die es dort noch nicht gibt: Kompromiss-Streudiagramme mit Pareto-Front,
Antwortkurven ueber eine Leiter, Boxplots ueber Szenarien und die
Detailtafel Wahrheit/mu/sigma/Phi/Aufenthalt.

Farbkonvention, damit alle Abbildungen zusammen lesbar bleiben
--------------------------------------------------------------
* **Farbe** codiert die Variante (Szenario bzw. Arm) — sie ist die Groesse,
  die man innerhalb eines Diagramms vergleicht.
* **Linienart** codiert das Modell — die Groesse, die man *zwischen*
  Diagrammen wiedererkennen will.

Das ist die einzige Aufteilung, die auch dann noch funktioniert, wenn zwei
Modelle mal drei Szenarien in ein Bild geraten; zwei Farbdimensionen tun das
nicht.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from common import (C_DARK, C_GEN, C_GREY, WHITE_INFERNO,      # constraints/common.py
                    draw_density, draw_free, draw_guided, save, style_axes)

from .metrics_ee import pareto            # noqa: E402

__all__ = ['save', 'draw_density', 'draw_free', 'draw_guided', 'style_axes',
           'WHITE_INFERNO', 'C_DARK', 'C_GEN', 'C_GREY',
           'ARM_FARBEN', 'SZEN_FARBEN', 'linienart', 'gitter', 'achse',
           'panel', 'antwortkurven', 'kompromiss', 'boxen', 'heatmap',
           'detailtafel']

ARM_FARBEN = {
    'frei':       '#9E9E9E',
    'kond':       '#1565C0',
    'kraft':      '#00C853',
    'kond+kraft': '#D81B60',
}
SZEN_FARBEN = {
    'zufall':  '#1565C0',
    'haelfte': '#00C853',
    'loch':    '#D81B60',
    'orakel':  '#8E24AA',
    'blind':   '#9E9E9E',
}
_LINIEN = ['-', '--', ':', '-.']


def linienart(i):
    return _LINIEN[i % len(_LINIEN)]


def farbe(name, i=0):
    return ARM_FARBEN.get(name) or SZEN_FARBEN.get(name) or f"C{i}"


# ── Grundgeruest ─────────────────────────────────────────────────────────────
def achse(ax, titel=None, xl=None, yl=None, fontsize=10):
    """Projektstil fuer ein gewoehnliches (nicht-raeumliches) Diagramm."""
    ax.set_facecolor('white')
    if titel:
        ax.set_title(titel, fontsize=fontsize, color=C_DARK)
    if xl:
        ax.set_xlabel(xl, fontsize=9, color=C_GREY)
    if yl:
        ax.set_ylabel(yl, fontsize=9, color=C_GREY)
    ax.tick_params(labelsize=7, colors=C_GREY)
    for sp in ax.spines.values():
        sp.set_color('#ccc')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')
    return ax


def gitter(n, max_cols=5, groesse=(3.6, 3.9)):
    """Achsenraster fuer n Panels. -> (fig, flache Achsenliste)"""
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, squeeze=False, facecolor='white',
                             figsize=(groesse[0] * cols, groesse[1] * rows))
    flach = [axes[i // cols][i % cols] for i in range(rows * cols)]
    for ax in flach[n:]:
        ax.axis('off')
    return fig, flach[:n]


def unbekannt_schraffur(ax, unbekannt):
    """Das unbekannte Gebiet als graue Flaeche unter der Dichte markieren.

    Bewusst grau und halbtransparent statt farbig: die Zieldichte traegt in
    diesen Abbildungen schon die Farbe (WHITE_INFERNO), und eine zweite
    Farbskala daneben macht beide unlesbar.
    """
    m = np.asarray(unbekannt, dtype=float)
    ax.imshow(np.ma.masked_where(m < 0.5, m), extent=[0, 1, 0, 1],
              origin='lower', cmap=matplotlib.colors.ListedColormap(['#78909C']),
              alpha=0.22, aspect='auto', zorder=0.5, interpolation='nearest')


# ── Panel ueber alle Holdout-Formen ──────────────────────────────────────────
def panel(eintraege, titel, out_dir, name, max_cols=5, untertitel=None):
    """Ein Panel je Form.

    `eintraege` ist eine Liste von dicts mit den Schluesseln

        shape       Formname (Panel-Titel, wenn `kopf` fehlt)
        d_map       (R,R) Dichte im Hintergrund
        curve       (T,2) erzeugte Bahn (numpy)
        frei        optional (T,2) Referenzbahn ohne Fuehrung
        unbekannt   optional (R,R) bool, wird grau hinterlegt
        kopf        optional Panel-Titel
    """
    fig, axes = gitter(len(eintraege), max_cols=max_cols)
    for i, (ax, e) in enumerate(zip(axes, eintraege)):
        draw_density(ax, e['d_map'])
        if e.get('unbekannt') is not None:
            unbekannt_schraffur(ax, e['unbekannt'])
        if e.get('frei') is not None:
            draw_free(ax, e['frei'], label='Reference')
        draw_guided(ax, e['curve'], label=e.get('label', 'generated'))
        style_axes(ax, e.get('kopf', f"'{e['shape']}'"), fontsize=8)
        if i == 0:
            ax.legend(frameon=True, fontsize=6, loc='upper right',
                      facecolor='white', edgecolor='#ddd', framealpha=0.9)
    kopf = titel + (f"\n{untertitel}" if untertitel else '')
    fig.suptitle(kopf, fontsize=13, color=C_DARK, y=1.005)
    fig.tight_layout()
    p = save(fig, out_dir, name)
    plt.close(fig)
    return p


# ── Antwortkurven ueber eine Leiter (kappa oder Ziellaenge) ──────────────────
def antwortkurven(ax, zeilen, x_key, y_key, gruppe_key, modell_key=None,
                  band=True, referenz=None, ref_label=None):
    """Mittelwert je Gruppe ueber die Leiter, mit Interquartilsband.

    Das Band ist der Interquartilsabstand ueber die Formen, nicht die
    Standardabweichung: die Verteilungen sind ueber 25 sehr verschiedene Formen
    hinweg schief, und ein symmetrisches Band suggeriert eine Symmetrie, die
    nicht da ist.
    """
    gruppen = sorted({z[gruppe_key] for z in zeilen})
    modelle = sorted({z[modell_key] for z in zeilen}) if modell_key else [None]
    for gi, g in enumerate(gruppen):
        for mi, m in enumerate(modelle):
            sub = [z for z in zeilen if z[gruppe_key] == g
                   and (m is None or z[modell_key] == m)]
            if not sub:
                continue
            xs = sorted({z[x_key] for z in sub})
            mid, lo, hi = [], [], []
            for x in xs:
                v = np.array([z[y_key] for z in sub if z[x_key] == x], dtype=float)
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    mid.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    continue
                mid.append(np.mean(v))
                lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
            lab = g if m is None else f"{g} / {m}"
            c = farbe(g, gi)
            ax.plot(xs, mid, linienart(mi), color=c, lw=2, marker='o', ms=4,
                    label=lab)
            if band and len(modelle) * len(gruppen) <= 4:
                ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
    if referenz is not None:
        ax.axhline(referenz, color=C_GREY, lw=1, ls=':', label=ref_label)
    ax.legend(frameon=True, fontsize=7, facecolor='white', edgecolor='#ddd')
    return ax


# ── Kompromiss-Streudiagramm mit Pareto-Front ────────────────────────────────
def kompromiss(ax, zeilen, x_key, y_key, farb_key, groesse_key=None,
               front=True, xl=None, yl=None, titel=None):
    """Ausbeutung gegen Erkundung, ein Punkt je Lauf, Front hervorgehoben."""
    gruppen = sorted({z[farb_key] for z in zeilen})
    for gi, g in enumerate(gruppen):
        sub = [z for z in zeilen if z[farb_key] == g]
        xs = [z[x_key] for z in sub]
        ys = [z[y_key] for z in sub]
        s = ([12 + 60 * _skaliert(sub, groesse_key)[i] for i in range(len(sub))]
             if groesse_key else 16)
        ax.scatter(xs, ys, s=s, color=farbe(g, gi), alpha=0.45, lw=0, label=str(g))
        if front:
            idx = pareto(list(zip(xs, ys)))
            pf = sorted([(xs[i], ys[i]) for i in idx])
            ax.plot([p[0] for p in pf], [p[1] for p in pf], '-',
                    color=farbe(g, gi), lw=1.6, alpha=0.9)
    achse(ax, titel, xl, yl)
    ax.legend(frameon=True, fontsize=7, facecolor='white', edgecolor='#ddd',
              title=farb_key, title_fontsize=7)
    return ax


def _skaliert(zeilen, key):
    v = np.array([z[key] for z in zeilen], dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(v)


# ── Boxplots ─────────────────────────────────────────────────────────────────
def boxen(ax, zeilen, wert_key, gruppe_key, unter_key=None, referenz=None,
          ref_label=None, titel=None, yl=None):
    """Verteilung ueber die Formen je Gruppe (und optional je Untergruppe)."""
    gruppen = sorted({z[gruppe_key] for z in zeilen})
    unter = sorted({z[unter_key] for z in zeilen}) if unter_key else [None]
    daten, labels, farben, pos = [], [], [], []
    p = 0.0
    for g in gruppen:
        for ui, u in enumerate(unter):
            sub = [z[wert_key] for z in zeilen if z[gruppe_key] == g
                   and (u is None or z[unter_key] == u)]
            sub = [v for v in sub if np.isfinite(v)]
            if not sub:
                continue
            daten.append(sub)
            labels.append(g if u is None else f"{g}\n{u}")
            farben.append(farbe(g, gruppen.index(g)))
            pos.append(p)
            p += 1.0
        p += 0.6
    if not daten:
        return ax
    bp = ax.boxplot(daten, positions=pos, widths=0.7, patch_artist=True,
                    medianprops=dict(color=C_DARK, lw=1.4),
                    flierprops=dict(marker='.', ms=3, mfc='#888', mec='none'),
                    whiskerprops=dict(color='#bbb'), capprops=dict(color='#bbb'))
    for patch, c in zip(bp['boxes'], farben):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor('#bbb')
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7)
    if referenz is not None:
        ax.axhline(referenz, color=C_GREY, lw=1, ls=':', label=ref_label)
        ax.legend(frameon=True, fontsize=7, facecolor='white', edgecolor='#ddd')
    achse(ax, titel, None, yl)
    return ax


# ── Heatmap ──────────────────────────────────────────────────────────────────
def heatmap(ax, matrix, xticks, yticks, titel=None, xl=None, yl=None,
            fmt='{:.3f}', cmap='inferno_r'):
    m = np.asarray(matrix, dtype=float)
    im = ax.imshow(m, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, fontsize=7)
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=7)
    lo, hi = np.nanmin(m), np.nanmax(m)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not np.isfinite(m[i, j]):
                continue
            hell = (m[i, j] - lo) / max(hi - lo, 1e-12) < 0.5
            ax.text(j, i, fmt.format(m[i, j]), ha='center', va='center',
                    fontsize=6.5, color=('#1A1A2E' if hell else 'white'))
    achse(ax, titel, xl, yl)
    ax.grid(False)
    return im


# ── Detailtafel: woher die Zieldichte kommt ──────────────────────────────────
def detailtafel(shape, wahrheit, mu, sd, phi, belegung_, curve, out_dir, name,
                unbekannt=None, titel=None):
    """Fuenf Felder nebeneinander: Wahrheit, mu, sigma, Phi, Aufenthalt.

    Diese Tafel beantwortet die Frage, die eine Metriktabelle nicht beantworten
    kann: *warum* faehrt die Bahn dorthin. Ohne sie ist ein hoher `lift` nicht
    von einem Artefakt der Zieldichte zu unterscheiden.
    """
    felder = [('True Density', wahrheit, WHITE_INFERNO),
              ('Belief  \u03bc', mu, WHITE_INFERNO),
              ('Uncertainty  \u03c3', sd, 'Blues'),
              ('Target Density  \u03a6', phi, WHITE_INFERNO),
              ('Occupancy  c', belegung_, 'Greens')]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.1), facecolor='white')
    for ax, (t, f, cm) in zip(axes, felder):
        f = np.asarray(f, dtype=float)
        ax.imshow(f / max(f.max(), 1e-12), extent=[0, 1, 0, 1], origin='lower',
                  cmap=cm, vmin=0, vmax=1, alpha=0.85, aspect='auto', zorder=0)
        if unbekannt is not None and t.startswith('True'):
            unbekannt_schraffur(ax, unbekannt)
        ax.plot(curve[:, 0], curve[:, 1], color=C_GEN, lw=1.8, alpha=0.95,
                zorder=3)
        style_axes(ax, t, fontsize=9)
    fig.suptitle(titel or f"'{shape}'", fontsize=12, color=C_DARK, y=1.02)
    fig.tight_layout()
    p = save(fig, out_dir, name)
    plt.close(fig)
    return p
