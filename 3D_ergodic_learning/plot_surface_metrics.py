r"""
plot_surface_metrics.py
=======================
Die `metriken.csv` aus `run_surface_eval.py` zu Diagrammen machen.

`run_surface_eval.py` schreibt eine Zeile je (Form x Flaeche) und druckt am Ende
eine Tabelle. Eine Tabelle mit 25 x 10 = 250 Zeilen liest niemand. Hier
entstehen daraus:

    01_pro_flaeche.png     Balken je Flaeche, Mittelwert mit Streuung, eine
                           Kachel je Metrik — beantwortet "welche Geometrie
                           faellt dem Netz schwer?"
    02_heatmap.png         Form x Flaeche als Waermebild — beantwortet "liegt es
                           an der Flaeche oder an der Dichte?" Eine ganze
                           schlechte Zeile ist eine schwierige Form, eine ganze
                           schlechte Spalte eine schwierige Flaeche.
    03_verteilung.png      Boxplots je Flaeche — zeigt Ausreisser, die der
                           Mittelwert verdeckt.
    metriken.html          Dieselben Zahlen interaktiv (Plotly): Metrik ueber
                           ein Menue umschaltbar, Zellwerte beim Ueberfahren.
    zusammenfassung.csv    Mittel / Median / Max je Flaeche und eine Gesamtzeile.

    python plot_surface_metrics.py --csv results/surfaces/metriken.csv \
                                   --out_dir results/surfaces/diagramme
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# Stilvorgaben aus CLAUDE.md
BLAU = '#1565C0'
GRUEN = '#00C853'
TEXT = '#1A1A2E'
GRAU = '#555555'
SPINE = '#cccccc'

# Metrik -> (Beschriftung, kleiner-ist-besser, Einheit)
METRIKEN = {
    'erg':          ('Ergodic error', True,  ''),
    'coverage':     ('Coverage (distance)', True,  ''),
    'standoff_err': ('Standoff error', True,  ''),
    'pointing_deg': ('Pointing error', True,  '\u00b0'),
    # None = neither large nor small is "better"
    'path_len':     ('Path length', None, ''),
    'hit_frac':     ('Hit surface fraction', None, ''),
}


def lade(csv_pfad, standoff_target):
    """metriken.csv -> (Zeilen, Formen, Flaechen) in Dateireihenfolge."""
    with open(csv_pfad, newline='', encoding='utf-8') as f:
        roh = list(csv.DictReader(f))
    if not roh:
        raise SystemExit(f'{csv_pfad} enthaelt keine Zeilen.')

    zeilen, formen, flaechen = [], [], []
    for r in roh:
        z = {'shape': r['shape'], 'surface': r['surface']}
        for k, v in r.items():
            if k in ('shape', 'surface'):
                continue
            try:
                z[k] = float(v)
            except (TypeError, ValueError):
                z[k] = float('nan')
        # Der rohe Standoff sagt wenig — trainiert wurde auf einen Sollwert,
        # interessant ist der Betrag der Abweichung davon.
        if 'standoff' in z:
            z['standoff_err'] = abs(z['standoff'] - standoff_target)
        zeilen.append(z)
        if z['shape'] not in formen:
            formen.append(z['shape'])
        if z['surface'] not in flaechen:
            flaechen.append(z['surface'])
    return zeilen, formen, flaechen


def labels_fuer(flaechen):
    """Sprechende Namen aus surfaces.py, mit Rueckfall auf den Schluessel."""
    out = {}
    try:
        import surfaces
        for k in flaechen:
            try:
                out[k] = surfaces.build(k).label
            except Exception:
                out[k] = k
    except Exception:
        out = {k: k for k in flaechen}
    return out


def matrix(zeilen, formen, flaechen, metrik):
    M = np.full((len(formen), len(flaechen)), np.nan)
    fi = {n: i for i, n in enumerate(formen)}
    si = {n: i for i, n in enumerate(flaechen)}
    for z in zeilen:
        if metrik in z:
            M[fi[z['shape']], si[z['surface']]] = z[metrik]
    return M


def _achse(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2, color=GRAU, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(SPINE)
    ax.tick_params(colors=GRAU, labelsize=8)
    ax.title.set_color(TEXT)


def bild_pro_flaeche(zeilen, flaechen, lab, metriken, pfad, standoff_target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(metriken)
    spalten = 3
    reihen = int(np.ceil(n / spalten))
    fig, axes = plt.subplots(reihen, spalten, figsize=(5.0 * spalten, 3.6 * reihen),
                             facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    x = np.arange(len(flaechen))
    for ax, m in zip(axes, metriken):
        titel, klein_besser, einheit = METRIKEN.get(m, (m, True, ''))
        mittel, streu = [], []
        for k in flaechen:
            werte = [z[m] for z in zeilen if z['surface'] == k and np.isfinite(z.get(m, np.nan))]
            mittel.append(np.mean(werte) if werte else np.nan)
            streu.append(np.std(werte) if werte else np.nan)
        mittel, streu = np.asarray(mittel), np.asarray(streu)

        # Die beste Flaeche gruen, der Rest blau — die Rangfolge soll ohne
        # Zahlenlesen sichtbar sein.
        if klein_besser is None or not np.isfinite(mittel).any():
            best = -1          # ungewertete Metrik: keine Flaeche wird hervorgehoben
        else:
            best = int(np.nanargmin(mittel) if klein_besser else np.nanargmax(mittel))
        farben = [GRUEN if i == best else BLAU for i in range(len(flaechen))]

        ax.bar(x, mittel, yerr=streu, color=farben, alpha=0.9,
               error_kw=dict(ecolor=GRAU, elinewidth=1, capsize=3, alpha=0.7))
        if m == 'standoff_err':
            ax.axhline(0.0, color=GRAU, lw=1, ls='--', alpha=0.6)
            ax.set_ylabel(f'|Standoff − {standoff_target:g}|', color=GRAU, fontsize=9)
        _achse(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([lab.get(k, k) for k in flaechen], rotation=38,
                           ha='right', fontsize=8)
        pfeil = {True: '↓', False: '↑', None: '·'}[klein_besser]
        ax.set_title(f'{titel} {pfeil}{einheit and "  [" + einheit + "]"}',
                     fontsize=11, loc='left', pad=8)

    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Mean per Target Surface  (Error bar = spread across holdout shapes)',
                 color=TEXT, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(pfad, dpi=150, facecolor='white')
    plt.close(fig)


def bild_heatmap(zeilen, formen, flaechen, lab, metriken, pfad):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metriken = [m for m in metriken if m in ('erg', 'coverage', 'standoff_err', 'pointing_deg')]
    spalten = 2
    reihen = int(np.ceil(len(metriken) / spalten))
    hoehe = max(3.0, 0.22 * len(formen) + 2.0)
    fig, axes = plt.subplots(reihen, spalten,
                             figsize=(5.6 * spalten, hoehe * reihen),
                             facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    for ax, m in zip(axes, metriken):
        M = matrix(zeilen, formen, flaechen, m)
        im = ax.imshow(M, aspect='auto', cmap='inferno_r', interpolation='nearest')
        ax.set_xticks(np.arange(len(flaechen)))
        ax.set_xticklabels([lab.get(k, k) for k in flaechen], rotation=38,
                           ha='right', fontsize=7)
        ax.set_yticks(np.arange(len(formen)))
        ax.set_yticklabels(formen, fontsize=6.5)
        ax.tick_params(colors=GRAU)
        for s in ax.spines.values():
            s.set_color(SPINE)
        titel, klein_besser, einheit = METRIKEN.get(m, (m, True, ''))
        pfeil = {True: '↓', False: '↑', None: '·'}[klein_besser]
        ax.set_title(f'{titel} {pfeil}', fontsize=11,
                     loc='left', color=TEXT, pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(colors=GRAU, labelsize=7)
        cb.outline.set_edgecolor(SPINE)

    for ax in axes[len(metriken):]:
        ax.axis('off')
    fig.suptitle('Shape \u00d7 Surface — dark is bad.  Whole row dark = difficult shape, '
                 'whole column dark = difficult geometry',
                 color=TEXT, fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(pfad, dpi=150, facecolor='white')
    plt.close(fig)


def bild_verteilung(zeilen, flaechen, lab, metriken, pfad):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metriken = [m for m in metriken if m in ('erg', 'coverage', 'standoff_err', 'pointing_deg')]
    fig, axes = plt.subplots(len(metriken), 1,
                             figsize=(max(7.0, 0.85 * len(flaechen) + 3), 3.1 * len(metriken)),
                             facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    for ax, m in zip(axes, metriken):
        daten = [[z[m] for z in zeilen
                  if z['surface'] == k and np.isfinite(z.get(m, np.nan))]
                 for k in flaechen]
        daten = [d if d else [np.nan] for d in daten]
        bp = ax.boxplot(daten, patch_artist=True, widths=0.6,
                        medianprops=dict(color=GRUEN, lw=2),
                        flierprops=dict(marker='o', markersize=3,
                                        markerfacecolor=GRAU, alpha=0.5,
                                        markeredgecolor='none'))
        for box in bp['boxes']:
            box.set(facecolor=BLAU, alpha=0.28, edgecolor=BLAU, linewidth=1.2)
        for w in bp['whiskers'] + bp['caps']:
            w.set(color=GRAU, linewidth=1)
        _achse(ax)
        ax.set_xticklabels([lab.get(k, k) for k in flaechen], rotation=30,
                           ha='right', fontsize=8)
        titel, klein_besser, _ = METRIKEN.get(m, (m, True, ''))
        ax.set_title(f'{titel} — Distribution over holdout shapes',
                     fontsize=11, loc='left', pad=8)

    fig.tight_layout()
    fig.savefig(pfad, dpi=150, facecolor='white')
    plt.close(fig)


def seite_interaktiv(zeilen, formen, flaechen, lab, metriken, pfad, titel):
    """Eine Plotly-Seite: Waermebild und Balken, Metrik ueber ein Menue."""
    import plotly.graph_objects as go

    fig = go.Figure()
    spuren_je_metrik = 2
    for m in metriken:
        M = matrix(zeilen, formen, flaechen, m)
        fig.add_trace(go.Heatmap(
            z=M, x=[lab.get(k, k) for k in flaechen], y=formen,
            colorscale='Inferno_r', visible=False, xaxis='x', yaxis='y',
            colorbar=dict(len=0.72, y=0.62, thickness=12),
            hovertemplate='%{y} on %{x}<br>%{z:.5f}<extra></extra>'))
        mittel = [float(np.nanmean(M[:, j])) if np.isfinite(M[:, j]).any() else np.nan
                  for j in range(len(flaechen))]
        fig.add_trace(go.Bar(
            x=[lab.get(k, k) for k in flaechen], y=mittel,
            marker_color=BLAU, visible=False, xaxis='x2', yaxis='y2',
            hovertemplate='%{x}<br>Mean %{y:.5f}<extra></extra>'))

    for i in range(spuren_je_metrik):
        fig.data[i].visible = True

    knoepfe = []
    for i, m in enumerate(metriken):
        sicht = [False] * (len(metriken) * spuren_je_metrik)
        sicht[i * spuren_je_metrik] = True
        sicht[i * spuren_je_metrik + 1] = True
        knoepfe.append(dict(label=METRIKEN.get(m, (m,))[0], method='update',
                            args=[{'visible': sicht}]))

    fig.update_layout(
        title=dict(text=titel, x=0.02, xanchor='left',
                   font=dict(color=TEXT, size=16)),
        paper_bgcolor='white', plot_bgcolor='white', showlegend=False,
        height=max(760, 22 * len(formen) + 400),
        margin=dict(l=140, r=40, t=110, b=60),
        xaxis=dict(domain=[0, 1], anchor='y', tickangle=-35,
                   gridcolor='#eeeeee', color=GRAU),
        yaxis=dict(domain=[0.34, 1.0], anchor='x', autorange='reversed',
                   color=GRAU),
        xaxis2=dict(domain=[0, 1], anchor='y2', tickangle=-35,
                    gridcolor='#eeeeee', color=GRAU),
        yaxis2=dict(domain=[0, 0.24], anchor='x2', title='Mean',
                    gridcolor='#eeeeee', color=GRAU),
        updatemenus=[dict(buttons=knoepfe, direction='down', x=0.0, y=1.14,
                          xanchor='left', bgcolor='white',
                          bordercolor=SPINE, font=dict(color=TEXT))],
    )
    fig.write_html(pfad, include_plotlyjs='cdn', full_html=True)


def zusammenfassung(zeilen, flaechen, lab, metriken, pfad):
    kopf = ['surface', 'label', 'n']
    for m in metriken:
        kopf += [f'{m}_mean', f'{m}_median', f'{m}_max']

    def block(auswahl):
        out = [len(auswahl)]
        for m in metriken:
            w = [z[m] for z in auswahl if np.isfinite(z.get(m, np.nan))]
            out += ([round(float(np.mean(w)), 6), round(float(np.median(w)), 6),
                     round(float(np.max(w)), 6)] if w else [float('nan')] * 3)
        return out

    with open(pfad, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(kopf)
        for k in flaechen:
            w.writerow([k, lab.get(k, k)] + block([z for z in zeilen if z['surface'] == k]))
        w.writerow(['TOTAL', 'all surfaces'] + block(zeilen))
    return kopf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', default=os.path.join(_here, 'results', 'surfaces', 'metriken.csv'))
    p.add_argument('--out_dir', default=None,
                   help='Standard: <Ordner der CSV>/diagramme')
    p.add_argument('--standoff_target', type=float, default=0.12,
                   help='Sollabstand aus dem Training; daraus wird standoff_err gebildet')
    p.add_argument('--titel', default='3D-Flaechenauswertung')
    a = p.parse_args()

    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(a.csv)), 'diagramme')
    os.makedirs(out_dir, exist_ok=True)

    zeilen, formen, flaechen = lade(a.csv, a.standoff_target)
    lab = labels_fuer(flaechen)
    metriken = [m for m in METRIKEN if any(m in z for z in zeilen)]

    bild_pro_flaeche(zeilen, flaechen, lab, metriken,
                     os.path.join(out_dir, '01_pro_flaeche.png'), a.standoff_target)
    bild_heatmap(zeilen, formen, flaechen, lab, metriken,
                 os.path.join(out_dir, '02_heatmap.png'))
    bild_verteilung(zeilen, flaechen, lab, metriken,
                    os.path.join(out_dir, '03_verteilung.png'))
    seite_interaktiv(zeilen, formen, flaechen, lab, metriken,
                     os.path.join(out_dir, 'metriken.html'), a.titel)
    zusammenfassung(zeilen, flaechen, lab, metriken,
                    os.path.join(out_dir, 'zusammenfassung.csv'))

    with open(os.path.join(out_dir, 'zusammenfassung.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(
            formen=formen, flaechen=flaechen, labels=lab, metriken=metriken,
            standoff_target=a.standoff_target,
            gesamt={m: dict(
                mittel=float(np.nanmean([z[m] for z in zeilen if m in z])),
                median=float(np.nanmedian([z[m] for z in zeilen if m in z])),
                max=float(np.nanmax([z[m] for z in zeilen if m in z])))
                for m in metriken}), f, indent=1)

    print(f'{len(zeilen)} rows — {len(formen)} shapes x {len(flaechen)} surfaces')
    for f_ in ('01_pro_flaeche.png', '02_heatmap.png', '03_verteilung.png',
               'metriken.html', 'zusammenfassung.csv', 'zusammenfassung.json'):
        print(f'  [{f_.split(".")[-1]:4s}] {os.path.join(out_dir, f_)}')


if __name__ == '__main__':
    main()
