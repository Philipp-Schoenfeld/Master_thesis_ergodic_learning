r"""
viz_laengen_db.py
=================
Die Laengen-Datenbank sichtbar machen.

Drei Ansichten:

  laengen_verlauf.png   Wie sich eine Bahn ueber die neunzehn Iterationsstaende
                        entwickelt — je Zeile eine Form, je Spalte ein Stand.
                        Alle Varianten teilen sich denselben Startpunkt, die
                        Laenge ist also die einzige Groesse, die sich aendert.

  laengen_statistik.png Verteilung der Laengen, Wachstum je Checkpoint und die
                        Spreizung innerhalb einer Form.

  laengen_uebersicht_*  Alle Formen im Raster, je Form der laengste Stand.

    python viz_laengen_db.py --db ergodic_dataset_length.db
"""
import argparse
import collections
import json
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def white_inferno():
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    r = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - r) * np.ones((n, 3)) + r * inf[:n, :3]
    return mc.LinearSegmentedColormap.from_list('wi', inf)


def _stil(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc'); s.set_linewidth(.5)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', default='ergodic_dataset_length.db')
    p.add_argument('--out', default='visualizations/laengen')
    p.add_argument('--formen', type=int, default=8,
                   help='Formen im Verlaufsbild.')
    p.add_argument('--res', type=int, default=64)
    p.add_argument('--cols', type=int, default=14)
    p.add_argument('--rows', type=int, default=10)
    a = p.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dichte_numpy import dichte_auf_gitter

    cmap = white_inferno()
    os.makedirs(a.out, exist_ok=True)

    con = sqlite3.connect(a.db)
    zeilen = con.execute(
        "SELECT shape_name, split, n_iters, length, density_params, trajectory,"
        " x0 FROM ergodic_pairs ORDER BY shape_name, n_iters").fetchall()
    con.close()
    if not zeilen:
        sys.exit('Datenbank leer.')

    proForm = collections.OrderedDict()
    for r in zeilen:
        proForm.setdefault(r[0], []).append(r)
    print(f'  {len(zeilen)} Zeilen, {len(proForm)} Formen')

    # ── 1. Verlauf ────────────────────────────────────────────────────────
    namen = list(proForm)[::max(1, len(proForm) // a.formen)][:a.formen]
    spalten = max(len(proForm[n]) for n in namen)
    fig, axes = plt.subplots(len(namen), spalten,
                             figsize=(1.05 * spalten, 1.2 * len(namen)),
                             facecolor='white', squeeze=False)
    for i, nm in enumerate(namen):
        eintraege = proForm[nm]
        d = dichte_auf_gitter(json.loads(eintraege[0][4]), a.res)
        d = d / max(d.max(), 1e-12)
        x0 = np.array(json.loads(eintraege[0][6]))
        for j in range(spalten):
            ax = axes[i][j]
            if j >= len(eintraege):
                ax.axis('off'); continue
            _, _, n_it, L, _, blob, _ = eintraege[j]
            xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
            ax.imshow(d, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                      alpha=.55, vmin=0)
            ax.plot(xy[:, 0], xy[:, 1], color='#1565C0', lw=.9, alpha=.9)
            ax.plot(*x0, marker='X', color='#D81B60', ms=4, mew=.7,
                    mec='white', zorder=5)
            _stil(ax)
            if i == 0:
                ax.set_title(f'{n_it}', fontsize=6.5, color='#555', pad=2)
            if j == 0:
                ax.set_ylabel(nm[:14], fontsize=6.5, color='#1A1A2E')
            ax.set_xlabel(f'{L:.1f}', fontsize=5.5, color='#8A8A9C', labelpad=1)
    fig.suptitle('Dieselbe Form, dieselbe Startposition — nur die Zahl der '
                 'SVGD-Iterationen waechst (Spalten). Unten die Pfadlaenge.',
                 fontsize=8.5, color='#555', y=.995)
    fig.tight_layout(rect=[0, 0, 1, .975], h_pad=.5, w_pad=.25)
    fig.savefig(os.path.join(a.out, 'laengen_verlauf.png'), dpi=180,
                facecolor='white')
    plt.close(fig)
    print('  laengen_verlauf.png')

    # ── 2. Statistik ──────────────────────────────────────────────────────
    L = np.array([r[3] for r in zeilen])
    proIter = collections.defaultdict(list)
    for r in zeilen:
        proIter[r[2]].append(r[3])
    its = sorted(proIter)
    mittel = [np.mean(proIter[i]) for i in its]
    spanne = [max(v[3] for v in e) / max(min(v[3] for v in e), 1e-9)
              for e in proForm.values()]

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6), facecolor='white')
    ax[0].hist(L, bins=50, color='#1565C0', alpha=.85)
    ax[0].set_title('Verteilung aller Pfadlaengen', fontsize=10, color='#1A1A2E')
    ax[0].set_xlabel('Laenge'); ax[0].set_ylabel('Zeilen')

    ax[1].plot(its, mittel, marker='o', ms=3.5, color='#1565C0')
    ax[1].set_xscale('log')
    ax[1].set_title('Mittlere Laenge je Iterationsstand', fontsize=10,
                    color='#1A1A2E')
    ax[1].set_xlabel('SVGD-Iterationen'); ax[1].set_ylabel('Laenge')

    ax[2].hist(spanne, bins=40, color='#00838F', alpha=.85)
    ax[2].set_title('Spreizung je Form (laengster / kuerzester Stand)',
                    fontsize=10, color='#1A1A2E')
    ax[2].set_xlabel('Faktor'); ax[2].set_ylabel('Formen')
    for b in ax:
        b.grid(alpha=.2); b.set_facecolor('white')
        for s in b.spines.values():
            s.set_color('#ccc')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'laengen_statistik.png'), dpi=170,
                facecolor='white')
    plt.close(fig)
    print('  laengen_statistik.png')

    # ── 3. Uebersicht ueber alle Formen ───────────────────────────────────
    proSeite = a.cols * a.rows
    letzte = [e[-1] for e in proForm.values()]
    seiten = (len(letzte) + proSeite - 1) // proSeite
    for s in range(seiten):
        teil = letzte[s * proSeite:(s + 1) * proSeite]
        zeil = (len(teil) + a.cols - 1) // a.cols
        fig, axes = plt.subplots(zeil, a.cols,
                                 figsize=(1.05 * a.cols, 1.15 * zeil),
                                 facecolor='white', squeeze=False)
        for k, (nm, _sp, n_it, Lg, dp, blob, x0s) in enumerate(teil):
            ax = axes[k // a.cols][k % a.cols]
            d = dichte_auf_gitter(json.loads(dp), a.res)
            d = d / max(d.max(), 1e-12)
            xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
            ax.imshow(d, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                      alpha=.55, vmin=0)
            ax.plot(xy[:, 0], xy[:, 1], color='#1565C0', lw=.7, alpha=.9)
            ax.plot(*np.array(json.loads(x0s)), marker='X', color='#D81B60',
                    ms=3.5, mew=.6, mec='white', zorder=5)
            _stil(ax)
            ax.set_title(f'{nm[:13]}  {Lg:.1f}', fontsize=4.2, color='#555',
                         pad=1.2)
        for j in range(len(teil), zeil * a.cols):
            axes[j // a.cols][j % a.cols].axis('off')
        fig.suptitle(f'Alle Formen, jeweils der laengste Stand '
                     f'({s + 1}/{seiten})', fontsize=9, color='#1A1A2E', y=.998)
        fig.tight_layout(rect=[0, 0, 1, .985], h_pad=.3, w_pad=.2)
        pfad = os.path.join(a.out, f'laengen_uebersicht_{s + 1}.png')
        fig.savefig(pfad, dpi=180, facecolor='white')
        plt.close(fig)
        print(f'  {os.path.basename(pfad)}  ({len(teil)} Formen)')

    print(f'\n  Laenge {L.min():.2f} bis {L.max():.2f}, im Mittel {L.mean():.2f}')
    print(f'  Spreizung je Form: Median {np.median(spanne):.2f}x, '
          f'groesste {max(spanne):.2f}x')


if __name__ == '__main__':
    main()
