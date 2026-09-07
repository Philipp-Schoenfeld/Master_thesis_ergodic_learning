r"""
viz_woerter.py
==============
Sichtpruefung der Wort-Zieldichten und ihrer geloesten Bahnen.

Die Pruefung, die dieses Bild beantworten soll, ist die aus Phase 0: faehrt die
ergodische Bahn die Buchstaben sichtbar ab, und steht jedes Wort vollstaendig im
Einheitsquadrat — oder hat der Rasterizer es an seiner Filtergrenze
abgeschnitten?

Ein Feld je Wort, gezeichnet aus der Datenbank (nicht neu geloest), im
Hausstil: weisser Grund, WHITE_INFERNO fuer die Zieldichte, Grundwahrheit in
Tiefblau, der konditionierte Startpunkt als Punkt am Bahnanfang.

    python viz_woerter.py --db ergodic_dataset_start_plus.db
"""
import argparse, json, os, sqlite3, sys

import numpy as np
import matplotlib
if 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# ── Hausstil (CLAUDE.md) ─────────────────────────────────────────────────────
_inferno = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
_n_white = 40
for _i in range(_n_white):
    _t = _i / _n_white
    _inferno[_i] = (1 - _t) * np.array([1, 1, 1, 1]) + _t * _inferno[_n_white]
WHITE_INFERNO = mcolors.LinearSegmentedColormap.from_list('white_inferno', _inferno)

GT_BLAU = '#1565C0'
TEXT_DUNKEL = '#1A1A2E'


def _lade(db, muster='wort_%'):
    """Je Wort einen Eintrag — den mit der niedrigsten id."""
    con = sqlite3.connect(db)
    q = ("SELECT shape_name, split, density_params, trajectory, x0 "
         "FROM ergodic_pairs WHERE shape_name LIKE ? ORDER BY id ASC")
    gesehen, out = set(), []
    for nm, sp, dp, blob, x0 in con.execute(q, (muster,)):
        wort = nm.rsplit('_x', 1)[0]
        if wort in gesehen:
            continue
        gesehen.add(wort)
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
        out.append((wort, sp, json.loads(dp), xy, np.array(json.loads(x0))))
    con.close()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', default=os.path.join(_here, 'ergodic_dataset_start_plus.db'))
    p.add_argument('--out', default=os.path.join(_here, 'visualizations',
                                                 'woerter_gitter.png'))
    p.add_argument('--res', type=int, default=110)
    p.add_argument('--cols', type=int, default=6)
    a = p.parse_args()

    from shape_library import pdf_on_grid

    eintraege = _lade(a.db)
    if not eintraege:
        raise SystemExit(f'keine Wort-Eintraege in {a.db}')
    n = len(eintraege)
    cols = a.cols
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.35 * cols, 2.55 * rows),
                             facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    # Wieviel des Trägers die Bahn tatsaechlich erreicht — die Zahl, die dem
    # Auge beim "faehrt sie die Buchstaben ab?" nachhilft.
    for ax, (wort, split, dp, xy, x0) in zip(axes, eintraege):
        d, _, _ = pdf_on_grid(dp, resolution=a.res)
        d = np.asarray(d, dtype=np.float64)
        d = d / max(d.max(), 1e-12)
        ax.imshow(d, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO,
                  vmin=0, vmax=1, alpha=0.55, aspect='equal', zorder=0)
        ax.plot(xy[:, 0], xy[:, 1], color=GT_BLAU, lw=2.5, alpha=0.9, zorder=2)
        ax.scatter([x0[0]], [x0[1]], s=42, facecolor='#FFFFFF',
                   edgecolor=GT_BLAU, linewidth=1.6, zorder=3)

        # Traegerabdeckung: Anteil der Zellen mit nennenswerter Dichte, denen
        # die Bahn naeher als eine halbe Zellbreite gekommen ist.
        gy, gx = np.nonzero(d > 0.12)
        if len(gx):
            zellen = np.stack([(gx + .5) / a.res, (gy + .5) / a.res], -1)
            dist = np.linalg.norm(zellen[:, None, :] - xy[None, ::2, :], axis=-1)
            abdeckung = float((dist.min(axis=1) < 0.05).mean())
        else:
            abdeckung = float('nan')

        rand = 'val' if split == 'val' else None
        titel = wort.replace('wort_', '').capitalize()
        ax.set_title(f'{titel}   {abdeckung*100:.0f} %'
                     + ('  (Holdout)' if rand else ''),
                     color=TEXT_DUNKEL, fontsize=9.5, pad=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.2)
        for s in ax.spines.values():
            s.set_color('#cc0000' if rand else '#cccccc')
            s.set_linewidth(1.6 if rand else 1.0)
        ax.set_facecolor('white')

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle('Wort-Zieldichten und ergodische Grundwahrheit — '
                 'Prozentzahl: Anteil des Traegers, den die Bahn erreicht',
                 color=TEXT_DUNKEL, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor='white')
    plt.close(fig)
    print(f'{n} Woerter -> {a.out}')


if __name__ == '__main__':
    main()
