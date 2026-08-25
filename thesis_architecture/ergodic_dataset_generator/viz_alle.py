r"""
viz_alle.py
===========
Jede Form und jede Bahn der Datenbank zeichnen — nicht eine Stichprobe.

Die Felder sind nach Gruppen sortiert und seitenweise ausgegeben, damit sich
die vier neuen flachen Familien gegen den bestehenden Bestand halten lassen.
Je Feld: Zieldichte als blasse Waermekarte, gefahrene Bahn in Blau, Startpunkt
als Kreuz. Rechnet ueber `dichte_numpy`, nicht ueber JAX — daneben laeuft
womoeglich noch ein Generierungslauf.

    python viz_alle.py --db ergodic_dataset_start.db --out visualizations/alle
"""
import argparse, json, os, sqlite3, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dichte_numpy import dichte_auf_gitter


def white_inferno():
    import matplotlib.colors as mc, matplotlib.pyplot as plt
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mc.LinearSegmentedColormap.from_list('white_inferno', inf)


def gruppe(name, split):
    if name.startswith('flat_val_'): return '5 flache Holdouts'
    if name.startswith('flat_ped_'):   return '1 Sockelformen'
    if name.startswith('flat_blur_'):  return '2 weichgezeichnet'
    if name.startswith('flat_broad_'): return '3 breite Moden'
    if name.startswith('flat_ring_'):  return '4 Konturformen'
    return '6 Holdout (bestehend)' if split == 'val' else '7 Bestand'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db',   default='ergodic_dataset_start.db')
    p.add_argument('--out',  default='visualizations/alle')
    p.add_argument('--cols', type=int, default=12)
    p.add_argument('--rows', type=int, default=10)
    p.add_argument('--res',  type=int, default=56)
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = white_inferno()
    os.makedirs(a.out, exist_ok=True)

    con = sqlite3.connect(a.db)
    rows = con.execute("SELECT shape_name, split, density_params, trajectory, x0 "
                       "FROM ergodic_pairs ORDER BY id ASC").fetchall()
    con.close()
    print(f'  {len(rows)} Zeilen in {a.db}', flush=True)

    nach_gruppe = {}
    for r in rows:
        nach_gruppe.setdefault(gruppe(r[0], r[1]), []).append(r)

    pro_seite = a.cols * a.rows
    seiten, anfahrt_stat = [], []
    for g in sorted(nach_gruppe):
        eintraege = nach_gruppe[g]
        n_seiten = (len(eintraege) + pro_seite - 1) // pro_seite
        for s in range(n_seiten):
            teil = eintraege[s * pro_seite:(s + 1) * pro_seite]
            zeilen = (len(teil) + a.cols - 1) // a.cols
            fig, axes = plt.subplots(zeilen, a.cols,
                                     figsize=(1.12 * a.cols, 1.24 * zeilen),
                                     facecolor='white', squeeze=False)
            for i, (nm, sp, dp, blob, x0s) in enumerate(teil):
                ax = axes[i // a.cols][i % a.cols]
                d = dichte_auf_gitter(json.loads(dp), a.res)
                d = d / max(d.max(), 1e-12)
                xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
                x0 = np.array(json.loads(x0s))
                ax.imshow(d, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                          alpha=.55, vmin=0)
                ax.plot(xy[:, 0], xy[:, 1], color='#1565C0', lw=.75, alpha=.9)
                ax.plot(*x0, marker='X', color='#D81B60', ms=4.0, mew=.7,
                        mec='white', zorder=5)
                ys, xs = np.nonzero(d > 0.15)
                if len(xs):
                    ziel = np.stack([xs / (a.res - 1), ys / (a.res - 1)], 1)
                    anfahrt_stat.append(float(np.min(np.linalg.norm(ziel - x0, axis=1))))
                ax.set_title(nm.replace('rand_', '').replace('_complex', 'c')[:15],
                             fontsize=3.6, color='#555', pad=1.1)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xticks([]); ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_color('#ccc'); sp_.set_linewidth(.4)
            for j in range(len(teil), zeilen * a.cols):
                axes[j // a.cols][j % a.cols].axis('off')
            titel = g[2:] + (f'  ({s + 1}/{n_seiten})' if n_seiten > 1 else '')
            fig.suptitle(f'{titel} — {len(eintraege)} Formen', fontsize=8,
                         color='#1A1A2E', y=.997)
            fig.tight_layout(rect=[0, 0, 1, .985], h_pad=.28, w_pad=.18)
            pfad = os.path.join(a.out, f'{g[0]}_{g[2:].replace(" ", "_")}_{s + 1}.png')
            fig.savefig(pfad, dpi=190, facecolor='white')
            plt.close(fig)
            seiten.append(pfad)
            print(f'    {os.path.basename(pfad)}  ({len(teil)} Felder)', flush=True)

    ad = np.array(anfahrt_stat)
    print(f'\n  {len(seiten)} Seiten in {a.out}/')
    print(f'  Anfahrtsweg vom Startpunkt zur Dichte: Median {np.median(ad):.3f}, '
          f'oberstes Zehntel ab {np.quantile(ad, .9):.3f}, groesster {ad.max():.3f}')


if __name__ == '__main__':
    main()
