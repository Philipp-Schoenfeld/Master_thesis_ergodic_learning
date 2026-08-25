r"""
viz_startpunkte.py
==================
Die Bahnen der Startpunkt-Datenbank zeichnen.

Je Feld: die Zieldichte, die gefahrene Bahn und der Startpunkt als Kreuz.
Der Startpunkt liegt jetzt irgendwo auf der Flaeche statt immer unten links —
die Bahn braucht also eine Anfahrt, bevor sie mit dem Abdecken beginnt. Genau
die soll hier sichtbar sein: sie ist der Teil, den ein startpunktkonditioniertes
Netz lernen muss und den das bisherige gar nicht kannte.

    python viz_startpunkte.py --db ergodic_dataset_start.db --n 24
"""
import argparse, json, os, sqlite3
import numpy as np


def white_inferno():
    import matplotlib.colors as mc, matplotlib.pyplot as plt
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mc.LinearSegmentedColormap.from_list('white_inferno', inf)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', default='ergodic_dataset_start.db')
    p.add_argument('--n', type=int, default=24)
    p.add_argument('--cols', type=int, default=6)
    p.add_argument('--res', type=int, default=96)
    p.add_argument('--out', default='visualizations/startpunkte.png')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shape_library import pdf_on_grid
    cmap = white_inferno()

    con = sqlite3.connect(a.db)
    rows = con.execute("SELECT shape_name, split, density_params, trajectory, x0 "
                       "FROM ergodic_pairs ORDER BY id ASC").fetchall()
    con.close()
    if not rows:
        print('  (Datenbank noch leer)'); return
    schritt = max(1, len(rows) // a.n)
    rows = rows[::schritt][:a.n]

    cols = a.cols
    zeilen = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(zeilen, cols, figsize=(2.12 * cols, 2.28 * zeilen),
                             facecolor='white', squeeze=False)
    anfahrten = []
    for i, (nm, sp, dp, blob, x0s) in enumerate(rows):
        ax = axes[i // cols][i % cols]
        d, _, _ = pdf_on_grid(json.loads(dp), resolution=a.res)
        d = np.asarray(d, dtype=float); d /= max(d.max(), 1e-12)
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
        x0 = np.array(json.loads(x0s))

        ax.imshow(d, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                  alpha=.55, vmin=0)
        ax.plot(xy[:, 0], xy[:, 1], color='#1565C0', lw=1.4, alpha=.9)
        ax.plot(*x0, marker='X', color='#D81B60', ms=9, mew=1.4,
                mec='white', zorder=5)
        # Wie weit ist der Startpunkt von der Dichte entfernt?
        R = d.shape[0]
        ys, xs = np.nonzero(d > 0.15)
        if len(xs):
            P = np.stack([xs / (R - 1), ys / (R - 1)], -1)
            dist = float(np.linalg.norm(P - x0, axis=1).min())
        else:
            dist = float('nan')
        anfahrten.append(dist)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=.2)
        for s in ax.spines.values():
            s.set_color('#ccc')
        ax.set_title(nm[:16], fontsize=8.5, color='#1A1A2E', pad=2)
        ax.set_xlabel(f'Anfahrt {dist:.2f}', fontsize=7.6, color='#555',
                      labelpad=1.5)

    for j in range(len(rows), zeilen * cols):
        axes[j // cols][j % cols].axis('off')

    fig.suptitle(f'Startpunkt-Datensatz — {len(rows)} von '
                 f'{schritt * len(rows)} Bahnen   '
                 f'(Kreuz = Startpunkt, Zahl = Abstand zur Zieldichte)',
                 color='#1A1A2E', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.03 / max(zeilen, 1) * 2.2])
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out, dpi=132, facecolor='white')
    print(f'[viz] {a.out}')
    v = np.array(anfahrten)
    print(f'  Anfahrt: min {np.nanmin(v):.2f}  Median {np.nanmedian(v):.2f}  '
          f'max {np.nanmax(v):.2f}')


if __name__ == '__main__':
    main()
