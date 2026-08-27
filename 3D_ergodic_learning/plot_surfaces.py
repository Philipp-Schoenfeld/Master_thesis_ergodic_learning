r"""
plot_surfaces.py
================
Die Ergebnisse von `run_surface_eval.py` zeichnen.

Je Panel: die Oberflaeche als Punktwolke, eingefaerbt mit der darauf
projizierten Zieldichte, dazu die vom Netz erzeugte Bahn und einige
Sensorachsen. Abgewandte Oberflaechenteile bleiben blassgrau — so ist sichtbar,
welcher Teil ueberhaupt beschriftet wurde.

    python plot_surfaces.py --json results/surfaces/bahnen.json
"""
import argparse, json, os
import numpy as np

LABEL = {'ebene_flach': 'Ebene, waagerecht', 'ebene_gekippt': 'Ebene, 50° gekippt',
         'ebene_diagonal': 'Ebene, Raumdiagonale', 'kugel': 'Kugel',
         'wuerfel': 'Würfel', 'ei': 'Eiform', 'bunny': 'Stanford-Bunny'}
ORDER = ['ebene_flach', 'ebene_gekippt', 'ebene_diagonal', 'kugel',
         'wuerfel', 'ei', 'bunny']
BLICK = {'ebene_flach': (26, -60), 'ebene_gekippt': (18, -70),
         'ebene_diagonal': (24, -50), 'kugel': (20, -60), 'wuerfel': (22, -52),
         'ei': (18, -62), 'bunny': (16, -80)}


def white_inferno():
    import matplotlib.colors as mc, matplotlib.pyplot as plt
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mc.LinearSegmentedColormap.from_list('white_inferno', inf)


def style3d(ax, lo=0.10, hi=0.90):
    # Enger als der Einheitswuerfel: die Oberflaechen sitzen in [0.14, 0.86],
    # und bei vollem Bereich verschenkt jedes Panel die Haelfte seiner Flaeche.
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_zlim(lo, hi)
    ax.set_box_aspect((1, 1, 1))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
        pane._axinfo['grid'].update(color='#dddddd', linewidth=.5)
        pane.line.set_color('#cccccc')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def rot6_to_R(r6):
    """(N,6) -> (N,3,3). Gram-Schmidt, wie in orientation.py."""
    a, b = r6[:, :3], r6[:, 3:]
    e1 = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
    b = b - (e1 * b).sum(-1, keepdims=True) * e1
    e2 = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', default='results/surfaces/bahnen.json')
    p.add_argument('--per_block', type=int, default=4)
    p.add_argument('--surface_dots', type=int, default=1400)
    p.add_argument('--arrows', type=int, default=9)
    p.add_argument('--arrow_len', type=float, default=0.09)
    p.add_argument('--out_prefix', default='results/surfaces/oberflaechen')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
    cmap = white_inferno()
    rng = np.random.default_rng(0)

    D = json.load(open(a.json))
    by = {}
    for e in D['eintraege']:
        by.setdefault(e['shape'], {})[e['surface']] = e
    names = list(by.keys())
    keys = [k for k in ORDER if k in by[names[0]]]
    nb = (len(names) + a.per_block - 1) // a.per_block

    for bi in range(nb):
        blk = names[bi * a.per_block:(bi + 1) * a.per_block]
        fig = plt.figure(figsize=(3.05 * len(keys), 3.25 * len(blk)),
                         facecolor='white')
        for r, nm in enumerate(blk):
            for c, k in enumerate(keys):
                e = by[nm][k]
                ax = fig.add_subplot(len(blk), len(keys),
                                     r * len(keys) + c + 1, projection='3d')
                B0 = np.asarray(e['bahn'])
                F0 = np.asarray(e.get('flaeche', e['partikel']))[:, :3]
                allp = np.vstack([B0, F0])
                lo = float(allp.min()) - 0.04
                hi = float(allp.max()) + 0.04
                style3d(ax, lo, hi)
                el, az = BLICK.get(k, (22, -60))
                ax.view_init(elev=el, azim=az)

                if 'flaeche' in e:
                    Q = np.asarray(e['flaeche']); wq = np.asarray(e['gewicht'])
                else:                                   # aeltere Laeufe
                    P = np.asarray(e['partikel']); Q, wq = P[:, :3], P[:, 3]
                if len(Q) > a.surface_dots:
                    sel = rng.choice(len(Q), a.surface_dots, replace=False)
                    Q, wq = Q[sel], wq[sel]
                lit = wq > 1e-3
                ax.scatter(Q[~lit, 0], Q[~lit, 1], Q[~lit, 2], s=1.8,
                           color='#cfcfd8', alpha=.35, linewidths=0,
                           depthshade=False)
                ax.scatter(Q[lit, 0], Q[lit, 1], Q[lit, 2], s=6.0, c=wq[lit],
                           cmap=cmap, vmin=0, vmax=1, alpha=.85, linewidths=0,
                           depthshade=False)

                B = np.asarray(e['bahn'])
                ax.plot(B[:, 0], B[:, 1], B[:, 2], color='#00A344', lw=2.0,
                        alpha=.95)
                # Durchscheinende Bahn (immer im Vordergrund)
                ax.plot(B[:, 0], B[:, 1], B[:, 2], color='#00A344', lw=2.0,
                        alpha=.25, zorder=10)

                if e.get('rot6'):
                    R = rot6_to_R(np.asarray(e['rot6']))
                    idx = np.linspace(0, len(B) - 1, a.arrows).astype(int)
                    ridx = np.linspace(0, len(R) - 1, a.arrows).astype(int)
                    for i, j in zip(idx, ridx):
                        d = R[j][:, 2] * a.arrow_len
                        ax.plot([B[i, 0], B[i, 0] + d[0]],
                                [B[i, 1], B[i, 1] + d[1]],
                                [B[i, 2], B[i, 2] + d[2]],
                                color='#1565C0', lw=1.1, alpha=.8)

                m = e['metrik']
                ax.set_title(LABEL.get(k, k) if r == 0 else '',
                             color='#1A1A2E', fontsize=10, pad=1)
                # In die Achse hinein statt darunter: ausserhalb wird die
                # Beschriftung bei mehreren Zeilen von der naechsten Achse
                # verdeckt und verschwindet aus allen Zeilen ausser der letzten.
                ax.text2D(0.5, 0.045,
                          f"erg {m['erg']:.4f} · so {m['standoff']:.3f} · "
                          f"{m['pointing_deg']:.0f}°",
                          transform=ax.transAxes, ha='center', va='bottom',
                          color='#555', fontsize=8.2, clip_on=False)
                if c == 0:
                    ax.text2D(-0.10, 0.5, nm, transform=ax.transAxes,
                              rotation=90, va='center', ha='center',
                              color='#1A1A2E', fontsize=10)

        fig.suptitle('Das trainierte 3D-CFM+ErgLoss-Netz auf projizierte '
                     f'Zieldichten — Formen {bi * a.per_block + 1}'
                     f'–{bi * a.per_block + len(blk)} von {len(names)}'
                     '   (unter jedem Bild: ergodischer Fehler, Standoff, '
                     'Blickwinkel)', color='#1A1A2E', fontsize=11.5)
        fig.tight_layout(rect=[0.012, 0.008, 1, 1 - 0.05 / len(blk) * 2.2])
        out = f'{a.out_prefix}_{bi + 1}.png'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=132, facecolor='white')
        plt.close(fig)
        print(f'[viz] {out}')


if __name__ == '__main__':
    main()
