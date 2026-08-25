r"""
plot_alle_missionen.py
======================
Jede gefahrene Bahn jeder Mission auf jeder Holdout-Form zeichnen.

Die Uebersichtsbilder des Auswertungslaufs zeigen nur vier Formen und nur die
glaubenskonditionierten Missionen. Hier kommt alles auf den Tisch: neun
Missionen mal zwoelf Formen, in Bloecken zu je vier Formen, damit die
einzelnen Bilder lesbar bleiben.

Liest `bahnen.json`, das `apply_cfm_belief.py --save_paths` schreibt.

    python plot_alle_missionen.py --json results/cfm_belief_lokal/bahnen.json
"""
import argparse, json, os
import numpy as np

MISS = ['orakel', 'glaube-1', 'glaube-R', 'zweistufig',
        'B-warm', 'B-kalt', 'B-auswahl', 'grad-R', 'maeher']
# Dieselben Rollen wie auf der Ergebnisseite, damit die Farben wiedererkennbar
# bleiben; maeher ist die neutrale Referenz.
FARBE = {'orakel': '#2a78d6', 'glaube-1': '#eb6834', 'glaube-R': '#1baf7a',
         'zweistufig': '#eda100', 'B-warm': '#e87ba4', 'B-kalt': '#008300',
         'B-auswahl': '#4a3aa7', 'grad-R': '#e34948', 'maeher': '#9A9AAC'}


def white_inferno():
    import matplotlib.colors as mc, matplotlib.pyplot as plt
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mc.LinearSegmentedColormap.from_list('white_inferno', inf)


def style(ax):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(alpha=0.2); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', default='results/cfm_belief_lokal/bahnen.json')
    p.add_argument('--per_block', type=int, default=4)
    p.add_argument('--out_prefix', default='results/cfm_belief/alle_missionen')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = white_inferno()

    D = json.load(open(a.json))
    formen = D['formen']
    nb = (len(formen) + a.per_block - 1) // a.per_block

    for bi in range(nb):
        blk = formen[bi * a.per_block:(bi + 1) * a.per_block]
        fig, axes = plt.subplots(len(blk), len(MISS),
                                 figsize=(1.72 * len(MISS), 1.86 * len(blk)),
                                 facecolor='white', squeeze=False)
        for r, e in enumerate(blk):
            truth = np.asarray(e['truth'])
            for c, m in enumerate(MISS):
                ax = axes[r][c]; style(ax)
                ax.imshow(truth, origin='lower', extent=[0, 1, 0, 1],
                          cmap=cmap, alpha=0.55, vmin=0.0)
                d = e['missionen'].get(m)
                if d is None:
                    ax.text(.5, .5, '—', ha='center', va='center',
                            color='#999', fontsize=13); continue
                xy = np.asarray(d['bahn'])
                ax.plot(xy[:, 0], xy[:, 1], color=FARBE[m], lw=1.5,
                        alpha=0.95, solid_capstyle='round')
                ax.set_xlabel(f"{d['coverage']:.4f} · L {d['path_len']:.1f}",
                              color='#555', fontsize=7.2, labelpad=1.5)
                if r == 0:
                    ax.set_title(m, color=FARBE[m], fontsize=10,
                                 fontweight='semibold')
            axes[r][0].set_ylabel(e['name'], color='#1A1A2E', fontsize=9.5)

        fig.suptitle('Alle Missionen des trainierten CFM+ErgLoss-Netzes — '
                     f'Formen {bi * a.per_block + 1}–{bi * a.per_block + len(blk)} '
                     f'von {len(formen)}   (unter jedem Bild: Abdeckungsabstand '
                     'und Weglänge)', color='#1A1A2E', fontsize=11.5)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.055 / len(blk) * 2.4])
        out = f'{a.out_prefix}_{bi + 1}.png'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=132, facecolor='white')
        plt.close(fig)
        print(f'[viz] {out}')


if __name__ == '__main__':
    main()
