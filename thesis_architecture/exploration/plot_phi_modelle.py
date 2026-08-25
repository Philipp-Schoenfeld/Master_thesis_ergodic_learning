r"""
plot_phi_modelle.py
===================
Die Zieldichten der verschiedenen Modelle nebeneinander zeichnen.

Zeilen: die Messmengen. Spalten: die Wahrheit, dann je ein Modell.
So ist unmittelbar zu sehen, welches Modell auf die Form zeigt und welches
im Wesentlichen das Muster der Messpunkte abbildet.

    python plot_phi_modelle.py --shapes A digit_5
"""
import argparse, json, os
import numpy as np

TITEL = {'ucb': r'UCB  $\mu+\kappa\sigma$', 'stretch': 'UCB, gespreizt',
         'mass': r'Massenanteil  $w$', 'ei': 'Erwarteter Zugewinn',
         'lse': r'$P(f>\tau)$  Niveaumenge',
         'eid': 'Informationsdichte (EID)', 'mi': 'GP-MI (gesättigt)'}
ORDER = ['ucb', 'stretch', 'mass', 'mi', 'ei', 'lse', 'eid']


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
    p.add_argument('--json', default='results/phi_modelle/felder.json')
    p.add_argument('--csv', default='results/phi_modelle/kennzahlen.csv')
    p.add_argument('--shapes', nargs='+', default=None)
    p.add_argument('--out_prefix', default='results/phi_modelle/phi')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    cmap = white_inferno()

    F = json.load(open(a.json))
    kz = list(csv.DictReader(open(a.csv)))
    shapes = a.shapes or list(F)

    for nm in shapes:
        if nm not in F:
            print(f'  (uebersprungen: {nm})'); continue
        e = F[nm]
        truth = np.asarray(e['_wahrheit'])
        modelle = [m for m in ORDER if m in e]
        priors = sorted(int(k) for k in e[modelle[0]])
        cols = 1 + len(modelle)
        fig, axes = plt.subplots(len(priors), cols,
                                 figsize=(1.86 * cols, 2.16 * len(priors)),
                                 facecolor='white', squeeze=False)
        for r, n in enumerate(priors):
            ax = axes[r][0]; style(ax)
            ax.imshow(truth, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                      alpha=.6, vmin=0, vmax=1)
            mp = np.asarray(e['_mess'][str(n)])
            ax.scatter(mp[:, 0], mp[:, 1], s=11, color='#1A1A2E', alpha=.8,
                       linewidths=0, zorder=3)
            ax.set_ylabel(f'{n} Messungen', color='#1A1A2E', fontsize=10)
            if r == 0:
                ax.set_title('Wahrheit + Messpunkte', color='#1A1A2E',
                             fontsize=9.5)
            for c, m in enumerate(modelle, start=1):
                ax = axes[r][c]; style(ax)
                ax.imshow(np.asarray(e[m][str(n)]), origin='lower',
                          extent=[0, 1, 0, 1], cmap=cmap, alpha=.75,
                          vmin=0, vmax=1)
                sel = [x for x in kz if x['shape'] == nm
                       and int(x['n_prior']) == n and x['modell'] == m]
                if sel:
                    ax.set_xlabel(f"treffer {float(sel[0]['treffer']):.3f}",
                                  color='#555', fontsize=8)
                if r == 0:
                    ax.set_title(TITEL.get(m, m), color='#1A1A2E', fontsize=9.5)

        fig.suptitle(f'{nm} — dieselbe Messung, sechs Modellierungen der '
                     'Zieldichte', color='#1A1A2E', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.05 / len(priors) * 2.3])
        out = f'{a.out_prefix}_{nm}.png'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=132, facecolor='white'); plt.close(fig)
        print(f'[viz] {out}')


if __name__ == '__main__':
    main()
