r"""
plot_variante_d.py
==================
Die Abdeckungsschuld beim Arbeiten zusehen.

Je Form eine Zeile: die Zieldichte in vier Runden, darunter die
Aufenthaltsdichte, aus der die Schuld gerechnet wird — und rechts die
Gesamtbahn gegen die Bahn von `glaube-R`, die dieselbe Strecke ohne
Umbuchung faehrt.

    python plot_variante_d.py --json results/cfm_prior_d/n12/bahnen.json
"""
import argparse, json, os
import numpy as np


def white_inferno():
    import matplotlib.colors as mc, matplotlib.pyplot as plt
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mc.LinearSegmentedColormap.from_list('white_inferno', inf)


def white_blue():
    import matplotlib.colors as mc
    return mc.LinearSegmentedColormap.from_list(
        'white_blue', ['#FFFFFF', '#BBD6F2', '#5B93D6', '#1F4E8C', '#10294D'])


def style(ax):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(alpha=.2); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', default='results/cfm_prior_d/n12/bahnen.json')
    p.add_argument('--shapes', nargs='+', default=['A', 'digit_5', 'korean_5'])
    p.add_argument('--out', default='results/cfm_prior_d/variante_d.png')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap, blue = white_inferno(), white_blue()

    D = json.load(open(a.json))
    by = {e['name']: e for e in D['formen']}
    shapes = [s for s in a.shapes if s in by]

    nr = 2 * len(shapes)
    fig, axes = plt.subplots(nr, 6, figsize=(14.6, 2.42 * nr),
                             facecolor='white', squeeze=False)
    for i, nm in enumerate(shapes):
        e = by[nm]
        d = e['missionen']['glaube-D']
        truth = np.asarray(e['truth'])
        runden, kap = d['runde'], d['kappa']
        pick = [0, 1, 2, len(runden) - 1][:4]

        for j, pi in enumerate(pick):
            ax = axes[2 * i][j]; style(ax)
            ax.imshow(np.asarray(d['phi'][pi]), origin='lower',
                      extent=[0, 1, 0, 1], cmap=cmap, alpha=.72,
                      vmin=0, vmax=1)
            if i == 0:
                ax.set_title(f'$\\Phi$ Runde {runden[pi] + 1}', fontsize=10,
                             color='#1A1A2E')
            ax.set_xlabel(f'$\\kappa$={kap[pi]:.2f}', color='#555', fontsize=8.5)

            ax2 = axes[2 * i + 1][j]; style(ax2)
            ax2.imshow(np.asarray(d['visit'][pi]), origin='lower',
                       extent=[0, 1, 0, 1], cmap=blue, alpha=.9, vmin=0, vmax=1)
            if i == 0:
                ax2.set_title('Aufenthaltsdichte' if j == 0 else '',
                              fontsize=9.5, color='#1F4E8C')

        # Rechts: die beiden Gesamtbahnen
        for j, (mis, col, lab) in enumerate(
                (('glaube-D', '#00838F', 'Variante D'),
                 ('glaube-R', '#1baf7a', 'glaube-R'))):
            ax = axes[2 * i][4 + j]; style(ax)
            ax.imshow(truth, origin='lower', extent=[0, 1, 0, 1], cmap=cmap,
                      alpha=.5, vmin=0)
            b = np.asarray(e['missionen'][mis]['bahn'])
            ax.plot(b[:, 0], b[:, 1], color=col, lw=1.5, alpha=.95)
            mm = e['missionen'][mis]
            ax.set_xlabel(f"cov {mm['coverage']:.4f} · L {mm['path_len']:.1f}",
                          color='#555', fontsize=8.5)
            if i == 0:
                ax.set_title(lab, color=col, fontsize=10, fontweight='semibold')
            axes[2 * i + 1][4 + j].axis('off')

        axes[2 * i][0].set_ylabel(nm, color='#1A1A2E', fontsize=10.5)

    fig.suptitle('Variante D — die Abdeckungsschuld nimmt der besuchten '
                 'Gegend ihre Anziehung', color='#1A1A2E', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.035 / len(shapes) * 2.1])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=132, facecolor='white')
    print(f'[viz] {a.out}')


if __name__ == '__main__':
    main()
