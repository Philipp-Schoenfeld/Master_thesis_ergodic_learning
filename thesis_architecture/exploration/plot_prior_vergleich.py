r"""
plot_prior_vergleich.py
=======================
Die drei Vorwissensmengen nebeneinanderstellen.

Der Sweep aus `run_job_cfm_prior.bash` schreibt drei getrennte
Ergebnisverzeichnisse. Getrennt betrachtet zeigen sie kaum etwas — der Punkt
ist der Vergleich. Diese Abbildung legt fuer *dieselbe* Form die drei
Durchlaeufe uebereinander:

    Zeilen  = 12, 30, 60 Vorabmessungen
    Spalten = die Zieldichte der ersten Runde, dann jede Mission

Weil derselbe Zufallskeim benutzt wurde, sind die zwoelf Vormessungen des
ersten Durchlaufs auch die ersten zwoelf des zweiten und dritten. Die Zeilen
sind also nicht drei unabhaengige Stichproben, sondern eine wachsende.

    python plot_prior_vergleich.py --shapes A digit_5
"""
import argparse, json, os
import numpy as np

MISS = ['orakel', 'glaube-1', 'glaube-R', 'zweistufig', 'B-warm', 'maeher']
FARBE = {'orakel': '#2a78d6', 'glaube-1': '#eb6834', 'glaube-R': '#1baf7a',
         'zweistufig': '#eda100', 'B-warm': '#e87ba4', 'maeher': '#9A9AAC',
         'glaube-D': '#00838F'}


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
    p.add_argument('--root', default='results/cfm_prior')
    p.add_argument('--priors', type=int, nargs='+', default=[12, 30, 60])
    p.add_argument('--shapes', nargs='+', default=['A', 'digit_5'])
    p.add_argument('--missions', nargs='+', default=MISS)
    p.add_argument('--out_prefix', default='results/cfm_belief/prior_vergleich')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = white_inferno()

    D = {n: json.load(open(f'{a.root}/n{n}/bahnen.json')) for n in a.priors}

    for name in a.shapes:
        cols = 1 + len(a.missions)
        fig, axes = plt.subplots(len(a.priors), cols,
                                 figsize=(1.78 * cols, 2.06 * len(a.priors)),
                                 facecolor='white', squeeze=False)
        for r, n in enumerate(a.priors):
            e = next(x for x in D[n]['formen'] if x['name'] == name)
            truth = np.asarray(e['truth'])
            prior = np.asarray(e['prior']) if e['prior'] else np.zeros((0, 2))

            # Spalte 0: die Zieldichte, auf die die erste Runde konditioniert
            ax = axes[r][0]; style(ax)
            phi = e['missionen']['glaube-R']['phi']
            ax.imshow(np.asarray(phi[0]), origin='lower', extent=[0, 1, 0, 1],
                      cmap=cmap, alpha=0.6, vmin=0.0, vmax=1.0)
            ax.scatter(prior[:, 0], prior[:, 1], s=13, color='#1A1A2E',
                       alpha=.75, linewidths=0, zorder=3)
            if r == 0:
                ax.set_title(r'$\Phi$ Runde 1', color='#1A1A2E', fontsize=10)
            ax.set_ylabel(f'{n} Messungen', color='#1A1A2E', fontsize=10.5)

            for c, m in enumerate(a.missions, start=1):
                ax = axes[r][c]; style(ax)
                ax.imshow(truth, origin='lower', extent=[0, 1, 0, 1],
                          cmap=cmap, alpha=0.5, vmin=0.0)
                d = e['missionen'].get(m)
                if d is None:
                    ax.text(.5, .5, '—', ha='center', va='center',
                            color='#999'); continue
                xy = np.asarray(d['bahn'])
                ax.plot(xy[:, 0], xy[:, 1], color=FARBE.get(m, '#444'),
                        lw=1.6, alpha=.95, solid_capstyle='round')
                ax.set_xlabel(f"{d['coverage']:.4f} · L {d['path_len']:.1f}",
                              color='#555', fontsize=7.4, labelpad=1.5)
                if r == 0:
                    ax.set_title(m, color=FARBE.get(m, '#444'), fontsize=10,
                                 fontweight='semibold')

        fig.suptitle(f'{name} — dieselbe Form, dreimal gefahren mit '
                     f'{", ".join(str(x) for x in a.priors)} Vorabmessungen',
                     color='#1A1A2E', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.045 / len(a.priors) * 2.2])
        out = f'{a.out_prefix}_{name}.png'
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=132, facecolor='white')
        plt.close(fig)
        print(f'[viz] {out}')


if __name__ == '__main__':
    main()
