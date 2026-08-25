r"""
plot_phi_kreuz.py
=================
Wie die Modellierung der Zieldichte und die Missionen zusammenwirken.

Zwei Ausgaben:

  bahnen_<Form>_n<N>.png   Raster: Zeilen sind die Phi-Modelle, Spalten die
                           Missionen. Dieselbe Form, dieselbe Messmenge — nur
                           die Zieldichte und der Planer wechseln.
  matrix_<Groesse>.png     Waermekarte derselben Anordnung, aber mit Zahlen
                           statt Bahnen, gemittelt ueber alle zwoelf Formen.

    python plot_phi_kreuz.py --shape A --priors 12 30 60
"""
import argparse, csv, json, os
import numpy as np

MODELLE = ['ucb', 'stretch', 'mass', 'mi', 'ei', 'lse', 'eid']
MNAME = {'ucb': 'UCB  μ+κσ', 'stretch': 'UCB gespreizt', 'mass': 'Massenanteil',
         'mi': 'GP-MI', 'ei': 'Erw. Zugewinn', 'lse': 'Niveaumenge',
         'eid': 'Informationsdichte'}
MISSIONEN = ['orakel', 'glaube-1', 'glaube-R', 'zweistufig', 'glaube-D',
             'B-warm', 'maeher']
FARBE = {'orakel': '#2a78d6', 'glaube-1': '#eb6834', 'glaube-R': '#1baf7a',
         'zweistufig': '#eda100', 'glaube-D': '#00838F', 'B-warm': '#e87ba4',
         'maeher': '#9A9AAC'}


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
    ax.grid(alpha=.2); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')


def lade(root, root_d, modell, n):
    """Alle Missionen eines (Modell, Messmenge) — aus beiden Verzeichnissen."""
    out = {}
    for base in (f'{root}/{modell}_n{n}', f'{root_d}/{modell}_n{n}'):
        f = f'{base}/bahnen.json'
        if not os.path.exists(f):
            continue
        for e in json.load(open(f))['formen']:
            d = out.setdefault(e['name'], {'truth': e['truth'], 'm': {}})
            d['m'].update(e['missionen'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='results/phi_kreuz')
    p.add_argument('--root_d', default='results/phi_kreuz_d')
    p.add_argument('--shape', default='A')
    p.add_argument('--priors', type=int, nargs='+', default=[12, 30, 60])
    p.add_argument('--groesse', default='ergodic', choices=['ergodic', 'coverage',
                                                            'belief_rmse'])
    p.add_argument('--out_dir', default='results/phi_kreuz')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = white_inferno()
    os.makedirs(a.out_dir, exist_ok=True)

    daten = {(m, n): lade(a.root, a.root_d, m, n)
             for m in MODELLE for n in a.priors}
    vorhanden = [m for m in MODELLE if daten[(m, a.priors[0])]]
    if not vorhanden:
        print('  (keine Daten gefunden)'); return

    # ── 1) Bahnen-Raster je Messmenge ────────────────────────────────────
    for n in a.priors:
        miss = [x for x in MISSIONEN
                if any(x in daten[(m, n)].get(a.shape, {'m': {}})['m']
                       for m in vorhanden)]
        if not miss:
            continue
        fig, axes = plt.subplots(len(vorhanden), len(miss),
                                 figsize=(1.74 * len(miss), 1.94 * len(vorhanden)),
                                 facecolor='white', squeeze=False)
        for r, mo in enumerate(vorhanden):
            e = daten[(mo, n)].get(a.shape)
            truth = np.asarray(e['truth']) if e else None
            for c, mi in enumerate(miss):
                ax = axes[r][c]; style(ax)
                if truth is not None:
                    ax.imshow(truth, origin='lower', extent=[0, 1, 0, 1],
                              cmap=cmap, alpha=.5, vmin=0)
                d = (e or {'m': {}})['m'].get(mi)
                if d is None:
                    ax.text(.5, .5, '—', ha='center', va='center', color='#bbb')
                else:
                    b = np.asarray(d['bahn'])
                    ax.plot(b[:, 0], b[:, 1], color=FARBE[mi], lw=1.4, alpha=.95)
                    ax.set_xlabel(f"{d['ergodic']:.4f}", color='#555',
                                  fontsize=7.2, labelpad=1.4)
                if r == 0:
                    ax.set_title(mi, color=FARBE[mi], fontsize=9.5,
                                 fontweight='semibold')
            axes[r][0].set_ylabel(MNAME.get(mo, mo), color='#1A1A2E', fontsize=9)
        fig.suptitle(f'{a.shape} — sieben Zieldichten × sieben Missionen, '
                     f'{n} Vorabmessungen   (unter jedem Bild: ergodischer Fehler)',
                     color='#1A1A2E', fontsize=11.5)
        fig.tight_layout(rect=[0, 0, 1, 1 - 0.05 / len(vorhanden) * 2.1])
        out = f'{a.out_dir}/bahnen_{a.shape}_n{n}.png'
        fig.savefig(out, dpi=132, facecolor='white'); plt.close(fig)
        print(f'[viz] {out}')

    # ── 2) Waermekarte der Zahlen ────────────────────────────────────────
    import matplotlib.colors as mc
    seq = mc.LinearSegmentedColormap.from_list(
        'blau', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
    fig, axes = plt.subplots(1, len(a.priors),
                             figsize=(4.6 * len(a.priors), 4.4), facecolor='white')
    axes = np.atleast_1d(axes)
    miss_all = [x for x in MISSIONEN
                if any(x in daten[(vorhanden[0], a.priors[0])].get(k, {'m': {}})['m']
                       for k in daten[(vorhanden[0], a.priors[0])])]
    for j, n in enumerate(a.priors):
        M = np.full((len(vorhanden), len(miss_all)), np.nan)
        for r, mo in enumerate(vorhanden):
            for c, mi in enumerate(miss_all):
                v = [f['m'][mi][a.groesse] for f in daten[(mo, n)].values()
                     if mi in f['m']]
                if v:
                    M[r, c] = float(np.mean(v))
        ax = axes[j]
        vmin, vmax = np.nanmin(M), np.nanpercentile(M, 92)
        ax.imshow(M, cmap=seq, vmin=vmin, vmax=vmax, aspect='auto')
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                if np.isnan(M[r, c]):
                    continue
                hell = (M[r, c] - vmin) / max(vmax - vmin, 1e-9) > .55
                ax.text(c, r, f'{M[r, c]:.4f}', ha='center', va='center',
                        fontsize=7.6, color='white' if hell else '#1A1A2E')
        ax.set_xticks(range(len(miss_all)))
        ax.set_xticklabels(miss_all, rotation=38, ha='right', fontsize=8.5)
        ax.set_yticks(range(len(vorhanden)))
        ax.set_yticklabels([MNAME.get(m, m) for m in vorhanden], fontsize=8.5)
        ax.set_title(f'{n} Vorabmessungen', fontsize=11, color='#1A1A2E')
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f'{a.groesse} über alle zwölf Formen — dunkler ist schlechter',
                 fontsize=12.5, color='#1A1A2E')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f'{a.out_dir}/matrix_{a.groesse}.png'
    fig.savefig(out, dpi=140, facecolor='white'); plt.close(fig)
    print(f'[viz] {out}')


if __name__ == '__main__':
    main()
