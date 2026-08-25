r"""
plot_phi_wirkung.py
===================
Was die Modellierung der Zieldichte in der gefahrenen Bahn bewirkt.

Links: wie gut die Zieldichte auf den wahren Traeger zeigt (ohne Netz
gerechnet). Rechts: der ergodische Fehler der tatsaechlich gefahrenen Bahn
gegen die wahre Dichte. Die Frage ist beide Male dieselbe — wird es besser,
wenn man mehr misst?
"""
import argparse, csv, os
import numpy as np

MOD = ['ucb', 'mi', 'ei', 'lse', 'mass', 'eid']
FARBE = {'ucb': '#e34948', 'mass': '#2a78d6', 'lse': '#1baf7a',
         'mi': '#eda100', 'eid': '#4a3aa7', 'ei': '#eb6834',
         'stretch': '#9A9AAC'}
NAME = {'ucb': 'UCB (bisher)', 'mass': 'Massenanteil', 'lse': 'Niveaumenge',
        'mi': 'GP-MI', 'eid': 'Informationsdichte', 'ei': 'Erw. Zugewinn',
        'stretch': 'UCB gespreizt'}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kennzahlen', default='results/phi_modelle/kennzahlen.csv')
    p.add_argument('--lauf', default='results/phi_lauf')
    p.add_argument('--out', default='results/phi_modelle/wirkung.png')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    kz = list(csv.DictReader(open(a.kennzahlen)))
    priors = sorted({int(r['n_prior']) for r in kz})

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3), facecolor='white')

    # (1) Zielgenauigkeit der Dichte, ohne Netz
    ax = axes[0]
    for m in MOD:
        y = [np.mean([float(r['treffer']) for r in kz
                      if r['modell'] == m and int(r['n_prior']) == n])
             for n in priors]
        if not np.isfinite(y).all():
            continue
        ax.plot(priors, y, 'o-', color=FARBE[m], lw=2.1, ms=5,
                label=NAME.get(m, m))
    fl = np.mean([float(r['traeger']) for r in kz])
    ax.axhline(fl, color='#888', ls='--', lw=1.4)
    ax.annotate('gleichverteilt', (priors[-1], fl), textcoords='offset points',
                xytext=(-6, 5), ha='right', color='#666', fontsize=9)
    ax.set_title('Zieldichte: Anteil auf dem wahren Träger\n(größer ist besser)',
                 fontsize=11, color='#1A1A2E')

    # (2) und (3) Ergodischer Fehler der gefahrenen Bahn
    for j, mis in enumerate(('glaube-1', 'glaube-R')):
        ax = axes[j + 1]
        for m in MOD:
            xs, ys = [], []
            for n in (12, 60):
                f = f'{a.lauf}/{m}_n{n}/metriken.csv'
                if not os.path.exists(f):
                    continue
                r = [x for x in csv.DictReader(open(f)) if x['mission'] == mis]
                if r:
                    xs.append(n)
                    ys.append(np.mean([float(x['ergodic']) for x in r]))
            if len(xs) == 2:
                ax.plot(xs, ys, 'o-', color=FARBE[m], lw=2.1, ms=6)
                d = (ys[0] - ys[1]) / ys[0] * 100
                ax.annotate(f'{d:+.0f} %', (xs[1], ys[1]),
                            textcoords='offset points', xytext=(8, -3),
                            color=FARBE[m], fontsize=9.5, fontweight='semibold')
        ax.set_title(f'Gefahrene Bahn: ergodischer Fehler\n{mis} '
                     '(kleiner ist besser)', fontsize=11, color='#1A1A2E')
        ax.set_xlim(6, 74); ax.set_xticks([12, 60])

    for ax in axes:
        ax.set_xlabel('Vorabmessungen', color='#555')
        ax.grid(alpha=.22)
        for s in ax.spines.values():
            s.set_color('#ccc')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Sieben Modellierungen der Zieldichte — was sie aus mehr '
                 'Messungen machen', fontsize=13, color='#1A1A2E')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor='white')
    print(f'[viz] {a.out}')


if __name__ == '__main__':
    main()
