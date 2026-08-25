r"""
plot_phi_erklaerung.py
======================
Die sieben Zieldichten an einem eindimensionalen Beispiel.

In zwei Dimensionen sieht man *dass* sich die Modelle unterscheiden, aber nur
schwer *wie*. An einem Schnitt durch einen Glauben mit drei Messungen und zwei
Huegeln wird jede Formel unmittelbar lesbar: wo hebt sie an, wo drueckt sie ab,
und was macht sie mit dem Sockel, den sigma vor der Fahrt ueberall bildet.
"""
import argparse, os
import numpy as np


def gp(x_obs, y_obs, xs, ls=0.09, var=1.0, noise=0.05):
    k = lambda a, b: var * np.exp(-0.5 * ((a[:, None] - b[None, :]) / ls) ** 2)
    K = k(x_obs, x_obs) + noise ** 2 * np.eye(len(x_obs))
    L = np.linalg.cholesky(K)
    al = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    Ks = k(xs, x_obs)
    mu = Ks @ al
    v = np.linalg.solve(L, Ks.T)
    sd = np.sqrt(np.clip(var - (v ** 2).sum(0), 1e-12, None))
    return mu, sd


def modelle(mu, sd, kappa=3.0, w=0.75, tau=0.25, xi=0.01, gamma=1.0, noise=0.05):
    from scipy.stats import norm
    n = lambda a: np.clip(a, 0, None) + 1e-6
    mx = lambda a: n(a) / n(a).max()
    out = {}
    out['ucb'] = mx(mu + kappa * sd)
    p = mu + kappa * sd
    out['stretch'] = mx((p - p.min()) / max(p.ptp(), 1e-12))
    m_, s_ = n(mu) / n(mu).sum(), n(sd) / n(sd).sum()
    out['mass'] = mx((1 - w) * m_ + w * s_)
    out['mi'] = mx(mu + kappa * (np.sqrt(gamma + sd ** 2) - np.sqrt(gamma)))
    b = mu.max()
    z = (mu - b - xi) / np.clip(sd, 1e-9, None)
    out['ei'] = mx((mu - b - xi) * norm.cdf(z) + sd * norm.pdf(z))
    out['lse'] = mx(norm.cdf((mu - tau) / np.clip(sd, 1e-9, None)))
    g = np.gradient(mu) ** 2 + np.gradient(sd) ** 2
    e_ = n(g) / n(g).sum()
    out['eid'] = mx(m_ + kappa * e_)
    return out


TITEL = {'ucb': r'UCB   $\mu+\kappa\sigma$', 'stretch': 'UCB, gespreizt',
         'mass': r'Massenanteil   $(1-w)\hat\mu+w\hat\sigma$', 'mi': 'GP-MI',
         'ei': 'Erwarteter Zugewinn', 'lse': r'Niveaumenge   $P(f>\tau)$',
         'eid': 'Informationsdichte (EID)'}
ORDER = ['ucb', 'stretch', 'mass', 'mi', 'ei', 'lse', 'eid']
FARBE = {'ucb': '#e34948', 'stretch': '#9A9AAC', 'mass': '#2a78d6',
         'mi': '#eda100', 'ei': '#eb6834', 'lse': '#1baf7a', 'eid': '#4a3aa7'}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='results/phi_modelle/erklaerung.png')
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xs = np.linspace(0, 1, 400)
    wahr = (0.95 * np.exp(-0.5 * ((xs - 0.24) / 0.055) ** 2)
            + 0.62 * np.exp(-0.5 * ((xs - 0.63) / 0.075) ** 2))
    xo = np.array([0.10, 0.26, 0.40])          # nur links gemessen
    yo = np.interp(xo, xs, wahr) + np.array([0.01, -0.02, 0.01])
    mu, sd = gp(xo, yo, xs)

    fig, axes = plt.subplots(2, 4, figsize=(15.4, 6.4), facecolor='white',
                             sharex=True)
    ax = axes[0][0]
    ax.fill_between(xs, np.clip(mu - sd, -.2, None), mu + sd, color='#5B93D6',
                    alpha=.22, lw=0, label=r'$\mu\pm\sigma$')
    ax.plot(xs, wahr, color='#1A1A2E', lw=2, label='Wahrheit')
    ax.plot(xs, mu, color='#1F4E8C', lw=2, label=r'$\mu$')
    ax.plot(xo, yo, 'o', color='#1A1A2E', ms=6, label='Messungen')
    ax.set_title('Der Glaube', fontsize=11.5, color='#1A1A2E')
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax.set_ylim(-0.25, 1.45)

    M = modelle(mu, sd)
    for i, k in enumerate(ORDER):
        ax = axes[(i + 1) // 4][(i + 1) % 4]
        ax.plot(xs, wahr, color='#bbb', lw=1.4)
        ax.fill_between(xs, 0, M[k], color=FARBE[k], alpha=.28, lw=0)
        ax.plot(xs, M[k], color=FARBE[k], lw=2.1)
        for x in xo:
            ax.axvline(x, color='#1A1A2E', lw=.7, alpha=.35, ls=':')
        anteil = M[k][xs > 0.5].sum() / M[k].sum()
        ax.set_title(TITEL[k], fontsize=10.5, color=FARBE[k])
        ax.set_xlabel(f'{anteil*100:.0f} % der Masse rechts der Messungen',
                      fontsize=8.6, color='#555')
        ax.set_ylim(-0.05, 1.15)

    for ax in axes.ravel():
        ax.grid(alpha=.2); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#ccc')
    fig.suptitle('Derselbe Glaube, sieben Zieldichten — gemessen wurde nur '
                 'links (gepunktete Linien)', fontsize=13, color='#1A1A2E')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor='white')
    print(f'[viz] {a.out}')
    for k in ORDER:
        print(f'  {k:8s} Masse rechts: {M[k][xs>0.5].sum()/M[k].sum()*100:5.1f} %')


if __name__ == '__main__':
    main()
