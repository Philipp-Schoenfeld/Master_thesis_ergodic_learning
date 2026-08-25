r"""
plot_startglaube.py
===================
Die Ausgangslage einer Mission sichtbar machen: was zu Beginn bekannt ist,
was unbekannt ist, und was daraus als Zieldichte wird.

Reproduziert exakt den Anfangszustand aus `apply_cfm_belief.py` — dieselben
Zufallszahlen, dasselbe Messrauschen, derselbe GP. Fuenf Spalten je Form:

    Wahrheit    das Feld, das der Roboter nicht kennt
    Messungen   die `--n_prior` Vorabmessungen, die er hat
    mu          was er daraus zu wissen glaubt
    sigma       wo er nichts weiss  (hell = unerforscht)
    Phi         mu + kappa*sigma — was das Netz tatsaechlich bekommt

    python plot_startglaube.py --shapes 4 --n_prior 12 --kappa 3.0
"""
import argparse, os, sys
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here),
           os.path.join(os.path.dirname(_here), 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.acquisition import ucb_density, is_degenerate
from common.belief import GPBelief
from common.data import load_truth
from common.observation import measure


def white_inferno():
    import matplotlib.colors as mc, matplotlib.pyplot as plt
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mc.LinearSegmentedColormap.from_list('white_inferno', inf)


def white_blue():
    """Eigene Skala fuer die Unsicherheit.

    sigma ist keine Zieldichte, und sie mit WHITE_INFERNO zu zeichnen laesst
    unerforschtes Gebiet wie einen Dichte-Hotspot aussehen — genau
    entgegengesetzt zur Bedeutung. Weiss heisst hier "schon gemessen",
    Blau heisst "unbekannt".
    """
    import matplotlib.colors as mc
    return mc.LinearSegmentedColormap.from_list(
        'white_blue', ['#FFFFFF', '#BBD6F2', '#5B93D6', '#1F4E8C', '#10294D'])


def style(ax):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(alpha=0.2); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=4)
    p.add_argument('--n_prior', type=int, nargs='+', default=[12])
    p.add_argument('--kappa', type=float, default=3.0)
    p.add_argument('--truth_res', type=int, default=128)
    p.add_argument('--gp_res', type=int, default=64)
    p.add_argument('--lengthscale', type=float, default=0.08)
    p.add_argument('--gp_noise', type=float, default=0.05)
    p.add_argument('--noise', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=os.path.join(_here, 'results', 'cfm_belief',
                                                 'startglaube.png'))
    a = p.parse_args()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = white_inferno()
    blue = white_blue()

    names, truths = load_truth(n=a.shapes, split='val',
                               resolution=a.truth_res, device='cpu')

    nrow = len(names) * len(a.n_prior)
    fig, axes = plt.subplots(nrow, 5, figsize=(13.2, 2.72 * nrow),
                             facecolor='white', squeeze=False)
    titles = ['Wahrheit\n(kennt der Roboter nicht)',
              'Vorabmessungen\n(alles, was er hat)',
              r'$\mu$ — was er zu wissen glaubt',
              r'$\sigma$ — wo er nichts weiß',
              r'$\Phi=\mu+\kappa\sigma$ ($\kappa$=%.1f)' % a.kappa]

    r = 0
    for npri in a.n_prior:
        for i, name in enumerate(names):
            truth = truths[i]
            # identisch zu apply_cfm_belief.main()
            b = GPBelief(grid_res=a.gp_res, lengthscale=a.lengthscale,
                         noise=a.gp_noise, device='cpu')
            g = torch.Generator().manual_seed(a.seed * 977 + i)
            pp = torch.rand(npri, 2, generator=g)
            if npri > 0:
                _, vv = measure(pp, truth, noise_std=a.noise)
                b.observe(pp, vv)
            else:
                # Ohne jede Vormessung ist mu = 0 und sigma ueberall gleich.
                # Phi ist dann fuer *jedes* kappa gleichverteilt — Erkundung
                # ist in diesem Zustand nicht definiert.
                vv = torch.zeros(0)

            mu, sd = b.posterior_grid()
            phi = ucb_density(mu, sd, kappa=a.kappa, norm='max')
            deg = is_degenerate(phi)

            fields = [truth.numpy(), truth.numpy(), mu.numpy(), sd.numpy(),
                      phi.numpy()]
            for c in range(5):
                ax = axes[r][c]; style(ax)
                f = fields[c]
                al = 0.18 if c == 1 else 0.55
                cm = blue if c == 3 else cmap
                ax.imshow(f, origin='lower', extent=[0, 1, 0, 1], cmap=cm,
                          alpha=0.85 if c == 3 else al, vmin=0.0,
                          vmax=1.0 if c in (0, 1, 4) else None)
                if c == 1:
                    v = vv.numpy()
                    ax.scatter(pp[:, 0], pp[:, 1], c=v, cmap=cmap, vmin=0,
                               vmax=1, s=110, edgecolors='#1A1A2E',
                               linewidths=1.2, zorder=3)
                if r == 0:
                    ax.set_title(titles[c], color='#1A1A2E', fontsize=10.5)
            lab = f'{name}\n{npri} Messungen'
            axes[r][0].set_ylabel(lab, color='#1A1A2E', fontsize=10.5)

            n_hit = int((vv > 0.15).sum()) if npri > 0 else 0
            axes[r][1].set_xlabel(
                (f'{n_hit} von {npri} treffen die Form' if npri > 0
                 else 'keine einzige Messung'), color='#555', fontsize=9)
            axes[r][3].set_xlabel(
                f'$\\sigma$ ∈ [{sd.min():.2f}, {sd.max():.2f}]  —  '
                f'{100 * float((sd > 0.95 * sd.max()).float().mean()):.0f} % '
                'unberührt', color='#555', fontsize=9)
            # Wie viel von Phi kommt aus der Unsicherheit statt aus Wissen?
            denom = float((mu.clamp(min=0) + a.kappa * sd).sum())
            share = float((a.kappa * sd).sum()) / max(denom, 1e-12)
            axes[r][4].set_xlabel(
                ('entartet: gleichverteilt' if deg else
                 f'{share * 100:.0f} % davon ist $\\kappa\\sigma$, '
                 f'nicht $\\mu$'), color='#555', fontsize=9)
            r += 1

    fig.suptitle('Der Glaube zu Missionsbeginn — was bekannt ist und '
                 'was unerforscht bleibt', color='#1A1A2E', fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.035 / nrow * 4])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=132, facecolor='white')
    print(f'[viz] {a.out}')


if __name__ == '__main__':
    main()
