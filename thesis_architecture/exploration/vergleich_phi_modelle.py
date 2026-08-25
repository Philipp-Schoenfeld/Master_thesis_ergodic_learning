r"""
vergleich_phi_modelle.py
========================
Wie stark haengt die Zieldichte ueberhaupt an den Messungen?

Diese Auswertung braucht **kein Netz und keine GPU**. Sie misst allein, was aus
dem Glauben als Zieldichte herauskommt — und beantwortet damit die Frage, warum
12, 30 oder 60 Vorabmessungen im gefahrenen Ergebnis kaum auseinandergehen.

Drei Kennzahlen je Modell und Messmenge:

    treffer   Anteil der Zielmasse, der auf dem *wahren* Traeger liegt.
              Eine gleichverteilte Zieldichte erreicht hier den Flaechenanteil
              des Traegers (etwa 0,2); eine perfekte 1,0. Das ist die Zahl, die
              sagt, ob Phi ueberhaupt auf die Form zeigt.
    korr      Pearson-Korrelation zwischen Phi und der Wahrheit.
    sigma%    Anteil der Zielmasse, der aus dem Unsicherheitsterm stammt.
              Nur dort definiert, wo sich das Modell additiv zerlegen laesst.

Und die eigentliche Frage: wie viel gewinnt `treffer` zwischen der kleinsten
und der groessten Messmenge? Ein Modell, bei dem das nahe null bleibt, kann
im gefahrenen Ergebnis keinen Unterschied erzeugen — egal wie gut der Planer ist.

    python vergleich_phi_modelle.py --shapes 12
"""
import argparse, json, os, sys
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.acquisition import phi_from_belief, PHI_MODELLE
from common.belief import GPBelief
from common.data import load_truth
from common.observation import measure

# Voreinstellungen je Modell — bewusst so gewaehlt, dass jedes Modell in seiner
# eigenen Groesse "mittelstark erkundet", damit der Vergleich nicht an einer
# willkuerlich schlechten Einstellung eines Konkurrenten haengt.
PARAMS = {'ucb': dict(kappa=3.0), 'stretch': dict(kappa=3.0),
          'mass': dict(w=0.5), 'ei': dict(xi=0.01), 'lse': dict(tau=0.25),
          'mi': dict(kappa=3.0, gamma=1.0),
          'eid': dict(kappa=3.0, noise=0.05)}


def kennzahlen(phi, truth, mu, sd, kappa=3.0, schwelle=0.15):
    """treffer, korr, sigma-Anteil."""
    t = truth
    if t.shape != phi.shape:
        t = torch.nn.functional.interpolate(truth[None, None], size=phi.shape,
                                            mode='bilinear',
                                            align_corners=True)[0, 0]
    supp = (t > schwelle)
    p = phi / phi.sum().clamp(min=1e-12)
    treffer = float(p[supp].sum())
    a, b = phi.reshape(-1), t.reshape(-1)
    korr = float(((a - a.mean()) * (b - b.mean())).mean()
                 / (a.std().clamp(min=1e-12) * b.std().clamp(min=1e-12)))
    anteil = float((kappa * sd).sum() / (mu.clamp(min=0) + kappa * sd).sum())
    return treffer, korr, anteil, float(supp.float().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shapes', type=int, default=12)
    p.add_argument('--priors', type=int, nargs='+', default=[12, 30, 60])
    p.add_argument('--modelle', nargs='+', default=sorted(PHI_MODELLE))
    p.add_argument('--gp_res', type=int, default=64)
    p.add_argument('--truth_res', type=int, default=128)
    p.add_argument('--lengthscale', type=float, default=0.08)
    p.add_argument('--gp_noise', type=float, default=0.05)
    p.add_argument('--noise', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'phi_modelle'))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    names, T = load_truth(n=a.shapes, split='val', resolution=a.truth_res)
    print(f"{len(names)} Formen, {len(a.priors)} Messmengen, "
          f"{len(a.modelle)} Modelle\n")

    rows, felder = [], {}
    for i, nm in enumerate(names):
        truth = T[i]
        for n in a.priors:
            b = GPBelief(grid_res=a.gp_res, lengthscale=a.lengthscale,
                         noise=a.gp_noise)
            g = torch.Generator().manual_seed(a.seed * 977 + i)
            pp = torch.rand(n, 2, generator=g)
            _, vv = measure(pp, truth, noise_std=a.noise)
            b.observe(pp, vv)
            mu, sd = b.posterior_grid()
            for m in a.modelle:
                phi = phi_from_belief(mu, sd, modell=m, **PARAMS.get(m, {}))
                tr, ko, an, fl = kennzahlen(phi, truth, mu, sd)
                rows.append(dict(shape=nm, n_prior=n, modell=m, treffer=tr,
                                 korr=ko, sigma_anteil=an, traeger=fl))
                if i < 3:
                    felder.setdefault(nm, {}).setdefault(m, {})[n] = \
                        phi.numpy().round(4).tolist()
            if i < 3:
                felder[nm].setdefault('_wahrheit', truth.numpy().round(4).tolist())
                felder[nm].setdefault('_mess', {})[n] = pp.numpy().round(4).tolist()

    import csv
    cp = os.path.join(a.out_dir, 'kennzahlen.csv')
    with open(cp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out_dir, 'felder.json'), 'w') as f:
        json.dump(felder, f)

    fl = float(np.mean([r['traeger'] for r in rows]))
    print(f"Der wahre Traeger belegt im Mittel {fl * 100:.1f} % der Flaeche.")
    print("Eine gleichverteilte Zieldichte erreicht also treffer = "
          f"{fl:.3f}.\n")
    hdr = ''.join(f"{n:>10d}" for n in a.priors)
    print(f"{'Modell':10s} {'':2s}" + hdr + f"{'Zuwachs':>12s}   sigma-Anteil")
    print('-' * (26 + 10 * len(a.priors) + 16))
    best = None
    for m in a.modelle:
        vals = [float(np.mean([r['treffer'] for r in rows
                               if r['modell'] == m and r['n_prior'] == n]))
                for n in a.priors]
        an = float(np.mean([r['sigma_anteil'] for r in rows
                            if r['modell'] == m and r['n_prior'] == a.priors[0]]))
        zuw = (vals[-1] - vals[0]) / max(vals[0], 1e-9) * 100
        print(f"{m:10s} {'':2s}" + ''.join(f"{v:10.3f}" for v in vals)
              + f"{zuw:11.0f} %   {an * 100:12.0f} %")
        if best is None or zuw > best[1]:
            best = (m, zuw)
    print(f"\n  [csv] {cp}")
    print(f"\nGroesster Zuwachs: {best[0]} mit {best[1]:.0f} %")


if __name__ == '__main__':
    main()
