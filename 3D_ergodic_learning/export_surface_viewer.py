r"""
export_surface_viewer.py
========================
Die Ergebnisse aus `bahnen.json` in eine kompakte Form fuer den interaktiven
Betrachter bringen.

Nichts wird neu gerechnet — nur ausgeduennt und quantisiert, damit 84 Szenen
zusammen in eine Seite passen:

  * Koordinaten als Ganzzahlen 0…1000 statt als Fliesskommazahlen. Bei einer
    Szene von einem Meter Kantenlaenge ist das ein Millimeter Aufloesung, weit
    unter allem, was man drehend erkennt.
  * Oberflaechenpunkte ausgeduennt, mit Vorrang fuer die beschrifteten.
  * Die 6D-Rotationen werden hier schon zu Sensorachsen ausgerechnet, damit
    der Betrachter kein Gram-Schmidt braucht.
"""
import argparse, json, os
import numpy as np


def rot6_to_axis(r6):
    """(N,6) -> (N,3): die dritte Spalte der Rotationsmatrix."""
    a, b = r6[:, :3], r6[:, 3:]
    e1 = a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-9)
    b = b - (e1 * b).sum(-1, keepdims=True) * e1
    e2 = b / np.linalg.norm(b, axis=-1, keepdims=True).clip(1e-9)
    return np.cross(e1, e2)


def q(x, lo=0.0, hi=1.0, n=1000):
    return np.clip(np.round((np.asarray(x) - lo) / (hi - lo) * n), 0, n).astype(int)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--json', default='results/surfaces/bahnen.json')
    p.add_argument('--out', default='results/surfaces/viewer.json')
    p.add_argument('--lit', type=int, default=1300,
                   help='beschriftete Oberflaechenpunkte je Szene')
    p.add_argument('--dark', type=int, default=700,
                   help='abgewandte Punkte je Szene')
    p.add_argument('--traj', type=int, default=200)
    p.add_argument('--frames', type=int, default=16)
    a = p.parse_args()

    D = json.load(open(a.json))
    rng = np.random.default_rng(0)
    szenen, formen, flaechen = {}, [], []

    for e in D['eintraege']:
        if e['shape'] not in formen:
            formen.append(e['shape'])
        if e['surface'] not in flaechen:
            flaechen.append(e['surface'])

        P = np.asarray(e['flaeche']); W = np.asarray(e['gewicht'])
        lit = np.flatnonzero(W > 1e-3); dark = np.flatnonzero(W <= 1e-3)
        if len(lit) > a.lit:
            lit = rng.choice(lit, a.lit, replace=False)
        if len(dark) > a.dark:
            dark = rng.choice(dark, a.dark, replace=False)
        keep = np.concatenate([lit, dark])

        B = np.asarray(e['bahn'])
        idx = np.linspace(0, len(B) - 1, min(a.traj, len(B))).astype(int)
        Bs = B[idx]

        ax = None
        if e.get('rot6'):
            R = rot6_to_axis(np.asarray(e['rot6']))
            fi = np.linspace(0, len(R) - 1, a.frames).astype(int)
            bi = np.linspace(0, len(Bs) - 1, a.frames).astype(int)
            ax = dict(p=q(Bs[bi]).flatten().tolist(),
                      d=q(R[fi], -1.0, 1.0).flatten().tolist())

        szenen[f"{e['shape']}|{e['surface']}"] = dict(
            p=q(P[keep]).flatten().tolist(),
            w=np.clip(np.round(W[keep] * 100), 0, 100).astype(int).tolist(),
            b=q(Bs).flatten().tolist(),
            a=ax,
            m={k: round(float(v), 5) for k, v in e['metrik'].items()
               if k not in ('shape', 'surface')})

    out = dict(formen=formen, flaechen=flaechen, szenen=szenen)
    with open(a.out, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    mb = os.path.getsize(a.out) / 2 ** 20
    print(f'{len(szenen)} Szenen, {len(formen)} Formen × {len(flaechen)} Flächen')
    print(f'[json] {a.out}  ({mb:.2f} MiB)')


if __name__ == '__main__':
    main()
