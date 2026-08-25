r"""
export_kreuz_viewer.py
======================
Das Phi-Kreuz in eine Form bringen, die sich im Browser auswaehlen laesst.

1512 Trajektorien gleichzeitig zu zeigen hilft niemandem. Diese Ausgabe legt
alle davon kompakt ab, damit der Betrachter zur Laufzeit zusammenstellen kann,
was gerade interessiert: welche Zieldichten, welche Missionen, wie viel
Vorwissen, welche Form.

Kompakt heisst hier: Koordinaten als Ganzzahlen 0…1000, Dichtefelder auf 48²
heruntergerechnet und auf 0…255 quantisiert. Bei einer Zeichenflaeche von
wenigen hundert Pixeln ist beides unter der Aufloesungsgrenze.
"""
import argparse, json, os
import numpy as np

MODELLE = ['ucb', 'stretch', 'mass', 'mi', 'ei', 'lse', 'eid']
MISSIONEN = ['orakel', 'glaube-1', 'glaube-R', 'zweistufig', 'B-warm', 'maeher']


def q(x, lo=0.0, hi=1.0, n=1000):
    return np.clip(np.round((np.asarray(x) - lo) / (hi - lo) * n), 0, n).astype(int)


def verkleinern(feld, ziel=48):
    """Mittelwert-Pooling auf `ziel`² — schnell und ohne Abhaengigkeiten."""
    a = np.asarray(feld, dtype=np.float64)
    r = a.shape[0]
    if r == ziel:
        return a
    k = r // ziel
    if k * ziel == r:
        return a.reshape(ziel, k, ziel, k).mean(axis=(1, 3))
    idx = np.linspace(0, r - 1, ziel).astype(int)
    return a[np.ix_(idx, idx)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='results/phi_kreuz')
    p.add_argument('--priors', type=int, nargs='+', default=[12, 30, 60])
    p.add_argument('--traj_pts', type=int, default=96)
    p.add_argument('--feld', type=int, default=48)
    p.add_argument('--out', default='results/phi_kreuz/viewer.json')
    a = p.parse_args()

    formen, truth, phi, traj, metrik = [], {}, {}, {}, {}
    for m in MODELLE:
        for n in a.priors:
            f = f'{a.root}/{m}_n{n}/bahnen.json'
            if not os.path.exists(f):
                print(f'  (fehlt: {m}_n{n})'); continue
            D = json.load(open(f))
            for e in D['formen']:
                nm = e['name']
                if nm not in truth:
                    formen.append(nm)
                    t = verkleinern(e['truth'], a.feld)
                    truth[nm] = q(t / max(t.max(), 1e-12), n=255).flatten().tolist()
                # Die Zieldichte der ersten Runde — sie haengt an Modell,
                # Vorwissen und Form, nicht an der Mission.
                gr = e['missionen'].get('glaube-R')
                if gr and gr.get('phi'):
                    ph = verkleinern(gr['phi'][0], a.feld)
                    phi[f'{m}|{n}|{nm}'] = q(ph, n=100).flatten().tolist()
                for mi, d in e['missionen'].items():
                    b = np.asarray(d['bahn'])
                    idx = np.linspace(0, len(b) - 1,
                                      min(a.traj_pts, len(b))).astype(int)
                    k = f'{m}|{n}|{nm}|{mi}'
                    traj[k] = q(b[idx]).flatten().tolist()
                    metrik[k] = [round(d['ergodic'], 5), round(d['coverage'], 5),
                                 round(d['path_len'], 2),
                                 round(d['belief_rmse'], 4)]

    out = dict(formen=formen, modelle=MODELLE, missionen=MISSIONEN,
               priors=a.priors, feld=a.feld, truth=truth, phi=phi,
               traj=traj, metrik=metrik)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'{len(traj)} Trajektorien, {len(phi)} Zieldichten, '
          f'{len(formen)} Formen')
    print(f'[json] {a.out}  ({os.path.getsize(a.out)/2**20:.2f} MiB)')


if __name__ == '__main__':
    main()
