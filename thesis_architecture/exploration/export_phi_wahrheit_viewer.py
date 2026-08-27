r"""
export_phi_wahrheit_viewer.py
==============================
Das Phi-Kreuz (Grundwahrheit-Variante, `apply_cfm_belief.py --prior_mode
wahrheit`) in eine Form bringen, die sich im Browser auswaehlen laesst.

Liest alle `results/phi_wahrheit[_SUFFIX]/<muster>_<phi>/bahnen.json` und legt
Trajektorien, Zieldichten und Metriken kompakt in einer JSON-Datei ab, damit
der Betrachter zur Laufzeit zusammenstellen kann, was gerade interessiert:
welches Vorwissen-Muster, welche Zieldichte (Phi-Modell), welche Form, welche
Mission.

Kompakt heisst hier: Trajektorien-Koordinaten als Ganzzahlen 0..1000,
Dichtefelder auf 48^2 heruntergerechnet und auf 0..255 quantisiert.
"""
import argparse, json, os, glob
import numpy as np

MUSTER = ['haelfte', 'quadranten', 'loch']
MODELLE = ['ucb', 'stretch', 'mass', 'ei', 'lse', 'mi', 'eid']
MISSIONEN = ['orakel', 'glaube-1', 'glaube-R', 'zweistufig', 'glaube-D', 'B-warm', 'maeher']


def q(x, lo=0.0, hi=1.0, n=1000):
    return np.clip(np.round((np.asarray(x, dtype=np.float64) - lo) / (hi - lo) * n), 0, n).astype(int)


def verkleinern(feld, ziel=48):
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
    p.add_argument('--root', default='results/phi_wahrheit')
    p.add_argument('--traj_pts', type=int, default=220)
    p.add_argument('--feld', type=int, default=48)
    p.add_argument('--out', default='results/phi_wahrheit/viewer.json')
    a = p.parse_args()

    formen, truth, maske, phi, traj, metrik, anytime = [], {}, {}, {}, {}, {}, {}
    gefunden = 0
    for mu in MUSTER:
        for mo in MODELLE:
            f = f'{a.root}/{mu}_{mo}/bahnen.json'
            if not os.path.exists(f):
                print(f'  (fehlt: {mu}_{mo})')
                continue
            gefunden += 1
            D = json.load(open(f))
            AT_PATH = f'{a.root}/{mu}_{mo}/anytime.json'
            AT = json.load(open(AT_PATH)) if os.path.exists(AT_PATH) else {}
            for shape_idx, e in enumerate(D['formen']):
                nm = e['name']
                if nm not in truth:
                    formen.append(nm)
                    t = verkleinern(e['truth'], a.feld)
                    tmax = float(np.max(t)) or 1e-12
                    truth[nm] = q(t / tmax, n=255).flatten().tolist()
                mk = verkleinern(e['maske'], a.feld)
                maske[f'{mu}|{nm}'] = q(mk, n=1).flatten().tolist()

                for mi, d in e['missionen'].items():
                    b = np.asarray(d['bahn'])
                    idx = np.linspace(0, len(b) - 1,
                                      min(a.traj_pts, len(b))).astype(int)
                    k = f'{mu}|{mo}|{nm}|{mi}'
                    traj[k] = q(b[idx]).flatten().tolist()
                    metrik[k] = [round(float(d['coverage']), 5),
                                 round(float(d['ergodic']), 5),
                                 round(float(d['belief_rmse']), 4),
                                 round(float(d['path_len']), 2)]
                    # Repraesentative Zieldichte dieser Zelle: erste Planungsrunde
                    # von glaube-R (haengt an Muster + Modell + Form, nicht Mission).
                    if mi == 'glaube-R' and d.get('phi'):
                        ph = verkleinern(np.asarray(d['phi'][0]), a.feld)
                        phi[f'{mu}|{mo}|{nm}'] = q(ph, n=255).flatten().tolist()

                    # Anytime-Kurve dieser Mission: (Weglaenge, Abdeckung) je
                    # Runde/Abtastpunkt — fuer Vergleiche bei gleichem Budget,
                    # unabhaengig davon wie lang die volle Bahn tatsaechlich ist.
                    rounds = AT.get(mi)
                    if rounds and shape_idx < len(rounds) and rounds[shape_idx]:
                        curve = rounds[shape_idx]
                        flat = []
                        for r in curve:
                            flat.append(round(float(r['path_len']), 3))
                            flat.append(round(float(r['coverage']), 5))
                            flat.append(round(float(r.get('belief_rmse', 0.0)), 4))
                        anytime[k] = flat

    formen = sorted(formen)
    out = dict(formen=formen, muster=MUSTER, modelle=MODELLE, missionen=MISSIONEN,
               feld=a.feld, metrik_felder=['coverage', 'ergodic', 'belief_rmse', 'path_len'],
               truth=truth, maske=maske, phi=phi, traj=traj, metrik=metrik, anytime=anytime)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    print(f'{gefunden} Zellen gelesen, {len(traj)} Trajektorien, {len(phi)} Zieldichten, '
          f'{len(formen)} Formen')
    print(f'[json] {a.out}  ({os.path.getsize(a.out)/2**20:.2f} MiB)')


if __name__ == '__main__':
    main()
