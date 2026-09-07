r"""
export_db_snapshots.py
======================
Beispiele aus der projizierten Datenbank fuer `render_3d_snapshots.py`
aufbereiten — die Sichtpruefung der Datenbank selbst.

Wozu: die Guete-Tabelle sagt, wie oft ein Strahl danebenging. Sie sagt nicht,
ob die Bahn *aussieht* wie eine Bahn auf dieser Flaeche. Ein Buchstabe mit
Loch, ein Torus, ein Helm mit offenen Riemen — dass die Zahlen dort in Ordnung
sind, heisst noch nicht, dass die projizierte Kurve dort liegt, wo sie soll.
Genau ein Beispiel je Gruppe reicht dafuer; hundert Bilder sieht sich niemand
an.

Die volle Dichtewolke wird neu gerechnet, nicht aus der Datenbank gelesen: dort
stehen nur die 512 Konditionierungspartikel, und mit denen sieht man der
Oberflaeche nicht an, wo die Zieldichte liegt. Weil die Blickrichtung je
Eintrag mitgeschrieben ist, laesst sich der Projektor exakt so
wiederherstellen, wie er beim Bau war — das ist der Zweck der Spalten
`view_x/y/z`.

    python export_db_snapshots.py --je_gruppe 1
    python render_3d_snapshots.py --in_dir results/db3d/schnappschuesse
"""
import argparse, json, os, sqlite3, sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'thesis_architecture', 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import surfaces                                            # noqa: E402
import project_db_3d as P                                  # noqa: E402


def _waehle(con, je_gruppe, seed=0):
    """Je Gruppe `je_gruppe` Eintraege — die mit mittlerer Guete.

    Nicht die besten und nicht die schlimmsten: der beste Eintrag einer Gruppe
    beweist nichts ueber die Gruppe, der schlimmste ist schon durch den
    Fehlschussfilter gegangen. Der Median ist der Fall, den die Gruppe
    tatsaechlich meistens liefert.
    """
    wahl = []
    for (g,) in con.execute('SELECT DISTINCT gruppe FROM ergodic_pairs_3d '
                            "WHERE split='train' ORDER BY gruppe"):
        rows = list(con.execute(
            "SELECT id, shape_name, surface, gruppe, view_x, view_y, view_z, "
            "miss_frac, jump_max, hit_frac, perp_max_deg "
            "FROM ergodic_pairs_3d WHERE split='train' AND gruppe=? "
            "ORDER BY miss_frac", (g,)))
        if not rows:
            continue
        mitte = len(rows) // 2
        schritt = max(len(rows) // (je_gruppe + 1), 1)
        for k in range(je_gruppe):
            i = min(max(mitte + (k - je_gruppe // 2) * schritt, 0), len(rows) - 1)
            wahl.append(rows[i])
    return wahl


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', default=P.DB_OUT)
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'db3d',
                                                     'schnappschuesse'))
    p.add_argument('--je_gruppe', type=int, default=1)
    p.add_argument('--n_surface', type=int, default=20000)
    p.add_argument('--dens_res', type=int, default=128)
    p.add_argument('--n_wolke', type=int, default=2600,
                   help='wie viele Oberflaechenpunkte ins Bild kommen')
    p.add_argument('--ziel_dreiecke', type=int, default=2600)
    a = p.parse_args()

    con = sqlite3.connect(a.db)
    surfaces.externe_aufnehmen(verbose=False)
    wahl = _waehle(con, a.je_gruppe)
    print(f'{len(wahl)} Beispiele aus {os.path.basename(a.db)}')

    gitter = P.Dichtegitter(a.dens_res)
    rs = np.random.default_rng(0)
    eintraege, meshes = [], {}

    for (rid, nm, sk, gruppe, vx, vy, vz, miss, jump, hit, perp) in wahl:
        dp = con.execute('SELECT density_params FROM dichten WHERE shape_name=?',
                         (nm,)).fetchone()
        if dp is None:
            print(f'  [!] {nm}: keine Dichte in der Tabelle — uebersprungen')
            continue
        dichte = json.loads(dp[0])
        bp, x0 = con.execute('SELECT traj_pos, x0_2d FROM ergodic_pairs_3d '
                             'WHERE id=?', (rid,)).fetchone()
        pos = np.frombuffer(bp, dtype=np.float32).reshape(-1, 3).astype(float)

        s = surfaces.build(sk)
        view = np.array([vx, vy, vz], dtype=float)
        proj = P.Projektor(s, view, P.Raycaster(s.mesh),
                           n_surface=a.n_surface, dens_res=a.dens_res, seed=0)
        wt = proj.gewicht(gitter(nm, dichte))

        # Beleuchtete Punkte bevorzugt, aber nicht ausschliesslich: ohne die
        # dunklen sieht man die Rueckseite der Flaeche nicht und kann nicht
        # beurteilen, ob die Bahn wirklich aussen liegt.
        hell = np.flatnonzero(wt > 1e-3)
        dunkel = np.flatnonzero(wt <= 1e-3)
        hell = rs.choice(hell, min(len(hell), int(a.n_wolke * 0.68)), replace=False)
        dunkel = rs.choice(dunkel, min(len(dunkel), a.n_wolke - len(hell)),
                           replace=False)
        keep = np.concatenate([hell, dunkel])

        eintraege.append(dict(
            shape=f'{nm}__{gruppe}', surface=sk,
            bahn=pos.round(4).tolist(),
            flaeche=proj.pts[keep].round(4).tolist(),
            gewicht=wt[keep].round(3).tolist(),
            # `render_3d_snapshots` schreibt drei Zahlen in den Titel. Hier gibt
            # es kein erzeugtes Ergebnis zu bewerten, also stehen dort die
            # Guetemasse der Datenbank — beschriftet bleiben sie ohnehin ueber
            # die Namen erg/coverage/pointing, was hier Fehlschuss, beleuchteter
            # Anteil und Abweichung von der Senkrechten heisst.
            metrik=dict(erg=float(miss), coverage=float(hit),
                        pointing_deg=float(perp)),
            gruppe=gruppe, view=[round(float(v), 4) for v in view],
            x0_2d=json.loads(x0) if x0 else None,
            start_pos=pos[0].round(4).tolist()))

        if sk not in meshes:
            from export_meshes import vereinfache
            ziel = 200 if sk.startswith('ebene') else a.ziel_dreiecke
            V, F = vereinfache(s.mesh, ziel)
            q = np.clip(np.round(V * 1000), 0, 1000).astype(int)
            meshes[sk] = dict(v=q.flatten().tolist(), f=F.flatten().tolist(),
                              n=len(V), t=len(F))
        print(f'  {gruppe:10s} {nm[:22]:22s} @ {sk:20s} '
              f'Fehlschuss {miss*100:4.1f} %  Sprung {jump:.3f}  '
              f'beschienen {hit*100:4.1f} %')

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, 'bahnen.json'), 'w') as f:
        json.dump(dict(eintraege=eintraege), f, separators=(',', ':'))
    with open(os.path.join(a.out_dir, 'meshes.json'), 'w') as f:
        json.dump(meshes, f, separators=(',', ':'))
    con.close()
    print(f'\n[json] {a.out_dir}')
    print('  weiter mit: python render_3d_snapshots.py --in_dir '
          f'{os.path.relpath(a.out_dir, _here)}')


if __name__ == '__main__':
    main()
