r"""
export_offset_check_viz.py
===========================
Datenexport fuer eine interaktive Vorher/Nachher-Ansicht zweier Fixes in
`project_db_3d.py`: fuer eine Stichprobe von Formen je Oberflaeche wird die
Bahn sowohl aus der DB vor dem Fehlschuss-Fix (Fallback auf den
naechstgelegenen Oberflaechenpunkt, riss lange Spruenge in die Bahn) als auch
aus der reparierten DB (Fehlschuss-Rohpunkte verworfen statt ausgewichen,
plus projizierter Startpunkt) geladen — zusammen mit den
Zielverteilungs-Partikeln und einer neutralen Kontext-Punktwolke der Flaeche.

Nur fuer den einmaligen Export in eine JSON-Datei gedacht, die ein
HTML/Three.js-Artefakt zum Durchklicken einliest. Kein Teil der Trainings- oder
Datenbank-Pipeline.

    python export_offset_check_viz.py
"""
import json
import os
import sqlite3
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'thesis_architecture', 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import surfaces  # noqa: E402

# Vorher: Fehlschuss-Fallback auf den naechstgelegenen Punkt (Bug).
DB_OLD = os.path.join(_here, 'ergodic_dataset_3d_PREJUMPFIX_tmp.db')
# Nachher: Fehlschuss-Rohpunkte verworfen, plus Startpunkt (Fix).
DB_NEW = os.path.join(_here, 'ergodic_dataset_3d_no_offset.db')
OUT = os.path.join(_here, 'results', 'db3d', 'offset_check_viz.json')

N_SURFACE_CTX = 2500     # neutrale Kontext-Punktwolke je Oberflaeche
N_PER_SURFACE = 4        # Formen je Oberflaeche in der Stichprobe
ROUND = 4


def _hat_spalte(con, name):
    cols = [r[1] for r in con.execute('PRAGMA table_info(ergodic_pairs_3d)')]
    return name in cols


def lade_eintrag(con, shape_name, surface):
    start_sel = ', start_pos' if _hat_spalte(con, 'start_pos') else ''
    row = con.execute(
        f'''SELECT traj_pos, particles, miss_frac, standoff_mean, standoff_sd,
                  jump_max, perp_max_deg, hit_frac{start_sel}
           FROM ergodic_pairs_3d WHERE shape_name=? AND surface=?''',
        (shape_name, surface)).fetchone()
    if row is None:
        return None
    pos = np.frombuffer(row[0], dtype=np.float32).reshape(-1, 3)
    parts = np.frombuffer(row[1], dtype=np.float32).reshape(-1, 4)
    start = np.frombuffer(row[8], dtype=np.float32) if start_sel else None
    return dict(pos=pos, parts=parts, start=start,
                guete=dict(miss_frac=row[2], standoff_mean=row[3],
                          standoff_sd=row[4], jump_max=row[5],
                          perp_max_deg=row[6], hit_frac=row[7],
                          n_points=len(pos)))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    con_old = sqlite3.connect(DB_OLD)
    con_new = sqlite3.connect(DB_NEW)

    surface_ctx = {}
    entries = []
    for sk in surfaces.KEYS:
        surf = surfaces.build(sk)
        P, _ = surf.sample(N_SURFACE_CTX, seed=0)
        surface_ctx[sk] = dict(label=surf.label,
                               pts=P.round(ROUND).tolist())

        shapes = [r[0] for r in con_new.execute(
            'SELECT DISTINCT shape_name FROM ergodic_pairs_3d WHERE surface=? '
            'ORDER BY id ASC', (sk,)).fetchall()]
        stride = max(1, len(shapes) // N_PER_SURFACE)
        wahl = shapes[::stride][:N_PER_SURFACE]

        for nm in wahl:
            neu = lade_eintrag(con_new, nm, sk)
            alt = lade_eintrag(con_old, nm, sk)
            if neu is None or alt is None:
                continue
            entries.append(dict(
                shape=nm, surface=sk,
                pos_new=neu['pos'].round(ROUND).tolist(),
                pos_old=alt['pos'].round(ROUND).tolist(),
                parts=neu['parts'].round(ROUND).tolist(),
                start=(neu['start'].round(ROUND).tolist()
                      if neu['start'] is not None else None),
                guete_new={k: round(v, 5) for k, v in neu['guete'].items()},
                guete_old={k: round(v, 5) for k, v in alt['guete'].items()},
            ))
        print(f"  {sk:16s} {len(wahl)} Formen exportiert")

    con_old.close()
    con_new.close()

    with open(OUT, 'w') as f:
        json.dump(dict(surfaces=surface_ctx, entries=entries), f)
    size_mb = os.path.getsize(OUT) / 2**20
    print(f"\n{len(entries)} Eintraege, {len(surface_ctx)} Oberflaechen "
          f"-> {OUT} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
