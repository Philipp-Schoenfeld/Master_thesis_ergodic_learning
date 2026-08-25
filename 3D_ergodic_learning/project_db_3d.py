r"""
project_db_3d.py
================
Aus der 2D-Datenbank eine echte 3D-Datenbank machen: Zielverteilung *und*
Pfad auf gekruemmte Oberflaechen projizieren, der Pfad als SE(3)-Bahn.

Das Verfahren
-------------
Beides — Dichte und Bahn — wird durch **denselben** Projektor geworfen. Damit
bleibt die Beziehung zwischen ihnen erhalten: eine Bahn, die in 2D ergodisch
zu ihrer Dichte lag, liegt nach der Projektion an denselben Stellen relativ
zur projizierten Dichte.

Fuer jeden Bahnpunkt (u,v) wird ein Strahl durch die Projektionsebene in die
Szene geschossen. Der erste Treffer liefert den Oberflaechenpunkt p und die
Flaechennormale n. Daraus:

    Position     x = p + standoff * n          (auf der Aussenseite)
    Blickachse   z = -n                        (senkrecht auf die Flaeche)
    Vorwaerts    x_achse = Tangente, auf die Tangentialebene projiziert
    Rest         y = z x x_achse

Damit ist die Bahn **an jeder Stelle senkrecht zur Oberflaeche** — nicht
naeherungsweise, sondern per Konstruktion.

Wo nichts getroffen wird
------------------------
Eine 2D-Bahn laeuft gelegentlich ueber den Rand der Silhouette hinaus; dort
gibt es keinen Treffer. Statt solche Punkte zu verwerfen (was Luecken in die
Bahn risse) wird auf den *naechstgelegenen* Oberflaechenpunkt zurueckgegriffen.
Der Anteil solcher Punkte wird je Eintrag mitgeschrieben — er ist das
wichtigste Guetemass dieser Datenbank.

    python project_db_3d.py --preview          # 25 Beispiele zum Pruefen
    python project_db_3d.py --build            # die Datenbank schreiben
"""
import argparse, json, os, sqlite3, sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'thesis_architecture', 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import surfaces                                            # noqa: E402

DB_IN = os.path.join(_root, 'thesis_architecture',
                     'ergodic_dataset_generator', 'ergodic_dataset_775.db')
DB_OUT = os.path.join(_here, 'ergodic_dataset_3d.db')


# ── Strahlen ─────────────────────────────────────────────────────────────────
class Raycaster:
    """open3d-Raytracer plus Nächster-Punkt-Anfrage fuer die Fehlschuesse."""

    def __init__(self, mesh):
        import open3d as o3d
        self.o3d = o3d
        tm = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.float32),
            o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.int32))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tm)
        self.mesh = mesh

    def shoot(self, origins, direction):
        o3d = self.o3d
        d = np.tile(np.asarray(direction, np.float32), (len(origins), 1))
        rays = o3d.core.Tensor(np.hstack([origins.astype(np.float32), d]),
                               dtype=o3d.core.float32)
        r = self.scene.cast_rays(rays)
        t = r['t_hit'].numpy()
        nrm = r['primitive_normals'].numpy()
        hit = np.isfinite(t)
        # Fehlschuesse liefern inf; ohne das Ersetzen entstuende inf*0 = nan.
        t_safe = np.where(hit, t, 0.0)
        pts = origins + t_safe[:, None] * np.asarray(direction)[None, :]
        return pts, nrm, hit

    def closest(self, query):
        o3d = self.o3d
        q = o3d.core.Tensor(query.astype(np.float32), dtype=o3d.core.float32)
        r = self.scene.compute_closest_points(q)
        pts = r['points'].numpy()
        nrm = r['primitive_normals'].numpy()
        return pts, nrm


def frames_from_normals(pos, nrm, eps=1e-8):
    """SE(3)-Rahmen: Blickachse entgegen der Normalen, Vorwaerts aus der Tangente.

    -> R (T,3,3) mit den Spalten (x, y, z); z zeigt auf die Flaeche.
    """
    z = -nrm / np.linalg.norm(nrm, axis=-1, keepdims=True).clip(eps)
    tan = np.gradient(pos, axis=0)
    tan = tan - (tan * z).sum(-1, keepdims=True) * z          # in die Tangentialebene
    bad = np.linalg.norm(tan, axis=-1) < 1e-6
    if bad.any():                       # Tangente parallel zur Normalen
        alt = np.tile(np.array([1.0, 0.0, 0.0]), (len(pos), 1))
        flip = np.abs(z[:, 0]) > 0.9
        alt[flip] = np.array([0.0, 1.0, 0.0])
        alt = alt - (alt * z).sum(-1, keepdims=True) * z
        tan[bad] = alt[bad]
    x = tan / np.linalg.norm(tan, axis=-1, keepdims=True).clip(eps)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def matrix_to_rot6(R):
    """Die ersten beiden Spalten — dieselbe Konvention wie `orientation.py`."""
    return np.concatenate([R[..., 0], R[..., 1]], axis=-1)


# ── Ein Eintrag ──────────────────────────────────────────────────────────────
def projiziere(surface, rc, dens2d, xy, standoff=0.12, n_particles=512,
               n_surface=20000, seed=0):
    """-> dict mit Bahn (T,3), Rahmen (T,6), Partikeln (N,4) und Guetemassen."""
    e1, e2, w = surfaces._frame(surface.view)

    # Ausdehnung der Flaeche in der Projektionsebene — dieselbe Bezugsgroesse
    # wie in `surfaces.project`, damit Dichte und Bahn deckungsgleich landen.
    P, _ = surface.sample(n_surface, seed=seed)
    a, bb = P @ e1, P @ e2
    a0, a1 = a.min(), a.max()
    b0, b1 = bb.min(), bb.max()

    # Startpunkte weit vor der Flaeche, entlang der Blickrichtung
    uv = np.asarray(xy, dtype=np.float64)
    org = (np.outer(a0 + uv[:, 0] * (a1 - a0), e1)
           + np.outer(b0 + uv[:, 1] * (b1 - b0), e2)) - 4.0 * w
    hitp, hitn, ok = rc.shoot(org, w)

    if (~ok).any():                     # Fehlschuesse: naechster Punkt
        cp, cn = rc.closest(org[~ok] + 4.0 * w)
        hitp[~ok], hitn[~ok] = cp, cn

    # Normalen nach aussen richten (dem Projektor zugewandt)
    flip = (hitn @ (-w)) < 0
    hitn[flip] *= -1.0

    pos = hitp + standoff * hitn
    R = frames_from_normals(pos, hitn)

    # Die Zielverteilung durch denselben Projektor
    pts, nrm, wt = surfaces.project(surface, dens2d, n_points=n_surface, seed=seed)
    parts = surfaces.particles_from_projection(pts, wt, n_particles, seed=seed)

    # ── Guetemasse ──────────────────────────────────────────────────────
    d_surf = np.linalg.norm(pos - hitp, axis=-1)
    sprung = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    senk = np.rad2deg(np.arccos(np.clip(
        (-R[..., 2] * hitn).sum(-1), -1, 1)))                # sollte 0 sein
    return dict(pos=pos.astype(np.float32),
                rot6=matrix_to_rot6(R).astype(np.float32),
                parts=parts.astype(np.float32),
                flaeche=pts.astype(np.float32), gewicht=wt.astype(np.float32),
                treffer_ok=ok,
                fehlschuss=float((~ok).mean()),
                standoff=float(d_surf.mean()),
                standoff_sd=float(d_surf.std()),
                sprung_max=float(sprung.max()),
                sprung_mittel=float(sprung.mean()),
                senk_max=float(senk.max()),
                getroffen=float((wt > 1e-3).mean()))


def lade_paare(db, splits=('train', 'val')):
    c = sqlite3.connect(db)
    q = ("SELECT shape_name, split, density_params, trajectory FROM ergodic_pairs "
         f"WHERE split IN ({','.join('?' * len(splits))}) ORDER BY id ASC")
    out = []
    for nm, sp, dp, blob in c.execute(q, splits):
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2).astype(np.float64)
        out.append((nm, sp, json.loads(dp), np.clip(xy, 0.0, 1.0)))
    c.close()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db_in', default=DB_IN)
    p.add_argument('--db_out', default=DB_OUT)
    p.add_argument('--surfaces', nargs='+', default=surfaces.KEYS)
    p.add_argument('--standoff', type=float, default=0.12)
    p.add_argument('--n_particles', type=int, default=512)
    p.add_argument('--dens_res', type=int, default=128)
    p.add_argument('--build', action='store_true')
    p.add_argument('--preview', type=int, default=0,
                   help='so viele Beispiele als JSON ablegen, ohne die DB zu schreiben')
    p.add_argument('--splits', nargs='+', default=['train', 'val'])
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'db3d'))
    a = p.parse_args()

    from shape_library import pdf_on_grid
    os.makedirs(a.out_dir, exist_ok=True)

    paare = lade_paare(a.db_in, tuple(a.splits))
    print(f'{len(paare)} Paare aus {os.path.basename(a.db_in)}')
    surf = {k: surfaces.build(k) for k in a.surfaces}
    rcs = {k: Raycaster(s.mesh) for k, s in surf.items()}
    print(f'{len(surf)} Oberflaechen: {", ".join(surf[k].label for k in a.surfaces)}')

    if a.preview:
        # Beispiele quer ueber Formen und Flaechen streuen
        wahl, i = [], 0
        while len(wahl) < a.preview:
            nm, sp, dp, xy = paare[(i * 37) % len(paare)]
            wahl.append((nm, sp, dp, xy, a.surfaces[len(wahl) % len(a.surfaces)]))
            i += 1
        eintraege = []
        for nm, sp, dp, xy, sk in wahl:
            d2, _, _ = pdf_on_grid(dp, resolution=a.dens_res)
            d2 = np.asarray(d2, np.float64); d2 /= max(d2.max(), 1e-12)
            r = projiziere(surf[sk], rcs[sk], d2, xy, a.standoff, a.n_particles)
            rs = np.random.default_rng(0)
            P, W = r['flaeche'], r['gewicht']
            lit = np.flatnonzero(W > 1e-3); dark = np.flatnonzero(W <= 1e-3)
            lit = rs.choice(lit, min(len(lit), 1300), replace=False)
            dark = rs.choice(dark, min(len(dark), 700), replace=False)
            keep = np.concatenate([lit, dark])
            eintraege.append(dict(
                shape=nm, split=sp, surface=sk,
                pos=r['pos'].round(4).tolist(), rot6=r['rot6'].round(4).tolist(),
                flaeche=P[keep].round(4).tolist(),
                gewicht=W[keep].round(3).tolist(),
                miss=(~r['treffer_ok']).astype(int).tolist(),
                guete={k: round(v, 5) for k, v in r.items()
                       if isinstance(v, float)}))
            g = eintraege[-1]['guete']
            print(f"  {nm[:18]:18s} {surf[sk].label:20s} "
                  f"Fehlschuss {g['fehlschuss']*100:5.1f} %  "
                  f"Standoff {g['standoff']:.3f}±{g['standoff_sd']:.3f}  "
                  f"Sprung max {g['sprung_max']:.3f}  "
                  f"senkrecht bis {g['senk_max']:.2f}°")
        f = os.path.join(a.out_dir, 'vorschau.json')
        json.dump(dict(eintraege=eintraege, standoff=a.standoff), open(f, 'w'))
        print(f'\n[json] {f}')
        fs = [e['guete']['fehlschuss'] for e in eintraege]
        sp = [e['guete']['sprung_max'] for e in eintraege]
        print(f"Fehlschuss im Mittel {np.mean(fs)*100:.1f} %, schlimmster "
              f"{np.max(fs)*100:.1f} %")
        print(f"Groesster Sprung {np.max(sp):.3f} (Bahnlaenge ~1)")
        return

    if not a.build:
        print('\nNichts zu tun — --preview N oder --build angeben.')
        return

    if os.path.exists(a.db_out):
        os.remove(a.db_out)
    con = sqlite3.connect(a.db_out)
    con.execute('''CREATE TABLE ergodic_pairs_3d (
        id INTEGER PRIMARY KEY AUTOINCREMENT, shape_name TEXT, split TEXT,
        surface TEXT, standoff REAL, density_params TEXT,
        traj_pos BLOB, traj_rot6 BLOB, particles BLOB,
        n_points INTEGER, n_particles INTEGER,
        miss_frac REAL, standoff_mean REAL, standoff_sd REAL,
        jump_max REAL, perp_max_deg REAL, hit_frac REAL)''')
    con.execute('CREATE INDEX idx_surface ON ergodic_pairs_3d(surface)')
    con.execute('CREATE INDEX idx_split ON ergodic_pairs_3d(split)')

    import time
    t0 = time.perf_counter()
    for i, (nm, sp, dp, xy) in enumerate(paare):
        d2, _, _ = pdf_on_grid(dp, resolution=a.dens_res)
        d2 = np.asarray(d2, np.float64); d2 /= max(d2.max(), 1e-12)
        for sk in a.surfaces:
            r = projiziere(surf[sk], rcs[sk], d2, xy, a.standoff, a.n_particles)
            con.execute('''INSERT INTO ergodic_pairs_3d
                (shape_name, split, surface, standoff, density_params,
                 traj_pos, traj_rot6, particles, n_points, n_particles,
                 miss_frac, standoff_mean, standoff_sd, jump_max,
                 perp_max_deg, hit_frac)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (nm, sp, sk, a.standoff, json.dumps(dp),
                 r['pos'].tobytes(), r['rot6'].tobytes(), r['parts'].tobytes(),
                 len(r['pos']), a.n_particles, r['fehlschuss'], r['standoff'],
                 r['standoff_sd'], r['sprung_max'], r['senk_max'], r['getroffen']))
        if (i + 1) % 100 == 0:
            con.commit()
            print(f'  {i+1}/{len(paare)}  ({time.perf_counter()-t0:.0f} s)')
    con.commit()
    n = con.execute('SELECT COUNT(*) FROM ergodic_pairs_3d').fetchone()[0]
    print(f'\n{n} Eintraege in {os.path.basename(a.db_out)} '
          f'({os.path.getsize(a.db_out)/2**20:.1f} MiB, '
          f'{time.perf_counter()-t0:.0f} s)')
    for row in con.execute('''SELECT surface, COUNT(*), AVG(miss_frac),
                              AVG(jump_max), MAX(perp_max_deg)
                              FROM ergodic_pairs_3d GROUP BY surface'''):
        print(f'  {row[0]:16s} {row[1]:5d}  Fehlschuss {row[2]*100:5.1f} %  '
              f'Sprung {row[3]:.3f}  senkrecht bis {row[4]:.2f}°')
    con.close()


if __name__ == '__main__':
    main()
