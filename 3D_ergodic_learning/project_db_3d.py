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

Was sich gegenueber der ersten Fassung geaendert hat
----------------------------------------------------
**Kein Kreuzprodukt mehr.** Frueher lief jede 2D-Bahn ueber jede Flaeche. Bei
79 Flaechen mit je zwei Blickwinkeln waeren das aus 1295 Paaren ueber 200 000
Eintraege — mehr Daten, aber kaum mehr Vielfalt, weil dieselbe Bahn 158-mal
wiederkehrt. Stattdessen wird je (Flaeche x Blickwinkel) eine feste Zahl von
2D-Paaren zufaellig gezogen.

**Der Projektor wird einmal je (Flaeche, Blickwinkel) gebaut, nicht je
Eintrag.** Das war der teure Fehler der ersten Fassung: `projiziere` rief
`surface.sample(20000)` auf, und `surfaces.project` gleich noch einmal — also
40 000 Flaechenstichproben je Eintrag, obwohl sich nichts daran je Eintrag
aendert. Was wirklich je Eintrag anfaellt, sind 201 Strahlen. Der Unterschied
ist der zwischen Stunden und Minuten.

**Die Dichtegitter werden zwischengespeichert.** Ein Wort mit 400
GMM-Komponenten auf einem 128er-Gitter sind 6,5 Millionen Auswertungen. Da
dieselbe Form beim Ziehen immer wieder vorkommt, wird das Gitter je Form genau
einmal gerechnet.

**Zwei Holdout-Achsen.** `val_form` ist eine unbekannte Dichte auf bekannter
Geometrie, `val_flaeche` eine bekannte Dichte auf zurueckgehaltener Geometrie,
`val_beides` beides zugleich. Ein einziger `val`-Topf haette nicht sagen
koennen, woran eine schlechte Zahl liegt.

    python project_db_3d.py --preview 40       # Beispiele pruefen
    python project_db_3d.py --probe            # Probebau, ~10 Paare je Winkel
    python project_db_3d.py --build            # der volle Bau
"""
import argparse, json, os, sqlite3, sys, time
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'thesis_architecture', 'ergodic_dataset_generator')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import surfaces                                            # noqa: E402

DB_IN = os.path.join(_root, 'thesis_architecture', 'ergodic_dataset_generator',
                     'ergodic_dataset_start_plus.db')
DB_OUT = os.path.join(_here, 'ergodic_dataset_3d.db')

MAX_FEHLSCHUSS = 0.35      # darueber wird der Eintrag gar nicht erst geschrieben


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


# ── Der Projektor: alles, was je (Flaeche, Blickwinkel) nur einmal anfaellt ──
class Projektor:
    """Vorberechnete Geometrie einer Flaeche unter einer Blickrichtung.

    Was hier liegt, haengt nicht von der Zieldichte und nicht von der Bahn ab:
    die Flaechenstichprobe, ihre Koordinaten in der Projektionsebene, die
    Gitterindizes, und welche Punkte dem Projektor zugewandt sind. Je Eintrag
    bleibt danach genau zweierlei zu tun — die Dichte an den Gitterindizes
    ablesen und 201 Strahlen schiessen.
    """

    def __init__(self, surface, view, rc, n_surface=20000, dens_res=128,
                 seed=0):
        self.surface, self.rc = surface, rc
        self.view = np.asarray(view, dtype=np.float64)
        self.view /= max(np.linalg.norm(self.view), 1e-12)
        self.e1, self.e2, self.w = surfaces._frame(self.view)

        P, N = surface.sample(n_surface, seed=seed)
        self.pts = P
        a, b = P @ self.e1, P @ self.e2
        self.a0, self.a1 = float(a.min()), float(a.max())
        self.b0, self.b1 = float(b.min()), float(b.max())
        u = (a - self.a0) / max(self.a1 - self.a0, 1e-12)
        v = (b - self.b0) / max(self.b1 - self.b0, 1e-12)
        R = int(dens_res)
        self.ix = np.clip((u * (R - 1)).round().astype(np.int32), 0, R - 1)
        self.iy = np.clip((v * (R - 1)).round().astype(np.int32), 0, R - 1)
        self.facing = (N @ (-self.w)) > 1e-6
        self.dens_res = R

    def gewicht(self, d2):
        """Die 2D-Dichte an den Flaechenpunkten; abgewandte bekommen null."""
        w = np.asarray(d2, dtype=np.float64)[self.iy, self.ix]
        return np.where(self.facing, w, 0.0)

    def strahlen(self, xy):
        """201 Strahlen auf die Bahnpunkte. -> (Trefferpunkte, Normalen, ok)"""
        uv = np.asarray(xy, dtype=np.float64)
        org = (np.outer(self.a0 + uv[:, 0] * (self.a1 - self.a0), self.e1)
               + np.outer(self.b0 + uv[:, 1] * (self.b1 - self.b0), self.e2)
               ) - 4.0 * self.w
        hitp, hitn, ok = self.rc.shoot(org, self.w)
        if (~ok).any():                     # Fehlschuesse: naechster Punkt
            cp, cn = self.rc.closest(org[~ok] + 4.0 * self.w)
            hitp[~ok], hitn[~ok] = cp, cn
        flip = (hitn @ (-self.w)) < 0       # Normalen nach aussen richten
        hitn[flip] *= -1.0
        return hitp, hitn, ok


# ── Ein Eintrag ──────────────────────────────────────────────────────────────
def projiziere(proj, d2, xy, standoff=0.12, n_particles=512, seed=0):
    """-> dict mit Bahn (T,3), Rahmen (T,6), Partikeln (N,4) und Guetemassen."""
    hitp, hitn, ok = proj.strahlen(xy)
    pos = hitp + standoff * hitn
    R = frames_from_normals(pos, hitn)

    wt = proj.gewicht(d2)
    parts = surfaces.particles_from_projection(proj.pts, wt, n_particles,
                                               seed=seed)

    # ── Guetemasse ──────────────────────────────────────────────────────
    d_surf = np.linalg.norm(pos - hitp, axis=-1)
    sprung = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
    senk = np.rad2deg(np.arccos(np.clip(
        (-R[..., 2] * hitn).sum(-1), -1, 1)))                # sollte 0 sein
    return dict(pos=pos.astype(np.float32),
                rot6=matrix_to_rot6(R).astype(np.float32),
                parts=parts.astype(np.float32),
                flaeche=proj.pts.astype(np.float32),
                gewicht=wt.astype(np.float32),
                treffer_ok=ok,
                fehlschuss=float((~ok).mean()),
                standoff=float(d_surf.mean()),
                standoff_sd=float(d_surf.std()),
                sprung_max=float(sprung.max()),
                sprung_mittel=float(sprung.mean()),
                senk_max=float(senk.max()),
                getroffen=float((wt > 1e-3).mean()))


# ── Die 2D-Quelle ────────────────────────────────────────────────────────────
def lade_paare(db, splits=('train', 'val', 'val_flat')):
    """-> Liste von dicts mit name, split, dichte, xy, x0.

    `x0` kommt jetzt mit. Ohne ihn koennte die Startpunkt-Konditionierung nicht
    trainiert werden: das Netz braucht den Startpunkt als *Eingang*, und der
    steht in der 2D-Datenbank, nicht in der projizierten. In 3D wird er zum
    ersten Bahnpunkt `pos[0]` — der 2D-Wert bleibt trotzdem erhalten, weil er
    der einzige Weg zurueck zur Quelle ist.
    """
    c = sqlite3.connect(db)
    q = ("SELECT shape_name, split, density_params, trajectory, x0 "
         "FROM ergodic_pairs "
         f"WHERE split IN ({','.join('?' * len(splits))}) ORDER BY id ASC")
    out = []
    for nm, sp, dp, blob, x0 in c.execute(q, splits):
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2).astype(np.float64)
        out.append(dict(name=nm, split=sp, dichte=json.loads(dp),
                        xy=np.clip(xy, 0.0, 1.0),
                        x0=np.asarray(json.loads(x0), dtype=np.float64)))
    c.close()
    return out


class Dichtegitter:
    """Dichtegitter je Form, einmal gerechnet.

    Der Schluessel ist der Formname und nicht die Zeilennummer: die vier
    Startpunkt-Varianten eines Wortes haben verschiedene Bahnen, aber genau
    dieselbe Dichte. Ohne diesen Zwischenspeicher wuerde ein Wort mit 400
    GMM-Komponenten bei jedem Zug neu ueber 16 384 Gitterzellen ausgewertet.
    """

    def __init__(self, res=128):
        from shape_library import pdf_on_grid
        self._pdf = pdf_on_grid
        self.res = int(res)
        self._c = {}
        self.treffer = self.fehlgriffe = 0

    def __call__(self, name, dichte):
        g = self._c.get(name)
        if g is None:
            self.fehlgriffe += 1
            d, _, _ = self._pdf(dichte, resolution=self.res)
            d = np.asarray(d, dtype=np.float64)
            g = (d / max(d.max(), 1e-12)).astype(np.float32)
            self._c[name] = g
        else:
            self.treffer += 1
        return g


# ── Schema ───────────────────────────────────────────────────────────────────
SCHEMA = '''
CREATE TABLE ergodic_pairs_3d (
    id INTEGER PRIMARY KEY AUTOINCREMENT, shape_name TEXT, split TEXT,
    surface TEXT, gruppe TEXT, standoff REAL, density_params TEXT,
    traj_pos BLOB, traj_rot6 BLOB, particles BLOB, start_pos BLOB,
    x0_2d TEXT, view_x REAL, view_y REAL, view_z REAL, view_id INTEGER,
    n_points INTEGER, n_particles INTEGER,
    miss_frac REAL, standoff_mean REAL, standoff_sd REAL,
    jump_max REAL, perp_max_deg REAL, hit_frac REAL);
CREATE INDEX idx_surface ON ergodic_pairs_3d(surface);
CREATE INDEX idx_split ON ergodic_pairs_3d(split);
CREATE INDEX idx_gruppe ON ergodic_pairs_3d(gruppe);
CREATE TABLE dichten (shape_name TEXT PRIMARY KEY, density_params TEXT);
CREATE TABLE blickwinkel (
    surface TEXT, view_id INTEGER, gruppe TEXT, heldout INTEGER,
    view_x REAL, view_y REAL, view_z REAL,
    treffer REAL, fehlschuss REAL, bestanden INTEGER,
    PRIMARY KEY (surface, view_id));
'''

# `density_params` steht in der Zeile weiterhin drin, aber leer: die Dichte
# selbst liegt einmal je Form in `dichten`. Bei 43 000 Eintraegen und rund
# 20 kB JSON je Wort waeren das sonst ueber 800 MB fuer Angaben, die sich
# 33-mal wiederholen. Die Spalte bleibt, damit bestehende Abfragen nicht
# brechen; wer die Dichte braucht, verbindet ueber `shape_name`.

EINFUEGEN = '''INSERT INTO ergodic_pairs_3d
    (shape_name, split, surface, gruppe, standoff, density_params,
     traj_pos, traj_rot6, particles, start_pos, x0_2d,
     view_x, view_y, view_z, view_id, n_points, n_particles,
     miss_frac, standoff_mean, standoff_sd, jump_max, perp_max_deg, hit_frac)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''


def _zeile(nm, split, sk, gruppe, standoff, r, x0, view, vid, n_particles):
    return (nm, split, sk, gruppe, standoff, '',
            r['pos'].tobytes(), r['rot6'].tobytes(), r['parts'].tobytes(),
            r['pos'][0].astype(np.float32).tobytes(),
            json.dumps([float(x0[0]), float(x0[1])]),
            float(view[0]), float(view[1]), float(view[2]), int(vid),
            len(r['pos']), n_particles, r['fehlschuss'], r['standoff'],
            r['standoff_sd'], r['sprung_max'], r['senk_max'], r['getroffen'])


# ── Bau ──────────────────────────────────────────────────────────────────────
def _flaechenliste(a):
    surfaces.externe_aufnehmen(verbose=False)
    train = a.surfaces if a.surfaces else surfaces.alle_keys()
    held = [] if a.ohne_heldout else surfaces.alle_keys(nur_heldout=True)
    return train, held


def bauen(a):
    paare = lade_paare(a.db_in, tuple(a.splits))
    bekannt = [p for p in paare if p['split'] == 'train']
    unbekannt = [p for p in paare if p['split'] != 'train']
    print(f'{len(paare)} 2D-Paare aus {os.path.basename(a.db_in)}: '
          f'{len(bekannt)} bekannte, {len(unbekannt)} zurueckgehaltene Dichten')

    train_f, held_f = _flaechenliste(a)
    print(f'{len(train_f)} Trainingsflaechen, {len(held_f)} zurueckgehaltene, '
          f'{a.views} Blickwinkel je Flaeche')

    if os.path.exists(a.db_out):
        os.remove(a.db_out)
    con = sqlite3.connect(a.db_out)
    con.executescript(SCHEMA)
    for p in paare:
        con.execute('INSERT OR IGNORE INTO dichten VALUES (?,?)',
                    (p['name'], json.dumps(p['dichte'])))
    con.commit()

    gitter = Dichtegitter(a.dens_res)
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    geschrieben = verworfen = 0
    notwinkel = []

    aufgaben = [(k, False) for k in train_f] + [(k, True) for k in held_f]
    for fi, (sk, heldout) in enumerate(aufgaben):
        s = surfaces.build(sk)
        gruppe = surfaces.gruppe_von(sk)
        rc = Raycaster(s.mesh)
        # Ebenen behalten ihre eigene Normale als Blickrichtung.
        #
        # Gemessen: auf `ebene_flach` mit ihrer eigenen Normalen liegt der
        # groesste Sprung bei 0,05 — mit zufaelligen Richtungen stieg die
        # Gruppe auf 0,31 im Mittel. Der Grund ist, dass eine Ebene keine Dicke
        # hat. Schraeg angeschaut ist ihre Silhouette eine Linie, die 2D-Dichte
        # faellt zum groessten Teil daneben, und die Bahn besteht aus
        # Naechster-Punkt-Ersatz. Das Guetetor faellt darauf herein, weil es
        # den Trefferanteil auf die Ausdehnung der *projizierten* Flaeche
        # bezieht — und die ist dann eben auch nur ein Streifen.
        #
        # Die zehn Ebenen bringen ihre Vielfalt ohnehin ueber ihre zehn
        # verschiedenen Lagen im Raum ein, nicht ueber den Blickwinkel: entlang
        # der Normalen ist die Projektion unverzerrt, und genau das ist der
        # Fall, den sie darstellen sollen. Sie bekommen deshalb einen Winkel
        # und dafuer das volle Paarbudget.
        ist_ebene = gruppe == 'ebene'
        if ist_ebene:
            winkel = [(s.view, surfaces.guete_richtung(
                s.mesh, s.view, n=a.winkel_strahlen), True)]
        else:
            winkel = surfaces.blickrichtungen(rng, s.mesh, anzahl=a.views,
                                              versuche=a.winkel_versuche,
                                              n=a.winkel_strahlen)
        for vid, (view, g, ok) in enumerate(winkel):
            if not ok:
                notwinkel.append((sk, vid, round(g['treffer'], 3),
                                  round(g['fehlschuss'], 3)))
            con.execute('INSERT INTO blickwinkel VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (sk, vid, gruppe, int(heldout), float(view[0]),
                         float(view[1]), float(view[2]), g['treffer'],
                         g['fehlschuss'], int(ok)))
            proj = Projektor(s, view, rc, n_surface=a.n_surface,
                             dens_res=a.dens_res, seed=vid)

            # Zwei Toepfe je Flaeche. Auf einer Trainingsflaeche liefert die
            # bekannte Dichte 'train' und die zurueckgehaltene 'val_form'; auf
            # einer zurueckgehaltenen Flaeche heissen dieselben beiden Toepfe
            # 'val_flaeche' und 'val_beides'.
            # Ebenen haben nur einen Winkel und bekommen dafuer das Budget
            # beider, damit die Gruppe nicht allein durch die Bauart schrumpft.
            f = a.views if ist_ebene else 1
            if heldout:
                toepfe = [('val_flaeche', bekannt, a.paare_pro_view_val * f),
                          ('val_beides', unbekannt, a.paare_pro_view_val * f)]
            else:
                toepfe = [('train', bekannt, a.paare_pro_view * f),
                          ('val_form', unbekannt, a.paare_pro_view_val * f)]

            for split, quelle, anzahl in toepfe:
                if not quelle or anzahl <= 0:
                    continue
                idx = rng.choice(len(quelle), size=min(anzahl, len(quelle)),
                                 replace=False)
                for j in idx:
                    p = quelle[j]
                    d2 = gitter(p['name'], p['dichte'])
                    r = projiziere(proj, d2, p['xy'], a.standoff,
                                   a.n_particles, seed=int(j))
                    if r['fehlschuss'] > a.max_fehlschuss:
                        verworfen += 1
                        continue
                    con.execute(EINFUEGEN, _zeile(
                        p['name'], split, sk, gruppe, a.standoff, r, p['x0'],
                        view, vid, a.n_particles))
                    geschrieben += 1
        con.commit()
        print(f'  [{fi+1:3d}/{len(aufgaben)}] {sk:22s} {gruppe:10s} '
              f'{geschrieben:7d} Eintraege  ({time.perf_counter()-t0:.0f} s)')

    con.commit()
    _bericht(con, a, geschrieben, verworfen, notwinkel, gitter, t0)
    con.close()


def _bericht(con, a, geschrieben, verworfen, notwinkel, gitter, t0):
    print(f'\n{geschrieben} Eintraege in {os.path.basename(a.db_out)} '
          f'({os.path.getsize(a.db_out)/2**20:.0f} MiB, '
          f'{time.perf_counter()-t0:.0f} s)')
    print(f'{verworfen} verworfen (Fehlschuss > {a.max_fehlschuss:.0%})')
    print(f'Dichtegitter: {gitter.fehlgriffe} gerechnet, '
          f'{gitter.treffer} aus dem Zwischenspeicher '
          f'({gitter.treffer/max(gitter.treffer+gitter.fehlgriffe,1):.0%} gespart)')

    print('\nEintraege je Split:')
    for sp, n in con.execute('SELECT split, COUNT(*) FROM ergodic_pairs_3d '
                             'GROUP BY split ORDER BY COUNT(*) DESC'):
        print(f'  {sp:14s} {n:7d}')

    print('\nGuete je Gruppe:')
    for row in con.execute('''SELECT gruppe, COUNT(*), AVG(miss_frac),
                              AVG(jump_max), MAX(perp_max_deg), AVG(hit_frac)
                              FROM ergodic_pairs_3d GROUP BY gruppe'''):
        print(f'  {row[0]:12s} {row[1]:7d}  Fehlschuss {row[2]*100:5.1f} %  '
              f'Sprung {row[3]:.3f}  senkrecht bis {row[4]:.2f}°  '
              f'beschienen {row[5]*100:4.1f} %')

    print('\nGuete je Flaeche (schlechteste 20 nach Fehlschuss):')
    for row in con.execute('''SELECT surface, gruppe, COUNT(*), AVG(miss_frac),
                              AVG(jump_max), MAX(perp_max_deg), AVG(hit_frac)
                              FROM ergodic_pairs_3d GROUP BY surface
                              ORDER BY AVG(miss_frac) DESC LIMIT 20'''):
        print(f'  {row[0]:22s} {row[1]:10s} {row[2]:6d}  '
              f'Fehlschuss {row[3]*100:5.1f} %  Sprung {row[4]:.3f}  '
              f'senkrecht bis {row[5]:.2f}°  beschienen {row[6]*100:4.1f} %')

    fehlend = [k for k in surfaces.alle_keys(mit_heldout=True)
               if not con.execute('SELECT 1 FROM ergodic_pairs_3d WHERE '
                                  'surface=? LIMIT 1', (k,)).fetchone()]
    if fehlend:
        print(f'\n[!] {len(fehlend)} Flaechen ohne einen einzigen Eintrag — '
              f'alle Zuege am Fehlschussfilter gescheitert:\n    '
              + ', '.join(fehlend))
    if notwinkel:
        print(f'\n[!] {len(notwinkel)} Blickwinkel nur mit Abstrichen '
              f'(Silhouette unter {surfaces.MIN_TREFFER:.0%} oder Netz '
              f'durchlaessig):')
        for sk, vid, tr, fs in notwinkel[:15]:
            print(f'    {sk:22s} Winkel {vid}  Treffer {tr:.2f}  '
                  f'Fehlschuss {fs:.3f}')


# ── Vorschau ─────────────────────────────────────────────────────────────────
def vorschau(a):
    paare = lade_paare(a.db_in, tuple(a.splits))
    train_f, held_f = _flaechenliste(a)
    keys = (train_f + held_f)
    gitter = Dichtegitter(a.dens_res)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    eintraege = []
    for i in range(a.preview):
        sk = keys[(i * 7) % len(keys)]
        p = paare[(i * 37) % len(paare)]
        s = surfaces.build(sk)
        view, g, ok = surfaces.zufaellige_blickrichtung(
            rng, s.mesh, versuche=a.winkel_versuche, n=a.winkel_strahlen)
        proj = Projektor(s, view, Raycaster(s.mesh), n_surface=a.n_surface,
                         dens_res=a.dens_res, seed=i)
        d2 = gitter(p['name'], p['dichte'])
        r = projiziere(proj, d2, p['xy'], a.standoff, a.n_particles, seed=i)

        rs = np.random.default_rng(0)
        P, W = r['flaeche'], r['gewicht']
        lit = np.flatnonzero(W > 1e-3); dark = np.flatnonzero(W <= 1e-3)
        lit = rs.choice(lit, min(len(lit), 1300), replace=False)
        dark = rs.choice(dark, min(len(dark), 700), replace=False)
        keep = np.concatenate([lit, dark])
        eintraege.append(dict(
            shape=p['name'], split=p['split'], surface=sk,
            gruppe=surfaces.gruppe_von(sk),
            view=[round(float(v), 4) for v in view],
            x0=[round(float(v), 4) for v in p['x0']],
            start_pos=r['pos'][0].round(4).tolist(),
            pos=r['pos'].round(4).tolist(), rot6=r['rot6'].round(4).tolist(),
            flaeche=P[keep].round(4).tolist(),
            gewicht=W[keep].round(3).tolist(),
            miss=(~r['treffer_ok']).astype(int).tolist(),
            guete={k: round(v, 5) for k, v in r.items()
                   if isinstance(v, float)}))
        q = eintraege[-1]['guete']
        print(f"  {p['name'][:18]:18s} {sk:20s} "
              f"Fehlschuss {q['fehlschuss']*100:5.1f} %  "
              f"Standoff {q['standoff']:.3f}±{q['standoff_sd']:.3f}  "
              f"Sprung max {q['sprung_max']:.3f}  "
              f"senkrecht bis {q['senk_max']:.2f}°")
    f = os.path.join(a.out_dir, 'vorschau.json')
    json.dump(dict(eintraege=eintraege, standoff=a.standoff), open(f, 'w'))
    print(f'\n[json] {f}')
    fs = np.array([e['guete']['fehlschuss'] for e in eintraege])
    sp = np.array([e['guete']['sprung_max'] for e in eintraege])
    se = np.array([e['guete']['senk_max'] for e in eintraege])
    print(f'Fehlschuss im Mittel {fs.mean()*100:.1f} %, schlimmster '
          f'{fs.max()*100:.1f} %')
    print(f'Groesster Sprung {sp.max():.3f} (Bahnlaenge ~1)')
    print(f'Groesste Abweichung von der Senkrechten {se.max():.3g}°')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db_in', default=DB_IN)
    p.add_argument('--db_out', default=DB_OUT)
    p.add_argument('--surfaces', nargs='+', default=None,
                   help='nur diese Flaechen (Voreinstellung: alle aus '
                        'surfaces.alle_keys())')
    p.add_argument('--ohne_heldout', action='store_true', default=False)
    p.add_argument('--standoff', type=float, default=0.12)
    p.add_argument('--n_particles', type=int, default=512)
    p.add_argument('--n_surface', type=int, default=20000)
    p.add_argument('--dens_res', type=int, default=128)
    p.add_argument('--views', type=int, default=2,
                   help='Blickwinkel je Flaeche')
    p.add_argument('--paare_pro_view', type=int, default=240,
                   help='zufaellig gezogene 2D-Paare je (Flaeche x Winkel)')
    p.add_argument('--paare_pro_view_val', type=int, default=24)
    p.add_argument('--max_fehlschuss', type=float, default=MAX_FEHLSCHUSS)
    p.add_argument('--winkel_versuche', type=int, default=24)
    p.add_argument('--winkel_strahlen', type=int, default=600)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--build', action='store_true')
    p.add_argument('--probe', action='store_true',
                   help='Probebau: 10 Paare je Winkel, eigene Ausgabedatei')
    p.add_argument('--preview', type=int, default=0,
                   help='so viele Beispiele als JSON ablegen, ohne die DB zu schreiben')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'val_flat'])
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'db3d'))
    a = p.parse_args()

    if a.preview:
        return vorschau(a)
    if a.probe:
        a.paare_pro_view = 10
        a.paare_pro_view_val = 4
        if a.db_out == DB_OUT:
            a.db_out = os.path.join(_here, 'ergodic_dataset_3d_probe.db')
        print('Probebau: 10 Trainings- und 4 Holdout-Paare je Winkel\n')
        return bauen(a)
    if not a.build:
        print('\nNichts zu tun — --preview N, --probe oder --build angeben.')
        return
    bauen(a)


if __name__ == '__main__':
    main()
