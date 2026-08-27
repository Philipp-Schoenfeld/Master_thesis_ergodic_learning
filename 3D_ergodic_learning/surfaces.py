r"""
surfaces.py
===========
Zielverteilungen auf gekruemmten Oberflaechen statt auf einer Ebene.

Das 3D-Netz ist auf Dichten trainiert, die in einer duennen Scheibe um
z = 0,5 liegen (`Z_SIGMA = 0,05` in `data_3d.py`). Eine Kugel, ein Wuerfel
oder der Stanford-Bunny sind damit **ausserhalb der Trainingsverteilung** —
genau das soll hier gemessen werden, nicht umgangen.

Das Verfahren ist fuer alle Oberflaechen dasselbe und in einem Satz erklaert:
die zweidimensionale Zieldichte wird wie von einem Diaprojektor auf die
Oberflaeche geworfen. Getroffen wird nur, was dem Projektor zugewandt ist; die
Rueckseite bleibt leer.

    p auf der Oberflaeche  ->  (u, v) = Koordinaten in der Projektionsebene
                           ->  Gewicht = Dichte(u, v),  falls n . (-w) > 0

`w` ist die Strahlrichtung des Projektors. Sie ist je Oberflaeche so gewaehlt,
dass die Projektion etwas Interessantes trifft — beim Wuerfel etwa entlang der
Raumdiagonale, damit drei Seiten gleichzeitig beschriftet werden.

Die Ebenen sind der Sonderfall, in dem die Projektion verlustfrei ist: liegt
`w` auf der Flaechennormalen, ist das Bild unverzerrt. Fuer die drei
Raumorientierungen bleibt es dabei — verglichen wird dann nicht die Verzerrung,
sondern was die Lage im Raum fuer eine Bahn bedeutet, die auf z = 0,5 trainiert
wurde.
"""

import os
import numpy as np

MARGIN = 0.14          # Abstand zum Rand des Einheitswuerfels
_THRESH = 1e-3         # ab hier gilt eine Zelle als getroffen


# ── Hilfen ───────────────────────────────────────────────────────────────────
def _fit_unit(v, margin=MARGIN):
    """In [margin, 1-margin]^3 einpassen, Seitenverhaeltnis erhalten."""
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = float(max((hi - lo).max(), 1e-9))
    return (v - (lo + hi) / 2.0) * ((1.0 - 2 * margin) / span) + 0.5


def _frame(w):
    """Orthonormales (e1, e2, w) zu einer Blickrichtung."""
    w = np.asarray(w, dtype=np.float64)
    w = w / max(np.linalg.norm(w), 1e-12)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, w); e1 /= max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(w, e1)
    return e1, e2, w


def _plane_mesh(normal, n=90):
    """Quadratisches Netz mit gegebener Normalen, Kantenlaenge 1."""
    import trimesh
    e1, e2, nrm = _frame(normal)
    a = np.linspace(-0.5, 0.5, n)
    U, V = np.meshgrid(a, a, indexing='ij')
    verts = (U[..., None] * e1 + V[..., None] * e2).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    f1 = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:]], -1).reshape(-1, 3)
    f2 = np.stack([idx[:-1, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=np.vstack([f1, f2]),
                           process=False)


def _egg_mesh(subdiv=4, taper=0.32, stretch=1.35):
    """Ovoid: eine Kugel, entlang z gestreckt und nach oben verjuengt."""
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    v = np.asarray(m.vertices, dtype=np.float64).copy()
    z = v[:, 2].copy()
    v[:, :2] *= (1.0 - taper * z)[:, None]
    v[:, 2] = z * stretch
    return trimesh.Trimesh(vertices=v, faces=m.faces, process=False)


def _prism_mesh(height=1.0, radius=0.62):
    """Gerades Dreiecksprisma: eine gleichseitige Dreiecksgrundflaeche,
    entlang z extrudiert.
    """
    import trimesh
    angles = np.deg2rad([90.0, 210.0, 330.0])
    tri = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    bottom = np.concatenate([tri, np.full((3, 1), -height / 2)], axis=1)
    top = np.concatenate([tri, np.full((3, 1), height / 2)], axis=1)
    verts = np.vstack([bottom, top])          # 0,1,2 unten; 3,4,5 oben
    faces = [[0, 2, 1], [3, 4, 5]]             # Grund- und Deckflaeche
    for i in range(3):
        j = (i + 1) % 3
        faces += [[i, j, j + 3], [i, j + 3, i + 3]]
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)


def _cone_mesh(radius=0.55, height=1.0, sections=64):
    """Gerader Kreiskegel, Spitze oben."""
    import trimesh
    m = trimesh.creation.cone(radius=radius, height=height, sections=sections)
    v = np.asarray(m.vertices, dtype=np.float64).copy()
    v[:, 2] -= height / 2          # trimesh baut den Kegel von z=0 bis z=height
    return trimesh.Trimesh(vertices=v, faces=m.faces, process=False)


def _torus_mesh(major=0.35, minor=0.16, major_sections=48, minor_sections=24):
    """Torus um die z-Achse."""
    import trimesh
    m = trimesh.creation.torus(major_radius=major, minor_radius=minor,
                               major_sections=major_sections,
                               minor_sections=minor_sections)
    return trimesh.Trimesh(vertices=np.asarray(m.vertices, dtype=np.float64),
                           faces=np.asarray(m.faces), process=False)


def _bunny_mesh():
    """Stanford-Bunny. Wird beim ersten Aufruf ueber open3d geholt."""
    import trimesh, open3d as o3d
    path = o3d.data.BunnyMesh().path
    m = trimesh.load(path, process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    m.fix_normals()
    # Aufrecht stellen: das Original liegt mit y nach oben.
    v = np.asarray(m.vertices, dtype=np.float64)
    m = trimesh.Trimesh(vertices=np.stack([v[:, 0], -v[:, 2], v[:, 1]], -1),
                        faces=m.faces, process=False)
    m.fix_normals()
    return m


# ── Die Oberflaechen ─────────────────────────────────────────────────────────
class Surface:
    """Ein Dreiecksnetz in [0,1]^3 samt Projektionsrichtung."""

    def __init__(self, key, label, mesh, view, note=''):
        import trimesh
        self.key, self.label, self.note = key, label, note
        v = _fit_unit(np.asarray(mesh.vertices, dtype=np.float64))
        self.mesh = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
        self.mesh.fix_normals()
        self.view = np.asarray(view, dtype=np.float64)
        self.view /= max(np.linalg.norm(self.view), 1e-12)

    def sample(self, n, seed=0):
        """n gleichverteilte Punkte auf der Flaeche samt Normalen."""
        import trimesh
        rng = np.random.default_rng(seed)
        pts, fid = trimesh.sample.sample_surface(self.mesh, n, seed=int(seed))
        nrm = np.asarray(self.mesh.face_normals)[fid]
        return np.asarray(pts, dtype=np.float64), np.asarray(nrm, dtype=np.float64)

    @property
    def area(self):
        return float(self.mesh.area)


def build(key):
    import trimesh
    if key == 'ebene_flach':
        return Surface(key, 'Ebene, waagerecht', _plane_mesh((0, 0, 1)),
                       view=(0, 0, -1),
                       note='die Trainingslage: Dichte in einer Scheibe um z = 0,5')
    if key == 'ebene_gekippt':
        nrm = np.array([0.0, np.sin(np.deg2rad(50)), np.cos(np.deg2rad(50))])
        return Surface(key, 'Ebene, 50° gekippt', _plane_mesh(nrm), view=-nrm,
                       note='dieselbe Dichte, um 50 Grad aus der Trainingslage gedreht')
    if key == 'ebene_diagonal':
        nrm = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        return Surface(key, 'Ebene, Raumdiagonale', _plane_mesh(nrm), view=-nrm,
                       note='Normale entlang (1,1,1) — keine Achse ausgezeichnet')
    if key == 'kugel':
        return Surface(key, 'Kugel', trimesh.creation.icosphere(subdivisions=4),
                       view=(0, 0, -1),
                       note='konstante Kruemmung, die Dichte trifft eine Halbkugel')
    if key == 'wuerfel':
        return Surface(key, 'Würfel', trimesh.creation.box(extents=(1, 1, 1)),
                       view=-np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
                       note='Projektion entlang der Raumdiagonale trifft drei Seiten')
    if key == 'ei':
        return Surface(key, 'Eiform', _egg_mesh(), view=(0, 0, -1),
                       note='wie die Kugel, aber ohne Symmetrie entlang der Achse')
    if key == 'bunny':
        return Surface(key, 'Stanford-Bunny', _bunny_mesh(), view=(0, -1, 0),
                       note='echtes Messnetz, konkav und konvex zugleich')
    if key == 'prisma':
        nrm = np.array([0.45, -0.3, 0.84]) / np.linalg.norm([0.45, -0.3, 0.84])
        return Surface(key, 'Dreiecksprisma', _prism_mesh(), view=-nrm,
                       note='ebene Facetten mit scharfen Kanten dazwischen — '
                            'die Projektion trifft Deckflaeche und zwei Seiten')
    if key == 'kegel':
        return Surface(key, 'Kegel', _cone_mesh(), view=(0, 0, -1),
                       note='Kruemmung nimmt zur Spitze hin zu — die Mantelflaeche '
                            'wird schmaler, waehrend die Projektion gleich breit bleibt')
    if key == 'torus':
        return Surface(key, 'Torus', _torus_mesh(), view=(0, 0, -1),
                       note='Loch in der Mitte: von oben trifft die Projektion nur '
                            'den Ring, die Innenseite bleibt zwangslaeufig unbeschienen')
    raise KeyError(key)


KEYS = ['ebene_flach', 'ebene_gekippt', 'ebene_diagonal',
        'kugel', 'wuerfel', 'ei', 'bunny', 'prisma', 'kegel', 'torus']


# ── Projektion ───────────────────────────────────────────────────────────────
def project(surface, dens2d, n_points=20000, seed=0):
    """Die 2D-Dichte auf die Oberflaeche werfen.

    dens2d: (R, R), indiziert [y, x], auf Maximum 1 normiert.

    Rueckgabe: (pts (M,3), nrm (M,3), w (M,)) — Punkte auf der Flaeche mit dem
    projizierten Gewicht. Abgewandte Punkte bekommen Gewicht 0 und bleiben so
    in der Ausgabe erhalten, damit die Zeichnung die ganze Oberflaeche zeigen
    kann und nicht nur den beschrifteten Teil.
    """
    pts, nrm = surface.sample(n_points, seed=seed)
    e1, e2, w = _frame(surface.view)

    # Koordinaten in der Projektionsebene, auf die Ausdehnung der Flaeche bezogen
    a = pts @ e1
    b = pts @ e2
    u = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    v = (b - b.min()) / max(b.max() - b.min(), 1e-12)

    R = dens2d.shape[-1]
    ix = np.clip((u * (R - 1)).round().astype(int), 0, R - 1)
    iy = np.clip((v * (R - 1)).round().astype(int), 0, R - 1)
    weight = np.asarray(dens2d, dtype=np.float64)[iy, ix]

    # Nur was dem Projektor zugewandt ist
    facing = (nrm @ (-w)) > 1e-6
    weight = np.where(facing, weight, 0.0)
    return pts, nrm, weight


def particles_from_projection(pts, weight, n_particles, seed=0,
                              mode='uniform', threshold=_THRESH):
    """Partikelwolke (N,4) in der Konvention des Trainings.

    `mode='uniform'`: Orte gleichverteilt ueber den getroffenen Teil der
    Oberflaeche, das Gewicht als viertes Merkmal — genau wie
    `data_3d.sample_particles(mode='uniform')`, auf das das Netz trainiert
    wurde.
    """
    rng = np.random.default_rng(seed)
    supp = weight > threshold
    if supp.sum() < 8:                      # nichts getroffen: ganze Flaeche
        supp = np.ones_like(weight, dtype=bool)
    p = (np.ones(supp.sum()) if mode == 'uniform' else weight[supp] + 1e-9)
    p = p / p.sum()
    idx = rng.choice(np.flatnonzero(supp), size=n_particles, replace=True, p=p)
    out = np.concatenate([pts[idx], weight[idx, None]], axis=-1)
    return out.astype(np.float32)
