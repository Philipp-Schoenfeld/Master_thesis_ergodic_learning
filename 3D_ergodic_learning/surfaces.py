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


def _build_alt(key):
    """Die zehn urspruenglichen Flaechen, unveraendert.

    Diese Funktion ist bewusst nicht angefasst worden. `run_surface_eval.py`,
    `plot_surfaces.py` und `export_surface_viewer.py` bauen darauf auf, und die
    Vergleichszahlen der bisherigen Auswertungen haengen daran, dass genau
    dieselben Netze mit genau denselben Blickrichtungen herauskommen.
    """
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

# `KEYS` bleibt absichtlich diese Zehnerliste. Die drei Auswertungsskripte
# durchlaufen sie ohne Argument; waere sie auf vierundachtzig Flaechen
# angewachsen, haetten sie ihr Verhalten stillschweigend geaendert. Die neue,
# vollstaendige Liste heisst `alle_keys()` und muss ausdruecklich angefordert
# werden.


# ═════════════════════════════════════════════════════════════════════════════
#  Die erweiterte Flaechenliste
# ═════════════════════════════════════════════════════════════════════════════
#
# Acht Gruppen, weil sie beim Training verschieden gewichtet werden sollen:
#
#   ebene         zehn Lagen derselben unverzerrten Projektion. `ebene_flach`
#                 ist die Trainingslage der bisherigen Laeufe und bekommt
#                 deshalb ein eigenes Gewicht — sie ist der Bezugspunkt, gegen
#                 den sich alles Neue messen lassen muss.
#   primitiv      die sieben bisherigen Koerper plus die dreissig aus
#                 koerper.py — konvex oder rotationssymmetrisch.
#   buchstabe     die sechsundzwanzig extrudierten Grossbuchstaben.
#   extern        gemessene und modellierte Netze, die das Aufnahmetor
#                 bestanden haben.
#   organismus    fuenfundvierzig Metaball-Koerper (organismen.py) — glatt und
#                 nicht konvex, weder Primitiv noch Rotationskoerper.
#   baugruppe     fuenfundzwanzig boolesche Verknuepfungen (baugruppen.py) —
#                 durchgehende Loecher, Kerben, mehrteilige Formen.
#   superquadrik  achtundzwanzig Formen einer stetigen Familie zwischen Kasten
#                 und Stern (superquadriken.py).
#   wort          fuenfundzwanzig kurze Wortkoerper (woerter.py) — mehrteilige,
#                 scharfkantige Silhouetten mit Zwischenraum statt nur Kanten.
#
# Zurueckgehalten wird ueber beide Achsen getrennt: Flaechen mit `heldout=True`
# im Verzeichnis kommen im Training nie vor.

GRUPPEN = ('ebene', 'primitiv', 'buchstabe', 'extern',
          'organismus', 'baugruppe', 'superquadrik', 'wort')


def _fibonacci_halbkugel(n, ausschluss=None, min_winkel=25.0):
    """n moeglichst gleichverteilte Richtungen auf der oberen Halbkugel.

    Die goldene Spirale verteilt Punkte auf der Kugel gleichmaessiger als
    Zufall oder ein Winkelraster, das an den Polen zusammenlaeuft. Richtungen,
    die einer bereits vergebenen zu nahe kommen, fallen heraus — sonst laege
    eine der neuen Ebenen praktisch auf `ebene_flach` und brächte nichts.
    """
    phi = np.pi * (3.0 - np.sqrt(5.0))
    m = 4 * n + 16                       # ueberziehen, danach aussortieren
    out = []
    aus = [np.asarray(a, np.float64) / np.linalg.norm(a)
           for a in (ausschluss or [])]
    kappa = np.cos(np.deg2rad(min_winkel))
    for i in range(m):
        z = 1.0 - (i + 0.5) / m          # 1 .. -1
        if z < 0.05:                     # nur die obere Halbkugel
            continue
        r = np.sqrt(max(1.0 - z * z, 0.0))
        a = phi * i
        v = np.array([r * np.cos(a), r * np.sin(a), z])
        v /= np.linalg.norm(v)
        if any(abs(float(v @ b)) > kappa for b in aus + out):
            continue
        out.append(v)
        if len(out) >= n:
            break
    return out


def _ebenen():
    """Zehn Ebenen: die drei bestehenden plus sieben neue Normalen."""
    fest = {
        'ebene_flach': np.array([0.0, 0.0, 1.0]),
        'ebene_gekippt': np.array([0.0, np.sin(np.deg2rad(50)),
                                   np.cos(np.deg2rad(50))]),
        'ebene_diagonal': np.ones(3) / np.sqrt(3),
    }
    eintraege = {}
    for k, n in fest.items():
        eintraege[k] = (n / np.linalg.norm(n), '')
    for i, v in enumerate(_fibonacci_halbkugel(7, ausschluss=list(fest.values()))):
        eintraege[f'ebene_fib_{i}'] = (
            v, 'gleichverteilte Normale aus der Fibonacci-Halbkugel')
    return eintraege


def _registry_bauen():
    import koerper
    import text_volumen
    import organismen
    import baugruppen
    import superquadriken
    import woerter
    reg = {}

    def _legen(k, eintrag):
        """Eintragen, aber niemals stillschweigend ueberschreiben.

        Aufgefallen war das an `kegel`: `koerper.py` hatte den Namen ebenfalls
        vergeben, und die spaeter eingetragene Fassung verdraengte die alte.
        Der Fehler war nach aussen unsichtbar — `build('kegel')` lieferte ueber
        die `KEYS`-Abkuerzung weiter das richtige Netz, aber die Flaechenzahl
        war um eins zu klein, und `build_erweitert` haette ein anderes Netz
        gebaut als `build`. Genau die Sorte Unterschied, die man in einer
        Guete-Tabelle mit vierundachtzig Zeilen nicht mehr findet.
        """
        if k in reg:
            raise KeyError(f'Flaechenschluessel {k!r} doppelt vergeben — '
                           f'die Gruppen {reg[k]["gruppe"]!r} und '
                           f'{eintrag["gruppe"]!r} benutzen ihn beide.')
        reg[k] = eintrag

    for k, (nrm, notiz) in _ebenen().items():
        if k in ('ebene_flach', 'ebene_gekippt', 'ebene_diagonal'):
            _legen(k, dict(gruppe='ebene', alt=True, view=None, notiz=notiz))
            continue
        _legen(k, dict(
            label=f'Ebene, Normale ({nrm[0]:+.2f},{nrm[1]:+.2f},{nrm[2]:+.2f})',
            bauer=(lambda v=nrm: _plane_mesh(v)), gruppe='ebene',
            view=-nrm, notiz=notiz, heldout=False))

    for k in ('kugel', 'wuerfel', 'ei', 'bunny', 'prisma', 'kegel', 'torus'):
        _legen(k, dict(gruppe='primitiv' if k != 'bunny' else 'extern',
                       alt=True, view=None, notiz=''))

    for tab, heldout in ((koerper.KOERPER, False),
                         (koerper.KOERPER_HELDOUT, True)):
        for k, (label, bauer, notiz) in tab.items():
            _legen(k, dict(label=label, bauer=bauer, gruppe='primitiv',
                           view=(0.0, 0.0, -1.0), notiz=notiz,
                           heldout=heldout))

    for tab, heldout in ((text_volumen.TEXT_KOERPER, False),
                         (text_volumen.TEXT_HELDOUT, True)):
        for k, (label, bauer, notiz) in tab.items():
            _legen(k, dict(label=label, bauer=bauer, gruppe='buchstabe',
                           view=(0.0, 0.0, -1.0), notiz=notiz,
                           heldout=heldout))

    # ── Erweiterung auf 200 Flaechen: vier neue, selbstgemachte Kategorien ──
    # Aus jeder ein kleiner Anteil zurueckgehalten (zwei je Kategorie), damit
    # `val_flaeche` (bekannte Dichte, unbekannte Geometrie) auch die neuen
    # Formfamilien prueft und nicht nur die urspruenglichen.
    _HELDOUT_NEU = {
        'organismus_00', 'organismus_38',
        'bg_hantel_var', 'bg_bogen_var',
        'sq_kreuzquer_lang', 'sq_fass_lang',
        'wort_vita', 'wort_ias',
    }
    for k, (label, bauer, notiz) in organismen.ORGANISMEN.items():
        _legen(k, dict(label=label, bauer=bauer, gruppe='organismus',
                       view=(0.0, 0.0, -1.0), notiz=notiz,
                       heldout=k in _HELDOUT_NEU))
    for k, (label, bauer, notiz) in baugruppen.BAUGRUPPEN.items():
        _legen(k, dict(label=label, bauer=bauer, gruppe='baugruppe',
                       view=(0.0, 0.0, -1.0), notiz=notiz,
                       heldout=k in _HELDOUT_NEU))
    for k, (label, bauer, notiz) in superquadriken.SUPERQUADRIKEN.items():
        _legen(k, dict(label=label, bauer=bauer, gruppe='superquadrik',
                       view=(0.0, 0.0, -1.0), notiz=notiz,
                       heldout=k in _HELDOUT_NEU))
    for k, (label, bauer, notiz) in woerter.WOERTER_KOERPER.items():
        _legen(k, dict(label=label, bauer=bauer, gruppe='wort',
                       view=(0.0, 0.0, -1.0), notiz=notiz,
                       heldout=k in _HELDOUT_NEU))
    return reg


_REGISTRY = None


def registry():
    """key -> Eintrag. Wird beim ersten Zugriff gebaut, danach behalten.

    Die externen Netze stehen nicht darin: sie muessen heruntergeladen und
    durchs Aufnahmetor geschickt werden, und beides soll nicht bei jedem
    Import passieren. `externe_aufnehmen()` traegt sie nach.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _registry_bauen()
    return _REGISTRY


def externe_aufnehmen(verbose=True):
    """Die externen Netze pruefen und die bestandenen ins Verzeichnis legen.

    -> (aufgenommen [keys], abgelehnt {key: begruendung})
    """
    import externe_netze
    reg = registry()
    ja, nein = externe_netze.aufnehmen(verbose=verbose)
    neu = []
    for k, (label, mesh, notiz) in ja.items():
        schl = f'extern_{k}'
        reg[schl] = dict(label=label, bauer=(lambda m=mesh: m),
                         gruppe='extern', view=(0.0, -1.0, 0.0), notiz=notiz,
                         heldout=False)
        neu.append(schl)
    return neu, nein


def alle_keys(gruppen=None, mit_heldout=False, nur_heldout=False):
    """Schluessel der erweiterten Liste, nach Gruppe gefiltert."""
    reg = registry()
    out = []
    for k, e in reg.items():
        h = bool(e.get('heldout', False))
        if nur_heldout and not h:
            continue
        if not mit_heldout and not nur_heldout and h:
            continue
        if gruppen and e['gruppe'] not in gruppen:
            continue
        out.append(k)
    return out


def gruppe_von(key):
    return registry()[key]['gruppe']


# ── Zufaellige Blickrichtungen ───────────────────────────────────────────────

def guete_richtung(mesh, richtung, n=600, seed=0):
    """Wie brauchbar ein Blickwinkel auf ein Netz ist.

    Zwei Zahlen, die verschiedene Dinge messen:

    * `treffer` — Anteil des Projektionsquadrats, unter dem ueberhaupt
      Geometrie liegt. Das ist der Flaechenanteil der Silhouette. Ist er
      klein, faellt der groesste Teil der 2D-Dichte neben den Koerper, und die
      projizierte Bahn besteht ueberwiegend aus Naechster-Punkt-Ersatz.
    * `fehlschuss` — Anteil der Strahlen, die auf einen *tatsaechlich
      vorhandenen* Oberflaechenpunkt gezielt sind und ihn trotzdem verfehlen.
      Bei einem geschlossenen Koerper ist das null; steigt der Wert, ist das
      Netz aus dieser Richtung durchlaessig.

    Die beiden zu trennen ist der Kern: eine Kugel hat aus jeder Richtung
    `treffer` = pi/4 und `fehlschuss` = 0. Ein offener Helm kann `treffer` =
    0,7 und trotzdem `fehlschuss` = 0,1 haben. Nur die zweite Zahl sagt etwas
    ueber die Netzqualitaet.
    """
    import open3d as o3d
    import trimesh
    e1, e2, w = _frame(richtung)
    szene = o3d.t.geometry.RaycastingScene()
    szene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.int32)))

    P, _ = trimesh.sample.sample_surface(mesh, n, seed=int(seed))
    P = np.asarray(P, dtype=np.float64)
    a, b = P @ e1, P @ e2
    a0, a1, b0, b1 = a.min(), a.max(), b.min(), b.max()

    def _anteil(u, v):
        org = (np.outer(u, e1) + np.outer(v, e2)) - 4.0 * w
        d = np.tile(w.astype(np.float32), (len(org), 1))
        rays = o3d.core.Tensor(np.hstack([org.astype(np.float32), d]),
                               dtype=o3d.core.float32)
        return float(np.isfinite(szene.cast_rays(rays)['t_hit'].numpy()).mean())

    rng = np.random.default_rng(int(seed) + 7)
    treffer = _anteil(rng.uniform(a0, a1, n), rng.uniform(b0, b1, n))
    fehlschuss = 1.0 - _anteil(a, b)
    return dict(treffer=treffer, fehlschuss=fehlschuss)


MIN_TREFFER = 0.55
MAX_FEHLSCHUSS = 0.25


def zufaellige_blickrichtung(rng, mesh, min_treffer=MIN_TREFFER,
                             max_fehlschuss=MAX_FEHLSCHUSS, versuche=24,
                             meiden=(), min_winkel=35.0, n=600):
    """Eine Blickrichtung aus der Halbkugel ziehen, bis sie taugt.

    -> (richtung (3,), guete dict, bestanden bool)

    Verworfen wird bei zu kleiner Silhouette oder zu durchlaessigem Netz, und
    ebenso, wenn die Richtung einer schon vergebenen zu nahe kommt — zwei fast
    gleiche Winkel auf derselben Flaeche waeren zwei fast gleiche Eintraege.

    Findet sich nach `versuche` Zuegen keine, die beide Schwellen haelt, wird
    die beste gefundene zurueckgegeben und `bestanden` ist False. Das ist
    Absicht: es gibt Koerper — der Armadillo etwa, mit seinen abstehenden
    Gliedmassen — deren Silhouette aus *keiner* Richtung 55 % des Quadrats
    fuellt. Sie hier stumm fallenzulassen hiesse, die Flaechenliste von einer
    Schwelle bestimmen zu lassen, statt von der Geometrie. Der Bauer schreibt
    stattdessen mit, welche Winkel nur mit Abstrichen genommen wurden, und die
    Guetefilter der Datenbank raeumen danach je Eintrag auf.
    """
    kappa = np.cos(np.deg2rad(min_winkel))
    gemieden = [np.asarray(m, np.float64) / max(np.linalg.norm(m), 1e-12)
                for m in meiden]
    beste, beste_g, beste_punkte = None, None, -np.inf
    for i in range(versuche):
        v = rng.normal(size=3)
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        v = v / nv
        if any(float(v @ m) > kappa for m in gemieden):
            continue
        g = guete_richtung(mesh, v, n=n, seed=int(rng.integers(0, 2 ** 31)))
        # Punktzahl nur fuer den Rueckfall: viel Silhouette, wenig Durchlass.
        punkte = g['treffer'] - 2.0 * g['fehlschuss']
        if punkte > beste_punkte:
            beste, beste_g, beste_punkte = v, g, punkte
        if g['treffer'] >= min_treffer and g['fehlschuss'] <= max_fehlschuss:
            return v, g, True
    if beste is None:                      # alle Zuege lagen zu nah an `meiden`
        v = rng.normal(size=3)
        v /= max(np.linalg.norm(v), 1e-12)
        return v, guete_richtung(mesh, v, n=n), False
    return beste, beste_g, False


def blickrichtungen(rng, mesh, anzahl=2, **kw):
    """`anzahl` moeglichst verschiedene brauchbare Richtungen."""
    out = []
    for _ in range(anzahl):
        v, g, ok = zufaellige_blickrichtung(
            rng, mesh, meiden=[r for r, _, _ in out], **kw)
        out.append((v, g, ok))
    return out


def build_erweitert(key):
    """Eine Flaeche aus der erweiterten Liste bauen."""
    e = registry()[key]
    if e.get('alt'):
        return _build_alt(key)
    return Surface(key, e['label'], e['bauer'](), view=e['view'],
                   note=e['notiz'])


def build(key):
    """Eine Flaeche bauen.

    Fuer die zehn urspruenglichen Schluessel laeuft das ueber `_build_alt` und
    liefert Netz *und* Blickrichtung unveraendert; die bestehenden
    Auswertungsskripte merken von der Erweiterung nichts. Alles andere kommt
    aus dem Verzeichnis, das dabei beim ersten Mal gebaut wird.
    """
    if key in KEYS:
        return _build_alt(key)
    return build_erweitert(key)


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
