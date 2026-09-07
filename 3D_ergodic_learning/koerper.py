r"""
koerper.py
==========
Dreissig selbstgebaute Zielkoerper plus zwei zurueckgehaltene.

Warum selbstgebaut und nicht heruntergeladen: die Flaechen sollen die
*Geometrie* durchmustern, gegen die eine ergodische Bahn gefuehrt werden muss —
ebene Facetten mit scharfen Kanten, einfache und doppelte Kruemmung,
Sattelpunkte, Loecher, Spitzen. Ein Netz aus dem Internet bringt davon je einen
zufaelligen Ausschnitt mit; hier steht jede Klasse ausdruecklich einmal da.

Vier Gruppen, in der Reihenfolge zunehmender Kruemmung:

* **platonisch/archimedisch** — nur ebene Facetten, alle Normalen konstant je
  Facette. Konvexe Huelle bekannter Eckpunkte.
* **Prismen und Spate** — Extrusion eines Vielecks, plus die beiden Scherungen
  des Quaders, bei denen keine Kante mehr rechtwinklig steht.
* **Pyramiden und Verwandte** — eine Spitze oder eine gegen die Grundflaeche
  verdrehte Deckflaeche; hier wird die Projektionsrichtung erstmals wichtig.
* **Rotationskoerper** — einfach gekruemmt (Zylinder, Kegel), doppelt gekruemmt
  (Ellipsoid, Tropfen), negativ gekruemmt (Hyperboloid, Katenoid), und mit Loch
  (Torus, Hohlzylinder).

Alle Bauer geben ein `trimesh.Trimesh` zurueck, um den Ursprung zentriert und
von der Groessenordnung eins. Das Einpassen in [0,1]^3 macht `surfaces._fit_unit`,
damit dort genau eine Stelle ueber die Groesse entscheidet.

Selbsttest:

    python koerper.py --pruefen
"""
import numpy as np

# ── Hilfen ───────────────────────────────────────────────────────────────────


def _hull(punkte):
    """Konvexe Huelle einer Punktwolke als wasserdichtes Netz.

    Fuer die konvexen Koerper ist das der kuerzeste korrekte Weg: eine
    Flaechenliste von Hand aufzuschreiben waere fehleranfaellig und muesste die
    Fuenf- und Sechsecke ohnehin triangulieren. `ConvexHull` liefert genau die
    Triangulierung, und `fix_normals` richtet sie nach aussen.
    """
    import trimesh
    from scipy.spatial import ConvexHull
    p = np.asarray(punkte, dtype=np.float64)
    h = ConvexHull(p)
    m = trimesh.Trimesh(vertices=p[h.vertices],
                        faces=_reindex(h, p), process=True)
    m.fix_normals()
    return m


def _reindex(hull, punkte):
    """Flaechen der Huelle auf die Teilmenge `hull.vertices` umnummerieren."""
    abbild = {alt: neu for neu, alt in enumerate(hull.vertices)}
    return np.array([[abbild[i] for i in f] for f in hull.simplices])


def _mesh(verts, faces):
    """Netz aus expliziten Listen; Normalen werden nach aussen gedreht."""
    import trimesh
    m = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64),
                        faces=np.asarray(faces, dtype=np.int64), process=False)
    m.fix_normals()
    return m


def _ring(n, radius=1.0, z=0.0, phase=0.0):
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False) + phase
    return np.stack([radius * np.cos(a), radius * np.sin(a),
                     np.full(n, float(z))], axis=1)


def _polygon_prisma(xy, hoehe):
    """Vieleck (n,2) entlang z extrudieren, um z=0 zentriert."""
    import trimesh
    from shapely.geometry import Polygon
    m = trimesh.creation.extrude_polygon(Polygon(np.asarray(xy)), height=hoehe)
    v = np.asarray(m.vertices, dtype=np.float64).copy()
    v[:, 2] -= hoehe / 2.0
    return _mesh(v, m.faces)


def _drehen(profil, sections=64):
    """Profilkurve (n,2) als (Radius, Hoehe) um die z-Achse drehen.

    Die Profile hier beginnen und enden auf der Achse (r = 0). Dort faellt der
    Ring zu einem Punkt zusammen, `revolve` schliesst ihn zu einem Faecher, und
    der Koerper ist ohne Deckel wasserdicht. Ein Profil, das mit r > 0 endet,
    ergaebe eine offene Roehre — deshalb laufen selbst Katenoid und
    Hohlprofile hier ueber die Achse zurueck.
    """
    import trimesh
    m = trimesh.creation.revolve(np.asarray(profil, dtype=np.float64),
                                 sections=sections)
    m.fix_normals()
    return m


# ── Gruppe 1: platonisch und archimedisch ────────────────────────────────────

def tetraeder():
    v = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
    return _hull(v / np.sqrt(3))


def oktaeder():
    v = np.zeros((6, 3))
    v[0, 0] = v[2, 1] = v[4, 2] = 1.0
    v[1, 0] = v[3, 1] = v[5, 2] = -1.0
    return _hull(v)


def dodekaeder():
    phi = (1 + np.sqrt(5)) / 2
    wuerfel = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1)
                        for c in (-1, 1)], dtype=np.float64)
    rechtecke = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            rechtecke += [[0, s1 / phi, s2 * phi],
                          [s1 / phi, s2 * phi, 0],
                          [s1 * phi, 0, s2 / phi]]
    return _hull(np.vstack([wuerfel, np.array(rechtecke)]) / np.sqrt(3))


def ikosaeder():
    phi = (1 + np.sqrt(5)) / 2
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [[0, s1, s2 * phi], [s1, s2 * phi, 0], [s1 * phi, 0, s2]]
    return _hull(np.array(v, dtype=np.float64) / np.sqrt(1 + phi ** 2))


def kuboktaeder():
    """Alle Permutationen von (+-1, +-1, 0) — Quadrate und Dreiecke im Wechsel."""
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            v += [[s1, s2, 0], [s1, 0, s2], [0, s1, s2]]
    return _hull(np.array(v, dtype=np.float64) / np.sqrt(2))


def oktaederstumpf():
    """Alle Permutationen von (0, +-1, +-2): sechs Quadrate, acht Sechsecke."""
    v = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            for a, b in ((1, 2), (2, 1)):
                v += [[0, s1 * a, s2 * b], [s1 * a, s2 * b, 0],
                      [s1 * b, 0, s2 * a]]
    return _hull(np.unique(np.array(v, dtype=np.float64), axis=0) / np.sqrt(5))


# ── Gruppe 2: Prismen und Spate ──────────────────────────────────────────────

def quader():
    import trimesh
    return _mesh(*_vf(trimesh.creation.box(extents=(1.0, 0.62, 0.42))))


def _vf(m):
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces)


def _schere(m, S):
    v, f = _vf(m)
    return _mesh(v @ np.asarray(S, dtype=np.float64).T, f)


def spat():
    """Parallelepiped: der Quader, entlang zweier Achsen geschert.

    Keine Kante steht mehr rechtwinklig auf einer anderen — die Projektion
    trifft drei Facetten unter drei verschiedenen Winkeln.
    """
    import trimesh
    S = np.array([[1.0, 0.42, 0.28], [0.0, 1.0, 0.35], [0.0, 0.0, 1.0]])
    return _schere(trimesh.creation.box(extents=(0.9, 0.7, 0.6)), S)


def rhomboeder():
    """Der Sonderfall des Spats mit drei gleich langen Kanten und gleichem
    Winkel zwischen je zweien — schief, aber vollstaendig symmetrisch."""
    import trimesh
    a = np.deg2rad(72.0)
    c = np.cos(a)
    # Gram-Matrix (1, c, c; c, 1, c; c, c, 1) ueber die Cholesky-Zerlegung in
    # drei Kantenvektoren gleicher Laenge und gleichen Zwischenwinkels.
    G = np.full((3, 3), c) + np.eye(3) * (1.0 - c)
    S = np.linalg.cholesky(G).T
    return _schere(trimesh.creation.box(extents=(0.8, 0.8, 0.8)), S)


def dreiecksplatte():
    """Flach liegendes Dreiecksprisma.

    Bewusst *nicht* dieselben Masse wie `surfaces.prisma`: das bestehende
    Dreiecksprisma ist ein aufrechter Stab mit Hoehe 1 und Umkreis 0,62. Waere
    dies eine zweite Ausgabe davon, haetten zwei Schluessel dasselbe Netz, und
    eine Auswertung nach Flaeche wuerde denselben Fall doppelt zaehlen. Als
    flache Platte ist es ein anderer Fall: die Deckflaeche traegt fast alles,
    die drei Seiten fast nichts.
    """
    a = np.deg2rad([90.0, 210.0, 330.0])
    return _polygon_prisma(np.stack([0.70 * np.cos(a), 0.70 * np.sin(a)], 1), 0.22)


def sechseckprisma():
    a = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    return _polygon_prisma(np.stack([0.58 * np.cos(a), 0.58 * np.sin(a)], 1), 0.8)


def sternprisma(zacken=5, r_aussen=0.62, r_innen=0.27):
    a = np.linspace(0, 2 * np.pi, 2 * zacken, endpoint=False) + np.pi / 2
    r = np.where(np.arange(2 * zacken) % 2 == 0, r_aussen, r_innen)
    return _polygon_prisma(np.stack([r * np.cos(a), r * np.sin(a)], 1), 0.35)


# ── Gruppe 3: Pyramiden und Verwandte ────────────────────────────────────────

def pyramide(seite=0.8, hoehe=0.9):
    h, s = hoehe / 2.0, seite / 2.0
    v = [[-s, -s, -h], [s, -s, -h], [s, s, -h], [-s, s, -h], [0, 0, h]]
    f = [[0, 2, 1], [0, 3, 2],                        # Grundflaeche
         [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    return _mesh(v, f)


def bipyramide(seite=0.8, hoehe=0.75):
    h, s = hoehe, seite / 2.0
    v = [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [0, 0, h], [0, 0, -h]]
    f = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
         [1, 0, 5], [2, 1, 5], [3, 2, 5], [0, 3, 5]]
    return _mesh(v, f)


def pyramidenstumpf(unten=0.9, oben=0.36, hoehe=0.8):
    h, a, b = hoehe / 2.0, unten / 2.0, oben / 2.0
    v = [[-a, -a, -h], [a, -a, -h], [a, a, -h], [-a, a, -h],
         [-b, -b, h], [b, -b, h], [b, b, h], [-b, b, h]]
    f = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
    for i in range(4):
        j = (i + 1) % 4
        f += [[i, j, j + 4], [i, j + 4, i + 4]]
    return _mesh(v, f)


def keil(laenge=1.0, breite=0.6, hoehe=0.7):
    """Dreiecksprisma liegend: eine geneigte Flaeche, eine senkrechte, eine
    waagerechte — drei sehr verschiedene Winkel zum Projektor."""
    l, b, h = laenge / 2.0, breite / 2.0, hoehe / 2.0
    v = [[-l, -b, -h], [l, -b, -h], [l, b, -h], [-l, b, -h],
         [-l, -b, h], [-l, b, h]]
    f = [[0, 2, 1], [0, 3, 2],          # Boden
         [0, 1, 4], [3, 5, 2],          # Stirnseiten (Dreiecke)
         [1, 2, 5], [1, 5, 4],          # geneigte Flaeche
         [0, 4, 5], [0, 5, 3]]          # senkrechte Rueckwand
    return _mesh(v, f)


def antiprisma(n=6, radius=0.6, hoehe=0.7):
    """Zwei gegeneinander verdrehte n-Ecke, dazwischen ein Band aus Dreiecken.

    Konvex, also ueber die Huelle gebaut: die Flaechenliste von Hand haette
    dieselben Dreiecke ergeben, nur mit mehr Gelegenheit, sich zu vertun.
    """
    o = _ring(n, radius, hoehe / 2.0, phase=np.pi / n)
    u = _ring(n, radius, -hoehe / 2.0)
    return _hull(np.vstack([o, u]))


def trapezoeder(n=5, radius=0.62, versatz=0.30, spitze=0.85):
    """n-seitiges Trapezoeder: 2n Drachenflaechen, zwei Spitzen.

    Der Dual des Antiprismas. Zwei gegeneinander verdrehte Ringe auf
    verschiedenen Hoehen plus zwei Spitzen auf der Achse.
    """
    o = _ring(n, radius, versatz, phase=np.pi / n)
    u = _ring(n, radius, -versatz)
    s = np.array([[0.0, 0.0, spitze], [0.0, 0.0, -spitze]])
    return _hull(np.vstack([o, u, s]))


# ── Gruppe 4: Rotationskoerper ───────────────────────────────────────────────

def zylinder(radius=0.5, hoehe=0.9):
    import trimesh
    return _mesh(*_vf(trimesh.creation.cylinder(radius=radius, height=hoehe,
                                                sections=64)))


def hohlzylinder(r_innen=0.28, r_aussen=0.55, hoehe=0.7):
    import trimesh
    return _mesh(*_vf(trimesh.creation.annulus(r_min=r_innen, r_max=r_aussen,
                                               height=hoehe, sections=64)))


def kegel_schlank(radius=0.30, hoehe=1.15):
    """Schlanker, hoher Kegel — schmaler als `surfaces.kegel` (0,55 x 1,0).

    Derselbe Grund wie bei der Dreiecksplatte: gleiche Masse waeren dasselbe
    Netz unter zweitem Namen. Schlank ist ausserdem der interessantere Fall,
    weil die Mantelflaeche steiler steht und die Normalen staerker streuen.
    """
    import trimesh
    m = trimesh.creation.cone(radius=radius, height=hoehe, sections=64)
    v, f = _vf(m)
    v[:, 2] -= hoehe / 2.0
    return _mesh(v, f)


def kegelstumpf(r_unten=0.58, r_oben=0.22, hoehe=0.85):
    h = hoehe / 2.0
    return _drehen([[0.0, -h], [r_unten, -h], [r_oben, h], [0.0, h]])


def torus_dick(major=0.30, minor=0.24):
    """Dicker Torus mit fast geschlossenem Loch.

    `surfaces.torus` ist schlank (0,35 / 0,16) und laesst von oben ein weites
    Loch offen. Hier ist der Ring so dick, dass die Projektion das Loch kaum
    noch findet — die negativ gekruemmte Innenseite kommt dafuer staerker zur
    Geltung.
    """
    import trimesh
    return _mesh(*_vf(trimesh.creation.torus(major_radius=major,
                                             minor_radius=minor,
                                             major_sections=48,
                                             minor_sections=24)))


def ellipsoid(skalen=(0.62, 0.42, 0.30), subdiv=4):
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    return _mesh(np.asarray(m.vertices) * np.asarray(skalen, dtype=np.float64),
                 m.faces)


def paraboloid(radius=0.6, hoehe=0.9, n=48):
    """z = (r/R)^2 * H, unten mit einer Scheibe geschlossen."""
    r = np.linspace(0.0, radius, n)
    z = (r / radius) ** 2 * hoehe - hoehe / 2.0
    profil = np.vstack([np.stack([r, z], 1),
                        [[radius, -hoehe / 2.0], [0.0, -hoehe / 2.0]]])
    return _drehen(profil)


def hyperboloid(hals=0.24, hoehe=0.9, weite=0.55, n=40):
    """Einschaliges Hyperboloid: r(z) = hals * sqrt(1 + (z/weite)^2).

    Der einzige Koerper hier mit durchweg *negativer* Gauss-Kruemmung: jede
    Tangentialebene schneidet die Flaeche. Die Taille traegt kaum Flaeche, die
    Enden viel — die Projektion trifft beides gleichzeitig.
    """
    z = np.linspace(-hoehe / 2.0, hoehe / 2.0, n)
    r = hals * np.sqrt(1.0 + (z / weite) ** 2)
    profil = np.vstack([[[0.0, z[0]]], np.stack([r, z], 1), [[0.0, z[-1]]]])
    return _drehen(profil)


def katenoid(hals=0.20, hoehe=0.85, n=40):
    """r(z) = hals * cosh(z / hals) — die Minimalflaeche unter den Spulen.

    Gegenueber dem Hyperboloid faellt die Taille schaerfer ab; die beiden
    stehen nebeneinander, weil sie sich in der Silhouette aehneln und in der
    Kruemmungsverteilung nicht.
    """
    z = np.linspace(-hoehe / 2.0, hoehe / 2.0, n)
    r = hals * np.cosh(z / hals)
    profil = np.vstack([[[0.0, z[0]]], np.stack([r, z], 1), [[0.0, z[-1]]]])
    return _drehen(profil)


def spindel(radius=0.42, hoehe=1.0, n=48):
    """Zwei Spitzen, dazwischen ein Kreisbogen — ein Zitronenkoerper."""
    t = np.linspace(-np.pi / 2, np.pi / 2, n)
    z = np.sin(t) * hoehe / 2.0
    r = np.cos(t) * radius
    return _drehen(np.stack([r, z], 1))


def tropfen(radius=0.45, hoehe=1.0, n=56):
    """Unten kugelig, oben in eine Spitze auslaufend.

    Das Profil laeuft mit einer Wurzel gegen die Achse, nicht mit einer dritten
    Potenz. Der Unterschied ist nicht kosmetisch: bei einer kubisch
    auslaufenden Spitze liegen die letzten Profilpunkte so dicht an der Achse,
    dass `revolve` dort entartete Dreiecke erzeugt, die beim Zusammenfassen
    doppelter Ecken wegfallen — das Netz war dann nicht mehr wasserdicht. Die
    Wurzel gibt der Spitze eine senkrechte Tangente und laesst alle Ringe
    endlich gross.
    """
    u = np.linspace(-1.0, 1.0, n)
    r = radius * 1.6 * np.sqrt(np.clip(1.0 - u ** 2, 0.0, None)) \
        * np.sqrt((1.0 - u) / 2.0)
    return _drehen(np.stack([r, u * hoehe / 2.0], 1))


def oloid(radius=0.5, n=96):
    """Konvexe Huelle zweier senkrecht zueinander stehender Kreise, deren
    Mittelpunkte um einen Radius versetzt sind.

    Abwickelbar und trotzdem in keiner Richtung eben — beim Abrollen beruehrt
    er den Boden entlang seiner ganzen Oberflaeche.
    """
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k1 = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n)], 1)
    k1[:, 0] -= radius / 2.0
    k2 = np.stack([radius * np.cos(a) + radius / 2.0, np.zeros(n),
                   radius * np.sin(a)], 1)
    return _hull(np.vstack([k1, k2]))


# ── Zurueckgehalten ──────────────────────────────────────────────────────────

def kapsel(radius=0.3, hoehe=0.7):
    import trimesh
    m = trimesh.creation.capsule(height=hoehe, radius=radius, count=[32, 32])
    v, f = _vf(m)
    v[:, 2] -= hoehe / 2.0
    return _mesh(v, f)


def halbkugel(radius=0.55, n=48):
    """Viertelkreisprofil, gedreht — Kuppel plus Bodenscheibe.

    Ausdruecklich ohne boolesche Verknuepfung gebaut: `manifold3d` ist in
    dieser Umgebung nicht installiert, und `trimesh` faellt ohne es beim
    Schneiden stillschweigend auf ein Ergebnis zurueck, das nicht mehr
    wasserdicht ist.
    """
    t = np.linspace(0.0, np.pi / 2, n)
    profil = np.vstack([[[0.0, 0.0]],
                        np.stack([radius * np.cos(t), radius * np.sin(t)], 1)])
    return _drehen(profil)


# ── Verzeichnis ──────────────────────────────────────────────────────────────
#  Schluessel -> (Bezeichnung, Bauer, Notiz)

KOERPER = {
    # platonisch / archimedisch
    'tetraeder':      ('Tetraeder', tetraeder, 'vier Dreiecke, die kleinste geschlossene Facettenflaeche'),
    'oktaeder':       ('Oktaeder', oktaeder, 'acht Dreiecke, Facetten paarweise parallel'),
    'dodekaeder':     ('Dodekaeder', dodekaeder, 'zwoelf Fuenfecke — fast eine Kugel aus ebenen Stuecken'),
    'ikosaeder':      ('Ikosaeder', ikosaeder, 'zwanzig Dreiecke, dichteste platonische Annaeherung an die Kugel'),
    'kuboktaeder':    ('Kuboktaeder', kuboktaeder, 'Quadrate und Dreiecke im Wechsel'),
    'oktaederstumpf': ('Oktaederstumpf', oktaederstumpf, 'sechs Quadrate, acht Sechsecke — fuellt den Raum lueckenlos'),
    # Prismen / Spate
    'quader':          ('Quader', quader, 'drei verschiedene Kantenlaengen, alle Winkel recht'),
    'spat':            ('Spat', spat, 'geschert: keine zwei Kanten mehr rechtwinklig'),
    'rhomboeder':      ('Rhomboeder', rhomboeder, 'schief, aber mit drei gleichen Kanten und Winkeln'),
    'dreiecksplatte':  ('Dreiecksplatte', dreiecksplatte, 'flach liegendes Dreiecksprisma: Deckflaeche traegt fast alles'),
    'sechseckprisma':  ('Sechseckprisma', sechseckprisma, 'sechs schmale Seiten, flache Deckflaeche'),
    'sternprisma':     ('Sternprisma', sternprisma, 'fuenf einspringende Ecken — die Silhouette ist nicht konvex'),
    # Pyramiden
    'pyramide':        ('Pyramide', pyramide, 'quadratische Grundflaeche, vier geneigte Dreiecke'),
    'bipyramide':      ('Bipyramide', bipyramide, 'zwei Pyramiden Ruecken an Ruecken, Spitzen auf der Achse'),
    'pyramidenstumpf': ('Pyramidenstumpf', pyramidenstumpf, 'Deckflaeche kleiner als Grundflaeche'),
    'keil':            ('Keil', keil, 'eine geneigte, eine senkrechte, eine waagerechte Flaeche'),
    'antiprisma':      ('Antiprisma', antiprisma, 'zwei verdrehte Sechsecke mit Dreiecksband dazwischen'),
    'trapezoeder':     ('Trapezoeder', trapezoeder, 'zehn Drachenflaechen zwischen zwei Spitzen'),
    # Rotationskoerper
    'zylinder':      ('Zylinder', zylinder, 'einfach gekruemmter Mantel, zwei ebene Deckel'),
    'hohlzylinder':  ('Hohlzylinder', hohlzylinder, 'Loch entlang der Achse: die Innenwand bleibt unbeschienen'),
    'kegel_schlank': ('Schlanker Kegel', kegel_schlank, 'steiler Mantel, Kruemmung nimmt zur Spitze hin zu'),
    'kegelstumpf':   ('Kegelstumpf', kegelstumpf, 'wie der Kegel, aber mit ebener Deckflaeche'),
    'torus_dick':    ('Dicker Torus', torus_dick, 'fast geschlossenes Loch, ausgepraegte Innenseite'),
    'ellipsoid':     ('Ellipsoid', ellipsoid, 'drei verschiedene Halbachsen, doppelt gekruemmt'),
    'paraboloid':    ('Paraboloid', paraboloid, 'Schale, deren Kruemmung nach aussen abnimmt'),
    'hyperboloid':   ('Hyperboloid', hyperboloid, 'durchweg negativ gekruemmt: jede Tangentialebene schneidet'),
    'katenoid':      ('Katenoid', katenoid, 'Spule mit scharf einschnuerender Taille'),
    'spindel':       ('Spindel', spindel, 'zwei Spitzen, dazwischen ein Kreisbogen'),
    'tropfen':       ('Tropfen', tropfen, 'unten kugelig, oben zur Spitze auslaufend'),
    'oloid':         ('Oloid', oloid, 'abwickelbar und trotzdem nirgends eben'),
}

KOERPER_HELDOUT = {
    'kapsel':    ('Kapsel', kapsel, 'Zylinder mit Kugelkappen — Uebergang ohne Kante'),
    'halbkugel': ('Halbkugel', halbkugel, 'Kuppel und ebene Scheibe an einer scharfen Kante'),
}


def baue(key):
    for tab in (KOERPER, KOERPER_HELDOUT):
        if key in tab:
            return tab[key][1]()
    raise KeyError(key)


# ── Selbsttest ───────────────────────────────────────────────────────────────

def pruefen(verbose=True):
    """Jeden Koerper bauen und auf Wasserdichtheit pruefen. -> Liste der Fehler."""
    fehler = []
    for tab, marke in ((KOERPER, ''), (KOERPER_HELDOUT, ' [Heldout]')):
        for k, (label, bauer, _) in tab.items():
            try:
                m = bauer()
            except Exception as e:                     # noqa: BLE001
                fehler.append((k, f'Bau fehlgeschlagen: {type(e).__name__}: {e}'))
                continue
            grund = []
            if not m.is_watertight:
                grund.append('nicht wasserdicht')
            if not m.is_winding_consistent:
                grund.append('Umlaufsinn uneinheitlich')
            if m.volume <= 1e-9:
                grund.append(f'Volumen {m.volume:.3g}')
            if len(m.faces) < 4:
                grund.append(f'nur {len(m.faces)} Dreiecke')
            if grund:
                fehler.append((k, ', '.join(grund)))
            if verbose:
                zeichen = 'ok ' if not grund else 'FEHLER'
                print(f'  {zeichen} {k:16s} {len(m.vertices):6d} Ecken '
                      f'{len(m.faces):6d} Dreiecke  V={m.volume:.4f}'
                      f'  A={m.area:.4f}{marke}')
    return fehler


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pruefen', action='store_true')
    a = p.parse_args()
    print(f'{len(KOERPER)} Koerper + {len(KOERPER_HELDOUT)} zurueckgehalten\n')
    f = pruefen()
    print()
    if f:
        for k, g in f:
            print(f'  FEHLER {k}: {g}')
        raise SystemExit(1)
    print('Alle Koerper wasserdicht.')
