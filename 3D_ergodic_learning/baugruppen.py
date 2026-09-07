r"""
baugruppen.py
=============
Fuenfundzwanzig boolesche Baugruppen aus zwei bis drei Grundkoerpern aus
`koerper.py` — die einzige Formfamilie hier, die durch Vereinigung/Differenz
mehrerer Teile entsteht statt durch eine einzelne Formel oder Extrusion.

Warum das eine eigene Kategorie ist: alles in `koerper.py`,
`superquadriken.py` und `organismen.py` ist *ein* zusammenhaengender Rand
einer einzigen Konstruktion. Ein Loch, das ganz durchgeht (nicht wie beim
Torus rundherum, sondern gerade durch einen sonst massiven Koerper), eine
Kerbe, die eine Kante durchschneidet, oder Zaehne, die aus einem Zylinder
herausstehen, entstehen erst durch die boolesche Verknuepfung mehrerer Teile.

`halbkugel()` in `koerper.py` wurde ausdruecklich *ohne* boolesche
Verknuepfung gebaut, weil `manifold3d` in der damaligen Umgebung fehlte und
`trimesh` beim Schneiden sonst still auf ein nicht mehr wasserdichtes
Ergebnis zurueckfaellt. `manifold3d` ist inzwischen installiert (siehe
Abschnitt in `LIZENZEN.md`); jede Baugruppe hier wird trotzdem, wie ueberall
sonst in diesem Modul, ausdruecklich auf Wasserdichtheit geprueft, statt dem
Ergebnis blind zu vertrauen.

Selbsttest:

    python baugruppen.py --pruefen
"""
import numpy as np
import koerper


def _verschieben(mesh, dx=0.0, dy=0.0, dz=0.0):
    import trimesh
    m = mesh.copy()
    m.apply_translation([dx, dy, dz])
    return m


def _skalieren(mesh, sx=1.0, sy=1.0, sz=1.0):
    m = mesh.copy()
    m.vertices = m.vertices * np.array([sx, sy, sz])
    return m


def _drehen_um_x(mesh, grad):
    m = mesh.copy()
    c, s = np.cos(np.deg2rad(grad)), np.sin(np.deg2rad(grad))
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    m.vertices = m.vertices @ R.T
    return m


def _drehen_um_y(mesh, grad):
    m = mesh.copy()
    c, s = np.cos(np.deg2rad(grad)), np.sin(np.deg2rad(grad))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    m.vertices = m.vertices @ R.T
    return m


def _drehen_um_z(mesh, grad):
    m = mesh.copy()
    c, s = np.cos(np.deg2rad(grad)), np.sin(np.deg2rad(grad))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    m.vertices = m.vertices @ R.T
    return m


def _boolesch(op, teile):
    """`op` in {'vereinigen', 'differenz', 'schnitt'}, manifold3d-Backend."""
    import trimesh
    fn = {'vereinigen': trimesh.boolean.union,
          'differenz': trimesh.boolean.difference,
          'schnitt': trimesh.boolean.intersection}[op]
    m = fn(teile, engine='manifold')
    m.fix_normals()
    return m


# ── Baugruppen ────────────────────────────────────────────────────────────────

def wuerfel_lochkreuz(radius=0.16):
    """Quader mit zwei senkrecht zueinander stehenden, durchgehenden Loechern."""
    q = _skalieren(koerper.quader(), 1.4, 1.4, 1.4)
    l1 = _skalieren(koerper.zylinder(radius=radius, hoehe=2.0), 1, 1, 1)
    l2 = _drehen_um_x(l1, 90.0)
    zwischenstand = _boolesch('differenz', [q, l1])
    return _boolesch('differenz', [zwischenstand, l2])


def torus_steg(radius=0.14):
    """Dicker Torus mit einem Zylindersteg durch das Loch."""
    t = koerper.torus_dick()
    steg = koerper.zylinder(radius=radius, hoehe=0.9)
    return _boolesch('vereinigen', [t, steg])


def kugel_facette(tiefe=0.25):
    """Ellipsoid mit einer flachen, abgeschnittenen Seite.

    Die Schnittbox ist in y/z absichtlich weit groesser als die Kugel (deckt
    dort alles), in x aber ein Halbraum ab der Schwelle `0.55 - tiefe` — sonst
    kappte eine symmetrische, zu kleine Box beide Pole statt einer einzelnen
    Seite, und die Tiefe haette, weil die Box die Kugel in x ohnehin ganz
    umschliesst, gar keine Wirkung auf das Ergebnis.
    """
    k = koerper.ellipsoid(skalen=(0.55, 0.55, 0.55))
    schwelle = 0.55 - tiefe
    schnitt = _verschieben(_skalieren(koerper.quader(), 2.0, 4.0, 4.0),
                           schwelle + 1.0, 0.0, 0.0)
    return _boolesch('differenz', [k, schnitt])


def pyramide_dorn(dornradius=0.10, dornhoehe=0.55):
    """Pyramide mit einem duennen Dorn auf der Spitze.

    Der Kegel muss ein Stueck in die Pyramide hineinreichen, sonst beruehren
    sich beide Teile nur in der Apexspitze — ein einzelner Punkt ist keine
    Beruehrflaeche, und die Vereinigung liefert zwei getrennte Koerper statt
    eines zusammenhaengenden.
    """
    p = koerper.pyramide()
    ueberlapp = 0.08
    dorn = _verschieben(koerper.kegel_schlank(radius=dornradius, hoehe=dornhoehe),
                        0, 0, 0.45 + dornhoehe / 2 - ueberlapp)
    return _boolesch('vereinigen', [p, dorn])


def hantel(abstand=0.55, kugelradius=0.30, stabradius=0.10):
    """Zwei Kugeln, durch einen duennen Stab verbunden — zwei getrennte
    Rundungen mit einer scharfen Taille dazwischen, das genaue Gegenteil
    einer Superquadrik-Taille."""
    import trimesh
    k1 = _verschieben(_skalieren(trimesh.creation.icosphere(subdivisions=3),
                                 kugelradius, kugelradius, kugelradius),
                      -abstand / 2, 0, 0)
    k2 = _verschieben(_skalieren(trimesh.creation.icosphere(subdivisions=3),
                                 kugelradius, kugelradius, kugelradius),
                      abstand / 2, 0, 0)
    # Achse des Zylinders (liegt entlang z) per echter Drehung auf x legen —
    # eine rohe Spaltenvertauschung waere eine Spiegelung und kehrte den
    # Umlaufsinn der Facetten um, was den Booleschen als "kein Volumen" gilt.
    stab = _drehen_um_y(koerper.zylinder(radius=stabradius, hoehe=abstand + 0.1), 90.0)
    return _boolesch('vereinigen', [_boolesch('vereinigen', [k1, k2]), stab])


def l_block(bissen_anteil=0.5):
    """Quader, aus dem ein Viertel entfernt ist — eine nicht konvexe,
    einspringende Kante statt nur einspringender Ecken wie beim Sternprisma."""
    q = _skalieren(koerper.quader(), 1.4, 1.4, 1.0)
    bissen = _verschieben(_skalieren(koerper.quader(), 1.0, 1.0, 1.2),
                          0.7 * bissen_anteil, 0.7 * bissen_anteil, 0.0)
    return _boolesch('differenz', [q, bissen])


def becher(wandstaerke=0.10):
    """Zylinder mit einer von oben eingelassenen Bohrung — oben offen,
    unten geschlossen, anders als der schon vorhandene Hohlzylinder mit
    durchgehendem Loch."""
    aussen = koerper.zylinder(radius=0.45, hoehe=0.7)
    innen = _verschieben(koerper.zylinder(radius=0.45 - wandstaerke, hoehe=0.7),
                         0, 0, 0.09)
    return _boolesch('differenz', [aussen, innen])


def kreuzstern(arm=1.5):
    """Drei duenne, orthogonale Balken, mittig verschmolzen — ein 3D-Kreuz,
    in jeder Blickrichtung eine andere zweidimensionale Kreuzsilhouette."""
    bx = _skalieren(koerper.quader(), arm, 0.26, 0.26)
    by = _skalieren(koerper.quader(), 0.26, arm, 0.26)
    bz = _skalieren(koerper.quader(), 0.26, 0.26, arm)
    return _boolesch('vereinigen', [_boolesch('vereinigen', [bx, by]), bz])


def ring_wuerfel(radius=0.26):
    """Quader mit einem einzelnen, zentralen, durchgehenden runden Loch —
    der einfachste Fall dieser Kategorie, als Gegenpol zum Lochkreuz.

    `radius` muss unter dem y-Halbmass des Quaders (0,31) bleiben — sonst
    steht der Lochrand fast tangential an der Seitenflaeche, und das
    Boolesche liefert dort zwei knapp getrennte statt einer durchgehenden
    Schale.
    """
    q = koerper.quader()
    loch = koerper.zylinder(radius=radius, hoehe=2.0)
    return _boolesch('differenz', [q, loch])


def ikosaeder_kerbe(tiefe=0.35):
    """Ikosaeder mit einer geraden Kerbe, die eine Kante durchschneidet."""
    ik = koerper.ikosaeder()
    ik = _skalieren(ik, 0.5, 0.5, 0.5)
    kerbe = _drehen_um_z(_skalieren(koerper.quader(), 2.0, 0.14, tiefe), 20.0)
    kerbe = _verschieben(kerbe, 0, 0, 0.28)
    return _boolesch('differenz', [ik, kerbe])


def zahnrad_grob(n_zaehne=8, radius=0.42, zahnlaenge=0.16):
    """Zylinder mit radial angesetzten Zaehnen — viele kleine, periodisch
    wiederkehrende einspringende Ecken, das hochfrequenteste Silhouetten-
    Muster dieser gesamten Erweiterung."""
    kern = koerper.zylinder(radius=radius, hoehe=0.30)
    zahn = _skalieren(koerper.quader(), zahnlaenge, 0.09, 0.28)
    zahn = _verschieben(zahn, radius + zahnlaenge / 2 - 0.02, 0, 0)
    teile = [kern]
    for i in range(n_zaehne):
        teile.append(_drehen_um_z(zahn, i * 360.0 / n_zaehne))
    out = teile[0]
    for t in teile[1:]:
        out = _boolesch('vereinigen', [out, t])
    return out


def bogen(offen_ab=0.0):
    """Dicker Torus, oberhalb von `offen_ab` weggeschnitten — ein
    Bogen/Torbogen statt eines geschlossenen Rings. Kleineres `offen_ab`
    schneidet mehr weg (schmalerer Bogen, groessere Oeffnung)."""
    t = koerper.torus_dick(major=0.45, minor=0.16)
    hoehe_box = 1.4
    schnitt = _verschieben(_skalieren(koerper.quader(), 2.0, hoehe_box, 1.0),
                           0, offen_ab + hoehe_box / 2, 0)
    return _boolesch('differenz', [t, schnitt])


def kapsel_ring(versatz=0.0):
    """Kapsel mit einem Ring um die Taille — zwei verschiedene Kruemmungs-
    arten (konvexe Kapselkappen, konkave Ringinnenseite) an einem Koerper."""
    k = koerper.kapsel(radius=0.28, hoehe=0.55)
    ring = _skalieren(koerper.torus_dick(major=0.34, minor=0.10), 1, 1, 1)
    ring = _verschieben(ring, 0, 0, versatz)
    return _boolesch('vereinigen', [k, ring])


# ── Verzeichnis: Grundform + zweite Variante mit anderen Parametern ──────────

_ARTEN = [
    ('bg_lochkreuz',    'Quader, Lochkreuz', wuerfel_lochkreuz,
     'zwei durchgehende, senkrecht zueinander stehende Bohrungen'),
    ('bg_torussteg',    'Torus mit Steg', torus_steg,
     'Zylinder quer durch das Loch des Torus'),
    ('bg_facette',      'Ellipsoid, Facette', kugel_facette,
     'ansonsten glatter Koerper mit einer einzelnen ebenen Schnittflaeche'),
    ('bg_dorn',         'Pyramide mit Dorn', pyramide_dorn,
     'zwei sehr verschiedene Massstaebe an einem Koerper vereinigt'),
    ('bg_hantel',       'Hantel', hantel,
     'zwei getrennte Rundungen, durch einen duennen Stab verbunden'),
    ('bg_lblock',       'L-Block', l_block,
     'einspringende Kante statt nur einspringender Ecken'),
    ('bg_becher',       'Becher', becher,
     'oben offene, unten geschlossene Bohrung'),
    ('bg_kreuzstern',   'Kreuzstern', kreuzstern,
     'drei orthogonale Balken, aus jeder Achse eine andere Kreuzsilhouette'),
    ('bg_ringwuerfel',  'Ring-Wuerfel', ring_wuerfel,
     'ein einzelnes zentrales, durchgehendes rundes Loch'),
    ('bg_kerbe',        'Ikosaeder mit Kerbe', ikosaeder_kerbe,
     'gerade Kerbe, die eine Kante des Ikosaeders durchschneidet'),
    ('bg_zahnrad',      'Zahnrad, grob', zahnrad_grob,
     'periodisch wiederkehrende Zaehne — die hochfrequenteste Silhouette hier'),
    ('bg_bogen',        'Bogen', bogen,
     'halber Torus — offener Bogen statt geschlossenem Ring'),
    ('bg_kapselring',   'Kapsel mit Ring', kapsel_ring,
     'konvexe Kappen und eine konkave Ringinnenseite am selben Koerper'),
]

BAUGRUPPEN = {}
for _key, _label, _fn, _notiz in _ARTEN:
    BAUGRUPPEN[_key] = (_label, _fn, _notiz + ' (Grundform)')

# Zweite Variante je Baugruppe mit verschobenen Parametern — bewusst kein
# Zufall: jede Variante bleibt benennbar und einzeln nachvollziehbar.
_VARIANTEN = {
    'bg_lochkreuz':   lambda: wuerfel_lochkreuz(radius=0.22),
    'bg_torussteg':   lambda: torus_steg(radius=0.20),
    'bg_facette':     lambda: kugel_facette(tiefe=0.40),
    'bg_dorn':        lambda: pyramide_dorn(dornradius=0.16, dornhoehe=0.85),
    'bg_hantel':      lambda: hantel(abstand=0.75, kugelradius=0.24, stabradius=0.08),
    'bg_lblock':      lambda: l_block(bissen_anteil=0.85),
    'bg_becher':      lambda: becher(wandstaerke=0.16),
    'bg_kreuzstern':  lambda: kreuzstern(arm=1.9),
    'bg_ringwuerfel': lambda: ring_wuerfel(radius=0.20),
    'bg_kerbe':       lambda: ikosaeder_kerbe(tiefe=0.55),
    'bg_zahnrad':     lambda: zahnrad_grob(n_zaehne=12, radius=0.38, zahnlaenge=0.20),
    'bg_bogen':       lambda: bogen(offen_ab=-0.2),
}
for _key, _fn in _VARIANTEN.items():
    _label, _, _notiz = BAUGRUPPEN[_key]
    BAUGRUPPEN[f'{_key}_var'] = (f'{_label}, Variante', _fn,
                                 _notiz.replace('(Grundform)', '(zweite Variante)'))


def baue(key):
    return BAUGRUPPEN[key][1]()


def pruefen(verbose=True):
    fehler = []
    for k, (label, bauer, _) in BAUGRUPPEN.items():
        try:
            m = bauer()
        except Exception as e:                              # noqa: BLE001
            fehler.append((k, f'Bau fehlgeschlagen: {type(e).__name__}: {e}'))
            continue
        grund = []
        if not m.is_watertight:
            grund.append('nicht wasserdicht')
        if not m.is_winding_consistent:
            grund.append('Umlaufsinn uneinheitlich')
        if m.volume <= 1e-9:
            grund.append(f'Volumen {m.volume:.3g}')
        if grund:
            fehler.append((k, ', '.join(grund)))
        if verbose:
            zeichen = 'ok ' if not grund else 'FEHLER'
            print(f'  {zeichen} {k:18s} {len(m.vertices):6d} Ecken '
                  f'{len(m.faces):6d} Dreiecke  V={m.volume:.4f}  Teile={m.body_count}')
    return fehler


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pruefen', action='store_true')
    p.parse_args()
    print(f'{len(BAUGRUPPEN)} Baugruppen\n')
    f = pruefen()
    print()
    if f:
        for k, g in f:
            print(f'  FEHLER {k}: {g}')
        raise SystemExit(1)
    print('Alle Baugruppen wasserdicht.')
