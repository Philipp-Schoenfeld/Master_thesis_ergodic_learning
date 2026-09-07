r"""
text_volumen.py
===============
Buchstaben und Ziffern als Volumenkoerper.

Ein extrudierter Buchstabe ist als Zielflaeche etwas anderes als jeder Koerper
aus `koerper.py`: seine Silhouette ist nicht konvex, sie hat Loecher (A, B, D,
O, P, Q, R, 0), einspringende Ecken (E, K, W) und duenne Stege (X, Y). Beim
Werfen der Dichte darauf entstehen dadurch Fehlschuesse an genau den Stellen,
an denen die 2D-Bahn ueber den Rand des Zeichens hinauslaeuft — das ist der
Fall, den die Datenbank ausdruecklich mitmessen soll.

Der Weg vom Zeichen zum Netz:

    TextPath  ->  Ringe (Aussenkonturen und Loecher)
              ->  shapely-Polygon nach der Even-Odd-Regel
              ->  trimesh.creation.extrude_polygon

Die Even-Odd-Regel ist der Punkt, an dem es schiefgehen kann: `to_polygons`
liefert alle Ringe gleichberechtigt, ohne zu sagen, welcher ein Loch ist. Ein
'B' hat zwei Loecher in einer Aussenkontur, ein '8' ebenso; nimmt man alle als
Aussenkonturen, wird der Buchstabe zu einem massiven Klotz. Die symmetrische
Differenz der Ringe loest das ohne Fallunterscheidung: was in einer geraden
Zahl von Ringen liegt, ist aussen, was in einer ungeraden liegt, ist innen.

    pip install shapely mapbox-earcut

`mapbox_earcut` wird nicht direkt importiert, aber `extrude_polygon` braucht es
zum Triangulieren von Polygonen mit Loechern und scheitert sonst.

Selbsttest:

    python text_volumen.py --pruefen
"""
import string

import numpy as np

# A-Z als Trainingsflaechen, zwei Ziffern zurueckgehalten. Die Ziffern sind
# bewusst keine Buchstaben: der Holdout soll eine Silhouette pruefen, deren
# Bauart das Netz kennt, deren Form es aber nie gesehen hat.
BUCHSTABEN = list(string.ascii_uppercase)
ZIFFERN_HELDOUT = ['1', '5']

TIEFE_ANTEIL = 0.35          # Tiefe, bezogen auf die Zeichenhoehe


def _ringe(zeichen, groesse=1.0):
    """Geschlossene Ringe der Glyphe als Liste von (n,2)-Feldern."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    prop = FontProperties(family='DejaVu Sans', weight='bold')
    pfad = TextPath((0.0, 0.0), zeichen, size=groesse, prop=prop)
    ringe = [np.asarray(r, dtype=np.float64)
             for r in pfad.to_polygons(closed_only=True)]
    return [r for r in ringe if len(r) >= 4]


def _polygon(zeichen, groesse=1.0):
    """Ringe nach der Even-Odd-Regel zu einer Flaeche verrechnen."""
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    ringe = _ringe(zeichen, groesse)
    if not ringe:
        raise ValueError(f'Glyphe {zeichen!r} liefert keine Kontur')
    flaeche = None
    for r in ringe:
        p = make_valid(Polygon(r))
        flaeche = p if flaeche is None else flaeche.symmetric_difference(p)
    return flaeche


def _als_netz(flaeche, tiefe):
    """Eine shapely-Flaeche extrudieren; Mehrteiler werden zusammengefuegt."""
    import trimesh
    from shapely.geometry import Polygon
    teile = ([flaeche] if isinstance(flaeche, Polygon)
             else [g for g in flaeche.geoms if isinstance(g, Polygon)])
    netze = [trimesh.creation.extrude_polygon(t, height=tiefe)
             for t in teile if t.area > 1e-9]
    if not netze:
        raise ValueError('nach dem Verrechnen der Ringe blieb keine Flaeche')
    m = netze[0] if len(netze) == 1 else trimesh.util.concatenate(netze)
    v = np.asarray(m.vertices, dtype=np.float64).copy()
    v -= (v.min(axis=0) + v.max(axis=0)) / 2.0
    out = trimesh.Trimesh(vertices=v, faces=m.faces, process=False)
    out.fix_normals()
    return out


def baue(zeichen, groesse=1.0, tiefe=None):
    """Ein Zeichen als extrudiertes, wasserdichtes Netz.

    Das Zeichen steht aufrecht in der xy-Ebene und ist entlang z extrudiert;
    von vorne betrachtet ist es also lesbar. `surfaces._fit_unit` skaliert es
    anschliessend in den Einheitswuerfel.
    """
    flaeche = _polygon(zeichen, groesse)
    hoehe = flaeche.bounds[3] - flaeche.bounds[1]
    return _als_netz(flaeche, tiefe if tiefe is not None
                     else max(hoehe, 1e-6) * TIEFE_ANTEIL)


# ── Verzeichnis ──────────────────────────────────────────────────────────────
#  Schluessel -> (Bezeichnung, Bauer, Notiz)

def _eintrag(zeichen, notiz):
    return (f'Buchstabe {zeichen}', (lambda c=zeichen: baue(c)), notiz)


_MIT_LOCH = set('ABDOPQR')
_SCHMAL = set('IJLT')

TEXT_KOERPER = {
    f'buchstabe_{c.lower()}': _eintrag(
        c,
        'Silhouette mit Loch — die Projektion faellt in der Mitte durch'
        if c in _MIT_LOCH else
        'schmale Silhouette — viel Rand, wenig Flaeche'
        if c in _SCHMAL else
        'nicht konvexe Silhouette mit einspringenden Ecken')
    for c in BUCHSTABEN
}

TEXT_HELDOUT = {
    f'ziffer_{d}': (f'Ziffer {d}', (lambda c=d: baue(c)),
                    'zurueckgehalten: bekannte Bauart, unbekannte Form')
    for d in ZIFFERN_HELDOUT
}


def pruefen(verbose=True):
    """Alle Zeichen bauen und pruefen. -> Liste der Fehler."""
    fehler = []
    for tab, marke in ((TEXT_KOERPER, ''), (TEXT_HELDOUT, ' [Heldout]')):
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
            if grund:
                fehler.append((k, ', '.join(grund)))
            if verbose:
                print(f"  {'ok ' if not grund else 'FEHLER'} {k:14s} "
                      f'{len(m.vertices):5d} Ecken {len(m.faces):5d} Dreiecke '
                      f'V={m.volume:.4f}{marke}')
    return fehler


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pruefen', action='store_true')
    p.parse_args()
    print(f'{len(TEXT_KOERPER)} Buchstaben + {len(TEXT_HELDOUT)} Ziffern '
          f'(zurueckgehalten)\n')
    f = pruefen()
    print()
    if f:
        for k, g in f:
            print(f'  FEHLER {k}: {g}')
        raise SystemExit(1)
    print('Alle Zeichen wasserdicht.')
