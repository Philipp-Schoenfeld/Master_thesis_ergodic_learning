r"""
externe_netze.py
================
Gemessene und modellierte Netze von aussen — als Gegenprobe zu den
selbstgebauten Koerpern.

Warum ueberhaupt: `koerper.py` und `text_volumen.py` liefern saubere Geometrie.
Jede Facette ist eben, jede Normale zeigt verlaesslich nach aussen, jedes Netz
ist wasserdicht. Echte Netze sind das nicht. Sie haben ungleichmaessige
Dreiecke, duenne Waende, offene Raender und stellenweise falsch orientierte
Normalen — und genau daran entscheidet sich, ob die Projektion in der
Datenbank belastbar ist oder nur auf Laborgeometrie funktioniert.

Zwei Quellen:

* **open3d** bringt geprueft heruntergeladene Modelle mit Pruefsumme mit
  (`o3d.data.*`). Sie landen in `~/open3d_data/` und nicht im Repo.
* **Stanford** liefert die grossen gescannten Modelle. Die kommen nur ueber das
  Unterkommando `--fetch`, ausdruecklich und einzeln — siehe unten.

Das Aufnahmetor
---------------
Ein heruntergeladenes Netz wird nicht ungeprueft in die Flaechenliste
uebernommen. Jedes wird einmal mit tausend Probestrahlen beschossen; wer
durchfaellt, wird mit Begruendung uebersprungen, statt spaeter mitten im
Datenbankbau Fehlschuesse zu produzieren, deren Ursache dann nicht mehr
sichtbar ist.

Gemessen werden zwei verschiedene Dinge, und die Unterscheidung ist wichtig:

* **Silhouettentreffer** — Strahlen, die auf die Projektion tatsaechlich
  vorhandener Oberflaechenpunkte gezielt werden. Ein geschlossener Koerper
  liefert hier praktisch 100 %. Faellt der Wert ab, schluepfen Strahlen durch
  Loecher im Netz: duennes Glas, offene Riemen, nicht geschlossene Raender.
  **Hierauf wirkt die Schwelle von 0,9.**
* **Flaechentreffer** — Strahlen auf einem gleichmaessigen Raster ueber das
  ganze Projektionsquadrat. Der Wert sagt etwas ueber die Silhouette, nicht
  ueber die Netzqualitaet: eine perfekte Kugel im Quadrat kommt geometrisch
  nie ueber pi/4 = 0,785 hinaus. Er wird berichtet, aber nicht als Tor
  benutzt; das Aussortieren schlechter Winkel macht spaeter
  `surfaces.zufaellige_blickrichtung`.

    python externe_netze.py --pruefen          # Aufnahmetor, laedt die o3d-Modelle
    python externe_netze.py --manifest         # Stanford: nur den Plan zeigen
    python externe_netze.py --fetch happy      # Stanford: wirklich herunterladen
"""
import hashlib
import json
import os

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_here, 'cache', 'netze')

ZIEL_DREIECKE = 4000        # worauf dezimiert wird
PROBE_STRAHLEN = 1000
SCHWELLE_SILHOUETTE = 0.90

# Ein Netz unter dieser Dreieckszahl traegt keine Geometrie, die die
# selbstgebauten Koerper nicht schon haetten. Aufgefallen ist das an der
# Holzkiste: nach dem Laden blieben zwoelf Dreiecke uebrig, denn ihre ganze
# Gestalt steckt in der Textur, nicht im Netz. Als Zielflaeche waere sie eine
# zweite Ausgabe von `koerper.quader` — mit dem Nachteil, dass niemand mehr
# sieht, dass es dieselbe Form ist.
MIN_DREIECKE = 100


# ── Laden und Aufbereiten ────────────────────────────────────────────────────

def _als_trimesh(pfad):
    """Datei -> einzelnes Trimesh. Szenen werden zusammengefasst."""
    import trimesh
    m = trimesh.load(pfad, process=True, force='mesh')
    if isinstance(m, trimesh.Scene):
        # glTF-Modelle kommen als Szene mit mehreren Knoten. `dump()` wendet die
        # Knotentransformationen an — ohne das lägen die Teile uebereinander.
        m = trimesh.util.concatenate(m.dump())
    m.fix_normals()
    return m


def dezimieren(m, ziel=ZIEL_DREIECKE):
    """Auf ~`ziel` Dreiecke eindampfen.

    Ueber open3d und nicht ueber trimesh: `trimesh.simplify_quadric_decimation`
    heisst in trimesh 4/5 anders und braucht `fast_simplification`, das hier
    nicht installiert ist. open3d bringt den Quadric-Decimator mit.
    """
    if len(m.faces) <= ziel:
        return m
    import open3d as o3d
    import trimesh
    tm = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(m.faces, dtype=np.int32)))
    tm = tm.simplify_quadric_decimation(target_number_of_triangles=int(ziel))
    tm.remove_degenerate_triangles()
    tm.remove_duplicated_vertices()
    tm.remove_unreferenced_vertices()
    out = trimesh.Trimesh(vertices=np.asarray(tm.vertices),
                          faces=np.asarray(tm.triangles), process=True)
    out.fix_normals()
    return out


def _aufrecht(m, achse='y'):
    """Modelle mit y nach oben in die z-nach-oben-Konvention drehen."""
    import trimesh
    if achse != 'y':
        return m
    v = np.asarray(m.vertices, dtype=np.float64)
    out = trimesh.Trimesh(vertices=np.stack([v[:, 0], -v[:, 2], v[:, 1]], -1),
                          faces=m.faces, process=False)
    out.fix_normals()
    return out


# ── Die Modelle ──────────────────────────────────────────────────────────────
#  Schluessel -> (Bezeichnung, o3d-Datenklasse, Hochachse, Notiz)

_O3D = {
    'armadillo':  ('Stanford-Armadillo', 'ArmadilloMesh', 'y',
                   'gescannt, stark konkav an Armen und Beinen'),
    'knoten':     ('Kleeblattknoten', 'KnotMesh', 'z',
                   'verschlungen: die Projektion trifft dieselbe Roehre mehrfach'),
    'affenkopf':  ('Suzanne', 'MonkeyModel', 'y',
                   'modelliert, glatte Freiformflaechen mit scharfen Kanten'),
    'avocado':    ('Avocado', 'AvocadoModel', 'y',
                   'glTF-Modell, weich und fast konvex'),
    'kiste':      ('Holzkiste', 'CrateModel', 'y',
                   'quaderfoermig mit gefasten Kanten'),
    'helm':       ('Beschaedigter Helm', 'DamagedHelmetModel', 'y',
                   'duenne Schale mit Visier — Kandidat fuers Aufnahmetor'),
    'fliegerhelm': ('Fliegerhelm', 'FlightHelmetModel', 'y',
                    'offene Riemen und Schnallen — Kandidat fuers Aufnahmetor'),
    'schwert':    ('Schwert', 'SwordModel', 'y',
                   'lang und duenn: extremes Seitenverhaeltnis'),
}


def _o3d_pfad(klasse):
    import open3d as o3d
    return getattr(o3d.data, klasse)().path


def lade_o3d(key, dezimiert=True):
    label, klasse, achse, _ = _O3D[key]
    m = _aufrecht(_als_trimesh(_o3d_pfad(klasse)), achse)
    return dezimieren(m) if dezimiert else m


# ── Stanford: Manifest und Abruf ─────────────────────────────────────────────
#
# Diese Dateien werden **nicht** beilaeufig geholt. Sie sind gross, sie liegen
# auf einem Universitaetsserver ohne Pruefsummen, und die grossen Scans stehen
# unter einer Lizenz, die kommerzielle Nutzung ausschliesst. Deshalb:
# `--manifest` zeigt, was geholt wuerde; `--fetch <name>` holt genau eines,
# rechnet danach den SHA-256 aus und schreibt ihn in `netze.lock`, sodass jeder
# weitere Abruf gegen den einmal festgestellten Wert prueft.
#
# Die erwartete Groesse stammt von der Stanford-Uebersichtsseite und ist ein
# Richtwert; abweicht sie um mehr als 20 %, bricht der Abruf ab.

STANFORD = {
    'happy': dict(
        label='Happy Buddha',
        url='http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz',
        mb=32.0,
        datei='happy_recon/happy_vrip.ply',
        lizenz='Stanford 3D Scanning Repository — frei fuer Forschung, '
               'mit Danksagung',
        achse='y',
        heldout=False),
    'drache': dict(
        label='Stanford-Drache',
        url='http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz',
        mb=12.0,
        datei='dragon_recon/dragon_vrip.ply',
        lizenz='Stanford 3D Scanning Repository — frei fuer Forschung, '
               'mit Danksagung',
        achse='y',
        heldout=True),
}

# Thai Statue und Asian Dragon stehen bewusst **nicht** in der Tabelle. Beide
# stammen von der XYZ-RGB-Seite desselben Repositoriums, und deren Lizenz ist
# genau die, die im Plan als Ausschlussgrund fuer "XYZRGB" genannt wird. Sie
# einzeln aufzunehmen und die Quelle im selben Atemzug auszuschliessen waere ein
# Widerspruch, den der Code nicht stillschweigend aufloesen sollte.


def _lock_pfad():
    return os.path.join(CACHE, 'netze.lock')


def _lock_lesen():
    p = _lock_pfad()
    return json.load(open(p)) if os.path.isfile(p) else {}


def manifest():
    """Was `--fetch` holen wuerde — ohne irgendetwas zu holen."""
    lock = _lock_lesen()
    zeilen = []
    for k, e in STANFORD.items():
        h = lock.get(k, {}).get('sha256')
        zeilen.append(
            f"  {k:8s} {e['label']:18s} ~{e['mb']:5.1f} MB"
            f"{'  [Heldout]' if e['heldout'] else ''}\n"
            f"           {e['url']}\n"
            f"           Lizenz: {e['lizenz']}\n"
            f"           SHA-256: {h if h else '(noch nicht festgestellt — '
                                              'wird beim ersten Abruf gesetzt)'}")
    return '\n'.join(zeilen)


def hole_stanford(key, bestaetigt=False):
    """Ein Stanford-Modell holen. Ohne `bestaetigt=True` passiert nichts."""
    import tarfile
    import urllib.request
    if key not in STANFORD:
        raise KeyError(f'{key!r} — bekannt sind {list(STANFORD)}')
    if not bestaetigt:
        raise SystemExit('Abruf nicht bestaetigt. --fetch setzt die '
                         'Bestaetigung; ohne sie wird nichts geladen.')
    e = STANFORD[key]
    os.makedirs(CACHE, exist_ok=True)
    ziel = os.path.join(CACHE, f'{key}.tar.gz')
    if not os.path.isfile(ziel):
        print(f"  lade {e['url']}")
        urllib.request.urlretrieve(e['url'], ziel)
    gross = os.path.getsize(ziel) / 2 ** 20
    if abs(gross - e['mb']) / e['mb'] > 0.2:
        raise SystemExit(f'  Groesse {gross:.1f} MB weicht mehr als 20 % von '
                         f"den erwarteten {e['mb']:.1f} MB ab — Abbruch.")
    h = hashlib.sha256(open(ziel, 'rb').read()).hexdigest()
    lock = _lock_lesen()
    bekannt = lock.get(key, {}).get('sha256')
    if bekannt and bekannt != h:
        raise SystemExit(f'  SHA-256 weicht ab!\n    erwartet {bekannt}\n'
                         f'    erhalten {h}')
    lock[key] = dict(sha256=h, mb=round(gross, 2), url=e['url'])
    json.dump(lock, open(_lock_pfad(), 'w'), indent=2)
    with tarfile.open(ziel) as t:
        t.extractall(CACHE)
    print(f'  {e["label"]}: {gross:.1f} MB, SHA-256 {h}')
    return os.path.join(CACHE, e['datei'])


def lade_stanford(key, dezimiert=True):
    e = STANFORD[key]
    pfad = os.path.join(CACHE, e['datei'])
    if not os.path.isfile(pfad):
        raise FileNotFoundError(
            f"{e['label']} ist nicht im Zwischenspeicher. Erst holen:\n"
            f"    python externe_netze.py --fetch {key}")
    m = _aufrecht(_als_trimesh(pfad), e['achse'])
    return dezimieren(m) if dezimiert else m


# ── Aufnahmetor ──────────────────────────────────────────────────────────────

def _rahmen(w):
    w = np.asarray(w, dtype=np.float64)
    w = w / max(np.linalg.norm(w), 1e-12)
    up = np.array([0., 0., 1.]) if abs(w[2]) < 0.9 else np.array([0., 1., 0.])
    e1 = np.cross(up, w); e1 /= max(np.linalg.norm(e1), 1e-12)
    return e1, np.cross(w, e1), w


def strahlenprobe(mesh, richtung, n=PROBE_STRAHLEN, seed=0):
    """Trefferanteile eines Netzes unter einer Blickrichtung.

    -> dict mit `silhouette` (Strahlen auf projizierte Oberflaechenpunkte) und
       `flaeche` (gleichmaessiges Raster ueber das Projektionsquadrat).
    """
    import open3d as o3d
    import trimesh
    e1, e2, w = _rahmen(richtung)

    szene = o3d.t.geometry.RaycastingScene()
    szene.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(mesh.vertices), dtype=o3d.core.float32),
        o3d.core.Tensor(np.asarray(mesh.faces), dtype=o3d.core.int32)))

    P, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    P = np.asarray(P, dtype=np.float64)
    a, b = P @ e1, P @ e2
    a0, a1, b0, b1 = a.min(), a.max(), b.min(), b.max()
    spann = float(max(a1 - a0, b1 - b0, 1e-9))

    def _schiessen(u, v):
        org = (np.outer(u, e1) + np.outer(v, e2)) - 6.0 * spann * w
        d = np.tile(w.astype(np.float32), (len(org), 1))
        rays = o3d.core.Tensor(np.hstack([org.astype(np.float32), d]),
                               dtype=o3d.core.float32)
        t = szene.cast_rays(rays)['t_hit'].numpy()
        return float(np.isfinite(t).mean())

    # (1) auf die Silhouette: die Strahlen zielen auf Punkte, die es gibt.
    sil = _schiessen(a, b)

    # (2) gleichmaessiges Raster ueber das ganze Quadrat.
    rng = np.random.default_rng(seed + 1)
    fl = _schiessen(rng.uniform(a0, a1, n), rng.uniform(b0, b1, n))
    return dict(silhouette=sil, flaeche=fl)


def pruefe_netz(mesh, richtungen=None, seed=0):
    """Aufnahmetor. -> (bestanden, dict der Messwerte, Begruendung)."""
    if richtungen is None:
        # Drei Achsen und eine Raumdiagonale: ein Netz, das nur aus einer
        # Richtung durchlaessig ist, faellt sonst nicht auf.
        richtungen = [(0, 0, -1), (0, -1, 0), (-1, 0, 0),
                      (-1, -1, -1)]
    sil, fl = [], []
    for r in richtungen:
        m = strahlenprobe(mesh, r, seed=seed)
        sil.append(m['silhouette']); fl.append(m['flaeche'])
    werte = dict(silhouette_min=float(np.min(sil)),
                 silhouette_mittel=float(np.mean(sil)),
                 flaeche_mittel=float(np.mean(fl)),
                 wasserdicht=bool(mesh.is_watertight),
                 dreiecke=int(len(mesh.faces)))
    if werte['dreiecke'] < MIN_DREIECKE:
        return False, werte, (
            f"nur {werte['dreiecke']} Dreiecke — die Gestalt steckt in der "
            f'Textur, nicht im Netz; als Zielflaeche waere das ein zweiter '
            f'Quader')
    if werte['silhouette_min'] < SCHWELLE_SILHOUETTE:
        return False, werte, (
            f"Silhouettentreffer {werte['silhouette_min']*100:.1f} % < "
            f'{SCHWELLE_SILHOUETTE*100:.0f} % — Strahlen schluepfen durch das '
            f'Netz (offene Raender oder duenne Schale)')
    return True, werte, ''


def aufnehmen(keys=None, verbose=True):
    """Alle externen Netze bauen, pruefen und die bestandenen zurueckgeben.

    -> (angenommen {key: (label, mesh, notiz)}, abgelehnt {key: begruendung})
    """
    keys = list(_O3D) if keys is None else list(keys)
    angenommen, abgelehnt = {}, {}
    for k in keys:
        label, _, _, notiz = _O3D[k]
        try:
            m = lade_o3d(k)
        except Exception as e:                          # noqa: BLE001
            abgelehnt[k] = f'Laden fehlgeschlagen: {type(e).__name__}: {e}'
            if verbose:
                print(f'  ABGELEHNT {k:12s} {abgelehnt[k]}')
            continue
        ok, werte, grund = pruefe_netz(m)
        if verbose:
            print(f"  {'ok       ' if ok else 'ABGELEHNT'} {k:12s} "
                  f"Silhouette {werte['silhouette_min']*100:5.1f} % "
                  f"Flaeche {werte['flaeche_mittel']*100:5.1f} % "
                  f"{werte['dreiecke']:5d} Dreiecke "
                  f"{'wasserdicht' if werte['wasserdicht'] else 'offen'}"
                  + (f'\n              -> {grund}' if grund else ''))
        if ok:
            angenommen[k] = (label, m, notiz)
        else:
            abgelehnt[k] = grund
    return angenommen, abgelehnt


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pruefen', action='store_true',
                   help='Aufnahmetor ueber alle open3d-Modelle laufen lassen '
                        '(laedt sie beim ersten Mal herunter)')
    p.add_argument('--manifest', action='store_true',
                   help='zeigen, was --fetch holen wuerde')
    p.add_argument('--fetch', type=str, default=None,
                   help='ein Stanford-Modell wirklich herunterladen')
    a = p.parse_args()

    if a.manifest:
        print('Stanford-Modelle (werden nur auf ausdrueckliche Anweisung '
              'geholt):\n')
        print(manifest())
    elif a.fetch:
        hole_stanford(a.fetch, bestaetigt=True)
    elif a.pruefen:
        print(f'Aufnahmetor: {PROBE_STRAHLEN} Probestrahlen je Richtung, '
              f'Schwelle {SCHWELLE_SILHOUETTE*100:.0f} % Silhouettentreffer\n')
        ja, nein = aufnehmen()
        print(f'\n{len(ja)} angenommen, {len(nein)} abgelehnt')
        for k, g in nein.items():
            print(f'  {k}: {g}')
    else:
        p.print_help()
