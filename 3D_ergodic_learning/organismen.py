r"""
organismen.py
=============
Fuenfundvierzig Metaball-Organismen: die einzige Formfamilie hier, die weder
konvex noch rotationssymmetrisch noch aus ebenen Facetten aufgebaut ist.

`koerper.py` deckt Ecken/Kanten und Rotationskoerper ab, `superquadriken.py`
eine stetige Familie zwischen Kasten und Stern, `woerter.py` mehrteilige,
scharfkantige Silhouetten. Was in keiner der drei vorkommt: eine glatte,
*organische*, wirklich nicht konvexe Oberflaeche mit Einbuchtungen zwischen
verschmolzenen Rundungen — das Profil, das am ehesten an gewachsene statt
konstruierte Formen erinnert.

Verfahren: ein Skalarfeld aus 2-5 verschmolzenen Gauss-Blobs

    F(p) = sum_i w_i * exp(-|p - c_i|^2 / (2 r_i^2))

auf einem 48^3-Gitter ausgewertet, die Oberflaeche F(p) = Schwelle per
`skimage.measure.marching_cubes` extrahiert. Reine Mathematik auf einem
selbst gewaehlten Feld — keine externe Geometrie, keine Lizenzfrage. Jede
Form ist ueber einen festen Seed reproduzierbar (`organismus_00` .. `_44`).

Selbsttest:

    python organismen.py --pruefen
"""
import numpy as np

N_ORGANISMEN = 39   # 39 statt 45: project_db_3d.py haengt beim Bau ueber
                    # surfaces.externe_aufnehmen() sechs bereits vorbereitete,
                    # aber bislang nie geholte Open3D-Netze an (siehe
                    # LIZENZEN.md) — sechs weniger hier haelt die Gesamtzahl
                    # aller vier neuen Kategorien bei 117 und damit die
                    # komplette Datenbank (77 bisherige + 117 + 6 Externe) bei
                    # der angepeilten runden 200.
GRID = 48
SCHWELLE = 1.0


def _feld(rng, n_blobs):
    """`n_blobs` Gauss-Blobs: Position (im Wuerfel [-0.7,0.7]^3), Radius,
    Gewicht. Positionen ueberlappend genug gezogen, dass die meisten
    Organismen zusammenhaengen, aber nicht alle — manche zerfallen in zwei
    durch einen Steg verbundene oder auch getrennte Klumpen, was fuer die
    Silhouette denselben Fall wie ein Wort mit Buchstabenabstand ist.
    """
    mitte = rng.uniform(-0.35, 0.35, size=(n_blobs, 3))
    # erster Blob im Ursprung verankert, damit immer ein Kern existiert
    mitte[0] = 0.0
    radien = rng.uniform(0.28, 0.5, size=n_blobs)
    gewichte = rng.uniform(0.8, 1.3, size=n_blobs)
    return mitte, radien, gewichte


def _auswerten(mitte, radien, gewichte, grid=GRID, spanne=1.0):
    ax = np.linspace(-spanne, spanne, grid)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    F = np.zeros_like(X)
    for c, r, w in zip(mitte, radien, gewichte):
        d2 = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2
        F += w * np.exp(-d2 / (2.0 * r * r))
    spacing = ax[1] - ax[0]
    return F, spacing, ax[0]


def organismus(seed=0, n_blobs=None):
    """Ein Metaball-Organismus zum ganzzahligen `seed`."""
    import trimesh
    from skimage.measure import marching_cubes
    rng = np.random.default_rng(seed)
    gezogen = int(rng.integers(2, 6))    # immer ziehen: haelt den RNG-Zustand
    nb = n_blobs if n_blobs is not None else gezogen  # gleich, ob n_blobs vorgegeben ist oder nicht
    mitte, radien, gewichte = _feld(rng, nb)
    F, spacing, ursprung = _auswerten(mitte, radien, gewichte)
    if F.max() <= SCHWELLE:
        # zu schwach verschmolzen/zu klein: Gewichte anheben und neu werten
        gewichte = gewichte * (SCHWELLE * 1.5 / F.max())
        F, spacing, ursprung = _auswerten(mitte, radien, gewichte)
    verts, faces, normals, _ = marching_cubes(F, level=SCHWELLE,
                                              spacing=(spacing,) * 3)
    verts = verts + ursprung
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m.fix_normals()
    # leichte Glaettung: das Voxelgitter hinterlaesst facettierte Kanten,
    # die kein Merkmal des Feldes sind, nur seiner Aufloesung.
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=8)
    m.fix_normals()
    return m


# ── Verzeichnis ──────────────────────────────────────────────────────────────

def _notiz(n_blobs):
    return (f'{n_blobs} verschmolzene Gauss-Blobs, glatt und nicht konvex — '
            'je nach Ueberlappung ein zusammenhaengender Koerper oder '
            'mehrere durch keinen Steg verbundene Teile')


def _nb_fuer(seed):
    """Nur die Blobzahl ziehen, ohne das Netz zu bauen — der volle Bau
    (Marching Cubes + Glaettung) waere fuer die reine Registrierung ein
    unnoetiger Vorlauf bei jedem Import."""
    return int(np.random.default_rng(seed).integers(2, 6))


ORGANISMEN = {}
for _i in range(N_ORGANISMEN):
    _seed = 1000 + _i
    _nb = _nb_fuer(_seed)
    ORGANISMEN[f'organismus_{_i:02d}'] = (
        f'Organismus {_i:02d}',
        (lambda s=_seed, n=_nb: organismus(s, n_blobs=n)),
        _notiz(_nb))


def baue(key):
    return ORGANISMEN[key][1]()


def pruefen(verbose=True):
    fehler = []
    for k, (label, bauer, _) in ORGANISMEN.items():
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
            print(f'  {zeichen} {k:16s} {len(m.vertices):6d} Ecken '
                  f'{len(m.faces):6d} Dreiecke  V={m.volume:.4f}  Teile={m.body_count}')
    return fehler


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pruefen', action='store_true')
    p.parse_args()
    print(f'{len(ORGANISMEN)} Organismen\n')
    f = pruefen()
    print()
    if f:
        for k, g in f:
            print(f'  FEHLER {k}: {g}')
        raise SystemExit(1)
    print('Alle Organismen wasserdicht.')
