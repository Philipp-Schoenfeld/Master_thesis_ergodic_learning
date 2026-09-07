r"""
superquadriken.py
==================
Achtundzwanzig Superquadriken: eine Formfamilie, kein Einzelentwurf je Koerper.

`koerper.py` deckt Ecken/Kanten (platonisch), Extrusion (Prismen) und
Kruemmung (Rotationskoerper) je durch benannte Einzelfaelle ab. Hier steht
stattdessen eine einzige Formel mit zwei Exponenten, die den Raum dazwischen
*stetig* durchmisst — vom Kasten ueber die Kugel bis zum eingeschnuerten,
nicht konvexen Stern:

    c(w, e) = sign(cos w) * |cos w|^e
    s(w, e) = sign(sin w) * |sin w|^e

    x = a1 * c(theta, e1) * c(phi, e2)
    y = a2 * c(theta, e1) * s(phi, e2)
    z = a3 * s(theta, e1)

`e1` regelt die Rundung an den Polen (klein = spitz/eingeschnuert, gross =
kastig), `e2` dieselbe Rundung im Querschnitt um die z-Achse. Gebaut wird auf
der Eckentopologie einer Ikosphaere: jeder Vertex traegt seinen eigenen
Kugelwinkel (theta, phi), die Formel ersetzt nur den Radius in diese
Richtung — dieselbe Technik wie `surfaces._egg_mesh` und `koerper.ellipsoid`,
nur mit zwei Exponenten statt einer festen Streckung. Reine Formel, keine
externe Geometrie, keine Lizenzfrage.

Selbsttest:

    python superquadriken.py --pruefen
"""
import numpy as np


def _c(w, e):
    return np.sign(np.cos(w)) * np.abs(np.cos(w)) ** e


def _s(w, e):
    return np.sign(np.sin(w)) * np.abs(np.sin(w)) ** e


def superquadrik(e1=1.0, e2=1.0, skalen=(0.5, 0.5, 0.5), subdiv=4):
    """Superquadrik auf Ikosphaeren-Topologie.

    `e1` < 1 schnuert die Pole ein (nicht konvex), `e1` > 1 flacht sie ab
    (kastig); `e2` tut dasselbe im Querschnitt um die z-Achse.
    """
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    v = np.asarray(m.vertices, dtype=np.float64)
    r = np.linalg.norm(v, axis=1, keepdims=True)
    d = v / np.clip(r, 1e-12, None)
    theta = np.arcsin(np.clip(d[:, 2], -1.0, 1.0))          # Pol-Winkel
    phi = np.arctan2(d[:, 1], d[:, 0])                      # Azimut
    a1, a2, a3 = skalen
    x = a1 * _c(theta, e1) * _c(phi, e2)
    y = a2 * _c(theta, e1) * _s(phi, e2)
    z = a3 * _s(theta, e1)
    verts = np.stack([x, y, z], axis=1)
    out = trimesh.Trimesh(vertices=verts, faces=m.faces, process=False)
    out.fix_normals()
    return out


# ── Kuratierte Parameterliste ────────────────────────────────────────────────
#  (e1, e2, Skalen, Kurzname, Notiz-Stamm) — vierzehn Formfamilien in je zwei
#  Skalierungsvarianten (isotrop / anisotrop) macht achtundzwanzig.

_ARTEN = [
    (0.25, 0.25, 'sq_stern3d',    'in allen drei Achsen eingeschnuerter Stern, stark nicht konvex'),
    (0.35, 1.00, 'sq_spindel',    'an den Polen spitz, im Querschnitt rund — Spindelquerschnitt'),
    (1.00, 0.35, 'sq_dreikant',   'rund an den Polen, im Querschnitt eingeschnuert — dreikantig wirkend'),
    (0.35, 0.35, 'sq_oktaederrund', 'zwischen Oktaeder und Kugel, alle Kanten leicht gerundet'),
    (1.00, 1.00, 'sq_kugel',      'Sonderfall e1=e2=1 — die Kugel selbst als Bezugspunkt der Familie'),
    (2.00, 2.00, 'sq_rundkasten', 'kastig mit stark gerundeten Kanten'),
    (4.00, 4.00, 'sq_kasten',     'nahe am Wuerfel, Kanten fast scharf'),
    (0.60, 2.00, 'sq_linse',      'spitze Pole, kastiger Querschnitt — linsenartige Kontur'),
    (2.00, 0.60, 'sq_tonne',      'kastige Pole, eingeschnuerter Querschnitt — tonnenartig'),
    (0.50, 4.00, 'sq_diamant',    'schlanke Pole, fast quadratischer Querschnitt'),
    (4.00, 0.50, 'sq_flachstern', 'kastige Pole, sternfoermiger Querschnitt'),
    (0.20, 4.00, 'sq_nadelkasten', 'sehr spitze Pole, kastiger Querschnitt — Nadelspitzen an einem Quader'),
    (1.50, 0.20, 'sq_kreuzquer',  'leicht gerundete Pole, stark eingeschnuerter Querschnitt — kreuzfoermig von oben'),
    (3.00, 1.50, 'sq_fass',       'kastige Pole, maessig gerundeter Querschnitt — fassartig'),
]

SUPERQUADRIKEN = {}
for e1, e2, name, notiz in _ARTEN:
    SUPERQUADRIKEN[name] = (
        f'Superquadrik e1={e1:.2f} e2={e2:.2f}',
        (lambda e1=e1, e2=e2: superquadrik(e1, e2, skalen=(0.55, 0.42, 0.5))),
        notiz + ' (isotrope Skalierung)')
    SUPERQUADRIKEN[f'{name}_lang'] = (
        f'Superquadrik e1={e1:.2f} e2={e2:.2f}, gestreckt',
        (lambda e1=e1, e2=e2: superquadrik(e1, e2, skalen=(0.70, 0.32, 0.42))),
        notiz + ' (anisotrope, gestreckte Skalierung)')


def baue(key):
    return SUPERQUADRIKEN[key][1]()


def pruefen(verbose=True):
    fehler = []
    for k, (label, bauer, _) in SUPERQUADRIKEN.items():
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
                  f'{len(m.faces):6d} Dreiecke  V={m.volume:.4f}  A={m.area:.4f}')
    return fehler


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pruefen', action='store_true')
    p.parse_args()
    print(f'{len(SUPERQUADRIKEN)} Superquadriken\n')
    f = pruefen()
    print()
    if f:
        for k, g in f:
            print(f'  FEHLER {k}: {g}')
        raise SystemExit(1)
    print('Alle Superquadriken wasserdicht.')
