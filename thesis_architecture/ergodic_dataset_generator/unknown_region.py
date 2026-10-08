r"""
unknown_region.py
==================
Erweitert eine bestehende Zieldichte um einen zufaelligen "unbekannten"
Flaechenbereich mit hoher Dichte, fuer das Explorations-Dataset.

Drei Schritte:

1. `sample_unknown_polygon` wuerfelt einen organischen, zufaellig platzierten
   Flaechenausschnitt (ueber `shape_rasterizer.random_smooth_polygon`), dessen
   tatsaechliche Flaeche per Verwerfungsstichprobe auf einen zufaelligen Wert
   zwischen `area_lo` und `area_hi` des Arbeitsraums [0,1]^2 fest genagelt wird.
2. `mit_unknown_region` (in `shape_library.py`) mischt diesen Bereich als
   zusaetzliches GMM in die bestehende Dichte, mit Gewichtsanteil `a` -- siehe
   dortige Begruendung fuer die Mischformel.
3. `build_unknown_variant` haengt zusaetzlich einen Maeander-Abdeckungspfad
   ueber den neuen Bereich an die bestehende heuristische Initialisierung an,
   damit der SVGD-Loeser nicht bei Null anfaengt, den Bereich zu entdecken.

Reine Flaechen-Abdeckung (Schritt 1) ist etwas anderes als Dichte-Gewicht
(Schritt 2): ein kleiner Bereich kann genauso "hohe Dichte" tragen wie ein
grosser -- das eine ist eine geometrische, das andere eine Wahrscheinlich-
keitsmasse-Eigenschaft.
"""
import numpy as np
from matplotlib.path import Path

from shape_rasterizer import random_smooth_polygon, render_filled_polygon, points_to_gmm
from shape_library import mit_unknown_region
from ergodic_solver import _analytical_waypoints, _gmm_waypoints, _downsample_or_pad

_GRID_RES = 200
_XS = np.linspace(0.0, 1.0, _GRID_RES)
_GX, _GY = np.meshgrid(_XS, _XS)
_GRID_PTS = np.column_stack([_GX.ravel(), _GY.ravel()])


def sample_unknown_polygon(rng, area_lo=0.40, area_hi=0.75, max_tries=500):
    """Zufaelliger organischer Flaechenausschnitt mit Flaeche in [area_lo, area_hi].

    Verwerfungsstichprobe statt exaktem Flaechen-Treffer: Groesse, Form und
    Lage werden alle zufaellig gewuerfelt, und nur Kandidaten, deren
    tatsaechliche (gerasterte) Flaeche in den Zielbereich faellt, werden
    akzeptiert. Bei den gewaehlten Parameterbereichen braucht das im Schnitt
    nur wenige Versuche (siehe `test_unknown_region.py`).

    Returns
    -------
    verts      : (n_points, 2) geschlossenes Polygon, in [0,1]^2 geclippt
    area_frac  : tatsaechliche, gerasterte Flaeche als Anteil von [0,1]^2
    """
    for _ in range(max_tries):
        n_verts = int(rng.integers(6, 12))
        r_mean  = float(rng.uniform(0.22, 0.44))
        r_var   = float(rng.uniform(0.03, 0.10))
        cx, cy  = rng.uniform(0.15, 0.85, size=2)
        # `random_smooth_polygon` clamps its radii to <= 0.44 internally (a
        # circle at that radius covers only ~0.61 of [0,1]^2), so reaching the
        # top of `area_hi` (0.75) needs an extra post-hoc scale around the new
        # center, applied *after* the library's own clamp.
        scale   = float(rng.uniform(1.0, 2.0))

        verts = random_smooth_polygon(n_verts=n_verts, rng=rng, r_mean=r_mean,
                                      r_var=r_var, n_points=300)
        verts = (verts - 0.5) * scale + np.array([cx, cy])
        verts = np.clip(verts, 0.0, 1.0)

        mask = Path(verts).contains_points(_GRID_PTS)
        area_frac = float(mask.mean())
        if area_lo <= area_frac <= area_hi:
            return verts, area_frac

    raise RuntimeError(
        f"Kein Polygon mit Flaeche in [{area_lo}, {area_hi}] nach "
        f"{max_tries} Versuchen gefunden.")


def lawnmower_path(poly_verts, spacing=0.04, n_samples_per_line=200):
    """Maeander-Abdeckungspfad, auf den Innenbereich von `poly_verts` geclippt.

    Reiner Heuristik-Init-Beitrag (vgl. die TSP-Lissajous-Pfade in
    `ergodic_solver._gmm_waypoints`) -- muss nicht exakt sein, der SVGD-Loeser
    verfeinert ihn ohnehin. Faellt auf den Flaechenschwerpunkt zurueck, falls
    der Bereich zu duenn fuer eine einzige Scanlinie waere (sollte bei den
    Flaechenanteilen aus `sample_unknown_polygon` praktisch nie vorkommen).
    """
    path = Path(poly_verts)
    ymin, ymax = poly_verts[:, 1].min(), poly_verts[:, 1].max()
    xmin, xmax = poly_verts[:, 0].min(), poly_verts[:, 0].max()

    ys = np.arange(ymin, ymax, spacing)
    xs_line = np.linspace(xmin, xmax, n_samples_per_line)
    segs = []
    for i, y in enumerate(ys):
        row = np.column_stack([xs_line, np.full_like(xs_line, y)])
        inside = path.contains_points(row)
        if not inside.any():
            continue
        idx = np.where(inside)[0]
        seg = row[idx[0]:idx[-1] + 1]
        if i % 2 == 1:
            seg = seg[::-1]
        segs.append(seg)

    if not segs:
        return poly_verts.mean(axis=0, keepdims=True)
    return np.vstack(segs)


def build_unknown_variant(base_shape_def, x0, rng, tsteps, dt,
                          area_lo=0.40, area_hi=0.75, a_range=(0.6, 0.8),
                          region_sigma=0.025, region_n_points=250,
                          lawnmower_spacing=0.04):
    """Baut eine (Dichte, Init-Trajektorie)-Variante mit unbekanntem Bereich.

    Returns
    -------
    new_shape_def : dict, Basisdichte + 'unknown_region'-Schluessel
    p_traj_init   : (tsteps+1, 2) Positions-Referenz fuer `custom_p_traj`
    info          : dict mit polygon, area_frac, a -- fuer Logging/DB
    """
    verts, area_frac = sample_unknown_polygon(rng, area_lo, area_hi)

    interior = render_filled_polygon(verts, n_points=region_n_points)
    region_gmm = points_to_gmm(interior, sigma=region_sigma)
    a = float(rng.uniform(*a_range))
    new_shape_def = mit_unknown_region(base_shape_def, region_gmm, a)

    if base_shape_def.get('type') == 'analytical':
        base_wp = _analytical_waypoints(x0, base_shape_def)
    else:
        base_wp = _gmm_waypoints(x0, base_shape_def)

    sweep = lawnmower_path(verts, spacing=lawnmower_spacing)
    last_pt = base_wp[-1]
    dist = float(np.linalg.norm(sweep[0] - last_pt))
    n_transit = max(5, int(round(dist * 120)))
    transit = np.linspace(last_pt, sweep[0], n_transit)[1:-1]

    combined = np.vstack([base_wp, transit, sweep]) if len(transit) else \
        np.vstack([base_wp, sweep])
    p_traj_init, _u_traj, _v0 = _downsample_or_pad(combined, tsteps)

    info = {'polygon': verts.tolist(), 'area_frac': area_frac, 'a': a}
    return new_shape_def, p_traj_init, info
