r"""
init_baselines.py
==================
Die drei Initialisierungen, die es im Projekt noch nicht gibt: die beiden
linearen (Diagonale, Gerade am oberen Rand) und die heuristische
Dichte-Peak-TSP-Abtastung. Lawnmower/Zufallsbahn/Orakel stehen bereits in
`exploration/common/baselines.py` und werden von dort importiert, nicht
kopiert.

Alle Funktionen geben eine (T,2)-Bahn auf [0,1]^2 zurueck — dieselbe
Konvention wie `common.baselines.lawnmower_path`/`random_path`, damit sie
unveraendert in `SvgdRefiner.refine` (das eine (T,2)-Startbahn erwartet)
weiterverwendet werden koennen.
"""

import numpy as np
import torch

from exploration_optimierung.mission import resample_arclength  # noqa: E402


def diagonal_path(n_points=128, margin=0.04):
    """Vollstaendige Diagonale von unten-links nach oben-rechts."""
    t = torch.linspace(margin, 1 - margin, n_points)
    return torch.stack([t, t], dim=-1)


def straight_top_path(n_points=128, margin=0.04, y=None):
    """Gerade Linie ueber den oberen Rand des Arbeitsraums.

    `y` per Default knapp unter der oberen Kante (`1 - margin`), nicht exakt
    auf ihr — sonst faellt ein Teil der Bahn mit dem Randstrafterm der
    SVGD-Verfeinerung zusammen und wird beim ersten Schritt nach innen
    gezogen, bevor ueberhaupt etwas gemessen wurde.
    """
    y = (1 - margin) if y is None else y
    x = torch.linspace(margin, 1 - margin, n_points)
    return torch.stack([x, torch.full_like(x, y)], dim=-1)


def linear_angle_path(angle_deg, n_points=128, margin=0.04):
    """Straight line through the workspace centre at `angle_deg` (measured
    from the +x axis), running from one edge of the margin-inset square to
    the opposite one -- i.e. the full chord of [margin, 1-margin]^2 in that
    direction. `diagonal_path` is the special case 45 deg, `straight_top_path`
    is *not* one (it does not pass through the centre). Angles in [0, 180)
    cover every distinct line; 180..360 would only reverse the direction.
    """
    theta = np.deg2rad(angle_deg)
    d = np.array([np.cos(theta), np.sin(theta)])
    half = 0.5 - margin
    t_max = min(half / abs(d[i]) for i in range(2) if abs(d[i]) > 1e-9)
    t = torch.linspace(-t_max, t_max, n_points)
    dir_t = torch.tensor(d, dtype=torch.float32)
    return 0.5 + t[:, None] * dir_t[None, :]


def linear_ray_path(start, angle_deg, length, n_points=128, margin=0.04):
    """Straight line leaving `start` at `angle_deg` (from the +x axis),
    `length` long in total, reflected specularly at the walls of the
    margin-inset square [margin, 1-margin]^2 (billiard path).

    Replanning counterpart of `linear_angle_path`: that one is a full chord
    through the workspace centre and therefore cannot begin at an arbitrary
    agent position. Here every candidate starts exactly at `start` (the point
    where the previous length unit ended) and the 30 candidates differ only in
    their heading, `angle_deg` in [0, 360). The first point is `start`
    itself even if it lies slightly outside the margin box; the line proper
    begins at the nearest point inside the box.
    """
    lo, hi = margin, 1.0 - margin
    start = np.asarray(start, dtype=np.float64).reshape(2)
    p = np.clip(start, lo, hi)
    th = np.deg2rad(angle_deg)
    d = np.array([np.cos(th), np.sin(th)])
    pts = [start.copy()]
    if np.abs(p - start).max() > 1e-12:
        pts.append(p.copy())
    remaining = float(length)
    for _ in range(1000):
        if remaining <= 1e-9:
            break
        for k in range(2):                      # turn away from a wall we sit on
            if (p[k] <= lo + 1e-9 and d[k] < 0) or (p[k] >= hi - 1e-9 and d[k] > 0):
                d[k] = -d[k]
        t_wall = [np.inf, np.inf]
        for k in range(2):
            if d[k] > 1e-12:
                t_wall[k] = (hi - p[k]) / d[k]
            elif d[k] < -1e-12:
                t_wall[k] = (lo - p[k]) / d[k]
        t = min(t_wall[0], t_wall[1], remaining)
        p = p + t * d
        pts.append(p.copy())
        remaining -= t
        for k in range(2):
            if t_wall[k] <= t + 1e-12:
                d[k] = -d[k]
    curve = torch.tensor(np.stack(pts), dtype=torch.float32)
    if curve.shape[0] < 2:
        curve = torch.cat([curve, curve], dim=0)
    return resample_arclength(curve, n_points)


def random_walk_path(n_points=128, seed=0, step_std=0.035, start=None):
    """Echte Irrfahrt (nicht die glatte Spline-Bahn aus `common.baselines`).

    Gaussche Schritte, an den Raendern reflektiert statt geklemmt — Klemmen
    wuerde die Bahn an den Kanten festkleben lassen und die Irrfahrt praktisch
    in eine Randbahn verwandeln. Anschliessend nach Bogenlaenge auf
    `n_points` neu abgetastet, damit die Punktdichte mit den uebrigen
    Baselines vergleichbar bleibt (eine rohe Irrfahrt haeuft Punkte dort, wo
    sie zoegert).
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    raw_n = max(n_points * 4, 512)
    steps = torch.randn(raw_n, 2, generator=g) * step_std
    pos = (start if start is not None
           else torch.tensor([0.5, 0.5])).clone().float()
    pts = [pos.clone()]
    for s in steps:
        nxt = pos + s
        # Reflexion an [0,1]^2: spiegeln statt abschneiden
        for d in range(2):
            if nxt[d] < 0:
                nxt[d] = -nxt[d]
            elif nxt[d] > 1:
                nxt[d] = 2 - nxt[d]
        nxt = nxt.clamp(0.0, 1.0)
        pts.append(nxt.clone())
        pos = nxt
    curve = torch.stack(pts, dim=0)
    return resample_arclength(curve, n_points)


# ─────────────────────────────────────────────────────────────────────────
# Heuristische Initialisierung: Dichte-Peaks greedy per TSP-naeher-Nachbar
# abgefahren. Bewusst kein gelerntes/optimierendes Verfahren — das ist der
# Punkt der Baseline: ein einfacher, nicht-gelernter Coverage-Heuristik, der
# unterscheidbar von Maeander (dichte-unabhaengig) und Irrfahrt (ziellos)
# ist.
# ─────────────────────────────────────────────────────────────────────────

def _find_peaks(phi, n_peaks=12, min_dist=0.10):
    """Lokale Dichtemaxima per iterativer Unterdrueckung des Umfelds.

    Kein echtes Non-Max-Suppression auf dem Gradienten — einfacher und
    robuster fuer eine Heuristik: das globale Maximum nehmen, eine Scheibe
    vom Radius `min_dist` um es herum auf 0 setzen, wiederholen. Terminiert
    von selbst, wenn keine positive Dichte mehr uebrig ist.
    """
    R = phi.shape[-1]
    ys, xs = torch.meshgrid(torch.linspace(0, 1, R), torch.linspace(0, 1, R),
                            indexing='ij')
    flat = phi.detach().clone().clamp(min=0.0).cpu()
    peaks = []
    for _ in range(n_peaks):
        idx = int(torch.argmax(flat))
        val = float(flat.reshape(-1)[idx])
        if val <= 1e-9:
            break
        iy, ix = divmod(idx, R)
        px, py = ix / (R - 1), iy / (R - 1)
        peaks.append((px, py))
        d2 = (xs - px) ** 2 + (ys - py) ** 2
        flat[d2 < min_dist ** 2] = 0.0
    return peaks


def _greedy_tour(points, start=None):
    """Naechster-Nachbar-Tour durch `points`, optional ab `start`."""
    pts = [tuple(p) for p in points]
    if start is not None:
        cur = (float(start[0]), float(start[1]))
    else:
        cur = pts.pop(0)
    tour = [cur]
    remaining = pts[:]
    while remaining:
        d = [((cur[0] - p[0]) ** 2 + (cur[1] - p[1]) ** 2) for p in remaining]
        j = min(range(len(remaining)), key=lambda i: d[i])
        cur = remaining.pop(j)
        tour.append(cur)
    return tour


def heuristic_peak_path(phi, n_points=128, n_peaks=12, min_dist=0.10,
                        start=None):
    """Greedy-TSP-Abtastung der Dichte-Peaks von `phi` (R,R).

    `phi` ist hier bewusst die *aktuelle* Zieldichte (mu+kappa*sigma unter
    der jeweiligen Wissensstufe, oder die volle Wahrheit bei
    `ground_truth`) — dieselbe Information, die auch das CFM-Netz zur
    Partikelziehung bekommt. Ein fairer Vergleich zwischen "gelernt" und
    "heuristisch" braucht denselben Informationsstand, sonst vergleicht man
    Wissen statt Verfahren.
    """
    peaks = _find_peaks(phi, n_peaks=n_peaks, min_dist=min_dist)
    if len(peaks) < 2:
        peaks = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
    tour = _greedy_tour(peaks, start=start)
    curve = torch.tensor(tour, dtype=torch.float32)
    if curve.shape[0] < 2:
        curve = torch.cat([curve, curve], dim=0)
    return resample_arclength(curve, n_points)


# ─────────────────────────────────────────────────────────────────────────
# GUI heuristic, extracted for reuse ("nimm unsere Heuristik, wie sie auch
# fuer die GUI berechnet wird" -- Philipp, 2026-09-17). This is a *different*
# algorithm from `heuristic_peak_path` above (which pre-dates this and was
# an intentionally simpler reimplementation, see the module docstring): here
# it is `exploration/interactive_sim.py::_heuristic_init_from_phi`, moved
# out of the `Mission` class so eval matrix and GUI run the exact same code
# instead of two implementations that could silently drift apart -- the same
# reasoning `svgd_refine.py`'s module docstring gives for why `SvgdRefiner`
# itself lives outside the GUI. `interactive_sim.py` is left untouched; it
# still has its own method, unrelated to this copy.
# ─────────────────────────────────────────────────────────────────────────

def gui_heuristic_path(phi, start_pos=(0.5, 0.5), n_points=128, nxi=25, deg=5):
    """TSP + local serpentine heuristic, identical in every numeric step to
    the GUI's own initializer.

    1. Top-weighted cells of `phi` (R,R) -> k-means cluster centers (peaks).
    2. Greedy nearest-neighbour tour through the centers, starting at
       `start_pos` (the domain center by default -- there is no "current
       agent position" for a from-scratch single-shot baseline, matching
       `_get_init`'s own fallback `sp_np = np.array([0.5, 0.5])` when no
       position is given).
    3. A local serpentine swing around each center, connected by straight
       transit segments.
    4. Interpolated to `nxi` B-spline control points (25, the project-wide
       default `NXI`) and rendered to a dense `n_points`-point curve via the
       B-spline basis -- the GUI itself skips this last rendering step
       because it hands the control points straight to `planner.plan(init=)`
       as a network warm-start, but every other baseline in this module
       (`heuristic_peak_path`, `diagonal_path`, ...) returns a dense curve,
       so this does too for a fair, uniform interface into `SvgdRefiner`.

    Falls back to a plain diagonal when `phi` has too little structure for
    k-means to find at least 2 centers -- same fallback the GUI uses.
    """
    phi_np = phi.detach().cpu().numpy().copy()
    R = phi_np.shape[-1]
    start_pos_np = np.asarray(start_pos, dtype=np.float64)

    flat = phi_np.ravel()
    n_peaks = min(8, max(3, int((flat > 0.3 * flat.max()).sum() / (R * 0.5))))
    top_idx = np.argsort(flat)[-n_peaks * R:]
    top_y, top_x = np.unravel_index(top_idx, phi_np.shape)
    top_pts = np.stack([top_x / (R - 1), top_y / (R - 1)], axis=-1)

    from scipy.cluster.vq import kmeans2
    n_centers = min(n_peaks, len(top_pts))
    if n_centers < 2:
        t = np.linspace(0, 1, nxi)[:, None]
        cps = 0.05 + 0.9 * np.tile(t, (1, 2))
    else:
        centers, _ = kmeans2(top_pts, n_centers, minit='points')

        curr = start_pos_np
        remaining = list(range(len(centers)))
        order = []
        while remaining:
            dists = [np.linalg.norm(curr - centers[i]) for i in remaining]
            best = remaining.pop(int(np.argmin(dists)))
            order.append(best)
            curr = centers[best]
        ordered = centers[order]

        all_pts = [np.array([start_pos_np])]
        for c in ordered:
            prev = all_pts[-1][-1]
            n_transit = max(3, int(np.linalg.norm(c - prev) * 40))
            transit = np.linspace(prev, c, n_transit)
            spread = 0.08
            n_swing = 12
            tau = np.linspace(-1, 1, n_swing)
            dx = np.array([spread, 0])
            dy = np.array([0, spread * 0.4])
            serpentine = c[None, :] + np.outer(tau, dx) + \
                np.outer(np.sin(2 * np.pi * tau), dy)
            serpentine = np.clip(serpentine, 0.02, 0.98)
            all_pts.extend([transit, serpentine])

        combined = np.vstack(all_pts)
        idx = np.linspace(0, len(combined) - 1, nxi).astype(int)
        cps = combined[idx]
        cps = np.clip(cps, 0.02, 0.98)

    from obstacles import bspline_basis_matrix
    B = bspline_basis_matrix(nxi, n_points, deg)
    dense = B @ cps
    return torch.from_numpy(dense).float()
