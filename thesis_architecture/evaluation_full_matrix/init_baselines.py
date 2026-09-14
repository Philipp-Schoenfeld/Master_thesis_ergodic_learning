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
