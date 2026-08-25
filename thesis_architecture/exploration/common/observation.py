r"""
observation.py
==============
Was eine abgefahrene Trajektorie ueber das wahre Feld verraet.

Dieser Baustein konstruiert die partielle Beobachtbarkeit, die der Datensatz
nicht mitbringt: die Datenbank enthaelt die *wahre* Dichte, hier wird daraus ein
Messprozess. Wie realistisch das Gesamtergebnis ist, entscheidet sich an dieser
Datei mehr als an jeder Modellwahl — deshalb sind die Annahmen hier explizit und
nicht in einer Trainingsschleife versteckt.

Modell: der Roboter misst entlang seiner Bahn punktweise, mit additivem Rauschen.
Ein Sensorradius > 0 verbreitert das zu mehreren Messpunkten pro Bahnpunkt, was
der Fussabdruck-Kopplung des 3D-Zweigs entspricht.

Differenzierbarkeit: der gemessene Wert wird per bilinearer Interpolation aus dem
wahren Gitter gelesen und ist damit differenzierbar im *Messort*. Variante B
braucht genau das.
"""

import torch
import torch.nn.functional as F


def sample_field(field, pts):
    """Bilineare Abfrage eines Gitterfeldes. field: (H,W), pts: (N,2) in [0,1]^2.

    Gibt (N,) zurueck und ist differenzierbar in `pts`. `grid_sample` erwartet
    Koordinaten in [-1,1] und die Reihenfolge (x, y), waehrend das Feld
    (Zeile=y, Spalte=x) indiziert ist — die Umrechnung hier ist die uebliche
    Fehlerquelle und deshalb an genau einer Stelle gekapselt.
    """
    # Das Wahrheitsgitter kommt aus dem Datenlader (CPU), die Abfragepunkte aus
    # dem Planer — und der laeuft beim Netz auf der GPU. Die Angleichung hier
    # statt an jeder Aufrufstelle: ein Device-Mismatch bricht sonst erst mitten
    # in einer langen Auswertung ab.
    field = field.to(pts.device)
    H, W = field.shape[-2:]
    g = pts.view(1, -1, 1, 2) * 2.0 - 1.0                  # (1, N, 1, 2)
    out = F.grid_sample(field.view(1, 1, H, W), g,
                        mode='bilinear', padding_mode='border',
                        align_corners=True)
    return out.view(-1)


def measure(curve, truth, noise_std=0.02, sensor_radius=0.0, n_ring=4,
            generator=None):
    """Messungen entlang einer Bahn.

    curve:  (T,2) abgefahrene Punkte
    truth:  (H,W) wahres Dichtefeld
    -> (M,2) Orte, (M,) Werte

    Mit `sensor_radius > 0` wird pro Bahnpunkt zusaetzlich ein Ring von
    `n_ring` Punkten gemessen: der Roboter sieht eine Umgebung, nicht nur den
    Punkt unter sich. Das ist der Unterschied zwischen "Spur hinterlassen" und
    "Flaeche abdecken" und aendert die noetige Bahnlaenge erheblich.
    """
    pts = curve
    if sensor_radius > 0 and n_ring > 0:
        ang = torch.arange(n_ring, device=curve.device, dtype=curve.dtype)
        ang = ang * (2 * torch.pi / n_ring)
        off = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        ring = curve.unsqueeze(1) + sensor_radius * off.unsqueeze(0)
        pts = torch.cat([curve, ring.reshape(-1, 2)], dim=0)
    pts = pts.clamp(0.0, 1.0)

    vals = sample_field(truth, pts)
    if noise_std > 0:
        n = torch.randn(vals.shape, device=vals.device, dtype=vals.dtype,
                        generator=generator)
        vals = vals + noise_std * n
    return pts, vals


def thin(pts, vals, max_points=96):
    """Gleichmaessige Ausduennung auf hoechstens `max_points` Messungen.

    Der GP kostet O(n^3) in der Zahl der Messungen. Eine Bahn mit 128 Punkten
    mal Sensorring liefert schnell mehrere hundert, von denen benachbarte
    ohnehin fast dieselbe Information tragen. Gleichmaessig statt zufaellig
    ausduennen, damit die raeumliche Abdeckung der Messmenge erhalten bleibt.
    """
    n = pts.shape[0]
    if n <= max_points:
        return pts, vals
    idx = torch.linspace(0, n - 1, max_points, device=pts.device).long()
    return pts[idx], vals[idx]
