r"""
dichte_numpy.py
===============
Dichteauswertung ohne JAX.

Warum das existiert: `pdf_on_grid` zieht XLA hoch. Neben einem laufenden
Generierungslauf ist das auf diesem Rechner der Weg in den OOM-Killer — genau
daran ist der erste Startpunkt-Lauf gestorben. Fuer das blosse Zeichnen von
Uebersichten reicht NumPy vollkommen aus.

Deckt dieselben drei Faelle ab wie `make_pdf_and_score`: GMM, analytische
Segmente, und den Sockel obendrauf.
"""
import numpy as np


def _gmm(d, X, Y):
    means = np.asarray(d['means'], float)
    covs  = np.asarray(d['covs'], float)
    w     = np.asarray(d['weights'], float); w = w / w.sum()
    out = np.zeros_like(X)
    for m, C, wk in zip(means, covs, w):
        det = C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0]
        if det <= 1e-18:
            continue
        inv = np.array([[C[1, 1], -C[0, 1]], [-C[1, 0], C[0, 0]]]) / det
        dx, dy = X - m[0], Y - m[1]
        q = inv[0, 0]*dx*dx + (inv[0, 1] + inv[1, 0])*dx*dy + inv[1, 1]*dy*dy
        out += wk * np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))
    return out


def _analytisch(d, X, Y):
    sig = float(d.get('sigma', 0.025))
    best = np.zeros_like(X)
    for a, b in np.asarray(d['segments'], float):
        ab = b - a
        l2 = max(float(ab @ ab), 1e-12)
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / l2, 0.0, 1.0)
        dx = X - (a[0] + t * ab[0])
        dy = Y - (a[1] + t * ab[1])
        np.maximum(best, np.exp(-(dx*dx + dy*dy) / (2 * sig**2)), out=best)
    return best


def dichte_auf_gitter(d, res=64):
    """(res, res) Dichtefeld ueber [0,1]^2 — dieselbe Konvention wie pdf_on_grid."""
    xs = np.linspace(0.0, 1.0, res)
    X, Y = np.meshgrid(xs, xs)
    p = _analytisch(d, X, Y) if d.get('type') == 'analytical' else _gmm(d, X, Y)
    ped = d.get('pedestal')
    if ped:
        c = np.asarray(ped['center'], float); sg = float(ped['sigma'])
        g = np.exp(-((X - c[0])**2 + (Y - c[1])**2) / (2 * sg**2))
        p = (1 - ped['weight']) * p / ped['z_base'] + ped['weight'] * g / ped['z_ped']
    return p + 1e-10
