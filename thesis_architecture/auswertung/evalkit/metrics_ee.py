r"""
metrics_ee.py
=============
Exploration und Exploitation — **einzeln gemessen**, und ein Verhaeltnis, das
sich zwischen Formen, Szenarien und Modellen vergleichen laesst.

Warum die beiden Seiten getrennt gemessen werden muessen
--------------------------------------------------------
Eine einzelne Kennzahl verdeckt genau den Zielkonflikt, um den es geht: eine
Bahn kann die bekannte Dichte perfekt abdecken und nichts dazulernen, oder viel
lernen und die bekannte Dichte verfehlen. Beide Faelle sehen in einem
gemittelten Skalar gleich aus. Deshalb steht hier links eine
Ausbeutungs-Spalte, rechts eine Erkundungs-Spalte, und erst danach ein
Verhaeltnis.

Die drei Bezugsgroessen
-----------------------
Aus dem Glauben ueber das Feld ergeben sich drei Dichten auf demselben Gitter:

    mu_hat    Posterior-Mittelwert, auf Masse 1 gebracht — *das Bekannte*.
              Ziel der Ausbeutung: dort soll die Bahn Zeit verbringen.
    sd_hat    Posterior-Standardabweichung, auf Masse 1 — *das Unbekannte*.
              Ziel der Erkundung.
    c         Aufenthaltsdichte der Bahn (Gauss-Kern der Sensorbreite um jeden
              Bahnpunkt, auf Masse 1). Das ist die Groesse, die die ergodische
              Theorie mit der Zieldichte vergleicht.

Warum Jensen-Shannon und nicht der ergodische Fehler als Leitmass
-----------------------------------------------------------------
Der ergodische Fehler (Fourier-gewichtete Koeffizientendifferenz) ist die
Groesse des Solvers und wird deshalb **mitberichtet**. Als Verhaeltniszaehler
taugt er nicht: er ist nach oben unbeschraenkt und haengt in seiner Skala an
der Form (eine ausgedehnte Dichte erzeugt systematisch groessere Werte als eine
kompakte). Ein Quotient zweier solcher Zahlen ist zwischen Formen nicht
vergleichbar.

Die Jensen-Shannon-Divergenz (Lin 1991) ist dagegen **symmetrisch und
beschraenkt**: in Bit gerechnet liegt sie immer in [0, 1], fuer jede Form und
jedes Szenario. Damit sind

    exploit_jsd = JSD(c || mu_hat)        (klein = deckt Bekanntes gut ab)
    explore_jsd = JSD(c || sd_hat)        (klein = geht ins Unbekannte)

zwei Zahlen auf **derselben** Skala, und ihr Verhaeltnis ist eine sinnvolle
Groesse:

    EEI = exploit_jsd / (exploit_jsd + explore_jsd)   in [0, 1]

    EEI = 0.5   die Aufenthaltsverteilung liegt gleich weit von beiden Zielen
    EEI < 0.5   ausbeutungslastig (naeher am Bekannten)
    EEI > 0.5   erkundungslastig

Diese Konstruktion — zwei Ziele, beide auf eine gemeinsame beschraenkte Skala
gebracht, dann als Anteil ausgedrueckt — ist die uebliche Art, ein
Zwei-Ziel-Verhaeltnis vergleichbar zu machen. Sie ersetzt nicht die
Einzelspalten, sie fasst sie zusammen.

Der zweite, anschaulichere Quotient: `lift`
-------------------------------------------
    lift = (Anteil der Aufenthaltsmasse im unbekannten Gebiet)
           / (Flaechenanteil des unbekannten Gebiets)

Ein `lift` von 1 heisst: die Bahn behandelt Unbekanntes wie jeder andere Ort
auch — sie erkundet nicht, sie faellt nur hinein. Groesser als 1 heisst
gezielte Erkundung, kleiner als 1 aktives Meiden. Das ist die klassische
Anreicherungs-Kennzahl und braucht keine Referenzlaeufe.

Der Vergleich zwischen Modellen und Szenarien: Abstand zum Utopiepunkt
----------------------------------------------------------------------
Ueber eine kappa-Leiter spannen die Paare (exploit_jsd, explore_jsd) eine
Kompromisskurve auf. Um zwei Modelle zu vergleichen, werden beide Achsen **je
Form** auf [0,1] gebracht (bestes und schlechtestes Ergebnis der eigenen
Leiter), und dann der Abstand zum Utopiepunkt (0,0) gemessen:

    d_utopie = sqrt(x^2 + y^2) / sqrt(2)   in [0, 1], klein ist besser

Das ist die uebliche Skalarisierung in der Mehrzieloptimierung (Abstand zum
Idealpunkt) und beantwortet die Frage "wer erreicht den besseren Kompromiss",
ohne vorher ein Gewichtsverhaeltnis festlegen zu muessen. Zusaetzlich wird die
Pareto-Front der Leiter markiert — Punkte, die von keinem anderen in *beiden*
Achsen geschlagen werden.

Und die Kontrolle, die keine dieser Zahlen ersetzt
--------------------------------------------------
`cov_wahr` misst die Abdeckung der **wahren** Dichte, die der Planer nie
gesehen hat, und `belief_rmse`, ob das Gelernte auch stimmt. Ein Modell, das
den schoensten EEI erreicht und die Wahrheit verfehlt, hat nichts gewonnen.
"""

import numpy as np
import torch

from exploration.common.metrics import (belief_rmse, coverage_vs_truth,
                                        information_gain)
from exploration.common.observation import measure, thin

LOG2 = float(np.log(2.0))


# ── Dichten auf dem Gitter ───────────────────────────────────────────────────
def belegung(curve, res, bandbreite, device=None):
    """Aufenthaltsdichte einer Bahn auf einem (res,res)-Gitter, Masse 1.

    Ein Gauss-Kern der Breite `bandbreite` um jeden Bahnpunkt, aufsummiert. Die
    Breite ist der Sensorradius: was beim Vorbeifahren erfasst wird, gilt als
    besucht, nicht nur der Punkt unter dem Roboter. Der Wert waechst mit der
    Zahl der Bahnpunkte in der Naehe, ist also ein Mass fuer die *dort
    verbrachte Zeit*.

    (Gleiche Konstruktion wie `visitation_field` in
    `exploration/apply_cfm_belief.py`; dort steckt sie in einem CLI-Skript, das
    beim Import `exploration/` auf `sys.path` zoege und damit das Modul
    `common` mehrdeutig machte — siehe Namensfalle in `evalkit/__init__.py`.)
    """
    device = device or curve.device
    a = torch.linspace(0, 1, res, device=device)
    ys, xs = torch.meshgrid(a, a, indexing='ij')
    zellen = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    d2 = torch.cdist(zellen, curve.to(device).float()) ** 2
    k = torch.exp(-d2 / (2.0 * bandbreite ** 2)).sum(dim=1)
    return _masse1(k.view(res, res))


def _masse1(x, floor=0.0):
    x = x.clamp(min=0.0) + floor
    return x / x.sum().clamp(min=1e-12)


def jsd(p, q, eps=1e-12):
    """Jensen-Shannon-Divergenz in **Bit**, also in [0, 1].

    Symmetrisch und beschraenkt — das ist der Grund, warum sie hier statt der
    KL-Divergenz steht: KL wird unendlich, sobald p Masse hat, wo q keine hat,
    und genau das passiert staendig (die Bahn faehrt durch Gebiete mit
    sigma = 0).
    """
    p = _masse1(p.flatten()).double()
    q = _masse1(q.flatten()).double()
    m = 0.5 * (p + q)

    def h(a):
        return -(a * (a + eps).log()).sum()

    return float((h(m) - 0.5 * (h(p) + h(q))) / LOG2)


# ── Eine Zeile: alle Metriken fuer eine erzeugte Bahn ────────────────────────
def bewerte(curve, *, mu, sd, wahrheit, glaube, erg=None, cps=None,
            sigma_schwelle=0.5, bandbreite=0.06, gp_noise=0.05,
            sensor_radius=0.06, max_obs=96):
    """Alle Exploration/Exploitation-Metriken einer Bahn. -> dict

    Args:
        curve: (T,2) dichte Bahn im Einheitsquadrat.
        mu, sd: (R,R) Posterior des Ausgangsglaubens.
        wahrheit: (R,R) wahre Dichte (nur der Auswerter kennt sie).
        glaube: der Ausgangsglaube, fuer den Informationsgewinn.
        erg: optional `constraints.common.ErgodicMetrics`, dann werden
            zusaetzlich die Solver-Energien gegen mu_hat und sd_hat berichtet.
        cps: (1,nxi,2) Kontrollpunkte, nur fuer `erg` gebraucht.
    """
    dev = curve.device
    res = mu.shape[-1]
    mu = mu.to(dev)
    sd = sd.to(dev)
    wahrheit = wahrheit.to(dev)

    # Entartete Bezugsdichten sauber melden statt Zahlen zu erfinden.
    # `orakel` hat sigma = 0 ueberall (nichts zu erkunden), `blind` hat mu = 0
    # ueberall (nichts auszubeuten). Beide sind als *Referenzen* gemeint; die
    # jeweils sinnlose Spalte wird NaN, damit sie in keinem Mittelwert landet.
    hat_mu = float(mu.clamp(min=0).sum()) > 1e-9
    hat_sd = float(sd.clamp(min=0).sum()) > 1e-9

    mu_hat = _masse1(mu)
    sd_hat = _masse1(sd)
    c = belegung(curve, res, bandbreite, device=dev)

    bekannt = sd <= sigma_schwelle
    unbekannt = ~bekannt
    flaeche_unbekannt = float(unbekannt.float().mean())
    dwell_unbekannt = float(c[unbekannt].sum()) if flaeche_unbekannt > 0 else 0.0

    nan = float('nan')
    e_jsd = jsd(c, mu_hat) if hat_mu else nan
    x_jsd = jsd(c, sd_hat) if hat_sd else nan

    # Mission nachfahren: was haette der Roboter unterwegs gemessen?
    glaube_nach = glaube.clone()
    pts, vals = measure(curve.detach().cpu(), wahrheit.cpu(),
                        noise_std=gp_noise, sensor_radius=sensor_radius)
    glaube_nach.observe(*thin(pts, vals, max_points=max_obs))

    zeile = {
        # ── Ausbeutung (klein ist besser, ausser dwell) ──────────────────
        'exploit_jsd': e_jsd,
        'exploit_cov': float(coverage_vs_truth(curve, mu_hat)) if hat_mu else nan,
        'dwell_bekannt': 1.0 - dwell_unbekannt,
        # ── Erkundung ────────────────────────────────────────────────────
        'explore_jsd': x_jsd,
        'explore_cov': float(coverage_vs_truth(curve, sd_hat)) if hat_sd else nan,
        'dwell_unbekannt': dwell_unbekannt,
        'neu_gesehen': _neu_gesehen(curve, unbekannt, sensor_radius),
        'info_gain': information_gain(glaube, glaube_nach),
        # ── Verhaeltnis ──────────────────────────────────────────────────
        'EEI': (e_jsd / max(e_jsd + x_jsd, 1e-12)
                if (hat_mu and hat_sd) else nan),
        'lift': (dwell_unbekannt / flaeche_unbekannt
                 if flaeche_unbekannt > 1e-9 else float('nan')),
        'flaeche_unbekannt': flaeche_unbekannt,
        # ── Kontrolle gegen die Wahrheit ─────────────────────────────────
        'cov_wahr': float(coverage_vs_truth(curve, wahrheit)),
        'belief_rmse': belief_rmse(glaube_nach, wahrheit),
        'laenge': float((curve[1:] - curve[:-1]).norm(dim=-1).sum()),
    }

    if erg is not None and cps is not None:
        zeile['exploit_erg'] = (
            erg.score(cps, erg.phi_for(mu_hat.detach().cpu().numpy()), mu_hat)[0]
            if hat_mu else nan)
        zeile['explore_erg'] = (
            erg.score(cps, erg.phi_for(sd_hat.detach().cpu().numpy()), sd_hat)[0]
            if hat_sd else nan)
    return zeile


def _neu_gesehen(curve, unbekannt, radius):
    """Anteil der unbekannten Zellen, die der Sensor unterwegs erfasst hat.

    Die anschaulichste Erkundungszahl: nicht "wie lange war die Bahn im
    Unbekannten", sondern "wie viel davon hat sie ueberhaupt zu Gesicht
    bekommen". Eine Bahn, die im Unbekannten kreist, hat hohen `dwell`, aber
    niedriges `neu_gesehen`.
    """
    if not bool(unbekannt.any()):
        return float('nan')
    res = unbekannt.shape[-1]
    dev = curve.device
    a = torch.linspace(0, 1, res, device=dev)
    ys, xs = torch.meshgrid(a, a, indexing='ij')
    zellen = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)[unbekannt.reshape(-1)]
    d = torch.cdist(zellen, curve.to(dev).float()).min(dim=1).values
    return float((d <= radius).float().mean())


# ── Aggregation ueber die kappa-Leiter ───────────────────────────────────────
def _normiere(werte):
    """Auf [0,1] mit dem Besten (0) und Schlechtesten (1) der eigenen Leiter."""
    w = np.asarray(werte, dtype=float)
    lo, hi = np.nanmin(w), np.nanmax(w)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(w)
    return (w - lo) / (hi - lo)


def utopie(zeilen, gruppe=('modell', 'szenario', 'shape'),
           x='exploit_jsd', y='explore_jsd'):
    """`d_utopie` und `balance` je Zeile ergaenzen (in-place) und zurueckgeben.

    Normiert wird **je Gruppe**, also je Form: nur innerhalb einer Form ist die
    kappa-Leiter ein fairer Vergleich. Ueber Formen hinweg wird erst danach
    gemittelt.

    `balance` = x_norm / (x_norm + y_norm): 0 heisst "reine Ausbeutung war der
    einzige Kompromiss, den dieser Punkt eingegangen ist".
    """
    schluessel = {}
    for z in zeilen:
        schluessel.setdefault(tuple(z[g] for g in gruppe), []).append(z)
    for _, gr in schluessel.items():
        xn = _normiere([g[x] for g in gr])
        yn = _normiere([g[y] for g in gr])
        for g, a, b in zip(gr, xn, yn):
            g['x_norm'], g['y_norm'] = float(a), float(b)
            g['d_utopie'] = float(np.hypot(a, b) / np.sqrt(2.0))
            g['balance'] = float(a / max(a + b, 1e-12))
    return zeilen


def pareto(punkte):
    """Indizes der Pareto-Front bei Minimierung **beider** Koordinaten."""
    p = np.asarray(punkte, dtype=float)
    front = []
    for i in range(len(p)):
        dominiert = np.any(np.all(p <= p[i], axis=1) & np.any(p < p[i], axis=1))
        if not dominiert:
            front.append(i)
    return front


def hypervolumen(punkte, ref=(1.0, 1.0)):
    """2D-Hypervolumen der Front gegen einen Referenzpunkt. Gross ist besser.

    Ein einzelner Vergleichswert fuer eine ganze Kompromisskurve — nuetzlich,
    wenn zwei Modelle sich an verschiedenen Stellen der Leiter ueberholen und
    kein einzelner kappa-Wert die Frage entscheidet.
    """
    p = np.asarray(punkte, dtype=float)
    p = p[np.all(np.isfinite(p), axis=1)]
    p = p[(p[:, 0] < ref[0]) & (p[:, 1] < ref[1])]
    if len(p) == 0:
        return 0.0
    p = p[pareto(p)]
    p = p[np.argsort(p[:, 0])]
    hv, y_vor = 0.0, ref[1]
    for x, y in p:
        if y < y_vor:
            hv += (ref[0] - x) * (y_vor - y)
            y_vor = y
    return float(hv)
