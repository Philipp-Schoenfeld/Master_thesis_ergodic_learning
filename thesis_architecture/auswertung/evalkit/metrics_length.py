r"""
metrics_length.py
=================
Rekonstruktion **nach Laenge**: wie gut bildet eine Bahn die Zieldichte ab,
wenn man ihr ein Wegbudget vorgibt?

Die Frage dahinter
------------------
Eine Bahn deckt eine Dichte umso besser ab, je laenger sie ist — das ist
trivial und macht jeden Vergleich bei *freier* Laenge wertlos. Interessant ist
die Kurve: **wie viel Rekonstruktionsguete bekommt man pro Wegeinheit**, und
verlaeuft diese Kurve fuer ein Modell besser als fuer ein anderes. Das ist die
gleiche Denkweise wie bei einer Rate-Distortion-Kurve, und es ist derselbe
Grund, aus dem `exploration/common/metrics.anytime_curve` Varianten bei
gleicher Weglaenge vergleicht statt am Endwert.

Zwei Fragen, die sauber getrennt bleiben muessen
------------------------------------------------
1. **Laengentreue** — wird die *angeforderte* Laenge ueberhaupt erreicht?
   Das ist eine Eigenschaft des Steuermechanismus (gelernte Konditionierung
   oder Inferenz-Kraft), nicht der Abdeckungsqualitaet.
2. **Rekonstruktion** — wie gut deckt die tatsaechlich gefahrene Bahn die
   Zieldichte ab? Das wird gegen die **erreichte**, nicht gegen die
   angeforderte Laenge aufgetragen. Sonst vergleicht man wieder Budgets statt
   Verfahren: ein Mechanismus ohne Laengenautoritaet faehrt bei jedem Ziel
   dieselbe Bahn und saehe in einer Auftragung ueber die Anforderung konstant
   gut aus.

Die Kennzahl fuer Frage 1: `autoritaet`
---------------------------------------
Die Steigung der Regression "erreichte Laenge ueber angeforderte Laenge".

    autoritaet = 1   perfekte Kontrolle
    autoritaet = 0   die Vorgabe hat keinerlei Wirkung

Ein einzelner relativer Fehler beantwortet das nicht: ein Mechanismus, der
immer die natuerliche Laenge der Form ausgibt, hat bei einem Ziel nahe dieser
Laenge einen winzigen Fehler und ist trotzdem voellig steuerlos. Die Steigung
zeigt das sofort, und `r2` sagt, wie sauber der Zusammenhang ist.

Die Kennzahl fuer Frage 2: `rec_jsd`
------------------------------------
Jensen-Shannon-Divergenz zwischen der Aufenthaltsdichte der Bahn und der
Zieldichte, in Bit und damit in [0,1] — beschraenkt und zwischen Formen
vergleichbar, anders als der ergodische Fehler, dessen Skala an der Form
haengt. Der ergodische Fehler und die Abdeckungsdistanz des Solvers laufen
trotzdem als Spalten mit: sie sind die Groessen, in denen der Rest der Arbeit
rechnet, und ein Ergebnis, das nur in einer neuen Metrik gut aussieht, waere
verdaechtig.

Und die Effizienz
-----------------
    nutzen_pro_weg = (rec_jsd(frei) - rec_jsd) / (laenge - laenge_frei)

Wie viel Rekonstruktionsgewinn eine zusaetzliche Wegeinheit gebracht hat,
gemessen gegen den ungefuehrten Arm derselben Form. Negativ heisst: laenger
gefahren und *schlechter* rekonstruiert — der Fall, den man sehen will, wenn
eine Laengenkraft die Bahn nur aufblaeht, statt sie sinnvoll zu verteilen.
"""

import numpy as np
import torch

from exploration.common.metrics import coverage_vs_truth

from .metrics_ee import belegung, jsd, _masse1


def rekonstruktion(curve, ziel, *, bandbreite=0.06, erg=None, cps=None):
    """Rekonstruktionsguete einer Bahn gegenueber einer Zieldichte. -> dict

    Args:
        curve: (T,2) dichte Bahn.
        ziel:  (R,R) Zieldichte (Maximum- oder summennormiert, egal).
        erg/cps: optional die Solver-Metriken mitfuehren.
    """
    dev = curve.device
    ziel = ziel.to(dev)
    ziel_hat = _masse1(ziel)
    c = belegung(curve, ziel.shape[-1], bandbreite, device=dev)

    zeile = {
        'rec_jsd': jsd(c, ziel_hat),
        'rec_cov': float(coverage_vs_truth(curve, ziel_hat)),
        'laenge': float((curve[1:] - curve[:-1]).norm(dim=-1).sum()),
    }
    if erg is not None and cps is not None:
        phi = erg.phi_for(ziel_hat.detach().cpu().numpy())
        e, cov, _ = erg.score(cps, phi, ziel_hat)
        zeile['E_erg'] = e
        zeile['cov_solver'] = cov
    return zeile


def laengentreue(laenge, ziel):
    """Signierter und absoluter relativer Laengenfehler."""
    rel = laenge / max(float(ziel), 1e-12) - 1.0
    return {'rel_fehler': rel, 'abs_fehler': abs(rel)}


# ── Aggregation ueber die Laengenleiter ──────────────────────────────────────
def autoritaet(ziele, erreicht):
    """Steigung und R^2 von "erreicht ueber angefordert".

    -> dict(autoritaet, r2, achsenabschnitt, spanne_erreicht)

    `spanne_erreicht` (max - min der erreichten Laengen) steht daneben, weil
    eine Steigung nahe 0 zwei Ursachen haben kann: keine Wirkung, oder eine
    Wirkung, die im Rauschen verschwindet. Eine Spanne nahe 0 entscheidet das.
    """
    x = np.asarray(ziele, dtype=float)
    y = np.asarray(erreicht, dtype=float)
    gut = np.isfinite(x) & np.isfinite(y)
    x, y = x[gut], y[gut]
    if len(x) < 2 or np.ptp(x) < 1e-9:
        return {'autoritaet': float('nan'), 'r2': float('nan'),
                'achsenabschnitt': float('nan'),
                'spanne_erreicht': float(np.ptp(y)) if len(y) else float('nan')}
    a, b = np.polyfit(x, y, 1)
    rest = y - (a * x + b)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((rest ** 2).sum()) / ss_tot if ss_tot > 1e-12 else float('nan')
    return {'autoritaet': float(a), 'r2': float(r2), 'achsenabschnitt': float(b),
            'spanne_erreicht': float(np.ptp(y))}


def nutzen_pro_weg(zeilen, frei_key='frei', arm_key='arm',
                   qual='rec_jsd', laenge='laenge'):
    """`nutzen_pro_weg` je Zeile ergaenzen, bezogen auf den freien Arm.

    Erwartet Zeilen, die alle zur selben (Modell, Form)-Kombination gehoeren
    und genau einen Arm `frei_key` enthalten.
    """
    frei = next((z for z in zeilen if z[arm_key] == frei_key), None)
    if frei is None:
        return zeilen
    for z in zeilen:
        dl = z[laenge] - frei[laenge]
        z['nutzen_pro_weg'] = ((frei[qual] - z[qual]) / dl
                               if abs(dl) > 1e-6 else float('nan'))
    return zeilen


def flaeche_unter(xs, ys):
    """Mittlere Guete ueber den gemeinsamen Laengenbereich (Trapezregel).

    Ein einzelner Vergleichswert fuer eine ganze Rekonstruktionskurve. Durch
    die Spannweite geteilt, damit Arme mit unterschiedlich weit reichenden
    Laengen nicht allein dadurch besser dastehen, dass ihre Kurve laenger ist.
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    gut = np.isfinite(x) & np.isfinite(y)
    x, y = x[gut], y[gut]
    if len(x) < 2 or np.ptp(x) < 1e-9:
        return float('nan')
    return float(np.trapezoid(y, x) / np.ptp(x))


def bei_budget(xs, ys, budget):
    """Guete bei fester erreichter Laenge, linear interpoliert.

    `None`, wenn der Arm dieses Budget gar nicht erreicht — dann ist er dort
    schlicht nicht vergleichbar, und das soll sichtbar bleiben statt durch den
    Endwert ersetzt zu werden (gleiche Konvention wie
    `exploration/common/metrics.at_budget`).
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    if budget < x[0] or budget > x[-1]:
        return None
    return float(np.interp(budget, x, y))
