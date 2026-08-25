r"""
acquisition.py
==============
Vom Glauben zur abzudeckenden Zielverteilung.

Hier sitzt die inhaltliche Kernentscheidung des ganzen Ordners. Die naive
Variante waere, zwei getrennte ergodische Verluste zu bilden — einen auf den
bekannten Anteil, einen auf die Unsicherheit. Das ist schlecht gestellt:
Ergodizitaet heisst, dass die Aufenthaltsverteilung *einer* Zielverteilung
gleicht. Fordert man Uebereinstimmung mit zweien gleichzeitig, ist das im
Allgemeinen unerfuellbar, und das Verhaeltnis der Verlustgewichte entscheidet
willkuerlich, welche staerker verfehlt wird.

Stattdessen werden die beiden Anteile *vor* dem ergodischen Verlust zu einer
Dichte verrechnet:

    Phi(x) = mu(x) + kappa * sigma(x)

Das ist die UCB-Akquisefunktion (Srinivas et al., GP-UCB) und macht aus dem
Problem wieder ein wohlgestelltes: eine Bahn, eine Zielverteilung, ein Verlust.
`kappa` ist dabei ein interpretierbarer Parameter statt eines Verhaeltnisses
zweier Verlustgewichte — gross heisst erkunden, klein heisst ausbeuten.
"""

import torch


def ucb_density(mu, sd, kappa=2.0, floor=1e-6, norm='sum'):
    """Phi = mu + kappa * sigma, auf nichtnegative Dichte gebracht.

    mu, sd: (R,R) -> (R,R)

    `norm` legt die Normierung fest und ist der Grund, warum eine fertig
    trainierte Zieldichte nicht ohne Weiteres in ein bestehendes Netz passt:

      'sum' — Integral 1, die uebliche Wahrscheinlichkeitsdichte. Das ist die
              Konvention der Varianten A-E, weil dort der ergodische Verlust
              die Dichte ohnehin nur ueber ihre Fourier-Koeffizienten sieht.
      'max' — Maximum 1. Das ist die Konvention, in der die *Trainingsdichten*
              des Hauptrunners vorliegen (`d_map /= d_map.max()` in
              `_load_shapes`), und damit auch die Skala des dritten
              Partikelkanals, auf den das trainierte Netz konditioniert wurde.

    Der Unterschied ist kein Detail: auf einem 48x48-Gitter liegt eine
    summennormierte Dichte um den Faktor ~2300 tiefer. Ein Netz, das Gewichte
    der Groessenordnung 1 gesehen hat, bekaeme eine Wolke, deren Gewichtskanal
    numerisch nicht von Null zu unterscheiden ist. Wer eine UCB-Dichte in ein
    bestehendes Netz gibt, braucht deshalb 'max'.

    `floor` hebt die gesamte Dichte an und verhindert, dass Gebiete mit
    Phi = 0 fuer die Partikelziehung voellig unsichtbar werden. Bei reiner
    Ausbeutung (kappa = 0) ueber einem duennen Feld ist das der Unterschied
    zwischen "deckt die Form ab" und "hat keine gueltige Zielverteilung".

    **Entartungsfall, der beim Testen auffiel:** ohne jede Vormessung ist
    mu = 0 und sigma ueberall gleich. Phi ist dann gleichverteilt, und zwar
    *fuer jedes* kappa — Erkundung ist in diesem Zustand nicht definiert, weil
    kein Ort informativer ist als ein anderer, und jede raumfuellende Bahn ist
    gleich gut. Der Vergleich verschiedener kappa braucht also einen Glauben
    mit Struktur, entweder aus Vormessungen (`n_prior > 0`) oder aus einem
    vorangegangenen Planungsschritt (Varianten C und D). `is_degenerate`
    macht diesen Zustand pruefbar, statt ihn stumm passieren zu lassen.
    """
    phi = mu + kappa * sd
    phi = phi.clamp(min=0.0) + floor
    if norm == 'max':
        return phi / phi.max().clamp(min=1e-12)
    return phi / phi.sum().clamp(min=1e-12)


def is_degenerate(phi, tol=1e-4):
    """True, wenn die Zieldichte praktisch gleichverteilt ist.

    Dann traegt sie keine Information und jeder Vergleich von kappa-Werten
    darauf ist bedeutungslos. Die Runner melden das, statt Zahlen auszugeben,
    die nach einem Ergebnis aussehen.
    """
    rng = float(phi.max() - phi.min())
    return rng < tol * float(phi.mean().clamp(min=1e-12)) * phi.numel()


def particles_from_density(phi, n_particles, device=None, mode='uniform',
                           threshold=1e-5, generator=None):
    """Partikelwolke (N,3) mit (x, y, mu) aus einer Dichte (R,R).

    Bewusst dieselbe Konvention wie `sample_particles` im Hauptrunner: bei
    `mode='uniform'` werden Orte gleichverteilt ueber dem Traeger gezogen und
    tragen den Dichtewert als Gewicht mit. Nur so passt die Wolke ohne weitere
    Anpassung in das bestehende Netz und in `ErgodicLoss(weighted_target=True)`.
    """
    device = device or phi.device
    R = phi.shape[-1]
    flat = phi.reshape(-1)
    w = (flat > threshold).float() if mode == 'uniform' else flat + 1e-7
    if float(w.sum()) <= 0:
        w = torch.ones_like(flat)

    idx = torch.multinomial(w, n_particles, replacement=True,
                            generator=generator)
    sy, sx = idx // R, idx % R
    jx = (torch.rand(n_particles, device=device, generator=generator) - .5) / (R - 1)
    jy = (torch.rand(n_particles, device=device, generator=generator) - .5) / (R - 1)
    px = (sx.float() / (R - 1) + jx).clamp(0, 1)
    py = (sy.float() / (R - 1) + jy).clamp(0, 1)
    return torch.stack([px, py, flat[idx]], dim=-1)


def kappa_schedule(step, n_steps, kappa0=3.0, kappa1=0.3):
    """Von Erkunden zu Ausbeuten ueber eine Mission hinweg.

    Ein fester Wert kann nicht ausdruecken, dass frueh erkundet und spaet
    abgedeckt werden soll — dieselbe Schwaeche, wegen der GP-UCB in der
    Literatur mit einem kappa-Zeitplan gefahren wird. Nur die Varianten mit
    mehreren Planungsschritten (C, D) koennen davon Gebrauch machen; A und E
    nutzen einen festen Wert.
    """
    if n_steps <= 1:
        return kappa0
    t = step / (n_steps - 1)
    return kappa0 + (kappa1 - kappa0) * t


# ── Alternative Modellierungen der Zieldichte ────────────────────────────────
#
# `ucb_density` ist additiv: Phi = mu + kappa*sigma. Das hat eine Eigenschaft,
# die beim Messen sofort auffaellt und beim Hinschreiben nicht: sigma liegt vor
# der Fahrt fast ueberall bei seinem Maximum, der Term kappa*sigma ist also im
# Wesentlichen ein **konstanter Sockel**. Mit kappa = 3 und einem mittleren
# sigma von 0,87 stehen 2,6 Sockel gegen ein mittleres mu von 0,09 — mu traegt
# rund 4 Prozent zur Zieldichte bei. Selbst bei 240 Messungen sind es erst 30.
#
# Deshalb aendert sich die geplante Bahn kaum, wenn man von 12 auf 60
# Vorabmessungen geht: das Gemessene bekommt schlicht keine Stimme. Die
# folgenden Modelle greifen an genau dieser Stelle an, auf jeweils andere Weise.
# `ucb_density` bleibt unveraendert, damit die gefahrenen Vergleiche gueltig
# bleiben.

def _norm_max(phi, floor=1e-6):
    phi = phi.clamp(min=0.0) + floor
    return phi / phi.max().clamp(min=1e-12)


def phi_ucb(mu, sd, kappa=2.0, **kw):
    """Phi = mu + kappa*sigma. Die bisherige Modellierung."""
    return _norm_max(mu + kappa * sd)


def phi_stretch(mu, sd, kappa=2.0, **kw):
    """Wie UCB, aber der Wertebereich wird auf [0,1] gespreizt.

    Der billigste Angriff auf den Sockel: nicht durch das Maximum teilen,
    sondern das Minimum abziehen und dann durch die Spanne teilen. Aus einem
    Kontrastverhaeltnis von 1,3 wird eines von unendlich — die Struktur, die
    vorher als 30-Prozent-Welligkeit auf einem Sockel sass, fuellt danach den
    ganzen Bereich.

    Der Preis ist begrifflich: die Zieldichte ist danach keine Ueberlagerung
    zweier Groessen mehr, sondern eine monotone Umskalierung davon. Die
    Ordnung der Orte bleibt, ihre Verhaeltnisse nicht.
    """
    phi = mu + kappa * sd
    lo, hi = phi.min(), phi.max()
    return _norm_max((phi - lo) / (hi - lo).clamp(min=1e-12))


def phi_mass(mu, sd, w=0.5, **kw):
    """Beide Anteile auf gleiche Masse gebracht, dann mit `w` gemischt.

        Phi = (1-w) * mu/sum(mu)  +  w * sigma/sum(sigma)

    Damit ist `w` kein Verhaeltnis zweier Groessen mit verschiedenen Einheiten
    mehr, sondern direkt der **Anteil der Zielmasse, der auf Erkundung
    entfaellt**. w = 0,5 heisst: die Haelfte der Aufenthaltszeit soll dem
    Unbekannten gelten, die andere dem Bekannten. Das ist die Groesse, die man
    eigentlich einstellen will, wenn man kappa einstellt.
    """
    m = mu.clamp(min=0.0); m = m / m.sum().clamp(min=1e-12)
    s = sd.clamp(min=0.0); s = s / s.sum().clamp(min=1e-12)
    return _norm_max((1.0 - w) * m + w * s)


def phi_ei(mu, sd, xi=0.01, best=None, **kw):
    """Erwarteter Zugewinn (Expected Improvement).

    Nicht "wo weiss ich wenig", sondern "wo koennte mehr sein als das Beste,
    das ich schon kenne". Ein Gebiet mit hohem sigma, dessen mu weit unter dem
    bisherigen Bestwert liegt, wird dadurch uninteressant — genau der
    Unterschied, der beim additiven UCB fehlt.

    Bezugsgroesse `best` ist per Voreinstellung das Maximum des
    Posterior-Mittelwerts.
    """
    import math
    b = mu.max() if best is None else best
    z = (mu - b - xi) / sd.clamp(min=1e-9)
    # Normalverteilung ohne SciPy
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    return _norm_max((mu - b - xi) * cdf + sd * pdf)


def phi_lse(mu, sd, tau=0.25, **kw):
    """Wahrscheinlichkeit, dass die Dichte eine Schwelle ueberschreitet.

        Phi = P(f(x) > tau) = Normal_cdf((mu - tau) / sigma)

    Die Zielgroesse ist hier nicht der Wert der Dichte, sondern ihr **Traeger**
    — die Frage "gehoert dieser Ort zur Form?". Fuer eine Abdeckungsaufgabe ist
    das oft die ehrlichere Frage, und sie haengt stark an den Messungen: jede
    Messung verschiebt mu und schrumpft sigma, beides veraendert die
    Wahrscheinlichkeit unmittelbar.

    Verwandt mit der Niveaumengen-Schaetzung (Gotovos et al. 2013).
    """
    import math
    z = (mu - tau) / sd.clamp(min=1e-9)
    return _norm_max(0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))))


def phi_mi(mu, sd, kappa=2.0, gamma=0.0, **kw):
    """GP-MI nach Contal et al. (2014).

        Phi = mu + kappa * ( sqrt(gamma + sigma^2) - sqrt(gamma) )

    `gamma` sammelt die bereits eingeholte Information. Waechst es, faellt der
    Zugewinn eines weiteren Besuchs derselben Gegend — der Term saettigt, statt
    wie kappa*sigma unbegrenzt zu locken. Bei gamma = 0 ist es exakt UCB, die
    Modellierung ist also eine echte Verallgemeinerung.
    """
    return _norm_max(mu + kappa * ((gamma + sd ** 2).sqrt()
                                   - torch.as_tensor(float(gamma)).sqrt()))


PHI_MODELLE = {'ucb': phi_ucb, 'stretch': phi_stretch, 'mass': phi_mass,
               'ei': phi_ei, 'lse': phi_lse, 'mi': phi_mi}


def phi_from_belief(mu, sd, modell='ucb', **kw):
    """Eine Zieldichte nach dem gewaehlten Modell. Immer auf Maximum 1."""
    if modell not in PHI_MODELLE:
        raise KeyError(f"unbekanntes Phi-Modell {modell!r}; "
                       f"bekannt: {sorted(PHI_MODELLE)}")
    return PHI_MODELLE[modell](mu, sd, **kw)


def phi_eid(mu, sd, kappa=2.0, noise=0.05, w_sigma=1.0, **kw):
    r"""Erwartete Informationsdichte nach Miller et al. (2016).

    "Ergodic Exploration of Distributed Information" dreht die Frage um: die
    Zieldichte einer ergodischen Bahn soll nicht das *Feld* sein, sondern die
    **erwartete Fisher-Information** einer Messung — also der Ort, an dem eine
    Messung am meisten ueber das noch Unbekannte verraet.

    Fuer einen Lageparameter ist die Fisher-Information einer verrauschten
    Messung des Signals s bekanntlich

        I(x) = || grad s(x) ||^2 / sigma_n^2

    — sie ist gross dort, wo das Signal sich *aendert*, nicht dort, wo es gross
    ist. Ein flaches Plateau, egal wie hoch, verraet ueber die Lage einer Form
    nichts; eine Kante verraet alles. Mit dem GP als Glauben tritt an die Stelle
    von s der Posterior-Mittelwert, und die Unsicherheit steuert einen zweiten
    Gradiententerm bei:

        EID(x) ~ ( ||grad mu||^2 + w * ||grad sigma||^2 ) / sigma_n^2

    **Ein Vorbehalt, der zur Sache gehoert.** Diese Groesse beantwortet die
    Frage "wo messe ich am meisten Neues", nicht "wo verbringe ich Zeit, um
    eine Dichte abzudecken". Als reine Zieldichte gefahren, laesst sie das
    Innere einer Form links liegen und faehrt nur deren Rand ab. Deshalb wird
    sie hier nicht pur benutzt, sondern gegen den Mittelwert gemischt, beide
    auf gleiche Masse gebracht:

        Phi = mu_hat + kappa * EID_hat

    Damit bleibt der kappa-Zeitplan sinnvoll: frueh dominiert die Information,
    spaet die aufgedeckte Dichte. Die reine Form erhaelt man mit kappa gross
    und einem mu, das noch flach ist — also genau am Missionsanfang.
    """
    gy, gx = torch.gradient(mu)
    eid = gx ** 2 + gy ** 2
    if w_sigma:
        sy, sx = torch.gradient(sd)
        eid = eid + w_sigma * (sx ** 2 + sy ** 2)
    eid = eid / max(float(noise) ** 2, 1e-12)

    m = mu.clamp(min=0.0); m = m / m.sum().clamp(min=1e-12)
    e = eid.clamp(min=0.0); e = e / e.sum().clamp(min=1e-12)
    return _norm_max(m + float(kappa) * e)


PHI_MODELLE['eid'] = phi_eid
