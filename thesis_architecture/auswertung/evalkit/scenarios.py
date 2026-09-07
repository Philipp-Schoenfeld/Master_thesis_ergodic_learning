r"""
scenarios.py
============
Die Wissensszenarien: *was* weiss der Planer vor der Fahrt ueber die Zieldichte?

Genau diese Frage trennt Exploration von Exploitation. Bei vollstaendigem
Wissen gibt es nichts zu erkunden, bei gar keinem Wissen nichts auszubeuten —
erst der **teilweise** bekannte Zustand macht das Verhaeltnis messbar. Die drei
gefragten Szenarien unterscheiden sich nicht in der Menge des Wissens, sondern
in seiner *Geometrie*, und das ist der Punkt:

    zufall    Zufaellig verteilte Punktmessungen ueber der ganzen Flaeche.
              Wissen ist ueberall gleich lueckenhaft, es gibt keine Kante.
              Der Glaube ist ein echter GP-Posterior: mu naehert die Wahrheit
              nur an, sigma faellt um jede Messung herum ab.

    haelfte   Die linke Haelfte ist exakt bekannt, die rechte gar nicht. Eine
              harte Kante quer durch das Gebiet. Das Unbekannte grenzt an den
              Rand — eine Bahn kann sich an der Kante entlanghangeln.

    loch      Alles ausser einer Scheibe in der Mitte ist exakt bekannt. Das
              Unbekannte ist **von Wissen umschlossen**; es gibt keine Kante
              zum Rand, an der man sich orientieren koennte, und der Weg
              hinein kostet immer einen Umweg durch bereits Bekanntes.

Dazu zwei Bezugspunkte, die die Zahlen erst interpretierbar machen:

    orakel    Alles bekannt (sigma = 0). Reine Ausbeutung, Obergrenze fuer die
              Exploitation-Metriken. Fuer *jedes* kappa dieselbe Zieldichte.
    blind     Nichts bekannt. mu = 0 und sigma ueberall gleich, Phi also
              gleichverteilt — reine Erkundung. `is_degenerate` meldet diesen
              Zustand; er ist als Referenz gemeint, nicht als Wettbewerber.

Umsetzung
---------
Nichts davon wird hier neu gebaut. `haelfte`, `loch` und `orakel` sind
`MaskiertesWissen` aus `exploration/common/belief.py` mit den Masken aus
`muster_maske`; `zufall` und `blind` sind ein gewoehnlicher `GPBelief` mit
bzw. ohne Vorabmessungen (`exploration/common/data.initial_belief`).

Der Unterschied zwischen den beiden Sorten ist inhaltlich und nicht nur
technisch: bei `zufall` ist das Wissen **weich** (ein Posterior mit Restfehler),
bei `haelfte`/`loch` **hart** (Grundwahrheit innen, gar nichts aussen). Damit
alle Szenarien trotzdem eine gemeinsame Definition von "bekannt" haben, wird
die bekannte Region einheitlich ueber die Unsicherheit erklaert:

    bekannt(x)  :=  sigma(x) <= sigma_schwelle

Bei den harten Masken faellt das exakt mit der Maske zusammen (sigma = 0
innen, 1 aussen); bei `zufall` liefert es die Umgebung der Messpunkte. Nur so
sind `dwell`-Anteil und `lift` ueber alle Szenarien hinweg dieselbe Groesse.
"""

import torch

from exploration.common.acquisition import is_degenerate, ucb_density
from exploration.common.belief import GPBelief, MaskiertesWissen, muster_maske
from exploration.common.data import initial_belief

# Order = order in tables and figures.
SZENARIEN = ['random', 'half', 'hole']
REFERENZEN = ['oracle', 'blind']
ALLE = SZENARIEN + REFERENZEN

# Internal builder keys (legacy German names kept internally)
_SZEN_INTERN = {
    'random':  'zufall',
    'half':    'haelfte',
    'hole':    'loch',
    'oracle':  'orakel',
    'blind':   'blind',
}

BESCHREIBUNG = {
    'random':  'random point measurements',
    'half':    'left half known',
    'hole':    'unknown hole in the centre',
    'oracle':  'everything known (reference)',
    'blind':   'nothing known (reference)',
}

# Default threshold: half the prior standard deviation
# (variance=1 -> sigma_prior=1). A location is considered known as soon as
# uncertainty has been at least halved.
SIGMA_THRESHOLD = 0.5
SIGMA_SCHWELLE = SIGMA_THRESHOLD  # backward-compat alias


def baue_glauben(szenario, wahrheit, *, gp_res=64, n_prior=40, lengthscale=0.08,
                 gp_noise=0.05, sigma_bekannt=0.0, seed=0, device='cpu'):
    """Initial belief for a scenario. -> `GPBelief`-like object.

    `wahrheit` is the (gp_res, gp_res) grid of the true density, normalised to
    maximum 1 -- same convention as in training (`d_map /= d_map.max()`).

    `gp_noise` is deliberately 0.05 instead of the class default 1e-2:
    measurement points on a path are densely spaced; otherwise the GP
    interpolates the noise and the posterior mean shoots far outside [0,1].
    """
    intern = _SZEN_INTERN.get(szenario, szenario)
    if intern in ('haelfte', 'loch', 'orakel'):
        muster = {'haelfte': 'haelfte', 'loch': 'loch', 'orakel': 'alles'}[intern]
        maske = muster_maske(muster, gp_res, device=device)
        return MaskiertesWissen(maske, wahrheit, sigma_bekannt=sigma_bekannt,
                                grid_res=gp_res, lengthscale=lengthscale,
                                noise=gp_noise, device=device)
    if intern == 'zufall':
        return initial_belief(wahrheit, n_prior=n_prior, grid_res=gp_res,
                              lengthscale=lengthscale, noise=gp_noise,
                              seed=seed, device=device)
    if intern == 'blind':
        return GPBelief(grid_res=gp_res, lengthscale=lengthscale,
                        noise=gp_noise, device=device)
    raise KeyError(f"unknown scenario {szenario!r}; known: {ALLE}")


def zieldichte(glaube, kappa, floor=1e-6):
    """(mu, sd, Phi) with Phi = mu + kappa*sigma, normalised to **Maximum 1**.

    The normalisation is not a stylistic choice: the third particle channel,
    on which the network was conditioned, carries density values with a maximum of 1.
    A sum-normalised density would be lower by a factor of ~4000 on a 64x64 grid
    and would be indistinguishable from zero by the network numerically.
    """
    mu, sd = glaube.posterior_grid()
    phi = ucb_density(mu, sd, kappa=kappa, floor=floor, norm='max')
    return mu, sd, phi


def bekannt_maske(sd, schwelle=SIGMA_THRESHOLD):
    """Boolean grid: where is the field considered known? Defined uniformly via sigma."""
    return sd <= schwelle


def anteil_unbekannt(sd, schwelle=SIGMA_THRESHOLD):
    """Area fraction of the unknown region -- the reference value for `lift`."""
    return float((~bekannt_maske(sd, schwelle)).float().mean())


def entartet(phi):
    """True if Phi is practically uniform (scenario `blind`)."""
    return bool(is_degenerate(phi))


def wahrheitsgitter(shape_name, gp_res=64, device='cpu'):
    """Goes through `shape_library` instead of the database so that the
    same density is created here as in `constraints/common.density_and_particles`
    and the numbers match the existing evaluation series.
    """
    from shape_library import get_shape, pdf_on_grid
    d_map, _, _ = pdf_on_grid(get_shape(shape_name), resolution=gp_res)
    d = torch.tensor(d_map, dtype=torch.float32, device=device)
    return d / d.max().clamp(min=1e-12)
