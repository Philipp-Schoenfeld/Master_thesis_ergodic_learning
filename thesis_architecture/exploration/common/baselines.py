r"""
baselines.py
============
Referenzpunkte, ohne die die Zahlen der Varianten nicht interpretierbar sind.

"Variante B erreicht Abdeckung 0.044" ist keine Aussage. "Variante B erreicht
78 % dessen, was mit vollstaendigem Vorwissen moeglich waere, und liegt damit
40 % vor einer Maeander-Abtastung" ist eine. Drei Bezugsgroessen:

* **Orakel** — plant gegen die *wahre* Dichte. Das ist die Decke: was waere
  erreichbar, wenn es gar nichts zu erkunden gaebe? Der Abstand dorthin ist der
  Preis der Unwissenheit, und genau den soll Erkundung klein halten.
* **Maeander** — deckt das Gebiet gleichmaessig ab, ohne jede Information zu
  nutzen. Die Untergrenze, die jede informationsgetriebene Methode schlagen
  muss, um ihren Aufwand zu rechtfertigen. Bemerkenswert oft tut sie das nicht.
* **Zufallsbahn** — glatte, aber ziellose Bewegung. Trennt "hat etwas gelernt"
  von "war irgendwo unterwegs".
"""

import torch


def oracle_path(truth, planner, n_particles=192, seed=0):
    """Plant gegen die wahre Dichte — die Decke des Erreichbaren."""
    from .acquisition import particles_from_density
    phi = truth.clamp(min=0.0)
    phi = phi / phi.sum().clamp(min=1e-12)
    cps = planner.plan(particles_from_density(phi, n_particles), 1)
    return planner.render(cps)[0]


def lanes_for_length(target, margin=0.06):
    """Bahnenzahl, mit der ein Maeander ungefaehr `target` lang wird.

    Ein Maeander mit L Bahnen ueber ein Quadrat der Kantenlaenge w hat rund
    L*w Laenge quer plus (L-1)*w/(L-1) = w laengs. Nach L aufgeloest, damit der
    Maeander dasselbe Wegbudget bekommt wie die geplanten Varianten — sonst
    vergleicht man ihn bei einer anderen Laenge als alle anderen.
    """
    w = 1.0 - 2 * margin
    return max(2, int(round((target - w) / max(w, 1e-9))))


def lawnmower_path(n_points=128, n_lanes=6, margin=0.06, target_length=None):
    """Maeander ueber das Einheitsquadrat.

    Die klassische Abdeckungsstrategie, die keinerlei Information benutzt. Wenn
    eine informationsgetriebene Variante sie bei gleichem Budget nicht schlaegt,
    hat der ganze Aufwand keinen Ertrag — deshalb steht sie in jeder Tabelle.
    """
    if target_length is not None:
        n_lanes = lanes_for_length(target_length, margin)
    per = max(2, n_points // n_lanes)
    xs, ys = [], []
    lanes = torch.linspace(margin, 1 - margin, n_lanes)
    for i, y in enumerate(lanes):
        x = torch.linspace(margin, 1 - margin, per)
        if i % 2:
            x = x.flip(0)
        xs.append(x)
        ys.append(torch.full((per,), float(y)))
    return torch.stack([torch.cat(xs), torch.cat(ys)], dim=-1)


def random_path(planner, seed=0, scale=0.28):
    """Glatte, aber ziellose Bahn ueber dieselbe B-Spline-Darstellung.

    Bewusst ueber die Spline-Basis statt als Irrfahrt: sonst vergleicht man
    nebenbei Glattheit statt Planungsguete.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    cps = (scale * torch.randn(1, planner.nxi, 2, generator=g) + 0.5).clamp(0.02, 0.98)
    return planner.render(cps)[0]
