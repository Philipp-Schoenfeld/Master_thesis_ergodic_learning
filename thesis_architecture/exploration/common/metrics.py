r"""
metrics.py
==========
Bewertung einer Mission gegen die Wahrheit, die nur der Auswerter kennt.

Drei Groessen, die verschiedene Fragen beantworten und deshalb alle drei
gebraucht werden:

* **coverage_vs_truth** — wie gut deckt die Bahn die *wahre* Dichte ab? Das ist
  die eigentliche Aufgabe, und der Planer hat sie nie gesehen.
* **information_gain** — wie viel Unsicherheit wurde aufgeloest? Eine Bahn kann
  die Wahrheit gut abdecken und trotzdem wenig gelernt haben (Glueck), oder viel
  lernen und schlecht abdecken (reine Erkundung).
* **belief_rmse** — wie nah ist der Glaube am Ende an der Wahrheit? Der
  Zwischenwert, der zeigt, ob das Gelernte auch stimmt.

Eine Mission ist nur dann gut, wenn alle drei stimmen. Ein einzelner Skalar
verdeckt genau den Zielkonflikt, um den es geht.
"""

import torch


def coverage_vs_truth(curve, truth, eps=1e-12):
    """Dichtegewichteter mittlerer Abstand zur naechsten Bahnstelle.

    curve: (T,2), truth: (R,R) -> Skalar (kleiner ist besser)

    Basisfrei und ohne Modenabschneidung — dasselbe Mass wie
    `coverage_distance` in der 2D-Auswertung, damit die Zahlen hier an die
    bestehende Reihe anschliessen.
    """
    # Auch der dtype wird angeglichen: Bahnen, die aus exportierten Listen
    # rekonstruiert werden, kommen als float64 zurueck, waehrend das Gitter
    # float32 ist — cdist verweigert die Mischung.
    curve = curve.to(torch.float32)
    truth = truth.to(device=curve.device, dtype=torch.float32)
    R = truth.shape[-1]
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, R, device=curve.device),
        torch.linspace(0, 1, R, device=curve.device), indexing='ij')
    cells = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    w = truth.reshape(-1).clamp(min=0.0)
    d = torch.cdist(cells, curve).min(dim=1).values
    return (d * w).sum() / w.sum().clamp(min=eps)


def information_gain(belief_before, belief_after):
    """Abgebaute Gesamtunsicherheit (groesser ist besser)."""
    return float(belief_before.total_uncertainty() -
                 belief_after.total_uncertainty())


def belief_rmse(belief, truth):
    """RMSE des Posterior-Mittelwerts gegen die Wahrheit auf dem Gitter.

    Wie in `coverage_vs_truth` wird das Wahrheitsgitter hier auf Geraet und
    dtype des Glaubens gebracht. Der Glaube liegt auf der GPU, sobald ein
    Netz-Planer im Spiel ist, die Wahrheit kommt vom Datenlader auf der CPU —
    und `interpolate` gibt sie unveraendert auf der CPU zurueck. Ohne diese
    Zeile bricht die Auswertung erst nach der ersten vollstaendigen Mission ab,
    also spaet genug, dass ein Cluster-Job dafuer schon eine GPU belegt hat.
    """
    mu, _ = belief.posterior_grid()
    t = truth.to(device=mu.device, dtype=mu.dtype)
    if t.shape != mu.shape:
        t = torch.nn.functional.interpolate(
            t[None, None], size=mu.shape, mode='bilinear',
            align_corners=True)[0, 0]
    return float(((mu - t) ** 2).mean().sqrt())


def path_length(curve):
    return float((curve[1:] - curve[:-1]).norm(dim=-1).sum())


def summarise(name, curve, truth, b0, b1, extra=None):
    """Ein Ergebnisdatensatz pro Mission."""
    row = {
        'variant': name,
        'coverage': float(coverage_vs_truth(curve, truth)),
        'info_gain': information_gain(b0, b1),
        'belief_rmse': belief_rmse(b1, truth),
        'path_len': path_length(curve),
        'n_obs': int(b1.n_obs),
    }
    if extra:
        row.update(extra)
    return row


def print_table(rows, title=""):
    """Vergleichstabelle ueber Missionen."""
    if not rows:
        print("  (keine Ergebnisse)")
        return
    if title:
        print(f"\n  {title}")
    cols = ['variant', 'coverage', 'info_gain', 'belief_rmse', 'path_len', 'n_obs']
    extra = [k for k in rows[0] if k not in cols]
    cols += extra
    head = f"  {'Variante':<26}" + "".join(f"{c:>13}" for c in cols[1:])
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        line = f"  {str(r['variant']):<26}"
        for c in cols[1:]:
            v = r.get(c, '')
            line += f"{v:>13.4f}" if isinstance(v, float) else f"{str(v):>13}"
        print(line)


# ── Anytime-Auswertung ─────────────────────────────────────────────────────
def anytime_curve(path, truth, belief0, n_points=12, noise=0.02,
                  sensor_radius=0.03, max_obs=96):
    """Guete als Funktion der bereits gefahrenen Weglaenge.

    Der entscheidende Baustein fuer einen fairen Vergleich. Die Varianten
    erzeugen unterschiedlich lange Bahnen — B faehrt rund dreimal so weit wie
    D —, und bei freier Weglaenge misst "weiter fahren" fast immer besser. Ein
    einzelner Endwert vergleicht dann Budgets statt Verfahren.

    Stattdessen wird die Bahn an wachsenden Praefixen ausgewertet. Das ergibt
    pro Variante eine Kurve, und zwei Kurven lassen sich bei *gleicher*
    Weglaenge vergleichen — oder man liest ab, welche Variante bei jedem Budget
    vorn liegt.

    -> Liste von dicts mit path_len, coverage, info_gain, belief_rmse
    """
    from .observation import measure, thin

    seg = (path[1:] - path[:-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros(1, device=path.device), seg.cumsum(0)])
    total = float(cum[-1])

    out = []
    for frac in torch.linspace(1.0 / n_points, 1.0, n_points):
        target = float(frac) * total
        # Suchwert auf dem Geraet von `cum` bilden: `searchsorted` verweigert
        # eine sortierte Folge auf der GPU mit einem Suchwert auf der CPU.
        tgt = torch.tensor(target, device=cum.device, dtype=cum.dtype)
        k = int(torch.searchsorted(cum, tgt).clamp(2, len(path)))
        prefix = path[:k]

        b = belief0.clone()
        pts, vals = measure(prefix, truth, noise_std=noise,
                            sensor_radius=sensor_radius)
        b.observe(*thin(pts, vals, max_points=max_obs))

        out.append({
            'path_len': float(cum[k - 1]),
            'coverage': float(coverage_vs_truth(prefix, truth)),
            'info_gain': information_gain(belief0, b),
            'belief_rmse': belief_rmse(b, truth),
        })
    return out


def at_budget(curve, budget, key='coverage'):
    """Wert einer Anytime-Kurve bei fester Weglaenge, linear interpoliert.

    Gibt None zurueck, wenn die Bahn das Budget gar nicht ausschoepft — dann
    ist die Variante bei diesem Budget schlicht nicht vergleichbar, und das
    soll sichtbar bleiben statt durch den Endwert ersetzt zu werden.
    """
    xs = [c['path_len'] for c in curve]
    ys = [c[key] for c in curve]
    if budget < xs[0]:
        return ys[0]
    if budget > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= budget:
            t = (budget - xs[i - 1]) / max(xs[i] - xs[i - 1], 1e-12)
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]
