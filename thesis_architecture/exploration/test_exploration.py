#!/usr/bin/env python3
r"""
test_exploration.py
===================
Pruefungen fuer den Explorations-Unterbau.

Schwerpunkt liegt auf den Eigenschaften, die stillschweigend brechen koennen und
dann Zahlen liefern, die nach einem Ergebnis aussehen:

* Haengt die Posterior-Varianz wirklich nur von den Messorten ab? Auf dieser
  Eigenschaft steht Variante B vollstaendig.
* Ist die Vorausschau konsistent mit dem tatsaechlichen Messprozess? Ein erster
  Lauf war um 40 % zu pessimistisch, weil der Sensorring fehlte.
* Faellt die Unsicherheit dort, wo gemessen wurde, und nur dort?
* Ist die Zieldichte bei flachem Prior entartet — und wird das gemeldet?
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.acquisition import (ucb_density, particles_from_density,  # noqa: E402
                                is_degenerate, kappa_schedule)
from common.belief import GPBelief                                    # noqa: E402
from common.observation import measure, sample_field, thin            # noqa: E402
from common import metrics                                            # noqa: E402

torch.manual_seed(0)
PASS, FAIL = "[ok]  ", "[FAIL]"
_fail = []


def check(name, ok, detail=""):
    print(f"{PASS if ok else FAIL} {name:<56} {detail}")
    if not ok:
        _fail.append(name)


def test_belief():
    b = GPBelief(grid_res=16, lengthscale=0.15)
    mu, sd = b.posterior_grid()
    check("leerer Glaube: mu = 0, sigma = sqrt(var)",
          float(mu.abs().max()) == 0 and abs(float(sd.mean()) - 1.0) < 1e-5)

    b.observe(torch.tensor([[0.5, 0.5]]), torch.tensor([1.0]))
    mu, sd = b.posterior_grid()
    c = sd.shape[0] // 2
    check("Unsicherheit faellt am Messort am staerksten",
          float(sd[c, c]) < float(sd[0, 0]),
          f"{float(sd[c,c]):.3f} vs Ecke {float(sd[0,0]):.3f}")
    check("Mittelwert zieht zur Messung", float(mu[c, c]) > 0.5,
          f"{float(mu[c,c]):.3f}")

    b2 = b.clone().observe(torch.tensor([[0.1, 0.1]]), torch.tensor([0.0]))
    check("clone() laesst das Original unberuehrt", b.n_obs == 1 and b2.n_obs == 2)


def test_variance_independent_of_values():
    """Die Eigenschaft, auf der Variante B steht."""
    pts = torch.rand(12, 2)
    b1 = GPBelief(grid_res=16).observe(pts, torch.rand(12))
    b2 = GPBelief(grid_res=16).observe(pts, torch.rand(12) * 100.0)
    _, s1 = b1.posterior_grid()
    _, s2 = b2.posterior_grid()
    check("Posterior-Varianz haengt nur von den ORTEN ab",
          bool(torch.allclose(s1, s2, atol=1e-6)),
          f"max |diff| = {float((s1-s2).abs().max()):.2e}")


def test_uncertainty_after():
    b = GPBelief(grid_res=16)
    pts = torch.rand(10, 2)
    pred = b.uncertainty_after(pts)
    after = float(b.clone().observe(pts, torch.rand(10)).total_uncertainty())
    check("uncertainty_after sagt das Ergebnis exakt voraus",
          abs(float(pred) - after) < 1e-3,
          f"{float(pred):.4f} vs {after:.4f}")

    p = torch.rand(8, 2, requires_grad=True)
    b.uncertainty_after(p).backward()
    check("uncertainty_after ist differenzierbar in den Orten",
          bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max()) > 0,
          f"|grad|max {float(p.grad.abs().max()):.2e}")

    many = b.uncertainty_after(torch.rand(60, 2))
    few = b.uncertainty_after(torch.rand(6, 2))
    check("mehr Messorte loesen mehr Unsicherheit auf",
          float(many) < float(few), f"{float(many):.1f} < {float(few):.1f}")


def test_observation():
    truth = torch.zeros(32, 32)
    truth[8:24, 8:24] = 1.0
    inside = torch.tensor([[0.5, 0.5]])
    outside = torch.tensor([[0.02, 0.02]])
    check("sample_field liest das Feld an der richtigen Stelle",
          float(sample_field(truth, inside)) > 0.9 and
          float(sample_field(truth, outside)) < 0.1,
          f"innen {float(sample_field(truth, inside)):.2f}, "
          f"aussen {float(sample_field(truth, outside)):.2f}")

    p = torch.rand(4, 2, requires_grad=True)
    sample_field(truth, p).sum().backward()
    check("sample_field ist differenzierbar im Ort",
          bool(torch.isfinite(p.grad).all()))

    curve = torch.rand(20, 2)
    pts0, _ = measure(curve, truth, noise_std=0.0, sensor_radius=0.0)
    pts1, _ = measure(curve, truth, noise_std=0.0, sensor_radius=0.05, n_ring=4)
    check("Sensorradius erzeugt zusaetzliche Messpunkte",
          pts1.shape[0] == 5 * pts0.shape[0],
          f"{pts0.shape[0]} -> {pts1.shape[0]}")

    t, v = thin(torch.rand(300, 2), torch.rand(300), max_points=64)
    check("thin begrenzt die Messmenge", t.shape[0] == 64 and v.shape[0] == 64)


def test_lookahead_matches_reality():
    """Der Fehler, den der erste Lauf von Variante B hatte."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'variant_b_diffsim'))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'run_b', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'variant_b_diffsim', 'run_b.py'))
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)

    truth = torch.zeros(32, 32); truth[8:24, 8:24] = 1.0
    curve = torch.rand(48, 2)
    radius = 0.04

    b = GPBelief(grid_res=20)
    probe = rb._probe_points(curve, 24, sensor_radius=radius)
    pred = float(b.uncertainty_after(probe))

    b2 = b.clone()
    pts, vals = measure(curve, truth, noise_std=0.0, sensor_radius=radius)
    b2.observe(*thin(pts, vals, max_points=probe.shape[0]))
    real = float(b2.total_uncertainty())

    rel = abs(pred - real) / max(real, 1e-9)
    check("Vorausschau mit Sensorring trifft die Wirklichkeit",
          rel < 0.25, f"Abweichung {rel:.1%}")

    probe_no_ring = rb._probe_points(curve, 24, sensor_radius=0.0)
    check("ohne Sensorring ist die Vorausschau pessimistischer",
          float(b.uncertainty_after(probe_no_ring)) > pred,
          "genau der Fehler des ersten Laufs")


def test_acquisition():
    b = GPBelief(grid_res=20)
    mu, sd = b.posterior_grid()
    phi = ucb_density(mu, sd, kappa=2.0)
    check("Phi ist eine normierte Dichte",
          abs(float(phi.sum()) - 1.0) < 1e-5 and bool((phi >= 0).all()))
    check("flacher Prior wird als entartet erkannt", is_degenerate(phi),
          "kappa hat dort keine Wirkung")

    b.observe(torch.tensor([[0.5, 0.5]]), torch.tensor([1.0]))
    mu, sd = b.posterior_grid()
    check("mit Struktur nicht mehr entartet",
          not is_degenerate(ucb_density(mu, sd, kappa=2.0)))

    hi = ucb_density(mu, sd, kappa=8.0)
    lo = ucb_density(mu, sd, kappa=0.0)
    c = sd.shape[0] // 2
    check("kleines kappa gewichtet den bekannten Ort staerker",
          float(lo[c, c]) > float(hi[c, c]),
          f"kappa=0: {float(lo[c,c]):.4f} vs kappa=8: {float(hi[c,c]):.4f}")

    p = particles_from_density(phi, 128)
    check("Partikel haben (x, y, mu) und liegen in [0,1]",
          p.shape == (128, 3) and float(p[:, :2].min()) >= 0
          and float(p[:, :2].max()) <= 1)

    ks = [kappa_schedule(i, 5, 3.0, 0.3) for i in range(5)]
    check("kappa faellt monoton ueber die Mission",
          all(ks[i] > ks[i + 1] for i in range(4)),
          f"{ks[0]:.1f} -> {ks[-1]:.1f}")


def test_metrics():
    truth = torch.zeros(24, 24); truth[8:16, 8:16] = 1.0
    good = torch.stack([torch.linspace(.35, .65, 40),
                        torch.linspace(.35, .65, 40)], dim=-1)
    bad = torch.stack([torch.linspace(.90, .98, 40),
                       torch.linspace(.90, .98, 40)], dim=-1)
    check("coverage_vs_truth belohnt die Bahn auf der Dichte",
          float(metrics.coverage_vs_truth(good, truth)) <
          float(metrics.coverage_vs_truth(bad, truth)),
          f"{float(metrics.coverage_vs_truth(good, truth)):.4f} vs "
          f"{float(metrics.coverage_vs_truth(bad, truth)):.4f}")

    b0 = GPBelief(grid_res=16)
    b1 = b0.clone().observe(torch.rand(20, 2), torch.rand(20))
    check("information_gain ist positiv, wenn gemessen wurde",
          metrics.information_gain(b0, b1) > 0,
          f"{metrics.information_gain(b0, b1):.1f}")


if __name__ == '__main__':
    print("\n=== Explorations-Unterbau ===\n")
    test_belief()
    test_variance_independent_of_values()
    test_uncertainty_after()
    test_observation()
    test_lookahead_matches_reality()
    test_acquisition()
    test_metrics()
    print()
    if _fail:
        print(f"{len(_fail)} Pruefung(en) fehlgeschlagen: {_fail}")
        sys.exit(1)
    print("Alle Pruefungen bestanden.")
