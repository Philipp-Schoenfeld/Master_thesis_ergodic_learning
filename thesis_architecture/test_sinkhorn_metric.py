#!/usr/bin/env python3
r"""
test_sinkhorn_metric.py
=======================
Checks for the Sinkhorn option in `ergodic_metric.ErgodicLoss`, plus the one
number that decides how to use it: the scale ratio between the two measures on
real data.

That ratio matters because `--lambda_erg` was tuned for the Fourier term over
the ablation w = 2..400. If Sinkhorn is an order of magnitude smaller, the same
lambda is an order of magnitude weaker, and a head-to-head run at "the same
weight" would compare nothing of the sort. The last check prints the factor to
carry over.
"""

import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (_here, os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    _p = os.path.normpath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

torch.manual_seed(0)
np.random.seed(0)

PASS, FAIL = "[ok]  ", "[FAIL]"
_failures = []


def check(name, ok, detail=""):
    print(f"{PASS if ok else FAIL} {name:<54} {detail}")
    if not ok:
        _failures.append(name)


def _particles_from_curve(curve, n=256, spread=0.02, seed=0):
    """A target cloud that the given curve covers well, for a 'good' case."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, curve.shape[1], (curve.shape[0], n), generator=g)
    pts = torch.gather(curve, 1, idx[..., None].expand(-1, -1, 2))
    pts = pts + spread * torch.randn(pts.shape, generator=g)
    mu = torch.ones(curve.shape[0], n, 1)
    return torch.cat([pts.clamp(0, 1), mu], dim=-1)


def test_defaults_unchanged():
    """The whole point of the flag: without it, nothing moves."""
    from ergodic_metric import ErgodicLoss

    L_old = ErgodicLoss(nxi=25, K=8, pts=64, weight=1.0)
    check("default metric is still fourier", L_old.metric == 'fourier',
          L_old.extra_repr())

    cps = torch.rand(4, 25, 2)
    parts = torch.rand(4, 64, 3)
    t = torch.rand(4)
    a = L_old(cps, parts, t)

    L_new = ErgodicLoss(nxi=25, K=8, pts=64, weight=1.0, metric='fourier')
    b = L_new(cps, parts, t)
    check("explicit metric='fourier' equals the old path",
          torch.allclose(a, b), f"{a.item():.6e} vs {b.item():.6e}")

    # return_parts must not change the value it returns.
    c, p = L_new(cps, parts, t, return_parts=True)
    check("return_parts leaves the value alone", torch.allclose(a, c),
          f"parts={sorted(p)}")


def test_sinkhorn_basics():
    from ergodic_metric import ErgodicLoss, sinkhorn_error
    from geomloss import SamplesLoss

    L = ErgodicLoss(nxi=25, K=8, pts=128, weight=1.0, metric='sinkhorn')
    check("sinkhorn loss builds", L._sinkhorn is not None, L.extra_repr())

    # A cloud drawn from the curve itself must score far lower than an
    # unrelated one — the basic sanity of a discrepancy measure.
    cps = torch.rand(3, 25, 2)
    curve = L.render(cps)
    near = _particles_from_curve(curve, spread=0.01)
    far = torch.cat([torch.rand(3, 256, 2), torch.ones(3, 256, 1)], dim=-1)

    e_near = L.coverage_error(cps, near)
    e_far = L.coverage_error(cps, far)
    check("sinkhorn: matching cloud scores below a random one",
          bool((e_near < e_far).all()),
          f"near {e_near.mean():.4e} vs far {e_far.mean():.4e}")
    check("sinkhorn error is non-negative", bool((e_near >= -1e-6).all()),
          f"min {e_near.min():.2e}")

    # Debiasing: a cloud equal to the path itself gives (near) zero.
    sk = SamplesLoss('sinkhorn', p=2, blur=0.05, scaling=0.9, debias=True,
                     backend='tensorized')
    same = torch.cat([curve, torch.ones(3, curve.shape[1], 1)], dim=-1)
    e_same = sinkhorn_error(curve, same, sk)
    check("debiased sinkhorn is ~0 for identical clouds",
          float(e_same.abs().max()) < 1e-6, f"max {e_same.abs().max():.2e}")


def test_mu_weighting():
    """mu must actually steer the target, or uniform sampling is being ignored."""
    from ergodic_metric import sinkhorn_error
    from geomloss import SamplesLoss

    sk = SamplesLoss('sinkhorn', p=2, blur=0.05, scaling=0.9, debias=True,
                     backend='tensorized')
    # Curve sitting in the left half of the square.
    t = torch.linspace(0, 1, 100)
    curve = torch.stack([0.25 + 0.05 * torch.cos(6 * t), 0.2 + 0.6 * t], -1)[None]

    # Particles spread over both halves, but mu concentrated on the left.
    xs = torch.cat([torch.rand(1, 128, 1) * 0.5,
                    0.5 + torch.rand(1, 128, 1) * 0.5], dim=1)
    ys = torch.rand(1, 256, 1)
    mu_left = torch.cat([torch.ones(1, 128, 1), 0.01 * torch.ones(1, 128, 1)], 1)
    mu_flat = torch.ones(1, 256, 1)

    p_left = torch.cat([xs, ys, mu_left], dim=-1)
    p_flat = torch.cat([xs, ys, mu_flat], dim=-1)

    e_left = sinkhorn_error(curve, p_left, sk, weighted=True)
    e_flat = sinkhorn_error(curve, p_flat, sk, weighted=True)
    check("mu weighting changes the target (left-heavy scores lower)",
          float(e_left) < float(e_flat),
          f"mu-left {float(e_left):.4e} vs mu-flat {float(e_flat):.4e}")

    e_unw = sinkhorn_error(curve, p_left, sk, weighted=False)
    check("weighted=False ignores mu", abs(float(e_unw) - float(e_flat)) < 1e-7,
          f"{float(e_unw):.4e} vs {float(e_flat):.4e}")


def test_gradients():
    """The term is only useful if it pushes the control points."""
    from ergodic_metric import ErgodicLoss

    for metric in ('fourier', 'sinkhorn', 'both'):
        L = ErgodicLoss(nxi=25, K=8, pts=128, weight=1.0, metric=metric)
        cps = torch.rand(2, 25, 2, requires_grad=True)
        parts = torch.cat([torch.rand(2, 128, 2), torch.rand(2, 128, 1)], -1)
        err = L.coverage_error(cps, parts).sum()
        err.backward()
        g = cps.grad
        ok = g is not None and torch.isfinite(g).all() and g.abs().max() > 0
        check(f"{metric}: gradient reaches the control points and is finite",
              bool(ok), f"|grad|_max = {g.abs().max():.3e}")

    # One descent step must lower the Sinkhorn error — the property the whole
    # term depends on.
    L = ErgodicLoss(nxi=25, K=8, pts=128, weight=1.0, metric='sinkhorn')
    cps = torch.rand(4, 25, 2, requires_grad=True)
    parts = torch.cat([torch.rand(4, 200, 2), torch.rand(4, 200, 1)], -1)
    e0 = L.coverage_error(cps, parts)
    e0.sum().backward()
    with torch.no_grad():
        stepped = cps - 0.5 * cps.grad
    e1 = L.coverage_error(stepped, parts)
    check("a descent step lowers the sinkhorn error",
          bool((e1 < e0).all()), f"{e0.mean():.4e} -> {e1.mean():.4e}")


def test_scale_on_real_data():
    """The number needed to translate lambda_erg between the two measures."""
    import sqlite3, json
    db = os.path.join(_here, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')
    if not os.path.isfile(db):
        print("      (Datenbank nicht gefunden, Skalenvergleich uebersprungen)")
        return

    sys.path.append(os.path.join(_here, 'ergodic_dataset_generator'))
    from shape_library import pdf_on_grid
    from flow_matching_runner_particles import sample_particles
    from ergodic_metric import ErgodicLoss

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT trajectory, density_params FROM ergodic_pairs "
                        "WHERE split='val' LIMIT 12").fetchall()
    conn.close()

    cps_list, grids = [], []
    for blob, params in rows:
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
        idx = np.linspace(0, len(xy) - 1, 25).astype(int)
        cps_list.append(xy[idx])
        d, _, _ = pdf_on_grid(json.loads(params), resolution=64)
        d = np.asarray(d, dtype=np.float32)
        grids.append(d / max(d.max(), 1e-12))

    cps = torch.tensor(np.stack(cps_list), dtype=torch.float32)
    grid_t = torch.tensor(np.stack(grids))
    parts = sample_particles(grid_t, torch.arange(len(rows)), 256, 'cpu',
                             mode='uniform')

    L = ErgodicLoss(nxi=25, K=8, pts=128, weight=1.0, metric='both')
    _, p = L.coverage_error(cps, parts, return_parts=True)
    f, s = p['fourier'].mean().item(), p['sinkhorn'].mean().item()

    # The value ratio is informative, but training is driven by the gradient, so
    # that is the number to carry over into lambda_erg.
    grads = {}
    for m in ('fourier', 'sinkhorn'):
        Lm = ErgodicLoss(nxi=25, K=8, pts=128, weight=1.0, metric=m)
        c = cps.clone().requires_grad_(True)
        Lm.coverage_error(c, parts).sum().backward()
        grads[m] = c.grad.norm().item()

    r_val = f / max(s, 1e-12)
    r_grad = grads['fourier'] / max(grads['sinkhorn'], 1e-12)

    print()
    print(f"      Solver-Trajektorien, {len(rows)} Holdout-Formen:")
    print(f"        Fourier  (K=8):   Wert {f:.5e}   |grad| {grads['fourier']:.5e}")
    print(f"        Sinkhorn (0.05):  Wert {s:.5e}   |grad| {grads['sinkhorn']:.5e}")
    print(f"        Verhaeltnis Fourier/Sinkhorn:  Wert {r_val:.2f}   Gradient {r_grad:.2f}")
    print(f"        -> fuer gleiche Gradientenstaerke lambda_erg beim Sinkhorn")
    print(f"           etwa {r_grad:.0f}x GROESSER waehlen als beim Fourier,")
    print(f"           also w={300 * r_grad:.0f} als Gegenstueck zu w=300.")
    print()
    check("beide Masse sind auf echten Daten positiv und endlich",
          f > 0 and s > 0 and np.isfinite(f) and np.isfinite(s))
    check("Gradientenverhaeltnis ist endlich und positiv",
          np.isfinite(r_grad) and r_grad > 0, f"{r_grad:.2f}")


if __name__ == '__main__':
    print("\n=== Sinkhorn-Option in ErgodicLoss ===\n")
    test_defaults_unchanged()
    test_sinkhorn_basics()
    test_mu_weighting()
    test_gradients()
    test_scale_on_real_data()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {_failures}")
        sys.exit(1)
    print("Alle Pruefungen bestanden.")
