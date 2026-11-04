#!/usr/bin/env python3
"""Checks for the DB improvement tools (surface_metrics, refine_db3d, keep_path).

Synthetic planar scene, CPU only: runs locally and as the gate of
run_job_db_refine.bash.

    python test_db_refine.py
"""
import os
import sqlite3
import sys
import tempfile
import types

import numpy as np
import torch

from data_surfaces import load_surface_db
from obstacles import bspline_basis_matrix
from refine_db3d import Refiner, apply_filter, open_dst, report, write_batch
from surface_metrics import standoff_pointing

FAILED = []


def check(name, ok, info=''):
    print(f"[{'ok' if ok else 'FAIL'}]   {name:<62} {info}")
    if not ok:
        FAILED.append(name)


def scene(B=6, N=512, seed=0):
    """Particles on the plane z = 0.5 with a Gaussian mu; teacher splines on the plane, axis -z."""
    g = np.random.default_rng(seed)
    xy = g.uniform(0.15, 0.85, (B, N, 2))
    mu = np.exp(-((xy - 0.5) ** 2).sum(-1) / 0.04)
    parts = np.concatenate([xy, np.full((B, N, 1), 0.5), mu[..., None]], -1).astype(np.float32)
    cps = np.concatenate([g.uniform(0.3, 0.7, (B, 25, 2)), np.full((B, 25, 1), 0.5)], -1)
    # R = [x, y, z] with z = (0, 0, -1) looking down on the plane; rot6 = first two columns.
    rot6 = np.tile(np.array([1., 0., 0., 0., -1., 0.]), (B, 25, 1))
    return (torch.tensor(np.concatenate([cps, rot6], -1), dtype=torch.float32),
            torch.from_numpy(parts))


def test_metrics():
    print("\n--- surface_metrics ---\n")
    _, parts = scene()
    q = torch.tensor([0.4, 0.5, 0.62]).expand(6, 10, 3).contiguous()
    axis = torch.tensor([0., 0., -1.]).expand(6, 10, 3)
    d, ang = standoff_pointing(q, axis, parts)
    check("plane distance 0.12 above the plane", (d - 0.12).abs().max().item() < 1e-4,
          f"{d.mean().item():.4f}")
    check("axis along the normal -> 0 deg", ang.max().item() < 0.5, f"{ang.max().item():.2f}")
    d0, ang0 = standoff_pointing(q * torch.tensor([1., 1., 0.]) + torch.tensor([0., 0., 0.5]),
                                 axis, parts)
    check("on the plane: distance ~0, still 0 deg (nearest-particle bias gone)",
          d0.max().item() < 1e-4 and ang0.max().item() < 0.5)
    tilt = torch.tensor([1., 0., -1.]).expand(6, 10, 3)
    _, a45 = standoff_pointing(q, tilt, parts)
    check("axis tilted 45 deg -> 45 deg", (a45 - 45).abs().max().item() < 0.5)


def test_refiner():
    print("\n--- Refiner ---\n")
    x1, parts = scene()
    ref = Refiner(torch.device('cpu'))
    x_off = ref.offset(x1)
    check("offset moves 0.12 along the normal (here +z)",
          torch.allclose(x_off[..., 2], torch.full_like(x_off[..., 2], 0.62), atol=1e-6))
    m0 = ref.metrics(x_off, parts)
    torch.manual_seed(0)
    x_r, loss = ref.run(x_off, parts, iters=60, lr=2e-3)
    m1 = ref.metrics(x_r, parts)
    with torch.no_grad():
        acc0, len0 = ref.shape_stats(x_off[..., :3])
        l0, _ = ref.objective(x_off, parts, (acc0.clamp(min=1e-6), len0.clamp(min=1e-3)))
    check("objective decreases for every entry", (loss <= l0 + 1e-6).all().item(),
          f"mean {l0.mean().item():.3f} -> {loss.mean().item():.3f}")
    check("Sinkhorn decreases on average", m1['erg'].mean() < m0['erg'].mean(),
          f"{m0['erg'].mean().item():.2e} -> {m1['erg'].mean().item():.2e}")
    check("start control point held", torch.equal(x_r[:, 0, :3], x_off[:, 0, :3]))
    check("all finite", torch.isfinite(x_r).all().item())
    return x1, parts, x_off, x_r, m0, m1, loss, ref


def test_db_roundtrip(x1, parts, x_off, x_r, m0, m1, loss, ref):
    print("\n--- destination DB round trip ---\n")
    from project_db_3d import SCHEMA, EINFUEGEN
    tmp = tempfile.mkdtemp()
    src, dst = os.path.join(tmp, 'src.db'), os.path.join(tmp, 'dst.db')
    con = sqlite3.connect(src)
    con.executescript(SCHEMA)
    Bs = bspline_basis_matrix(25, 201, 5)
    splits = ['train', 'train', 'train', 'val_form', 'val_flaeche', 'val_beides']
    for i in range(6):
        pos = (Bs @ x1[i, :, :3].numpy()).astype(np.float32)
        rot = (Bs @ x1[i, :, 3:].numpy()).astype(np.float32)
        con.execute(EINFUEGEN, (f's{i}', splits[i], 'ebene_flach', 'ebene', 0.0, '',
                                pos.tobytes(), rot.tobytes(), parts[i].numpy().tobytes(),
                                pos[0].tobytes(), '[0,0]', 0., 0., 1., 0, 201, 512,
                                0., 0., 0., 0., 0., 1.))
    con.commit(); con.close()

    entries = load_surface_db(src, target_cps=True, keep_path=True)
    check("keep_path adds the stored path", entries[0]['path_pos'].shape == (201, 3)
          and entries[0]['path_rot6'].shape == (201, 6))
    c = open_dst(src, dst)
    np_ = lambda d: {k: v.numpy() for k, v in d.items()}
    write_batch(c, entries, x_r, np_(m0), np_(m0), np_(m1), loss.numpy(),
                Bs.astype(np.float64), 0.12)
    a = types.SimpleNamespace(filter_pct=50.0, standoff=0.12, max_so_dev=1.0)
    apply_filter(c, a)
    c.close()
    out = load_surface_db(dst, target_cps=True)
    by = {e['name']: e for e in out}
    check("filter drops train entries only (p50 of 3 train -> 1 dropped)",
          sum(e['split'] == 'train' for e in out) == 2 and len(out) == 5, f"{len(out)} left")
    err = max(np.abs(by[n]['x1'] - x_r[int(n[1:])].numpy()).max() for n in by)
    check("loader recovers the refined control points exactly", err < 1e-3, f"max |d| {err:.1e}")
    check("start column = refined cp0",
          all(np.allclose(by[n]['start'], x_r[int(n[1:]), 0, :3].numpy(), atol=1e-6) for n in by))
    report(dst, os.path.join(tmp, 'rep'), [])
    check("report written", os.path.exists(os.path.join(tmp, 'rep', 'refine_histograms.png')))


if __name__ == '__main__':
    test_metrics()
    args = test_refiner()
    test_db_roundtrip(*args)
    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) FAILED: {FAILED}")
        sys.exit(1)
    print("All checks passed.")
