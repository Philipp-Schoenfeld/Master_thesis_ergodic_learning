r"""
generate_dataset_adam_svgd.py
==============================
Training database solved with Adam-SVGD, stored as B-spline control points
(Philipp's request, 2026-10-24).

Solve space (`--solve_space`):

* ``waypoints`` (default): Adam-SVGD on T = `--T` (128) raw waypoints, then a
  least-squares projection onto a B-spline with `--nxi` control points at the
  same time parameter, control point 0 fixed to x0. Measured on 48 densities
  (`study_waypoint_projection.py`): at nxi 40 smoother (bending energy 443 vs
  699) AND at least as ergodic (K=10 error 2.66e-4 vs 2.83e-4) as the best
  direct B-spline solve (nxi 50); the projection reproduces the waypoint curve
  to ~6e-4, i.e. the gain comes from solving in waypoint space.
* ``bspline``: Adam-SVGD directly on the nxi control points.

The stored `ergodic_error` is always that of the stored B-spline (128-point
render, K=8); in waypoint mode the raw solve's error, the raw waypoints and
the projection error are stored as well.

Densities: the 1172 base shapes of `ergodic_dataset_improved_final.db` plus
ONE large-area variant per shape -- the existing `#unk0` row of the same DB
(the base density overlaid with a random high-density area, see
`unknown_region.py`), reused byte-for-byte so the new DB targets exactly the
same distributions as before. `#unk1` is dropped. Split and start point x0
are copied from the source row as well.

Start points: `--n_starts` (default 3) trajectories per density. Start 0 is
the source row's x0 (already uniform over the workspace, from the
`--x0_mode ueberall` generation) and keeps the source name; starts k >= 1 are
drawn uniformly from [X0_MARGIN, 1-X0_MARGIN]^2 (seeded per density and k)
and are stored as `<name>@s<k>`. Only with several starts per density does
the CFM see the SAME density with DIFFERENT starts -- with one start each, the
trajectory is predictable from the density alone and the start conditioning
can be ignored. Extra starts of holdout densities get split `val_start`
(kept for evaluating the start dependence, but not part of the runner's
`val` holdout set and its per-epoch visualisation).

Solver, per trajectory:

* init: a straight line from x0 through the workspace centre to the opposite
  edge of the margin box (`linear_init_curve`) -- no heuristic, no shape
  knowledge. Point 0 is x0 (and so is control point 0 of its exact
  least-squares B-spline fit in bspline mode -- clamped B-splines reproduce
  linear functions).
* `svgd_batched.BatchedSvgdTorch` -- the project's "Adam-SVGD" (same solver
  as `adam_svgd` in `evaluation_full_matrix/run_init_comparison.py`): an
  8-particle swarm per density, median-heuristic RBF kernel, Adam on the
  energy W_ERGODIC * ergodic(K=8) + W_SMOOTH * accel^2 + W_BOUNDARY * box,
  best-energy particle as the result. All densities of a batch run as one
  GPU batch.
* the start is a HARD pin: control point 0 is x0 in every particle and its
  update is zeroed (a projection), instead of the solver's soft W_START term,
  so every stored trajectory starts exactly at x0 -- the condition the
  start-conditioned CFM is trained on.
* stops when the ergodic error has plateaued (best-so-far improved by less
  than `--tol` relative over the last `--patience` iterations, checked every
  `--check_every`), after at least `--min_iters`, at the latest after
  `--max_iters` (10000).

Stored per row (schema of `generate_dataset_length.init_db` plus extra
columns): `trajectory` = the B-spline rendered at tsteps+1 = 401 points (so
every old consumer that subsamples waypoints keeps working), and the exact
`control_points` (nxi, 2) float32 -- `flow_matching_runner_start.py
--target_cps` trains on those directly instead of on subsampled waypoints.

    # local smoke test (CPU, seconds)
    python generate_dataset_adam_svgd.py --out /tmp/x.db --limit 4 \
        --max_iters 300 --device cpu

    # full run (cluster, one GPU)
    python generate_dataset_adam_svgd.py --nxi 25 \
        --out ergodic_dataset_adam_svgd_nxi25.db
"""
import argparse
import json
import os
import signal
import sqlite3
import sys
import time
import zlib

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_arch, 'evaluation_full_matrix'),
           os.path.join(_arch, 'exploration'), os.path.join(_root, 'SE3_SVGD'),
           os.path.join(_root, 'src'), os.path.join(_root, 'bsplinax-main')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                        # noqa: E402

from generate_dataset_length import init_db                         # noqa: E402

SRC_DB = 'ergodic_dataset_improved_final.db'
N_POINTS = 128          # solver render resolution (project default, `run_svgd_convergence.N_POINTS`)
DEGREE = 5
TSTEPS_STORE = 400      # stored dense trajectory has tsteps+1 points, as in the source DB
INIT_MARGIN = 0.04      # same margin box as `init_baselines.linear_angle_path`
X0_MARGIN = 0.03        # extra starts: same range as the source x0 (`--x0_margin 0.03`)

EXTRA_COLUMNS = (
    ('control_points', 'BLOB'),
    ('nxi', 'INTEGER'),
    ('ergodic_error', 'REAL'),
    ('init_ergodic_error', 'REAL'),
    ('converged', 'INTEGER'),
    ('erg_history', 'BLOB'),
    ('solver', 'TEXT'),
    ('solver_config', 'TEXT'),
    ('raw_waypoints', 'BLOB'),
    ('raw_ergodic_error', 'REAL'),
    ('fit_rmse', 'REAL'),
)


# ── densities ────────────────────────────────────────────────────────────────

def load_source_rows(src_db, variants=('base', 'unk0')):
    """-> list of dict(name, split, density_params (str), x0 (2,), dt), in source id order."""
    conn = sqlite3.connect(src_db)
    rows = conn.execute("SELECT shape_name, split, density_params, x0, dt"
                        " FROM ergodic_pairs ORDER BY id ASC").fetchall()
    conn.close()
    out = []
    for name, split, dp, x0, dt in rows:
        kind = name.split('#', 1)[1] if '#' in name else 'base'
        if kind in variants:
            out.append(dict(name=name, split=split, density_params=dp,
                            x0=np.asarray(json.loads(x0), dtype=np.float64), dt=float(dt)))
    return out


def expand_starts(rows, n_starts, seed):
    """Start 0 = the source row itself; starts k >= 1 as `<name>@s<k>` with a
    uniform random x0. `density` names the source row the grid belongs to."""
    out = []
    for r in rows:
        out.append(dict(r, density=r['name']))
        for k in range(1, n_starts):
            rng = np.random.default_rng(zlib.crc32(f"{seed}:x0:{r['name']}:{k}".encode()))
            split = 'val_start' if r['split'] == 'val' else r['split']
            out.append(dict(r, name=f"{r['name']}@s{k}", split=split, density=r['name'],
                            x0=rng.uniform(X0_MARGIN, 1.0 - X0_MARGIN, size=2)))
    return out


def density_grid(density_params, res):
    from shape_library import pdf_on_grid
    d, _, _ = pdf_on_grid(json.loads(density_params), resolution=res)
    d = np.clip(np.asarray(d, dtype=np.float64), 0.0, None)
    return d / max(d.sum(), 1e-300)


def target_coeffs(grids, K):
    """(C, R, R) normalised density grids -> (C, K*K) phi_k, same k ordering as the solver."""
    from ergodic_energy_torch import make_k_grid, target_coeffs_from_grid
    k_idx, _ = make_k_grid(K)
    return target_coeffs_from_grid(torch.as_tensor(grids, dtype=torch.float64),
                                   torch.as_tensor(k_idx)).numpy()


# ── init ─────────────────────────────────────────────────────────────────────

def linear_init_curve(x0, n_points=N_POINTS, margin=INIT_MARGIN):
    """Straight line from x0 through the centre (0.5, 0.5) up to the opposite
    edge of [margin, 1-margin]^2. Falls back to the 45 deg diagonal if x0 sits
    on the centre."""
    x0 = np.asarray(x0, dtype=np.float64)
    d = np.array([0.5, 0.5]) - x0
    n = np.linalg.norm(d)
    d = d / n if n > 1e-3 else np.array([1.0, 1.0]) / np.sqrt(2.0)
    t_hits = [((1.0 - margin if d[i] > 0 else margin) - x0[i]) / d[i]
              for i in range(2) if abs(d[i]) > 1e-9]
    t_max = max(min(t_hits), 1e-3)
    t = np.linspace(0.0, t_max, n_points)
    return x0[None, :] + t[:, None] * d[None, :]


# ── solver ───────────────────────────────────────────────────────────────────

def ergodic_error(bs, X, phik):
    """Solver-consistent ergodic error 0.5 * sum Lambda (c - phi)^2 of dense curves X (C, T, 2)."""
    from svgd_batched import K_FREQ
    X = X.to(bs.cdt)
    cx, _ = bs._cos_sin(np.pi * X[..., 0])
    cy, _ = bs._cos_sin(np.pi * X[..., 1])
    c = torch.matmul(cx, cy.transpose(1, 2)).reshape(X.shape[0], K_FREQ * K_FREQ) / X.shape[1]
    return 0.5 * (bs.Lambda * (c - phik.to(bs.cdt)) ** 2).sum(-1)


def solve_pinned(bs, init_cps, phik, starts, seeds, max_iters=10000, check_every=50,
                 patience=1000, tol=0.01, min_iters=500, batch=4096, on_done=None,
                 log_every=0, lr=None, lr_half_life=None, lr_min=1e-4):
    """Adam-SVGD with a hard start pin and per-density early stopping.

    `BatchedSvgdTorch.run`'s update loop (its `energy_grad` and `_svgd_delta`
    are used as they are), with three changes: control point 0 is projected
    onto `starts` (jitter-free, zero update), each density stops on its own
    once its ergodic error plateaus, and finished densities leave the GPU batch
    while waiting ones join (refills happen at check boundaries, so every
    density's own iteration count stays a multiple of `check_every`; Adam's
    bias correction uses that per-density count).

    init_cps (C, nxi, 2), phik (C, K*K), starts (C, 2), seeds (C,) -> per
    density dict(cps (nxi, 2), n_iters, erg, erg_init, converged, history),
    also passed to `on_done(index, result)` as soon as it is final.

    `lr`: Adam step size (default `svgd_batched.ADAM_LR` = 2e-3).
    `lr_half_life`: optional exponential decay of the step size per density,
    lr * 0.5 ** (n / lr_half_life) with that density's own iteration count n,
    floored at `lr_min`. None = constant step size (previous behaviour).
    """
    from svgd_batched import (ADAM_B1, ADAM_B2, ADAM_EPS, ADAM_LR, CP_CLIP, N_PARTICLES,
                              UPDATE_CLAMP)
    assert max_iters % check_every == 0 and patience % check_every == 0
    lr0 = ADAM_LR if lr is None else float(lr)
    dev, dt = bs.dev, bs.dt
    C, nxi, _ = init_cps.shape
    N, D = N_PARTICLES, nxi * 2
    w = patience // check_every
    B = bs.B.to(dt)

    queue = list(range(C))
    act = []                          # density indices currently on the GPU
    P = m = v = tl = ph = st = None
    hist, best = {}, {}
    results = [None] * C

    def _join(idx):
        nonlocal P, m, v, tl, ph, st
        jit = np.stack([np.random.default_rng(int(seeds[i])).normal(scale=bs.jitter, size=(N, nxi, 2))
                        for i in idx])
        jit[:, :, 0, :] = 0.0
        P_new = torch.as_tensor(init_cps[idx][:, None] + jit, device=dev, dtype=dt)
        P_new[:, :, 0, :] = torch.as_tensor(starts[idx], device=dev, dtype=dt)[:, None]
        P_new = P_new.reshape(len(idx), N, D)
        z = torch.zeros_like(P_new)
        ph_new = torch.as_tensor(phik[idx], device=dev, dtype=dt)
        st_new = torch.as_tensor(starts[idx], device=dev, dtype=dt)
        tl_new = torch.zeros(len(idx), device=dev, dtype=dt)
        if P is None:
            P, m, v, tl, ph, st = P_new, z, z.clone(), tl_new, ph_new, st_new
        else:
            P, m, v = torch.cat([P, P_new]), torch.cat([m, z]), torch.cat([v, z.clone()])
            tl, ph, st = torch.cat([tl, tl_new]), torch.cat([ph, ph_new]), torch.cat([st, st_new])
        e0 = ergodic_error(bs, torch.matmul(B, torch.as_tensor(init_cps[idx], device=dev, dtype=dt)),
                           ph_new)
        for j, i in enumerate(idx):
            hist[i] = [float(e0[j])]
            best[i] = float(e0[j])
        act.extend(idx)

    def _refill():
        free = batch - len(act)
        if free > 0 and queue:
            take = queue[:free]
            del queue[:free]
            _join(take)

    _refill()
    it = 0
    t0 = time.perf_counter()
    while act:
        A = len(act)
        e, g = bs.energy_grad(P.reshape(A, N, nxi, 2), ph, None)
        delta = bs._svgd_delta(P, -g.reshape(A, N, D))
        delta[:, :, 0:2] = 0.0
        mx = delta.abs().amax(dim=(1, 2))
        delta = delta * torch.where(mx > UPDATE_CLAMP, UPDATE_CLAMP / torch.clamp(mx, min=1e-300),
                                    torch.ones_like(mx))[:, None, None]
        tl = tl + 1.0
        m = ADAM_B1 * m + (1 - ADAM_B1) * delta
        v = ADAM_B2 * v + (1 - ADAM_B2) * delta ** 2
        m_hat = m / (1 - ADAM_B1 ** tl)[:, None, None]
        v_hat = v / (1 - ADAM_B2 ** tl)[:, None, None]
        step = lr0
        if lr_half_life:
            step = torch.clamp(lr0 * 0.5 ** (tl / lr_half_life), min=lr_min)[:, None, None]
        P = torch.clamp(P + step * m_hat / (torch.sqrt(v_hat) + ADAM_EPS), *CP_CLIP)
        P[:, :, 0:2] = st[:, None, :]
        it += 1
        if it % check_every:
            continue

        e, _ = bs.energy_grad(P.reshape(A, N, nxi, 2), ph, None, want_grad=False)
        b = e.argmin(dim=1)
        cps_best = P[torch.arange(A, device=dev), b].reshape(A, nxi, 2)
        erg = ergodic_error(bs, torch.matmul(B, cps_best), ph).cpu().numpy()
        n_loc = tl.cpu().numpy().astype(int)
        keep = []
        for j, i in enumerate(act):
            hist[i].append(float(erg[j]))
            best[i] = min(best[i], float(erg[j]))
            run_min = np.minimum.accumulate(hist[i])
            plateau = (n_loc[j] >= min_iters and len(run_min) > w
                       and (run_min[-1 - w] - run_min[-1]) < tol * run_min[-1 - w])
            if plateau or n_loc[j] >= max_iters:
                results[i] = dict(cps=cps_best[j].detach().cpu().numpy().astype(np.float64),
                                  n_iters=int(n_loc[j]), erg=float(erg[j]), erg_init=hist[i][0],
                                  converged=bool(plateau),
                                  history=np.asarray(hist[i], dtype=np.float32))
                if on_done is not None:
                    on_done(i, results[i])
            else:
                keep.append(j)
        if len(keep) < A:
            kt = torch.as_tensor(keep, device=dev, dtype=torch.long)
            P, m, v, tl, ph, st = P[kt], m[kt], v[kt], tl[kt], ph[kt], st[kt]
            act[:] = [act[j] for j in keep]
        _refill()
        if log_every and it % log_every == 0:
            done = sum(r is not None for r in results)
            print(f"    it={it:6d}  active={len(act):5d}  done={done}/{C}  "
                  f"{time.perf_counter() - t0:7.1f}s", flush=True)
    return results


def make_solver(nxi, device, compute_dtype, w_smooth_scale=1.0):
    from obstacles import bspline_basis_matrix
    from svgd_batched import BatchedSvgdTorch, W_SMOOTH
    B32 = bspline_basis_matrix(nxi, N_POINTS, DEGREE)
    cdt = {'float64': torch.float64, 'float32': torch.float32}[compute_dtype]
    return BatchedSvgdTorch(B32, device=device, dtype=torch.float64, compute_dtype=cdt,
                            smoothness_weight=W_SMOOTH * w_smooth_scale), B32


def init_cps_for(B32, x0s):
    curves = np.stack([linear_init_curve(x0) for x0 in x0s])                     # (C, T, 2)
    C, T, _ = curves.shape
    B64 = B32.astype(np.float64)
    P0 = np.linalg.lstsq(B64, curves.transpose(1, 0, 2).reshape(T, C * 2), rcond=None)[0]
    P0 = P0.reshape(-1, C, 2).transpose(1, 0, 2)
    P0[:, 0, :] = np.asarray(x0s)
    return P0


def project_to_bspline(Y, x0, BT):
    """Least-squares B-spline fit of the waypoint curve Y (T, 2) at equal time
    parameter (BT: (T, nxi) basis), control point 0 fixed to x0."""
    rest = np.linalg.lstsq(BT[:, 1:], Y - BT[:, :1] @ x0[None, :], rcond=None)[0]
    return np.clip(np.vstack([x0[None, :], rest]), 0.0, 1.0)


def make_backend(solve_space, nxi, device, compute_dtype, T=N_POINTS, w_smooth_scale=1.0):
    """-> (bs, init_fn, finalize_fn) for `solve_pinned`.

    init_fn(x0s) gives the start parameters (C, D, 2): control points
    (bspline) or T waypoints. finalize_fn(res, x0, phik_row) turns one solve
    result into the stored B-spline: res['cps'] (nxi, 2) and res['erg'] = its
    ergodic error on the 128-point render; waypoint mode also sets 'raw'
    (T, 2), 'raw_erg' and 'fit_rmse' (mean distance projection <-> waypoints).

    Waypoint mode runs `BatchedSvgdTorch` with B = identity; for T != 128 the
    smoothness weight is scaled by (T/128)^3 (sum of squared second
    differences ~ T^-3), i.e. the same continuous objective. `w_smooth_scale`
    multiplies the smoothness weight in both modes (1.0 = W_SMOOTH = 15).
    """
    bs_spline, B32 = make_solver(nxi, 'cpu', 'float64')     # scores the stored B-spline
    if solve_space == 'bspline':
        bs, _ = make_solver(nxi, device, compute_dtype, w_smooth_scale)

        def finalize(res, x0, phik_row):
            res.update(raw=None, raw_erg=None, fit_rmse=None)
            return res
        return bs, (lambda x0s: init_cps_for(B32, x0s)), finalize

    from obstacles import bspline_basis_matrix
    from svgd_batched import BatchedSvgdTorch, W_SMOOTH
    cdt = {'float64': torch.float64, 'float32': torch.float32}[compute_dtype]
    bs = BatchedSvgdTorch(np.eye(T, dtype=np.float32), device=device, dtype=torch.float64,
                          compute_dtype=cdt,
                          smoothness_weight=W_SMOOTH * w_smooth_scale * (T / N_POINTS) ** 3)
    BT = bspline_basis_matrix(nxi, T, DEGREE).astype(np.float64)
    B64 = torch.as_tensor(B32, dtype=torch.float64)

    def init(x0s):
        return np.stack([linear_init_curve(x0, n_points=T) for x0 in x0s])

    def finalize(res, x0, phik_row):
        Y = res['cps']
        cps = project_to_bspline(Y, np.asarray(x0, dtype=np.float64), BT)
        erg = ergodic_error(bs_spline, torch.matmul(B64, torch.as_tensor(cps))[None],
                            torch.as_tensor(phik_row)[None])
        res.update(raw=Y, raw_erg=res['erg'], cps=cps, erg=float(erg[0]),
                   fit_rmse=float(np.linalg.norm(BT @ cps - Y, axis=1).mean()))
        return res
    return bs, init, finalize


# ── trajectory metrics (also used by study_nxi_adam_svgd.py) ─────────────────

def curve_metrics(cps, n_dense=2048):
    """Geometric smoothness of the B-spline of `cps` (nxi, 2), independent of the
    parameterisation: path length, bending energy int kappa^2 ds, 95th
    percentile of the curvature, and the solver's own smoothness term
    sum(accel^2) on the N_POINTS render."""
    from obstacles import bspline_basis_matrix
    nxi = cps.shape[0]
    X = bspline_basis_matrix(nxi, n_dense, DEGREE).astype(np.float64) @ cps
    h = 1.0 / (n_dense - 1)
    d1 = np.gradient(X, h, axis=0)
    d2 = np.gradient(d1, h, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    kappa = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-9) ** 3
    ok = speed > 1e-6 * max(speed.max(), 1e-12)
    ds = speed * h
    Xr = bspline_basis_matrix(nxi, N_POINTS, DEGREE).astype(np.float64) @ cps
    acc = Xr[2:] - 2 * Xr[1:-1] + Xr[:-2]
    return dict(path_len=float(ds.sum()),
                bending=float((kappa[ok] ** 2 * ds[ok]).sum()),
                kappa_p95=float(np.percentile(kappa[ok], 95)) if ok.any() else 0.0,
                accel_sq=float((acc ** 2).sum()))


# ── DB ───────────────────────────────────────────────────────────────────────

def open_out_db(path):
    conn = init_db(path)
    have = {r[1] for r in conn.execute("PRAGMA table_info(ergodic_pairs)")}
    for col, typ in EXTRA_COLUMNS:
        if col not in have:
            conn.execute(f"ALTER TABLE ergodic_pairs ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def save_row(conn, row, res, B_store, config):
    cps = res['cps']
    traj = (B_store.astype(np.float64) @ cps).astype(np.float32)
    length = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
    conn.execute(
        "INSERT INTO ergodic_pairs (shape_name, split, density_params, trajectory, x0, dt, tsteps,"
        " n_iters, length, generated_at, control_points, nxi, ergodic_error, init_ergodic_error,"
        " converged, erg_history, solver, solver_config, raw_waypoints, raw_ergodic_error, fit_rmse)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (row['name'], row['split'], row['density_params'], traj.tobytes(),
         json.dumps([float(v) for v in row['x0']]), row['dt'], TSTEPS_STORE, res['n_iters'],
         length, time.strftime('%Y-%m-%dT%H:%M:%S'),
         cps.astype(np.float32).tobytes(), int(cps.shape[0]), res['erg'], res['erg_init'],
         int(res['converged']), res['history'].tobytes(),
         config.get('solver_name', f"adam_svgd_pinned_{config.get('solve_space', 'bspline')}"),
         json.dumps(config),
         None if res.get('raw') is None else np.asarray(res['raw'], dtype=np.float32).tobytes(),
         res.get('raw_erg'), res.get('fit_rmse')))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src_db', default=SRC_DB)
    p.add_argument('--out', required=True)
    p.add_argument('--variants', default='base,unk0',
                   help="Comma-separated row kinds of the source DB: 'base' and/or 'unk0', 'unk1'.")
    p.add_argument('--nxi', type=int, default=40)
    p.add_argument('--solve_space', default='waypoints', choices=['waypoints', 'bspline'],
                   help='waypoints: solve on --T raw waypoints, store their B-spline '
                        'projection; bspline: solve on the nxi control points directly.')
    p.add_argument('--T', type=int, default=N_POINTS,
                   help='Waypoints of the waypoint solve (smoothness weight scaled by (T/128)^3).')
    p.add_argument('--lr', type=float, default=2e-3,
                   help='Adam step size of the SVGD update (svgd_batched.ADAM_LR = 2e-3).')
    p.add_argument('--w_smooth_scale', type=float, default=1.0,
                   help='Factor on the smoothness weight W_SMOOTH = 15.')
    p.add_argument('--n_starts', type=int, default=3,
                   help='Trajectories per density: the source x0 + (n_starts-1) uniform '
                        'random starts (stored as <name>@s<k>).')
    p.add_argument('--max_iters', type=int, default=10000)
    p.add_argument('--check_every', type=int, default=50)
    p.add_argument('--patience', type=int, default=1000,
                   help='Plateau window: stop once the best ergodic error improved by less '
                        'than --tol (relative) over the last this many iterations.')
    p.add_argument('--tol', type=float, default=0.01)
    p.add_argument('--min_iters', type=int, default=500)
    p.add_argument('--batch', type=int, default=4096,
                   help='Trajectories on the GPU at the same time (finished ones are replaced).')
    p.add_argument('--grid_res', type=int, default=128)
    p.add_argument('--compute_dtype', default='float64', choices=['float64', 'float32'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--limit', type=int, default=None, help='Only the first N densities (tests).')
    p.add_argument('--names', default=None, help='Comma-separated shape names (tests).')
    a = p.parse_args()

    rows = load_source_rows(a.src_db, tuple(a.variants.split(',')))
    if a.names:
        wanted = set(a.names.split(','))
        rows = [r for r in rows if r['name'] in wanted]
    if a.limit:
        rows = rows[:a.limit]
    n_dens = len(rows)
    rows = expand_starts(rows, a.n_starts, a.seed)
    conn = open_out_db(a.out)
    done = {r[0] for r in conn.execute("SELECT shape_name FROM ergodic_pairs")}
    todo = [r for r in rows if r['name'] not in done]
    print(f"  source   : {a.src_db}  ({n_dens} densities, variants {a.variants}, "
          f"{a.n_starts} starts each = {len(rows)} trajectories)")
    print(f"  output   : {a.out}  (already done: {len(rows) - len(todo)})")
    space = (f"{a.T} waypoints -> B-spline nxi={a.nxi}" if a.solve_space == 'waypoints'
             else f"B-spline nxi={a.nxi}")
    print(f"  solver   : Adam-SVGD on {space}, lr {a.lr:g}, W_SMOOTH x{a.w_smooth_scale:g}, "
          f"hard start pin, max_iters={a.max_iters}, "
          f"plateau tol={a.tol:.1%} over {a.patience} it, {a.compute_dtype} on {a.device}",
          flush=True)
    if not todo:
        return

    config = dict(nxi=a.nxi, n_points=N_POINTS, degree=DEGREE, max_iters=a.max_iters,
                  check_every=a.check_every, patience=a.patience, tol=a.tol,
                  min_iters=a.min_iters, grid_res=a.grid_res, compute_dtype=a.compute_dtype,
                  init='linear_through_centre', start_pin='hard', seed=a.seed,
                  n_starts=a.n_starts, x0_margin=X0_MARGIN, solve_space=a.solve_space,
                  T=a.T if a.solve_space == 'waypoints' else None, lr=a.lr,
                  w_smooth_scale=a.w_smooth_scale)

    # One grid per density, shared by all its starts.
    t0 = time.perf_counter()
    dens = list(dict.fromkeys(r['density'] for r in todo))
    dp = {r['density']: r['density_params'] for r in todo}
    grids = np.stack([density_grid(dp[d], a.grid_res) for d in dens])
    from svgd_batched import K_FREQ
    phik_d = target_coeffs(grids, K_FREQ)
    del grids
    pos = {d: i for i, d in enumerate(dens)}
    phik = phik_d[[pos[r['density']] for r in todo]]
    print(f"  {len(dens)} density grids + phi_k in {time.perf_counter() - t0:.1f}s "
          f"for {len(todo)} trajectories", flush=True)

    bs, init_fn, finalize = make_backend(a.solve_space, a.nxi, a.device, a.compute_dtype, T=a.T,
                                         w_smooth_scale=a.w_smooth_scale)
    from obstacles import bspline_basis_matrix
    B_store = bspline_basis_matrix(a.nxi, TSTEPS_STORE + 1, DEGREE)
    x0s = np.stack([r['x0'] for r in todo])
    P0 = init_fn(x0s)
    seeds = np.array([zlib.crc32(f"{a.seed}:{r['name']}".encode()) & 0x7FFFFFFF for r in todo])

    stats = dict(n=0, conv=0, iters=0)

    def on_done(i, res):
        res = finalize(res, todo[i]['x0'], phik[i])
        save_row(conn, todo[i], res, B_store, config)
        stats['n'] += 1
        stats['conv'] += int(res['converged'])
        stats['iters'] += res['n_iters']
        if stats['n'] % 50 == 0 or stats['n'] == len(todo):
            conn.commit()
            el = time.perf_counter() - t1
            print(f"  [{stats['n']}/{len(todo)}] {todo[i]['name']:<28} it={res['n_iters']:5d} "
                  f"E={res['erg_init']:.4f}->{res['erg']:.4f}  converged {stats['conv']}/{stats['n']}"
                  f"  mean it {stats['iters'] / stats['n']:.0f}  {el / 60:.1f} min", flush=True)

    # A preemption/time limit between commits only loses the uncommitted rows;
    # a restart skips everything already stored.
    def _term(signum, frame):
        conn.commit()
        print("  [!] SIGTERM -- committed, exiting", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term)

    t1 = time.perf_counter()
    solve_pinned(bs, P0, phik, x0s, seeds, max_iters=a.max_iters, check_every=a.check_every,
                 patience=a.patience, tol=a.tol, min_iters=a.min_iters, batch=a.batch,
                 on_done=on_done, log_every=10 * a.check_every, lr=a.lr)
    conn.commit()
    tot = conn.execute("SELECT count(*), sum(converged), avg(n_iters), avg(ergodic_error),"
                       " avg(length) FROM ergodic_pairs").fetchone()
    conn.close()
    print(f"\n  solved {stats['n']} trajectories in {(time.perf_counter() - t1) / 60:.1f} min")
    print(f"  DB now: {tot[0]} rows, converged {tot[1]}, mean n_iters {tot[2]:.0f}, "
          f"mean ergodic error {tot[3]:.4f}, mean path length {tot[4]:.2f}")


if __name__ == '__main__':
    main()
