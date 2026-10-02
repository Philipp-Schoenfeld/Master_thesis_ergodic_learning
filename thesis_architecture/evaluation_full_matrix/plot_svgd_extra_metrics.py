r"""
plot_svgd_extra_metrics.py
===========================
Three further views on the SVGD-convergence DB (same 3 x 3 layout as
`plot_svgd_convergence.py`: strategies x knowledge states), all derived from
the stored intermediate states -- no SVGD is re-run.

1. Iterations to target (`iterations_to_target_*.png`)
   Fraction of runs whose ergodic error E has reached a per-shape target
   eps(shape) = eps_factor * E_ref(shape) by iteration k (first passage).
   E_ref(shape) = median over ALL runs of that shape (every method, knowledge
   state, strategy; they share the true density as SVGD target) of E at the
   last stored iteration. The warm-start speed-up is read off as how far left
   a curve rises. Default eps_factor 1.5 (at 1.1 many baselines never get
   there, at 3 everything is trivially fast); the CSV lists 1.1 / 1.25 / 1.5 /
   2 / 3.
2. Swept target mass (`swept_mass_*.png`)
   Fraction of the true density's mass within `--sensor_radius` (0.06, the
   mission's sensor footprint) of the path
   (`metrics_explore_exploit.swept_mass_fraction`), per iteration.
3. Diversity and displacement
   `diversity_*.png`: mean pairwise symmetric Chamfer distance between the
   n_init paths of one (shape, knowledge, strategy, method) group, per
   iteration (do different inits end in the same solution?).
   `displacement_*.png`: Chamfer distance of every path to its own state at
   iteration 0 (the B-spline fit of the initialisation), i.e. how far SVGD
   had to move it.
   Chamfer = 0.5 * (mean nearest-neighbour distance a->b + b->a) on the
   rendered 128-point curves, in workspace units ([0,1]^2).

Metrics 2/3 are evaluated on a fixed iteration grid (every 5th iteration up to
100, every 50th afterwards) and cached in `extra_metrics_cache_<N>iters.npz`
(delete or pass --recompute after the DB changed).

Usage:
    python plot_svgd_extra_metrics.py --out_tag svgd_convergence_20261001
    python plot_svgd_extra_metrics.py --out_tag ... --eps_factor 2 --sensor_radius 0.05
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import torch                                                         # noqa: E402
import matplotlib                                                    # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                      # noqa: E402

import svgd_convergence_db as sdb                                    # noqa: E402
from plot_svgd_convergence import (                                  # noqa: E402
    COND_ORDER, COND_TITLE, FIRST_RUN_ITERS, INK, METHOD_STYLE, MUTED, STRAT_ORDER,
    STRATEGY_TITLE, band, broadcast_shared, mark_first_run_end, style_axes)

EPS_TABLE_FACTORS = (1.1, 1.25, 1.5, 2.0, 3.0)
DEFAULT_RADIUS = 0.06           # sensor footprint of the mission (variant_runner)


# ── metric kernels (torch, batched) ─────────────────────────────────────────

def chamfer(a, b):
    """Symmetric Chamfer distance. a: (..., Ta, 2), b: (..., Tb, 2) -> (...)."""
    d = torch.cdist(a, b)
    return 0.5 * (d.min(dim=-1).values.mean(dim=-1) + d.min(dim=-2).values.mean(dim=-1))


def grid_cells(res, device):
    """Cell centres of the res x res density grid, same order as
    `metrics_explore_exploit.swept_mass_fraction`."""
    ys, xs = torch.meshgrid(torch.linspace(0, 1, res, device=device),
                            torch.linspace(0, 1, res, device=device), indexing='ij')
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def swept_mass_batch(curves, cells, w, radius, chunk=32):
    """curves: (S, T, 2) -> (S,) fraction of mass within `radius` of each curve.
    Batched version of `swept_mass_fraction` (checked against it in the tests)."""
    out = []
    wsum = w.sum().clamp(min=1e-12)
    for i in range(0, curves.shape[0], chunk):
        c = curves[i:i + chunk]
        d = torch.cdist(cells.unsqueeze(0).expand(c.shape[0], -1, -1), c)   # (c, N, T)
        covered = (d.min(dim=-1).values <= radius).to(w.dtype)               # (c, N)
        out.append((covered * w).sum(dim=-1) / wsum)
    return torch.cat(out)


def pairwise_diversity(x, chunk=4):
    """x: (n, G, T, 2) curves of n runs at G iterations -> (G,) mean pairwise
    Chamfer distance over the n(n-1) ordered pairs."""
    n, G, T, _ = x.shape
    x = x.permute(1, 0, 2, 3)                                  # (G, n, T, 2)
    out = []
    for i in range(0, G, chunk):
        xc = x[i:i + chunk]
        g = xc.shape[0]
        a = xc[:, :, None].expand(g, n, n, T, 2).reshape(-1, T, 2)
        b = xc[:, None].expand(g, n, n, T, 2).reshape(-1, T, 2)
        dm = chamfer(a, b).reshape(g, n, n)
        out.append((dm.sum(dim=(1, 2)) - dm.diagonal(dim1=1, dim2=2).sum(dim=1)) / (n * (n - 1)))
    return torch.cat(out)


def first_passage(E, eps):
    """E: (..., n_iters+1), eps broadcastable to E[..., 0] -> first iteration with
    E <= eps (float, nan if never)."""
    hit = E <= np.asarray(eps)[..., None]
    return np.where(hit.any(axis=-1), hit.argmax(axis=-1), np.nan).astype(float)


def iteration_grid(n_iters):
    g = np.unique(np.concatenate([np.arange(0, 100, 5), np.arange(100, n_iters + 1, 50),
                                  [n_iters]]))
    return g[g <= n_iters]


# ── derived-metric pass over the DB ─────────────────────────────────────────

def compute_derived(conn, grid, radius, device):
    """One streaming pass. -> dict with per-run swept mass / displacement on
    `grid` plus per-group diversity."""
    runs, swept, disp = [], [], []
    groups = {}
    B = None
    cells = w = None
    truth_cache = {}
    t0 = time.time()
    n = 0
    for r in sdb.iter_runs(conn, columns=('cps',), include_ext=True):
        if B is None:
            B = torch.as_tensor(np.array(sdb.load_basis(conn, r['nxi'], r['n_points'])), device=device)
        shape = r['shape']
        if shape not in truth_cache:
            t = torch.as_tensor(np.array(sdb.load_truth(conn, shape)), device=device)
            truth_cache[shape] = (grid_cells(t.shape[-1], device), t.reshape(-1).clamp(min=0.0))
        cells, w = truth_cache[shape]
        cps = torch.as_tensor(r['cps'][grid], device=device)                 # (G, nxi, 2)
        curves = torch.einsum('pi,gid->gpd', B, cps)                         # (G, T, 2)
        swept.append(swept_mass_batch(curves, cells, w, radius).cpu().numpy())
        disp.append(chamfer(curves, curves[:1].expand_as(curves)).cpu().numpy())
        runs.append((shape, r['knowledge_condition'], r['strategy'], r['method'], r['init_idx']))
        groups.setdefault((shape, r['knowledge_condition'], r['strategy'], r['method']),
                          {})[r['init_idx']] = curves.cpu()
        n += 1
        if n % 500 == 0:
            print(f"[extra] {n} runs processed, {time.time() - t0:.0f}s", flush=True)
    div_keys, div = [], []
    for key, d in sorted(groups.items()):
        x = torch.stack([d[i] for i in sorted(d)]).to(device)                # (n, G, T, 2)
        div_keys.append(key)
        div.append(pairwise_diversity(x).cpu().numpy())
    print(f"[extra] derived metrics for {n} runs, {len(div_keys)} groups, "
          f"{time.time() - t0:.0f}s", flush=True)
    return dict(run_keys=np.array(runs, dtype='U40'), swept=np.stack(swept),
                disp=np.stack(disp), grid=grid, radius=radius,
                div_keys=np.array(div_keys, dtype='U40'), div=np.stack(div))


def load_or_compute(conn, grid, radius, device, cache_path, recompute):
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    if os.path.isfile(cache_path) and not recompute:
        z = np.load(cache_path)
        if (np.array_equal(z['grid'], grid) and float(z['radius']) == radius
                and len(z['run_keys']) == n_runs):
            print(f"[extra] using cache {os.path.basename(cache_path)}")
            return {k: z[k] for k in z.files}
        print("[extra] cache does not match (grid/radius/runs) -> recomputing")
    d = compute_derived(conn, grid, radius, device)
    np.savez_compressed(cache_path, **d)
    return d


# ── series builders ─────────────────────────────────────────────────────────

def per_run_series(keys, values):
    """keys: (R, 5) [shape, cond, strat, method, idx], values: (R, G) ->
    {(cond, strat, method): (mean, band)}: mean over shapes of the per-shape
    mean over inits; band = mean over shapes of the per-shape std."""
    by = {}
    for (shape, c, st, m, _idx), v in zip(keys, values):
        by.setdefault((c, st, m), {}).setdefault(shape, []).append(v)
    out = {}
    for k, per_shape in by.items():
        means = np.stack([np.mean(v, axis=0) for v in per_shape.values()])
        stds = np.stack([np.std(v, axis=0) for v in per_shape.values()])
        out[k] = (means.mean(axis=0), stds.mean(axis=0))
    return broadcast_shared(out)


def diversity_series(keys, values):
    """keys: (n, 4) [shape, cond, strat, method], values (n, G) -> mean over
    shapes, band = std across shapes."""
    by = {}
    for (shape, c, st, m), v in zip(keys, values):
        by.setdefault((c, st, m), []).append(v)
    out = {k: (np.mean(v, axis=0), np.std(v, axis=0)) for k, v in by.items()}
    return broadcast_shared(out)


# ── figures ─────────────────────────────────────────────────────────────────

def grid_figure(series, x, ylabel, title, path, logy=False, ylim=None, band_on=True,
                last_iter=None):
    conds = [c for c in COND_ORDER if any(k[0] == c for k in series)]
    strats = [s for s in STRAT_ORDER if any(k[1] == s for k in series)]
    if not conds or not strats:
        return
    fig, axes = plt.subplots(len(strats), len(conds), figsize=(4.4 * len(conds), 3.4 * len(strats)),
                             facecolor='white', squeeze=False, sharex=True)
    for i, st in enumerate(strats):
        for j, c in enumerate(conds):
            ax = axes[i][j]
            style_axes(ax)
            for m, sty in METHOD_STYLE.items():
                if (c, st, m) not in series:
                    continue
                mean, spread = series[(c, st, m)]
                if band_on:
                    band(ax, x, mean, spread, sty) if logy else _linear_band(ax, x, mean, spread, sty)
                else:
                    ax.plot(x, mean, color=sty['color'], lw=sty['lw'], label=sty['label'], alpha=0.95)
            mark_first_run_end(ax, last_iter if last_iter is not None else int(x[-1]))
            if logy:
                ax.set_yscale('log')
            if ylim is not None:
                ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(COND_TITLE[c], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{STRATEGY_TITLE[st]}\n{ylabel}", fontsize=9, color=INK)
            if i == len(strats) - 1:
                ax.set_xlabel('SVGD iteration', fontsize=9, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), frameon=False,
                   fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, fontsize=10, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def _linear_band(ax, x, mean, spread, sty):
    ax.plot(x, mean, color=sty['color'], lw=sty['lw'], label=sty['label'], alpha=0.95)
    ax.fill_between(x, np.maximum(mean - spread, 0.0), mean + spread, color=sty['color'],
                    alpha=0.18, linewidth=0)


# ── iterations to target ────────────────────────────────────────────────────

def load_E(conn):
    rows = []
    for r in sdb.iter_runs(conn, columns=('E_total',), include_ext=True):
        rows.append((r['shape'], r['knowledge_condition'], r['strategy'], r['method'],
                     r['init_idx'], r['E_total']))
    return rows


def reference_levels(rows):
    """shape -> median final E over all runs of that shape."""
    by = {}
    for shape, *_rest, E in rows:
        by.setdefault(shape, []).append(E[-1])
    return {s: float(np.median(v)) for s, v in by.items()}


def target_tables(rows, ref, factor):
    """-> {(cond, strat, method): first-passage iterations of all its runs
    (nan = never reached)}, shared baselines broadcast."""
    by = {}
    for shape, c, st, m, _i, E in rows:
        by.setdefault((c, st, m), []).append(first_passage(E, ref[shape] * factor))
    return broadcast_shared({k: np.array(v) for k, v in by.items()})


def survival_series(fp, n_iters):
    x = np.arange(n_iters + 1)
    out = {}
    for k, v in fp.items():
        frac = np.array([(v <= it).sum() for it in x], dtype=float) / len(v)   # nan <= it is False
        out[k] = (frac, None)
    return x, out


def write_target_csv(rows, ref, path, n_iters):
    out = []
    for f in EPS_TABLE_FACTORS:
        fp = target_tables(rows, ref, f)
        for (c, st, m), v in sorted(fp.items()):
            ok = ~np.isnan(v)
            out.append({'eps_factor': f, 'knowledge_condition': c, 'strategy': st, 'method': m,
                        'n_runs': len(v), 'fraction_reached': float(ok.mean()),
                        'median_iters_reached': float(np.median(v[ok])) if ok.any() else '',
                        'mean_iters_reached': float(v[ok].mean()) if ok.any() else '',
                        'q25_iters_reached': float(np.percentile(v[ok], 25)) if ok.any() else '',
                        'q75_iters_reached': float(np.percentile(v[ok], 75)) if ok.any() else '',
                        'last_iteration': n_iters})
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def write_derived_csv(d, path, grid):
    sel = [i for i, g in enumerate(grid) if g in (0, 10, 25, 100, 250, 500, 1000, 2000, 3000)]
    metrics = {
        'swept_mass': per_run_series(d['run_keys'], d['swept']),
        'displacement': per_run_series(d['run_keys'], d['disp']),
        'diversity': diversity_series(d['div_keys'], d['div']),
    }
    rows = []
    for name, series in metrics.items():
        for (c, st, m), (mean, spread) in sorted(series.items()):
            row = {'metric': name, 'knowledge_condition': c, 'strategy': st, 'method': m}
            for i in sel:
                row[f'mean_iter{grid[i]}'] = float(mean[i])
                row[f'spread_iter{grid[i]}'] = float(spread[i])
            rows.append(row)
    keys = list(rows[0])
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--eps_factor', type=float, default=1.5,
                    help='Target = eps_factor * median final E of the shape (default 1.5).')
    ap.add_argument('--sensor_radius', type=float, default=DEFAULT_RADIUS)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--only', type=str, default='target,swept,diversity,displacement',
                    help='Comma-separated subset of: target, swept, diversity, displacement.')
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(',')}

    out_dir = os.path.join(_here, 'results', args.out_tag)
    conn = sdb.open_db(os.path.join(out_dir, 'svgd_convergence.db'))
    plots = os.path.join(out_dir, 'plots')
    os.makedirs(plots, exist_ok=True)
    n_runs, n_iters = conn.execute("SELECT COUNT(*), MAX(n_iters) FROM runs").fetchone()
    ext_total = conn.execute("SELECT MAX(n_iters_total) FROM runs_ext").fetchone()[0]
    n_iters = max(n_iters, ext_total or 0)
    n_ext = len(sdb.ext_run_ids(conn))
    if 0 < n_ext < n_runs:
        raise SystemExit(f"extension incomplete ({n_ext}/{n_runs} runs) -- finish "
                         "extend_svgd_convergence.py first")
    sfx = f'_{n_iters}iters' if n_iters > FIRST_RUN_ITERS else ''
    cfg = sdb.get_meta(conn, 'config', {})
    n_init = cfg.get('n_init', '?')
    n_shapes = conn.execute("SELECT COUNT(DISTINCT shape) FROM runs").fetchone()[0]

    if 'target' in only:
        rows = load_E(conn)
        ref = reference_levels(rows)
        fp = target_tables(rows, ref, args.eps_factor)
        x, series = survival_series(fp, n_iters)
        tag = f"{args.eps_factor:g}".replace('.', 'p')
        grid_figure(series, x, 'Fraction of runs at target',
                    f"Iterations to target: fraction of runs with E <= {args.eps_factor:g} x E_ref(shape) "
                    f"by iteration k ({n_shapes} shapes x {n_init} inits pooled; "
                    f"E_ref = median final E of the shape)",
                    os.path.join(plots, f'iterations_to_target_x{tag}{sfx}.png'),
                    ylim=(0, 1.02), band_on=False, last_iter=n_iters)
        write_target_csv(rows, ref, os.path.join(out_dir, f'iterations_to_target{sfx}.csv'), n_iters)
        for (c, st, m), v in sorted(fp.items()):
            if st == 'eid' or m != 'cfm':
                ok = ~np.isnan(v)
                print(f"[target x{args.eps_factor:g}] {c:12s} {st:4s} {m:12s} reached "
                      f"{ok.mean():.2f}, median iterations "
                      f"{np.median(v[ok]) if ok.any() else float('nan'):.0f}")
        del rows

    if only & {'swept', 'diversity', 'displacement'}:
        grid = iteration_grid(n_iters)
        cache = os.path.join(out_dir, f'extra_metrics_cache_{n_iters}iters.npz')
        d = load_or_compute(conn, grid, args.sensor_radius, args.device, cache, args.recompute)
        x = d['grid']
        r_note = f"sensor radius {args.sensor_radius:g}"
        if 'swept' in only:
            grid_figure(per_run_series(d['run_keys'], d['swept']), x,
                        'Swept target mass (fraction)',
                        f"Swept target mass vs. SVGD iterations ({r_note}) -- mean over {n_shapes} "
                        f"shapes; band: +-1 std across initialisations (n={n_init})",
                        os.path.join(plots, f'swept_mass{sfx}.png'), ylim=(0, 1.02),
                        last_iter=n_iters)
        if 'diversity' in only:
            grid_figure(diversity_series(d['div_keys'], d['div']), x,
                        'Mean pairwise Chamfer distance',
                        f"Diversity of the {n_init} paths per group vs. SVGD iterations -- mean over "
                        f"{n_shapes} shapes; band: +-1 std across shapes",
                        os.path.join(plots, f'diversity{sfx}.png'), logy=True, last_iter=n_iters)
        if 'displacement' in only:
            grid_figure(per_run_series(d['run_keys'], d['disp']), x,
                        'Chamfer distance to iteration-0 path',
                        f"Displacement from the initial path vs. SVGD iterations -- mean over "
                        f"{n_shapes} shapes; band: +-1 std across initialisations (n={n_init})",
                        os.path.join(plots, f'displacement{sfx}.png'), last_iter=n_iters)
        write_derived_csv(d, os.path.join(out_dir, f'extra_metrics_summary{sfx}.csv'), x)
    print(f"[extra] wrote figures to {plots}")


if __name__ == '__main__':
    main()
