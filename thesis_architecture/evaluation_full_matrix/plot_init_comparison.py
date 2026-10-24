r"""
plot_init_comparison.py
=========================
Per-shape + aggregate mean+-std-vs-SVGD-iteration panels for `run_init_
comparison.py`'s DB (`init_comparison_db.py`), direct analogue of `plot_
smoothness_comparison.py` widened from ~5 to 14 (7 method x 2 solver)
variants, plus a `summary.csv` with ALL 6 requested metrics at the final
iteration:

  ergodic error    `ergodic_energy_torch.ergodic_term` (K=10, W_ERGODIC=600,
                    the project's standard spectral quality metric; what the
                    solvers actually optimise) -- recomputed from the stored
                    states, like `plot_smoothness_comparison.py` already
                    does, not cached in the DB.
  coverage         `ergodic_energy_torch.coverage_distance` -- an
                    independent, non-spectral "how far is the target mass
                    from the path" check. Computed only at the FINAL logged
                    state (its O(T x grid-cells) cdist is much costlier per
                    call than the Fourier-based ergodic term; a full
                    per-iteration series for 14 variants x 25 shapes x 1500
                    states was judged not worth the wall-clock here).
  smoothness       stored `smooth_series` (computed by the generator).
  path length      stored `path_len_series`.
  outside fraction stored `outside_series` (fraction of waypoints below
                    `--outside_thresh` of the shape's own max density).
  n90              iteration at which the per-candidate ergodic-error series
                    first reaches 90% of its own total improvement,
                    `argmin_i { E(i) <= E(0) - 0.9*(E(0)-E(final)) }` --
                    matches Philipp's reference image's own legend literally.

Usage
-----
    python plot_init_comparison.py --out_tag init_comparison_YYYYMMDD
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
import torch                                                        # noqa: E402

import init_comparison_db as idb                                    # noqa: E402
import viz                                                           # noqa: E402
from ergodic_energy_torch import (ergodic_term, make_k_grid,        # noqa: E402
                                  target_coeffs_from_grid, coverage_distance, K_DEFAULT)

METHODS = ('cfm', 'selfsup', 'ebm', 'dfl', 'linear', 'random_walk', 'heuristic')
SOLVERS = ('tsvec_svgd', 'adam_svgd')

#: Validated categorical palette (dataviz skill, `scripts/validate_palette.js`;
#: the default 8-hue theme, slots 1-7, light mode). The project's earlier ad-hoc
#: colors FAILED the validator outright (lightness-band, chroma-floor and CVD
#: checks on #F9A825/#6D4C41/#9E9E9E/#8E24AA vs #1565C0); this 7-color subset
#: PASSES every adjacent-pair gate (worst CVD Delta E 9.1, worst normal-vision
#: Delta E 19.6) -- adjacent is the relevant pairlist for the bar/line/heatmap
#: forms these plots use. It still fails two ALL-pairs combinations (green vs
#: orange, violet vs blue) at 7 series, which the skill flags as expected past
#: 3 series and legal only with secondary encoding -- covered here by the
#: direct method-name labels on every chart (scatter annotations, breakdown
#: y-ticks, bar labels) plus solver-encoded linestyle/hatch/marker-shape.
METHOD_COLORS = {
    'cfm':          '#2a78d6',   # blue
    'selfsup':      '#eb6834',   # orange
    'ebm':          '#1baf7a',   # aqua
    'dfl':          '#eda100',   # yellow
    'linear':       '#e87ba4',   # magenta
    'random_walk':  '#008300',   # green
    'heuristic':    '#6250d6',   # violet
}
METHOD_LABELS = {
    'cfm': 'CFM', 'selfsup': 'Self-supervised', 'ebm': 'Energy-Policy (EBM)',
    'dfl': 'Decision-Focused Learning', 'linear': 'Linear', 'random_walk': 'Random walk',
    'heuristic': 'Heuristic (dataset-gen)',
}
SOLVER_LINESTYLES = {'tsvec_svgd': '-', 'adam_svgd': (0, (5, 2))}
SOLVER_LABELS = {'tsvec_svgd': 'TSVEC-SVGD', 'adam_svgd': 'Adam-SVGD'}


def _style(ax):
    ax.set_facecolor('white')
    ax.grid(alpha=0.2, color='#ccc')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555', labelsize=8)
    ax.xaxis.label.set_color('#1A1A2E')
    ax.yaxis.label.set_color('#1A1A2E')


def load_series(conn, shape, variant, series):
    rows = list(idb.iter_runs(conn, shape=shape, variant=variant))
    if not rows:
        return None, None
    n_states = max(r['n_states'] for r in rows)
    steps = np.arange(n_states)
    mat = np.full((len(rows), n_states), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        mat[i, :len(r[series])] = r[series]
    return steps, mat


def n90_from_series(err_mat):
    """err_mat: (n_cand, n_states) -> (n_cand,) int, iteration reaching 90%
    of that candidate's own (err[0] - err[final]) improvement. `n_states-1`
    ("never reached 90%") if the series never improves or is all-NaN."""
    n_cand, n_states = err_mat.shape
    out = np.full(n_cand, n_states - 1, dtype=np.int32)
    for c in range(n_cand):
        e = err_mat[c]
        valid = np.flatnonzero(np.isfinite(e))
        if valid.size < 2:
            continue
        e0, efin = e[valid[0]], e[valid[-1]]
        target = e0 - 0.9 * (e0 - efin)
        hit = valid[e[valid] <= target]
        out[c] = hit[0] if hit.size else valid[-1]
    return out


class ErgodicSeriesCache:
    """Recomputes `ergodic_term` from stored states, cached per (shape,
    variant) -- mirrors `plot_smoothness_comparison.ErgodicSeriesCache`.

    `ergodic_term`/`coverage_distance` are NOT stored in the DB (see module
    docstring) -- they're recomputed from the raw per-iteration states every
    time a plot script runs: ~5M (candidate, state) pairs through a B-spline
    render + 100-mode Fourier transform, CPU-only when there's no local GPU.
    That cost was being paid fresh on every one of this session's plot
    iterations (overview dashboard, ranking chart, per-metric plots, ...),
    each a separate process with no memory of the last. `_series`/
    `_coverage` are therefore ALSO persisted to `cache_path` (a pickle next
    to the DB, invalidated if the DB file's mtime moved past the cache's) --
    the in-memory dicts stay the source of truth within one run, the pickle
    just avoids repeating the expensive part across runs."""

    def __init__(self, conn, cache_path=None):
        self.conn = conn
        self.cache_path = cache_path
        k_idx, Lambda = make_k_grid(K_DEFAULT)
        self.k_idx = torch.as_tensor(k_idx, dtype=torch.float32)
        self.Lambda = torch.as_tensor(Lambda, dtype=torch.float32)
        self._phi_k, self._basis, self._series, self._dense, self._coverage = {}, {}, {}, {}, {}
        self._load_disk_cache()

    def _db_mtime(self):
        row = self.conn.execute("PRAGMA database_list").fetchone()
        db_file = row[2] if row else None
        return os.path.getmtime(db_file) if db_file and os.path.exists(db_file) else None

    def _load_disk_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, 'rb') as f:
                payload = pickle.load(f)
            db_mtime = self._db_mtime()
            if db_mtime is not None and payload.get('db_mtime') != db_mtime:
                print(f"[ErgodicSeriesCache] {self.cache_path} is older than the DB -- "
                     "ignoring (will recompute and overwrite).")
                return
            self._series = payload.get('series', {})
            self._coverage = payload.get('coverage', {})
            print(f"[ErgodicSeriesCache] loaded {len(self._series)} ergodic + "
                 f"{len(self._coverage)} coverage entries from {self.cache_path}")
        except Exception as exc:                                      # noqa: BLE001
            print(f"[ErgodicSeriesCache] failed to load {self.cache_path} ({exc}) -- recomputing.")

    def save(self):
        """Persist the in-memory ergodic/coverage caches -- call after a
        script has touched every (shape, variant) it needs (e.g. right after
        writing summary.csv), so later plot scripts in the same results dir
        start warm instead of repeating the ~5M-pair recomputation."""
        if not self.cache_path:
            return
        payload = dict(db_mtime=self._db_mtime(), series=self._series, coverage=self._coverage)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(payload, f)
        print(f"[ErgodicSeriesCache] saved {len(self._series)} ergodic + "
             f"{len(self._coverage)} coverage entries -> {self.cache_path}")

    def phi_k(self, shape):
        if shape not in self._phi_k:
            truth = torch.as_tensor(idb.load_truth(self.conn, shape), dtype=torch.float32)
            self._phi_k[shape] = target_coeffs_from_grid(truth, self.k_idx)
        return self._phi_k[shape]

    def basis(self, nxi):
        if nxi not in self._basis:
            row = self.conn.execute(
                "SELECT matrix, n_points FROM basis WHERE nxi=?", (nxi,)).fetchone()
            self._basis[nxi] = np.frombuffer(row[0], dtype=np.float32).reshape(row[1], nxi)
        return self._basis[nxi]

    def dense_states(self, shape, variant):
        """-> list of (cand_idx row dict, dense (n_states, T, 2) np) -- shared
        decode step for ergodic_term AND coverage_distance (final state).
        Cached per (shape, variant): `load()` and `coverage_final()` used to
        each decode+B-spline-render the full state log independently (same
        state_codec unpack + einsum, twice per cell) -- cheap on the cluster's
        24GB allocation this comparison already runs under, but pure waste
        regardless of where it runs."""
        key = (shape, variant)
        if key not in self._dense:
            rows = list(idb.iter_runs(self.conn, shape=shape, variant=variant, with_states=True))
            self._dense[key] = [
                (r, np.einsum('pi,sid->spd', self.basis(r['nxi']), r['cps'])
                 if r['log_space'] == 'cps' else r['cps'])
                for r in rows]
        return self._dense[key]

    def load(self, shape, variant):
        """-> (n_states,) step indices, (n_cand, n_states) float array."""
        key = (shape, variant)
        if key in self._series:
            return self._series[key]
        decoded = self.dense_states(shape, variant)
        if not decoded:
            self._series[key] = (None, None)
            return self._series[key]
        phi_k = self.phi_k(shape)
        n_states = max(r['n_states'] for r, _ in decoded)
        mat = np.full((len(decoded), n_states), np.nan, dtype=np.float32)
        for i, (r, dense) in enumerate(decoded):
            flat = torch.as_tensor(dense, dtype=torch.float32)
            chunks = [ergodic_term(flat[c0:c0 + 2500], self.k_idx, self.Lambda, phi_k)
                     for c0 in range(0, flat.shape[0], 2500)]
            mat[i, :r['n_states']] = torch.cat(chunks).numpy()
        out = (np.arange(n_states), mat)
        self._series[key] = out
        return out

    def coverage_final(self, shape, variant):
        """-> (n_cand,) float, `coverage_distance` at each candidate's FINAL
        logged state only (see module docstring for why not every state)."""
        key = (shape, variant)
        if key in self._coverage:
            return self._coverage[key]
        decoded = self.dense_states(shape, variant)
        if not decoded:
            self._coverage[key] = np.array([])
            return self._coverage[key]
        truth = torch.as_tensor(idb.load_truth(self.conn, shape), dtype=torch.float32)
        finals = torch.stack([torch.as_tensor(dense[-1], dtype=torch.float32) for _r, dense in decoded])
        out = coverage_distance(finals, truth).numpy()
        self._coverage[key] = out
        return out


def _metrics(ergo):
    return (
        ('ergodic_series', 'Ergodic error (lower = better coverage)', ergo.load, None),
        ('smooth_series', 'Smoothness energy (lower = smoother)',
         lambda shape, variant: load_series(ergo.conn, shape, variant, 'smooth_series'),
         dict(value='symlog', linthresh=0.3)),
        ('path_len_series', 'Path length', lambda shape, variant: load_series(ergo.conn, shape, variant, 'path_len_series'), None),
        ('outside_series', 'Fraction of steps outside target support',
         lambda shape, variant: load_series(ergo.conn, shape, variant, 'outside_series'), None),
    )


def plot_variant_panel(ax, conn, shapes, variants, loader, per_shape_mode=False, shape=None):
    for method in METHODS:
        for solver in SOLVERS:
            variant = f"{method}__{solver}"
            if variant not in variants:
                continue
            if per_shape_mode:
                steps, mat = loader(shape, variant)
                if steps is None:
                    continue
                valid = ~np.all(np.isnan(mat), axis=0)
                s, mean, std = steps[valid], np.nanmean(mat[:, valid], axis=0), np.nanstd(mat[:, valid], axis=0)
            else:
                per_shape_means, steps_ref = [], None
                for sh in shapes:
                    steps, mat = loader(sh, variant)
                    if steps is None:
                        continue
                    valid = ~np.all(np.isnan(mat), axis=0)
                    per_shape_means.append(np.nanmean(mat[:, valid], axis=0))
                    steps_ref = steps[valid]
                if not per_shape_means:
                    continue
                n = min(len(m) for m in per_shape_means)
                M = np.stack([m[:n] for m in per_shape_means])
                s, mean, std = steps_ref[:n], M.mean(axis=0), M.std(axis=0)
            color = METHOD_COLORS[method]
            ax.plot(s, mean, color=color, lw=1.8, ls=SOLVER_LINESTYLES[solver],
                   label=f"{METHOD_LABELS[method]} + {SOLVER_LABELS[solver]}")
            ax.fill_between(s, mean - std, mean + std, color=color, alpha=0.12)


def plot_shape_series(conn, shape, variants, out_path, ergo):
    metrics = _metrics(ergo)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4), facecolor='white')
    for ax, (_key, ylabel, loader, yscale) in zip(axes, metrics):
        plot_variant_panel(ax, conn, [shape], variants, loader, per_shape_mode=True, shape=shape)
        ax.set_xlabel('SVGD iteration')
        ax.set_ylabel(ylabel)
        if yscale is not None:
            ax.set_yscale(**yscale)
        _style(ax)
    axes[0].legend(fontsize=5.5, loc='upper right', frameon=False, ncol=2)
    fig.suptitle(f'Init-method x solver comparison -- {shape} (mean +- std over candidates)',
                color='#1A1A2E', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)


def plot_aggregate_series(conn, shapes, variants, out_path, ergo):
    metrics = _metrics(ergo)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4), facecolor='white')
    for ax, (_key, ylabel, loader, yscale) in zip(axes, metrics):
        plot_variant_panel(ax, conn, shapes, variants, loader, per_shape_mode=False)
        ax.set_xlabel('SVGD iteration')
        ax.set_ylabel(ylabel)
        if yscale is not None:
            ax.set_yscale(**yscale)
        _style(ax)
    axes[0].legend(fontsize=5.5, loc='upper right', frameon=False, ncol=2)
    fig.suptitle(f'Init-method x solver comparison -- averaged over {len(shapes)} holdout shapes',
                color='#1A1A2E', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)


def write_summary_csv(conn, shapes, variants, out_path, ergo):
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['shape', 'method', 'solver', 'n_cand',
                   'ergodic_final_mean', 'ergodic_final_std',
                   'coverage_final_mean', 'coverage_final_std',
                   'smooth_final_mean', 'smooth_final_std',
                   'path_len_final_mean', 'path_len_final_std',
                   'outside_final_mean', 'outside_final_std',
                   'n90_mean', 'n90_std'])
        for shape in shapes:
            for method in METHODS:
                for solver in SOLVERS:
                    variant = f"{method}__{solver}"
                    if variant not in variants:
                        continue
                    rows = list(idb.iter_runs(conn, shape=shape, variant=variant))
                    if not rows:
                        continue
                    sm = np.array([r['smooth_series'][np.isfinite(r['smooth_series'])][-1] for r in rows])
                    pl = np.array([r['path_len_series'][np.isfinite(r['path_len_series'])][-1] for r in rows])
                    out = np.array([r['outside_series'][np.isfinite(r['outside_series'])][-1] for r in rows])
                    _steps, emat = ergo.load(shape, variant)
                    last_valid = np.where(~np.all(np.isnan(emat), axis=0))[0][-1]
                    e_final = emat[:, last_valid]
                    n90 = n90_from_series(emat)
                    cov = ergo.coverage_final(shape, variant)
                    w.writerow([shape, method, solver, len(rows),
                               float(np.nanmean(e_final)), float(np.nanstd(e_final)),
                               float(cov.mean()), float(cov.std()),
                               float(sm.mean()), float(sm.std()),
                               float(pl.mean()), float(pl.std()),
                               float(out.mean()), float(out.std()),
                               float(n90.mean()), float(n90.std())])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    args = ap.parse_args()

    out_dir = os.path.join(_here, 'results', args.out_tag)
    db_path = os.path.join(out_dir, 'init_comparison.db')
    conn = idb.open_db(db_path)
    cfg = idb.get_meta(conn, 'config', {})
    shapes = cfg.get('shapes') or sorted({r[0] for r in conn.execute("SELECT DISTINCT shape FROM runs")})
    variants = sorted({r[0] for r in conn.execute("SELECT DISTINCT variant FROM runs")})
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'per_shape'), exist_ok=True)

    ergo = ErgodicSeriesCache(conn, cache_path=os.path.join(out_dir, 'metric_cache.pkl'))

    write_summary_csv(conn, shapes, variants, os.path.join(out_dir, 'summary.csv'), ergo)
    print(f"[plot_init] summary.csv written ({len(shapes)} shapes x {len(variants)} variants)")

    plot_aggregate_series(conn, shapes, variants, os.path.join(plot_dir, 'aggregate.png'), ergo)
    print("[plot_init] aggregate.png written")

    for shape in shapes:
        plot_shape_series(conn, shape, variants, os.path.join(plot_dir, 'per_shape', f'{shape}.png'), ergo)
    print(f"[plot_init] {len(shapes)} per-shape plots written -> {plot_dir}")

    ergo.save()


if __name__ == '__main__':
    main()
