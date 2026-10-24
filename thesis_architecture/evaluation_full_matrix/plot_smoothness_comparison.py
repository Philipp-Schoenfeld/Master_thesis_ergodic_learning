r"""
plot_smoothness_comparison.py
===============================
Plots and summary tables for `run_smoothness_comparison.py`'s DB
(`smoothness_db.py`): for each holdout shape, how smoothness, path length AND
ergodic error (coverage quality against the TRUE density) of the
30-trajectory DISTRIBUTION evolve over the SVGD iterations, per variant --
mean plus a shaded standard-deviation band (Philipp's meeting note: "immer
Mean und Standard Deviation plotten"), aggregated over all shapes, plus a
bonus visualisation of the trajectory distribution itself (mean curve +
pointwise positional covariance) at a few selected steps, the literal "Mean
und Covariance einer Verteilung von 30 Trajektorien" from the same note.

The ergodic-error panel (added 2026-10-09, Philipp's request: "füge ... die
ergodicity ein, damit man einen möglicherweise existierenden trade off
beobachten kann") is NOT stored in the DB -- unlike smoothness/path length
(computed once by `run_smoothness_comparison.py` and written to
`smooth_series`/`path_len_series`), it is recomputed here directly from the
already-logged SVGD states (`states` blob, every intermediate state of every
trajectory -- no SVGD re-run needed) against each shape's target Fourier
coefficients, using the project's standard ergodic-error metric
(`ergodic_energy_torch.ergodic_term`, K=10, W_ERGODIC=600, the same number
reported everywhere else in the project). One batched `ergodic_term` call per
(shape, variant) cell (all candidates x all states at once), so this is cheap
despite not being cached in the DB.

Usage
-----
    python plot_smoothness_comparison.py --out_tag smoothness_comparison_YYYYMMDD
"""
import argparse
import csv
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_arch, 'ergodic_validation'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
import torch                                                        # noqa: E402

import smoothness_db as sdb                                         # noqa: E402
import viz                                                          # noqa: E402
from ergodic_energy_torch import (ergodic_term, make_k_grid,        # noqa: E402
                                  target_coeffs_from_grid, K_DEFAULT)

VARIANT_COLORS = {
    'cfm_only':               '#9E9E9E',
    'cfm_svgd':                '#1565C0',
    'linear_svgd_bspline':     '#F9A825',
    'linear_svgd_raw':         '#8E24AA',
    'linear_svgd_raw_smooth':  '#00C853',
    # round 2 (2026-10-09): sun_* = dynamics-model solver (sun_refine.py),
    # *_svgd = direct preconditioned SVGD on points (tsvec_svgd.py, no
    # dynamics), see run_smoothness_comparison_v2.py's module docstring.
    'cfm_bspline_svgd':            '#1565C0',
    'linear_bspline_svgd':          '#F9A825',
    'linear_waypoint_svgd':         '#8E24AA',
    'linear_waypoint_svgd_smooth':  '#00C853',
    'linear_sun_pointmass':         '#6D4C41',
    'linear_sun_jerk':              '#00ACC1',
    # Adam-SVGD (svgd_batched.BatchedSvgdTorch, the GUI's refiner)
    'cfm_adam_bspline':            '#0D47A1',
    'linear_adam_bspline':         '#E65100',
    'linear_adam_waypoint':        '#AD1457',
    'linear_adam_waypoint_smooth': '#2E7D32',
}
VARIANT_LABELS = {
    'cfm_only':               'CFM inference only',
    'cfm_svgd':                'CFM + SVGD (B-spline)',
    'linear_svgd_bspline':     'Linear init + SVGD (B-spline)',
    'linear_svgd_raw':         'Linear init + SVGD (raw waypoints)',
    'linear_svgd_raw_smooth':  'Linear init + SVGD + smoothness force (raw waypoints)',
    'cfm_bspline_svgd':            'CFM + TSVEC-SVGD (B-spline)',
    'linear_bspline_svgd':          'Linear init + TSVEC-SVGD (B-spline)',
    'linear_waypoint_svgd':         'Linear init + TSVEC-SVGD (raw waypoints)',
    'linear_waypoint_svgd_smooth':  'Linear init + TSVEC-SVGD + smoothness (raw waypoints)',
    'linear_sun_pointmass':         'Linear init + Sun flow matching (Sinkhorn flow, point mass)',
    'linear_sun_jerk':              'Linear init + Sun flow matching (Stein flow, jerk-penalised)',
    'cfm_adam_bspline':            'CFM + Adam-SVGD (B-spline)',
    'linear_adam_bspline':         'Linear init + Adam-SVGD (B-spline)',
    'linear_adam_waypoint':        'Linear init + Adam-SVGD (waypoints)',
    'linear_adam_waypoint_smooth': 'Linear init + Adam-SVGD + smoothness force (waypoints)',
}
#: distinct dash pattern + linewidth per variant, on top of colour -- several
#: of these curves land almost exactly on top of each other (e.g. the three
#: linear-init variants in the ergodic-error panel), and a single solid line
#: per colour then makes the ones drawn first look "missing" rather than
#: "coincide". Different on/off periods mean each line still peeks through at
#: some x, regardless of z-order.
VARIANT_LINESTYLES = {
    'cfm_svgd':                '-',
    'linear_svgd_bspline':     (0, (1, 1)),       # fine dots
    'linear_svgd_raw':         (0, (5, 2)),       # dashed
    'linear_svgd_raw_smooth':  (0, (4, 1, 1, 1)),  # dash-dot
    'cfm_bspline_svgd':            '-',
    'linear_bspline_svgd':          (0, (1, 1)),
    'linear_waypoint_svgd':         (0, (5, 2)),
    'linear_waypoint_svgd_smooth':  (0, (4, 1, 1, 1)),
    'linear_sun_pointmass':         (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
    'linear_sun_jerk':              (0, (2, 1)),
    'cfm_adam_bspline':            '-',
    'linear_adam_bspline':         (0, (1, 1)),
    'linear_adam_waypoint':        (0, (5, 2)),
    'linear_adam_waypoint_smooth': (0, (4, 1, 1, 1)),
}
VARIANT_LW = {
    'cfm_svgd':                2.0,
    'linear_svgd_bspline':     2.6,
    'linear_svgd_raw':         2.0,
    'linear_svgd_raw_smooth':  1.6,
    'cfm_bspline_svgd':            2.0,
    'linear_bspline_svgd':          2.6,
    'linear_waypoint_svgd':         2.0,
    'linear_waypoint_svgd_smooth':  1.6,
    'linear_sun_pointmass':         2.2,
    'linear_sun_jerk':              1.8,
    'cfm_adam_bspline':            2.0,
    'linear_adam_bspline':         2.6,
    'linear_adam_waypoint':        2.0,
    'linear_adam_waypoint_smooth': 1.6,
}


def _style(ax):
    ax.set_facecolor('white')
    ax.grid(alpha=0.2, color='#ccc')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(colors='#555', labelsize=8)
    ax.xaxis.label.set_color('#1A1A2E')
    ax.yaxis.label.set_color('#1A1A2E')


def load_series(conn, shape, variant, series='smooth_series'):
    """-> (n_states,) step indices, (n_cand, n_states) float array (NaN where
    `--metric_stride` skipped a state)."""
    rows = list(sdb.iter_runs(conn, shape=shape, variant=variant))
    if not rows:
        return None, None
    n_states = max(r['n_states'] for r in rows)
    steps = np.arange(n_states)
    mat = np.full((len(rows), n_states), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        mat[i, :len(r[series])] = r[series]
    return steps, mat


#: log-spaced state-index subset `ErgodicSeriesCache.load` evaluates `ergodic_term` at, instead
#: of every stored state (see its docstring). Independent of `compute_convergence.py`'s own grid
#: (same idea, different consumer/cache) -- kept separate rather than shared to avoid a circular
#: import (that module imports `ErgodicSeriesCache` from this one).
_N_CHECKPOINTS = 150


def _checkpoint_indices(n_states):
    """Log-spaced indices in [0, n_states - 1], always including both ends."""
    n_iters = n_states - 1
    if n_iters <= 0:
        return np.array([0])
    idx = np.unique(np.round(np.geomspace(1, n_iters, _N_CHECKPOINTS - 1)).astype(int))
    return np.concatenate([[0], idx])


class ErgodicSeriesCache:
    """Computes `ergodic_term` (the project's standard ergodic-error metric,
    K=10/W_ERGODIC=600) from the STORED states -- see module docstring --
    and caches the result per (shape, variant), since both the per-shape and
    the aggregate plot need it."""

    def __init__(self, conn):
        self.conn = conn
        self.k_idx, Lambda = make_k_grid(K_DEFAULT)
        self.k_idx = torch.as_tensor(self.k_idx, dtype=torch.float32)
        self.Lambda = torch.as_tensor(Lambda, dtype=torch.float32)
        self._phi_k = {}
        self._basis = {}
        self._series = {}

    def phi_k(self, shape):
        if shape not in self._phi_k:
            truth = torch.as_tensor(sdb.load_truth(self.conn, shape), dtype=torch.float32)
            self._phi_k[shape] = target_coeffs_from_grid(truth, self.k_idx)
        return self._phi_k[shape]

    def basis(self, nxi):
        if nxi not in self._basis:
            row = self.conn.execute(
                "SELECT matrix, n_points FROM basis WHERE nxi=?", (nxi,)).fetchone()
            self._basis[nxi] = np.frombuffer(row[0], dtype=np.float32).reshape(row[1], nxi)
        return self._basis[nxi]

    def load(self, shape, variant):
        """-> (n_states,) step indices, (n_cand, n_states) float array (NaN outside a log-spaced
        checkpoint subset, see `_checkpoint_indices` -- evaluating `ergodic_term` at EVERY stored
        state does not scale with `n_iters`: at 4000 iterations it took ~35-40 min/shape instead
        of the few minutes the original 2000-iteration runs took (2026-10-12, Philipp's
        4000-iteration continuation run). A log-spaced subset renders visually identically (the
        plots are log-x anyway) at a cost independent of `n_iters`, like `compute_convergence.py`
        already does for its own, separate checkpoint grid."""
        key = (shape, variant)
        if key in self._series:
            return self._series[key]
        rows = list(sdb.iter_runs(self.conn, shape=shape, variant=variant, with_states=True))
        if not rows:
            self._series[key] = (None, None)
            return self._series[key]
        phi_k = self.phi_k(shape)
        n_states = max(r['n_states'] for r in rows)
        mat = np.full((len(rows), n_states), np.nan, dtype=np.float32)
        groups = {}
        for i, r in enumerate(rows):
            groups.setdefault((r['n_states'], r['log_space'], r['nxi']), []).append((i, r))
        for (ns, log_space, nxi), items in groups.items():
            idx = _checkpoint_indices(ns)
            states = np.stack([r['cps'][idx] for _, r in items])       # (G, len(idx), nxi, 2)
            if log_space == 'cps':
                dense = np.einsum('pi,gsid->gspd', self.basis(nxi), states)   # (G, K, T, 2)
            else:
                dense = states
            G, K, T = dense.shape[0], dense.shape[1], dense.shape[2]
            flat = torch.as_tensor(dense.reshape(G * K, T, 2), dtype=torch.float32)
            # `fourier_basis` materialises a (batch, T, M, 2) intermediate (M =
            # K_DEFAULT**2 = 100 modes); chunk it the same way
            # `run_mission_eval.py::ergodic_E_batch` does for its own large batches.
            chunks = []
            for c0 in range(0, flat.shape[0], 2500):
                chunks.append(ergodic_term(flat[c0:c0 + 2500], self.k_idx, self.Lambda, phi_k))
            e = torch.cat(chunks).numpy().reshape(G, K)
            for g, (i, _r) in enumerate(items):
                mat[i, idx] = e[g]
        out = (np.arange(n_states), mat)
        self._series[key] = out
        return out


#: (series label used in plots/summary, y-axis label, loader(shape, variant) ->
#: (steps, mat), yscale kwargs or None). Smoothness energy spans two regimes
#: that a linear axis can't show together -- several variants stay near 0
#: throughout (B-spline parameterisation, or the smoothness force) while the
#: unconstrained raw-waypoint variant grows past 20; a plain linear axis
#: flattens the near-0 group into visual noise. `symlog` keeps a linear
#: region near 0 (`linthresh`, where the near-0 variants' differences are
#: still what matters) and compresses everything beyond it logarithmically,
#: so both regimes stay legible on one axis.
def _metrics(ergo):
    return (
        ('smooth_series', 'Smoothness energy (lower = smoother)',
         lambda shape, variant: load_series(ergo.conn, shape, variant, 'smooth_series'),
         dict(value='symlog', linthresh=0.3)),
        ('path_len_series', 'Path length',
         lambda shape, variant: load_series(ergo.conn, shape, variant, 'path_len_series'),
         None),
        ('ergodic_series', 'Ergodic error vs. true density (lower = better coverage)',
         ergo.load, None),
    )


def _apply_yscale(ax, yscale):
    if yscale is not None:
        ax.set_yscale(**yscale)


def plot_shape_series(conn, shape, variants, out_path, ergo):
    metrics = _metrics(ergo)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), facecolor='white')
    for ax, (_key, ylabel, loader, yscale) in zip(axes, metrics):
        for variant in variants:
            steps, mat = loader(shape, variant)
            if steps is None:
                continue
            valid = ~np.all(np.isnan(mat), axis=0)
            s, m = steps[valid], mat[:, valid]
            mean, std = np.nanmean(m, axis=0), np.nanstd(m, axis=0)
            color = VARIANT_COLORS[variant]
            if variant == 'cfm_only':
                ax.axhline(float(mean[0]), color=color, lw=1.5, ls='--',
                          label=VARIANT_LABELS[variant])
                ax.axhspan(float(mean[0] - std[0]), float(mean[0] + std[0]),
                          color=color, alpha=0.08)
            else:
                ax.plot(s, mean, color=color, lw=VARIANT_LW[variant],
                       ls=VARIANT_LINESTYLES[variant], label=VARIANT_LABELS[variant])
                ax.fill_between(s, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel('SVGD iteration')
        ax.set_ylabel(ylabel)
        _apply_yscale(ax, yscale)
        _style(ax)
    axes[0].legend(fontsize=6.5, loc='upper right', frameon=False)
    fig.suptitle(f'Smoothness/ergodicity comparison -- {shape} (mean ± std over 30 trajectories)',
                color='#1A1A2E', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)


def plot_aggregate_series(conn, shapes, variants, out_path, ergo):
    """Mean-of-per-shape-means across all holdout shapes, +/- std across shapes."""
    metrics = _metrics(ergo)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), facecolor='white')
    for ax, (_key, ylabel, loader, yscale) in zip(axes, metrics):
        for variant in variants:
            per_shape_means = []
            steps_ref = None
            for shape in shapes:
                steps, mat = loader(shape, variant)
                if steps is None:
                    continue
                valid = ~np.all(np.isnan(mat), axis=0)
                per_shape_means.append(np.nanmean(mat[:, valid], axis=0))
                steps_ref = steps[valid]
            if not per_shape_means:
                continue
            n = min(len(m) for m in per_shape_means)
            M = np.stack([m[:n] for m in per_shape_means])
            s = steps_ref[:n]
            mean, std = M.mean(axis=0), M.std(axis=0)
            color = VARIANT_COLORS[variant]
            if variant == 'cfm_only':
                ax.axhline(float(mean[0]), color=color, lw=1.5, ls='--',
                          label=VARIANT_LABELS[variant])
            else:
                ax.plot(s, mean, color=color, lw=VARIANT_LW[variant],
                       ls=VARIANT_LINESTYLES[variant], label=VARIANT_LABELS[variant])
                ax.fill_between(s, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_xlabel('SVGD iteration')
        ax.set_ylabel(ylabel)
        _apply_yscale(ax, yscale)
        _style(ax)
    axes[0].legend(fontsize=6.5, loc='upper right', frameon=False)
    fig.suptitle(f'Smoothness/ergodicity comparison -- averaged over {len(shapes)} holdout '
                'shapes (mean ± std across shapes)', color='#1A1A2E', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)


def plot_trajectory_distribution(conn, shape, variant, step_fracs, out_path):
    """Ergodicity view of the 30-trajectory distribution at a few steps (replaces the former
    mean-trajectory + covariance-ellipse plot, 2026-10-11: the pointwise mean of 30 different
    trajectories is not a trajectory -- it collapses towards the centre -- and the ellipses
    covered the density). Top row: all 30 trajectories (thin) and the MEDIAN candidate by
    Hellinger distance (bold; an honest representative, not the best). Bottom row: that
    candidate's time-averaged visitation density with the target support as contour -- what
    ergodicity is about: time spent vs. target mass. Numbers from
    ergodic_validation/ergodicity_metrics.py (median over the 30 candidates)."""
    import ergodicity_metrics as em
    from scipy import ndimage
    rows = list(sdb.iter_runs(conn, shape=shape, variant=variant, with_states=True))
    if not rows:
        return
    truth = sdb.load_truth(conn, shape)
    nxi = rows[0]['nxi']
    log_space = rows[0]['log_space']
    n_states = rows[0]['n_states']
    B = None
    if log_space == 'cps':
        basis_row = conn.execute(
            "SELECT matrix, n_points FROM basis WHERE nxi=?", (nxi,)).fetchone()
        B = np.frombuffer(basis_row[0], dtype=np.float32).reshape(basis_row[1], nxi)
    steps = sorted(set(int(round(f * (n_states - 1))) for f in step_fracs))
    fig, axes = plt.subplots(2, len(steps), figsize=(3.0 * len(steps), 6.3), facecolor='white',
                             squeeze=False)
    R = truth.shape[0]
    for j, step in enumerate(steps):
        states = np.stack([r['cps'][min(step, r['cps'].shape[0] - 1)] for r in rows])  # (C, nxi, 2)
        dense = (np.einsum('pi,cid->cpd', B, states) if log_space == 'cps' else states)
        m = em.evaluate(dense, truth)
        h = m['hellinger']
        i_med = int(np.argsort(h)[len(h) // 2])
        ax = axes[0, j]
        viz.style_axes(ax)
        viz.draw_density(ax, truth)
        for c in range(dense.shape[0]):
            ax.plot(dense[c, :, 0], dense[c, :, 1], color='#00C853', lw=0.7, alpha=0.3)
        ax.plot(dense[i_med, :, 0], dense[i_med, :, 1], color='#00C853', lw=2.2, alpha=0.95)
        ax.set_title(f'step {step}/{n_states - 1}: Hellinger {np.median(h):.2f}, '
                     f'time outside {np.median(m["low_excess"]):+.2f}', color='#1A1A2E', fontsize=8)
        ax = axes[1, j]
        vis = ndimage.gaussian_filter(em.visitation(dense[i_med]), em.BLUR * em.GRID)
        ax.imshow(vis, origin='lower', extent=[0, 1, 0, 1], cmap=viz.CMAP)
        ax.contour(np.linspace(0, 1, R), np.linspace(0, 1, R), truth, levels=[0.1],
                   colors='#1565C0', linewidths=0.8)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f'time spent, median candidate (H {h[i_med]:.2f})', color='#1A1A2E', fontsize=8)
    fig.suptitle(f'{VARIANT_LABELS.get(variant, variant)} -- {shape}: 30 trajectories (bold: median '
                 'candidate) and its time-averaged visitation density (blue: target support)',
                 color='#1A1A2E', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=120, facecolor='white')
    plt.close(fig)


def write_summary_csv(conn, shapes, variants, out_path, ergo):
    """Writes summary.csv AND the same rows into the DB's `summary` table
    (`smoothness_db.save_summary`) -- the DB stays the complete, durable record of a run,
    the CSV the quick plain-text one."""
    db_rows = []
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['shape', 'variant', 'n_cand', 'smooth_final_mean', 'smooth_final_std',
                   'path_len_final_mean', 'path_len_final_std',
                   'ergodic_final_mean', 'ergodic_final_std'])
        for shape in shapes:
            for variant in variants:
                rows = list(sdb.iter_runs(conn, shape=shape, variant=variant))
                if not rows:
                    continue
                sm = np.array([r['smooth_series'][np.isfinite(r['smooth_series'])][-1]
                              for r in rows])
                pl = np.array([r['path_len_series'][np.isfinite(r['path_len_series'])][-1]
                              for r in rows])
                _steps, emat = ergo.load(shape, variant)
                e_final = emat[:, np.where(~np.all(np.isnan(emat), axis=0))[0][-1]]
                vals = [shape, variant, len(rows), float(sm.mean()), float(sm.std()),
                       float(pl.mean()), float(pl.std()),
                       float(np.nanmean(e_final)), float(np.nanstd(e_final))]
                w.writerow(vals)
                db_rows.append(dict(zip(
                    ['shape', 'variant', 'n_cand', 'smooth_final_mean', 'smooth_final_std',
                     'path_len_final_mean', 'path_len_final_std', 'ergodic_final_mean',
                     'ergodic_final_std'], vals)))
    sdb.save_summary(conn, db_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--dist_steps', type=str, default='0,0.1,0.25,1.0',
                    help='Step fractions for the trajectory-distribution plot.')
    args = ap.parse_args()

    out_dir = os.path.join(_here, 'results', args.out_tag)
    db_path = os.path.join(out_dir, 'smoothness_comparison.db')
    conn = sdb.open_db(db_path)
    cfg = sdb.get_meta(conn, 'config', {})
    shapes = cfg.get('shapes') or sorted({r[0] for r in conn.execute("SELECT DISTINCT shape FROM runs")})
    variants = cfg.get('variants') or sorted({r[0] for r in conn.execute("SELECT DISTINCT variant FROM runs")})
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'per_shape'), exist_ok=True)

    ergo = ErgodicSeriesCache(conn)

    write_summary_csv(conn, shapes, variants, os.path.join(out_dir, 'summary.csv'), ergo)
    print(f"[plot_smooth] summary.csv written ({len(shapes)} shapes x {len(variants)} variants)")

    plot_aggregate_series(conn, shapes, variants, os.path.join(plot_dir, 'aggregate.png'), ergo)
    print("[plot_smooth] aggregate.png written")

    step_fracs = [float(x) for x in args.dist_steps.split(',')]
    # any variant that actually ran SVGD (more than the single `cfm_only`
    # state) gets a distribution plot -- generic over both experiment rounds'
    # variant names, not a hardcoded round-1 list.
    dist_variants = [v for v in variants if v != 'cfm_only']
    for shape in shapes:
        plot_shape_series(conn, shape, variants, os.path.join(plot_dir, 'per_shape', f'{shape}.png'), ergo)
        for variant in dist_variants:
            plot_trajectory_distribution(
                conn, shape, variant, step_fracs,
                os.path.join(plot_dir, 'per_shape', f'{shape}_{variant}_dist.png'))
    print(f"[plot_smooth] {len(shapes)} per-shape plots written -> {plot_dir}")


if __name__ == '__main__':
    main()
