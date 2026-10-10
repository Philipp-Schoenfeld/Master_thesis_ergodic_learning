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
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Ellipse                              # noqa: E402
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
    'linear_sun_pointmass':         'Linear init + Sun SVGD (point-mass dynamics, waypoints)',
    'linear_sun_jerk':              'Linear init + Sun SVGD (jerk-penalised dynamics, waypoints)',
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
        """-> (n_states,) step indices, (n_cand, n_states) float array."""
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
            states = np.stack([r['cps'] for _, r in items])            # (G, ns, nxi, 2)
            if log_space == 'cps':
                dense = np.einsum('pi,gsid->gspd', self.basis(nxi), states)   # (G, ns, T, 2)
            else:
                dense = states
            G, T = dense.shape[0], dense.shape[2]
            flat = torch.as_tensor(dense.reshape(G * ns, T, 2), dtype=torch.float32)
            # `fourier_basis` materialises a (batch, T, M, 2) intermediate (M =
            # K_DEFAULT**2 = 100 modes) -- for one full (shape, variant) cell
            # (G*ns can be ~18000) that is ~1.8 GB in one shot, same concern
            # `run_mission_eval.py::ergodic_E_batch` already chunks for.
            chunks = []
            for c0 in range(0, flat.shape[0], 2500):
                chunks.append(ergodic_term(flat[c0:c0 + 2500], self.k_idx, self.Lambda, phi_k))
            e = torch.cat(chunks).numpy().reshape(G, ns)
            for g, (i, _r) in enumerate(items):
                mat[i, :ns] = e[g]
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
    """Mean trajectory + pointwise positional covariance (as ellipses) of the
    30-trajectory distribution, at a few steps -- the literal "Mean und
    Covariance einer Verteilung von 30 Trajektorien" from Philipp's note."""
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
    fig, axes = plt.subplots(1, len(steps), figsize=(3.2 * len(steps), 3.4), facecolor='white',
                             squeeze=False)
    axes = axes[0]
    for ax, step in zip(axes, steps):
        viz.style_axes(ax)
        viz.draw_density(ax, truth)
        states = np.stack([r['cps'][min(step, r['cps'].shape[0] - 1)] for r in rows])  # (C, nxi, 2)
        dense = (np.einsum('pi,cid->cpd', B, states) if log_space == 'cps' else states)
        mean = dense.mean(axis=0)
        for c in range(dense.shape[0]):
            ax.plot(dense[c, :, 0], dense[c, :, 1], color=VARIANT_COLORS[variant],
                   lw=0.6, alpha=0.18)
        ax.plot(mean[:, 0], mean[:, 1], color='#1A1A2E', lw=2.0)
        stride = max(1, dense.shape[1] // 10)
        for p in range(0, dense.shape[1], stride):
            pts = dense[:, p, :]
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)          # ascending: vals[1] = major axis
            vals = np.clip(vals, 0, None)
            angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))   # direction of vals[1]
            w = 2 * np.sqrt(vals[1]) * 2.0            # major axis, along `angle`
            h = 2 * np.sqrt(vals[0]) * 2.0            # minor axis; 2 std, scaled x2 for visibility
            ell = Ellipse(mean[p], width=max(w, 1e-3), height=max(h, 1e-3), angle=angle,
                         facecolor=VARIANT_COLORS[variant], alpha=0.25, edgecolor='none')
            ax.add_patch(ell)
        ax.set_title(f'step {step}/{n_states - 1}', color='#1A1A2E', fontsize=9)
    fig.suptitle(f'{VARIANT_LABELS[variant]} -- {shape}: mean trajectory + '
                'positional covariance (30 samples)', color='#1A1A2E', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)


def write_summary_csv(conn, shapes, variants, out_path, ergo):
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
                w.writerow([shape, variant, len(rows), float(sm.mean()), float(sm.std()),
                           float(pl.mean()), float(pl.std()),
                           float(np.nanmean(e_final)), float(np.nanstd(e_final))])


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
