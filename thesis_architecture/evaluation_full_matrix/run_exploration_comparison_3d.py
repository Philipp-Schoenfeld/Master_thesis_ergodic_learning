#!/usr/bin/env python3
r"""
run_exploration_comparison_3d.py
================================
The exploration comparison of `run_exploration_comparison.py` (five-round annealed replanning
mission, zero prior knowledge, kappa 1 -> 0.05, full execution per round, 25 heldout shapes x 7
variants x {RBF, Attentive-Kernel} belief, same seeds) in 3D: the heldout densities lie on a planar
patch in the unit cube (`plane3d.py`, default the flat training plane z = 0.5) and every method
plans, measures and is scored on 3D coordinates with the 3D-trained networks. Nothing is planned
in 2D and lifted; the figures show the patch in its own (u, v) chart, so they look like the 2D ones.

Methods (same shape / seed / belief / budget per row):
  networks + 3D TSVEC-SVGD refine (cloud cutoff 0.2, plane constraint as inference force)
    unet630      U-Net D512, LM-teacher DB, Sinkhorn lambda = 630 (3D counterpart of 2D unet630)
    dit630       DiT-B D768, same data / loss / schedule (counterpart of 2D dit630)
    dits_old     DiT-S D384, old LM DB (best 3D active-sensing net of 2026-11-04)
    nooffset200  U-Net D384, Fourier ergodic loss, 200-surface DB (3D GUI default)
    selfsup      3D self-supervised single-pass generator (no start input) + TSVEC
  search without a prior
    lintsvec     straight rays in the plane + the same 3D TSVEC-SVGD
    linadam      straight rays + 3D Adam-SVGD (reference SVGD, beta 100, lr 2e-3)
  model-free (no network, no SVGD), all `--walk_len` plane units per round
    rw           random walk on the patch
    lawn         lawnmower, one coarse sweep per round
    gpgoal       straight lines to points drawn from the GP posterior mean
    linray       best straight ray by 3D ergodic error against the round's target
    tsp          NEW: the `--tsp_points` highest points of the GP posterior mean (non-maximum
                 suppression `--tsp_min_dist`), connected by the shortest open travelling-salesman
                 path from the agent (exact Held-Karp up to 12 points), cut to the budget
    tsp_fill     the same, but the next-highest maxima are added until the shortest tour reaches
                 the budget (nearest neighbour + 2-opt beyond 12 points), so it drives as far as
                 the other baselines

Metrics: the columns of the 2D runner (same names, so `plot_exploration_summary.py --dir <out>`
works unchanged) with lengths / radii in plane units and ergodic errors from the 3D cube metric
(K = 10 per axis, not comparable to the 2D E values), plus `plane_dev_max` (largest distance of the
selected candidate from the plane before the final snap, plane units; 0 for the path baselines).

    python run_exploration_comparison_3d.py --smoke                 # 2 shapes, 2 rounds, tiny budget
    python run_exploration_comparison_3d.py --dry_run               # missions still to compute
    python run_exploration_comparison_3d.py                         # everything (resumable)
    python run_exploration_comparison_3d.py --plane tilted          # 50 deg tilted plane (own DB)
    python run_exploration_comparison_3d.py --plot_only
    python run_exploration_comparison_3d.py --merge_from other.db   # add-only merge of a job's DB
"""
import argparse
import csv
import math
import os
import sqlite3
import time
import types

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402

import mission_plane3d as mp            # noqa: E402  (sets up the 3D import path first)
import plane3d as p3                    # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))

NETWORK_METHODS = ('unet630', 'dit630', 'dits_old', 'nooffset200', 'selfsup')
RAY_METHODS = ('linadam', 'lintsvec')
PATH_METHODS = ('rw', 'lawn', 'gpgoal', 'linray', 'tsp', 'tsp_fill')
METHODS = PATH_METHODS + RAY_METHODS + NETWORK_METHODS
DEFAULT_METHODS = METHODS
KERNELS = ('rbf', 'ak')
METHOD_LABEL = {'rw': 'Random walk', 'lawn': 'Lawnmower', 'gpgoal': 'Line to GP-sampled point',
                'linray': 'Best straight ray (no search)', 'tsp': 'TSP through GP maxima',
                'tsp_fill': 'TSP through GP maxima, budget-filled',
                'linadam': 'Linear init + Adam-SVGD', 'lintsvec': 'Linear init + TSVEC (no prior)',
                **{k: v[1] for k, v in mp.NETS3D.items()}}
METHOD_COLOR = {'unet630': '#1565C0', 'dit630': '#6A1B9A', 'dits_old': '#AB47BC',
                'nooffset200': '#26C6DA', 'selfsup': '#C62828', 'rw': '#888888', 'lawn': '#E65100',
                'gpgoal': '#F9A825', 'linray': '#8D6E63', 'tsp': '#D81B60', 'tsp_fill': '#880E4F', 'linadam': '#00897B',
                'lintsvec': '#2E7D32'}
CUTOFF = 0.20                     # sample_annealed_mission_overview.CUT_MID
TEXT, MUTED, SPINE = '#1A1A2E', '#555555', '#CCCCCC'

METRIC_COLS = [
    'E_truth', 'E_truth_explore', 'E_truth_exploit', 'E_belief', 'E_plan',
    'swept03', 'swept06', 'swept10', 'swept_hi50', 'swept_hi75', 'cov',
    'belief_rmse', 'belief_corr', 'belief_tv', 'belief_iou_hi', 'belief_nll',
    'belief_cov95', 'belief_sd_mean', 'n_obs',
    'path_len', 'seg_len', 'out_gt_frac', 'in_hi50_frac', 'in_hi75_frac',
    'seg_out_gt_frac', 'seg_in_hi50_frac', 'seg_in_hi75_frac', 'seg_smooth', 'seg_curv',
    'revisit_frac', 'E_init_best', 'E_init_med', 'E_ref_best', 'E_ref_med',
    't_target', 't_init', 't_refine', 't_belief', 't_round', 'plane_dev_max',
]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS truths (shape TEXT PRIMARY KEY, res INTEGER NOT NULL, truth BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shape TEXT NOT NULL, kernel TEXT NOT NULL, method TEXT NOT NULL,
    variant INTEGER NOT NULL, seed INTEGER, n_rounds INTEGER, n_init INTEGER, n_iters INTEGER,
    seconds REAL, reached90 INTEGER, reached99 INTEGER,
    path_len_90 REAL, path_len_99 REAL, created_at REAL,
    UNIQUE(shape, kernel, method, variant));
CREATE TABLE IF NOT EXISTS rounds (
    run_id INTEGER NOT NULL REFERENCES runs(id), round INTEGER NOT NULL, kappa REAL,
    start_x REAL, start_y REAL, start_z REAL,
    {', '.join(f'{c} REAL' for c in METRIC_COLS)},
    segment BLOB,       -- (n, 3) float32, the path driven in this round, 3D WORLD coordinates
    phi BLOB,           -- (G, G) float16, target density the round was planned for (patch grid)
    mu BLOB, sd BLOB,   -- (G, G) float16, belief after the round's measurements (patch grid)
    PRIMARY KEY (run_id, round));
"""


def to_blob(x, dtype):
    return np.ascontiguousarray(np.asarray(x, dtype=dtype)).tobytes()


def _f(v):
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


class ResultDB:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.execute('PRAGMA journal_mode=WAL')
        have = {r[1] for r in self.conn.execute('PRAGMA table_info(rounds)')}
        for c in METRIC_COLS:
            if c not in have:
                self.conn.execute(f'ALTER TABLE rounds ADD COLUMN {c} REAL')
        self.conn.commit()

    def has_run(self, shape, kernel, method, variant):
        return self.conn.execute('SELECT 1 FROM runs WHERE shape=? AND kernel=? AND method=? AND variant=?',
                                 (shape, kernel, method, variant)).fetchone() is not None

    def put_truth(self, shape, truth):
        t = truth.detach().cpu().numpy().astype(np.float32)
        self.conn.execute('INSERT OR REPLACE INTO truths VALUES (?,?,?)', (shape, t.shape[-1], to_blob(t, np.float32)))
        self.conn.commit()

    def put_run(self, shape, kernel, method, variant, res, cfg):
        cur = self.conn.execute(
            'INSERT INTO runs (shape,kernel,method,variant,seed,n_rounds,n_init,n_iters,seconds,reached90,'
            'reached99,path_len_90,path_len_99,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (shape, kernel, method, variant, res['seed'], cfg.n_rounds, cfg.n_init, cfg.n_iters,
             res['seconds'], res['reached90'], res['reached99'], res['len90'], res['len99'], time.time()))
        rid = cur.lastrowid
        cols = ['run_id', 'round', 'kappa', 'start_x', 'start_y', 'start_z'] + METRIC_COLS + ['segment', 'phi', 'mu', 'sd']
        for r, row in enumerate(res['rows']):
            st = res['starts'][r]
            vals = [rid, r, res['kappas'][r], float(st[0]), float(st[1]), float(st[2])] + \
                   [_f(row.get(c)) for c in METRIC_COLS] + \
                   [to_blob(res['segments'][r], np.float32), to_blob(res['phis'][r], np.float16),
                    to_blob(res['mus'][r], np.float16), to_blob(res['sds'][r], np.float16)]
            self.conn.execute(f'INSERT INTO rounds ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})', vals)
        self.conn.commit()

    def load_run(self, shape, kernel, method, variant=0):
        run = self.conn.execute('SELECT id FROM runs WHERE shape=? AND kernel=? AND method=? AND variant=?',
                                (shape, kernel, method, variant)).fetchone()
        if run is None:
            return None
        rows, segs, phis, kappas = [], [], [], []
        for rec in self.conn.execute(f'SELECT round,kappa,{",".join(METRIC_COLS)},segment,phi FROM rounds '
                                     'WHERE run_id=? ORDER BY round', (run[0],)):
            d = dict(zip(['round', 'kappa'] + METRIC_COLS, rec[:2 + len(METRIC_COLS)]))
            d = {k: (float('nan') if v is None else v) for k, v in d.items()}
            d['swept'] = d['swept06']
            rows.append(d)
            kappas.append(d['kappa'])
            segs.append(torch.from_numpy(np.frombuffer(rec[-2], dtype=np.float32).reshape(-1, 3).copy()))
            phi = np.frombuffer(rec[-1], dtype=np.float16).astype(np.float32)
            phis.append(torch.from_numpy(phi.reshape(int(math.isqrt(phi.size)), -1)))
        return dict(rows=rows, segments=segs, phis=phis, kappas=kappas)

    def load_truth(self, shape):
        res, blob = self.conn.execute('SELECT res, truth FROM truths WHERE shape=?', (shape,)).fetchone()
        return torch.from_numpy(np.frombuffer(blob, dtype=np.float32).reshape(res, res).copy())

    def merge_from(self, src):
        """Add-only: runs of `src` whose key is missing here, with all their rounds; missing truths."""
        self.conn.execute('ATTACH DATABASE ? AS src', (os.path.abspath(src),))
        src_cols = {r[1] for r in self.conn.execute('PRAGMA src.table_info(rounds)')}
        cols = [c for c in [r[1] for r in self.conn.execute('PRAGMA main.table_info(rounds)')]
                if c in src_cols and c != 'run_id']
        new = self.conn.execute(
            'SELECT s.id FROM src.runs s WHERE NOT EXISTS (SELECT 1 FROM main.runs d WHERE d.shape=s.shape '
            'AND d.kernel=s.kernel AND d.method=s.method AND d.variant=s.variant) ORDER BY s.id').fetchall()
        rcols = [r[1] for r in self.conn.execute('PRAGMA main.table_info(runs)') if r[1] != 'id']
        for (sid,) in new:
            cur = self.conn.execute(f'INSERT INTO main.runs ({",".join(rcols)}) SELECT {",".join(rcols)} '
                                    'FROM src.runs WHERE id=?', (sid,))
            self.conn.execute(f'INSERT INTO main.rounds (run_id,{",".join(cols)}) SELECT ?,{",".join(cols)} '
                              'FROM src.rounds WHERE run_id=?', (cur.lastrowid, sid))
        self.conn.execute('INSERT OR IGNORE INTO main.truths SELECT * FROM src.truths')
        self.conn.execute('INSERT OR IGNORE INTO main.meta SELECT * FROM src.meta')
        self.conn.commit()
        self.conn.execute('DETACH DATABASE src')
        return len(new)


# ===========================================================================
# One mission
# ===========================================================================

def path_source(method, a, plane, ee, seed):
    if method == 'rw':
        return dict(path_fn=mp.random_walk_fn(plane, a.walk_len, seed))
    if method == 'lawn':
        return dict(path_fn=mp.lawnmower_fn(plane, a.walk_len, a.n_rounds))
    if method == 'gpgoal':
        return dict(path_fn=mp.gp_goal_fn(plane, a.walk_len, seed, source=a.gp_goal_source))
    if method == 'linray':
        return dict(path_fn=mp.best_ray_fn(plane, ee, a.n_init, a.walk_len))
    if method in ('tsp', 'tsp_fill'):
        return dict(path_fn=mp.tsp_fn(plane, a.walk_len, seed, a.tsp_points, a.tsp_min_dist, a.tsp_source,
                                      fill=method == 'tsp_fill'))
    if method in RAY_METHODS:
        return dict(init_fn=mp.ray_init_fn(plane, a.n_init, a.walk_len))
    return {}


def run_one(method, kernel, shape, truth, planner, ctx, a, kappas, seed):
    plane, ee, device = ctx.plane, ctx.ee, ctx.device
    evl = mp.Evaluator3D(truth, plane, ee, device)
    rows, mus, sds = [], [], []

    def on_round(r, belief, segments, diag):
        driven = torch.cat(segments, dim=0)
        earlier = torch.cat(segments[:-1], dim=0) if r > 0 else None
        row, mu, sd = evl.round_metrics(driven, segments[-1], earlier, belief, diag)
        rows.append(row)
        mus.append(mu.numpy())
        sds.append(sd.numpy())

    start_uv = (0.06, 0.06) if method == 'lawn' else (0.5, 0.5)
    t0 = time.time()
    segments, phis = mp.run_mission(
        f'{kernel}/{method}/{shape}', truth, plane, kernel, planner, ctx.refiner_for(method), ee,
        ctx.B32, ctx.settings, device, a.n_rounds, a.n_init, a.n_iters, kappas, start_uv, seed,
        cutoff=CUTOFF if method in NETWORK_METHODS else 0.0, on_round=on_round,
        **path_source(method, a, plane, ee, seed))
    seconds = time.time() - t0
    for r, row in enumerate(rows):
        row['E_plan'] = ee.score(torch.cat(segments[:r + 1]).to(device),
                                 ee.target_coeffs(phis[r].to(device), plane))['E_ergodic_total']
    full = torch.cat(segments, dim=0)
    lens = {}
    for thr in (0.90, 0.99):
        frac, reached = evl.steps_to_coverage(full, thr)
        lens[thr] = (int(reached), frac * mp.path_length(full) / plane.s if reached else float('nan'))
    starts = [plane.to_world(np.array(start_uv))] + [s[-1].numpy() for s in segments[:-1]]
    return dict(rows=rows, segments=[s.numpy() for s in segments], phis=[p.numpy() for p in phis],
                mus=mus, sds=sds, kappas=kappas, starts=starts, seconds=seconds, seed=seed,
                reached90=lens[0.90][0], reached99=lens[0.99][0], len90=lens[0.90][1], len99=lens[0.99][1])


def make_context(a, device):
    plane = p3.Plane(a.plane)
    B_np = mp.bspline_basis_matrix(mp.NXI, mp.N_POINTS, mp.DEGREE)
    ctx = types.SimpleNamespace(
        plane=plane, device=device, ee=p3.Ergodic3D(a.K, device=device),
        B32=torch.as_tensor(B_np, dtype=torch.float32, device=device),
        settings=mp.default_settings(ak_fit_iters=a.ak_fit_iters),
        bs_tsvec=p3.SolverBatch3D(p3.TsvecSvgd3D(B_np, plane, K=a.K, device=device, w_plane=a.w_plane)),
        bs_adam=p3.SolverBatch3D(p3.AdamSvgd3D(B_np, plane, K=a.K, device=device)))
    ctx.refiner_for = lambda m: ctx.bs_adam if m == 'linadam' else ctx.bs_tsvec
    return ctx


# ===========================================================================
# Figures (project style, plane chart)
# ===========================================================================

def white_inferno():
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    ramp = np.linspace(0, 1, 60)[:, None]
    inf[:60, :3] = (1 - ramp) * np.ones((60, 3)) + ramp * inf[:60, :3]
    return mcolors.LinearSegmentedColormap.from_list('white_inferno', inf)


ROUND_CMAP = mcolors.LinearSegmentedColormap.from_list('explore_to_exploit', ['#5E35B1', '#1E88E5', '#00ACC1', '#00C853'])


def round_colors(n):
    return ['#00C853'] if n == 1 else [mcolors.to_hex(ROUND_CMAP(t)) for t in np.linspace(0, 1, n)]


def _style(ax):
    ax.set_facecolor('white')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(SPINE)


def _uv(plane, seg):
    return plane.to_uv(torch.as_tensor(seg).double()).numpy()


TABLE_ROWS = (('E_truth', 'E_truth ↓', '{:.2f}'), ('E_belief', 'E_belief ↓', '{:.2f}'),
              ('swept', 'swept mass ↑', '{:.0%}'), ('cov', 'coverage dist ↓', '{:.3f}'),
              ('path_len', 'path length', '{:.1f}'), ('out_gt_frac', 'path outside GT ↓', '{:.0%}'),
              ('in_hi50_frac', 'path in dense GT ↑', '{:.0%}'), ('belief_rmse', 'belief RMSE ↓', '{:.3f}'),
              ('belief_corr', 'belief corr ↑', '{:.2f}'), ('E_init_best', 'E_target raw ↓', '{:.2f}'),
              ('E_ref_best', 'E_target after SVGD ↓', '{:.2f}'), ('plane_dev_max', 'off-plane dev.', '{:.1e}'))


def _metrics_table(ax, rows):
    ax.axis('off')
    n = len(rows)
    cells = [['–' if row.get(k, float('nan')) != row.get(k, float('nan')) else f.format(row[k]) for row in rows]
             for k, _, f in TABLE_ROWS]
    tab = ax.table(cellText=cells, rowLabels=[lab for _, lab, _ in TABLE_ROWS],
                   colLabels=[f'R{r}' for r in range(n)], cellLoc='center', loc='center', bbox=[0.30, 0.04, 0.70, 0.88])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor(SPINE)
        cell.set_linewidth(0.6)
        cell.get_text().set_color(TEXT)
        if i == 0:
            cell.set_facecolor('#F3F3F6')
        if j == n - 1 and i > 0:
            cell.get_text().set_fontweight('bold')
    ax.set_title('Metrics per round (last = final)', fontsize=10, color=TEXT)


def _draw_summary(ax, plane, truth, segs, colors, kappas, cmap, legend=True):
    ax.imshow(truth.numpy(), origin='lower', extent=[0, 1, 0, 1], cmap=cmap, vmin=0,
              vmax=float(truth.max()) or 1.0, alpha=0.55, zorder=1)
    for r, (seg, col, kap) in enumerate(zip(segs, colors, kappas)):
        uv = _uv(plane, seg)
        ax.plot(uv[:, 0], uv[:, 1], color=col, lw=2.0, alpha=0.9, zorder=3, label=f'Round {r}: κ = {kap:.3g}')
    _style(ax)
    if legend:
        ax.legend(loc='upper right', fontsize=7, frameon=True, facecolor='white', edgecolor='#ddd',
                  framealpha=0.85, handlelength=1.5)


def plot_shape(db, shape, kernel, methods, out_dir, plane):
    """Round-by-round overview of variant 0, one row per method (the 2D `plot_rows` layout)."""
    res = [(m, db.load_run(shape, kernel, m, 0)) for m in methods]
    res = [(m, r) for m, r in res if r is not None]
    if not res:
        return
    truth = db.load_truth(shape)
    n_rounds = len(res[0][1]['segments'])
    cmap, colors = white_inferno(), round_colors(n_rounds)
    fig, axes = plt.subplots(len(res), n_rounds + 2, squeeze=False, facecolor='white',
                             figsize=(3.4 * (n_rounds + 1) + 5.6, 3.7 * len(res)),
                             gridspec_kw=dict(width_ratios=[1] * (n_rounds + 1) + [1.65]))
    for i, (m, run) in enumerate(res):
        for r in range(n_rounds):
            ax = axes[i, r]
            ax.imshow(run['phis'][r].numpy(), origin='lower', extent=[0, 1, 0, 1], cmap=cmap, vmin=0,
                      vmax=1.0, alpha=0.55, zorder=1)
            uv = _uv(plane, run['segments'][r])
            ax.plot(uv[:, 0], uv[:, 1], color=colors[r], lw=2.2, alpha=0.95, zorder=3)
            ax.plot(uv[0, 0], uv[0, 1], 'o', color=colors[r], ms=5, zorder=4)
            _style(ax)
            ax.text(0.03, 0.97, f'κ = {run["kappas"][r]:.3g}', transform=ax.transAxes, fontsize=9, color=TEXT,
                    ha='left', va='top', zorder=5, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#ccc', alpha=0.85))
            if i == 0:
                ax.set_title(f'Round {r}', fontsize=10, color=TEXT)
            if r == 0:
                ax.set_ylabel(METHOD_LABEL[m], fontsize=10.5, color=TEXT)
        _draw_summary(axes[i, n_rounds], plane, truth, run['segments'], colors, run['kappas'], cmap)
        if i == 0:
            axes[i, n_rounds].set_title('All rounds (ground truth)', fontsize=10, color=TEXT)
        _metrics_table(axes[i, n_rounds + 1], run['rows'])
    kern = {'rbf': 'RBF', 'ak': 'Attentive-Kernel'}[kernel]
    d = plane.describe()
    fig.suptitle(f'3D exploration comparison on the {d["plane"]} plane (normal {d["normal"]}): {kern} belief, '
                 f'cloud cutoff {CUTOFF:g}\nshape {shape}; planned and scored in 3D world coordinates, drawn in '
                 "the plane's (u, v) chart; κ annealed, full execution, zero prior knowledge",
                 fontsize=12.5, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.95 if len(res) > 1 else 0.90])
    out = os.path.join(out_dir, kernel, f'{shape}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def plot_variants(db, shape, kernel, methods, out_dir, plane):
    n_var = db.conn.execute('SELECT MAX(variant)+1 FROM runs WHERE shape=? AND kernel=?', (shape, kernel)).fetchone()[0]
    if not n_var:
        return
    truth, cmap = db.load_truth(shape), white_inferno()
    fig, axes = plt.subplots(len(methods), n_var, squeeze=False, facecolor='white',
                             figsize=(2.3 * n_var, 2.5 * len(methods)))
    for i, m in enumerate(methods):
        for v in range(n_var):
            ax = axes[i, v]
            run = db.load_run(shape, kernel, m, v)
            if run is None:
                ax.axis('off')
                continue
            _draw_summary(ax, plane, truth, run['segments'], round_colors(len(run['segments'])),
                          run['kappas'], cmap, legend=False)
            fin = run['rows'][-1]
            ax.set_title(f'E={fin["E_truth"]:.1f}  swept {fin["swept06"]:.0%}', fontsize=8, color=TEXT)
            if v == 0:
                ax.set_ylabel(METHOD_LABEL[m], fontsize=9, color=TEXT)
    fig.suptitle(f'{shape}, {kernel.upper()} belief, {plane.key} plane: {n_var} variants per method '
                 '(all rounds on the ground truth)', fontsize=11, color=TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, 'variants', kernel, f'{shape}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=90, facecolor='white')
    plt.close(fig)


def plot_world(db, shape, kernel, methods, out_dir, plane):
    """The same missions (variant 0) in 3D world coordinates: the patch, the ground truth on it and
    the driven paths, one 3D panel per method -- the check that nothing lives in 2D."""
    runs = [(m, db.load_run(shape, kernel, m, 0)) for m in methods]
    runs = [(m, r) for m, r in runs if r is not None]
    if not runs:
        return
    truth = db.load_truth(shape)
    R = truth.shape[-1]
    pts = plane.grid_world(R).numpy()[::3]
    w = truth.reshape(-1).numpy()[::3]
    keep = w > 0.05
    ncol = min(4, len(runs))
    nrow = int(math.ceil(len(runs) / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 4.0 * nrow), facecolor='white')
    cmap = white_inferno()
    cor = plane.corners()
    for k, (m, run) in enumerate(runs):
        ax = fig.add_subplot(nrow, ncol, k + 1, projection='3d')
        ax.plot(cor[:, 0], cor[:, 1], cor[:, 2], color='#999', lw=0.8)
        ax.scatter(pts[keep, 0], pts[keep, 1], pts[keep, 2], c=w[keep], cmap=cmap, vmin=0, vmax=1, s=2,
                   alpha=0.55, linewidths=0)
        for seg, col in zip(run['segments'], round_colors(len(run['segments']))):
            s = seg.numpy()
            ax.plot(s[:, 0], s[:, 1], s[:, 2], color=col, lw=1.6, alpha=0.95)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_xlabel('x', fontsize=8, color=MUTED)
        ax.set_ylabel('y', fontsize=8, color=MUTED)
        ax.set_zlabel('z', fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_title(METHOD_LABEL[m], fontsize=9, color=TEXT)
    fig.suptitle(f'{shape}, {kernel.upper()} belief: driven paths in 3D world coordinates ({plane.key} plane)',
                 fontsize=11, color=TEXT)
    fig.tight_layout()
    out = os.path.join(out_dir, 'world3d', kernel, f'{shape}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=90, facecolor='white')
    plt.close(fig)


# ===========================================================================
# Tables / aggregate curves (same files as the 2D runner)
# ===========================================================================

def _nan(v):
    return float('nan') if v is None else float(v)


def fetch_table(db):
    cols = ['r.round', 'u.shape', 'u.kernel', 'u.method', 'u.variant'] + [f'r.{c}' for c in METRIC_COLS]
    names = ['round', 'shape', 'kernel', 'method', 'variant'] + METRIC_COLS
    return names, [dict(zip(names, rec)) for rec in
                   db.conn.execute(f'SELECT {",".join(cols)} FROM rounds r JOIN runs u ON u.id=r.run_id')]


CURVE_PANELS = (('E_truth', 'E vs. ground truth ↓'), ('E_belief', 'E vs. belief ↓'),
                ('swept06', 'GT mass within 0.06 of path ↑'), ('belief_rmse', 'belief RMSE ↓'),
                ('path_len', 'path length'), ('out_gt_frac', 'path outside GT ↓'),
                ('in_hi50_frac', 'path in dense GT (≥ 0.5) ↑'), ('belief_corr', 'belief corr ↑'))


def aggregate(db, out_dir, methods, ref='unet630'):
    names, recs = fetch_table(db)
    if not recs:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics_all.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(names)
        for d in recs:
            w.writerow([d[n] for n in names])
    n_rounds = max(d['round'] for d in recs) + 1
    cells = {}
    for d in recs:
        cells.setdefault((d['kernel'], d['method'], d['round']), {}).setdefault(d['shape'], []).append(d)
    keys = ['n_shapes', 'n_runs', 'mean', 'median', 'std_total', 'std_between_shapes', 'std_within_shape', 'min', 'max']
    with open(os.path.join(out_dir, 'summary_by_round.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'round', 'metric'] + keys)
        for (k, m, r), per in sorted(cells.items()):
            for c in METRIC_COLS:
                allv = np.array([_nan(d[c]) for ds in per.values() for d in ds])
                if np.isnan(allv).all():
                    continue
                sm = np.array([np.nanmean([_nan(d[c]) for d in ds]) for ds in per.values()])
                wi = np.array([np.nanstd([_nan(d[c]) for d in ds]) for ds in per.values()])
                w.writerow([k, m, r, c, len(per), len(allv), np.nanmean(allv), np.nanmedian(allv), np.nanstd(allv),
                            np.nanstd(sm), np.nanmean(wi), np.nanmin(allv), np.nanmax(allv)])
    with open(os.path.join(out_dir, 'summary_by_shape_final.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'shape', 'metric', 'n_variants', 'mean', 'std', 'min', 'max'])
        for (k, m, r), per in sorted(cells.items()):
            if r != n_rounds - 1:
                continue
            for shape, ds in sorted(per.items()):
                for c in METRIC_COLS:
                    v = np.array([_nan(d[c]) for d in ds])
                    if not np.isnan(v).all():
                        w.writerow([k, m, shape, c, len(v), np.nanmean(v), np.nanstd(v), np.nanmin(v), np.nanmax(v)])
    tcols = ['t_target', 't_init', 't_refine', 't_belief', 't_round']
    with open(os.path.join(out_dir, 'timing_by_method.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'n_rounds_timed'] + [f'{c}_mean' for c in tcols] + [f'{c}_median' for c in tcols])
        for k in KERNELS:
            for m in methods:
                sel = [d for d in recs if d['kernel'] == k and d['method'] == m and d['t_round'] is not None]
                if sel:
                    cols = {c: np.array([d[c] for d in sel], dtype=float) for c in tcols}
                    w.writerow([k, m, len(sel)] + [float(np.nanmean(cols[c])) for c in tcols]
                               + [float(np.nanmedian(cols[c])) for c in tcols])
    # mean path length per round of every method (to budget-match the baselines: --walk_len)
    with open(os.path.join(out_dir, 'seg_len_by_method.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'n_rounds', 'seg_len_mean', 'seg_len_median', 'plane_dev_max_max'])
        for k in KERNELS:
            for m in methods:
                sel = [d for d in recs if d['kernel'] == k and d['method'] == m]
                if sel:
                    sl = np.array([_nan(d['seg_len']) for d in sel])
                    pd = np.array([_nan(d['plane_dev_max']) for d in sel])
                    w.writerow([k, m, len(sel), float(np.nanmean(sl)), float(np.nanmedian(sl)),
                                float(np.nanmax(pd)) if not np.isnan(pd).all() else ''])
    final = {(d['kernel'], d['method'], d['shape'], d['variant']): d for d in recs if d['round'] == n_rounds - 1}
    with open(os.path.join(out_dir, f'paired_vs_{ref}_final.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'metric', 'better_is', 'n_pairs', f'share_better_than_{ref}', f'mean_diff_vs_{ref}'])
        for k in KERNELS:
            for m in methods:
                if m == ref:
                    continue
                for c, lower in (('E_truth', True), ('E_belief', True), ('swept06', False), ('belief_rmse', True),
                                 ('belief_corr', False), ('out_gt_frac', True), ('in_hi50_frac', False), ('cov', True)):
                    diffs = np.array([_nan(final[key][c]) - _nan(final[(k, ref) + key[2:]][c]) for key in final
                                      if key[0] == k and key[1] == m and (k, ref) + key[2:] in final])
                    diffs = diffs[~np.isnan(diffs)]
                    if len(diffs):
                        w.writerow([k, m, c, 'lower' if lower else 'higher', len(diffs),
                                    float(((diffs < 0) if lower else (diffs > 0)).mean()), float(diffs.mean())])
    for mode in ('mean', 'spread'):
        plot_curves(cells, n_rounds, out_dir, [m for m in methods if any(key[1] == m for key in cells)], mode)
    print(f'  tables + aggregate plots in {out_dir}')


def plot_curves(cells, n_rounds, out_dir, methods, mode):
    fig, axes = plt.subplots(len(KERNELS), len(CURVE_PANELS), squeeze=False, facecolor='white',
                             figsize=(3.3 * len(CURVE_PANELS), 3.2 * len(KERNELS)))
    xs = np.arange(n_rounds)
    for i, k in enumerate(KERNELS):
        for j, (c, title) in enumerate(CURVE_PANELS):
            ax = axes[i, j]
            for m in methods:
                mean, sem = [], []
                for r in range(n_rounds):
                    per = cells.get((k, m, r), {})
                    fn = np.nanmean if mode == 'mean' else np.nanstd
                    v = np.array([fn([_nan(d[c]) for d in ds]) for ds in per.values()])
                    v = v[~np.isnan(v)] if len(v) else v
                    mean.append(v.mean() if len(v) else np.nan)
                    sem.append(v.std() / math.sqrt(len(v)) if len(v) > 1 else 0.0)
                mean, sem = np.array(mean), np.array(sem)
                ax.plot(xs, mean, color=METHOD_COLOR[m], lw=2, marker='o', ms=3.5, label=METHOD_LABEL[m])
                ax.fill_between(xs, mean - sem, mean + sem, color=METHOD_COLOR[m], alpha=0.15, lw=0)
            ax.set_title(f'{title}\n{k.upper()} belief' if j == 0 else title, fontsize=9, color=TEXT)
            ax.set_xlabel('round', fontsize=8, color=MUTED)
            ax.set_xticks(xs)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=8, colors=MUTED)
            for s in ax.spines.values():
                s.set_color(SPINE)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=min(len(methods), 6), fontsize=9, frameon=False)
    fig.suptitle(('Mean over heldout shapes of the per-shape variant mean ± s.e.m. over shapes'
                  if mode == 'mean' else 'Spread between variants: std over the variants of a shape, averaged over shapes')
                 + ' (3D, path driven so far)', fontsize=11, color=TEXT)
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    fig.savefig(os.path.join(out_dir, 'aggregate_curves.png' if mode == 'mean' else 'aggregate_variant_spread.png'),
                dpi=120, facecolor='white')
    plt.close(fig)


# ===========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', type=str, default=','.join(DEFAULT_METHODS),
                   help='Comma-separated; known: ' + ', '.join(METHODS) + '. Finished rows are skipped.')
    p.add_argument('--kernels', type=str, default=','.join(KERNELS))
    p.add_argument('--plane', type=str, default='flat', choices=sorted(p3.PLANE_NORMALS),
                   help="Orientation of the patch: 'flat' (z = 0.5, the 3D training plane), 'tilted' "
                        "(50 deg), 'diagonal' (normal (1,1,1)). Each plane gets its own DB / folder.")
    p.add_argument('--n_shapes', type=int, default=None)
    p.add_argument('--shape_names', type=str, default=None)
    p.add_argument('--n_variants', type=int, default=7)
    p.add_argument('--n_rounds', type=int, default=5)
    p.add_argument('--n_init', type=int, default=8)
    p.add_argument('--n_iters', type=int, default=500, help='SVGD iterations per round (TSVEC and Adam).')
    p.add_argument('--kappa0', type=float, default=1.0)
    p.add_argument('--kappa1', type=float, default=0.05)
    p.add_argument('--walk_len', type=float, default=6.1,
                   help='Path budget per round of the model-free baselines and length of the rays, plane '
                        'units (6.1 = the 2D benchmark; see seg_len_by_method.csv for the 3D networks).')
    p.add_argument('--K', type=int, default=p3.K_DEFAULT, help='Fourier modes per axis of the 3D metric / SVGD.')
    p.add_argument('--w_plane', type=float, default=p3.W_PLANE, help='Weight of the plane attachment term in TSVEC.')
    p.add_argument('--ak_fit_iters', type=int, default=200)
    p.add_argument('--gp_goal_source', type=str, default='mean', choices=['mean', 'phi'])
    p.add_argument('--tsp_points', type=int, default=12,
                   help="'tsp': number of belief maxima per round (exact TSP up to 12); 'tsp_fill' starts with "
                        "these and adds maxima until the tour reaches the budget.")
    p.add_argument('--tsp_min_dist', type=float, default=0.10,
                   help="'tsp': non-maximum suppression radius between the picked points, plane units.")
    p.add_argument('--tsp_source', type=str, default='mean', choices=['mean', 'phi'],
                   help="'tsp' picks the maxima of the GP posterior mean ('mean') or of the round's target ('phi').")
    p.add_argument('--ckpt', action='append', default=[],
                   help='Checkpoint override "method=path" (repeatable); default: newest match of NETS3D.')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--max_minutes', type=float, default=None)
    p.add_argument('--db', type=str, default=None)
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--plot_only', action='store_true')
    p.add_argument('--no_plots', action='store_true', help='Skip the per-shape figures (tables still written).')
    p.add_argument('--merge_from', type=str, default=None,
                   help='Add-only merge of another 3D comparison DB into --db, then plots + tables.')
    p.add_argument('--smoke', action='store_true', help='2 shapes, 2 rounds, tiny budget, separate DB/folder.')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.smoke:
        a.n_rounds, a.n_init, a.n_iters, a.n_variants = 2, 2, 20, 1
        a.n_shapes = a.n_shapes or 2
    tag = f'exploration_comparison_3d_{a.plane}' + ('_smoke' if a.smoke else '')
    out_dir = a.out_dir or os.path.join(_here, 'plots', tag)
    db_path = a.db or os.path.join(out_dir, 'exploration_comparison.db')
    methods, kernels = a.methods.split(','), a.kernels.split(',')
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f'unknown method {m}; known: {METHODS}')
    overrides = dict(s.split('=', 1) for s in a.ckpt)
    db = ResultDB(db_path)
    if a.merge_from:
        print(f'merged {db.merge_from(a.merge_from)} new runs from {a.merge_from} into {db_path}')
        a.plot_only = True
    device = a.device
    labels = a.shape_names.split(',') if a.shape_names else None
    names, truths = mp.load_truth(labels=labels, n=a.n_shapes or 10_000)
    print(f'{len(names)} heldout shapes, plane={a.plane}, methods={methods}, kernels={kernels}, db={db_path}')
    plane = p3.Plane(a.plane)

    if not a.plot_only:
        kappas = [a.kappa0 if a.n_rounds <= 1 else a.kappa0 + (a.kappa1 - a.kappa0) * r / (a.n_rounds - 1)
                  for r in range(a.n_rounds)]
        todo = {m: [(k, i, n, v) for v in range(a.n_variants) for i, n in enumerate(names)
                    for k in kernels if not db.has_run(n, k, m, v)] for m in methods}
        if a.dry_run:
            for m in methods:
                print(f'  {m}: {len(todo[m])} missions to compute')
            return
        for m in methods:                       # fail fast: every network must be on disk
            if todo[m] and m in NETWORK_METHODS:
                print(f'  {m}: {mp.find_checkpoint3d(m, overrides.get(m))}')
        ctx = make_context(a, device)
        meta = dict(n_variants=a.n_variants, n_rounds=a.n_rounds, n_init=a.n_init, n_iters=a.n_iters,
                    kappa0=a.kappa0, kappa1=a.kappa1, walk_len=a.walk_len, cutoff=CUTOFF, K=a.K,
                    w_plane=a.w_plane, tsp_points=a.tsp_points, tsp_min_dist=a.tsp_min_dist,
                    tsp_source=a.tsp_source, units='plane units (world / scale)', **plane.describe())
        for k, v in meta.items():
            db.conn.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (k, str(v)))
        for n, t in zip(names, truths):
            db.put_truth(n, t)
        t_start, stop = time.time(), False
        for method in methods:
            if not todo[method]:
                continue
            planner = mp.build_planner3d(method, plane, device, ctx.settings, overrides.get(method))
            if planner is not None:
                print(f'  {method}: {planner.info}')
                db.conn.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (f'ckpt_{method}', planner.info['ckpt']))
                db.conn.commit()
            print(f'\n=== {method}: {len(todo[method])} missions ===', flush=True)
            for k, i, n, v in todo[method]:
                if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                    print('  time budget reached, stopping cleanly (resume later)')
                    stop = True
                    break
                res = run_one(method, k, n, truths[i], planner, ctx, a, kappas, i * 100 + v)
                db.put_run(n, k, method, v, res, a)
                last = res['rows'][-1]
                print(f'  [{k}/{method}] {n} v{v}: {res["seconds"]:.0f}s  E_truth={last["E_truth"]:.2f}  '
                      f'swept={last["swept06"]:.0%}  path={last["path_len"]:.1f}', flush=True)
            del planner
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
            if stop:
                break

    print('\n=== plots + tables ===')
    done = {r[0] for r in db.conn.execute('SELECT DISTINCT shape FROM runs')}
    have = {r[0] for r in db.conn.execute('SELECT DISTINCT method FROM runs')}
    shown = [m for m in METHODS if m in have and m in methods] or [m for m in METHODS if m in have]
    if not a.no_plots:
        for n in names:
            if n in done:
                for k in kernels:
                    plot_shape(db, n, k, shown, out_dir, plane)
                    plot_variants(db, n, k, shown, out_dir, plane)
                    plot_world(db, n, k, shown, out_dir, plane)
    aggregate(db, out_dir, shown)


if __name__ == '__main__':
    main()
