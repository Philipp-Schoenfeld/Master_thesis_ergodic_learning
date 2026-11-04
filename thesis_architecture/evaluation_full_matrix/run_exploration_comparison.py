#!/usr/bin/env python3
r"""
run_exploration_comparison.py
=============================
The five-round annealed replanning mission of `sample_annealed_mission_overview.py`
(zero prior knowledge, kappa 1 -> 0.05, full execution per round) applied to ALL
heldout shapes for five methods and both belief kernels, with per-round metrics,
all trajectories in one SQLite DB, and per-shape overview plots.

Methods (same shape / seed / belief for every method, so rows differ only by the planner):
  unet630   U-Net lambda=630, CFM init + TSVEC-SVGD refine (cloud cutoff 0.2)
  dit630    DiT lambda=630, CFM init + TSVEC-SVGD refine (cloud cutoff 0.2)
  rw        random walk, no network, no SVGD; `--walk_len` path length per round
  lawn      lawnmower, one budget-matched sweep cut into equal rounds, no network, no SVGD
  linadam   `--n_init` straight rays (billiard reflection) from the current position,
            refined by `--n_iters` iterations of the reference Adam-SVGD
            (`common.adam_svgd.AdamSvgd`: one joint swarm of the rays, Adam on Liu & Wang's SVGD
            direction, K=10, beta=100, lr=2e-3 = `linear_adam_bspline` of
            run_smoothness_comparison_v2.py, start hard-pinned) against the round's target

Belief kernels: 'rbf' and 'ak' (Attentive Kernel). 5 methods x 2 kernels x 25 shapes.

Metrics per round (of the whole path driven so far unless prefixed `seg_`; all against the
ground truth (GT) the mission never sees, normalised to max 1, except E_belief / E_plan):
  E_truth / _explore / _exploit   ergodic error vs. the GT (K=10, solver weighting)
  E_belief                        ergodic error vs. the belief mean AFTER the round's update
                                  (clipped >= 0, scaled to max 1)
  E_plan                          ergodic error vs. the target density the round was planned for
  swept03/06/10                   share of GT mass within 0.03 / 0.06 / 0.10 of the path
  swept_hi50 / swept_hi75         share of the dense GT cells (GT >= 0.5 / 0.75) within 0.06
  cov                             density-weighted mean distance to the path
  belief_rmse / corr / tv / iou_hi / nll / cov95 / sd_mean
                                  belief mean vs. GT (tv = total variation of the normalised
                                  densities, iou_hi = IoU of {>= 0.5} sets, nll / cov95 =
                                  calibration of the belief's sigma)
  path_len, seg_len               cumulative / this round's path length
  out_gt_frac                     share of the path outside the GT support (GT < 0.05)
  in_hi50_frac / in_hi75_frac     share of the path inside dense GT regions (>= 0.5 / 0.75)
  seg_out_gt_frac, seg_in_hi50_frac   the same for this round's segment only
  seg_smooth, seg_curv            smoothness energy (solver's acceleration term) and mean
                                  absolute turning angle per unit length of the segment
  revisit_frac                    share of this segment's points within 0.03 of earlier rounds
  E_init_best / E_ref_best (+ _med)   best / median candidate vs. the round's target before /
                                  after SVGD (NaN for random walk and lawnmower)
Per run: reached 90 % / 99 % GT-mass coverage and the path length at which it was reached.

    python run_exploration_comparison.py --smoke           # 2 shapes, 2 rounds, tiny budget
    python run_exploration_comparison.py                   # everything (resumable)
    python run_exploration_comparison.py --plot_only       # plots + tables from the DB
"""

import argparse
import csv
import io
import math
import os
import sqlite3
import time
import types

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

import sample_annealed_mission as sam                                       # noqa: E402
import sample_annealed_mission_overview as ov                               # noqa: E402
from common.acquisition import kappa_schedule                               # noqa: E402
from common.data import load_truth                                          # noqa: E402
from common.metrics import path_length                                      # noqa: E402
from ergodic_energy_torch import coeffs_from_points                         # noqa: E402,F401
from init_baselines import linear_ray_path                                  # noqa: E402
from metrics_explore_exploit import (ExploreExploitErgodic, smoothness_energy,   # noqa: E402
                                     steps_to_full_coverage)
from run_ideal_matrix import PLANNER_BUILDERS                               # noqa: E402
from run_mission_eval import _truth_like, belief_quality                    # noqa: E402
from svgd_batched import BatchedTsvecSvgdTorch                              # noqa: E402
import run_svgd_convergence as rsc                                          # noqa: E402
import variant_runner as vr                                                 # noqa: E402
from common.adam_svgd import AdamSvgd                                       # noqa: E402
from ergodic_energy_torch import W_SMOOTH                                   # noqa: E402
from run_smoothness_comparison_v2 import ADAM_BETA, ADAM_K, ADAM_LR         # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))

METHODS = ('unet630', 'dit630', 'rw', 'lawn', 'linadam')
KERNELS = ('rbf', 'ak')
METHOD_LABEL = {'unet630': 'U-Net λ=630', 'dit630': 'DiT λ=630', 'rw': 'Random walk',
                'lawn': 'Lawnmower', 'linadam': 'Linear init + Adam-SVGD'}
METHOD_COLOR = {'unet630': '#1565C0', 'dit630': '#6A1B9A', 'rw': '#888888',
                'lawn': '#E65100', 'linadam': '#00897B'}
NET_OF = {'unet630': 'unet630', 'dit630': 'dit630', 'rw': 'rw', 'lawn': 'lawn',
          'linadam': 'linadam'}
CUTOFF = ov.CUT_MID

GT_SUPPORT = 0.05            # path point counts as "outside the GT" below this GT value
HI_LEVELS = (0.5, 0.75)      # "dense GT region" thresholds
REVISIT_RADIUS = 0.03

METRIC_COLS = [
    'E_truth', 'E_truth_explore', 'E_truth_exploit', 'E_belief', 'E_plan',
    'swept03', 'swept06', 'swept10', 'swept_hi50', 'swept_hi75', 'cov',
    'belief_rmse', 'belief_corr', 'belief_tv', 'belief_iou_hi', 'belief_nll',
    'belief_cov95', 'belief_sd_mean', 'n_obs',
    'path_len', 'seg_len', 'out_gt_frac', 'in_hi50_frac', 'in_hi75_frac',
    'seg_out_gt_frac', 'seg_in_hi50_frac', 'seg_in_hi75_frac', 'seg_smooth', 'seg_curv',
    'revisit_frac',
    'E_init_best', 'E_init_med', 'E_ref_best', 'E_ref_med',
]
# aliases the overview table / plot code expects
ALIASES = {'swept': 'swept06', 'belief_corr': 'belief_corr'}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS truths (
    shape TEXT PRIMARY KEY, res INTEGER NOT NULL, truth BLOB NOT NULL);   -- (res, res) float32
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shape TEXT NOT NULL, kernel TEXT NOT NULL, method TEXT NOT NULL,
    variant INTEGER NOT NULL, seed INTEGER, n_rounds INTEGER, n_init INTEGER, n_iters INTEGER,
    seconds REAL, reached90 INTEGER, reached99 INTEGER,
    path_len_90 REAL, path_len_99 REAL, created_at REAL,
    UNIQUE(shape, kernel, method, variant));
CREATE TABLE IF NOT EXISTS rounds (
    run_id INTEGER NOT NULL REFERENCES runs(id), round INTEGER NOT NULL, kappa REAL,
    start_x REAL, start_y REAL,
    {', '.join(f'{c} REAL' for c in METRIC_COLS)},
    segment BLOB,       -- (n, 2) float32, the path driven in this round
    phi BLOB,           -- (R, R) float16, target density the round was planned for
    mu BLOB, sd BLOB,   -- (G, G) float16, belief after the round's measurements
    PRIMARY KEY (run_id, round));
"""


def to_blob(x, dtype):
    return np.ascontiguousarray(np.asarray(x, dtype=dtype)).tobytes()


class ResultDB:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.execute('PRAGMA journal_mode=WAL')

    def has_run(self, shape, kernel, method, variant):
        return self.conn.execute(
            'SELECT 1 FROM runs WHERE shape=? AND kernel=? AND method=? AND variant=?',
            (shape, kernel, method, variant)).fetchone() is not None

    def put_truth(self, shape, truth):
        t = truth.detach().cpu().numpy().astype(np.float32)
        self.conn.execute('INSERT OR REPLACE INTO truths VALUES (?,?,?)',
                          (shape, t.shape[-1], to_blob(t, np.float32)))
        self.conn.commit()

    def put_run(self, shape, kernel, method, variant, res, cfg):
        cur = self.conn.execute(
            'INSERT INTO runs (shape,kernel,method,variant,seed,n_rounds,n_init,n_iters,seconds,'
            'reached90,reached99,path_len_90,path_len_99,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (shape, kernel, method, variant, res['seed'], cfg.n_rounds, cfg.n_init, cfg.n_iters,
             res['seconds'], res['reached90'], res['reached99'], res['len90'], res['len99'],
             time.time()))
        rid = cur.lastrowid
        for r, row in enumerate(res['rows']):
            cols = ['run_id', 'round', 'kappa', 'start_x', 'start_y'] + METRIC_COLS + \
                   ['segment', 'phi', 'mu', 'sd']
            vals = [rid, r, res['kappas'][r], res['starts'][r][0], res['starts'][r][1]] + \
                   [_f(row.get(c)) for c in METRIC_COLS] + \
                   [to_blob(res['segments'][r], np.float32), to_blob(res['phis'][r], np.float16),
                    to_blob(res['mus'][r], np.float16), to_blob(res['sds'][r], np.float16)]
            self.conn.execute(f'INSERT INTO rounds ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})',
                              vals)
        self.conn.commit()

    def load_run(self, shape, kernel, method, variant=0):
        run = self.conn.execute(
            'SELECT id FROM runs WHERE shape=? AND kernel=? AND method=? AND variant=?',
            (shape, kernel, method, variant)).fetchone()
        if run is None:
            return None
        rows, segs, phis, kappas = [], [], [], []
        q = (f'SELECT round,kappa,{",".join(METRIC_COLS)},segment,phi FROM rounds '
             'WHERE run_id=? ORDER BY round')
        for rec in self.conn.execute(q, (run[0],)):
            d = dict(zip(['round', 'kappa'] + METRIC_COLS, rec[:2 + len(METRIC_COLS)]))
            d = {k: (float('nan') if v is None else v) for k, v in d.items()}
            d['swept'] = d['swept06']
            rows.append(d)
            kappas.append(d['kappa'])
            seg = np.frombuffer(rec[-2], dtype=np.float32).reshape(-1, 2).copy()
            segs.append(torch.from_numpy(seg))
            phi = np.frombuffer(rec[-1], dtype=np.float16).astype(np.float32)
            phis.append(torch.from_numpy(phi.reshape(int(math.isqrt(phi.size)), -1)))
        return dict(rows=rows, segments=segs, phis=phis, kappas=kappas)

    def load_truth(self, shape):
        res, blob = self.conn.execute('SELECT res, truth FROM truths WHERE shape=?', (shape,)).fetchone()
        return torch.from_numpy(np.frombuffer(blob, dtype=np.float32).reshape(res, res).copy())


def _f(v):
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


class RefAdamSvgdBatch:
    """`bs.run` contract of `sample_annealed_mission.run_mission` on top of the reference
    Adam-SVGD (`common.adam_svgd.AdamSvgd`): all candidates form ONE joint swarm against the
    shared target (one shape per call), the start is hard-pinned, the target is the K=10 Fourier
    coefficient vector (same as TSVEC). `seeds` is unused (no stochastic element)."""

    def __init__(self, B, device):
        self.B = np.asarray(B, dtype=np.float32)
        self.T, self.nxi = self.B.shape
        self.solver = AdamSvgd(B=self.B, K=ADAM_K, device=device, w_smooth=W_SMOOTH,
                               beta=ADAM_BETA, lr=ADAM_LR)

    def run(self, init_curves, phi_k, start, seeds, n_iters, record=False):
        curves = np.asarray(init_curves, dtype=np.float64)
        C = curves.shape[0]
        rhs = curves.transpose(1, 0, 2).reshape(self.T, C * 2)
        P0 = np.linalg.lstsq(self.B.astype(np.float64), rhs, rcond=None)[0].reshape(
            self.nxi, C, 2).transpose(1, 0, 2).astype(np.float32)
        phi_k = np.asarray(phi_k, dtype=np.float64)
        phi_k = phi_k[0] if phi_k.ndim == 2 else phi_k
        start = None if start is None else np.asarray(start, dtype=np.float64).reshape(-1, 2)[0]
        out = self.solver.run(P0, phi_k, n_iters=n_iters, record=False, start=start)
        return dict(cps=None, final_cps=out['final'].detach().cpu().numpy().astype(np.float32),
                    final_energy=out['energy'].detach().cpu().numpy())


# ===========================================================================
# Metrics
# ===========================================================================

class Evaluator:
    """Per-shape metric machinery: ground-truth masks and grid distances are built once."""

    def __init__(self, truth, ee, device):
        self.truth, self.ee, self.device = truth.float().cpu(), ee, device
        self.R = truth.shape[-1]
        ys, xs = torch.meshgrid(torch.linspace(0, 1, self.R), torch.linspace(0, 1, self.R),
                                indexing='ij')
        self.cells = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
        self.w = self.truth.reshape(-1).clamp(min=0.0)
        self.w_sum = float(self.w.sum().clamp(min=1e-12))
        self.hi = {lv: (self.truth.reshape(-1) >= lv) for lv in HI_LEVELS}
        self.phi_k_truth = ee.target_coeffs(self.truth)

    def truth_at(self, pts):
        idx = (pts.cpu().clamp(0, 1) * (self.R - 1)).round().long()
        return self.truth[idx[:, 1], idx[:, 0]]

    def path_stats(self, pts, prefix=''):
        t = self.truth_at(pts)
        out = {f'{prefix}out_gt_frac': float((t < GT_SUPPORT).float().mean()),
               f'{prefix}in_hi50_frac': float((t >= 0.5).float().mean()),
               f'{prefix}in_hi75_frac': float((t >= 0.75).float().mean())}
        return out

    def coverage(self, driven):
        d = torch.cdist(self.cells, driven.cpu().float()).min(dim=1).values
        out = {'cov': float((d * self.w).sum() / self.w_sum)}
        for key, rad in (('swept03', 0.03), ('swept06', 0.06), ('swept10', 0.10)):
            out[key] = float(((d <= rad).float() * self.w).sum() / self.w_sum)
        for lv in HI_LEVELS:
            m = self.hi[lv]
            out[f'swept_hi{int(lv * 100)}'] = (float((d[m] <= 0.06).float().mean())
                                               if m.any() else float('nan'))
        return out

    def belief(self, belief):
        mu, sd = belief.posterior_grid()
        belief._cache = None
        mu, sd = mu.float(), sd.float()
        bq = belief_quality(mu, sd, self.truth)
        mu_n = mu.clamp(min=0.0)
        mu_n = mu_n / mu_n.max().clamp(min=1e-6)
        t = _truth_like(self.truth, mu_n)
        a, b = mu_n / mu_n.sum().clamp(min=1e-12), t / t.sum().clamp(min=1e-12)
        ma, mb = mu_n >= HI_LEVELS[0], t >= HI_LEVELS[0]
        iou = float((ma & mb).sum() / (ma | mb).sum().clamp(min=1))
        return mu, sd, mu_n, dict(
            belief_rmse=bq['rmse'], belief_corr=bq['corr'], belief_nll=bq['nll'],
            belief_cov95=bq['cov95'], belief_sd_mean=bq['sd_mean'],
            belief_tv=float(0.5 * (a - b).abs().sum()), belief_iou_hi=iou)

    def round_metrics(self, driven, seg, earlier, belief, diag):
        sc = self.ee.score(driven, self.phi_k_truth)
        mu, sd, mu_n, bm = self.belief(belief)
        row = dict(E_truth=sc['E_ergodic_total'], E_truth_explore=sc['E_ergodic_explore'],
                   E_truth_exploit=sc['E_ergodic_exploit'],
                   E_belief=self.ee.score(driven, self.ee.target_coeffs(mu_n))['E_ergodic_total'],
                   path_len=path_length(driven), seg_len=path_length(seg),
                   n_obs=int(belief.n_obs), **bm, **self.coverage(driven),
                   **self.path_stats(driven), **self.path_stats(seg, 'seg_'),
                   seg_smooth=smoothness_energy(seg), seg_curv=turning_per_length(seg),
                   revisit_frac=(float((torch.cdist(seg, earlier).min(dim=1).values
                                        <= REVISIT_RADIUS).float().mean())
                                 if earlier is not None else 0.0))
        row.update(diag)
        return row, mu, sd


def turning_per_length(seg):
    v = seg[1:] - seg[:-1]
    ang = torch.atan2(v[:, 1], v[:, 0])
    turn = torch.remainder(ang[1:] - ang[:-1] + math.pi, 2 * math.pi) - math.pi
    return float(turn.abs().sum() / max(path_length(seg), 1e-9))


# ===========================================================================
# One mission
# ===========================================================================

def run_one(method, kernel, shape, truth, planner, bs, B32, ee, base_args, a, kappas, device, seed):
    setup = ov.SETUPS[setup_key(kernel, method)]
    args = types.SimpleNamespace(**vars(base_args))
    args.belief_kernel, args.ak_fit_iters, args.np_ckpt = kernel, a.ak_fit_iters, None
    evl = Evaluator(truth, ee, device)
    rows, mus, sds = [], [], []

    def on_round(r, belief, segments, diag):
        driven = torch.cat(segments, dim=0)
        earlier = torch.cat(segments[:-1], dim=0) if r > 0 else None
        row, mu, sd = evl.round_metrics(driven, segments[-1], earlier, belief, diag)
        rows.append(row)
        mus.append(mu.cpu().numpy())
        sds.append(sd.cpu().numpy())

    kw = {}
    if method in ('rw', 'lawn'):
        kw['path_fn'] = ov.baseline_path_fn(setup, a, seed)
    elif method == 'linadam':
        kw['init_fn'] = lambda r, start, phi: torch.stack(
            [linear_ray_path(start.detach().cpu().numpy(), 360.0 * i / a.n_init, a.walk_len, rsc.N_POINTS)
             for i in range(a.n_init)])
    start_pos = (0.06, 0.06) if method == 'lawn' else (0.5, 0.5)
    t0 = time.time()
    segments, phis, _ = sam.run_mission(
        f'{kernel}/{method}/{shape}', truth, planner, bs, ee, B32, args, device, a.n_rounds,
        a.n_init, a.n_iters, kappas, start_pos, seed,
        cutoff=CUTOFF if method in ('unet630', 'dit630') else 0.0, on_round=on_round, **kw)
    seconds = time.time() - t0
    for r, row in enumerate(rows):      # E vs. the round's own planning target (needs phis)
        row['E_plan'] = ee.score(torch.cat(segments[:r + 1], dim=0),
                                 ee.target_coeffs(phis[r]))['E_ergodic_total']
    full = torch.cat(segments, dim=0)
    lens = {}
    for thr in (0.90, 0.99):
        frac, reached = steps_to_full_coverage(full, truth, threshold=thr, sensor_radius=0.06)
        lens[thr] = (int(reached), float(frac) * path_length(full) if reached else float('nan'))
    starts = [start_pos] + [s[-1].tolist() for s in segments[:-1]]
    return dict(rows=rows, segments=[s.numpy() for s in segments], phis=[p.numpy() for p in phis],
                mus=mus, sds=sds, kappas=kappas, starts=starts, seconds=seconds, seed=seed,
                reached90=lens[0.90][0], reached99=lens[0.99][0], len90=lens[0.90][1],
                len99=lens[0.99][1])


# ===========================================================================
# Plots / tables from the DB
# ===========================================================================

def setup_key(kernel, method):
    return f'{kernel}_{method}' if method in ('rw', 'lawn', 'linadam') else f'{kernel}_{method}_mid'


def plot_shape(db, shape, kernel, methods, out_dir):
    """Round-by-round overview of variant 0 (one row per method)."""
    results = []
    truth = db.load_truth(shape)
    for m in methods:
        res = db.load_run(shape, kernel, m, 0)
        if res is None:
            return
        res.update(setup=setup_key(kernel, m), truth=truth)
        results.append(res)
    kern = {'rbf': 'RBF', 'ak': 'Attentive-Kernel'}[kernel]
    ov.plot_rows(results, f'Exploration comparison: {kern} belief (cloud cutoff {CUTOFF:g})',
                 os.path.join(out_dir, kernel, f'{shape}.png'), shape)


def plot_variants(db, shape, kernel, methods, out_dir):
    """All variants of every method on the ground truth: one row per method, one panel per
    variant (all rounds overlaid), so the spread between the random restarts is visible."""
    n_var = db.conn.execute('SELECT MAX(variant)+1 FROM runs WHERE shape=? AND kernel=?',
                            (shape, kernel)).fetchone()[0]
    if not n_var:
        return
    truth = db.load_truth(shape)
    cmap = sam.white_inferno()
    fig, axes = plt.subplots(len(methods), n_var, squeeze=False, facecolor='white',
                             figsize=(2.3 * n_var, 2.5 * len(methods)))
    for i, m in enumerate(methods):
        for v in range(n_var):
            ax = axes[i, v]
            res = db.load_run(shape, kernel, m, v)
            if res is None:
                ax.axis('off')
                continue
            sam.plot_summary(ax, truth, [torch.as_tensor(s) for s in res['segments']],
                             sam.round_colors(len(res['segments'])), res['kappas'], cmap=cmap)
            ax.get_legend().remove()
            fin = res['rows'][-1]
            ax.set_title(f'E={fin["E_truth"]:.1f}  swept {fin["swept06"]:.0%}', fontsize=8,
                         color=ov.TEXT)
            if v == 0:
                ax.set_ylabel(METHOD_LABEL[m], fontsize=9, color=ov.TEXT)
    fig.suptitle(f'{shape}, {kernel.upper()} belief: {n_var} variants per method '
                 '(all rounds on the ground truth)', fontsize=11, color=ov.TEXT)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(out_dir, 'variants', kernel, f'{shape}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=90, facecolor='white')
    plt.close(fig)


def fetch_table(db):
    cols = ['r.round', 'u.shape', 'u.kernel', 'u.method', 'u.variant'] + \
        [f'r.{c}' for c in METRIC_COLS]
    cur = db.conn.execute(f'SELECT {",".join(cols)} FROM rounds r JOIN runs u ON u.id=r.run_id')
    names = ['round', 'shape', 'kernel', 'method', 'variant'] + METRIC_COLS
    return names, [dict(zip(names, rec)) for rec in cur]


def _nan(v):
    return float('nan') if v is None else float(v)


def aggregate(db, out_dir, methods):
    """Tables + curves over all (shape, variant) runs in the DB. Two sources of spread are kept
    apart: between shapes (std of the per-shape variant means) and within a shape (std over the
    variants of one shape, averaged over the shapes)."""
    names, recs = fetch_table(db)
    os.makedirs(out_dir, exist_ok=True)
    if not recs:
        return
    with open(os.path.join(out_dir, 'metrics_all.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(names)
        for d in recs:
            w.writerow([d[n] for n in names])
    n_rounds = max(d['round'] for d in recs) + 1
    # (kernel, method, round) -> shape -> list of rows (one per variant)
    cells = {}
    for d in recs:
        cells.setdefault((d['kernel'], d['method'], d['round']), {}).setdefault(d['shape'], []).append(d)

    def stats(per_shape, c):
        """-> dict of statistics of metric c, or None if it is NaN everywhere."""
        all_v = np.array([_nan(d[c]) for ds in per_shape.values() for d in ds])
        if np.isnan(all_v).all():
            return None
        shape_means = np.array([np.nanmean([_nan(d[c]) for d in ds]) for ds in per_shape.values()])
        within = np.array([np.nanstd([_nan(d[c]) for d in ds]) for ds in per_shape.values()])
        return dict(n_shapes=len(per_shape), n_runs=len(all_v), mean=np.nanmean(all_v),
                    median=np.nanmedian(all_v), std_total=np.nanstd(all_v),
                    std_between_shapes=np.nanstd(shape_means), std_within_shape=np.nanmean(within),
                    min=np.nanmin(all_v), max=np.nanmax(all_v))

    keys = ['n_shapes', 'n_runs', 'mean', 'median', 'std_total', 'std_between_shapes',
            'std_within_shape', 'min', 'max']
    with open(os.path.join(out_dir, 'summary_by_round.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'round', 'metric'] + keys)
        for (k, m, r), per_shape in sorted(cells.items()):
            for c in METRIC_COLS:
                st = stats(per_shape, c)
                if st:
                    w.writerow([k, m, r, c] + [st[x] for x in keys])
    # per shape, final round: mean / std over the variants
    with open(os.path.join(out_dir, 'summary_by_shape_final.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'shape', 'metric', 'n_variants', 'mean', 'std', 'min', 'max'])
        for (k, m, r), per_shape in sorted(cells.items()):
            if r != n_rounds - 1:
                continue
            for shape, ds in sorted(per_shape.items()):
                for c in METRIC_COLS:
                    v = np.array([_nan(d[c]) for d in ds])
                    if not np.isnan(v).all():
                        w.writerow([k, m, shape, c, len(v), np.nanmean(v), np.nanstd(v),
                                    np.nanmin(v), np.nanmax(v)])
    # paired comparison of the final round, pairing (shape, variant): the same seed means the
    # same belief noise for every method
    final = {(d['kernel'], d['method'], d['shape'], d['variant']): d
             for d in recs if d['round'] == n_rounds - 1}
    with open(os.path.join(out_dir, 'paired_vs_unet630_final.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'method', 'metric', 'better_is', 'n_pairs', 'share_better_than_unet630',
                    'mean_diff_vs_unet630'])
        for k in KERNELS:
            for m in methods:
                if m == 'unet630':
                    continue
                for c, lower in (('E_truth', True), ('E_belief', True), ('swept06', False),
                                 ('belief_rmse', True), ('belief_corr', False),
                                 ('out_gt_frac', True), ('in_hi50_frac', False), ('cov', True)):
                    diffs = [_nan(final[key][c]) - _nan(final[(k, 'unet630') + key[2:]][c])
                             for key in final if key[0] == k and key[1] == m
                             and (k, 'unet630') + key[2:] in final]
                    diffs = np.array([x for x in diffs if x == x])
                    if len(diffs):
                        w.writerow([k, m, c, 'lower' if lower else 'higher', len(diffs),
                                    float(((diffs < 0) if lower else (diffs > 0)).mean()),
                                    float(diffs.mean())])
    plot_curves(cells, n_rounds, out_dir, methods, 'mean')
    plot_curves(cells, n_rounds, out_dir, methods, 'spread')
    print(f'  tables + aggregate plots in {out_dir}')


CURVE_PANELS = (('E_truth', 'E vs. ground truth ↓'), ('E_belief', 'E vs. belief ↓'),
                ('swept06', 'GT mass within 0.06 of path ↑'), ('belief_rmse', 'belief RMSE ↓'),
                ('path_len', 'path length'), ('out_gt_frac', 'path outside GT ↓'),
                ('in_hi50_frac', 'path in dense GT (≥ 0.5) ↑'), ('belief_corr', 'belief corr ↑'))


def plot_curves(cells, n_rounds, out_dir, methods, mode):
    """mode 'mean': mean over shapes of the per-shape variant means (band: s.e.m. over shapes);
    mode 'spread': mean over shapes of the std over the variants of a shape."""
    fig, axes = plt.subplots(len(KERNELS), len(CURVE_PANELS), squeeze=False, facecolor='white',
                             figsize=(3.3 * len(CURVE_PANELS), 3.2 * len(KERNELS)))
    xs = np.arange(n_rounds)
    for i, k in enumerate(KERNELS):
        for j, (c, title) in enumerate(CURVE_PANELS):
            ax = axes[i, j]
            for m in methods:
                mean, sem = [], []
                for r in range(n_rounds):
                    per_shape = cells.get((k, m, r), {})
                    if mode == 'mean':
                        v = np.array([np.nanmean([_nan(d[c]) for d in ds]) for ds in per_shape.values()])
                    else:
                        v = np.array([np.nanstd([_nan(d[c]) for d in ds]) for ds in per_shape.values()])
                    v = v[~np.isnan(v)] if len(v) else v
                    mean.append(v.mean() if len(v) else np.nan)
                    sem.append(v.std() / math.sqrt(len(v)) if len(v) > 1 else 0.0)
                mean, sem = np.array(mean), np.array(sem)
                ax.plot(xs, mean, color=METHOD_COLOR[m], lw=2, marker='o', ms=3.5,
                        label=METHOD_LABEL[m])
                ax.fill_between(xs, mean - sem, mean + sem, color=METHOD_COLOR[m], alpha=0.15, lw=0)
            ax.set_title(f'{title}\n{k.upper()} belief' if j == 0 else title, fontsize=9, color=ov.TEXT)
            ax.set_xlabel('round', fontsize=8, color=ov.MUTED)
            ax.set_xticks(xs)
            ax.grid(alpha=0.2)
            ax.tick_params(labelsize=8, colors=ov.MUTED)
            for s in ax.spines.values():
                s.set_color('#ccc')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(methods), fontsize=9, frameon=False)
    fig.suptitle('Mean over heldout shapes of the per-shape variant mean ± s.e.m. over shapes '
                 '(path driven so far)' if mode == 'mean' else
                 'Spread between variants: std over the variants of a shape, averaged over shapes '
                 '(± s.e.m. over shapes)', fontsize=11, color=ov.TEXT)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    name = 'aggregate_curves.png' if mode == 'mean' else 'aggregate_variant_spread.png'
    fig.savefig(os.path.join(out_dir, name), dpi=120, facecolor='white')
    plt.close(fig)


# ===========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', type=str, default=','.join(METHODS))
    p.add_argument('--kernels', type=str, default=','.join(KERNELS))
    p.add_argument('--n_shapes', type=int, default=None, help='Only the first N heldout shapes (default: all).')
    p.add_argument('--shape_names', type=str, default=None, help='Comma-separated explicit shapes.')
    p.add_argument('--n_variants', type=int, default=7,
                   help='Independent variants (seeds) per method, kernel and shape: different '
                        'network samples / SVGD jitter / random walks / measurement noise.')
    p.add_argument('--n_rounds', type=int, default=5)
    p.add_argument('--n_init', type=int, default=8, help='Candidates per round.')
    p.add_argument('--n_iters', type=int, default=500, help='SVGD iterations per round (TSVEC and Adam).')
    p.add_argument('--kappa0', type=float, default=1.0)
    p.add_argument('--kappa1', type=float, default=0.05)
    p.add_argument('--walk_len', type=float, default=6.1,
                   help='Path length per round of random walk / lawnmower / linear rays.')
    p.add_argument('--ak_fit_iters', type=int, default=200)
    p.add_argument('--max_minutes', type=float, default=None, help='Stop cleanly after this wall time.')
    p.add_argument('--db', type=str, default=None)
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--plot_only', action='store_true')
    p.add_argument('--smoke', action='store_true', help='2 shapes, 2 rounds, tiny budget, separate DB/folder.')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.smoke:
        a.n_rounds, a.n_init, a.n_iters, a.n_variants = 2, 2, 20, 2
        a.n_shapes = a.n_shapes or 2
    tag = 'exploration_comparison' + ('_smoke' if a.smoke else '')
    out_dir = a.out_dir or os.path.join(_here, 'plots', tag)
    db_path = a.db or os.path.join(out_dir, f'{tag}.db')
    methods, kernels = a.methods.split(','), a.kernels.split(',')
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f'unknown method {m}; known: {METHODS}')
    db = ResultDB(db_path)
    device = a.device

    labels = a.shape_names.split(',') if a.shape_names else None
    names, truths = load_truth(labels=labels, n=a.n_shapes or 10_000, split='val',
                               resolution=rsc.TRUTH_RES, device=device)
    names = list(names)
    print(f'{len(names)} heldout shapes, methods={methods}, kernels={kernels}, db={db_path}')

    if not a.plot_only:
        kappas = [kappa_schedule(r, a.n_rounds, a.kappa0, a.kappa1) for r in range(a.n_rounds)]
        base_args, _, _ = vr.build_strategy_args(sam.STRATEGY_NAME, device)
        ee = ExploreExploitErgodic(device=device)
        B_np = rsc.basis_matrix(rsc.NXI, rsc.N_POINTS, rsc.DEGREE)
        B32 = torch.as_tensor(B_np, dtype=torch.float32, device=device)
        bs_tsvec = BatchedTsvecSvgdTorch(B_np, device)
        bs_adam = RefAdamSvgdBatch(B_np, device)
        meta = dict(n_variants=a.n_variants, n_rounds=a.n_rounds, n_init=a.n_init, n_iters=a.n_iters, kappa0=a.kappa0,
                    kappa1=a.kappa1, walk_len=a.walk_len, cutoff=CUTOFF, gt_support=GT_SUPPORT,
                    hi_levels=list(HI_LEVELS), adam_k=ADAM_K, adam_beta=ADAM_BETA,
                    adam_lr=ADAM_LR)
        for k, v in meta.items():
            db.conn.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (k, str(v)))
        for n, t in zip(names, truths):
            db.put_truth(n, t)
        t_start = time.time()
        stop = False
        for method in methods:
            # variant outermost: a stopped run still holds complete variants of every shape
            todo = [(k, i, n, v) for v in range(a.n_variants) for i, n in enumerate(names)
                    for k in kernels if not db.has_run(n, k, method, v)]
            if not todo:
                continue
            planner = None
            if method in ('unet630', 'dit630'):
                planner = PLANNER_BUILDERS['particles'](ov.find_checkpoint(NET_OF[method]), device)
            bs = bs_adam if method == 'linadam' else bs_tsvec
            print(f'\n=== {method}: {len(todo)} missions ===')
            for k, i, n, v in todo:
                if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                    print('  time budget reached, stopping cleanly (resume later)')
                    stop = True
                    break
                res = run_one(method, k, n, truths[i].detach().cpu(), planner, bs, B32, ee, base_args,
                              a, kappas, device, i * 100 + v)
                db.put_run(n, k, method, v, res, a)
                last = res['rows'][-1]
                print(f'  [{k}/{method}] {n} v{v}: {res["seconds"]:.0f}s  E_truth={last["E_truth"]:.2f}  '
                      f'swept={last["swept06"]:.0%}  path={last["path_len"]:.1f}')
            del planner
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            if stop:
                break

    print('\n=== plots + tables ===')
    done = {r[0] for r in db.conn.execute('SELECT DISTINCT shape FROM runs')}
    for n in names:
        if n in done:
            for k in kernels:
                plot_shape(db, n, k, methods, out_dir)
                plot_variants(db, n, k, methods, out_dir)
    aggregate(db, out_dir, methods)


if __name__ == '__main__':
    main()
