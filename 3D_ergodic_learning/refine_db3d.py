#!/usr/bin/env python3
"""Refine the teacher paths of a projected 3D DB on the training objective.

Phases 1-3 of the DB improvement plan (November 2026):

  1. Refinement. Every entry's LSQ control points are optimised per entry with
     Adam on exactly the objective the networks are trained on: debiased
     Sinkhorn at the beam footprint (lambda 630) + orientation loss (pointing,
     standoff 0.12 +- 0.03, angular smoothness; lambda 0.012). Control point 0
     (the start) is held. The projected paths of `project_db_3d.py` lie *on*
     the surface (standoff 0, axis = -normal) and were optimised in 2D for the
     planar density; as CFM targets they pull the network away from what the
     ErgLoss asks for.
  2. Standoff. Before refining, every path is moved 0.12 along its surface
     normal (variant b, = `project_db_3d.py --standoff 0.12`) and the refinement
     then optimises for standoff 0.12 (variant a). This removes the conflict
     between on-surface targets and `--standoff_target 0.12` in training.
  3. Filter. Train entries whose refined Sinkhorn error is above the
     `--filter_pct` percentile, or whose standoff misses the target by more
     than `--max_so_dev`, or that went non-finite, are dropped. Holdout entries
     are never dropped, so the holdout stays comparable to the source DB.

Shape guard: Sinkhorn alone rewards ever longer, wigglier paths. The
refinement penalises control-polygon length and mean squared second difference
growing beyond `--slack` times their initial value.

The destination DB has the source schema plus `refine_*` columns; the paths are
stored as 201 points sampled from the refined spline, so `load_surface_db(...,
target_cps=True)` recovers the refined control points exactly. The run is
resumable: rows already in the destination are skipped.

    python refine_db3d.py --src ergodic_dataset_3d_lm.db --dst ergodic_dataset_3d_lm_ref.db
"""
import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from data_surfaces import load_surface_db
from ergodic_metric import ErgodicLoss
from obstacles import bspline_basis_matrix
from orientation import rot6d_to_matrix, sensor_axis
from orientation_energy import OrientationLoss
from surface_metrics import standoff_pointing

SPLITS = ('train', 'val_form', 'val_flaeche', 'val_beides')
N_STORE = 201
EXTRA_COLS = ['refine_erg0', 'refine_erg_off', 'refine_erg1', 'refine_so0', 'refine_so1',
              'refine_pt0', 'refine_pt1', 'refine_len0', 'refine_len1', 'refine_loss1']


class Refiner:
    """Per-entry Adam on the training objective, batched over entries."""

    def __init__(self, device, nxi=25, deg=5, pts=128, lambda_erg=630.0, lambda_ori=0.012,
                 w_point=0.1, w_standoff=300.0, w_angsmooth=2.0, standoff=0.12, band=0.03,
                 w_shape=1.0, slack=1.5):
        self.dev, self.nxi, self.standoff = device, nxi, standoff
        self.lambda_erg, self.lambda_ori = lambda_erg, lambda_ori
        self.w_shape, self.slack = w_shape, slack
        self.erg = ErgodicLoss(nxi=nxi, nd=3, pts=pts, deg=deg, t_power=0.0,
                               ergodic_on='footprint', metric='sinkhorn',
                               sinkhorn_blur=0.05).to(device)
        self.ori = OrientationLoss(nxi=nxi, pts=pts, deg=deg, t_power=0.0, w_point=w_point,
                                   w_standoff=w_standoff, w_angsmooth=w_angsmooth,
                                   standoff_target=standoff, standoff_band=band).to(device)
        self.B = torch.from_numpy(bspline_basis_matrix(nxi, pts, deg)).float().to(device)

    def offset(self, x1, standoff=None):
        """Move on-surface control points `standoff` against their sensor axis (= along the normal)."""
        s = self.standoff if standoff is None else standoff
        z = sensor_axis(rot6d_to_matrix(x1[..., 3:]), 2)
        z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return torch.cat([x1[..., :3] - s * z, x1[..., 3:]], dim=-1)

    @staticmethod
    def shape_stats(pos):
        acc = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]).pow(2).sum(-1).mean(-1)
        length = (pos[:, 1:] - pos[:, :-1]).norm(dim=-1).sum(-1)
        return acc, length

    def objective(self, x, parts, ref_stats):
        erg = self.erg.coverage_error(x[..., :3], parts, rot6d=x[..., 3:])
        ori = sum(self.ori.terms(x, parts).values())
        acc, length = self.shape_stats(x[..., :3])
        acc0, len0 = ref_stats
        shape = (torch.relu(acc / acc0 - self.slack) + torch.relu(length / len0 - self.slack))
        total = self.lambda_erg * erg + self.lambda_ori * ori + self.w_shape * shape
        return total, erg

    def run(self, x_init, parts, iters=400, lr=2e-3):
        """x_init (B, nxi, 9), parts (B, N, 4) -> best x (B, nxi, 9), best loss (B,)."""
        x_init = x_init.float()
        with torch.no_grad():
            acc0, len0 = self.shape_stats(x_init[..., :3])
            ref = (acc0.clamp(min=1e-6), len0.clamp(min=1e-3))
        cp0 = x_init[:, :1, :3]
        free = x_init[:, 1:, :3].clone().requires_grad_(True)
        rot = x_init[..., 3:].clone().requires_grad_(True)
        opt = torch.optim.Adam([free, rot], lr=lr)
        best = x_init.clone()
        best_l = torch.full((x_init.shape[0],), float('inf'), device=x_init.device)
        for _ in range(iters + 1):
            x = torch.cat([torch.cat([cp0, free], dim=1), rot], dim=-1)
            total, _ = self.objective(x, parts, ref)
            with torch.no_grad():
                better = torch.isfinite(total) & (total < best_l)
                best[better] = x.detach()[better]
                best_l[better] = total.detach()[better]
            opt.zero_grad(set_to_none=True)
            total.nan_to_num(0.0).sum().backward()
            opt.step()
        return best, best_l

    @torch.no_grad()
    def metrics(self, x, parts):
        """-> dict of (B,) : Sinkhorn footprint error, PCA standoff, PCA pointing, polygon length."""
        x = x.float()
        erg = self.erg.coverage_error(x[..., :3], parts, rot6d=x[..., 3:])
        curve = torch.einsum('pi,bid->bpd', self.B, x[..., :3])
        axis = sensor_axis(rot6d_to_matrix(torch.einsum('pi,bic->bpc', self.B, x[..., 3:])), 2)
        dist, ang = standoff_pointing(curve, axis, parts)
        _, length = self.shape_stats(x[..., :3])
        return dict(erg=erg, so=dist.mean(-1), so_sd=dist.std(-1), pt=ang.mean(-1), len=length)


def refiner_from_args(a, dev):
    return Refiner(dev, lambda_erg=a.lambda_erg, lambda_ori=a.lambda_ori, w_point=a.w_point,
                   w_standoff=a.w_standoff, w_angsmooth=a.w_angsmooth, standoff=a.standoff,
                   band=a.standoff_band, w_shape=a.w_shape, slack=a.slack)


def add_refine_args(p):
    p.add_argument('--iters', type=int, default=400)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--standoff', type=float, default=0.12)
    p.add_argument('--standoff_band', type=float, default=0.03)
    p.add_argument('--lambda_erg', type=float, default=630.0)
    p.add_argument('--lambda_ori', type=float, default=0.012)
    p.add_argument('--w_point', type=float, default=0.1)
    p.add_argument('--w_standoff', type=float, default=300.0)
    p.add_argument('--w_angsmooth', type=float, default=2.0)
    p.add_argument('--w_shape', type=float, default=1.0)
    p.add_argument('--slack', type=float, default=1.5)


def pick(entries, limit, seed=0):
    """`limit` entries spread evenly over the splits (all if limit is None)."""
    if not limit:
        return entries
    rng = np.random.default_rng(seed)
    splits = sorted({e['split'] for e in entries})
    out = []
    for sp in splits:
        pool = [e for e in entries if e['split'] == sp]
        out += [pool[i] for i in sorted(rng.permutation(len(pool))[:max(1, limit // len(splits))])]
    return out


# ---------------------------------------------------------------------------
# destination DB
# ---------------------------------------------------------------------------
def open_dst(src, dst):
    new = not os.path.exists(dst)
    con = sqlite3.connect(dst)
    con.execute("ATTACH DATABASE ? AS src", (src,))
    if new:
        for (sql,) in con.execute("SELECT sql FROM src.sqlite_master WHERE type='table' "
                                  "AND name NOT LIKE 'sqlite_%'").fetchall():
            con.execute(sql)
        for (sql,) in con.execute("SELECT sql FROM src.sqlite_master WHERE type='index' "
                                  "AND sql IS NOT NULL").fetchall():
            con.execute(sql)
        for c in EXTRA_COLS:
            con.execute(f"ALTER TABLE ergodic_pairs_3d ADD COLUMN {c} REAL")
        for t in ('dichten', 'blickwinkel'):
            if con.execute("SELECT 1 FROM src.sqlite_master WHERE name=?", (t,)).fetchone():
                con.execute(f"INSERT INTO main.{t} SELECT * FROM src.{t}")
        con.execute("CREATE TABLE refine_meta (key TEXT PRIMARY KEY, value TEXT)")
        con.commit()
    return con


def write_batch(con, batch, x_ref, before, off, after, loss, Bs, standoff):
    src_cols = [r[1] for r in con.execute("PRAGMA src.table_info(ergodic_pairs_3d)")]
    cols = src_cols + EXTRA_COLS
    ids = [e['id'] for e in batch]
    rows = {r[0]: list(r) for r in con.execute(
        f"SELECT * FROM src.ergodic_pairs_3d WHERE id IN ({','.join('?' * len(ids))})", ids)}
    ci = {c: i for i, c in enumerate(src_cols)}
    x = x_ref.cpu().numpy().astype(np.float64)
    out = []
    for i, e in enumerate(batch):
        r = rows[e['id']]
        pos = (Bs @ x[i, :, :3]).astype(np.float32)
        rot = (Bs @ x[i, :, 3:]).astype(np.float32)
        r[ci['traj_pos']], r[ci['traj_rot6']] = pos.tobytes(), rot.tobytes()
        r[ci['start_pos']] = x[i, 0, :3].astype(np.float32).tobytes()
        r[ci['n_points']] = N_STORE
        r[ci['standoff']] = standoff
        r[ci['standoff_mean']] = float(after['so'][i])
        r[ci['standoff_sd']] = float(after['so_sd'][i])
        r += [float(before['erg'][i]), float(off['erg'][i]), float(after['erg'][i]),
              float(before['so'][i]), float(after['so'][i]), float(before['pt'][i]),
              float(after['pt'][i]), float(before['len'][i]), float(after['len'][i]),
              float(loss[i])]
        out.append(r)
    con.executemany(f"INSERT INTO main.ergodic_pairs_3d ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})", out)
    con.commit()


def apply_filter(con, a):
    if con.execute("SELECT 1 FROM refine_meta WHERE key='filtered'").fetchone():
        print("filter already applied")
        return
    erg = np.array([v for (v,) in con.execute(
        "SELECT refine_erg1 FROM ergodic_pairs_3d WHERE split='train'")], dtype=float)
    thr = float(np.nanpercentile(erg, a.filter_pct))
    q = ("FROM ergodic_pairs_3d WHERE split='train' AND (refine_erg1 IS NULL "
         "OR refine_erg1 != refine_erg1 OR refine_erg1 > ? OR ABS(refine_so1 - ?) > ?)")
    par = (thr, a.standoff, a.max_so_dev)
    n_drop = con.execute("SELECT COUNT(*) " + q, par).fetchone()[0]
    con.execute("DELETE " + q, par)
    stats = dict(filter_pct=a.filter_pct, erg_threshold=thr, max_so_dev=a.max_so_dev,
                 dropped=n_drop, train_before=len(erg), train_after=len(erg) - n_drop)
    con.execute("INSERT INTO refine_meta VALUES ('filtered', ?)", (json.dumps(stats),))
    con.commit()
    con.execute("DETACH DATABASE src")
    con.execute("VACUUM")
    print(f"filter: dropped {n_drop} of {len(erg)} train entries "
          f"(Sinkhorn > {thr:.3e} = p{a.filter_pct:g}, or |standoff - {a.standoff}| > {a.max_so_dev})")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def report(dst, out_dir, examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    con = sqlite3.connect(dst)
    cols = ['split', 'gruppe'] + EXTRA_COLS
    rows = con.execute(f"SELECT {','.join(cols)} FROM ergodic_pairs_3d").fetchall()
    meta = dict(con.execute("SELECT key, value FROM refine_meta").fetchall())
    con.close()
    D = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
    num = {c: D[c].astype(float) for c in EXTRA_COLS}

    def line(name, m):
        f = lambda c: np.nanmedian(num[c][m])
        return (f"| {name} | {m.sum()} | {f('refine_erg0'):.3e} | {f('refine_erg_off'):.3e} | "
                f"{f('refine_erg1'):.3e} | {f('refine_so0'):.3f} | {f('refine_so1'):.3f} | "
                f"{f('refine_pt0'):.1f} | {f('refine_pt1'):.1f} | {f('refine_len0'):.2f} | "
                f"{f('refine_len1'):.2f} |")

    md = ["# DB refinement report", "",
          "Medians. erg = Sinkhorn footprint error; so / pt = standoff / pointing against a "
          "local PCA plane (surface_metrics.py); len = control-polygon length. "
          "0 = projected teacher (on the surface), off = moved 0.12 along the normal, "
          "1 = refined.", "",
          "| subset | n | erg0 | erg_off | erg1 | so0 | so1 | pt0 [deg] | pt1 [deg] | len0 | len1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    md.append(line('all', np.ones(len(rows), bool)))
    for s in sorted(set(D['split'])):
        md.append(line(s, D['split'] == s))
    for g in sorted(set(D['gruppe'])):
        md.append(line(f"group {g}", D['gruppe'] == g))
    if 'filtered' in meta:
        md += ["", f"Filter: `{meta['filtered']}`"]
    open(os.path.join(out_dir, 'refine_report.md'), 'w').write("\n".join(md) + "\n")
    print("\n".join(md))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor='white')
    spec = [('Sinkhorn footprint error', 'refine_erg0', 'refine_erg1', True),
            ('Standoff to local plane', 'refine_so0', 'refine_so1', False),
            ('Pointing angle to normal [deg]', 'refine_pt0', 'refine_pt1', False)]
    for ax, (title, c0, c1, log) in zip(axes, spec):
        ax.set_facecolor('white'); ax.grid(alpha=0.2)
        for s in ax.spines.values():
            s.set_color('#cccccc')
        v0, v1 = num[c0][np.isfinite(num[c0])], num[c1][np.isfinite(num[c1])]
        lo, hi = min(v0.min(), v1.min()), max(v0.max(), v1.max())
        bins = (np.geomspace(max(lo, 1e-6), hi, 60) if log else np.linspace(lo, hi, 60))
        ax.hist(v0, bins=bins, color='#1565C0', alpha=0.6, label='projected teacher')
        ax.hist(v1, bins=bins, color='#00C853', alpha=0.6, label='refined')
        if log:
            ax.set_xscale('log')
        ax.set_title(title, color='#1A1A2E', fontsize=10)
        ax.tick_params(colors='#555555', labelsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle('DB targets before and after refinement', color='#1A1A2E', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'refine_histograms.png'), dpi=150, facecolor='white')
    plt.close(fig)

    if examples:
        n = len(examples)
        fig = plt.figure(figsize=(4.6 * min(n, 3), 4.6 * ((n + 2) // 3)), facecolor='white')
        for k, ex in enumerate(examples):
            ax = fig.add_subplot((n + 2) // 3, min(n, 3), k + 1, projection='3d')
            p = ex['parts']
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=6, c='#444444', alpha=0.3, lw=0)
            ax.plot(*ex['before'].T, color='#1565C0', lw=2.5, alpha=0.9, label='projected teacher')
            ax.plot(*ex['after'].T, color='#00C853', lw=2.2, alpha=0.95, label='refined')
            ax.set_title(f"{ex['name']} / {ex['surface']}\nSinkhorn {ex['e0']:.1e} -> {ex['e1']:.1e}",
                         fontsize=8, color='#1A1A2E')
            ax.tick_params(labelsize=6, colors='#555555')
            if k == 0:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'refine_examples.png'), dpi=150, facecolor='white')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', default='ergodic_dataset_3d_lm.db')
    p.add_argument('--dst', default='ergodic_dataset_3d_lm_ref.db')
    p.add_argument('--limit', type=int, default=None, help='only this many entries (smoke test)')
    p.add_argument('--filter_pct', type=float, default=98.0)
    p.add_argument('--max_so_dev', type=float, default=0.06)
    p.add_argument('--report_dir', default=os.path.join(_here, 'results', 'db_refine'))
    p.add_argument('--n_examples', type=int, default=6)
    p.add_argument('--device', default=None)
    add_refine_args(p)
    a = p.parse_args()

    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    t0 = time.time()
    entries = pick(load_surface_db(a.src, splits=SPLITS, target_cps=True), a.limit)
    print(f"{len(entries)} entries loaded in {time.time() - t0:.0f} s")
    con = open_dst(a.src, a.dst)
    done = {i for (i,) in con.execute("SELECT id FROM main.ergodic_pairs_3d")}
    # After the filter the dropped rows are missing from the destination on
    # purpose; a resumed run must not refine them again.
    filtered = con.execute("SELECT 1 FROM refine_meta WHERE key='filtered'").fetchone()
    todo = [] if filtered else [e for e in entries if e['id'] not in done]
    print(f"{len(done)} already refined, {len(todo)} to do")

    ref = refiner_from_args(a, dev)
    Bs = bspline_basis_matrix(25, N_STORE, 5).astype(np.float64)
    rng = np.random.default_rng(0)
    pool = [e['id'] for e in todo if e['split'] != 'train']
    ex_ids = set(rng.choice(pool, min(a.n_examples, len(pool)), replace=False).tolist()
                 if pool else [])
    examples, t1 = [], time.time()
    for b0 in range(0, len(todo), a.batch):
        batch = todo[b0:b0 + a.batch]
        x1 = torch.from_numpy(np.stack([e['x1'] for e in batch])).to(dev)
        parts = torch.from_numpy(np.stack([e['parts'] for e in batch])).to(dev)
        x_off = ref.offset(x1)
        before, off = ref.metrics(x1, parts), ref.metrics(x_off, parts)
        x_ref, loss = ref.run(x_off, parts, a.iters, a.lr)
        after = ref.metrics(x_ref, parts)
        cpu = lambda d: {k: v.cpu().numpy() for k, v in d.items()}
        write_batch(con, batch, x_ref, cpu(before), cpu(off), cpu(after), loss.cpu().numpy(),
                    Bs, a.standoff)
        for i, e in enumerate(batch):
            if e['id'] in ex_ids:
                Bc = bspline_basis_matrix(25, 256, 5)
                examples.append(dict(name=e['name'], surface=e['surface'], parts=e['parts'],
                                     before=Bc @ e['x1'][:, :3],
                                     after=Bc @ x_ref[i, :, :3].cpu().numpy(),
                                     e0=float(before['erg'][i]), e1=float(after['erg'][i])))
        n = b0 + len(batch)
        el = time.time() - t1
        print(f"  {n}/{len(todo)}  {el / n * 1000:.1f} ms/entry  ETA {el / n * (len(todo) - n) / 60:.0f} min"
              f"  | batch median Sinkhorn {before['erg'].median().item():.2e} -> "
              f"{after['erg'].median().item():.2e}  standoff {after['so'].median().item():.3f}"
              f"  pointing {after['pt'].median().item():.1f} deg", flush=True)

    apply_filter(con, a)
    con.close()
    report(a.dst, a.report_dir, examples)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {a.dst}")


if __name__ == '__main__':
    main()
