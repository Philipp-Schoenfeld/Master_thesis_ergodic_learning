#!/usr/bin/env python3
r"""
check_obstacle_runs.py
======================
Independent collision check of an obstacle-benchmark DB (`run_exploration_comparison.py
--obstacles ...`): every stored driven segment is re-checked against the obstacles recorded in the
DB's `meta` table (densified to 0.002, physical radius), next to the per-method timing, path length
and final errors. Exit code 1 if any driven path enters a physical obstacle.

    python check_obstacle_runs.py --db plots/exploration_comparison_obstacles_dice6_r0.08/....db
"""
import argparse
import ast
import sqlite3
import sys

import numpy as np

import sample_annealed_mission  # noqa: E402,F401  (sets up the import paths of the pipeline)
import workspace_obstacles as wo  # noqa: E402
from obstacles import CircleObstacle, CompositeObstacle


def load_obstacles(conn, layout):
    meta = dict(conn.execute('SELECT key, value FROM meta').fetchall())
    if 'obstacle_centers' not in meta:
        # convergence-test DBs (sweep.db) carry no meta data: fall back to the named layout
        if layout == 'none':
            raise SystemExit('DB has no obstacle meta data; pass --layout (e.g. dice6)')
        obst = wo.make_obstacles(layout, wo.DEFAULT_RADIUS, wo.DEFAULT_MARGIN)
        return obst, dict(obstacle_radius=wo.DEFAULT_RADIUS, obstacle_margin=wo.DEFAULT_MARGIN,
                          w_obstacle='?')
    centers = ast.literal_eval(meta['obstacle_centers'])
    r, m = float(meta['obstacle_radius']), float(meta['obstacle_margin'])
    return CompositeObstacle([CircleObstacle(c, r, m) for c in centers]), meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', required=True)
    ap.add_argument('--layout', default='none', help='Layout for DBs without obstacle meta data (sweep.db).')
    a = ap.parse_args()
    conn = sqlite3.connect(f'file:{a.db}?mode=ro', uri=True)
    obst, meta = load_obstacles(conn, a.layout)
    print(f'{len(obst.obstacles)} obstacles, radius {meta["obstacle_radius"]}, '
          f'margin {meta["obstacle_margin"]}, w_obstacle {meta.get("w_obstacle")}')
    q = ('SELECT u.method, u.kernel, u.id, u.seconds FROM runs u ORDER BY u.method, u.kernel')
    stats = {}
    for method, kernel, rid, sec in conn.execute(q).fetchall():
        rows = conn.execute('SELECT round, segment, seg_len, obs_viol_ref, E_truth, E_truth_free '
                            'FROM rounds WHERE run_id=? ORDER BY round', (rid,)).fetchall()
        pen, inside_len, total_len = 0.0, 0.0, 0.0
        for _, blob, seg_len, *_ in rows:
            seg = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2).astype(np.float64)
            st = wo.path_stats(seg, obst)
            pen = max(pen, st['pen_max'])
            inside_len += st['len_in_frac'] * (seg_len or 0.0)
            total_len += seg_len or 0.0
        viol_ref = [r[3] for r in rows if r[3] is not None and r[3] == r[3]]
        s = stats.setdefault((method, kernel), dict(n=0, pen=[], in_len=[], sec=[], len=[],
                                                    vref=[], E=[], Ef=[]))
        s['n'] += 1
        s['pen'].append(pen)
        s['in_len'].append(inside_len)
        s['sec'].append(sec or np.nan)
        s['len'].append(total_len / max(len(rows), 1))
        s['vref'] += viol_ref
        s['E'].append(rows[-1][4])
        s['Ef'].append(rows[-1][5])
    print(f'{"method":14}{"kern":5}{"runs":>5}{"max pen":>9}{"runs hit":>9}{"s/mission":>10}'
          f'{"len/round":>10}{"viol pre-clear":>15}{"E_truth":>9}{"E_free":>8}')
    bad = False
    for (m, k), s in sorted(stats.items()):
        pen = np.array(s['pen'])
        hit = int((pen > 0).sum())
        bad |= hit > 0
        vref = np.array([v for v in s['vref'] if v is not None], dtype=float)
        vref = vref if len(vref) else np.array([np.nan])
        print(f'{m:14}{k:5}{s["n"]:5}{pen.max():9.4f}{hit:9}{np.nanmean(s["sec"]):10.1f}'
              f'{np.mean(s["len"]):10.2f}{np.nanmean(vref):9.4f}/{np.nanmax(vref):.3f}'
              f'{np.nanmean(np.array(s["E"], dtype=float)):9.2f}{np.nanmean(np.array(s["Ef"], dtype=float)):8.2f}')
    print('COLLISION-FREE' if not bad else 'COLLISIONS FOUND')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
