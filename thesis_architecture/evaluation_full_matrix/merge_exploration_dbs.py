#!/usr/bin/env python3
r"""
merge_exploration_dbs.py
========================
Fold the runs of a second exploration-comparison DB (e.g. the one a cluster job of
`run_exploration_comparison.py --db ...` wrote for the missing methods) into the main DB
(`plots/exploration_comparison/exploration_comparison.db`).

Add-only: runs whose (shape, kernel, method, variant) key already exists in the target are skipped,
new runs get fresh ids together with all their `rounds` rows; missing `truths` are copied. The
target is never modified otherwise. The target is migrated to the current column set first (add-only); columns missing in the source become NULL.

    python merge_exploration_dbs.py --src plots/exploration_comparison_extra/exploration_comparison_extra.db
    python merge_exploration_dbs.py --src other.db --dst main.db --dry_run
"""
import argparse
import os

import run_exploration_comparison as rec

_here = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DST = os.path.join(_here, 'plots', 'exploration_comparison', 'exploration_comparison.db')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', default=DEFAULT_DST)
    ap.add_argument('--dry_run', action='store_true')
    a = ap.parse_args()
    if not os.path.isfile(a.src):
        raise SystemExit(f'source DB not found: {a.src}')
    dst = rec.ResultDB(a.dst)                       # creates / migrates the target (add-only)
    conn = dst.conn
    conn.execute('ATTACH DATABASE ? AS src', (os.path.abspath(a.src),))    # source stays untouched

    new_runs = conn.execute(
        'SELECT s.id, s.shape, s.kernel, s.method, s.variant FROM src.runs s WHERE NOT EXISTS ('
        'SELECT 1 FROM main.runs d WHERE d.shape=s.shape AND d.kernel=s.kernel AND d.method=s.method '
        'AND d.variant=s.variant) ORDER BY s.id').fetchall()
    by_method = {}
    for _, shape, kernel, method, variant in new_runs:
        by_method[method] = by_method.get(method, 0) + 1
    print(f'{len(new_runs)} new runs in {a.src}: {by_method or "-"}')
    skipped = conn.execute('SELECT COUNT(*) FROM src.runs').fetchone()[0] - len(new_runs)
    print(f'{skipped} runs already present in the target (skipped)')
    if a.dry_run or not new_runs:
        return

    run_cols = [r[1] for r in conn.execute('PRAGMA table_info(runs)') if r[1] != 'id']
    round_cols = [r[1] for r in conn.execute('PRAGMA table_info(rounds)') if r[1] != 'run_id']
    src_round_cols = {r[1] for r in conn.execute('PRAGMA src.table_info(rounds)')}
    sel_round = ','.join(c if c in src_round_cols else f'NULL AS {c}' for c in round_cols)
    with conn:
        conn.execute('INSERT OR IGNORE INTO main.truths SELECT * FROM src.truths')
        for sid, *_ in new_runs:
            cur = conn.execute(f'INSERT INTO main.runs ({",".join(run_cols)}) '
                               f'SELECT {",".join(run_cols)} FROM src.runs WHERE id=?', (sid,))
            conn.execute(f'INSERT INTO main.rounds (run_id,{",".join(round_cols)}) '
                         f'SELECT ?,{sel_round} FROM src.rounds WHERE run_id=?', (cur.lastrowid, sid))
    conn.execute('DETACH DATABASE src')
    total = conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]
    print(f'merged; target now holds {total} runs')


if __name__ == '__main__':
    main()
