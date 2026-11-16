"""Rebuilds the combined 10-method comparison data when the main DB is not available.

Takes metrics_all.csv of the first run (5 methods, tracked in git) and appends all rounds of the
extra DBs (plots/exploration_comparison_extra_{a,b,c}), then writes
plots/exploration_comparison_combined/{metrics_all.csv, exploration_comparison.db (runs only)}.
Afterwards: python plot_exploration_summary.py --dir plots/exploration_comparison_combined
Note: the reach/path-length columns exist only for the new methods (old runs table is missing).
"""
import csv
import os
import sqlite3

BASE = 'plots/exploration_comparison'
OUT = 'plots/exploration_comparison_combined'
TAGS = 'abc'

os.makedirs(OUT, exist_ok=True)
with open(f'{BASE}/metrics_all.csv', newline='') as f:
    rows = list(csv.DictReader(f))
cols = list(rows[0].keys())
out_db_path = f'{OUT}/exploration_comparison.db'
if os.path.exists(out_db_path):
    os.remove(out_db_path)
out_db = sqlite3.connect(out_db_path)
out_db.execute('create table runs(shape,kernel,method,variant,reached90,reached99,'
               'path_len_90,path_len_99,seconds)')
for t in TAGS:
    c = sqlite3.connect(f'plots/exploration_comparison_extra_{t}/exploration_comparison_extra.db')
    sel = ['r.round', 'u.shape', 'u.kernel', 'u.method', 'u.variant'] + [f'r.{k}' for k in cols[5:]]
    for rec in c.execute(f'SELECT {",".join(sel)} FROM rounds r JOIN runs u ON u.id=r.run_id'):
        rows.append({k: ('' if v is None else v) for k, v in zip(cols, rec)})
    out_db.executemany('insert into runs values(?,?,?,?,?,?,?,?,?)', c.execute(
        'select shape,kernel,method,variant,reached90,reached99,path_len_90,path_len_99,seconds '
        'from runs'))
out_db.commit()
with open(f'{OUT}/metrics_all.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print(len(rows), 'rows,', sorted({r['method'] for r in rows}))
