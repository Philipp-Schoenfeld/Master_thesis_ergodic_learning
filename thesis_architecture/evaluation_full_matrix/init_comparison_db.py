r"""
init_comparison_db.py -- SQLite store for the init-method x solver comparison
================================================================================
Same storage convention as `smoothness_db.py` (one row = one (shape, variant,
cand_idx) trajectory, every intermediate solver state packed via
`state_codec.pack_states` so mean/variance of any metric can be recomputed
without re-running a solver), kept as a separate module/schema so the
existing smoothness-comparison results and scripts are untouched.

`variant` here encodes BOTH the initialization method and the solver,
`f"{method}__{solver}"` (e.g. `cfm__tsvec_svgd`, `heuristic__adam_svgd`) --
14 combinations for 7 methods x 2 solvers.

One extra column vs. `smoothness_db.py`: `outside_series` -- the fraction of
a trajectory's waypoints lying outside the target density's support (below
`--outside_thresh` of that shape's own max density) at every logged state,
one of the two new metrics this comparison adds (the other, time-to-
convergence/n90, is derived downstream from `ergodic_series`, which -- like
in `plot_smoothness_comparison.py` -- is computed from the stored `states`
at analysis time rather than cached here).
"""
import json
import os
import sqlite3
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shape TEXT NOT NULL,
    variant TEXT NOT NULL,
    method TEXT NOT NULL,
    solver TEXT NOT NULL,
    cand_idx INTEGER NOT NULL,
    init_param REAL,
    n_iters INTEGER NOT NULL,
    nxi INTEGER NOT NULL,
    n_points INTEGER NOT NULL,
    log_space TEXT NOT NULL,
    smoothness_weight REAL NOT NULL,
    n_states INTEGER NOT NULL,
    init_curve BLOB NOT NULL,
    states BLOB NOT NULL,
    smooth_series BLOB NOT NULL,
    path_len_series BLOB NOT NULL,
    outside_series BLOB NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(shape, variant, cand_idx)
);
CREATE INDEX IF NOT EXISTS idx_runs_lookup ON runs(shape, variant);
CREATE INDEX IF NOT EXISTS idx_runs_method_solver ON runs(shape, method, solver);
CREATE TABLE IF NOT EXISTS truths (
    shape TEXT PRIMARY KEY, res INTEGER NOT NULL, density BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS basis (
    nxi INTEGER NOT NULL, n_points INTEGER NOT NULL, degree INTEGER NOT NULL,
    matrix BLOB NOT NULL, PRIMARY KEY (nxi, n_points, degree)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def open_db(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()


def set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, default=str)))


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def save_basis(conn, B, nxi, n_points, degree):
    conn.execute("INSERT OR REPLACE INTO basis VALUES (?,?,?,?)", (nxi, n_points, degree, f32(B)))


def save_truth(conn, shape, density):
    d = np.asarray(density, dtype=np.float32)
    conn.execute("INSERT OR REPLACE INTO truths VALUES (?,?,?)", (shape, d.shape[-1], d.tobytes()))


def load_truth(conn, shape):
    res, blob = conn.execute("SELECT res, density FROM truths WHERE shape=?", (shape,)).fetchone()
    return np.frombuffer(blob, dtype=np.float32).reshape(res, res)


def existing_cands(conn, shape, variant):
    """Set of cand_idx already stored for (shape, variant) -- resume support."""
    return {r[0] for r in conn.execute(
        "SELECT cand_idx FROM runs WHERE shape=? AND variant=?", (shape, variant))}


def save_runs(conn, shape, method, solver, rows):
    """`rows`: list of dicts with cand_idx, init_param, n_iters, nxi, n_points,
    log_space, smoothness_weight, n_states, init_curve (array), states
    (packed bytes), smooth_series, path_len_series, outside_series (arrays).
    One transaction."""
    variant = f"{method}__{solver}"
    now = time.time()
    conn.executemany(
        """INSERT OR REPLACE INTO runs
           (shape, variant, method, solver, cand_idx, init_param, n_iters, nxi, n_points,
            log_space, smoothness_weight, n_states, init_curve, states, smooth_series,
            path_len_series, outside_series, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(shape, variant, method, solver, int(r['cand_idx']),
          None if r.get('init_param') is None else float(r['init_param']),
          int(r['n_iters']), int(r['nxi']), int(r['n_points']), r['log_space'],
          float(r['smoothness_weight']), int(r['n_states']), f32(r['init_curve']),
          r['states'], f32(r['smooth_series']), f32(r['path_len_series']),
          f32(r['outside_series']), now)
         for r in rows])
    conn.commit()


def iter_runs(conn, with_states=False, **filters):
    """Yield run rows (decoded arrays); `with_states=True` adds 'cps'
    (n_states, nxi, 2) float32, decoded via `state_codec.unpack_states`."""
    from state_codec import unpack_states
    cols = ['shape', 'variant', 'method', 'solver', 'cand_idx', 'init_param', 'n_iters', 'nxi',
            'n_points', 'log_space', 'smoothness_weight', 'n_states', 'init_curve',
            'smooth_series', 'path_len_series', 'outside_series']
    if with_states:
        cols.append('states')
    where, params = [], []
    for k, v in filters.items():
        if v is None:
            continue
        where.append(f'{k}=?')
        params.append(v)
    q = (f"SELECT {','.join(cols)} FROM runs" +
        (' WHERE ' + ' AND '.join(where) if where else '') + ' ORDER BY shape, variant, cand_idx')
    for row in conn.execute(q, params):
        d = dict(zip(cols, row))
        d['init_curve'] = np.frombuffer(d['init_curve'], dtype=np.float32).reshape(-1, 2)
        d['smooth_series'] = np.frombuffer(d['smooth_series'], dtype=np.float32)
        d['path_len_series'] = np.frombuffer(d['path_len_series'], dtype=np.float32)
        d['outside_series'] = np.frombuffer(d['outside_series'], dtype=np.float32)
        if with_states:
            d['cps'] = unpack_states(d.pop('states'), d['n_states'], d['nxi'])
        yield d
