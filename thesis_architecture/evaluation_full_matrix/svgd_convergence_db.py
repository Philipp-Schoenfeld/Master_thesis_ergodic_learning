r"""
svgd_convergence_db.py -- SQLite store for the SVGD-convergence experiment
============================================================================
One row in `runs` = one initialisation that was refined by SVGD for
`n_iters` iterations. The row keeps

  * `init_curve`  -- the raw dense initialisation (T,2), exactly as produced
                     by the CFM / random-walk / linear generator,
  * `cps`         -- the B-spline control points (n_iters+1, nxi, 2) of EVERY
                     intermediate state (index 0 = B-spline fit of the init,
                     index i = solver output after iteration i),
  * `E_total`, `E_explore`, `E_exploit`, `path_len` -- per-state arrays
                     (n_iters+1,), computed against the TRUE density,

so any other metric can be recomputed later from the stored trajectories
(`render_states` turns control points back into the dense curves; the
`basis` table holds the exact B-spline basis that was used).

Supporting tables make the DB self-contained: `truths` (true density grid per
shape), `targets` (SVGD target density per shape/knowledge/strategy), `meta`
(JSON config). Everything numeric is stored as float32 blobs.
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
    knowledge_condition TEXT NOT NULL,
    strategy TEXT NOT NULL,          -- lse | ucb | eid (target density + CFM setting)
    method TEXT NOT NULL,            -- cfm | random_walk | linear
    init_idx INTEGER NOT NULL,
    init_param REAL,                 -- linear: angle in degrees, random_walk: seed
    n_iters INTEGER NOT NULL,
    nxi INTEGER NOT NULL,
    n_points INTEGER NOT NULL,
    E_raw_init REAL,                 -- E_ergodic_total of the raw dense init
    init_curve BLOB NOT NULL,        -- (n_points, 2) float32
    cps BLOB NOT NULL,               -- (n_iters+1, nxi, 2) float32
    E_total BLOB NOT NULL,           -- (n_iters+1,) float32
    E_explore BLOB NOT NULL,
    E_exploit BLOB NOT NULL,
    path_len BLOB NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(shape, knowledge_condition, strategy, method, init_idx)
);
CREATE INDEX IF NOT EXISTS idx_runs_group
    ON runs(knowledge_condition, strategy, method, shape);
-- Continuation of a run beyond its original n_iters (extend_svgd_convergence.py):
-- states start_iter .. n_iters_total (n_iters_total - start_iter + 1 entries, the
-- ORIGINAL rows in `runs` stay untouched). The continuation is an exact re-run of
-- the whole trajectory with the same seed; `prefix_max_abs_diff` records how far
-- the re-run's first states deviate from the stored ones (0.0 = bitwise equal).
CREATE TABLE IF NOT EXISTS runs_ext (
    run_id INTEGER PRIMARY KEY REFERENCES runs(id),
    start_iter INTEGER NOT NULL,
    n_iters_total INTEGER NOT NULL,
    cps BLOB NOT NULL,
    E_total BLOB NOT NULL,
    E_explore BLOB NOT NULL,
    E_exploit BLOB NOT NULL,
    path_len BLOB NOT NULL,
    prefix_max_abs_diff REAL NOT NULL,
    prefix_E_max_rel_diff REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS truths (
    shape TEXT PRIMARY KEY, res INTEGER NOT NULL, density BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS targets (
    shape TEXT NOT NULL, knowledge_condition TEXT NOT NULL, strategy TEXT NOT NULL,
    res INTEGER NOT NULL, density BLOB NOT NULL,
    PRIMARY KEY (shape, knowledge_condition, strategy)
);
CREATE TABLE IF NOT EXISTS basis (
    nxi INTEGER NOT NULL, n_points INTEGER NOT NULL, degree INTEGER NOT NULL,
    matrix BLOB NOT NULL, PRIMARY KEY (nxi, n_points, degree)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_ARRAY_COLS = ('cps', 'E_total', 'E_explore', 'E_exploit', 'path_len')


def open_db(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()


def existing_keys(conn, shape, knowledge_condition, strategy):
    """{(method, init_idx)} already stored for one block -- resume support."""
    cur = conn.execute(
        "SELECT method, init_idx FROM runs WHERE shape=? AND knowledge_condition=? "
        "AND strategy=?", (shape, knowledge_condition, strategy))
    return set(cur.fetchall())


def save_run(conn, r, commit=False):
    """`r`: dict with keys shape, knowledge_condition, strategy, method,
    init_idx, init_param, n_iters, nxi, init_curve (T,2), E_raw_init, cps,
    E_total, E_explore, E_exploit, path_len."""
    conn.execute(
        """INSERT OR REPLACE INTO runs
           (shape, knowledge_condition, strategy, method, init_idx, init_param,
            n_iters, nxi, n_points, E_raw_init, init_curve, cps, E_total,
            E_explore, E_exploit, path_len, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (r['shape'], r['knowledge_condition'], r['strategy'], r['method'],
         int(r['init_idx']), None if r.get('init_param') is None else float(r['init_param']),
         int(r['n_iters']), int(r['nxi']), int(r['init_curve'].shape[0]),
         float(r['E_raw_init']), _f32(r['init_curve']), _f32(r['cps']),
         _f32(r['E_total']), _f32(r['E_explore']), _f32(r['E_exploit']),
         _f32(r['path_len']), time.time()))
    if commit:
        conn.commit()


def save_run_ext(conn, run_id, start_iter, n_iters_total, cps, E_total, E_explore,
                 E_exploit, path_len, prefix_max_abs_diff, prefix_E_max_rel_diff,
                 commit=False):
    """States `start_iter .. n_iters_total` of run `run_id` (arrays with
    n_iters_total - start_iter + 1 entries along axis 0)."""
    conn.execute(
        """INSERT OR REPLACE INTO runs_ext
           (run_id, start_iter, n_iters_total, cps, E_total, E_explore, E_exploit,
            path_len, prefix_max_abs_diff, prefix_E_max_rel_diff, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (int(run_id), int(start_iter), int(n_iters_total), _f32(cps), _f32(E_total),
         _f32(E_explore), _f32(E_exploit), _f32(path_len),
         float(prefix_max_abs_diff), float(prefix_E_max_rel_diff), time.time()))
    if commit:
        conn.commit()


def ext_run_ids(conn):
    return {r[0] for r in conn.execute("SELECT run_id FROM runs_ext")}


def save_truth(conn, shape, density):
    d = np.asarray(density, dtype=np.float32)
    conn.execute("INSERT OR REPLACE INTO truths VALUES (?,?,?)",
                 (shape, d.shape[-1], d.tobytes()))


def save_target(conn, shape, knowledge_condition, strategy, density):
    d = np.asarray(density, dtype=np.float32)
    conn.execute("INSERT OR REPLACE INTO targets VALUES (?,?,?,?,?)",
                 (shape, knowledge_condition, strategy, d.shape[-1], d.tobytes()))


def save_basis(conn, B, nxi, n_points, degree):
    conn.execute("INSERT OR REPLACE INTO basis VALUES (?,?,?,?)",
                 (nxi, n_points, degree, _f32(B)))


def set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                 (key, json.dumps(value, default=str)))


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def load_basis(conn, nxi, n_points, degree=5):
    row = conn.execute("SELECT matrix FROM basis WHERE nxi=? AND n_points=? AND degree=?",
                       (nxi, n_points, degree)).fetchone()
    return np.frombuffer(row[0], dtype=np.float32).reshape(n_points, nxi)


def load_truth(conn, shape):
    res, blob = conn.execute("SELECT res, density FROM truths WHERE shape=?",
                             (shape,)).fetchone()
    return np.frombuffer(blob, dtype=np.float32).reshape(res, res)


def load_target(conn, shape, knowledge_condition, strategy):
    res, blob = conn.execute(
        "SELECT res, density FROM targets WHERE shape=? AND knowledge_condition=? "
        "AND strategy=?", (shape, knowledge_condition, strategy)).fetchone()
    return np.frombuffer(blob, dtype=np.float32).reshape(res, res)


def _decode(row, cols, with_states=True):
    d = dict(zip(cols, row))
    n1 = d['n_iters'] + 1
    d['init_curve'] = np.frombuffer(d['init_curve'], dtype=np.float32).reshape(
        d['n_points'], 2)
    for k in _ARRAY_COLS:
        if k not in d:
            continue
        if d[k] is None:
            continue
        a = np.frombuffer(d[k], dtype=np.float32)
        d[k] = a.reshape(n1, d['nxi'], 2) if k == 'cps' else a.reshape(n1)
    return d


SHARED = 'all'


def iter_runs(conn, columns=('E_total',), expand_shared=False, include_ext=False,
              **filters):
    """Yield decoded run dicts. `columns`: which per-state arrays to load
    (`'cps'` is the heavy one, ~200 KB per run); metadata columns and the
    init curve are always included. `filters`: any of shape,
    knowledge_condition, strategy, method, init_idx (AND-combined).

    Baseline rows that do not depend on knowledge/strategy (truth-target mode
    of run_svgd_convergence.py) are stored once with knowledge_condition /
    strategy == 'all'. With `expand_shared=True` a knowledge_condition /
    strategy filter also matches those rows, and they are yielded with the
    requested label (plus `shared=True`).

    `include_ext=True` appends the continuation states from `runs_ext` (if a
    run has one) to every requested per-state array and sets `n_iters` to the
    extended total, so `d['E_total']` has n_iters+1 entries covering the whole
    trajectory; `d['n_iters_stored']` keeps the original count."""
    meta_cols = ['id', 'shape', 'knowledge_condition', 'strategy', 'method',
                 'init_idx', 'init_param', 'n_iters', 'nxi', 'n_points',
                 'E_raw_init', 'init_curve']
    cols = meta_cols + [c for c in columns if c in _ARRAY_COLS]
    where, params = [], []
    for k, v in filters.items():
        if v is None:
            continue
        if expand_shared and k in ('knowledge_condition', 'strategy'):
            where.append(f"({k}=? OR {k}='{SHARED}')")
        else:
            where.append(f'{k}=?')
        params.append(v)
    q = f"SELECT {','.join(cols)} FROM runs" + \
        (' WHERE ' + ' AND '.join(where) if where else '') + ' ORDER BY id'
    for row in conn.execute(q, params):
        d = _decode(row, cols)
        d['n_iters_stored'] = d['n_iters']
        if include_ext:
            _append_ext(conn, d, [c for c in columns if c in _ARRAY_COLS])
        d['shared'] = False
        if expand_shared:
            for k in ('knowledge_condition', 'strategy'):
                if d[k] == SHARED and filters.get(k) is not None:
                    d[k] = filters[k]
                    d['shared'] = True
        yield d


def _append_ext(conn, d, array_cols):
    if not array_cols:
        return
    q = ("SELECT start_iter, n_iters_total, " + ",".join(array_cols) +
         " FROM runs_ext WHERE run_id=?")
    row = conn.execute(q, (d['id'],)).fetchone()
    if row is None:
        return
    start, total = row[0], row[1]
    if start != d['n_iters'] + 1:
        raise ValueError(f"run {d['id']}: extension starts at {start}, "
                         f"stored states end at {d['n_iters']}")
    n_ext = total - start + 1
    for col, blob in zip(array_cols, row[2:]):
        a = np.frombuffer(blob, dtype=np.float32)
        a = a.reshape(n_ext, d['nxi'], 2) if col == 'cps' else a.reshape(n_ext)
        d[col] = np.concatenate([d[col], a], axis=0)
    d['n_iters'] = total


def render_states(cps, B):
    """Control points (..., nxi, 2) -> dense curves (..., T, 2) via the
    stored basis matrix B (T, nxi)."""
    return np.einsum('pi,...id->...pd', B, cps)
