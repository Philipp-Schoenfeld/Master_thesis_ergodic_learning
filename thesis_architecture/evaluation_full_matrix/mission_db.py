r"""
mission_db.py -- SQLite store for the replanning-mission experiment
======================================================================
`run_mission_eval.py` plans a batch of candidate trajectories, drives ONE length
unit of the best one, updates the belief and replans, until the driven path has
covered 99 % of the ground-truth mass. Everything is stored in one SQLite file
("shard") per (knowledge condition, strategy, method) so that the 27 independent
mission sets never share a writer:

  rounds      one row per (shape, round): scalar metrics measured AFTER driving
              the unit (swept mass, ergodic error vs. the ground truth and vs.
              the target density, information gain, ...), the planning target
              density, the belief fields the planner saw (float16), the driven
              segment and the measurements that updated the belief (so a run can
              be resumed / replayed exactly),
  candidates  one row per (shape, round, candidate): the raw initialisation, the
              control points of EVERY SVGD state (`svgd_batched.pack_states`:
              16-bit quantised, delta coded, zlib) and the ergodic error against
              the planning target along the iterations (every `E_stride`-th
              state plus the last one),
  truths / basis / meta   make a shard self-contained.

Ergodic errors use `ExploreExploitErgodic` (K=10, W=600, the number reported
everywhere else in the project); J = E + LAMBDA_LEN_J * path length
(`metrics_explore_exploit.add_J`).
"""
import glob
import json
import os
import sqlite3
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shape TEXT NOT NULL,
    knowledge_condition TEXT NOT NULL,
    strategy TEXT NOT NULL,
    method TEXT NOT NULL,
    round INTEGER NOT NULL,           -- 0-based; n_exec = round + 1 length units driven
    n_exec INTEGER NOT NULL,
    start_x REAL, start_y REAL,       -- where the plan had to begin (end of the previous unit)
    selected_idx INTEGER,             -- candidate that was driven
    start_gap REAL,                   -- |first control point - start| of the driven candidate
    swept_mass REAL,                  -- fraction of the true mass within the coverage radius of the driven path
    reached99 INTEGER,                -- swept_mass >= threshold
    E_truth REAL, E_truth_explore REAL, E_truth_exploit REAL,   -- driven path vs. ground truth
    E_target REAL,                    -- driven path vs. the target density AFTER the belief update
    E_target_plan REAL,               -- driven path vs. the target density the round was planned for
    J_truth REAL, J_target REAL,      -- E + LAMBDA_LEN_J * path_len
    cov REAL, cov_norm REAL,
    path_len REAL, seg_len REAL,
    info_gain REAL,                   -- uncertainty removed by this unit (sum of sigma over the grid)
    info_gain_cum REAL,               -- ... since the initial belief
    unc_before REAL, unc_after REAL, belief_rmse REAL, n_obs INTEGER,
    E_cand_best REAL, E_cand_median REAL, E_cand_worst REAL,    -- final-iteration E vs. plan target
    target BLOB,                      -- (R, R) float32 planning target density
    mu_plan BLOB, sd_plan BLOB,       -- (R, R) float16 belief the planner saw
    segment BLOB,                     -- (n, 2) float32 driven unit
    obs_pts BLOB, obs_vals BLOB,      -- measurements that updated the belief
    created_at REAL NOT NULL,
    UNIQUE(shape, knowledge_condition, strategy, method, round)
);
CREATE INDEX IF NOT EXISTS idx_rounds_shape ON rounds(shape, round);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shape TEXT NOT NULL,
    knowledge_condition TEXT NOT NULL,
    strategy TEXT NOT NULL,
    method TEXT NOT NULL,
    round INTEGER NOT NULL,
    cand_idx INTEGER NOT NULL,
    selected INTEGER NOT NULL,
    init_param REAL,                  -- linear: heading in degrees, random walk: seed
    n_iters INTEGER NOT NULL,
    nxi INTEGER NOT NULL,
    n_states INTEGER NOT NULL,        -- stored SVGD states (state 0 = fit of the init)
    state_stride INTEGER NOT NULL,
    E_stride INTEGER NOT NULL,
    E_init REAL, E_final REAL,        -- raw init / final state vs. the planning target
    init_curve BLOB NOT NULL,         -- (n_points, 2) float32
    states BLOB NOT NULL,             -- svgd_batched.pack_states
    E_series BLOB NOT NULL,           -- float32, iterations 0, E_stride, 2 E_stride, ..., n_iters
    UNIQUE(shape, knowledge_condition, strategy, method, round, cand_idx)
);
CREATE INDEX IF NOT EXISTS idx_cand_round
    ON candidates(shape, round, selected);
CREATE TABLE IF NOT EXISTS truths (
    shape TEXT PRIMARY KEY, res INTEGER NOT NULL, density BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS basis (
    nxi INTEGER NOT NULL, n_points INTEGER NOT NULL, degree INTEGER NOT NULL,
    matrix BLOB NOT NULL, PRIMARY KEY (nxi, n_points, degree)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

#: scalar columns of `rounds` (everything but blobs / ids), in table order.
ROUND_SCALARS = (
    'shape', 'knowledge_condition', 'strategy', 'method', 'round', 'n_exec',
    'start_x', 'start_y', 'selected_idx', 'start_gap', 'swept_mass', 'reached99',
    'E_truth', 'E_truth_explore', 'E_truth_exploit', 'E_target', 'E_target_plan',
    'J_truth', 'J_target', 'cov', 'cov_norm', 'path_len', 'seg_len', 'info_gain',
    'info_gain_cum', 'unc_before', 'unc_after', 'belief_rmse', 'n_obs',
    'E_cand_best', 'E_cand_median', 'E_cand_worst')
ROUND_BLOBS = ('target', 'mu_plan', 'sd_plan', 'segment', 'obs_pts', 'obs_vals')


def shard_path(root, cond, strategy, method):
    return os.path.join(root, 'shards', f'{cond}__{strategy}__{method}.db')


def list_shards(root):
    return sorted(glob.glob(os.path.join(root, 'shards', '*.db')))


def open_db(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()


def f16(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float16)).tobytes()


def set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                 (key, json.dumps(value, default=str)))


def get_meta(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def save_basis(conn, B, nxi, n_points, degree):
    conn.execute("INSERT OR REPLACE INTO basis VALUES (?,?,?,?)",
                 (nxi, n_points, degree, f32(B)))


def load_basis(conn, nxi, n_points, degree=5):
    row = conn.execute("SELECT matrix FROM basis WHERE nxi=? AND n_points=? AND degree=?",
                       (nxi, n_points, degree)).fetchone()
    return np.frombuffer(row[0], dtype=np.float32).reshape(n_points, nxi)


def save_truth(conn, shape, density):
    d = np.asarray(density, dtype=np.float32)
    conn.execute("INSERT OR REPLACE INTO truths VALUES (?,?,?)",
                 (shape, d.shape[-1], d.tobytes()))


def load_truth(conn, shape):
    res, blob = conn.execute("SELECT res, density FROM truths WHERE shape=?",
                             (shape,)).fetchone()
    return np.frombuffer(blob, dtype=np.float32).reshape(res, res)


# -- writing -----------------------------------------------------------------

def save_round(conn, row, candidates):
    """Insert one (shape, round) row plus its candidate rows in ONE transaction
    (the commit is what makes a resumed run see either all of it or none).

    `row`: dict with the keys of ROUND_SCALARS and ROUND_BLOBS (numpy arrays
    are converted); `candidates`: list of dicts with cand_idx, selected,
    init_param, n_iters, nxi, n_states, state_stride, E_stride, E_init, E_final,
    init_curve (array), states (bytes), E_series (array)."""
    cols = list(ROUND_SCALARS) + list(ROUND_BLOBS) + ['created_at']
    vals = [row[c] for c in ROUND_SCALARS]
    for c in ROUND_BLOBS:
        v = row[c]
        if c in ('mu_plan', 'sd_plan'):
            vals.append(f16(v))
        elif isinstance(v, (bytes, type(None))):
            vals.append(v)
        else:
            vals.append(f32(v))
    vals.append(time.time())
    conn.execute(
        f"INSERT OR REPLACE INTO rounds ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        vals)
    for c in candidates:
        conn.execute(
            """INSERT OR REPLACE INTO candidates
               (shape, knowledge_condition, strategy, method, round, cand_idx, selected,
                init_param, n_iters, nxi, n_states, state_stride, E_stride, E_init,
                E_final, init_curve, states, E_series)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row['shape'], row['knowledge_condition'], row['strategy'], row['method'],
             row['round'], int(c['cand_idx']), int(c['selected']),
             None if c.get('init_param') is None else float(c['init_param']),
             int(c['n_iters']), int(c['nxi']), int(c['n_states']), int(c['state_stride']),
             int(c['E_stride']), float(c['E_init']), float(c['E_final']),
             f32(c['init_curve']), c['states'], f32(c['E_series'])))
    conn.commit()


# -- reading -----------------------------------------------------------------

def round_numbers(conn):
    """{shape: [completed round numbers]} -- resume support."""
    out = {}
    for shape, r in conn.execute("SELECT shape, round FROM rounds ORDER BY shape, round"):
        out.setdefault(shape, []).append(r)
    return out


def load_replay_rows(conn, shape):
    """Rows needed to rebuild the state of a shape: ordered by round, with the
    driven segment, the measurements and the reached flag."""
    rows = []
    for r, seg, op, ov, reached, n_exec in conn.execute(
            "SELECT round, segment, obs_pts, obs_vals, reached99, n_exec FROM rounds "
            "WHERE shape=? ORDER BY round", (shape,)):
        rows.append(dict(
            round=r, reached99=bool(reached), n_exec=n_exec,
            segment=np.frombuffer(seg, dtype=np.float32).reshape(-1, 2),
            obs_pts=np.frombuffer(op, dtype=np.float32).reshape(-1, 2),
            obs_vals=np.frombuffer(ov, dtype=np.float32)))
    return rows


def iter_rounds(conn, columns=ROUND_SCALARS, **filters):
    """Yield dicts of the requested `rounds` columns (scalars and/or blobs;
    blobs come back decoded to float32 arrays, mu/sd to float32 from float16),
    ordered by (shape, round). `filters`: equality filters on any scalar."""
    cols = list(columns)
    where, params = [], []
    for k, v in filters.items():
        if v is None:
            continue
        where.append(f'{k}=?')
        params.append(v)
    q = f"SELECT {','.join(cols)} FROM rounds" + \
        (' WHERE ' + ' AND '.join(where) if where else '') + ' ORDER BY shape, round'
    for row in conn.execute(q, params):
        d = dict(zip(cols, row))
        for c in ROUND_BLOBS:
            if c in d and d[c] is not None:
                if c in ('mu_plan', 'sd_plan'):
                    a = np.frombuffer(d[c], dtype=np.float16).astype(np.float32)
                    d[c] = a.reshape(int(round(np.sqrt(a.size))), -1)
                elif c == 'target':
                    a = np.frombuffer(d[c], dtype=np.float32)
                    d[c] = a.reshape(int(round(np.sqrt(a.size))), -1)
                elif c in ('segment', 'obs_pts'):
                    d[c] = np.frombuffer(d[c], dtype=np.float32).reshape(-1, 2)
                else:
                    d[c] = np.frombuffer(d[c], dtype=np.float32)
        yield d


def iter_candidates(conn, with_states=False, **filters):
    """Yield candidate rows. `with_states=True` decodes the SVGD states to a
    (n_states, nxi, 2) float32 array under 'cps'."""
    from state_codec import unpack_states
    cols = ['shape', 'knowledge_condition', 'strategy', 'method', 'round', 'cand_idx',
            'selected', 'init_param', 'n_iters', 'nxi', 'n_states', 'state_stride',
            'E_stride', 'E_init', 'E_final', 'init_curve', 'E_series']
    if with_states:
        cols.append('states')
    where, params = [], []
    for k, v in filters.items():
        if v is None:
            continue
        where.append(f'{k}=?')
        params.append(v)
    q = f"SELECT {','.join(cols)} FROM candidates" + \
        (' WHERE ' + ' AND '.join(where) if where else '') + \
        ' ORDER BY shape, round, cand_idx'
    for row in conn.execute(q, params):
        d = dict(zip(cols, row))
        d['init_curve'] = np.frombuffer(d['init_curve'], dtype=np.float32).reshape(-1, 2)
        d['E_series'] = np.frombuffer(d['E_series'], dtype=np.float32)
        if with_states:
            d['cps'] = unpack_states(d.pop('states'), d['n_states'], d['nxi'])
        yield d
