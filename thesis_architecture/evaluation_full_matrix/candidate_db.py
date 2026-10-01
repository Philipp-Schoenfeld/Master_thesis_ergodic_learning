r"""
candidate_db.py -- persist ALL raw candidates' B-spline control points
==========================================================================
Philipp's request (2026-10-01): today only the selected/refined winner's
dense rendered trajectory gets saved (`save_trajectory_files`); the other
29 raw candidates per decision point are generated, scored, and discarded.
That already cost real GPU time twice in this session (extend_svgd500.py,
generate_raw_pool_metrics.py both had to re-generate candidates from
scratch just to answer questions the original run could have answered for
free). Going forward, opt in with `--save_candidates_db` to persist every
candidate's B-spline CONTROL POINTS (not the dense rendered curve -- nxi
control points x 2D is far smaller, and the curve is always exactly
reconstructible from them via the project's own bspline render step) to a
SQLite database next to the run's other outputs. Anything computed from a
rendered/refined curve (any metric, at any threshold, any SVGD budget) can
be redone later from this DB without touching the network again.

Opt-in and additive by design: a run without the flag behaves exactly as
before (see CLAUDE.md's own convention on this). The DB lives at
`results/<out_tag>/candidates.db`.
"""
import os
import sqlite3
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shape TEXT NOT NULL,
    knowledge_condition TEXT NOT NULL,
    representation TEXT NOT NULL,
    strategy TEXT NOT NULL,
    replan_scheme TEXT NOT NULL,
    replan_round INTEGER NOT NULL,
    candidate_idx INTEGER NOT NULL,
    is_selected INTEGER NOT NULL,
    pre_svgd_score REAL,
    nxi INTEGER NOT NULL,
    nd INTEGER NOT NULL,
    control_points BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_lookup
    ON candidates(shape, knowledge_condition, representation, strategy, replan_scheme);
"""


def open_db(out_tag_dir):
    """One connection per process, kept open for the whole run -- SQLite
    handles concurrent single-writer access fine for this access pattern
    (one process, many sequential inserts, periodic commits)."""
    path = os.path.join(out_tag_dir, 'candidates.db')
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def save_candidates(conn, shape, knowledge_condition, representation, strategy,
                    replan_scheme, replan_round, cps, selected_idx, scores=None):
    """`cps`: (n_candidates, nxi, nd) tensor/array of B-spline control
    points, as returned by `planner.plan()` BEFORE rendering/refinement.
    `selected_idx`: which candidate was chosen (pre_svgd winner, or None if
    not yet decided -- e.g. post_svgd, where the winner is only known after
    all are refined; pass the post-refinement winner's index there instead,
    it's still useful as a record even though it's not the pre-SVGD pick).
    `scores`: optional per-candidate pre-SVGD score list, same order as cps.
    """
    cps_np = cps.detach().cpu().numpy() if hasattr(cps, 'detach') else np.asarray(cps)
    n, nxi, nd = cps_np.shape
    now = time.time()
    rows = [
        (shape, knowledge_condition, representation, strategy, replan_scheme,
         replan_round, i, 1 if i == selected_idx else 0,
         float(scores[i]) if scores is not None else None,
         nxi, nd, cps_np[i].astype(np.float64).tobytes(), now)
        for i in range(n)
    ]
    conn.executemany(
        """INSERT INTO candidates
           (shape, knowledge_condition, representation, strategy, replan_scheme,
            replan_round, candidate_idx, is_selected, pre_svgd_score, nxi, nd,
            control_points, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    conn.commit()


def load_candidates(conn, shape=None, knowledge_condition=None, representation=None,
                    strategy=None, replan_scheme=None):
    """Returns a list of dicts, each with a decoded (nxi, nd) numpy array
    under 'control_points'. Filters are AND-combined; omit any to widen."""
    clauses, params = [], []
    for col, val in [('shape', shape), ('knowledge_condition', knowledge_condition),
                     ('representation', representation), ('strategy', strategy),
                     ('replan_scheme', replan_scheme)]:
        if val is not None:
            clauses.append(f'{col} = ?')
            params.append(val)
    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    cur = conn.execute(f'SELECT * FROM candidates {where} ORDER BY id', params)
    cols = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        d['control_points'] = np.frombuffer(d['control_points'], dtype=np.float64).reshape(d['nxi'], d['nd'])
        out.append(d)
    return out
