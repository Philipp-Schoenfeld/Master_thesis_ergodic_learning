r"""
generate_dataset_lm.py
=======================
Training database with the teacher method recommended by the teacher study
(`thesis_architecture/teacher_study`, Kontext/2026-10-24_teacher_study.md,
config `lm_K16_cont_o2_w15`), on the densities and start points of
`generate_dataset_adam_svgd.py`:

* densities: 1172 base shapes of `ergodic_dataset_improved_final.db` + their
  `#unk0` large-area variant; `--n_starts` (3) start points each (source x0 +
  uniform random starts `<name>@s<k>`, extra starts of val -> `val_start`),
  i.e. the same rows, names and splits as the Adam-SVGD DB
  (`load_source_rows` / `expand_starts` are shared).
* per trajectory (density, start), exactly the study pipeline
  (`teacher_study.pipeline.run_cell`, reused, not copied): 30 chords from x0
  to a ring as initial curves -> batched Levenberg-Marquardt on the TSVEC
  residual energy with K=16 Fourier modes and the smoothness continuation
  150 > 50 > 15 (acceleration force), 128 waypoints, start pinned ->
  least-squares projection onto 25 control points (degree 5, control point 0
  = x0) -> polish stages (100 / 300 Adam steps in control-point space).
* selection (the study's rule, `analyze_teacher_study.select`): among the
  pool stages bs25 / bs25_pol100 / bs25_pol300 of the 30 candidates, drop
  diverged ones and those with points outside [0.03, 0.97]; take the
  SMOOTHEST (`smooth`) one whose Hellinger distance is at most `--eps` (0.05)
  above the best. If the boundary filter leaves nothing, the candidates with
  the fewest outside points are used (`selection_fallback` = 1).
  One filter on top of the study: candidates whose CONTROL POINTS leave
  [0, 1] are dropped as well. The study only scored the curve, but ~70% of
  the unpolished bs25 projections have control points outside [0, 1] (the
  degree-5 control polygon reaches beyond the curve), and the training
  augmentation clamps targets to [0, 1], which would distort exactly those
  curves. The polish stages keep their control points in [0.02, 0.98], so the
  pool keeps polished versions of the same candidates.

Stored per row: the schema of `generate_dataset_adam_svgd.py`
(`control_points` = the chosen 25 control points, `trajectory` = their
B-spline at 401 points, `ergodic_error` = K=8 error of that B-spline exactly as
the validator recomputes it, raw waypoints of the chosen candidate, ...) plus
`hellinger`, `smooth`, `selected_stage`, `selected_cand`, `n_eligible`,
`selection_fallback`.

Parallel parts: `--part i --n_parts n` solves every n-th trajectory into
`<out>.part<i>.db`; `--merge` joins the parts into `<out>`. Resumable: rows
already in the (part) DB are skipped.

    python generate_dataset_lm.py --out /tmp/x.db --names A,B --device cpu     # test
    python generate_dataset_lm.py --out ergodic_dataset_lm_nxi25.db --part 0 --n_parts 3
    python generate_dataset_lm.py --out ergodic_dataset_lm_nxi25.db --merge
"""
import argparse
import glob
import json
import os
import signal
import sqlite3
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, os.path.join(_arch, 'teacher_study')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import generate_dataset_adam_svgd as gen                           # noqa: E402
import pipeline as pl                                               # noqa: E402  (teacher_study)
import torch                                                        # noqa: E402

CONFIG_NAME = 'lm_K16_cont_o2_w15'
POOL = ('bs25', 'bs25_pol100', 'bs25_pol300')
LM_COLUMNS = (('hellinger', 'REAL'), ('smooth', 'REAL'), ('selected_stage', 'TEXT'),
              ('selected_cand', 'INTEGER'), ('n_eligible', 'INTEGER'),
              ('selection_fallback', 'INTEGER'))


def teacher_cfg():
    """The study's recommended config, with only the stages the selection pool needs."""
    from configs import full
    from dataclasses import replace
    cfg = next(c for c in full() if c.name == CONFIG_NAME)
    return replace(cfg, projections=((pl.NXI, pl.DEGREE),), polish_iters=(100, 300))


def shape_data(name, density_params, x0):
    """`pipeline.ShapeData` as `pipeline.load_shapes` builds it (density at TRUTH_RES, max-normalised)."""
    from shape_library import pdf_on_grid
    d, _, _ = pdf_on_grid(json.loads(density_params), resolution=pl.TRUTH_RES)
    d = np.asarray(d, dtype=np.float32)
    x0 = np.asarray(x0, dtype=np.float64)
    return pl.ShapeData(name, d / max(float(d.max()), 1e-12), x0, x0[None])


def select(stages, finals, eps, max_outside=0.0):
    """The study's rule on one cell -> (stage, cand, n_eligible, fallback)."""
    cand = []
    for st in POOL:
        m = stages[st]
        P = np.asarray(finals[st])
        cps_ok = ((P >= 0.0) & (P <= 1.0)).all(axis=(1, 2))
        for c in range(len(m['hellinger'])):
            cand.append((st, c, m['hellinger'][c], m['smooth'][c], m['outside'][c], m['diverged'][c],
                         bool(cps_ok[c])))
    ok = [x for x in cand if x[5] == 0 and x[6] and x[4] <= max_outside]
    fallback = not ok
    if fallback:
        live = [x for x in cand if x[5] == 0 and x[6]] or [x for x in cand if x[5] == 0] or cand
        least = min(x[4] for x in live)
        ok = [x for x in live if x[4] == least]
    h_best = min(x[2] for x in ok)
    elig = [x for x in ok if x[2] <= h_best + eps]
    st, c = min(elig, key=lambda x: x[3])[:2]
    return st, c, len(elig), fallback


def solve_one(cfg, sd, n_init, device, eps, seed=0):
    """One trajectory: the study's cell + selection -> dict(cps (25,2), raw (128,2), stage, cand,
    hellinger, smooth, n_eligible, fallback, init (128,2))."""
    stages, finals = pl.run_cell(cfg, sd, n_init, device, seed=seed)
    st, c, n_el, fb = select(stages, finals, eps)
    return dict(cps=np.clip(np.asarray(finals[st][c], dtype=np.float64), 0.0, 1.0), raw=np.asarray(finals['wp'][c], dtype=np.float64),
                stage=st, cand=int(c), hellinger=float(stages[st]['hellinger'][c]),
                smooth=float(stages[st]['smooth'][c]), n_eligible=n_el, fallback=fb,
                init=pl.make_inits(cfg, sd, n_init, seed)[c])


class K8Scorer:
    """K=8 ergodic error of a curve against the 128-grid density -- the definition the Adam-SVGD DB
    stores and `validate_adam_svgd_db.py` recomputes."""

    def __init__(self, nxi):
        from svgd_batched import K_FREQ
        self.K = K_FREQ
        self.bs, B32 = gen.make_solver(nxi, 'cpu', 'float64')
        self.B = torch.as_tensor(B32, dtype=torch.float64)
        self.cache = {}

    def phik(self, density_params):
        if density_params not in self.cache:
            self.cache[density_params] = gen.target_coeffs(gen.density_grid(density_params, 128)[None], self.K)[0]
        return torch.as_tensor(self.cache[density_params])

    def curve(self, X, density_params):
        return float(gen.ergodic_error(self.bs, torch.as_tensor(X, dtype=torch.float64)[None],
                                       self.phik(density_params)[None])[0])

    def cps(self, P, density_params):
        return self.curve(torch.matmul(self.B, torch.as_tensor(P, dtype=torch.float64)), density_params)


def open_lm_db(path):
    conn = gen.open_out_db(path)
    have = {r[1] for r in conn.execute("PRAGMA table_info(ergodic_pairs)")}
    for col, typ in LM_COLUMNS:
        if col not in have:
            conn.execute(f"ALTER TABLE ergodic_pairs ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


def merge(out):
    parts = sorted(glob.glob(f"{out}.part*.db"))
    if not parts:
        sys.exit(f"no parts {out}.part*.db")
    if os.path.exists(out):
        sys.exit(f"{out} exists, not overwritten")
    conn = open_lm_db(out)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(ergodic_pairs)") if r[1] != 'id']
    seen = set()
    for pth in parts:
        src = sqlite3.connect(pth)
        n = 0
        for row in src.execute(f"SELECT {', '.join(cols)} FROM ergodic_pairs ORDER BY id"):
            if row[0] in seen:
                continue
            conn.execute(f"INSERT INTO ergodic_pairs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", row)
            seen.add(row[0])
            n += 1
        src.close()
        conn.commit()
        print(f"  {os.path.basename(pth):<48} {n:6d} rows")
    print(f"  merged {len(seen)} rows -> {out}")
    conn.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src_db', default=gen.SRC_DB)
    p.add_argument('--out', required=True)
    p.add_argument('--variants', default='base,unk0')
    p.add_argument('--n_starts', type=int, default=3)
    p.add_argument('--n_init', type=int, default=30, help='candidates (chords) per trajectory')
    p.add_argument('--eps', type=float, default=0.05, help='Hellinger tolerance of the selection rule')
    p.add_argument('--seed', type=int, default=20261024, help='start-point seed (same as the Adam-SVGD DB)')
    p.add_argument('--part', type=int, default=None)
    p.add_argument('--n_parts', type=int, default=1)
    p.add_argument('--merge', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--names', default=None)
    a = p.parse_args()
    if a.merge:
        return merge(a.out)

    cfg = teacher_cfg()
    rows = gen.load_source_rows(a.src_db, tuple(a.variants.split(',')))
    if a.names:
        wanted = set(a.names.split(','))
        rows = [r for r in rows if r['name'] in wanted]
    if a.limit:
        rows = rows[:a.limit]
    rows = gen.expand_starts(rows, a.n_starts, a.seed)
    if a.part is not None:
        rows = rows[a.part::a.n_parts]
    path = a.out if a.part is None else f"{a.out}.part{a.part}.db"
    conn = open_lm_db(path)
    done = {r[0] for r in conn.execute("SELECT shape_name FROM ergodic_pairs")}
    todo = [r for r in rows if r['name'] not in done]
    print(f"  method   : teacher study {CONFIG_NAME}: {a.n_init} chords -> LM (K={cfg.K}, "
          f"continuation {cfg.schedule}, {cfg.n_iters} its) on {pl.N_POINTS} waypoints -> "
          f"B-spline nxi={pl.NXI} + polish {cfg.polish_iters} -> selection eps={a.eps}, boundary filter")
    print(f"  output   : {path}  ({len(rows)} trajectories in this part, done {len(rows) - len(todo)})",
          flush=True)
    if not todo:
        return

    config = dict(solver_name='lm_teacher', method=CONFIG_NAME, study_cfg=cfg.to_dict(), nxi=pl.NXI,
                  n_points=pl.N_POINTS, degree=pl.DEGREE, n_init=a.n_init, eps=a.eps, pool=list(POOL),
                  max_outside=0.0, truth_res=pl.TRUTH_RES, grid_res=128, n_starts=a.n_starts,
                  x0_margin=gen.X0_MARGIN, seed=a.seed, solve_space='lm_waypoints',
                  max_iters=cfg.n_iters)
    from obstacles import bspline_basis_matrix
    B_store = bspline_basis_matrix(pl.NXI, gen.TSTEPS_STORE + 1, pl.DEGREE)
    B128 = pl.basis(pl.NXI, pl.N_POINTS, pl.DEGREE)
    k8 = K8Scorer(pl.NXI)

    def _term(signum, frame):
        conn.commit()
        print("  [!] SIGTERM -- committed, exiting", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term)

    t0 = time.perf_counter()
    sd_cache = {}
    n_fb = 0
    for i, r in enumerate(todo):
        if r['density'] not in sd_cache:
            sd_cache.clear()            # rows of one density are adjacent (expand_starts)
            sd_cache[r['density']] = shape_data(r['density'], r['density_params'], r['x0'])
        base = sd_cache[r['density']]
        sd = pl.ShapeData(r['name'], base.density, np.asarray(r['x0'], dtype=np.float64),
                          np.asarray(r['x0'], dtype=np.float64)[None])
        out = solve_one(cfg, sd, a.n_init, a.device, a.eps)
        dp = r['density_params']
        pol = int(out['stage'].split('_pol')[1]) if '_pol' in out['stage'] else 0
        res = dict(cps=out['cps'], n_iters=cfg.n_iters + pol, erg=k8.cps(out['cps'], dp),
                   erg_init=k8.curve(out['init'], dp), converged=True,
                   history=np.array([k8.curve(out['init'], dp), k8.curve(out['raw'], dp)], dtype=np.float32),
                   raw=out['raw'], raw_erg=k8.curve(out['raw'], dp),
                   fit_rmse=float(np.linalg.norm(B128 @ out['cps'] - out['raw'], axis=1).mean()))
        gen.save_row(conn, r, res, B_store, config)
        conn.execute("UPDATE ergodic_pairs SET hellinger=?, smooth=?, selected_stage=?, selected_cand=?,"
                     " n_eligible=?, selection_fallback=? WHERE shape_name=?",
                     (out['hellinger'], out['smooth'], out['stage'], out['cand'], out['n_eligible'],
                      int(out['fallback']), r['name']))
        n_fb += int(out['fallback'])
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            conn.commit()
            el = time.perf_counter() - t0
            print(f"  [{i + 1}/{len(todo)}] {r['name']:<30} {out['stage']:<12} H={out['hellinger']:.3f} "
                  f"smooth={out['smooth']:.3f}  {el / (i + 1):.2f} s/traj, rest ~"
                  f"{el / (i + 1) * (len(todo) - i - 1) / 3600:.2f} h, fallbacks {n_fb}", flush=True)
    conn.commit()
    conn.close()
    print(f"  done: {len(todo)} trajectories in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()
