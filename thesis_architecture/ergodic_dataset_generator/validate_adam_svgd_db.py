r"""
validate_adam_svgd_db.py
=========================
Is a DB from `generate_dataset_adam_svgd.py` fit for training
`flow_matching_runner_start.py --target_cps --start_inpaint`?

Checks, each printed as PASS / WARN / FAIL (exit code 1 if any FAIL, so a
dataset job can stop an `afterok`-chained training):

  schema        extra columns present, one nxi, blob sizes, finite, in [0, 1]
  complete      every (source row x start) present exactly once, nothing extra
  splits        base/unk0 split = source split; extra starts of val -> val_start
  leakage       no train row shares its density with a val/val_flat/val_start
                row (compared by density content, not by name)
  density       all starts of a density carry the source's density_params
  start pin     control point 0 == stored x0 == first trajectory point
  starts        the starts of one density really differ; x0 coverage
  solver        converged fraction, iterations, iteration-cap hits, ergodic
                error init -> final, outliers
  recompute     ergodic error recomputed from the stored control points and
                the density grid == stored value (catches row/density mix-ups)
  geometry      path length, bending energy, curvature, control points on the
                clip bound
  start effect  Chamfer distance between the trajectories of different starts
                of one density (if ~0 the start would not matter)
  runner        `_load_shapes(target_cps=True)` returns the stored control
                points; one augmented batch through the CFM+ErgLoss loss with
                --start_inpaint is finite and keeps start == control point 0

`--sensitivity K` additionally re-solves K train densities to measure how
smooth (density, start) -> trajectory is: same x0 with another solver seed,
and x0 shifted by 0.01 / 0.03. Small changes should give similar curves
(learnable); compare to the distance between different starts.

Writes <out>/validation.json and <out>/validation_examples.png.

    python validate_adam_svgd_db.py --db ergodic_dataset_adam_svgd_nxi25.db \
        --out results_adam_svgd_validation --sensitivity 32
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import generate_dataset_adam_svgd as gen                           # noqa: E402
import torch                                                        # noqa: E402

RESULTS = []


def report(status, name, msg):
    RESULTS.append(dict(status=status, check=name, msg=msg))
    print(f"  [{status:4s}] {name:<13} {msg}", flush=True)


def check(cond, name, msg, warn_only=False):
    report('PASS' if cond else ('WARN' if warn_only else 'FAIL'), name, msg)


def dkey(dp):
    """Density identity by content (canonical JSON), independent of the name."""
    return hashlib.sha1(json.dumps(json.loads(dp), sort_keys=True).encode()).hexdigest()


def render(cps, n=256):
    from obstacles import bspline_basis_matrix
    return bspline_basis_matrix(cps.shape[0], n, gen.DEGREE).astype(np.float64) @ cps


def chamfer(a, b):
    """Footprint distance: where the curves are, regardless of the order."""
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def route(a, b):
    """Route distance: mean pointwise distance at equal curve parameter -- what
    the CFM regresses (control points in order), unlike `chamfer`."""
    return float(np.linalg.norm(a - b, axis=-1).mean())


def load_db(path):
    conn = sqlite3.connect(path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(ergodic_pairs)")]
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM ergodic_pairs ORDER BY id").fetchall()
    conn.close()
    return cols, [dict(zip(cols, r)) for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', required=True)
    p.add_argument('--src_db', default=gen.SRC_DB)
    p.add_argument('--variants', default='base,unk0')
    p.add_argument('--n_starts', type=int, default=None,
                   help='Expected starts per density (default: from solver_config).')
    p.add_argument('--subset', action='store_true',
                   help='DB was built for a subset of the source (--names/--limit): '
                        'only check that present rows are correct, not that all exist.')
    p.add_argument('--out', default='results_adam_svgd_validation')
    p.add_argument('--sensitivity', type=int, default=0)
    p.add_argument('--n_examples', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cols, rows = load_db(a.db)
    print(f"  DB: {a.db}  ({len(rows)} rows)")

    # ── schema ───────────────────────────────────────────────────────────────
    missing = [c for c, _ in gen.EXTRA_COLUMNS if c not in cols]
    check(not missing, 'schema', f"extra columns missing: {missing}" if missing
          else "all extra columns present")
    if missing:
        return finish(a)
    nxis = {r['nxi'] for r in rows}
    check(len(nxis) == 1, 'schema', f"nxi values: {sorted(nxis)}")
    nxi = next(iter(nxis))
    cfg = json.loads(rows[0]['solver_config'])
    n_starts = a.n_starts or cfg.get('n_starts', 1)
    bad_size = [r['shape_name'] for r in rows
                if len(r['control_points']) != nxi * 2 * 4
                or len(r['trajectory']) != (gen.TSTEPS_STORE + 1) * 2 * 4]
    check(not bad_size, 'schema', f"blob sizes ok ({nxi}x2 cps, {gen.TSTEPS_STORE + 1}x2 traj)"
          if not bad_size else f"wrong blob sizes: {bad_size[:5]}")
    for r in rows:
        r['cps'] = np.frombuffer(r['control_points'], dtype=np.float32).reshape(nxi, 2).astype(np.float64)
        r['traj'] = np.frombuffer(r['trajectory'], dtype=np.float32).reshape(-1, 2)
        r['x0v'] = np.asarray(json.loads(r['x0']), dtype=np.float64)
    finite = all(np.isfinite(r['cps']).all() and np.isfinite(r['traj']).all()
                 and np.isfinite(r['ergodic_error']) for r in rows)
    check(finite, 'schema', "all control points / trajectories / errors finite")
    lo = min(r['cps'].min() for r in rows)
    hi = max(r['cps'].max() for r in rows)
    check(lo >= 0.0 and hi <= 1.0, 'schema', f"control points in [{lo:.3f}, {hi:.3f}]")

    # ── completeness / splits / density consistency ──────────────────────────
    src = gen.load_source_rows(a.src_db, tuple(a.variants.split(',')))
    expected = {r['name']: r for r in gen.expand_starts(src, n_starts, cfg.get('seed', 20261024))}
    names = [r['shape_name'] for r in rows]
    dup = len(names) - len(set(names))
    extra = [n for n in names if n not in expected]
    absent = [n for n in expected if n not in set(names)]
    check(dup == 0 and not extra, 'complete', f"{dup} duplicate names, {len(extra)} unexpected names"
          + (f" e.g. {extra[:3]}" if extra else ""))
    if not a.subset:
        check(not absent, 'complete', f"{len(expected) - len(absent)}/{len(expected)} expected "
              f"trajectories present" + (f", missing e.g. {absent[:3]}" if absent else ""))
    wrong_split = [r['shape_name'] for r in rows if r['shape_name'] in expected
                   and r['split'] != expected[r['shape_name']]['split']]
    check(not wrong_split, 'splits', "splits as expected (extra starts of val -> val_start)"
          if not wrong_split else f"wrong split: {wrong_split[:5]}")
    from collections import Counter
    print(f"           split counts: {dict(Counter(r['split'] for r in rows))}")
    wrong_dens = [r['shape_name'] for r in rows if r['shape_name'] in expected
                  and r['density_params'] != expected[r['shape_name']]['density_params']]
    check(not wrong_dens, 'density', "every row carries its source density_params"
          if not wrong_dens else f"density mismatch: {wrong_dens[:5]}")
    x0_wrong = [r['shape_name'] for r in rows if r['shape_name'] in expected
                and np.abs(r['x0v'] - expected[r['shape_name']]['x0']).max() > 1e-9]
    check(not x0_wrong, 'density', "x0 equals the expected start (source x0 or seeded draw)"
          if not x0_wrong else f"x0 mismatch: {x0_wrong[:5]}")

    # leakage by density content
    by_key = defaultdict(set)
    for r in rows:
        r['dkey'] = dkey(r['density_params'])
        by_key[r['dkey']].add(r['split'])
    leak = [k for k, s in by_key.items() if 'train' in s and len(s) > 1]
    names_leak = sorted({r['shape_name'] for r in rows if r['dkey'] in set(leak)})
    check(not leak, 'leakage', f"{len(leak)} densities appear in train AND a holdout split"
          + (f": {names_leak[:8]}" if leak else ""), warn_only=True)
    same_dens_names = defaultdict(set)
    for r in rows:
        same_dens_names[r['dkey']].add(r['shape_name'].split('@')[0])
    clones = [sorted(v) for v in same_dens_names.values() if len(v) > 1]
    check(not clones, 'leakage', f"{len(clones)} groups of differently named shapes share one "
          f"density" + (f", e.g. {clones[:3]}" if clones else ""), warn_only=True)

    # ── start pin ────────────────────────────────────────────────────────────
    pin = max(np.abs(r['cps'][0] - r['x0v']).max() for r in rows)
    pin_t = max(np.abs(r['traj'][0] - r['x0v']).max() for r in rows)
    check(pin < 1e-5 and pin_t < 1e-5, 'start pin',
          f"max |cp0 - x0| = {pin:.1e}, max |traj[0] - x0| = {pin_t:.1e}")

    # ── starts ───────────────────────────────────────────────────────────────
    groups = defaultdict(list)
    for r in rows:
        groups[r['shape_name'].split('@')[0]].append(r)
    dmin = [min(np.linalg.norm(g[i]['x0v'] - g[j]['x0v']) for i in range(len(g))
                for j in range(i + 1, len(g))) for g in groups.values() if len(g) > 1]
    if dmin:
        check(np.percentile(dmin, 5) > 0.02, 'starts',
              f"min start distance within a density: p5 {np.percentile(dmin, 5):.3f}, "
              f"median {np.median(dmin):.3f}", warn_only=True)
    X0 = np.stack([r['x0v'] for r in rows])
    H, _, _ = np.histogram2d(X0[:, 0], X0[:, 1], bins=4, range=[[0, 1], [0, 1]])
    check(H.min() > 0.4 * H.mean(), 'starts', f"x0 4x4 histogram min/mean/max "
          f"{H.min():.0f}/{H.mean():.0f}/{H.max():.0f}", warn_only=True)

    # ── solver quality ───────────────────────────────────────────────────────
    conv = np.array([r['converged'] for r in rows], dtype=bool)
    its = np.array([r['n_iters'] for r in rows])
    erg = np.array([r['ergodic_error'] for r in rows])
    e0 = np.array([r['init_ergodic_error'] for r in rows])
    cap = int((its >= cfg['max_iters']).sum())
    check(conv.mean() >= 0.9, 'solver', f"converged {conv.mean():.1%}, iteration cap hit "
          f"{cap} ({cap / len(rows):.1%}), iterations median {np.median(its):.0f} "
          f"p95 {np.percentile(its, 95):.0f}", warn_only=True)
    ratio = erg / np.maximum(e0, 1e-12)
    check(np.percentile(ratio, 99) < 0.2, 'solver',
          f"ergodic error init -> final: median {np.median(e0):.4f} -> {np.median(erg):.2e} "
          f"(ratio p50 {np.median(ratio):.3f}, p99 {np.percentile(ratio, 99):.3f})")
    for var in ('base', 'unk0'):
        for k in range(n_starts):
            sel = [r['ergodic_error'] for r in rows
                   if (('#unk0' in r['shape_name']) == (var == 'unk0'))
                   and (('@s' in r['shape_name']) == (k > 0))
                   and (k == 0 or r['shape_name'].endswith(f'@s{k}'))]
            if sel:
                print(f"           erg {var:4s} start {k}: n={len(sel):5d} median {np.median(sel):.2e}"
                      f"  p95 {np.percentile(sel, 95):.2e}")
    thr = 5 * np.median(erg)
    out_names = [rows[i]['shape_name'] for i in np.argsort(-erg)[:5]]
    check((erg > thr).mean() < 0.02, 'solver', f"{(erg > thr).sum()} rows with error > 5x median "
          f"({thr:.2e}); worst: {out_names}", warn_only=True)

    # ── recompute ergodic error from stored control points ───────────────────
    from svgd_batched import K_FREQ
    dens = {}
    for r in rows:
        dens.setdefault(r['dkey'], r['density_params'])
    keys = list(dens)
    grid_res = cfg.get('grid_res', 128)
    print(f"           recomputing {len(keys)} density grids ...", flush=True)
    grids = np.stack([gen.density_grid(dens[k], grid_res) for k in keys])
    phik_d = gen.target_coeffs(grids, K_FREQ)
    kpos = {k: i for i, k in enumerate(keys)}
    bs, B32 = gen.make_solver(nxi, 'cpu', 'float64')
    cps_all = torch.as_tensor(np.stack([r['cps'] for r in rows]))
    X = torch.matmul(torch.as_tensor(B32, dtype=torch.float64), cps_all)
    e_re = gen.ergodic_error(bs, X, torch.as_tensor(phik_d[[kpos[r['dkey']] for r in rows]])).numpy()
    rel = np.abs(e_re - erg) / np.maximum(erg, 1e-12)
    # float32 storage of the control points moves the error by ~1e-6 relative
    check(np.percentile(rel, 99) < 0.02, 'recompute',
          f"stored vs recomputed ergodic error: rel. diff median {np.median(rel):.1e}, "
          f"max {rel.max():.1e}")

    # ── projection (waypoint solve space) ────────────────────────────────────
    if cfg.get('solve_space') == 'waypoints':
        fit = np.array([r['fit_rmse'] for r in rows])
        raw = np.array([r['raw_ergodic_error'] for r in rows])
        has_raw = all(r['raw_waypoints'] is not None for r in rows)
        check(has_raw and np.percentile(fit, 99) < 0.01, 'projection',
              f"B-spline projection vs solved waypoints: mean distance median {np.median(fit):.1e}, "
              f"p99 {np.percentile(fit, 99):.1e}; raw waypoints stored: {has_raw}")
        ratio_p = erg / np.maximum(raw, 1e-12)
        check(np.median(ratio_p) < 1.3, 'projection',
              f"ergodic error projected / raw: median {np.median(ratio_p):.2f}, "
              f"p95 {np.percentile(ratio_p, 95):.2f}", warn_only=True)

    # ── teacher-study selection (generate_dataset_lm.py) ─────────────────────
    if cfg.get('solver_name') == 'lm_teacher':
        fit = np.array([r['fit_rmse'] for r in rows])
        fb = np.array([r['selection_fallback'] for r in rows])
        hel = np.array([r['hellinger'] for r in rows])
        smo = np.array([r['smooth'] for r in rows])
        stg = Counter(r['selected_stage'] for r in rows)
        print(f"           selected stages: {dict(stg)}; Hellinger median {np.median(hel):.3f}, "
              f"smooth median {np.median(smo):.3f}; 25-CP curve vs LM waypoints: mean distance "
              f"median {np.median(fit):.1e}, p99 {np.percentile(fit, 99):.1e}")
        check(all(r['raw_waypoints'] is not None for r in rows), 'selection',
              "raw LM waypoints of the chosen candidate stored for every row")
        check(fb.mean() < 0.05, 'selection', f"boundary-filter fallback in {int(fb.sum())} rows "
              f"({fb.mean():.1%}; no candidate fully inside [0.03, 0.97])", warn_only=True)

    # ── geometry ─────────────────────────────────────────────────────────────
    met = [gen.curve_metrics(r['cps']) for r in rows]
    for r, m in zip(rows, met):
        r.update(m)
    L = np.array([m['path_len'] for m in met])
    bend = np.array([m['bending'] for m in met])
    kp = np.array([m['kappa_p95'] for m in met])
    on_clip = np.mean([((r['cps'][1:] <= 0.0201) | (r['cps'][1:] >= 0.9799)).any(axis=1).mean()
                       for r in rows])
    print(f"           path length median {np.median(L):.2f} [p5 {np.percentile(L, 5):.2f}, "
          f"p95 {np.percentile(L, 95):.2f}], bending median {np.median(bend):.0f} "
          f"p95 {np.percentile(bend, 95):.0f}, kappa_p95 median {np.median(kp):.1f}")
    check(on_clip < 0.15, 'geometry', f"control points on the clip bound: {on_clip:.1%}",
          warn_only=True)
    check(np.percentile(L, 1) > 0.5, 'geometry', f"no degenerate (too short) trajectories: "
          f"p1 path length {np.percentile(L, 1):.2f}")

    # ── start effect ─────────────────────────────────────────────────────────
    cd_starts, rt_starts = [], []
    for g in groups.values():
        if len(g) < 2:
            continue
        cur = [render(r['cps']) for r in g]
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                cd_starts.append(chamfer(cur[i], cur[j]))
                rt_starts.append(route(cur[i], cur[j]))
    cd_starts, rt_starts = np.array(cd_starts), np.array(rt_starts)
    # reference: trajectories of two different densities
    rng = np.random.default_rng(a.seed)
    cd_dens, rt_dens = [], []
    for _ in range(300):
        i, j = rng.choice(len(rows), 2, replace=False)
        if rows[i]['dkey'] != rows[j]['dkey']:
            ci, cj = render(rows[i]['cps']), render(rows[j]['cps'])
            cd_dens.append(chamfer(ci, cj))
            rt_dens.append(route(ci, cj))
    ref = dict(chamfer_starts=float(np.median(cd_starts)) if len(cd_starts) else None,
               route_starts=float(np.median(rt_starts)) if len(rt_starts) else None,
               chamfer_densities=float(np.median(cd_dens)), route_densities=float(np.median(rt_dens)))
    if len(cd_starts):
        check(np.median(cd_starts) > 0.01, 'start effect',
              f"different starts of one density: Chamfer median {ref['chamfer_starts']:.4f}, "
              f"route median {ref['route_starts']:.4f}  (different densities: Chamfer "
              f"{ref['chamfer_densities']:.4f}, route {ref['route_densities']:.4f})")

    # ── runner compatibility ─────────────────────────────────────────────────
    runner_check(a, rows, nxi)

    # ── sensitivity ──────────────────────────────────────────────────────────
    sens = None
    if a.sensitivity > 0:
        sens = sensitivity(a, rows, nxi, cfg, phik_d, kpos, ref)

    plot_examples(a, rows, groups, grids, kpos, nxi)
    with open(os.path.join(a.out, 'validation.json'), 'w') as f:
        json.dump(dict(db=a.db, results=RESULTS, sensitivity=sens,
                       stats=dict(n=len(rows), converged=float(conv.mean()),
                                  iters_median=float(np.median(its)), erg_median=float(np.median(erg)),
                                  path_len_median=float(np.median(L)),
                                  distances=ref)),
                  f, indent=1)
    return finish(a)


def runner_check(a, rows, nxi):
    sys.path.insert(0, os.path.dirname(_here))
    import flow_matching_runner_start as runner
    from flow_matching_cond_particles_start import (ParticleCrossAttnFlowNetwork,
                                                    compute_particle_cfm_loss)
    runner._DB_PATH = os.path.abspath(a.db)
    tr, va, trd, vad = runner._load_shapes(nxi, grid_res=64, target_cps=True)
    by_name = {r['shape_name']: r for r in rows}
    n_train = sum(r['split'] == 'train' for r in rows)
    n_val = sum(r['split'] == 'val' for r in rows)
    check(len(tr) == n_train and len(va) == n_val, 'runner',
          f"_load_shapes: {len(tr)} train / {len(va)} val (DB: {n_train} / {n_val}; "
          f"val_start/val_flat excluded as intended)")
    diff = max(np.abs(np.asarray(v, dtype=np.float64) - by_name[k]['cps']).max()
               for k, v in list(tr.items())[:500])
    check(diff < 1e-6, 'runner', f"loaded targets == stored control points (max diff {diff:.1e})")

    torch.manual_seed(a.seed)
    keys = list(tr)[:16]
    x1 = torch.tensor(np.stack([tr[k] for k in keys]), dtype=torch.float32)
    grids = torch.stack([torch.tensor(trd[k], dtype=torch.float32) for k in keys])
    parts = runner.sample_particles(grids, torch.arange(len(keys)), 64, 'cpu', mode='uniform')
    xa, pa = runner.augment_batch_torch(x1, parts, p_flip=0.0)
    model = ParticleCrossAttnFlowNetwork(nxi=nxi, D=32, start_inpaint=True)
    from ergodic_metric import ErgodicLoss
    erg = ErgodicLoss(nxi=nxi, K=8, pts=128, deg=gen.DEGREE, weight=1300.0, t_power=2.0,
                      weighted_target=True, metric='fourier')
    loss, parts_ = compute_particle_cfm_loss(model, xa, pa, p_drop=0.1, ergodic=erg)
    loss.backward()
    gfin = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    check(bool(torch.isfinite(loss)) and gfin, 'runner',
          f"augmented batch -> CFM+ErgLoss (inpaint) loss {float(loss):.3f}, gradients finite")


def sensitivity_lm(a, pick, cfg, ref, rng):
    """`sensitivity` for a `generate_dataset_lm.py` DB: re-run the same teacher pipeline + selection.
    LM is deterministic given the chord inits, so 'same' (unchanged x0) checks reproducibility."""
    import generate_dataset_lm as glm
    tcfg = glm.teacher_cfg()
    k8 = glm.K8Scorer(cfg['nxi'])
    out = {}
    for tag, shift in (('same', 0.0), ('dx0.01', 0.01), ('dx0.03', 0.03)):
        cd, rt, de = [], [], []
        for r in pick:
            d = rng.normal(size=2)
            d /= np.linalg.norm(d)
            x0 = np.clip(r['x0v'] + shift * d, gen.X0_MARGIN, 1 - gen.X0_MARGIN)
            sd = glm.shape_data(r['shape_name'], r['density_params'], x0)
            s = glm.solve_one(tcfg, sd, cfg['n_init'], a.device, cfg['eps'])
            x, y = render(r['cps']), render(s['cps'])
            cd.append(chamfer(x, y))
            rt.append(route(x, y))
            de.append(k8.cps(s['cps'], r['density_params']) / max(r['ergodic_error'], 1e-12))
        out[tag] = dict(chamfer_median=float(np.median(cd)), route_median=float(np.median(rt)),
                        route_p90=float(np.percentile(rt, 90)), erg_ratio_median=float(np.median(de)))
        print(f"           sensitivity {tag:7s}: to stored trajectory Chamfer {np.median(cd):.4f}, "
              f"route {np.median(rt):.4f} (p90 {np.percentile(rt, 90):.4f}); erg ratio "
              f"{np.median(de):.2f}", flush=True)
    check(out['same']['route_median'] < 0.01, 'sensitivity',
          f"re-running the pipeline on the stored start reproduces the trajectory "
          f"(route {out['same']['route_median']:.4f})", warn_only=True)
    rs = ref['route_starts'] or np.inf
    check(out['dx0.01']['route_median'] < 0.5 * rs, 'sensitivity',
          f"x0 + 0.01 changes the route by {out['dx0.01']['route_median']:.4f} vs "
          f"{rs:.4f} between different starts of a density", warn_only=True)
    return out


def sensitivity(a, rows, nxi, cfg, phik_d, kpos, ref):
    """Re-solve K train densities (start 0): other seed, x0 + 0.01, x0 + 0.03."""
    rng = np.random.default_rng(a.seed)
    cand = [r for r in rows if r['split'] == 'train' and '@s' not in r['shape_name']]
    pick = [cand[i] for i in rng.choice(len(cand), min(a.sensitivity, len(cand)), replace=False)]
    if cfg.get('solver_name') == 'lm_teacher':
        return sensitivity_lm(a, pick, cfg, ref, rng)
    bs, init_fn, finalize = gen.make_backend(cfg.get('solve_space', 'bspline'), nxi, a.device,
                                             cfg.get('compute_dtype', 'float64'),
                                             T=cfg.get('T') or gen.N_POINTS,
                                             w_smooth_scale=cfg.get('w_smooth_scale', 1.0))
    out = {}
    for tag, shift, seed_off in (('seed', 0.0, 1), ('dx0.01', 0.01, 0), ('dx0.03', 0.03, 0)):
        x0s = []
        for r in pick:
            d = rng.normal(size=2)
            d /= np.linalg.norm(d)
            x0s.append(np.clip(r['x0v'] + shift * d, gen.X0_MARGIN, 1 - gen.X0_MARGIN))
        x0s = np.stack(x0s)
        seeds = np.array([(zlib.crc32(f"{cfg.get('seed')}:{r['shape_name']}".encode()) + seed_off)
                          & 0x7FFFFFFF for r in pick])
        phik = phik_d[[kpos[r['dkey']] for r in pick]]
        res = gen.solve_pinned(bs, init_fn(x0s), phik, x0s, seeds,
                               max_iters=cfg['max_iters'], check_every=cfg['check_every'],
                               patience=cfg['patience'], tol=cfg['tol'], min_iters=cfg['min_iters'],
                               lr=cfg.get('lr'))
        res = [finalize(r, x0, ph) for r, x0, ph in zip(res, x0s, phik)]
        cur = [(render(r['cps']), render(s['cps'])) for r, s in zip(pick, res)]
        cd = np.array([chamfer(x, y) for x, y in cur])
        rt = np.array([route(x, y) for x, y in cur])
        de = np.array([s['erg'] / max(r['ergodic_error'], 1e-12) for r, s in zip(pick, res)])
        out[tag] = dict(chamfer_median=float(np.median(cd)), route_median=float(np.median(rt)),
                        route_p90=float(np.percentile(rt, 90)), erg_ratio_median=float(np.median(de)))
        print(f"           sensitivity {tag:7s}: to stored trajectory Chamfer {np.median(cd):.4f}, "
              f"route {np.median(rt):.4f} (p90 {np.percentile(rt, 90):.4f}); erg ratio "
              f"{np.median(de):.2f}", flush=True)
    rs = ref['route_starts'] or np.inf
    check(out['dx0.01']['route_median'] < 0.5 * rs, 'sensitivity',
          f"x0 + 0.01 changes the route by {out['dx0.01']['route_median']:.4f} vs "
          f"{rs:.4f} between different starts of a density", warn_only=True)
    return out


def plot_examples(a, rows, groups, grids, kpos, nxi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from flow_matching_runner_start import WHITE_INFERNO
    rng = np.random.default_rng(a.seed)
    gnames = [g for g in groups if len(groups[g]) > 1] or list(groups)
    pick = [gnames[i] for i in rng.choice(len(gnames), min(a.n_examples, len(gnames)), replace=False)]
    n_cols = min(4, len(pick))
    n_rows = (len(pick) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.8 * n_rows),
                             facecolor='white', squeeze=False)
    styles = [dict(lw=2.5, alpha=0.9), dict(lw=1.6, alpha=0.6, ls='--'), dict(lw=1.6, alpha=0.6, ls=':')]
    for ax, gname in zip(axes.flat, pick):
        g = sorted(groups[gname], key=lambda r: r['shape_name'])
        d = grids[kpos[g[0]['dkey']]]
        ax.set_facecolor('white')
        ax.imshow(d / d.max(), extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO,
                  vmin=0, vmax=1, alpha=0.55, aspect='auto', zorder=0)
        for k, r in enumerate(g):
            X = render(r['cps'], 512)
            st = styles[min(k, len(styles) - 1)]
            ax.plot(X[:, 0], X[:, 1], color='#1565C0', zorder=2,
                    label=f"start {k}" if ax is axes.flat[0] else None, **st)
            ax.scatter([r['x0v'][0]], [r['x0v'][1]], color='#1565C0', marker='*', s=110,
                       edgecolors='white', linewidths=0.6, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#ccc')
        errs = ', '.join(f"{r['ergodic_error']:.1e}" for r in g)
        ax.set_title(f"{gname}\nE = {errs}", fontsize=7, color='#1A1A2E')
    for ax in list(axes.flat)[len(pick):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=7, frameon=True, facecolor='white', edgecolor='#ddd')
    fig.suptitle('Adam-SVGD training DB: ground truth for each start point of a density',
                 fontsize=11, color='#1A1A2E', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, 'validation_examples.png'), dpi=130, bbox_inches='tight',
                facecolor='white')
    plt.close()


def finish(a):
    n_fail = sum(r['status'] == 'FAIL' for r in RESULTS)
    n_warn = sum(r['status'] == 'WARN' for r in RESULTS)
    print(f"\n  {n_fail} FAIL, {n_warn} WARN, {len(RESULTS) - n_fail - n_warn} PASS"
          f"  -> {'NOT fit for training' if n_fail else 'fit for training'}")
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
