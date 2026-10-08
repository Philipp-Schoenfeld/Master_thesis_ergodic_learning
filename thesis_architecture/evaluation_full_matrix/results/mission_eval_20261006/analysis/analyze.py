r"""
analyze.py -- summary numbers of the replanning-mission run mission_eval_20261006
================================================================================
Reads the shard DBs (scalar columns only, no SVGD state blobs) and writes
`analysis.json`, the data behind the results page:

  * per-unit curves (swept mass, E_truth) per method and per strategy; a
    finished mission keeps its last value so every mean is over all missions,
  * paired CFM-vs-baseline comparisons per (knowledge, strategy, shape) at
    fixed numbers of executed units (win/tie/loss counts + Wilcoxon p),
  * executions until 99 % swept mass (unreached = 41, i.e. worse than the cap),
  * candidate level: ergodic error of the raw initialisations vs. after SVGD,
    and the median E along the SVGD iterations, per method.

Usage:  python analyze.py   (from this folder or anywhere)
"""
import glob
import json
import os
import sqlite3

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
METHODS = ('cfm', 'linear', 'random_walk')
STRATS = ('eid', 'ucb', 'lse')
CONDS = ('half_known', 'ten_samples', 'none_known')
N_MAX = 40
UNREACHED = 41
CHECK_UNITS = (1, 3, 5, 10, 15)
THRESHOLDS = (90, 95)
EXAMPLES = dict(cond='none_known', strategy='eid',
                shapes=('A', 'organic_10', 'cjk_0', 'rand_gmm_complex_10'))


def load_rounds():
    rows = []
    for f in sorted(glob.glob(os.path.join(RUN, 'shards', '*.db'))):
        c = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
        rows += c.execute(
            "SELECT knowledge_condition, strategy, method, shape, round, swept_mass, "
            "E_truth, reached99, E_cand_best, E_cand_median FROM rounds").fetchall()
        c.close()
    return rows


def mission_series(rows):
    """{(cond, strat, method, shape): dict(swept=(N_MAX,), E=(N_MAX,), exec99)}"""
    tmp = {}
    for cond, strat, meth, shape, r, sw, e, reached, *_ in rows:
        tmp.setdefault((cond, strat, meth, shape), []).append((r, sw, e, reached))
    out = {}
    for k, v in tmp.items():
        v.sort()
        sw = np.array([x[1] for x in v])
        e = np.array([x[2] for x in v])
        hit = [x[0] + 1 for x in v if x[3]]
        pad = N_MAX - len(v)
        out[k] = dict(swept=np.r_[sw, np.full(pad, sw[-1])],
                      E=np.r_[e, np.full(pad, e[-1])],
                      exec99=hit[0] if hit else UNREACHED)
        for thr in THRESHOLDS:
            idx = np.nonzero(sw >= thr / 100.0)[0]
            out[k][f'exec{thr}'] = int(idx[0]) + 1 if len(idx) else UNREACHED
    return out


def curves(ms, key_fn, groups, field):
    res = {}
    for g in groups:
        arr = np.stack([v[field] for k, v in ms.items() if key_fn(k) == g])
        res[g] = dict(mean=arr.mean(0).tolist(), q25=np.percentile(arr, 25, 0).tolist(),
                      q75=np.percentile(arr, 75, 0).tolist(), median=np.median(arr, 0).tolist(),
                      n=int(arr.shape[0]))
    return res


def paired(ms, base, field, unit=None, lower_better=True):
    a, b = [], []
    for (cond, strat, meth, shape), v in ms.items():
        if meth != 'cfm':
            continue
        w = ms.get((cond, strat, base, shape))
        if w is None:
            continue
        if field.startswith('exec'):
            a.append(v[field]); b.append(w[field])
        else:
            a.append(v[field][unit - 1]); b.append(w[field][unit - 1])
    a, b = np.array(a, float), np.array(b, float)
    d = (a - b) if lower_better else (b - a)          # < 0 => CFM better
    nz = d[d != 0]
    p = float(wilcoxon(a, b).pvalue) if len(nz) > 5 else float('nan')
    return dict(cfm_better=int((d < 0).sum()), tie=int((d == 0).sum()), cfm_worse=int((d > 0).sum()),
                median_cfm=float(np.median(a)), median_base=float(np.median(b)),
                mean_diff=float((a - b).mean()), p=p, n=int(len(a)))


def candidate_stats():
    """E_init / E_final per method over all candidates + median E_series."""
    res = {}
    for m in METHODS:
        e0, e1, best0, best1, series = [], [], [], [], []
        for f in sorted(glob.glob(os.path.join(RUN, 'shards', f'*__{m}.db'))):
            c = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
            for ei, ef in c.execute("SELECT E_init, E_final FROM candidates"):
                e0.append(ei); e1.append(ef)
            for b0, b1 in c.execute("SELECT MIN(E_init), MIN(E_final) FROM candidates "
                                    "GROUP BY shape, round"):
                best0.append(b0); best1.append(b1)
            # E_series of every 7th candidate is enough for the median curve
            for (blob,) in c.execute("SELECT E_series FROM candidates WHERE id % 7 = 0"):
                series.append(np.frombuffer(blob, dtype=np.float32))
            c.close()
        e0, e1 = np.array(e0), np.array(e1)
        s = np.stack(series)
        res[m] = dict(n_cands=int(len(e0)),
                      E_init_median=float(np.median(e0)), E_final_median=float(np.median(e1)),
                      E_init_q=[float(x) for x in np.percentile(e0, [10, 25, 50, 75, 90])],
                      E_final_q=[float(x) for x in np.percentile(e1, [10, 25, 50, 75, 90])],
                      best_init_median=float(np.median(best0)), best_final_median=float(np.median(best1)),
                      series_median=np.median(s, 0).tolist(),
                      series_q25=np.percentile(s, 25, 0).tolist(),
                      series_q75=np.percentile(s, 75, 0).tolist())
    return res


def examples():
    """Truth (downsampled to 32x32) and driven path of a few shapes, per method."""
    res = {}
    for m in METHODS:
        f = os.path.join(RUN, 'shards', f"{EXAMPLES['cond']}__{EXAMPLES['strategy']}__{m}.db")
        c = sqlite3.connect(f'file:{f}?mode=ro', uri=True)
        for shape in EXAMPLES['shapes']:
            r, blob = c.execute("SELECT res, density FROM truths WHERE shape=?", (shape,)).fetchone()
            d = np.frombuffer(blob, dtype=np.float32).reshape(r, r)
            if shape not in res:
                step = max(1, r // 32)
                res[shape] = dict(truth=np.round(d[::step, ::step] / max(d.max(), 1e-9), 3).tolist(),
                                  paths={})
            segs, units, reached = [], 0, 0
            for blob_s, rch in c.execute("SELECT segment, reached99 FROM rounds WHERE shape=? "
                                         "ORDER BY round", (shape,)):
                seg = np.frombuffer(blob_s, dtype=np.float32).reshape(-1, 2)
                segs.append(seg[::2])
                units += 1
                reached = rch
            pts = np.concatenate(segs)
            res[shape]['paths'][m] = dict(pts=np.round(pts, 4).tolist(), units=units,
                                          reached=int(reached))
        c.close()
    return res


def main():
    rows = load_rounds()
    ms = mission_series(rows)
    out = dict(n_missions=len(ms))
    out['curves_method'] = {f: curves(ms, lambda k: k[2], METHODS, f) for f in ('swept', 'E')}
    out['curves_strategy'] = {f: curves(ms, lambda k: k[1], STRATS, f) for f in ('swept', 'E')}
    out['curves_method_strategy'] = {
        s: {f: curves({k: v for k, v in ms.items() if k[1] == s}, lambda k: k[2], METHODS, f)
            for f in ('swept', 'E')} for s in STRATS}

    reach = {}
    for (cond, strat, meth, shape), v in ms.items():
        reach.setdefault(f'{cond}|{strat}|{meth}', []).append(v['exec99'])
    out['exec99'] = {k: dict(reached=int(sum(x < UNREACHED for x in v)), n=len(v),
                             median=float(np.median(v)), mean_capped=float(np.mean(v)),
                             values=sorted(v))
                     for k, v in reach.items()}

    cmp_ = {}
    for base in ('linear', 'random_walk'):
        c = {f: paired(ms, base, f) for f in ('exec99', 'exec95', 'exec90')}
        for u in CHECK_UNITS:
            c[f'E_u{u}'] = paired(ms, base, 'E', u)
            c[f'swept_u{u}'] = paired(ms, base, 'swept', u, lower_better=False)
        cmp_[base] = c
    out['paired'] = cmp_
    # paired per strategy at unit 3 and 5
    out['paired_strategy'] = {
        s: {base: {f'{fld}_u{u}': paired({k: v for k, v in ms.items() if k[1] == s}, base, fld, u,
                                         lower_better=(fld == 'E'))
                   for fld in ('E', 'swept') for u in (3, 5)}
            for base in ('linear', 'random_walk')}
        for s in STRATS}

    hard = {}
    for (cond, strat, meth, shape), v in ms.items():
        hard.setdefault(shape, [0, 0, []])
        hard[shape][1] += 1
        if v['exec99'] == UNREACHED:
            hard[shape][0] += 1
            hard[shape][2].append(float(v['swept'][-1]))
    out['unreached_by_shape'] = {k: dict(unreached=a, n=n, final_swept_median=float(np.median(f)) if f else None)
                                 for k, (a, n, f) in sorted(hard.items(), key=lambda x: -x[1][0])}
    out['exec_thresholds'] = {
        f: {m: dict(median=float(np.median([v[f] for k, v in ms.items() if k[2] == m])),
                    mean=float(np.mean([v[f] for k, v in ms.items() if k[2] == m])))
            for m in METHODS}
        for f in ('exec90', 'exec95', 'exec99')}
    out['exec_thresholds_strategy'] = {
        f: {s: float(np.median([v[f] for k, v in ms.items() if k[1] == s])) for s in STRATS}
        for f in ('exec90', 'exec95', 'exec99')}
    out['examples'] = examples()
    out['candidates'] = candidate_stats()
    with open(os.path.join(HERE, 'analysis.json'), 'w') as f:
        json.dump(out, f)
    print(json.dumps({k: out[k] for k in ('n_missions', 'paired')}, indent=1)[:6000])
    for m, v in out['candidates'].items():
        print(m, {k: v[k] for k in ('n_cands', 'E_init_median', 'E_final_median',
                                    'best_init_median', 'best_final_median')})


if __name__ == '__main__':
    main()
