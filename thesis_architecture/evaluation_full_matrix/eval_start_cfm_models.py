r"""
eval_start_cfm_models.py -- incremental, lossless comparison of start-conditioned CFM checkpoints
================================================================================================
Same test as `eval_start_cfm_model.py` (parts A/B/C on the holdout rows of the LM teacher DB), but
each model is evaluated ON ITS OWN and every raw number is stored, so models that finish training
later are added without re-running anything and without changing the cases they are compared on.

    run     evaluate one checkpoint -> results/<out_tag>/models/<label>/{meta.json, A.npz, B.npz, C.npz}
    report  read every stored model -> results/<out_tag>/{report.md, summary.json, *.png}

Fairness across runs:
  * The protocol (case list, sample counts, solver iterations, seeds) is written to `protocol.json`
    by the first run; a later run with different settings or a different DB aborts.
  * Every random draw (conditioning particles, flow noise) is seeded per CASE, not per model:
    all models see the same particles and the same initial noise (common random numbers).
  * No time budget: every model runs the full protocol, so all models cover the same cases.
  * Model-independent baselines (ground truth, chord inits, chords + LM) are computed once and stored
    under `baselines/`.
  * Each part is written atomically when it is done; a re-run of the same label skips finished parts
    (resume after preemption / time limit).

Stored per model (metric keys = all of `teacher_study.pipeline.stage_metrics`):
  A.npz  m__<key> (n_cases, n_samples), curves (n_cases, n_samples, 128, 2), t_sample (n_cases,) [s]
  B.npz  m__<key> (n_warm, n_iters, n_cand), curves (n_warm, n_iters, n_cand, 128, 2)
  C.npz  m__<key> (n_grid_cases, n_grid, n_grid_samples), starts (n_grid, 2), curves (...)

    python eval_start_cfm_models.py run --out_tag cfm_models --label unet-l630 --ckpt <x.pt> --db <lm.db>
    python eval_start_cfm_models.py report --out_tag cfm_models --ref unet-l1300-600ep
"""
import argparse
import glob
import json
import os
import socket
import time

import numpy as np
import torch

from eval_start_cfm_model import RESULTS, C_GT, C_CHORD, load_cases, metrics_row, seed_of, _style, pl

PROTOCOL_KEYS = ('db', 'n_samples', 'steps', 'n_cand', 'lm_iters', 'n_grid_cases', 'grid_side',
                 'n_grid_samples', 'seed')
MODEL_COLORS = ['#00C853', '#EF6C00', '#8E24AA', '#00ACC1', '#C62828', '#6D4C41', '#F9A825', '#3949AB']


# ---------------------------------------------------------------------------------------------
# storage helpers
# ---------------------------------------------------------------------------------------------

def _atomic_savez(path, **arrays):
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _stack_metrics(per_item):
    """list of stage_metrics dicts (each key -> (n,)) -> {m__key: (len(list), n)}"""
    return {f'm__{k}': np.stack([d[k] for d in per_item]) for k in per_item[0]}


def _load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _metrics(npz):
    return {k[3:]: v for k, v in npz.items() if k.startswith('m__')}


# ---------------------------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------------------------

def _protocol(a, cases):
    val = [c['name'] for c in cases if c['split'] == 'val']
    return dict(
        db=os.path.basename(a.db), n_samples=a.n_samples, steps=a.steps, n_cand=a.n_cand,
        lm_iters=list(a.lm_iters), n_grid_cases=a.n_grid_cases, grid_side=a.grid_side,
        n_grid_samples=a.n_grid_samples, seed=a.seed,
        cases=[dict(name=c['name'], density=c['density'], split=c['split'], kind=c['kind'],
                    start=('stored' if c['start_idx'] == 0 else 'new')) for c in cases],
        warm_cases=val, grid_cases=val[:a.n_grid_cases])


def _check_protocol(out, proto):
    path = os.path.join(out, 'protocol.json')
    if not os.path.exists(path):
        _atomic_json(path, proto)
        print(f"[eval] protocol written: {len(proto['cases'])} cases, {len(proto['warm_cases'])} warm-start "
              f"densities, {len(proto['grid_cases'])} grid densities", flush=True)
        return
    old = json.load(open(path))
    diff = [k for k in PROTOCOL_KEYS + ('cases', 'warm_cases', 'grid_cases') if old.get(k) != proto.get(k)]
    if diff:
        raise SystemExit(f"[eval] protocol mismatch in {diff}: this run would not be comparable to the "
                         f"models already stored in {out}")


def _sync(dev):
    if str(dev).startswith('cuda'):
        torch.cuda.synchronize()


def _sample(planner, case, n, dev, seed, start=None):
    """n CFM samples conditioned on the case's start -> ((n, 128, 2) float64 curves, seconds for plan())."""
    import apply_cfm_belief as acb
    torch.manual_seed(seed)
    parts = acb.phi_particles(case['truth'].to(dev), 256, mode='uniform', device=dev)
    x0 = case['sd'].x0 if start is None else start
    _sync(dev)
    t0 = time.perf_counter()
    cps = planner.plan(parts, n_candidates=n, start=torch.as_tensor(x0, dtype=torch.float32))
    _sync(dev)
    dt = time.perf_counter() - t0
    return planner.render(cps).detach().cpu().numpy().astype(np.float64), dt


def _lm_trajectory(c, X0, lm_iters, dev):
    """Levenberg-Marquardt (TSVEC energy, K=16, w=15, start pinned) from X0 -> per-iteration metrics + curves."""
    from common.tsvec_svgd import TsvecSvgd
    sd = c['sd']
    energy = TsvecSvgd(B=None, K=16, device=dev, w_smooth=15.0, beta=1.0, smooth_order=2)
    phi_k = pl._phi_k(sd.density, energy.k_idx, dev)
    X0 = np.clip(X0, 0.02, 0.98)
    X0[:, 0] = sd.x0
    P, done, ms, curves = X0.astype(np.float32), 0, [], []
    for it in lm_iters:
        if it > done:
            P = pl._levenberg_marquardt(energy, P, phi_k, it - done, start=sd.x0)
            P = P.detach().cpu().numpy() if torch.is_tensor(P) else P
            done = it
        X = np.asarray(P, dtype=np.float64)
        ms.append(metrics_row(X, c))
        curves.append(X.astype(np.float32))
    return ms, np.stack(curves)


def _part_B(c_list, inits_fn, a, dev):
    """inits_fn(case) -> (n_cand, 128, 2) initial curves; -> B.npz arrays."""
    per_case, curves = [], []
    t0 = time.time()
    for i, c in enumerate(c_list):
        ms, X = _lm_trajectory(c, inits_fn(c), a.lm_iters, dev)
        per_case.append(_stack_metrics(ms))
        curves.append(X)
        if (i + 1) % 8 == 0:
            print(f"[eval]   B {i + 1}/{len(c_list)}  {(time.time() - t0) / (i + 1):.1f} s/density", flush=True)
    keys = per_case[0].keys()
    return dict({k: np.stack([p[k] for p in per_case]) for k in keys}, curves=np.stack(curves))


# ---------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------

def run(a):
    out = os.path.join(RESULTS, a.out_tag)
    mdir = os.path.join(out, 'models', a.label)
    bdir = os.path.join(out, 'baselines')
    dev = a.device
    t_start = time.time()
    elapsed = lambda: (time.time() - t_start) / 60.0

    cases = load_cases(a.db, None, a.seed)
    proto = _protocol(a, cases)
    os.makedirs(out, exist_ok=True)
    _check_protocol(out, proto)
    os.makedirs(mdir, exist_ok=True)
    os.makedirs(bdir, exist_ok=True)
    by_name = {c['name']: c for c in cases}
    warm = [by_name[n] for n in proto['warm_cases']]
    grid = [by_name[n] for n in proto['grid_cases']]

    # ---------------------------------------------------------- baselines (model-independent, once)
    p = os.path.join(bdir, 'A_ground_truth.npz')
    if not os.path.exists(p):
        X = [pl.render(c['gt_cps'][None], pl.N_POINTS, pl.DEGREE) for c in cases]
        _atomic_savez(p, **_stack_metrics([metrics_row(x, c) for x, c in zip(X, cases)]),
                      curves=np.stack(X).astype(np.float32))
    p = os.path.join(bdir, 'A_chord.npz')
    if not os.path.exists(p):
        X = [pl.chord_inits(c['sd'].x0, a.n_samples) for c in cases]
        _atomic_savez(p, **_stack_metrics([metrics_row(x, c) for x, c in zip(X, cases)]),
                      curves=np.stack(X).astype(np.float32))
    p = os.path.join(bdir, 'B_chord.npz')
    if not os.path.exists(p):
        print("[eval] baseline: chords + LM", flush=True)
        _atomic_savez(p, **_part_B(warm, lambda c: pl.chord_inits(c['sd'].x0, a.n_cand), a, dev))
    print(f"[eval] baselines ready  {elapsed():.1f} min", flush=True)

    # ---------------------------------------------------------- model
    from apply_cfm_belief import CfmPlanner
    planner = CfmPlanner(ckpt=a.ckpt, steps=a.steps, device=dev)
    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    objective = ck.get('objective', 'cfm')
    backbone = ck.get('backbone', 'unet')
    if objective == 'diffusion':
        backbone = f'{backbone}-diff'      # own family in tables and plots; sampled by the DDPM sampler (eta from the model)
    meta = dict(label=a.label, ckpt=os.path.abspath(a.ckpt), backbone=backbone, objective=objective,
                diff_pred=ck.get('diff_pred'), diff_eta=ck.get('diff_eta'),
                lambda_erg=float(ck.get('lambda_erg', 0.0)), epoch=ck.get('epoch'),
                n_params=int(sum(t.numel() for t in planner.model.parameters())),
                start_inpaint=bool(getattr(planner.model, 'start_inpaint', False)),
                dit_depth=ck.get('dit_depth'), dit_heads=ck.get('dit_heads'), D=ck.get('D'),
                host=socket.gethostname(),
                gpu=torch.cuda.get_device_name(0) if str(dev).startswith('cuda') else 'cpu')
    del ck
    _atomic_json(os.path.join(mdir, 'meta.json'), meta)
    print(f"[eval] model {a.label}: {meta['backbone']}, lambda={meta['lambda_erg']}, epoch={meta['epoch']}, "
          f"{meta['n_params'] / 1e6:.1f}M params, {meta['gpu']}", flush=True)

    # A: generation quality, every case, n_samples each
    p = os.path.join(mdir, 'A.npz')
    if not os.path.exists(p):
        ms, X, ts = [], [], []
        t0 = time.time()
        for i, c in enumerate(cases):
            x, dt = _sample(planner, c, a.n_samples, dev, seed_of(a.seed, c['name'], 'A'))
            ms.append(metrics_row(x, c))
            X.append(x.astype(np.float32))
            ts.append(dt)
            if (i + 1) % 36 == 0:
                print(f"[eval]   A {i + 1}/{len(cases)}  {(time.time() - t0) / (i + 1):.1f} s/case", flush=True)
        _atomic_savez(p, **_stack_metrics(ms), curves=np.stack(X), t_sample=np.array(ts))
    print(f"[eval] A done  {elapsed():.1f} min", flush=True)

    # C: start conditioning on a grid of start points
    p = os.path.join(mdir, 'C.npz')
    if not os.path.exists(p):
        g = np.linspace(0.05, 0.95, a.grid_side)
        starts = np.array([[gx, gy] for gx in g for gy in g])
        per_case, X = [], []
        for c in grid:
            ms, xs = [], []
            for s in starts:
                x, _ = _sample(planner, c, a.n_grid_samples, dev, seed_of(a.seed, 'grid', c['name'], *s), start=s)
                ms.append(metrics_row(x, c, x0=s))
                xs.append(x.astype(np.float32))
            per_case.append(_stack_metrics(ms))
            X.append(np.stack(xs))
        keys = per_case[0].keys()
        _atomic_savez(p, **{k: np.stack([q[k] for q in per_case]) for k in keys}, starts=starts,
                      curves=np.stack(X))
    print(f"[eval] C done  {elapsed():.1f} min", flush=True)

    # B: warm start of the teacher solver from n_cand model samples
    p = os.path.join(mdir, 'B.npz')
    if not os.path.exists(p):
        _atomic_savez(p, **_part_B(
            warm, lambda c: _sample(planner, c, a.n_cand, dev, seed_of(a.seed, c['name'], 'B'))[0], a, dev))
    print(f"[eval] B done -> {mdir}  {elapsed():.1f} min", flush=True)


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------

def _boot_ci(per_case, groups, n_boot=2000, seed=0):
    """Mean of per_case values with a 95% cluster-bootstrap CI (resampling densities, not cases)."""
    uniq = sorted(set(groups))
    idx = {g: np.flatnonzero(np.array(groups) == g) for g in uniq}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        means.append(np.mean(np.concatenate([per_case[idx[uniq[j]]] for j in pick])))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(per_case)), float(lo), float(hi)


def _load_models(out):
    models = {}
    for d in sorted(glob.glob(os.path.join(out, 'models', '*'))):
        if not os.path.exists(os.path.join(d, 'meta.json')):
            continue
        m = dict(meta=json.load(open(os.path.join(d, 'meta.json'))))
        for part in ('A', 'B', 'C'):
            p = os.path.join(d, f'{part}.npz')
            if os.path.exists(p):
                m[part] = _load_npz(p)
        models[m['meta']['label']] = m
    return dict(sorted(models.items(), key=lambda kv: (kv[1]['meta']['backbone'], kv[1]['meta']['lambda_erg'],
                                                       kv[1]['meta']['epoch'] or 0)))


def report(a):
    out = os.path.join(RESULTS, a.out_tag)
    proto = json.load(open(os.path.join(out, 'protocol.json')))
    models = _load_models(out)
    base = {k: _load_npz(os.path.join(out, 'baselines', f'{k}.npz'))
            for k in ('A_ground_truth', 'A_chord', 'B_chord') if os.path.exists(os.path.join(out, 'baselines', f'{k}.npz'))}
    cases = proto['cases']
    dens = [c['density'] for c in cases]
    lm_iters = proto['lm_iters']
    groups = {'all': np.ones(len(cases), bool)}
    for key, vals in (('split', ('val', 'val_start')), ('kind', ('base', 'unk0'))):   # split=val == stored start
        for v in vals:
            groups[f'{key}={v}'] = np.array([c[key] == v for c in cases])

    # A tables: method -> group -> metrics (mean over samples, then cases)
    methods = {'ground_truth': _metrics(base['A_ground_truth'])} if 'A_ground_truth' in base else {}
    methods.update({lab: _metrics(m['A']) for lab, m in models.items() if 'A' in m})
    if 'A_chord' in base:
        methods['chord'] = _metrics(base['A_chord'])
    keys = ('hellinger', 'ergodic_k10', 'coverage', 'smooth', 'jerk_rms', 'curv_p95', 'outside', 'start_err')
    tab = {}
    for meth, M in methods.items():
        for g, mask in groups.items():
            d = {k: float(M[k][mask].mean()) for k in keys if k in M}
            d['hellinger_best'] = float(M['hellinger'][mask].min(1).mean())
            d['n_cases'] = int(mask.sum())
            tab[f'{meth}|{g}'] = d
    # paired comparison against the reference model, CI over densities
    ref = a.ref if a.ref in models else next(iter(models), None)
    paired = {}
    for lab, m in models.items():
        if lab == ref or 'A' not in m or 'A' not in models[ref]:
            continue
        Mr, Mm = _metrics(models[ref]['A']), _metrics(m['A'])
        for k in ('hellinger', 'ergodic_k10', 'smooth', 'outside'):
            diff = Mm[k].mean(1) - Mr[k].mean(1)
            mean, lo, hi = _boot_ci(diff, dens)
            paired.setdefault(lab, {})[k] = dict(diff=mean, ci=[lo, hi], better=float(np.mean(diff < 0)))
    # absolute Hellinger per model with CI (for the lambda plot)
    hell_ci = {lab: _boot_ci(_metrics(m['A'])['hellinger'].mean(1), dens) for lab, m in models.items() if 'A' in m}
    smooth_ci = {lab: _boot_ci(_metrics(m['A'])['smooth'].mean(1), dens) for lab, m in models.items() if 'A' in m}

    # B: warm start
    warm = {}
    B_all = {lab: _metrics(m['B']) for lab, m in models.items() if 'B' in m}
    if 'B_chord' in base:
        B_all['chord'] = _metrics(base['B_chord'])
    for lab, M in B_all.items():
        warm[lab] = {str(it): {k: float(M[k][:, j].mean()) for k in ('hellinger', 'ergodic_k10', 'smooth', 'outside')}
                     for j, it in enumerate(lm_iters)}
    speed = {}
    if 'chord' in B_all and 100 in lm_iters:
        target = B_all['chord']['hellinger'][:, lm_iters.index(100)].mean(1)
        for lab, M in B_all.items():
            H = M['hellinger'].mean(2)                                    # (n_warm, n_iters)
            its = np.array([min([it for j, it in enumerate(lm_iters) if H[w, j] <= target[w]], default=np.inf)
                            for w in range(len(target))])
            speed[lab] = dict(median_iters=float(np.median(its)), share_reached=float(np.mean(np.isfinite(its))),
                              share_at_iter0=float(np.mean(its == 0)))

    # C: start conditioning
    startc = {}
    name_idx = {c['name']: i for i, c in enumerate(cases)}
    for lab, m in models.items():
        d = {}
        if 'A' in m:
            X = m['A']['curves']
            d['start_err_max'] = float(m['A']['m__start_err'].max())
            d['route_dist_within_case'] = float(np.mean([
                np.mean([np.linalg.norm(x[i] - x[j], axis=-1).mean() for i in range(len(x)) for j in range(i + 1, len(x))])
                for x in X]))
            by_d = {}
            for c in cases:
                by_d.setdefault(c['density'], []).append(X[name_idx[c['name']]][0])
            d['route_dist_between_starts'] = float(np.mean([
                np.linalg.norm(v[p] - v[q], axis=-1).mean() for v in by_d.values()
                for p in range(len(v)) for q in range(p + 1, len(v))]))
            d['t_sample_median_s'] = float(np.median(m['A']['t_sample']))
            d['gpu'] = m['meta']['gpu']
        if 'C' in m:
            d['grid_start_err_max'] = float(m['C']['m__start_err'].max())
            d['grid_outside_mean'] = float(m['C']['m__outside'].mean())
            d['grid_hellinger_mean'] = float(m['C']['m__hellinger'].mean())
        startc[lab] = d

    summary = dict(protocol={k: proto[k] for k in PROTOCOL_KEYS}, n_cases=len(cases), reference=ref,
                   models={lab: m['meta'] for lab, m in models.items()}, tables=tab, paired_vs_ref=paired,
                   hellinger_ci=hell_ci, smooth_ci=smooth_ci, warm_start=warm, warm_speed=speed,
                   start_conditioning=startc)
    _atomic_json(os.path.join(out, 'summary.json'), summary)
    _write_report(summary, out)
    _plot(summary, models, out)
    print(f"[report] {len(models)} models -> {out}", flush=True)


def _write_report(s, out):
    p = s['protocol']
    L = [f"# Start-conditioned CFM comparison `{os.path.basename(out)}`", '',
         f"{s['n_cases']} holdout cases x {p['n_samples']} samples, {p['steps']} Euler steps; warm start: "
         f"{p['n_cand']} initial curves per density; same particles and flow noise for every model.", '',
         '## Models', '', '| label | backbone | lambda_erg | epochs | params | eval GPU | t per plan() [s] |',
         '|---|---|---|---|---|---|---|']
    for lab, m in s['models'].items():
        L.append(f"| {lab} | {m['backbone']} | {m['lambda_erg']:g} | {m['epoch']} | {m['n_params'] / 1e6:.1f}M | "
                 f"{m['gpu']} | {s['start_conditioning'].get(lab, {}).get('t_sample_median_s', float('nan')):.3f} |")
    L += ['', 'Plan time = median wall-clock of one call generating all samples of a case; only comparable '
          'between models evaluated on the same GPU type.', '',
          '## A: generation quality (mean over samples, then cases)', '',
          '| group | method | cases | Hellinger | best-of-N Hell. | K=10 error | coverage | smooth | jerk RMS | outside | start err |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    order = ['ground_truth'] + list(s['models']) + ['chord']
    for g in dict.fromkeys(k.split('|')[1] for k in s['tables']):
        for meth in order:
            v = s['tables'].get(f'{meth}|{g}')
            if v:
                L.append(f"| {g} | {meth} | {v['n_cases']} | {v['hellinger']:.3f} | {v['hellinger_best']:.3f} | "
                         f"{v['ergodic_k10']:.3g} | {v.get('coverage', float('nan')):.3f} | {v['smooth']:.3f} | "
                         f"{v.get('jerk_rms', float('nan')):.3g} | {v['outside']:.4f} | {v['start_err']:.1e} |")
    if s['paired_vs_ref']:
        L += ['', f"## Paired against `{s['reference']}` (model minus reference, mean over cases, "
                  "95% cluster-bootstrap CI over densities; negative = better)", '',
              '| model | metric | difference | 95% CI | model better in |', '|---|---|---|---|---|']
        for lab, d in s['paired_vs_ref'].items():
            for k, v in d.items():
                L.append(f"| {lab} | {k} | {v['diff']:+.4g} | [{v['ci'][0]:+.4g}, {v['ci'][1]:+.4g}] | {v['better']:.0%} |")
    its = list(next(iter(s['warm_start'].values()), {}).keys())
    L += ['', '## B: warm start (Levenberg-Marquardt, K=16, w=15), Hellinger after n iterations', '',
          '| init | ' + ' | '.join(f'{i}' for i in its) + ' |', '|---|' + '---|' * len(its)]
    for lab, d in s['warm_start'].items():
        L.append(f"| {lab} | " + ' | '.join(f"{d[i]['hellinger']:.3f}" for i in its) + ' |')
    if s['warm_speed']:
        L += ['', 'LM iterations until the Hellinger distance chords reach after 100 iterations (per density):', '',
              '| init | median iterations | reached within 200 | already at 0 |', '|---|---|---|---|']
        for lab, v in s['warm_speed'].items():
            L.append(f"| {lab} | {v['median_iters']:g} | {v['share_reached']:.0%} | {v['share_at_iter0']:.0%} |")
    L += ['', '## C: start conditioning', '', '```', json.dumps(s['start_conditioning'], indent=1), '```']
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L) + '\n')


def _plot(s, models, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labs = list(s['models'])
    col = {lab: MODEL_COLORS[i % len(MODEL_COLORS)] for i, lab in enumerate(labs)}
    col.update(ground_truth=C_GT, chord=C_CHORD)
    # 1: quality bars, all cases
    meths = ['ground_truth'] + labs
    panels = [('hellinger', 'Hellinger distance to target'), ('ergodic_k10', 'Fourier error K=10'),
              ('smooth', 'Smoothness energy'), ('outside', 'Fraction outside [0.03, 0.97]')]
    fig, axes = plt.subplots(1, 4, figsize=(4 + 1.1 * len(meths) * 4 / 2, 4.2), facecolor='white')
    for ax, (k, title) in zip(axes, panels):
        _style(ax)
        vals = [s['tables'][f'{m}|all'][k] for m in meths]
        ax.bar(range(len(meths)), vals, color=[col[m] for m in meths], width=0.7)
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels([m.replace('ground_truth', 'GT') for m in meths], rotation=35, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10, color='#1A1A2E')
    fig.suptitle(f"Generation quality on all {s['n_cases']} holdout cases (mean over samples and cases)",
                 fontsize=11, color='#1A1A2E', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'quality_bars.png'), dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    # 2: metrics against lambda_erg, one line per (backbone, epochs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='white')
    for ax, (ci, title) in zip(axes, [(s['hellinger_ci'], 'Hellinger distance to target'),
                                      (s['smooth_ci'], 'Smoothness energy')]):
        _style(ax)
        fam = {}
        for lab, m in s['models'].items():
            fam.setdefault((m['backbone'], m['epoch']), []).append(lab)
        for (bb, ep), ls in fam.items():
            ls = sorted(ls, key=lambda l: s['models'][l]['lambda_erg'])
            x = [s['models'][l]['lambda_erg'] for l in ls]
            y = np.array([ci[l] for l in ls])
            ax.errorbar(x, y[:, 0], yerr=[y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]], color=col[ls[0]], lw=2,
                        marker='o' if bb == 'unet' else 's', ms=6, capsize=3, label=f'{bb}, {ep} epochs')
        if 'ground_truth|all' in s['tables'] and ci is s['hellinger_ci']:
            ax.axhline(s['tables']['ground_truth|all']['hellinger'], color=C_GT, ls='--', lw=1.5, label='Ground truth')
        ax.set_xlabel('ErgLoss weight lambda_erg', fontsize=9, color='#555')
        ax.set_title(title, fontsize=10, color='#1A1A2E')
    axes[0].legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#ddd')
    fig.suptitle('Effect of the ErgLoss weight (mean, 95% CI over densities)', fontsize=11, color='#1A1A2E', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'lambda_effect.png'), dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    # 3: warm start curves
    if s['warm_start']:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='white')
        for ax, (k, title) in zip(axes, [('hellinger', 'Hellinger distance'), ('ergodic_k10', 'Fourier error K=10'),
                                         ('smooth', 'Smoothness energy')]):
            _style(ax)
            for lab, d in s['warm_start'].items():
                its = [int(i) for i in d]
                ax.plot(its, [d[str(i)][k] for i in its], color=col[lab], lw=2, marker='o', ms=4,
                        ls='--' if lab == 'chord' else '-', label=lab)
            ax.set_xlabel('Levenberg-Marquardt iterations', fontsize=9, color='#555')
            ax.set_title(title, fontsize=10, color='#1A1A2E')
            if k != 'hellinger':
                ax.set_yscale('log')
        axes[0].legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#ddd')
        fig.suptitle('Warm start: solver quality after n iterations, by initialisation', fontsize=11,
                     color='#1A1A2E', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'warm_start.png'), dpi=140, bbox_inches='tight', facecolor='white')
        plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='evaluate one checkpoint and store all raw results')
    r.add_argument('--ckpt', required=True)
    r.add_argument('--label', required=True, help='model name in the report, e.g. unet-l630')
    r.add_argument('--db', required=True)
    r.add_argument('--out_tag', required=True)
    r.add_argument('--n_samples', type=int, default=30)
    r.add_argument('--steps', type=int, default=100)
    r.add_argument('--n_cand', type=int, default=90, help='initial curves per warm-start density')
    r.add_argument('--lm_iters', type=int, nargs='+', default=[0, 5, 10, 25, 50, 100, 200])
    r.add_argument('--n_grid_cases', type=int, default=8)
    r.add_argument('--grid_side', type=int, default=5)
    r.add_argument('--n_grid_samples', type=int, default=8)
    r.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    r.add_argument('--seed', type=int, default=0)
    q = sub.add_parser('report', help='build tables and plots from every stored model')
    q.add_argument('--out_tag', required=True)
    q.add_argument('--ref', default=None, help='label of the reference model for the paired table')
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()
