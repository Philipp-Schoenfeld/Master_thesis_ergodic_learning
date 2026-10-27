r"""
eval_start_cfm_model.py -- structured ~45 min test of a start-conditioned CFM checkpoint
=========================================================================================
Built for the model trained on the teacher-study DB (`run_job_start_lm.bash`, run name
`..._CPS_INPAINT_LMTEACHER_...`), but works for any `start_cond` checkpoint.

Why a new script: `run_init_comparison.py` samples the CFM WITHOUT a start point, which for a
start-conditioned network means no FiLM conditioning and no pinned control point 0 -- not the model
that was trained. Here every sample is conditioned on the start x0 of its test case.

Test cases: the holdout rows of the training DB (`--db`, default the LM DB): split `val` (24 base
densities + their `#unk0` large-area variants, the stored start) and `val_start` (the same densities
with two OTHER start points, never seen in training). Ground truth = the stored teacher trajectory.
All metrics come from `teacher_study.pipeline.stage_metrics` (Hellinger, TV, coverage, K=10 Fourier
error, smoothness, acceleration/jerk RMS, curvature, fraction outside [0.03, 0.97], start error).

  A  generation quality     N samples per case, no solver: new model vs. old model (`--old_ckpt`,
                            the L500 start model) vs. ground truth vs. chord baseline
  B  warm start             Levenberg-Marquardt on the TSVEC energy (K=16, w=15, start pinned, the
                            teacher method's solver) from 30 initial curves: new model / old model /
                            chords; metrics after 0 ... 200 iterations; iterations to reach the
                            quality chords reach after 100
  C  start conditioning     start error, how much the generated curves change between the three
                            starts of one density, a 5x5 grid of starts

A part that would exceed `--budget_min` is skipped; sizes are set by --n_cases / --n_warm.

    python eval_start_cfm_model.py --ckpt <new.pt> --old_ckpt ../transfer/netz2d_startpunkt.pt \
        --db ../ergodic_dataset_generator/ergodic_dataset_lm_nxi25.db --out_tag lm_model_eval
"""
import argparse
import json
import os
import sqlite3
import sys
import time
import zlib

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
for _p in (_here, os.path.join(_arch, 'exploration'), os.path.join(_arch, 'teacher_study'),
           os.path.join(_arch, 'ergodic_dataset_generator'), _arch,
           os.path.join(os.path.dirname(_arch), 'bsplinax-main'), os.path.join(os.path.dirname(_arch), 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline as pl                                              # noqa: E402  (teacher_study)
import generate_dataset_lm as glm                                  # noqa: E402

RESULTS = os.path.join(_here, 'results')
C_NEW, C_OLD, C_GT, C_CHORD = '#00C853', '#EF6C00', '#1565C0', '#777777'


def load_cases(db, n_cases, seed):
    """Holdout cases: split val (stored start) and val_start (other starts), `n_cases` densities."""
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT shape_name, split, density_params, x0, control_points, nxi FROM ergodic_pairs"
                        " WHERE split IN ('val','val_start') ORDER BY id").fetchall()
    conn.close()
    dens = list(dict.fromkeys(r[0].split('@')[0] for r in rows))
    if n_cases is not None and n_cases < len(dens):
        pick = np.random.default_rng(seed).choice(len(dens), n_cases, replace=False)
        keep = {dens[i] for i in pick}
        rows = [r for r in rows if r[0].split('@')[0] in keep]
    cases = []
    for name, split, dp, x0, cps, nxi in rows:
        x0 = np.asarray(json.loads(x0), dtype=np.float64)
        sd = glm.shape_data(name, dp, x0)
        cases.append(dict(name=name, density=name.split('@')[0], split=split,
                          start_idx=int(name.split('@s')[1]) if '@s' in name else 0,
                          kind='unk0' if '#unk0' in name else 'base', sd=sd,
                          gt_cps=np.frombuffer(cps, dtype=np.float32).reshape(nxi, 2).astype(np.float64),
                          truth=torch.as_tensor(sd.density)))
    return cases


def sample(planner, case, n, device, seed, start=None):
    """n CFM samples for a case, conditioned on its start -> (n, 128, 2) float64 dense curves."""
    import apply_cfm_belief as acb
    torch.manual_seed(seed)
    parts = acb.phi_particles(case['truth'].to(device), 256, mode='uniform', device=device)
    x0 = case['sd'].x0 if start is None else start
    cps = planner.plan(parts, n_candidates=n, start=torch.as_tensor(x0, dtype=torch.float32))
    return planner.render(cps).detach().cpu().numpy().astype(np.float64)


def metrics_row(X, case, x0=None):
    return pl.stage_metrics(np.clip(X, 0.0, 1.0), case['sd'].density, case['sd'].x0 if x0 is None else x0)


def seed_of(*parts):
    return zlib.crc32('|'.join(str(p) for p in parts).encode()) & 0x7FFFFFFF


KEYS = ('hellinger', 'ergodic_k10', 'coverage', 'smooth', 'acc_rms', 'jerk_rms', 'curv_p95',
        'outside', 'path_len_arc', 'start_err')


def summarise(rows, group):
    """rows: list of dict(method, group-keys..., metric arrays (n,)) -> {(method, group): {key: mean, key_best: ..}}"""
    out = {}
    for r in rows:
        out.setdefault((r['method'], r[group]), []).append(r)
    res = {}
    for k, rs in out.items():
        d = {}
        for key in KEYS:
            m = np.array([x[key].mean() for x in rs])
            d[key] = float(m.mean())
            d[key + '_med'] = float(np.median(m))
        d['hellinger_best'] = float(np.mean([x['hellinger'].min() for x in rs]))
        d['n_cases'] = len(rs)
        res[k] = d
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--old_ckpt', default=None)
    ap.add_argument('--db', required=True)
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--n_cases', type=int, default=None, help='number of holdout DENSITIES (each with its starts)')
    ap.add_argument('--n_samples', type=int, default=10)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--n_warm', type=int, default=24, help='densities of the warm-start part (stored start)')
    ap.add_argument('--n_cand', type=int, default=30, help='initial curves per warm-start cell')
    ap.add_argument('--lm_iters', type=int, nargs='+', default=[0, 5, 10, 25, 50, 100, 200])
    ap.add_argument('--budget_min', type=float, default=45.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    out = os.path.join(RESULTS, a.out_tag)
    os.makedirs(out, exist_ok=True)
    t_start = time.time()
    elapsed = lambda: (time.time() - t_start) / 60.0
    dev = a.device

    from apply_cfm_belief import CfmPlanner
    new = CfmPlanner(ckpt=a.ckpt, steps=a.steps, device=dev)
    old = CfmPlanner(ckpt=a.old_ckpt, steps=a.steps, device=dev) if a.old_ckpt else None
    print(f"[eval] new model: nxi={new.nxi}, start_cond={new.start_cond}, "
          f"start_inpaint={getattr(new.model, 'start_inpaint', False)}"
          + (f"; old model: nxi={old.nxi}, start_inpaint={getattr(old.model, 'start_inpaint', False)}" if old else ''),
          flush=True)
    cases = load_cases(a.db, a.n_cases, a.seed)
    print(f"[eval] {len(cases)} cases ({len({c['density'] for c in cases})} densities), "
          f"n_samples={a.n_samples}, steps={a.steps}, device={dev}", flush=True)

    # ------------------------------------------------------------------ A: generation quality
    rows, gens = [], {}
    t0 = time.time()
    for i, c in enumerate(cases):
        if elapsed() > 0.4 * a.budget_min:
            print(f"[eval] part A: time share used up after {i}/{len(cases)} cases", flush=True)
            break
        sd = c['sd']
        base = dict(case=c['name'], density=c['density'], split=c['split'], kind=c['kind'],
                    start=('stored' if c['start_idx'] == 0 else 'new'))
        gt = pl.render(c['gt_cps'][None], pl.N_POINTS, pl.DEGREE)
        rows.append(dict(method='ground_truth', **base, **metrics_row(gt, c)))
        rows.append(dict(method='chord', **base, **metrics_row(pl.chord_inits(sd.x0, a.n_samples), c)))
        Xn = sample(new, c, a.n_samples, dev, seed_of(a.seed, c['name'], 'new'))
        rows.append(dict(method='new', **base, **metrics_row(Xn, c)))
        gens[c['name']] = Xn
        if old is not None and c['split'] == 'val':       # the old model only on the stored starts
            Xo = sample(old, c, a.n_samples, dev, seed_of(a.seed, c['name'], 'old'))
            rows.append(dict(method='old', **base, **metrics_row(Xo, c)))
        if (i + 1) % 20 == 0:
            print(f"[eval] A {i + 1}/{len(cases)}  {(time.time() - t0) / (i + 1):.1f} s/case  "
                  f"{elapsed():.1f} min", flush=True)
    n_done = len({r['case'] for r in rows})
    print(f"[eval] part A done: {n_done} cases, {elapsed():.1f} min", flush=True)

    tables = {}
    for grp in ('split', 'kind', 'start'):
        tables[grp] = {f'{m}|{g}': v for (m, g), v in summarise(rows, grp).items()}
    # paired comparison new vs old on the stored starts
    paired = {}
    if old is not None:
        by = {(r['method'], r['case']): r for r in rows}
        shared = sorted({c for (m, c) in by if m == 'old'})
        for key in ('hellinger', 'ergodic_k10', 'smooth', 'outside'):
            paired[key] = dict(
                new=float(np.mean([by[('new', c)][key].mean() for c in shared])),
                old=float(np.mean([by[('old', c)][key].mean() for c in shared])),
                new_better=float(np.mean([by[('new', c)][key].mean() < by[('old', c)][key].mean() for c in shared])),
                n=len(shared))

    # ------------------------------------------------------------------ C: start conditioning
    startc = {}
    if gens:
        err = np.concatenate([r['start_err'] for r in rows if r['method'] == 'new'])
        startc['start_err_max'], startc['start_err_mean'] = float(err.max()), float(err.mean())
        # route distance between the mean curve of different starts of one density, and within a case
        by_d = {}
        for c in cases:
            if c['name'] in gens:
                by_d.setdefault(c['density'], []).append(gens[c['name']])
        d_start, d_within = [], []
        for v in by_d.values():
            for X in v:
                d_within.append(np.mean([np.linalg.norm(X[i] - X[j], axis=-1).mean()
                                         for i in range(len(X)) for j in range(i + 1, len(X))]))
            for p in range(len(v)):
                for q in range(p + 1, len(v)):
                    d_start.append(np.linalg.norm(v[p][0] - v[q][0], axis=-1).mean())
        startc['route_dist_between_starts'] = float(np.mean(d_start)) if d_start else None
        startc['route_dist_within_case'] = float(np.mean(d_within))
        # 5x5 grid of starts on a few densities
        vals = [c for c in cases if c['split'] == 'val'][:4]
        g_err, g_out = [], []
        for c in vals:
            for gx in np.linspace(0.05, 0.95, 5):
                for gy in np.linspace(0.05, 0.95, 5):
                    X = sample(new, c, 4, dev, seed_of(a.seed, 'grid', c['name'], gx, gy), start=np.array([gx, gy]))
                    m = metrics_row(X, c, x0=np.array([gx, gy]))
                    g_err.append(m['start_err'].max())
                    g_out.append(m['outside'].mean())
        startc['grid_start_err_max'], startc['grid_outside_mean'] = float(np.max(g_err)), float(np.mean(g_out))
        print(f"[eval] part C: {startc}  {elapsed():.1f} min", flush=True)

    # ------------------------------------------------------------------ B: warm start
    warm = {}
    from common.tsvec_svgd import TsvecSvgd
    wcases = [c for c in cases if c['split'] == 'val'][:a.n_warm]
    inits = [('new', new), ('old', old), ('chord', None)]
    t_b = time.time()
    for wi, c in enumerate(wcases):
        if elapsed() > 0.92 * a.budget_min:
            print(f"[eval] part B: time budget reached after {wi}/{len(wcases)} densities", flush=True)
            break
        sd = c['sd']
        energy = TsvecSvgd(B=None, K=16, device=dev, w_smooth=15.0, beta=1.0, smooth_order=2)
        phi_k = pl._phi_k(sd.density, energy.k_idx, dev)
        for name, planner in inits:
            if name == 'old' and old is None:
                continue
            X0 = (pl.chord_inits(sd.x0, a.n_cand) if planner is None
                  else sample(planner, c, a.n_cand, dev, seed_of(a.seed, c['name'], 'warm', name)))
            X0 = np.clip(X0, 0.02, 0.98)
            X0[:, 0] = sd.x0
            P, done = X0.astype(np.float32), 0
            for it in a.lm_iters:
                if it > done:
                    P = pl._levenberg_marquardt(energy, P, phi_k, it - done, start=sd.x0)
                    P = P.detach().cpu().numpy() if torch.is_tensor(P) else P
                    done = it
                m = metrics_row(np.asarray(P, dtype=np.float64), c)
                warm.setdefault(name, {}).setdefault(it, []).append(
                    {k: float(m[k].mean()) for k in ('hellinger', 'ergodic_k10', 'smooth', 'outside')})
        if (wi + 1) % 4 == 0:
            print(f"[eval] B {wi + 1}/{len(wcases)}  {(time.time() - t_b) / (wi + 1):.1f} s/density  "
                  f"{elapsed():.1f} min", flush=True)
    warm_sum = {n: {str(it): {k: float(np.mean([x[k] for x in v])) for k in v[0]} for it, v in d.items()}
                for n, d in warm.items()}
    # iterations needed to match what chords reach after 100 iterations (per density, then averaged)
    speed = {}
    if 'chord' in warm and 100 in a.lm_iters:
        n_w = len(warm['chord'][100])
        for n in warm:
            its = []
            for j in range(n_w):
                target = warm['chord'][100][j]['hellinger']
                hit = [it for it in a.lm_iters if warm[n][it][j]['hellinger'] <= target]
                its.append(min(hit) if hit else float('inf'))
            its = np.array(its)
            speed[n] = dict(median_iters=float(np.median(its)),
                            share_reached_within_budget=float(np.mean(np.isfinite(its))),
                            share_at_iter0=float(np.mean(its == 0)))
    print(f"[eval] part B done: {elapsed():.1f} min", flush=True)

    res = dict(args=vars(a), n_cases=n_done, tables=tables, paired_new_vs_old=paired, start_conditioning=startc,
               warm_start=warm_sum, warm_speed=speed, minutes=elapsed())
    json.dump(res, open(os.path.join(out, 'results.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(out, 'generated.npz'), **{k: v.astype(np.float32) for k, v in gens.items()})
    write_report(res, out)
    plot_all(rows, warm_sum, cases, gens, out)
    print(f"[eval] done in {elapsed():.1f} min -> {out}", flush=True)


def write_report(res, out):
    L = [f"# Start-conditioned CFM evaluation `{res['args']['out_tag']}`", '',
         f"{res['n_cases']} holdout cases, {res['args']['n_samples']} samples each, {res['minutes']:.1f} min.", '',
         '## A: generation quality (mean over samples, then cases)', '',
         '| group | method | cases | Hellinger | best-of-N Hellinger | K=10 error | smooth | outside | start err |',
         '|---|---|---|---|---|---|---|---|---|']
    for grp, tab in res['tables'].items():
        for k, v in sorted(tab.items(), key=lambda kv: (kv[0].split('|')[1], kv[0].split('|')[0])):
            m, g = k.split('|')
            L.append(f"| {grp}={g} | {m} | {v['n_cases']} | {v['hellinger']:.3f} | {v['hellinger_best']:.3f} | "
                     f"{v['ergodic_k10']:.2e} | {v['smooth']:.3f} | {v['outside']:.3f} | {v['start_err']:.1e} |")
    if res['paired_new_vs_old']:
        L += ['', '## New vs old model, paired on the stored starts', '', '| metric | new | old | new better in |', '|---|---|---|---|']
        for k, v in res['paired_new_vs_old'].items():
            L.append(f"| {k} | {v['new']:.4g} | {v['old']:.4g} | {v['new_better']:.0%} of {v['n']} |")
    L += ['', '## C: start conditioning', '', '```', json.dumps(res['start_conditioning'], indent=1), '```',
          '', '## B: warm start (LM, K=16, w=15)', '', '| init | ' + ' | '.join(f"H@{it}" for it in res['warm_start'].get('chord', {})) + ' |',
          '|---|' + '---|' * len(res['warm_start'].get('chord', {}))]
    for n, d in res['warm_start'].items():
        L.append(f"| {n} | " + ' | '.join(f"{v['hellinger']:.3f}" for v in d.values()) + ' |')
    if res['warm_speed']:
        L += ['', 'Iterations to match the Hellinger distance chords reach after 100 iterations:', '', '```',
              json.dumps(res['warm_speed'], indent=1), '```']
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(L))


def _style(ax):
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2, lw=0.4, color='gray')
    for s in ax.spines.values():
        s.set_color('#ccc')
    ax.tick_params(labelsize=8, colors='#555')


def plot_all(rows, warm, cases, gens, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = dict(new=C_NEW, old=C_OLD, ground_truth=C_GT, chord=C_CHORD)
    lab = dict(new='New model (LM teacher DB)', old='Old model (L500)', ground_truth='Ground truth (teacher)',
               chord='Chords (no solver)')
    methods = [m for m in ('ground_truth', 'new', 'old', 'chord') if any(r['method'] == m for r in rows)]
    # 1: metric bars on the stored starts of the holdout densities
    panels = [('hellinger', 'Hellinger distance to target'), ('smooth', 'Smoothness energy'),
              ('outside', 'Fraction of points outside [0.03, 0.97]'), ('start_err', 'Start error')]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), facecolor='white')
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        vals = [np.mean([r[key].mean() for r in rows if r['method'] == m and r['split'] == 'val']) for m in methods]
        ax.bar(range(len(methods)), vals, color=[col[m] for m in methods], width=0.7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(['GT', 'New', 'Old', 'Chord'][:0] or [m.replace('ground_truth', 'GT').title() for m in methods],
                           fontsize=8)
        ax.set_title(title, fontsize=10, color='#1A1A2E')
    fig.suptitle('Generation quality on the holdout densities (stored start points, mean over cases)',
                 fontsize=11, color='#1A1A2E', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'quality_bars.png'), dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    # 2: warm start curves
    if warm:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor='white')
        for ax, (key, title) in zip(axes, [('hellinger', 'Hellinger distance'), ('ergodic_k10', 'Fourier error K=10'),
                                           ('smooth', 'Smoothness energy')]):
            _style(ax)
            for n, d in warm.items():
                its = [int(i) for i in d]
                ax.plot(its, [d[str(i)][key] for i in its], color=col[n], lw=2, marker='o', ms=4, label=lab[n])
            ax.set_xlabel('Levenberg-Marquardt iterations', fontsize=9, color='#555')
            ax.set_title(title, fontsize=10, color='#1A1A2E')
            if key != 'hellinger':
                ax.set_yscale('log')
        axes[0].legend(fontsize=8, frameon=True, facecolor='white', edgecolor='#ddd')
        fig.suptitle('Warm start: quality after solver iterations, by initialisation', fontsize=11,
                     color='#1A1A2E', y=1.03)
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'warm_start.png'), dpi=140, bbox_inches='tight', facecolor='white')
        plt.close()
    # 3: gallery -- one density, three starts, generated vs ground truth
    from analyze_teacher_study import WHITE_INFERNO
    by_d = {}
    for c in cases:
        if c['name'] in gens:
            by_d.setdefault(c['density'], []).append(c)
    pick = [d for d, v in by_d.items() if len(v) >= 3][:6]
    if pick:
        fig, axes = plt.subplots(len(pick), 3, figsize=(9, 3.1 * len(pick)), facecolor='white', squeeze=False)
        for i, d in enumerate(pick):
            for j, c in enumerate(sorted(by_d[d], key=lambda x: x['start_idx'])[:3]):
                ax = axes[i][j]
                ax.set_facecolor('white')
                ax.imshow(c['sd'].density, extent=[0, 1, 0, 1], origin='lower', cmap=WHITE_INFERNO,
                          vmin=0, vmax=1, alpha=0.55, aspect='auto')
                X = gens[c['name']]
                for k in range(len(X)):
                    ax.plot(X[k, :, 0], X[k, :, 1], color=C_NEW, lw=2.2 if k == 0 else 1.0,
                            alpha=0.95 if k == 0 else 0.3)
                g = pl.render(c['gt_cps'][None], 256, pl.DEGREE)[0]
                ax.plot(g[:, 0], g[:, 1], color=C_GT, lw=2.5, alpha=0.9)
                ax.scatter([c['sd'].x0[0]], [c['sd'].x0[1]], color=C_GT, marker='*', s=130,
                           edgecolors='white', linewidths=0.6, zorder=4)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color('#ccc')
                ax.set_title(f"{d}  start {j}", fontsize=8, color='#1A1A2E')
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'start_gallery.png'), dpi=130, bbox_inches='tight', facecolor='white')
        plt.close()


if __name__ == '__main__':
    main()
