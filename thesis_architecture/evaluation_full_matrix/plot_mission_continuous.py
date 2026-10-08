r"""
plot_mission_continuous.py
===========================
Ergodic error E and J = E + 0.02 * path length of the DRIVEN path vs. the
ground truth as a continuous function of the distance driven -- including the
development inside each length unit, not only one value per executed unit
(that is what `plot_mission_eval.py` shows) -- together with the ALTERNATIVES
that were not driven.

Line: the driven path. Its units are concatenated exactly as the runner does
(`st.driven`) and the metric is evaluated for every prefix of that path. E is a
time average over the path points, so the Fourier coefficients of all prefixes
come from one cumulative sum (K = 10, the project metric of
`ExploreExploitErgodic`). At the end of each unit the value equals the stored
`E_truth` / `J_truth` (checked on load).

Band: the 30 candidates of every planning round. Each candidate's final SVGD
state is rendered into the unit it would have driven (same steps as
`run_mission_eval.MissionSet.execute`: snap to the start, clamp, trim to one
length unit, resample by arc length -- checked: the selected candidate
reproduces the stored segment), appended to the path driven so far, and scored
the same way. Per shape and position the mean +-1 std across the 30
candidates is taken; the band is the mean of that interval over the shapes.
All candidates of a round start from the same driven history, so the band
closes at every replanning point and opens up inside the unit.

Layout as `plot_svgd_convergence.py`: rows = target-density strategy, columns =
knowledge state, one colour per initialisation method, mean over the holdout
shapes. Missions that reached 99 % earlier keep their last value (no spread).

Optional right axis (`--svgd_conv`): at every replanning point, the number of
SVGD iterations until the planning metric has converged. The metric is the one
SVGD optimises: E against the round's planning target (stored `E_series`) for
the E figure, E + 0.02 * length of the full plan (from the stored SVGD states)
for the J figure. Two criteria, each applied identically to every method, one
figure pair each:

  plateau  first logged iteration after which the candidate's own metric
           improves by less than `--conv_tol` (relative) over the next
           `--conv_window` iterations (a standard optimiser stopping rule);
  target   time-to-target: first logged iteration at which the metric drops to
           a common quality level -- per planning round r, the median final
           value (after n_iters) over ALL candidates of all methods, knowledge
           states, strategies and shapes in round r (later rounds plan for more
           concentrated targets and end higher, so one global level would
           mostly measure the round, not the initialisation).

Candidates that never get there count as n_iters. Per shape and round the
median over the 30 candidates is taken; the markers are the mean over the
shapes still running.

Outputs (in <out_root>/<out_tag>/plots/):
    overview_E_truth_continuous_<n>units.png
    overview_J_truth_continuous_<n>units.png
    (with --svgd_conv: overview_{E,J}_truth_continuous_<n>units_svgd_conv_{plateau,target}.png)
plus the curves as <out_tag>/analysis/continuous_<n>units.npz.

Usage:
    python plot_mission_continuous.py --out_tag mission_eval_20261006 --max_units 8
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import glob
import multiprocessing as mp
import sqlite3

import numpy as np
import torch

import plot_mission_eval as pme                                     # noqa: E402  (sets sys.path)

plt = pme.plt


def _setup():
    torch.set_num_threads(1)


def score_prefixes(F_prev_sum, n_prev, s_prev, pts, ee, phi, lam):
    """E and J of (history + pts) for every prefix of `pts`.
    F_prev_sum: (M,) sum of basis values over the history points."""
    from ergodic_energy_torch import fourier_basis
    F = fourier_basis(pts, ee.k_idx)
    c = (F_prev_sum + F.cumsum(0)) / (n_prev + torch.arange(1, len(pts) + 1, dtype=F.dtype)).unsqueeze(1)
    E = (ee.w * 0.5 * ee.Lambda * (c - phi) ** 2).sum(-1).double().numpy()
    step = np.r_[0.0, np.linalg.norm(np.diff(pts.numpy(), axis=0), axis=1)]
    s = s_prev + np.cumsum(step)
    return E, E + lam * s, s, F


def iters_to_convergence(metric, stride, n_iters, window, tol):
    """First logged iteration after which `metric` improves by less than `tol`
    (relative to its current value) over the next `window` iterations; n_iters if
    that never happens inside the run."""
    w = max(1, int(round(window / stride)))
    m = np.asarray(metric, dtype=np.float64)
    if len(m) <= w:
        return float(n_iters)
    gain = m[:-w] - m[w:]
    ok = np.nonzero(gain < tol * np.abs(m[:-w]))[0]
    return float(ok[0] * stride) if len(ok) else float(n_iters)


def iters_to_target(metric, stride, n_iters, level):
    """First logged iteration at which `metric` <= `level`; n_iters if never."""
    ok = np.nonzero(np.asarray(metric) <= level)[0]
    return float(ok[0] * stride) if len(ok) else float(n_iters)


def shard_curves(task):
    """One shard -> {shape: dict(E, J, E_lo, E_hi, J_lo, J_hi)} on x_grid (length units)."""
    path, x_grid, max_units = task['path'], task['x_grid'], task['max_units']
    conv = task.get('conv')                    # None or dict(window, tol)
    from common.metrics import trim_to_length, path_length
    from exploration_optimierung.mission import LENGTH_UNIT, PTS_PER_UNIT, resample_arclength
    from metrics_explore_exploit import ExploreExploitErgodic, LAMBDA_LEN_J
    from run_svgd_convergence import basis_matrix, NXI, N_POINTS, DEGREE
    from state_codec import unpack_states

    ee = ExploreExploitErgodic(device='cpu')
    B = basis_matrix(NXI, N_POINTS, DEGREE).astype(np.float64)
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    out, worst_row, worst_seg, n_cands = {}, 0.0, 0.0, 0
    shapes = [r[0] for r in conn.execute("SELECT DISTINCT shape FROM rounds ORDER BY shape")]
    if task.get('n_shapes'):
        shapes = shapes[:task['n_shapes']]
    for shape in shapes:
        res, blob = conn.execute("SELECT res, density FROM truths WHERE shape=?", (shape,)).fetchone()
        truth = torch.from_numpy(np.frombuffer(blob, dtype=np.float32).reshape(res, res).copy())
        phi = ee.target_coeffs(truth)
        rows = conn.execute("SELECT round, segment, E_truth, J_truth, start_x, start_y FROM rounds "
                            "WHERE shape=? ORDER BY round", (shape,)).fetchall()
        segs = [torch.from_numpy(np.frombuffer(r[1], dtype=np.float32).reshape(-1, 2).copy()) for r in rows]
        # -- driven path, every prefix
        pts = torch.cat(segs)
        E, J, s, F = score_prefixes(torch.zeros(ee.k_idx.shape[0]), 0, 0.0, pts, ee, phi, LAMBDA_LEN_J)
        ends = np.cumsum([len(q) for q in segs]) - 1
        for k, i in enumerate(ends):
            worst_row = max(worst_row, abs(E[i] - rows[k][2]) / max(abs(rows[k][2]), 1e-9),
                            abs(J[i] - rows[k][3]) / max(abs(rows[k][3]), 1e-9))
        Fcum = F.cumsum(0)
        xu = s / LENGTH_UNIT
        keep = np.r_[True, np.diff(xu) > 0]
        rec = dict(E=np.interp(x_grid, xu[keep], E[keep], left=np.nan),
                   J=np.interp(x_grid, xu[keep], J[keep], left=np.nan))
        for key in ('E', 'J'):                    # alternatives: default = no spread
            rec[key + '_mean'] = rec[key].copy()
            rec[key + '_std'] = np.zeros_like(rec[key])
        rec['series'] = {}                        # round -> (E (n_cand, n_log), J, stride, n_iters)
        # -- alternatives of every round inside the plotted range
        for k, row in enumerate(rows):
            r = row[0]
            if r >= max_units:
                break
            n_prev = 0 if k == 0 else int(ends[k - 1]) + 1
            F_prev = torch.zeros(ee.k_idx.shape[0]) if k == 0 else Fcum[n_prev - 1]
            s_prev = 0.0 if k == 0 else float(s[n_prev - 1])
            start = np.array([row[4], row[5]], dtype=np.float32)
            cand_E, cand_J, cand_x, conv_E, conv_J = [], [], [], [], []
            for sel, n_states, nxi, blob_s, e_blob, e_stride, n_it in conn.execute(
                    "SELECT selected, n_states, nxi, states, E_series, E_stride, n_iters FROM candidates "
                    "WHERE shape=? AND round=? ORDER BY cand_idx", (shape, r)):
                all_states = unpack_states(blob_s, n_states, nxi)
                cps = all_states[-1].astype(np.float64)
                if conv is not None:
                    e_it = np.frombuffer(e_blob, dtype=np.float32).astype(np.float64)
                    plans = np.einsum('pi,sid->spd', B, all_states[::e_stride][:len(e_it)].astype(np.float64))
                    plan_len = np.linalg.norm(np.diff(plans, axis=1), axis=-1).sum(-1)
                    conv_E.append(e_it.astype(np.float32))
                    conv_J.append((e_it + LAMBDA_LEN_J * plan_len).astype(np.float32))
                    conv_meta = (int(e_stride), int(n_it))
                curve = torch.as_tensor(B @ cps, dtype=torch.float32)
                curve[0] = torch.as_tensor(start)
                curve = curve.clamp(0.0, 1.0)
                seg = trim_to_length(curve, LENGTH_UNIT)
                n_pts = max(8, int(round(PTS_PER_UNIT * path_length(seg) / LENGTH_UNIT)))
                seg = resample_arclength(seg, n_pts)
                if sel:
                    if seg.shape == segs[k].shape:
                        worst_seg = max(worst_seg, float((seg - segs[k]).abs().max()))
                    else:
                        worst_seg = max(worst_seg, 1.0)
                Ec, Jc, sc, _ = score_prefixes(F_prev, n_prev, s_prev, seg, ee, phi, LAMBDA_LEN_J)
                cand_E.append(Ec); cand_J.append(Jc); cand_x.append(sc / LENGTH_UNIT)
                n_cands += 1
            if not cand_E:
                continue
            if conv_E:
                rec['series'][r] = (np.stack(conv_E), np.stack(conv_J)) + conv_meta
            m = (x_grid > r + 1e-9) & (x_grid <= r + 1 + 1e-9)
            xg = x_grid[m]
            for key, vals in (('E', cand_E), ('J', cand_J)):
                G = np.stack([np.interp(xg, cx, v) for cx, v in zip(cand_x, vals)])
                rec[key + '_mean'][m] = G.mean(0)
                rec[key + '_std'][m] = G.std(0)
        out[shape] = rec
    conn.close()
    return dict(key=task['key'], curves=out, worst_row=worst_row, worst_seg=worst_seg, n_cands=n_cands)


def plot_grid(curves, metric, x_grid, max_units, n_shapes, n_cand, out_path, conv=None):
    label = {'E': 'Ergodic error E (vs. ground truth)', 'J': 'J = E + 0.02 * path length'}[metric]
    fig, axes, conds, strats = pme.grid_axes({k: True for k in curves}, figsize_unit=(4.6, 3.4))
    for i, st in enumerate(strats):
        for j, cnd in enumerate(conds):
            ax = axes[i][j]
            pme.style_axes(ax)
            for u in range(1, max_units):
                ax.axvline(u, color=pme.MUTED, lw=0.6, ls=':', alpha=0.6)
            for mth, sty in pme.METHOD_STYLE.items():
                per = curves.get((cnd, st, mth))
                if not per:
                    continue
                drv = np.nanmean(np.stack([v[metric] for v in per.values()]), 0)
                cm = np.stack([v[metric + '_mean'] for v in per.values()])
                cs = np.stack([v[metric + '_std'] for v in per.values()])
                lo = np.nanmean(cm - cs, 0)
                hi = np.nanmean(cm + cs, 0)
                lo = np.maximum(lo, np.nanmean(cm, 0) * 0.05)
                ok = np.isfinite(drv)
                ax.fill_between(x_grid[ok], lo[ok], hi[ok], color=sty['color'], alpha=0.16, linewidth=0)
                ax.plot(x_grid[ok], drv[ok], color=sty['color'], lw=sty['lw'], label=sty['label'],
                        alpha=0.95)
            ax.set_yscale('log')
            ax.set_xlim(0, max_units)
            ax.set_xticks(range(0, max_units + 1))
            if conv is not None:
                ax2 = ax.twinx()
                for mth, sty in pme.METHOD_STYLE.items():
                    per = curves.get((cnd, st, mth))
                    if not per:
                        continue
                    C = np.stack([v[f"conv_{conv['key']}_{metric}"] for v in per.values()])
                    with np.errstate(all='ignore'):
                        cm = np.nanmean(C, 0)
                    xs = np.arange(max_units)
                    ok = np.isfinite(cm)
                    ax2.plot(xs[ok], cm[ok], color=sty['color'], lw=0.9, ls='--', marker='s', ms=4.5,
                             alpha=0.55, mec='white', mew=0.6)
                ax2.set_ylim(0, conv['n_iters'] * 1.05)
                ax2.tick_params(colors=pme.MUTED, labelsize=8)
                for sp in ax2.spines.values():
                    sp.set_color(pme.SPINE)
                if j == len(conds) - 1:
                    ax2.set_ylabel('SVGD iterations to convergence\n(squares, dashed)', fontsize=8,
                                   color=pme.INK)
    title = (f"{label} of the driven path vs. distance driven -- mean over {n_shapes} holdout shapes; "
             f"line: driven path; band: +-1 std across the {n_cand} alternative candidates of each "
             f"round; dotted: replanning")
    if conv is not None:
        title += f"\nright axis (squares): {conv['desc'][metric]}"

    if conv is not None:                       # legend entry for the right-axis markers
        axes[0][0].plot([], [], color=pme.MUTED, lw=0.9, ls='--', marker='s', ms=4.5,
                        label='SVGD iterations to convergence (right axis)')
    pme.finish_grid(fig, axes, conds, strats, label if metric == 'J' else 'Ergodic error E (total)',
                    title, ncols_legend=4 if conv is not None else 3)
    if conv is not None:                       # two-line title: lift it clear of the legend
        fig.suptitle(title, fontsize=10, color=pme.INK, y=1.07)
    fig.savefig(out_path, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def convergence_variants(curves, conv, max_units):
    """Turn the per-candidate SVGD metric series into the two convergence
    criteria (stored per shape as conv_<key>_<E|J>, one value per replanning
    point) and return the plot descriptions."""
    final = {}
    for per in curves.values():
        for rec in per.values():
            for r, ser in rec['series'].items():
                for i, m in enumerate(('E', 'J')):
                    final.setdefault((m, r), []).append(ser[i][:, -1])
    level = {k: float(np.median(np.concatenate(v))) for k, v in final.items()}
    tol, win, n_it = conv['tol'], conv['window'], conv['n_iters']
    rules = {
        'plateau': lambda x, stride, n, m, r: iters_to_convergence(x, stride, n, win, tol),
        'target': lambda x, stride, n, m, r: iters_to_target(x, stride, n, level[(m, r)]),
    }
    summary = {}
    for key, rule in rules.items():
        for (cnd, st, mth), per in curves.items():
            for rec in per.values():
                for m, i in (('E', 0), ('J', 1)):
                    arr = np.full(max_units, np.nan)
                    for r, ser in rec['series'].items():
                        vals = [rule(x, ser[2], ser[3], m, r) for x in ser[i]]
                        arr[r] = float(np.median(vals))
                        summary.setdefault((key, m, mth), []).extend(vals)
                    rec[f'conv_{key}_{m}'] = arr
    for (key, m, mth), v in sorted(summary.items()):
        v = np.asarray(v)
        print(f"[continuous] conv {key:7s} {m} {mth:11s}: median {np.median(v):6.0f} iters, "
              f"not converged within {n_it}: {np.mean(v >= n_it) * 100:5.1f} %", flush=True)
    for m in ('E', 'J'):
        lv = ', '.join(f"r{r + 1} {level[(m, r)]:.3f}" for r in range(max_units) if (m, r) in level)
        print(f"[continuous] time-to-target levels {m} (median final value of all candidates per round): {lv}",
              flush=True)
    for per in curves.values():
        for rec in per.values():
            rec.pop('series', None)
    base = "SVGD iterations until "
    tail = f"(median over the candidates, mean over shapes; capped at {n_it})"
    return [
        dict(key='plateau', n_iters=n_it, desc={
            m: f"{base}the planning metric improves by < {tol * 100:g} % over the next {win} iterations "
               f"[plateau criterion] {tail}" for m in ('E', 'J')}),
        dict(key='target', n_iters=n_it, desc={
            m: f"{base}the planning metric reaches the median final value of all candidates of all "
               f"methods in that round [time-to-target criterion] {tail}" for m in ('E', 'J')}),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--out_root', type=str, default=os.path.join(pme._here, 'results'))
    ap.add_argument('--max_units', type=int, default=8, help='x range in length units.')
    ap.add_argument('--res', type=int, default=50, help='Grid points per length unit.')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--svgd_conv', action='store_true',
                    help='Add a right axis with SVGD iterations to convergence at every replanning '
                         'point (writes *_svgd_conv.png, the plain figures stay untouched).')
    ap.add_argument('--conv_window', type=int, default=100, help='Look-ahead window in SVGD iterations.')
    ap.add_argument('--conv_tol', type=float, default=0.05,
                    help='Converged once the relative improvement over the window drops below this.')
    ap.add_argument('--n_iters', type=int, default=1500, help='SVGD iterations per round in the run.')
    ap.add_argument('--shards', type=str, default='',
                    help='Comma-separated shard names (cond__strategy__method) to restrict to (smoke test).')
    ap.add_argument('--n_shapes', type=int, default=0, help='Only the first n shapes per shard (smoke test).')
    a = ap.parse_args()

    root = os.path.join(a.out_root, a.out_tag)
    x_grid = np.linspace(1.0 / a.res, a.max_units, a.max_units * a.res)
    conv = dict(window=a.conv_window, tol=a.conv_tol, n_iters=a.n_iters) if a.svgd_conv else None
    only = {x for x in a.shards.split(',') if x}
    tasks = []
    for f in sorted(glob.glob(os.path.join(root, 'shards', '*.db'))):
        name = os.path.basename(f)[:-3]
        if only and name not in only:
            continue
        tasks.append(dict(key=tuple(name.split('__')), path=f, x_grid=x_grid,
                          max_units=a.max_units, conv=conv, n_shapes=a.n_shapes))
    curves, n_shapes, worst_row, worst_seg, n_cands = {}, 0, 0.0, 0.0, 0
    with mp.get_context('spawn').Pool(a.workers, initializer=_setup) as pool:
        for res in pool.imap_unordered(shard_curves, tasks):
            curves[res['key']] = res['curves']
            n_shapes = max(n_shapes, len(res['curves']))
            worst_row, worst_seg = max(worst_row, res['worst_row']), max(worst_seg, res['worst_seg'])
            n_cands += res['n_cands']
            print(f"[continuous] {'/'.join(res['key'])}: {len(res['curves'])} missions, "
                  f"{res['n_cands']} alternatives", flush=True)
    print(f"[continuous] driven path vs. stored E_truth/J_truth: worst relative deviation {worst_row:.1e}; "
          f"re-rendered selected candidate vs. stored segment: worst {worst_seg:.1e}; "
          f"{n_cands} alternatives scored", flush=True)
    if worst_row > 1e-3 or worst_seg > 1e-2:
        raise AssertionError('consistency check failed')

    variants = [None]
    if conv is not None:
        variants = convergence_variants(curves, conv, a.max_units)
    plots = os.path.join(root, 'plots')
    os.makedirs(plots, exist_ok=True)
    smoke = bool(only or a.n_shapes)
    for var in variants:
        suffix = (f"_svgd_conv_{var['key']}" if var else '') + ('_smoke' if smoke else '')
        for metric in ('E', 'J'):
            p = os.path.join(plots, f'overview_{metric}_truth_continuous_{a.max_units}units{suffix}.png')
            plot_grid(curves, metric, x_grid, a.max_units, n_shapes, 30, p, conv=var)
            print(f"[continuous] wrote {p}", flush=True)
    if smoke:
        return
    os.makedirs(os.path.join(root, 'analysis'), exist_ok=True)
    fields = ('E', 'J', 'E_mean', 'E_std', 'J_mean', 'J_std')
    if conv is not None:
        fields += tuple(f"conv_{v['key']}_{m}" for v in variants for m in ('E', 'J'))
    suffix = '_svgd_conv' if conv is not None else ''
    np.savez_compressed(
        os.path.join(root, 'analysis', f'continuous_{a.max_units}units{suffix}.npz'), x_units=x_grid,
        **{f"{k[0]}__{k[1]}__{k[2]}__{fld}": np.stack([v[fld] for v in per.values()])
           for k, per in curves.items()
           for fld in fields})


if __name__ == '__main__':
    mp.freeze_support()
    main()
