r"""
extend_svgd_convergence.py
===========================
Continue every SVGD run of `run_svgd_convergence.py` to a larger total
iteration count (default: 3000) so the convergence curves can be followed
from iteration 0 to 3000.

Why a re-run and not a "continuation from the stored state": the DB keeps the
control points of the best particle per iteration, but not the other 7
particles of the swarm nor the Adam moments, so the solver state at
iteration 1000 is not stored. The runs are however fully deterministic
(task seed -> jitter, SVGD itself has no randomness): re-running a run from
its stored raw initialisation with the same seed reproduces the stored
states bit for bit (checked on stored runs: max |diff| = 0.0). So each run is
re-computed to `--total_iters`, the first `n_iters+1` states are compared with
the stored ones, and ONLY the new states (iterations n_iters+1 .. total) are
written to the table `runs_ext`. Rows of `runs` are never modified. The
deviation of the re-run's prefix is stored per run (`prefix_max_abs_diff`,
`prefix_E_max_rel_diff`); 0.0 means the extension is an exact continuation of
the stored trajectory.

All inputs come from the DB itself (raw init, target density, config), no GPU
or checkpoint needed. Resumable: runs that already have a `runs_ext` row are
skipped.

Example
-------
    # smoke test (seconds)
    python extend_svgd_convergence.py --out_tag smoke_svgd_conv_real2 \
        --total_iters 1100 --limit_runs 4 --workers 4

    # full extension (~7 h, NOT started automatically)
    python extend_svgd_convergence.py --out_tag svgd_convergence_20261001
"""

import argparse
import collections
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                       # noqa: E402

import run_svgd_convergence as rsc                                 # noqa: E402
import svgd_convergence_db as sdb                                  # noqa: E402


def extend_task(task):
    """Worker: re-run to `total_iters`, compare the prefix, return the new
    states. Same refiner call (seed, init, target) as `run_svgd_convergence`."""
    try:
        stored_n = task['stored_n_iters']
        res = rsc.run_task(dict(
            shape=task['shape'], cond=task['cond'], strategy=task['strategy'],
            method=task['method'], init_idx=task['init_idx'],
            init_param=None, init_curve=task['init_curve'], phi=task['phi'],
            phi_k_truth=task['phi_k_truth'], n_iters=task['total_iters'],
            nxi=task['nxi'], seed=task['seed'], refiner=task['refiner']))
        cps = res['cps']
        prefix_diff = float(np.abs(cps[:stored_n + 1] - task['stored_cps']).max())
        e_old = task['stored_E_total']
        e_new = res['E_total'][:stored_n + 1]
        e_rel = float((np.abs(e_new - e_old) / np.maximum(np.abs(e_old), 1e-12)).max())
        sl = slice(stored_n + 1, None)
        return dict(run_id=task['run_id'], start_iter=stored_n + 1,
                    n_iters_total=task['total_iters'], cps=cps[sl],
                    E_total=res['E_total'][sl], E_explore=res['E_explore'][sl],
                    E_exploit=res['E_exploit'][sl], path_len=res['path_len'][sl],
                    prefix_max_abs_diff=prefix_diff, prefix_E_max_rel_diff=e_rel)
    except Exception:                                              # noqa: BLE001
        import traceback
        return {'error': traceback.format_exc(), 'key': task.get('run_id')}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', type=str, required=True)
    ap.add_argument('--total_iters', type=int, default=3000,
                    help='Total SVGD iterations after the extension (default 3000).')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device for the (cheap) target-coefficient computation; the '
                         'original run used the default (GPU if available).')
    ap.add_argument('--limit_runs', type=int, default=None,
                    help='Only extend this many runs (smoke tests).')
    ap.add_argument('--time_budget_h', type=float, default=None,
                    help='Stop submitting after this many hours; re-run to resume.')
    args = ap.parse_args()

    from metrics_explore_exploit import ExploreExploitErgodic

    db_path = os.path.join(_here, 'results', args.out_tag, 'svgd_convergence.db')
    if not os.path.isfile(db_path):
        raise FileNotFoundError(db_path)
    conn = sdb.open_db(db_path)
    cfg = sdb.get_meta(conn, 'config', {})
    svgd_target = cfg.get('svgd_target', 'belief')   # DBs before the flag existed: belief
    # Verlaengert wird immer mit dem Refiner, mit dem die DB entstanden ist
    # (DBs vor dem --refiner-Flag: der bisherige TSVEC-Refiner).
    refiner = cfg.get('refiner', 'tsvec')
    done = sdb.ext_run_ids(conn)

    ee = ExploreExploitErgodic(device=args.device)
    truth_cache = {}

    def truth_and_phi_k(shape):
        if shape not in truth_cache:
            t = sdb.load_truth(conn, shape)
            t = np.array(t)                       # writable copy (DB blobs are read-only)
            pk = ee.target_coeffs(torch.as_tensor(t)).detach().cpu().numpy()
            truth_cache[shape] = (t, pk)
        return truth_cache[shape]

    todo = []
    for r in conn.execute("SELECT id, shape, knowledge_condition, strategy, method, "
                          "init_idx, n_iters FROM runs ORDER BY id"):
        if r[0] in done:
            continue
        if r[6] >= args.total_iters:
            raise ValueError(f"run {r[0]} already has {r[6]} >= {args.total_iters} iterations")
        todo.append(r)
    if args.limit_runs is not None:
        todo = todo[:args.limit_runs]
    print(f"[extend] {len(todo)} runs to extend to {args.total_iters} iterations "
          f"({len(done)} already extended), svgd_target={svgd_target}", flush=True)
    if not todo:
        return

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(args.workers, initializer=rsc._worker_init)
    pending = collections.deque()
    max_pending = args.workers * 2
    t0 = time.time()
    n_done = n_fail = 0
    worst_prefix = 0.0

    def drain(block=False):
        nonlocal n_done, n_fail, worst_prefix
        while pending and (block or pending[0].ready()):
            res = pending.popleft().get()
            if 'error' in res:
                n_fail += 1
                print(f"[extend] FAILED run {res['key']}:\n{res['error']}", flush=True)
                continue
            sdb.save_run_ext(conn, **res)
            worst_prefix = max(worst_prefix, res['prefix_max_abs_diff'])
            n_done += 1
            if n_done % 25 == 0:
                conn.commit()
                el = (time.time() - t0) / 60
                print(f"[extend] {n_done}/{len(todo)} runs extended, {n_fail} failed, "
                      f"{el:.1f} min, worst prefix deviation so far {worst_prefix:.3e}",
                      flush=True)

    try:
        for (rid, shape, cond, strat, method, idx, stored_n) in todo:
            if (args.time_budget_h is not None
                    and (time.time() - t0) / 3600.0 > args.time_budget_h):
                print("[extend] time budget reached, stopping submission.", flush=True)
                break
            run = next(sdb.iter_runs(conn, columns=('cps', 'E_total'), shape=shape,
                                     knowledge_condition=cond, strategy=strat,
                                     method=method, init_idx=idx))
            truth, phi_k = truth_and_phi_k(shape)
            phi = truth if svgd_target == 'truth' or cond == sdb.SHARED \
                else sdb.load_target(conn, shape, cond, strat)
            pending.append(pool.apply_async(extend_task, ({
                'run_id': rid, 'shape': shape, 'cond': cond, 'strategy': strat,
                'method': method, 'init_idx': idx, 'init_curve': run['init_curve'],
                'phi': phi, 'phi_k_truth': phi_k, 'nxi': run['nxi'],
                'stored_n_iters': stored_n, 'stored_cps': run['cps'],
                'stored_E_total': run['E_total'], 'total_iters': args.total_iters,
                'refiner': refiner,
                'seed': rsc.task_seed(shape, cond, strat, method, idx)},)))
            while len(pending) > max_pending:
                drain(block=False)
                if len(pending) > max_pending:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        print("[extend] interrupted -- storing finished runs, dropping pending ones.",
              flush=True)
        pool.terminate()
        pending.clear()
    finally:
        if pending:
            drain(block=True)
        conn.commit()
        pool.close()
        pool.join()
    print(f"[extend] done: {n_done} runs extended, {n_fail} failed, "
          f"{(time.time() - t0) / 60:.1f} min; worst prefix deviation "
          f"{worst_prefix:.3e} (0.0 = exact continuation) -> {db_path}", flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    mp.freeze_support()
    main()
