r"""
test_mission_eval.py -- self-test of the replanning-mission pipeline
=====================================================================
Needs the GPU for part 7 (real CFM checkpoint); everything else runs on CPU.
~3-5 minutes.   Run:  python test_mission_eval.py   (exit code 0 = all passed)

 1. batched solvers == reference `SvgdRefiner` (logged control points), with and
    without the start pin; torch fp64 == numpy; mixed precision within tolerance
 2. state codec round trip
 3. `linear_ray_path` geometry
 4. DB round trip (rounds, candidates, states)
 5. mini mission, dry run (random planner): row counts, every round starts where
    the previous unit ended, every unit is one length unit long, resume is
    bit-identical to a straight-through run -- covers all four methods
    (cfm, selfsup, random_walk, linear)
 6. figures / tables of `plot_mission_eval.py` are produced from that run
 7. mini mission with the REAL start-conditioned CFM network (needs GPU)
 8. mini mission with the REAL self-supervised checkpoint, on CPU: despite
    having no start conditioning (unlike cfm), the start-point force applied
    during SVGD refinement still pulls its candidates to the agent position
    (small start_gap after refinement)
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import subprocess
import sys
import tempfile
import time

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, os.path.join(_arch, 'exploration'), _arch,
           os.path.join(_arch, 'ergodic_dataset_generator'),
           os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

import mission_db as mdb
import run_svgd_convergence as rsc
import svgd_batched as sb
from common.svgd_refine import SvgdRefiner
from init_baselines import linear_ray_path, random_walk_path
from state_codec import pack_states, unpack_states

RUNNER = os.path.join(_here, 'run_mission_eval.py')
PLOTTER = os.path.join(_here, 'plot_mission_eval.py')


def run_cli(script, *args):
    t = time.time()
    r = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise AssertionError(f"{os.path.basename(script)} failed with exit code {r.returncode}")
    return r.stdout, time.time() - t


def test_solvers():
    R = 64
    xs = np.linspace(0, 1, R)
    X, Y = np.meshgrid(xs, xs)
    phi = np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / 0.02) + 0.5 * np.exp(
        -((X - 0.75) ** 2 + (Y - 0.3) ** 2) / 0.01)
    phi /= phi.max()
    B = rsc.basis_matrix()
    phik = SvgdRefiner(0, backend='tsvec')._phi_k(phi)
    C, n = 4, 120
    inits = np.stack([random_walk_path(128, seed=s).numpy().astype(np.float64) for s in range(C)])
    seeds = [11, 12, 13, 14]
    start = np.array([0.5, 0.5])
    for st in (None, start):
        np_out = sb.BatchedSvgd(B).run(inits, phik, st, seeds, n)
        worst = 0.0
        for c in range(C):
            log = []
            SvgdRefiner(seed=seeds[c], backend='tsvec').refine(inits[c], phi, n, nxi=25, start=st,
                                                               trajectory_log=log)
            worst = max(worst, float(np.abs(np.stack(log) - np_out['cps'][c]).max()))
        assert worst < 1e-6, f"numpy batched vs reference: {worst}"
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        for label, kw, tol in (('fp64', {}, 1e-6), ('mixed', dict(compute_dtype=torch.float32), 5e-4)):
            t_out = sb.BatchedSvgdTorch(B, dev, torch.float64, **kw).run(inits, phik, st, seeds, n)
            d = float(np.abs(t_out['cps'].cpu().numpy() - np_out['cps']).max())
            assert d < tol, f"torch {label} vs numpy batched: {d}"
        print(f"ok  solvers vs reference (start pin {'on' if st is not None else 'off'}): "
              f"numpy-vs-ref max|d|={worst:.1e}, torch ({dev}) fp64/mixed within tolerance")
    # the pin really pulls: with start the first control point sits at the start
    pinned = sb.BatchedSvgd(B).run(inits, phik, start, seeds, 400)['final_cps'][:, 0]
    free = sb.BatchedSvgd(B).run(inits, phik, None, seeds, 400)['final_cps'][:, 0]
    assert np.abs(pinned - start).max() < 0.02 < np.abs(free - start).max(), (pinned, free)
    print("ok  start pin: first control point within 0.02 of the start (free run drifts away)")


def test_sun_solver():
    """Batched Sun refiner (`BatchedSunTorch`) == single `SvgdRefiner(backend='sun')`,
    and the start point is the initial state of the dynamics."""
    R = 64
    xs = np.linspace(0, 1, R)
    X, Y = np.meshgrid(xs, xs)
    phi = np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / 0.02) + 0.5 * np.exp(
        -((X - 0.75) ** 2 + (Y - 0.3) ** 2) / 0.01)
    phi /= phi.max()
    B = rsc.basis_matrix()
    C, n = 3, 60
    inits = np.stack([random_walk_path(128, seed=s).numpy().astype(np.float64) for s in range(C)])
    start = np.array([0.5, 0.5])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    for st in (None, start):
        out = sb.BatchedSunTorch(B, dev).run(inits, np.repeat(phi[None], C, axis=0), st,
                                             [0] * C, n)
        cps = out['cps'].cpu().numpy()
        assert cps.shape == (C, n + 1, 25, 2), cps.shape
        worst = 0.0
        for c in range(C):
            log = []
            SvgdRefiner(seed=0, backend='sun').refine(inits[c], phi, n, nxi=25, start=st,
                                                      trajectory_log=log)
            worst = max(worst, float(np.abs(np.stack(log) - cps[c]).max()))
        assert worst < 1e-5, f"batched Sun vs single Sun: {worst}"
        assert not np.allclose(cps[:, 0], cps[:, -1]), "Sun refiner did not move anything"
    assert np.abs(out['final_cps'][:, 0] - start).max() < 0.02, out['final_cps'][:, 0]
    print(f"ok  Sun refiner: batched == single (max|d|={worst:.1e}), start = initial state")


def test_codec():
    a = np.cumsum(np.random.default_rng(0).normal(scale=1e-3, size=(1501, 25, 2)), axis=0) + 0.5
    b = unpack_states(pack_states(a), 1501, 25)
    assert np.abs(a - b).max() < 2e-5
    a2 = np.random.default_rng(1).uniform(-0.4, 1.4, size=(5, 25, 2))     # jumps > int16 delta range
    assert np.abs(a2 - unpack_states(pack_states(a2), 5, 25)).max() < 2e-5
    print("ok  state codec round trip (incl. wide-delta fallback)")


def test_linear_ray():
    for ang in (0, 37, 90, 200, 315):
        c = linear_ray_path((0.5, 0.5), ang, 2.83, 128).numpy()
        assert np.allclose(c[0], 0.5, atol=1e-6)
        assert c.min() >= 0.04 - 1e-5 and c.max() <= 0.96 + 1e-5
        L = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
        assert abs(L - 2.83) < 0.05, L
        d = c[3] - c[0]
        assert abs(np.degrees(np.arctan2(d[1], d[0])) % 360 - ang % 360) < 2.0, ang
    c = linear_ray_path((0.02, 0.97), 100, 2.0, 64).numpy()               # start outside the margin box
    assert np.allclose(c[0], [0.02, 0.97], atol=1e-6) and np.isfinite(c).all()
    print("ok  linear_ray_path: starts at the position, stays in the box, length and heading right")


def test_db():
    with tempfile.TemporaryDirectory() as d:
        conn = mdb.open_db(os.path.join(d, 'x.db'))
        R = 64
        row = {k: 0.5 for k in mdb.ROUND_SCALARS}
        row.update(shape='S', knowledge_condition='none_known', strategy='ucb', method='cfm',
                   round=0, n_exec=1, selected_idx=1, reached99=0, n_obs=64,
                   target=np.random.rand(R, R), mu_plan=np.random.rand(R, R),
                   sd_plan=np.random.rand(R, R), segment=np.random.rand(72, 2),
                   obs_pts=np.random.rand(64, 2), obs_vals=np.random.rand(64))
        cps = np.random.rand(21, 25, 2).astype(np.float32)
        cands = [dict(cand_idx=i, selected=(i == 1), init_param=float(i), n_iters=20, nxi=25,
                      n_states=21, state_stride=1, E_stride=10, E_init=1.0, E_final=0.5,
                      init_curve=np.random.rand(128, 2), states=pack_states(cps),
                      E_series=np.array([3, 2, 1.0])) for i in range(3)]
        mdb.save_round(conn, row, cands)
        (back,) = list(mdb.iter_rounds(conn, columns=('shape', 'round', 'segment', 'target', 'mu_plan')))
        assert back['segment'].shape == (72, 2) and back['target'].shape == (R, R)
        cs = list(mdb.iter_candidates(conn, with_states=True))
        assert len(cs) == 3 and cs[0]['cps'].shape == (21, 25, 2)
        assert np.abs(cs[0]['cps'] - cps).max() < 2e-5 and cs[1]['selected'] == 1
        assert mdb.round_numbers(conn) == {'S': [0]}
        conn.close()
    print("ok  DB round trip (rounds, candidates, states)")


def check_mission_db(root, cond, strat, method, shapes, n_rounds):
    conn = mdb.open_db(mdb.shard_path(root, cond, strat, method))
    rows = list(mdb.iter_rounds(conn, columns=('shape', 'round', 'start_x', 'start_y', 'segment',
                                               'seg_len', 'start_gap', 'swept_mass', 'E_truth',
                                               'info_gain', 'reached99')))
    assert len(rows) == len(shapes) * n_rounds, (len(rows), len(shapes), n_rounds)
    for sh in shapes:
        rr = [r for r in rows if r['shape'] == sh]
        assert [r['round'] for r in rr] == list(range(n_rounds))
        for a, b in zip(rr[:-1], rr[1:]):
            end = a['segment'][-1]
            assert np.allclose([b['start_x'], b['start_y']], end, atol=1e-5), "round does not start where the unit ended"
        assert all(1.34 < r['seg_len'] <= 1.4143 for r in rr), [r['seg_len'] for r in rr]
        assert all(r['start_gap'] < 0.1 for r in rr)
        sm = [r['swept_mass'] for r in rr]
        assert all(y >= x - 1e-9 for x, y in zip(sm, sm[1:])), "swept mass must be non-decreasing"
        assert np.isfinite([r['E_truth'] for r in rr]).all() and np.isfinite([r['info_gain'] for r in rr]).all()
    cs = list(mdb.iter_candidates(conn, with_states=False, shape=shapes[0], round=0))
    conn.close()
    assert len(cs) > 1 and sum(c['selected'] for c in cs) == 1
    return rows


def test_mission_dry(tmp):
    common = ['--dry_run', '--refiner', 'tsvec', '--shapes', 'A,digit_5', '--conditions', 'none_known,half_known',
              '--strategies', 'ucb', '--n_init', '3', '--n_iters', '30', '--workers', '3',
              '--parallel_sets', '3', '--out_root', tmp]
    run_cli(RUNNER, *common, '--max_rounds', '4', '--out_tag', 'full')
    run_cli(RUNNER, *common, '--max_rounds', '2', '--out_tag', 'res')
    out, _ = run_cli(RUNNER, *common, '--max_rounds', '4', '--out_tag', 'res')
    assert 'resumed' in out
    for cond in ('none_known', 'half_known'):
        for m in ('cfm', 'selfsup', 'random_walk', 'linear'):
            check_mission_db(os.path.join(tmp, 'full'), cond, 'ucb', m, ['A', 'digit_5'], 4)
            a = check_mission_db(os.path.join(tmp, 'res'), cond, 'ucb', m, ['A', 'digit_5'], 4)
            b = check_mission_db(os.path.join(tmp, 'full'), cond, 'ucb', m, ['A', 'digit_5'], 4)
            for x, y in zip(a, b):
                for k in ('swept_mass', 'E_truth', 'info_gain', 'seg_len'):
                    assert x[k] == y[k], (cond, m, k, x[k], y[k])
    print("ok  dry-run mission: 8 sets (2 conditions x 4 methods) x 2 shapes x 4 rounds; "
          "units start where the last ended, "
          "one length unit each; resumed run == straight-through run (bit-identical)")


def test_mission_dry_sun(tmp):
    """The default refiner (Sun) end to end; results land in <out_tag>_sun."""
    run_cli(RUNNER, '--dry_run', '--refiner', 'sun', '--shapes', 'A', '--conditions', 'half_known',
            '--strategies', 'ucb', '--methods', 'random_walk', '--n_init', '3', '--n_iters', '30',
            '--workers', '2', '--parallel_sets', '1', '--max_rounds', '2', '--out_root', tmp,
            '--out_tag', 'sun')
    assert not os.path.exists(os.path.join(tmp, 'sun'))
    rows = check_mission_db(os.path.join(tmp, 'sun_sun'), 'half_known', 'ucb', 'random_walk', ['A'], 2)
    assert max(r['start_gap'] for r in rows) < 0.05, [r['start_gap'] for r in rows]
    print("ok  dry-run mission with the Sun refiner: 2 rounds stored under <out_tag>_sun")


def test_plots(tmp):
    out, dt = run_cli(PLOTTER, '--out_root', tmp, '--out_tag', 'full', '--paths', '--svgd_rounds', '0,2')
    plots = os.path.join(tmp, 'full', 'plots')
    for f in ('overview_E_truth.png', 'overview_J_target.png', 'overview_swept_mass.png',
              'overview_info_gain.png', 'executions_to_99_cdf.png', 'executions_to_99_box.png',
              'svgd_convergence_round1.png'):
        assert os.path.getsize(os.path.join(plots, f)) > 5000, f
    for f in ('missions.csv', 'summary.csv'):
        assert os.path.getsize(os.path.join(tmp, 'full', 'tables', f)) > 100, f
    # 2 conditions x 1 strategy x 4 methods (cfm, selfsup, random_walk, linear)
    assert len(os.listdir(os.path.join(plots, 'paths'))) == 8
    print(f"ok  plots and tables written ({dt:.0f} s)")


def test_mission_real(tmp):
    out, dt = run_cli(RUNNER, '--refiner', 'tsvec', '--shapes', 'A,rand_gmm_10', '--conditions', 'half_known',
                      '--strategies', 'eid', '--methods', 'cfm,random_walk', '--n_init', '6',
                      '--n_iters', '100', '--max_rounds', '3', '--workers', '3',
                      '--parallel_sets', '2', '--out_root', tmp, '--out_tag', 'real')
    for m in ('cfm', 'random_walk'):
        rows = check_mission_db(os.path.join(tmp, 'real'), 'half_known', 'eid', m, ['A', 'rand_gmm_10'], 3)
        assert max(r['start_gap'] for r in rows) < 0.05, [r['start_gap'] for r in rows]
    conn = mdb.open_db(mdb.shard_path(os.path.join(tmp, 'real'), 'half_known', 'eid', 'cfm'))
    cs = list(mdb.iter_candidates(conn, shape='A', round=1))
    # the start-conditioned network begins its plan at the previous end point
    rows = {r['round']: r for r in mdb.iter_rounds(conn, columns=('shape', 'round', 'start_x', 'start_y'),
                                                  shape='A')}
    s1 = np.array([rows[1]['start_x'], rows[1]['start_y']])
    conn.close()
    assert all(np.abs(c['init_curve'][0] - s1).max() < 1e-3 for c in cs), "CFM plan does not begin at the start"
    print(f"ok  mission with the real CFM checkpoint ({dt:.0f} s): plans begin at the previous end point, "
          f"start pin satisfied, 3 rounds stored")


def test_mission_real_selfsup(tmp):
    """Real self-supervised checkpoint, CPU (cheap: a single forward pass, no
    ODE integration) -- unlike cfm it has no start conditioning
    (`SelfsupPlanner.start_cond == False`), so this specifically checks that
    the start-point force applied during SVGD refinement (the same one every
    method relies on) still pulls its candidates to the agent position.

    `n_iters=500` (not the usual 100 of `test_mission_real`'s quick CFM check):
    measured on this exact setup, selfsup's start_gap was still ~0.17-0.38 at
    100 iterations (vs. cfm's ~0, since cfm already begins exactly at the
    start by construction) and down to ~0.01-0.03 at 500 -- production runs
    use 1500/3000 iterations, so this is a lower bound on how well it
    converges there, not the real-run number."""
    out, dt = run_cli(RUNNER, '--refiner', 'tsvec', '--shapes', 'A,rand_gmm_10', '--conditions', 'half_known',
                      '--strategies', 'eid', '--methods', 'selfsup', '--n_init', '6',
                      '--n_iters', '500', '--max_rounds', '3', '--workers', '3',
                      '--parallel_sets', '2', '--device', 'cpu', '--out_root', tmp, '--out_tag', 'real_selfsup')
    rows = check_mission_db(os.path.join(tmp, 'real_selfsup'), 'half_known', 'eid', 'selfsup',
                            ['A', 'rand_gmm_10'], 3)
    print(f"ok  mission with the real self-supervised checkpoint ({dt:.0f} s, CPU): no start "
          f"conditioning in the network itself, but the refiner's start-point force still pins "
          f"it within check_mission_db's start_gap < 0.1, 3 rounds stored "
          f"(worst start_gap: {max(r['start_gap'] for r in rows):.3f})")


def main():
    torch.set_num_threads(1)
    t0 = time.time()
    test_solvers()
    test_sun_solver()
    test_codec()
    test_linear_ray()
    test_db()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        test_mission_dry(tmp)
        test_mission_dry_sun(tmp)
        test_plots(tmp)
        test_mission_real_selfsup(tmp)
        if torch.cuda.is_available():
            test_mission_real(tmp)
        else:
            print("skip real-CFM-checkpoint mission test (no GPU)")
    print(f"ALL OK ({time.time() - t0:.0f} s)")


if __name__ == '__main__':
    main()
