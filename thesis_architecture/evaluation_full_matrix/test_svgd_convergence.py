r"""
test_svgd_convergence.py -- self-test for the SVGD-convergence pipeline.
CPU only, no checkpoint, ~1 minute.  Run:  python test_svgd_convergence.py
Checks (1) the trajectory log hook leaves `SvgdRefiner.refine` unchanged and
its last state equals the returned curve, (2) the batched metric equals
`ExploreExploitErgodic.score`, (3) DB round trip, (4) the angled line inits,
(7) an end-to-end `--dry_run` CLI pass of `run_svgd_convergence.py` with all
four methods (cfm, selfsup, random_walk, linear) via `DummyPlanner` stand-ins
for the two network methods -- no checkpoint needed, exercises the same
NETWORK_METHODS / shared-baseline dispatch the real run uses.
"""
import os
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

import run_svgd_convergence as rsc
import svgd_convergence_db as sdb
from common.svgd_refine import SvgdRefiner
from init_baselines import linear_angle_path, random_walk_path
from metrics_explore_exploit import ExploreExploitErgodic

RUNNER = os.path.join(_here, 'run_svgd_convergence.py')


def run_cli(*args):
    t = time.time()
    r = subprocess.run([sys.executable, RUNNER, *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise AssertionError(f"run_svgd_convergence.py failed with exit code {r.returncode}")
    return r.stdout, time.time() - t


def test_cli_dry_run():
    """End-to-end `--dry_run` pass through `main()`: both NETWORK_METHODS
    (cfm, selfsup) get a `DummyPlanner` stand-in (no checkpoint needed), so
    this exercises the exact method-dispatch code real runs use, including
    the random_walk/linear SHARED-baseline path, without needing a GPU."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        out_tag = 'selftest_dry'
        out, dt = run_cli('--dry_run', '--shapes', 'A', '--conditions', 'none_known,half_known',
                          '--strategies', 'ucb', '--n_init', '2', '--n_iters', '5', '--workers', '2',
                          '--out_tag', out_tag)
        assert 'done:' in out
        import run_svgd_convergence as rsc2
        import svgd_convergence_db as sdb2
        out_dir = os.path.join(_here, 'results', out_tag + '_sun')
        try:
            conn = sdb2.open_db(os.path.join(out_dir, 'svgd_convergence.db'))
            methods_seen = {r[0] for r in conn.execute("SELECT DISTINCT method FROM runs")}
            assert methods_seen == set(rsc2.METHODS), methods_seen
            # cfm/selfsup: once per (cond, strat) = 2x1 blocks x 2 inits; random_walk/linear: SHARED, once per shape
            for m in rsc2.NETWORK_METHODS:
                n = conn.execute("SELECT COUNT(*) FROM runs WHERE method=?", (m,)).fetchone()[0]
                assert n == 2 * 2, (m, n)
            for m in ('random_walk', 'linear'):
                n = conn.execute("SELECT COUNT(*) FROM runs WHERE method=? AND knowledge_condition=?",
                                 (m, sdb2.SHARED)).fetchone()[0]
                assert n == 2, (m, n)
            conn.close()
        finally:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
    print(f"ok  --dry_run CLI end-to-end ({dt:.0f} s): all four methods stored, "
          f"cfm/selfsup per (condition, strategy), random_walk/linear shared once per shape")


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    R = 64
    xs = np.linspace(0, 1, R)
    X, Y = np.meshgrid(xs, xs)
    phi = np.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / 0.02) + 0.5 * np.exp(
        -((X - 0.75) ** 2 + (Y - 0.3) ** 2) / 0.01)
    phi = phi / phi.max()
    init = random_walk_path(128, seed=3).numpy().astype(np.float64)
    n = 15

    # (1) hook is read-only and consistent with the returned curve
    out_plain = SvgdRefiner(seed=7).refine(init, phi, n, nxi=25)
    log = []
    out_log = SvgdRefiner(seed=7).refine(init, phi, n, nxi=25, trajectory_log=log)
    assert np.array_equal(out_plain, out_log), "trajectory_log changed the result"
    assert len(log) == n + 1, len(log)
    B = rsc.basis_matrix().astype(np.float64)
    assert np.allclose(B @ log[-1], out_log, atol=1e-5), "last logged state != returned curve"
    assert not np.allclose(log[0], log[-1]), "SVGD did not move anything"
    print("ok  trajectory_log: result unchanged, n+1 states, last == returned curve")

    # (2) batched metric == reference scorer
    ee = ExploreExploitErgodic(device='cpu')
    truth = torch.as_tensor(phi, dtype=torch.float32)
    phi_k = ee.target_coeffs(truth)
    curves = np.einsum('pi,sid->spd', B, np.stack(log)).astype(np.float32)
    m = rsc.score_states(ee, curves, phi_k.numpy())
    for s in (0, 7, n):
        ref = ee.score(torch.as_tensor(curves[s]), phi_k)
        for key, mk in (('E_total', 'E_ergodic_total'), ('E_explore', 'E_ergodic_explore'),
                        ('E_exploit', 'E_ergodic_exploit')):
            assert abs(m[key][s] - ref[mk]) <= 1e-4 * max(1.0, abs(ref[mk])), (key, s)
    print("ok  score_states == ExploreExploitErgodic.score")

    # (3) DB round trip
    with tempfile.TemporaryDirectory() as d:
        conn = sdb.open_db(os.path.join(d, 't.db'))
        sdb.save_basis(conn, rsc.basis_matrix(), 25, 128, 5)
        sdb.save_truth(conn, 'S', phi)
        sdb.save_target(conn, 'S', 'none_known', 'ucb', phi)
        row = dict(shape='S', knowledge_condition='none_known', strategy='ucb',
                   method='random_walk', init_idx=3, init_param=3.0, n_iters=n, nxi=25,
                   init_curve=init.astype(np.float32), E_raw_init=1.5,
                   cps=np.stack(log), **m)
        sdb.save_run(conn, row, commit=True)
        assert sdb.existing_keys(conn, 'S', 'none_known', 'ucb') == {('random_walk', 3)}
        (back,) = list(sdb.iter_runs(conn, columns=('cps', 'E_total', 'path_len')))
        assert back['cps'].shape == (n + 1, 25, 2) and back['E_total'].shape == (n + 1,)
        assert np.allclose(back['cps'], np.stack(log), atol=1e-6)
        assert np.allclose(sdb.render_states(back['cps'], sdb.load_basis(conn, 25, 128)),
                           curves, atol=1e-5)
        assert sdb.load_truth(conn, 'S').shape == (R, R)
        assert sdb.load_target(conn, 'S', 'none_known', 'ucb').shape == (R, R)
        conn.close()
    print("ok  DB round trip (states re-render to the same curves)")

    # (4) line inits
    for a in np.linspace(0, 180, 30, endpoint=False):
        c = linear_angle_path(a).numpy()
        assert c.min() >= 0.04 - 1e-5 and c.max() <= 0.96 + 1e-5
        assert np.allclose(c.mean(axis=0), 0.5, atol=1e-4)
    print("ok  linear_angle_path stays inside the margin box, centred")
    # (5) extension table round trip + iter_runs(include_ext=True)
    with tempfile.TemporaryDirectory() as d:
        conn = sdb.open_db(os.path.join(d, 't.db'))
        row = dict(shape='S', knowledge_condition='ground_truth', strategy='ucb',
                   method='cfm', init_idx=0, init_param=None, n_iters=n, nxi=25,
                   init_curve=init.astype(np.float32), E_raw_init=1.5,
                   cps=np.stack(log), **m)
        sdb.save_run(conn, row, commit=True)
        rid = conn.execute("SELECT id FROM runs").fetchone()[0]
        n_ext = 10
        ext = dict(cps=np.random.rand(n_ext, 25, 2), E_total=np.arange(n_ext) + 100.0,
                   E_explore=np.zeros(n_ext), E_exploit=np.zeros(n_ext),
                   path_len=np.ones(n_ext))
        sdb.save_run_ext(conn, rid, n + 1, n + n_ext, prefix_max_abs_diff=0.0,
                         prefix_E_max_rel_diff=0.0, commit=True, **ext)
        assert sdb.ext_run_ids(conn) == {rid}
        (plain,) = list(sdb.iter_runs(conn, columns=('E_total', 'cps')))
        (full,) = list(sdb.iter_runs(conn, columns=('E_total', 'cps'), include_ext=True))
        assert plain['E_total'].shape == (n + 1,) and plain['n_iters'] == n
        assert full['E_total'].shape == (n + n_ext + 1,) and full['n_iters'] == n + n_ext
        assert full['cps'].shape == (n + n_ext + 1, 25, 2) and full['n_iters_stored'] == n
        assert np.allclose(full['E_total'][:n + 1], plain['E_total'])
        assert np.allclose(full['E_total'][n + 1:], ext['E_total'])
        conn.close()
    print("ok  runs_ext round trip: stored + extension states concatenate to n_iters_total+1")
    # (6) extra metrics: batched kernels vs reference implementations
    import plot_svgd_extra_metrics as pem
    from metrics_explore_exploit import swept_mass_fraction
    truth_t = torch.as_tensor(phi, dtype=torch.float32)
    cur_t = torch.as_tensor(curves[:6])
    cells = pem.grid_cells(R, 'cpu')
    w = truth_t.reshape(-1).clamp(min=0.0)
    for radius in (0.03, 0.06):
        got = pem.swept_mass_batch(cur_t, cells, w, radius, chunk=4)
        ref = torch.tensor([swept_mass_fraction(c, truth_t, sensor_radius=radius) for c in cur_t])
        assert torch.allclose(got, ref, atol=1e-6), (radius, got, ref)
    a = torch.rand(3, 20, 2)
    b = a + torch.tensor([0.1, 0.0])
    assert torch.allclose(pem.chamfer(a, a), torch.zeros(3), atol=1e-6)
    assert torch.allclose(pem.chamfer(a, b), pem.chamfer(b, a), atol=1e-6)
    assert (pem.chamfer(a, b) <= 0.1 + 1e-6).all()            # a shift by 0.1 moves nothing further
    x = torch.rand(5, 3, 16, 2)                                # 5 runs, 3 iterations
    div = pem.pairwise_diversity(x, chunk=2)
    manual = torch.stack([torch.stack([pem.chamfer(x[i, g], x[j, g]) for i in range(5)
                                       for j in range(5) if i != j]).mean() for g in range(3)])
    assert torch.allclose(div, manual, atol=1e-6)
    assert torch.allclose(pem.pairwise_diversity(x[:1].repeat(4, 1, 1, 1)), torch.zeros(3), atol=1e-6)
    E = np.array([[5.0, 3.0, 1.0, 0.5], [5.0, 4.0, 3.9, 3.8]])
    fp = pem.first_passage(E, np.array([1.0, 1.0]))
    assert fp[0] == 2 and np.isnan(fp[1])
    assert float(pem.first_passage(np.array([2.0, 0.1]), 0.5)) == 1.0
    g = pem.iteration_grid(3000)
    assert g[0] == 0 and g[-1] == 3000 and len(g) == 79, len(g)
    print("ok  extra metrics: swept mass == reference, Chamfer/diversity sanity, first passage, grid")
    # (7) end-to-end --dry_run CLI, all four methods
    test_cli_dry_run()
    print("ALL OK")


if __name__ == '__main__':
    main()
