r"""
run_all_tests.py -- every correctness check of the ergodic solvers and the smoothness pipeline
=============================================================================================
  1. ergodic_validation/test_ergodicity_metrics.py   the independent ergodicity metrics
  2. ergodic_validation/reference_sun_check.py       our Sun port == official Sun et al. tutorial
  3. exploration/common/test_tsvec_svgd.py           TSVEC: Jacobian (finite differences), kernel,
                                                     descent, temperature regression
  4. exploration/common/test_adam_svgd.py            Adam-SVGD: Gaussian sampling, repulsion and
                                                     energy gradient vs. autograd, no collapse
  5. exploration/common/test_sun_sinkhorn.py         Sinkhorn flow: gradient vs. autodiff, warm
                                                     start, mass fractions, GPU twin == JAX
  6. evaluation_full_matrix/test_smoothness_comparison_v2.py   pipeline end-to-end (dry run + plots)
The solver bench (validate_solvers.py) is separate because it takes ~30-60 min.
Run: python run_all_tests.py   (exit 0 = all passed)
"""
import os
import subprocess
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
TESTS = [
    os.path.join(_here, 'test_ergodicity_metrics.py'),
    os.path.join(_here, 'reference_sun_check.py'),
    os.path.join(_arch, 'exploration', 'common', 'test_tsvec_svgd.py'),
    os.path.join(_arch, 'exploration', 'common', 'test_adam_svgd.py'),
    os.path.join(_arch, 'exploration', 'common', 'test_sun_sinkhorn.py'),
    os.path.join(_arch, 'evaluation_full_matrix', 'test_smoothness_comparison_v2.py'),
]


def main():
    failed = []
    for t in TESTS:
        t0 = time.time()
        r = subprocess.run([sys.executable, '-u', t], cwd=os.path.dirname(t), capture_output=True, text=True)
        ok = r.returncode == 0
        lines = [l for l in r.stdout.splitlines() if l.startswith('[') or 'PASSED' in l or 'OK' in l]
        print(f"{'PASS' if ok else 'FAIL'}  {os.path.relpath(t, _arch)}  ({time.time() - t0:.0f}s)")
        for l in lines:
            print('      ' + l)
        if not ok:
            failed.append(t)
            print(r.stdout[-2000:])
            print(r.stderr[-3000:])
    print(f"\n{len(TESTS) - len(failed)}/{len(TESTS)} test files passed")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
