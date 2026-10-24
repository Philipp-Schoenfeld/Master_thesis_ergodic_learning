"""Seed a new smoothness-comparison DB with already-computed, still valid cells.

Copies `--src` to `--dst` (only if `--dst` does not exist yet, so a resumed job never
overwrites its own progress), then deletes every run whose variant is not in `--keep`
or whose `n_iters` differs from `--n_iters`. `run_smoothness_comparison_v2.py` then
skips the kept (shape, variant, candidate) cells and only computes the missing ones.

Example (round 3):
  python seed_smoothness_db.py --src results/smoothness_comparison_v2_20261009/smoothness_comparison.db \
      --dst results/smoothness_comparison_v3_20261011/smoothness_comparison.db \
      --keep cfm_only,linear_sun_pointmass,linear_sun_jerk --n_iters 2000
"""
import argparse
import os
import shutil
import sqlite3
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--src', required=True, help='Existing DB with valid cells.')
    ap.add_argument('--dst', required=True, help='New DB (only created if missing).')
    ap.add_argument('--keep', required=True, help='Comma-separated variants to keep.')
    ap.add_argument('--n_iters', type=int, required=True,
                    help='Keep only runs computed with exactly this many iterations '
                         '(cfm_only runs have n_iters = 0 and are always kept).')
    args = ap.parse_args()

    if os.path.exists(args.dst):
        print(f"[seed] {args.dst} exists -> resuming, nothing copied.")
        return 0
    if not os.path.exists(args.src):
        print(f"[seed] source {args.src} not found -> every cell will be computed from scratch.")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    shutil.copyfile(args.src, args.dst)
    keep = [v for v in args.keep.split(',') if v]
    conn = sqlite3.connect(args.dst)
    marks = ','.join('?' * len(keep))
    n0 = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.execute(f"DELETE FROM runs WHERE variant NOT IN ({marks}) "
                 f"OR (n_iters != ? AND variant != 'cfm_only')", keep + [args.n_iters])
    conn.commit()
    n1 = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.execute("VACUUM")
    for variant, n in conn.execute("SELECT variant, COUNT(*) FROM runs GROUP BY variant ORDER BY variant"):
        print(f"[seed] kept {n:5d} runs of {variant}")
    print(f"[seed] {n1}/{n0} runs kept -> {args.dst}")
    conn.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
