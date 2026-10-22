"""Merge several smoothness-comparison DBs (parallel partial runs) into one.

Normally each source DB holds a DISJOINT set of variants for the same shapes (one process
for the GPU-based variants, one per CPU-bound sun variant, as `run_smoothness_comparison_v2.py`
writes a fresh DB containing only the `--variants` it was given) -- a plain "insert every row
from every source, in order" merge is then safe no matter the order.

`continue_smoothness_comparison.py` breaks that assumption: each of its per-process output DBs
is a full COPY of the source run (every variant), with only its own `--variants` subset actually
extended -- the rest are stale duplicates of the pre-continuation state. With overlapping
sources, pass `--src_variants` (one comma-separated variant list per source, in the same order
as `--src_tags`, `|`-separated between sources) to pull ONLY those variants from each source;
rows for any variant not named in any list are taken from the first source that has them, so
untouched variants (e.g. `cfm_only`, never touched by `continue_smoothness_comparison.py`) still
come through once. Omit `--src_variants` for the disjoint case (previous behaviour, unchanged).

The result is a regular `smoothness_comparison.db` that `plot_smoothness_comparison.py` reads.
Truths and bases are copied; the `config` meta entry of the first source is kept with its variant
list replaced by the union (ordered like `run_smoothness_comparison_v2.VARIANTS`).

Usage:
  python merge_smoothness_dbs.py --out_tag smoothness_comparison_v3_20261011 \
      --src_tags smoothness_comparison_v3_gpu_20261011,smoothness_comparison_v3_sunpm_20261011,...

  # overlapping sources (continue_smoothness_comparison.py's per-process copies):
  python merge_smoothness_dbs.py --out_tag smoothness_comparison_v6cont_20261012 \
      --src_tags smoothness_comparison_v6cont_gpu_20261012,smoothness_comparison_v6cont_sunpm_20261012,smoothness_comparison_v6cont_sunjerk_20261012 \
      --src_variants "cfm_bspline_svgd,linear_bspline_svgd,...|linear_sun_pointmass|linear_sun_jerk"
"""
import argparse
import json
import os
import sqlite3
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

RUN_COLS = ('shape, variant, cand_idx, init_param, n_iters, nxi, n_points, log_space, '
            'smoothness_weight, n_states, init_curve, states, smooth_series, '
            'path_len_series, created_at')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True)
    ap.add_argument('--src_tags', required=True, help='Comma-separated result folders to merge.')
    ap.add_argument('--src_variants', default=None,
                    help="Per-source variant filter for overlapping sources: one comma-separated "
                         "variant list per --src_tags entry, '|'-separated between sources. "
                         "Rows for variants not named in any list come from the first source "
                         "that has them. Omit for disjoint sources (default: every row kept).")
    args = ap.parse_args()

    import smoothness_db as sdb
    from run_smoothness_comparison_v2 import VARIANTS

    tags = args.src_tags.split(',')
    filters = (args.src_variants.split('|') if args.src_variants
              else [None] * len(tags))
    if len(filters) != len(tags):
        raise ValueError(f"--src_variants has {len(filters)} groups, --src_tags has {len(tags)}")
    seen_variants = set()

    out_path = os.path.join(_here, 'results', args.out_tag, 'smoothness_comparison.db')
    conn = sdb.open_db(out_path)
    first_cfg = None
    for tag, vfilter in zip(tags, filters):
        src = os.path.join(_here, 'results', tag, 'smoothness_comparison.db')
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        conn.execute("ATTACH DATABASE ? AS src", (src,))
        if vfilter is None:
            conn.execute(f"INSERT OR REPLACE INTO runs ({RUN_COLS}) SELECT {RUN_COLS} FROM src.runs")
        else:
            wanted = [v for v in vfilter.split(',') if v]
            # also pull any variant this source has that no source (including this filter) has
            # claimed yet, so untouched variants in an overlapping copy still come through once.
            row = conn.execute("SELECT value FROM src.meta WHERE key='config'").fetchone()
            src_variants = json.loads(row[0])['variants'] if row else []
            take = set(wanted) | (set(src_variants) - seen_variants)
            marks = ','.join('?' * len(take))
            conn.execute(f"INSERT OR REPLACE INTO runs ({RUN_COLS}) SELECT {RUN_COLS} FROM "
                        f"src.runs WHERE variant IN ({marks})", list(take))
            seen_variants |= take
        conn.execute("INSERT OR REPLACE INTO truths SELECT * FROM src.truths")
        conn.execute("INSERT OR IGNORE INTO basis SELECT * FROM src.basis")
        row = conn.execute("SELECT value FROM src.meta WHERE key='config'").fetchone()
        if first_cfg is None and row:
            first_cfg = json.loads(row[0])
        conn.commit()
        conn.execute("DETACH DATABASE src")
        print(f"[merge] {tag}: merged")
    present = {r[0] for r in conn.execute("SELECT DISTINCT variant FROM runs")}
    if first_cfg is not None:
        first_cfg['variants'] = [v for v in VARIANTS if v in present]
        sdb.set_meta(conn, 'config', first_cfg)
        conn.commit()
    for variant, n in conn.execute("SELECT variant, COUNT(*) FROM runs GROUP BY variant ORDER BY variant"):
        print(f"[merge] {n:5d} runs of {variant}")
    conn.close()
    print(f"[merge] -> {out_path}")


if __name__ == '__main__':
    main()
