r"""
run_teacher_study.py -- run a grid of teacher-trajectory configurations over holdout shapes
============================================================================================
Writes, per output tag (`results/<out_tag>/`):

  metrics.csv           one row per (shape, config, stage, candidate): all stage metrics
  finals/<shape>__<cfg>.npz   waypoints ('wp') and control points ('bs*') per stage
  config.json           the grid and the settings of the run
  done.txt              finished (shape, config) cells -- a re-run resumes where it stopped

The reference row `teacher_db` (the trajectory stored in the DB, resampled and projected the same
way) is written for every shape. Nothing here starts automatically: the full grids take hours.

Examples
--------
    # seconds, no checkpoint: exercises every code path
    python run_teacher_study.py --grid smoke --shapes A --n_init 4 --out_tag smoke --device cpu

    # pilot (about 40-60 min on the local RTX 2070 SUPER) -- ask before starting
    python run_teacher_study.py --grid pilot --out_tag pilot_20261024
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import pipeline as pl                                                      # noqa: E402
from configs import GRIDS, PILOT_SHAPES                                   # noqa: E402

RESULTS = os.path.join(_here, 'results')
KEY_COLS = ('shape', 'config', 'solver', 'stage', 'cand')


def _write_rows(writer, shape, cfg_name, solver, stages):
    n = 0
    for stage, m in stages.items():
        C = len(next(iter(m.values())))
        for c in range(C):
            writer.writerow({'shape': shape, 'config': cfg_name, 'solver': solver, 'stage': stage, 'cand': c,
                             **{k: float(v[c]) for k, v in m.items()}})
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out_tag', required=True, help='results go to teacher_study/results/<out_tag>/')
    ap.add_argument('--grid', default='pilot', choices=sorted(GRIDS), help='named configuration grid')
    ap.add_argument('--shapes', default=None,
                    help='comma-separated holdout shape names (default: pilot shapes for grid pilot, '
                         'all 25 holdout shapes otherwise)')
    ap.add_argument('--split', default='val', help="DB split the shapes come from ('val' = the 25 holdout shapes, "
                                                   "'train' = the training shapes)")
    ap.add_argument('--n_shapes', type=int, default=None,
                    help='random subset of that many shapes of the split (seeded by --shape_seed)')
    ap.add_argument('--shape_seed', type=int, default=0)
    ap.add_argument('--configs', default=None, help='comma-separated config names to run (default: all)')
    ap.add_argument('--n_init', type=int, default=30, help='candidates per cell (swarm size)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--time_budget_h', type=float, default=None, help='stop submitting cells after this time')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    grid = GRIDS[args.grid]()
    if args.configs:
        want = [c.strip() for c in args.configs.split(',') if c.strip()]
        unknown = set(want) - {c.name for c in grid}
        if unknown:
            raise KeyError(f'unknown configs {sorted(unknown)}; known: {[c.name for c in grid]}')
        grid = [c for c in grid if c.name in want]
    if args.shapes:
        labels = [s.strip() for s in args.shapes.split(',') if s.strip()]
    elif args.grid == 'pilot':
        labels = list(PILOT_SHAPES)
    else:
        labels = None
    if args.n_shapes is not None and labels is None:
        names = pl.list_shape_names(args.split)
        pick = np.random.default_rng(args.shape_seed).choice(len(names), size=min(args.n_shapes, len(names)), replace=False)
        labels = [names[i] for i in sorted(pick)]
    shapes = pl.load_shapes(labels, split=args.split)
    print(f'[teacher] {len(shapes)} shapes, {len(grid)} configs, n_init={args.n_init}, device={args.device}', flush=True)

    out = os.path.join(RESULTS, args.out_tag)
    os.makedirs(os.path.join(out, 'finals'), exist_ok=True)
    done_path = os.path.join(out, 'done.txt')
    done = set(open(done_path).read().split('\n')) if os.path.exists(done_path) else set()
    csv_path = os.path.join(out, 'metrics.csv')
    new_file = not os.path.exists(csv_path)
    json.dump(dict(grid=[c.to_dict() for c in grid], shapes=[s.name for s in shapes], n_init=args.n_init,
                   device=args.device, seed=args.seed, started=time.strftime('%Y-%m-%d %H:%M:%S')),
              open(os.path.join(out, 'config.json'), 'w'), indent=1)

    t0 = time.time()
    fields = None
    f = open(csv_path, 'a', newline='')
    writer = None
    try:
        for si, sd in enumerate(shapes):
            cells = [('teacher_db', 'db', None)] + [(c.name, c.solver, c) for c in grid]
            for name, solver, cfg in cells:
                cell_id = f'{sd.name}__{name}'
                if cell_id in done:
                    continue
                if args.time_budget_h is not None and (time.time() - t0) / 3600.0 > args.time_budget_h:
                    print('[teacher] time budget reached, stopping.', flush=True)
                    raise StopIteration
                t_cell = time.time()
                if cfg is None:
                    stages, finals = pl.teacher_reference(sd)
                else:
                    stages, finals = pl.run_cell(cfg, sd, args.n_init, args.device, seed=args.seed)
                if writer is None:
                    first = next(iter(stages.values()))
                    fields = list(KEY_COLS) + list(first.keys())
                    writer = csv.DictWriter(f, fieldnames=fields)
                    if new_file:
                        writer.writeheader()
                n = _write_rows(writer, sd.name, name, solver, stages)
                f.flush()
                np.savez_compressed(os.path.join(out, 'finals', f'{cell_id}.npz'), **finals)
                done.add(cell_id)
                with open(done_path, 'a') as d:
                    d.write(cell_id + '\n')
                print(f'[teacher] shape {si + 1}/{len(shapes)} {sd.name} / {name}: {n} rows, '
                      f'{time.time() - t_cell:.1f}s, {(time.time() - t0) / 60:.1f} min total', flush=True)
    except StopIteration:
        pass
    except KeyboardInterrupt:
        print('[teacher] interrupted -- finished cells are already written.', flush=True)
    finally:
        f.close()
    print(f'[teacher] done in {(time.time() - t0) / 60:.1f} min -> {out}', flush=True)


if __name__ == '__main__':
    main()
