r"""
recompute_thresholds.py -- steps_to_full_coverage at looser thresholds
=========================================================================
Philipp's request (2026-09-18): the project default (99%) makes
steps_to_full_coverage essentially binary (see analyze.py section 12 /
report section 12). Recompute it at 80% and 90% too, from the already-saved
winning trajectories (no new generation needed) -- local env is enough,
no torch network/GPU/SVGD required here, just the swept-coverage calc.
"""
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for p in (_here, os.path.join(_arch, 'exploration'), _arch,
         os.path.join(_arch, 'ergodic_dataset_generator'),
         os.path.join(_root, 'SE3_SVGD'), os.path.join(_root, 'src')):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from common.data import load_truth
from metrics_explore_exploit import steps_to_full_coverage

RAW = os.path.join(_here, 'results', 'best_of_30_paired_20260918', 'raw')
SHAPES = ['A', 'a_lc', 'digit_5', 'greek_upper_0', 'korean_5',
         'rand_ana_poly_10', 'rand_gmm_10', 'rand_gmm_20']
CONDS = ['ground_truth', 'half_known', 'ten_samples', 'none_known']

METHODS = {
    'CFM particles': lambda cond: os.path.join(RAW, cond, 'particles',
        'particles_mass_tuned_svgd25__no_replan'),
    'CFM spectral': lambda cond: os.path.join(RAW, cond, 'spectral',
        'spectral_mass_tuned_svgd25__no_replan'),
    'heuristic_tuned': lambda cond: os.path.join(RAW, cond, 'heuristic_tuned',
        'heuristic_tuned_mass__mass_tuned_svgd0_svgd1000'),
    'linear_waypoints_tuned': lambda cond: os.path.join(RAW, cond, 'linear_waypoints_tuned',
        'linear_waypoints_tuned_mass__mass_tuned_svgd0_svgd500'),
    'lawnmower': lambda cond: os.path.join(RAW, 'shared', 'lawnmower', 'lawnmower_fixed'),
}

names, truths = load_truth(labels=SHAPES, n=999, split='val', resolution=96, device='cpu')
truth_by_name = dict(zip(names, truths))

thresholds = [0.80, 0.90, 0.99]
results = []
for method, path_fn in METHODS.items():
    for cond in CONDS:
        d = path_fn(cond)
        for shape in SHAPES:
            traj_path = os.path.join(d, shape, 'trajectory.npy')
            if not os.path.isfile(traj_path):
                print('MISSING', traj_path)
                continue
            curve = torch.as_tensor(np.load(traj_path), dtype=torch.float32)
            truth = truth_by_name[shape]
            row = {'method': method, 'knowledge_condition': cond, 'shape': shape}
            for th in thresholds:
                frac, reached = steps_to_full_coverage(curve, truth, threshold=th,
                                                        knowledge_condition=cond)
                row[f'steps_{int(th*100)}'] = frac
                row[f'reached_{int(th*100)}'] = float(reached)
            results.append(row)

import csv
_analysis_dir = os.path.dirname(os.path.abspath(__file__))
out_csv = os.path.join(_analysis_dir, 'tables', '18_steps_thresholds_raw.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f'{len(results)} rows -> {out_csv}')
