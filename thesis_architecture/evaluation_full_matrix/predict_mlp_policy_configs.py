#!/usr/bin/env python3
r"""
predict_mlp_policy_configs.py
================================
Turns "what would Policy A (the trained MLP value model) pick as its first
move under knowledge_condition X?" into one fixed `(phi_model, param,
svgd_iters)` config per knowledge_condition -- the same shape as an entry in
`variant_runner.STRATEGIES` -- so it can be run through the ordinary
single-trajectory evaluation_full_matrix pipeline (`run_ideal_matrix.py`)
exactly like `eid_optuna_ideal_v2` or any hand-tuned strategy.

Read-only / cheap: builds the round-0 belief for each of the 24 holdout
shapes under each of the 4 `variant_runner.KNOWLEDGE_CONDITIONS`
(ground_truth/none_known/half_known/ten_samples), turns each into a
`policy.features.Zustand` exactly as `exploration_optimierung/mission.py`
does for round 0 of a mission, and asks the loaded value model
(`policy/ablage/policy_a_maske.pt`) for its predicted quality of all 48
trained candidates. No CFM planner, no SVGD, no trajectory generation --
just GP-posterior computation and one MLP forward pass per shape/condition,
seconds not minutes.

The per-condition config is the candidate with the best *mean* predicted
quality across all 24 shapes (lower = better, same selection rule the policy
itself uses at inference time -- see `policy.model.WertRichtlinie.__call__`),
mirroring how `eid_optuna_ideal_v2` is one fixed config found by averaging
over many shapes, not a per-shape choice.

Caveat: Policy A was trained under a randomly sized unknown region
(`--zufallsmaske`, `unbekannt_bereich=[0.5, 0.9]`, i.e. 50-90% of the grid
withheld). `half_known` (~50%) sits at the edge of that training
distribution; `ground_truth` (0% unknown) and `none_known` (100% unknown)
are outside it entirely, and `ten_samples` is a sparse point-prior rather
than a masked region -- a structurally different kind of belief the region
mask never produced. The predictions below are the policy's best guess under
condition it never saw, not a validated recommendation for those cases.

    python predict_mlp_policy_configs.py
"""

import json
import os
import sys

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

from common.data import load_truth                               # noqa: E402
import variant_runner as vr                                       # noqa: E402
from exploration_optimierung.policy.features import Zustand       # noqa: E402
from exploration_optimierung.policy.model import WertRichtlinie   # noqa: E402

POLICY_CKPT = os.path.join(_arch, 'exploration_optimierung', 'policy',
                           'ablage', 'policy_a_maske.pt')
OUT_JSON = os.path.join(_here, 'mlp_policy_predicted_configs.json')

#: Matches `policy_orakel_maske.json`'s `n_max` (the mission length the
#: masked policy was trained/evaluated under) so the `runde`/`rest`/
#: `weg_laenge` features sit on the same scale as training.
N_MAX = 10


def predict(device='cpu', truth_res=96, seed=0):
    names, truths = load_truth(labels=None, n=999, split='val',
                               resolution=truth_res, device=device)
    policy = WertRichtlinie.laden(POLICY_CKPT, device=device)
    K = len(policy.kandidaten)

    per_condition = {}
    for cond in vr.KNOWLEDGE_CONDITIONS:
        scores = np.zeros(K, dtype=np.float64)
        for i, name in enumerate(names):
            belief = vr.build_belief(cond, truths[i], seed=seed, device=device)
            mu, sd = belief.posterior_grid()
            z = Zustand(mu=mu, sd=sd, visit=None, driven=None, runde=0,
                       n_max=N_MAX, n_obs=belief.n_obs, name=name)
            w = policy.werte([z])          # (1, K)
            scores += w[0].detach().cpu().numpy().astype(np.float64)
        scores /= len(names)

        order = np.argsort(scores)         # smaller predicted value = better
        best_idx = int(order[0])
        best = policy.kandidaten[best_idx]
        top5 = [dict(kandidat=list(policy.kandidaten[j]),
                    mean_predicted_value=float(scores[j]))
               for j in order[:5]]

        per_condition[cond] = dict(
            phi_model=best[0], param=float(best[1]), svgd_iters=int(best[2]),
            mean_predicted_value=float(scores[best_idx]),
            n_shapes=len(names), top5=top5)
        print(f"[{cond:>12}] predicted: model={best[0]:<7} param={best[1]:<10.4f} "
             f"svgd_iters={int(best[2]):<4d}  mean_predicted_value={scores[best_idx]:.4f}")

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict(policy_ckpt=os.path.relpath(POLICY_CKPT, _root),
                       n_max=N_MAX, shapes=names,
                       per_condition=per_condition), f, indent=2)
    print(f"-> {OUT_JSON}")
    return per_condition


if __name__ == '__main__':
    predict(device='cuda' if torch.cuda.is_available() else 'cpu')
