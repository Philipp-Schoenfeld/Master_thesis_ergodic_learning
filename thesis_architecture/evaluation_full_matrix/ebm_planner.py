r"""
ebm_planner.py
===============
Gives the Ranking-NCE energy-based policy (`src/methods/EBM_RankingNCE`,
the "second generative head alongside CFM" per its own training job script)
the same duck-typed planner interface as `apply_cfm_belief.CfmPlanner` /
`selfsup_planner.SelfsupPlanner` (`.nxi`, `.start_cond`, `.device`,
`.plan(particles, n_candidates, start=None) -> cps`, `.render(cps) -> curves`)
so it drops into `run_init_comparison.py` wherever a planner is expected.

The EBM is not a generator by itself -- `evaluate_holdout.py` (its own
holdout-eval script) only ever *refines* a GIVEN warm start via Langevin or
CEM, and `cem_optimize`'s own docstring is explicit that it is "intended for
*local* search around a given warm start ... not global search from
scratch" since a Ranking-NCE energy only has well-shaped gradients/levels
near the data manifold it was contrasted against. So `EbmPlanner.plan`
seeds CEM from `n_candidates` evenly-spaced linear-chord curves (the same
parameter-free baseline `init_baselines.linear_angle_path` already provides
elsewhere in this comparison, fit to B-spline control points) with a modest
`init_std` -- a plausible-but-generic starting manifold, not any other
method's learned sample (so this stays a standalone baseline, not CFM-plus-
EBM) and not pure noise (which the docstring above warns will not reliably
descend).
"""
import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_root, 'src', 'methods', 'EBM_RankingNCE')):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

#: The only fully-trained Ranking-NCE checkpoint (500 epochs, D=256,
#: n_particles=256, n_negatives=100).
DEFAULT_EBM_CKPT = os.path.join(
    _root, 'src', 'methods', 'EBM_RankingNCE', 'checkpoints',
    'ebm_ebm_rankingnce_particles_start_date_10_13_18h43min_nxi25_D256_N256_C100_final.pt')

CEM_ITERS = 10
CEM_SAMPLES = 256
CEM_INIT_STD = 0.15


class EbmPlanner:
    def __init__(self, ckpt=None, device='cpu', pts=128, deg=5):
        from ebm_ranking_nce_core import EnergyNet
        from obstacles import bspline_basis_matrix

        self.device = torch.device(device)
        ckpt = ckpt or DEFAULT_EBM_CKPT
        ck = torch.load(ckpt, map_location=device, weights_only=True)
        self.nxi = ck.get('nxi', 25)
        nd = ck.get('nd', 2)
        D = ck.get('D', 256)
        self.n_particles = ck.get('n_particles', 256)
        self.sample_mode = ck.get('sample_mode', 'uniform')
        self.start_cond = bool(ck.get('use_start_cond', True))
        self.cfg_weight = 1.0          # unused, kept for interface parity

        self.model = EnergyNet(context_mode='particles', nxi=self.nxi, nd=nd, D=D,
                               use_sigma_cond=True, use_start_cond=self.start_cond).to(self.device)
        self.model.load_state_dict(ck['model_state_dict'])
        self.model.eval()

        self.B = torch.from_numpy(
            bspline_basis_matrix(self.nxi, pts, deg)).float().to(self.device)
        self.last_wallclock = 0.0

    def render(self, cps):
        return torch.einsum('pi,kid->kpd', self.B, cps.float().to(self.device))

    def _linear_init_cps(self, n_candidates):
        """n_candidates evenly-spaced linear chords, fit to this planner's
        B-spline control points -- see module docstring for why CEM is
        seeded from these rather than noise."""
        from init_baselines import linear_angle_path
        angles = [180.0 * i / max(n_candidates, 1) for i in range(n_candidates)]
        curves = torch.stack([linear_angle_path(a, self.B.shape[0]) for a in angles]).to(self.device)
        B_np = self.B.detach().cpu().numpy().astype(np.float64)
        rhs = curves.detach().cpu().numpy().transpose(1, 0, 2).reshape(B_np.shape[0], -1)
        cps = np.linalg.lstsq(B_np, rhs, rcond=None)[0].reshape(self.nxi, n_candidates, 2)
        return torch.as_tensor(cps.transpose(1, 0, 2), dtype=torch.float32, device=self.device)

    def plan(self, particles, n_candidates=1, start=None, **_ignored):
        """`particles`: (n_particles, 2/3) or (1, n_particles, ...) cloud
        sampled from the target density (same convention as CfmPlanner).
        `start`: (2,) agent start position, used for CEM's start-conditioning
        if the checkpoint was trained with it; ignored otherwise."""
        from ebm_ranking_nce_core import cem_optimize

        t0 = time.perf_counter()
        parts = particles.to(self.device)
        if parts.dim() == 2:
            parts = parts.unsqueeze(0)
        if parts.shape[0] == 1 and n_candidates > 1:
            parts = parts.expand(n_candidates, *parts.shape[1:])

        init_cps = self._linear_init_cps(n_candidates)
        start_t = None
        if self.start_cond and start is not None:
            start_t = torch.as_tensor(start, dtype=torch.float32, device=self.device)
            if start_t.dim() == 1:
                start_t = start_t.unsqueeze(0).expand(n_candidates, -1)

        with torch.no_grad():
            best_cps, _best_energy = cem_optimize(
                self.model, parts, self.nxi, 2, k_indices=None,
                n_iters=CEM_ITERS, n_samples=CEM_SAMPLES, elite_frac=0.1,
                init_mean=init_cps, init_std=CEM_INIT_STD, start=start_t)
        self.last_wallclock = time.perf_counter() - t0
        return best_cps.detach()
