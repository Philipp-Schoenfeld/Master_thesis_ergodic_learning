r"""
selfsup_planner.py
===================
Thin wrapper that gives the self-supervised single-pass generator
(`flow_matching_particles_selfsupervised.SelfSupervisedParticleGenerator`) the
same duck-typed interface as `apply_cfm_belief.CfmPlanner`
(`.nxi`, `.start_cond`, `.cfg_weight`, `.device`,
`.plan(particles, n_candidates, start=None) -> cps`, `.render(cps) -> curves`),
so it drops into `run_svgd_convergence.build_planner_curves` and
`run_mission_eval.MissionSet.network_candidates` wherever a CFM planner is
expected -- no separate code path.

`start_cond` is always False: the checkpoint has no start conditioning, and
`.plan` ignores `start` exactly like `CfmPlanner.plan` already does for a
non-start-conditioned CFM checkpoint ("the value is ignored instead of
raising"). That is correct here too -- the actual start-point binding happens
uniformly for every method during SVGD/Sun refinement (`BatchedSvgdTorch`/
`BatchedSunTorch`.run(..., starts, ...) pulls the first control point to the
agent position over the iterations), so nothing is lost by not hard-pinning
it at generation time.

Measured (`test_mission_eval.test_mission_real_selfsup`, `--refiner tsvec`,
CPU): the selected candidate's start_gap after refinement was ~0.17-0.38 at
100 iterations, ~0.01-0.08 at 500 -- unlike cfm/random_walk/linear, which all
already begin exactly at the agent position by construction, selfsup's raw
output starts wherever the network put it and genuinely needs refinement
iterations to close that gap. Production runs use 1500 (mission_eval) or 3000
(svgd_convergence) iterations, well past where this was still shrinking, but
a short/smoke run with few iterations will show a visibly larger start_gap
for selfsup than for the other methods -- expected, not a bug.

`SelfSupervisedParticleGenerator.generate()` already handles the three input
shapes a caller may pass (a single cloud, a pre-batched cloud of size 1, a
fully pre-batched cloud) exactly like the already-fixed CFM path does (see the
CLAUDE.md pitfall note on `generate_particle_trajectories`'s batch mismatch),
so no further shape handling is needed here.
"""
import os
import time

import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)

#: The only fully trained self-supervised checkpoint (500 epochs, D=384,
#: n_particles=256, n_candidates=8, diversity_weight=10). Known caveat: its
#: `diversity` metric measured ~0.08 (near the "ignores the noise input"
#: floor), so its `n_init` candidates are noticeably more similar to each
#: other than the CFM planner's -- an honest property of the current
#: checkpoint, not a bug in this wrapper.
DEFAULT_SELFSUP_CKPT = os.path.join(
    _arch, 'checkpoints',
    'selfsup_selfsup_particles_date_08_12_10h13min_nxi25_D384_N256_K8_div10_final.pt')


class SelfsupPlanner:
    def __init__(self, ckpt=None, device='cpu', pts=128, deg=5):
        from flow_matching_particles_selfsupervised import SelfSupervisedParticleGenerator
        from obstacles import bspline_basis_matrix

        self.device = torch.device(device)
        ckpt = ckpt or DEFAULT_SELFSUP_CKPT
        ck = torch.load(ckpt, map_location=device, weights_only=True)
        self.nxi = ck.get('nxi', 25)
        nd = ck.get('nd', 2)
        D = ck.get('D', 384)
        self.start_cond = False
        self.cfg_weight = 1.0          # unused, kept for interface parity with CfmPlanner
        self.n_particles = ck.get('n_particles', 256)

        self.model = SelfSupervisedParticleGenerator(
            nxi=self.nxi, nd=nd, D=D).to(self.device)
        self.model.load_state_dict(ck['model_state_dict'])
        self.model.eval()

        self.B = torch.from_numpy(
            bspline_basis_matrix(self.nxi, pts, deg)).float().to(self.device)
        self.last_wallclock = 0.0

    def render(self, cps):
        return torch.einsum('pi,kid->kpd', self.B, cps.float().to(self.device))

    def plan(self, particles, n_candidates=1, start=None, **_ignored):
        """`start` and any other CFM-only kwargs (length, obstacle, ...) are
        accepted and ignored -- see module docstring."""
        t0 = time.perf_counter()
        with torch.no_grad():
            cps = self.model.generate(particles.to(self.device),
                                      num_samples=n_candidates, device=str(self.device))
        self.last_wallclock = time.perf_counter() - t0
        return cps.detach()
