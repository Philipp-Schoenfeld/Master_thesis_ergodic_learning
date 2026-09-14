r"""
spectral_planner.py
====================
`SpectralPlanner` — the missing wrapper for the spectral-conditioned CFM
checkpoint, parallel to `apply_cfm_belief.CfmPlanner` (particle-conditioned).

Why this didn't already exist: `model_zoo.py::load_model`, `common/planner.py
::ModelPlanner` and `apply_cfm_belief.py::CfmPlanner` only branch on
`selfsupervised`/`length_cond`/`start_cond` flags in the checkpoint — none of
them recognise a spectral-conditioned checkpoint, and
`flow_matching_cond_spectral_crossattn.py::generate_spectral_trajectories`
has never been called from outside `flow_matching_runner_spectral.py`'s own
training/visualisation loop.

Conditioning input. `flow_matching_runner_spectral.py::trajectory_to_spectral`
builds the network's condition from a *reference trajectory's* time-averaged
Fourier statistic c_k, because at training time that is what's available (the
database stores trajectories, not densities). At inference time here there is
no reference trajectory for a partially-known belief — only a density grid —
so this planner instead feeds the network phi_k = integral(mu(x) F_k(x) dx),
computed directly from the belief/target grid via
`ergodic_energy_torch.target_coeffs_from_grid`. Both are estimates of the same
quantity (an ergodic trajectory's c_k should converge to the target's phi_k),
but the network never saw phi_k-shaped inputs during training — this is a
genuine train/inference mismatch, not a solved problem, and any spectral-arm
result should be read with that caveat attached.

Caveat that matters just as much: this checkpoint's `holdout_labels`
(`sigma, rand_poly_7, G, spiral_2cw, star_5, W, lissajous_1_3, heart, 5, phi`)
come from a different, smaller synthetic-shape dataset than
`shape_library.VALIDATION_SHAPES` (letters/GMMs/CJK/organic blobs) used
everywhere else in this evaluation. Running it against the same 25 shapes as
the particle model tests genuine out-of-distribution generalisation, not a
like-for-like representation comparison — flag this wherever spectral-arm
numbers are reported.

No start-point conditioning exists in this architecture at all (confirmed:
the checkpoint carries no `start_cond`-equivalent flag, and
`SpectralCrossAttnFlowNetwork.forward` takes no start argument). Replanning
therefore pins the first point post-hoc, exactly like `SvgdRefiner.refine`
already does for the particle-conditioned non-start-cond fallback.
"""

import os
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
if _arch not in sys.path:
    sys.path.insert(0, _arch)

from ergodic_energy_torch import make_k_grid, target_coeffs_from_grid  # noqa: E402


class SpectralPlanner:
    """Spectral-coefficient-conditioned CFM planner.

    Interface deliberately mirrors `apply_cfm_belief.CfmPlanner` (`.plan`,
    `.render`, `.last_wallclock`) but `.plan` takes a *density grid*
    (mu, or any (R,R) target density) instead of a particle cloud — spectral
    conditioning has no use for point samples.
    """

    def __init__(self, ckpt_path, device='cpu', pts=128, deg=5, steps=100):
        from obstacles import bspline_basis_matrix
        from flow_matching_cond_spectral_crossattn import (
            SpectralCrossAttnFlowNetwork, generate_spectral_trajectories)

        self.device = torch.device(device)
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.nxi = ck['nxi']
        self.nd = ck['nd']
        self.S = ck['S']
        self.cfg_weight = ck.get('cfg_weight', 1.0)
        self.steps = steps
        self.start_cond = False   # this architecture never supports it

        self.model = SpectralCrossAttnFlowNetwork(
            nxi=self.nxi, nd=self.nd, D=ck['D'], S=self.S,
            n_lambda=ck['n_lambda'], predict_lambda=ck['predict_lambda'],
        ).to(self.device)
        self.model.load_state_dict(self._patched_state_dict(ck['model_state_dict']))
        self.model.eval()
        self._gen = generate_spectral_trajectories

        K = int(np.ceil(np.sqrt(self.S)))
        k_idx_full, Lambda_full = make_k_grid(K)          # (K^2,2),(K^2,)
        # Two dtypes, two uses: `k_idx` (float) feeds the cosine basis in
        # `target_coeffs_from_grid`; `k_idx_long` indexes the network's
        # positional-encoding table (`FrequencyPositionalEncoding2D`), which
        # requires integer indices -- passing the float version throws
        # "tensors used as indices must be long, int, byte or bool".
        self.k_idx = torch.from_numpy(k_idx_full[:self.S]).float().to(self.device)
        self.k_idx_long = torch.from_numpy(k_idx_full[:self.S]).long().to(self.device)
        self.Lambda = torch.from_numpy(Lambda_full[:self.S]).float().to(self.device)

        self.B = torch.from_numpy(
            bspline_basis_matrix(self.nxi, pts, deg)).float().to(self.device)
        self.last_wallclock = 0.0

    def _patched_state_dict(self, sd):
        """`cond_spectral_crossattn_ep900.pt` predates the `Lambda_k` input
        channel of `SpectralTokenizer.freq_mlp` (checked directly: this is
        the *only* shape mismatch between the checkpoint and the current
        `SpectralCrossAttnFlowNetwork` — every other key matches exactly).

        The checkpoint's `freq_mlp.0.weight` is (D,1): the network was
        trained with c_k as the sole input. Zero-padding it to (D,2) instead
        of leaving the new column at its random init reproduces the
        checkpoint's exact trained function (the padded column contributes
        exactly 0 regardless of Lambda_k) rather than injecting untrained
        noise into every forward pass.
        """
        key = 'spectral_tokenizer.freq_mlp.0.weight'
        own = self.model.state_dict()
        if key in sd and key in own and sd[key].shape != own[key].shape:
            old_w = sd[key]
            if old_w.shape[1] == 1 and own[key].shape[1] == 2:
                print(f"[SpectralPlanner] patching {key}: {tuple(old_w.shape)} -> "
                     f"{tuple(own[key].shape)} (zero-padded Lambda_k column; "
                     "checkpoint predates that input channel)")
                padded = torch.zeros_like(own[key])
                padded[:, :1] = old_w
                sd = dict(sd)
                sd[key] = padded
            else:
                raise RuntimeError(
                    f"unexpected shape mismatch for {key}: ckpt {tuple(old_w.shape)} "
                    f"vs model {tuple(own[key].shape)} -- not the known "
                    "Lambda_k-channel drift, needs a fresh look before patching blindly")
        return sd

    def spectral_from_density(self, density):
        """(R,R) density -> (S,2) [phi_k, Lambda_k] condition, see module docstring."""
        phi_k = target_coeffs_from_grid(density.to(self.device), self.k_idx)
        return torch.stack([phi_k, self.Lambda], dim=-1)

    def render(self, cps):
        return torch.einsum('pi,kid->kpd', self.B, cps.float().to(self.device))

    def plan(self, density, n_candidates=1, start=None):
        """`density`: (R,R) target grid. `start` is accepted for interface
        parity with CfmPlanner but only applied as a hard post-hoc snap of
        the first control point — this architecture has no learned start
        conditioning (see module docstring)."""
        t0 = time.perf_counter()
        spec = self.spectral_from_density(density)
        x, _lambda0 = self._gen(
            self.model, spec, self.k_idx_long, num_samples=n_candidates,
            nxi=self.nxi, nd=self.nd, steps=self.steps,
            device=str(self.device), cfg_weight=self.cfg_weight)
        if start is not None:
            with torch.no_grad():
                x[:, 0] = start.to(self.device)
        self.last_wallclock = time.perf_counter() - t0
        return x.detach()
