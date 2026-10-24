r"""
ak_belief.py
============
`GPBelief` with the Attentive Kernel (Chen, Khardon, Liu, RSS 2022) instead of
the stationary RBF kernel.

The AK mixes 10 RBF primitives (lengthscales 0.005-0.25 domain widths) with
location-dependent weights from a small network, so the belief can be sharp
at stroke edges and smooth in empty regions. Only that network is fitted, by
maximising the marginal likelihood after every `observe`, warm-started from
the previous fit (as in the authors' online setting).

Amplitude and noise stay fixed at `variance` and `noise`: the AK has
k(x, x) = amplitude exactly (normalised attention weights), so the prior
standard deviation and hence the scale of sigma match the RBF belief, and the
UCB kappa schedules tuned for it stay comparable. Prior mean 0 as in
`GPBelief`. Posterior variance still depends only on the measurement
locations, so `uncertainty_after` keeps its meaning for a fixed kernel.
"""

import copy
import math
import os
import sys

import torch

_arch = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _arch not in sys.path:
    sys.path.insert(0, _arch)

from attentive_kernel_belief import BatchedAttentiveKernelGP  # noqa: E402
from common.belief import GPBelief                            # noqa: E402


class AKBelief(GPBelief):
    """Args (in addition to GPBelief's):
        fit_iters: Adam steps on the marginal likelihood per `observe`.
        lr:        Adam learning rate of the kernel network (reference: 1e-3).
        seed:      initialisation of the kernel network.
    """

    def __init__(self, fit_iters=200, lr=1e-3, seed=0, **kw):
        super().__init__(**kw)
        self.fit_iters = fit_iters
        self.lr = lr
        gen = torch.Generator(device=self.device).manual_seed(seed)
        self.ak = BatchedAttentiveKernelGP(1, dtype=self.dtype, device=self.device,
                                           generator=gen)
        self.ak.raw_amplitude.data.fill_(math.log(math.expm1(self.var)))
        self.ak.requires_grad_(False)
        self._opt = None

    def _nn_params(self):
        return [p for n, p in self.ak.named_parameters() if n.startswith('nn_')]

    def _k(self, a, b):
        return self.ak.kernel(a.unsqueeze(0), b.unsqueeze(0))[0]

    def clone(self):
        b = copy.copy(self)
        b.ak = copy.deepcopy(self.ak)
        b.X, b.y = self.X.clone(), self.y.clone()
        b._cache, b._opt = None, None
        return b

    def observe(self, pts, vals):
        super().observe(pts, vals)
        self._fit()
        return self

    def _fit(self):
        params = self._nn_params()
        if self._opt is None:
            self._opt = torch.optim.Adam(params, lr=self.lr)
        X, y = self.X.detach(), self.y.detach().unsqueeze(-1)
        eye = (self.noise ** 2) * torch.eye(X.shape[0], device=self.device, dtype=self.dtype)
        for p in params:
            p.requires_grad_(True)
        with torch.enable_grad():
            for _ in range(self.fit_iters):
                L = torch.linalg.cholesky(self._k(X, X) + eye)
                alpha = torch.cholesky_solve(y, L)
                nll = 0.5 * (y * alpha).sum() + torch.log(torch.diagonal(L)).sum()
                self._opt.zero_grad(set_to_none=True)
                nll.backward()
                self._opt.step()
        for p in params:
            p.requires_grad_(False)
        self._cache = None
