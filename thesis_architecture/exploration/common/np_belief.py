r"""
np_belief.py
============
Belief from the pretrained attentive neural process
(`neural_process_belief.py`, trained by `train_belief_np.py`) instead of a GP.

`mu` and `sigma` are the NP's predictive mean and standard deviation given
all measurements so far: one forward pass, no fit. Without measurements the
NP has no context to attend to, so the prior is the same as `GPBelief`'s
(mu = 0, sigma = sqrt(variance)).

Two differences to a GP the caller has to keep in mind:
- `uncertainty_after` is not available: the NP's sigma depends on the
  measured values, not only on the locations, so it cannot be looked ahead.
- The NP was meta-trained on 32-128 uniformly scattered measurements with
  noise 0.01; missions measure along paths, with more points and more noise.
"""

import copy
import glob
import os
import sys

import torch

_arch = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _arch not in sys.path:
    sys.path.insert(0, _arch)

from neural_process_belief import load_np  # noqa: E402
from common.belief import GPBelief         # noqa: E402


def latest_np_ckpt():
    found = sorted(glob.glob(os.path.join(_arch, 'checkpoints', 'belief_np_date_*_final.pt')),
                   key=os.path.getmtime)
    if not found:
        raise FileNotFoundError('no checkpoints/belief_np_date_*_final.pt; pass np_ckpt')
    return found[-1]


_MODELS = {}


def _model(ckpt, device):
    key = (os.path.abspath(ckpt), str(device))
    if key not in _MODELS:
        _MODELS[key] = load_np(ckpt, device)[0]
    return _MODELS[key]


class NPBelief(GPBelief):
    def __init__(self, ckpt=None, **kw):
        super().__init__(**kw)
        self.ckpt = ckpt or latest_np_ckpt()
        self.model = _model(self.ckpt, self.device)

    def clone(self):
        b = copy.copy(self)
        b.X, b.y = self.X.clone(), self.y.clone()
        return b

    def posterior(self, pts):
        pts = pts.to(device=self.device, dtype=self.dtype)
        if self.X.shape[0] == 0:
            return super().posterior(pts)
        ctx = torch.cat([self.X, self.y.unsqueeze(-1)], dim=-1).float().unsqueeze(0)
        with torch.no_grad():
            mean, std = self.model(ctx, pts.float().unsqueeze(0))
        return mean[0].to(self.dtype), std[0].to(self.dtype)

    def uncertainty_after(self, pts):
        raise NotImplementedError(
            'NPBelief: sigma depends on the measured values, no look-ahead possible')
