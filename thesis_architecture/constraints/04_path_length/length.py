r"""
length.py
=========
Constraint 4: **target path length** as an inference-time force.

    E = 0.5 * (L/L_target - 1)^2        (mode 'exact', two-sided)
    E = 0.5 * relu(L/L_target - 1)^2    (mode 'cap',   budget only)

The repo already answers this question by *training*: the `length_cond`
checkpoints (`flow_matching_cond_particles_length.py`) take the target length
as a FiLM input with its own CFG channel. This module is the inference-only
complement -- it needs no retraining and works with any checkpoint, including
ones that never saw a length signal.

**What the holdout table does and does not measure.** Both arms of that table
come from the *same* checkpoint in the *same* mode: `common.guided_generate`
never passes `length`, so the model always receives its `null_length_token`
and its learned length channel is switched off throughout. The comparison is
therefore free-vs-force, with seed, conditioning particles and CFG weight held
identical -- a clean measurement of what the force alone does.

It is *not* a comparison against the learned conditioning. That one would be
the genuinely interesting result -- a learned conditioning can redistribute the
path to spend its budget well because the network re-plans the whole curve,
while a post-hoc force can only stretch or shrink what the flow already drew --
but it needs a third arm that calls `generate_particle_trajectories` with
`length=`/`length_cfg_weight=`, and that has not been run. Do not read the
current numbers as evidence about it.

Note the relative (dimensionless) form: penalising `L - L_target` directly
would make the guidance weight depend on how long the paths happen to be for
a given shape, so a weight tuned on a compact letter would misbehave on a
sprawling one. `L/L_target - 1` keeps one weight usable everywhere.

Arc length couples neighbouring curve points, so the gradient goes through
`common.curve_energy_grad` (autograd), not the pointwise shortcut.
"""

import torch

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import arc_length


class TargetLength:
    """Drive the rendered curve's arc length to `target`.

    mode='exact' pulls from both sides; mode='cap' only penalises exceeding
    the budget, which is the more realistic robot constraint (a battery or a
    time budget bounds the path from above, not below).
    """

    def __init__(self, target, mode='exact'):
        assert mode in ('exact', 'cap'), mode
        self.target = float(target)
        self.mode = mode

    def __repr__(self):
        return f"TargetLength(target={self.target:.3f}, mode={self.mode!r})"

    def energy(self, curve):
        rel = arc_length(curve) / self.target - 1.0
        if self.mode == 'cap':
            rel = torch.clamp(rel, min=0.0)
        return 0.5 * (rel ** 2).sum()

    def length(self, curve):
        return arc_length(curve)[0].item()

    def rel_error(self, curve):
        """Signed relative deviation from the target, the reported metric."""
        return self.length(curve) / self.target - 1.0
