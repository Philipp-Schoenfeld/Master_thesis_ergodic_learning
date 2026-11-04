"""
planner3d.py
============
The trained 3D SE(3) flow-matching network as a planner.

It is conditioned on a particle cloud ``(x, y, z, weight)`` that lives *on the
target surface* (not in a volume: that is how the network was trained) and,
optionally, on a start point.  It returns B-spline control points for the
position (25 x 3) and a 6D orientation per control point; the sensor axis
(third column of the rotation) points at the surface.
"""

import numpy as np
import torch

import paths3d

paths3d.setup()
from flow_matching_cond_particles_crossattn import (  # noqa: E402
    ParticleCrossAttnFlowNetwork, generate_particle_trajectories)
from orientation import rot6d_to_matrix                # noqa: E402
from obstacles import bspline_basis_matrix             # noqa: E402  (the 3D one)


class Planner3D:
    def __init__(self, ckpt=paths3d.DEFAULT_CKPT, device=None, steps=100,
                 cfg_weight=2.0, pts=128, deg=5):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.ckpt_info = {k: ck.get(k) for k in ('epoch', 'nxi', 'D', 'n_particles',
                                                  'orientation', 'start_cond', 'db3d')}
        self.nxi = int(ck.get('nxi', 25))
        self.n_particles = int(ck.get('n_particles', 512))
        self.start_cond = bool(ck.get('start_cond', False))
        self.orientation = bool(ck.get('orientation', False))
        self.steps, self.cfg_weight = steps, cfg_weight
        self.model = ParticleCrossAttnFlowNetwork(
            nxi=self.nxi, nd=3, D=int(ck.get('D', 384)),
            predict_orientation=self.orientation, start_cond=self.start_cond).to(self.device)
        self.model.load_state_dict(ck['model_state_dict'])
        self.model.eval()
        self.B = torch.from_numpy(bspline_basis_matrix(self.nxi, pts, deg)).float().to(self.device)

    @torch.no_grad()
    def plan(self, particles, start=None, generator=None):
        """particles (N, 4) -> (cps (nxi, 3), rot6 (nxi, 6) or None), on the CPU.

        ``start`` (3,) pins the first control point and conditions the network;
        None runs the network's null-start token."""
        parts = torch.as_tensor(np.asarray(particles), dtype=torch.float32, device=self.device)
        kw = {}
        if start is not None and self.start_cond:
            kw['start'] = torch.as_tensor(np.asarray(start), dtype=torch.float32,
                                          device=self.device).reshape(-1)
        cps, rot6 = generate_particle_trajectories(
            self.model, parts, num_samples=1, nxi=self.nxi, nd=3, steps=self.steps,
            device=str(self.device), cfg_weight=self.cfg_weight, generator=generator, **kw)
        return cps[0].detach().cpu(), (None if rot6 is None else rot6[0].detach().cpu())

    @torch.no_grad()
    def render(self, cps, rot6=None):
        """Control points -> dense curve (T, 3) and unit sensor axes (T, 3)."""
        cps = cps.to(self.device).float()
        curve = self.B @ cps
        if rot6 is None:
            return curve.cpu(), None
        R = rot6d_to_matrix(rot6.to(self.device).float().reshape(-1, 6)).reshape(self.nxi, 3, 3)
        axes = self.B @ R[:, :, 2]
        axes = axes / axes.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return curve.cpu(), axes.cpu()
