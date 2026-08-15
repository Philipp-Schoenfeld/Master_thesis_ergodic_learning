r"""
model_zoo.py
============
Loading and sampling for the trained generators, behind one interface.

Both the evaluation script and the visualisation need to turn a checkpoint into
"give me n trajectories for this conditioning", and the two model families do
that very differently — the flow-matching nets integrate an ODE with CFG, the
self-supervised generator is a single forward pass. Keeping that difference in
one place means a new model type is added once, not twice, and it keeps the two
consumers free of import cycles.
"""

import time
import torch


def load_model(ckpt_path, device):
    """Build the right model for a checkpoint and load its weights.

    Returns (model, kind, meta) where kind is 'selfsup' or 'flow'. The type is
    read from the checkpoint's own `selfsupervised` flag, so nothing has to be
    inferred from the file name.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    nxi = ckpt.get('nxi', 25)
    nd = ckpt.get('nd', 2)
    D = ckpt.get('D', 384)
    meta = dict(
        nxi=nxi, nd=nd, D=D,
        n_particles=ckpt.get('n_particles', 256),
        epoch=ckpt.get('epoch'),
        cfg_weight=ckpt.get('cfg_weight', 2.0),
        n_candidates=ckpt.get('n_candidates'),
        diversity_weight=ckpt.get('diversity_weight'),
        lambda_erg=ckpt.get('lambda_erg', 0.0),
        use_obstacle=ckpt.get('use_obstacle', False),
    )

    if ckpt.get('selfsupervised', False):
        from flow_matching_particles_selfsupervised import SelfSupervisedParticleGenerator
        model = SelfSupervisedParticleGenerator(nxi=nxi, nd=nd, D=D).to(device)
        kind = 'selfsup'
    else:
        from flow_matching_cond_particles_crossattn import ParticleCrossAttnFlowNetwork
        model = ParticleCrossAttnFlowNetwork(nxi=nxi, nd=nd, D=D).to(device)
        kind = 'flow'

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, kind, meta


def describe(kind, meta):
    """Short human-readable configuration string for titles and logs."""
    bits = [f"D={meta['D']}", f"N={meta['n_particles']}"]
    if kind == 'selfsup':
        bits.append(f"K={meta['n_candidates']} div={meta['diversity_weight']:g}"
                    if meta['n_candidates'] else 'selbstüberwacht')
    elif meta['lambda_erg']:
        bits.append(f"ergodic loss w={meta['lambda_erg']:g}")
    return ', '.join(bits)


@torch.no_grad()
def generate(model, kind, particles, n_samples, meta, steps, device, seed,
             obstacle=None, obstacle_weight=20.0, obstacle_t_start=0.3,
             polish_steps=250):
    """n trajectories for one conditioning. Returns (cps, seconds).

    `obstacle` applies the inference-time repulsion. The flow models steer around
    it during integration; the single-pass generator has no integration to steer,
    so its output is pushed out afterwards by the same penalty descent — the
    guarantee (no penetration) is identical, only the route there differs.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    g = torch.Generator(device=dev).manual_seed(seed)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if kind == 'selfsup':
        cps = model.generate(particles, num_samples=n_samples,
                             device=str(dev), generator=g)
        if obstacle is not None:
            from obstacles import basis_torch, polish_out_of_obstacle
            B = basis_torch(meta['nxi'], 256, 5, device=dev)
            cps = polish_out_of_obstacle(cps, obstacle, B, max_iters=polish_steps)
    else:
        from flow_matching_cond_particles_crossattn import generate_particle_trajectories
        cps, _ = generate_particle_trajectories(
            model, particles, num_samples=n_samples,
            nxi=meta['nxi'], nd=meta['nd'], steps=steps, device=str(dev),
            cfg_weight=meta['cfg_weight'], generator=g,
            obstacle=obstacle, obstacle_weight=obstacle_weight,
            obstacle_t_start=obstacle_t_start, polish_steps=polish_steps)

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    return cps, time.perf_counter() - t0
