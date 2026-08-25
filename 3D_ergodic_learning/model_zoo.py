r"""
model_zoo.py  —  3D port
========================
Loading and sampling for the trained 3D generators, behind one interface.

Identical in structure to the 2D version. The only substantive addition is a
guard: a checkpoint is refused if its stored `dim` is not 3, because a 2D
checkpoint would load into a 3D network only to produce silent nonsense — the
state dict shapes differ in exactly two tensors (the MPD input convolution and
the head output layer), and everything else would match.
"""

import time
import torch


def load_model(ckpt_path, device):
    """Build the right model for a checkpoint and load its weights.

    Returns (model, kind, meta) where kind is 'selfsup' or 'flow'.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    dim = ckpt.get('dim', 2)
    if dim != 3:
        raise ValueError(
            f"{ckpt_path} was trained in {dim}D. This folder only loads 3D "
            f"checkpoints — the 2D ones belong to thesis_architecture/.")

    nxi = ckpt.get('nxi', 25)
    nd = ckpt.get('nd', 3)
    D = ckpt.get('D', 384)
    meta = dict(
        nxi=nxi, nd=nd, D=D,
        n_particles=ckpt.get('n_particles', 512),
        epoch=ckpt.get('epoch'),
        cfg_weight=ckpt.get('cfg_weight', 2.0),
        n_candidates=ckpt.get('n_candidates'),
        diversity_weight=ckpt.get('diversity_weight'),
        lambda_erg=ckpt.get('lambda_erg', 0.0),
        erg_K=ckpt.get('erg_K'),
        grid_res=ckpt.get('grid_res', 64),
        z_plane=ckpt.get('z_plane', 0.5),
        z_sigma=ckpt.get('z_sigma', 0.05),
        use_obstacle=ckpt.get('use_obstacle', False),
        orientation=ckpt.get('orientation', False),
        frame_mode=ckpt.get('frame_mode', 'lookat'),
        ergodic_on=ckpt.get('ergodic_on', 'position'),
        standoff_target=ckpt.get('standoff_target', 0.12),
        standoff_band=ckpt.get('standoff_band', 0.03),
    )

    if ckpt.get('selfsupervised', False):
        from flow_matching_particles_selfsupervised import SelfSupervisedParticleGenerator
        model = SelfSupervisedParticleGenerator(
            nxi=nxi, nd=nd, D=D,
            predict_orientation=meta['orientation']).to(device)
        kind = 'selfsup'
    else:
        from flow_matching_cond_particles_crossattn import ParticleCrossAttnFlowNetwork
        model = ParticleCrossAttnFlowNetwork(
            nxi=nxi, nd=nd, D=D,
            predict_orientation=meta['orientation']).to(device)
        kind = 'flow'

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, kind, meta


def describe(kind, meta):
    """Short human-readable configuration string for titles and logs."""
    bits = [f"D={meta['D']}", f"N={meta['n_particles']}", f"R={meta['grid_res']}"]
    if meta.get('orientation'):
        bits.append(f"SE(3) auf {meta['ergodic_on']}")
    if kind == 'selfsup':
        bits.append(f"K={meta['n_candidates']} div={meta['diversity_weight']:g}"
                    if meta['n_candidates'] else 'selbstueberwacht')
    elif meta['lambda_erg']:
        bits.append(f"ergodic loss w={meta['lambda_erg']:g}")
    return ', '.join(bits)


@torch.no_grad()
def generate(model, kind, particles, n_samples, meta, steps, device, seed,
             obstacle=None, obstacle_weight=20.0, obstacle_t_start=0.3,
             polish_steps=250):
    """n trajectories for one conditioning. Returns (cps, rot6d, seconds).

    `rot6d` is None for position-only checkpoints.

    `obstacle` applies the inference-time repulsion. The flow model steers around
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
        cps, rot6d = model.generate(particles, num_samples=n_samples,
                                    device=str(dev), generator=g)
        if obstacle is not None:
            from obstacles import basis_torch, polish_out_of_obstacle
            B = basis_torch(meta['nxi'], 256, 5, device=dev)
            cps = polish_out_of_obstacle(cps, obstacle, B, max_iters=polish_steps)
    else:
        from flow_matching_cond_particles_crossattn import generate_particle_trajectories
        cps, rot6d = generate_particle_trajectories(
            model, particles, num_samples=n_samples,
            nxi=meta['nxi'], nd=meta['nd'], steps=steps, device=str(dev),
            cfg_weight=meta['cfg_weight'], generator=g,
            obstacle=obstacle, obstacle_weight=obstacle_weight,
            obstacle_t_start=obstacle_t_start, polish_steps=polish_steps)

    if dev.type == 'cuda':
        torch.cuda.synchronize()
    return cps, rot6d, time.perf_counter() - t0
