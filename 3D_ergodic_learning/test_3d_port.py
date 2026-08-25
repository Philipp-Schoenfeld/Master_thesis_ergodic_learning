#!/usr/bin/env python3
r"""
test_3d_port.py
===============
Sanity checks for the 3D port. Runs on CPU in a few seconds, no database and no
checkpoint needed for most of it.

The checks that matter are the ones that would otherwise fail silently:

1. **Degeneracy check.** A 3D energy evaluated on data confined to the plane
   z = const, with only k3 = 0 modes kept, must equal the 2D energy on the same
   data. This is the single most valuable test in the file: it proves the 3D
   basis reduces to the 2D one instead of quietly computing something else.
2. **Shape plumbing.** Every module gets the tensor ranks it expects.
3. **Obstacle gradient.** Finite differences against the analytic gradient, and
   the chain rule through the B-spline basis against autograd.
4. **Symmetry breaking.** A straight line through the sphere centre is the
   degenerate case the polish loop exists for; it must still clear.
"""

import math
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

torch.manual_seed(0)
np.random.seed(0)

PASS, FAIL = "[ok]  ", "[FAIL]"
_failures = []


def check(name, ok, detail=""):
    print(f"{PASS if ok else FAIL} {name:<52} {detail}")
    if not ok:
        _failures.append(name)


# ── 1. Energy reduces to the 2D case on planar data ──────────────────────────
def test_planar_reduction():
    from ergodic_energy_torch import make_k_grid, fourier_basis

    K = 6
    k3, _ = make_k_grid(K, nd=3, exponent=-1.5)
    k2, _ = make_k_grid(K, nd=2, exponent=-1.5)

    # Points confined to z = 0: cos(pi*k3*0) = 1, so the 3D basis restricted to
    # the k3 = 0 slice must equal the 2D basis exactly.
    T = 40
    xy = torch.rand(T, 2, dtype=torch.float64)
    xyz = torch.cat([xy, torch.zeros(T, 1, dtype=torch.float64)], dim=1)

    F3 = fourier_basis(xyz, torch.tensor(k3))          # (T, K^3)
    F2 = fourier_basis(xy, torch.tensor(k2))           # (T, K^2)

    # k3 grid is odometer-ordered with k3 slowest-varying last, so the k3 == 0
    # entries are exactly those whose third index is zero.
    sel = torch.tensor(k3[:, 2] == 0)
    err = (F3[:, sel] - F2).abs().max().item()
    check("3D basis on z=0 equals 2D basis (k3=0 slice)", err < 1e-12,
          f"max |diff| = {err:.2e}")

    # And at z = 0.5 the cos(pi*k3*0.5) factor kills odd k3 but keeps k3=0.
    xyz_h = torch.cat([xy, torch.full((T, 1), 0.5, dtype=torch.float64)], dim=1)
    F3h = fourier_basis(xyz_h, torch.tensor(k3))
    err_h = (F3h[:, sel] - F2).abs().max().item()
    check("same at z=0.5 (the plane the DB is lifted to)", err_h < 1e-12,
          f"max |diff| = {err_h:.2e}")


# ── 2. Lambda exponent convention ────────────────────────────────────────────
def test_lambda():
    from ergodic_energy_torch import make_k_grid, LAMBDA_EXPONENT
    check("Lambda exponent is -(d+1)/2 = -2.0 in 3D",
          abs(LAMBDA_EXPONENT + 2.0) < 1e-12, f"{LAMBDA_EXPONENT}")
    k, lam = make_k_grid(3)
    manual = (1.0 + (k ** 2).sum(1)) ** LAMBDA_EXPONENT
    check("Lambda matches (1+|k|^2)^exponent",
          np.abs(lam - manual).max() < 1e-15)
    check("K^3 modes, not K^2", len(k) == 27, f"{len(k)} modes at K=3")


# ── 3. Shape plumbing through every module ───────────────────────────────────
def test_shapes():
    from flow_matching_cond_particles_crossattn import (
        ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss,
        generate_particle_trajectories)
    from flow_matching_particles_selfsupervised import (
        SelfSupervisedParticleGenerator, compute_selfsupervised_loss)
    from ergodic_energy_torch import ErgodicEnergy
    from obstacles import bspline_basis_matrix

    B, nxi, nd, N, D = 3, 25, 3, 64, 32
    x1 = torch.rand(B, nxi, nd)
    parts = torch.rand(B, N, nd + 1)

    net = ParticleCrossAttnFlowNetwork(nxi=nxi, nd=nd, D=D)
    loss, comps = compute_particle_cfm_loss(net, x1, parts, p_drop=0.1)
    check("CFM net forward + loss", loss.ndim == 0 and 'cfm' in comps,
          f"loss={loss.item():.4f}")

    gen, _ = generate_particle_trajectories(net, parts[0], num_samples=2,
                                            nxi=nxi, nd=nd, steps=3)
    check("CFM generation shape", tuple(gen.shape) == (2, nxi, nd), tuple(gen.shape))

    basis = torch.from_numpy(bspline_basis_matrix(nxi, 100, 5))
    energy = ErgodicEnergy(K=4, basis=basis)
    ssg = SelfSupervisedParticleGenerator(nxi=nxi, nd=nd, D=D)
    phi = torch.rand(B, 4 ** 3)
    sl, sp = compute_selfsupervised_loss(ssg, parts, phi, energy,
                                         n_candidates=2, diversity_weight=1.0)
    check("selfsup loss with K=2 candidates", sl.ndim == 0 and 'diversity' in sp,
          f"E={sp['energy'].item():.2f}")

    out, out_rot = ssg.generate(parts[0], num_samples=4)
    check("selfsup generation shape", tuple(out.shape) == (4, nxi, nd), tuple(out.shape))
    check("no orientation output when the head is off", out_rot is None)

    # And with the head on: the state gains a 6D block end to end.
    ssg_o = SelfSupervisedParticleGenerator(nxi=nxi, nd=nd, D=D,
                                            predict_orientation=True)
    o_cps, o_rot = ssg_o.generate(parts[0], num_samples=4)
    check("selfsup orientation output shape",
          tuple(o_rot.shape) == (4, nxi, 6), tuple(o_rot.shape))

    net_o = ParticleCrossAttnFlowNetwork(nxi=nxi, nd=nd, D=D,
                                         predict_orientation=True)
    x1_o = torch.rand(B, nxi, nd + 6)
    l_o, c_o = compute_particle_cfm_loss(net_o, x1_o, parts, p_drop=0.1)
    check("CFM loss splits position and rotation", 'cfm_rot' in c_o,
          f"pos={c_o['cfm'].item():.3f} rot={c_o['cfm_rot'].item():.3f}")
    g_o, g_rot = generate_particle_trajectories(net_o, parts[0], num_samples=2,
                                                nxi=nxi, nd=nd, steps=3)
    check("CFM generation returns positions and 6D",
          tuple(g_o.shape) == (2, nxi, nd) and tuple(g_rot.shape) == (2, nxi, 6),
          f"{tuple(g_o.shape)} / {tuple(g_rot.shape)}")

    e, terms = energy(x1, phi[:1].expand(B, -1), return_terms=True)
    check("energy returns per-sample vector", tuple(e.shape) == (B,), tuple(e.shape))
    check("energy has 3 terms without obstacle", set(terms) ==
          {'smooth', 'ergodic', 'boundary'})


# ── 4. Obstacle: gradient, chain rule, degenerate polish ─────────────────────
def test_obstacle():
    from obstacles import (SphereObstacle, bspline_basis_matrix,
                           curve_repulsion_grad, polish_out_of_obstacle,
                           max_violation)

    obs = SphereObstacle()
    P = torch.rand(200, 3, dtype=torch.float64, requires_grad=True)

    E = obs.penalty(P)
    E.backward()
    ana = obs.grad_penalty(P.detach())
    err = (P.grad - ana).abs().max().item()
    check("analytic grad matches autograd", err < 1e-12, f"max |diff| = {err:.2e}")

    # Finite differences on a point guaranteed to be inside.
    p = torch.tensor([[0.52, 0.5, 0.5]], dtype=torch.float64)
    g = obs.grad_penalty(p)[0]
    eps, fd = 1e-6, []
    for j in range(3):
        d = torch.zeros(1, 3, dtype=torch.float64); d[0, j] = eps
        fd.append(((obs.penalty(p + d) - obs.penalty(p - d)) / (2 * eps)).item())
    err = max(abs(a - b) for a, b in zip(g.tolist(), fd))
    check("finite differences match", err < 1e-7, f"max |diff| = {err:.2e}")

    nxi, T = 25, 128
    B = torch.from_numpy(bspline_basis_matrix(nxi, T, 5)).double()
    cps = torch.rand(2, nxi, 3, dtype=torch.float64, requires_grad=True)
    curve = torch.einsum('pi,bid->bpd', B, cps)
    obs.penalty(curve).backward()
    manual = curve_repulsion_grad(cps.detach(), obs, B, normalize=False)
    err = (cps.grad - manual).abs().max().item()
    check("chain rule B^T g equals autograd", err < 1e-11, f"max |diff| = {err:.2e}")

    # The degenerate case the symmetry breaker exists for: a straight line
    # through the sphere centre, where the radial gradient pushes along the
    # curve and plain descent stalls.
    cases = {
        'x-axis through centre': (torch.tensor([0.05, 0.5, 0.5]),
                                  torch.tensor([0.95, 0.5, 0.5])),
        'space diagonal':        (torch.tensor([0.05, 0.05, 0.05]),
                                  torch.tensor([0.95, 0.95, 0.95])),
        'yz diagonal':           (torch.tensor([0.5, 0.05, 0.05]),
                                  torch.tensor([0.5, 0.95, 0.95])),
    }
    Bf = torch.from_numpy(bspline_basis_matrix(nxi, T, 5))
    for name, (a, b) in cases.items():
        t = torch.linspace(0, 1, nxi)[:, None]
        line = (a[None] * (1 - t) + b[None] * t)[None]
        before = max_violation(line, obs, Bf)
        after = max_violation(polish_out_of_obstacle(line, obs, Bf), obs, Bf)
        check(f"polish clears: {name}", after <= 1e-6,
              f"{before:.4f} -> {after:.2e}")


# ── 5. Data helpers ──────────────────────────────────────────────────────────
def test_data_helpers():
    from data_3d import _lift_to_plane, augment_batch, Z_PLANE

    xy = np.random.rand(50, 2).astype(np.float32)
    p = _lift_to_plane(xy)
    check("2D lift adds a constant z", p.shape == (50, 3) and
          np.allclose(p[:, 2], Z_PLANE), f"z={p[0, 2]}")

    x = torch.rand(4, 25, 3)
    x[..., 2] = 0.5                                   # planar
    parts = torch.rand(4, 32, 4)
    parts[..., 2] = 0.5
    xa, pa = augment_batch(x, parts, rot_range=20.0, noise_std=0.0,
                           trans_range=0.0, scale_range=(1.0, 1.0), rot_full=False)
    spread = (xa[..., 2] - xa[..., 2].mean(dim=1, keepdim=True)).abs().max().item()
    check("z-rotation keeps planar data planar", spread < 1e-5,
          f"max z spread = {spread:.2e}")
    check("particle mu is preserved", torch.allclose(pa[..., 3], parts[..., 3]))

    xf, _ = augment_batch(x, parts, rot_range=20.0, noise_std=0.0,
                          trans_range=0.0, scale_range=(1.0, 1.0), rot_full=True)
    spread_f = (xf[..., 2] - xf[..., 2].mean(dim=1, keepdim=True)).abs().max().item()
    check("--rot_full does tilt out of plane", spread_f > 1e-3,
          f"max z spread = {spread_f:.3f}")


# ── 6. Planarity metric ──────────────────────────────────────────────────────
def test_planarity():
    from ergodic_energy_torch import planarity

    t = torch.linspace(0, 2 * math.pi, 100)
    flat = torch.stack([0.5 + 0.3 * torch.cos(t),
                        0.5 + 0.3 * torch.sin(t),
                        torch.full_like(t, 0.5)], dim=-1)[None]
    helix = torch.stack([0.5 + 0.3 * torch.cos(t),
                         0.5 + 0.3 * torch.sin(t),
                         torch.linspace(0.2, 0.8, 100)], dim=-1)[None]
    pf, ph = planarity(flat).item(), planarity(helix).item()
    check("planarity ~ 0 for a flat circle", pf < 1e-6, f"{pf:.2e}")
    check("planarity > 0 for a helix", ph > 1e-3, f"{ph:.4f}")


# ── 7. Rotation representation (Stufe 1) ─────────────────────────────────────
def test_rot6d():
    from orientation import (rot6d_to_matrix, matrix_to_rot6d, identity_rot6d,
                             geodesic_angle)

    x = torch.randn(200, 6, dtype=torch.float64)
    R = rot6d_to_matrix(x)
    I = R.transpose(-1, -2) @ R
    orth = (I - torch.eye(3, dtype=torch.float64)).abs().max().item()
    det = torch.linalg.det(R)
    check("6D -> rotation is orthonormal", orth < 1e-12, f"max |R^T R - I| = {orth:.2e}")
    check("6D -> rotation has det +1", (det - 1).abs().max().item() < 1e-12)

    # Round trip: matrix -> 6D -> matrix must be the identity map.
    back = rot6d_to_matrix(matrix_to_rot6d(R))
    err = (back - R).abs().max().item()
    check("matrix -> 6D -> matrix round trip", err < 1e-12, f"max |diff| = {err:.2e}")

    e = identity_rot6d((5,), dtype=torch.float64)
    err = (rot6d_to_matrix(e) - torch.eye(3, dtype=torch.float64)).abs().max().item()
    check("identity encoding decodes to I", err < 1e-15)

    # The head must start at the identity: zero-initialised last layer plus the
    # identity offset, so a freshly built model contributes no rotation at all.
    from flow_matching_cond_particles_crossattn import OrientationHead
    h = OrientationHead(D=32)
    out = h(torch.randn(4, 32, 25))
    err = (rot6d_to_matrix(out) - torch.eye(3)).abs().max().item()
    check("OrientationHead starts at identity", err < 1e-6, f"max |diff| = {err:.2e}")

    # Geodesic angle against a known rotation.
    th = torch.tensor([0.7], dtype=torch.float64)
    c, s = torch.cos(th), torch.sin(th)
    Rz = torch.tensor([[[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]],
                      dtype=torch.float64).reshape(1, 3, 3)
    ang = geodesic_angle(torch.eye(3, dtype=torch.float64)[None], Rz)
    check("geodesic angle matches the known rotation",
          abs(ang.item() - 0.7) < 1e-6, f"{ang.item():.6f} vs 0.7")


# ── 8. Frames (Stufe 0) ──────────────────────────────────────────────────────
def test_frames():
    from orientation import (rmf_frames, lookat_frames, curve_tangents,
                             sensor_axis, geodesic_angle)

    t = torch.linspace(0, 2 * math.pi, 120, dtype=torch.float64)
    circle = torch.stack([0.5 + 0.3 * torch.cos(t),
                          0.5 + 0.3 * torch.sin(t),
                          torch.full_like(t, 0.5)], dim=-1)[None]

    R = rmf_frames(circle)
    I = R.transpose(-1, -2) @ R
    check("RMF frames are orthonormal",
          (I - torch.eye(3, dtype=torch.float64)).abs().max().item() < 1e-9)
    tang = curve_tangents(circle)
    check("RMF body-z is the tangent",
          (sensor_axis(R) - tang).abs().max().item() < 1e-9)

    # The defining property: less accumulated twist than Frenet on a curve where
    # Frenet spins. A straight line is the extreme case — Frenet is undefined,
    # RMF is constant.
    line = torch.stack([torch.linspace(0.1, 0.9, 60, dtype=torch.float64),
                        torch.full((60,), 0.5, dtype=torch.float64),
                        torch.full((60,), 0.5, dtype=torch.float64)], dim=-1)[None]
    Rl = rmf_frames(line)
    drift = geodesic_angle(Rl[:, :-1], Rl[:, 1:]).sum().item()
    check("RMF has no twist on a straight line", drift < 1e-6, f"drift = {drift:.2e}")

    # Look-at: body-z must equal the requested direction exactly.
    d = torch.randn(1, 120, 3, dtype=torch.float64)
    d = d / d.norm(dim=-1, keepdim=True)
    Rk = lookat_frames(circle, d)
    err = (sensor_axis(Rk) - d).abs().max().item()
    check("look-at body-z equals the target direction", err < 1e-9,
          f"max |diff| = {err:.2e}")


# ── 9. Surface field and orientation energy (Stufe 2) ────────────────────────
def test_surface_field():
    from orientation import SurfaceField, lookat_frames, sensor_axis
    from orientation_energy import (pointing_term, standoff_term,
                                    angular_smoothness_term, pointing_error_deg)

    # A slab of occupied voxels around z = 0.5, like the planar targets.
    R = 32
    vol = np.zeros((R, R, R), dtype=np.float32)
    zc = R // 2
    vol[zc - 1:zc + 2, 8:24, 8:24] = 1.0
    field = SurfaceField(vol)

    # A point well above the slab: distance ~ its height, direction ~ -z.
    p = torch.tensor([[[0.5, 0.5, 0.8]]])
    dist = field.distance(p)[0, 0].item()
    dirn = field.direction(p)[0, 0]
    check("distance above the slab is about the height",
          abs(dist - 0.3) < 0.05, f"{dist:.3f} vs ~0.30")
    check("direction from above points down",
          dirn[2].item() < -0.9, f"dz = {dirn[2].item():.3f}")

    # Inside the slab the distance is zero and the fallback direction is used.
    q = torch.tensor([[[0.5, 0.5, 0.5]]])
    check("distance inside the slab is zero", field.distance(q)[0, 0].item() < 1e-6)

    # Pointing term is zero for a perfectly aimed frame and positive otherwise.
    curve = torch.tensor([[[0.5, 0.5, 0.8], [0.55, 0.5, 0.8], [0.6, 0.5, 0.8]]])
    d = field.direction(curve)
    Rg = lookat_frames(curve, d)
    good = pointing_term(Rg, d, w=1.0).item()
    check("pointing term is 0 for a perfectly aimed frame", good < 1e-6,
          f"{good:.2e}")
    flipped = lookat_frames(curve, -d)
    bad = pointing_term(flipped, d, w=1.0).item()
    check("pointing term is 2 per point when reversed",
          abs(bad - 2 * curve.shape[1]) < 1e-5, f"{bad:.3f}")
    check("pointing error in degrees agrees",
          abs(pointing_error_deg(Rg, d).item()) < 1e-3)

    # Standoff: inside the band costs nothing, outside costs quadratically.
    inside = standoff_term(torch.full((1, 10), 0.12), target=0.12, band=0.03, w=1.0)
    outside = standoff_term(torch.full((1, 10), 0.30), target=0.12, band=0.03, w=1.0)
    check("standoff is free inside the band", inside.item() < 1e-9)
    check("standoff grows outside the band",
          abs(outside.item() - 0.5 * (0.15 ** 2) * 10) < 1e-6, f"{outside.item():.4f}")

    # Angular smoothness: zero for a constant rotation sequence.
    const = torch.eye(3).expand(1, 12, 3, 3)
    check("angular smoothness is 0 for a constant frame",
          angular_smoothness_term(const, w=1.0).item() < 1e-9)

    # Footprint depends on orientation — the property that lets orientation
    # influence coverage rather than only decorate it.
    a_down = torch.tensor([[[0., 0., -1.]]])
    a_side = torch.tensor([[[1., 0., 0.]]])
    fp1 = field.footprint(p, a_down)
    fp2 = field.footprint(p, a_side)
    check("footprint moves when the sensor tilts",
          (fp1 - fp2).abs().max().item() > 0.1,
          f"shift = {(fp1 - fp2).norm().item():.3f}")


# ── 10. SE3Energy composition ────────────────────────────────────────────────
def test_se3_energy():
    from orientation import SurfaceField, matrix_to_rot6d, lookat_frames
    from orientation_energy import SE3Energy
    from ergodic_energy_torch import ErgodicEnergy
    from obstacles import bspline_basis_matrix

    R = 32
    vol = np.zeros((R, R, R), dtype=np.float32)
    vol[R // 2 - 1:R // 2 + 2, 8:24, 8:24] = 1.0
    field = SurfaceField(vol)

    nxi, T = 25, 60
    basis = torch.from_numpy(bspline_basis_matrix(nxi, T, 5))
    base = ErgodicEnergy(K=4, basis=basis)
    e_se3 = SE3Energy(base=base, ergodic_on='position')

    cps = torch.rand(3, nxi, 3)
    phi = torch.rand(3, 4 ** 3)

    # Without orientation the composed energy must equal the base energy
    # exactly — otherwise enabling the module would silently change results.
    a = base(cps, phi)
    b = e_se3(cps, phi)
    check("SE3Energy without orientation equals ErgodicEnergy",
          (a - b).abs().max().item() < 1e-5, f"max |diff| = {(a-b).abs().max():.2e}")

    curve = torch.einsum('ti,bid->btd', basis, cps)
    rot6d = matrix_to_rot6d(lookat_frames(cps, field.direction(cps)))
    tot, terms = e_se3(cps, phi, rot6d=rot6d, field=field, return_terms=True)
    check("SE3Energy adds the three orientation terms",
          {'point', 'standoff', 'angsmooth'} <= set(terms), sorted(terms))
    check("SE3Energy total is finite", bool(torch.isfinite(tot).all()))

    # 'footprint' must actually change the ergodic term, or the option is a lie.
    e_fp = SE3Energy(base=base, ergodic_on='footprint')
    _, t_fp = e_fp(cps, phi, rot6d=rot6d, field=field, return_terms=True)
    diff = (terms['ergodic'] - t_fp['ergodic']).abs().max().item()
    check("ergodic_on='footprint' changes the coverage term", diff > 1e-6,
          f"max |diff| = {diff:.3f}")


# ── 11. Augmentation carries the frames ──────────────────────────────────────
def test_augment_orientation():
    from data_3d import augment_batch
    from orientation import rot6d_to_matrix, identity_rot6d, sensor_axis

    B, nxi = 4, 25
    pos = torch.rand(B, nxi, 3)
    rot = identity_rot6d((B, nxi))
    x = torch.cat([pos, rot], dim=-1)
    parts = torch.rand(B, 32, 4)

    xa, _ = augment_batch(x, parts, rot_range=45.0, noise_std=0.0,
                          trans_range=0.0, scale_range=(1.0, 1.0), p_flip=0.0)
    check("augmented state keeps its 9 channels", xa.shape[-1] == 9, xa.shape)

    Ra = rot6d_to_matrix(xa[..., 3:])
    I = Ra.transpose(-1, -2) @ Ra
    check("rotated frames stay orthonormal",
          (I - torch.eye(3)).abs().max().item() < 1e-5)

    # A z-rotation must leave body-z untouched and turn body-x by the same angle
    # the positions turned — that consistency is the whole point.
    az = sensor_axis(Ra)
    check("z-rotation leaves the body-z axis alone",
          (az - torch.tensor([0., 0., 1.])).abs().max().item() < 1e-5,
          f"max |diff| = {(az - torch.tensor([0., 0., 1.])).abs().max():.2e}")


def test_geodesic_angle_gradient():
    """The backward pass at identical rotations.

    Regression test for a NaN that only showed up through training: the forward
    value was a correct 0 and every printed term looked healthy, then every
    weight in the network turned NaN after a single optimiser step. The cause
    was d/dx sqrt(x) at x = 0 inside `geodesic_angle`, reached whenever two
    consecutive frames coincide — which is exactly the zero-initialised state
    every `--orientation` run starts from, so it fired on step one, every time.
    """
    from orientation import (geodesic_angle, rot6d_to_matrix, identity_rot6d,
                             rot_path_length)
    from orientation_energy import angular_smoothness_term

    x = identity_rot6d((1, 8)).clone().requires_grad_(True)
    R = rot6d_to_matrix(x)
    ang = geodesic_angle(R[:, :-1], R[:, 1:])
    check("identical frames report ~zero angle", float(ang.abs().max()) < 1e-5,
          f"max {float(ang.abs().max()):.2e} rad")
    ang.sum().backward()
    check("geodesic_angle gradient is finite at identical frames",
          bool(torch.isfinite(x.grad).all()))

    # The two consumers of it, along the same path.
    for name, fn in (("rot_path_length", rot_path_length),
                     ("angular_smoothness_term", angular_smoothness_term)):
        y = identity_rot6d((1, 8)).clone().requires_grad_(True)
        fn(rot6d_to_matrix(y)).sum().backward()
        check(f"{name} gradient is finite at identical frames",
              bool(torch.isfinite(y.grad).all()))

    # Accuracy must not have been traded away for the fix.
    th = 0.7
    c, s_ = math.cos(th), math.sin(th)
    Ra = torch.eye(3)[None, None]
    Rb = torch.tensor([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])[None, None]
    err = abs(float(geodesic_angle(Ra, Rb)) - th)
    check("a real angle is still exact", err < 1e-6, f"error {err:.2e} rad")



# ── 10. Orientation rendering (viz_3d.draw_frames) ───────────────────────────
def _render_rgb(draw, figsize=(3.2, 3.2), dpi=70):
    """Draw onto a fresh 3D axis and return the RGB buffer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    draw(ax)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _orange_mask(rgb, tol=90):
    """Pixels close to SENSOR_ORANGE (#EF6C00).

    Not a sufficient test on its own where the density is drawn: inferno passes
    straight through orange, so a plain colour match also picks up the target
    volume. On a bare axis it is unambiguous; over a full panel use
    `_added_orange` instead, which is what the failing first version of this
    test taught us.
    """
    import viz_3d
    tgt = np.array([int(viz_3d.SENSOR_ORANGE[i:i + 2], 16) for i in (1, 3, 5)])
    return (np.abs(rgb.astype(int) - tgt).sum(axis=-1) < tol)


def _added_orange(with_rgb, without_rgb):
    """Orange pixels present only in the first render.

    Differencing the two isolates the arrows from anything orange the rest of
    the panel already contained.
    """
    return (_orange_mask(with_rgb) & ~_orange_mask(without_rgb)).sum()


def test_draw_frames():
    """Rendering checks for the sensor-axis arrows.

    This is the code path that only ever runs once a `--orientation` run reaches
    its first visualisation step, hours into training. Everything else about the
    orientation stack is covered numerically above; what is left is whether the
    arrows actually reach the canvas, sit on the curve, and follow R.
    """
    import viz_3d

    T = 64
    s = np.linspace(0, 1, T)
    curve = np.stack([0.2 + 0.6 * s, 0.5 + 0.0 * s, 0.5 + 0.0 * s], axis=1)

    def R_const(vec):
        """(T, 3, 3) frames whose column 2 is `vec` everywhere."""
        v = np.asarray(vec, dtype=float)
        v = v / np.linalg.norm(v)
        a = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e0 = np.cross(a, v); e0 /= np.linalg.norm(e0)
        e1 = np.cross(v, e0)
        return np.tile(np.stack([e0, e1, v], axis=1), (T, 1, 1))

    # None must be a no-op, not a crash: that is the branch every non-SE(3) run
    # takes, and the existing 3D visualisation relies on it.
    empty = _render_rgb(lambda ax: viz_3d.draw_frames(ax, curve, None))
    check("draw_frames(R=None) draws nothing", _orange_mask(empty).sum() == 0)

    up = _render_rgb(lambda ax: viz_3d.draw_frames(ax, curve, R_const([0, 0, 1])))
    n_up = _orange_mask(up).sum()
    check("arrows reach the canvas", n_up > 0, f"{n_up} orange px")

    # Direction must come from R, not from the curve: same curve, opposite axis.
    down = _render_rgb(lambda ax: viz_3d.draw_frames(ax, curve, R_const([0, 0, -1])))
    ys_up = np.where(_orange_mask(up))[0]
    ys_dn = np.where(_orange_mask(down))[0]
    check("arrow direction follows R (up vs down differ)",
          abs(ys_up.mean() - ys_dn.mean()) > 3.0,
          f"mean row {ys_up.mean():.1f} vs {ys_dn.mean():.1f}")

    # `axis` must select a column of R, so asking for a different column of the
    # same frames must change the picture.
    a2 = _render_rgb(lambda ax: viz_3d.draw_frames(ax, curve, R_const([0, 0, 1]), axis=2))
    a0 = _render_rgb(lambda ax: viz_3d.draw_frames(ax, curve, R_const([0, 0, 1]), axis=0))
    check("axis= selects a different column of R",
          not np.array_equal(_orange_mask(a2), _orange_mask(a0)))

    # Subsampling: n_arrows controls how many are drawn, and a curve shorter
    # than n_arrows must not over-draw or index out of range.
    few = _render_rgb(lambda ax: viz_3d.draw_frames(ax, curve, R_const([0, 0, 1]), n_arrows=3))
    many = _render_rgb(lambda ax: viz_3d.draw_frames(ax, curve, R_const([0, 0, 1]), n_arrows=40))
    check("n_arrows controls arrow count",
          _orange_mask(few).sum() < _orange_mask(many).sum(),
          f"{_orange_mask(few).sum()} px vs {_orange_mask(many).sum()} px")

    short = np.stack([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]])
    Rs = R_const([0, 0, 1])[:2]
    ok = True
    try:
        _render_rgb(lambda ax: viz_3d.draw_frames(ax, short, Rs, n_arrows=14))
    except Exception as e:
        ok = False
        detail = repr(e)
    check("curve shorter than n_arrows does not raise", ok,
          "" if ok else detail)

    # Arrows must sit on the curve. Render the curve alone and the arrows alone,
    # then compare their horizontal extent: a wrong position argument (a common
    # quiver slip) puts them somewhere else entirely.
    only_curve = _render_rgb(
        lambda ax: ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                           color='#000000', lw=2))
    dark = (only_curve.astype(int).sum(axis=-1) < 200)
    xs_curve = np.where(dark)[1]
    xs_arrow = np.where(_orange_mask(up))[1]
    overlap = (xs_arrow.min() >= xs_curve.min() - 12 and
               xs_arrow.max() <= xs_curve.max() + 12)
    check("arrows sit along the curve, not elsewhere", overlap,
          f"curve x[{xs_curve.min()},{xs_curve.max()}] "
          f"arrows x[{xs_arrow.min()},{xs_arrow.max()}]")


def test_panel_with_orientation():
    """The wiring the runners actually use: panel(gen_R=...) end to end."""
    import viz_3d

    nxi, T = 25, 128
    cps = np.random.RandomState(0).rand(1, nxi, 3) * 0.6 + 0.2
    curve = viz_3d.cp_to_bspline(cps[0], pts=T)
    R = np.tile(np.eye(3), (T, 1, 1))
    particles = np.concatenate(
        [np.random.RandomState(1).rand(64, 3), np.ones((64, 1))], axis=1)
    volume = np.zeros((8, 8, 8), dtype=np.float32); volume[3:5, 3:5, 3:5] = 1.0

    def draw(ax, gen_R):
        viz_3d.panel(ax, base=None, gen_cps=cps, particles=particles,
                     volume=volume, title='t', bspline_pts=T, gen_R=gen_R)

    with_R = _render_rgb(lambda ax: draw(ax, R), figsize=(4, 4))
    without = _render_rgb(lambda ax: draw(ax, None), figsize=(4, 4))

    added = _added_orange(with_R, without)
    check("panel(gen_R=...) adds arrows over the full panel", added > 0,
          f"{added} px orange only with gen_R")
    check("panel(gen_R=None) adds none the other way round",
          _added_orange(without, with_R) == 0)

    # A second render with the same arguments must be identical, or the check
    # above is measuring plot jitter rather than the arrows.
    again = _render_rgb(lambda ax: draw(ax, None), figsize=(4, 4))
    check("panel rendering is deterministic", np.array_equal(without, again))


# ── 11. Stufe 3: orientation as a CFM training objective ─────────────────────
def _fake_batch(B=3, nxi=25, N=96, seed=0):
    """A 9-channel batch and a particle cloud whose mass sits on a z-plane."""
    g = torch.Generator().manual_seed(seed)
    from orientation import identity_rot6d
    pos = torch.rand(B, nxi, 3, generator=g) * 0.5 + 0.25
    x1 = torch.cat([pos, identity_rot6d((B, nxi))], dim=-1)
    q = torch.rand(B, N, 3, generator=g)
    q[..., 2] = 0.5                                   # planar target
    mu = torch.ones(B, N, 1)
    return x1, torch.cat([q, mu], dim=-1)


def test_particle_surface():
    """The surface must come from the particles, and mu must gate occupancy."""
    from orientation_energy import ParticleSurface

    # One occupied particle at a known place; distance and direction are exact.
    q = torch.tensor([[[0.5, 0.5, 0.5], [0.9, 0.9, 0.9]]])
    mu = torch.tensor([[[1.0], [0.01]]])              # second one is empty space
    parts = torch.cat([q, mu], dim=-1)
    surf = ParticleSurface(parts, mu_thresh=0.5)

    p = torch.tensor([[[0.5, 0.5, 0.2]]])             # 0.3 below the occupied one
    check("ParticleSurface distance is exact",
          abs(float(surf.distance(p)) - 0.3) < 1e-6, f"{float(surf.distance(p)):.4f}")
    d = surf.direction(p)[0, 0]
    check("ParticleSurface direction points at the occupied particle",
          bool(torch.allclose(d, torch.tensor([0., 0., 1.]), atol=1e-6)),
          f"{d.tolist()}")

    # Lowering the threshold admits the low-mu particle and changes the answer,
    # which is what proves mu is actually consulted.
    loose = ParticleSurface(parts, mu_thresh=0.001)
    p2 = torch.tensor([[[0.95, 0.95, 0.95]]])
    check("mu_thresh gates which particles count",
          float(loose.distance(p2)) < float(surf.distance(p2)),
          f"loose {float(loose.distance(p2)):.3f} vs strict {float(surf.distance(p2)):.3f}")

    # Differentiable in the query point — the whole reason it exists.
    p3 = torch.tensor([[[0.5, 0.5, 0.2]]], requires_grad=True)
    surf.distance(p3).sum().backward()
    check("ParticleSurface is differentiable in the query point",
          bool(torch.isfinite(p3.grad).all()) and float(p3.grad.abs().max()) > 0)


def test_orientation_loss():
    from orientation_energy import OrientationLoss

    x1, parts = _fake_batch()
    L = OrientationLoss(nxi=25, pts=64, weight=1.0, t_power=0.0)
    t = torch.ones(x1.shape[0])

    val, comp = L(x1, parts, t, return_parts=True)
    check("OrientationLoss returns its three terms",
          set(comp) == {'point', 'standoff', 'angsmooth'}, str(sorted(comp)))
    check("OrientationLoss value is finite", bool(torch.isfinite(val)),
          f"{float(val):.4f}")

    # Gradient must reach both blocks: positions (through standoff) and the
    # rotation block (through pointing). A term that only moves one of the two
    # would look healthy in the log and do half the job.
    x = x1.clone().requires_grad_(True)
    L(x, parts, t).backward()
    g_pos = x.grad[..., :3].abs().max()
    g_rot = x.grad[..., 3:].abs().max()
    check("gradient reaches the position block", float(g_pos) > 0, f"{float(g_pos):.3e}")
    check("gradient reaches the rotation block", float(g_rot) > 0, f"{float(g_rot):.3e}")

    # Pointing at the surface must score better than pointing away from it.
    from orientation import matrix_to_rot6d
    B, nxi = x1.shape[0], x1.shape[1]
    down = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    up = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    x_at = x1.clone(); x_away = x1.clone()
    # Curve sits above the plane, so the surface is in -z.
    x_at[..., 2] = 0.62; x_away[..., 2] = 0.62
    x_at[..., 3:] = matrix_to_rot6d(up).expand(B, nxi, 6)
    x_away[..., 3:] = matrix_to_rot6d(down).expand(B, nxi, 6)
    p_at = L.terms(x_at, parts)['point'].mean()
    p_away = L.terms(x_away, parts)['point'].mean()
    check("pointing at the surface beats pointing away",
          float(p_at) < float(p_away), f"{float(p_at):.3f} vs {float(p_away):.3f}")

    # Standoff: inside the band costs nothing, outside costs. Needs a *dense*
    # target, or the nearest particle sits diagonally rather than straight below
    # and the measured distance exceeds the intended height by the in-plane gap.
    gx, gy = torch.meshgrid(torch.linspace(0, 1, 40), torch.linspace(0, 1, 40),
                            indexing='ij')
    dense_q = torch.stack([gx.reshape(-1), gy.reshape(-1),
                           torch.full((1600,), 0.5)], dim=-1)[None].expand(B, -1, -1)
    dense = torch.cat([dense_q, torch.ones(B, 1600, 1)], dim=-1)

    x_in = x1.clone(); x_in[..., 2] = 0.5 + L.standoff_target
    x_out = x1.clone(); x_out[..., 2] = 0.5 + L.standoff_target + 5 * L.standoff_band
    d_in = float(L.terms(x_in, dense)['standoff'].mean())
    d_out = float(L.terms(x_out, dense)['standoff'].mean())
    check("standoff is free inside the band", d_in < 1e-3, f"cost {d_in:.2e}")
    check("standoff charges outside the band", d_out > 1.0, f"cost {d_out:.1f}")


def test_footprint_coupling():
    """Without it, orientation cannot influence coverage at all."""
    from ergodic_metric import ErgodicLoss
    from orientation import matrix_to_rot6d

    x1, parts = _fake_batch()
    B, nxi = x1.shape[0], x1.shape[1]
    rot_a = matrix_to_rot6d(torch.eye(3)).expand(B, nxi, 6).contiguous()
    tilt = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    rot_b = matrix_to_rot6d(tilt).expand(B, nxi, 6).contiguous()

    pos = ErgodicLoss(nxi=nxi, K=4, pts=64, nd=3, ergodic_on='position')
    e_a = pos.coverage_error(x1[..., :3], parts, rot6d=rot_a)
    e_b = pos.coverage_error(x1[..., :3], parts, rot6d=rot_b)
    check("ergodic_on='position' ignores orientation entirely",
          bool(torch.allclose(e_a, e_b)), "identical, as it must be")

    fp = ErgodicLoss(nxi=nxi, K=4, pts=64, nd=3, ergodic_on='footprint')
    f_a = fp.coverage_error(x1[..., :3], parts, rot6d=rot_a)
    f_b = fp.coverage_error(x1[..., :3], parts, rot6d=rot_b)
    check("ergodic_on='footprint' makes coverage depend on orientation",
          not bool(torch.allclose(f_a, f_b)),
          f"{float(f_a.mean()):.4f} vs {float(f_b.mean()):.4f}")

    check("footprint without rot6d is refused, not silently wrong",
          _raises(lambda: fp.coverage_error(x1[..., :3], parts)))

    # And it must stay differentiable through the rotation.
    r = rot_a.clone().requires_grad_(True)
    fp.coverage_error(x1[..., :3], parts, rot6d=r).sum().backward()
    check("footprint coverage is differentiable in the rotation",
          bool(torch.isfinite(r.grad).all()) and float(r.grad.abs().max()) > 0,
          f"|grad|max {float(r.grad.abs().max()):.3e}")


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


def test_cfm_loss_integration():
    """The defaults must reproduce the previous loss exactly."""
    from flow_matching_cond_particles_crossattn import (
        ParticleCrossAttnFlowNetwork, compute_particle_cfm_loss)

    torch.manual_seed(0)
    x1, parts = _fake_batch(B=2, nxi=25, N=64)
    model = ParticleCrossAttnFlowNetwork(nxi=25, nd=3, D=32,
                                         predict_orientation=True)

    torch.manual_seed(1)
    a, pa = compute_particle_cfm_loss(model, x1, parts)
    torch.manual_seed(1)
    b, pb = compute_particle_cfm_loss(model, x1, parts, orientation=None,
                                      w_cfm_rot=1.0)
    check("default call is unchanged by the new arguments",
          torch.allclose(a, b), f"{float(a):.6f} vs {float(b):.6f}")
    check("no orientation term appears by default", 'ori' not in pa,
          str(sorted(pa)))

    from orientation_energy import OrientationLoss
    ori = OrientationLoss(nxi=25, pts=64, weight=1.0)
    torch.manual_seed(1)
    c, pc = compute_particle_cfm_loss(model, x1, parts, orientation=ori)
    check("orientation term is added and logged per component",
          'ori' in pc and 'ori_point' in pc and 'ori_standoff' in pc,
          str(sorted(k for k in pc if k.startswith('ori'))))
    check("adding the orientation term changes the loss",
          not torch.allclose(a, c), f"{float(a):.4f} -> {float(c):.4f}")

    torch.manual_seed(1)
    d, _ = compute_particle_cfm_loss(model, x1, parts, w_cfm_rot=0.0)
    check("w_cfm_rot=0 drops the imitation term", float(d) < float(a),
          f"{float(a):.4f} -> {float(d):.4f}")


if __name__ == '__main__':
    print("\n=== 3D port sanity checks ===\n")
    test_planar_reduction()
    test_lambda()
    test_shapes()
    test_obstacle()
    test_data_helpers()
    test_planarity()
    print("\n--- orientation (Stufe 0 / 1 / 2) ---\n")
    test_rot6d()
    test_frames()
    test_surface_field()
    test_se3_energy()
    test_augment_orientation()
    test_geodesic_angle_gradient()
    print("\n--- orientation rendering ---\n")
    test_draw_frames()
    test_panel_with_orientation()
    print("\n--- Stufe 3: orientation as a CFM objective ---\n")
    test_particle_surface()
    test_orientation_loss()
    test_footprint_coupling()
    test_cfm_loss_integration()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED: {_failures}")
        sys.exit(1)
    print("All checks passed.")
