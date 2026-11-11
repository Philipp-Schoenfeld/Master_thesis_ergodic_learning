"""
render_obstacle_demo.py
=======================
Offline MP4 of the 3D active-sensing simulation with obstacles, rendered in MuJoCo.

Runs the same pipeline as the control centre without a GUI: belief -> particle cloud -> flow
matching plan (repelled from the obstacles during inference) -> snapping onto the surface
(repelled again in every refinement step) -> the Panda drives the plan -> measure -> replan.

Two spheres are placed on the path the network would take *without* obstacles; that path is drawn
in blue next to the plan that is actually driven (red = current round, green = earlier rounds).
Both plans use the same seed and the same particle cloud, so the blue line shows exactly what the
obstacles changed.

    python render_obstacle_demo.py --out obstacle_demo.mp4 [--rounds 3] [--split val_form --entry 0]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPL3 = os.path.dirname(_HERE)
if _EXPL3 not in sys.path:
    sys.path.insert(0, _EXPL3)

import paths3d                                                     # noqa: E402

paths3d.setup()
import figures3d as F                                              # noqa: E402
import mission3d as M                                              # noqa: E402
from mujoco_3d import scene3d as S                                 # noqa: E402
from mujoco_3d.ik import TaskPriorityIK                            # noqa: E402
from mujoco_3d.joint_map3d import JointMap3D                       # noqa: E402
from mujoco_3d.run_mujoco_3d import LiveArm3D, PathOverlay3D, _placement_for  # noqa: E402
from obstacles import SphereObstacle, CompositeObstacle            # noqa: E402
from planner3d import Planner3D                                    # noqa: E402
from surface3d import make_lit                                     # noqa: E402

SIM_FPS = 60.0
AGENT_SPEED = 0.6              # figure units / s, as in the control centre
CAPTION = ('grey spheres: obstacles   blue: plan without obstacles   '
           'red: current plan   green: driven')


def arclength(path):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])


def thin(pts, cap):
    pts = np.asarray(pts)
    if len(pts) <= cap:
        return pts
    idx = np.unique(np.append(np.arange(0, len(pts), int(np.ceil(len(pts) / cap))), len(pts) - 1))
    return pts[idx]


def make_mission(planner, fig, obstacle, seed, device):
    a = M.Args(n_particles=planner.n_particles)
    a.snap, a.obstacle = True, obstacle
    return M.Mission3D(planner, fig, a, variant='C', seed=seed, device=device)


def place_obstacles(planner, fig, seed, device, n, radius, margin):
    """Spheres on the unobstructed round-1 plan (evenly spread along it)."""
    torch.manual_seed(seed)
    m = make_mission(planner, fig, None, seed, device)
    mu, sd = m.belief.posterior()
    curve, _, _ = m.plan_round(m.phi_plain(mu, sd, m.args.kappa0))
    arc = arclength(curve)
    fracs = np.linspace(0.0, 1.0, n + 2)[1:-1] if n > 1 else [0.5]
    centres = [curve[int(np.searchsorted(arc, f * arc[-1]))] for f in fracs]
    return [SphereObstacle(center=c, radius=radius, margin=margin) for c in centres]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='obstacle_demo.mp4')
    ap.add_argument('--preview', default=None, help='also save one frame (PNG) here')
    ap.add_argument('--preview_step', type=int, default=700, help='simulation step of the preview frame')
    ap.add_argument('--split', default='val_form')
    ap.add_argument('--entry', type=int, default=0)
    ap.add_argument('--match', default=None,
                    help="pick the first entry of the split whose 'shape|surface' contains this text")
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--n_obstacles', type=int, default=2)
    ap.add_argument('--radius', type=float, default=0.10, help='obstacle radius (figure units)')
    ap.add_argument('--margin', type=float, default=None,
                    help='clearance added to the radius (figure units); default: eraser radius + 15 mm in the scene')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--cam', type=float, nargs=3, default=(215.0, -20.0, 1.25),
                    metavar=('AZIMUTH', 'ELEVATION', 'DISTANCE'))
    ap.add_argument('--max_steps', type=int, default=0, help='stop after this many sim steps (test)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planner = Planner3D(device=device)
    entries = F.list_entries(args.split)
    if args.match:
        entries = [e for e in entries if args.match.lower() in ('%s|%s' % (e['shape'], e['surface'])).lower()]
    entry = entries[args.entry]
    fig = F.build_figure(entry)
    lit = make_lit(fig)
    print('[demo] figure: %s' % fig.title, flush=True)

    # -- scene ---------------------------------------------------------------------
    base_model = S.build_spec().compile()
    placement, reach = _placement_for(entry, fig, lit, base_model)
    model = S.build_model(fig, placement, os.path.join(S.CACHE_DIR, 'scenes'))
    model.vis.global_.offwidth, model.vis.global_.offheight = args.width, args.height
    ik = TaskPriorityIK(model, tool_axis=(0, 0, 1))
    ik.posture = S.home_pose(model).copy()
    pts_w, axes_w = S.surface_samples(lit, placement, 1500)
    jmap = JointMap3D(ik, pts_w, axes_w, S.home_pose(model), n_restarts=3)
    print('[demo] scale %.2f, %.0f%% reachable, joint map %.0f%%'
          % (placement.scale, 100 * reach, 100 * jmap.valid_fraction), flush=True)

    # -- obstacles + mission -----------------------------------------------------------
    margin = args.margin if args.margin is not None else (S.ERASER_RADIUS + 0.015) / placement.scale
    print('[demo] obstacle radius %.3f + clearance %.3f (figure units)' % (args.radius, margin), flush=True)
    obstacles = place_obstacles(planner, fig, args.seed, device, args.n_obstacles, args.radius, margin)
    obs_comp = CompositeObstacle(obstacles)
    obs_rows = np.array([[*o.center, o.radius, o.margin] for o in obstacles])
    print('[demo] obstacles: %s' % [tuple(np.round(o.center, 3)) for o in obstacles], flush=True)
    torch.manual_seed(args.seed)
    mission = make_mission(planner, fig, obs_comp, args.seed, device)
    gen = mission.rounds()

    # -- arm, renderer, writer -----------------------------------------------------------
    n0 = np.asarray(fig.start, dtype=np.float64)
    start_axis = S.limit_axes(placement.direction(-mission.lit.project(n0[None])[1][0]), -S.FACE_DIR)
    arm = LiveArm3D(model, jmap, placement.point(fig.start), np.ravel(start_axis), fps=SIM_FPS)
    overlay = PathOverlay3D(None, placement, lit)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = placement.shift
    cam.azimuth, cam.elevation, cam.distance = args.cam

    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()
    writer = imageio.get_writer(args.out, fps=args.fps, codec='libx264', quality=8,
                                macro_block_size=1, pixelformat='yuv420p')
    every = max(1, int(round(SIM_FPS / args.fps)))
    centres_w = [placement.point(np.array(o.center)) for o in obstacles]
    radii_w = [o.radius * placement.scale for o in obstacles]
    state = dict(step=0, frames=0, min_tip=np.inf, preview_done=False)

    def render_frame(seg, k, ghost, round_txt):
        driven = np.concatenate(mission.driven, axis=0) if mission.driven else np.zeros((0, 3))
        past = thin(driven, 700)
        cur = thin(seg[:max(k, 1)], 300)
        pre = thin(seg[max(k - 1, 0):], 200)
        renderer.update_scene(arm.data, camera=cam)
        overlay.add_geoms(renderer.scene, past, cur, pre, obs_rows, ghost=ghost)
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img, 'RGBA')
        d.rectangle([0, 0, 190, 34], fill=(255, 255, 255, 215))
        d.text((12, 8), round_txt, fill=(26, 26, 46), font=font)
        d.rectangle([0, args.height - 34, args.width, args.height], fill=(255, 255, 255, 215))
        d.text((12, args.height - 27), CAPTION, fill=(26, 26, 46), font=font)
        writer.append_data(np.asarray(img))
        state['frames'] += 1
        return img

    def tick(seg, k, ghost, txt, force=False):
        state['step'] += 1
        tip, _ = arm.tool_state()
        for c, r in zip(centres_w, radii_w):
            state['min_tip'] = min(state['min_tip'], float(np.linalg.norm(tip - c)) - r)
        if force or state['step'] % every == 0:
            img = render_frame(seg, k, ghost, txt)
            if args.preview and not state['preview_done'] and state['step'] >= args.preview_step:
                img.save(args.preview)
                state['preview_done'] = True

    t_start = time.perf_counter()
    dt = 1.0 / SIM_FPS
    seg = ghost = None
    done = False
    for r in range(args.rounds):
        rng_pre = copy.deepcopy(mission.rng.bit_generator.state)
        torch.manual_seed(args.seed + r)
        info = next(gen)                       # also commits the previous round
        seg, axes = info['seg'], info['axes']
        # the plan without obstacles: same rng state, same noise, same particle cloud
        rng_post = copy.deepcopy(mission.rng.bit_generator.state)
        mission.rng.bit_generator.state = rng_pre
        torch.manual_seed(args.seed + r)
        mission.args.obstacle = None
        ghost, _, _ = mission.plan_round(torch.from_numpy(info['phi']).to(device))
        mission.args.obstacle = obs_comp
        mission.rng.bit_generator.state = rng_post
        mission.lit.set_obstacle(obs_comp)
        txt = 'Round %d / %d' % (r + 1, args.rounds)
        print('[demo] %s: plan %d points, %.2f long (%.1fs elapsed)'
              % (txt, len(seg), arclength(seg)[-1], time.perf_counter() - t_start), flush=True)
        arc = arclength(seg)
        s = 0.0
        while s < arc[-1] and not done:
            k = int(min(np.searchsorted(arc, s), len(seg) - 1))
            goal = placement.point(seg[k])
            axis = np.ravel(S.limit_axes(placement.direction(axes[k]), -S.FACE_DIR))
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            arm.step(goal, axis)
            tip, _ = arm.tool_state()
            lag = float(np.linalg.norm(tip - (goal - axis * S.STANDOFF)))
            s += AGENT_SPEED * dt * float(np.clip(1.0 - (lag - 0.015) / 0.05, 0.1, 1.0))
            tick(seg, k, ghost, txt)
            done = bool(args.max_steps and state['step'] >= args.max_steps)
        if done:
            break
    if seg is not None and not done:
        mission.commit(*mission.pending)
        for _ in range(int(1.5 * SIM_FPS)):          # hold the final state for 1.5 s
            arm.step(placement.point(seg[-1]), np.ravel(S.limit_axes(
                placement.direction(axes[-1]), -S.FACE_DIR)))
            tick(seg, len(seg), ghost, 'Done', force=(state['step'] % every == 0))
    writer.close()

    if not mission.driven:
        print('[demo] stopped before the first round was committed')
        return
    driven = np.concatenate(mission.driven, axis=0)
    clear = [float(np.linalg.norm(driven - np.array(o.center), axis=1).min() - o.radius) for o in obstacles]
    print('[demo] driven path vs. obstacles (figure units, clearance to the raw radius): %s'
          % np.round(clear, 3).tolist())
    print('[demo] closest eraser tip to an obstacle surface in the scene: %.1f mm (tool radius 18 mm)'
          % (1e3 * state['min_tip']))
    if not getattr(fig, 'is_volume', False):
        import trimesh
        from scipy.spatial import cKDTree
        dense, _ = trimesh.sample.sample_surface(fig.mesh, 300000, seed=5)
        dist = cKDTree(dense).query(driven.astype(np.float64))[0]
        print('[demo] driven path vs. figure surface: max %.1f mm, mean %.2f mm in the scene (scale %.2f)'
              % (1e3 * dist.max() * placement.scale, 1e3 * dist.mean() * placement.scale, placement.scale))
    print('[demo] surface constraint (mesh_alignment, |SDF| max / mean |cos(t,n)|) of the last plan: %s'
          % {k: round(float(v), 4) for k, v in mission.last_report.items()})
    print('[demo] wrote %s: %d frames, %.1fs of video, %.0fs wall-clock'
          % (args.out, state['frames'], state['frames'] / args.fps, time.perf_counter() - t_start))


if __name__ == '__main__':
    main()
