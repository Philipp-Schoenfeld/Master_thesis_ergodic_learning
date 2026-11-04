"""
scene3d.py
==========
MuJoCo scene for the 3D active-sensing simulation: the Panda holding the
eraser tool, plus the target figure.

The figure is the triangle mesh of the held-out target form (exactly as the
dataset defines it), rigidly rotated, scaled and shifted into the robot's
workspace so that the side the density was projected onto faces the robot.  The
projected target density is drawn on that side as coloured points.  A scene is
built per figure (mesh baked in world coordinates); the viewer is relaunched
when the figure changes.
"""

import os
import sys

import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPL3 = os.path.dirname(_HERE)
if _EXPL3 not in sys.path:
    sys.path.insert(0, _EXPL3)

from mujoco_3d.ik import TaskPriorityIK                    # noqa: E402

CACHE_DIR = os.path.join(_HERE, 'cache')

# -- Eraser tool (same geometry as the 2D scene) --------------------------------
ERASER_RADIUS = 0.018
ERASER_HALF_LEN = 0.035
ERASER_CENTRE = 0.095
ERASER_TIP = ERASER_CENTRE + ERASER_HALF_LEN
GRIPPER_CTRL = ERASER_RADIUS / 0.04 * 255.0
N_ARM_JOINTS = 7

#: Gap between the eraser tip and the figure surface (m).
STANDOFF = 0.006
#: Panda joint velocity limits (rad/s).
JOINT_VMAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]) * float(os.environ.get('MJ3D_VSCALE', 1.0))
MAX_TIP_SPEED = 0.20
#: Largest tool re-orientation rate (rad/s).
MAX_AXIS_RATE = float(os.environ.get('MJ3D_AXRATE', 3.0))

#: Density sites: radius (m), lift above the surface (m), minimum density shown.
SITE_RADIUS = 0.0035
SITE_LIFT = 0.0008
MIN_DENSITY_SHOWN = 0.03
#: Cube workspace: edge thickness (m) and the most density sites drawn (strongest first).
CUBE_EDGE_RADIUS = 0.0025
MAX_VOLUME_SITES = 3000


def _quat_z_to(direction):
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, np.asarray(direction, dtype=np.float64))
    return quat


def _white_inferno_lut(n=256):
    import matplotlib
    inf = matplotlib.colormaps['inferno'](np.linspace(0, 1, n))[:, :3]
    blend = 60
    ramp = np.linspace(0, 1, blend)[:, None]
    inf[:blend] = (1 - ramp) * 1.0 + ramp * inf[:blend]
    return inf


LUT = _white_inferno_lut()


def build_spec(mesh_path=None, sites=None, edges=None):
    """Panda + eraser; optionally the figure mesh (world-frame OBJ), density
    sites ``sites = (positions (K,3), rgba (K,4))`` and the edges of a cube
    workspace ``edges = [(start, end), ...]`` (world frame)."""
    panda_scene = os.path.join(_HERE, 'menagerie', 'franka_emika_panda', 'scene.xml')
    spec = mujoco.MjSpec.from_file(panda_scene)

    hand = next(b for b in spec.bodies if b.name == 'hand')
    direction = np.array([0.0, 0.0, 1.0])
    quat = _quat_z_to(direction)
    eraser = hand.add_geom()
    eraser.name = 'eraser_geom'
    eraser.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    eraser.size = [ERASER_RADIUS, ERASER_HALF_LEN, 0]
    eraser.pos = (ERASER_CENTRE * direction).tolist()
    eraser.quat = quat.tolist()
    eraser.rgba = [0.93, 0.36, 0.51, 1.0]
    eraser.contype = 0
    eraser.conaffinity = 0
    felt = hand.add_geom()
    felt.name = 'eraser_pad'
    felt.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    felt.size = [ERASER_RADIUS * 1.02, 0.004, 0]
    felt.pos = ((ERASER_TIP - 0.004) * direction).tolist()
    felt.quat = quat.tolist()
    felt.rgba = [0.18, 0.18, 0.22, 1.0]
    felt.contype = 0
    felt.conaffinity = 0
    tip = hand.add_site()
    tip.name = 'eraser_tip'
    tip.type = mujoco.mjtGeom.mjGEOM_SPHERE
    tip.size = [0.004, 0.004, 0.004]
    tip.pos = (ERASER_TIP * direction).tolist()
    tip.quat = quat.tolist()
    tip.rgba = [0.0, 0.78, 0.33, 1.0]

    if mesh_path is not None:
        mesh = spec.add_mesh()
        mesh.name = 'figure'
        mesh.file = mesh_path
        body = spec.worldbody.add_body()
        body.name = 'figure_body'
        g = body.add_geom()
        g.name = 'figure_mesh'
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = 'figure'
        g.rgba = [0.78, 0.80, 0.84, 1.0]
        g.contype = 0
        g.conaffinity = 0
    if edges is not None:
        body = spec.worldbody.add_body()
        body.name = 'cube_body'
        for a, b in edges:
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.size = [CUBE_EDGE_RADIUS, 0, 0]
            g.fromto = [*a, *b]
            g.rgba = [0.45, 0.47, 0.55, 0.9]
            g.contype = 0
            g.conaffinity = 0
    if sites is not None:
        pos, rgba = sites
        body = spec.worldbody.add_body()
        body.name = 'density_body'
        for k in range(len(pos)):
            s = body.add_site()
            s.type = mujoco.mjtGeom.mjGEOM_SPHERE
            s.size = [SITE_RADIUS, 0, 0]
            s.pos = pos[k].tolist()
            s.rgba = rgba[k].tolist()
    return spec


# -- Placement ------------------------------------------------------------------
def rotation_between(a, b):
    """Smallest rotation matrix taking unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        helper = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


class Placement:
    """Rigid map figure frame -> world: ``x_w = scale * R @ (x - c) + t``."""

    def __init__(self, R, scale, centre, shift):
        self.R, self.scale = np.asarray(R, float), float(scale)
        self.centre, self.shift = np.asarray(centre, float), np.asarray(shift, float)

    def point(self, x):
        return self.scale * (np.asarray(x, float) - self.centre) @ self.R.T + self.shift

    def direction(self, d):
        return np.asarray(d, float) @ self.R.T


#: Where the figure may sit (world frame, m); tried in this order of preference.
POSITIONS = ((0.60, 0.70), (0.55, 0.62))      # (x, z) of the figure centre
SCALE_CANDIDATES = (0.64, 0.56, 0.48, 0.40, 0.32, 0.26)
#: A placement counts as usable when its joint map covers at least this share of the lit surface.
MIN_MAP_COVER = 0.85
#: Direction (world) the lit side should face: towards the robot.
FACE_DIR = np.array([-1.0, 0.0, 0.0])
#: The tool axis may deviate at most this far (deg) from the direction towards the
#: robot's side.  0 = constant axis (what the arm tracks best, measured: mean tip
#: error 4-15 mm vs 10-50 mm with 20 deg); larger values follow the surface normal
#: more closely but the arm loses the path at the edges of thick figures.
MAX_TILT_DEG = float(os.environ.get('MJ3D_CONE', 0.0))


def figure_rotation(view):
    """Rotation that makes the lit side (outward normals ~ -view) face the robot."""
    return rotation_between(-np.asarray(view, float), FACE_DIR)


def limit_axes(axes, ref, max_deg=None):
    """Clamp unit vectors ``axes`` (..., 3) into the cone of half-angle
    ``max_deg`` around the unit vector ``ref``."""
    max_deg = MAX_TILT_DEG if max_deg is None else max_deg
    a = np.atleast_2d(np.asarray(axes, float))
    ref = np.asarray(ref, float) / np.linalg.norm(ref)
    cos = np.clip(a @ ref, -1.0, 1.0)
    ang = np.arccos(cos)
    lim = np.radians(max_deg)
    out = a.copy()
    over = ang > lim
    if over.any():
        perp = a[over] - cos[over, None] * ref
        perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
        out[over] = np.cos(lim) * ref + np.sin(lim) * perp
    return out.reshape(np.shape(axes))


def servo_lead_time(model, n_joints=N_ARM_JOINTS):
    """Look-ahead that cancels the position servos' velocity lag (kd / kp)."""
    kp = np.asarray(model.actuator_gainprm[:n_joints, 0], dtype=np.float64)
    kd = -np.asarray(model.actuator_biasprm[:n_joints, 2], dtype=np.float64)
    valid = kp > 0
    return float(np.median(kd[valid] / kp[valid])) if valid.any() else 0.0


def home_pose(model):
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    return np.asarray(model.key_qpos[key][:N_ARM_JOINTS], dtype=np.float64).copy()


def reachable_fraction(ik, q_home, pts_w, axes_w):
    """Fraction of sample points reachable with the tool along ``axes_w``
    (position within 2 mm, axis within 3 deg)."""
    ok = 0
    q = np.asarray(q_home, float).copy()
    for p, a in zip(pts_w, axes_w):
        q2, pe, ae = ik.solve(q, p - a * STANDOFF, a, iterations=40, max_step=0.3)
        if pe < 2e-3 and ae < np.radians(3.0):
            ok += 1
            q = q2
        else:
            q = np.asarray(q_home, float).copy()
    return ok / max(len(pts_w), 1)


def surface_samples(lit, placement, n, seed=0):
    """World-frame points on the lit surface and the (cone-limited) tool axes there."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(lit.pts), size=min(n, len(lit.pts)), replace=False)
    pts = placement.point(lit.pts[pick])
    axes = limit_axes(-placement.direction(lit.nrm[pick]), -FACE_DIR)
    return pts, axes


def choose_placement(model, figure, lit, n_sample=300, verbose=False):
    """Best placement: fraction of the lit surface covered by a smooth joint map
    Prefers the largest size whose map is usable (>= MIN_MAP_COVER), else the best coverage.  (A joint map is the right test: it fails where the arm
    cannot follow the surface on one IK branch, not just where a point is reachable
    from some seed.)"""
    from mujoco_3d.joint_map3d import JointMap3D
    q_home = home_pose(model)
    ik = TaskPriorityIK(model, tool_axis=(0, 0, 1))
    ik.posture = q_home.copy()
    R = figure_rotation(figure.view)
    centre = 0.5 * (figure.pts.min(axis=0) + figure.pts.max(axis=0))
    best = None
    # a shrunk figure (figure.shrink < 1) keeps its smaller real-world size: the sizes the
    # full-size figure would have been given are scaled down with it
    cap = SCALE_CANDIDATES[0] * getattr(figure, 'shrink', 1.0) + 1e-9
    for scale in [c for c in SCALE_CANDIDATES if c <= cap] or SCALE_CANDIDATES[-1:]:   # big to small: first usable size wins
        for x, z in POSITIONS:
            pl = Placement(R, scale, centre, np.array([x, 0.0, z]))
            if pl.point(figure.pts)[:, 2].min() < 0.04:        # keep above the floor
                continue
            pts_w, axes_w = surface_samples(lit, pl, n_sample)
            frac = JointMap3D(ik, pts_w, axes_w, q_home, n_restarts=2).valid_fraction
            if verbose:
                print('  scale %.2f x %.2f z %.2f -> map %.0f%%' % (scale, x, z, 100 * frac))
            if best is None or frac > best[1]:
                best = (pl, frac)
            if frac >= MIN_MAP_COVER:
                return pl, frac
    if best is None:
        raise RuntimeError('no placement above the floor')
    return best


def density_sites(figure, placement, lit_only=True):
    """Coloured points on the lit surface carrying the target density."""
    keep = figure.lit & (figure.w >= MIN_DENSITY_SHOWN) if lit_only else figure.w >= MIN_DENSITY_SHOWN
    if getattr(figure, 'is_volume', False) and keep.sum() > MAX_VOLUME_SITES:
        thr = np.sort(figure.w[keep])[-MAX_VOLUME_SITES]
        keep = keep & (figure.w >= thr)
    p = figure.pts[keep] + figure.nrm[keep] * (SITE_LIFT / placement.scale)
    pos = placement.point(p)
    idx = np.clip((figure.w[keep] * 255).astype(int), 0, 255)
    rgba = np.concatenate([LUT[idx], np.ones((len(idx), 1))], axis=1)
    return pos, rgba


def build_model(figure, placement, work_dir):
    """Compile the scene for one figure; returns the MjModel."""
    os.makedirs(work_dir, exist_ok=True)
    if getattr(figure, 'is_volume', False):            # cube workspace: edges, no mesh
        edges = [(placement.point(a), placement.point(b)) for a, b in figure.edges]
        return build_spec(sites=density_sites(figure, placement), edges=edges).compile()
    import trimesh
    mesh = figure.mesh
    verts = placement.point(np.asarray(mesh.vertices, dtype=np.float64))
    out = trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces), process=False)
    path = os.path.join(work_dir, 'figure_%d.obj' % os.getpid())
    out.export(path)
    spec = build_spec(mesh_path=path, sites=density_sites(figure, placement))
    return spec.compile()
