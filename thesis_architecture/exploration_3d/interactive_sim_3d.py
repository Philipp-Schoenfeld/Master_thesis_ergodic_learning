"""
interactive_sim_3d.py
=====================
Control centre of the 3D active-sensing simulation (the 3D counterpart of the
2D ``exploration/interactive_sim.py``).

Pick a held-out target (form on a surface), press Play: the belief about the
density on the surface turns into a particle cloud, the trained SE(3) flow
matching network plans a trajectory on the surface, the agent drives it, senses
the density under it, updates the belief and replans.  "Open MuJoCo" shows the
same run with the Panda on the figure as defined by the dataset.
"""

import os
import threading
import time
import traceback

import numpy as np
import matplotlib

matplotlib.use(os.environ.get('MPLBACKEND', 'TkAgg'))
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.colors import LinearSegmentedColormap       # noqa: E402
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox   # noqa: E402

import paths3d                                               # noqa: E402

paths3d.setup()
import figures3d as F                                        # noqa: E402
import mission3d as M                                        # noqa: E402
import net_zoo3d                                             # noqa: E402
from planner3d import Planner3D                              # noqa: E402
from obstacles import SphereObstacle, CompositeObstacle   # noqa: E402  (the 3D one)

SPLITS = ('val_form', 'val_beides', 'val_flaeche', 'volume')   # first entry = the cube surface
SURFACE_SPLITS = SPLITS[:3]
SPLIT_LABEL = {'val_beides': 'new form + new surface', 'val_form': 'new form',
               'val_flaeche': 'new surface', 'volume': 'cube volume'}
SOLVER_DESC = {
    'A': 'A - plan once, execute once',
    'B': 'B - plan, drive a length unit, replan (coverage debt)',
    'C': 'C - receding horizon: execute every plan, replan',
    'D': 'D - plan long, execute a short part, replan (coverage debt)',
    'E': 'E - 5 rounds with a manual kappa per round',
}
PHI_DEFAULTS = {'ucb': 0.5, 'eid': 0.6, 'mass': 0.47, 'lse': 0.67}
PHI_LABELS = ['ucb', 'mass', 'eid', 'lse']
VIEWS = ['Truth', 'Mean', 'Sigma', 'Phi']
FRAME_MS = 55
#: Agent speed in figure units per second (the unit cube; ~0.2 m/s on the robot).
DEFAULT_SPEED = 0.6
PANEL_ZOOM = 1.0                      # two main panels share the width the single one had
#: Reveal radius of the initial point samples (single measurements, hence small).
PRIOR_RADIUS = 0.015

#: Belief kernels (belief3d.KERNELS) and their buttons.  The neural process only exists for
#: 2D inputs (no 3D checkpoint), so its button reports that instead of switching.
KERNEL_LABELS = {'rbf': 'RBF kernel', 'ak': 'AK kernel', 'np': 'NP kernel'}
KERNEL_ON, KERNEL_OFF = '#9FA8DA', '#F2F2F5'

#: Spherical obstacles placed by clicking on the figure.
MAX_OBSTACLES = 10
DEFAULT_OBS_RADIUS = 0.08
PICK_PX = 14                          # click tolerance (screen pixels) when picking a figure point

INK, GREY = '#1A1A2E', '#555555'
GREEN, RED = '#00C853', '#FF1744'


def white_inferno():
    base = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    ramp = np.linspace(0, 1, 60)[:, None]
    base[:60, :3] = (1 - ramp) + ramp * base[:60, :3]
    return LinearSegmentedColormap.from_list('white_inferno', base)


CMAP = white_inferno()
#: Inside the cube low values must stay visible against the white background.
VOLUME_CMAP = plt.get_cmap('inferno')


def volume_alpha(v):
    return 0.06 + 0.8 * np.clip(v, 0.0, 1.0) ** 1.2


class App:
    def __init__(self, ckpt, device=None, seed=0, fig_sel=None, agent=None, shared_path=None,
                 show=True, shared_obs=None):
        self.planner = Planner3D(ckpt, device=device)
        self.device = device
        #: Loaded checkpoint and its registry entry (`net_zoo3d.py`), switched by the
        #: "Network" button; unregistered checkpoints get a generic entry from their metadata.
        self.ckpt = ckpt
        self.net_entry = net_zoo3d.entry_for(ckpt) or net_zoo3d.custom_entry(ckpt)
        self.fig_net, self.net_radio, self.net_file_txt = None, None, None
        self.seed = seed
        self.fig_sel, self.agent, self.shared_path = fig_sel, agent, shared_path
        self.shared_obs = shared_obs             # obstacle spheres for the MuJoCo process
        self.entries = {s: F.list_entries(s) for s in SPLITS}
        self.split_i = 0
        self.entry_i = 0
        self._surface_sel = (0, 0)               # (split, entry) to return to from the cube volume
        self.solver = 'C'
        self.phi_ui = 'ucb'
        self.kappa = PHI_DEFAULTS['ucb']
        self.tau = PHI_DEFAULTS['lse']
        self.mass_w = PHI_DEFAULTS['mass']
        self.cutoff = 0.0
        self.sensor_radius = 0.05
        self.speed = DEFAULT_SPEED
        self.e_kappas = list(M.E_DEFAULT_KAPPAS)
        self.snap = True
        self.net_orientation = False
        self.truth_known = False                 # plan against the true density (oracle) instead of the belief
        self.belief_kernel = 'rbf'
        self.obstacles = []                      # SphereObstacle list; repels every plan
        self.obs_radius = DEFAULT_OBS_RADIUS     # radius of the next obstacle placed
        self.place_mode = False                  # clicking the figure places an obstacle
        self._obs_artists = []
        self._obs_dirty = False                  # obstacles changed while a plan was being computed
        self._press_xy = None
        self.view_mode = 'Truth'
        self.version = 0
        self.busy = False
        self.playing = False
        self.finished = False
        self.cur = None
        self.play_idx = 0
        self.play_arc = 0.0
        self._last_play_t = time.perf_counter()
        self.driven = np.zeros((0, 3))
        self.info_last = None
        self.p_mj = None
        self.know_fig = None
        self._know_t = 0.0

        self._build_gui()
        self._load_figure()
        self.timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self.timer.add_callback(self._tick)
        if show:
            self.timer.start()
            plt.show()

    # -- selection -------------------------------------------------------------
    @property
    def split(self):
        return SPLITS[self.split_i]

    @property
    def entry(self):
        return self.entries[self.split][self.entry_i]

    def _shapes(self):
        seen, out = set(), []
        for e in self.entries[self.split]:
            if e['shape'] not in seen:
                seen.add(e['shape'])
                out.append(e['shape'])
        return out

    def _change_shape(self, d):
        shapes = self._shapes()
        cur = shapes.index(self.entry['shape'])
        name = shapes[(cur + d) % len(shapes)]
        self.entry_i = next(i for i, e in enumerate(self.entries[self.split]) if e['shape'] == name)
        self._load_figure()

    def _change_surface(self, d):
        same = [i for i, e in enumerate(self.entries[self.split]) if e['shape'] == self.entry['shape']]
        k = same.index(self.entry_i)
        self.entry_i = same[(k + d) % len(same)]
        self._load_figure()

    @property
    def volume_mode(self):
        return self.split == 'volume'

    def _change_split(self, event):
        if self.volume_mode:                       # the split only applies to surface figures
            return
        self.split_i = (self.split_i + 1) % len(SURFACE_SPLITS)
        self.entry_i = 0
        self._load_figure()

    def _on_mode(self, event):
        """Switch between the held-out surface figures and the cube workspace."""
        if self.volume_mode:
            self.split_i, self.entry_i = self._surface_sel
        else:
            self._surface_sel = (self.split_i, self.entry_i)
            self.split_i, self.entry_i = SPLITS.index('volume'), 0
        self._load_figure()

    # -- figure / mission ------------------------------------------------------
    def _load_figure(self):
        self.status_txt.set_text('Loading %s ...' % self.entry['shape'])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.figure = F.build_figure(self.entry)
        self.metrics = M.Metrics3D(self.figure, device='cpu')
        self.lit_idx = np.flatnonzero(self.figure.lit)
        self._setup_axes()
        self.version += 1
        self._sync_selection_texts()
        if self.fig_sel is not None:
            with self.fig_sel.get_lock():
                self.fig_sel[:3] = [self.version, self.split_i, self.entry_i]
        self.reset()

    def _setup_axes(self):
        ax = self.ax
        ax.clear()
        f = self.figure
        P = f.pts[self.lit_idx]
        vol = self.volume_mode
        self.sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=f.w[self.lit_idx], cmap=CMAP, vmin=0,
                             vmax=1, s=6 if vol else 7, alpha=None if vol else 0.9,
                             depthshade=False, linewidths=0)
        self._paint(self.sc, f.w[self.lit_idx])
        if vol:
            self._draw_cube(ax)
        U = f.pts[~f.lit]
        if len(U):
            ax.scatter(U[::4, 0], U[::4, 1], U[::4, 2], c='#dddddd', s=2, alpha=0.15,
                       depthshade=False, linewidths=0)
        self.cloud_sc = ax.scatter([], [], [], c='#444444', s=6, alpha=0.3, depthshade=False)
        self.line_past, = ax.plot([], [], [], color=GREEN, lw=2.2, alpha=0.95)
        self.line_cur, = ax.plot([], [], [], color=RED, lw=2.4)
        self.line_pre, = ax.plot([], [], [], color=RED, lw=1.8, alpha=0.6)
        self.agent_pt, = ax.plot([], [], [], 'o', color='#1565C0', ms=8, mec='white')
        self.start_pt, = ax.plot([f.start[0]], [f.start[1]], [f.start[2]], '*', color='#1565C0',
                                 ms=12, mec='white')
        c, r = self._extent()
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1), zoom=PANEL_ZOOM)
        elev, azim = self._camera()
        ax.view_init(elev=elev, azim=azim)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((1, 1, 1, 0))
        ax.set_xlabel('x', color=GREY)
        ax.set_ylabel('y', color=GREY)
        ax.set_zlabel('z', color=GREY)
        ax.tick_params(colors=GREY, labelsize=7)
        self.title_txt = ax.set_title(f.title, color=INK, fontsize=11)
        self._setup_belief_axes(c, r, elev, azim)
        self._obs_artists = []                    # ax.clear() already dropped the old ones
        self._draw_obstacles()

    def _setup_belief_axes(self, c, r, elev, azim):
        """Second main panel: the GP belief (posterior mean) with the same path artists."""
        ax, f = self.ax_b, self.figure
        ax.clear()
        P = f.pts[self.lit_idx]
        vol = self.volume_mode
        self.sc_b = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.zeros(len(P)), cmap=CMAP, vmin=0,
                               vmax=1, s=6 if vol else 7, alpha=None if vol else 0.9,
                               depthshade=False, linewidths=0)
        if vol:
            self._draw_cube(ax)
        U = f.pts[~f.lit]
        if len(U):
            ax.scatter(U[::4, 0], U[::4, 1], U[::4, 2], c='#dddddd', s=2, alpha=0.15,
                       depthshade=False, linewidths=0)
        self.line_past_b, = ax.plot([], [], [], color=GREEN, lw=2.2, alpha=0.95)
        self.line_cur_b, = ax.plot([], [], [], color=RED, lw=2.4)
        self.line_pre_b, = ax.plot([], [], [], color=RED, lw=1.8, alpha=0.6)
        self.agent_pt_b, = ax.plot([], [], [], 'o', color='#1565C0', ms=8, mec='white')
        ax.plot([f.start[0]], [f.start[1]], [f.start[2]], '*', color='#1565C0', ms=12, mec='white')
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1), zoom=PANEL_ZOOM)
        ax.view_init(elev=elev, azim=azim)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((1, 1, 1, 0))
        ax.set_xlabel('x', color=GREY)
        ax.set_ylabel('y', color=GREY)
        ax.set_zlabel('z', color=GREY)
        ax.tick_params(colors=GREY, labelsize=7)
        self.title_b = ax.set_title('GP belief (mean)', color=INK, fontsize=11)

    def _sync_main_views(self, event):
        """Rotating one main panel rotates the other one too."""
        panels = (self.ax, self.ax_b)
        if event.inaxes not in panels:
            return
        other = self.ax_b if event.inaxes is self.ax else self.ax
        other.view_init(elev=event.inaxes.elev, azim=event.inaxes.azim)
        self.fig.canvas.draw_idle()

    def _camera(self):
        """(elev, azim) in degrees: onto the lit side of a surface figure, an oblique view of the cube."""
        f = self.figure
        cam = getattr(f, 'camera_dir', None)
        if cam is None:
            cam = -f.view
        return (float(np.degrees(np.arcsin(np.clip(cam[2], -1, 1)))),
                float(np.degrees(np.arctan2(cam[1], cam[0]))))

    def _extent(self):
        """Axis centre and half-width.  A shrunk figure is shown inside the full workspace
        (so it looks smaller), anything else is fitted to its bounding box."""
        f = self.figure
        if getattr(f, 'shrink', 1.0) < 1.0:
            return np.full(3, 0.5), 0.4
        lo, hi = f.pts.min(axis=0), f.pts.max(axis=0)
        return 0.5 * (lo + hi), 0.5 * float((hi - lo).max())

    def _draw_cube(self, ax):
        for a, b in self.figure.edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color='#9E9E9E', lw=0.9, alpha=0.8)

    def _paint(self, sc, values, cmap=CMAP, rgba=None):
        """Colour a point scatter.  In the cube the opacity grows with the value: white
        (zero density) points in front would otherwise hide the coloured ones behind."""
        if not self.volume_mode:
            if rgba is None:
                sc.set_array(values)
            else:
                sc.set_array(None)
                sc.set_facecolor(rgba)
            return
        if rgba is None:
            v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
            rgba = (VOLUME_CMAP if cmap is CMAP else cmap)(v)
            rgba[:, 3] = volume_alpha(v)
        sc.set_array(None)
        sc.set_facecolor(rgba)

    def _args(self):
        a = M.Args(phi_model=self.phi_ui, sensor_radius=self.sensor_radius,
                   phi_cutoff=self.cutoff, n_particles=self.planner.n_particles,
                   kappas=list(self.e_kappas), phi_tau=self.tau)
        a.snap, a.net_orientation = self.snap, self.net_orientation
        a.truth_known = self.truth_known
        a.belief_kernel = self.belief_kernel
        a.obstacle = CompositeObstacle(list(self.obstacles)) if self.obstacles else None
        if self.phi_ui == 'mass':
            w = min(max(self.mass_w, 0.0), 0.98)
            a.kappa0 = w / max(1e-6, 1.0 - w)
            a.kappa1 = a.kappa0
        else:
            a.kappa0 = self.kappa
            a.kappa1 = 0.6 * self.kappa
        return a

    def reset(self, event=None):
        self.mission = M.Mission3D(self.planner, self.figure, self._args(), variant=self.solver,
                                   seed=self.seed)
        self.gen = self.mission.rounds()
        self.driven = np.zeros((0, 3))
        self.cur, self.finished, self.playing = None, False, False
        self.play_idx, self.play_arc = 0, 0.0
        self.info_last = None
        self.known = np.zeros(len(self.figure.pts), dtype=bool)
        self._reveal(self.figure.pts[self.mission.prior_idx], radius=PRIOR_RADIUS)
        self.b_play.label.set_text('Play')
        self.status_txt.set_text('Prior knowledge - Play plans round 1')
        self.round_txt.set_text('')
        self._publish_agent(self.figure.start, self._axis_at(self.figure.start))
        self._publish_path()
        self._update_scatter()                        # truth / prior belief until the first plan arrives
        self._advance_round()
        self._redraw()
        self._update_knowledge_window(force=True)

    def _axis_at(self, p):
        _, n = self.mission.lit.project(np.atleast_2d(p))
        return -n[0]

    # -- round generator (background thread) ----------------------------------
    def _advance_round(self):
        self.busy = True
        self._busy_t0 = time.perf_counter()
        gen, result = self.gen, {}
        self._plan_result = result

        def worker():
            try:
                result['val'] = next(gen)
            except StopIteration:
                result['stop'] = True
            except Exception as exc:                   # noqa: BLE001
                traceback.print_exc()
                result['err'] = exc

        self._plan_thread = threading.Thread(target=worker, daemon=True)
        self._plan_thread.start()

    def _poll_plan(self):
        elapsed = time.perf_counter() - self._busy_t0
        step = getattr(self.mission, 'current_step', 'inference')
        label = {'inference': 'Inference', 'refine': 'Surface constraint'}.get(
            step, 'Clamping to cube' if self.volume_mode else 'Snapping to surface')
        self.prog_txt.set_text('%s ... %.1fs' % (label, elapsed))
        if self._plan_thread.is_alive():
            return
        self.busy = False
        self.prog_txt.set_text('')
        res = self._plan_result
        if 'err' in res:
            self.finished, self.playing = True, False
            self.b_play.label.set_text('Play')
            self.status_txt.set_text('Planning error: %s' % res['err'])
            return
        if res.get('stop'):
            self.cur, self.finished, self.playing = None, True, False
            self.b_play.label.set_text('Play')
            self.status_txt.set_text('Mission finished - Reset for a new run')
            return
        self.cur = res['val']
        self.info_last = self.cur
        self.play_idx, self.play_arc = 0, 0.0
        self.status_txt.set_text('Solver %s - round %d/%d (kappa %.3f)' % (
            self.solver, self.cur['round'] + 1, self.cur['n_rounds'], self.cur['kappa']))
        self._update_scatter()
        self._redraw()                                # show the new plan right away
        self._update_knowledge_window(force=True)
        if self._obs_dirty:                           # obstacles changed while this plan was computed
            self._obs_dirty = False
            self._replan_current()

    # -- playback ----------------------------------------------------------------
    def _on_play(self, event):
        if self.finished:
            return
        self.playing = not self.playing
        self.b_play.label.set_text('Pause' if self.playing else 'Play')
        self._last_play_t = time.perf_counter()

    def _advance_index(self, seg, idx):
        if len(seg) < 2:
            return len(seg)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
        now = time.perf_counter()
        dt = FRAME_MS / 1000.0 if idx <= 0 else min(now - self._last_play_t, 0.5)
        if idx <= 0:
            self.play_arc = 0.0
        self._last_play_t = now
        self.play_arc += self.speed * dt * self._arm_pace()
        new_idx = int(np.searchsorted(arc, self.play_arc, side='left'))
        return int(min(len(seg), max(idx, new_idx)))

    def _arm_pace(self):
        """1 while the MuJoCo arm keeps up with the agent, down to 0.1 when it lags
        (the agent waits for the arm instead of running away from it)."""
        if self.agent is None or self.p_mj is None or not self.p_mj.is_alive():
            return 1.0
        with self.agent.get_lock():
            lag = float(self.agent[7])
        return float(np.clip(1.0 - (lag - 0.015) / 0.05, 0.1, 1.0))

    def _tick(self):
        try:
            self._tick_inner()
        except Exception:
            traceback.print_exc()
            self.timer.stop()
            self.playing = False
            self.status_txt.set_text('Error - simulation stopped (see console)')
            self.fig.canvas.draw_idle()

    def _tick_inner(self):
        if self.busy:
            self._poll_plan()
            self._redraw_light()
            return
        if not (self.playing and not self.finished and self.cur is not None):
            return
        seg = self.cur['seg']
        new_idx = self._advance_index(seg, self.play_idx)
        self._reveal(seg[self.play_idx:new_idx])
        self.play_idx = new_idx
        if self.play_idx >= len(seg):
            self.driven = np.concatenate([self.driven, seg], axis=0)
            self.cur = None
            self._advance_round()                     # commits + observes, plans the next round
            self._update_metrics()
        self._redraw()
        self._update_knowledge_window()

    # -- display -------------------------------------------------------------------
    def _field(self):
        info = self.info_last
        if self.view_mode == 'Truth' or info is None:
            return self.figure.w
        return {'Mean': np.clip(info['mu'], 0, 1), 'Sigma': np.clip(info['sd'], 0, 1),
                'Phi': info['phi']}[self.view_mode]

    def _reveal(self, pts, radius=None):
        """Mark the lit surface within the sensor radius of ``pts`` as known."""
        if pts is None or len(pts) == 0:
            return
        idx = self.mission.sensor.footprint(np.atleast_2d(pts), self.mission.args.sensor_radius
                                            if radius is None else radius)
        self.known[idx] = True

    def _belief_mean(self):
        """GP posterior mean at planning time of the current round (the prior before the first
        plan), clipped to the density range."""
        if self.info_last is not None:
            return np.clip(self.info_last['mu'], 0, 1)
        mu, _ = self.mission.belief.posterior()
        return np.clip(mu.cpu().numpy(), 0, 1)

    def _update_scatter(self):
        self._paint(self.sc, self._field()[self.lit_idx])
        self._paint(self.sc_b, self._belief_mean()[self.lit_idx])
        self.title_b.set_text('GP belief (mean)%s' % ('  [truth known]'
                                                      if self.truth_known else ''))
        if self.info_last is not None:
            p = self.info_last['parts'][:, :3]
            self.cloud_sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

    def _set_line(self, line, pts):
        pts = np.asarray(pts)
        if len(pts):
            line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
        else:
            line.set_data_3d([], [], [])

    def _redraw_light(self):
        self.fig.canvas.draw_idle()

    def _redraw(self):
        cur, pre, pos, axis = np.zeros((0, 3)), np.zeros((0, 3)), None, None
        if self.cur is not None:
            seg = self.cur['seg']
            cur, pre = seg[:max(self.play_idx, 1)], seg[max(self.play_idx - 1, 0):]
            k = max(self.play_idx - 1, 0)
            pos = seg[k]
            axis = self.cur['axes'][k]
        elif len(self.driven):
            pos = self.driven[-1]
        if pos is None:
            pos = self.figure.start
        if axis is None:
            axis = self._axis_at(pos)
        for past, cu, pr, ag in ((self.line_past, self.line_cur, self.line_pre, self.agent_pt),
                                 (self.line_past_b, self.line_cur_b, self.line_pre_b, self.agent_pt_b)):
            self._set_line(past, self.driven)
            self._set_line(cu, cur)
            self._set_line(pr, pre)
            ag.set_data_3d([pos[0]], [pos[1]], [pos[2]])
        self._publish_agent(pos, axis)
        self._publish_path()
        self.fig.canvas.draw_idle()

    def _update_metrics(self):
        if len(self.driven) < 2:
            return
        mu, sd = self.mission.belief.posterior()
        mu, sd = mu.cpu().numpy(), sd.cpu().numpy()
        lit = self.figure.lit
        L = float(np.linalg.norm(np.diff(self.driven, axis=0), axis=1).sum())
        self.metric_txt.set_text(
            'Ergodic metric (vs. truth)  %.4f\nCoverage (mass-weighted dist)  %.4f\n'
            'Belief error (lit, MAE)  %.3f\nMean sigma (lit)  %.3f\nPath length  %.2f   Observations  %d' % (
                self.metrics.ergodic(self.driven), self.metrics.coverage(self.driven),
                float(np.abs(np.clip(mu, 0, 1)[lit] - self.figure.w[lit]).mean()),
                float(sd[lit].mean()), L, self.mission.belief.n_obs))

    # -- shared memory --------------------------------------------------------------
    def _publish_agent(self, pos, axis):
        if self.agent is None:
            return
        with self.agent.get_lock():
            self.agent[:7] = [pos[0], pos[1], pos[2], axis[0], axis[1], axis[2],
                              self.sensor_radius]

    def _publish_path(self):
        if self.shared_path is None:
            return
        from mujoco_3d.run_mujoco_3d import (PATH_HEADER, PATH_MAX_PAST, PATH_MAX_CURRENT,
                                             PATH_MAX_PREVIEW)

        def thin(pts, cap):
            if len(pts) <= cap:
                return pts
            stride = int(np.ceil((len(pts) - 1) / (cap - 1)))
            idx = np.arange(0, len(pts), stride)
            if idx[-1] != len(pts) - 1:
                if len(idx) < cap:
                    idx = np.append(idx, len(pts) - 1)
                else:
                    idx[-1] = len(pts) - 1
            return pts[idx]

        cur, pre = np.zeros((0, 3)), np.zeros((0, 3))
        if self.cur is not None:
            seg = self.cur['seg']
            cur, pre = seg[:self.play_idx], seg[self.play_idx:]
        past, cur, pre = thin(self.driven, PATH_MAX_PAST), thin(cur, PATH_MAX_CURRENT), thin(pre, PATH_MAX_PREVIEW)
        off_cur = PATH_HEADER + 3 * PATH_MAX_PAST
        off_pre = off_cur + 3 * PATH_MAX_CURRENT
        with self.shared_path.get_lock():
            buf = np.frombuffer(self.shared_path.get_obj(), dtype=np.float64)
            buf[0], buf[1], buf[2] = len(past), len(cur), len(pre)
            buf[PATH_HEADER:PATH_HEADER + 3 * len(past)] = np.asarray(past, np.float64).ravel()
            buf[off_cur:off_cur + 3 * len(cur)] = np.asarray(cur, np.float64).ravel()
            buf[off_pre:off_pre + 3 * len(pre)] = np.asarray(pre, np.float64).ravel()

    def _on_open_mujoco(self, event):
        if self.fig_sel is None or self.agent is None:
            print('[control centre] MuJoCo needs main.py (shared memory) - start "python main.py".')
            return
        import multiprocessing as mp
        if self.p_mj is None or not self.p_mj.is_alive():
            from mujoco_3d.run_mujoco_3d import run_mujoco_3d
            self.p_mj = mp.Process(target=run_mujoco_3d,
                                   args=(self.fig_sel, self.agent, self.shared_path),
                                   kwargs=dict(shared_obs=self.shared_obs))
            self.p_mj.start()
            self._publish_obstacles()
        self._open_knowledge_window()

    # -- knowledge window ---------------------------------------------------------
    KNOW_PANELS = ('Belief (mean)', 'Uncertainty (sigma)', 'Target density Phi',
                   'Known vs. unknown')

    def _open_knowledge_window(self):
        if self.know_fig is not None and plt.fignum_exists(self.know_fig.number):
            return
        fig = plt.figure(figsize=(12.5, 9.5), facecolor='white')
        fig.canvas.manager.set_window_title('3D Active Sensing - Knowledge')
        self.know_fig = fig
        self.know_axes, self.know_sc = [], []
        for i, name in enumerate(self.KNOW_PANELS):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d', facecolor='white')
            ax.set_title(name, color=INK, fontsize=10)
            self.know_axes.append(ax)
        self.know_note = fig.text(0.5, 0.01, '', ha='center', color=GREY, fontsize=9)
        fig.canvas.mpl_connect('button_release_event', self._sync_know_views)
        fig.canvas.mpl_connect('close_event', lambda e: setattr(self, 'know_fig', None))
        self._build_know_panels()
        self._update_knowledge_window(force=True)
        fig.show()

    def _build_know_panels(self):
        f = self.figure
        P = f.pts[self.lit_idx]
        c, r = self._extent()
        elev, azim = self._camera()
        self.know_cmaps = [CMAP, plt.get_cmap('viridis'), CMAP, CMAP]
        vol = self.volume_mode
        self.know_sc = []
        for ax, cm, name in zip(self.know_axes, self.know_cmaps, self.KNOW_PANELS):
            ax.clear()
            ax.set_title(name, color=INK, fontsize=10)
            self.know_sc.append(ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.zeros(len(P)), cmap=cm,
                                           vmin=0, vmax=1, s=3 if vol else 6, depthshade=False,
                                           linewidths=0))
            if vol:
                self._draw_cube(ax)
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.set_box_aspect((1, 1, 1), zoom=1.2)
            ax.view_init(elev=elev, azim=azim)
            for a in (ax.xaxis, ax.yaxis, ax.zaxis):
                a.set_pane_color((1, 1, 1, 0))
            ax.tick_params(colors=GREY, labelsize=6)
        self._know_version = self.version

    def _sync_know_views(self, event):
        src = event.inaxes
        if src not in getattr(self, 'know_axes', []):
            return
        for ax in self.know_axes:
            if ax is not src:
                ax.view_init(elev=src.elev, azim=src.azim)
        self.know_fig.canvas.draw_idle()

    def _update_knowledge_window(self, force=False):
        fig = self.know_fig
        if fig is None or not plt.fignum_exists(fig.number):
            return
        now = time.perf_counter()
        if not force and now - self._know_t < 0.6:
            return
        self._know_t = now
        if getattr(self, '_know_version', None) != self.version:
            self._build_know_panels()
        info, lit = self.info_last, self.lit_idx
        n = len(lit)
        if info is None:
            mu = sd = phi = np.zeros(n)
            sd = np.ones(n)
        else:
            mu, sd, phi = np.clip(info['mu'], 0, 1)[lit], np.clip(info['sd'], 0, 1)[lit], info['phi'][lit]
        known = self.known[lit]
        truth = self.figure.w[lit]
        # unknown surface is black, known surface shows the ground truth
        rgba = np.zeros((n, 4))
        rgba[:, 3] = 1.0
        rgba[known] = CMAP(truth[known])
        if self.volume_mode:                          # faint grey = unknown, opacity = density where known
            rgba[~known] = (0.55, 0.55, 0.55, 0.05)
            rgba[known] = VOLUME_CMAP(np.clip(truth[known], 0, 1))
            rgba[known, 3] = volume_alpha(truth[known])
        for sc, vals, cm in zip(self.know_sc[:3], (mu, sd, phi), self.know_cmaps):
            self._paint(sc, vals, cm)
        self._paint(self.know_sc[3], None, rgba=rgba)    # explicit colours, not a colour map
        self.know_note.set_text('Known: %.0f%% of the %s   |   belief, sigma and Phi are the state '
                                'at planning time of the current round'
                                % (100.0 * known.mean(), 'cube' if self.volume_mode else 'lit surface'))
        fig.canvas.draw_idle()

    # -- widgets --------------------------------------------------------------------
    def _build_gui(self):
        self.fig = plt.figure(figsize=(15.5, 8.6), facecolor='white')
        self.fig.canvas.manager.set_window_title('3D Active Sensing - Control Centre')
        # two 3D panels side by side: the selected view (truth by default) and the GP belief
        self.ax = self.fig.add_axes([0.005, 0.11, 0.300, 0.83], projection='3d', facecolor='white')
        self.ax_b = self.fig.add_axes([0.305, 0.11, 0.300, 0.83], projection='3d', facecolor='white')
        self.fig.canvas.mpl_connect('button_release_event', self._sync_main_views)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        T = self.fig.text
        x0, w = 0.63, 0.35

        def btn(rect, label, cb):
            b = Button(self.fig.add_axes(rect), label, color='#F2F2F5', hovercolor='#E2E2EA')
            b.label.set_color(INK)
            b.label.set_fontsize(9)
            b.on_clicked(cb)
            return b

        self.status_txt = T(x0, 0.955, '', color=INK, fontsize=11, weight='bold')
        self.round_txt = T(x0, 0.93, '', color=GREY, fontsize=9)
        self.prog_txt = T(x0 + 0.20, 0.93, '', color=GREY, fontsize=9)

        # held-out selection
        T(x0, 0.895, 'Held-out target', color=GREY, fontsize=9)
        self.b_shape_prev = btn([x0, 0.855, 0.03, 0.03], '<', lambda e: self._change_shape(-1))
        self.shape_txt = T(x0 + 0.037, 0.862, '', color=INK, fontsize=10)
        self.b_shape_next = btn([x0 + 0.31, 0.855, 0.03, 0.03], '>', lambda e: self._change_shape(1))
        self.b_surf_prev = btn([x0, 0.815, 0.03, 0.03], '<', lambda e: self._change_surface(-1))
        self.surf_txt = T(x0 + 0.037, 0.822, '', color=INK, fontsize=10)
        self.b_surf_next = btn([x0 + 0.31, 0.815, 0.03, 0.03], '>', lambda e: self._change_surface(1))
        self.b_split = btn([x0, 0.775, 0.34, 0.03], '', self._change_split)

        # solver / phi / view
        T(x0, 0.74, 'Solver', color=GREY, fontsize=9)
        self.r_solver = RadioButtons(self.fig.add_axes([x0, 0.585, 0.34, 0.15], facecolor='white'),
                                     list(SOLVER_DESC.values()), active=2)
        self.r_solver.on_clicked(self._on_solver)
        T(x0, 0.565, 'Acquisition Phi', color=GREY, fontsize=9)
        self.r_phi = RadioButtons(self.fig.add_axes([x0, 0.485, 0.16, 0.075], facecolor='white'),
                                  PHI_LABELS, active=0)
        self.r_phi.on_clicked(self._on_phi)
        T(x0 + 0.18, 0.565, 'Show', color=GREY, fontsize=9)
        self.r_view = RadioButtons(self.fig.add_axes([x0 + 0.18, 0.485, 0.16, 0.075], facecolor='white'),
                                   VIEWS, active=0)
        self.r_view.on_clicked(self._on_view)
        for r in (self.r_solver, self.r_phi, self.r_view):
            for lab in r.labels:
                lab.set_fontsize(8.5)
                lab.set_color(INK)

        # sliders
        def slider(y, label, lo, hi, init, cb, fmt='%.2f'):
            s = Slider(self.fig.add_axes([x0 + 0.08, y, 0.21, 0.022]), label, lo, hi, valinit=init,
                       valfmt=fmt, color='#9FA8DA')
            s.label.set_fontsize(8.5)
            s.valtext.set_fontsize(8.5)
            s.on_changed(cb)
            return s

        self.s_kappa = slider(0.445, 'kappa', 0.0, 3.0, self.kappa, self._on_tuning)
        self.s_cut = slider(0.415, 'cloud cutoff', 0.0, 0.9, self.cutoff, self._on_cutoff)
        self.s_radius = slider(0.385, 'sensor radius', 0.02, 0.15, self.sensor_radius, self._on_radius, '%.3f')
        self.s_speed = slider(0.355, 'speed', 0.1, 1.5, self.speed, self._on_speed)

        # solver E schedule
        T(x0, 0.33, 'Solver E: kappa per round', color=GREY, fontsize=9)
        self.e_boxes = []
        for i in range(M.E_ROUNDS):
            tb = TextBox(self.fig.add_axes([x0 + 0.015 + i * 0.066, 0.295, 0.055, 0.028]), '',
                         initial='%g' % self.e_kappas[i])
            tb.on_submit(lambda text, i=i: self._on_e_kappa(i, text))
            self.e_boxes.append(tb)

        # actions
        self.b_play = btn([x0, 0.235, 0.07, 0.04], 'Play', self._on_play)
        self.b_reset = btn([x0 + 0.08, 0.235, 0.07, 0.04], 'Reset', self.reset)
        self.b_mj = btn([x0 + 0.16, 0.235, 0.10, 0.04], 'Open MuJoCo', self._on_open_mujoco)
        self.b_know = btn([x0 + 0.27, 0.235, 0.09, 0.04], 'Knowledge',
                          lambda e: self._open_knowledge_window())

        self.b_snap = btn([x0, 0.195, 0.17, 0.03], '', self._on_snap)
        self.b_orient = btn([x0 + 0.18, 0.195, 0.16, 0.03], '', self._on_orient)
        self.b_mode = btn([x0, 0.157, 0.34, 0.03], '', self._on_mode)
        self.b_kernel = {}
        for i, key in enumerate(KERNEL_LABELS):
            self.b_kernel[key] = btn([x0 + i * 0.1167, 0.119, 0.1067, 0.03], KERNEL_LABELS[key],
                                     lambda e, key=key: self._on_kernel(key))
        self.b_net = btn([x0, 0.088, 0.17, 0.026], '', self._open_net_window)
        self.b_truth = btn([x0 + 0.18, 0.088, 0.16, 0.026], '', self._on_truth)
        self.metric_txt = T(x0, 0.003, '', color=INK, fontsize=8.5, family='monospace', va='bottom')
        self._sync_net_ui()
        self.note_txt = T(0.02, 0.012, '', color=GREY, fontsize=8.5, va='bottom')

        # obstacles: click on the figure to place a sphere, the plan is repelled from it
        self.obs_txt = T(0.02, 0.088, '', color=GREY, fontsize=8.5, va='bottom')
        self.b_place = btn([0.02, 0.048, 0.19, 0.032], '', self._on_place_toggle)
        self.b_obs_undo = btn([0.22, 0.048, 0.10, 0.032], 'Remove last', self._on_obstacle_undo)
        self.b_obs_clear = btn([0.33, 0.048, 0.10, 0.032], 'Clear all', self._on_obstacle_clear)
        self.s_obs_r = Slider(self.fig.add_axes([0.50, 0.054, 0.10, 0.022]), 'radius', 0.03, 0.25,
                              valinit=self.obs_radius, valfmt='%.2f', color='#9FA8DA')
        self.s_obs_r.label.set_fontsize(8.5)
        self.s_obs_r.valtext.set_fontsize(8.5)
        self.s_obs_r.on_changed(lambda v: setattr(self, 'obs_radius', float(v)))
        self._sync_obstacle_ui()

    def _sync_selection_texts(self):
        e = self.entry
        self.shape_txt.set_text(e['shape'])
        same = [x for x in self.entries[self.split] if x['shape'] == e['shape']]
        k = same.index(e)
        self.surf_txt.set_text('%s  (%d/%d)' % (self.figure.label, k + 1, len(same)))
        vol = self.volume_mode
        self.b_snap.label.set_text('Positions: %s' % (
            ('clamped to cube' if vol else 'snapped to surface') if self.snap else 'raw network'))
        self.b_orient.label.set_text('Tool axis: %s' % (
            'network' if self.net_orientation else ('fixed' if vol else 'surface normal')))
        self.b_split.label.set_text('Split: n/a (cube volume)' if vol else
                                    'Split: %s  (click to change)' % SPLIT_LABEL[self.split])
        for key, b in self.b_kernel.items():
            on = key == self.belief_kernel
            b.color = KERNEL_ON if on else KERNEL_OFF          # the colour a button returns to after hover
            b.ax.set_facecolor(KERNEL_ON if on else KERNEL_OFF)
            b.label.set_weight('bold' if on else 'normal')
            b.label.set_color(GREY if key == 'np' else INK)
        self.b_truth.label.set_text('Ground truth: %s' % ('known' if self.truth_known else 'unknown'))
        self.b_truth.color = KERNEL_ON if self.truth_known else KERNEL_OFF
        self.b_truth.ax.set_facecolor(KERNEL_ON if self.truth_known else KERNEL_OFF)
        self.b_truth.label.set_weight('bold' if self.truth_known else 'normal')
        self.b_mode.label.set_text('Target: %s  (click to switch)' % (
            'cube volume (explore the space)' if vol else 'surface figure'))
        self.note_txt.set_text('Green: earlier rounds   Red: current round   Dark dots: particle cloud'
                               '   Grey spheres: obstacles')

    def _on_snap(self, event):
        self.snap = not self.snap
        self._after_plan_setting()

    def _on_orient(self, event):
        self.net_orientation = not self.net_orientation
        self._after_plan_setting()

    def _after_plan_setting(self):
        self._apply_live_args()
        self._sync_selection_texts()
        self._replan_current()
        self.fig.canvas.draw_idle()

    def _replan_current(self):
        """Plan the current round again with the new setting.  If the agent is already
        part-way along the old plan, that part counts as driven (and is observed) and
        the new plan starts where the agent is."""
        if self.busy or self.cur is None:
            return                                  # applies to the next round instead
        info = self.cur
        if self.play_idx > 1:
            part, ax = info['seg'][:self.play_idx], info['axes'][:self.play_idx]
            self.driven = np.concatenate([self.driven, part], axis=0)
            self.mission.commit(part, ax)
        self.cur = None
        self.busy = True
        self._busy_t0 = time.perf_counter()
        mission, result = self.mission, {}
        self._plan_result = result

        def worker():
            try:
                result['val'] = mission.replan(info)
            except Exception as exc:                # noqa: BLE001
                traceback.print_exc()
                result['err'] = exc

        self._plan_thread = threading.Thread(target=worker, daemon=True)
        self._plan_thread.start()

    # -- obstacles ---------------------------------------------------------------------
    def _agent_pos(self):
        """Current position of the agent (end of the driven part of the plan, else the start)."""
        if self.cur is not None:
            return np.asarray(self.cur['seg'][max(self.play_idx - 1, 0)], dtype=np.float64)
        if len(self.driven):
            return np.asarray(self.driven[-1], dtype=np.float64)
        return np.asarray(self.figure.start, dtype=np.float64)

    def _draw_obstacles(self):
        """(Re)draw the obstacle spheres in both main panels."""
        for art in self._obs_artists:
            try:
                art.remove()
            except Exception:  # noqa: BLE001 -- the axes may have been cleared already
                pass
        self._obs_artists = []
        if self.obstacles:
            comp = CompositeObstacle(self.obstacles)
            for ax in (self.ax, self.ax_b):
                self._obs_artists += comp.draw(ax, n_mesh=20)

    def _publish_obstacles(self):
        """Obstacle spheres (centre, radius, margin) into shared memory for the MuJoCo view."""
        if self.shared_obs is None:
            return
        from mujoco_3d.run_mujoco_3d import OBS_MAX, OBS_STRIDE
        rows = [[*o.center, o.radius, o.margin] for o in self.obstacles[:OBS_MAX]]
        with self.shared_obs.get_lock():
            buf = np.frombuffer(self.shared_obs.get_obj(), dtype=np.float64)
            buf[0] = len(rows)
            buf[1:1 + OBS_STRIDE * len(rows)] = np.asarray(rows, np.float64).ravel()

    def _sync_obstacle_ui(self):
        on = self.place_mode
        self.b_place.label.set_text('Place obstacles: %s' % ('ON - click the figure' if on else 'off'))
        self.b_place.color = KERNEL_ON if on else KERNEL_OFF
        self.b_place.ax.set_facecolor(KERNEL_ON if on else KERNEL_OFF)
        self.b_place.label.set_weight('bold' if on else 'normal')
        n = len(self.obstacles)
        self.obs_txt.set_text('Obstacles: %d / %d   (the plan is repelled during inference and '
                              'in every snapping step)' % (n, MAX_OBSTACLES))

    def _obstacles_changed(self):
        """New plan for the current round with the changed obstacles."""
        self._draw_obstacles()
        self._publish_obstacles()
        self._sync_obstacle_ui()
        self._apply_live_args()
        if self.busy:
            self._obs_dirty = True                  # replan as soon as the running plan arrives
        else:
            self._replan_current()
        self.fig.canvas.draw_idle()

    def _on_place_toggle(self, event=None):
        self.place_mode = not self.place_mode
        self._sync_obstacle_ui()
        self.fig.canvas.draw_idle()

    def _on_obstacle_undo(self, event=None):
        if self.obstacles:
            self.obstacles.pop()
            self._obstacles_changed()

    def _on_obstacle_clear(self, event=None):
        if self.obstacles:
            self.obstacles.clear()
            self._obstacles_changed()

    def _on_mouse_press(self, event):
        self._press_xy = (event.x, event.y) if event.inaxes in (self.ax, self.ax_b) else None

    def _on_mouse_release(self, event):
        """A click (no drag: a drag rotates the view) on a main panel places an obstacle."""
        press, self._press_xy = self._press_xy, None
        if not self.place_mode or press is None or event.button != 1:
            return
        if event.inaxes not in (self.ax, self.ax_b) or event.x is None:
            return
        if np.hypot(event.x - press[0], event.y - press[1]) > 4:
            return
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if tb is not None and getattr(tb, 'mode', '') not in ('', None):
            return                                  # zoom / pan tool active
        p = self._pick_point(event)
        if p is None:
            self.status_txt.set_text('No figure point under the cursor - click on the figure')
            self.fig.canvas.draw_idle()
            return
        self._add_obstacle(p)

    def _pick_point(self, event):
        """Figure point under the cursor: among the points within PICK_PX pixels of the click
        the one closest to the camera, or None."""
        from mpl_toolkits.mplot3d import proj3d
        ax = event.inaxes
        P = self.figure.pts[self.lit_idx].astype(np.float64)
        x2, y2, _ = proj3d.proj_transform(P[:, 0], P[:, 1], P[:, 2], ax.get_proj())
        xy = ax.transData.transform(np.column_stack([x2, y2]))
        near = np.flatnonzero(np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y) < PICK_PX)
        if not len(near):
            return None
        el, az = np.radians(ax.elev), np.radians(ax.azim)
        cam = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        return P[near[np.argmax(P[near] @ cam)]]

    def _add_obstacle(self, p):
        if len(self.obstacles) >= MAX_OBSTACLES:
            self.status_txt.set_text('At most %d obstacles - remove one first' % MAX_OBSTACLES)
            self.fig.canvas.draw_idle()
            return
        new = SphereObstacle(center=p, radius=self.obs_radius, margin=M.OBSTACLE_MARGIN)
        guard = new.effective_radius + 0.01
        for q in (self._agent_pos(), self.figure.start):
            if np.linalg.norm(np.asarray(q, dtype=np.float64) - np.asarray(p)) < guard:
                self.status_txt.set_text('Obstacle would cover the agent - click elsewhere')
                self.fig.canvas.draw_idle()
                return
        self.obstacles.append(new)
        self._obstacles_changed()

    def _on_kernel(self, key):
        """Switch the belief kernel (the mission restarts with the new belief)."""
        if key == 'np':
            self.status_txt.set_text('NP kernel: the neural process is 2D only (no 3D checkpoint)')
            self.fig.canvas.draw_idle()
            return
        if key == self.belief_kernel:
            return
        self.belief_kernel = key
        self._sync_selection_texts()
        self.reset()

    def _on_truth(self, event=None):
        """Toggle 'ground truth known': plan against the true density instead of the GP belief
        (the mission restarts).  The belief is still updated and shown."""
        self.truth_known = not self.truth_known
        self._sync_selection_texts()
        self.reset()

    # -- network checkpoint ----------------------------------------------------------
    def _net_choices(self):
        """``[(entry, path)]`` offered in the selection window (+ the loaded one if unregistered)."""
        nets = net_zoo3d.available()
        if all(e.key != self.net_entry.key for e, _ in nets):
            nets.append((self.net_entry, self.ckpt))
        return nets

    def _sync_net_ui(self):
        """Button label and, if open, the selection window follow the loaded network."""
        self.b_net.label.set_text('Network [%s]' % self.net_entry.key)
        if self.fig_net is not None and self.net_radio is not None:
            keys = [e.key for e, _ in self._net_choices()]
            self.net_radio.eventson = False
            self.net_radio.set_active(keys.index(self.net_entry.key))
            self.net_radio.eventson = True
            self.net_file_txt.set_text('loaded: %s' % os.path.basename(self.ckpt))
            self.fig_net.canvas.draw_idle()

    def _open_net_window(self, event=None):
        if self.fig_net is not None:
            try:
                self.fig_net.canvas.manager.show()
            except Exception:  # noqa: BLE001 -- raising the window is cosmetic
                pass
            return
        nets = self._net_choices()
        fig = plt.figure(figsize=(12.5, 1.5 + 0.85 * len(nets)), facecolor='white')
        try:
            fig.canvas.manager.set_window_title('Network checkpoint')
        except Exception:  # noqa: BLE001 -- title is cosmetic
            pass
        fig.text(0.02, 0.965, 'Planner network (3D SE(3), 25 control points, 512 surface particles)'
                 ' - click to load and restart the mission', fontsize=11, fontweight='bold',
                 color=INK, va='top')
        ax_r = fig.add_axes([0.02, 0.08, 0.33, 0.80], facecolor='white')
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        keys = [e.key for e, _ in nets]
        self.net_radio = RadioButtons(ax_r, [e.label for e, _ in nets],
                                      active=keys.index(self.net_entry.key))
        for lbl in self.net_radio.labels:
            lbl.set_fontsize(9)
        self.net_radio.on_clicked(self._on_net_pick)
        ax_d = fig.add_axes([0.37, 0.08, 0.62, 0.80])
        ax_d.axis('off')
        for (entry, _), lbl in zip(nets, self.net_radio.labels):
            ax_d.text(0.0, lbl.get_position()[1], entry.description, fontsize=8.5, color=GREY,
                      va='center', ha='left', transform=ax_d.transAxes, linespacing=1.35)
        self.net_file_txt = fig.text(0.02, 0.02, 'loaded: %s' % os.path.basename(self.ckpt),
                                     fontsize=8, color=GREY)
        self.fig_net = fig
        fig.canvas.mpl_connect('close_event', self._on_net_closed)
        plt.show(block=False)

    def _on_net_closed(self, event):
        self.fig_net, self.net_radio, self.net_file_txt = None, None, None

    def _on_net_pick(self, label):
        for entry, path in self._net_choices():
            if entry.label == label:
                self._switch_net(entry, path)
                return

    def _switch_net(self, entry, path):
        """Load another planner checkpoint and restart the mission with it."""
        if entry.key == self.net_entry.key or os.path.abspath(path) == os.path.abspath(self.ckpt):
            return
        self.status_txt.set_text('Loading network [%s] ...' % entry.key)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        try:
            planner = Planner3D(path, device=self.device)
        except Exception as exc:  # noqa: BLE001 -- report, keep the old network
            traceback.print_exc()
            self.status_txt.set_text('Network [%s] could not be loaded - keeping [%s]: %s'
                                     % (entry.key, self.net_entry.key, str(exc)[:60]))
            self._sync_net_ui()
            return
        print('[network] switched to %s (%s)' % (entry.key, os.path.basename(path)))
        self.planner, self.ckpt, self.net_entry = planner, path, entry
        self._free_gpu()
        self._sync_net_ui()
        self.reset()

    @staticmethod
    def _free_gpu():
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _on_solver(self, label):
        self.solver = label.split(' ')[0]
        self.reset()

    def _on_phi(self, label):
        self.phi_ui = label
        if label == 'lse':
            self.s_kappa.label.set_text('tau')
            self.s_kappa.set_val(self.tau)
        elif label == 'mass':
            self.s_kappa.label.set_text('w')
            self.s_kappa.set_val(self.mass_w)
        else:
            self.s_kappa.label.set_text('kappa')
            self.s_kappa.set_val(PHI_DEFAULTS[label])
        self.reset()

    def _on_view(self, label):
        self.view_mode = label
        self._update_scatter()
        self.fig.canvas.draw_idle()

    def _on_tuning(self, val):
        if self.phi_ui == 'lse':
            self.tau = val
        elif self.phi_ui == 'mass':
            self.mass_w = val
        else:
            self.kappa = val
        self._apply_live_args()

    def _on_cutoff(self, val):
        self.cutoff = val
        self._apply_live_args()

    def _on_radius(self, val):
        self.sensor_radius = val
        self._apply_live_args()

    def _on_speed(self, val):
        self.speed = val

    def _on_e_kappa(self, i, text):
        try:
            self.e_kappas[i] = max(float(text), 0.0)
        except ValueError:
            self.e_boxes[i].set_val('%g' % self.e_kappas[i])
            return
        self._apply_live_args()

    def _apply_live_args(self):
        """Changes apply to the rounds planned from now on."""
        a, new = self.mission.args, self._args()
        for k in ('phi_tau', 'sensor_radius', 'phi_cutoff', 'kappa0', 'kappa1', 'kappas', 'snap', 'net_orientation',
                  'obstacle'):
            setattr(a, k, getattr(new, k))


def run_interactive_sim_3d(ckpt, device, seed, fig_sel, agent, shared_path, shared_obs=None):
    app = App(ckpt, device=device, seed=seed, fig_sel=fig_sel, agent=agent, shared_path=shared_path,
              shared_obs=shared_obs)
    return app


if __name__ == '__main__':
    import sys
    App(sys.argv[1] if len(sys.argv) > 1 else paths3d.DEFAULT_CKPT)
