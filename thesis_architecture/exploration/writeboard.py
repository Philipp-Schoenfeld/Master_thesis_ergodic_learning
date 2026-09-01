import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

_here = os.path.dirname(os.path.abspath(__file__))
_arch = os.path.dirname(_here)
_root = os.path.dirname(_arch)
for _p in (_here, _arch, os.path.join(_arch, 'ergodic_dataset_generator'), os.path.join(_root, 'SE3_SVGD')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault('MPLBACKEND', 'TkAgg')

def white_inferno():
    import matplotlib.colors as mcolors
    inf = plt.get_cmap('inferno')(np.linspace(0, 1, 256))
    n = 60
    ramp = np.linspace(0, 1, n)[:, None]
    inf[:n, :3] = (1 - ramp) * np.ones((n, 3)) + ramp * inf[:n, :3]
    return mcolors.LinearSegmentedColormap.from_list('white_inferno', inf)

def style(ax):
    ax.set_facecolor('white')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#ccc')

class Writeboard:
    TRUTH_RES = 96
    
    def __init__(self, truth_array, agent_info_array, shared_obstacles):
        self.truth_array = truth_array
        self.agent_info = agent_info_array
        self.shared_obstacles = shared_obstacles
        
        self.fig = plt.figure(figsize=(10, 8), facecolor='white')
        self.ax = self.fig.add_axes([0.05, 0.05, 0.9, 0.9])
        style(self.ax)
        self.ax.set_title("Writeboard (Draw Target, Agent Erases)", fontsize=14, pad=10)
        
        self.cmap = white_inferno()
        self.grid = np.zeros((self.TRUTH_RES, self.TRUTH_RES), dtype=np.float64)
        
        self.img = self.ax.imshow(
            self.grid, origin='lower', extent=[0, 1, 0, 1], cmap=self.cmap, vmin=0, vmax=1
        )
        
        # Agent visual elements
        # Eraser icon (rectangle) + radius outline
        self.agent_radius_circle = mpatches.Circle((0.5, 0.5), 0.06, facecolor='none',
                                                  edgecolor='#FF5722', alpha=0.5, lw=2,
                                                  ls='--', zorder=4)
        # Simple eraser shape (a pink/gray rectangle)
        self.agent_eraser = mpatches.Rectangle((0.5-0.03, 0.5-0.015), 0.06, 0.03,
                                               facecolor='#E0E0E0', edgecolor='#757575',
                                               alpha=0.9, lw=1.5, zorder=5)
                                               
        self.ax.add_patch(self.agent_radius_circle)
        self.ax.add_patch(self.agent_eraser)
        
        self.agent_radius_circle.set_visible(False)
        self.agent_eraser.set_visible(False)
        
        # Ghost patches for drag and drop
        self.ax_obs_template = self.fig.add_axes([0.01, 0.45, 0.04, 0.1])
        self.ax_obs_template.set_axis_off()
        self.ax_obs_template.set_xlim(0, 1)
        self.ax_obs_template.set_ylim(0, 1)
        self.ax_obs_template.set_aspect('equal')
        template_radius = 0.4
        self.ax_obs_template.add_patch(mpatches.Circle((0.5, 0.5), template_radius, facecolor='#9E9E9E', alpha=0.75, edgecolor='#424242', lw=1.5, zorder=6))
        self.ax_obs_template.text(0.5, -0.15, 'Drag Obstacle', ha='center', va='top', fontsize=8, color='#424242', transform=self.ax_obs_template.transAxes)

        self.obs_drag_patch = mpatches.Circle((0.5, 0.5), 0.05, facecolor='#9E9E9E', alpha=0.5, edgecolor='#424242', lw=1.5, zorder=7, visible=False)
        self.ax.add_patch(self.obs_drag_patch)
        self._dragging_obs = False
        
        # Pre-allocate 10 obstacle patches for visualization
        self.obs_patches = []
        for _ in range(10):
            patch = mpatches.Circle((0, 0), 0.05, facecolor='#9E9E9E', alpha=0.75, edgecolor='#424242', lw=1.5, zorder=6, visible=False)
            self.ax.add_patch(patch)
            self.obs_patches.append(patch)
        
        # Drawing state
        self.is_drawing = False
        self.last_mouse_event = None
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        
        ax_reset = self.fig.add_axes([0.4, 0.01, 0.1, 0.035])
        self.b_reset = Button(ax_reset, 'Reset Target')
        self.b_reset.on_clicked(self._on_reset)
        
        ax_clear_obs = self.fig.add_axes([0.55, 0.01, 0.15, 0.035])
        self.b_clear_obs = Button(ax_clear_obs, 'Clear Obstacles')
        self.b_clear_obs.on_clicked(self._on_clear_obs)
        
        self.timer = self.fig.canvas.new_timer(interval=50) # 20fps
        self.timer.add_callback(self._tick)
        self.timer.start()
        
        plt.show()

    def _on_press(self, event):
        if event.inaxes == self.ax:
            self.is_drawing = True
            self.last_mouse_event = event
            self._apply_brush(event)
        elif event.inaxes == self.ax_obs_template:
            self._dragging_obs = True
            self.obs_drag_patch.set_visible(False)
            
    def _on_release(self, event):
        self.is_drawing = False
        self.last_mouse_event = None
        if self._dragging_obs:
            self._dragging_obs = False
            self.obs_drag_patch.set_visible(False)
            if event.inaxes == self.ax:
                if self.shared_obstacles is not None:
                    with self.shared_obstacles.get_lock():
                        arr = np.frombuffer(self.shared_obstacles.get_obj(), dtype=np.float64)
                        for i in range(10):
                            if arr[i*3+2] == 0.0: # empty slot
                                arr[i*3] = event.xdata
                                arr[i*3+1] = event.ydata
                                arr[i*3+2] = 0.05 # default radius
                                break
            self.fig.canvas.draw_idle()
        
    def _on_reset(self, event):
        with self.truth_array.get_lock():
            np_truth = np.frombuffer(self.truth_array.get_obj(), dtype=np.float64).reshape((self.TRUTH_RES, self.TRUTH_RES))
            np_truth[:] = 0.0
            self.grid[:] = 0.0
        self.img.set_data(self.grid)
        self.fig.canvas.draw_idle()
        
    def _on_clear_obs(self, event):
        if self.shared_obstacles is not None:
            with self.shared_obstacles.get_lock():
                arr = np.frombuffer(self.shared_obstacles.get_obj(), dtype=np.float64)
                arr[:] = 0.0
        
    def _on_motion(self, event):
        if self.is_drawing and event.inaxes == self.ax:
            self.last_mouse_event = event
            self._apply_brush(event)
        elif self._dragging_obs:
            if event.inaxes == self.ax:
                self.obs_drag_patch.center = (event.xdata, event.ydata)
                self.obs_drag_patch.set_visible(True)
            else:
                self.obs_drag_patch.set_visible(False)
            self.fig.canvas.draw_idle()
            
    def _apply_brush(self, event):
        if event.xdata is None or event.ydata is None:
            return
            
        res = self.TRUTH_RES
        x_idx = int(np.clip(event.xdata * res, 0, res - 1))
        y_idx = int(np.clip(event.ydata * res, 0, res - 1))
        
        sigma = 2.0
        y, x = np.ogrid[-y_idx:res-y_idx, -x_idx:res-x_idx]
        blob = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        with self.truth_array.get_lock():
            # Read shared memory
            np_truth = np.frombuffer(self.truth_array.get_obj(), dtype=np.float64).reshape((res, res))
            # Modify
            np_truth += blob * 0.25
            np.clip(np_truth, 0, 1, out=np_truth)
            # Update local display grid instantly for responsiveness
            self.grid[:] = np_truth[:]
            
        self.img.set_data(self.grid)
        self.fig.canvas.draw_idle()

    def _tick(self):
        # Apply brush continuously if mouse is held down
        if self.is_drawing and self.last_mouse_event is not None:
            self._apply_brush(self.last_mouse_event)
            
        # Update from shared memory
        res = self.TRUTH_RES
        
        with self.agent_info.get_lock():
            ax, ay, a_rad, eraser_mode = self.agent_info[:]
            
        with self.truth_array.get_lock():
            np_truth = np.frombuffer(self.truth_array.get_obj(), dtype=np.float64).reshape((res, res))
            self.grid[:] = np_truth[:]
            
        self.img.set_data(self.grid)
        
        # Update agent position
        if eraser_mode > 0.5: # True
            self.agent_radius_circle.set_visible(True)
            self.agent_eraser.set_visible(True)
            self.agent_radius_circle.center = (ax, ay)
            self.agent_radius_circle.set_radius(a_rad)
            self.agent_eraser.set_xy((ax - a_rad * 0.8, ay - a_rad * 0.4))
            self.agent_eraser.set_width(a_rad * 1.6)
            self.agent_eraser.set_height(a_rad * 0.8)
            self.ax.set_title("Writeboard (Eraser Mode: ACTIVE)", color='#D32F2F', fontsize=14, pad=10)
        else:
            self.agent_radius_circle.set_visible(False)
            self.agent_eraser.set_visible(False)
            self.ax.set_title("Writeboard (Eraser Mode: INACTIVE)", color='#1A1A2E', fontsize=14, pad=10)
            
        # Update obstacles
        if self.shared_obstacles is not None:
            with self.shared_obstacles.get_lock():
                arr = np.frombuffer(self.shared_obstacles.get_obj(), dtype=np.float64).copy()
            for i in range(10):
                x, y, r = arr[i*3], arr[i*3+1], arr[i*3+2]
                if r > 0.0:
                    self.obs_patches[i].center = (x, y)
                    self.obs_patches[i].set_radius(r)
                    self.obs_patches[i].set_visible(True)
                else:
                    self.obs_patches[i].set_visible(False)
            
        self.fig.canvas.draw_idle()

def run_writeboard(truth_array, agent_info_array, shared_obstacles):
    Writeboard(truth_array, agent_info_array, shared_obstacles)
