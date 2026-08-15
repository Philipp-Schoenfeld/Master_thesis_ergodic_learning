import os
os.environ['MPLBACKEND'] = 'TkAgg'
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
print("Using backend:", plt.get_backend())

from shape_library import TEST_NEW_SHAPES, get_shape, pdf_on_grid, make_pdf_and_score
from ergodic_solver import run_ergodic_coverage

class InteractiveDrawer:
    def __init__(self, shape_name, shape_def, tsteps=200, dt=0.05, num_iters=200):
        self.shape_name = shape_name
        self.shape_def = shape_def
        self.tsteps = tsteps
        self.dt = dt
        self.num_iters = num_iters
        
        self.path = []
        self.drawing = False
        self.solved = False
        
        self.pdf_grid, self.gx, self.gy = pdf_on_grid(shape_def, resolution=100)
        
        self.fig, self.ax = plt.subplots(figsize=(7, 7), facecolor='#0f0f1a')
        self.ax.set_facecolor('#0f0f1a')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Draw Initial Path: {shape_name}', color='white')
        
        self.ax.imshow(self.pdf_grid, origin='lower', extent=[0, 1, 0, 1],
                       cmap='inferno', aspect='equal',
                       vmin=self.pdf_grid.min(), vmax=self.pdf_grid.max())
        
        self.line, = self.ax.plot([], [], 'w--', lw=2)
        self.traj_lines = []
        
        # Connect events
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Buttons
        ax_solve = plt.axes([0.1, 0.02, 0.2, 0.05])
        self.btn_solve = Button(ax_solve, 'Solve')
        self.btn_solve.on_clicked(self.solve)
        
        ax_clear = plt.axes([0.35, 0.02, 0.2, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_clear.on_clicked(self.clear)
        
        ax_next = plt.axes([0.7, 0.02, 0.2, 0.05])
        self.btn_next = Button(ax_next, 'Save & Next')
        self.btn_next.on_clicked(self.next_shape)
        
        self.finished = False

    def on_press(self, event):
        if event.inaxes != self.ax: return
        if self.solved: return # don't draw if already solved
        self.drawing = True
        self.path = [(event.xdata, event.ydata)]
        self.update_line()

    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax: return
        self.path.append((event.xdata, event.ydata))
        self.update_line()

    def on_release(self, event):
        self.drawing = False

    def update_line(self):
        if not self.path:
            self.line.set_data([], [])
        else:
            self.line.set_data(*zip(*self.path))
        self.fig.canvas.draw_idle()

    def clear(self, event):
        self.path = []
        self.solved = False
        self.update_line()
        for l in self.traj_lines:
            l.remove()
        self.traj_lines = []
        self.ax.set_title(f'Draw Initial Path: {self.shape_name}', color='white')
        self.fig.canvas.draw_idle()

    def solve(self, event):
        if len(self.path) < 2:
            print("Please draw a longer path first.")
            return
            
        self.ax.set_title('Solving... Please wait.', color='yellow')
        self.fig.canvas.draw_idle()
        plt.pause(0.1) # allow GUI to update
        
        path = np.array(self.path)
        
        # Interpolate drawn path to exact tsteps + 1
        diffs = np.diff(path, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.concatenate(([0], np.cumsum(dists)))
        total_dist = cum_dists[-1]
        
        if total_dist < 1e-5:
            return
            
        idx_eval = np.linspace(0, total_dist, self.tsteps + 1)
        p_traj = np.column_stack([
            np.interp(idx_eval, cum_dists, path[:, 0]),
            np.interp(idx_eval, cum_dists, path[:, 1])
        ])
        
        from scipy.ndimage import gaussian_filter1d
        p_traj = gaussian_filter1d(p_traj, sigma=3, axis=0, mode='nearest')
        
        # Build score fn right before solving (lazy)
        _, score_fn = make_pdf_and_score(self.shape_def)
        
        # Run ergodic coverage
        t0 = time.perf_counter()
        traj_xy, _ = run_ergodic_coverage(
            score_fn, x0=p_traj[0], shape_def=self.shape_def, 
            custom_p_traj=p_traj, dt=self.dt, tsteps=self.tsteps, num_iters=self.num_iters, verbose=True
        )
        elapsed = time.perf_counter() - t0
        
        # Plot optimized trajectory
        n = len(traj_xy)
        for i in range(n - 1):
            alpha = 0.3 + 0.7 * i / n
            l, = self.ax.plot(traj_xy[i:i+2, 0], traj_xy[i:i+2, 1],
                              color='#00E5FF', lw=1.5, alpha=alpha, zorder=10)
            self.traj_lines.append(l)
            
        self.solved = True
        
        # Auto-save immediately so the user doesn't lose it if they click X
        self.save_figure()
        
        self.ax.set_title(f'Optimized in {elapsed:.1f}s. Redraw or Next.', color='white')
        self.fig.canvas.draw()
        plt.pause(0.01)

    def save_figure(self):
        save_dir = 'visualizations/interactive_draw'
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'{self.shape_name}.png')
        self.fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor=self.fig.get_facecolor())
        print(f"  -> Saved drawn trajectory to {filename}")

    def next_shape(self, event):
        self.finished = True
        plt.close(self.fig)


def run_interactive():
    # Only test the 15 complex test shapes for now
    test_shapes = TEST_NEW_SHAPES
    print(f"Found {len(test_shapes)} test shapes for interactive drawing.")
    
    for name in test_shapes:
        shape_def = get_shape(name)
        
        # Determine tsteps/iters based on complexity
        if len(shape_def['means']) > 5:
            tsteps = 400
            num_iters = 2000
        else:
            tsteps = 200
            num_iters = 1000
            
        drawer = InteractiveDrawer(name, shape_def, tsteps=tsteps, num_iters=num_iters)
        plt.show() # Blocks until the figure is closed
        print(f"Finished {name}")

if __name__ == '__main__':
    run_interactive()
