"""
review_dataset.py
=================
Interactive UI for reviewing generated ergodic trajectories.
Displays shapes that have status 'pending' in the database.
Allows you to accept or reject them.

Usage:
  python review_dataset.py
"""

import os
os.environ['MPLBACKEND'] = 'TkAgg'
import json
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from shape_library import pdf_on_grid

_here = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_here, 'ergodic_dataset.db')


class ReviewUI:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.pending_items = self._fetch_pending()
        self.current_idx = 0

        if not self.pending_items:
            print("No pending shapes to review. All caught up!")
            self.conn.close()
            return

        self.fig, self.axes = plt.subplots(1, 2, figsize=(11, 5), facecolor='#0f0f1a')
        plt.subplots_adjust(bottom=0.2)
        
        # Setup Buttons
        ax_accept = plt.axes([0.6, 0.05, 0.15, 0.075])
        ax_reject = plt.axes([0.4, 0.05, 0.15, 0.075])
        ax_quit = plt.axes([0.8, 0.05, 0.1, 0.075])

        self.btn_accept = Button(ax_accept, 'Accept', color='#2E7D32', hovercolor='#1B5E20')
        self.btn_reject = Button(ax_reject, 'Reject', color='#C62828', hovercolor='#b71c1c')
        self.btn_quit = Button(ax_quit, 'Quit', color='#424242', hovercolor='#212121')

        # Button styling
        for btn in (self.btn_accept, self.btn_reject, self.btn_quit):
            btn.label.set_color('white')
            btn.label.set_fontweight('bold')

        self.btn_accept.on_clicked(self.accept)
        self.btn_reject.on_clicked(self.reject)
        self.btn_quit.on_clicked(self.quit)

        # Plot first shape
        self.update_plot()
        plt.show()

    def _fetch_pending(self):
        """Fetch all rows that are still 'pending'."""
        cur = self.conn.cursor()
        # Verify the column exists, if not, wait for it (should be created already)
        try:
            cur.execute("SELECT id, shape_name, split, density_params, trajectory FROM ergodic_pairs WHERE status = 'pending' ORDER BY id ASC")
            return cur.fetchall()
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}")
            print("Ensure you have run the ALTER TABLE command to add the 'status' column.")
            return []

    def update_plot(self):
        if self.current_idx >= len(self.pending_items):
            self.fig.suptitle("Review Complete!", color='white', fontsize=16, fontweight='bold')
            for ax in self.axes:
                ax.clear()
                ax.set_facecolor('#0f0f1a')
                ax.axis('off')
            self.fig.canvas.draw()
            return

        row_id, shape_name, split, params_json, traj_blob = self.pending_items[self.current_idx]
        shape_def = json.loads(params_json)
        traj_xy = np.frombuffer(traj_blob, dtype=np.float32).reshape(-1, 2)

        # Reconstruct full shape_def to match what pdf_on_grid expects
        if shape_def.get('type') == 'analytical':
            # The loaded dict has 'type', 'segments', 'sigma'
            pass 
        else:
            shape_def['type'] = 'gmm'
            shape_def['means'] = np.array(shape_def['means'])
            shape_def['covs'] = np.array(shape_def['covs'])
            shape_def['weights'] = np.array(shape_def['weights'])

        pdf_grid, gx, gy = pdf_on_grid(shape_def, resolution=80)

        for ax in self.axes:
            ax.clear()
            ax.set_facecolor('#0f0f1a')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.imshow(pdf_grid, origin='lower', extent=[0, 1, 0, 1],
                      cmap='inferno', aspect='equal',
                      vmin=pdf_grid.min(), vmax=pdf_grid.max())

        # Title
        progress = f"[{self.current_idx + 1}/{len(self.pending_items)}]"
        self.fig.suptitle(f'{progress} Shape: {shape_name} (Split: {split})', 
                          color='white', fontsize=14, fontweight='bold', y=0.95)

        # Plot 1: Target Density
        self.axes[0].set_title('Target Density  φ(x)', color='white', fontsize=11)
        
        # Plot 2: Ergodic Trajectory
        self.axes[1].set_title('Ergodic Trajectory', color='white', fontsize=11)
        n = len(traj_xy)
        for i in range(n - 1):
            alpha = 0.5 + 0.5 * i / n
            self.axes[1].plot(traj_xy[i:i+2, 0], traj_xy[i:i+2, 1],
                         color='#FF00FF', lw=2.5, alpha=alpha)
        
        self.axes[1].scatter(traj_xy[0, 0], traj_xy[0, 1], s=80, c='white', zorder=5)
        self.axes[1].scatter(traj_xy[-1, 0], traj_xy[-1, 1], s=80, c='#FFD700', zorder=5, marker='*')

        self.fig.canvas.draw()

    def set_status(self, status):
        if self.current_idx >= len(self.pending_items):
            return
            
        row_id = self.pending_items[self.current_idx][0]
        cur = self.conn.cursor()
        cur.execute("UPDATE ergodic_pairs SET status = ? WHERE id = ?", (status, row_id))
        self.conn.commit()
        
        print(f"Marked shape ID {row_id} as '{status}'.")
        self.current_idx += 1
        self.update_plot()

    def accept(self, event):
        self.set_status('accepted')

    def reject(self, event):
        self.set_status('rejected')

    def quit(self, event):
        print("Exiting review...")
        plt.close()


if __name__ == '__main__':
    print("Launching Dataset Review UI...")
    ReviewUI(_DB_PATH)
