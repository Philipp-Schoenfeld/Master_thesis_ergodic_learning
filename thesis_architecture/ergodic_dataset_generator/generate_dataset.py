"""
generate_dataset.py
===================
Ergodic dataset generator — produces (density, trajectory) pairs.

Run modes:
  python generate_dataset.py --mode preview   # 5 shapes, full visualisation
  python generate_dataset.py --mode full      # 775 shapes

Outputs
  ergodic_dataset_775.db          SQLite database
  visualizations/preview/         PNG per shape
  visualizations/preview_grid.png Summary grid
"""

import argparse
import json
import os
import random
import sqlite3
import time

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

_here    = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_here, 'ergodic_dataset_775.db')
_VIZ_DIR = os.path.join(_here, 'visualizations')

os.makedirs(os.path.join(_VIZ_DIR, 'preview'), exist_ok=True)
os.makedirs(os.path.join(_VIZ_DIR, 'full'),    exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────

def init_db(db_path=_DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ergodic_pairs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            shape_name     TEXT    NOT NULL,
            split          TEXT    NOT NULL DEFAULT 'train',
            density_params TEXT    NOT NULL,
            trajectory     BLOB    NOT NULL,
            x0             TEXT    NOT NULL,
            dt             REAL    NOT NULL,
            tsteps         INTEGER NOT NULL,
            generated_at   TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def store_pair(conn, shape_name, split, shape_def, traj_xy, x0, dt, tsteps):
    if shape_def.get('type') == 'analytical':
        params = {
            'type': 'analytical',
            'segments': shape_def['segments'],
            'sigma': shape_def.get('sigma', 0.025)
        }
    else:
        params = {
            'means':   shape_def['means'],
            'covs':    shape_def['covs'],
            'weights': [float(w) for w in shape_def['weights']],
        }
    # Der Sockel gehoert zwingend mit in die Datenbank: ohne ihn baut
    # `make_pdf_and_score` beim Laden eine andere Dichte als die, gegen die
    # der Solver optimiert hat.
    if shape_def.get('pedestal'):
        params['pedestal'] = shape_def['pedestal']
    params_json = json.dumps(params)
    traj_blob = traj_xy.astype(np.float32).tobytes()
    conn.execute(
        """INSERT INTO ergodic_pairs
           (shape_name, split, density_params, trajectory, x0, dt, tsteps, generated_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (shape_name, split, params_json, traj_blob,
         json.dumps(list(x0)), dt, tsteps,
         time.strftime('%Y-%m-%dT%H:%M:%S'))
    )
    conn.commit()


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_pair(shape_name, shape_def, traj_xy, init_traj, save_path, resolution=80):
    """Save a figure showing the GMM density contour + ergodic trajectory."""
    from shape_library import pdf_on_grid

    pdf_grid, gx, gy = pdf_on_grid(shape_def, resolution=resolution)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor='#0f0f1a')
    fig.suptitle(f'Shape: {shape_name}', color='white', fontsize=13,
                 fontweight='bold', y=1.01)

    for ax in axes:
        ax.set_facecolor('#0f0f1a')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        # Smooth density heatmap
        ax.imshow(pdf_grid, origin='lower', extent=[0, 1, 0, 1],
                  cmap='inferno', aspect='equal',
                  vmin=pdf_grid.min(), vmax=pdf_grid.max())

    # 1. Target density
    axes[0].set_title('Target Density  φ(x)', color='white', fontsize=10)
    
    # Plot init_traj on the density map
    if init_traj is not None:
        axes[0].plot(init_traj[:, 0], init_traj[:, 1], color='#FF0000', lw=1.5, alpha=0.8, label='Initialization')

    # 2. Trajectory
    axes[1].set_title('Ergodic Trajectory', color='white', fontsize=10)

    n = len(traj_xy)
    for i in range(n - 1):
        alpha = 0.5 + 0.5 * i / n
        axes[1].plot(traj_xy[i:i+2, 0], traj_xy[i:i+2, 1],
                     color='#FF00FF', lw=2.5, alpha=alpha)
    axes[1].scatter(traj_xy[0, 0], traj_xy[0, 1],
                    s=80, c='white', zorder=5)
    axes[1].scatter(traj_xy[-1, 0], traj_xy[-1, 1],
                    s=80, c='#FFD700', zorder=5, marker='*')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'    Saved → {os.path.relpath(save_path, _here)}')


def plot_preview_grid(results, save_path, resolution=70):
    """Save a summary grid of all preview shapes."""
    from shape_library import pdf_on_grid

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8),
                              facecolor='#0f0f1a',
                              gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    fig.suptitle('Ergodic Dataset — Preview', color='white',
                 fontsize=14, fontweight='bold', y=1.01)

    for col, (name, shape_def, traj_xy, init_traj) in enumerate(results):
        pdf_grid, gx, gy = pdf_on_grid(shape_def, resolution=resolution)

        for row in range(2):
            ax = axes[row, col]
            ax.set_facecolor('#0f0f1a')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect('equal'); ax.axis('off')
            ax.imshow(pdf_grid, origin='lower', extent=[0, 1, 0, 1],
                      cmap='inferno', aspect='equal')
            if row == 0:
                ax.set_title(name, color='white', fontsize=10, pad=4)
                # Plot init_traj on the density map
                if init_traj is not None:
                    ax.plot(init_traj[:, 0], init_traj[:, 1], color='white', lw=1.0, linestyle='--', alpha=0.6)

        # row 1: add trajectory
        n_pts = len(traj_xy)
        for i in range(n_pts - 1):
            alpha = 0.5 + 0.5 * i / n_pts
            axes[1, col].plot(traj_xy[i:i+2, 0], traj_xy[i:i+2, 1],
                              color='#FF00FF', lw=2.5, alpha=alpha)
        axes[1, col].scatter(*traj_xy[0], s=50, c='white', zorder=5)
        axes[1, col].scatter(*traj_xy[-1], s=60, c='#FFD700', zorder=5, marker='*')

    axes[0, 0].text(-0.08, 0.5, 'Density', transform=axes[0, 0].transAxes,
                    color='white', fontsize=9, va='center', rotation=90)
    axes[1, 0].text(-0.08, 0.5, 'Trajectory', transform=axes[1, 0].transAxes,
                    color='white', fontsize=9, va='center', rotation=90)

    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f'  Grid saved → {os.path.relpath(save_path, _here)}')


def get_initial_trajectory(shape_def, x0=(0.1, 0.2), dt=0.05, tsteps=200):
    from ergodic_solver import _generate_initial_trajectory, _build_lqr
    import numpy as np
    import jax.numpy as jnp
    if 'means' in shape_def or shape_def.get('type') == 'analytical':
        p_traj, u_traj_np, v0 = _generate_initial_trajectory(x0, shape_def, tsteps, dt)
        x0j = jnp.array([x0[0], x0[1], v0[0], v0[1]])
        u_traj = jnp.array(u_traj_np)
    else:
        T = dt * tsteps
        x0j = jnp.array([x0[0], x0[1], 2.0*(0.5-x0[0])/T, 2.0*(0.5-x0[1])/T])
        u_traj = jnp.zeros((tsteps, 2))
    
    pm, _, _ = _build_lqr(dt)
    init_traj = np.array(pm.traj_sim(x0j, u_traj))[:, :2]
    return init_traj


def plot_all_targets(shape_names, save_dir, resolution=70):
    from shape_library import get_shape, pdf_on_grid
    import math
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = 100
    for b_idx in range(math.ceil(len(shape_names) / batch_size)):
        batch = shape_names[b_idx * batch_size : (b_idx + 1) * batch_size]
        cols = 10
        rows = math.ceil(len(batch) / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), facecolor='#0f0f1a')
        fig.suptitle(f'Target Distributions & Init Trajectories (Batch {b_idx + 1})', color='white', fontsize=16, fontweight='bold', y=1.02)
        
        # Flatten axes properly
        import numpy as np
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        from tqdm import tqdm
        print(f"  Rendering density grid batch {b_idx+1}...")
        for i, name in enumerate(tqdm(batch, leave=False)):
            ax = axes_flat[i]
            shape_def = get_shape(name)
            pdf_grid, _, _ = pdf_on_grid(shape_def, resolution=resolution)
            
            rng = np.random.default_rng(abs(hash(name)) % (2**31))
            x0 = tuple(rng.uniform(0.05, 0.25, size=2))
            init_traj = get_initial_trajectory(shape_def, x0=x0)
            
            ax.set_facecolor('#0f0f1a')
            ax.axis('off')
            ax.imshow(pdf_grid, origin='lower', extent=[0, 1, 0, 1], cmap='inferno')
            ax.plot(init_traj[:, 0], init_traj[:, 1], color='#a500ff', linewidth=1.0, alpha=0.9)
            ax.plot(init_traj[0, 0], init_traj[0, 1], 'wo', markersize=3)
            ax.set_title(name, color='white', fontsize=8, pad=2)
            
        # Turn off unused axes
        for j in range(len(batch), len(axes_flat)):
            axes_flat[j].axis('off')
            axes_flat[j].set_facecolor('#0f0f1a')
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'all_targets_grid_{b_idx + 1}.png')
        plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved target grid → {os.path.relpath(save_path, _here)}")


# ── Generation ────────────────────────────────────────────────────────────────

def generate_shapes(shape_names, split, conn, viz_dir, solver_kwargs, verbose=True):
    a, b = getattr(_ARGS, 'shapes_from', None), getattr(_ARGS, 'shapes_to', None)
    if a is not None or b is not None:
        shape_names = list(shape_names)[a or 0:b]
        print(f'  Abschnitt {a or 0}:{b if b is not None else "Ende"} '
              f'-> {len(shape_names)} Formen')
    """Generate ergodic trajectories for a list of shape names."""
    from shape_library import get_shape, make_pdf_and_score
    from ergodic_solver import run_ergodic_coverage

    os.makedirs(viz_dir, exist_ok=True)

    # Fetch already existing shapes for this split to resume gracefully
    existing_shapes = set(row[0] for row in conn.execute("SELECT shape_name FROM ergodic_pairs WHERE split=?", (split,)).fetchall())

    results = []
    from tqdm import tqdm
    for name in tqdm(shape_names, desc='Total Progress', position=0):
        if name in existing_shapes:
            if verbose:
                print(f'\n  [{name}] Already exists in {split}, skipping...')
            continue

        if verbose:
            print(f'\n  [{name}] Building shape & score function …')
        shape_def = get_shape(name)
        _, score_fn = make_pdf_and_score(shape_def)

        rng = np.random.default_rng(abs(hash(name)) % (2**31))
        if getattr(_ARGS, 'x0_mode', 'ecke') == 'ueberall':
            m = getattr(_ARGS, 'x0_margin', 0.03)
            x0 = tuple(rng.uniform(m, 1.0 - m, size=2))
        else:
            x0 = tuple(rng.uniform(0.05, 0.25, size=2))

        # Increase iterations for complex shapes
        kwargs = solver_kwargs.copy()
        if name.startswith('test_gmm_') and int(name.split('_')[-1]) >= 6:
            if kwargs.get('num_iters') == 600:
                kwargs['num_iters'] = 1500
            if kwargs.get('tsteps') == 200:
                kwargs['tsteps'] = 400

        print(f'  [{name}] Running ergodic solver  x0={x0} (iters={kwargs["num_iters"]}, tsteps={kwargs["tsteps"]}) …')
        t0 = time.perf_counter()
        traj_xy, init_traj = run_ergodic_coverage(
            score_fn, x0=x0, shape_def=shape_def, verbose=verbose, **kwargs
        )
        elapsed = time.perf_counter() - t0
        print(f'  [{name}] Done in {elapsed:.1f}s  traj_shape={traj_xy.shape}')

        store_pair(conn, name, split, shape_def, traj_xy,
                   x0, kwargs['dt'], kwargs['tsteps'])

        png = os.path.join(viz_dir, f'{name}_iters{kwargs["num_iters"]}_scale{kwargs["score_scale"]}.png')
        plot_pair(name, shape_def, traj_xy, init_traj, png)
        results.append((name, shape_def, traj_xy, init_traj))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',       choices=['preview', 'full', 'test_new', 'test_complex', 'flat'], default='preview',
                   help="'preview' = 5 shapes; 'full' = 775 shapes; 'test_new' = 10 custom test GMMs; "
                        "'test_complex' = 30 highly complex shapes; 'flat' = 400 flache Formen "
                        "(Sockel, weichgezeichnet, breite Moden, Konturen) + 12 flache Holdouts. "
                        "Ergaenzt einen bestehenden Datensatz, ohne die Splits 'train'/'val' zu veraendern.")
    p.add_argument('--dt',         type=float, default=0.05)
    p.add_argument('--tsteps',     type=int,   default=200,
                   help="Trajectory timesteps (output shape: tsteps+1)")
    p.add_argument('--num_iters',  type=int,   default=600,
                   help='Number of SVGD iterations (default: 600)')
    p.add_argument('--step_size',  type=float, default=0.01)
    p.add_argument('--score_scale',type=float, default=1.0,
                   help="Multiplier for the target density score function (default: 1.0)")
    p.add_argument('--h',          type=float, default=0.01,
                   help="RBF kernel bandwidth")
    p.add_argument('--x0_mode', choices=['ecke', 'ueberall'], default='ecke',
                   help="Wo der Startpunkt gezogen wird. 'ecke' ist das "
                        "bisherige Verhalten (unten links, [0.05, 0.25]^2); "
                        "'ueberall' zieht gleichverteilt ueber die ganze "
                        "Flaeche, sodass die Bahn eine echte Anfahrt braucht.")
    p.add_argument('--x0_margin', type=float, default=0.03,
                   help='Randabstand bei --x0_mode ueberall.')
    p.add_argument('--seed', type=int, default=None,
                   help='Zufallskeim fuer die Startpunkte.')
    p.add_argument('--shapes_from', type=int, default=None,
                   help='Nur Formen ab diesem Index bearbeiten — fuer das '
                        'Aufteilen des Laufs auf mehrere Jobs.')
    p.add_argument('--shapes_to', type=int, default=None)
    p.add_argument('--db',         type=str,   default=_DB_PATH)
    return p.parse_args()


_ARGS = None


def main():
    global _ARGS
    args = parse_args()
    _ARGS = args
    if args.seed is not None:
        import numpy as _np
        _np.random.seed(args.seed)
    from shape_library import (get_shape, PREVIEW_SHAPES,
                               VALIDATION_SHAPES, train_shape_names)

    conn = init_db(args.db)
    solver_kwargs = dict(
        dt=args.dt, tsteps=args.tsteps,
        num_iters=args.num_iters,
        step_size=args.step_size, h=args.h,
        score_scale=args.score_scale
    )

    if args.mode == 'preview':
        print('\n' + '='*60)
        print('  PREVIEW MODE — generating 5 shapes')
        print('='*60)
        results = generate_shapes(
            PREVIEW_SHAPES, split='preview',
            conn=conn, viz_dir=os.path.join(_VIZ_DIR, 'preview'),
            solver_kwargs=solver_kwargs, verbose=True,
        )
        grid_path = os.path.join(_VIZ_DIR, f'preview_grid_iters{args.num_iters}_scale{args.score_scale}.png')
        plot_preview_grid(results, grid_path)
        print(f'\n  Preview complete. Check visualizations/preview/ and {os.path.basename(grid_path)}')

    elif args.mode == 'test_new':
        from shape_library import TEST_NEW_SHAPES
        print('\n' + '='*60)
        print('  TEST_NEW MODE — generating custom test distributions')
        print('='*60)
        os.makedirs(os.path.join(_VIZ_DIR, 'test_new'), exist_ok=True)
        results = generate_shapes(
            TEST_NEW_SHAPES, split='test',
            conn=conn, viz_dir=os.path.join(_VIZ_DIR, 'test_new'),
            solver_kwargs=solver_kwargs, verbose=True,
        )
        grid_path = os.path.join(_VIZ_DIR, f'test_new_grid_iters{args.num_iters}_scale{args.score_scale}.png')
        plot_preview_grid(results, grid_path)
        print(f'\n  Test_new complete. Check visualizations/test_new/ and {os.path.basename(grid_path)}')

    elif args.mode == 'test_complex':
        from shape_library import TEST_COMPLEX_SHAPES
        print('\n' + '='*60)
        print('  TEST_COMPLEX MODE — generating 30 highly complex test distributions')
        print('='*60)
        os.makedirs(os.path.join(_VIZ_DIR, 'test_complex'), exist_ok=True)
        results = generate_shapes(
            TEST_COMPLEX_SHAPES, split='test',
            conn=conn, viz_dir=os.path.join(_VIZ_DIR, 'test_complex'),
            solver_kwargs=solver_kwargs, verbose=True,
        )
        grid_path = os.path.join(_VIZ_DIR, f'test_complex_grid_iters{args.num_iters}_scale{args.score_scale}.png')
        plot_preview_grid(results, grid_path)
        print(f'\n  Test_complex complete. Check visualizations/test_complex/ and {os.path.basename(grid_path)}')

    elif args.mode == 'flat':
        from shape_library import flat_shape_names
        train_names = flat_shape_names('train')
        val_names   = flat_shape_names('val')
        print('\n' + '='*60)
        print('  FLAT MODE — %d flache Trainingsformen + %d flache Holdouts'
              % (len(train_names), len(val_names)))
        print('='*60)
        print('  Diese Formen ergaenzen einen bestehenden Datensatz. Der Split')
        print("  'train' waechst, 'val' bleibt unangetastet; die flachen")
        print("  Holdouts liegen im eigenen Split 'val_flat'.")
        generate_shapes(
            train_names, split='train',
            conn=conn, viz_dir=os.path.join(_VIZ_DIR, 'flat'),
            solver_kwargs=solver_kwargs, verbose=False,
        )
        generate_shapes(
            val_names, split='val_flat',
            conn=conn, viz_dir=os.path.join(_VIZ_DIR, 'flat'),
            solver_kwargs=solver_kwargs, verbose=False,
        )
        print('\n  Flache Formen fertig.')

    else:  # full
        print('\n' + '='*60)
        print('  FULL MODE — generating 775 shapes')
        print('='*60)
        val_names   = VALIDATION_SHAPES
        train_names = train_shape_names(750)
        all_names = val_names + train_names
        print(f'  Train: {len(train_names)}  |  Val: {len(val_names)}  |  Total: {len(all_names)}')
        
        # Pre-visualize target distributions and initial trajectories
        plot_all_targets(all_names, os.path.join(_VIZ_DIR, 'all_targets'))
        print('\n  Visualizations saved to visualizations/all_targets/')
        print('  Automatically proceeding to Ergodic trajectory generation...\n')
        
        generate_shapes(
            train_names, split='train',
            conn=conn, viz_dir=os.path.join(_VIZ_DIR, 'full'),
            solver_kwargs=solver_kwargs, verbose=False,
        )
        generate_shapes(
            val_names, split='val',
            conn=conn, viz_dir=os.path.join(_VIZ_DIR, 'full'),
            solver_kwargs=solver_kwargs, verbose=False,
        )
        print('\n  Full dataset generation complete.')

    conn.close()


if __name__ == '__main__':
    main()
