import sys

with open('flow_matching_runner_ergodic.py', 'r') as f:
    content = f.read()

# 1. Imports
content = content.replace(
    'import argparse, os, random, sqlite3, sys, math\nimport matplotlib.pyplot as plt',
    'import argparse, os, random, sqlite3, sys, math, json\nimport matplotlib.pyplot as plt'
)

# 2. Add pdf_on_grid
content = content.replace(
    'from flow_matching_cond_spectral_crossattn import (',
    'from ergodic_dataset_generator.shape_library import pdf_on_grid\nfrom flow_matching_cond_spectral_crossattn import ('
)

# 3. Data loader
old_loader = """def _load_shapes(nxi):
    conn = sqlite3.connect(_DB_PATH)
    cur  = conn.cursor()
    # Only load shapes from train and val split; testing shapes can be ignored or loaded if needed
    cur.execute("SELECT trajectory, shape_name, split FROM ergodic_pairs WHERE split IN ('train', 'val') ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    train_shapes = {}
    val_shapes = {}
    
    for blob, label, split in rows:
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
        idx = np.linspace(0, len(xy) - 1, nxi).astype(int)
        
        if split == 'val':
            key = label if label not in val_shapes else f"{label}#{len(val_shapes)}"
            val_shapes[key] = xy[idx]
        else:
            key = label if label not in train_shapes else f"{label}#{len(train_shapes)}"
            train_shapes[key] = xy[idx]
            
    return train_shapes, val_shapes"""

new_loader = """def _load_shapes(nxi):
    conn = sqlite3.connect(_DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT trajectory, shape_name, split, density_params FROM ergodic_pairs WHERE split IN ('train', 'val') ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    train_shapes = {}
    val_shapes = {}
    train_densities = {}
    val_densities = {}
    
    for blob, label, split, density_params in rows:
        xy = np.frombuffer(blob, dtype=np.float32).reshape(-1, 2)
        idx = np.linspace(0, len(xy) - 1, nxi).astype(int)
        
        if split == 'val':
            key = label if label not in val_shapes else f"{label}#{len(val_shapes)}"
            val_shapes[key] = xy[idx]
            val_densities[key] = density_params
        else:
            key = label if label not in train_shapes else f"{label}#{len(train_shapes)}"
            train_shapes[key] = xy[idx]
            train_densities[key] = density_params
            
    return train_shapes, val_shapes, train_densities, val_densities"""
content = content.replace(old_loader, new_loader)

# 4. _save_viz signature
content = content.replace(
    'def _save_viz(model, holdout_shapes, holdout_spec, title, ep_num, args, device, tag=None):',
    'def _save_viz(model, holdout_shapes, holdout_spec, holdout_densities, title, ep_num, args, device, tag=None):'
)
content = content.replace(
    'run_str  = f"S{args.S}_nxi{args.nxi}_D{args.D}_flip{args.p_flip}"',
    'run_str  = f"ergodic_S{args.S}_nxi{args.nxi}_D{args.D}_flip{args.p_flip}"'
)
content = content.replace(
    'visualise_set(model, holdout_shapes, holdout_spec, title, viz_path, args, device, max_cols=5)',
    'visualise_set(model, holdout_shapes, holdout_spec, holdout_densities, title, viz_path, args, device, max_cols=5)'
)

# 5. train wrapper calls
content = content.replace(
    'def train(model, x1_clean, loss_fn, args, holdout_shapes=None, holdout_spec=None):',
    'def train(model, x1_clean, loss_fn, args, holdout_shapes=None, holdout_spec=None, holdout_densities=None):'
)
content = content.replace(
    '_save_viz(model, holdout_shapes, holdout_spec,\n                          f"Holdout Shapes - Epoch {ep+1}", ep + 1, args, x1_clean.device)',
    '_save_viz(model, holdout_shapes, holdout_spec, holdout_densities,\n                          f"Holdout Shapes - Epoch {ep+1}", ep + 1, args, x1_clean.device)'
)
content = content.replace(
    '_save_viz(model, holdout_shapes, holdout_spec,\n                      f"Emergency Holdout - Epoch {ep+1}", ep + 1,\n                      args, x1_clean.device, tag="emergency")',
    '_save_viz(model, holdout_shapes, holdout_spec, holdout_densities,\n                      f"Emergency Holdout - Epoch {ep+1}", ep + 1,\n                      args, x1_clean.device, tag="emergency")'
)

# 6. _draw_traj and visualise_set
old_viz = """def _draw_traj(ax, base, gen_cps, title, bspline_pts=512, bspline_deg=5):
    ax.set_facecolor('white')
    if len(base) >= 6:
        ax.plot(*cp_to_bspline(base, bspline_pts, bspline_deg).T,
                color='#1565C0', lw=2.5, label='Ground Truth', zorder=2)
        ax.scatter(base[:, 0], base[:, 1],
                   color='#1565C0', s=12, alpha=0.5, zorder=2)
    for i, cp in enumerate(gen_cps):
        main_alpha = 0.85 if i == 0 else 0.2
        scatter_alpha = 0.4 if i == 0 else 0.1
        if len(cp) >= 6:
            ax.plot(*cp_to_bspline(cp, bspline_pts, bspline_deg).T,
                    color='#EF5350', lw=1.8, alpha=main_alpha,
                    label='Generated' if i == 0 else '', zorder=3)
        ax.scatter(cp[:, 0], cp[:, 1],
                   color='#EF5350', s=8, alpha=scatter_alpha, zorder=3)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.10, lw=0.5)
    ax.set_title(title, fontsize=10, color='#1A1A2E', pad=6)
    ax.legend(frameon=False, fontsize=7, loc='upper left')


def visualise_set(model, shapes_dict, spectral_dict, title_prefix, save_path,
                  args, device, max_cols=5):
    \"\"\"Generate & plot trajectories for a set of shapes (spectral conditioned).\"\"\"
    labels = list(shapes_dict.keys())
    n = len(labels)
    if n == 0:
        return
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 5 * n_rows),
                              facecolor='white', squeeze=False)
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold',
                 color='#1A1A2E', y=1.01)

    for idx, lbl in enumerate(labels):
        ax = axes[idx // n_cols][idx % n_cols]
        base = shapes_dict[lbl]
        spec, k_idx = spectral_dict[lbl]
        spec_t = torch.tensor(spec, dtype=torch.float32)
        k_idx_t = torch.tensor(k_idx, dtype=torch.long)

        gen, lam = generate_spectral_trajectories(
            model, spec_t, k_idx_t,
            num_samples=args.n_gen, nxi=args.nxi, nd=args.nd,
            steps=args.steps, device=str(device),
            cfg_weight=args.cfg_weight,
        )
        gen = gen.cpu().numpy()
        _draw_traj(ax, base, gen, f"'{lbl}'",
                   args.bspline_pts, args.bspline_deg)

        # Show lambda if predicted
        if lam is not None:
            lam_str = ", ".join(f"{v:.2f}" for v in lam[0].cpu().numpy())
            ax.text(0.02, 0.02, f"lam=[{lam_str}]",
                    transform=ax.transAxes, fontsize=5, alpha=0.5,
                    verticalalignment='bottom')

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")"""

new_viz = """def _draw_traj(ax, base, gen_cps, density_json, title, bspline_pts=512, bspline_deg=5):
    ax.set_facecolor('#0f0f1a')
    
    if density_json:
        shape_def = json.loads(density_json)
        if shape_def.get('type') != 'analytical':
            shape_def['type'] = 'gmm'
            shape_def['means'] = np.array(shape_def['means'])
            shape_def['covs'] = np.array(shape_def['covs'])
            shape_def['weights'] = np.array(shape_def['weights'])
        pdf_grid, _, _ = pdf_on_grid(shape_def, resolution=50)
        ax.imshow(pdf_grid, origin='lower', extent=[0, 1, 0, 1], cmap='inferno', aspect='equal')
        
    if len(base) >= 6:
        ax.plot(*cp_to_bspline(base, bspline_pts, bspline_deg).T,
                color='#1565C0', lw=2.5, label='Ground Truth', zorder=2)
        ax.scatter(base[:, 0], base[:, 1],
                   color='#1565C0', s=12, alpha=0.5, zorder=2)
    for i, cp in enumerate(gen_cps):
        main_alpha = 0.85 if i == 0 else 0.2
        scatter_alpha = 0.4 if i == 0 else 0.1
        if len(cp) >= 6:
            ax.plot(*cp_to_bspline(cp, bspline_pts, bspline_deg).T,
                    color='#FF00FF', lw=1.8, alpha=main_alpha,
                    label='Generated' if i == 0 else '', zorder=3)
        ax.scatter(cp[:, 0], cp[:, 1],
                   color='white', s=8, alpha=scatter_alpha, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, color='white', pad=6)


def visualise_set(model, shapes_dict, spectral_dict, density_dict, title_prefix, save_path,
                  args, device, max_cols=5):
    \"\"\"Generate & plot trajectories for a set of shapes (spectral conditioned).\"\"\"
    labels = list(shapes_dict.keys())
    n = len(labels)
    if n == 0:
        return
    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.5 * n_cols, 5 * n_rows),
                              facecolor='#0f0f1a', squeeze=False)
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold',
                 color='white', y=1.01)

    for idx, lbl in enumerate(labels):
        ax = axes[idx // n_cols][idx % n_cols]
        base = shapes_dict[lbl]
        spec, k_idx = spectral_dict[lbl]
        density = density_dict[lbl]
        spec_t = torch.tensor(spec, dtype=torch.float32)
        k_idx_t = torch.tensor(k_idx, dtype=torch.long)

        gen, lam = generate_spectral_trajectories(
            model, spec_t, k_idx_t,
            num_samples=args.n_gen, nxi=args.nxi, nd=args.nd,
            steps=args.steps, device=str(device),
            cfg_weight=args.cfg_weight,
        )
        gen = gen.cpu().numpy()
        _draw_traj(ax, base, gen, density, f"'{lbl}'",
                   args.bspline_pts, args.bspline_deg)

        # Show lambda if predicted
        if lam is not None:
            lam_str = ", ".join(f"{v:.2f}" for v in lam[0].cpu().numpy())
            ax.text(0.02, 0.02, f"lam=[{lam_str}]",
                    transform=ax.transAxes, fontsize=5, color='white', alpha=0.5,
                    verticalalignment='bottom')

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {save_path}")"""

content = content.replace(old_viz, new_viz)

# 7. main run updates
content = content.replace(
    'train_shapes, holdout_shapes = _load_shapes(args.nxi)',
    'train_shapes, holdout_shapes, train_densities, holdout_densities = _load_shapes(args.nxi)'
)
content = content.replace(
    'train(model, x1, compute_spectral_cfm_loss, args,\n              holdout_shapes=holdout_shapes, holdout_spec=holdout_spec)',
    'train(model, x1, compute_spectral_cfm_loss, args,\n              holdout_shapes=holdout_shapes, holdout_spec=holdout_spec, holdout_densities=holdout_densities)'
)
content = content.replace(
    'viz_train_spec = {k: train_spec[k] for k in viz_train_keys}',
    'viz_train_spec = {k: train_spec[k] for k in viz_train_keys}\n    viz_train_densities = {k: train_densities[k] for k in viz_train_keys}'
)
content = content.replace(
    'visualise_set(\n        model, viz_train, viz_train_spec,\n        \'Training Shapes (Spectral Cross-Attn)\',',
    'visualise_set(\n        model, viz_train, viz_train_spec, viz_train_densities,\n        \'Training Shapes (Spectral Cross-Attn)\','
)
content = content.replace(
    'visualise_set(\n        model, holdout_shapes, holdout_spec,\n        \'HELD-OUT Shapes (Spectral Cross-Attn)\',',
    'visualise_set(\n        model, holdout_shapes, holdout_spec, holdout_densities,\n        \'HELD-OUT Shapes (Spectral Cross-Attn)\','
)


with open('flow_matching_runner_ergodic.py', 'w') as f:
    f.write(content)

