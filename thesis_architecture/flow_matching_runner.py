#!/usr/bin/env python3
"""
flow_matching_runner.py — Modular flow-matching trainer + B-spline visualiser.

Datasets : polynomial_N | h_shape | N_and_H | ergodic_db | all_shapes
Models   : patch_unet | cond_patch_unet

# Single-shape (unconditional):
python flow_matching_runner.py --dataset polynomial_N
python flow_matching_runner.py --run_all

# All-shapes conditional (single training, shape-conditioned generation):
python flow_matching_runner.py --dataset all_shapes --model cond_patch_unet
"""

import argparse, os, sqlite3, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, '..')
for _p in (os.path.join(_root, 'bsplinax-main'), os.path.join(_root, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bsplinax.bspline import BsplineBasisClamped
from init_strategies.polynomial import init_particles as poly_init
from init_strategies.h_shape    import init_particles as h_init
from flow_matching_patch_unet import (
    PatchUNetFlowNetwork, compute_cfm_loss, generate_trajectories as euler_generate,
)
from flow_matching_cond_patch_unet import (
    CondPatchUNetFlowNetwork, compute_cond_cfm_loss, generate_cond_trajectories,
)

_DB_PATH = os.path.join(_here, 'Trajectory_data_generator', 'stein_coverage_results.db')

# ===========================================================================
# Helpers
# ===========================================================================

def cp_to_bspline(cps, pts=512, deg=5):
    nxi = cps.shape[0]
    B = np.array(BsplineBasisClamped(degree=deg, num_control_points=nxi,
                                     num_phase_points=pts,
                                     compute_derivatives=False).B)
    return B @ cps


def _remap_x(traj, lo, hi):
    out = traj.copy(); out[:, 0] = lo + traj[:, 0] * (hi - lo); return out


def _load_ergodic_raw(n, nxi):
    conn = sqlite3.connect(_DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT trajectory, shape FROM runs ORDER BY RANDOM() LIMIT ?", (n,))
    rows = cur.fetchall(); conn.close()
    trajs = []
    for blob, sh in rows:
        shape = tuple(map(int, sh.split(',')))
        xy = np.frombuffer(blob, dtype=np.float32).reshape(shape)[:, :2]
        idx = np.linspace(0, len(xy)-1, nxi).astype(int)
        trajs.append(xy[idx])
    return np.array(trajs)   # (n, nxi, 2)


# ===========================================================================
# Dataset loaders  (unconditional: return x1, base)
# ===========================================================================

def _load_poly_N(nxi, nd, batch_size, noise_std, device, **kw):
    assert nd == 2
    pts, base = poly_init(N=batch_size, T=nxi, degree=5, noise_std=noise_std)
    x1 = torch.tensor(pts.reshape(batch_size, nxi, nd), dtype=torch.float32).to(device)
    return x1, base


def _load_h_shape(nxi, nd, batch_size, noise_std, device, **kw):
    assert nd == 2
    pts, base = h_init(N=batch_size, T=nxi, noise_std=noise_std)
    x1 = torch.tensor(pts.reshape(batch_size, nxi, nd), dtype=torch.float32).to(device)
    return x1, base


def _load_N_and_H(nxi, nd, batch_size, noise_std, device, **kw):
    assert nd == 2
    half = batch_size // 2
    pN_raw, bN_raw = poly_init(N=half,           T=nxi, degree=5, noise_std=noise_std)
    pH_raw, bH_raw = h_init  (N=batch_size-half, T=nxi, noise_std=noise_std)
    N_lo, N_hi, H_lo, H_hi = 0.05, 0.43, 0.57, 0.95
    pN = np.array([_remap_x(r.reshape(nxi, nd), N_lo, N_hi).ravel() for r in pN_raw])
    pH = np.array([_remap_x(r.reshape(nxi, nd), H_lo, H_hi).ravel() for r in pH_raw])
    pts  = np.concatenate([pN, pH], axis=0)
    x1   = torch.tensor(pts.reshape(batch_size, nxi, nd), dtype=torch.float32).to(device)
    bN   = _remap_x(bN_raw, N_lo, N_hi)
    bH   = _remap_x(bH_raw, H_lo, H_hi)
    base = np.concatenate([bN, np.full((3, 2), np.nan), bH], axis=0)
    return x1, base


def _load_ergodic_db(nxi, nd, batch_size, noise_std, device, **kw):
    assert nd == 2
    trajs = _load_ergodic_raw(batch_size, nxi)
    base  = trajs[0].copy()
    noisy = np.clip(trajs + np.random.normal(0, noise_std, trajs.shape), 0.02, 0.98)
    return torch.tensor(noisy, dtype=torch.float32).to(device), base


# ---------------------------------------------------------------------------
# Conditional dataset: all shapes mixed, returns (x1, base_dict, ref_cps)
# ---------------------------------------------------------------------------

def _load_all_shapes(nxi, nd, batch_size, noise_std, device, **kw):
    """
    Mixed batch from N, H and ergodic shapes.
    Returns (x1_batch, bases_dict, ref_cps_batch) where:
      - x1_batch       : (B, nxi, nd)  noisy training trajectories
      - bases_dict     : {name: ndarray(nxi,2)}  one reference per shape for viz
      - ref_cps_batch  : (B, nxi, nd)  per-sample reference (conditioning signal)
    """
    assert nd == 2
    third = batch_size // 3
    counts = [third, third, batch_size - 2*third]   # N, H, ergodic

    all_x1, all_ref = [], []

    # --- N-shape ---
    pN, bN = poly_init(N=counts[0], T=nxi, degree=5, noise_std=noise_std)
    x1N = pN.reshape(counts[0], nxi, nd)
    all_x1.append(x1N)
    all_ref.append(np.tile(bN[None], (counts[0], 1, 1)))

    # --- H-shape ---
    pH, bH = h_init(N=counts[1], T=nxi, noise_std=noise_std)
    x1H = pH.reshape(counts[1], nxi, nd)
    all_x1.append(x1H)
    all_ref.append(np.tile(bH[None], (counts[1], 1, 1)))

    # --- Ergodic: load batch, use mean as prototype reference ---
    ergo = _load_ergodic_raw(counts[2], nxi)                    # (n, nxi, 2)
    ergo_proto = ergo.mean(axis=0)                              # (nxi, 2) — prototype
    noisy_ergo = np.clip(ergo + np.random.normal(0, noise_std, ergo.shape), 0.02, 0.98)
    all_x1.append(noisy_ergo)
    all_ref.append(np.tile(ergo_proto[None], (counts[2], 1, 1)))

    # Shuffle
    x1_all  = np.concatenate(all_x1, axis=0)
    ref_all = np.concatenate(all_ref, axis=0)
    perm    = np.random.permutation(batch_size)

    x1_tensor  = torch.tensor(x1_all[perm],  dtype=torch.float32).to(device)
    ref_tensor = torch.tensor(ref_all[perm], dtype=torch.float32).to(device)

    bases = {'polynomial_N': bN, 'h_shape': bH, 'ergodic_db': ergo_proto}
    return x1_tensor, bases, ref_tensor


TRAJECTORY_DATASETS = {
    'polynomial_N': _load_poly_N,
    'h_shape':      _load_h_shape,
    'N_and_H':      _load_N_and_H,
    'ergodic_db':   _load_ergodic_db,
    'all_shapes':   _load_all_shapes,   # conditional
}

DATASET_TITLES = {
    'polynomial_N': 'Polynomial N-Shape',
    'h_shape':      'H-Shape (Linear)',
    'N_and_H':      'N + H  (side-by-side)',
    'ergodic_db':   'Ergodic DB Sample',
    'all_shapes':   'All Shapes — Conditional',
}

# ===========================================================================
# Model registry
# ===========================================================================

MODEL_BUILDERS = {
    'patch_unet':      lambda nxi, nd, D, **kw: PatchUNetFlowNetwork(nxi=nxi, nd=nd, D=D),
    'cond_patch_unet': lambda nxi, nd, D, **kw: CondPatchUNetFlowNetwork(nxi=nxi, nd=nd, D=D),
}
MODEL_LOSS_FNS = {
    'patch_unet':      compute_cfm_loss,
    'cond_patch_unet': compute_cond_cfm_loss,
}

# ===========================================================================
# Training
# ===========================================================================

def train(model, x1, loss_fn, epochs, lr, log_every=500, ref_cps=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model, x1, ref_cps) if ref_cps is not None else loss_fn(model, x1)
        loss.backward(); opt.step()
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  ep {ep:5d} | loss {loss.item():.6f}")


# ===========================================================================
# Visualization helpers
# ===========================================================================

def _draw_traj(ax, base, train_samples, gen_cps, title,
               bspline_pts=512, bspline_deg=5):
    ax.set_facecolor('white')
    for i in range(min(6, len(train_samples))):
        cp = train_samples[i]
        ax.plot(cp[:,0], cp[:,1], color='#90CAF9', lw=0.7, alpha=0.3)

    # Target (blue) — handle NaN segments
    if np.any(np.isnan(base)):
        segs = np.split(base, np.where(np.any(np.isnan(base), axis=1))[0])
        for k, seg in enumerate(segs):
            seg = seg[~np.any(np.isnan(seg), axis=1)]
            if len(seg) >= 6:
                ax.plot(*cp_to_bspline(seg, bspline_pts, bspline_deg).T,
                        color='#1565C0', lw=2.2, label='Target' if k==0 else '')
    else:
        if len(base) >= 6:
            ax.plot(*cp_to_bspline(base, bspline_pts, bspline_deg).T,
                    color='#1565C0', lw=2.2, label='Target')

    # Generated (red)
    for i, cp in enumerate(gen_cps):
        if len(cp) >= 6:
            ax.plot(*cp_to_bspline(cp, bspline_pts, bspline_deg).T,
                    color='#EF5350', lw=1.8, alpha=0.85,
                    label='Generated' if i==0 else '')
        ax.scatter(cp[:,0], cp[:,1], color='#EF5350', s=10, alpha=0.4, zorder=3)

    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(True, alpha=0.10, lw=0.5)
    ax.set_title(title, fontsize=10, color='#1A1A2E', pad=6)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')


def _draw_arch(ax, model_name, nxi, nd, D, params):
    ax.set_facecolor('white'); ax.set_xlim(0,10); ax.set_ylim(-0.5,12); ax.axis('off')

    def box(x,y,w,h,lbl,sub='',col='#EEE'):
        ax.add_patch(plt.Rectangle((x-w/2,y-h/2),w,h,
                     facecolor=col, edgecolor='#555', lw=1.0, zorder=2))
        ax.text(x, y+(0.1 if sub else 0), lbl, ha='center', va='center',
                fontsize=7, fontweight='bold', color='#222', zorder=3)
        if sub:
            ax.text(x, y-0.18, sub, ha='center', va='center',
                    fontsize=5.5, color='#555', zorder=3)

    def arr(x0,y0,x1,y1):
        ax.annotate('', xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=0.9))

    cx, bw, bh = 5.0, 2.9, 0.65
    is_cond = model_name == 'cond_patch_unet'

    rows = [
        (11.3,'Input x',     f'(B,{nxi},{nd}) noisy CPs','white'),
        (10.0,'① Patchify',  f'→(B,{D},{nxi})',          '#FFF8E1'),
        (8.70,'② Time Emb', f't→R^{D}',                  '#F3E5F5'),
    ]
    if is_cond:
        rows.append((8.70,'③ Shape Enc',f'ref_cps→R^{D} (+ Time)','#E8EAF6'))

    rows += [
        (7.40,'Enc1 s=1',    f'(B,{D},{nxi})',            '#E3F2FD'),
        (6.30,'Enc2 s=2',    f'(B,{2*D},{nxi}//2)',       '#BBDEFB'),
        (5.20,'Enc3 s=2',    f'(B,{4*D},{nxi}//4)',       '#90CAF9'),
        (4.10,'Bottleneck',  f'(B,{4*D},{nxi}//4) ×2',   '#E8EAF6'),
        (3.00,'Dec1+skip',   f'→(B,{2*D},{nxi}//2)',      '#E8F5E9'),
        (1.90,'Dec2+skip',   f'→(B,{D},{nxi})',           '#C8E6C9'),
        (0.70,'④ MLP Head', f'→vθ(B,{nxi},{nd})',         '#FFF3E0'),
    ]

    prev = None
    for (y, lbl, sub, col) in rows:
        box(cx, y, bw, bh, lbl, sub, col)
        if prev is not None and abs(prev - y) < 1.8:
            arr(cx, prev-bh/2, cx, y+bh/2)
        prev = y

    if is_cond:
        # Side box for ref_cps input to ShapeEncoder
        box(8.2, 8.70, 2.0, bh, 'ref_cps', f'(B,{nxi},{nd})', '#F3E5F5')
        arr(8.2, 8.70-bh/2+0.1, cx-bw/2, 8.70)

    ax.set_title(f'{model_name}\n{params:,} params', fontsize=8, color='#333', pad=5)


# ===========================================================================
# Run functions
# ===========================================================================

def run_single(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = TRAJECTORY_DATASETS[args.dataset]
    result = loader(nxi=args.nxi, nd=args.nd, batch_size=args.batch_size,
                    noise_std=args.noise_std, device=device)
    x1, base = result[0], result[1]
    train_np = x1.cpu().numpy()

    model  = MODEL_BUILDERS[args.model](nxi=args.nxi, nd=args.nd, D=args.D).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {args.model}: {params:,} params")

    train(model, x1, MODEL_LOSS_FNS[args.model], args.epochs, args.lr, args.log_every)

    gen_t, _ = euler_generate(model=model, num_samples=args.n_gen,
                              nxi=args.nxi, nd=args.nd, steps=args.steps,
                              device=str(device))
    gen_np = gen_t.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14,6), facecolor='white',
                             gridspec_kw={'width_ratios':[1.1,0.9]})
    _draw_traj(axes[0], base, train_np, gen_np,
               DATASET_TITLES[args.dataset], args.bspline_pts, args.bspline_deg)
    _draw_arch(axes[1], args.model, args.nxi, args.nd, args.D, params)
    plt.suptitle(f'Flow Matching — {args.model} | {DATASET_TITLES[args.dataset]}',
                 fontsize=12, fontweight='bold', color='#1A1A2E', y=1.01)
    plt.tight_layout(); plt.show()


def run_all(args):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = ['polynomial_N','h_shape','N_and_H','ergodic_db']
    fig = plt.figure(figsize=(16, 28), facecolor='white')
    fig.suptitle(f'Flow Matching — {args.model} — All Datasets',
                 fontsize=14, fontweight='bold', color='#1A1A2E')
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.40, wspace=0.30,
                            width_ratios=[1.1,0.9])

    for row, ds in enumerate(datasets):
        print(f"\n[{row+1}/4] {DATASET_TITLES[ds]}")
        loader = TRAJECTORY_DATASETS[ds]
        x1, base = loader(nxi=args.nxi, nd=args.nd, batch_size=args.batch_size,
                          noise_std=args.noise_std, device=device)
        train_np = x1.cpu().numpy()
        model  = MODEL_BUILDERS[args.model](nxi=args.nxi, nd=args.nd, D=args.D).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train(model, x1, MODEL_LOSS_FNS[args.model], args.epochs, args.lr, args.log_every)
        gen_t, _ = euler_generate(model=model, num_samples=args.n_gen,
                                  nxi=args.nxi, nd=args.nd, steps=args.steps, device=str(device))
        ax_t = fig.add_subplot(gs[row, 0])
        ax_a = fig.add_subplot(gs[row, 1])
        _draw_traj(ax_t, base, train_np, gen_t.cpu().numpy(),
                   DATASET_TITLES[ds], args.bspline_pts, args.bspline_deg)
        _draw_arch(ax_a, args.model, args.nxi, args.nd, args.D, params)
    plt.show()


def run_conditional(args):
    """
    Train ONE conditional model on all shapes simultaneously.
    At inference, condition on each shape's reference CPs to generate
    shape-specific trajectories — no retraining needed.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f"  Conditional training — all shapes  |  {args.model}")
    print(f"{'='*55}")

    # Load mixed dataset
    x1, bases_dict, ref_cps = _load_all_shapes(
        nxi=args.nxi, nd=args.nd, batch_size=args.batch_size,
        noise_std=args.noise_std, device=device)

    model  = MODEL_BUILDERS[args.model](nxi=args.nxi, nd=args.nd, D=args.D).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {args.model}: {params:,} params\n")

    # Train once on all shapes
    train(model, x1, compute_cond_cfm_loss, args.epochs, args.lr,
          args.log_every, ref_cps=ref_cps)

    # ------------------------------------------------------------------
    # Generate: condition on each shape's reference CPs
    # ------------------------------------------------------------------
    shape_names = list(bases_dict.keys())   # N, H, ergodic
    # Also add "N+H side by side" as a 4th inference panel
    bN = bases_dict['polynomial_N']
    bH = bases_dict['h_shape']

    viz_shapes = {
        'polynomial_N': bN,
        'h_shape':      bH,
        'ergodic_db':   bases_dict['ergodic_db'],
    }
    # N+H: generate N (left) and H (right) separately, combine for display
    bN_nh = _remap_x(bN, 0.05, 0.43)
    bH_nh = _remap_x(bH, 0.57, 0.95)
    viz_shapes['N_and_H'] = np.concatenate(
        [bN_nh, np.full((3,2), np.nan), bH_nh], axis=0)

    print("\nGenerating conditioned trajectories...")
    fig = plt.figure(figsize=(18, 20), facecolor='white')
    fig.suptitle(
        f'Conditional Flow Matching — {args.model}\n'
        f'Trained once on all shapes, conditioned at inference\n'
        f'{params:,} params  |  nξ={args.nxi}  D={args.D}',
        fontsize=12, fontweight='bold', color='#1A1A2E')

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.35)

    for col, (ds_name, base_ref) in enumerate(viz_shapes.items()):
        print(f"  → {DATASET_TITLES.get(ds_name, ds_name)}")

        if ds_name == 'N_and_H':
            # Generate N with N condition + H with H condition
            ref_N = torch.tensor(bN_nh, dtype=torch.float32)
            ref_H = torch.tensor(bH_nh, dtype=torch.float32)
            genN  = generate_cond_trajectories(model, ref_N, 1, args.nxi, args.nd,
                                               args.steps, str(device)).cpu().numpy()
            genH  = generate_cond_trajectories(model, ref_H, 1, args.nxi, args.nd,
                                               args.steps, str(device)).cpu().numpy()
            gen_cps = np.concatenate([genN, genH], axis=0)
        else:
            ref_t = torch.tensor(base_ref, dtype=torch.float32)
            gen_t = generate_cond_trajectories(model, ref_t, args.n_gen,
                                               args.nxi, args.nd, args.steps, str(device))
            gen_cps = gen_t.cpu().numpy()

        ax_traj = fig.add_subplot(gs[0, col])
        _draw_traj(ax_traj, base_ref, np.zeros((0, args.nxi, 2)), gen_cps,
                   DATASET_TITLES.get(ds_name, ds_name),
                   args.bspline_pts, args.bspline_deg)

    # Architecture diagram (spans bottom row)
    ax_arch = fig.add_subplot(gs[1, :])
    _draw_arch(ax_arch, args.model, args.nxi, args.nd, args.D, params)
    plt.tight_layout(); plt.show()


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',       default='patch_unet',  choices=list(MODEL_BUILDERS))
    p.add_argument('--dataset',     default='polynomial_N',choices=list(TRAJECTORY_DATASETS))
    p.add_argument('--run_all',     action='store_true',   help='Train all 4 shapes separately')
    p.add_argument('--nxi',         type=int,   default=20)
    p.add_argument('--nd',          type=int,   default=2)
    p.add_argument('--D',           type=int,   default=128)
    p.add_argument('--epochs',      type=int,   default=3000)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--n_gen',       type=int,   default=1)
    p.add_argument('--bspline_pts', type=int,   default=512)
    p.add_argument('--bspline_deg', type=int,   default=5)
    p.add_argument('--noise_std',   type=float, default=0.02)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--steps',       type=int,   default=100)
    p.add_argument('--log_every',   type=int,   default=500)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'all_shapes' or args.model == 'cond_patch_unet':
        args.model = 'cond_patch_unet'
        run_conditional(args)
    elif args.run_all:
        run_all(args)
    else:
        run_single(args)
