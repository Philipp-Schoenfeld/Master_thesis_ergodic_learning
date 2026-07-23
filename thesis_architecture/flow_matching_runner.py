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

import argparse, math, os, sqlite3, sys
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
from flow_matching_cond_mpd_unet import CondMpdUNetFlowNetwork
from flow_matching_cond_mpd_film_unet import CondMpdFiLMUNetFlowNetwork
from flow_matching_tsvec_unet import TSVECFlowNetwork
import concurrent.futures

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


def _make_N_base(nxi):
    # N waypoints
    n_pts = np.array([
        [0.25, 0.15],
        [0.25, 0.85],
        [0.75, 0.15],
        [0.75, 0.85],
    ])
    diffs = np.diff(n_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0], np.cumsum(dists)))
    cum = cum / cum[-1]
    t = np.linspace(0, 1, nxi)
    x = np.interp(t, cum, n_pts[:, 0])
    y = np.interp(t, cum, n_pts[:, 1])
    return np.stack([x, y], axis=-1)


def _make_N_base(nxi):
    # N waypoints
    n_pts = np.array([
        [0.25, 0.15],
        [0.25, 0.85],
        [0.75, 0.15],
        [0.75, 0.85],
    ])
    diffs = np.diff(n_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0], np.cumsum(dists)))
    cum = cum / cum[-1]
    t = np.linspace(0, 1, nxi)
    x = np.interp(t, cum, n_pts[:, 0])
    y = np.interp(t, cum, n_pts[:, 1])
    return np.stack([x, y], axis=-1)


# ===========================================================================
# Dataset loaders  (unconditional: return x1, base)
# ===========================================================================

def _load_poly_N(nxi, nd, batch_size, noise_std, device, **kw):
    assert nd == 2
    base = _make_N_base(nxi)
    particles = []
    for _ in range(batch_size):
        noise = np.random.normal(0.0, noise_std, (nxi, 2))
        traj = np.clip(base + noise, 0.02, 0.98)
        particles.append(traj.ravel())
    x1 = torch.tensor(np.array(particles).reshape(batch_size, nxi, nd),
                      dtype=torch.float32).to(device)
    return x1, base


def _load_h_shape(nxi, nd, batch_size, noise_std, device, **kw):
    assert nd == 2
    pts, base = h_init(N=batch_size, T=nxi, noise_std=noise_std)
    x1 = torch.tensor(pts.reshape(batch_size, nxi, nd), dtype=torch.float32).to(device)
    return x1, base


def _make_NH_base(nxi):
    """
    Single continuous N+H trajectory: N left side → N diagonal → N right top,
    then a bridge stroke to H top-left, then H drawn top-down.
    Uses nxi control points total for the combined shape.
    """
    N_lo, N_hi, H_lo, H_hi = 0.05, 0.43, 0.57, 0.95
    # N waypoints (remapped to left half)
    n_pts = np.array([
        [N_lo + 0.25*(N_hi-N_lo), 0.15],   # N bottom-left
        [N_lo + 0.25*(N_hi-N_lo), 0.85],   # N top-left
        [N_lo + 0.75*(N_hi-N_lo), 0.15],   # N bottom-right (diagonal down)
        [N_lo + 0.75*(N_hi-N_lo), 0.85],   # N top-right
    ])
    # Bridge: N top-right → H top-left
    bridge = np.array([
        [H_lo + 0.25*(H_hi-H_lo), 0.85],   # H top-left
    ])
    # H waypoints (remapped to right half), starting from top-left going down
    h_pts = np.array([
        [H_lo + 0.25*(H_hi-H_lo), 0.15],   # H bottom-left (down left leg)
        [H_lo + 0.25*(H_hi-H_lo), 0.50],   # H crossbar left
        [H_lo + 0.75*(H_hi-H_lo), 0.50],   # H crossbar right
        [H_lo + 0.75*(H_hi-H_lo), 0.85],   # H top-right
        [H_lo + 0.75*(H_hi-H_lo), 0.15],   # H bottom-right
    ])
    all_pts = np.concatenate([n_pts, bridge, h_pts], axis=0)
    # Arc-length parameterisation → uniform T samples
    diffs = np.diff(all_pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0], np.cumsum(dists)))
    cum = cum / cum[-1]
    t = np.linspace(0, 1, nxi)
    x = np.interp(t, cum, all_pts[:, 0])
    y = np.interp(t, cum, all_pts[:, 1])
    return np.stack([x, y], axis=-1)  # (nxi, 2)


def _load_N_and_H(nxi, nd, batch_size, noise_std, device, **kw):
    assert nd == 2
    base_traj = _make_NH_base(nxi)  # (nxi, 2) single connected trajectory
    particles = []
    for _ in range(batch_size):
        noise = np.random.normal(0.0, noise_std, (nxi, 2))
        traj = np.clip(base_traj + noise, 0.02, 0.98)
        particles.append(traj.ravel())
    x1 = torch.tensor(np.array(particles).reshape(batch_size, nxi, nd),
                       dtype=torch.float32).to(device)
    return x1, base_traj


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
    q = batch_size // 4
    counts = [q, q, q, batch_size - 3 * q]

    all_x1, all_ref = [], []

    # --- N-shape ---
    bN = _make_N_base(nxi)
    if counts[0] > 0:
        n_particles = []
        for _ in range(counts[0]):
            noise = np.random.normal(0.0, noise_std, (nxi, 2))
            n_particles.append(np.clip(bN + noise, 0.02, 0.98))
        all_x1.append(np.array(n_particles))
        all_ref.append(np.tile(bN[None], (counts[0], 1, 1)))

    # --- H-shape ---
    _, bH = h_init(N=0, T=nxi, noise_std=noise_std)
    if counts[1] > 0:
        pH, _ = h_init(N=counts[1], T=nxi, noise_std=noise_std)
        x1H = pH.reshape(counts[1], nxi, nd)
        all_x1.append(x1H)
        all_ref.append(np.tile(bH[None], (counts[1], 1, 1)))

    # --- N+H combined shape ---
    bNH = _make_NH_base(nxi)                                     # (nxi, 2)
    if counts[2] > 0:
        nh_particles = []
        for _ in range(counts[2]):
            noise = np.random.normal(0.0, noise_std, (nxi, 2))
            nh_particles.append(np.clip(bNH + noise, 0.02, 0.98))
        all_x1.append(np.array(nh_particles))
        all_ref.append(np.tile(bNH[None], (counts[2], 1, 1)))

    # --- Ergodic: load exactly 1 trajectory and use it as the prototype ---
    ergo_raw = _load_ergodic_raw(1, nxi)                        # (1, nxi, 2)
    ergo_proto = ergo_raw[0]                                    # (nxi, 2) — prototype
    # Tile it to counts[3] copies
    ergo_base = np.tile(ergo_proto[None], (counts[3], 1, 1))    # (counts[3], nxi, 2)
    noisy_ergo = np.clip(ergo_base + np.random.normal(0, noise_std, ergo_base.shape), 0.02, 0.98)
    all_x1.append(noisy_ergo)
    all_ref.append(ergo_base)

    # Shuffle
    x1_all  = np.concatenate(all_x1, axis=0)
    ref_all = np.concatenate(all_ref, axis=0)
    perm    = np.random.permutation(batch_size)

    x1_tensor  = torch.tensor(x1_all[perm],  dtype=torch.float32).to(device)
    ref_tensor = torch.tensor(ref_all[perm], dtype=torch.float32).to(device)

    bases = {'polynomial_N': bN, 'h_shape': bH, 'N_and_H': bNH, 'ergodic_db': ergo_proto}
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
    'N_and_H':      'N+H Connected Trajectory',
    'ergodic_db':   'Ergodic DB Sample',
    'all_shapes':   'All Shapes — Conditional',
}

# ===========================================================================
# Model registry
# ===========================================================================

MODEL_BUILDERS = {
    'patch_unet':      lambda nxi, nd, D, **kw: PatchUNetFlowNetwork(nxi=nxi, nd=nd, D=D),
    'cond_patch_unet': lambda nxi, nd, D, **kw: CondPatchUNetFlowNetwork(nxi=nxi, nd=nd, D=D),
    'cond_mpd_unet':   lambda nxi, nd, D, **kw: CondMpdUNetFlowNetwork(nxi=nxi, nd=nd, D=D),
    'cond_mpd_film_unet': lambda nxi, nd, D, **kw: CondMpdFiLMUNetFlowNetwork(nxi=nxi, nd=nd, D=D),
    'tsvec_unet':      lambda nxi, nd, D, **kw: TSVECFlowNetwork(nxi=nxi, nd=nd, D=D),
}
MODEL_LOSS_FNS = {
    'patch_unet':      compute_cfm_loss,
    'cond_patch_unet': compute_cond_cfm_loss,
    'cond_mpd_unet':   compute_cond_cfm_loss,
    'cond_mpd_film_unet': compute_cond_cfm_loss,
    'tsvec_unet':      'tsvec_custom',  # handled by train_tsvec, not generic train
}

# ===========================================================================
# Training
# ===========================================================================

def train(model, x1, loss_fn, epochs, lr, log_every=500, ref_cps=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    use_cuda = x1.device.type == 'cuda'
    
    from tqdm import tqdm
    pbar = tqdm(range(epochs), desc="Training", unit="ep")
    
    for ep in pbar:
        opt.zero_grad()
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu', dtype=torch.bfloat16):
            loss = loss_fn(model, x1, ref_cps) if ref_cps is not None else loss_fn(model, x1)
        loss.backward()
        opt.step()
        
        if ep % 50 == 0 or ep == epochs - 1:
            pbar.set_postfix(loss=f"{loss.item():.4f}")


# ---------------------------------------------------------------------------
# TSVEC-specific loss & training (stratified time, smoothness reg, LR sched)
# ---------------------------------------------------------------------------

def compute_tsvec_loss(model, x1_batch, ref_cps_batch, smooth_weight=0.1):
    """
    TSVEC CFM loss with:
    - Stratified uniform time sampling (divides [0,1] into B bins for even coverage)
    - B-Spline smoothness regularization via finite differences on predicted velocity
    """
    B, nxi, nd = x1_batch.shape
    device = x1_batch.device

    # --- Stratified time sampling ---
    # Divide [0,1] into B equal bins, sample one t per bin → guaranteed uniform coverage
    bin_edges = torch.linspace(0, 1, B + 1, device=device)
    t = bin_edges[:B] + torch.rand(B, device=device) / B  # (B,) in [i/B, (i+1)/B)
    t = t.clamp(1e-5, 1 - 1e-5)  # avoid exact 0 and 1

    x0    = torch.randn_like(x1_batch)
    t_exp = t.view(B, 1, 1)
    xt    = (1 - t_exp) * x0 + t_exp * x1_batch
    ut    = x1_batch - x0

    vt = model(xt, t, ref_cps_batch)

    # --- Flow matching loss ---
    flow_loss = torch.mean((vt - ut) ** 2)

    # --- Smoothness regularization ---
    # Penalize second-order finite differences of predicted velocity field
    # This encourages Ck-smooth B-Spline control point velocities
    if smooth_weight > 0 and nxi >= 3:
        # First differences: Δv[i] = v[i+1] - v[i]
        dv = vt[:, 1:, :] - vt[:, :-1, :]
        # Second differences: Δ²v[i] = Δv[i+1] - Δv[i] (curvature proxy)
        ddv = dv[:, 1:, :] - dv[:, :-1, :]
        smooth_loss = torch.mean(ddv ** 2)
    else:
        smooth_loss = 0.0

    return flow_loss + smooth_weight * smooth_loss


def train_tsvec(model, x1, ref_cps, epochs, lr, log_every=500, smooth_weight=0.1):
    """
    TSVEC-specific training loop with:
    - AdamW optimizer (decoupled weight decay for attention layers)
    - Warmup (5% of epochs) + Cosine annealing LR schedule
    - Gradient clipping (max_norm=1.0)
    - bfloat16 autocast
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # LR Schedule: linear warmup then cosine decay
    warmup_epochs = max(1, epochs // 20)  # 5% warmup
    
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return ep / warmup_epochs  # linear warmup from 0 to lr
        # Cosine decay from lr to 0
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    
    model.train()
    use_cuda = x1.device.type == 'cuda'
    
    from tqdm import tqdm
    pbar = tqdm(range(epochs), desc="TSVEC Training", unit="ep")
    
    for ep in pbar:
        opt.zero_grad()
        with torch.autocast(device_type='cuda' if use_cuda else 'cpu', dtype=torch.bfloat16):
            loss = compute_tsvec_loss(model, x1, ref_cps, smooth_weight=smooth_weight)
        loss.backward()
        
        # Gradient clipping to prevent explosion from attention layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()
        scheduler.step()
        
        if ep % 50 == 0 or ep == epochs - 1:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")


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
    is_cond = 'cond' in model_name or 'tsvec' in model_name

    is_tsvec = 'tsvec' in model_name

    rows = [
        (11.3,'Input x',     f'(B,{nxi},{nd}) noisy CPs','white'),
        (10.0,'① Patchify',  f'→(B,{D},{nxi})',          '#FFF8E1'),
        (8.70,'② Time Emb', f't→R^{D}',                  '#F3E5F5'),
    ]
    if is_cond:
        enc_label = '③ Shape Enc (Unpooled)' if is_tsvec else '③ Shape Enc'
        enc_sub   = f'ref_cps→R^(H×{D})' if is_tsvec else f'ref_cps→R^{D} (+ Time)'
        rows.append((8.70, enc_label, enc_sub, '#E8EAF6'))

    rows += [
        (7.40,'Enc1 s=1',    f'(B,{D},{nxi})',            '#E3F2FD'),
        (6.30,'Enc2 s=2',    f'(B,{2*D},{nxi}//2)',       '#BBDEFB'),
        (5.20,'Enc3 s=2',    f'(B,{4*D},{nxi}//4)',       '#90CAF9'),
    ]

    if is_tsvec:
        # Expanded bottleneck showing the Conference Room internals
        rows += [
            (4.60,'BN Conv1 (FiLM)',   f'(B,{4*D},{nxi}//4)',      '#E8EAF6'),
            (3.90,'Self-Attn + PE',    f'Zero-Init | 4 heads',     '#FFE0B2'),
            (3.20,'Cross-Attn + PE',   f'Q:tokens, KV:ref_cps',    '#FFCCBC'),
            (2.50,'BN Conv2 (FiLM)',   f'(B,{4*D},{nxi}//4)',      '#E8EAF6'),
        ]
    else:
        rows.append((4.10,'Bottleneck',  f'(B,{4*D},{nxi}//4) ×2', '#E8EAF6'))

    rows += [
        (1.70 if is_tsvec else 3.00,'Dec1+skip',   f'→(B,{2*D},{nxi}//2)',      '#E8F5E9'),
        (0.90 if is_tsvec else 1.90,'Dec2+skip',   f'→(B,{D},{nxi})',           '#C8E6C9'),
        (0.10 if is_tsvec else 0.70,'④ MLP Head', f'→vθ(B,{nxi},{nd})',         '#FFF3E0'),
    ]

    prev = None
    for (y, lbl, sub, col) in rows:
        box(cx, y, bw, bh, lbl, sub, col)
        if prev is not None and abs(prev - y) < 1.8:
            arr(cx, prev-bh/2, cx, y+bh/2)
        prev = y

    if is_cond:
        # Side box for ref_cps input to ShapeEncoder
        ref_sub = f'(B,{nxi},{nd}) → KV seq' if is_tsvec else f'(B,{nxi},{nd})'
        box(8.2, 8.70, 2.0, bh, 'ref_cps', ref_sub, '#F3E5F5')
        arr(8.2, 8.70-bh/2+0.1, cx-bw/2, 8.70)
        
        if is_tsvec:
            # Arrow from ref_cps encoder to the Cross-Attention block
            arr(8.2, 8.70-bh/2, 8.2, 3.20+bh/2)
            arr(8.2, 3.20, cx+bw/2, 3.20)

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

    if args.load_model and os.path.isfile(args.load_model):
        print(f"  Loading pretrained model from {args.load_model}")
        ckpt = torch.load(args.load_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded successfully.")
    else:
        train(model, x1, MODEL_LOSS_FNS[args.model], args.epochs, args.lr, args.log_every)
        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model) if os.path.dirname(args.save_model) else '.', exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'nxi': args.nxi, 'nd': args.nd, 'D': args.D}, args.save_model)
            print(f"  Model saved to {args.save_model}")

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
    Train ONLY cond_mpd_unet conditional model on all four shapes simultaneously.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    print(f"\n{'='*65}")
    print(f"  Conditional training — cond_mpd_unet")
    print(f"{'='*65}")

    # Load mixed dataset
    x1, bases_dict, ref_cps = _load_all_shapes(
        nxi=args.nxi, nd=args.nd, batch_size=args.batch_size,
        noise_std=args.noise_std, device=device)

    model_name = 'cond_mpd_unet'
    model = MODEL_BUILDERS[model_name](nxi=args.nxi, nd=args.nd, D=args.D).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {model_name}      : {params:,} params\n")

    if args.load_model and os.path.isfile(args.load_model):
        print(f"  Loading pretrained model from {args.load_model}")
        ckpt = torch.load(args.load_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded successfully (trained for {ckpt.get('epochs', '?')} epochs)")
    else:
        print(f"[{model_name}] Starting training...")
        train(model, x1.clone(), MODEL_LOSS_FNS[model_name], args.epochs, args.lr,
              args.log_every, ref_cps=ref_cps.clone())
        print(f"[{model_name}] Finished training!")
        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model) if os.path.dirname(args.save_model) else '.', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'nxi': args.nxi,
                'nd': args.nd,
                'D': args.D,
                'epochs': args.epochs,
                'lr': args.lr
            }, args.save_model)
            print(f"  Model saved to {args.save_model}")

    # ------------------------------------------------------------------
    # Generate: condition on each shape's reference CPs
    # ------------------------------------------------------------------
    shape_names = list(bases_dict.keys())
    bN = bases_dict['polynomial_N']
    bH = bases_dict['h_shape']

    viz_shapes = {
        'polynomial_N': bN,
        'h_shape':      bH,
        'N_and_H':      bases_dict['N_and_H'],
        'ergodic_db':   bases_dict['ergodic_db'],
    }

    print("\nGenerating conditioned trajectories...")
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    fig.suptitle(
        f'{model_name} Ergodic Trajectory Generation\n'
        f'Trained on all shapes, Batch Size={args.batch_size}, nξ={args.nxi}, D={args.D}',
        fontsize=14, fontweight='bold', color='#1A1A2E')

    # 2 rows: Row 0 for generations, Row 1 for Arch Diagram
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.35)

    def draw_model_row(row_idx, current_model, title_prefix):
        for col, (ds_name, base_ref) in enumerate(viz_shapes.items()):
            ref_t = torch.tensor(base_ref, dtype=torch.float32)
            gen_t = generate_cond_trajectories(current_model, ref_t, args.n_gen, args.nxi, args.nd, args.steps, str(device))
            gen_cps = gen_t.cpu().numpy()

            ax_traj = fig.add_subplot(gs[row_idx, col])
            _draw_traj(ax_traj, base_ref, np.zeros((0, args.nxi, 2)), gen_cps,
                       f"{title_prefix}: {DATASET_TITLES.get(ds_name, ds_name)}",
                       args.bspline_pts, args.bspline_deg)

    # Draw model on Row 0
    draw_model_row(0, model, model_name)

    # Architecture diagram on Row 1 (spanning all 4 columns for aesthetics)
    ax_arch = fig.add_subplot(gs[1, :])
    _draw_arch(ax_arch, model_name, args.nxi, args.nd, args.D, params)
    
    plt.tight_layout(); plt.show()


def run_overfit(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"  Extreme Overfitting — Ergodic Single Sample")
    print(f"{'='*65}")

    ergo_raw = _load_ergodic_raw(1, args.nxi)
    ergo_proto = ergo_raw[0]
    
    ergo_base = np.tile(ergo_proto[None], (args.batch_size, 1, 1))
    noisy_ergo = np.clip(ergo_base + np.random.normal(0, args.noise_std, ergo_base.shape), 0.02, 0.98)
    x1 = torch.tensor(noisy_ergo, dtype=torch.float32).to(device)
    ref_cps = torch.tensor(ergo_base, dtype=torch.float32).to(device)

    models_to_train = ['cond_mpd_unet', 'cond_mpd_film_unet']
    trained_models = {}

    for m_name in models_to_train:
        print(f"\nTraining {m_name}...")
        model = MODEL_BUILDERS[m_name](nxi=args.nxi, nd=args.nd, D=args.D).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {params:,}")
        
        if args.load_model and os.path.isfile(f"{args.load_model}_{m_name}.pt"):
            print(f"  Loading pretrained {m_name} from {args.load_model}_{m_name}.pt")
            ckpt = torch.load(f"{args.load_model}_{m_name}.pt", map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            train(model, x1, MODEL_LOSS_FNS[m_name], args.epochs, args.lr, args.log_every, ref_cps=ref_cps)
            if args.save_model:
                os.makedirs(os.path.dirname(args.save_model) if os.path.dirname(args.save_model) else '.', exist_ok=True)
                torch.save({'model_state_dict': model.state_dict()}, f"{args.save_model}_{m_name}.pt")
                print(f"  Saved to {args.save_model}_{m_name}.pt")
        trained_models[m_name] = (model, params)

    print("\nGenerating conditioned trajectories...")
    fig = plt.figure(figsize=(14, 6 * len(models_to_train)), facecolor='white')
    fig.suptitle(f'Extreme Overfitting — Ergodic DB', fontsize=14, fontweight='bold', color='#1A1A2E')
    gs = gridspec.GridSpec(len(models_to_train), 2, figure=fig, hspace=0.40, wspace=0.30, width_ratios=[1.1, 0.9])

    for row, m_name in enumerate(models_to_train):
        model, params = trained_models[m_name]
        
        ref_t = torch.tensor(ergo_proto[None], dtype=torch.float32).to(device)
        gen_t = generate_cond_trajectories(model, ref_t, args.n_gen, args.nxi, args.nd, args.steps, str(device))
        gen_cps = gen_t.cpu().numpy()

        ax_traj = fig.add_subplot(gs[row, 0])
        _draw_traj(ax_traj, ergo_proto, x1.cpu().numpy(), gen_cps, f"Overfit: {m_name}", args.bspline_pts, args.bspline_deg)

        ax_arch = fig.add_subplot(gs[row, 1])
        _draw_arch(ax_arch, m_name, args.nxi, args.nd, args.D, params)

    plt.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model',       default='patch_unet',  choices=list(MODEL_BUILDERS))
    p.add_argument('--dataset',     default='polynomial_N',choices=list(TRAJECTORY_DATASETS))
    p.add_argument('--run_all',     action='store_true',   help='Train all 4 shapes separately')
    p.add_argument('--overfit',     action='store_true',   help='Overfit cond_mpd models on single ergodic sample')
    p.add_argument('--nxi',         type=int,   default=20)
    p.add_argument('--nd',          type=int,   default=2)
    p.add_argument('--D',           type=int,   default=384)
    p.add_argument('--epochs',      type=int,   default=10000)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--n_gen',       type=int,   default=1)
    p.add_argument('--bspline_pts', type=int,   default=512)
    p.add_argument('--bspline_deg', type=int,   default=5)
    p.add_argument('--noise_std',   type=float, default=0.0)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--steps',       type=int,   default=100)
    p.add_argument('--log_every',   type=int,   default=500)
    p.add_argument('--save_model',  type=str,   default=None, help='Path to save trained model (.pt)')
    p.add_argument('--load_model',  type=str,   default=None, help='Path to load pretrained model (.pt), skips training')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.overfit:
        run_overfit(args)
    elif args.dataset == 'all_shapes' or args.model == 'cond_patch_unet':
        args.model = 'cond_patch_unet'
        run_conditional(args)
    elif args.run_all:
        run_all(args)
    else:
        run_single(args)
