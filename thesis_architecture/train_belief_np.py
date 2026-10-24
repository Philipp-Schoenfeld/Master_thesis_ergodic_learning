#!/usr/bin/env python3
"""
train_belief_np.py
==================
Meta-trains the attentive neural process of neural_process_belief.py: from K
noisy point measurements of a target density, predict the density anywhere.

Tasks come from the train split of ergodic_dataset_775.db with the same
measurement model as the AK belief (uniform random locations, K in
[k_min, k_max], Gaussian noise). Half of the query points are uniform, half
follow the density, so the network does not learn to predict zero everywhere.
Validation uses the val split with fixed measurements and reports the same
metrics for the AK-GP on identical measurements as a reference.
"""

import argparse, json, os, signal, sqlite3, sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_here, 'ergodic_dataset_generator'))
from shape_library import pdf_on_grid
from checkpoint_rotation import nach_zwischenstand, nach_endstand
from attentive_kernel_belief import measure_density, fit_belief_grids
from neural_process_belief import AttentiveNeuralProcess, gaussian_nll

_DB_PATH = os.path.join(_here, 'ergodic_dataset_generator', 'ergodic_dataset_775.db')
_VAL_KS = (32, 128)


def _load_grids(split, grid_res=128):
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute("SELECT density_params FROM ergodic_pairs WHERE split = ? ORDER BY id ASC",
                        (split,)).fetchall()
    conn.close()
    grids = []
    for (params,) in rows:
        d, _, _ = pdf_on_grid(json.loads(params), resolution=grid_res)
        grids.append(d / d.max() if d.max() > 0 else d)
    return torch.tensor(np.stack(grids), dtype=torch.float32)


def _augment(grids):
    """Random element of the dihedral group D4, exact on the grid."""
    grids = torch.rot90(grids, int(torch.randint(4, ())), dims=(1, 2))
    flip = torch.rand(grids.shape[0], device=grids.device) < 0.5
    return torch.where(flip[:, None, None], grids.flip(-1), grids)


def _density_queries(grids, n):
    B, H, W = grids.shape
    flat = grids.reshape(B, -1)
    idx = torch.multinomial(flat + 1e-7, n, replacement=True)
    xy = torch.stack([(idx % W).float() / (W - 1), (idx // W).float() / (H - 1)], dim=-1)
    return xy, torch.gather(flat, 1, idx)


def _belief_metrics(belief, true):
    """Mean relative L2 error and mean Pearson correlation over shapes."""
    err = ((belief - true).pow(2).mean((1, 2)) / true.pow(2).mean((1, 2))).sqrt()
    b = belief.flatten(1) - belief.flatten(1).mean(1, keepdim=True)
    t = true.flatten(1) - true.flatten(1).mean(1, keepdim=True)
    corr = (b * t).sum(1) / (b.norm(dim=1) * t.norm(dim=1)).clamp(min=1e-12)
    return err.mean().item(), corr.mean().item()


def _save(model, opt, sched, step, args, path, val=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': sched.state_dict(),
        'step': step, 'config': model.config, 'val': val,
        'k_min': args.k_min, 'k_max': args.k_max, 'noise': args.noise,
        'wandb_id': wandb.run.id if args.use_wandb and _WANDB_OK and wandb.run else None,
    }, path)


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")
    run_str = f"{timestamp}_D{args.D}_K{args.k_min}-{args.k_max}_n{args.noise:g}"
    stem = os.path.splitext(args.save_model)[0]
    os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)

    print(f"  Loading density grids... (device={device})")
    train_grids = _load_grids('train').to(device)
    val_grids = _load_grids('val').to(device)
    val_true = F.interpolate(val_grids[:, None], size=args.val_grid, mode='bilinear',
                             align_corners=True)[:, 0]
    val_meas = {}
    gen = torch.Generator(device=device).manual_seed(1234)
    for k in _VAL_KS:
        val_meas[k] = measure_density(val_grids, k, args.noise, gen)
    print(f"  {train_grids.shape[0]} train / {val_grids.shape[0]} val shapes")

    ak_ref = {}
    if args.ak_reference:
        for k in _VAL_KS:
            ak = fit_belief_grids(val_meas[k], grid_res=args.val_grid, n_iters=300, generator=gen)
            ak_ref[k] = _belief_metrics(ak, val_true)
            print(f"  AK reference K={k}: rel. L2 {ak_ref[k][0]:.3f}, corr {ak_ref[k][1]:.3f}")

    model = AttentiveNeuralProcess(D=args.D, heads=args.heads, n_self=args.n_self,
                                   n_cross=args.n_cross, fourier_scale=args.fourier_scale).to(device)
    print(f"  NP params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    start = 0
    resume_id = None
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        sched.load_state_dict(ckpt['scheduler_state_dict'])
        start, resume_id = ckpt['step'] + 1, ckpt.get('wandb_id')
        run_str = os.path.basename(args.resume).split('_ep')[0][len(os.path.basename(stem)) + 1:]
        print(f"  Resumed from step {start}")

    if args.use_wandb and _WANDB_OK:
        wandb.init(project=args.wandb_project, name=f"belief_np_{run_str}", config=vars(args),
                   id=resume_id, resume="allow" if resume_id else None)
        for k, (err, corr) in ak_ref.items():
            wandb.run.summary[f'ak_ref/K{k}_rel_l2'] = err
            wandb.run.summary[f'ak_ref/K{k}_corr'] = corr

    class TerminateInterrupt(Exception):
        pass

    def _sigterm(signum, frame):
        raise TerminateInterrupt()
    signal.signal(signal.SIGTERM, _sigterm)

    def validate():
        model.eval()
        out = {}
        for k in _VAL_KS:
            err, corr = _belief_metrics(model.predict_grid(val_meas[k], args.val_grid), val_true)
            out[f'val/K{k}_rel_l2'], out[f'val/K{k}_corr'] = err, corr
        model.train()
        return out

    model.train()
    step = start
    try:
        for step in range(start, args.steps):
            idx = torch.randint(train_grids.shape[0], (args.batch,), device=device)
            grids = _augment(train_grids[idx])
            k = int(torch.randint(args.k_min, args.k_max + 1, ()))
            ctx = measure_density(grids, k, args.noise)
            q_uni = measure_density(grids, args.n_query // 2, 0.0)
            xq_den, y_den = _density_queries(grids, args.n_query - args.n_query // 2)
            xq = torch.cat([q_uni[..., :2], xq_den], dim=1)
            yq = torch.cat([q_uni[..., 2], y_den], dim=1)

            mean, std = model(ctx, xq)
            loss = gaussian_nll(mean, std, yq)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if (step + 1) % args.log_every == 0:
                log = {'train/nll': loss.item(), 'train/lr': sched.get_last_lr()[0], 'step': step + 1}
                if (step + 1) % args.val_every == 0:
                    log.update(validate())
                    print(f"  step {step + 1}: nll {loss.item():.4f}  " + "  ".join(
                        f"{n.split('/')[1]} {v:.3f}" for n, v in log.items() if n.startswith('val/')),
                        flush=True)
                if args.use_wandb and _WANDB_OK:
                    wandb.log(log)

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                path = f"{stem}_{run_str}_ep{step + 1:06d}.pt"
                _save(model, opt, sched, step, args, path)
                nach_zwischenstand(stem, run_str, path)
    except (KeyboardInterrupt, TerminateInterrupt):
        path = f"{stem}_{run_str}_ep{step + 1:06d}.pt"
        _save(model, opt, sched, step, args, path)
        nach_zwischenstand(stem, run_str, path)
        print(f"\n  [!] Interrupted at step {step + 1}, saved -> {path}")
        sys.exit(0)

    val = validate()
    final = f"{stem}_{run_str}_final.pt"
    _save(model, opt, sched, args.steps - 1, args, final, val={**val, **{
        f'ak_ref/K{k}_{m}': v for k, (e, c) in ak_ref.items() for m, v in (('rel_l2', e), ('corr', c))}})
    nach_endstand(stem, run_str, final)
    print(f"  Final: " + "  ".join(f"{n} {v:.3f}" for n, v in val.items()))
    print(f"  Saved -> {final}")
    if args.use_wandb and _WANDB_OK:
        wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--D', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--n_self', type=int, default=2, help='Self-attention layers over the context.')
    p.add_argument('--n_cross', type=int, default=2, help='Cross-attention layers query -> context.')
    p.add_argument('--fourier_scale', type=float, default=4.0,
                   help='Std of the Fourier frequencies (cycles per domain width).')
    p.add_argument('--k_min', type=int, default=32, help='Minimum number of measurements per task.')
    p.add_argument('--k_max', type=int, default=128, help='Maximum number of measurements per task.')
    p.add_argument('--noise', type=float, default=0.01,
                   help='Measurement noise std on the max-normalised density.')
    p.add_argument('--n_query', type=int, default=512, help='Query points per task.')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--val_grid', type=int, default=64)
    p.add_argument('--ak_reference', action='store_true', default=False,
                   help='Fit the AK-GP once on the val measurements as a reference.')
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--val_every', type=int, default=1000)
    p.add_argument('--save_every', type=int, default=5000)
    p.add_argument('--save_model', type=str, default=os.path.join(_here, 'checkpoints', 'belief_np.pt'))
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--use_wandb', action='store_true', default=False)
    p.add_argument('--wandb_project', type=str, default='flow-matching')
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
