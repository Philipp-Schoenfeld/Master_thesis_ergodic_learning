#!/usr/bin/env python3
r"""
train_explorer.py
=================
Amortisierte Planung ueber Glaubenszustaenden — das Netz, das den
GradientPlanner in den Varianten A-E ersetzt.

WAS HIER TRAINIERT WIRD, UND WARUM NICHT FUENFMAL
-------------------------------------------------
A, C, D und E unterscheiden sich nicht im Modell, sondern in der Missions-
schleife *zur Laufzeit*. Fuenf getrennte Trainings waeren also drei identische
Kopien. Genuin verschieden sind drei Ziele:

  belief     volle Trajektorie aus einer UCB-Glaubensdichte.   -> A, D, E
  segment    kurzer Abschnitt mit erzwungenem Startpunkt und
             Beruecksichtigung der bereits gefahrenen Bahn.    -> C
  lookahead  wie `belief`, plus differenzierbare Vorausschau
             auf die verbleibende Unsicherheit.                -> B

Dazu kommt als Kontrolle das vorhandene, auf *wahren* Dichten trainierte Netz:
uebertraegt es sich ohne Weiteres auf Glaubensdichten, waere dieses Training
ueberfluessig — das ist eine Aussage, die man messen und nicht annehmen sollte.

SELBSTUEBERWACHT, WEIL ES KEINE LABELS GIBT
-------------------------------------------
Die Datenbank enthaelt Solver-Trajektorien fuer die *wahren* Dichten. Fuer eine
UCB-Glaubensdichte existiert keine Referenzloesung, und sie zu erzeugen hiesse,
den GradientPlanner tausendfach offline laufen zu lassen. Stattdessen wird
direkt gegen die ergodische Energie optimiert — dasselbe Muster wie in
`flow_matching_particles_selfsupervised.py`.

GLAUBENSZUSTAENDE WERDEN VORBERECHNET
-------------------------------------
Ein Glaubenszustand haengt nicht vom Modell ab. Die teure GP-Inversion muss
deshalb nur einmal laufen, nicht in jedem Trainingsschritt: `--objective belief`
und `segment` arbeiten auf einem Cache aus Phi-Gittern. Nur `lookahead` braucht
den GP live, weil sein Gradient durch die Vorausschau laeuft.

    python train_explorer.py --objective belief --epochs 400 --use_wandb
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_expl = os.path.normpath(os.path.join(_here, '..'))
sys.path.insert(0, _expl)

import common  # noqa: E402,F401  (richtet sys.path ein)
from common.acquisition import ucb_density, particles_from_density  # noqa: E402
from common.belief import GPBelief                                  # noqa: E402
from common.data import DEFAULT_DB                                  # noqa: E402
from common.observation import measure, thin                        # noqa: E402

from ergodic_metric import ErgodicLoss                              # noqa: E402
from flow_matching_particles_selfsupervised import (                # noqa: E402
    SelfSupervisedParticleGenerator)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


class TerminateInterrupt(Exception):
    pass


def _sigterm(signum, frame):
    raise TerminateInterrupt()


# ===========================================================================
# Glaubenszustaende
# ===========================================================================

def make_belief(truth, rng, grid_res, lengthscale, noise, planner=None):
    """Ein zufaelliger Glaubenszustand.

    Zwei Sorten, absichtlich gemischt: Streumessungen stehen fuer "ein paar
    Stichproben genommen", eine abgefahrene Bahn fuer "schon ein Stueck der
    Mission hinter sich". Trainiert man nur auf der ersten, sieht das Netz die
    Glaubenszustaende nie, die in den Varianten C und D tatsaechlich auftreten —
    dort folgt jede Planung auf eine bereits gefahrene Strecke.
    """
    b = GPBelief(grid_res=grid_res, lengthscale=lengthscale, noise=noise)
    n = int(rng.integers(0, 40))
    if n:
        pts = torch.from_numpy(rng.random((n, 2)).astype(np.float32))
        if rng.random() < 0.5:
            # Zusammenhaengende Bahn statt Streupunkten.
            t = torch.linspace(0, 1, n)[:, None]
            a = torch.from_numpy(rng.random((1, 2)).astype(np.float32))
            c = torch.from_numpy(rng.random((1, 2)).astype(np.float32))
            pts = (a + t * (c - a) + 0.05 * torch.randn(n, 2)).clamp(0, 1)
        p, v = measure(pts, truth, noise_std=noise, sensor_radius=0.03)
        b.observe(*thin(p, v, max_points=64))
    return b


def build_cache(truths, n_states, grid_res, lengthscale, noise, kappa_range,
                seed=0, verbose=True):
    """Vorberechnete Phi-Gitter. -> (Phi (M,R,R), shape_idx (M,))"""
    rng = np.random.default_rng(seed)
    phis, idx = [], []
    t0 = time.perf_counter()
    for m in range(n_states):
        i = int(rng.integers(0, len(truths)))
        b = make_belief(truths[i], rng, grid_res, lengthscale, noise)
        mu, sd = b.posterior_grid()
        k = float(rng.uniform(*kappa_range))
        phis.append(ucb_density(mu, sd, kappa=k))
        idx.append(i)
        if verbose and (m + 1) % max(1, n_states // 10) == 0:
            print(f"    {m+1}/{n_states} Glaubenszustaende "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return torch.stack(phis), torch.tensor(idx)


# ===========================================================================
# Verluste
# ===========================================================================

def regularisers(cps, boundary_w=10.0, smooth_w=0.5, target_len=None,
                 length_w=20.0, basis=None):
    out = (cps - cps.clamp(0.0, 1.0)).pow(2).sum(dim=(1, 2))
    acc = cps[:, 2:] - 2 * cps[:, 1:-1] + cps[:, :-2]
    reg = boundary_w * out + smooth_w * acc.pow(2).sum(dim=(1, 2))
    if target_len is not None and basis is not None:
        curve = torch.einsum('pi,bid->bpd', basis, cps)
        L = (curve[:, 1:] - curve[:, :-1]).norm(dim=-1).sum(dim=1)
        reg = reg + length_w * (L - target_len).pow(2)
    return reg


def diversity_reward(cps, weight):
    """Abstossung zwischen gleichzeitig erzeugten Kandidaten."""
    if weight <= 0 or cps.shape[0] < 2:
        return cps.new_zeros(())
    flat = cps.flatten(1)
    d = torch.cdist(flat, flat)
    n = d.shape[0]
    off = d[~torch.eye(n, dtype=torch.bool, device=d.device)]
    return -weight * off.mean()


# ===========================================================================
# Training
# ===========================================================================

def run(args):
    import signal
    signal.signal(signal.SIGTERM, _sigterm)

    dev = torch.device(args.device or
                       ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"  Geraet: {dev}")

    # ── wahre Dichten ──────────────────────────────────────────────────────
    from common.data import load_truth
    names, truths = load_truth(n=args.n_shapes, split='train',
                               resolution=args.grid_res, db_path=args.db)
    print(f"  {len(names)} Trainingsformen, Gitter {args.grid_res}^2")

    nxi = args.seg_nxi if args.objective == 'segment' else args.nxi
    model = SelfSupervisedParticleGenerator(nxi=nxi, nd=2, D=args.D).to(dev)
    print(f"  Modell: {sum(p.numel() for p in model.parameters()):,} Parameter, "
          f"nxi={nxi}")

    erg = ErgodicLoss(nxi=nxi, K=args.erg_K, pts=args.erg_pts, weight=1.0,
                      t_power=0.0, metric=args.metric).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)
    start_ep = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimizer_state_dict'])
        sched.load_state_dict(ck['scheduler_state_dict'])
        start_ep = ck['epoch'] + 1
        print(f"  Fortsetzung ab Epoche {start_ep}")

    # ── Glaubenszustaende ──────────────────────────────────────────────────
    beliefs = None
    if args.objective == 'lookahead':
        print(f"  Vorausschau-Ziel: GP bleibt live (Gradient laeuft hindurch)")
        rng = np.random.default_rng(args.seed)
        beliefs = [make_belief(truths[int(rng.integers(0, len(truths)))], rng,
                               args.grid_res, args.lengthscale, args.noise)
                   for _ in range(args.n_states)]
        b_idx = torch.tensor([int(rng.integers(0, len(truths)))
                              for _ in range(args.n_states)])
        phis = torch.stack([ucb_density(*b.posterior_grid(),
                                        kappa=float(rng.uniform(*args.kappa)))
                            for b in beliefs])
    else:
        cache_f = os.path.join(args.cache_dir,
                               f"phis_{args.objective}_{args.n_states}_"
                               f"{args.grid_res}_{args.n_shapes}_{args.seed}.pt")
        if os.path.isfile(cache_f):
            d = torch.load(cache_f, weights_only=False)
            phis, b_idx = d['phis'], d['idx']
            print(f"  Glaubens-Cache geladen: {tuple(phis.shape)}")
        else:
            print(f"  Baue {args.n_states} Glaubenszustaende...")
            phis, b_idx = build_cache(truths, args.n_states, args.grid_res,
                                      args.lengthscale, args.noise, args.kappa,
                                      seed=args.seed)
            os.makedirs(args.cache_dir, exist_ok=True)
            torch.save({'phis': phis, 'idx': b_idx}, cache_f)
            print(f"  -> {cache_f}")

    phis = phis.to(dev)

    if args.use_wandb and _WANDB:
        wandb.init(project=args.wandb_project, name=args.run_str,
                   config=vars(args))

    basis = erg.B
    from tqdm import tqdm
    pbar = tqdm(range(start_ep, args.epochs), desc=f"Explorer/{args.objective}",
                unit='ep', initial=start_ep, total=args.epochs)
    avg = float('nan')

    try:
        for ep in pbar:
            perm = torch.randperm(phis.shape[0])
            tot, nb, parts_log = 0.0, 0, {}

            for s in range(0, len(perm), args.mini_batch):
                sel = perm[s:s + args.mini_batch]
                if len(sel) < 2:
                    continue
                phi_b = phis[sel]

                pcloud = torch.stack([
                    particles_from_density(phi_b[j], args.n_particles,
                                           device=dev)
                    for j in range(phi_b.shape[0])])

                z = torch.randn(phi_b.shape[0], nxi, 2, device=dev)
                start = None
                if args.objective == 'segment':
                    start = torch.rand(phi_b.shape[0], 1, 2, device=dev)

                opt.zero_grad(set_to_none=True)
                cps = model(z, pcloud)
                if start is not None:
                    cps = torch.cat([start, cps[:, 1:]], dim=1)

                e = erg.coverage_error(cps, pcloud)
                r = regularisers(cps, target_len=args.target_length,
                                 basis=basis)
                loss = (args.w_erg * e + r).mean()
                loss = loss + diversity_reward(cps, args.diversity_weight)
                comp = {'erg': e.mean().detach(), 'reg': r.mean().detach()}

                if args.objective == 'lookahead':
                    curve = torch.einsum('pi,bid->bpd', basis, cps)
                    idx = torch.linspace(0, curve.shape[1] - 1,
                                         args.n_probe).long()
                    unc = torch.stack([
                        beliefs[int(sel[j])].uncertainty_after(
                            curve[j, idx].clamp(0, 1).cpu())
                        for j in range(cps.shape[0])]).to(dev)
                    loss = loss + args.w_unc * unc.mean()
                    comp['unc'] = unc.mean().detach()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip_grad)
                opt.step()

                tot += float(loss)
                nb += 1
                for k, v in comp.items():
                    parts_log[k] = parts_log.get(k, 0.0) + float(v)

            sched.step()
            avg = tot / max(nb, 1)
            aparts = {k: v / max(nb, 1) for k, v in parts_log.items()}
            if ep % 5 == 0 or ep == args.epochs - 1:
                pbar.set_postfix(loss=f"{avg:.4f}",
                                 **{k: f"{v:.3e}" for k, v in aparts.items()})
            if args.use_wandb and _WANDB:
                wandb.log({'train/loss': avg, 'epoch': ep + 1,
                           'train/lr': sched.get_last_lr()[0],
                           **{f'train/{k}': v for k, v in aparts.items()}})

            if args.save_every > 0 and (ep + 1) % args.save_every == 0:
                save(model, opt, sched, ep, avg, args)

    except (KeyboardInterrupt, TerminateInterrupt):
        print(f"\n  [!] Abbruch bei Epoche {ep+1} — Notfall-Checkpoint...")
        save(model, opt, sched, ep, avg, args, tag='emergency')
        sys.exit(0)

    save(model, opt, sched, args.epochs - 1, avg, args, tag='final')
    print("  Training abgeschlossen.")


def save(model, opt, sched, ep, loss, args, tag=None):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    suffix = tag if tag else f"ep{ep+1:04d}"
    path = os.path.join(args.ckpt_dir, f"explorer_{args.run_str}_{suffix}.pt")
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': sched.state_dict(),
                'epoch': ep, 'loss': loss,
                'objective': args.objective, 'nxi': args.nxi,
                'seg_nxi': args.seg_nxi, 'D': args.D,
                'n_particles': args.n_particles, 'erg_K': args.erg_K,
                'metric': args.metric, 'kappa': args.kappa,
                'grid_res': args.grid_res, 'lengthscale': args.lengthscale,
                'target_length': args.target_length,
                'args': vars(args)}, path)
    print(f"  Checkpoint -> {path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--objective', required=True,
                   choices=['belief', 'segment', 'lookahead'],
                   help='belief -> A/D/E, segment -> C, lookahead -> B')
    p.add_argument('--n_shapes', type=int, default=400)
    p.add_argument('--n_states', type=int, default=4000,
                   help='Vorberechnete Glaubenszustaende.')
    p.add_argument('--grid_res', type=int, default=32)
    p.add_argument('--lengthscale', type=float, default=0.08)
    p.add_argument('--noise', type=float, default=0.02)
    p.add_argument('--kappa', type=float, nargs=2, default=[0.0, 6.0],
                   help='kappa wird je Zustand aus diesem Bereich gezogen, '
                        'damit das Netz den ganzen Erkunden-Ausbeuten-Bogen '
                        'sieht statt nur einer Einstellung.')
    p.add_argument('--nxi', type=int, default=25)
    p.add_argument('--seg_nxi', type=int, default=10)
    p.add_argument('--D', type=int, default=384)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--erg_K', type=int, default=8)
    p.add_argument('--erg_pts', type=int, default=128)
    p.add_argument('--metric', default='fourier', choices=['fourier', 'sinkhorn'])
    p.add_argument('--w_erg', type=float, default=100.0)
    p.add_argument('--w_unc', type=float, default=0.01,
                   help='Vorausschau-Gewicht. Der Unsicherheitsterm liegt bei '
                        'einigen hundert, der ergodische bei 1e-2 — ohne diese '
                        'Skalierung erschlaegt er alles.')
    p.add_argument('--n_probe', type=int, default=32)
    p.add_argument('--target_length', type=float, default=4.0,
                   help='Ziel-Weglaenge, damit die Varianten spaeter bei '
                        'gleichem Budget vergleichbar sind.')
    p.add_argument('--diversity_weight', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--mini_batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--db', default=DEFAULT_DB)
    p.add_argument('--ckpt_dir', default=os.path.join(_expl, 'checkpoints'))
    p.add_argument('--cache_dir', default=os.path.join(_expl, 'cache'))
    p.add_argument('--resume', default=None)
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', default='exploration-2d')
    a = p.parse_args()

    if not hasattr(a, 'timestamp'):
        a.timestamp = datetime.now().strftime("date_%m_%d_%Hh%Mmin")
    a.run_str = (f"{a.objective}_{a.timestamp}_D{a.D}_N{a.n_particles}"
                 f"_S{a.n_states}_K{a.erg_K}_k{a.kappa[0]:g}-{a.kappa[1]:g}"
                 f"_L{a.target_length:g}_{a.metric}")
    return a


if __name__ == '__main__':
    run(parse_args())
