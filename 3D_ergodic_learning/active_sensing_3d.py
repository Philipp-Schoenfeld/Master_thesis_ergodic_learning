#!/usr/bin/env python3
"""3D active-sensing missions on surfaces with an unknown density.

Port of the 2D replanning missions (thesis_architecture/evaluation_full_matrix/
run_mission_eval.py) to the 3D surface setting. One mission = one holdout entry
(surface + true density mu on the 512 surface particles) and one initialiser:

  1. belief: an exact RBF-GP over the particle positions (prior mean 0, noise
     0.05) is updated with what the sensor measured so far,
  2. target density: UCB, phi = clip(mu_post + kappa * sd) * (1 - visited), with
     kappa annealed linearly kappa0 -> kappa1 over the rounds; peak-normalised
     and used as the particle weights of the planner,
  3. `n_cand` candidate trajectories from the current position:
         <network>  start-conditioned CFM (pos 3 + rot6 6, control point 0 = position),
         linear     straight rays on a Fibonacci sphere, reflected at the box walls,
                    sensor axis looking at the nearest particle,
  4. every candidate is refined with the training objective against phi
     (refine_db3d.Refiner: Sinkhorn footprint coverage + orientation + shape
     guard, start held); the best candidate (lowest objective against phi; the
     planner never sees the truth) is driven completely,
  5. the sensor measures along it: every particle within `r_sense` of a beam
     footprint (position + distance to the nearest particle * sensor axis) is
     read with Gaussian noise; the belief is updated; back to 1. until the swept
     truth mass reaches `reach` (0.99) or `rounds` are used.

Stored per (method, condition, mission, round): Sinkhorn footprint error of the
whole driven path against the TRUE density (E_truth), swept mass, path length,
belief RMSE, objective against phi. Conditions: none_known, ten_samples (10 random
particles read before round 0).

    python active_sensing_3d.py --db3d ergodic_dataset_3d_lm.db \
        --models "DiT-S, refined DB=ckpt.pt" --methods "DiT-S, refined DB" linear
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from ergodic_metric import sinkhorn_error
from obstacles import bspline_basis_matrix
from orientation import rot6d_to_matrix, sensor_axis
from orientation_energy import ParticleSurface

_here = os.path.dirname(os.path.abspath(__file__))
NXI, PTS, DEG = 25, 128, 5
LO, HI = 0.02, 0.98


# ---------------------------------------------------------------------------
# belief
# ---------------------------------------------------------------------------
class GPBelief:
    """Exact GP over the particle positions; the density is read at the particles only."""

    def __init__(self, xyz, noise=0.05, lengthscale=0.1, signal=1.0):
        x = torch.as_tensor(xyz, dtype=torch.float64)
        d2 = torch.cdist(x, x).pow(2)
        self.K = signal ** 2 * torch.exp(-0.5 * d2 / lengthscale ** 2)
        self.noise, self.signal = noise, signal
        self.idx, self.val = [], []

    def observe(self, idx, vals):
        self.idx += [int(i) for i in idx]
        self.val += [float(v) for v in vals]

    @property
    def n_obs(self):
        return len(self.idx)

    def posterior(self):
        """-> mean (N,), sd (N,) as float32 numpy arrays."""
        N = self.K.shape[0]
        if not self.idx:
            return np.zeros(N, np.float32), np.full(N, self.signal, np.float32)
        i = torch.as_tensor(self.idx)
        y = torch.as_tensor(self.val, dtype=torch.float64)
        L = torch.linalg.cholesky(self.K[i][:, i] + self.noise ** 2 * torch.eye(len(i), dtype=torch.float64))
        Ks = self.K[i]                                           # (n, N)
        mu = Ks.T @ torch.cholesky_solve(y[:, None], L)[:, 0]
        v = torch.linalg.solve_triangular(L, Ks, upper=False)
        sd = (self.signal ** 2 - v.pow(2).sum(0)).clamp(min=1e-8).sqrt()
        return mu.float().numpy(), sd.float().numpy()


def kappa_schedule(r, rounds, k0, k1):
    return k0 if rounds <= 1 else k0 + (k1 - k0) * r / (rounds - 1)


def ucb_density(mu, sd, visit, kappa):
    """Peak-normalised UCB target with a visitation discount; never exactly zero."""
    phi = np.clip(mu + kappa * sd, 0.0, None) * np.clip(1.0 - visit, 0.0, 1.0)
    return np.clip(phi / max(phi.max(), 1e-9), 1e-4, None).astype(np.float32)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def rot6_from_axis(z):
    """(..., 3) unit sensor axes -> (..., 6) with column 2 of the rotation = z."""
    ref = torch.zeros_like(z)
    ref[..., 2] = 1.0
    flat = z[..., 2].abs() > 0.9
    ref[flat] = torch.tensor([1.0, 0.0, 0.0], dtype=z.dtype, device=z.device)
    b1 = ref - (ref * z).sum(-1, keepdim=True) * z
    b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    b2 = torch.cross(z, b1, dim=-1)
    return torch.cat([b1, b2], dim=-1)


def fibonacci_sphere(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)


def reflect(x, lo=LO, hi=HI):
    u = (x - lo) / (hi - lo)
    u = 1.0 - np.abs(1.0 - np.mod(u, 2.0))
    return lo + u * (hi - lo)


def linear_candidates(start, xyz, n, length, rng):
    """(n, NXI, 9): straight rays from `start`, reflected at the walls, looking at the nearest particle."""
    d = fibonacci_sphere(n)
    d = d @ np.linalg.qr(rng.normal(size=(3, 3)))[0]             # random rotation of the ray set
    s = np.linspace(0.0, length, NXI)
    pos = reflect(start[None, None, :] + d[:, None, :] * s[None, :, None]).astype(np.float32)
    pos[:, 0] = start
    p = torch.from_numpy(pos)
    P = torch.from_numpy(np.asarray(xyz, np.float32))
    j = torch.cdist(p.reshape(-1, 3), P).argmin(-1)
    z = (P[j] - p.reshape(-1, 3))
    z = (z / z.norm(dim=-1, keepdim=True).clamp(min=1e-9)).reshape(n, NXI, 3)
    return torch.cat([p, rot6_from_axis(z)], dim=-1)


class Sensor:
    """Beam footprint on the full surface, read-out and sweep metrics."""

    def __init__(self, device):
        self.B = torch.from_numpy(bspline_basis_matrix(NXI, PTS * 2, DEG)).float().to(device)
        self.dev = device

    @torch.no_grad()
    def curve(self, x, parts):
        """x (NXI, 9), parts (N, 4) -> positions (T, 3), footprints (T, 3)."""
        x = x.to(self.dev).float()
        pos = self.B @ x[:, :3]
        ax = sensor_axis(rot6d_to_matrix(self.B @ x[:, 3:]), 2)
        ax = ax / ax.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        surf = ParticleSurface(parts[None].to(self.dev).float(), 0.0)   # all particles
        foot = surf.footprint(pos[None], ax[None])[0]
        return pos, foot


def within(foot, xyz, r):
    """Boolean (N,) - particles within r of any footprint point. foot (T,3), xyz (N,3) tensors."""
    return (torch.cdist(xyz, foot).min(-1).values <= r)


# ---------------------------------------------------------------------------
# planners
# ---------------------------------------------------------------------------
@torch.no_grad()
def net_candidates(model, phi_parts, start, n, steps, cfg, seed, device):
    from flow_matching_cond_particles_crossattn import generate_particle_trajectories
    g = torch.Generator(device=device).manual_seed(seed)
    cps, rot6 = generate_particle_trajectories(
        model, phi_parts.to(device), num_samples=n, steps=steps, device=str(device),
        cfg_weight=cfg, start=start.to(device), generator=g)
    return torch.cat([cps, rot6], dim=-1).float()


# ---------------------------------------------------------------------------
# missions
# ---------------------------------------------------------------------------
class Mission:
    def __init__(self, entry, cond, a, rng, device):
        self.e = entry
        self.name, self.split, self.group = entry['name'], entry['split'], entry['gruppe']
        parts = np.asarray(entry['parts'], np.float32)
        self.xyz = parts[:, :3]
        self.truth = parts[:, 3] / max(parts[:, 3].max(), 1e-9)
        self.xyz_t = torch.from_numpy(self.xyz).to(device)
        self.belief = GPBelief(self.xyz, a.gp_noise, a.gp_lengthscale)
        self.rng = rng
        self.pos = np.asarray(entry['start'], np.float32)
        self.visit = np.zeros(len(self.xyz), np.float32)
        self.seen = np.zeros(len(self.xyz), bool)
        self.foot_all = []                        # list of (T, 3) tensors
        self.length = 0.0
        self.done = False
        self.rows = []
        self.paths = []
        if cond == 'ten_samples':
            idx = rng.choice(len(self.xyz), 10, replace=False)
            self.belief.observe(idx, self.read(idx))
        self.mass0 = float(self.truth.sum())

    def read(self, idx):
        return np.clip(self.truth[idx] + self.rng.normal(0, self.belief.noise, len(idx)), 0, None)

    def swept(self):
        return float(self.truth[self.seen].sum() / self.truth.sum())


def sinkhorn_truth(refiner, foot_cat, parts_truth):
    c = foot_cat[None].float()
    return float(sinkhorn_error(c, parts_truth[None].float(), refiner.erg._sinkhorn, True, 3)[0])


def run_cell(method, cond, entries, a, sensor, refiner, models, device):
    """Lockstep missions of all entries for one (method, condition)."""
    rng_master = np.random.default_rng(a.seed)
    ms = [Mission(e, cond, a, np.random.default_rng(rng_master.integers(1 << 30)), device)
          for e in entries]
    truth_parts = [torch.from_numpy(np.concatenate([m.xyz, m.truth[:, None]], 1)).to(device) for m in ms]
    for r in range(a.rounds):
        act = [i for i, m in enumerate(ms) if not m.done]
        if not act:
            break
        kappa = kappa_schedule(r, a.rounds, a.kappa0, a.kappa1)
        t0 = time.time()
        phis, cands = [], []
        for i in act:
            m = ms[i]
            mu, sd = m.belief.posterior()
            phi = ucb_density(mu, sd, m.visit, kappa)
            parts = torch.from_numpy(np.concatenate([m.xyz, phi[:, None]], 1))
            phis.append(parts)
            seed = a.seed * 1000003 + r * 1009 + i
            if method == 'linear':
                x = linear_candidates(m.pos, m.xyz, a.n_cand, a.lin_len, np.random.default_rng(seed))
            else:
                x = net_candidates(models[method], parts, torch.from_numpy(m.pos), a.n_cand,
                                   a.steps, a.cfg_weight, seed, device).cpu()
            cands.append(x)
        # refinement in chunks over missions x candidates
        X = torch.stack(cands)                                    # (M, n, NXI, 9)
        Pm = torch.stack(phis)                                    # (M, N, 4)
        M, n = X.shape[:2]
        per = max(1, a.chunk // n)
        Xr, Lr = [], []
        for s in range(0, M, per):
            xb = X[s:s + per].reshape(-1, NXI, 9).to(device)
            pb = Pm[s:s + per].to(device)[:, None].expand(-1, n, -1, -1).reshape(-1, Pm.shape[1], 4)
            if a.refine_iters > 0:
                xb, lb = refiner.run(xb, pb.contiguous(), iters=a.refine_iters, lr=a.refine_lr)
            else:
                with torch.no_grad():
                    ref = tuple(q.clamp(min=1e-6) for q in refiner.shape_stats(xb[..., :3]))
                    lb, _ = refiner.objective(xb, pb.contiguous(), ref)
            Xr.append(xb.cpu().reshape(-1, n, NXI, 9))
            Lr.append(lb.cpu().reshape(-1, n))
        Xr, Lr = torch.cat(Xr), torch.cat(Lr)
        t_plan = time.time() - t0
        for k, i in enumerate(act):
            m = ms[i]
            lk = torch.nan_to_num(Lr[k], nan=float('inf'))
            j = int(lk.argmin())
            best = Xr[k, j]
            pos, foot = sensor.curve(best, Pm[k])
            near = within(foot, m.xyz_t, a.r_sense).cpu().numpy()
            idx = np.nonzero(near)[0]
            m.belief.observe(idx, m.read(idx))
            m.seen |= within(foot, m.xyz_t, a.r_cov).cpu().numpy()
            d2 = torch.cdist(m.xyz_t, foot).min(-1).values.cpu().numpy()
            m.visit = np.maximum(m.visit, np.exp(-0.5 * (d2 / a.r_sense) ** 2)).astype(np.float32)
            m.foot_all.append(foot)
            m.length += float((pos[1:] - pos[:-1]).norm(dim=-1).sum())
            m.pos = pos[-1].cpu().numpy().astype(np.float32)
            mu, _ = m.belief.posterior()
            m.rows.append(dict(round=r, e_truth=sinkhorn_truth(refiner, torch.cat(m.foot_all), truth_parts[i]),
                               swept=m.swept(), length=m.length, belief_rmse=float(np.sqrt(((mu - m.truth) ** 2).mean())),
                               n_obs=m.belief.n_obs, loss_sel=float(lk[j]), kappa=kappa))
            m.paths.append(dict(pos=pos.cpu().numpy(), foot=foot.cpu().numpy()))
            m.done = m.swept() >= a.reach
        print(f"  [{method} / {cond}] round {r}: {len(act)} missions, plan+refine {t_plan:.0f} s, "
              f"mean swept {np.mean([m.swept() for m in ms]):.3f}, "
              f"mean E_truth {np.mean([m.rows[-1]['e_truth'] for m in ms]):.3e}", flush=True)
    return ms


def save_cell(out_dir, method, cond, ms):
    tag = f"{method.replace(' ', '_').replace(',', '').replace('/', '-')}__{cond}"
    rec = [dict(name=m.name, split=m.split, group=m.group, rows=m.rows) for m in ms]
    json.dump(rec, open(os.path.join(out_dir, tag + '.json'), 'w'))
    arr = {}
    for i, m in enumerate(ms):
        for r, p in enumerate(m.paths):
            arr[f'pos_{i}_{r}'] = p['pos'].astype(np.float16)
            arr[f'foot_{i}_{r}'] = p['foot'].astype(np.float16)
    np.savez_compressed(os.path.join(out_dir, tag + '_paths.npz'), **arr)
    return tag


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db3d', default='ergodic_dataset_3d_lm.db')
    p.add_argument('--models', nargs='*', default=[], help='"label=checkpoint.pt" ...')
    p.add_argument('--methods', nargs='+', required=True, help='network labels and/or "linear"')
    p.add_argument('--conditions', nargs='+', default=['none_known', 'ten_samples'])
    p.add_argument('--splits', nargs='+', default=['val_form', 'val_flaeche', 'val_beides'])
    p.add_argument('--per_split', type=int, default=9)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--reach', type=float, default=0.99)
    p.add_argument('--n_cand', type=int, default=30)
    p.add_argument('--steps', type=int, default=100, help='CFM integration steps')
    p.add_argument('--cfg_weight', type=float, default=2.0)
    p.add_argument('--refine_iters', type=int, default=150)
    p.add_argument('--refine_lr', type=float, default=2e-3)
    p.add_argument('--chunk', type=int, default=300, help='candidates per refinement batch')
    p.add_argument('--kappa0', type=float, default=1.0)
    p.add_argument('--kappa1', type=float, default=0.05)
    p.add_argument('--gp_noise', type=float, default=0.05)
    p.add_argument('--gp_lengthscale', type=float, default=0.1)
    p.add_argument('--r_sense', type=float, default=0.06)
    p.add_argument('--r_cov', type=float, default=0.06)
    p.add_argument('--lin_len', type=float, default=6.5, help='length of the linear rays')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'active_sensing_3d'))
    p.add_argument('--device', default=None)
    a = p.parse_args()

    from data_surfaces import load_surface_db
    from eval_lm3d_compare import load_model, pick_entries
    from refine_db3d import Refiner

    dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(a.out_dir, exist_ok=True)
    entries = pick_entries(load_surface_db(a.db3d, splits=tuple(a.splits)), a.per_split, a.seed)
    print(f"{len(entries)} missions per cell: "
          + ", ".join(f"{s}={sum(e['split'] == s for e in entries)}" for s in a.splits), flush=True)
    paths = dict(m.split('=', 1) for m in a.models)
    models = {}
    for meth in a.methods:
        if meth != 'linear':
            models[meth], info = load_model(paths[meth], dev)
            print(f"loaded {meth}: {info}", flush=True)
    refiner = Refiner(dev)
    sensor = Sensor(dev)
    json.dump(dict(args=vars(a)), open(os.path.join(a.out_dir, 'config.json'), 'w'), indent=1)
    for meth in a.methods:
        for cond in a.conditions:
            tag = f"{meth.replace(' ', '_').replace(',', '').replace('/', '-')}__{cond}"
            if os.path.exists(os.path.join(a.out_dir, tag + '.json')):
                print(f"skip {tag} (done)", flush=True)
                continue
            t0 = time.time()
            ms = run_cell(meth, cond, entries, a, sensor, refiner, models, dev)
            save_cell(a.out_dir, meth, cond, ms)
            print(f"done {tag} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
