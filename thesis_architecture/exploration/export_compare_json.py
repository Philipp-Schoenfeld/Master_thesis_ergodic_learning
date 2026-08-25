#!/usr/bin/env python3
r"""
export_compare_json.py
======================
Exportiert dieselbe Mission einmal mit Gradientenplaner und einmal mit Netz,
plus die Zieldichte, gegen die tatsaechlich geplant wurde.

Der letzte Punkt ist der, den bisher kein Bild zeigt: die Varianten planen nicht
gegen die wahre Dichte, sondern gegen Phi = mu + kappa*sigma aus dem Glauben.
Ohne diese Ebene sieht man nur, dass eine Bahn die Form verfehlt — nicht, ob sie
das Falsche getroffen hat oder ob ihr Ziel schon falsch war.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import common  # noqa: E402,F401
from common.acquisition import ucb_density, particles_from_density  # noqa: E402
from common.baselines import lawnmower_path, oracle_path            # noqa: E402
from common.data import initial_belief, load_truth                  # noqa: E402
from common.planner import GradientPlanner                          # noqa: E402
from common import metrics                                          # noqa: E402


def _load(d, m):
    spec = importlib.util.spec_from_file_location(
        m, os.path.join(_here, d, m + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def thin_path(p, n=110):
    p = p.detach().cpu().numpy()
    if len(p) <= n:
        return p
    return p[np.linspace(0, len(p) - 1, n).astype(int)]


def r(a, n=4):
    return [round(float(v), n) for v in np.asarray(a).reshape(-1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--belief_ckpt', required=True)
    p.add_argument('--segment_ckpt', required=True)
    p.add_argument('--lookahead_ckpt', required=True)
    p.add_argument('--shapes', type=int, default=6)
    p.add_argument('--grid_res', type=int, default=48)
    p.add_argument('--floor_res', type=int, default=40)
    p.add_argument('--n_prior', type=int, default=12)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--kappa', type=float, default=2.0)
    p.add_argument('--segments', type=int, default=3)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--lambda_cov', type=float, default=20000.0)
    p.add_argument('--target_length', type=float, default=4.0)
    p.add_argument('--grad_steps', type=int, default=200)
    p.add_argument('--out', default='results/eval/compare_export.json')
    a = p.parse_args()

    from evaluate_trained import ExplorerPlanner
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nets = {'belief': ExplorerPlanner(a.belief_ckpt, device=dev),
            'segment': ExplorerPlanner(a.segment_ckpt, device=dev),
            'lookahead': ExplorerPlanner(a.lookahead_ckpt, device=dev)}

    mods = {k: _load(d, m) for k, (d, m) in {
        'A': ('variant_a_combined', 'run_a'),
        'B': ('variant_b_diffsim', 'run_b'),
        'C': ('variant_c_receding', 'run_c'),
        'D': ('variant_d_belief_cond', 'run_d'),
        'E': ('variant_e_two_stage', 'run_e')}.items()}

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    D_EXEC = 0.35
    share = {'A': 1., 'B': 1., 'C': 1. / a.segments,
             'D': 1. / (a.rounds * D_EXEC), 'E': .5, 'ref': 1.}
    kw = dict(n_prior=a.n_prior, grid_res=a.grid_res, n_particles=a.n_particles)

    out = {'kappa': a.kappa, 'shapes': [],
           'variants': ['A', 'B', 'C', 'D', 'E'],
           'titles': {'A': 'A · kombinierte Dichte',
                      'B': 'B · diff. Vorausschau',
                      'C': 'C · receding horizon',
                      'D': 'D · glaubenskonditioniert',
                      'E': 'E · zweistufig'}}

    def down(g, f):
        idx = np.linspace(0, g.shape[0] - 1, f).astype(int)
        return g[np.ix_(idx, idx)]

    for i, name in enumerate(names):
        truth = T[i]
        b0 = initial_belief(truth, n_prior=a.n_prior, grid_res=a.grid_res, seed=i)
        mu, sd = b0.posterior_grid()
        phi = ucb_density(mu, sd, kappa=a.kappa)

        rec = {'label': name,
               'truth': r(down(truth.cpu().numpy(), a.floor_res), 3),
               'belief': r(down((phi / phi.max()).cpu().numpy(), a.floor_res), 3),
               'floor_res': a.floor_res, 'paths': {}, 'cov': {}}

        gp = GradientPlanner(steps=a.grad_steps, seed=i,
                             target_length=a.target_length)
        rec['paths']['Orakel'] = r(thin_path(
            oracle_path(truth, gp, a.n_particles, seed=i)))
        rec['paths']['Maeander'] = r(thin_path(
            lawnmower_path(target_length=a.target_length)))

        for pk in ('grad', 'netz'):
            def mk(seed, nxi=25, who='ref'):
                if pk == 'grad':
                    return GradientPlanner(nxi=nxi, steps=a.grad_steps, seed=seed,
                                           target_length=a.target_length * share[who])
                return nets['segment' if who == 'C' else
                            'lookahead' if who == 'B' else 'belief']

            per = {}
            per['A'] = mods['A'].run_mission(truth, mk(i, who='A'),
                                             kappa=a.kappa, seed=i, **kw)[0]
            if pk == 'grad':
                per['B'] = mods['B'].run_mission(
                    truth, GradientPlanner(steps=1, seed=i,
                                           target_length=a.target_length),
                    lambda_cov=a.lambda_cov, steps=a.grad_steps, seed=i, **kw)[0]
            else:
                per['B'] = mods['A'].run_mission(truth, mk(i, who='B'),
                                                 kappa=a.kappa, seed=i, **kw)[0]
            per['C'] = mods['C'].run_mission(
                truth, lambda sd_, nx: mk(sd_, nx, 'C'),
                n_segments=a.segments, seed=i, **kw)[0]
            per['D'] = mods['D'].run_mission(
                truth, lambda sd_: mk(sd_, who='D'), rounds=a.rounds,
                execute_frac=D_EXEC, seed=i, **kw)[0]
            per['E'] = mods['E'].run_mission(
                truth, lambda sd_: mk(sd_, who='E'), seed=i, **kw)[0]

            for v, path in per.items():
                rec['paths'][f'{pk}_{v}'] = r(thin_path(path))
                rec['cov'][f'{pk}_{v}'] = round(
                    float(metrics.coverage_vs_truth(path, truth)), 4)

        for k in ('Orakel', 'Maeander'):
            pth = torch.tensor(np.asarray(rec['paths'][k]).reshape(-1, 2))
            rec['cov'][k] = round(float(metrics.coverage_vs_truth(pth, truth)), 4)

        out['shapes'].append(rec)
        print(f"  {i+1}/{len(names)} {name}", flush=True)

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), separators=(',', ':'))
    print(f"\n  -> {a.out}  ({os.path.getsize(a.out)/1024:.0f} KB)")


if __name__ == '__main__':
    main()
