#!/usr/bin/env python3
r"""
evaluate_trained.py
===================
Auswertung der trainierten Explorer-Netze in den Varianten A-E: Metriken bei
festem Wegbudget, visuelle Vergleiche, und die Amortisierungsmessung.

Drei Fragen, die getrennt beantwortet werden:

**1. Welche Missionsschleife ist die beste?** Planer festhalten, Variante
wechseln. Ausgewertet ueber Anytime-Kurven bei mehreren Wegbudgets, mit Orakel
und Maeander als Bezugspunkten und gepaarter Auszaehlung pro Form.

**2. Zahlt sich Amortisierung aus?** Variante festhalten, Planer wechseln:
Gradientenplaner (der "Solver") gegen das trainierte Netz. Gemessen wird nicht
nur die Guete, sondern vor allem die Wanduhrzeit *pro Planungsschritt* — bei
Variante C faellt die dutzendfach pro Mission an, und genau dort entscheidet
sich, ob Amortisierung mehr ist als eine Bequemlichkeit.

**3. Braucht es glaubenskonditioniertes Training ueberhaupt?** Das auf *wahren*
Dichten trainierte Netz laeuft als Kontrolle mit. Uebertraegt es sich ohne
Weiteres auf UCB-Glaubensdichten, war das Training hier ueberfluessig — eine
Aussage, die man messen und nicht annehmen sollte.

Die Bilder folgen den Projektkonventionen: WHITE_INFERNO fuer die Dichte,
#1565C0 fuer Referenzbahnen, #00C853 fuer erzeugte, dunkle Punkte fuer Partikel.

    python evaluate_trained.py --auto            # nimmt die neuesten Checkpoints
    python evaluate_trained.py --belief_ckpt ... --segment_ckpt ... --lookahead_ckpt ...
"""

import argparse
import csv
import glob
import importlib.util
import os
import statistics as st
import sys
import time

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import common  # noqa: E402,F401
from common import metrics                                        # noqa: E402
from common.acquisition import ucb_density, particles_from_density  # noqa: E402
from common.baselines import lawnmower_path, oracle_path, random_path  # noqa: E402
from common.data import initial_belief, load_truth                # noqa: E402
from common.planner import BasePlanner, GradientPlanner           # noqa: E402


# ===========================================================================
# Planer um ein trainiertes Explorer-Netz
# ===========================================================================

class ExplorerPlanner(BasePlanner):
    """Amortisierter Planer: ein Vorwaertsdurchgang statt einer Optimierung."""

    def __init__(self, checkpoint, device='cpu', pts=128, deg=5):
        from obstacles import bspline_basis_matrix
        from flow_matching_particles_selfsupervised import (
            SelfSupervisedParticleGenerator)

        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        self.meta = ck
        self.objective = ck.get('objective', 'belief')
        self.nxi = ck.get('seg_nxi' if self.objective == 'segment' else 'nxi', 25)
        self.device = torch.device(device)
        # Basis auf CPU: render() wird auf den CPU-Kontrollpunkten aufgerufen.
        self.B = torch.from_numpy(bspline_basis_matrix(self.nxi, pts, deg))
        self.model = SelfSupervisedParticleGenerator(
            nxi=self.nxi, nd=2, D=ck.get('D', 384)).to(device)
        self.model.load_state_dict(ck['model_state_dict'])
        self.model.eval()
        self.last_wallclock = 0.0
        self.name = os.path.basename(checkpoint)

    @torch.no_grad()
    def plan(self, particles, n_candidates=1, init=None, history=None,
             start=None):
        t0 = time.perf_counter()
        p = particles.to(self.device).unsqueeze(0).expand(n_candidates, -1, -1)
        z = torch.randn(n_candidates, self.nxi, 2, device=self.device)
        cps = self.model(z, p)
        if start is not None:
            cps = torch.cat([start.to(self.device).view(1, 1, 2).expand(
                n_candidates, 1, 2), cps[:, 1:]], dim=1)
        self.last_wallclock = time.perf_counter() - t0
        # Auf CPU zurueckgeben: die Missionsschleife (Glaube, Messung, Metrik)
        # laeuft dort, und der Gradientenplaner tut es auch. Beide Pfade muessen
        # sich identisch verhalten, sonst vergleicht die Auswertung nebenbei
        # Geraete statt Verfahren.
        return cps.clamp(0.0, 1.0).cpu()


def newest(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


# ===========================================================================
# Bilder
# ===========================================================================

def _white_inferno():
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    c = plt.colormaps['inferno'](np.linspace(0, 1, 256))
    for i in range(40):
        t = i / 40
        c[i] = (1 - t) * np.array([1, 1, 1, 1]) + t * c[40]
    return mcolors.LinearSegmentedColormap.from_list('white_inferno', c)


def plot_missions(rows, out_path, suptitle):
    """Raster: Formen als Zeilen, Varianten als Spalten.

    Je Feld die Zieldichte, aus der geplant wurde, darunter die gefahrene Bahn.
    Der Vergleich der *Zeilen* zeigt, wie unterschiedlich die Varianten dieselbe
    Form angehen; der Vergleich mit der wahren Dichte im Hintergrund, ob sie
    das Richtige getroffen haben.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    WI = _white_inferno()
    shapes = sorted({r['shape'] for r in rows})
    variants = [v for v in ['Orakel', 'A', 'B', 'C', 'D', 'E', 'Maeander']
                if any(r['variant'] == v for r in rows)]
    nr, nc = len(shapes), len(variants)
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 2.9 * nr),
                             facecolor='white', squeeze=False)

    for i, sh in enumerate(shapes):
        for j, va in enumerate(variants):
            ax = axes[i][j]
            ax.set_facecolor('white')
            hit = [r for r in rows if r['shape'] == sh and r['variant'] == va]
            if not hit:
                ax.axis('off')
                continue
            r = hit[0]
            ax.imshow(r['truth'], origin='lower', extent=[0, 1, 0, 1],
                      cmap=WI, alpha=0.55)
            if r.get('particles') is not None:
                p = r['particles']
                ax.scatter(p[:, 0], p[:, 1], s=6, c='#444444', alpha=0.3,
                           linewidths=0)
            c = r['path']
            ax.plot(c[:, 0], c[:, 1], color='#00C853', lw=2.2, alpha=0.95)
            ax.plot(c[0, 0], c[0, 1], 'o', color='#1565C0', ms=5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.2); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('#ccc')
            if i == 0:
                ax.set_title(va, color='#1A1A2E', fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(sh[:14], color='#555', fontsize=9)
            ax.text(0.02, 0.02, f"cov {r['coverage']:.3f}", fontsize=7,
                    color='#555', transform=ax.transAxes)

    fig.suptitle(suptitle, color='#1A1A2E', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)
    return out_path


def plot_anytime(curves, out_path, title):
    """Anytime-Kurven: Guete ueber gefahrener Weglaenge.

    Die eigentliche Vergleichsdarstellung. Ein einzelner Endwert vergleicht
    Budgets; diese Kurven zeigen, welche Variante bei *jedem* Budget vorn liegt.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    palette = {'Orakel': '#1565C0', 'A': '#00C853', 'B': '#EF6C00',
               'C': '#7B1FA2', 'D': '#00838F', 'E': '#C2185B',
               'Maeander': '#9E9E9E', 'Zufall': '#BDBDBD'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor='white')
    keys = [('coverage', 'Abdeckung der wahren Dichte (kleiner besser)'),
            ('info_gain', 'Informationsgewinn (groesser besser)'),
            ('belief_rmse', 'Glaubensfehler RMSE (kleiner besser)')]

    for ax, (key, lab) in zip(axes, keys):
        for name, cs in curves.items():
            xs = [st.mean(c[i]['path_len'] for c in cs) for i in range(len(cs[0]))]
            ys = [st.mean(c[i][key] for c in cs) for i in range(len(cs[0]))]
            ax.plot(xs, ys, label=name, color=palette.get(name, '#666'),
                    lw=2.2 if name in 'ABCDE' else 1.6,
                    ls='--' if name in ('Maeander', 'Zufall', 'Orakel') else '-',
                    alpha=0.95)
        ax.set_xlabel('gefahrene Weglaenge', color='#555')
        ax.set_title(lab, color='#1A1A2E', fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_facecolor('white')
        for s in ax.spines.values():
            s.set_color('#ccc')
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, color='#1A1A2E', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor='white')
    plt.close(fig)
    return out_path


# ===========================================================================
# Hauptlauf
# ===========================================================================

def _load(d, m):
    spec = importlib.util.spec_from_file_location(
        m, os.path.join(_here, d, m + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--auto', action='store_true',
                   help='Neueste Checkpoints aus checkpoints/ nehmen.')
    p.add_argument('--belief_ckpt', default=None)
    p.add_argument('--segment_ckpt', default=None)
    p.add_argument('--lookahead_ckpt', default=None)
    p.add_argument('--control_ckpt', default=None,
                   help='Auf WAHREN Dichten trainiertes Netz als Kontrolle.')
    p.add_argument('--shapes', type=int, default=12)
    p.add_argument('--grid_res', type=int, default=48)
    p.add_argument('--n_prior', type=int, default=12)
    p.add_argument('--n_particles', type=int, default=192)
    p.add_argument('--kappa', type=float, default=2.0)
    p.add_argument('--segments', type=int, default=3)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--lambda_cov', type=float, default=20000.0)
    p.add_argument('--target_length', type=float, default=4.0)
    p.add_argument('--budgets', type=float, nargs='+', default=[0.4, 0.7, 1.0])
    p.add_argument('--anytime_points', type=int, default=12)
    p.add_argument('--grad_steps', type=int, default=200,
                   help='Optimierungsschritte des Gradientenplaners — die '
                        'Groesse, gegen die das Netz antritt.')
    p.add_argument('--viz_shapes', type=int, default=4)
    p.add_argument('--out_dir', default=os.path.join(_here, 'results', 'eval'))
    p.add_argument('--device', default=None)
    a = p.parse_args()

    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ck_dir = os.path.join(_here, 'checkpoints')
    if a.auto:
        a.belief_ckpt = a.belief_ckpt or newest(f"{ck_dir}/explorer_belief_*.pt")
        a.segment_ckpt = a.segment_ckpt or newest(f"{ck_dir}/explorer_segment_*.pt")
        a.lookahead_ckpt = a.lookahead_ckpt or newest(f"{ck_dir}/explorer_lookahead_*.pt")

    os.makedirs(a.out_dir, exist_ok=True)
    print("\n=== Auswertung der trainierten Explorer ===")
    for n, c in (('belief', a.belief_ckpt), ('segment', a.segment_ckpt),
                 ('lookahead', a.lookahead_ckpt), ('control', a.control_ckpt)):
        print(f"  {n:10s} {os.path.basename(c) if c else '— nicht vorhanden'}")

    nets = {}
    for n, c in (('belief', a.belief_ckpt), ('segment', a.segment_ckpt),
                 ('lookahead', a.lookahead_ckpt)):
        if c and os.path.isfile(c):
            nets[n] = ExplorerPlanner(c, device=dev)

    mods = {'A': _load('variant_a_combined', 'run_a'),
            'B': _load('variant_b_diffsim', 'run_b'),
            'C': _load('variant_c_receding', 'run_c'),
            'D': _load('variant_d_belief_cond', 'run_d'),
            'E': _load('variant_e_two_stage', 'run_e')}

    names, T = load_truth(n=a.shapes, resolution=a.grid_res)
    print(f"\n  {len(names)} Holdout-Formen\n")

    D_EXEC = 0.35
    share = {'A': 1.0, 'B': 1.0, 'C': 1.0 / a.segments,
             'D': 1.0 / (a.rounds * D_EXEC), 'E': 0.5, 'ref': 1.0}

    def grad_planner(seed, nxi=25, who='ref'):
        return GradientPlanner(nxi=nxi, steps=a.grad_steps, seed=seed,
                               target_length=a.target_length * share[who])

    def net_planner(kind):
        return lambda *args_, **kw: nets[kind]

    common_kw = dict(n_prior=a.n_prior, grid_res=a.grid_res,
                     n_particles=a.n_particles)

    results, viz_rows, plan_times = [], [], {}
    curves = {}

    for planner_kind in (['Gradient'] + (['Netz'] if nets else [])):
        paths = {}
        for i, truth in enumerate(T):
            def mk(seed, nxi=25, who='ref'):
                if planner_kind == 'Gradient':
                    return grad_planner(seed, nxi, who)
                kind = ('segment' if who == 'C' else
                        'lookahead' if who == 'B' else 'belief')
                return nets.get(kind, nets.get('belief'))

            per = {}
            per['A'] = mods['A'].run_mission(truth, mk(i, who='A'),
                                             kappa=a.kappa, seed=i, **common_kw)[0]
            per['B'] = mods['B'].run_mission(
                truth, grad_planner(i, who='B') if planner_kind == 'Gradient'
                else mk(i, who='B'),
                lambda_cov=a.lambda_cov, steps=a.grad_steps, seed=i,
                **common_kw)[0] if planner_kind == 'Gradient' else \
                mods['A'].run_mission(truth, mk(i, who='B'), kappa=a.kappa,
                                      seed=i, **common_kw)[0]
            per['C'] = mods['C'].run_mission(
                truth, lambda sd, nx: mk(sd, nx, 'C'),
                n_segments=a.segments, seed=i, **common_kw)[0]
            per['D'] = mods['D'].run_mission(
                truth, lambda sd: mk(sd, who='D'), rounds=a.rounds,
                execute_frac=D_EXEC, seed=i, **common_kw)[0]
            per['E'] = mods['E'].run_mission(
                truth, lambda sd: mk(sd, who='E'), seed=i, **common_kw)[0]

            pl = grad_planner(i)
            per['Orakel'] = oracle_path(truth, pl, a.n_particles, seed=i)
            per['Maeander'] = lawnmower_path(target_length=a.target_length)
            per['Zufall'] = random_path(pl, seed=i)

            for k, v in per.items():
                paths.setdefault(k, []).append(v)
            print(f"  [{planner_kind}] {i+1}/{len(T)} {names[i]}")

        b0s = [initial_belief(T[i], n_prior=a.n_prior, grid_res=a.grid_res,
                              seed=i) for i in range(len(T))]
        cur = {k: [metrics.anytime_curve(pp, T[i], b0s[i],
                                         n_points=a.anytime_points)
                   for i, pp in enumerate(v)] for k, v in paths.items()}
        curves[planner_kind] = cur

        driven = {k: st.mean(c[-1]['path_len'] for c in cs) for k, cs in cur.items()}
        ref_len = min(driven.values()) * 0.98
        for k in cur:
            for f in a.budgets:
                b = f * ref_len
                vals = [metrics.at_budget(c, b, 'coverage') for c in cur[k]]
                inf = [metrics.at_budget(c, b, 'info_gain') for c in cur[k]]
                ok = [x for x in vals if x is not None]
                results.append({
                    'planner': planner_kind, 'variant': k, 'budget_frac': f,
                    'budget': round(b, 3),
                    'coverage': round(st.mean(ok), 5) if ok else None,
                    'info_gain': round(st.mean(x for x in inf if x is not None), 3)
                    if any(x is not None for x in inf) else None,
                    'n': len(ok), 'driven': round(driven[k], 3)})

        # Bilder fuer die ersten Formen
        for i in range(min(a.viz_shapes, len(T))):
            b0 = b0s[i]
            mu, sd = b0.posterior_grid()
            phi = ucb_density(mu, sd, kappa=a.kappa)
            pc = particles_from_density(phi, 120).cpu().numpy()
            for k in ['Orakel', 'A', 'B', 'C', 'D', 'E', 'Maeander']:
                if k not in paths:
                    continue
                viz_rows.append({
                    'planner': planner_kind, 'shape': names[i], 'variant': k,
                    'truth': T[i].cpu().numpy(), 'particles': pc,
                    'path': paths[k][i].cpu().numpy(),
                    'coverage': float(metrics.coverage_vs_truth(paths[k][i], T[i])),
                })

    # ── Ausgabe ────────────────────────────────────────────────────────────
    print("\n  Abdeckung bei festem Wegbudget (kleiner ist besser)")
    hdr = f"  {'Planer/Variante':<24}" + "".join(f"{f:g}x".rjust(12) for f in a.budgets)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for pk in curves:
        for k in ['Orakel', 'A', 'B', 'C', 'D', 'E', 'Maeander', 'Zufall']:
            sel = [r for r in results if r['planner'] == pk and r['variant'] == k]
            if not sel:
                continue
            line = f"  {pk[:4]}/{k:<18}"
            for f in a.budgets:
                v = next((r['coverage'] for r in sel if r['budget_frac'] == f), None)
                line += f"{v:12.4f}" if v is not None else f"{'n/a':>12}"
            print(line)

    # Amortisierung
    if 'Netz' in curves:
        print("\n  Amortisierung: Wanduhrzeit pro Planungsschritt")
        gp = grad_planner(0)
        pc = particles_from_density(
            ucb_density(*initial_belief(T[0], n_prior=a.n_prior,
                                        grid_res=a.grid_res).posterior_grid(),
                        kappa=a.kappa), a.n_particles)
        gp.plan(pc, 1)
        t_grad = gp.last_wallclock
        for kind, npl in nets.items():
            npl.plan(pc, 1)
            print(f"    Gradient {t_grad*1000:8.1f} ms  vs  Netz/{kind} "
                  f"{npl.last_wallclock*1000:6.1f} ms  "
                  f"-> {t_grad/max(npl.last_wallclock,1e-9):.0f}x")

    # ── Dateien ────────────────────────────────────────────────────────────
    csv_p = os.path.join(a.out_dir, 'metriken.csv')
    with open(csv_p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    figs = []
    for pk, cur in curves.items():
        figs.append(plot_anytime(
            {k: v for k, v in cur.items()},
            os.path.join(a.out_dir, f'anytime_{pk.lower()}.png'),
            f"Anytime-Vergleich — Planer: {pk}"))
    for pk in curves:
        rws = [r for r in viz_rows if r['planner'] == pk]
        if rws:
            figs.append(plot_missions(
                rws, os.path.join(a.out_dir, f'missionen_{pk.lower()}.png'),
                f"Missionen — Planer: {pk}"))

    print(f"\n  Metriken -> {csv_p}")
    for f in figs:
        print(f"  Bild     -> {f}")
    print()


if __name__ == '__main__':
    main()
