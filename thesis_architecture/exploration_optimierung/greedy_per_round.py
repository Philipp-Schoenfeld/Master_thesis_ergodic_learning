r"""
greedy_per_round.py
====================
Pilot: reoptimiert den Phi-Parameter (tau/kappa/w) **jede Runde neu**, statt
ihn wie in `optimize.py` ueber die ganze Laengeneinheit-Mission fest zu
lassen.

Eine echte gemeinsame Optimierung von n_max unabhaengigen Rundenwerten ist
mit Rastersuche nicht rechenbar (16 Rasterpunkte ^ 6 Runden ~ 16,7 Mio.
Kombinationen fuer ein einziges Modell). Machbar ist eine **gierige
Ein-Runden-Optimierung**: an jeder Runde werden alle 16 Rasterpunkte batched
geplant und ausgefuehrt (dieselbe Batching-Idee wie in `LaengenMission`, hier
ueber Kandidaten statt ueber Formen), der mit der besten resultierenden
Abdeckung des **Glaubens** (mu_hat) wird committed, der Rest verworfen, dann
geht es zur naechsten Runde. Das ist nicht global optimal (kurzsichtig, sieht
nicht voraus, was ein anderer Wert fuer spaetere Runden bedeuten wuerde),
aber der einzige Ansatz, der ueberhaupt in vertretbarer Zeit lokal laeuft.

Die Auswahl darf **nicht** gegen die wahre Zieldichte gehen -- ein echter
Agent kennt sie nicht, und danach auszuwaehlen waere ein Blick auf das
Ergebnis (vgl. den Docstring von `apply_cfm_belief.best_candidate`, der aus
demselben Grund gegen Phi statt gegen die Wahrheit auswaehlt). Verwendet wird
deshalb `coverage_vs_truth(bahn, mu_hat)` als Auswahlkriterium; die
Wahrheit kommt erst danach ins Spiel, rein zur Auswertung von `cov`/`J`.

Vergleicht das Ergebnis gegen die feste Optimal-Policy-Einstellung
(dieselben Parameter wie `interactive_sim.OPTIMAL_POLICY['mit_svgd']`) auf
denselben Formen und demselben Seed.

    python -m exploration_optimierung.greedy_per_round \
        --model niveau --svgd_iters 25 --n_shapes 5 --n_max 6 --seed 0
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import DEFAULT_CKPT, RESULTS_DIR
from .mission import (LENGTH_UNIT, PHI_MODELS, LaengenMission,
                      blind_coverage, build_mission_args, build_planner,
                      load_holdout, param_grid, refine_batch,
                      resample_arclength)
from .mission import visitation_recent  # noqa: F401 (re-export fuer Klarheit)

import apply_cfm_belief as acb                                   # noqa: E402
from common.belief import GPBelief                               # noqa: E402
from common.metrics import coverage_vs_truth, path_length, trim_to_length  # noqa: E402
from common.observation import measure, thin                     # noqa: E402

#: Die feste OPTIMAL_POLICY['mit_svgd']-Einstellung je Modell, dieselben
#: Werte wie in `exploration/interactive_sim.py` (niveau ist der
#: veroeffentlichte Gewinner, die anderen drei zum Vergleich).
FIXED_PARAM = {'niveau': 0.6067, 'ucb': 0.4597, 'mass': 0.53, 'eid': 2.4662}


def run_fixed(planner, truths, names, model, param, n_max, svgd_iters, seed):
    """Baseline: EIN fester Parameter ueber die ganze Mission (wie optimize.py)."""
    args = build_mission_args(str(truths.device), phi_model=model, param=param)
    m = LaengenMission(planner, truths, names, args, svgd_iters=svgd_iters,
                       seed=seed)
    rows = m.run(n_max)
    return rows


def run_greedy(planner, truths, names, model, n_max, svgd_iters, seed,
              n_candidates=16, nxi_refine=25):
    """Pro Form und Runde: `n_candidates` Parameterwerte batched planen und
    ausfuehren, den mit dem besten resultierenden bezogenen
    Abdeckungsfehler je Form committen. -> (rows, chosen_params)

    rows: eine Zeile je (Form, Runde), Schema wie `LaengenMission.run` (also
    direkt kompatibel mit `objective.per_n`/`score_trace`).
    chosen_params: {shape_name: [param_runde_1, ..., param_runde_n_max]}
    """
    pname, _internal = PHI_MODELS[model]
    grid = param_grid(pname, n_candidates)
    S = truths.shape[0]
    device = truths.device

    cov_blind = [blind_coverage(truths[i]) for i in range(S)]
    beliefs = [GPBelief(grid_res=64, lengthscale=0.08, noise=0.05,
                        device=str(device)) for _ in range(S)]
    driven = [None] * S
    chosen = {names[i]: [] for i in range(S)}
    plan_s_total = [0.0] * S
    svgd_s_total = [0.0] * S
    rows = []

    torch.manual_seed(seed)
    for r in range(n_max):
        for i in range(S):
            here = driven[i]
            visit = (visitation_recent(here, 64, 0.06, str(device),
                                       half_life=3.0 * LENGTH_UNIT)
                     if here is not None else None)
            mu, sd = beliefs[i].posterior_grid()
            mu_hat = mu.clamp(min=0.0)
            mu_hat = mu_hat / mu_hat.sum().clamp(min=1e-12)

            phis = []
            kappa_of = []
            for p in grid:
                a = build_mission_args(str(device), phi_model=model, param=p)
                phi, _ = acb.debt_density(mu, sd, visit, a.kappa, a)
                phis.append(phi)
                kappa_of.append(a)
            parts = torch.stack([
                acb.phi_particles(phi, kappa_of[0].n_particles,
                                  mode=kappa_of[0].phi_mode, device=str(device))
                for phi in phis])

            pos = here[-1] if here is not None else None
            starts = (pos.unsqueeze(0).repeat(len(grid), 1)
                     if pos is not None else None)

            t0 = time.perf_counter()
            with torch.no_grad():
                cps = planner.plan(parts, n_candidates=len(grid), start=starts)
                curves = planner.render(cps)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            plan_dt = time.perf_counter() - t0

            t0 = time.perf_counter()
            refined = refine_batch(curves, phis, svgd_iters, nxi=nxi_refine)
            svgd_dt = time.perf_counter() - t0

            best = None
            for k, p in enumerate(grid):
                curve = torch.as_tensor(refined[k], device=device,
                                        dtype=torch.float32).clamp(0.0, 1.0)
                if pos is not None:
                    gap = float((curve[0] - pos.to(device)).norm())
                    if gap > 1e-6:
                        al = torch.linspace(0, 1, 8, device=device,
                                            dtype=curve.dtype).unsqueeze(-1)
                        link = (pos.to(device).unsqueeze(0) * (1 - al)
                               + curve[0].unsqueeze(0) * al)
                        curve = torch.cat([link, curve], dim=0)
                seg = trim_to_length(curve, LENGTH_UNIT)
                n_pts = max(8, int(round(72 * path_length(seg) / LENGTH_UNIT)))
                seg = resample_arclength(seg, n_pts)
                full = torch.cat([here, seg], dim=0) if here is not None else seg
                # Auswahl NUR gegen den Glauben (mu_hat), nie gegen die
                # Wahrheit -- ein echter Agent kennt die Wahrheit nicht.
                # Nach der Wahrheit auszuwaehlen waere ein Blick auf das
                # Ergebnis (vgl. `apply_cfm_belief.best_candidate`-Docstring)
                # und wuerde den Vergleich mit der festen Einstellung
                # unbrauchbar machen.
                cov_online = float(coverage_vs_truth(full, mu_hat))
                if best is None or cov_online < best[0]:
                    best = (cov_online, seg, p)

            _cov_online_best, seg, p_best = best
            # Reporting/J: wie ueberall sonst im Projekt gegen die Wahrheit
            # gemessen -- das ist die erlaubte, nachtraegliche Auswertung,
            # nicht die Entscheidung selbst.
            full_final = torch.cat([here, seg], dim=0) if here is not None else seg
            cov_final = float(coverage_vs_truth(full_final, truths[i]))
            pts, vals = measure(seg, truths[i], noise_std=0.02,
                                sensor_radius=0.06)
            beliefs[i].observe(*thin(pts, vals, max_points=64))
            driven[i] = seg if here is None else torch.cat([here, seg], dim=0)
            chosen[names[i]].append(p_best)
            plan_s_total[i] += plan_dt
            svgd_s_total[i] += svgd_dt

            rows.append(dict(
                shape=names[i], n_exec=r + 1, cov=cov_final,
                cov_norm=cov_final / max(cov_blind[i], 1e-12),
                param=p_best, plan_s=plan_s_total[i], svgd_s=svgd_s_total[i]))
    return rows, chosen


def run_greedy_wide(planner, truths, names, models, param_points,
                    svgd_buckets, n_max, seed, nxi_refine=25, pool=None,
                    plan_batch=96):
    """A-breit: pro Runde wird nicht nur der Parameter, sondern auch das
    Phi-Modell und die SVGD-Iterationszahl gemeinsam neu gewaehlt --
    Kandidatenraum ist das Kreuzprodukt (Modell x Parameter-Gitter x
    SVGD-Bucket). Auswahlkriterium wie `run_greedy`: Abdeckung von mu_hat, nie
    der Wahrheit.

    Die Schleife selbst steht seit dem Ausbau zum Trainingsdatensatz in
    `policy/oracle.py` (`orakel_rollout`) — dort wird zusaetzlich *jede*
    Kandidatenbewertung mitgeschrieben, was Option A als Trainingssignal
    braucht. Hier bleibt nur der Aufruf, damit es die Auswahllogik nicht
    zweimal gibt und beide Wege dieselben Zahlen liefern.

    -> (rows, chosen) mit chosen[form] = [(modell, param, svgd), ...] je Runde.
    """
    from .policy.features import aktionsraster
    from .policy.oracle import orakel_rollout

    candidates = aktionsraster(models, param_points, svgd_buckets)
    args = build_mission_args(str(truths.device), phi_model=models[0],
                              param=FIXED_PARAM.get(models[0], 1.0))
    rows, chosen, _m = orakel_rollout(planner, truths, names, args, candidates,
                                      n_max, seed=seed, pool=pool,
                                      plan_batch=plan_batch)
    # Spaltennamen der bisherigen Ausgabe beibehalten.
    for row in rows:
        row['model'] = row.pop('gewaehlt_modell', None)
        row['param'] = row.pop('gewaehlt_param', None)
        row['svgd'] = row.pop('gewaehlt_svgd', None)
    return rows, chosen


def _j_at_n(rows, n, n_max, lambda_len=0.02, lambda_time=0.004):
    """J = q + lambda_len*n + lambda_time*t(n), gemittelt ueber alle Formen
    in `rows` bei genau Rundenzahl n (dieselbe Formel wie objective.py)."""
    sub = [r for r in rows if r['n_exec'] == n]
    q = float(np.mean([r['cov_norm'] for r in sub]))
    t = float(np.mean([r['plan_s'] + r['svgd_s'] for r in sub])) * (n / n_max)
    return q + lambda_len * n + lambda_time * t, q, t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='niveau', choices=sorted(PHI_MODELS))
    p.add_argument('--svgd_iters', type=int, default=25)
    p.add_argument('--n_shapes', type=int, default=5)
    p.add_argument('--n_max', type=int, default=6)
    p.add_argument('--n_candidates', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--device', default=None)
    p.add_argument('--wide', action='store_true',
                   help="A-breit: Modell + Parameter + SVGD gemeinsam je "
                        "Runde neu waehlen (statt nur der Parameter eines "
                        "festen Modells)")
    p.add_argument('--models', nargs='*', default=sorted(PHI_MODELS),
                   help="nur mit --wide: welche Phi-Modelle im Kandidatenraum")
    p.add_argument('--param_points', type=int, default=4,
                   help="nur mit --wide: Parameterpunkte je Modell (grober "
                        "als --n_candidates, weil zusaetzlich mit Modell x "
                        "SVGD multipliziert wird)")
    p.add_argument('--svgd_buckets', nargs='*', type=int, default=[0, 25, 100],
                   help="nur mit --wide: SVGD-Iterationswerte im Kandidatenraum")
    a = p.parse_args(argv)

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if a.wide:
        k = len(a.models) * a.param_points * len(a.svgd_buckets)
        print(f"Device: {device}   A-breit: Modelle={a.models}   "
             f"Parameterpunkte/Modell={a.param_points}   "
             f"SVGD-Buckets={a.svgd_buckets}   Formen: {a.n_shapes}   "
             f"n_max: {a.n_max}   Kandidaten/Runde: {k}")
    else:
        print(f"Device: {device}   Modell: {a.model}   SVGD: {a.svgd_iters}   "
             f"Formen: {a.n_shapes}   n_max: {a.n_max}   Kandidaten/Runde: {a.n_candidates}")

    t_load0 = time.perf_counter()
    planner = build_planner(ckpt=a.ckpt, device=device)
    names, truths = load_holdout(resolution=96, device=device, limit=a.n_shapes)
    print(f"Formen: {names}  (Laden {time.perf_counter() - t_load0:.1f}s)")

    fixed_param = FIXED_PARAM[a.model]

    t0 = time.perf_counter()
    rows_fixed = run_fixed(planner, truths, names, a.model, fixed_param,
                           a.n_max, a.svgd_iters, a.seed)
    t_fixed = time.perf_counter() - t0
    J_fixed, q_fixed, time_fixed = _j_at_n(rows_fixed, a.n_max, a.n_max)
    print(f"\nFest ({a.model}={fixed_param:.4f} ueber alle {a.n_max} Runden): "
         f"J={J_fixed:.4f}  q={q_fixed:.4f}  t(n)={time_fixed:.2f}s  "
         f"[{t_fixed:.1f}s Rechenzeit]")

    t0 = time.perf_counter()
    if a.wide:
        k = len(a.models) * a.param_points * len(a.svgd_buckets)
        rows_greedy, chosen = run_greedy_wide(
            planner, truths, names, a.models, a.param_points,
            a.svgd_buckets, a.n_max, a.seed)
    else:
        k = a.n_candidates
        rows_greedy, chosen = run_greedy(planner, truths, names, a.model,
                                         a.n_max, a.svgd_iters, a.seed,
                                         n_candidates=a.n_candidates)
    t_greedy = time.perf_counter() - t0
    J_greedy, q_greedy, time_greedy = _j_at_n(rows_greedy, a.n_max, a.n_max)
    print(f"Greedy (pro Runde neu optimiert):                        "
         f"J={J_greedy:.4f}  q={q_greedy:.4f}  t(n)={time_greedy:.2f}s  "
         f"[{t_greedy:.1f}s Rechenzeit, {k}x teurer als Fest]")

    delta = J_fixed - J_greedy
    print(f"\nDelta J (Fest - Greedy): {delta:+.4f}  "
         f"({'Greedy besser' if delta > 0 else 'Fest besser oder gleich'})")

    print("\nGewaehlte Konfiguration je Form und Runde (Greedy):")
    for shape, params in chosen.items():
        if a.wide:
            cells = "  ".join(f"{m}/{p:.2f}/s{s}" for (m, p, s) in params)
        else:
            cells = "  ".join(f"{p:.3f}" for p in params)
        print(f"  {shape:<20} {cells}")

    out = dict(
        wide=a.wide, model=a.model, svgd_iters=a.svgd_iters,
        models=a.models if a.wide else None,
        param_points=a.param_points if a.wide else None,
        svgd_buckets=a.svgd_buckets if a.wide else None,
        n_candidates=k, n_shapes=a.n_shapes, n_max=a.n_max, seed=a.seed,
        shapes=names, fixed_param=fixed_param,
        J_fixed=J_fixed, q_fixed=q_fixed, time_s_fixed=time_fixed,
        wallclock_fixed_s=t_fixed,
        J_greedy=J_greedy, q_greedy=q_greedy, time_s_greedy=time_greedy,
        wallclock_greedy_s=t_greedy,
        delta_J=delta,
        chosen_params={s: [list(c) for c in cs] if a.wide else cs
                      for s, cs in chosen.items()},
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = 'wide' if a.wide else f'{a.model}_svgd{a.svgd_iters}'
    out_path = os.path.join(RESULTS_DIR, f'greedy_pilot_{tag}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nGeschrieben: {out_path}")
    return out


if __name__ == '__main__':
    main()
