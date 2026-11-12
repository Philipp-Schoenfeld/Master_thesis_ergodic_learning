r"""
select_best_models.py
=====================
Which checkpoint of each family goes into the final exploration comparison
(`run_exploration_comparison.py`)? Answered from the active-sensing runs that already exist
(`results/as_models_20261028/summary.json`, written by `plot_active_sensing_models.py`): nothing
is recomputed.

Families (only flow-matching planners are candidates for the two flow-matching families):
  DiT           dit630                                   (the only 2D DiT)
  U-Net (CFM)   unet630, unet1890, unet0, lm1300, old500 (diffusion U-Net `diff630` is not a flow
                                                          matching model and is left out)
  self-supervised  selfsup                               (one trained checkpoint)

Rule: the model with the lowest ergodic error against the ground truth after the last round,
averaged over the knowledge conditions of the run (`none_known`, `ten_samples`); ties within one
standard error are reported but not resolved by this script -- the result is a recommendation,
checked in `best_models.json`, and overridable with `--methods` in the runner. The table lists
every candidate with its mean standard error so the margin between them is visible (the
differences between the U-Nets are within one SEM; see `results/as_models_20261028/report.md`
for the paired CIs).

    python select_best_models.py                  # print table, write best_models.json
    python select_best_models.py --print_methods  # only the --methods string for the runner
"""
import argparse
import json
import os

_here = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(_here, 'results', 'as_models_20261028', 'summary.json')
OUT = os.path.join(_here, 'best_models.json')

#: family -> candidate method keys of run_exploration_comparison.py; the JSON uses the
#: model keys of the as_models run for the lookup
FAMILIES = {
    'dit': {'dit630': 'dit630'},
    'unet': {'unet630': 'unet630', 'unet1890': 'unet1890', 'unet0': 'unet0',
             'lm1300': 'lm1300', 'old500': 'old500'},
    'selfsup': {'selfsup': None},        # no as_models entry: the single checkpoint is the best
}
#: methods that are always part of the final comparison, independent of the selection
ALWAYS = ('rw', 'lawn', 'gpgoal', 'linray', 'linadam', 'lintsvec')
#: already in the DB of the first full run
FINISHED = ('unet630', 'dit630', 'rw', 'lawn', 'linadam')


def score(conds, key):
    vals, sems = [], []
    for c, models in conds.items():
        m = models.get(key)
        if m is not None:
            vals.append(m['E_final'])
            sems.append(m['E_final_sem'])
    if not vals:
        return None, None, 0
    return sum(vals) / len(vals), sum(sems) / len(sems), len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--summary', default=SUMMARY)
    ap.add_argument('--print_methods', action='store_true',
                    help='Print only the comma-separated --methods string for the runner.')
    a = ap.parse_args()
    with open(a.summary, encoding='utf-8') as f:
        summ = json.load(f)
    conds = summ['conds']

    chosen, table = {}, []
    for fam, cands in FAMILIES.items():
        scored = []
        for method, key in cands.items():
            if key is None:
                scored.append((method, None, None, 0))
                continue
            mean, sem, n = score(conds, key)
            if mean is not None:
                scored.append((method, mean, sem, n))
        measured = [s for s in scored if s[1] is not None]
        if measured:
            best = min(measured, key=lambda s: s[1])
        else:
            best = scored[0]
        chosen[fam] = best[0]
        for m, mean, sem, n in sorted(scored, key=lambda s: (s[1] is None, s[1] or 0)):
            table.append((fam, m, mean, sem, n, m == best[0]))

    methods = list(ALWAYS) + [chosen[f] for f in ('dit', 'unet', 'selfsup')]
    # the first run's reference pair stays in the comparison (paired baseline of its tables)
    for m in ('unet630', 'dit630'):
        if m not in methods:
            methods.append(m)
    methods = list(dict.fromkeys(methods))
    to_run = [m for m in methods if m not in FINISHED]

    if a.print_methods:
        print(','.join(methods))
        return
    print(f'{"family":<9}{"model":<10}{"E_final (mean of conditions)":>32}{"mean SEM":>10}{"n cond":>8}')
    for fam, m, mean, sem, n, best in table:
        if mean is None:
            print(f'{fam:<9}{m:<10}{"-- (single checkpoint)":>32}{"":>10}{"":>8}  <- chosen')
        else:
            print(f'{fam:<9}{m:<10}{mean:>32.3f}{sem:>10.3f}{n:>8}{"  <- chosen" if best else ""}')
    print('\nchosen:', chosen)
    print('methods of the final comparison:', ','.join(methods))
    print('still to compute:', ','.join(to_run) or '-')
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(dict(rule='lowest final-round E vs. ground truth, mean over knowledge conditions '
                            'of results/as_models_20261028',
                       chosen=chosen, methods=methods, to_run=to_run,
                       table=[dict(family=fam, model=m, E_final=mean, sem=sem, n_conditions=n,
                                   chosen=best) for fam, m, mean, sem, n, best in table]),
                  f, indent=1)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
